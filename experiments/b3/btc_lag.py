"""H021: do Polymarket 'Bitcoin up or down' prices lag Binance enough to profit after fees?

Data: marketlenstrade/polymarket-historical-data (CC BY 4.0, "Marketlens, https://marketlens.trade"),
folder bitcoin-5m-15m-hourly-2026-06-15: full order-book history of all 408 markets of one day
plus the Binance BTC trade tape. Pass the folder path as argv[1].

Fair value of 'Up' at time t: P = Phi( ln(S_t / S_open) / (sigma * sqrt(tau)) ), tau = time to close,
sigma = realised volatility of 1-second Binance log returns over the trailing 30 minutes.
Strategy: when fair - ask > margin buy Up at the ask; when bid - fair > margin buy Down at 1 - bid.
Execution happens at the book `lat` seconds later (latency). Fee per share = 0.07 * p * (1 - p).
One position per market per side (no pyramiding), 1 share; settle at the official outcome.
"""
import sys, json
import numpy as np, pandas as pd
from math import erf, sqrt

D = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "btc_lag_result.json"

ref = pd.read_parquet(f"{D}/reference-BTC.parquet")
ref["price"] = ref["price"].astype(float)
t0 = int(ref.timestamp.min() // 1000); t1 = int(ref.timestamp.max() // 1000) + 1
sec = np.arange(t0, t1)
idx = np.searchsorted(ref.timestamp.values, sec * 1000, side="right") - 1
px = ref.price.values[np.clip(idx, 0, None)]                 # last price at each second
lr = np.r_[0, np.diff(np.log(px))]
win = 1800
c2 = np.cumsum(lr ** 2)
var = (c2 - np.r_[np.zeros(win), c2[:-win]]) / np.minimum(np.arange(1, len(c2) + 1), win)
sig = np.sqrt(np.maximum(var, 1e-12))                        # per-sqrt-second


def S(ts):                                                   # BTC price at unix second ts
    return px[np.clip(int(ts) - t0, 0, len(px) - 1)]


def Phi(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def fair_up(t_ms, open_s, close_s):
    ts = t_ms / 1000; tau = close_s - ts
    if tau <= 0:
        return None
    s = sig[np.clip(int(ts) - t0, 0, len(sig) - 1)]
    return Phi(np.log(S(ts) / S(open_s)) / (s * np.sqrt(tau)))


def book_series(path):
    """Top of book (t, bid, ask, ask_size, bid_size) at every snapshot/trade event."""
    h = pd.read_parquet(path)
    bids, asks = {}, {}; out = []
    for r in h.itertuples(index=False):
        if r.event_type == "snapshot":
            bids = {float(x["price"]): float(x["size"]) for x in json.loads(r.bids)} if isinstance(r.bids, str) else {}
            asks = {float(x["price"]): float(x["size"]) for x in json.loads(r.asks)} if isinstance(r.asks, str) else {}
        elif r.event_type == "delta":
            book = bids if r.side == "BUY" else asks
            if r.size == 0:
                book.pop(float(r.price), None)
            else:
                book[float(r.price)] = float(r.size)
            continue                                         # book exact only at trades/snapshots
        if bids and asks:
            b = max(bids); a = min(asks)
            out.append((r.t, b, a, asks[a], bids[b]))
    return np.array(out) if out else np.zeros((0, 5))


m = pd.read_csv(f"{D}/markets.csv")
m["open_s"] = [int(x.timestamp()) for x in pd.to_datetime(m.open_time)]
m["close_s"] = [int(x.timestamp()) for x in pd.to_datetime(m.close_time)]
fee = lambda p: 0.07 * p * (1 - p)

records = []          # per evaluation point: series, market, tau, mid, fair, outcome
books = {}
for r in m.itertuples():
    B = book_series(f"{D}/{r.file}")
    books[r.market_id] = B
    up = 1.0 if r.winning_outcome == "Up" else 0.0
    for t, b, a, *_ in B:
        f = fair_up(t, r.open_s, r.close_s)
        if f is None or t / 1000 < r.open_s:
            continue
        records.append((r.series, r.market_id, r.close_s - t / 1000, (a + b) / 2, a - b, f, up))
rec = pd.DataFrame(records, columns=["series", "mid_id", "tau", "mid", "spread", "fair", "up"])


def brier(p, y):
    return float(np.mean((p - y) ** 2))


summary = {"n_markets": int(len(m)), "n_points": int(len(rec)),
           "brier_market_mid": brier(rec.mid, rec.up), "brier_fair_model": brier(rec.fair, rec.up),
           "mean_abs_gap": float(np.mean(np.abs(rec.mid - rec.fair))),
           "median_spread": float(rec.spread.median())}


TRADES = []


def backtest(margin, lat, stale=0, tag=None):
    pnl = []; edge = []; sizes = []
    for r in m.itertuples():
        B = books[r.market_id]
        if len(B) == 0:
            continue
        up = 1.0 if r.winning_outcome == "Up" else 0.0
        done = {"up": False, "down": False}
        for k, (t, b, a, asz, bsz) in enumerate(B):
            if t / 1000 < r.open_s:
                continue
            f = fair_up(t - stale * 1000, r.open_s, r.close_s) if t - stale * 1000 >= r.open_s * 1000 else None
            if f is None:
                if t / 1000 >= r.close_s:
                    break
                continue
            # execution at the book `lat` seconds later
            j = np.searchsorted(B[:, 0], t + lat * 1000)
            if j >= len(B):
                break
            _, b2, a2, asz2, bsz2 = B[j]
            if not done["up"] and f - a > margin:
                p = a2; pnl.append(up - p - fee(p)); edge.append(f - a); sizes.append(asz2); done["up"] = True
                if tag: TRADES.append((tag, r.series, r.open_s, r.close_s - t / 1000, 'up', p, f, up - p - fee(p), asz2))
            if not done["down"] and b - f > margin:
                p = 1 - b2; pnl.append((1 - up) - p - fee(p)); edge.append(b - f); sizes.append(bsz2); done["down"] = True
                if tag: TRADES.append((tag, r.series, r.open_s, r.close_s - t / 1000, 'down', p, 1 - f, (1 - up) - p - fee(p), bsz2))
    pnl = np.array(pnl)
    return dict(margin=margin, latency_s=lat, n_trades=int(len(pnl)),
                mean_pnl_per_share=float(pnl.mean()) if len(pnl) else None,
                se=float(pnl.std() / np.sqrt(len(pnl))) if len(pnl) > 1 else None,
                total_pnl_1share=float(pnl.sum()) if len(pnl) else 0.0,
                median_top_size=float(np.median(sizes)) if sizes else None)


summary["backtests"] = [backtest(mg, lat) for mg in [0.03, 0.05, 0.10, 0.20] for lat in [0, 1, 3, 10]]
# robustness: stale Binance signal (60 s / 300 s old) -> if still profitable, the edge is not latency
summary["stale_signal"] = [backtest(mg, 3, stale=st) for mg in [0.10, 0.20] for st in [60, 300]]
backtest(0.10, 3, tag="m10"); backtest(0.20, 3, tag="m20")
pd.DataFrame(TRADES, columns=["tag", "series", "open_s", "tau", "side", "price", "fair", "pnl", "top_size"]).to_csv(OUT.replace(".json", "_trades.csv"), index=False)
# baseline without Binance at all: buy the favourite whenever the mid is in a band (pure favourite-longshot bias)
def fav_baseline(lo, hi, lat=3):
    pnl = []
    for r in m.itertuples():
        B = books[r.market_id]; up = 1.0 if r.winning_outcome == "Up" else 0.0
        for k, (t, b, a, asz, bsz) in enumerate(B):
            if t / 1000 < r.open_s or t / 1000 >= r.close_s:
                continue
            mid = (a + b) / 2
            j = np.searchsorted(B[:, 0], t + lat * 1000)
            if j >= len(B):
                break
            _, b2, a2, _, _ = B[j]
            if lo <= mid <= hi:
                p = a2; pnl.append(up - p - fee(p)); break
            if lo <= 1 - mid <= hi:
                p = 1 - b2; pnl.append((1 - up) - p - fee(p)); break
    pnl = np.array(pnl)
    return dict(band=[lo, hi], n=int(len(pnl)), mean=float(pnl.mean()), se=float(pnl.std() / np.sqrt(len(pnl))))
summary["favourite_baseline_no_binance"] = [fav_baseline(.7, .9), fav_baseline(.8, .95), fav_baseline(.9, .99)]
# calibration by bucket: realized frequency vs market mid vs fair
rec["bucket"] = pd.cut(rec.mid, [0, .1, .3, .5, .7, .9, 1.0])
summary["calibration"] = rec.groupby("bucket", observed=True).agg(n=("up", "size"), mid=("mid", "mean"),
                                                                  fair=("fair", "mean"), realized=("up", "mean")).reset_index().astype({"bucket": str}).to_dict("records")
json.dump(summary, open(OUT, "w"), indent=1)
print(json.dumps(summary, indent=1, ensure_ascii=False))
