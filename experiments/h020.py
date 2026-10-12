"""H020 (control): CAPE timing rule vs buy-and-hold on real S&P 500 data (Shiller via datasets/s-and-p-500)."""
import os
import numpy as np, pandas as pd
from common import save

URL = "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv"
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sp500.csv")

if __name__ == "__main__":
    if not os.path.exists(P):
        os.makedirs(os.path.dirname(P), exist_ok=True)
        import urllib.request; urllib.request.urlretrieve(URL, P)
    df = pd.read_csv(P, parse_dates=["Date"])
    df = df[(df["Long Interest Rate"] > 0) & (df["Dividend"] > 0)].reset_index(drop=True)
    y = df["Long Interest Rate"].values / 100
    stock = (df["SP500"].values[1:] + df["Dividend"].values[1:] / 12) / df["SP500"].values[:-1] - 1
    bond = y[:-1] / 12 - 8.0 * (y[1:] - y[:-1])          # ~10y bond, duration ~8
    rf = y[:-1] / 12
    cape = df["PE10"].values
    med = pd.Series(cape).rolling(120, min_periods=120).median().values
    sig = (cape < med)[:-1]                              # decided at end of previous month
    valid = ~np.isnan(med[:-1]) & (cape[:-1] > 0)
    dates = df["Date"].values[1:]
    lines, data = [f"Данные: {str(df['Date'].iloc[0])[:7]} – {str(df['Date'].iloc[-1])[:7]}, месяцев {len(df)}\n"], {}
    for lab, lo, hi in [("1891–1980", "1891", "1981"),
                        ("1981–конец данных (вне выборки)", "1981", "2100")]:
        m = valid & (dates >= np.datetime64(lo)) & (dates < np.datetime64(hi))
        s, b, f, g = stock[m], bond[m], rf[m], sig[m]
        switch = np.r_[0, np.abs(np.diff(g.astype(int)))]
        strat = np.where(g, s, b) - 0.001 * switch
        def stats(r):
            ex = r - f
            return 12 * r.mean(), np.sqrt(12) * ex.mean() / ex.std(), np.prod(1 + r) ** (12 / len(r)) - 1
        bh = stats(s); st = stats(strat); bo = stats(b)
        data[lab] = dict(buyhold=bh, cape_rule=st, bonds=bo, months=int(m.sum()), in_stocks=float(g.mean()))
        lines.append(f"## {lab}: месяцев {m.sum()}, правило в акциях {100*g.mean():.0f}% времени")
        for n, (mu, sh, cagr) in [("купил и держи акции", bh), ("правило CAPE", st), ("только облигации", bo)]:
            lines.append(f"  {n:22s} годовая доходность (геом.) {100*cagr:5.2f}%   Шарп {sh:.2f}")
    save("h020", "\n".join(lines) + "\n", data)
