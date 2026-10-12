"""Simple Swiss-system chess tournament simulator.

Model: every player has a hidden "true" strength (Elo scale) and a public
rating = true strength + noise. Game outcome follows the Elo formula with a
draw probability that is highest between equal players and a small white
advantage. Pairing is a simplified Dutch system (NOT a full FIDE-compliant
implementation): score groups, top half vs bottom half, no rematches,
leftovers float down.
"""
import numpy as np
from scipy.optimize import minimize

DEFAULT = dict(N=100, rounds=9, sd_true=250.0, sd_noise=75.0, mean=2000.0,
               dmax=0.35, white_adv=30.0, rule="dutch", win_pts=1.0,
               draw_pts=0.5, pairing_key="rating")


def expected(a, b):
    return 1.0 / (1.0 + 10 ** (-(a - b) / 400.0))


def play(sw, sb, cfg, rng):
    """Return score of white (1, 0.5, 0) in chess terms."""
    e = expected(sw + cfg["white_adv"], sb)
    pd = cfg["dmax"] * (1 - abs(2 * e - 1))
    u = rng.random()
    if u < e - pd / 2:
        return 1.0
    if u < e + pd / 2:
        return 0.5
    return 0.0


def make_players(cfg, rng):
    s = rng.normal(cfg["mean"], cfg["sd_true"], cfg["N"])
    r = s + rng.normal(0, cfg["sd_noise"], cfg["N"])
    return s, r


# ---------------------------------------------------------------- pairing
def _pair_group(G, played, rule, rng):
    pairs, fl = [], []
    if rule in ("adjacent", "random"):
        H = list(G)
        if rule == "random":
            rng.shuffle(H)
        used = set()
        for i, p in enumerate(H):
            if p in used:
                continue
            for q in H[i + 1:]:
                if q not in used and q not in played[p]:
                    pairs.append((p, q)); used.update((p, q)); break
            else:
                fl.append(p)
        return pairs, fl
    m = len(G) // 2
    S1, S2 = G[:m], G[m:]
    used = set()
    for k, p in enumerate(S1):
        if rule == "dutch":
            cand = S2[k:] + S2[:k][::-1]
        elif rule == "fold":
            t = len(S2) - 1 - k
            cand = sorted(S2, key=lambda q: abs(S2.index(q) - t))
        else:
            raise ValueError(rule)
        for q in cand:
            if q not in used and q not in played[p]:
                pairs.append((p, q)); used.update((p, q)); break
        else:
            fl.append(p)
    fl += [q for q in S2 if q not in used]
    pos = {p: i for i, p in enumerate(G)}
    fl.sort(key=pos.get)
    return pairs, fl


def pair_round(order, score, played, rule, rng, use_groups=True):
    pairs, floaters = [], []
    if use_groups:
        groups, cur, cs = [], [], None
        for p in order:
            if cs is None or score[p] != cs:
                if cur:
                    groups.append(cur)
                cur, cs = [], score[p]
            cur.append(p)
        groups.append(cur)
    else:
        groups = [list(order)]
    for g in groups:
        G = floaters + g
        pr, floaters = _pair_group(G, played, rule, rng)
        pairs += pr
    # leftovers at the bottom: greedy, allow rematch only as last resort
    L = floaters
    while len(L) >= 2:
        p = L.pop(0)
        j = next((i for i, q in enumerate(L) if q not in played[p]), 0)
        pairs.append((p, L.pop(j)))
    return pairs


# ---------------------------------------------------------------- BT fit
def bt_fit(N, games, prior_mean=None, prior_sd=400.0):
    """Elo/Bradley-Terry MAP estimate. games: list of (a, b, score_a)."""
    if prior_mean is None:
        prior_mean = np.full(N, 2000.0)
    A = np.array([g[0] for g in games]); B = np.array([g[1] for g in games])
    S = np.array([g[2] for g in games])
    k = np.log(10) / 400.0

    def f(x):
        d = x[A] - x[B]
        p = 1 / (1 + np.exp(-k * d))
        p = np.clip(p, 1e-12, 1 - 1e-12)
        ll = np.sum(S * np.log(p) + (1 - S) * np.log(1 - p))
        lp = -0.5 * np.sum(((x - prior_mean) / prior_sd) ** 2)
        g = np.zeros(N)
        r = k * (S - p)
        np.add.at(g, A, r); np.add.at(g, B, -r)
        g -= (x - prior_mean) / prior_sd ** 2
        return -(ll + lp), -g

    res = minimize(f, prior_mean.copy(), jac=True, method="L-BFGS-B")
    return res.x


# ---------------------------------------------------------------- tournament
def run_tournament(cfg, rng, s=None, r=None, adaptive=False):
    N = cfg["N"]
    if s is None:
        s, r = make_players(cfg, rng)
    score = np.zeros(N); chess = np.zeros(N)
    played = [set() for _ in range(N)]
    opps = [[] for _ in range(N)]
    res = [[] for _ in range(N)]          # chess result 1/0.5/0 per round
    cum = [[] for _ in range(N)]
    whites = np.zeros(N, int); last = np.zeros(N, int)
    games = []
    for rd in range(cfg["rounds"]):
        if adaptive and rd > 0:
            est = bt_fit(N, games, prior_mean=r, prior_sd=cfg["sd_noise"] + 1)
            order = sorted(range(N), key=lambda i: -est[i])
            pairs = pair_round(order, score, played, "adjacent", rng, use_groups=False)
        else:
            order = sorted(range(N), key=lambda i: (-score[i], -r[i]))
            pairs = pair_round(order, score, played, cfg["rule"], rng)
        for a, b in pairs:
            ca = 2 * whites[a] - len(opps[a]); cb = 2 * whites[b] - len(opps[b])
            if ca < cb or (ca == cb and (last[a] < last[b] or (last[a] == last[b] and rng.random() < .5))):
                w, bl = a, b
            else:
                w, bl = b, a
            x = play(s[w], s[bl], cfg, rng)
            whites[w] += 1; last[w] = 1; last[bl] = -1
            for p, q, v in ((w, bl, x), (bl, w, 1 - x)):
                played[p].add(q); opps[p].append(q); res[p].append(v)
                chess[p] += v
                score[p] += cfg["win_pts"] if v == 1 else (cfg["draw_pts"] if v == .5 else 0)
            games.append((w, bl, x))
        for p in range(N):
            cum[p].append(score[p])
    return dict(s=s, r=r, score=score, chess=chess, opps=opps, res=res,
                cum=cum, whites=whites, games=games, cfg=cfg)


# ---------------------------------------------------------------- tiebreaks
def tiebreaks(T, with_bt=True):
    N = len(T["s"]); sc = T["score"]; r = T["r"]
    tb = {}
    bh = np.array([sum(sc[o] for o in T["opps"][p]) for p in range(N)])
    tb["BH"] = bh
    tb["BH-C1"] = np.array([bh[p] - min(sc[o] for o in T["opps"][p]) for p in range(N)])
    tb["BH-Med"] = np.array([bh[p] - min(sc[o] for o in T["opps"][p]) - max(sc[o] for o in T["opps"][p]) for p in range(N)])
    tb["SB"] = np.array([sum(sc[o] * v for o, v in zip(T["opps"][p], T["res"][p])) for p in range(N)])
    tb["ARO"] = np.array([np.mean([r[o] for o in T["opps"][p]]) for p in range(N)])
    tb["Wins"] = np.array([sum(v == 1 for v in T["res"][p]) for p in range(N)], float)
    tb["Progressive"] = np.array([sum(T["cum"][p]) for p in range(N)])
    tb["OwnRating"] = r.copy()
    frac = np.clip(T["chess"] / len(T["res"][0]), 0.01, 0.99)
    tb["TPR"] = tb["ARO"] + 400 * np.log10(frac / (1 - frac))
    if with_bt:
        tb["BT-flat"] = bt_fit(N, T["games"], prior_sd=1000.0)
        tb["BT-prior"] = bt_fit(N, T["games"], prior_mean=r, prior_sd=T["cfg"]["sd_noise"] + 1)
    tb["Random"] = np.random.default_rng().random(N)
    return tb


def tie_concordance(T, tb):
    """For every pair tied on score: did the tiebreak put the truly stronger
    player first? Returns {name: (agree, total)}; ties in tiebreak count 1/2."""
    s, sc = T["s"], T["score"]
    out = {k: [0.0, 0] for k in tb}
    for v in np.unique(sc):
        idx = np.where(sc == v)[0]
        if len(idx) < 2:
            continue
        i, j = np.triu_indices(len(idx), 1)
        a, b = idx[i], idx[j]
        sgn_true = np.sign(s[a] - s[b])
        for k, x in tb.items():
            d = np.sign(x[a] - x[b])
            agree = np.where(d == 0, 0.5, (d == sgn_true).astype(float))
            out[k][0] += agree.sum(); out[k][1] += len(a)
    return out


def final_order(T, tbvec):
    N = len(T["s"])
    return sorted(range(N), key=lambda i: (-T["score"][i], -tbvec[i]))
