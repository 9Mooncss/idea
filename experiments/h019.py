"""H019: CS2 Major Swiss stage: Valve's Buchholz ("difficulty") sorting vs seed-only.

Implements the Swiss rules of Valve's Major Supplemental Rulebook
(github.com/ValveSoftware/counter-strike_rules_and_regs, commit of 2026-10-07):
round 1: seed i vs i+8; rounds 2-3: highest seed vs lowest available seed without
rematch; later rounds (6-team groups): the priority table; elimination and
advancement matches are Bo3, others Bo1. Mid-stage seed = (W-L record, Difficulty
Score = sum of opponents' current W minus L, initial seed).
"""
import numpy as np
from common import sim, save

N = 16


def p_win(a, b):
    return 1 / (1 + 10 ** (-(a - b) / 400))


def series(s, x, y, bo, rng):
    need = bo // 2 + 1; wx = wy = 0
    while wx < need and wy < need:
        if rng.random() < p_win(s[x], s[y]):
            wx += 1
        else:
            wy += 1
    return (x, y) if wx > wy else (y, x)


PRIORITY = [((1, 6), (2, 5), (3, 4)), ((1, 6), (2, 4), (3, 5)), ((1, 5), (2, 6), (3, 4)),
            ((1, 5), (2, 4), (3, 6)), ((1, 4), (2, 6), (3, 5)), ((1, 4), (2, 5), (3, 6)),
            ((1, 6), (2, 3), (4, 5)), ((1, 5), (2, 3), (4, 6)), ((1, 3), (2, 6), (4, 5)),
            ((1, 3), (2, 5), (4, 6)), ((1, 4), (2, 3), (5, 6)), ((1, 3), (2, 4), (5, 6)),
            ((1, 2), (3, 6), (4, 5)), ((1, 2), (3, 5), (4, 6)), ((1, 2), (3, 4), (5, 6))]


def table_pairs(g, opps):
    for row in PRIORITY:
        pr = [(g[a - 1], g[b - 1]) for a, b in row]
        if all(y not in opps[x] for x, y in pr):
            return pr
    return [(g[a - 1], g[b - 1]) for a, b in PRIORITY[0]]


def stage(s, seed_rank, rng, use_buchholz):
    W = np.zeros(N, int); L = np.zeros(N, int); opps = [[] for _ in range(N)]
    by_seed = list(np.argsort(seed_rank))
    pairs = [(by_seed[i], by_seed[i + 8]) for i in range(8)]
    while True:
        for x, y in pairs:
            bo = 3 if (W[x] == 2 or L[x] == 2) else 1
            w, l = series(s, x, y, bo, rng)
            W[w] += 1; L[l] += 1; opps[x].append(y); opps[y].append(x)
        active = [t for t in range(N) if W[t] < 3 and L[t] < 3]
        if not active:
            break
        diff = W - L
        pairs = []
        for rec in sorted({(W[t], L[t]) for t in active}, reverse=True):
            g = [t for t in active if (W[t], L[t]) == rec]
            key = (lambda t: (-sum(diff[o] for o in opps[t]), seed_rank[t])) if use_buchholz else (lambda t: seed_rank[t])
            g.sort(key=key)
            if len(g) == 6 and len(opps[g[0]]) >= 3:
                pairs += table_pairs(g, opps); continue
            while g:
                x = g.pop(0)
                cand = [y for y in reversed(g) if y not in opps[x]] or list(reversed(g))
                y = cand[0]; g.remove(y); pairs.append((x, y))
    return [t for t in range(N) if W[t] == 3]


def run(c, rng):
    s = rng.normal(0, c["sd"], N)
    seed_rank = np.argsort(np.argsort(-(s + rng.normal(0, c["seed_noise"], N))))
    top8 = set(np.argsort(-s)[:8]); best = int(np.argmax(s))
    out = {}
    for lab, ub in [("Valve (Бухгольц)", True), ("только посев", False)]:
        adv = stage(s, seed_rank, rng, ub)
        out[lab] = (len(top8 & set(adv)), best in adv)
    return out


if __name__ == "__main__":
    lines, data = ["Среднее число настоящих топ-8 среди 8 прошедших | доля турниров, где прошёл сильнейший\n"], {}
    for sd, noise in [(150, 75), (100, 50), (200, 100)]:
        R = sim(run, dict(sd=sd, seed_noise=noise), 20000, seed=71)
        lines.append(f"## разброс силы {sd} Эло, ошибка посева {noise}")
        for k in R[0]:
            a = np.array([r[k][0] for r in R]); b = np.mean([r[k][1] for r in R])
            data[f"{sd}/{k}"] = (a.mean(), b)
            lines.append(f"  {k:18s} {a.mean():.3f} ± {1.96*a.std()/np.sqrt(len(a)):.3f} | {b:.3f}")
        d = np.array([r["Valve (Бухгольц)"][0] - r["только посев"][0] for r in R])
        lines.append(f"  разница (Valve − посев): {d.mean():+.3f} ± {1.96*d.std()/np.sqrt(len(d)):.3f}")
    save("h019", "\n".join(lines) + "\n", data)
