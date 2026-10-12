"""UEFA Champions League league phase (2024+ format) simulator.

36 teams in 4 pots of 9 (by a noisy coefficient). Each team plays 2 opponents
from every pot (one home, one away). No country restrictions (simplification).
Goals ~ Poisson. Tie-break order (UEFA, league phase): 1 GD, 2 GF, 3 away GF,
4 wins, 5 away wins, 6 opponents' points, 7 opponents' GD, 8 opponents' GF,
9 disciplinary points (lower better), 10 club coefficient.
"""
import numpy as np

T = 36


def draw(rng, coef):
    pots = np.argsort(-coef).reshape(4, 9)
    games = []                                  # (home, away)
    for a in range(4):
        p = rng.permutation(9)                  # own pot: one 9-cycle
        for k in range(9):
            games.append((pots[a][p[k]], pots[a][p[(k + 1) % 9]]))
        for b in range(a + 1, 4):
            while True:
                s1, s2 = rng.permutation(9), rng.permutation(9)
                if np.all(s1 != s2):
                    break
            for i in range(9):
                games.append((pots[a][i], pots[b][s1[i]]))   # a home
                games.append((pots[b][s2[i]], pots[a][i]))   # a away
    return np.array(games), pots


def make_teams(rng, sd=0.35, coef_noise=0.15):
    s = rng.normal(0, sd, T)
    coef = s + rng.normal(0, coef_noise, T)
    return s, coef


def simulate(games, s, S, rng, mu=np.log(1.35), home=0.12):
    h, a = games[:, 0], games[:, 1]
    d = (s[h] - s[a]) / 2
    gh = rng.poisson(np.exp(mu + home + d), (S, len(games)))
    ga = rng.poisson(np.exp(mu - home - d), (S, len(games)))
    return gh, ga


def table(games, gh, ga, coef, rng, depth=10):
    """Return ranking order (S, 36) best first, plus stats dict."""
    S = gh.shape[0]
    H = np.zeros((len(games), T)); A = np.zeros((len(games), T))
    H[np.arange(len(games)), games[:, 0]] = 1; A[np.arange(len(games)), games[:, 1]] = 1
    hw, dr, aw = (gh > ga) * 1.0, (gh == ga) * 1.0, (gh < ga) * 1.0
    pts = (3 * hw + dr) @ H + (3 * aw + dr) @ A
    gf = gh @ H + ga @ A; gc = ga @ H + gh @ A; gd = gf - gc
    agf = ga @ A
    w = hw @ H + aw @ A; awin = aw @ A
    Opp = H.T @ A + A.T @ H                      # 36x36 opponent matrix
    opp_pts = pts @ Opp.T; opp_gd = gd @ Opp.T; opp_gf = gf @ Opp.T
    disc = rng.integers(0, 15, (S, T)).astype(float)
    coefm = np.broadcast_to(coef, (S, T))
    rnd = rng.random((S, T))
    keys = [pts, gd, gf, agf, w, awin, opp_pts, opp_gd, opp_gf, -disc, coefm]
    used = keys[:1 + depth] + [rnd]
    order = np.lexsort([-k for k in used[::-1]], axis=-1)   # last key = primary
    stats = dict(pts=pts, gd=gd, gf=gf, agf=agf, w=w, awin=awin, opp_pts=opp_pts,
                 opp_gd=opp_gd, opp_gf=opp_gf)
    return order, stats
