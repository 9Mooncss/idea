"""Sanity checks of the simulator: rematches, colour balance, draw rate, score-group pairing."""
import numpy as np
import swiss
rng = np.random.default_rng(0)
c = dict(swiss.DEFAULT)
rem = 0; colour = []; draws = []; samegroup = []; e_err = []
for t in range(200):
    T = swiss.run_tournament(c, rng)
    rem += sum(len(o) - len(set(o)) for o in T["opps"]) // 2
    colour += list(np.abs(2 * T["whites"] - c["rounds"]))
    draws += [g[2] == 0.5 for g in T["games"]]
    # expected vs actual score check (true strengths)
    for w, b, x in T["games"]:
        e_err.append(x - swiss.expected(T["s"][w] + c["white_adv"], T["s"][b]))
print("rematches per tournament:", rem / 200)
print("colour |W-B| distribution:", {k: colour.count(k) for k in sorted(set(colour))})
print("overall draw rate:", np.mean(draws))
print("mean(actual - expected) score of white:", np.mean(e_err), "(should be ~0)")
