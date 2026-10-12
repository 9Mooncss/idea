import numpy as np
from common import swiss


def tie_run(cfg, rng, with_bt=False):
    T = swiss.run_tournament(cfg, rng)
    tb = swiss.tiebreaks(T, with_bt=with_bt)
    return {k: v for k, v in swiss.tie_concordance(T, tb).items()}


def aggregate(results):
    agg = {}
    for r in results:
        for k, (a, n) in r.items():
            x = agg.setdefault(k, [0.0, 0, []]); x[0] += a; x[1] += n
            if n:
                x[2].append(a / n)
    out = {}
    for k, (a, n, per) in agg.items():
        per = np.array(per)
        # CI from tournament-level variation (pairs inside one tournament are dependent)
        out[k] = (a / n, 1.96 * per.std() / np.sqrt(len(per)))
    return out
