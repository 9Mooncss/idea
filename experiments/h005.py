"""H005: pairing rule inside score groups."""
import numpy as np
from scipy.stats import kendalltau
from common import sim, cfg, save, ci, swiss


def run(c, rng):
    T = swiss.run_tournament(c, rng)
    tb = swiss.tiebreaks(T, with_bt=False)
    order = swiss.final_order(T, tb["BH-C1"])
    s = T["s"]; rank = np.empty(len(s)); rank[order] = np.arange(len(s))
    return kendalltau(-rank, s)[0], order[0] == int(np.argmax(s))


if __name__ == "__main__":
    lines, data = ["правило | тау Кендалла | доля правильного победителя\n"], {}
    for setting, base in {"100 игроков, 9 туров": {}, "128 игроков, 7 туров": dict(N=128, rounds=7)}.items():
        lines.append(f"\n## {setting}")
        for rule in ["dutch", "fold", "adjacent", "random"]:
            R = sim(run, cfg(rule=rule, **base), 4000, seed=5)
            t = np.array([r[0] for r in R]); w = np.mean([r[1] for r in R])
            data[f"{setting}/{rule}"] = (t.mean(), w)
            lines.append(f"  {rule:9s} {t.mean():.4f} ± {1.96*t.std()/np.sqrt(len(t)):.4f} | {w:.3f} ± {ci(w,len(R)):.3f}")
    save("h005", "\n".join(lines) + "\n", data)
