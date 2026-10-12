"""H010: adaptive (posterior-based) pairing vs Dutch."""
import numpy as np
from scipy.stats import kendalltau
from common import sim, cfg, save, ci, swiss


def run(c, rng, adaptive=False):
    T = swiss.run_tournament(c, rng, adaptive=adaptive)
    tb = swiss.tiebreaks(T, with_bt=False)
    order = swiss.final_order(T, tb["BH-C1"])
    s = T["s"]; rank = np.empty(len(s)); rank[order] = np.arange(len(s))
    return kendalltau(-rank, s)[0], order[0] == int(np.argmax(s))


if __name__ == "__main__":
    lines, data = ["вариант | тау | доля правильного победителя\n"], {}
    for lab, c in {"базовая (шум 75)": cfg(), "шумные рейтинги (150)": cfg(sd_noise=150.0)}.items():
        lines.append(f"\n## {lab}")
        for ad in [False, True]:
            R = sim(run, c, 2000, seed=10, adaptive=ad)
            t = np.array([r[0] for r in R]); p = np.mean([r[1] for r in R])
            data[f"{lab}/{ad}"] = (t.mean(), p)
            lines.append(f"  {'адаптивная' if ad else 'голландская':11s} {t.mean():.4f} ± {1.96*t.std()/np.sqrt(len(t)):.4f} | {p:.3f} ± {ci(p,len(R)):.3f}")
    save("h010", "\n".join(lines) + "\n", data)
