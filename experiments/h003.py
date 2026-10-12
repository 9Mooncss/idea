"""H003: ranking by Bradley-Terry fit vs points + BH-C1."""
import numpy as np
from scipy.stats import kendalltau
from common import sim, cfg, save, ci, swiss


def run(c, rng):
    T = swiss.run_tournament(c, rng)
    tb = swiss.tiebreaks(T, with_bt=True)
    s = T["s"]; best = int(np.argmax(s)); top3 = set(np.argsort(-s)[:3])
    out = {}
    for name, order in {
        "очки + BH-C1": swiss.final_order(T, tb["BH-C1"]),
        "очки + собств. рейтинг": swiss.final_order(T, tb["OwnRating"]),
        "модель БТ (без рейтинга)": list(np.argsort(-tb["BT-flat"])),
        "модель БТ (с рейтингом)": list(np.argsort(-tb["BT-prior"])),
        "только стартовый рейтинг": list(np.argsort(-T["r"])),
    }.items():
        rank = np.empty(len(s)); rank[order] = np.arange(len(s))
        tau = kendalltau(-rank, s)[0]
        out[name] = (tau, order[0] == best, len(top3 & set(order[:3])) / 3)
    return out


if __name__ == "__main__":
    lines, data = [], {}
    for label, c in {"базовая": cfg(), "шумные рейтинги (150)": cfg(sd_noise=150.0),
                     "40 игроков, 7 туров": cfg(N=40, rounds=7)}.items():
        R = sim(run, c, 1500, seed=3)
        lines.append(f"\n## {label}  (тау Кендалла | доля правильного 1-го места | доля верных в топ-3)")
        data[label] = {}
        for k in R[0]:
            t = np.array([r[k][0] for r in R]); w = np.mean([r[k][1] for r in R]); t3 = np.mean([r[k][2] for r in R])
            data[label][k] = (t.mean(), w, t3)
            lines.append(f"  {k:26s} {t.mean():.4f} ± {1.96*t.std()/np.sqrt(len(t)):.4f} | {w:.3f} ± {ci(w,len(R)):.3f} | {t3:.3f}")
    save("h003", "\n".join(lines) + "\n", data)
