"""H009: underrated player penalty (same true strength, rating -300)."""
import numpy as np
from common import sim, cfg, save, swiss


def run(c, rng, shift=0.0):
    s, r = swiss.make_players(c, rng)
    i = int(np.argsort(s)[int(0.8 * len(s))])   # 80th percentile player
    r = r.copy(); r[i] = s[i] - shift
    T = swiss.run_tournament(c, rng, s=s, r=r)
    tb = swiss.tiebreaks(T, with_bt=False)
    order = swiss.final_order(T, tb["BH-C1"])
    return float(T["chess"][i]), order.index(i) + 1, float(np.mean(s[T["opps"][i]]))


if __name__ == "__main__":
    lines, data = [], {}
    for rule_cfg, lab in [(cfg(), "голландская, 9 туров"), (cfg(rounds=5), "голландская, 5 туров")]:
        lines.append(f"\n## {lab}")
        base = None
        for shift in [0.0, 150.0, 300.0]:
            R = np.array(sim(run, rule_cfg, 6000, seed=9, shift=shift))
            m = R.mean(0); e = 1.96 * R.std(0) / np.sqrt(len(R))
            data[f"{lab}/{shift}"] = list(m)
            lines.append(f"  занижение {shift:.0f}: очки {m[0]:.3f} ± {e[0]:.3f}, место {m[1]:.1f} ± {e[1]:.1f}, ср. настоящая сила соперников {m[2]:.0f}")
    lines.append("\n(одни и те же зёрна случайности для всех вариантов занижения: состав участников совпадает, отличается только рейтинг одного игрока)")
    save("h009", "\n".join(lines) + "\n", data)
