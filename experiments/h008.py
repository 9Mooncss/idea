"""H008: extra white and P(1st) for top-5 players."""
import numpy as np
from common import sim, cfg, save, ci, swiss


def run(c, rng):
    T = swiss.run_tournament(c, rng)
    tb = swiss.tiebreaks(T, with_bt=False)
    win = swiss.final_order(T, tb["BH-C1"])[0]
    top = np.argsort(-T["s"])[:5]
    return [(int(T["whites"][p]), p == win, float(T["chess"][p])) for p in top]


if __name__ == "__main__":
    lines, data = [], {}
    for wa in [30.0, 0.0]:
        R = [x for r in sim(run, cfg(white_adv=wa), 8000, seed=8) for x in r]
        lines.append(f"\n## преимущество белых {wa} Эло")
        for w in [4, 5]:
            sel = [x for x in R if x[0] == w]
            p = np.mean([x[1] for x in sel]); sc = np.mean([x[2] for x in sel])
            data[f"{wa}/{w}"] = (len(sel), p, sc)
            lines.append(f"  белых={w}: n={len(sel)}, P(1-е место)={p:.4f} ± {ci(p,len(sel)):.4f}, средние очки={sc:.3f}")
        lines.append(f"  прочие числа белых: n={sum(1 for x in R if x[0] not in (4,5))}")
    save("h008", "\n".join(lines) + "\n", data)
