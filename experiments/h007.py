"""H007: 3-1-0 scoring vs 1-1/2-0."""
import numpy as np
from scipy.stats import kendalltau
from common import sim, cfg, save, ci, swiss
from h005 import run

if __name__ == "__main__":
    lines, data = ["доля ничьих между равными | очки | тау | доля правильного победителя\n"], {}
    for d in [0.15, 0.35, 0.55]:
        for lab, (w, dr) in {"1-½-0": (1.0, 0.5), "3-1-0": (3.0, 1.0)}.items():
            R = sim(run, cfg(dmax=d, win_pts=w, draw_pts=dr), 4000, seed=7)
            t = np.array([r[0] for r in R]); p = np.mean([r[1] for r in R])
            data[f"{d}/{lab}"] = (t.mean(), p)
            lines.append(f"  {d:.2f} | {lab} | {t.mean():.4f} ± {1.96*t.std()/np.sqrt(len(t)):.4f} | {p:.3f} ± {ci(p,len(R)):.3f}")
    save("h007", "\n".join(lines) + "\n", data)
