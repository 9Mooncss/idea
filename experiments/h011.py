"""H011 + H012: tournament schemes for best-of-N selection with an LLM judge."""
import numpy as np
from common import sim, save
import judge_sim

if __name__ == "__main__":
    lines, data = [], {}
    for N in [16, 64]:
        for sigma in [0.5, 1.0]:
            for drift in [0.0, 0.7]:
                c = dict(N=N, sigma=sigma, bias=0.3, drift=drift)
                R = sim(judge_sim.run, c, 3000, seed=21)
                lines.append(f"\n## N={N}, шум судьи {sigma}, дрейф шкалы {drift}, позиционная предвзятость 0.3")
                lines.append("  схема | вызовов судьи (среднее) | доля верного выбора ±95% | средняя потеря качества")
                rows = []
                for k in R[0]:
                    ok = np.mean([r[k][0] for r in R]); reg = np.mean([r[k][1] for r in R]); calls = np.mean([r[k][2] for r in R])
                    rows.append((calls, k, ok, reg))
                for calls, k, ok, reg in sorted(rows):
                    e = 1.96 * np.sqrt(ok * (1 - ok) / len(R))
                    lines.append(f"  {k:32s} {calls:7.1f} | {ok:.3f} ± {e:.3f} | {reg:.3f}")
                data[f"{N}/{sigma}/{drift}"] = rows
    save("h011", "\n".join(lines) + "\n", data)
