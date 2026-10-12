"""H019 robustness: does Valve's Buchholz rule help when the seeding is poor?"""
import numpy as np
from common import sim, save
from h019 import run

if __name__ == "__main__":
    lines, data = ["Разница в числе настоящих топ-8 среди прошедших (Valve − только посев), разброс силы 150\n"], {}
    for noise in [0, 75, 150, 300, 100000]:
        R = sim(run, dict(sd=150, seed_noise=noise), 20000, seed=72)
        d = np.array([r["Valve (Бухгольц)"][0] - r["только посев"][0] for r in R])
        v = np.mean([r["Valve (Бухгольц)"][0] for r in R])
        lab = "случайный посев" if noise > 10000 else f"ошибка посева {noise}"
        data[lab] = (d.mean(), v)
        lines.append(f"  {lab:20s} Valve {v:.3f}; разница {d.mean():+.3f} ± {1.96*d.std()/np.sqrt(len(d)):.3f}")
        print(lines[-1], flush=True)
    save("h019_robust", "\n".join(lines) + "\n", data)
