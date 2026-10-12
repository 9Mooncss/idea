"""H018: hiring with harsh/lenient interviewers: pairwise comparisons vs average score."""
import numpy as np
from common import sim, save
from swiss import bt_fit

M, TOP = 100, 10      # candidates, number to hire


def run(c, rng):
    q = rng.normal(0, 1, M); best = set(np.argsort(-q)[:TOP])
    nr = 15; bias = rng.normal(0, c["bias"], nr)
    budget = c["views"] * M                         # interviewer-time units
    # absolute: each rating = 1 time unit, rating = q + rater bias + noise, rounded 1..5
    k = budget // M
    sc = np.zeros(M)
    for i in range(M):
        r = rng.integers(0, nr, k)
        sc[i] = np.mean(np.clip(np.round(3 + (q[i] + bias[r] + rng.normal(0, c["noise"], k))), 1, 5))
    hire_abs = set(np.argsort(-(sc + 1e-6 * rng.random(M)))[:TOP])
    # pairwise: each comparison costs 1.5 units; rater bias cancels
    ncmp = int(budget / 1.5)
    a = rng.integers(0, M, ncmp); b = (a + rng.integers(1, M, ncmp)) % M
    win = (q[a] + rng.normal(0, c["noise"], ncmp)) > (q[b] + rng.normal(0, c["noise"], ncmp))
    games = [(int(x), int(y), 1.0 if w else 0.0) for x, y, w in zip(a, b, win)]
    est = bt_fit(M, games, prior_mean=np.zeros(M), prior_sd=400.0)
    hire_pw = set(np.argsort(-est)[:TOP])
    return len(hire_abs & best) / TOP, len(hire_pw & best) / TOP


if __name__ == "__main__":
    lines, data = ["Доля верно нанятых из настоящего топ-10 (из 100 кандидатов), при равном времени интервьюеров\n"], {}
    for views in [3, 6]:
        for bias in [0.0, 0.5, 1.0]:
            R = np.array(sim(run, dict(views=views, bias=bias, noise=0.7), 400, seed=61))
            m = R.mean(0); e = 1.96 * R.std(0) / np.sqrt(len(R))
            data[f"{views}/{bias}"] = list(m)
            lines.append(f"  оценок на кандидата {views}, разброс строгости {bias}: баллы {m[0]:.3f} ± {e[0]:.3f} | попарно {m[1]:.3f} ± {e[1]:.3f}")
    save("h018", "\n".join(lines) + "\n", data)
