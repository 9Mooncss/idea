"""H014: do deep UCL tie-breakers (criteria 6-10) move top-8/top-24 probabilities?"""
import numpy as np
import ucl_sim as u
from common import save

SD, MU = 0.5, np.log(1.5)
if __name__ == "__main__":
    rng = np.random.default_rng(41)
    lines = ["Для каждой жеребьёвки: 20000 сезонов на одних и тех же исходах матчей, три способа упорядочить таблицу.",
             "Показана максимальная по 36 командам разница вероятностей (п.п.).\n"]
    mx = {"5": [], "2": []}
    for d in range(20):
        s, coef = u.make_teams(rng, sd=SD); g, pots = u.draw(rng, coef)
        gh, ga = u.simulate(g, s, 20000, rng, mu=MU)
        prob = {}
        for depth in [10, 5, 2]:
            order, _ = u.table(g, gh, ga, coef, np.random.default_rng(d), depth=depth)
            rank = np.argsort(order, axis=1)
            prob[depth] = ((rank < 8).mean(0), (rank < 24).mean(0))
        row = []
        for dep in [5, 2]:
            d8 = np.abs(prob[10][0] - prob[dep][0]).max(); d24 = np.abs(prob[10][1] - prob[dep][1]).max()
            mx[str(dep)].append((d8, d24)); row.append(f"без критериев {dep+1}–10: топ-8 {100*d8:.2f}, топ-24 {100*d24:.2f}")
        lines.append(f"жеребьёвка {d+1:2d}: " + " | ".join(row))
    se = 100 * np.sqrt(0.25 / 20000)
    for dep in ["5", "2"]:
        a = np.array(mx[dep])
        lines.append(f"\nСредний максимум разницы без критериев {int(dep)+1}–10: топ-8 {100*a[:,0].mean():.2f} п.п., топ-24 {100*a[:,1].mean():.2f} п.п.")
    lines.append(f"(для сравнения: случайная погрешность одной вероятности ≈ {se:.2f} п.п.; одни и те же исходы матчей сравниваются напрямую, поэтому разница почти не зашумлена)")
    save("h014", "\n".join(lines) + "\n", {k: v for k, v in mx.items()})
