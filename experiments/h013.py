"""H013: does UCL criterion 6 (opponents' points) favour the weaker team?"""
import numpy as np
import ucl_sim as u
from common import save

SD, MU = 0.5, np.log(1.5)
if __name__ == "__main__":
    rng = np.random.default_rng(31)
    seasons = 0; agree = {k: [0.0, 0] for k in ["opp_pts", "opp_gd", "gd", "w"]}
    per_season = []
    agree15 = [0.0, 0]; decide = {"8/9": [0, 0], "24/25": [0, 0]}
    opp_by_pot = np.zeros(4); cnt_pot = np.zeros(4)
    for d in range(400):
        s, coef = u.make_teams(rng, sd=SD); g, pots = u.draw(rng, coef)
        Opp = np.zeros((36, 36))
        for h, a in g:
            Opp[h, a] = Opp[a, h] = 1
        for k in range(4):
            opp_by_pot[k] += (Opp[pots[k]] @ s).mean() / 8; cnt_pot[k] += 1
        gh, ga = u.simulate(g, s, 250, rng, mu=MU)
        order, st = u.table(g, gh, ga, coef, rng)
        seasons += gh.shape[0]
        i, j = np.triu_indices(36, 1)
        tied = st["pts"][:, i] == st["pts"][:, j]
        sgn = np.sign(s[i] - s[j])[None, :]
        a_s, n_s = 0.0, 0
        for k in agree:
            dd = np.sign(st[k][:, i] - st[k][:, j])
            ag = np.where(dd == 0, 0.5, (dd == sgn) * 1.0)
            agree[k][0] += ag[tied].sum(); agree[k][1] += tied.sum()
            if k == "opp_pts":
                a_s, n_s = ag[tied].sum(), tied.sum()
        per_season.append((a_s, n_s))
        same15 = tied.copy()
        for k in ["gd", "gf", "agf", "w", "awin"]:
            same15 &= st[k][:, i] == st[k][:, j]
        dd = np.sign(st["opp_pts"][:, i] - st["opp_pts"][:, j])
        ag = np.where(dd == 0, 0.5, (dd == sgn) * 1.0)
        agree15[0] += ag[same15].sum(); agree15[1] += same15.sum()
        # how often criteria 6+ decide the 8/9 and 24/25 boundaries
        for lab, pos in [("8/9", 7), ("24/25", 23)]:
            x, y = order[:, pos], order[:, pos + 1]
            eq = np.ones(gh.shape[0], bool)
            for k in ["pts", "gd", "gf", "agf", "w", "awin"]:
                eq &= np.take_along_axis(st[k], x[:, None], 1)[:, 0] == np.take_along_axis(st[k], y[:, None], 1)[:, 0]
            decide[lab][0] += eq.sum(); decide[lab][1] += len(eq)
    a = np.array(per_season)
    m = a[:, 0].sum() / a[:, 1].sum()
    res = a[:, 0] - m * a[:, 1]
    se = np.sqrt(len(a) / (len(a) - 1) * (res ** 2).sum()) / a[:, 1].sum()
    lines = [f"Симулировано сезонов: {seasons} (400 жеребьёвок × 250), матчей: {seasons*144}",
             "Средняя настоящая сила соперников по корзинам (1 = сильнейшая):",
             "  " + ", ".join(f"корзина {k+1}: {opp_by_pot[k]/cnt_pot[k]:+.4f}" for k in range(4)),
             "\nСогласие с настоящей силой среди пар с равными очками:"]
    for k, (x, n) in agree.items():
        lines.append(f"  {k:8s} {100*x/n:5.1f}%  (пар {n})")
    lines.append(f"  95% ДИ для opp_pts (по жеребьёвкам): {100*(m-1.96*se):.1f} – {100*(m+1.96*se):.1f}")
    lines.append(f"\nСреди пар, равных по критериям 1–5 (где критерий 6 реально решает): {100*agree15[0]/max(agree15[1],1):.1f}% (пар {agree15[1]})")
    for lab, (x, n) in decide.items():
        lines.append(f"Граница {lab}: решается критериями 6+ в {100*x/n:.2f}% сезонов")
    save("h013", "\n".join(lines) + "\n", dict(agree=agree, agree15=agree15, decide=decide))
