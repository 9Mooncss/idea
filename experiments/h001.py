"""H001/H002/H004 share this: tiebreak concordance in several settings."""
from common import sim, cfg, save
from tie_common import tie_run, aggregate

SETTINGS = {
    "базовая (100 игроков, 9 туров, шум 75, ничьи 35%)": cfg(),
    "малый (40 игроков, 7 туров)": cfg(N=40, rounds=7),
    "большой (200 игроков, 9 туров)": cfg(N=200, rounds=9),
    "точные рейтинги (шум 30)": cfg(sd_noise=30.0),
    "шумные рейтинги (шум 150)": cfg(sd_noise=150.0),
    "мало ничьих (15%)": cfg(dmax=0.15),
    "много ничьих (55%)": cfg(dmax=0.55),
}
if __name__ == "__main__":
    lines, data = ["Согласие тай-брейка с настоящей силой среди пар с равными очками, % (±95% ДИ)\n"], {}
    for name, c in SETTINGS.items():
        agg = aggregate(sim(tie_run, c, 2000, seed=1, with_bt=True))
        data[name] = agg
        lines.append(f"\n## {name}")
        for k, (m, e) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
            lines.append(f"  {k:12s} {100*m:5.1f} ± {100*e:.1f}")
    save("h001", "\n".join(lines) + "\n", data)
