"""Follow-up to H001/H004: in homogeneous fields (rating-restricted sections),
are Buchholz-type tiebreaks worse than a coin flip under Dutch pairing?"""
from common import sim, cfg, save
from tie_common import tie_run, aggregate

KEYS = ["BH-C1", "BH", "SB", "ARO", "Progressive", "OwnRating", "Random"]
if __name__ == "__main__":
    lines = ["разброс силы / шум рейтинга / жеребьёвка : согласие тай-брейков, %\n"]; data = {}
    for sd in [50.0, 100.0, 150.0]:
        for noise in [20.0, 50.0, 75.0]:
            for rule in ["dutch", "random"]:
                if rule == "random" and noise != 50.0:
                    continue
                agg = aggregate(sim(tie_run, cfg(sd_true=sd, sd_noise=noise, rule=rule), 1500, seed=11))
                data[f"{sd}/{noise}/{rule}"] = agg
                lines.append(f"sd={sd:.0f} шум={noise:.0f} {rule:6s}: " +
                             "  ".join(f"{k} {100*agg[k][0]:.1f}±{100*agg[k][1]:.1f}" for k in KEYS))
    save("extra_homogeneous", "\n".join(lines) + "\n", data)
