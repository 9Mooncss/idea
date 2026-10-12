"""H002: own rating vs best result-based tiebreak as rating noise grows."""
from common import sim, cfg, save
from tie_common import tie_run, aggregate

RESULT_BASED = ["BH", "BH-C1", "BH-Med", "SB", "ARO", "Progressive", "Wins", "BT-flat"]
if __name__ == "__main__":
    lines = ["Шум рейтинга | Собств. рейтинг | лучший турнирный тай-брейк\n"]; data = {}
    for noise in [0, 50, 100, 150, 200, 250, 300, 400]:
        agg = aggregate(sim(tie_run, cfg(sd_noise=float(noise)), 1500, seed=2, with_bt=True))
        best = max(RESULT_BASED, key=lambda k: agg[k][0])
        data[noise] = agg
        lines.append(f"{noise:4d} | {100*agg['OwnRating'][0]:5.1f} ± {100*agg['OwnRating'][1]:.1f} | "
                     f"{best} {100*agg[best][0]:5.1f} ± {100*agg[best][1]:.1f}")
    save("h002", "\n".join(lines) + "\n", data)
