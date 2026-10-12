"""H006: rounds needed for P(true best wins) >= 50% vs log2(N)."""
import numpy as np
from common import sim, cfg, save, swiss


def run(c, rng):
    T = swiss.run_tournament(c, rng)
    tb = swiss.tiebreaks(T, with_bt=False)
    return swiss.final_order(T, tb["BH-C1"])[0] == int(np.argmax(T["s"]))


def p_win(N, R, sd):
    return float(np.mean(sim(run, cfg(N=N, rounds=R, sd_true=sd), 600 if N < 256 else 300, seed=6)))


if __name__ == "__main__":
    lines, data = [], {}
    for sd in [250.0, 150.0]:
        lines.append(f"\n## разброс силы sd={sd}")
        Ns, Rs = [], []
        for N in [16, 32, 64, 128, 256, 512]:
            R = max(3, int(np.log2(N)) - 2); row = {}
            row[R] = p_win(N, R, sd)
            if row[R] >= 0.5:          # go down until below 0.5
                while R > 2 and row[R] >= 0.5:
                    R -= 1; row[R] = p_win(N, R, sd)
                need = R + 1
            else:                      # go up until >= 0.5
                while row[R] < 0.5 and R < N - 1 and R < 25:
                    R += 1; row[R] = p_win(N, R, sd)
                need = R if row[R] >= 0.5 else None
            row = sorted(row.items())
            print(sd, N, need, row, flush=True)
            data[f"{sd}/{N}"] = row
            lines.append(f"  N={N:4d}: туров нужно {need}   (туры, доля: {[(r, round(p, 3)) for r, p in row]})")
            if need:
                Ns.append(np.log2(N)); Rs.append(need)
        slope = np.polyfit(Ns, Rs, 1)[0] if len(Ns) > 2 else float('nan')
        data[f"{sd}/slope"] = slope
        lines.append(f"  наклон по log2(N): {slope:.2f}")
    save("h006", "\n".join(lines) + "\n", data)
