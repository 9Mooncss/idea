"""Re-check of the batch-1 'homogeneous field' finding.

1. Confidence intervals + number of tournaments / games / tied pairs for every
   group of extra_homogeneous.py (same seeds -> same tournaments).
2. Same scenarios paired by the official bbpPairings engine (Dutch, FIDE).
   Both engines use identical player sets (same seed per tournament).
3. bbpPairings checker (-c) run on tournaments paired by our simplified engine
   to count where our pairing deviates from the FIDE Dutch rules.
"""
import os, subprocess, tempfile, sys
import numpy as np
from common import sim, cfg, save, swiss
import bbp_sim

KEYS = ["BH-C1", "BH", "SB", "ARO", "Progressive", "OwnRating", "Random"]


def one(c, rng, engine="ours"):
    s, r = swiss.make_players(c, rng)
    if engine == "ours":
        T = swiss.run_tournament(c, rng, s=s, r=r)
    else:
        T = bbp_sim.run_tournament_bbp(c, rng, s=s, r=r)
    tb = swiss.tiebreaks(T, with_bt=False)
    conc = swiss.tie_concordance(T, tb)
    rem = sum(len(o) - len(set(o)) for o in T["opps"]) // 2
    return conc, len(T["games"]), rem


def summarize(R):
    out = {"tournaments": len(R), "games": int(sum(x[1] for x in R)),
           "rematches": int(sum(x[2] for x in R)),
           "tied_pairs": int(sum(x[0]["BH"][1] for x in R))}
    for k in KEYS:
        a = np.array([x[0][k][0] for x in R]); n = np.array([x[0][k][1] for x in R])
        m = a.sum() / n.sum()
        # cluster (per-tournament) standard error of a ratio estimator
        res = a - m * n
        se = np.sqrt(len(R) / (len(R) - 1) * np.sum(res ** 2)) / n.sum()
        out[k] = (m, m - 1.96 * se, m + 1.96 * se)
    return out


def fmt(label, o):
    s = (f"{label}\n    турниров {o['tournaments']}, партий {o['games']}, пар с равными очками {o['tied_pairs']}, "
         f"повторных встреч {o['rematches']}\n")
    for k in KEYS:
        m, lo, hi = o[k]
        if k == "Random":
            verdict = "(контроль)"
        elif lo <= 0.5 <= hi:
            verdict = "интервал включает 50% → эффекта не доказано"
        elif hi < 0.5:
            verdict = "ХУЖЕ монетки (доказано)"
        else:
            verdict = "лучше монетки"
        s += f"    {k:12s} {100*m:5.1f}%  [95% ДИ {100*lo:5.1f} – {100*hi:5.1f}]  {verdict}\n"
    return s


def trf_from_ours(T, path):
    """Write our simplified-engine tournament as TRF (players renumbered by rating)."""
    N = len(T["s"]); half = N // 2
    order = np.argsort(-T["r"], kind="stable"); pid = np.empty(N, int); pid[order] = np.arange(1, N + 1)
    hist = [[] for _ in range(N)]
    for gi, (w, b, x) in enumerate(T["games"]):
        hist[pid[w] - 1].append((pid[b], "w", {1.0: "1", .5: "=", 0.0: "0"}[x]))
        hist[pid[b] - 1].append((pid[w], "b", {1.0: "0", .5: "=", 0.0: "1"}[x]))
    sc = np.zeros(N); sc[pid - 1] = T["score"]
    bbp_sim.write_trf(path, N, np.sort(T["r"])[::-1], hist, sc, T["cfg"]["rounds"])


def check_ours(c, rng):
    T = swiss.run_tournament(c, rng)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "t.trf"); trf_from_ours(T, p)
        r = subprocess.run([bbp_sim.BBP, "--dutch", p, "-c"], capture_output=True, text=True)
        txt = (r.stdout + r.stderr).strip()
    return txt


if __name__ == "__main__":
    n_bbp = int(os.environ.get("N_BBP", "1000"))
    lines, data = [], {}
    lines.append("# Часть 1. Наша упрощённая жеребьёвка, все группы из extra_homogeneous (1500 турниров в каждой)\n")
    for sd in [50.0, 100.0, 150.0]:
        for noise in [20.0, 50.0, 75.0]:
            for rule in ["dutch", "random"]:
                if rule == "random" and noise != 50.0:
                    continue
                o = summarize(sim(one, cfg(sd_true=sd, sd_noise=noise, rule=rule), 1500, seed=11))
                data[f"ours/{sd}/{noise}/{rule}"] = o
                lines.append(fmt(f"  разброс силы {sd:.0f}, шум рейтинга {noise:.0f}, жеребьёвка {rule}", o))
                print(lines[-1], flush=True)
    lines.append(f"\n# Часть 2. Те же сценарии: наша жеребьёвка против официальной bbpPairings (Dutch), "
                 f"{n_bbp} одинаковых турниров\n")
    for sd, noise in [(100.0, 75.0), (50.0, 20.0)]:
        for eng in ["ours", "bbp"]:
            o = summarize(sim(one, cfg(sd_true=sd, sd_noise=noise), n_bbp, seed=12, engine=eng))
            data[f"cmp/{sd}/{noise}/{eng}"] = o
            lines.append(fmt(f"  разброс силы {sd:.0f}, шум {noise:.0f}, движок: "
                             f"{'наш упрощённый' if eng == 'ours' else 'bbpPairings'}", o))
            print(lines[-1], flush=True)
    lines.append("\n# Часть 3. Проверка нашей жеребьёвки программой bbpPairings (-c), 20 турниров (разброс 100, шум 75)\n")
    rngs = np.random.SeedSequence(13).spawn(20)
    bad = 0; examples = []
    for sq in rngs:
        txt = check_ours(cfg(sd_true=100.0, sd_noise=75.0), np.random.default_rng(sq))
        if txt:
            bad += 1; examples.append(txt)
    lines.append(f"  турниров с замечаниями: {bad} из 20")
    for e in examples[:2]:
        lines.append("  пример вывода проверки:\n    " + "\n    ".join(e.splitlines()[:12]))
    data["check"] = {"bad": bad, "examples": examples[:5]}
    save("verify_finding", "\n".join(lines) + "\n", data)
