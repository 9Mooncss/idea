"""Same tournament model as swiss.py, but pairings come from the official
FIDE-endorsed engine bbpPairings (Dutch system), called once per round.

Set BBP=/path/to/bbpPairings.exe or install it with install_bbp.sh.
"""
import os, subprocess, tempfile
import numpy as np
import swiss

BBP = os.environ.get("BBP", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "bbp", "bbpPairings.exe"))


def write_trf(path, N, rating, hist, score, rounds_total):
    """hist[p] = list of (opp_id, colour 'w'/'b', result '1'/'='/'0'); ids 1..N."""
    lines = ["012 sim", f"XXR {rounds_total}", "XXC white1"]
    for p in range(N):
        pid = p + 1
        line = f"001 {pid:4d}      {'P%04d' % pid:<33s}{int(round(rating[p])):4d}"
        line = line.ljust(80) + f"{score[p]:4.1f}" + f" {pid:4d}"
        line = line.ljust(91)
        for opp, col, res in hist[p]:
            line += f"{opp:4d} {col} {res}  "
        lines.append(line.rstrip())
    open(path, "w").write("\n".join(lines) + "\n")


def bbp_pair(trf, out):
    r = subprocess.run([BBP, "--dutch", trf, "-p", out], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"bbpPairings rc={r.returncode}: {r.stderr[:300]}")
    rows = open(out).read().split()
    n = int(rows[0]); vals = list(map(int, rows[1:1 + 2 * n]))
    return [(vals[2 * i] - 1, vals[2 * i + 1] - 1) for i in range(n)]   # (white, black), 0-based


def run_tournament_bbp(cfg, rng, s=None, r=None):
    N = cfg["N"]
    if s is None:
        s, r = swiss.make_players(cfg, rng)
    # pairing numbers = order by rating (descending), as in a real event
    order = np.argsort(-r, kind="stable")
    s, r = s[order], r[order]
    score = np.zeros(N); chess = np.zeros(N)
    hist = [[] for _ in range(N)]
    opps = [[] for _ in range(N)]; res = [[] for _ in range(N)]; cum = [[] for _ in range(N)]
    whites = np.zeros(N, int); games = []
    with tempfile.TemporaryDirectory() as td:
        trf, out = os.path.join(td, "t.trf"), os.path.join(td, "p.txt")
        for rd in range(cfg["rounds"]):
            write_trf(trf, N, r, hist, score, cfg["rounds"])
            pairs = bbp_pair(trf, out)
            for w, b in pairs:
                if b < 0:
                    raise RuntimeError("bye not supported (use even N)")
                x = swiss.play(s[w], s[b], cfg, rng)
                whites[w] += 1
                for p, q, v, col in ((w, b, x, "w"), (b, w, 1 - x, "b")):
                    hist[p].append((q + 1, col, {1.0: "1", 0.5: "=", 0.0: "0"}[v]))
                    opps[p].append(q); res[p].append(v); chess[p] += v; score[p] += v
                games.append((w, b, x))
            for p in range(N):
                cum[p].append(score[p])
    return dict(s=s, r=r, score=score, chess=chess, opps=opps, res=res, cum=cum,
                whites=whites, games=games, cfg=cfg)
