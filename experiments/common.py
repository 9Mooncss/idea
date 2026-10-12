import json, os, sys
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import swiss

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RES, exist_ok=True)


def _worker(args):
    fn, cfg, seeds, extra = args
    out = []
    for sd in seeds:
        out.append(fn(cfg, np.random.default_rng(sd), **extra))
    return out


def sim(fn, cfg, n, seed=0, **extra):
    """Run fn(cfg, rng) n times in parallel, return list of results."""
    seeds = np.random.SeedSequence(seed).spawn(n)
    chunks = [seeds[i::4] for i in range(4)]
    with Pool(4) as p:
        parts = p.map(_worker, [(fn, cfg, c, extra) for c in chunks])
    return [x for part in parts for x in part]


def cfg(**kw):
    c = dict(swiss.DEFAULT); c.update(kw); return c


def ci(p, n):
    return 1.96 * np.sqrt(max(p * (1 - p), 1e-9) / n)


def save(name, text, data):
    open(os.path.join(RES, name + ".txt"), "w").write(text)
    json.dump(data, open(os.path.join(RES, name + ".json"), "w"), indent=1, default=float)
    print(text)
