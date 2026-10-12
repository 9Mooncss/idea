"""H015: prefix-cache eviction for LLM serving: LRU vs LFU vs GDSF (cost-aware).

Each request reuses one shared prefix (system prompt / document). A hit saves the
prefill cost of that prefix: cost(L) = L + L^2 / L0 (L0=inf -> linear).
Memory used = L tokens. Popularity ~ Zipf; optional popularity churn.
"""
import heapq, itertools
from collections import OrderedDict
import numpy as np
from common import save


def workload(rng, P=3000, n=150000, alpha=1.0, churn=False):
    L = np.clip(rng.lognormal(np.log(1000), 1.3, P), 50, 120000).astype(int)
    w = 1.0 / np.arange(1, P + 1) ** alpha
    perm = rng.permutation(P)
    reqs = []
    chunk = 5000
    for start in range(0, n, chunk):
        if churn and start:
            k = int(0.15 * P); idx = rng.choice(P, k, replace=False)
            perm[idx] = perm[rng.permutation(idx)]       # shuffle popularity of 15% items
        p = np.empty(P); p[perm] = w; p /= p.sum()
        reqs.append(rng.choice(P, chunk, p=p))
    return L, np.concatenate(reqs)


def run_policy(policy, L, reqs, C, cost):
    used = 0; saved = 0.0; total = 0.0
    if policy == "LRU":
        cache = OrderedDict()
        for k in reqs:
            c = cost[k]; total += c
            if k in cache:
                saved += c; cache.move_to_end(k); continue
            if L[k] > C:
                continue
            while used + L[k] > C:
                old, _ = cache.popitem(last=False); used -= L[old]
            cache[k] = 1; used += L[k]
        return saved / total
    # LFU and GDSF via heap with lazy deletion
    freq = {}; prio = {}; heap = []; clock = 0.0; cnt = itertools.count()
    for k in reqs:
        c = cost[k]; total += c
        f = freq.get(k, 0) + 1; freq[k] = f
        if k in prio:
            saved += c
        elif L[k] > C:
            continue
        else:
            while used + L[k] > C:
                pv, _, old = heapq.heappop(heap)
                if prio.get(old) != pv:
                    continue
                del prio[old]; used -= L[old]
                if policy == "GDSF":
                    clock = pv
            used += L[k]
        pv = f if policy == "LFU" else clock + f * c / L[k]
        prio[k] = pv; heapq.heappush(heap, (pv, next(cnt), k))
    return saved / total


if __name__ == "__main__":
    lines, data = ["Доля сэкономленных вычислений префилла (чем больше, тем лучше)\n"], {}
    for churn in [False, True]:
        for L0 in [np.inf, 8000.0]:
            for frac in [0.02, 0.05, 0.15]:
                res = {p: [] for p in ["LRU", "LFU", "GDSF"]}
                for rep in range(3):
                    rng = np.random.default_rng(100 + rep)
                    L, reqs = workload(rng, churn=churn)
                    cost = L + (L.astype(float) ** 2 / L0 if np.isfinite(L0) else 0)
                    C = int(frac * L.sum())
                    for p in res:
                        res[p].append(run_policy(p, L, reqs, C, cost))
                m = {p: np.mean(v) for p, v in res.items()}
                key = f"{'смена популярности' if churn else 'стабильная популярность'}, {'линейная' if not np.isfinite(L0) else 'квадратичная (L0=8000)'} стоимость, кэш {int(frac*100)}% всех префиксов"
                data[key] = m
                gain = 100 * (m["GDSF"] / m["LRU"] - 1)
                lines.append(f"{key}:\n   LRU {m['LRU']:.3f}  LFU {m['LFU']:.3f}  GDSF {m['GDSF']:.3f}   → GDSF против LRU: {gain:+.1f}% (относительно)")
                print(lines[-1], flush=True)
    save("h015", "\n".join(lines) + "\n", data)
