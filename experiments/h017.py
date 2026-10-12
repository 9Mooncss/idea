"""H017: finding the best ad creative: Swiss duels vs uniform, successive halving, Thompson."""
import numpy as np
from common import sim, save
from swiss import pair_round

K = 16


def uniform(ctr, B, rng):
    n = B // K; c = rng.binomial(n, ctr)
    return int(np.argmax(c + rng.random(K) * 1e-3))


def halving(ctr, B, rng):
    alive = list(range(K)); R = int(np.log2(K)); clicks = np.zeros(K); imps = np.zeros(K)
    for r in range(R):
        n = B // R // len(alive)
        for a in alive:
            clicks[a] += rng.binomial(n, ctr[a]); imps[a] += n
        rate = clicks / np.maximum(imps, 1)
        alive = sorted(alive, key=lambda a: -rate[a])[:len(alive) // 2]
    return alive[0]


def thompson(ctr, B, rng, batch=500):
    a = np.ones(K); b = np.ones(K)
    for _ in range(B // batch):
        draws = rng.beta(a[:, None], b[:, None], (K, 20)).argmax(0)    # 20 sub-batches
        cnt = np.bincount(draws, minlength=K) * (batch // 20)
        c = rng.binomial(cnt, ctr); a += c; b += cnt - c
    return int(np.argmax(a / (a + b)))


def swiss_duels(ctr, B, rng):
    R = int(np.log2(K)) + 1; n = B // (R * K)
    score = np.zeros(K); played = [set() for _ in range(K)]
    clicks = np.zeros(K); imps = np.zeros(K); tie = rng.random(K)
    for r in range(R):
        order = sorted(range(K), key=lambda i: (-score[i], tie[i]))
        for x, y in pair_round(order, score, played, "dutch", rng):
            cx, cy = rng.binomial(n, ctr[x]), rng.binomial(n, ctr[y])
            clicks[[x, y]] += [cx, cy]; imps[[x, y]] += n
            score[x if cx > cy else y if cy > cx else (x if rng.random() < .5 else y)] += 1
            played[x].add(y); played[y].add(x)
    rate = clicks / np.maximum(imps, 1)
    return max(range(K), key=lambda i: (score[i], rate[i]))


def run(c, rng):
    ctr = 0.02 * np.exp(rng.normal(0, 0.1, K)); best = int(np.argmax(ctr))
    out = {}
    for name, f in [("равномерно", uniform), ("деление пополам", halving),
                    ("Thompson sampling", thompson), ("швейцарка дуэлей", swiss_duels)]:
        ch = f(ctr, c["B"], rng)
        out[name] = (ch == best, (ctr[best] - ctr[ch]) / ctr[best])
    return out


if __name__ == "__main__":
    lines, data = ["16 креативов, CTR ≈ 2% с разбросом ~10%. Доля верного выбора лучшего | средняя потеря CTR (отн.)\n"], {}
    for B in [100_000, 400_000, 1_600_000]:
        R = sim(run, dict(B=B), 2000, seed=51)
        lines.append(f"## бюджет {B:,} показов".replace(",", " "))
        for k in R[0]:
            ok = np.mean([r[k][0] for r in R]); reg = np.mean([r[k][1] for r in R])
            data[f"{B}/{k}"] = (ok, reg)
            lines.append(f"  {k:20s} {ok:.3f} ± {1.96*np.sqrt(ok*(1-ok)/len(R)):.3f} | {100*reg:.2f}%")
    save("h017", "\n".join(lines) + "\n", data)
