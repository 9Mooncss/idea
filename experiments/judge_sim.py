"""H011/H012: choosing the best of N LLM answers with a noisy judge model.

Latent quality q ~ N(0,1). One pairwise call (i shown first, j second): the judge
perceives q+e (e ~ N(0, sigma) independently per answer per call) and prefers the
first answer by a position bias b. One pointwise call: score 1..10 =
round(5.5 + 1.5*(q + e + c)), c ~ N(0, drift) is a per-call scale shift.
"""
import numpy as np
from swiss import pair_round


class Judge:
    def __init__(self, q, sigma, bias, drift, rng):
        self.q, self.s, self.b, self.d, self.rng = q, sigma, bias, drift, rng
        self.calls = 0

    def pair(self, i, j):                      # True if i (shown first) wins
        self.calls += 1
        e = self.rng.normal(0, self.s, 2)
        return self.q[i] + e[0] + self.b > self.q[j] + e[1]

    def point(self, i):
        self.calls += 1
        v = self.q[i] + self.rng.normal(0, self.s) + self.rng.normal(0, self.d)
        return float(np.clip(np.round(5.5 + 1.5 * v), 1, 10))


def match(J, i, j, mode):
    if mode == "1":                            # one call, random order
        return (i if J.pair(i, j) else j) if J.rng.random() < .5 else (j if J.pair(j, i) else i)
    if mode == "2":                            # both orders, third call if split
        a = J.pair(i, j); b = not J.pair(j, i)
        if a == b:
            return i if a else j
        return match(J, i, j, "1")
    if mode == "bo3":
        w = {i: 0, j: 0}; k = 0
        while max(w.values()) < 2:
            x, y = (i, j) if k % 2 == 0 else (j, i)
            win = x if J.pair(x, y) else y; w[win] += 1; k += 1
        return max(w, key=w.get)


def knockout(J, cand, mode):
    cand = list(cand); J.rng.shuffle(cand)
    while len(cand) > 1:
        nxt = [match(J, cand[k], cand[k + 1], mode) for k in range(0, len(cand) - 1, 2)]
        if len(cand) % 2:
            nxt.append(cand[-1])
        cand = nxt
    return cand[0]


def swiss_select(J, N, rounds):
    score = np.zeros(N); played = [set() for _ in range(N)]; first = np.zeros(N)
    opps = [[] for _ in range(N)]
    tie = J.rng.random(N)
    for _ in range(rounds):
        order = sorted(range(N), key=lambda i: (-score[i], tie[i]))
        for a, b in pair_round(order, score, played, "dutch", J.rng):
            if first[a] > first[b]:            # balance who is shown first
                a, b = b, a
            first[a] += 1
            w = a if J.pair(a, b) else b
            score[w] += 1
            played[a].add(b); played[b].add(a); opps[a].append(b); opps[b].append(a)
    bh = np.array([sum(score[o] for o in opps[i]) for i in range(N)])
    return max(range(N), key=lambda i: (score[i], bh[i], tie[i]))


def round_robin(J, N):
    w = np.zeros(N)
    for i in range(N):
        for j in range(i + 1, N):
            win = match(J, i, j, "1"); w[win] += 1
    return int(np.argmax(w + 1e-6 * J.rng.random(N)))


def pointwise(J, N, k):
    m = np.array([np.mean([J.point(i) for _ in range(k)]) for i in range(N)])
    return int(np.argmax(m + 1e-6 * J.rng.random(N)))


def hybrid(J, N, m, mode="2"):
    s = np.array([J.point(i) for i in range(N)]) + 1e-6 * J.rng.random(N)
    top = list(np.argsort(-s)[:m])
    return knockout(J, top, mode)


def methods(N):
    L = int(np.log2(N))
    M = {"олимпийка, 1 вызов на матч": lambda J: knockout(J, range(N), "1"),
         "олимпийка, оба порядка": lambda J: knockout(J, range(N), "2"),
         "олимпийка, до 2 побед": lambda J: knockout(J, range(N), "bo3")}
    for R in sorted({L, L + 2, 2 * L}):
        M[f"швейцарка, {R} туров"] = (lambda R: lambda J: swiss_select(J, N, R))(R)
    if N <= 16:
        M["круговой (все со всеми)"] = lambda J: round_robin(J, N)
    for k in [1, 3, 6]:
        M[f"баллы, {k} раз(а) каждому"] = (lambda k: lambda J: pointwise(J, N, k))(k)
    for m in [4, 8]:
        M[f"гибрид: баллы → финал топ-{m}"] = (lambda m: lambda J: hybrid(J, N, m))(m)
    return M


def run(cfg, rng):
    N = cfg["N"]
    q = rng.normal(0, 1, N); best = int(np.argmax(q))
    out = {}
    for name, f in methods(N).items():
        J = Judge(q, cfg["sigma"], cfg["bias"], cfg["drift"], rng)
        c = f(J)
        out[name] = (c == best, q[best] - q[c], J.calls)
    return out
