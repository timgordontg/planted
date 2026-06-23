"""
Discovery methods — the things `planted` grades.

A *method* looks at a return series and proposes "analogs": pairs of historical
windows it believes are the same kind of market state. Each proposed match
carries a p-value and a fired/abstained decision. The benchmark then checks
those matches against the planted ground truth.

Write your own by subclassing ``Method`` and implementing ``match``; then score
it with ``planted.run_benchmark(YourMethod())``. Three reference methods ship
here so you can see the spread before writing anything:

* ``RandomMatcher``      — honest but useless: fires at random with rate alpha.
                           Recovers nothing; demonstrates the benchmark's floor.
* ``UngatedNN``          — the "tourist": always claims its nearest neighbor is
                           an analog. Looks productive, but hallucinates analogs
                           in pure noise.
* ``SurrogateGatedNN``   — the "practitioner": same nearest-neighbor search, but
                           a match only counts if it is closer than a
                           structure-destroyed surrogate would produce. This is
                           surrogate-data hypothesis testing, and it is what lets
                           the method *abstain* when there is nothing to find.

The window embedding helpers are shared and public so a custom method can reuse
them or roll its own representation.
"""

import math
import random
from statistics import mean, pstdev


# ---------------------------------------------------------------------------
# Shared representation: embed rolling windows into an interpretable space
# ---------------------------------------------------------------------------

def _autocorr1(x):
    m = mean(x)
    denom = sum((v - m) ** 2 for v in x)
    if denom == 0:
        return 0.0
    return sum((x[i] - m) * (x[i + 1] - m) for i in range(len(x) - 1)) / denom


def window_features(w):
    """Embed one return window as a 6-d interpretable feature vector:
    [mean, volatility, return autocorr, |return| autocorr, downside semi-dev,
    excess kurtosis]. These are deliberately hand-crafted and legible — a
    learned embedding is an obvious upgrade and a fair thing to benchmark."""
    m = mean(w)
    s = pstdev(w) or 1e-9
    absr = [abs(x) for x in w]
    down = [x for x in w if x < 0]
    semidev = (sum(x * x for x in down) / len(w)) ** 0.5 if down else 0.0
    k = sum(((x - m) / s) ** 4 for x in w) / len(w) - 3.0
    return [m, s, _autocorr1(w), _autocorr1(absr), semidev, k]


def embed_all(rets, win, stride):
    """Slide a window of length ``win`` by ``stride`` and embed each placement.
    Returns (feature_rows, spans) where spans[i] = (start, end)."""
    feats, spans = [], []
    for start in range(0, len(rets) - win + 1, stride):
        feats.append(window_features(rets[start:start + win]))
        spans.append((start, start + win))
    return feats, spans


def standardize(feats):
    """Z-score each feature column so no single feature dominates the distance."""
    cols = list(zip(*feats))
    mu = [mean(c) for c in cols]
    sd = [pstdev(c) or 1e-9 for c in cols]
    return [[(r[j] - mu[j]) / sd[j] for j in range(len(r))] for r in feats]


def _dist(a, b):
    return math.sqrt(sum((a[j] - b[j]) ** 2 for j in range(len(a))))


def nearest_neighbors(Z, spans, min_gap):
    """For each window, its nearest neighbor that is at least ``min_gap`` steps
    away in time. The time gap forces a "match" to be a genuine *recurrence*
    elsewhere in history rather than the adjacent (overlapping) window — this is
    what prevents temporal-adjacency leakage in both matcher and scorer."""
    out = []
    for i in range(len(Z)):
        best, arg = float("inf"), -1
        si0 = spans[i][0]
        for j in range(len(Z)):
            if abs(si0 - spans[j][0]) < min_gap:
                continue
            d = _dist(Z[i], Z[j])
            if d < best:
                best, arg = d, j
        out.append((best, arg))
    return out


def block_bootstrap(rets, block, rng):
    """Structure-destroying surrogate via circular block resampling. Preserves
    the marginal distribution and short-range stylized facts (within a block)
    but destroys the long-range regime persistence a matcher tries to exploit.
    Comparing real matches against this surrogate is the null hypothesis test."""
    T = len(rets)
    out = []
    while len(out) < T:
        start = rng.randrange(T)
        out.extend(rets[start:start + block])
    return out[:T]


# ---------------------------------------------------------------------------
# Method interface + reference methods
# ---------------------------------------------------------------------------

# A record is a 4-tuple: (query_index, neighbor_index, p_value, fired_bool).

class Method:
    """Base class. Subclass and implement ``match``.

    ``match(rets, seed)`` must return ``(records, spans)`` where ``records`` is a
    list of ``(query_idx, neighbor_idx, p_value, fired)`` and ``spans`` is the
    window-span list from ``embed_all`` (so the scorer can look up labels).
    """

    name = "method"

    def __init__(self, win=50, stride=15, min_gap=100, alpha=0.05):
        self.win, self.stride = win, stride
        self.min_gap, self.alpha = min_gap, alpha

    def match(self, rets, seed=0):
        raise NotImplementedError


class RandomMatcher(Method):
    """Honest-but-useless control. Picks a random distant neighbor and fires
    independently with probability ``alpha``. Should score ~0 recovery and ~alpha
    fire-rate everywhere — the floor any real method must clear."""

    name = "random"

    def match(self, rets, seed=0):
        rng = random.Random(seed)
        _, spans = embed_all(rets, self.win, self.stride)
        n = len(spans)
        records = []
        for i in range(n):
            cands = [j for j in range(n)
                     if abs(spans[i][0] - spans[j][0]) >= self.min_gap]
            nb = rng.choice(cands) if cands else i
            fired = rng.random() < self.alpha
            records.append((i, nb, rng.random(), fired))
        return records, spans


class UngatedNN(Method):
    """The "tourist". Nearest-neighbor analog finder with NO significance gate —
    it always declares its nearest neighbor a real analog. Productive-looking,
    but it cannot abstain, so it hallucinates analogs in pure noise."""

    name = "ungated-nn"

    def match(self, rets, seed=0):
        feats, spans = embed_all(rets, self.win, self.stride)
        Z = standardize(feats)
        nn = nearest_neighbors(Z, spans, self.min_gap)
        return [(i, nn[i][1], 0.0, True) for i in range(len(nn))], spans


class SurrogateGatedNN(Method):
    """The "practitioner". Same nearest-neighbor search, but a match only fires
    if its distance is in the left tail of distances produced on block-bootstrap
    surrogates of the *same* series (structure destroyed). The surrogate gives a
    calibrated p-value, so on a structureless world the method correctly stays
    near its alpha fire-rate instead of inventing analogs."""

    name = "surrogate-gated-nn"

    def __init__(self, n_surrogate=8, surrogate_block=10, **kw):
        super().__init__(**kw)
        self.n_surrogate = n_surrogate
        self.surrogate_block = surrogate_block

    def match(self, rets, seed=0):
        rng = random.Random(seed)
        feats, spans = embed_all(rets, self.win, self.stride)
        Z = standardize(feats)
        obs = nearest_neighbors(Z, spans, self.min_gap)

        # Null distribution of NN distances under structure-destroyed surrogates.
        null_d = []
        for _ in range(self.n_surrogate):
            sur = block_bootstrap(rets, self.surrogate_block, rng)
            f2, sp2 = embed_all(sur, self.win, self.stride)
            null_d.extend(d for d, _ in nearest_neighbors(standardize(f2),
                                                          sp2, self.min_gap))
        null_d.sort()
        Nn = len(null_d)

        def pval(d):
            lo, hi = 0, Nn
            while lo < hi:                       # fraction of surrogate d <= d
                mid = (lo + hi) // 2
                if null_d[mid] <= d:
                    lo = mid + 1
                else:
                    hi = mid
            return max(1.0 / (Nn + 1), lo / Nn)

        records = []
        for i, (d, arg) in enumerate(obs):
            p = pval(d)
            records.append((i, arg, p, p <= self.alpha))
        return records, spans


#: The default roster the leaderboard runs.
ROSTER = [RandomMatcher, UngatedNN, SurrogateGatedNN]
