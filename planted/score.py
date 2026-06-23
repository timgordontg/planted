"""
Scoring — how `planted` grades a method's matches against planted truth.

Two axes that pull against each other, plus the diagnostics that keep them
honest:

* RECOVERY        — among the matches a method *fired*, how often do the two
                    windows actually share a planted regime? Reported as a
                    chance-corrected skill score in [-something, 1]: 0 means "no
                    better than guessing", 1 means "perfect". On a NULL world
                    (no structure) an honest method scores ~0 here.

* NOISE-REJECTION — on a NULL world the only correct fire-rate is ~alpha. A
                    method that fires far more than alpha is hallucinating
                    analogs in noise; this score penalizes that over-firing.

* CALIBRATION     — on a NULL world, the per-match p-values should be ~Uniform.
                    Score = 1 - KS distance from uniform. A method whose
                    p-values are calibrated under the null is trustworthy.

The headline number, COMPOSITE = recovery x noise_rejection, nets skill against
hallucination: you cannot win it by finding analogs everywhere, only by finding
real ones and abstaining on noise.
"""

import math
import random
from statistics import mean


def _label_of(spans, idx, labels):
    """Majority planted regime label inside a window's span."""
    a, b = spans[idx]
    seg = labels[a:b]
    return max(set(seg), key=seg.count)


def fired_agreements(records, spans, labels):
    """For each *fired* match, 1 if the two windows share a planted regime else
    0. The raw material of recovery; pooling these across many worlds gives a
    low-variance, unbiased skill estimate (see ``recovery_skill``)."""
    return [1 if _label_of(spans, q, labels) == _label_of(spans, nb, labels)
            else 0 for (q, nb, p, fired) in records if fired]


def expected_agreement(spans, labels, min_gap=0):
    """Chance level for label agreement between two windows paired *the way the
    method is allowed to pair them*: uniformly at random among partners at least
    ``min_gap`` steps apart in time.

    The naive baseline is the Simpson/Gini coincidence ``sum_k p_k^2`` over
    window labels — the probability two *unconditionally* random windows share a
    regime. But every method here only matches windows >= ``min_gap`` apart (no
    adjacent-window leakage), and under persistent regimes that time constraint
    shifts the agreement rate of a random pair away from ``sum_k p_k^2``. Scoring
    against the unconditional baseline therefore leaves a residual skill in an
    honest-but-useless matcher (empirically ~ -0.07 here): the metric fools
    itself in exactly the way this benchmark exists to expose.

    So the baseline is computed *conditional on the gap*: for each window, the
    fraction of its gap-respecting partners that share its label, averaged over
    windows. This is the agreement a random matcher achieves by construction, so
    a random matcher centers at exactly 0 skill. ``min_gap=0`` recovers the
    unconditional Simpson index. Exact, O(n^2) — negligible beside the match."""
    labs = [_label_of(spans, i, labels) for i in range(len(spans))]
    n = len(labs)
    if n < 2:
        return 0.0
    starts = [s[0] for s in spans]
    total_share, total_pairs = 0, 0
    for i in range(n):
        for j in range(n):
            if i == j or abs(starts[i] - starts[j]) < min_gap:
                continue
            total_pairs += 1
            if labs[i] == labs[j]:
                total_share += 1
    return total_share / total_pairs if total_pairs else 0.0


def recovery_skill(agreements, expected):
    """Cohen's-kappa-style skill from pooled agreement outcomes.

    skill = (agreement_rate - expected) / (1 - expected)

    where ``expected`` is the random-pairing agreement (``expected_agreement``).
    Signed (NOT clamped at 0) so that on a true null it is an *unbiased*
    estimator centered on 0 — clamping would bias the null upward and flatter a
    useless method. Returns (skill, standard_error)."""
    n = len(agreements)
    if n == 0:
        return 0.0, 0.0
    denom = (1.0 - expected) or 1e-9
    rate = mean(agreements)
    skill = (rate - expected) / denom
    se = math.sqrt(rate * (1.0 - rate) / n) / denom
    return skill, se


def fire_rate(records):
    """Fraction of windows for which the method fired a match."""
    return sum(1 for r in records if r[3]) / len(records) if records else 0.0


def noise_rejection(records, alpha):
    """1.0 if fire-rate <= alpha (correctly abstaining), else alpha/fire-rate.
    A method firing at 10x alpha on noise scores 0.1 — it is mostly hallucinating."""
    fr = fire_rate(records)
    return (1.0, fr) if fr <= alpha else (alpha / fr, fr)


def calibration(records):
    """1 - Kolmogorov-Smirnov distance between the method's p-values and a
    Uniform[0,1]. 1.0 means perfectly calibrated under the null."""
    ps = sorted(p for (_, _, p, _) in records)
    n = len(ps)
    if n == 0:
        return 0.0
    ks = max(max(abs((i + 1) / n - p), abs(i / n - p)) for i, p in enumerate(ps))
    return 1.0 - ks


def composite(recovery, nr_score):
    """Headline score: skill on structure x abstention on noise."""
    return max(0.0, recovery) * nr_score
