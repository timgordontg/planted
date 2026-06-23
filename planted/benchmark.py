"""
The benchmark: run a method across many planted worlds and grade it.

``run_benchmark`` sweeps the structure dial ``sep`` from 0 (pure noise) to 1
(strong regimes), generating many independent worlds per setting, and reports:

* a RECOVERY curve   — chance-corrected skill vs structure strength. Should sit
                       at ~0 when sep=0 and rise as structure appears.
* a FIRE-RATE curve  — fraction of windows that fired vs structure strength.
                       Should sit at ~alpha on noise and rise with structure.
* a SCORECARD        — headline scalars: recovery on noise (want ~0), fire-rate
                       on noise (want ~alpha), recovery with structure, and the
                       composite.

``leaderboard`` runs the whole roster and prints one comparison table — the
fastest way to see why a calibrated method beats a tourist that "finds" analogs
everywhere.
"""

from statistics import mean, pstdev

from .worlds import make_world, regimes_from_sep, REGIME_TARGETS
from .score import (fired_agreements, expected_agreement, recovery_skill,
                    fire_rate, composite)

DEFAULT_SEPS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
N_REGIMES = len(REGIME_TARGETS)


def run_benchmark(method, seps=DEFAULT_SEPS, n_seeds=12, T=2000, progress=None):
    """Grade ``method`` across the structure sweep.

    Returns a dict with ``recovery`` and ``fire_rate`` curves (lists of
    ``(sep, value, error)``) and a ``scorecard`` of headline scalars.
    ``progress`` is an optional callback ``progress(sep)`` for CLI feedback.
    """
    # The chance baseline must match how the method is allowed to pair windows:
    # only partners >= min_gap apart in time (see score.expected_agreement).
    min_gap = getattr(method, "min_gap", 0)
    rec_curve, fire_curve = [], []
    for sep in seps:
        regimes = regimes_from_sep(sep)
        pooled, expected, fires = [], [], []
        for s in range(n_seeds):
            rets, labels = make_world(seed=1000 + s, T=T, regimes=regimes)
            records, spans = method.match(rets, seed=s)
            ag = fired_agreements(records, spans, labels)
            pooled.extend(ag)
            expected.extend([expected_agreement(spans, labels, min_gap)] * len(ag))
            fires.append(fire_rate(records))
        exp = mean(expected) if expected else 1.0 / N_REGIMES
        skill, se = recovery_skill(pooled, exp)
        rec_curve.append((sep, skill, se))
        fire_curve.append((sep, mean(fires), pstdev(fires)))
        if progress:
            progress(sep)

    noise_rec, noise_se = rec_curve[0][1], rec_curve[0][2]   # recovery at sep=0
    noise_fire = fire_curve[0][1]                            # fire-rate at sep=0
    hi = [(v, se) for sep, v, se in rec_curve if sep >= 0.6]
    struct_rec = mean(v for v, _ in hi) if hi else rec_curve[-1][1]
    # Pooled SE of the averaged high-structure recovery (independent worlds).
    struct_se = (sum(se * se for _, se in hi) ** 0.5 / len(hi)
                 if hi else rec_curve[-1][2])
    alpha = getattr(method, "alpha", 0.05)
    nr = 1.0 if noise_fire <= alpha else alpha / noise_fire
    scorecard = {
        "method": method.name,
        "noise_recovery": noise_rec,     # want ~0  (no false skill on noise)
        "noise_recovery_se": noise_se,   # standard error of the above
        "noise_fire_rate": noise_fire,   # want ~alpha
        "alpha": alpha,
        "struct_recovery": struct_rec,   # want > 0 (real skill on structure)
        "struct_recovery_se": struct_se, # standard error of the above
        "noise_rejection": nr,
        "composite": composite(struct_rec, nr),
    }
    return {"recovery": rec_curve, "fire_rate": fire_curve, "scorecard": scorecard}


def leaderboard(methods, n_seeds=10, T=2000, verbose=True):
    """Run a roster and return scorecards sorted by composite (best first)."""
    cards = []
    for cls in methods:
        method = cls() if isinstance(cls, type) else cls
        if verbose:
            print(f"  scoring {method.name} ...", flush=True)
        res = run_benchmark(method, n_seeds=n_seeds, T=T)
        cards.append(res["scorecard"])
    cards.sort(key=lambda c: c["composite"], reverse=True)
    if verbose:
        _print_table(cards)
    return cards


def _print_table(cards):
    print("\n  PLANTED LEADERBOARD  (higher composite = better)")
    print("  " + "-" * 74)
    print(f"  {'method':<20}{'composite':>11}{'noise rec.':>12}"
          f"{'noise fire':>12}{'struct rec.':>13}")
    print(f"  {'(want)':<20}{'high':>11}{'~0':>12}{'~alpha':>12}{'high':>13}")
    print("  " + "-" * 74)
    for c in cards:
        print(f"  {c['method']:<20}{c['composite']:>11.3f}"
              f"{c['noise_recovery']:>12.3f}{c['noise_fire_rate']:>12.3f}"
              f"{c['struct_recovery']:>13.3f}")
    print("  " + "-" * 74)
    print("  Read it like this: the tourist scores real-looking 'struct rec.'")
    print("  but its 'noise fire' is ~1.0 — it screams 'analog!' even in pure")
    print("  noise — so the composite collapses. Only abstaining on noise wins.")
