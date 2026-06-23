"""
Null sanity checks — the benchmark grading *itself*.

These are the tests that make `planted` falsifiable: if the framework were
fooling itself, they would fail. They assert that on pure-noise worlds an honest
method recovers ~nothing and fires at ~alpha, that a no-gate "tourist"
hallucinates analogs in that same noise, and that real structure is genuinely
recovered. Pure stdlib `unittest` — `python -m unittest` runs them, no install.
Takes ~a minute: the null claims are pooled estimates, so the suite spends real
seeds to resolve them rather than asserting against a hand-tuned threshold.
"""

import unittest

from planted.benchmark import run_benchmark
from planted.methods import SurrogateGatedNN, UngatedNN, RandomMatcher
from planted.worlds import make_world, regimes_from_sep, stylized_facts, \
    is_marketlike

# Small config so the suite runs in a few seconds while staying meaningful.
# Recovery is a pooled estimate with a standard error, so the null checks assert
# *statistical* consistency with zero (within K_SE standard errors) rather than
# against an arbitrary cutoff — that is robust to the sampling noise at this seed
# count instead of papering over it with a loose threshold.
CFG = dict(n_seeds=16, T=1500)
K_SE = 3.0


class NullSanity(unittest.TestCase):

    def test_gated_does_not_fool_itself_on_noise(self):
        """On pure noise, recovery is consistent with zero and fire-rate near
        alpha (the method abstains instead of inventing analogs)."""
        card = run_benchmark(SurrogateGatedNN(), seps=(0.0,), **CFG)["scorecard"]
        self.assertLessEqual(
            abs(card["noise_recovery"]), K_SE * card["noise_recovery_se"],
            "gated method's noise recovery is not consistent with zero skill")
        self.assertLess(card["noise_fire_rate"], 0.12,
                        "gated method over-fires on pure noise")

    def test_gated_recovers_real_structure(self):
        """With strong regimes, recovery is positive beyond sampling noise."""
        card = run_benchmark(SurrogateGatedNN(), seps=(1.0,), **CFG)["scorecard"]
        # struct_recovery averages sep>=0.6; here only sep=1.0 is present.
        self.assertGreater(
            card["struct_recovery"], K_SE * card["struct_recovery_se"],
            "gated method fails to recover planted structure beyond noise")

    def test_tourist_hallucinates_on_noise(self):
        """The ungated tourist fires almost everywhere, even in pure noise."""
        card = run_benchmark(UngatedNN(), seps=(0.0,), **CFG)["scorecard"]
        self.assertGreater(card["noise_fire_rate"], 0.8,
                           "tourist should fire ~everywhere; gate may be leaking")

    def test_tourist_composite_collapses(self):
        """Hallucination is punished: the tourist's composite is near zero."""
        card = run_benchmark(UngatedNN(), **CFG)["scorecard"]
        self.assertLess(card["composite"], 0.05,
                        "noise-rejection penalty failed to punish the tourist")

    def test_random_is_a_floor(self):
        """Honest-but-useless random scores ~0 recovery (no false skill) — even
        on a *structured* world, because the chance baseline is conditioned on
        the same gap-respecting pairing the matcher uses."""
        card = run_benchmark(RandomMatcher(), seps=(1.0,), **CFG)["scorecard"]
        self.assertLessEqual(
            abs(card["struct_recovery"]), K_SE * card["struct_recovery_se"],
            "random matcher shows skill inconsistent with zero")

    def test_null_world_is_marketlike(self):
        """A NULL world must still look like a market (fat tails, vol cluster)."""
        rets, _ = make_world(seed=7, T=3000, regimes=None)
        self.assertTrue(is_marketlike(stylized_facts(rets)),
                        "null world fails stylized-facts validation")

    def test_sep_zero_equals_no_structure(self):
        """sep=0 collapses all regimes — the structured generator's own null."""
        regimes = regimes_from_sep(0.0)
        self.assertTrue(all(r == regimes[0] for r in regimes))


if __name__ == "__main__":
    unittest.main()
