"""Interface conformance: every method returns well-formed records + spans."""

import unittest

from planted.worlds import make_world, regimes_from_sep
from planted.methods import ROSTER, embed_all


class MethodInterface(unittest.TestCase):

    def setUp(self):
        self.rets, _ = make_world(seed=3, T=1200, regimes=regimes_from_sep(1.0))

    def test_records_are_well_formed(self):
        for cls in ROSTER:
            method = cls()
            records, spans = method.match(self.rets, seed=0)
            n_windows = len(embed_all(self.rets, method.win, method.stride)[1])
            self.assertEqual(len(records), n_windows)
            self.assertEqual(len(spans), n_windows)
            for q, nb, p, fired in records:
                self.assertTrue(0 <= q < n_windows)
                self.assertTrue(0 <= nb < n_windows)
                self.assertTrue(0.0 <= p <= 1.0)
                self.assertIn(fired, (True, False))

    def test_analog_respects_time_gap(self):
        """Fired analogs must be at least min_gap apart in time (no leakage)."""
        for cls in ROSTER:
            method = cls()
            records, spans = method.match(self.rets, seed=0)
            for q, nb, p, fired in records:
                if fired:
                    gap = abs(spans[q][0] - spans[nb][0])
                    self.assertGreaterEqual(gap, method.min_gap)


if __name__ == "__main__":
    unittest.main()
