"""
The committed S&P 500 snapshot must be real, intact, and market-shaped.

These guard the `spx` demo's data so it can never quietly drift into something
unrealistic: the series must load, be the right size, and clear planted's *own*
market-likeness bar — the same one every synthetic world must pass. That last
check is the demo's quiet flex: the real S&P 500 and the invented worlds look
alike by the same yardstick.
"""

import unittest

from planted.spx_data import returns, SPAN, LOG_RETURNS
from planted.worlds import stylized_facts, is_marketlike


class SpxData(unittest.TestCase):

    def test_loads_a_returns_copy(self):
        r = returns()
        self.assertEqual(len(r), len(LOG_RETURNS))
        self.assertIsNot(r, LOG_RETURNS, "returns() should hand back a fresh list")
        self.assertGreater(len(r), 2000, "expected a multi-year daily series")

    def test_values_are_plausible_daily_returns(self):
        r = returns()
        # Daily log-returns: tiny, centered near zero, none absurdly large.
        self.assertTrue(all(abs(x) < 0.25 for x in r), "implausible daily move")
        self.assertLess(abs(sum(r) / len(r)), 0.01, "daily mean drift too large")

    def test_span_is_well_formed(self):
        d0, d1 = SPAN
        self.assertLess(d0, d1, "span should run oldest -> newest")

    def test_real_spx_is_marketlike(self):
        """Real S&P 500 clears the same stylized-facts bar as a planted world."""
        self.assertTrue(is_marketlike(stylized_facts(returns())),
                        "committed S&P 500 series fails market-likeness check")


if __name__ == "__main__":
    unittest.main()
