"""
planted — a market that knows the right answer.

A benchmark for pattern-discovery methods on financial time series. It generates
synthetic markets with *planted* ground truth (and sometimes none at all), then
measures whether a method recovers the real structure AND refuses to hallucinate
structure that isn't there.

Zero dependencies, pure standard library. Quick start:

    import planted
    planted.leaderboard(planted.ROSTER)          # score the reference methods

    rets = planted.load_returns("yourdata.csv")  # bring your own series
    print(planted.format_explore(planted.explore(rets)))
"""

from .worlds import (make_world, regimes_from_sep, stylized_facts,
                     is_marketlike, REGIME_TARGETS)
from .methods import (Method, RandomMatcher, UngatedNN, SurrogateGatedNN,
                      ROSTER, window_features, embed_all)
from .score import (recovery_skill, noise_rejection, calibration, composite,
                    fire_rate, fired_agreements)
from .benchmark import run_benchmark, leaderboard
from .data import load_returns, explore, format_explore

__version__ = "0.1.0"

__all__ = [
    "make_world", "regimes_from_sep", "stylized_facts", "is_marketlike",
    "REGIME_TARGETS", "Method", "RandomMatcher", "UngatedNN", "SurrogateGatedNN",
    "ROSTER", "window_features", "embed_all", "recovery_skill",
    "noise_rejection", "calibration", "composite", "fire_rate",
    "fired_agreements", "run_benchmark", "leaderboard", "load_returns",
    "explore", "format_explore", "__version__",
]
