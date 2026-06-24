"""
Write your own method in ~20 lines, then score it against planted ground truth.

Run:  python examples/custom_method.py

This defines a deliberately simple method — nearest-neighbor on a 2-feature
representation (volatility + drift only), gated by the same surrogate test the
built-ins use — and drops it straight into the benchmark. Change `features`
below and re-run to see how representation choices move the recovery / fire-rate
tradeoff. That is the whole point: an open frame where a real idea gets a real,
ground-truthed result, ad hoc.
"""

import os
import sys
from statistics import mean, pstdev

# Run straight from a fresh clone (no install): put the repo root on the path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from planted import run_benchmark
from planted.methods import SurrogateGatedNN, embed_all, standardize, \
    nearest_neighbors, block_bootstrap


def features(window):
    """Your representation. Return a feature vector for one return window.
    Here: just volatility and drift. Try adding skew, autocorrelation, etc."""
    return [pstdev(window) or 1e-9, mean(window)]


class MyMethod(SurrogateGatedNN):
    """Reuse the surrogate gate; swap in a custom window representation."""

    name = "my-2-feature-nn"

    def _embed(self, rets):
        feats, spans = [], []
        for start in range(0, len(rets) - self.win + 1, self.stride):
            feats.append(features(rets[start:start + self.win]))
            spans.append((start, start + self.win))
        return standardize(feats), spans

    def match(self, rets, seed=0):
        import random
        rng = random.Random(seed)
        Z, spans = self._embed(rets)
        obs = nearest_neighbors(Z, spans, self.min_gap)
        null_d = []
        for _ in range(self.n_surrogate):
            sur = block_bootstrap(rets, self.surrogate_block, rng)
            Zs, sps = self._embed(sur)
            null_d.extend(d for d, _ in nearest_neighbors(Zs, sps, self.min_gap))
        null_d.sort()
        Nn = len(null_d)

        def pval(d):
            lo, hi = 0, Nn
            while lo < hi:
                mid = (lo + hi) // 2
                if null_d[mid] <= d:
                    lo = mid + 1
                else:
                    hi = mid
            return max(1.0 / (Nn + 1), lo / Nn)

        out = []
        for i, (d, arg) in enumerate(obs):
            p = pval(d)
            out.append((i, arg, p, p <= self.alpha))
        return out, spans


if __name__ == "__main__":
    card = run_benchmark(MyMethod(), n_seeds=8)["scorecard"]
    print(f"\n{card['method']}:")
    print(f"  composite             = {card['composite']:.3f}")
    print(f"  recovery (structure)  = {card['struct_recovery']:.3f}")
    print(f"  recovery (noise)      = {card['noise_recovery']:.3f}  (want ~0)")
    print(f"  fire-rate (noise)     = {card['noise_fire_rate']:.3f}  (want ~0.05)")
    print("\nNow edit features() above and re-run. Fewer features usually means")
    print("less recovery but easier calibration — the core tradeoff.")
