"""
Synthetic markets with *planted* ground truth.

The whole point of `planted` is that we generate price worlds where we already
know the answer, so any discovery method can be graded against truth instead of
against a pretty-but-unfalsifiable backtest.

Two kinds of world come out of here:

* a STRUCTURED world  — a hidden Markov regime schedule drives the returns, and
  we keep the regime-label sequence as ground truth. A good method should
  recover it.
* a NULL world        — statistically a market (fat tails, volatility
  clustering) but with NO regime structure to find. A good method should find
  *nothing* here. This is the trap that catches methods that fool themselves.

Generative model (all stdlib, no numpy):
  returns_t = mu(state_t) + sigma_t * vol_scale(state_t) * z_t
  sigma_t^2 = omega + (alpha + gamma*[eps_{t-1} < 0]) * eps_{t-1}^2   # GJR-GARCH
              + beta * sigma_{t-1}^2                                  # (leverage)
  z_t ~ standardized Student-t(df)                                   # fat tails
  state_t ~ sticky K-state Markov chain                              # regimes

The GJR-GARCH recursion gives volatility clustering *and* the leverage effect
(a negative shock adds ``gamma`` to the ARCH coefficient, so down moves raise
tomorrow's volatility more than up moves); the Student-t innovations give fat
tails; the Markov chain gives recurring regimes. Together they reproduce the
"stylized facts" of real return series (see `stylized_facts`).
"""

import math
import random
from statistics import mean, pstdev


# ---------------------------------------------------------------------------
# Generative model
# ---------------------------------------------------------------------------

def _student_t(rng, df):
    """One draw from a Student-t(df) rescaled to unit variance (df > 2).

    Built from a normal over a chi-square: t = z / sqrt(chi2(df)/df). The final
    factor rescales to unit variance so `df` only controls tail fatness, not
    overall scale.
    """
    z = rng.gauss(0.0, 1.0)
    g = rng.gammavariate(df / 2.0, 2.0)          # chi-square(df)
    t = z / math.sqrt(g / df)
    return t * math.sqrt((df - 2.0) / df)


def make_world(seed, T=3000, regimes=None, df=8.0,
               garch=(3.0e-6, 0.02, 0.10, 0.90), p_stay=0.985):
    """Generate one synthetic world.

    Parameters
    ----------
    seed : int            reproducibility.
    T : int               number of time steps (≈ trading days).
    regimes : list[(mu, vol_scale)] | None
        One (drift, volatility-multiplier) pair per regime. ``None`` (or a
        single regime) produces a NULL world: market-like but structureless.
    df : float            Student-t degrees of freedom (smaller => fatter tails).
    garch : (omega, alpha, gamma, beta)
        GJR-GARCH(1,1) parameters. ``gamma`` is the leverage term: a negative
        shock adds it to the ARCH coefficient, so down moves raise next-period
        volatility more than up moves. Defaults are calibrated so unconditional
        volatility is ≈ 1%/day, the scale of real equity returns.
    p_stay : float        Markov self-transition probability (regime stickiness).

    Returns
    -------
    (returns, labels) : (list[float], list[int])
        ``labels[t]`` is the ground-truth regime id at time ``t``.
    """
    rng = random.Random(seed)
    if regimes is None:
        regimes = [(0.0, 1.0)]
    K = len(regimes)
    omega, alpha, gamma, beta = garch

    def next_state(s):
        # Sticky chain: stay with prob p_stay, else jump to a uniform other.
        if K == 1 or rng.random() < p_stay:
            return s
        return rng.choice([k for k in range(K) if k != s])

    state = rng.randrange(K)
    # Under symmetric innovations a negative shock occurs half the time, so the
    # expected ARCH coefficient is alpha + gamma/2 — that is what enters the
    # unconditional variance (and the stationarity budget alpha+gamma/2+beta<1).
    sigma2 = omega / max(1e-9, (1.0 - alpha - gamma / 2.0 - beta))
    eps_prev = 0.0
    rets, labels = [], []
    for _ in range(T):
        state = next_state(state)
        mu, vscale = regimes[state]
        # GJR asymmetry: a negative prior shock adds gamma to the ARCH term.
        arch = alpha + (gamma if eps_prev < 0.0 else 0.0)
        sigma2 = omega + arch * eps_prev * eps_prev + beta * sigma2
        sigma = math.sqrt(sigma2) * vscale
        eps = sigma * _student_t(rng, df)
        rets.append(mu + eps)
        eps_prev = eps
        labels.append(state)
    return rets, labels


# ---------------------------------------------------------------------------
# Difficulty knob: interpolate regimes from "all identical" to "well separated"
# ---------------------------------------------------------------------------

# Three archetypal regimes (drift, volatility multiplier). Regimes are separated
# mostly by DRIFT and only mildly by volatility (0.95 / 1.25 / 1.05). That is
# deliberate: a wide vol spread would give the regimes away by a blunt mixture
# of variances and inflate the marginal excess kurtosis into the dozens. A
# single regime (a NULL world) keeps textbook kurtosis (~4); a three-regime
# world runs higher from the vol mixture (~15-18) — which is itself realistic
# for a series spanning calm and crisis. Separating on drift keeps the marginal
# distribution market-shaped while leaving the regimes genuinely discoverable:
# they must be *inferred*, not read straight off a volatility gauge.
REGIME_TARGETS = [
    (0.0018, 0.95),   # calm: steady drift up, slightly low vol
    (-0.0024, 1.25),  # stress: drift down, elevated vol
    (0.0000, 1.05),   # choppy: no drift, near-baseline vol
]
_MEAN_MU = mean(m for m, _ in REGIME_TARGETS)
_MEAN_V = mean(v for _, v in REGIME_TARGETS)


def regimes_from_sep(sep):
    """Regimes at separation ``sep`` in [0, 1].

    ``sep = 0`` collapses every regime onto the same parameters (a NULL world:
    structure strength zero). ``sep = 1`` is full separation. Sweeping ``sep``
    is how we turn the "amount of real structure" dial — the x-axis of the
    benchmark curves.
    """
    return [(_MEAN_MU + sep * (mu - _MEAN_MU),
             _MEAN_V + sep * (v - _MEAN_V)) for mu, v in REGIME_TARGETS]


# ---------------------------------------------------------------------------
# Stylized-facts validator: does a world actually look like a market?
# ---------------------------------------------------------------------------

def _autocorr(x, lag):
    n = len(x)
    m = mean(x)
    denom = sum((v - m) ** 2 for v in x)
    if denom == 0:
        return 0.0
    num = sum((x[i] - m) * (x[i + lag] - m) for i in range(n - lag))
    return num / denom


def excess_kurtosis(x):
    """Kurtosis minus 3. Positive => fatter tails than a normal."""
    m = mean(x)
    s = pstdev(x)
    if s == 0:
        return 0.0
    return sum(((v - m) / s) ** 4 for v in x) / len(x) - 3.0


def stylized_facts(rets):
    """Quantitative fingerprint a return series must roughly match to count as
    market-like. These are the canonical "stylized facts" of asset returns:

    * fat tails            : excess kurtosis > 0
    * no linear pred.      : return autocorrelation ≈ 0
    * volatility clustering: |return| autocorrelation > 0
    * leverage effect      : today's return vs tomorrow's |return| is negative
                             (down moves are followed by higher volatility)
    """
    absr = [abs(r) for r in rets]
    a, b = rets[:-1], absr[1:]
    ma, mb = mean(a), mean(b)
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(len(a)))
    va = math.sqrt(sum((v - ma) ** 2 for v in a))
    vb = math.sqrt(sum((v - mb) ** 2 for v in b))
    leverage = cov / (va * vb) if va and vb else 0.0
    return {
        "excess_kurtosis": excess_kurtosis(rets),
        "ret_autocorr_l1": _autocorr(rets, 1),
        "absret_autocorr_l1": _autocorr(absr, 1),
        "leverage": leverage,
    }


def is_marketlike(sf):
    """True if a stylized-facts dict clears minimal market-likeness thresholds."""
    return (sf["excess_kurtosis"] > 0.5 and
            abs(sf["ret_autocorr_l1"]) < 0.15 and
            sf["absret_autocorr_l1"] > 0.05)
