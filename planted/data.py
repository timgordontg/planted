"""
Bring your own data.

`planted`'s benchmark needs planted ground truth, which real markets don't come
with. But the *method machinery* works on any return series — so you can point a
method at your own data and get a real, ad-hoc answer to the only question that
survives without ground truth:

    "Does this method find more structure in my series than in
     structure-destroyed copies of it?"

That is exactly the surrogate-data test the gated method uses internally, run on
your data directly. If a method fires far more on your real series than on its
block-bootstrap surrogates, the recurrence structure it found is unlikely to be
noise. If it fires the same on both, it was fooling itself.

INPUT FORMAT (specified, so results are unambiguous)
----------------------------------------------------
A CSV file. By default the LAST column is used; pass ``col`` to choose another
by name or index. Values are read as either prices or returns:

  * kind="price"  : values are levels; we convert to log returns,
                    r_t = ln(P_t / P_{t-1}).
  * kind="return" : values are already period returns (e.g. 0.012 for +1.2%).
  * kind="auto"   : guess — if values are all positive and span a wide range we
                    treat them as prices, otherwise as returns. Override if the
                    guess is wrong.

A header row is auto-detected (used when present to resolve column names).
Blank cells and non-numeric rows are skipped.
"""

import csv
import math
import random
from statistics import mean, pstdev

from .methods import SurrogateGatedNN, block_bootstrap, embed_all
from .score import fire_rate


def _looks_like_prices(vals):
    """Heuristic: prices are positive and span a wide multiplicative range;
    returns are small and centered near zero."""
    if any(v <= 0 for v in vals):
        return False
    spread = max(vals) / min(vals)
    return spread > 1.5 and abs(mean(vals)) > 0.1


def load_returns(path, col=None, kind="auto"):
    """Load a return series from a CSV per the documented format.

    Parameters
    ----------
    path : str           path to the CSV file.
    col : str | int | None
        Column to read. ``None`` => last column. A string matches a header name;
        an int is a 0-based column index.
    kind : {"auto", "price", "return"}
        How to interpret the values (see module docstring).

    Returns
    -------
    list[float] : the return series.
    """
    rows = []
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if row:
                rows.append(row)
    if not rows:
        raise ValueError(f"{path}: file is empty")

    # Header detection: first row is a header if any cell is non-numeric.
    def _is_num(x):
        try:
            float(x)
            return True
        except ValueError:
            return False

    header = None
    if any(not _is_num(c) for c in rows[0]):
        header, rows = rows[0], rows[1:]

    # Resolve column index.
    if col is None:
        idx = len(rows[0]) - 1
    elif isinstance(col, int):
        idx = col
    else:
        if header is None or col not in header:
            raise ValueError(f"{path}: no column named {col!r}; "
                             f"header is {header}")
        idx = header.index(col)

    vals = []
    for row in rows:
        if idx < len(row) and _is_num(row[idx]) and row[idx].strip() != "":
            vals.append(float(row[idx]))
    if len(vals) < 200:
        raise ValueError(f"{path}: only {len(vals)} usable values; need >= 200 "
                         f"for a meaningful surrogate test")

    if kind == "auto":
        kind = "price" if _looks_like_prices(vals) else "return"
    if kind == "price":
        return [math.log(vals[i] / vals[i - 1]) for i in range(1, len(vals))]
    return vals


def explore(rets, method=None, n_surrogate=20, seed=0):
    """Run a method on a real (unlabeled) series and surrogate-test the result.

    Compares the method's fire-rate on the real series against its fire-rate on
    block-bootstrap surrogates (same marginals + short-range stylized facts, no
    long-range structure). Returns a dict of results and is safe to print.
    """
    method = method or SurrogateGatedNN()
    sf = _quick_facts(rets)

    records, _ = method.match(rets, seed=seed)
    real_fr = fire_rate(records)

    rng = random.Random(seed)
    sur_frs = []
    for _ in range(n_surrogate):
        sur = block_bootstrap(rets, getattr(method, "surrogate_block", 10), rng)
        rec_s, _ = method.match(sur, seed=seed)
        sur_frs.append(fire_rate(rec_s))
    sur_mean = mean(sur_frs)
    sur_sd = pstdev(sur_frs) or 1e-9
    z = (real_fr - sur_mean) / sur_sd       # how anomalous is the real fire-rate
    # one-sided empirical p: fraction of surrogates firing >= the real series
    p = (sum(1 for f in sur_frs if f >= real_fr) + 1) / (n_surrogate + 1)

    return {
        "n_returns": len(rets),
        "stylized_facts": sf,
        "method": method.name,
        "real_fire_rate": real_fr,
        "surrogate_fire_rate": sur_mean,
        "surrogate_sd": sur_sd,
        "z_score": z,
        "p_value": p,
        "verdict": _verdict(p),
    }


def _verdict(p):
    if p <= 0.01:
        return "strong evidence of real recurrence structure"
    if p <= 0.05:
        return "evidence of real recurrence structure"
    if p <= 0.15:
        return "weak / inconclusive"
    return "indistinguishable from noise — the method found nothing real"


def _quick_facts(rets):
    from .worlds import stylized_facts, is_marketlike
    sf = stylized_facts(rets)
    sf["marketlike"] = is_marketlike(sf)
    return sf


def format_explore(res):
    """Pretty one-screen readout of an ``explore`` result."""
    sf = res["stylized_facts"]
    lines = [
        f"  series length      : {res['n_returns']} returns",
        f"  stylized facts     : kurtosis={sf['excess_kurtosis']:+.2f}  "
        f"vol-cluster={sf['absret_autocorr_l1']:+.3f}  "
        f"leverage={sf['leverage']:+.3f}  marketlike={sf['marketlike']}",
        f"  method             : {res['method']}",
        f"  fire-rate (real)   : {res['real_fire_rate']:.3f}",
        f"  fire-rate (surrog) : {res['surrogate_fire_rate']:.3f} "
        f"+/- {res['surrogate_sd']:.3f}",
        f"  z-score            : {res['z_score']:+.2f}   "
        f"(real vs structure-destroyed)",
        f"  surrogate p-value  : {res['p_value']:.3f}",
        f"  VERDICT            : {res['verdict']}",
    ]
    return "\n".join(lines)
