# planted

**A market that knows the right answer.**

`planted` is a benchmark for pattern-discovery methods on financial time series.
It generates synthetic markets with *planted* ground truth — and, crucially,
sometimes with **no structure at all** — then grades a method on two axes that
pull against each other:

- **Recovery** — when there *is* real structure, does the method find it?
- **Noise-rejection** — when there is **nothing to find**, does the method
  *admit it* and abstain, instead of hallucinating an "analog" anyway?

Most backtests only measure the first. That is how a method can look brilliant
and be fooling itself: on pure noise it still "discovers" patterns, and you never
notice because the noise was never labelled. `planted` labels the noise. A method
that screams *"analog!"* in a structureless world is caught and its score
collapses — no matter how good it looked on the structured worlds.

> The headline metric is **noise-rejection, not recovery.** Finding patterns is
> easy. Knowing when there are none is the hard, valuable part — and the part
> that survives contact with a real market.

Zero dependencies. Pure Python standard library. Runs anywhere.

---

## The one-line demo

```bash
python -m planted demo
```

```
planted :: demo — does the method know when to say 'nothing here'?

  surrogate-gated-nn   composite=0.153  recovery(structure)=0.185  fire-rate(noise)=0.060
  ungated-nn           composite=0.007  recovery(structure)=0.145  fire-rate(noise)=1.000

  The tourist (ungated) finds 'analogs' even in pure noise, so its
  composite collapses. Only abstaining on noise earns the score.
```

Two methods, same nearest-neighbour search. One adds a significance gate; one
doesn't. On structured worlds they look comparable. On **pure-noise** worlds the
ungated "tourist" fires on *100% of windows* — it cannot help itself — so its
headline composite collapses to ~0.

![Fire-rate on pure-noise worlds, by method](figures/hallucination.svg)

This is the whole thesis in one chart: **lower is honest.** The dashed line is
the honest floor (`alpha = 0.05`). A calibrated method sits on it. A method that
"finds something" everywhere sits at the top — and that is a liability, not a
feature.

---

## The leaderboard

```bash
python -m planted bench
```

```
  PLANTED LEADERBOARD  (higher composite = better)
  --------------------------------------------------------------------------
  method                composite  noise rec.  noise fire  struct rec.
  (want)                     high          ~0      ~alpha         high
  --------------------------------------------------------------------------
  surrogate-gated-nn        0.153      -0.056       0.060        0.185
  ungated-nn                0.007       0.007       1.000        0.145
  random                    0.000      -0.042       0.062       -0.042
  --------------------------------------------------------------------------
```

Read across the **ungated-nn** row: its `struct rec.` (0.145) looks like real
skill — but its `noise fire` is `1.000`. It fires on everything, so its apparent
recovery is just the half of a broken clock that happens to be right. The
composite nets the two and the tourist drops to ~0. **random** is the floor: an
honest-but-useless control that recovers nothing and correctly fires at ~`alpha`.

---

## How a calibrated method behaves

Sweep the structure dial from `sep = 0` (pure noise) to `sep = 1` (strong
regimes) and watch a properly-gated method:

**Recovery rises only when real structure exists** — and sits at ~0 on noise, so
it never claims skill it doesn't have:

![Recovery skill vs structure strength](figures/recovery.svg)

**Fire-rate stays pinned at the noise floor** until genuine regimes appear:

![Fire-rate vs structure strength](figures/fire_rate.svg)

Those two curves *are* the definition of a trustworthy discovery method: it does
nothing on nothing, and more as there is more to do.

---

## How it works

Three pieces, each a small, readable module:

**1. Worlds with ground truth** (`planted/worlds.py`).
Returns are generated from a regime-switching **GJR-GARCH** process — GARCH
volatility clustering plus a leverage term, so a down move raises tomorrow's
volatility more than an equal up move — with Student-t (fat-tailed) innovations
and a sticky Markov chain of hidden regimes. The regime labels are kept as
ground truth. Every world — structured or null — must pass a **stylized-facts
validator** (fat tails, volatility clustering, ~zero return autocorrelation,
leverage effect) so it actually looks like a market. A null world is
statistically a market with *no regime to find*: the trap.

**2. Methods** (`planted/methods.py`).
A *method* proposes "analogs" — pairs of historical windows it believes are the
same kind of market state — each with a p-value and a fired/abstained decision.
Three reference methods ship so you can see the spread:

| method | what it is |
| --- | --- |
| `RandomMatcher` | honest-but-useless control; fires at random rate `alpha`. The floor. |
| `UngatedNN` | the *tourist*: always calls its nearest neighbour an analog. Cannot abstain. |
| `SurrogateGatedNN` | the *practitioner*: a match only counts if it is closer than a structure-destroyed **surrogate** of the same series would produce. |

The gate is surrogate-data hypothesis testing: block-bootstrap the series to
destroy long-range regime structure while preserving its marginals and
short-range stylized facts, then keep a match only if it is anomalously close
relative to that null. That is what lets a method *abstain*.

**3. Scoring** (`planted/score.py`).
`recovery` is a chance-corrected skill score: among the matches a method *fired*,
how often do the two windows actually share a planted regime, above what random
pairing would give? `noise-rejection` penalizes firing above `alpha` on null
worlds. The headline `composite = recovery × noise-rejection` cannot be won by
finding analogs everywhere — only by finding real ones *and* abstaining on noise.

> **The chance baseline is conditioned on how the method is allowed to pair
> windows** (matches must be ≥ `min_gap` apart in time). Scoring against the
> naive unconditional baseline leaves a residual phantom "skill" in a random
> matcher — the benchmark fooling *itself*. `planted` corrects for this so an
> honest-but-useless method centers at exactly zero skill. See
> `score.expected_agreement`.

---

## Write your own method

Subclass `Method`, implement `match`, drop it into the benchmark. The full
example is in [`examples/custom_method.py`](examples/custom_method.py) — a custom
window representation in ~20 lines:

```python
from planted import run_benchmark
from planted.methods import SurrogateGatedNN

class MyMethod(SurrogateGatedNN):
    name = "my-method"
    # ...swap in your own window features / representation...

card = run_benchmark(MyMethod(), n_seeds=8)["scorecard"]
print(card["composite"], card["noise_recovery"], card["struct_recovery"])
```

Change the representation, re-run, and read a real, ground-truthed result —
including whether your clever new feature just made the method hallucinate more.

---

## The benchmark grades itself

The most important tests in this repo are the **null-sanity checks**
([`tests/test_null_sanity.py`](tests/test_null_sanity.py)): they assert that on
pure noise an honest method's recovery is statistically consistent with **zero**,
that the tourist hallucinates in that same noise, and that real structure is
genuinely recovered. They are what make `planted` *falsifiable* — if the
framework were fooling itself, they would fail.

```bash
python -m unittest discover -s tests
```

Pure stdlib `unittest`, no install required.

---

## Install

```bash
pip install planted          # once published
# or, from source:
git clone https://github.com/timgordontg/planted && cd planted
python -m planted demo
```

No dependencies. Python 3.9+.

---

## What this is — and how it relates to my work

`planted` is a **methodology demo, not a trading system.** I build machine-learning
methods for financial markets; the models that act on real markets — and the
features and signals behind them — stay private. What's open here is the part I
think is most transferable and most often skipped: *how you prove a
pattern-discovery method is honest before you trust it.*

The discipline at its center — **measure whether a method abstains on noise, not
just whether it finds patterns** — is one I apply in my own research. A method that
"discovers" structure in everything is worse than useless on a real market: it is
a confident liar, and a backtest that only rewards finding patterns will always be
gamed by it. The thing that separates a real edge from an overfit story is knowing
when to say *"there's nothing here"* — and you can only measure that if you build
worlds where that is the right answer. `planted` builds those worlds, labels the
noise, and scores honesty as a first-class metric.

So no edge is given away here, and none is implied: this is the **evaluation layer**
made public — a way to show how I think about honest method-building, and to find
people who care about the same problem. If that's you, get in touch.

## License

MIT © Tim Gordon
