# planted

### Can a pattern-finder tell a real signal from random noise?

Most can't. **`planted`** is a 30-second test that proves it.

*An open finding from my machine-learning research on financial markets — by Tim Gordon.*

---

> Show a fortune teller anything — even TV static — and they'll confidently "see"
> a pattern in it. A surprising amount of market-prediction software does the exact
> same thing: it finds "signals" in pure randomness, looks brilliant in a backtest,
> then loses money the moment it meets data it has never seen.

## The a-ha (run it yourself, ~10 seconds)

```bash
python -m planted demo
```

```
planted :: I just fed two pattern-finders 2,000 days of PURE RANDOM
           NOISE. There is no pattern in it. I made sure.

  surrogate-gated-nn   found a "pattern" in    6% of it   ->  basically said "nothing here"  [ok]
  ungated-nn           found a "pattern" in  100% of it   ->  "PATTERN!" ...in random noise  [x]
```

Same math under the hood. The only difference: **one method is allowed to say
*"nothing here."*** The other can't — so it "finds" a pattern in *every single
window* of pure noise. That second one is the one that looks like a genius in a
backtest and goes broke live.

![What each method does on PURE NOISE — lower is honest](figures/hallucination.svg)

**That's the whole idea: finding patterns is easy. Knowing when there aren't any is
the hard part — and the only part that survives a real market.**

## Why this exists (and who I am)

I'm an ML researcher building methods for financial markets. The models that
actually make money stay private — but *how I keep myself honest* doesn't have to.

`planted` is that part, in the open. It builds a tiny synthetic market where I can
**plant** a real pattern — or plant *nothing at all* — and then check whether a
method is honest enough to tell the difference. No edge is given away here, and
none is implied. It's the discipline, not the secret sauce.

If you build, or hire for, machine learning that knows a signal from a story —
**say hi.**

## Play with it

```bash
git clone https://github.com/timgordontg/planted && cd planted

python -m planted demo     # the a-ha above (~10s): honesty vs hallucination on pure noise
python -m planted bench    # the full scoreboard — now with REAL patterns mixed in
```

`bench` is where it gets satisfying. The honest method and the hallucinator look
*almost the same* at finding real patterns — until you score what they do on noise:

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

Look at **ungated-nn**: its `struct rec.` (0.145) is nearly as high as the honest
method's (0.185) — real-looking skill! But `noise fire` is `1.000`: it fires on
*everything*. Once you penalize that, its score collapses to ~0. The hallucinator
is exposed only because we measured honesty, not just hits.

## ...okay, how does it actually work?

Three small, readable, zero-dependency modules:

- **`worlds.py`** — builds synthetic markets that genuinely behave like real ones
  (fat tails, volatility clustering, the leverage effect — a regime-switching
  GJR-GARCH process). Some get a hidden pattern planted in them; some get nothing.
  *We keep the answer key.*
- **`methods.py`** — the pattern-finders. The honest one only counts a match if it's
  stronger than the same hunt run on a **scrambled** copy of the data (a
  surrogate-data significance test). That's what earns it the right to say "nothing
  here."
- **`score.py`** — grades two things at once: **recovery** (did it find the real
  patterns?) and **noise-rejection** (did it shut up on noise?). The headline score
  is their product, so you cannot win by finding patterns everywhere.

And it **grades itself.** The checks in
[`tests/test_null_sanity.py`](tests/test_null_sanity.py) fail if the framework is
fooling *itself* — they prove an honest method scores ~0 on pure noise. Run them:

```bash
python -m unittest discover -s tests
```

No dependencies. Python 3.9+. Pure standard library.

## License

MIT © Tim Gordon
