# planted

**Can a machine tell a real pattern from random noise?**

Most market-prediction software can't — and that's exactly how it fools people.
`planted` is a 10-second demo that proves it.

*An open piece of my machine-learning research on financial markets. — Tim Gordon*

---

> Show a fortune teller pure TV static and they'll confidently "see" a pattern in
> it. A surprising amount of trading software does the same thing — and only the
> losses, much later, give it away.

## The a-ha

I gave two pattern-finders **2,000 days of pure random noise.** There is no pattern.
I made sure. Here's how often each one "found" one anyway:

![How often each method found a "pattern" in pure random noise — lower is more honest](figures/hallucination.svg)

One shrugged and said *"nothing here."* The other found a pattern in **every single
day** of pure noise — the kind of method that looks like a genius in a backtest and
goes broke the moment it meets the real world.

> **Finding patterns is easy. Knowing when there aren't any is the whole game.**

Run it yourself — about ten seconds:

```bash
git clone https://github.com/timgordontg/planted && cd planted
python -m planted demo
```

No setup, no dependencies. Pure Python.

## It's not just made-up data — here's the real S&P 500

Fair question: sure, it works on your invented markets — but what about real money?

```bash
python -m planted spx     # the same two finders, on 12 years of real S&P 500
```

I ran both finders on every trading day of the S&P 500 from 2014 to 2026 — then on
the **same** data shuffled into meaningless noise, to see if they could tell the
difference:

| "I've seen this day before"… | …on the real S&P 500 | …on the shuffle |
|---|---|---|
| the honest finder | **2% of days** | 5% of days |
| the tourist | **100% of days** | 100% of days |

The tourist calls almost every day a repeat of an earlier one — in the real market
**and** in pure noise. It can't tell them apart. The honest finder barely fires on
either: a handful of days, the same low rate you'd get by chance. By its test,
day-to-day S&P moves don't repeat in a way you could trade — so it stays quiet,
instead of selling you a pattern that isn't there.

*(Nothing here is graded against an answer key — real markets don't come with one.
The demo only asks whether a method claims structure it can't back up. That honest
question is the whole point.)*

## Why I built it

I build machine-learning methods for financial markets. The models that make money
stay private — but *how I keep them honest* shouldn't be a secret, and it's the part
worth sharing. `planted` is a tiny world where I can hide a real pattern, or hide
nothing at all, and prove whether a method is honest enough to know the difference.

That discipline — refusing to believe a pattern until it beats pure chance — is what
separates a real edge from a story that falls apart on live data.

**If you build, or hire for, ML that knows a signal from a story — I'd love to talk.**

## Go deeper

```bash
python -m planted bench    # the full scoreboard — now with real patterns mixed in
```

<details>
<summary>How it works (for the curious)</summary>

<br>

`planted` invents synthetic markets that behave like the real thing — fat tails,
volatility clustering, the works — and secretly labels each one as *has a pattern*
or *pure noise*. A method hunts for repeating structure; the honest one only counts
a find if it is stronger than the same hunt run on a **scrambled** copy of the data.
Every method is then scored on two things at once: did it find the real patterns,
**and** did it stay quiet on the noise? The headline score multiplies the two — so
you cannot win by crying "pattern!" at everything.

It even grades itself: `python -m unittest discover -s tests` runs checks that fail
if the framework is fooling *itself*. Under a thousand lines of pure standard
library — readable in one sitting.

</details>

## License

MIT © Tim Gordon
