"""
Command line entry point.

    python -m planted demo                 the a-ha: honesty vs hallucination on noise
    python -m planted bench                full leaderboard, with real patterns mixed in
    python -m planted spx                  the same test on 12 years of real S&P 500
    python -m planted figures              regenerate the README figures

Run with no arguments for this help.
"""

import sys


def _demo():
    from statistics import mean
    from .worlds import make_world
    from .methods import UngatedNN, SurrogateGatedNN
    from .score import fire_rate
    print("planted :: I just fed two pattern-finders 2,000 days of PURE RANDOM")
    print("           NOISE. There is no pattern in it. I made sure.\n")
    print("           thinking...\n")
    pct = {}
    for cls in (SurrogateGatedNN, UngatedNN):
        m = cls()
        frs = [fire_rate(m.match(make_world(seed=1000 + s, T=2000, regimes=None)[0],
                                 seed=s)[0]) for s in range(5)]
        pct[m.name] = round(100 * mean(frs))
    g, u = pct["surrogate-gated-nn"], pct["ungated-nn"]
    print(f'  surrogate-gated-nn   found a "pattern" in {g:>4}% of it   ->  '
          f'basically said "nothing here"  [ok]')
    print(f'  ungated-nn           found a "pattern" in {u:>4}% of it   ->  '
          f'"PATTERN!" ...in random noise  [x]')
    print('\n  Same math under the hood. The only difference: one is allowed to')
    print('  say "nothing here." The other can\'t — so it hallucinates structure')
    print("  that isn't there. That second one is what looks like genius in a")
    print("  backtest and goes broke on data it has never seen.\n")
    print("  Full scoreboard, with REAL patterns mixed in:  python -m planted bench")


def _bench():
    from .benchmark import leaderboard
    from .methods import ROSTER
    print("planted :: leaderboard — scoring the reference roster\n")
    leaderboard(ROSTER)


def _spx():
    import random
    from statistics import mean
    from .spx_data import returns, SPAN
    from .methods import SurrogateGatedNN, UngatedNN, block_bootstrap
    from .score import fire_rate

    spx = returns()
    d0, d1 = SPAN

    def fires_on(series):
        honest = fire_rate(SurrogateGatedNN().match(series, seed=0)[0])
        tourist = fire_rate(UngatedNN().match(series, seed=0)[0])
        return round(100 * honest), round(100 * tourist)

    # Same finders, run on the actual market...
    h_real, t_real = fires_on(spx)
    # ...and on the same market shuffled into structureless noise.
    shuffles = [block_bootstrap(spx, 10, random.Random(100 + s)) for s in range(5)]
    h_shuf = round(mean(fires_on(s)[0] for s in shuffles))
    t_shuf = round(mean(fires_on(s)[1] for s in shuffles))

    print(f"planted :: this time I ran the same two finders on the REAL S&P 500 —")
    print(f"           every trading day from {d0} to {d1} ({len(spx):,} days).")
    print("           Nobody planted anything here. This is the actual market.\n")
    print(f'  honest finder   said "I\'ve seen this before" about {h_real:>4}% of days')
    print(f'  the tourist     said "I\'ve seen this before" about {t_real:>4}% of days\n')
    print("  Now the tell. I shuffled that same S&P 500 into meaningless noise")
    print("  and asked again:\n")
    print(f"  honest finder   on the shuffled noise: {h_shuf:>4}%   <- a trickle. not fooled.")
    print(f"  the tourist     on the shuffled noise: {t_shuf:>4}%   <- same. can't tell.\n")
    print("  The tourist calls almost every day a repeat of an earlier one — in the")
    print("  real market AND in pure noise. It literally cannot tell them apart. The")
    print("  honest finder barely fires on either: a handful of days, the same low")
    print("  rate you'd get by chance. By its test, day-to-day S&P moves don't repeat")
    print("  in a way you could trade — so it stays quiet, instead of selling you a")
    print("  pattern that isn't there. That restraint is the whole skill.\n")
    print("  How that honest finder works:")
    print("    python -m planted bench")


def _figures():
    from .figures import generate
    generate()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "help"
    try:
        if cmd == "demo":
            _demo()
        elif cmd == "bench":
            _bench()
        elif cmd == "spx":
            _spx()
        elif cmd == "figures":
            _figures()
        else:
            print(__doc__)
    except KeyboardInterrupt:
        # demo/bench run for a while; a clean exit beats a stack trace on Ctrl-C.
        print("\nplanted: interrupted", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
