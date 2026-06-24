"""
Command line entry point.

    python -m planted demo                 the a-ha: honesty vs hallucination on noise
    python -m planted bench                full leaderboard, with real patterns mixed in
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
