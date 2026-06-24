"""
Command line entry point.

    python -m planted demo                 quick practitioner-vs-tourist contrast
    python -m planted bench                full leaderboard over the roster
    python -m planted figures              regenerate the README figures

Run with no arguments for this help.
"""

import sys


def _demo():
    from .methods import UngatedNN, SurrogateGatedNN
    from .benchmark import run_benchmark
    print("planted :: demo — does the method know when to say 'nothing here'?\n")
    for cls in (SurrogateGatedNN, UngatedNN):
        card = run_benchmark(cls(), n_seeds=12)["scorecard"]
        print(f"  {card['method']:<20} "
              f"composite={card['composite']:.3f}  "
              f"recovery(structure)={card['struct_recovery']:.3f}  "
              f"fire-rate(noise)={card['noise_fire_rate']:.3f}")
    print("\n  The tourist (ungated) finds 'analogs' even in pure noise, so its")
    print("  composite collapses. Only abstaining on noise earns the score.")


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
