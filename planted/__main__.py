"""
Command line entry point.

    python -m planted demo                 quick practitioner-vs-tourist contrast
    python -m planted bench                full leaderboard over the roster
    python -m planted explore FILE [...]   surrogate-test a method on YOUR data
    python -m planted figures              regenerate the README figures

Run with no arguments for this help.
"""

import sys


def _demo():
    from .methods import UngatedNN, SurrogateGatedNN
    from .benchmark import run_benchmark
    print("planted :: demo — does the method know when to say 'nothing here'?\n")
    for cls in (SurrogateGatedNN, UngatedNN):
        card = run_benchmark(cls(), n_seeds=8)["scorecard"]
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


def _explore(args):
    from .data import load_returns, explore, format_explore
    from .methods import SurrogateGatedNN
    if not args:
        print("usage: python -m planted explore FILE [--col NAME|IDX] "
              "[--kind auto|price|return]")
        return
    path = args[0]
    col, kind = None, "auto"
    i = 1
    while i < len(args):
        if args[i] == "--col" and i + 1 < len(args):
            col = args[i + 1]
            col = int(col) if col.lstrip("-").isdigit() else col
            i += 2
        elif args[i] == "--kind" and i + 1 < len(args):
            kind = args[i + 1]
            i += 2
        else:
            i += 1
    rets = load_returns(path, col=col, kind=kind)
    print(f"planted :: explore — surrogate test on {path}\n")
    print(format_explore(explore(rets, method=SurrogateGatedNN())))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "help"
    if cmd == "demo":
        _demo()
    elif cmd == "bench":
        _bench()
    elif cmd == "figures":
        _figures()
    elif cmd == "explore":
        _explore(argv[1:])
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
