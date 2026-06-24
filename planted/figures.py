"""
Figures — tiny dependency-free SVG charting.

GitHub renders SVG inline in a README, so we can ship real, sharp vector charts
without numpy or matplotlib. This module knows how to draw exactly two things:
a line chart with an optional shaded error band, and a bar chart. ``generate``
runs the benchmark and writes the README figures.
"""

import html

W, H = 680, 420
ML, MR, MT, MB = 74, 24, 56, 64      # margins
PW, PH = W - ML - MR, H - MT - MB    # plot area

INK = "#1b2733"
GRID = "#e6ebf0"
MUTED = "#5b6b7b"
PALETTE = {"blue": "#2563eb", "red": "#dc2626", "green": "#16a34a",
           "amber": "#d97706"}


def _x(v, lo, hi):
    return ML + (v - lo) / (hi - lo) * PW if hi > lo else ML


def _y(v, lo, hi):
    return MT + (1 - (v - lo) / (hi - lo)) * PH if hi > lo else MT + PH


def _header(title, subtitle):
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
         f'font-family="-apple-system,Segoe UI,Helvetica,Arial,sans-serif">',
         f'<rect width="{W}" height="{H}" fill="white"/>',
         f'<text x="{ML}" y="26" font-size="17" font-weight="700" '
         f'fill="{INK}">{html.escape(title)}</text>']
    if subtitle:
        s.append(f'<text x="{ML}" y="44" font-size="12" fill="{MUTED}">'
                 f'{html.escape(subtitle)}</text>')
    return s


def _y_axis(s, ylo, yhi, fmt="{:.2f}", ticks=5):
    for k in range(ticks + 1):
        v = ylo + (yhi - ylo) * k / ticks
        y = _y(v, ylo, yhi)
        s.append(f'<line x1="{ML}" y1="{y:.1f}" x2="{ML + PW}" y2="{y:.1f}" '
                 f'stroke="{GRID}"/>')
        s.append(f'<text x="{ML - 10}" y="{y + 4:.1f}" font-size="11" '
                 f'text-anchor="end" fill="{MUTED}">{fmt.format(v)}</text>')


def _legend(s, items):
    x = ML
    y = H - 18
    for label, color in items:
        s.append(f'<rect x="{x}" y="{y - 9}" width="12" height="12" rx="2" '
                 f'fill="{color}"/>')
        s.append(f'<text x="{x + 18}" y="{y + 1}" font-size="12" '
                 f'fill="{INK}">{html.escape(label)}</text>')
        x += 24 + 8 * len(label) + 16
    return s


def line_chart(title, subtitle, xlabel, ylabel, series, ylo, yhi,
               xlo=0.0, xhi=1.0, refs=()):
    """series: list of dicts {label, color, points:[(x,y)], band:[(x,lo,hi)]?}.
    refs: list of (yvalue, color, dash-label) horizontal reference lines."""
    s = _header(title, subtitle)
    _y_axis(s, ylo, yhi)
    # x ticks
    steps = 5
    for k in range(steps + 1):
        xv = xlo + (xhi - xlo) * k / steps
        x = _x(xv, xlo, xhi)
        s.append(f'<text x="{x:.1f}" y="{MT + PH + 20}" font-size="11" '
                 f'text-anchor="middle" fill="{MUTED}">{xv:.1f}</text>')
    # axis titles
    s.append(f'<text x="{ML + PW / 2:.0f}" y="{H - 38}" font-size="12" '
             f'text-anchor="middle" fill="{INK}">{html.escape(xlabel)}</text>')
    s.append(f'<text transform="translate(20,{MT + PH / 2:.0f}) rotate(-90)" '
             f'font-size="12" text-anchor="middle" '
             f'fill="{INK}">{html.escape(ylabel)}</text>')
    # reference lines
    for yv, color, lab in refs:
        y = _y(yv, ylo, yhi)
        s.append(f'<line x1="{ML}" y1="{y:.1f}" x2="{ML + PW}" y2="{y:.1f}" '
                 f'stroke="{color}" stroke-dasharray="5 4" stroke-width="1.3"/>')
        s.append(f'<text x="{ML + PW - 4}" y="{y - 5:.1f}" font-size="11" '
                 f'text-anchor="end" fill="{color}">{html.escape(lab)}</text>')
    # series
    for ser in series:
        color = PALETTE.get(ser["color"], ser["color"])
        band = ser.get("band")
        if band:
            top = " ".join(f"{_x(x, xlo, xhi):.1f},{_y(hi, ylo, yhi):.1f}"
                           for x, lo, hi in band)
            bot = " ".join(f"{_x(x, xlo, xhi):.1f},{_y(lo, ylo, yhi):.1f}"
                           for x, lo, hi in reversed(band))
            s.append(f'<polygon points="{top} {bot}" fill="{color}" '
                     f'fill-opacity="0.13"/>')
        pts = " ".join(f"{_x(x, xlo, xhi):.1f},{_y(y, ylo, yhi):.1f}"
                       for x, y in ser["points"])
        s.append(f'<polyline points="{pts}" fill="none" stroke="{color}" '
                 f'stroke-width="2.4"/>')
        for x, y in ser["points"]:
            s.append(f'<circle cx="{_x(x, xlo, xhi):.1f}" '
                     f'cy="{_y(y, ylo, yhi):.1f}" r="3.2" fill="{color}"/>')
    _legend(s, [(ser["label"], PALETTE.get(ser["color"], ser["color"]))
                for ser in series])
    s.append("</svg>")
    return "\n".join(s)


def bar_chart(title, subtitle, ylabel, bars, ylo, yhi, refs=()):
    """bars: list of (label, value, color)."""
    s = _header(title, subtitle)
    _y_axis(s, ylo, yhi, fmt="{:.0%}")
    s.append(f'<text transform="translate(20,{MT + PH / 2:.0f}) rotate(-90)" '
             f'font-size="12" text-anchor="middle" '
             f'fill="{INK}">{html.escape(ylabel)}</text>')
    for yv, color, lab in refs:
        y = _y(yv, ylo, yhi)
        s.append(f'<line x1="{ML}" y1="{y:.1f}" x2="{ML + PW}" y2="{y:.1f}" '
                 f'stroke="{color}" stroke-dasharray="5 4" stroke-width="1.3"/>')
        s.append(f'<text x="{ML + PW - 4}" y="{y - 5:.1f}" font-size="11" '
                 f'text-anchor="end" fill="{color}">{html.escape(lab)}</text>')
    n = len(bars)
    slot = PW / n
    bw = slot * 0.5
    for i, (label, val, color) in enumerate(bars):
        c = PALETTE.get(color, color)
        cx = ML + slot * (i + 0.5)
        y = _y(val, ylo, yhi)
        h = _y(ylo, ylo, yhi) - y
        s.append(f'<rect x="{cx - bw / 2:.1f}" y="{y:.1f}" width="{bw:.1f}" '
                 f'height="{h:.1f}" rx="3" fill="{c}"/>')
        s.append(f'<text x="{cx:.1f}" y="{y - 7:.1f}" font-size="12" '
                 f'font-weight="700" text-anchor="middle" '
                 f'fill="{c}">{val:.0%}</text>')
        s.append(f'<text x="{cx:.1f}" y="{MT + PH + 20}" font-size="11.5" '
                 f'text-anchor="middle" fill="{INK}">{html.escape(label)}</text>')
    s.append("</svg>")
    return "\n".join(s)


def generate(outdir="figures", n_seeds=10):
    """Run the benchmark and write the README figures. Returns the file paths."""
    import os
    from .benchmark import run_benchmark
    from .methods import UngatedNN, SurrogateGatedNN

    print("  generating figures (running benchmark) ...", flush=True)
    gated = run_benchmark(SurrogateGatedNN(), n_seeds=n_seeds,
                          progress=lambda sep: print(f"    gated sep={sep:.1f}",
                                                     flush=True))
    # Hero figure: the a-ha — how often each method "finds a pattern" in PURE
    # NOISE (sep=0 fire-rate). Two players, plain labels, lower is more honest.
    noise_bars = []
    for cls, label, color in [(SurrogateGatedNN, "The honest method", "blue"),
                              (UngatedNN, "The hallucinator", "red")]:
        res = run_benchmark(cls(), seps=(0.0,), n_seeds=n_seeds)
        noise_bars.append((label, res["fire_rate"][0][1], color))
    noise_bars.sort(key=lambda b: b[1])

    os.makedirs(outdir, exist_ok=True)
    paths = {}

    hero = bar_chart(
        'How often each "found a pattern" in PURE RANDOM NOISE',
        "There is nothing to find. Lower is more honest.",
        "of pure noise flagged as a “pattern”",
        noise_bars, 0.0, 1.0,
        refs=[(0.05, MUTED, "honest floor (5%)")])
    paths["hallucination"] = f"{outdir}/hallucination.svg"
    open(paths["hallucination"], "w").write(hero)

    rec = gated["recovery"]
    rec_series = [{
        "label": "surrogate-gated-nn", "color": "blue",
        "points": [(sep, v) for sep, v, _ in rec],
        "band": [(sep, v - e, v + e) for sep, v, e in rec],
    }]
    rec_fig = line_chart(
        "Recovery: real skill only when real structure exists",
        "Chance-corrected skill vs structure strength (band = +/-1 SE). "
        "At sep=0 (pure noise) skill sits at ~0 — no fooling itself.",
        "structure strength (0 = pure noise, 1 = strong regimes)",
        "recovery skill", rec_series, -0.1, 0.5,
        refs=[(0.0, MUTED, "no skill")])
    paths["recovery"] = f"{outdir}/recovery.svg"
    open(paths["recovery"], "w").write(rec_fig)

    fr = gated["fire_rate"]
    fr_series = [{
        "label": "surrogate-gated-nn", "color": "blue",
        "points": [(sep, v) for sep, v, _ in fr],
    }]
    fr_fig = line_chart(
        "Fire-rate: pinned at the noise floor, rising only with structure",
        "Fraction of windows fired vs structure strength. Sits at alpha on "
        "noise, climbs as genuine regimes appear.",
        "structure strength (0 = pure noise, 1 = strong regimes)",
        "fire-rate", fr_series, 0.0, 0.25,
        refs=[(0.05, MUTED, "alpha = 0.05")])
    paths["fire_rate"] = f"{outdir}/fire_rate.svg"
    open(paths["fire_rate"], "w").write(fr_fig)

    for k, p in paths.items():
        print(f"    wrote {p}", flush=True)
    return paths
