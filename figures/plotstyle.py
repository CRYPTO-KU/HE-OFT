#!/usr/bin/env python3
"""Paper figures whose fonts and width match the document exactly.

The same rule as draw.io figures: size the figure to the real column width in
inches, set the font sizes in points, save as PDF, and include it with NO width
option. Then `8pt` in the figure is 8pt on the page.

    import plotstyle as ps
    ps.use("paper/.paper-meta.yml")

    fig, ax = ps.figure(width="column", ratio=0.68)
    ax.plot(x, y, color=ps.C[0], label="ours")
    ax.set_xlabel("Training steps"); ax.set_ylabel("Max logits")
    ax.legend()
    ps.save(fig, "paper/figures/max-logits.pdf")

`save` prints the exact \\includegraphics line to paste. Never add a width to
it: `width=\\columnwidth` rescales the figure and its fonts with it.

Self-test:  python3 plotstyle.py --demo out.pdf
"""
import pathlib
import sys

import matplotlib
import matplotlib.pyplot as plt

_HERE = pathlib.Path(__file__).resolve().parent
_STYLE = _HERE.parent / "templates" / "paper.mplstyle"

# Ordered categorical palette. The first three are the most distinguishable
# from each other and in grayscale; use them for two- and three-series plots.
# Tuned against the Kimi K2 figure aesthetic, nudged for deuteranopia.
C = [
    "#4E79A7",  # 0 blue      baseline / reference
    "#E1575A",  # 1 red       ours / the thing being argued for
    "#F0A64B",  # 2 amber     second baseline
    "#5BA36B",  # 3 green     ablation
    "#8E6BB5",  # 4 purple
    "#4BB3B8",  # 5 teal
    "#B5793F",  # 6 brown
    "#9AA1AC",  # 7 gray      "everything else", never a headline series
]

# Softer variants, for fills, bands and confidence intervals.
C_LIGHT = ["#A8C3DC", "#F0AAAB", "#F8D2A5", "#AED4B7",
           "#C7B5DA", "#A5D9DB", "#DABC9F", "#CDD2D8"]

GRAY = "#9AA1AC"
INK = "#202020"

# Sequential and diverging ramps, for heatmaps.
SEQ = "viridis"
DIV = "RdBu_r"

_META = {}

# Fallback widths in points when no .paper-meta.yml is available.
_DEFAULT = {"column_width_pt": 241.14749, "text_width_pt": 506.295,
            "caption_pt": 8, "body_pt": 9, "font_family": "Times New Roman"}

_PT_PER_IN = 72.27  # TeX points, which is what \columnwidth reports


def _read_meta(path):
    meta = dict(_DEFAULT)
    if path is None:
        return meta
    p = pathlib.Path(path)
    if not p.exists():
        print(f"plotstyle: no {p}, using defaults "
              f"({_DEFAULT['column_width_pt']:.1f}pt column)", file=sys.stderr)
        return meta
    try:
        import yaml
        raw = yaml.safe_load(p.read_text()) or {}
    except ImportError:                      # parse the handful of keys we need
        raw = {}
        for line in p.read_text().split("\n"):
            if ":" in line and not line.strip().startswith("#"):
                k, _, v = line.partition(":")
                raw[k.strip()] = v.split("#")[0].strip()
    for k in _DEFAULT:
        if raw.get(k) not in (None, "", "FILL"):
            meta[k] = raw[k]
    for k in ("column_width_pt", "text_width_pt", "caption_pt", "body_pt"):
        try:
            meta[k] = float(meta[k])
        except (TypeError, ValueError):
            meta[k] = _DEFAULT[k]
    return meta


def use(meta_path="paper/.paper-meta.yml"):
    """Apply the style, with font family and sizes taken from the paper."""
    global _META
    _META = _read_meta(meta_path)
    if _STYLE.exists():
        plt.style.use(str(_STYLE))

    cap = _META["caption_pt"]
    fam = str(_META["font_family"])
    # Serif families that stand in for the paper's, in order of preference.
    serif = [fam, "Times New Roman", "Nimbus Roman", "Liberation Serif",
             "STIXGeneral", "DejaVu Serif"]
    sans = [fam, "Helvetica", "Arial", "Nimbus Sans", "DejaVu Sans"]
    is_sans = any(s in fam.lower() for s in ("sans", "helvetica", "arial"))

    matplotlib.rcParams.update({
        "font.family": "sans-serif" if is_sans else "serif",
        "font.serif": serif,
        "font.sans-serif": sans,
        # Math in a figure should match the document's math, not matplotlib's.
        "mathtext.fontset": "dejavusans" if is_sans else "stix",
        "font.size": cap,
        "axes.labelsize": cap,
        "axes.titlesize": cap,
        "xtick.labelsize": cap - 1,
        "ytick.labelsize": cap - 1,
        "legend.fontsize": cap - 1,
        "axes.prop_cycle": matplotlib.cycler(color=C),
    })
    return _META


def width_in(width="column"):
    """Figure width in inches for 'column', 'text', a point value, or inches."""
    if not _META:
        use(None)
    if width == "column":
        return _META["column_width_pt"] / _PT_PER_IN
    if width in ("text", "full"):
        return _META["text_width_pt"] / _PT_PER_IN
    w = float(width)
    return w / _PT_PER_IN if w > 12 else w      # >12 reads as points


def figure(width="column", ratio=0.68, nrows=1, ncols=1, **kw):
    """A figure sized to the real column width. ratio is height/width."""
    if not _META:
        use(None)
    w = width_in(width)
    return plt.subplots(nrows, ncols, figsize=(w, w * ratio), **kw)


def save(fig, path, verbose=True):
    """Save as vector PDF and print the include line."""
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, format="pdf")
    if verbose:
        stem = p.with_suffix("").as_posix()
        for pre in ("paper/", "./"):
            if stem.startswith(pre):
                stem = stem[len(pre):]
        print(f"wrote {p}")
        print(f"  \\includegraphics{{{stem}}}   <- no width option, ever")
    plt.close(fig)
    return p


def logx(ax, minor_labels=False):
    """Log x-axis without the 2x10^10 minor-tick clutter."""
    ax.set_xscale("log")
    if not minor_labels:
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    return ax


def logy(ax, minor_labels=False):
    ax.set_yscale("log")
    if not minor_labels:
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    return ax


def band(ax, x, lo, hi, color, alpha=0.18, **kw):
    """A dispersion band. Always pair a mean line with one of these."""
    return ax.fill_between(x, lo, hi, color=color, alpha=alpha,
                           linewidth=0, **kw)


def finish(ax, xlabel=None, ylabel=None, legend=True, loc="best"):
    """The touches every paper figure wants, applied consistently."""
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    for s in ax.spines.values():
        s.set_linewidth(0.7)
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(loc=loc)
    return ax


def _demo(out):
    import math
    use(None)
    fig, (a, b) = figure(width="text", ratio=0.34, ncols=2)

    xs = list(range(0, 16000, 100))
    a.plot(xs, [0.004 * (x / 1000) ** 3.1 + 2 for x in xs],
           color=C[1], label="Vanilla run")
    a.set_yscale("linear")
    finish(a, "Training steps", "Max logits", loc="upper left")

    for i, (fl, lab) in enumerate([(1.2, "1.2e+20 FLOPs"), (2.2, "2.2e+20 FLOPs"),
                                   (4.5, "4.5e+20 FLOPs"), (9.0, "9.0e+20 FLOPs")]):
        t = [10 ** (10.3 + 0.06 * k) for k in range(12)]
        y = [1.75 - 0.055 * math.log10(fl) - 0.09 * math.log10(v / 4e10)
             + 0.22 * (math.log10(v / 6e10)) ** 2 for v in t]
        b.plot(t, y, ":", marker="s", color=C[i], label=lab, markersize=3)
    logx(b)                      # suppresses the 2x10^10 minor-tick clutter
    finish(b, "Training tokens", "Validation loss", loc="lower left")

    save(fig, out)


if __name__ == "__main__":
    if "--demo" in sys.argv:
        _demo(sys.argv[sys.argv.index("--demo") + 1])
    else:
        print(__doc__)
