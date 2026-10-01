"""Plot style for the experiments section.

One import, one palette, one figure size. Every plot in the paper and in the
technical report goes through this module, so a font size is decided once.

    import heoft_plot as hp
    fig, ax = hp.panels(4)                 # the full-width four-panel figure
    ...
    hp.save(fig, "fig_trends.pdf")

Sizes come from page.yml, measured off IEEEtran with the journal option. A label in a saved figure is 8pt on the page, the same size as the
caption beneath it, because the figure is saved at its final width and included
with no width option. Never pass width= to \\includegraphics for these, and
never \\resizebox them. Both rescale the fonts and the plot stops matching the
paper.

Put as little text inside a plot as possible. A panel carries its axis labels
and its data. Everything else goes to the legend or to the caption, and the
caption states the comparison and the conditions and nothing else. Comment and
analysis belong in the body of the section.
"""
from pathlib import Path

import matplotlib.pyplot as plt

import plotstyle as ps

_HERE = Path(__file__).resolve().parent
ps.use(_HERE / "page.yml")
plt.style.use(_HERE / "paper.mplstyle")

# Every glyph in a figure is 8pt, the caption size. paper.mplstyle drops ticks
# and legends to 7, and this override raises them, so nothing in a plot is
# smaller than the caption beneath it.
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "figure.titlesize": 8,
})

# The SANZO palette, the same hex values as the paper's LaTeX color
# definitions. A figure and a diagram that sit on the same page use
# the same blue.
BLUE = "#5B7FA6"        # paperblue, the protocol
SAGE = "#5B7B6E"        # paperneutral, the reference or the ceiling
TERRA = "#D4826A"       # paperamber, the contrast
TAN = "#C6A87D"         # sanzotan, the clients
GREY = "#8B9EA8"        # sanzobluegrey, a floor or a chance line
RAMP = ["#D3DEEA", "#A9BDD3", "#7B9BBF", "#5B7FA6"]

# One name for one thing, so a series is the same colour in every panel.
SERIES = {
    "selected": BLUE,      # A_sel, the arrangement the estimator picks
    "alone": TAN,          # A_loc, a client on its own
    "disclosed": SAGE,     # A_dis, the model this protocol declines to build
    "pooled": GREY,        # A_pool, the centralised ceiling
    "cost": TERRA,
}


def panels(n=4, ratio=0.70, width="text"):
    """A row of n panels at the width they will occupy on the page.

    n=4 at text width gives four panels across both columns, which costs about
    0.19 of a page against 0.27 for the same four as separate column figures.
    """
    fig, axes = plt.subplots(
        1, n, figsize=(ps.width_in(width), ps.width_in(width) * ratio / n))
    return fig, (axes if n > 1 else [axes])


def column(ratio=0.68):
    """One panel, one column wide."""
    return ps.figure(width="column", ratio=ratio)


def label(ax, letter):
    """Panel letters go top left, at caption size, with no box."""
    ax.set_title(f"({letter})", loc="left")


def save(fig, name):
    """Save into figures/ and report whether the fonts match the caption."""
    ps.save(fig, str(_HERE / name))


logx, logy, band, finish = ps.logx, ps.logy, ps.band, ps.finish
