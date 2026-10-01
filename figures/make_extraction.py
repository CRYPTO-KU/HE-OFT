#!/usr/bin/env python3
"""Figure 4 of the paper, extraction. One column, one record.

    python3 figures/make_extraction.py

Fidelity of a copy of the served head against the queries spent, per parameter
of the head (C d), for the label-only interface HE-OFT serves and for an
interface that returns scores. Mean over three seeds and both arrangements.

  results/extraction_budget/results.csv
    strategy random   labels, a linear fit to uniformly drawn queries
    strategy logits   scores, a linear solve from d + 1 answers
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import heoft_plot as hp  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TASKS = [("ag_news", "AG-News", hp.BLUE, "o"), ("dbpedia_14", "DBpedia", hp.SAGE, "s"),
         ("banking77", "Banking77", hp.TERRA, "^")]


def main():
    rows = list(csv.DictReader(open(REPO / "results/extraction_budget/results.csv")))
    fig, ax = hp.column(ratio=0.52)
    hollow = []   # one hollow marker per task, gathered into one legend entry
    for task, name, col, mk in TASKS:
        acc = defaultdict(list)
        for r in rows:
            if r["task"] == task and r["strategy"] == "random":
                acc[int(r["queries"])].append(float(r["fidelity"]))
        par = next(int(r["C"]) * int(r["d"]) for r in rows if r["task"] == task)
        pts = sorted((q / par, sum(v) / len(v)) for q, v in acc.items() if q > 0)
        ax.plot(*zip(*pts), marker=mk, color=col, label=name)
        sc = [(int(r["queries"]) / par, float(r["fidelity"])) for r in rows
              if r["task"] == task and r["strategy"] == "logits"]
        q, f = sc[0][0], sum(x[1] for x in sc) / len(sc)
        # scores: an open marker at d + 1 answers, the linear solve
        hollow += ax.plot([q], [f], marker=mk, color=col, markerfacecolor="white",
                          markeredgecolor=col, markeredgewidth=1.2,
                          linestyle="none", markersize=6)
        print(name, "labels", [(round(x, 2), round(y, 3)) for x, y in pts],
              "scores", round(q, 3), round(f, 3))
    hp.logx(ax)
    # plain tick labels, because 10^-2 sets its exponent below the caption size
    ax.set_xticks([0.01, 0.1, 1, 10, 100])
    ax.set_xticklabels(["0.01", "0.1", "1", "10", "100"])
    ax.minorticks_off()
    ax.set_xlabel("queries per parameter of the head")
    ax.set_ylabel("fidelity of the copy")
    ax.set_ylim(0, 1.05)
    # colour and shape are the task; a line is labels, a hollow marker scores.
    # The scores entry shows all three hollow markers, so each one in the plot
    # can be matched to its task.
    from matplotlib.legend_handler import HandlerTuple
    h, l = ax.get_legend_handles_labels()
    h += [tuple(hollow)]
    l += ["scores, $d+1$ answers"]
    ax.legend(h, l, loc="lower right", frameon=False, handlelength=2.4,
              labelspacing=0.2, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)})
    hp.save(fig, "fig_extraction.pdf")


if __name__ == "__main__":
    main()
