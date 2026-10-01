#!/usr/bin/env python3
"""Figure 3 of the paper, selection. One column, one record per source.

    python3 figures/make_regret.py

For each task of Table I, the accuracy of the arrangement each selection rule
serves, averaged over the three seeds, with a black mark at the better of the
two servable arrangements chosen per seed. A rule that always picks the better
arrangement reaches the mark.

  text tasks   results/accuracy_text/results.csv
  CIFAR-100    results/accuracy_vision/results.csv

The counts over all 27 cells, CIFAR-10 partitions included, are printed for the
text and come from results/accuracy_vision/cifar10_matched_full.csv too.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import heoft_plot as hp  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SHARED, PERSONAL = "A_headonly", "B_personal"
RULES = [  # legend label, mode in the records, colour
    ("vote", "sel_federated", hp.TERRA),
    ("always shared", SHARED, hp.GREY),
    ("estimator", "sel_gp_rarefill", hp.BLUE),
]
TASKS = [("ag_news", "AG-News"), ("trec", "TREC"), ("dbpedia_14", "DBpedia"),
         ("banking77", "Banking77"), ("cifar100", "CIFAR-100")]


def load():
    out = defaultdict(lambda: defaultdict(dict))
    for x in csv.DictReader(open(REPO / "results/accuracy_text/results.csv")):
        out[x["task"]][x["seed"]][x["mode"]] = float(x["acc_mean"])
    for x in csv.DictReader(open(REPO / "results/accuracy_vision/results.csv")):
        out[x["dataset"]][x["seed"]][x["mode"]] = float(x["acc_mean"])
    return out


def mean(v):
    return sum(v) / len(v)


def main():
    d = load()
    fig, ax = hp.column(ratio=0.55)
    width = 0.26
    for i, (label, mode, colour) in enumerate(RULES):
        xs, ys = [], []
        for t, (task, _) in enumerate(TASKS):
            xs.append(t + (i - 1) * width)
            ys.append(mean([s[mode] for s in d[task].values()]))
        ax.bar(xs, ys, width=width, color=colour, label=label, linewidth=0)
        print(label, [round(y, 3) for y in ys])
    for t, (task, _) in enumerate(TASKS):
        best = mean([max(s[SHARED], s[PERSONAL]) for s in d[task].values()])
        ax.plot([t - 1.5 * width, t + 1.5 * width], [best, best], color="black", linewidth=1.0)
        print("better", task, round(best, 3))
    ax.set_xticks(range(len(TASKS)))
    ax.set_xticklabels([name for _, name in TASKS])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy served")
    ax.legend(loc="upper left", ncol=3, frameon=False, handlelength=1.0, columnspacing=0.8)
    hp.save(fig, "fig_regret.pdf")


if __name__ == "__main__":
    main()
