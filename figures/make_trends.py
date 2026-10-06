#!/usr/bin/env python3
"""Figure 4 of the technical report. Four panels, full width, one record behind each.

    python3 figures/make_trends.py

  (a) accuracy against the number of clients   results/accuracy_text/nsweep.csv
  (b) accuracy against label skew              results/accuracy_text/sensitivity.csv
  (c) accuracy against local steps             results/accuracy_text/sensitivity.csv
  (d) test time against the label space        results/fhe/cost_vs_d.csv, d = 768
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import heoft_plot as hp  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SEL, LOC, DIS = "sel_gp_rarefill", "local", "current"


def rows(p):
    return list(csv.DictReader(open(REPO / p)))


def mean_by(rs, key, mode):
    """mean accuracy over seeds, keyed by one axis, for one mode"""
    acc = defaultdict(list)
    for r in rs:
        if r["mode"] == mode:
            acc[float(r[key])].append(float(r["acc_mean"]))
    return sorted((k, sum(v) / len(v)) for k, v in acc.items())


# The deck asks for figures that stay readable in black and white. In greyscale
# the blue of "selected" and the sage of "disclosed" land on the same mid-grey,
# so each series carries a line style as well as a colour.
DASH = {"selected": "-", "disclosed": (0, (5, 2)), "alone": (0, (1, 1.6)),
        "pooled": (0, (3, 1.5, 1, 1.5)), "cost": (0, (5, 2))}


def line(ax, pts, name, marker):
    x, y = zip(*pts)
    ax.plot(x, y, marker=marker, color=hp.SERIES[name], label=name,
            linestyle=DASH.get(name, "-"))


def main():
    ns = rows("results/accuracy_text/nsweep.csv")
    sn = rows("results/accuracy_text/sensitivity.csv")

    fig, ax = hp.panels(4, ratio=0.78, width="text")

    # (a) federation size. One seed, which the table already states.
    for m, name, mk in ((SEL, "selected", "o"), (DIS, "disclosed", "s"),
                        (LOC, "alone", "^")):
        line(ax[0], mean_by(ns, "N", m), name, mk)
    ax[0].set_xscale("log")
    ax[0].set_xticks([10, 20, 50])
    ax[0].set_xticklabels(["10", "20", "50"])
    ax[0].minorticks_off()   # a log axis labels minor ticks as 2x10^1
    ax[0].set_xlabel(r"clients $N$")
    ax[0].set_ylabel("accuracy")
    hp.label(ax[0], "a")

    # (b) label skew, at the default K. Mean over three seeds.
    #
    # sensitivity.csv holds the skew axis at alpha 0.05, 0.30 and 1.00. The
    # default cell, alpha = 0.10, comes from the main accuracy record, three
    # seeds. Panel (b) shows three-seed means, panels (a) and (c) seed 42.
    hl = [r for r in rows("results/accuracy_text/results.csv")
          if r["task"] == "dbpedia_14"]
    sk = [r for r in sn if r["K"] == "200"]
    sk += [dict(r, alpha="0.1") for r in hl]
    for m, name, mk in ((SEL, "selected", "o"), (DIS, "disclosed", "s"),
                        (LOC, "alone", "^")):
        line(ax[1], mean_by(sk, "alpha", m), name, mk)
    ax[1].set_xscale("log")
    ax[1].set_xticks([0.05, 0.1, 0.3, 1.0])
    ax[1].set_xticklabels(["0.05", "0.1", "0.3", "1"])
    ax[1].minorticks_off()
    ax[1].set_xlabel(r"label skew $\alpha$")
    hp.label(ax[1], "b")

    # (c) local steps, at the default skew. Seed 42, and its default cell is
    # that seed's value, which is the N=10 row of the client sweep.
    st = [r for r in sn if r["alpha"] == "0.1"]
    st += [dict(r, K="200") for r in ns if r["N"] == "10"]
    for m, name, mk in ((SEL, "selected", "o"), (DIS, "disclosed", "s"),
                        (LOC, "alone", "^")):
        line(ax[2], mean_by(st, "K", m), name, mk)
    ax[2].set_xticks([100, 200, 400])
    ax[2].set_xlabel(r"local steps $K$")
    hp.label(ax[2], "c")

    # (d) the test time of one query against the label space, at N = 10 and
    # d = 768, the rows of cost_vs_d.csv that Table 7 and the
    # abstract use. Four curves: colour is what the querier receives, the line
    # style says whether the time is measured on one core or projected on a GPU.
    #   label, CPU     query_total_ms, measured end to end
    #   label, GPU     each server bootstrap replaced by the 2.28 s measured
    #                  under Phantom on an H100, the rest left on the processor
    #   scores, CPU    query encryption + head application + key switch +
    #                  querier decryption, the query without the argmax
    #   scores, GPU    the head's products and rotations at Phantom's published
    #                  A100 figures at ring degree 2^16, 8.5647 and 8.4920 ms,
    #                  the rest left on the processor
    q = sorted((r for r in rows("results/fhe/cost_vs_d.csv")
                if r["d"] == "768"), key=lambda r: int(r["C"]))
    C = [int(r["C"]) for r in q]
    ms = lambda r, *k: sum(float(r[x]) for x in k) / 1000
    lab_cpu = [ms(r, "query_total_ms") for r in q]
    lab_gpu = [int(r["server_bootstraps"]) * 2.28
               + ms(r, "query_total_ms") - ms(r, "in_bootstrap_ms") for r in q]
    sco_cpu = [ms(r, "query_encrypt_ms", "head_apply_ms", "key_switch_ms",
                  "querier_decrypt_ms") for r in q]
    sco_gpu = [ms(r, "query_encrypt_ms", "key_switch_ms", "querier_decrypt_ms")
               + int(r["head_ct_ct_products"]) * 8.5647e-3
               + int(r["head_rotations"]) * 8.4920e-3 for r in q]
    # Distinct colours, because panel (d) measures seconds and the other three
    # measure accuracy. One name for one thing extends to colour.
    for ys, col, mk, ls, lab in (
            (lab_cpu, hp.TERRA, "o", "-", "label, CPU"),
            (lab_gpu, hp.TERRA, "o", (0, (5, 2)), "label, GPU"),
            (sco_cpu, hp.GREY, "s", "-", "scores, CPU"),
            (sco_gpu, hp.GREY, "s", (0, (5, 2)), "scores, GPU")):
        ax[3].plot(C, ys, marker=mk, color=col, linestyle=ls, label=lab)
    ax[3].set_xscale("log")
    ax[3].set_yscale("log")
    # 77 and 100 are too close on a log axis for two labels, so 77 keeps its
    # marker and loses its label, as in the other panels' unlabelled points.
    ax[3].set_xticks([4, 14, 100])
    ax[3].set_xticklabels(["4", "14", "100"])
    ax[3].set_yticks([1, 10, 100, 1000])
    ax[3].set_yticklabels(["1", "10", "100", "1000"])
    ax[3].minorticks_off()
    ax[3].set_ylim(0.4, 3000)
    ax[3].set_xlabel("classes")
    ax[3].set_ylabel("seconds per query")
    hp.label(ax[3], "d")
    print("\npanel (d), seconds, C =", C)
    for name, ys in (("label CPU", lab_cpu), ("label GPU", lab_gpu),
                     ("scores CPU", sco_cpu), ("scores GPU", sco_gpu)):
        print(f"  {name:11s}", [round(y, 2) for y in ys])

    # Panels (a) to (c) draw the same three series, so they share one legend
    # above the row. Inside any panel it covers data. Panel (d) draws different
    # series and keeps its own, in the corner its curves leave empty.
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.40, 1.17), columnspacing=1.4)
    # Panel (d)'s four curves all rise across the panel and leave no free
    # corner, so its legend sits above it, two by two, level with the shared one.
    h4, l4 = ax[3].get_legend_handles_labels()
    fig.legend(h4, l4, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.87, 1.27), columnspacing=1.0,
               handlelength=1.8, labelspacing=0.2)
    hp.save(fig, "fig_trends.pdf")

    # the numbers the caption and the prose must agree with
    print("\nselected, by axis")
    print("  N     ", [(int(k), round(v, 3)) for k, v in mean_by(ns, "N", SEL)])
    print("  alpha ", [(k, round(v, 3)) for k, v in mean_by(sk, "alpha", SEL)])
    print("  K     ", [(int(k), round(v, 3)) for k, v in mean_by(st, "K", SEL)])


if __name__ == "__main__":
    main()
