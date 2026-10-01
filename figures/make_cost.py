"""The per-operation cost figure of the technical report, from the Lattigo grid.

Reads results/fhe/cost_grid.json, which holds every protocol operation
over the cross product of ring degree and federation size, and writes
figures/fig_cost.pdf.

Left panel: the per-query arithmetic, which the ring degree sets and the number
of clients does not touch. Right panel: the key switch that returns a label,
which is the one per-query operation that grows with both, because every client
contributes one share.

One measurement source throughout, so the two panels are read against each
other. The encrypted reciprocal is not shown: it runs on a deeper chain at a
different ring degree, and it is paid once per aggregation rather than per query.

Run:  python figures/make_cost.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
GRID = ROOT / "results" / "fhe" / "cost_grid.json"

TAN, BLUE, SAGE, GREY, TERRA = "#C6A87D", "#5B7FA6", "#5B7B6E", "#8B9EA8", "#D4826A"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42,
})

RING_OPS = [                                   # measured at the default N = 10
    ("ct_x_ct_mul_relin_rescale_ms", "product, both encrypted", BLUE),
    ("rotation_ms",                  "rotation",                SAGE),
    ("pt_x_ct_mul_rescale_ms",       "product, one plaintext",  TAN),
    ("ciphertext_add_ms",            "addition",                GREY),
]
NS = [(5, TAN), (10, BLUE), (20, TERRA)]       # key switch, one series per N
LOGNS = [14, 15, 16]


def label(v):
    return f"{v:,.0f}" if v >= 100 else f"{v:.1f}"


def main():
    rows = json.loads(GRID.read_text())
    at = {(r["log_n"], r["n_parties"]): r for r in rows}
    idx = np.arange(len(LOGNS))

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)

    w = 0.78 / len(RING_OPS)
    for k, (key, name, colour) in enumerate(RING_OPS):
        vals = [at[(g, 10)][key] for g in LOGNS]
        pos = idx + (k - (len(RING_OPS) - 1) / 2) * w
        axl.bar(pos, vals, w, color=colour, label=name, edgecolor="none")
        for p, v in zip(pos, vals):
            axl.annotate(label(v), (p, v), textcoords="offset points",
                         xytext=(0, 2), ha="center", fontsize=6,
                         color="#444444", rotation=90)

    w = 0.78 / len(NS)
    for k, (n, colour) in enumerate(NS):
        vals = [at[(g, n)]["key_switch_to_querier_ms"] for g in LOGNS]
        pos = idx + (k - (len(NS) - 1) / 2) * w
        axr.bar(pos, vals, w, color=colour, label=f"$N={n}$", edgecolor="none")
        for p, v in zip(pos, vals):
            axr.annotate(label(v), (p, v), textcoords="offset points",
                         xytext=(0, 2), ha="center", fontsize=6,
                         color="#444444", rotation=90)

    for ax in (axl, axr):
        ax.set_xticks(idx)
        ax.set_xticklabels([f"$2^{{{g}}}$" for g in LOGNS])
        ax.set_xlabel("ring degree")
        ax.set_yscale("log")
        ax.grid(axis="y", which="major", lw=0.4, color="#DDDDDD")
        ax.set_axisbelow(True)

    axl.set_ylabel("time (ms, log scale)")
    axl.set_title(r"per-query arithmetic  (at $N=10$)", loc="left")
    axl.legend(frameon=False, loc="upper left", ncol=2, columnspacing=1.0,
               handlelength=1.2, handletextpad=0.5)
    axr.set_title("key switch returning one label", loc="left")
    axr.legend(frameon=False, loc="upper left", ncol=3, columnspacing=1.0,
               handlelength=1.2, handletextpad=0.5)
    axl.set_ylim(0.3, 4e4)

    fig.tight_layout(w_pad=1.4)
    fig.savefig(OUT / "fig_cost.pdf")
    print(f"wrote {OUT / 'fig_cost.pdf'}")

    print("\nkey switch to the querier (ms), grid:")
    print("       " + "".join(f"{'N='+str(n):>10s}" for n, _ in NS))
    for g in LOGNS:
        print(f"  2^{g} " + "".join(
            f"{at[(g, n)]['key_switch_to_querier_ms']:10.1f}" for n, _ in NS))


if __name__ == "__main__":
    main()
