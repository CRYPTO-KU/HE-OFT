# Reproducing the paper

Every number in the technical report traces to a file in `results/`. This page gives the command that produces each file. A run writes to
`outputs/<experiment>/`, and `experiments/compare.py` checks it against the
record:

```bash
python experiments/compare.py results/accuracy_text/results.csv \
    outputs/accuracy_text/ag_news_N10_a0.1_K200.csv --keys task seed mode
```

The default settings of every script are those of the paper: N = 10 clients,
Dirichlet concentration alpha = 0.1, K = 200 local steps, adapter rank 8,
learning rate 5e-4, batch size 32, seeds 42, 43 and 44. Table and figure
numbers are those of the technical report.

## 0. Setup

```bash
pip install -r requirements.txt && pip install -e .
python experiments/prefetch.py        # once, on a machine with internet access
cd fhe && go build -o heoft-fhe . && cd ..
```

On a Slurm cluster, `slurm/gpu.sbatch <experiment> [flags]` runs one Python
experiment and `slurm/cpu.sbatch [flags]` one Go measurement. Create
`outputs/slurm/` before the first job.

## 1. Accuracy (run these first)

The accuracy runs save, per task and seed, the states, class counts and logits
of every client to `outputs/accuracy_{text,vision}/artifacts/`. Every attack
experiment in Section 3 reads these files.

| element | command | record |
|---|---|---|
| Tables 2 and 3, text rows | `python experiments/accuracy_text.py` | `results/accuracy_text/results.csv` |
| Tables 2 and 3, CIFAR-100 row | `python experiments/accuracy_vision.py --datasets cifar100` | `results/accuracy_vision/results.csv` |
| pooled column, text | `python experiments/pooled_text.py` | `results/pooled_text/results.csv`, rows `matched_total` |
| pooled column, CIFAR-100 | `python experiments/pooled_vision.py --datasets cifar100` | `results/pooled_vision/results.csv`, rows `matched_total` |
| Table 6, CIFAR-10 on published partitions | `python experiments/accuracy_vision.py --datasets cifar10 --clients <N> --alpha <alpha> --max-test 10000`, for (N, alpha) in (5, 0.1), (5, 0.3), (20, 0.04), (20, 0.16) | `results/accuracy_vision/cifar10_matched_full.csv` |
| Fig. 4(a), client count | `python experiments/accuracy_text.py --tasks dbpedia_14 --seeds 42 --clients <N>`, for N in 10, 20, 50 | `results/accuracy_text/nsweep.csv` |
| Fig. 4(b, c) and Table 4, skew and local steps | `python experiments/accuracy_text.py --tasks dbpedia_14 --alpha <a>` for a in 0.05, 0.3, 1.0, and `--seeds 42 --steps <K>` for K in 100, 400 | `results/accuracy_text/sensitivity.csv` |
| Fig. 5 and Table 5, selection | read from the accuracy records above | `results/accuracy_{text,vision}/` |
| head-row coverage | `python experiments/head_coverage.py` | `results/head_coverage/results.csv` |

The modes of the accuracy records and their names in the paper are listed in
the README.

## 2. Cryptographic cost (Go)

Run from `fhe/` after `go build -o heoft-fhe .`. No mode uses a GPU, and the
bootstrapping modes need up to 200 GB of memory. `fhe/README.md` describes every
mode, its memory and its wall time. Modes marked "stdout" print their record as
a CSV block, which `fhe/README.md` shows how to extract.

| element | command | record |
|---|---|---|
| Table 7 and Fig. 4(d), and the abstract's query time and traffic, CPU rows | `./heoft-fhe -cost-vs-d` | `results/fhe/cost_vs_d.csv`, rows at d = 768 |
| cost against the feature dimension | `./heoft-fhe -cost-vs-d` | `results/fhe/cost_vs_d.csv`, all rows |
| query cost at five label-space sizes | `./heoft-fhe -cost-vs-d -dims 768 -classes 4,6,14,77,100 -csv ../outputs/fhe/query_cost.csv` | `results/fhe/query_cost.csv` |
| Fig. 6, per-operation cost | `./heoft-fhe -cost-grid` | `results/fhe/cost_grid.json` |
| Table 8, communication | `./heoft-fhe -comm-cost` | `results/fhe/comm_grid.json`, chain `serving` |
| Table 9, server-side bootstrapping against collective refresh | `./heoft-fhe -serve-btp` and `./heoft-fhe -serve-tournament` (stdout) | `results/fhe/argmax_tournament_btp.csv`, `results/fhe/argmax_tournament.csv` |
| the stored bootstrapping-key figure the report corrects | `./heoft-fhe -btp-keys` | `results/fhe/btp_keys.json` |
| the encrypted reciprocal of the per-class totals | `./heoft-fhe -protocol-cost` | `results/fhe/protocol_cost.json` |
| the argmax index premium | `./heoft-fhe -serve-index` and `-serve-index-btp` (stdout) | `results/fhe/argmax_index.csv`, `results/fhe/argmax_index_btp.csv` |
| the argmax as a sequential fold | `./heoft-fhe -serve-argmax` (stdout) | `results/fhe/argmax_cost.csv` |
| refresh and decryption primitives | `./heoft-fhe -serve` (stdout) | `results/fhe/serve_primitives.csv` |
| the selection cost | `./heoft-fhe -selection-cost` (stdout) | `results/fhe/selection_cost.csv` |
| the smudging margin | `./heoft-fhe -smudge-noise -smudge-log2 25 -smudge-reps 3 -smudge-trials 10` | `results/fhe/smudge_noise.csv` |
| one real end-to-end query | `python experiments/export_head.py`, then `./heoft-fhe -serve-real -logn 15 -real-parties 10 -real-export ../outputs/export_head/ag_news_s42_A.json` and the same with `_B.json` (stdout) | `results/fhe/real_query.csv`, `results/fhe/real_query/` |

The GPU rows of the paper do not come from this code. They replace each
server-side bootstrap with its time under Phantom, a GPU CKKS library, and
leave the rest of the query on the processor.

Byte counts, levels and refresh counts reproduce exactly. Errors and decoded
values change in their last digits, since keys and noise are sampled afresh in
every run, and timings depend on the machine.

## 3. Extraction and membership

Each script reads the artifacts of Section 1 at the default settings.

| element | command | record |
|---|---|---|
| Fig. 7, extraction against queries | `python experiments/extraction_budget.py` | `results/extraction_budget/results.csv` |
| extraction cost against the size of the head | `python experiments/extraction_scale.py` | `results/extraction_scale/results.csv` |
| Table 11, answer format and output noise | `python experiments/extraction_defence.py` | `results/extraction_defence/results.csv` |
| Table 10, membership against the disclosed model | `python experiments/membership_disclosed.py --tasks <t> --seeds <s>`, one call per task in ag_news, dbpedia_14, banking77, cifar100 and seed | `results/membership_disclosed/results.csv` |
| membership against a copy of the head | `python experiments/membership_extracted.py` | `results/membership_extracted/results.csv` |
| label-only membership against the serving interface | `python experiments/membership_label_only.py` | `results/membership_label_only/results.csv` |
| what a head row carries about its client | `python experiments/row_leakage.py` | `results/row_leakage/results.csv` |

`membership_disclosed.py` trains 64 shadow federations per cell, which took
about 4 hours per text cell and 6.6 hours per CIFAR-100 cell on one A40.
`--shard 0/2` and `--shard 1/2` split a cell over two jobs.

## 4. Figures

```bash
pip install -e ".[figures]"
python figures/make_trends.py       # Fig. 4
python figures/make_regret.py       # Fig. 5
python figures/make_extraction.py   # Fig. 7
python figures/make_cost.py         # Fig. 6
```

The scripts read `results/` and write the PDFs next to themselves.
