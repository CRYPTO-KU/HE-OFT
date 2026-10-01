#!/usr/bin/env python
"""Compare a run's output with the paper's record.

Rows are matched on the key columns. For every numeric column present in both
files, the script prints the largest absolute difference over the matched rows.
Record rows the run did not produce are counted and not treated as errors, so a
run of one seed can be checked against a three-seed record.

    python experiments/compare.py results/accuracy_text/results.csv \
        outputs/accuracy_text/ag_news_N10_a0.1_K200.csv --keys task seed mode
"""
import argparse
import csv
import sys


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load(path, keys):
    rows = {}
    for r in csv.DictReader(open(path)):
        k = tuple(str(num(r[c])) if num(r[c]) is not None else r[c] for c in keys)
        rows[k] = r
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("record")
    ap.add_argument("run")
    ap.add_argument("--keys", nargs="+", required=True)
    ap.add_argument("--tol", type=float, default=0.0,
                    help="largest difference accepted, default exact")
    a = ap.parse_args()
    rec, run = load(a.record, a.keys), load(a.run, a.keys)
    both = sorted(set(rec) & set(run))
    if not both:
        sys.exit("no rows match on the key columns")
    cols = [c for c in rec[both[0]] if c in run[both[0]] and c not in a.keys
            and num(rec[both[0]][c]) is not None]
    worst = 0.0
    print(f"matched {len(both)} rows, record has {len(rec) - len(both)} more, "
          f"run has {len(run) - len(both)} more")
    for c in cols:
        d = max(abs(num(rec[k][c]) - num(run[k][c])) for k in both
                if num(rec[k][c]) is not None and num(run[k][c]) is not None)
        worst = max(worst, d)
        print(f"  {c:<28} max |diff| = {d:.6g}")
    ok = worst <= a.tol
    print("PASS" if ok else f"FAIL: largest difference {worst:.6g} > tol {a.tol}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
