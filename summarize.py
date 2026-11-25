#!/usr/bin/env python
"""Average the per-run metrics written by train.py / evaluate.py.

  python summarize.py results/metrics/detection.csv
  python summarize.py results/metrics/*.csv --output results/summary.csv
  python summarize.py results/metrics/detection.csv --filter eval_set=genimage_test classifier=3-MLP

Rows are grouped by configuration, evaluation set and post-processing; for each metric
the mean and standard deviation over runs are reported (the paper reports means). If a
configuration was evaluated more than once on the same set, only its latest results are kept.
"""

import argparse

import pandas as pd

from dejavu.metrics import METRICS

KEYS = ["backbone", "classifier", "task", "preprocess", "layer", "num_prompts", "train_generators",
        "eval_set", "eval_degradation", "model"]


def summarize(df):
    keys = [k for k in KEYS if k in df.columns]
    df = df.drop_duplicates(subset=keys + ["run"], keep="last")
    metrics = [m for m in METRICS if m in df.columns]
    g = df.groupby(keys, dropna=False, sort=False)
    out = g[metrics].mean().add_suffix("_mean").join(g[metrics].std().add_suffix("_std"))
    out["n_runs"] = g.size()
    order = [c for m in metrics for c in (f"{m}_mean", f"{m}_std")] + ["n_runs"]
    return out[order].reset_index()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="+", help="metrics CSV(s) written by train.py / evaluate.py")
    p.add_argument("--filter", nargs="*", default=[], metavar="COL=VALUE", help="Keep only matching rows.")
    p.add_argument("--output", default=None, help="Write the summary to this CSV.")
    args = p.parse_args()

    df = pd.concat([pd.read_csv(f) for f in args.csv], ignore_index=True)
    for f in args.filter:
        col, val = f.split("=", 1)
        df = df[df[col].astype(str) == val]
    out = summarize(df)
    if args.output:
        out.to_csv(args.output, index=False)
        print(f"Saved {len(out)} rows to {args.output}")
    cols = [c for c in out.columns if not c.endswith("_std") and c not in ("accuracy_mean", "precision_mean",
                                                                           "recall_mean")]
    with pd.option_context("display.max_rows", 500, "display.width", 250, "display.max_columns", 30):
        print(out[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
