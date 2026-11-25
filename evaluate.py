#!/usr/bin/env python
"""Evaluate already-trained classifiers (see train.py) on other evaluation sets or on
post-processed images, without retraining.

The configuration is identified by the same options used with train.py; all its saved
runs are evaluated (or those given with --runs). Results are appended to
<out_dir>/metrics/<task>.csv with the evaluation set and post-processing in the
`eval_set` / `eval_degradation` columns.

Examples
--------
  # out-of-domain evaluation of DejaVu on the three tasks (Figures 3, 5, 8; Table 10)
  python evaluate.py --backbones DejaVu --tasks detection attribution detection+attribution --eval_sets ood

  # classifiers trained on a single prompt, evaluated out-of-domain (Figure 10)
  python evaluate.py --backbones all --num_prompts 1 --eval_sets ood_validation

  # robustness: GenImage test images JPEG-compressed at QF 75 (features from
  # `extract_features.py --splits genimage_test --jpeg 75`), Section 5.3
  python evaluate.py --backbones DejaVu --eval_sets genimage_test --jpeg 75
"""

import argparse
import sys

from dejavu.data import expand_eval_sets
from dejavu.pipeline import TrainOptions, evaluate
from dejavu.transforms import DEGRADATIONS, degradation_tag
from train import add_config_args, configs_from_args


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(p)
    p.add_argument("--eval_sets", nargs="+", required=True, metavar="SET",
                   help="Evaluation sets: genimage, ood, ood_validation, ood_test or split names "
                        "(e.g. synthbuster_test, ood_attribution_validation).")
    p.add_argument("--runs", type=int, nargs="+", default=None, help="Runs to evaluate (default: all saved).")
    d = p.add_argument_group("Post-processed images (robustness); at most one")
    dg = d.add_mutually_exclusive_group()
    dg.add_argument("--jpeg", type=int, metavar="QF")
    dg.add_argument("--blur", type=float, metavar="SIGMA")
    dg.add_argument("--resize", type=float, metavar="R")
    dg.add_argument("--crop", type=float, metavar="C")
    dg.add_argument("--center_crop", type=float, metavar="C")
    args = p.parse_args()

    degradation = next(((n, getattr(args, n)) for n in DEGRADATIONS if getattr(args, n) is not None), None)
    for cfg in configs_from_args(args, p):
        opts = TrainOptions(seed=args.seed)
        try:
            evaluate(cfg, opts, args.features_dir, args.out_dir, expand_eval_sets(args.eval_sets, cfg.task),
                     degradation_tag(degradation), args.runs)
        except (FileNotFoundError, KeyError, ValueError) as e:
            sys.exit(f"[ERROR] {e}")


if __name__ == "__main__":
    main()
