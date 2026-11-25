#!/usr/bin/env python
"""Train classifiers on precomputed features of GenImage train and evaluate them.

Every combination of --backbones x --tasks x --classifiers x --layers x --num_prompts
is trained. MLPs are trained --runs times (default 10, as in the paper); LR/SVM/kNN
once (or once per run with --num_prompts, since every run draws its own prompts).
After training, each run is evaluated on --eval_sets.

Outputs (under --out_dir):
  <task>/<backbone>/<config>/classifiers/run<r>.{pth,pkl}      trained classifiers
  <task>/<backbone>/<config>/predictions/<eval_set>__run<r>.npz labels, predictions, scores
  <task>/<backbone>/<config>/confusion_matrices/<eval_set>.{csv,png}  (attribution / joint)
  metrics/<task>.csv                  one row per (configuration, evaluation set, run)
  metrics/<task>_per_generator.csv    detection: one row per generator (paired with all reals)
Use summarize.py to average the runs.

Examples
--------
  # DejaVu (MetaCLIP2-ViT-bigG/14 + 3-MLP), the three tasks, GenImage validation + test
  python train.py --backbones DejaVu --classifiers 3-MLP --tasks detection attribution detection+attribution

  # all classifiers on all backbones (Tables 4-9), also evaluated out-of-domain
  python train.py --backbones all --classifiers all --tasks detection --eval_sets genimage ood

  # training-set size: 1, 5, ..., 800 prompts (Section 4.4)
  python train.py --backbones all --classifiers 3-MLP --tasks detection --num_prompts 1 5 10 50 100 300 500 800

  # cross-model generalization: train on VQDM + Wukong only (Section 5.2)
  python train.py --backbones DejaVu --tasks detection --train_generators vqdm wukong --eval_sets genimage_test
"""

import argparse
import sys

from dejavu.backbones import BACKBONE_IDS, expand_backbones
from dejavu.classifiers import CLASSIFIERS
from dejavu.data import GENERATORS, TASKS, expand_eval_sets
from dejavu.pipeline import Config, TrainOptions, train
from dejavu.transforms import PREPROCESSINGS


def add_config_args(p):
    p.add_argument("--backbones", nargs="+", required=True, metavar="NAME",
                   help=f"Backbone ids/paper names, all, or a family (clip, metaclip2, dino). "
                        f"Ids: {', '.join(BACKBONE_IDS)} (alias 'DejaVu' = MetaCLIP2-ViT-bigG-14).")
    p.add_argument("--classifiers", nargs="+", default=["3-MLP"], metavar="CLF",
                   help=f"{', '.join(CLASSIFIERS)} or all (default: 3-MLP).")
    p.add_argument("--tasks", nargs="+", default=["detection"], choices=TASKS, help="Task(s) (default: detection).")
    p.add_argument("--features_dir", default="features", help="Folder of the extracted features (default: %(default)s).")
    p.add_argument("--out_dir", default="results", help="Output folder (default: %(default)s).")
    p.add_argument("--preprocess", default="default", choices=PREPROCESSINGS,
                   help="Use the features extracted with this preprocessing (Table 12).")
    p.add_argument("--layers", nargs="+", default=[None], metavar="LAYER",
                   help="Use the features of intermediate layer(s), e.g. block22 (Figure 11).")
    p.add_argument("--num_prompts", nargs="+", type=int, default=[None], metavar="N",
                   help="Train on the images of N random prompts out of 800 (Section 4.4); "
                        "each run draws its own prompts. 800 = the whole training set.")
    p.add_argument("--train_generators", nargs="+", default=None, choices=GENERATORS, metavar="GEN",
                   help=f"Detection only: train on these generators (+ all real images), Section 5.2. "
                        f"Choices: {', '.join(GENERATORS)}.")
    p.add_argument("--n_jobs", type=int, default=-1, help="Parallel jobs of the sklearn grid search.")
    p.add_argument("--seed", type=int, default=0, help="Base seed of MLP training (seed + run).")


def configs_from_args(args, parser):
    classifiers = CLASSIFIERS if "all" in args.classifiers else args.classifiers
    bad = [c for c in classifiers if c not in CLASSIFIERS]
    if bad:
        parser.error(f"Unknown classifier(s) {bad}. Choices: {', '.join(CLASSIFIERS)}")
    try:
        backbones = expand_backbones(args.backbones)
    except ValueError as e:
        parser.error(str(e))
    prompts = [None if (n is None or n >= 800) else n for n in args.num_prompts]
    if args.train_generators and args.tasks != ["detection"]:
        parser.error("--train_generators is only available for the detection task.")
    return [Config(b, c, t, args.preprocess, l, n, args.train_generators)
            for b in backbones for t in args.tasks for c in classifiers for l in args.layers for n in prompts]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(p)
    p.add_argument("--eval_sets", nargs="*", default=["genimage"], metavar="SET",
                   help="Evaluation sets after training (default: genimage = genimage_validation + "
                        "genimage_test). 'ood' / 'ood_validation' / 'ood_test' select the out-of-domain "
                        "sets of each task. Pass no value to skip evaluation.")
    p.add_argument("--runs", type=int, default=10, help="MLP runs (default: %(default)s).")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--prompt_seed", type=int, default=42,
                   help="Seed of the prompt subsets (prompt_seed + run), as in the paper.")
    p.add_argument("--skip_existing", action="store_true", help="Skip configurations already trained.")
    args = p.parse_args()

    for cfg in configs_from_args(args, p):
        opts = TrainOptions(runs=args.runs, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                            seed=args.seed, prompt_seed=args.prompt_seed, n_jobs=args.n_jobs)
        try:
            eval_sets = expand_eval_sets(args.eval_sets, cfg.task)
            train(cfg, opts, args.features_dir, args.out_dir, eval_sets, args.skip_existing)
        except (FileNotFoundError, KeyError, ValueError) as e:
            sys.exit(f"[ERROR] {e}")


if __name__ == "__main__":
    main()
