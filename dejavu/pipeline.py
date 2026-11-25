"""Training and evaluation of a classifier configuration on precomputed features."""

import os
import random
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .classifiers import (MLP_CLASSIFIERS, build_classifier, class_weights, classifier_file,
                          load_classifier, predict, write_json)
from .data import GENERATORS, load_features, load_split, task_classes, variant_tag
from .metrics import (METRICS, append_rows, compute_metrics, confusion, per_generator_metrics,
                      save_confusion_matrices)


@dataclass
class Config:
    """Identifies one trained classifier configuration (and its output folder)."""
    backbone: str
    classifier: str
    task: str
    preprocess: str = None
    layer: str = None
    num_prompts: int = None          # training-set size experiment (None = all 800 prompts)
    train_generators: list = None    # cross-model experiment (None = all 8 generators)

    @property
    def feature_variant(self):
        return variant_tag(self.preprocess, self.layer)

    @property
    def name(self):
        parts = [self.classifier]
        if self.feature_variant:
            parts.append(self.feature_variant)
        if self.num_prompts:
            parts.append(f"prompts{self.num_prompts}")
        if self.train_generators:
            parts.append("train-" + "+".join(sorted(self.train_generators)))
        return "__".join(parts)

    def run_dir(self, out_dir):
        return os.path.join(out_dir, self.task, self.backbone, self.name)

    def id_columns(self):
        return {"backbone": self.backbone, "classifier": self.classifier, "task": self.task,
                "preprocess": self.preprocess or "default", "layer": self.layer or "final",
                "num_prompts": self.num_prompts or 800,
                "train_generators": "+".join(sorted(self.train_generators)) if self.train_generators else "all"}


@dataclass
class TrainOptions:
    runs: int = 10
    epochs: int = 50
    batch_size: int = 64
    lr: float = 3e-4
    seed: int = 0              # MLP initialization / shuffling: seed + run
    prompt_seed: int = 42      # prompt subsets of the training-size experiment: prompt_seed + run
    n_jobs: int = -1
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Labels and features
# ---------------------------------------------------------------------------

def task_labels(df, task):
    """Filter a split for a task and return (df, integer labels)."""
    if task == "detection":
        return df, (df.label == "fake").astype(int).to_numpy()
    if task == "attribution":
        df = df[df.label == "fake"].reset_index(drop=True)
        return df, df.model.map(GENERATORS.index).to_numpy()
    if task == "detection+attribution":
        y = [0 if l == "real" else 1 + GENERATORS.index(m) for l, m in zip(df.label, df.model)]
        return df, np.asarray(y)
    raise ValueError(task)


def gather(df, index, feats):
    """L2-normalized features of the rows of df, in order."""
    missing = [p for p in df.image_path if p not in index]
    if missing:
        raise KeyError(f"{len(missing)} images have no features, e.g. {missing[0]}")
    X = feats[[index[p] for p in df.image_path]]
    return F.normalize(X, p=2, dim=-1).numpy().astype(np.float32)


def sample_prompts(df, num_prompts, seed):
    """Keep the fake and real images of `num_prompts` random prompts (ImageNet classes)."""
    rng = random.Random(seed)
    fake, real = df[df.label == "fake"], df[df.label == "real"]
    prompts = fake.prompt.unique().tolist()
    chosen = rng.sample(prompts, min(num_prompts, len(prompts)))
    return pd.concat([real[real.prompt.isin(chosen)], fake[fake.prompt.isin(chosen)]]).reset_index(drop=True)


def training_set(cfg, run, opts):
    df = load_split("genimage_train")
    if cfg.num_prompts:
        df = sample_prompts(df, cfg.num_prompts, opts.prompt_seed + run)
    if cfg.train_generators:
        df = df[(df.label == "real") | df.model.isin(cfg.train_generators)].reset_index(drop=True)
    return task_labels(df, cfg.task)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class Evaluator:
    """Evaluates the runs of a configuration on several sets, collecting metrics, predictions
    and confusion matrices; `finalize()` writes everything to disk."""

    def __init__(self, cfg, out_dir, features_dir, eval_sets, degradation_tag=""):
        self.cfg, self.out_dir, self.run_dir = cfg, out_dir, cfg.run_dir(out_dir)
        self.classes = task_classes(cfg.task)
        self.degradation_tag = degradation_tag
        self.variant = variant_tag(cfg.preprocess, cfg.layer, degradation_tag)
        self.sets = {}
        for name in eval_sets:
            index, feats = load_features(features_dir, cfg.backbone, name, self.variant)
            df, y = task_labels(load_split(name), cfg.task)
            self.sets[name] = (df, gather(df, index, feats), y)
            del feats
        self.rows, self.gen_rows, self.cms = [], [], {n: [] for n in eval_sets}

    def evaluate(self, clf, run, seed=None):
        binary = self.cfg.task == "detection"
        for name, (df, X, y) in self.sets.items():
            y_pred, scores = predict(clf, X)
            m = compute_metrics(y, y_pred, scores, binary)
            cols = {**self.cfg.id_columns(), "eval_set": name,
                    "eval_degradation": self.degradation_tag or "none", "run": run, "seed": seed}
            self.rows.append({**cols, **m})
            tag = name + (f"__{self.degradation_tag}" if self.degradation_tag else "")
            print(f"    [{tag:<40}] run {run}: "
                  + "  ".join(f"{k}={m[k]:.3f}" for k in ("balanced_accuracy", "f1_score", "ap", "auc")))
            pred_dir = os.path.join(self.run_dir, "predictions")
            os.makedirs(pred_dir, exist_ok=True)
            np.savez_compressed(os.path.join(pred_dir, f"{tag}__run{run}.npz"), image_path=df.image_path.to_numpy(),
                                y_true=y, y_pred=y_pred, scores=scores.astype(np.float32), classes=self.classes)
            if binary:
                for r in per_generator_metrics(y, y_pred, scores, df.model):
                    self.gen_rows.append({**cols, **r})
            else:
                self.cms[name].append(confusion(y, y_pred, len(self.classes)))

    def finalize(self):
        task = self.cfg.task
        append_rows(os.path.join(self.out_dir, "metrics", f"{task}.csv"), self.rows)
        append_rows(os.path.join(self.out_dir, "metrics", f"{task}_per_generator.csv"), self.gen_rows)
        for name, cms in self.cms.items():
            if cms:
                tag = name + (f"__{self.degradation_tag}" if self.degradation_tag else "")
                save_confusion_matrices(cms, self.classes, os.path.join(self.run_dir, "confusion_matrices", tag),
                                        title=f"{self.cfg.backbone} / {self.cfg.name} / {tag}")
        print_summary(self.rows)


def print_summary(rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    g = df.groupby("eval_set", sort=False)[["balanced_accuracy", "f1_score", "ap", "auc"]]
    mean, std = g.mean(), g.std().fillna(0)
    print("  Mean ± std over runs:")
    for name in mean.index:
        print(f"    {name:<40} " + "  ".join(
            f"{k}={mean.loc[name, k]:.3f}±{std.loc[name, k]:.3f}" for k in mean.columns))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg, opts, features_dir, out_dir, eval_sets, skip_existing=False):
    run_dir = cfg.run_dir(out_dir)
    is_mlp = cfg.classifier in MLP_CLASSIFIERS
    # sklearn classifiers are deterministic: one fit, unless each run has its own prompt subset
    runs = opts.runs if (is_mlp or cfg.num_prompts) else 1
    if skip_existing and all(os.path.exists(classifier_file(run_dir, cfg.classifier, r)) for r in range(runs)):
        print(f"  [skip] {run_dir} already trained")
        return
    print(f"\n=== {cfg.backbone} | {cfg.task} | {cfg.name} -> {run_dir}")
    index, feats = load_features(features_dir, cfg.backbone, "genimage_train", cfg.feature_variant)
    evaluator = Evaluator(cfg, out_dir, features_dir, eval_sets) if eval_sets else None
    classes = task_classes(cfg.task)
    write_json(os.path.join(run_dir, "config.json"),
               {"config": asdict(cfg), "options": {k: v for k, v in asdict(opts).items() if k != "device"},
                "classes": classes, "features_dir": os.path.abspath(features_dir)})

    X = None
    for run in range(runs):
        df, y = training_set(cfg, run, opts)
        if X is None or cfg.num_prompts:  # the training set changes across runs only with num_prompts
            X = gather(df, index, feats)
        seed = opts.seed + run
        torch.manual_seed(seed)
        np.random.seed(seed)
        w = class_weights(y, len(classes))
        clf = build_classifier(cfg.classifier, X.shape[1], len(classes), opts.device, opts.n_jobs)
        n_real, n_fake = int((df.label == "real").sum()), int((df.label == "fake").sum())
        print(f"  run {run} ({run + 1}/{runs}): training {cfg.classifier} on {len(df)} images "
              f"(real {n_real}, fake {n_fake})", flush=True)
        clf.fit(X, y, weights=w, epochs=opts.epochs, batch_size=opts.batch_size, lr=opts.lr)
        if getattr(clf, "best_params", None):
            print(f"    best params (5-fold CV): {clf.best_params}")
        path = classifier_file(run_dir, cfg.classifier, run)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        clf.save(path)
        if evaluator:
            evaluator.evaluate(clf, run, seed)
    if evaluator:
        evaluator.finalize()


def evaluate(cfg, opts, features_dir, out_dir, eval_sets, degradation_tag="", runs=None):
    """Evaluate the saved classifiers of a configuration (e.g. on OOD sets or degraded images)."""
    run_dir = cfg.run_dir(out_dir)
    clf_dir = os.path.join(run_dir, "classifiers")
    files = os.listdir(clf_dir) if os.path.isdir(clf_dir) else []
    available = sorted(int(f[3:].split(".")[0]) for f in files if f.startswith("run"))
    if not available:
        raise FileNotFoundError(f"No trained classifiers in {run_dir}/classifiers. Train them with train.py first.")
    runs = available if runs is None else [r for r in runs if r in available]
    print(f"\n=== Evaluating {cfg.backbone} | {cfg.task} | {cfg.name} (runs {runs})"
          + (f" on degraded images ({degradation_tag})" if degradation_tag else ""))
    evaluator = Evaluator(cfg, out_dir, features_dir, eval_sets, degradation_tag)
    for run in runs:
        clf = load_classifier(classifier_file(run_dir, cfg.classifier, run), opts.device)
        evaluator.evaluate(clf, run, opts.seed + run)
    evaluator.finalize()


__all__ = ["Config", "TrainOptions", "train", "evaluate", "METRICS"]
