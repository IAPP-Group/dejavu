"""Evaluation metrics (Section 3.4), per-generator metrics, confusion matrices, CSV logging."""

import fcntl
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)

METRICS = ["balanced_accuracy", "f1_score", "ap", "auc", "accuracy", "precision", "recall"]


def compute_metrics(y_true, y_pred, scores, binary):
    with warnings.catch_warnings():
        # expected on OOD sets that hold only some of the classes, or a single one
        warnings.simplefilter("ignore", UserWarning)
        return _compute_metrics(y_true, y_pred, scores, binary)


def _compute_metrics(y_true, y_pred, scores, binary):
    """Balanced accuracy ('Acc.' in the paper), support-weighted F1, AP, AUC (+ plain accuracy,
    weighted precision/recall). Binary: AP/AUC on the fake-class score. Multi-class: one-vs-rest,
    macro-averaged over the classes present in y_true (so that OOD sets holding only some of
    the training classes are supported); AUC is NaN if a single class is present."""
    y_true, y_pred, scores = np.asarray(y_true), np.asarray(y_pred), np.asarray(scores)
    out = {
        "n_samples": len(y_true),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    present = sorted(set(y_true.tolist()))
    if binary:
        if len(present) > 1:
            out["ap"] = average_precision_score(y_true, scores[:, 1])
            out["auc"] = roc_auc_score(y_true, scores[:, 1])
        else:
            out["ap"] = out["auc"] = float("nan")
    else:
        out["ap"] = float(np.mean([average_precision_score(y_true == c, scores[:, c]) for c in present]))
        out["auc"] = (float(np.mean([roc_auc_score(y_true == c, scores[:, c]) for c in present]))
                      if len(present) > 1 else float("nan"))
    return out


def per_generator_metrics(y_true, y_pred, scores, models):
    """Detection metrics of every generator, each paired with all the real images of the set
    (as in the cross-model and robustness tables). Real sources get their real recall."""
    y_true, y_pred, scores, models = map(np.asarray, (y_true, y_pred, scores, models))
    real = y_true == 0
    rows = []
    for m in sorted(set(models.tolist())):
        own = models == m
        mask = own if real[own].all() else (own | real)
        r = compute_metrics(y_true[mask], y_pred[mask], scores[mask], binary=True)
        r.update(model=m, n_samples=int(own.sum()))
        rows.append(r)
    return rows


def confusion(y_true, y_pred, num_classes):
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))


def save_confusion_matrices(cms, classes, path_stem, title=""):
    """Save the summed counts and the run-averaged row-normalized matrix (CSV) and a PNG plot.
    Rows: true class, columns: predicted class. Rows of classes absent from the set are NaN."""
    os.makedirs(os.path.dirname(path_stem), exist_ok=True)
    cms = np.stack(cms).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # rows of classes absent from the set
        mean_norm = np.nanmean(cms / cms.sum(axis=2, keepdims=True), axis=0)
    pd.DataFrame(cms.sum(0).astype(int), index=classes, columns=classes).to_csv(f"{path_stem}_counts.csv")
    pd.DataFrame(mean_norm, index=classes, columns=classes).to_csv(f"{path_stem}.csv", float_format="%.4f")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        keep = ~np.isnan(mean_norm).all(axis=1)
        m = mean_norm[keep]
        fig, ax = plt.subplots(figsize=(0.8 * len(classes) + 2, 0.7 * keep.sum() + 1.8))
        im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
        ax.set_yticks(range(keep.sum()), [c for c, k in zip(classes, keep) if k])
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if m[i, j] > 0.5 else "black")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        if title:
            ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(f"{path_stem}.png", dpi=150)
        plt.close(fig)
    except Exception as e:  # plotting is optional
        print(f"  [warn] could not plot confusion matrix: {e}")


def append_rows(csv_path, rows):
    """Append rows to a CSV (creating it, and new columns, if needed); safe across processes."""
    if not rows:
        return
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    lock_path = os.path.join(os.path.dirname(csv_path), "." + os.path.basename(csv_path) + ".lock")
    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        new = pd.DataFrame(rows)
        if os.path.exists(csv_path):
            new = pd.concat([pd.read_csv(csv_path), new], ignore_index=True)
        new.to_csv(csv_path, index=False)
        fcntl.flock(lock, fcntl.LOCK_UN)
