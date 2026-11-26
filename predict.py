#!/usr/bin/env python
"""Apply a trained classifier (e.g. the released DejaVu checkpoints) to your own images.

Unlike evaluate.py, which works on the registered splits and precomputed features, this
script takes any images, extracts their features with the classifier's backbone and
writes one prediction per image. If the images are labelled, it also computes the metrics.

Input (one of):
  --images   image files and/or folders (searched recursively); no labels.
  --csv      a CSV with an `image_path` column (absolute, or relative to --image_root,
             default: the CSV's folder) and, optionally, the ground truth:
               label   real / fake (or 0 / 1)          -> detection and joint-task metrics
               model   generator of each fake image    -> attribution and joint-task metrics
                       (one of the 8 GenImage generators; other generators are predicted
                       but left out of the attribution metrics)

The backbone, task, preprocessing and layer are read from the JSON next to the checkpoint
(checkpoints/dejavu_det.json for checkpoints/dejavu_det.pth, or the config.json of a
results/<task>/<backbone>/<config>/ folder for its classifiers/run<r>.pth).

Outputs (--out, default: predictions/<checkpoint name>/):
  predictions.csv       image_path, [label, model,] prediction, score of every class
  metrics.json          only with ground truth (+ per-generator detection metrics)

Examples
--------
  python predict.py --checkpoint checkpoints/dejavu_det.pth --images my_images/
  python predict.py --checkpoint checkpoints/dejavu_attr.pth --csv my_dataset.csv --out predictions/my_dataset
  python predict.py --checkpoint results/detection/RN50x64/3-MLP/classifiers/run0.pth --images a.png b.jpg
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dejavu.backbones import LoadedBackbone, get_backbone
from dejavu.classifiers import load_classifier, predict
from dejavu.data import GENERATORS, task_classes
from dejavu.metrics import compute_metrics, per_generator_metrics
from dejavu.transforms import build_transforms, jpeg

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def checkpoint_config(path):
    """Configuration of a checkpoint: <stem>.json next to it, or the config.json of its run folder."""
    stem = os.path.splitext(path)[0]
    for p in (stem + ".json", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(path))), "config.json")):
        if os.path.isfile(p):
            with open(p) as f:
                return json.load(f)["config"]
    return {}


def list_images(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                out += [os.path.join(root, f) for f in sorted(files) if f.lower().endswith(IMAGE_EXTS)]
        elif os.path.isfile(p):
            out.append(p)
        else:
            raise FileNotFoundError(f"No such file or folder: {p}")
    if not out:
        raise FileNotFoundError(f"No images ({', '.join(IMAGE_EXTS)}) found in {paths}")
    return pd.DataFrame({"image_path": out})


def read_csv(path, image_root=None):
    df = pd.read_csv(path)
    if "image_path" not in df.columns:
        raise ValueError(f"{path} has no `image_path` column (columns: {list(df.columns)})")
    root = image_root or os.path.dirname(os.path.abspath(path))
    df["file"] = [p if os.path.isabs(p) else os.path.join(root, p) for p in df.image_path]
    if "label" in df.columns:
        df["label"] = df.label.map(lambda v: {"0": "real", "1": "fake"}.get(str(v).strip().lower(),
                                                                            str(v).strip().lower()))
        bad = sorted(set(df.label) - {"real", "fake"})
        if bad:
            raise ValueError(f"`label` must be real/fake or 0/1, found {bad[:5]}")
    if "model" in df.columns:
        df["model"] = df.model.astype(str).str.strip().str.lower()
    return df


class Images(Dataset):
    def __init__(self, files, transform, compress=None):
        self.files, self.transform = list(files), transform
        self.compress = [False] * len(self.files) if compress is None else list(compress)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        try:
            image = Image.open(self.files[i]).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Cannot open image '{self.files[i]}': {e}") from e
        if self.compress[i]:
            image = jpeg(image, 96)
        return self.transform(image)


@torch.no_grad()
def extract(backbone, files, preprocess, layer, batch_size, num_workers, compress=None):
    _, eval_tf = build_transforms(backbone, preprocess)
    loader = DataLoader(Images(files, eval_tf, compress), batch_size=batch_size, num_workers=num_workers,
                        pin_memory=backbone.device == "cuda")
    feats = []
    for x in tqdm(loader, unit="batch", desc="features"):
        out = backbone.encode(x, [layer] if layer else None)
        feats.append(out[layer] if layer else out)
    return F.normalize(torch.cat(feats).float(), p=2, dim=-1).numpy()   # as in training


# ---------------------------------------------------------------------------
# Ground truth and metrics
# ---------------------------------------------------------------------------

def targets(df, task):
    """Integer ground truth for the rows that have it (-1 elsewhere), or None if there is none."""
    has_label, has_model = "label" in df.columns, "model" in df.columns
    if task == "detection":
        return (df.label == "fake").astype(int).to_numpy() if has_label else None
    gen = df.model.map({g: i for i, g in enumerate(GENERATORS)}) if has_model else None
    if task == "attribution":
        if gen is None:
            return None
        fake = (df.label == "fake") if has_label else pd.Series(True, index=df.index)
        return np.where(fake & gen.notna(), gen.fillna(-1), -1).astype(int)
    if not has_label:
        return None
    if gen is None:
        gen = pd.Series(np.nan, index=df.index)
    y = np.where(df.label == "real", 0, np.where(gen.notna(), gen.fillna(-1) + 1, -1))
    return y.astype(int)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="Classifier file (.pth or .pkl).")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", nargs="+", metavar="PATH", help="Image files and/or folders (recursive).")
    src.add_argument("--csv", help="CSV with an image_path column and optional label / model columns.")
    p.add_argument("--image_root", default=None, help="Folder of the relative paths of --csv (default: its folder).")
    p.add_argument("--out", default=None, help="Output folder (default: predictions/<checkpoint name>).")
    c = p.add_argument_group("Configuration (only if the checkpoint has no JSON)")
    c.add_argument("--backbone", default=None)
    c.add_argument("--task", default=None, choices=["detection", "attribution", "detection+attribution"])
    c.add_argument("--preprocess", default=None)
    c.add_argument("--layer", default=None)
    o = p.add_argument_group("Other")
    o.add_argument("--fake_qf96", action="store_true",
                   help="JPEG-compress the fake images (label column of --csv) at QF=96 before extraction, as "
                        "done for the GenImage splits in the paper. Off by default (as for the OOD benchmarks).")
    o.add_argument("--batch_size", type=int, default=32)
    o.add_argument("--num_workers", type=int, default=4)
    o.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32",
                   help="Precision of HuggingFace backbones (paper: float32).")
    args = p.parse_args()

    try:
        cfg = checkpoint_config(args.checkpoint)
        backbone_id = args.backbone or cfg.get("backbone")
        task = args.task or cfg.get("task")
        preprocess = args.preprocess or cfg.get("preprocess") or "default"
        layer = args.layer or cfg.get("layer")
        if not backbone_id or not task:
            raise ValueError("No JSON configuration found next to the checkpoint: pass --backbone and --task.")
        classes = task_classes(task)
        df = list_images(args.images) if args.images else read_csv(args.csv, args.image_root)
        files = df.pop("file") if "file" in df.columns else df.image_path
        missing = [f for f in files if not os.path.isfile(f)]
        if missing:
            raise FileNotFoundError(f"{len(missing)} images not found, e.g. {missing[0]}")
        if args.fake_qf96 and "label" not in df.columns:
            raise ValueError("--fake_qf96 needs a --csv with a `label` column.")
    except (FileNotFoundError, ValueError, KeyError) as e:
        sys.exit(f"[ERROR] {e}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    out_dir = args.out or os.path.join("predictions", name)
    print(f"Checkpoint: {args.checkpoint}\nBackbone: {backbone_id}  Task: {task}  Preprocess: {preprocess}  "
          f"Layer: {layer or 'final'}  Device: {device}\nImages: {len(df)}")

    clf = load_classifier(args.checkpoint, device)
    spec = get_backbone(backbone_id)
    print(f"=== Loading {spec.paper_name} ({spec.checkpoint})")
    backbone = LoadedBackbone(spec, device, getattr(torch, args.dtype))
    X = extract(backbone, files, preprocess, layer, args.batch_size, args.num_workers,
                (df.label == "fake").to_numpy() if args.fake_qf96 else None)
    del backbone
    y_pred, scores = predict(clf, X)
    if scores.shape[1] != len(classes):
        sys.exit(f"[ERROR] The classifier has {scores.shape[1]} outputs but task '{task}' has {len(classes)} classes.")

    out = df.copy()
    out["prediction"] = [classes[i] for i in y_pred]
    for j, cl in enumerate(classes):
        out[f"score_{cl}"] = scores[:, j]
    os.makedirs(out_dir, exist_ok=True)
    out.to_csv(os.path.join(out_dir, "predictions.csv"), index=False, float_format="%.6f")
    print(f"\nPredictions -> {os.path.join(out_dir, 'predictions.csv')}")
    print(out.prediction.value_counts().rename_axis("predicted").to_string())

    y = targets(df, task)
    if y is None:
        print("No ground truth (label / model columns): metrics not computed.")
        return
    keep = y >= 0
    dropped = ~keep & ((df.label == "fake").to_numpy() if "label" in df.columns else True)
    if task == "attribution" and "label" in df.columns and (df.label == "real").any():
        print(f"{(df.label == 'real').sum()} real images: predicted, not used in the attribution metrics.")
    if dropped.any():
        print(f"[warn] {dropped.sum()} fake images have no `model` among the 8 GenImage generators: "
              f"predicted, but left out of the metrics.")
    if not keep.any():
        print("No images with a usable ground truth: metrics not computed.")
        return
    binary = task == "detection"
    metrics = {"checkpoint": args.checkpoint, "backbone": backbone_id, "task": task,
               **compute_metrics(y[keep], y_pred[keep], scores[keep], binary)}
    if binary and "model" in df.columns:
        metrics["per_generator"] = per_generator_metrics(y, y_pred, scores, df.model.where(df.label == "fake", "real"))
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    print("Metrics: " + "  ".join(f"{k}={metrics[k]:.4f}" for k in ("balanced_accuracy", "f1_score", "ap", "auc")))
    print(f"Metrics -> {os.path.join(out_dir, 'metrics.json')}")


if __name__ == "__main__":
    main()
