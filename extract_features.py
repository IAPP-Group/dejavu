#!/usr/bin/env python
"""Extract frozen-backbone features for the DejaVu splits.

Features are saved to <features_dir>/<backbone>/<split>[__<variant>].pt, where the
variant encodes the preprocessing, the intermediate layer and the post-processing
operation (e.g. `genimage_test__qf75.pt`, `genimage_train__defake.pt`,
`genimage_train__block22.pt`). Feature files are keyed by the dataset-relative image
paths of the split CSVs, so they do not depend on where the datasets are stored.

Examples
--------
  # final-layer features of DejaVu's backbone on GenImage train/validation/test
  python extract_features.py --backbones MetaCLIP2-ViT-bigG-14 --splits genimage --data_config configs/data_paths.yaml

  # all 22 backbones, GenImage and the four OOD benchmarks (validation and test)
  python extract_features.py --backbones all --splits genimage ood --data_config configs/data_paths.yaml

  # alternative preprocessing / every intermediate layer
  python extract_features.py --backbones DejaVu --splits genimage --preprocess defake --data_config ...
  python extract_features.py --backbones DejaVu --splits genimage --all_layers --data_config ...

  # robustness: GenImage test images JPEG-compressed at QF 75
  python extract_features.py --backbones DejaVu --splits genimage_test --jpeg 75 --data_config ...
"""

import argparse
import os
import random
import sys
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dejavu.backbones import BACKBONE_IDS, LoadedBackbone, expand_backbones, get_backbone
from dejavu.data import (EXTRACTION_GROUPS, SPLITS, ImageDataset, PathResolver, expand_extraction_splits,
                         feature_path, save_features, variant_tag)
from dejavu.transforms import DEGRADATIONS, PREPROCESSINGS, build_transforms, degradation_tag


def get_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbones", nargs="+", required=True, metavar="NAME",
                   help="Backbone ids/paper names, 'all', or a family: clip, metaclip2, dino. "
                        f"Ids: {', '.join(BACKBONE_IDS)} (alias 'DejaVu' = MetaCLIP2-ViT-bigG-14).")
    p.add_argument("--splits", nargs="+", default=["genimage"], metavar="SPLIT",
                   help=f"Splits or groups ({', '.join(EXTRACTION_GROUPS)}). Default: genimage. "
                        f"Splits: {', '.join(n for n, s in SPLITS.items() if s.extractable)}.")
    p.add_argument("--data_config", default=None, help="YAML with dataset roots (see configs/data_paths.yaml).")
    p.add_argument("--data_root", default=None,
                   help="Folder holding one sub-folder per dataset (GenImage, ImageNet, ...); "
                        "an alternative/default for the roots of --data_config.")
    p.add_argument("--features_dir", default="features", help="Output folder (default: %(default)s).")

    m = p.add_argument_group("Feature variant")
    m.add_argument("--preprocess", default="default", choices=PREPROCESSINGS,
                   help="Image preprocessing (default: the backbone's own).")
    lg = m.add_mutually_exclusive_group()
    lg.add_argument("--layers", nargs="+", default=None, metavar="LAYER",
                    help="Intermediate layer(s) instead of the final embedding: block<i> for ViTs "
                         "([CLS] token), layer1..layer4 for CLIP ResNets. Extracted in a single pass.")
    lg.add_argument("--all_layers", action="store_true", help="Every intermediate layer (one file each).")

    d = p.add_argument_group("Post-processing (robustness, Section 5.3); at most one")
    dg = d.add_mutually_exclusive_group()
    dg.add_argument("--jpeg", type=int, metavar="QF", help=f"JPEG quality factor (paper: {DEGRADATIONS['jpeg'][2]}).")
    dg.add_argument("--blur", type=float, metavar="SIGMA", help=f"Gaussian blur sigma (paper: {DEGRADATIONS['blur'][2]}).")
    dg.add_argument("--resize", type=float, metavar="R", help=f"Resize factor (paper: {DEGRADATIONS['resize'][2]}).")
    dg.add_argument("--crop", type=float, metavar="C", help=f"Random crop, kept side fraction (paper: {DEGRADATIONS['crop'][2]}).")
    dg.add_argument("--center_crop", type=float, metavar="C", help=f"Center crop, kept side fraction (paper: {DEGRADATIONS['center_crop'][2]}).")

    o = p.add_argument_group("Other")
    o.add_argument("--fake_qf96", choices=["auto", "yes", "no"], default="auto",
                   help="JPEG QF=96 compression of fake images before extraction. auto (default): as in "
                        "the paper, applied to the GenImage splits and not to the OOD benchmarks.")
    o.add_argument("--batch_size", type=int, default=32)
    o.add_argument("--num_workers", type=int, default=4)
    o.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32",
                   help="Precision of HuggingFace backbones (paper: float32).")
    o.add_argument("--seed", type=int, default=0, help="Seed (random crops, UnivFD augmentation).")
    o.add_argument("--overwrite", action="store_true", help="Re-extract existing feature files.")
    return p


def get_degradation(args):
    for name in DEGRADATIONS:
        value = getattr(args, name)
        if value is not None:
            return (name, value)
    return None


def main():
    args = get_parser().parse_args()
    try:
        backbones = expand_backbones(args.backbones)
        splits = expand_extraction_splits(args.splits)
    except ValueError as e:
        sys.exit(f"[ERROR] {e}")
    degradation = get_degradation(args)
    dtag = degradation_tag(degradation)
    resolver = PathResolver(args.data_config, args.data_root)
    fake_qf96 = {"auto": None, "yes": True, "no": False}[args.fake_qf96]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Backbones: {backbones}\nSplits: {splits}\nPreprocess: {args.preprocess}  Degradation: {dtag or 'none'}  "
          f"Device: {device}")

    for bid in backbones:
        spec = get_backbone(bid)

        def out_file(split, layer=None):
            return feature_path(args.features_dir, bid, split, variant_tag(args.preprocess, layer, dtag))

        model = None
        for split in splits:
            layers_hint = args.layers or ([None] if not args.all_layers else None)
            if (not args.overwrite and layers_hint is not None
                    and all(os.path.exists(out_file(split, l)) for l in layers_hint)):
                print(f"[skip] {bid} / {split}: already extracted")
                continue
            if model is None:
                print(f"\n=== Loading {spec.paper_name} ({spec.checkpoint})")
                model = LoadedBackbone(spec, device, getattr(torch, args.dtype))
                train_tf, eval_tf = build_transforms(model, args.preprocess)
                layers = model.layer_names if args.all_layers else args.layers
            if layers and not args.overwrite and all(os.path.exists(out_file(split, l)) for l in layers):
                print(f"[skip] {bid} / {split}: already extracted")
                continue

            random.seed(args.seed)
            torch.manual_seed(args.seed)
            ds = ImageDataset(split, resolver, train_tf if SPLITS[split].is_train else eval_tf,
                              degradation=degradation, fake_qf96=fake_qf96)
            loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                                shuffle=False, pin_memory=device == "cuda")
            print(f"--- {bid} / {split}: {len(ds)} images (fake QF96: {ds.fake_qf96})")
            t0 = time.time()
            paths, labels = [], []
            feats = {l: [] for l in layers} if layers else []
            for rel_paths, x, y in tqdm(loader, unit="batch", leave=False):
                out = model.encode(x, layers)
                if layers:
                    for l in layers:
                        feats[l].append(out[l])
                else:
                    feats.append(out)
                paths += list(rel_paths)
                labels.append(y)
            labels = torch.cat(labels)
            meta = {"backbone": bid, "split": split, "preprocess": args.preprocess, "degradation": dtag or None,
                    "fake_qf96": ds.fake_qf96, "dtype": args.dtype}
            for l in (layers or [None]):
                f = torch.cat(feats[l] if l else feats)
                save_features(out_file(split, l), paths, f, labels, {**meta, "layer": l})
            print(f"    saved {len(paths)} x {f.shape[1]} features -> {out_file(split, (layers or [None])[-1])}"
                  + (f" (+{len(layers) - 1} layer files)" if layers and len(layers) > 1 else "")
                  + f" in {time.time() - t0:.0f}s")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
