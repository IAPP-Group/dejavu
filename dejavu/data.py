"""Splits, image path resolution, image datasets and feature files."""

import os
from dataclasses import dataclass, field

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import apply_degradation, jpeg

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS_DIR = os.path.join(PACKAGE_ROOT, "data", "splits")

# The 8 GenImage generators = classes of the attribution task (alphabetical order).
GENERATORS = ["adm", "biggan", "glide", "midjourney", "sdv4", "sdv5", "vqdm", "wukong"]
TASKS = ["detection", "attribution", "detection+attribution"]
OOD_DATASETS = ["synthbuster", "aigibench", "cospy", "communityforensics"]


def task_classes(task):
    if task == "detection":
        return ["real", "fake"]
    if task == "attribution":
        return list(GENERATORS)
    if task == "detection+attribution":
        return ["real"] + GENERATORS
    raise ValueError(f"Unknown task '{task}'. Choices: {TASKS}")


# ---------------------------------------------------------------------------
# Split registry
# ---------------------------------------------------------------------------

@dataclass
class Split:
    name: str
    # Fake images are JPEG-compressed at QF=96 before feature extraction, to match the
    # compression of the (JPEG) ImageNet real images. As in the experiments of the
    # paper this is done for all GenImage splits, and never for the OOD benchmarks.
    fake_qf96: bool = False
    is_train: bool = False
    # Splits whose feature files hold this split's images. Empty = the split itself.
    feature_sources: list = field(default_factory=list)

    @property
    def csv(self):
        return os.path.join(SPLITS_DIR, f"{self.name}.csv")

    @property
    def sources(self):
        return self.feature_sources or [self.name]

    @property
    def extractable(self):
        return not self.feature_sources


SPLITS = {}
for _s in ["train", "validation", "test"]:
    SPLITS[f"genimage_{_s}"] = Split(f"genimage_{_s}", fake_qf96=True, is_train=_s == "train")
for _d in OOD_DATASETS:
    for _s in ["validation", "test"]:
        SPLITS[f"{_d}_{_s}"] = Split(f"{_d}_{_s}")
for _s in ["validation", "test"]:
    # images of the GenImage generators found in the OOD benchmarks (+ their real images for
    # the joint task): their features are those extracted for the four OOD datasets
    for _t in ["ood_attribution", "ood_detection_attribution"]:
        SPLITS[f"{_t}_{_s}"] = Split(f"{_t}_{_s}", feature_sources=[f"{d}_{_s}" for d in OOD_DATASETS])

EXTRACTION_GROUPS = {
    "genimage": ["genimage_train", "genimage_validation", "genimage_test"],
    "ood_validation": [f"{d}_validation" for d in OOD_DATASETS],
    "ood_test": [f"{d}_test" for d in OOD_DATASETS],
}
EXTRACTION_GROUPS["ood"] = EXTRACTION_GROUPS["ood_validation"] + EXTRACTION_GROUPS["ood_test"]
EXTRACTION_GROUPS["all"] = EXTRACTION_GROUPS["genimage"] + EXTRACTION_GROUPS["ood"]


def ood_eval_sets(task, splits=("validation", "test")):
    """OOD evaluation sets of a task (Sections 4.1.1, 4.2.1, 4.3.1)."""
    if task == "detection":
        return [f"{d}_{s}" for s in splits for d in OOD_DATASETS]
    prefix = "ood_attribution" if task == "attribution" else "ood_detection_attribution"
    return [f"{prefix}_{s}" for s in splits]


def expand_eval_sets(names, task):
    """Expand 'genimage', 'ood', 'ood_validation', 'ood_test' into split names for `task`."""
    out = []
    for n in names:
        if n == "genimage":
            out += ["genimage_validation", "genimage_test"]
        elif n == "ood":
            out += ood_eval_sets(task)
        elif n in ("ood_validation", "ood_test"):
            out += ood_eval_sets(task, (n.split("_")[1],))
        elif n in SPLITS:
            out.append(n)
        else:
            raise ValueError(f"Unknown evaluation set '{n}'. Choices: genimage, ood, ood_validation, "
                             f"ood_test, {', '.join(SPLITS)}")
    return list(dict.fromkeys(out))


def expand_extraction_splits(names):
    out = []
    for n in names:
        if n in EXTRACTION_GROUPS:
            out += EXTRACTION_GROUPS[n]
        elif n in SPLITS:
            out += SPLITS[n].sources
        else:
            raise ValueError(f"Unknown split '{n}'. Choices: {', '.join(list(EXTRACTION_GROUPS) + list(SPLITS))}")
    return list(dict.fromkeys(out))


def load_split(name):
    return pd.read_csv(SPLITS[name].csv)


# ---------------------------------------------------------------------------
# Image path resolution
# ---------------------------------------------------------------------------

# Official GenImage release: folder, relative to the GenImage root, of the `<generator>_imagenet`
# folder of the split CSVs (whose fakes are in its `ai/` sub-folder, as in the official `val/ai`).
OFFICIAL_GENIMAGE_LAYOUT = {
    "adm_imagenet": "ADM/imagenet_ai_0508_adm/val",
    "biggan_imagenet": "BigGAN/imagenet_ai_0419_biggan/val",
    "glide_imagenet": "glide/imagenet_glide/val",
    "midjourney_imagenet": "Midjourney/imagenet_midjourney/val",
    "sdv4_imagenet": "stable_diffusion_v_1_4/imagenet_ai_0419_sdv4/val",
    "sdv5_imagenet": "stable_diffusion_v_1_5/imagenet_ai_0424_sdv5/val",
    "vqdm_imagenet": "VQDM/imagenet_ai_0419_vqdm/val",
    "wukong_imagenet": "wukong/imagenet_ai_0424_wukong/val",
}
DATASET_KEYS = ["GenImage", "ImageNet", "Synthbuster", "RAISE", "AIGIBench", "CO-SPY", "CommunityForensics"]


class PathResolver:
    """Map the dataset-relative paths of the split CSVs (e.g. `GenImage/adm_imagenet/ai/0_adm_153.PNG`,
    `ImageNet/ILSVRC2012_val_00000001_n01751748.JPEG`) to files on disk.

    Each CSV path starts with a dataset key (see DATASET_KEYS) and the rest is the path inside
    that dataset's root. The root is taken from the YAML config (`roots:`), or is
    `<data_root>/<key>` when only `data_root` is given. Optionally, `genimage_layout` maps the
    first folder of GenImage paths (`<generator>_imagenet`) to another folder, e.g. the
    official GenImage structure (OFFICIAL_GENIMAGE_LAYOUT, used with `genimage_layout: official`).
    """

    def __init__(self, config_path=None, data_root=None):
        cfg = {}
        if config_path:
            import yaml
            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}
        data_root = data_root or cfg.get("data_root")
        self.roots = {k: os.path.join(data_root, k) for k in DATASET_KEYS} if data_root else {}
        self.roots.update({k: v for k, v in (cfg.get("roots") or {}).items() if v})
        layout = cfg.get("genimage_layout") or {}
        self.genimage_layout = dict(OFFICIAL_GENIMAGE_LAYOUT) if layout == "official" else dict(layout)

    def __call__(self, rel_path):
        key, rest = rel_path.split("/", 1)
        if key not in self.roots:
            raise KeyError(f"No root configured for dataset '{key}' (path {rel_path}). "
                           f"Set it under `roots:` in the data config, or pass --data_root.")
        if key == "GenImage" and self.genimage_layout:
            folder, rest = rest.split("/", 1)
            rest = os.path.join(self.genimage_layout.get(folder, folder), rest)
        return os.path.join(self.roots[key], rest)


class ImageDataset(Dataset):
    """Images of a split, with the optional QF=96 compression of fakes and one degradation."""

    def __init__(self, split, resolver, transform, degradation=None, fake_qf96=None):
        self.df = load_split(split)
        self.resolver = resolver
        self.transform = transform
        self.degradation = degradation
        self.fake_qf96 = SPLITS[split].fake_qf96 if fake_qf96 is None else fake_qf96

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        rel_path = self.df.image_path.iat[idx]
        is_fake = self.df.label.iat[idx] == "fake"
        path = self.resolver(rel_path)
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Cannot open image '{path}': {e}") from e
        if is_fake and self.fake_qf96:
            image = jpeg(image, 96)
        image = apply_degradation(image, self.degradation)
        return rel_path, self.transform(image), int(is_fake)


# ---------------------------------------------------------------------------
# Feature files: <features_dir>/<backbone>/<split>[__<variant>].pt
# ---------------------------------------------------------------------------

def variant_tag(preprocess=None, layer=None, degradation_tag=""):
    """e.g. ('defake', 'block22', 'qf75') -> 'defake__block22__qf75' (empty for the default setting)."""
    parts = [p for p in (None if preprocess in (None, "default") else preprocess, layer, degradation_tag) if p]
    return "__".join(parts)


def feature_path(features_dir, backbone_id, split, variant=""):
    name = split + (f"__{variant}" if variant else "") + ".pt"
    return os.path.join(features_dir, backbone_id, name)


def save_features(path, rel_paths, features, labels, meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"paths": list(rel_paths), "features": features, "labels": labels, "meta": meta}, path)


def load_features(features_dir, backbone_id, split, variant=""):
    """Return {rel_path: index} and the [N, D] feature tensor of all source files of `split`."""
    paths, feats = [], []
    for source in SPLITS[split].sources:
        p = feature_path(features_dir, backbone_id, source, variant)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Feature file not found: {p}\n"
                                    f"Extract it first with extract_features.py (split '{source}').")
        d = torch.load(p, map_location="cpu")
        paths += d["paths"]
        feats.append(d["features"].float())
    return {p: i for i, p in enumerate(paths)}, torch.cat(feats)
