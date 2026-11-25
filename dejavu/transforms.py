"""Image preprocessing pipelines and post-processing operations (robustness tests)."""

import io
import random
from io import BytesIO

import cv2
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

PREPROCESSINGS = ["default", "openai", "defake", "univfd"]


# ---------------------------------------------------------------------------
# Preprocessing pipelines (Section 5.1 "Impact of preprocessing", Appendix A.3)
# ---------------------------------------------------------------------------

def openai_clip_preprocess(size=224):
    """OpenAI CLIP: shorter side to `size` (bicubic) + center crop + CLIP normalization."""
    return T.Compose([
        T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(size),
        T.Lambda(lambda img: img.convert("RGB")),
        T.ToTensor(),
        T.Normalize(CLIP_MEAN, CLIP_STD),
    ])


def defake_preprocess(size=224):
    """De-Fake: direct (bilinear) resize to size x size, normalization with mean = std = 0.5."""
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def univfd_preprocess(train, size=224, load_size=256):
    """UnivFD (official code). Train: shorter side to 256 + blur/JPEG augmentation +
    random crop + horizontal flip. Test: center crop only, no resize/augmentation."""
    ops = []
    if train:
        ops += [T.Lambda(lambda img: TF.resize(img, load_size, interpolation=Image.BILINEAR)),
                T.Lambda(_univfd_augment),
                T.RandomCrop(size),
                T.RandomHorizontalFlip()]
    else:
        ops += [T.CenterCrop(size)]
    ops += [T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD)]
    return T.Compose(ops)


def _univfd_augment(img):
    img = np.array(img)
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    if random.random() < 0.5:  # blur_prob
        sigma = random.random() * 3.0  # blur_sig = (0, 3)
        for c in range(3):
            gaussian_filter(img[:, :, c], output=img[:, :, c], sigma=sigma)
    if random.random() < 0.5:  # jpg_prob
        method = random.choice(("cv2", "pil"))
        quality = random.choice(range(30, 101))
        if method == "cv2":
            _, enc = cv2.imencode(".jpg", img[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            img = cv2.imdecode(enc, 1)[:, :, ::-1]
        else:
            buf = BytesIO()
            Image.fromarray(img).save(buf, format="jpeg", quality=quality)
            img = np.array(Image.open(buf))
    return Image.fromarray(np.ascontiguousarray(img))


def build_transforms(backbone, preprocess="default"):
    """Return (train_transform, eval_transform) for a LoadedBackbone.

    The alternative pipelines use the backbone's own native input resolution
    (e.g. 224, or 336 for ViT-L/14@336px) so that positional embeddings match.
    """
    if preprocess in (None, "default"):
        return backbone.default_preprocess, backbone.default_preprocess
    size = backbone.native_resolution
    if preprocess == "openai":
        t = openai_clip_preprocess(size)
        return t, t
    if preprocess == "defake":
        t = defake_preprocess(size)
        return t, t
    if preprocess == "univfd":
        return univfd_preprocess(True, size), univfd_preprocess(False, size)
    raise ValueError(f"Unknown preprocessing '{preprocess}'. Choices: {PREPROCESSINGS}")


# ---------------------------------------------------------------------------
# Post-processing operations (Section 5.3 "Robustness Evaluation")
# ---------------------------------------------------------------------------

def jpeg(image: Image.Image, qf: int) -> Image.Image:
    buf = io.BytesIO()
    image.save(buf, "JPEG", quality=qf, optimize=True)
    buf.seek(0)
    return Image.open(buf).copy()


def gaussian_blur(image: Image.Image, sigma: float) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(sigma))


def resize(image: Image.Image, factor: float) -> Image.Image:
    w, h = max(1, int(image.width * factor)), max(1, int(image.height * factor))
    return image.resize((w, h), Image.LANCZOS)


def random_crop(image: Image.Image, ratio: float) -> Image.Image:
    """Random crop keeping `ratio` of each side."""
    w, h = image.size
    nw, nh = int(w * ratio), int(h * ratio)
    left, top = random.randint(0, w - nw), random.randint(0, h - nh)
    return image.crop((left, top, left + nw, top + nh))


def center_crop(image: Image.Image, ratio: float) -> Image.Image:
    """Center crop keeping `ratio` of each side."""
    w, h = image.size
    nw, nh = int(w * ratio), int(h * ratio)
    left, top = (w - nw) // 2, (h - nh) // 2
    return image.crop((left, top, left + nw, top + nh))


# name -> (function, filename tag, values evaluated in the paper)
DEGRADATIONS = {
    "jpeg":        (jpeg,          "qf",     [95, 90, 85, 80, 75, 70, 65, 60]),
    "blur":        (gaussian_blur, "blur",   [1, 2, 3, 4]),
    "resize":      (resize,        "resize", [0.5, 0.7, 0.9, 1.1, 1.3, 1.5]),
    "crop":        (random_crop,   "crop",   [0.9, 0.7, 0.5, 0.3]),
    "center_crop": (center_crop,   "ccrop",  [0.9, 0.7, 0.5, 0.3]),
}


def format_value(v) -> str:
    """2.0 -> '2', 0.5 -> '0.5' (used in feature file names)."""
    v = float(v)
    return str(int(v)) if v.is_integer() else str(v)


def degradation_tag(degradation):
    """('jpeg', 75) -> 'qf75'; None -> ''."""
    if not degradation:
        return ""
    name, value = degradation
    return f"{DEGRADATIONS[name][1]}{format_value(value)}"


def apply_degradation(image, degradation):
    if not degradation:
        return image
    name, value = degradation
    fn = DEGRADATIONS[name][0]
    return fn(image, int(value) if name == "jpeg" else float(value))
