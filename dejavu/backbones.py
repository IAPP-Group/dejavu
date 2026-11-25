"""Registry and loading of the 22 frozen backbones studied in the paper.

Each backbone has a filename-safe canonical id (used in feature file names and
output folders), the name used in the paper, and the checkpoint it is loaded
from. Paper names (e.g. ``MetaCLIP2-ViT-bigG/14``, ``RN50×64``,
``DINOv3-ViT-H+/16``) and the short names of the original code base
(e.g. ``MetaCLIP2-Giant``) are accepted as aliases everywhere.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Backbone:
    id: str          # canonical, filename-safe id
    paper_name: str  # name used in the paper
    family: str      # "clip" | "metaclip2" | "dinov2" | "dinov3"
    checkpoint: str  # OpenAI CLIP model name or HuggingFace repo id
    legacy_name: str  # name used by the original (pre-release) code
    feature_dim: int


_B = Backbone
BACKBONES = [
    # OpenAI CLIP (https://github.com/openai/CLIP)
    _B("ViT-B-32",       "ViT-B/32",       "clip", "ViT-B/32",       "ViT-B32",       512),
    _B("ViT-B-16",       "ViT-B/16",       "clip", "ViT-B/16",       "ViT-B16",       512),
    _B("ViT-L-14",       "ViT-L/14",       "clip", "ViT-L/14",       "ViT-L14",       768),
    _B("ViT-L-14@336px", "ViT-L/14@336px", "clip", "ViT-L/14@336px", "ViT-L14@336px", 768),
    _B("RN50",           "RN50",           "clip", "RN50",           "RN50",          1024),
    _B("RN101",          "RN101",          "clip", "RN101",          "RN101",         512),
    _B("RN50x4",         "RN50×4",         "clip", "RN50x4",         "RN50x4",        640),
    _B("RN50x16",        "RN50×16",        "clip", "RN50x16",        "RN50x16",       768),
    _B("RN50x64",        "RN50×64",        "clip", "RN50x64",        "RN50x64",       1024),
    # MetaCLIP 2 (HuggingFace transformers)
    _B("MetaCLIP2-ViT-S-16",    "MetaCLIP2-ViT-S/16",    "metaclip2", "facebook/metaclip-2-worldwide-s16",           "MetaCLIP2-S16",   384),
    _B("MetaCLIP2-ViT-M-16",    "MetaCLIP2-ViT-M/16",    "metaclip2", "facebook/metaclip-2-worldwide-m16",           "MetaCLIP2-M16",   512),
    _B("MetaCLIP2-ViT-H-14",    "MetaCLIP2-ViT-H/14",    "metaclip2", "facebook/metaclip-2-worldwide-huge-quickgelu", "MetaCLIP2-H14",   1024),
    _B("MetaCLIP2-ViT-bigG-14", "MetaCLIP2-ViT-bigG/14", "metaclip2", "facebook/metaclip-2-worldwide-giant",         "MetaCLIP2-Giant", 1280),
    # DINOv2 (open weights)
    _B("DINOv2-ViT-S-14", "DINOv2-ViT-S/14", "dinov2", "facebook/dinov2-small", "DINOv2-S", 384),
    _B("DINOv2-ViT-B-14", "DINOv2-ViT-B/14", "dinov2", "facebook/dinov2-base",  "DINOv2-B", 768),
    _B("DINOv2-ViT-L-14", "DINOv2-ViT-L/14", "dinov2", "facebook/dinov2-large", "DINOv2-L", 1024),
    _B("DINOv2-ViT-g-14", "DINOv2-ViT-g/14", "dinov2", "facebook/dinov2-giant", "DINOv2-G", 1536),
    # DINOv3 (gated on the HuggingFace Hub: accept the license, then `huggingface-cli login`)
    _B("DINOv3-ViT-S-16",     "DINOv3-ViT-S/16",  "dinov3", "facebook/dinov3-vits16-pretrain-lvd1689m",     "DINOv3-ViT-S16",     384),
    _B("DINOv3-ViT-B-16",     "DINOv3-ViT-B/16",  "dinov3", "facebook/dinov3-vitb16-pretrain-lvd1689m",     "DINOv3-ViT-B16",     768),
    _B("DINOv3-ViT-L-16",     "DINOv3-ViT-L/16",  "dinov3", "facebook/dinov3-vitl16-pretrain-lvd1689m",     "DINOv3-ViT-L16",     1024),
    _B("DINOv3-ViT-Hplus-16", "DINOv3-ViT-H+/16", "dinov3", "facebook/dinov3-vith16plus-pretrain-lvd1689m", "DINOv3-ViT-H16Plus", 1280),
    _B("DINOv3-ViT-7B-16",    "DINOv3-ViT-7B/16", "dinov3", "facebook/dinov3-vit7b16-pretrain-lvd1689m",    "DINOv3-ViT-7B16",    4096),
]

BACKBONE_IDS = [b.id for b in BACKBONES]
DEJAVU_BACKBONE = "MetaCLIP2-ViT-bigG-14"
FAMILIES = {
    "clip": [b.id for b in BACKBONES if b.family == "clip"],
    "metaclip2": [b.id for b in BACKBONES if b.family == "metaclip2"],
    "dino": [b.id for b in BACKBONES if b.family in ("dinov2", "dinov3")],
}


def _norm(name: str) -> str:
    return (name.lower().replace("×", "x").replace("+", "plus")
            .replace("/", "").replace("-", "").replace("_", "").replace(" ", ""))


_ALIASES = {}
for _b in BACKBONES:
    for _name in (_b.id, _b.paper_name, _b.legacy_name):
        _ALIASES[_norm(_name)] = _b
_ALIASES[_norm("DejaVu")] = next(b for b in BACKBONES if b.id == DEJAVU_BACKBONE)


def get_backbone(name: str) -> Backbone:
    """Resolve a canonical id, a paper name or a legacy name to a Backbone."""
    try:
        return _ALIASES[_norm(name)]
    except KeyError:
        raise ValueError(f"Unknown backbone '{name}'. Available: {', '.join(BACKBONE_IDS)}") from None


def expand_backbones(names):
    """Expand 'all', family names ('clip', 'metaclip2', 'dino') and aliases."""
    out = []
    for name in names:
        if name == "all":
            out += BACKBONE_IDS
        elif name in FAMILIES:
            out += FAMILIES[name]
        else:
            out.append(get_backbone(name).id)
    return list(dict.fromkeys(out))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

class LoadedBackbone:
    """A frozen backbone with a uniform interface over OpenAI CLIP and HF models."""

    def __init__(self, spec: Backbone, device: str = "cuda", dtype: torch.dtype = torch.float32):
        self.spec = spec
        self.device = device
        self.processor = None
        if spec.family == "clip":
            import clip
            self.model, self.default_preprocess = clip.load(spec.checkpoint, device=device)
        else:
            from transformers import AutoModel
            # the paper uses float32; half precision is offered to fit DINOv3-ViT-7B/16 on smaller GPUs
            kwargs = {} if dtype == torch.float32 else {"dtype": dtype}
            try:
                self.model = AutoModel.from_pretrained(spec.checkpoint, **kwargs).to(device)
            except Exception as e:  # most often: gated DINOv3 repo without HF login
                raise RuntimeError(
                    f"Could not load {spec.checkpoint}. DINOv3 checkpoints are gated: accept the "
                    f"license at https://huggingface.co/{spec.checkpoint} and run `huggingface-cli login`. "
                    f"MetaCLIP 2 needs a recent transformers version (see requirements.txt). "
                    f"Underlying error: {e}") from e
            if spec.family == "metaclip2":
                from transformers import AutoProcessor
                self.processor = AutoProcessor.from_pretrained(spec.checkpoint)
            else:
                from transformers import AutoImageProcessor
                self.processor = AutoImageProcessor.from_pretrained(spec.checkpoint)
            processor = self.processor
            self.default_preprocess = (
                lambda img: processor(images=img, return_tensors="pt")["pixel_values"][0])
        self.model.eval()
        self.input_dtype = next(self.model.parameters()).dtype

    # -- geometry ------------------------------------------------------------

    @property
    def native_resolution(self) -> int:
        """Square input size the backbone was trained for (used by the alternative preprocessings)."""
        if self.spec.family == "clip":
            return self.model.visual.input_resolution
        image_processor = getattr(self.processor, "image_processor", self.processor)
        size = getattr(image_processor, "crop_size", None) or getattr(image_processor, "size", None)
        if hasattr(size, "get"):
            return size.get("height") or size.get("shortest_edge")
        return size

    def _blocks(self):
        f = self.spec.family
        if f == "metaclip2":
            return list(self.model.vision_model.encoder.layers)
        if f == "dinov2":
            return list(self.model.encoder.layer)
        if f == "dinov3":
            return list(self.model.model.layer)
        if self.spec.id.startswith("ViT"):
            return list(self.model.visual.transformer.resblocks)
        return [m for n, m in self.model.visual.named_children() if n.startswith("layer")]

    @property
    def layer_names(self):
        """Intermediate layers: block0..blockN-1 (ViTs) or layer1..layer4 (CLIP ResNets)."""
        if self.spec.family == "clip" and not self.spec.id.startswith("ViT"):
            return [n for n, _ in self.model.visual.named_children() if n.startswith("layer")]
        return [f"block{i}" for i in range(len(self._blocks()))]

    # -- forward ---------------------------------------------------------------

    def _forward(self, x):
        f = self.spec.family
        if f == "metaclip2":
            out = self.model.get_image_features(pixel_values=x)
            # depending on the transformers version this is a tensor or a
            # BaseModelOutputWithPooling whose pooler_output holds the projected features
            return out if torch.is_tensor(out) else out.pooler_output
        if f in ("dinov2", "dinov3"):
            return self.model(pixel_values=x).pooler_output  # layer-normed [CLS] token
        return self.model.encode_image(x)  # projected CLIP image embedding

    def _pool_intermediate(self, feats):
        if isinstance(feats, tuple):
            feats = feats[0]
        if self.spec.family != "clip":
            return feats[:, 0, :]          # HF ViTs are batch-first: [CLS] token
        if self.spec.id.startswith("ViT"):
            return feats[0]                # OpenAI ViT is sequence-first: [CLS] token
        return F.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)  # ResNet stage: GAP

    @torch.no_grad()
    def encode(self, x, layers=None):
        """Return the final embedding [B, D], or {layer: [B, D_l]} if `layers` is given."""
        x = x.to(self.device, dtype=self.input_dtype)
        if not layers:
            return self._forward(x).float().cpu()
        modules = dict(zip(self.layer_names, self._blocks()))
        unknown = [l for l in layers if l not in modules]
        if unknown:
            raise ValueError(f"Unknown layer(s) {unknown} for {self.spec.id}. Available: {self.layer_names}")
        captured, handles = {}, []
        for name in layers:
            handles.append(modules[name].register_forward_hook(
                lambda m, i, o, name=name: captured.__setitem__(name, o)))
        try:
            self._forward(x)
        finally:
            for h in handles:
                h.remove()
        return {name: self._pool_intermediate(captured[name]).float().cpu() for name in layers}
