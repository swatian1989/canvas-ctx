"""Pathology foundation model encoders with graceful degradation.

The paper uses MUSK. MUSK, UNI, Virchow and CONCH are gated on HuggingFace and
need approved access plus a token. Phikon and CTransPath are not. On a CPU-only
machine, encoding is the bottleneck, so embeddings are always cached to disk and
the classifier head is trained on cached features.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable

import torch

logger = logging.getLogger(__name__)


@dataclass
class EncoderSpec:
    name: str
    hf_repo: str | None
    gated: bool
    input_px: int
    embed_dim: int
    notes: str = ""


ENCODERS: dict[str, EncoderSpec] = {
    "musk":       EncoderSpec("musk", "xiangjx/musk", True, 384, 2048,
                              "paper default; 224 patches resized to 384"),
    "uni":        EncoderSpec("uni", "MahmoodLab/UNI", True, 224, 1024, "benchmark"),
    "virchow":    EncoderSpec("virchow", "paige-ai/Virchow", True, 224, 2560, "benchmark"),
    "conch":      EncoderSpec("conch", "MahmoodLab/CONCH", True, 224, 512),
    "phikon":     EncoderSpec("phikon", "owkin/phikon", False, 224, 768,
                              "LAPTOP DEFAULT, ungated"),
    "ctranspath": EncoderSpec("ctranspath", None, False, 224, 768,
                              "weights from the TransPath GitHub release"),
    "resnet50":   EncoderSpec("resnet50", None, False, 224, 2048,
                              "ImageNet floor, the paper's weakest benchmark"),
}


def resolve_encoder(requested: str) -> EncoderSpec:
    """Pick an encoder, falling back if a gated one is unavailable.

    Never raises on a missing token. Falls back and logs loudly, so an overnight
    run does not die at hour six on an auth error.
    """
    if requested not in ENCODERS:
        raise ValueError(f"unknown encoder '{requested}'. Options: {list(ENCODERS)}")
    spec = ENCODERS[requested]
    if spec.gated and not os.environ.get("HF_TOKEN"):
        logger.warning(
            "%s is gated and HF_TOKEN is not set. Falling back to phikon. "
            "Request access at https://huggingface.co/%s and export HF_TOKEN "
            "to match the paper.", spec.name, spec.hf_repo)
        return ENCODERS["phikon"]
    return spec


def _require_hf_token(spec: EncoderSpec) -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            f"{spec.name} is gated on HuggingFace. Request access at "
            f"https://huggingface.co/{spec.hf_repo} and export HF_TOKEN before "
            f"retrying. resolve_encoder() should have fallen back to phikon "
            f"already -- only call load_encoder() on a gated spec directly if "
            f"you intend to bypass that fallback."
        )
    return token


def _load_resnet50() -> tuple[torch.nn.Module, Callable]:
    """ImageNet-pretrained ResNet-50, average-pooled to a 2048-d vector.

    Uses the IMAGENET1K_V2 weight set's own ``.transforms()``: the exact
    resize/crop/normalisation that weight set was trained with, read from
    torchvision rather than hand-copied, so it can't drift out of sync.
    """
    from torchvision.models import ResNet50_Weights, resnet50

    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.fc = torch.nn.Identity()  # drop the classifier head, keep pooled features
    model.eval()
    tv_transform = weights.transforms()

    def preprocess(image) -> torch.Tensor:
        return tv_transform(image)

    return model, preprocess


class _CLSToken(torch.nn.Module):
    """Wraps a HF ViT-style backbone to return its CLS token embedding."""

    def __init__(self, backbone: torch.nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=pixel_values)
        return out.last_hidden_state[:, 0]


def _load_phikon() -> tuple[torch.nn.Module, Callable]:
    """Owkin Phikon, ungated ViT-B/16 pretrained on histology. CLS token, 768-d.

    `AutoImageProcessor` loads Phikon's own preprocessor config (its own
    mean/std, resize, rescale), not a generic ImageNet transform.
    """
    from transformers import AutoImageProcessor, ViTModel

    repo = ENCODERS["phikon"].hf_repo
    processor = AutoImageProcessor.from_pretrained(repo)
    backbone = ViTModel.from_pretrained(repo, add_pooling_layer=False)
    backbone.eval()

    def preprocess(image) -> torch.Tensor:
        return processor(image, return_tensors="pt")["pixel_values"][0]

    return _CLSToken(backbone), preprocess


def _load_uni() -> tuple[torch.nn.Module, Callable]:
    """MahmoodLab UNI, gated ViT-L/16. Needs approved HF access + HF_TOKEN.

    `HF_TOKEN` only needs to be present in the environment: huggingface_hub
    (>=0.19, pinned >=0.21 here) reads it automatically for `hf-hub:` loads,
    so it is never passed or logged explicitly.
    """
    spec = ENCODERS["uni"]
    _require_hf_token(spec)
    import timm
    from timm.data import create_transform, resolve_data_config

    model = timm.create_model(
        f"hf-hub:{spec.hf_repo}", pretrained=True, init_values=1e-5,
        dynamic_img_size=True,
    )
    model.eval()
    tv_transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

    def preprocess(image) -> torch.Tensor:
        return tv_transform(image)

    return model, preprocess


class _MuskGlobalEmbedding(torch.nn.Module):
    """Wraps MUSK to return only its global (CLS) image embedding.

    ``ms_aug=False``: MUSK's multi-scale augmentation is meant for retrieval
    quality, not for a cached, resumable, reproducible embedding store -- a
    cached embedding must be a deterministic function of the patch.
    """

    def __init__(self, backbone: torch.nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.backbone(
            image=pixel_values, with_head=False, out_norm=False,
            ms_aug=False, return_global=True,
        )[0]


def _load_musk() -> tuple[torch.nn.Module, Callable]:
    """MUSK (Xiang et al.), gated BEiT3-based encoder. [PAPER] default on hpc.

    Requires the `musk` package from https://github.com/lilab-stanford/MUSK
    (``pip install git+https://github.com/lilab-stanford/MUSK.git``); it is
    not on PyPI under the `musk` name and is not in requirements.txt because
    it is hpc-tier only. Forced to CPU/float32 here regardless of the
    upstream example's `cuda`/`float16`, since this project is CPU-only.
    """
    spec = ENCODERS["musk"]
    _require_hf_token(spec)
    try:
        import musk.modeling  # noqa: F401  (registers musk_large_patch16_384 with timm)
        from musk import utils as musk_utils
    except ImportError as exc:
        raise RuntimeError(
            "MUSK requires the `musk` package: "
            "pip install git+https://github.com/lilab-stanford/MUSK.git"
        ) from exc
    import torchvision.transforms as T
    from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
    from timm.models import create_model

    model = create_model("musk_large_patch16_384")
    musk_utils.load_model_and_may_interpolate("hf_hub:xiangjx/musk", model, "model|module", "")
    model.to(device="cpu", dtype=torch.float32)
    model.eval()

    tv_transform = T.Compose([
        T.Resize(384, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.CenterCrop((384, 384)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    ])

    def preprocess(image) -> torch.Tensor:
        return tv_transform(image)

    return _MuskGlobalEmbedding(model), preprocess


_LOADERS: dict[str, Callable[[], tuple[object, Callable]]] = {
    "resnet50": _load_resnet50,
    "phikon": _load_phikon,
    "uni": _load_uni,
    "musk": _load_musk,
}


def load_encoder(spec: EncoderSpec) -> tuple[object, Callable]:
    """Return (model, preprocess), each carrying the encoder's OWN statistics.

    `preprocess` maps a PIL RGB image to a `(3, H, W)` float tensor ready to
    stack into a batch and feed to `model`. Never share a transform across
    encoders: Phikon, UNI and MUSK each ship their own mean/std, and treating
    them as interchangeable with a generic ImageNet transform degrades
    embeddings silently rather than raising.
    """
    if spec.name not in _LOADERS:
        raise NotImplementedError(
            f"load_encoder: '{spec.name}' is not wired up. Supported: "
            f"{sorted(_LOADERS)}. virchow/conch/ctranspath are spec'd in "
            f"ENCODERS but not requested for this phase -- ask before adding."
        )
    logger.info("loading encoder %s (repo=%s, gated=%s)",
                spec.name, spec.hf_repo, spec.gated)
    return _LOADERS[spec.name]()
