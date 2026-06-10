"""Config for the two-stage piece classifier (occupancy + piece identity).

Torch-free on purpose: the training script's ``--dry-run`` and the dataset-prep
tooling load this without importing torch, so it runs on the laptop. The heavy
training/inference code (``piece_cnn``, ``piece_cnn_onnx``) reads the same config.

Design follows chesscog (Wölflein & Lange 2021): one classifier decides
occupancy (empty vs occupied) on a square-footprint crop, a second classifies the
12 piece identities on a taller crop that captures the piece rising above its
square. See ``docs/perception_piece_cnn.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

# Stage backbones we support (torchvision). InceptionV3 (chesscog's piece head)
# is intentionally omitted from the default to keep ONNX export simple and GPU
# cost low; add it here + in piece_cnn.build_model if exact replication is needed.
SUPPORTED_BACKBONES = ("resnet18", "resnet34", "mobilenet_v3_small", "efficientnet_b0")

Size = tuple[int, int]  # (height, width)


@dataclass(frozen=True)
class StageConfig:
    """One classifier stage (occupancy or piece)."""

    backbone: str
    input_size: Size  # (height, width) the crop is resized to
    top_pad_ratio: float  # upward crop padding (fraction of square height)

    def __post_init__(self) -> None:
        if self.backbone not in SUPPORTED_BACKBONES:
            raise ValueError(
                f"unsupported backbone {self.backbone!r}; "
                f"expected one of {SUPPORTED_BACKBONES}"
            )
        h, w = self.input_size
        if h <= 0 or w <= 0:
            raise ValueError(f"input_size must be positive, got {self.input_size}")
        if self.top_pad_ratio < 0:
            raise ValueError(f"top_pad_ratio must be >= 0, got {self.top_pad_ratio}")


@dataclass(frozen=True)
class TwoStageConfig:
    """Parsed two-stage piece-classifier config (see configs/perception/piece_cnn.yaml)."""

    occupancy: StageConfig
    piece: StageConfig
    pretrained: bool
    camera: str  # which camera frame the classifier crops from (side/oblique)
    batch_size: int
    head_epochs: int  # phase 1: train the new head only
    full_epochs: int  # phase 2: fine-tune the whole network
    head_lr: float
    full_lr: float
    weight_decay: float
    seed: int
    device: str
    val_fraction: float


def _as_size(raw: Any, name: str) -> Size:
    if not isinstance(raw, list | tuple) or len(raw) != 2:
        raise ValueError(f"{name} must be a [height, width] pair, got {raw!r}")
    return (int(raw[0]), int(raw[1]))


def _stage_from(raw: Any, *, default_backbone: str, default_pad: float) -> StageConfig:
    data = raw if isinstance(raw, Mapping) else {}
    return StageConfig(
        backbone=str(data.get("backbone", default_backbone)),
        input_size=_as_size(data.get("input", [100, 100]), "input"),
        top_pad_ratio=float(data.get("top_pad_ratio", default_pad)),
    )


def load_two_stage_config(path: str | Path) -> TwoStageConfig:
    """Load a :class:`TwoStageConfig` from a YAML config file."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    root = raw.get("perception", {}) if isinstance(raw, Mapping) else {}
    training = root.get("training", {})
    training = training if isinstance(training, Mapping) else {}
    return TwoStageConfig(
        occupancy=_stage_from(
            root.get("occupancy", {}),
            default_backbone="resnet18",
            default_pad=0.3,
        ),
        piece=_stage_from(
            root.get("piece", {}),
            default_backbone="resnet34",
            default_pad=1.0,
        ),
        pretrained=bool(root.get("pretrained", True)),
        camera=str(root.get("camera", "observation.images.side")),
        batch_size=int(training.get("batch_size", 64)),
        head_epochs=int(training.get("head_epochs", 1)),
        full_epochs=int(training.get("full_epochs", 3)),
        head_lr=float(training.get("head_lr", 1e-3)),
        full_lr=float(training.get("full_lr", 1e-4)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
        seed=int(training.get("seed", 1000)),
        device=str(training.get("device", "cuda")),
        val_fraction=float(training.get("val_fraction", 0.15)),
    )


def smoke_two_stage_config(config: TwoStageConfig) -> TwoStageConfig:
    """A cheap profile: 1 epoch per phase, tiny batch, CPU-friendly.

    Used by ``scripts/train_piece_cnn.py --smoke`` to confirm the training code
    runs end-to-end for a few cents of GPU (or on CPU for a tiny dataset).
    """
    return replace(
        config,
        batch_size=min(config.batch_size, 8),
        head_epochs=1,
        full_epochs=1,
    )
