"""Two-stage piece classifier: training, evaluation, export, and torch inference.

This is the **cloud-GPU** side (torch + torchvision). It trains two crop
classifiers, chesscog-style (Wölflein & Lange 2021):

1. occupancy: empty vs occupied, on a near-square crop;
2. piece identity: the 12 colour+type classes, on a taller crop.

Each stage uses an ImageNet-pretrained backbone, fine-tuned in two phases (head
only, then the whole network), matching chesscog's transfer recipe. Models export
to ONNX for laptop inference (``piece_cnn_onnx``). Crop extraction, labels, and
normalization come from the torch-free :mod:`piece_dataset`, shared with the ONNX
path so train and inference preprocess identically.

Import this module only where torch is installed (the cloud GPU); the laptop runs
the torch-free dry-runs and ONNX inference instead. See
``docs/perception_piece_cnn.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
from torch.utils.data import DataLoader, Dataset

from chess_robot.chess.board_mapper import GroundedGrid
from chess_robot.chess.board_state import BoardState
from chess_robot.perception.board_perception import CameraFrames
from chess_robot.perception.piece_cnn_config import StageConfig, TwoStageConfig
from chess_robot.perception.piece_dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    OCCUPANCY_CLASSES,
    PIECE_CLASSES,
    board_from_piece_predictions,
    normalize_batch,
    occupancy_inference_crops,
    occupied_from_predictions,
    piece_inference_crops,
    train_val_split,
)
from chess_robot.perception.piece_locator import PieceClassifier

_BACKBONES = {
    "resnet18": (tvm.resnet18, "ResNet18_Weights"),
    "resnet34": (tvm.resnet34, "ResNet34_Weights"),
    "mobilenet_v3_small": (tvm.mobilenet_v3_small, "MobileNet_V3_Small_Weights"),
    "efficientnet_b0": (tvm.efficientnet_b0, "EfficientNet_B0_Weights"),
}


def build_model(name: str, num_classes: int, pretrained: bool) -> tuple[Any, Any]:
    """Build a backbone with a fresh ``num_classes`` head; return ``(model, head)``.

    ``head`` is the final linear layer so the two-phase trainer can freeze the
    backbone and train the head alone in phase 1.
    """
    if name not in _BACKBONES:
        raise ValueError(f"unsupported backbone {name!r}; expected {tuple(_BACKBONES)}")
    constructor, weights_attr = _BACKBONES[name]
    weights = getattr(tvm, weights_attr).DEFAULT if pretrained else None
    model = constructor(weights=weights)
    if name in ("resnet18", "resnet34"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        head = model.fc
    else:  # mobilenet_v3_small / efficientnet_b0: classifier is a Sequential
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        head = model.classifier[-1]
    return model, head


class _ArrayDataset(Dataset):
    """Wrap ``N x H x W x 3`` uint8 crops + int labels as a torch dataset.

    Training applies light augmentation (flip, brightness/contrast jitter, noise);
    all samples are ImageNet-normalized to match inference.
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        *,
        train: bool,
        seed: int = 0,
    ) -> None:
        self._images = images
        self._labels = labels.astype(np.int64)
        self._train = train
        self._mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        self._generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return len(self._labels)

    def _augment(self, image: Any) -> Any:
        if torch.rand(1, generator=self._generator).item() < 0.5:
            image = torch.flip(image, dims=[2])
        brightness = 0.8 + 0.4 * torch.rand(1, generator=self._generator).item()
        image = torch.clamp(image * brightness, 0.0, 1.0)
        noise = 0.02 * torch.randn(image.shape, generator=self._generator)
        return torch.clamp(image + noise, 0.0, 1.0)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        array = self._images[index].astype(np.float32) / 255.0
        image = torch.from_numpy(array).permute(2, 0, 1)
        if self._train:
            image = self._augment(image)
        image = (image - self._mean) / self._std
        return image, int(self._labels[index])


def _set_backbone_frozen(model: Any, head: Any, frozen: bool) -> None:
    for param in model.parameters():
        param.requires_grad_(not frozen)
    for param in head.parameters():
        param.requires_grad_(True)


def _run_epochs(
    model: Any,
    *,
    epochs: int,
    phase: str,
    train_loader: Any,
    val_loader: Any,
    optimizer: Any,
    criterion: Any,
    device: str,
) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        seen = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)
            seen += len(labels)
        val_accuracy, _ = evaluate(model, val_loader, device)
        history.append(
            {
                "phase": phase,
                "epoch": epoch,
                "train_loss": total_loss / max(seen, 1),
                "val_accuracy": val_accuracy,
            }
        )
    return history


def train_two_phase(
    model: Any,
    head: Any,
    train_loader: Any,
    val_loader: Any,
    *,
    config: TwoStageConfig,
    device: str,
) -> list[dict[str, Any]]:
    """Phase 1 trains the head (backbone frozen), phase 2 fine-tunes everything."""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, Any]] = []

    _set_backbone_frozen(model, head, frozen=True)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.head_lr,
        weight_decay=config.weight_decay,
    )
    history += _run_epochs(
        model,
        epochs=config.head_epochs,
        phase="head",
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
    )

    _set_backbone_frozen(model, head, frozen=False)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.full_lr, weight_decay=config.weight_decay
    )
    history += _run_epochs(
        model,
        epochs=config.full_epochs,
        phase="full",
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
    )
    return history


def evaluate(model: Any, loader: Any, device: str) -> tuple[float, np.ndarray]:
    """Return ``(accuracy, confusion_matrix)`` over a loader."""
    model.eval()
    correct = 0
    total = 0
    confusion: dict[tuple[int, int], int] = {}
    num_classes = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            predictions = logits.argmax(dim=1).cpu().numpy()
            truth = labels.numpy()
            num_classes = max(num_classes, int(logits.shape[1]))
            for true_label, pred in zip(truth, predictions, strict=True):
                confusion[(int(true_label), int(pred))] = (
                    confusion.get((int(true_label), int(pred)), 0) + 1
                )
                correct += int(true_label == pred)
                total += 1
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for (true_label, pred), count in confusion.items():
        matrix[true_label, pred] = count
    return (correct / total if total else 0.0), matrix


def predict_indices(model: Any, crops: np.ndarray, device: str) -> np.ndarray:
    """Argmax class indices for a uint8 ``N x H x W x 3`` batch (inference)."""
    if len(crops) == 0:
        return np.empty((0,), dtype=np.int64)
    tensor = torch.from_numpy(normalize_batch(crops)).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
    return logits.argmax(dim=1).cpu().numpy().astype(np.int64)


def export_onnx(model: Any, path: str | Path, input_size: tuple[int, int], device: str) -> None:
    """Export a stage model to ONNX with a dynamic batch axis for laptop inference.

    Prefers the legacy TorchScript exporter (``dynamo=False``): it honors
    ``dynamic_axes`` cleanly, writes a single ``.onnx`` file, and avoids the newer
    dynamo exporter's opset down-conversion step (which logs a noisy, harmless
    failure when converting 18 -> 17). Falls back to the dynamo exporter at opset
    18 if the legacy path is unavailable in this torch version.
    """
    model.eval()
    dummy = torch.randn(1, 3, input_size[0], input_size[1], device=device)
    dynamic_axes = {"input": {0: "batch"}, "logits": {0: "batch"}}
    try:
        torch.onnx.export(
            model,
            dummy,
            str(path),
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=17,
            dynamo=False,
        )
    except Exception:  # legacy exporter unavailable: use the dynamo path at opset 18
        torch.onnx.export(
            model,
            dummy,
            str(path),
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=18,
            dynamo=True,
        )


def save_checkpoint(
    model: Any, path: str | Path, *, stage: StageConfig, classes: tuple[str, ...]
) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "backbone": stage.backbone,
            "input_size": list(stage.input_size),
            "classes": list(classes),
        },
        str(path),
    )


def load_stage_model(
    stage: StageConfig, num_classes: int, checkpoint_path: str | Path, device: str
) -> Any:
    """Rebuild a stage model and load weights from a checkpoint."""
    model, _ = build_model(stage.backbone, num_classes, pretrained=False)
    state = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(state["state_dict"])
    model.to(device)
    model.eval()
    return model


def _train_one_stage(
    stage: StageConfig,
    images: np.ndarray,
    labels: np.ndarray,
    *,
    num_classes: int,
    classes: tuple[str, ...],
    config: TwoStageConfig,
    device: str,
    out_dir: Path,
    name: str,
    export: bool,
    val_images: np.ndarray | None = None,
    val_labels: np.ndarray | None = None,
    init_checkpoint: Path | None = None,
) -> dict[str, Any]:
    if len(labels) == 0:
        raise ValueError(f"no {name} samples in dataset; prepare data first")
    if val_images is not None and val_labels is not None and len(val_labels) > 0:
        # honest held-out evaluation on a separate set; train on all of `images`
        train_x, train_y = images, labels
        val_x, val_y = val_images, val_labels
    else:
        train_x, train_y, val_x, val_y = train_val_split(
            images, labels, val_fraction=config.val_fraction, seed=config.seed
        )
        if len(val_y) == 0:  # tiny datasets: evaluate on the training set
            val_x, val_y = train_x, train_y
    train_loader = DataLoader(
        _ArrayDataset(train_x, train_y, train=True, seed=config.seed),
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        _ArrayDataset(val_x, val_y, train=False),
        batch_size=config.batch_size,
        shuffle=False,
    )
    model, head = build_model(stage.backbone, num_classes, pretrained=config.pretrained)
    if init_checkpoint is not None and init_checkpoint.exists():
        # warm-start from an existing checkpoint (e.g. the chesscog base) for the
        # few-shot fine-tune, instead of ImageNet weights.
        state = torch.load(str(init_checkpoint), map_location=device)
        model.load_state_dict(state["state_dict"])
    history = train_two_phase(
        model, head, train_loader, val_loader, config=config, device=device
    )
    accuracy, confusion = evaluate(model, val_loader, device)
    checkpoint_path = out_dir / f"{name}.pt"
    save_checkpoint(model, checkpoint_path, stage=stage, classes=classes)
    onnx_path = out_dir / f"{name}.onnx"
    if export:
        export_onnx(model, onnx_path, stage.input_size, device)
    return {
        "val_accuracy": accuracy,
        "confusion": confusion,
        "history": history,
        "checkpoint": str(checkpoint_path),
        "onnx": str(onnx_path) if export else None,
    }


def run_training(
    config: TwoStageConfig,
    dataset: Any,
    *,
    stage: str,
    out_dir: str | Path,
    export: bool = True,
    device: str | None = None,
    val_dataset: Any | None = None,
    init_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Train the requested stage(s) and write checkpoints + ONNX to ``out_dir``.

    If ``val_dataset`` is given, each stage trains on all of ``dataset`` and is
    evaluated on the separate ``val_dataset`` (honest held-out accuracy); otherwise
    an internal split of ``dataset`` is used. If ``init_dir`` is given, each stage
    warm-starts from ``init_dir/{occupancy,piece}.pt`` (the chesscog base) instead
    of ImageNet weights, for the few-shot fine-tune.
    """
    resolved_device = device or config.device
    torch.manual_seed(config.seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    init = Path(init_dir) if init_dir is not None else None
    results: dict[str, Any] = {}
    if stage in ("occupancy", "both"):
        results["occupancy"] = _train_one_stage(
            config.occupancy,
            dataset.occupancy_images,
            dataset.occupancy_labels,
            num_classes=len(OCCUPANCY_CLASSES),
            classes=OCCUPANCY_CLASSES,
            config=config,
            device=resolved_device,
            out_dir=out,
            name="occupancy",
            export=export,
            val_images=None if val_dataset is None else val_dataset.occupancy_images,
            val_labels=None if val_dataset is None else val_dataset.occupancy_labels,
            init_checkpoint=None if init is None else init / "occupancy.pt",
        )
    if stage in ("piece", "both"):
        results["piece"] = _train_one_stage(
            config.piece,
            dataset.piece_images,
            dataset.piece_labels,
            num_classes=len(PIECE_CLASSES),
            classes=PIECE_CLASSES,
            config=config,
            device=resolved_device,
            out_dir=out,
            name="piece",
            export=export,
            val_images=None if val_dataset is None else val_dataset.piece_images,
            val_labels=None if val_dataset is None else val_dataset.piece_labels,
            init_checkpoint=None if init is None else init / "piece.pt",
        )
    return results


class TorchTwoStageClassifier(PieceClassifier):
    """Run the two trained stages with torch; returns occupancy + piece identity.

    Crops the configured (side/oblique) camera with :mod:`piece_dataset`, runs the
    occupancy model over all squares, then the piece model over the occupied ones.
    Implements :class:`PieceClassifier`, so it drops into ``ComposedBoardPerception``.
    """

    def __init__(
        self,
        occupancy_model: Any,
        piece_model: Any,
        config: TwoStageConfig,
        *,
        device: str = "cpu",
    ) -> None:
        self._occupancy_model = occupancy_model
        self._piece_model = piece_model
        self._config = config
        self._device = device
        occupancy_model.to(device).eval()
        piece_model.to(device).eval()

    @classmethod
    def from_checkpoints(
        cls,
        config: TwoStageConfig,
        occupancy_checkpoint: str | Path,
        piece_checkpoint: str | Path,
        *,
        device: str = "cpu",
    ) -> TorchTwoStageClassifier:
        occupancy_model = load_stage_model(
            config.occupancy, len(OCCUPANCY_CLASSES), occupancy_checkpoint, device
        )
        piece_model = load_stage_model(
            config.piece, len(PIECE_CLASSES), piece_checkpoint, device
        )
        return cls(occupancy_model, piece_model, config, device=device)

    def classify(self, frames: CameraFrames, grid: GroundedGrid) -> BoardState:
        image = np.asarray(frames[self._config.camera])
        squares, occupancy_crops = occupancy_inference_crops(
            image,
            grid,
            size=self._config.occupancy.input_size,
            top_pad_ratio=self._config.occupancy.top_pad_ratio,
        )
        if not squares:
            return BoardState.empty()
        occupancy_pred = predict_indices(self._occupancy_model, occupancy_crops, self._device)
        occupied = occupied_from_predictions(squares, occupancy_pred.tolist())
        if not occupied:
            return BoardState.empty()
        piece_crops = piece_inference_crops(
            image,
            grid,
            occupied,
            size=self._config.piece.input_size,
            top_pad_ratio=self._config.piece.top_pad_ratio,
        )
        piece_pred = predict_indices(self._piece_model, piece_crops, self._device)
        return board_from_piece_predictions(occupied, piece_pred.tolist())
