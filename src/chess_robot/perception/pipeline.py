"""Assemble a runnable board-perception pipeline from config.

Ties the pieces together: the trained two-stage piece classifier (ONNX, run with
onnxruntime on the laptop) plus square grounding, behind the
:class:`BoardPerception` interface. ``configs/perception/perception.yaml`` points
to the model files and the per-camera board-corner calibration, so swapping the
chesscog base for the few-shot fine-tuned model, or recalibrating for a new
board, is a config change with no code edits.

Grounding currently uses the calibration bootstrap (hand-supplied corners +
homography); the learned YOLO corner detector slots in here later as a
``SquareGrounder`` without touching callers. See ``docs/perception_piece_cnn.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from chess_robot.perception.board_perception import (
    DEFAULT_GROUNDING_CAMERA,
    DEFAULT_PIECE_CAMERA,
    ComposedBoardPerception,
)
from chess_robot.perception.piece_locator import PieceClassifier
from chess_robot.perception.square_grounding import (
    BoardCorners,
    CornerSquareGrounder,
    FixedCornerDetector,
    SquareGrounder,
)

_DEFAULT_PIECE_CNN_CONFIG = "configs/perception/piece_cnn.yaml"


@dataclass(frozen=True)
class PerceptionConfig:
    """Parsed perception-pipeline config (see configs/perception/perception.yaml)."""

    piece_cnn_config: Path
    occupancy_onnx: str
    piece_onnx: str
    grounding_camera: str
    piece_camera: str
    overhead_corners: BoardCorners | None
    side_corners: BoardCorners | None


def _corners_from(raw: Any) -> BoardCorners | None:
    """Build :class:`BoardCorners` from a ``{a1: [x, y], ...}`` map, or None.

    Returns None when calibration is absent/incomplete (any corner missing or
    null), so an uncalibrated board is reported clearly rather than guessed.
    """
    if not isinstance(raw, Mapping):
        return None
    points: list[tuple[float, float]] = []
    for name in ("a1", "h1", "h8", "a8"):
        value = raw.get(name)
        if not isinstance(value, list | tuple) or len(value) != 2:
            return None
        points.append((float(value[0]), float(value[1])))
    return BoardCorners(a1=points[0], h1=points[1], h8=points[2], a8=points[3])


def _calibration_block(root: Any) -> Mapping[str, Any]:
    """The corner-calibration block: from ``calibration_file`` if set, else inline.

    A referenced-but-missing ``calibration_file`` yields an empty block (treated as
    uncalibrated), which is the expected state before the board is calibrated.
    """
    calibration_file = root.get("calibration_file")
    if calibration_file:
        path = Path(str(calibration_file))
        if not path.exists():
            return {}
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        block = data.get("calibration", {}) if isinstance(data, Mapping) else {}
        return block if isinstance(block, Mapping) else {}
    inline = root.get("calibration", {})
    return inline if isinstance(inline, Mapping) else {}


def load_perception_config(path: str | Path) -> PerceptionConfig:
    """Load a :class:`PerceptionConfig` from a YAML config file."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    root = raw.get("perception", {}) if isinstance(raw, Mapping) else {}
    models = root.get("models", {}) if isinstance(root.get("models"), Mapping) else {}
    cameras = root.get("cameras", {}) if isinstance(root.get("cameras"), Mapping) else {}
    calibration = _calibration_block(root)
    return PerceptionConfig(
        piece_cnn_config=Path(str(root.get("piece_cnn_config", _DEFAULT_PIECE_CNN_CONFIG))),
        occupancy_onnx=str(models.get("occupancy_onnx", "")),
        piece_onnx=str(models.get("piece_onnx", "")),
        grounding_camera=str(cameras.get("grounding", DEFAULT_GROUNDING_CAMERA)),
        piece_camera=str(cameras.get("piece", DEFAULT_PIECE_CAMERA)),
        overhead_corners=_corners_from(calibration.get("overhead")),
        side_corners=_corners_from(calibration.get("side")),
    )


def _corner_dict(corners: BoardCorners) -> dict[str, list[float]]:
    return {
        "a1": list(corners.a1),
        "h1": list(corners.h1),
        "h8": list(corners.h8),
        "a8": list(corners.a8),
    }


def write_calibration_file(
    path: str | Path,
    *,
    overhead: BoardCorners | None = None,
    side: BoardCorners | None = None,
) -> None:
    """Write a lab-specific calibration YAML (consumed via ``calibration_file``)."""
    block: dict[str, Any] = {}
    if overhead is not None:
        block["overhead"] = _corner_dict(overhead)
    if side is not None:
        block["side"] = _corner_dict(side)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump({"calibration": block}, sort_keys=False), encoding="utf-8")


def _grounder(corners: BoardCorners) -> SquareGrounder:
    return CornerSquareGrounder(FixedCornerDetector(corners))


def _load_onnx_classifier(config: PerceptionConfig) -> PieceClassifier:
    # Lazy import: onnxruntime (the `perception` extra) is only needed at run time.
    from chess_robot.perception.piece_cnn_config import load_two_stage_config
    from chess_robot.perception.piece_cnn_onnx import OnnxTwoStageClassifier

    paths = {"occupancy_onnx": config.occupancy_onnx, "piece_onnx": config.piece_onnx}
    for label, path in paths.items():
        if not path or not Path(path).exists():
            raise FileNotFoundError(
                f"{label} not found at {path!r}; set it in the perception config "
                "and place the weights under models/ (see models/README.md)"
            )
    cnn_config = load_two_stage_config(config.piece_cnn_config)
    return OnnxTwoStageClassifier.from_paths(cnn_config, config.occupancy_onnx, config.piece_onnx)


def build_board_perception(
    config: PerceptionConfig, *, classifier: PieceClassifier | None = None
) -> ComposedBoardPerception:
    """Build the composed perception pipeline from config.

    ``classifier`` can be injected (tests / a different backend); otherwise the
    ONNX two-stage classifier is loaded from the config's model paths. The piece
    camera must be calibrated (it is what the CNN crops from). If the overhead
    camera is also calibrated it grounds/highlights squares; otherwise the piece
    camera is grounded for both.
    """
    if config.side_corners is None:
        raise ValueError(
            "piece camera is not calibrated: fill calibration.side (a1,h1,h8,a8) "
            "in the perception config with scripts/annotate_corners.py"
        )
    if classifier is None:
        classifier = _load_onnx_classifier(config)
    piece_grounder = _grounder(config.side_corners)
    if config.overhead_corners is not None and config.grounding_camera != config.piece_camera:
        return ComposedBoardPerception(
            _grounder(config.overhead_corners),
            classifier,
            grounding_camera=config.grounding_camera,
            piece_grounder=piece_grounder,
            piece_camera=config.piece_camera,
        )
    return ComposedBoardPerception(
        piece_grounder, classifier, grounding_camera=config.piece_camera
    )


def load_frame(path: str | Path) -> np.ndarray:
    """Load a camera frame from a ``.npy`` array or an image file (``.png`` etc.)."""
    file = Path(path)
    if file.suffix == ".npy":
        return np.load(file)
    from PIL import Image  # lazy: only for image-file inputs

    return np.asarray(Image.open(file).convert("RGB"))
