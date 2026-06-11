"""Perception validation metrics

The board-perception model is validated **independently** of manipulation,
because the deterministic move logic trusts its output (see
``docs/evaluation.md``). This module implements the perception metrics:

- ``square_grounding_accuracy``    — square-name -> region (IoU >= threshold)
- ``occupancy_accuracy``           — occupied/empty per square
- ``piece_classification_accuracy``— piece type per occupied square
- ``capture_detection_accuracy``   — target-occupied (capture) flagged correctly
- ``zero_shot_board_generalization``— board-read accuracy on **held-out** boards

:func:`evaluate_perception` runs a :class:`BoardPerception` over labeled
:class:`PerceptionSample` s and micro-averages each metric. Metrics with no data
report ``None`` rather than a misleading ``0.0``. This is separate from the
manipulation evaluation
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chess_robot.chess.board_mapper import GroundedGrid, ImageRegion
from chess_robot.chess.board_state import NUM_SQUARES, BoardState, Square
from chess_robot.perception.board_perception import BoardPerception, CameraFrames
from chess_robot.perception.square_grounding import BoardCorners, grid_from_corners


def region_iou(a: ImageRegion, b: ImageRegion) -> float:
    """Intersection-over-union of two axis-aligned image regions."""
    inter_w = max(0.0, min(a.x_max, b.x_max) - max(a.x_min, b.x_min))
    inter_h = max(0.0, min(a.y_max, b.y_max) - max(a.y_min, b.y_min))
    intersection = inter_w * inter_h
    union = a.width * a.height + b.width * b.height - intersection
    return intersection / union if union > 0 else 0.0


def _grounding_counts(
    predicted: GroundedGrid, truth: GroundedGrid, iou_threshold: float
) -> tuple[int, int]:
    truth_regions = truth.regions
    if not truth_regions:
        return 0, 0
    predicted_regions = predicted.regions or {}
    correct = 0
    for square, truth_region in truth_regions.items():
        predicted_region = predicted_regions.get(square)
        if (
            predicted_region is not None
            and region_iou(predicted_region, truth_region) >= iou_threshold
        ):
            correct += 1
    return correct, len(truth_regions)


def _occupancy_counts(predicted: BoardState, truth: BoardState) -> tuple[int, int]:
    correct = 0
    for index in range(NUM_SQUARES):
        square = Square.from_index(index)
        if predicted.is_occupied(square) == truth.is_occupied(square):
            correct += 1
    return correct, NUM_SQUARES


def _piece_counts(predicted: BoardState, truth: BoardState) -> tuple[int, int]:
    correct = 0
    total = 0
    for index in range(NUM_SQUARES):
        square = Square.from_index(index)
        truth_piece = truth.piece_at(square)
        if truth_piece is None:
            continue
        total += 1
        predicted_piece = predicted.piece_at(square)
        if predicted_piece is not None and predicted_piece.piece_type is truth_piece.piece_type:
            correct += 1
    return correct, total


def _capture_counts(
    predicted: BoardState, truth: BoardState, targets: Iterable[Square]
) -> tuple[int, int]:
    correct = 0
    total = 0
    for target in targets:
        total += 1
        if predicted.is_occupied(target) == truth.is_occupied(target):
            correct += 1
    return correct, total


def _board_read_counts(predicted: BoardState, truth: BoardState) -> tuple[int, int]:
    """Per-square exact read: both empty, or both occupied with the same type."""
    correct = 0
    for index in range(NUM_SQUARES):
        square = Square.from_index(index)
        truth_piece = truth.piece_at(square)
        predicted_piece = predicted.piece_at(square)
        if truth_piece is None and predicted_piece is None:
            correct += 1
        elif (
            truth_piece is not None
            and predicted_piece is not None
            and predicted_piece.piece_type is truth_piece.piece_type
        ):
            correct += 1
    return correct, NUM_SQUARES


def _ratio(correct: int, total: int) -> float | None:
    return correct / total if total > 0 else None


def square_grounding_accuracy(
    predicted: GroundedGrid, truth: GroundedGrid, *, iou_threshold: float = 0.5
) -> float | None:
    return _ratio(*_grounding_counts(predicted, truth, iou_threshold))


def occupancy_accuracy(predicted: BoardState, truth: BoardState) -> float | None:
    return _ratio(*_occupancy_counts(predicted, truth))


def piece_classification_accuracy(
    predicted: BoardState, truth: BoardState
) -> float | None:
    return _ratio(*_piece_counts(predicted, truth))


def capture_detection_accuracy(
    predicted: BoardState, truth: BoardState, targets: Iterable[Square]
) -> float | None:
    return _ratio(*_capture_counts(predicted, truth, targets))


@dataclass(frozen=True)
class PerceptionSample:
    """One labeled perception example.

    ``frames`` is the model input (may be empty for the metadata bootstrap).
    ``ground_truth_grid`` is optional (only scored when present). ``held_out``
    marks an unseen board type for zero-shot measurement. ``capture_targets`` are
    the target squares to score capture detection on.
    """

    ground_truth_board: BoardState
    frames: CameraFrames = field(default_factory=dict)
    ground_truth_grid: GroundedGrid | None = None
    board_type: str = "default"
    held_out: bool = False
    capture_targets: tuple[Square, ...] = ()


@dataclass(frozen=True)
class PerceptionReport:
    """Micro-averaged perception metrics. ``None`` == no data for that metric."""

    square_grounding_accuracy: float | None
    occupancy_accuracy: float | None
    piece_classification_accuracy: float | None
    capture_detection_accuracy: float | None
    zero_shot_board_generalization: float | None
    num_samples: int
    num_held_out: int


def evaluate_perception(
    perception: BoardPerception,
    samples: Iterable[PerceptionSample],
    *,
    iou_threshold: float = 0.5,
) -> PerceptionReport:
    """Run ``perception`` over ``samples`` and micro-average the metrics."""
    g_c = g_t = 0
    o_c = o_t = 0
    p_c = p_t = 0
    cap_c = cap_t = 0
    z_c = z_t = 0
    num_samples = 0
    num_held_out = 0

    for sample in samples:
        num_samples += 1
        predicted = perception.perceive(sample.frames)

        if sample.ground_truth_grid is not None:
            c, t = _grounding_counts(
                predicted.grid, sample.ground_truth_grid, iou_threshold
            )
            g_c += c
            g_t += t

        c, t = _occupancy_counts(predicted.board_state, sample.ground_truth_board)
        o_c += c
        o_t += t

        c, t = _piece_counts(predicted.board_state, sample.ground_truth_board)
        p_c += c
        p_t += t

        if sample.capture_targets:
            c, t = _capture_counts(
                predicted.board_state, sample.ground_truth_board, sample.capture_targets
            )
            cap_c += c
            cap_t += t

        if sample.held_out:
            num_held_out += 1
            c, t = _board_read_counts(
                predicted.board_state, sample.ground_truth_board
            )
            z_c += c
            z_t += t

    return PerceptionReport(
        square_grounding_accuracy=_ratio(g_c, g_t),
        occupancy_accuracy=_ratio(o_c, o_t),
        piece_classification_accuracy=_ratio(p_c, p_t),
        capture_detection_accuracy=_ratio(cap_c, cap_t),
        zero_shot_board_generalization=_ratio(z_c, z_t),
        num_samples=num_samples,
        num_held_out=num_held_out,
    )


def _corners_from_dict(corners: dict[str, Any]) -> BoardCorners:
    def point(name: str) -> tuple[float, float]:
        value = corners[name]
        return float(value[0]), float(value[1])

    return BoardCorners(
        a1=point("a1"), h1=point("h1"), h8=point("h8"), a8=point("a8")
    )


def _sample_from_record(
    record: dict[str, Any], frame_loader: Callable[[str], object] | None
) -> PerceptionSample:
    grid: GroundedGrid | None = None
    corners = record.get("overhead_corners")
    if corners is not None:
        grid = grid_from_corners(_corners_from_dict(corners))

    frames: dict[str, object] = {}
    for camera_key, manifest_field in (
        ("observation.images.overhead", "overhead"),
        ("observation.images.side", "side"),
    ):
        path = record.get(manifest_field)
        if path is not None:
            frames[camera_key] = frame_loader(path) if frame_loader else path

    targets = tuple(
        Square.from_name(name) for name in record.get("capture_targets", [])
    )
    return PerceptionSample(
        ground_truth_board=BoardState.from_fen(record["fen"]),
        frames=frames,
        ground_truth_grid=grid,
        board_type=record.get("board_type", "default"),
        held_out=bool(record.get("held_out", False)),
        capture_targets=targets,
    )


def load_perception_samples(
    manifest_path: str | Path,
    *,
    frame_loader: Callable[[str], object] | None = None,
) -> list[PerceptionSample]:
    """Load :class:`PerceptionSample` s from a JSONL manifest (one record/line).

    Each record: ``fen`` (required); optional ``overhead`` / ``side`` image
    paths, ``overhead_corners`` (``{a1,h1,h8,a8: [x, y]}``), ``board_type``,
    ``held_out``, and ``capture_targets`` (square names). ``frame_loader`` maps an
    image path to a frame (e.g. a decoded numpy array); if ``None``, the path
    string is kept as the frame (fine for the metadata/bootstrap path).
    """
    text = Path(manifest_path).read_text(encoding="utf-8")
    samples: list[PerceptionSample] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            samples.append(_sample_from_record(json.loads(stripped), frame_loader))
    return samples
