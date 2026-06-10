"""Tests for perception validation metrics"""

import json
from pathlib import Path

import pytest

from chess_robot.chess.board_mapper import GroundedGrid, ImageRegion
from chess_robot.chess.board_state import BoardState, Piece, PieceColor, PieceType, Square
from chess_robot.eval.perception_metrics import (
    PerceptionSample,
    capture_detection_accuracy,
    evaluate_perception,
    load_perception_samples,
    occupancy_accuracy,
    piece_classification_accuracy,
    region_iou,
    square_grounding_accuracy,
)
from chess_robot.perception.board_perception import (
    OCCUPANCY_SOURCE_PERCEPTION,
    BoardPerception,
    ComposedBoardPerception,
    PerceivedBoard,
)
from chess_robot.perception.piece_locator import MetadataPieceClassifier
from chess_robot.perception.square_grounding import (
    BoardCorners,
    CornerSquareGrounder,
    FixedCornerDetector,
    grid_from_corners,
)

_CORNERS = BoardCorners(a1=(0.0, 0.0), h1=(640.0, 0.0), h8=(640.0, 640.0), a8=(0.0, 640.0))


class _ConstantPerception(BoardPerception):
    """Returns a fixed PerceivedBoard regardless of input (test double)."""

    def __init__(self, predicted: PerceivedBoard) -> None:
        self._predicted = predicted

    def perceive(self, frames: object = None) -> PerceivedBoard:
        return self._predicted


# --- region_iou ---------------------------------------------------------------


def test_region_iou_identical_and_disjoint() -> None:
    a = ImageRegion(0.0, 0.0, 10.0, 10.0)
    assert region_iou(a, a) == pytest.approx(1.0)
    far = ImageRegion(100.0, 100.0, 110.0, 110.0)
    assert region_iou(a, far) == pytest.approx(0.0)


def test_region_iou_partial_overlap() -> None:
    a = ImageRegion(0.0, 0.0, 10.0, 10.0)
    b = ImageRegion(5.0, 5.0, 15.0, 15.0)
    # intersection 25, union 175
    assert region_iou(a, b) == pytest.approx(25.0 / 175.0)


# --- grounding accuracy -------------------------------------------------------


def test_square_grounding_accuracy_perfect_and_degraded() -> None:
    truth = grid_from_corners(_CORNERS)
    assert square_grounding_accuracy(truth, truth) == pytest.approx(1.0)

    regions = dict(truth.regions or {})
    regions[Square("a", 1)] = ImageRegion(5000.0, 5000.0, 5010.0, 5010.0)  # far off
    degraded = GroundedGrid(regions=regions)
    assert square_grounding_accuracy(degraded, truth) == pytest.approx(63.0 / 64.0)


# --- occupancy / piece / capture ----------------------------------------------


def test_occupancy_accuracy() -> None:
    truth = BoardState.standard_starting_position()
    assert occupancy_accuracy(truth, truth) == pytest.approx(1.0)
    degraded = truth.without_piece(Square("a", 1))  # one square now wrong
    assert occupancy_accuracy(degraded, truth) == pytest.approx(63.0 / 64.0)


def test_piece_classification_accuracy() -> None:
    truth = BoardState.standard_starting_position()
    assert piece_classification_accuracy(truth, truth) == pytest.approx(1.0)
    # wrong type on one occupied square (32 occupied total)
    wrong = truth.with_piece(Square("a", 1), Piece(PieceType.QUEEN, PieceColor.WHITE))
    assert piece_classification_accuracy(wrong, truth) == pytest.approx(31.0 / 32.0)


def test_capture_detection_accuracy() -> None:
    truth = BoardState.standard_starting_position()
    targets = (Square("e", 1), Square("e", 4))  # occupied, empty
    assert capture_detection_accuracy(truth, truth, targets) == pytest.approx(1.0)
    # predict e1 empty -> capture mis-detected on one of two targets
    wrong = truth.without_piece(Square("e", 1))
    assert capture_detection_accuracy(wrong, truth, targets) == pytest.approx(0.5)


def test_no_data_metrics_return_none() -> None:
    assert square_grounding_accuracy(GroundedGrid(), GroundedGrid()) is None
    assert capture_detection_accuracy(
        BoardState.empty(), BoardState.empty(), ()
    ) is None


# --- evaluate_perception ------------------------------------------------------


def test_evaluate_perfect_composed_perception() -> None:
    board = BoardState.standard_starting_position()
    perception = ComposedBoardPerception(
        CornerSquareGrounder(FixedCornerDetector(_CORNERS)),
        MetadataPieceClassifier(board),
    )
    sample = PerceptionSample(
        ground_truth_board=board,
        frames={
            "observation.images.overhead": "overhead",
            "observation.images.side": "side",
        },
        ground_truth_grid=grid_from_corners(_CORNERS),
        capture_targets=(Square("e", 1), Square("e", 4)),
    )
    report = evaluate_perception(perception, [sample])
    assert report.square_grounding_accuracy == pytest.approx(1.0)
    assert report.occupancy_accuracy == pytest.approx(1.0)
    assert report.piece_classification_accuracy == pytest.approx(1.0)
    assert report.capture_detection_accuracy == pytest.approx(1.0)
    assert report.zero_shot_board_generalization is None  # no held-out samples
    assert report.num_samples == 1


def test_zero_shot_uses_held_out_subset() -> None:
    empty_prediction = PerceivedBoard(
        BoardState.empty(), GroundedGrid(), OCCUPANCY_SOURCE_PERCEPTION
    )
    perception = _ConstantPerception(empty_prediction)
    held_out = PerceptionSample(
        ground_truth_board=BoardState.standard_starting_position(), held_out=True
    )
    report = evaluate_perception(perception, [held_out])
    # 32 empty squares read correctly, 32 occupied squares missed -> 0.5
    assert report.zero_shot_board_generalization == pytest.approx(0.5)
    assert report.occupancy_accuracy == pytest.approx(0.5)
    assert report.piece_classification_accuracy == pytest.approx(0.0)
    assert report.square_grounding_accuracy is None  # no ground-truth grid
    assert report.num_held_out == 1


def test_load_perception_samples_from_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "eval.jsonl"
    record = {
        "overhead": "b/o.jpg",
        "side": "b/s.jpg",
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "overhead_corners": {
            "a1": [0, 0],
            "h1": [640, 0],
            "h8": [640, 640],
            "a8": [0, 640],
        },
        "board_type": "boardB",
        "held_out": True,
        "capture_targets": ["e4"],
    }
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")

    samples = load_perception_samples(manifest)
    assert len(samples) == 1
    sample = samples[0]
    assert sample.board_type == "boardB"
    assert sample.held_out is True
    assert sample.ground_truth_board == BoardState.standard_starting_position()
    assert sample.capture_targets == (Square("e", 4),)
    assert sample.ground_truth_grid is not None
    # no frame_loader -> paths kept as-is
    assert sample.frames["observation.images.overhead"] == "b/o.jpg"
