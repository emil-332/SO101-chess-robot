"""Tests for the perception pipeline factory (config -> BoardPerception)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from chess_robot.chess.board_state import BoardState
from chess_robot.perception.board_renderer import ColorPieceClassifier, render_board
from chess_robot.perception.piece_locator import PieceClassifier
from chess_robot.perception.pipeline import (
    PerceptionConfig,
    build_board_perception,
    load_frame,
    load_perception_config,
)
from chess_robot.perception.square_grounding import BoardCorners

_REPO = Path(__file__).resolve().parents[1]
_CONFIG = _REPO / "configs" / "perception" / "perception.yaml"
IMAGE_SIZE = (240, 240)
SIDE = "observation.images.side"
OVERHEAD = "observation.images.overhead"


def _corners() -> BoardCorners:
    h, w = IMAGE_SIZE
    mx, my = 0.08 * w, 0.08 * h
    return BoardCorners(a1=(mx, h - my), h1=(w - mx, h - my), h8=(w - mx, my), a8=(mx, my))


def _config(*, overhead: BoardCorners | None, side: BoardCorners | None) -> PerceptionConfig:
    return PerceptionConfig(
        piece_cnn_config=_REPO / "configs" / "perception" / "piece_cnn.yaml",
        occupancy_onnx="",
        piece_onnx="",
        grounding_camera=OVERHEAD,
        piece_camera=SIDE,
        overhead_corners=overhead,
        side_corners=side,
    )


def _color_classifier() -> PieceClassifier:
    return ColorPieceClassifier(camera=SIDE)


def test_shipped_config_loads_uncalibrated() -> None:
    config = load_perception_config(_CONFIG)
    assert config.occupancy_onnx == "models/piece_cnn_chesscog_2026-06-11/occupancy.onnx"
    assert config.piece_onnx == "models/piece_cnn_chesscog_2026-06-11/piece.onnx"
    assert config.piece_camera == SIDE
    # ships uncalibrated (null corners) so it cannot silently guess geometry
    assert config.side_corners is None
    assert config.overhead_corners is None


def test_build_requires_piece_calibration() -> None:
    with pytest.raises(ValueError, match="not calibrated"):
        build_board_perception(_config(overhead=None, side=None), classifier=_color_classifier())


def test_single_camera_pipeline_recovers_board() -> None:
    board = BoardState.standard_starting_position()
    rendered = render_board(board, _corners(), image_size=IMAGE_SIZE)
    perception = build_board_perception(
        _config(overhead=None, side=_corners()), classifier=_color_classifier()
    )
    result = perception.perceive({SIDE: rendered.image})
    assert result.board_state == board
    assert result.source == "perception"


def test_two_camera_pipeline_grounds_both() -> None:
    board = BoardState.from_fen("8/8/8/3QQ3/3qq3/8/8/8")
    rendered = render_board(board, _corners(), image_size=IMAGE_SIZE)
    overhead_frame = np.zeros((*IMAGE_SIZE, 3), dtype=np.uint8)
    perception = build_board_perception(
        _config(overhead=_corners(), side=_corners()), classifier=_color_classifier()
    )
    result = perception.perceive({OVERHEAD: overhead_frame, SIDE: rendered.image})
    assert result.board_state == board
    assert result.grids is not None
    assert OVERHEAD in result.grids and SIDE in result.grids


def test_load_frame_npy(tmp_path: Path) -> None:
    array = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    path = tmp_path / "frame.npy"
    np.save(path, array)
    assert np.array_equal(load_frame(path), array)
