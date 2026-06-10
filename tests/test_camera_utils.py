"""Tests for camera / crop utilities"""

import numpy as np
import pytest

from chess_robot.chess.board_mapper import GroundedGrid, ImageRegion
from chess_robot.chess.board_state import (
    BoardState,
    Piece,
    PieceColor,
    PieceType,
    Square,
)
from chess_robot.perception.camera_utils import (
    LabeledCrop,
    corners_from_points,
    crop_box,
    extract_square_crops,
    highlight_squares,
    labeled_square_crops,
    square_label,
)
from chess_robot.perception.square_grounding import BoardCorners, grid_from_corners

_OVERHEAD = BoardCorners(a1=(0.0, 0.0), h1=(640.0, 0.0), h8=(640.0, 640.0), a8=(0.0, 640.0))


def test_corners_from_points_orders_a1_h1_h8_a8() -> None:
    corners = corners_from_points([(0.0, 0.0), (8.0, 0.0), (8.0, 8.0), (0.0, 8.0)])
    assert corners == BoardCorners((0.0, 0.0), (8.0, 0.0), (8.0, 8.0), (0.0, 8.0))


def test_corners_from_points_requires_four() -> None:
    with pytest.raises(ValueError):
        corners_from_points([(0.0, 0.0)])


def test_crop_box_extends_upward_and_clips_top() -> None:
    region = ImageRegion(0.0, 240.0, 80.0, 320.0)
    assert crop_box(region, 640, 640, top_pad_ratio=1.0) == (0, 160, 80, 320)
    # near the top edge the upward extension clips to 0
    top = ImageRegion(0.0, 0.0, 80.0, 80.0)
    assert crop_box(top, 640, 640, top_pad_ratio=1.0) == (0, 0, 80, 80)


def test_crop_box_side_padding_and_right_clip() -> None:
    region = ImageRegion(560.0, 0.0, 640.0, 80.0)
    box = crop_box(region, 640, 640, top_pad_ratio=0.0, side_pad_ratio=0.5)
    assert box == (520, 0, 640, 80)


def test_extract_square_crops_shapes() -> None:
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    crops = extract_square_crops(image, grid_from_corners(_OVERHEAD), top_pad_ratio=0.0)
    assert len(crops) == 64
    assert crops[Square("a", 1)].shape == (80, 80, 3)


def test_square_label() -> None:
    board = BoardState.from_map(
        {Square("b", 1): Piece(PieceType.KNIGHT, PieceColor.WHITE)}
    )
    assert square_label(board, Square("b", 1)) == "white_knight"
    assert square_label(board, Square("c", 3)) == "empty"


def test_labeled_square_crops_auto_labels_from_board() -> None:
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    board = BoardState.standard_starting_position()
    labeled = labeled_square_crops(
        image, grid_from_corners(_OVERHEAD), board, top_pad_ratio=0.0
    )
    assert len(labeled) == 64
    by_square = {crop.square: crop for crop in labeled}
    assert isinstance(by_square[Square("e", 1)], LabeledCrop)
    assert by_square[Square("e", 1)].label == "white_king"
    assert by_square[Square("e", 4)].label == "empty"


def test_highlight_squares_draws_borders_and_copies() -> None:
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    grid = grid_from_corners(_OVERHEAD)
    out = highlight_squares(image, grid, [Square("a", 1)], color=(255, 0, 0), thickness=2)
    # a1 region is the bottom-left 80x80 block at (0,0)-(80,80)
    assert tuple(out[0, 0]) == (255, 0, 0)  # top-left border pixel
    assert tuple(out[40, 40]) == (0, 0, 0)  # interior untouched
    assert tuple(image[0, 0]) == (0, 0, 0)  # input not mutated (copy)


def test_highlight_squares_skips_ungrounded() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    out = highlight_squares(image, GroundedGrid(), [Square("a", 1)])
    assert np.array_equal(out, image)
