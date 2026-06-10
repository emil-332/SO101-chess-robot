"""Tests for square grounding / homography geometry"""

import pytest

from chess_robot.chess.board_state import FILES, RANKS, Square
from chess_robot.perception.square_grounding import (
    BoardCorners,
    CornerSquareGrounder,
    FixedCornerDetector,
    grid_from_corners,
)


def _axis_aligned_corners(size: float = 80.0) -> BoardCorners:
    span = size * 8
    return BoardCorners(a1=(0.0, 0.0), h1=(span, 0.0), h8=(span, span), a8=(0.0, span))


def test_axis_aligned_grounding_matches_regular_grid() -> None:
    grid = grid_from_corners(_axis_aligned_corners(80.0))
    regions = grid.regions
    assert regions is not None
    assert len(regions) == 64

    a1 = regions[Square("a", 1)]
    assert (a1.x_min, a1.y_min, a1.x_max, a1.y_max) == pytest.approx(
        (0.0, 0.0, 80.0, 80.0)
    )
    h8 = regions[Square("h", 8)]
    assert (h8.x_min, h8.y_min, h8.x_max, h8.y_max) == pytest.approx(
        (560.0, 560.0, 640.0, 640.0)
    )
    # b1 spans canonical [1,2]x[0,1] -> centre at (120, 40)
    assert regions[Square("b", 1)].center == pytest.approx((120.0, 40.0))


def test_grid_covers_all_64_squares() -> None:
    regions = grid_from_corners(_axis_aligned_corners()).regions
    assert regions is not None
    for file in FILES:
        for rank in RANKS:
            assert Square(file, rank) in regions


def test_corner_square_grounder_uses_detector() -> None:
    grounder = CornerSquareGrounder(FixedCornerDetector(_axis_aligned_corners(100.0)))
    regions = grounder.ground(image=None).regions
    assert regions is not None
    a1 = regions[Square("a", 1)]
    assert (a1.x_min, a1.y_min, a1.x_max, a1.y_max) == pytest.approx(
        (0.0, 0.0, 100.0, 100.0)
    )


def test_perspective_grounding_is_monotonic() -> None:
    # Trapezoid (oblique-ish): top edge narrower than bottom. Square ordering must
    # still hold: files advance +x along the bottom, ranks advance +y upward.
    corners = BoardCorners(
        a1=(0.0, 0.0), h1=(800.0, 0.0), h8=(600.0, 400.0), a8=(200.0, 400.0)
    )
    regions = grid_from_corners(corners).regions
    assert regions is not None
    a1c = regions[Square("a", 1)].center
    h1c = regions[Square("h", 1)].center
    a8c = regions[Square("a", 8)].center
    assert h1c[0] > a1c[0]
    assert a8c[1] > a1c[1]


def test_grid_exposes_exact_quads() -> None:
    grid = grid_from_corners(_axis_aligned_corners(80.0))
    quads = grid.quads
    assert quads is not None
    assert len(quads) == 64
    a1 = quads[Square("a", 1)]
    assert a1.corners == (
        pytest.approx((0.0, 0.0)),
        pytest.approx((80.0, 0.0)),
        pytest.approx((80.0, 80.0)),
        pytest.approx((0.0, 80.0)),
    )
    # the AABB region is the quad's bounding box
    regions = grid.regions
    assert regions is not None
    assert a1.bounding_region() == regions[Square("a", 1)]


def test_board_corners_reject_non_distinct() -> None:
    with pytest.raises(ValueError):
        BoardCorners(a1=(0.0, 0.0), h1=(0.0, 0.0), h8=(8.0, 8.0), a8=(0.0, 8.0))


def test_board_corners_reject_non_convex_order() -> None:
    # Self-intersecting "bowtie" ordering (edges a1-h1 and h8-a8 cross): not a board.
    with pytest.raises(ValueError):
        BoardCorners(a1=(0.0, 0.0), h1=(8.0, 8.0), h8=(8.0, 0.0), a8=(0.0, 8.0))


def test_board_corners_rotated_round_trips() -> None:
    corners = _axis_aligned_corners(80.0)
    assert corners.rotated(4) == corners
    rotated = corners.rotated(1)
    assert rotated != corners
    assert rotated.a1 == corners.a8  # one quarter turn relabels the corners
