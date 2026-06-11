"""Tests for the board mapper (  1.3)."""

import pytest

from chess_robot.chess.board_mapper import (
    BoardMapper,
    BoardPoint,
    GroundedGrid,
    ImageRegion,
    SquareNotGroundedError,
)
from chess_robot.chess.board_state import Square


def _region_grid() -> GroundedGrid:
    return GroundedGrid(
        regions={
            Square("b", 1): ImageRegion(10.0, 20.0, 30.0, 50.0),
            Square("c", 3): ImageRegion(100.0, 100.0, 120.0, 130.0),
        }
    )


def test_region_for_valid_square() -> None:
    mapper = BoardMapper(_region_grid())
    region = mapper.region_for("b1")
    assert region == ImageRegion(10.0, 20.0, 30.0, 50.0)
    assert region.center == (20.0, 35.0)
    assert region.width == 20.0
    assert region.height == 30.0


def test_accepts_square_or_name_equivalently() -> None:
    mapper = BoardMapper(_region_grid())
    assert mapper.region_for("b1") == mapper.region_for(Square("b", 1))


def test_region_for_invalid_name_raises_value_error() -> None:
    mapper = BoardMapper(_region_grid())
    with pytest.raises(ValueError):
        mapper.region_for("z9")


def test_region_for_ungrounded_square_raises() -> None:
    mapper = BoardMapper(_region_grid())
    with pytest.raises(SquareNotGroundedError):
        mapper.region_for("a1")


def test_coordinate_missing_when_only_regions_grounded() -> None:
    mapper = BoardMapper(_region_grid())
    assert not mapper.has_coordinate("b1")
    with pytest.raises(SquareNotGroundedError):
        mapper.coordinate_for("b1")


def test_regular_board_coordinates_orientation() -> None:
    grid = GroundedGrid.from_regular_board_coordinates(
        a1_center=BoardPoint(0.0, 0.0, 0.1), square_size=0.05
    )
    mapper = BoardMapper(grid)
    assert mapper.coordinate_for("a1") == BoardPoint(0.0, 0.0, 0.1)
    # files advance +x, ranks advance +y
    h1 = mapper.coordinate_for("h1")
    assert h1.x == pytest.approx(0.35)
    assert h1.y == pytest.approx(0.0)
    assert h1.z == pytest.approx(0.1)
    a8 = mapper.coordinate_for("a8")
    assert a8.x == pytest.approx(0.0)
    assert a8.y == pytest.approx(0.35)
    assert not mapper.has_region("a1")


def test_regular_board_coordinates_rejects_nonpositive_size() -> None:
    with pytest.raises(ValueError):
        GroundedGrid.from_regular_board_coordinates(BoardPoint(0.0, 0.0), 0.0)


def test_degenerate_image_region_rejected() -> None:
    with pytest.raises(ValueError):
        ImageRegion(10.0, 10.0, 10.0, 20.0)  # zero width
    with pytest.raises(ValueError):
        ImageRegion(10.0, 10.0, 20.0, 5.0)  # negative height


def test_has_region_and_coordinate_flags() -> None:
    mapper = BoardMapper(_region_grid())
    assert mapper.has_region("b1")
    assert not mapper.has_region("a1")
    assert not mapper.has_coordinate("b1")
