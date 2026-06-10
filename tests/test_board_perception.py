"""Tests for the board-perception interface, bootstrap, and composition."""

import pytest

from chess_robot.chess.board_mapper import GroundedGrid, ImageRegion
from chess_robot.chess.board_state import (
    BoardState,
    Piece,
    PieceColor,
    PieceType,
    Square,
)
from chess_robot.perception.board_perception import (
    OCCUPANCY_SOURCE_METADATA,
    OCCUPANCY_SOURCE_PERCEPTION,
    BoardPerception,
    ComposedBoardPerception,
    MetadataBoardPerception,
    PerceivedBoard,
    cross_check_occupancy,
)
from chess_robot.perception.piece_locator import (
    CameraFrames,
    MetadataPieceClassifier,
    PieceClassifier,
)
from chess_robot.perception.square_grounding import (
    BoardCorners,
    CornerSquareGrounder,
    FixedCornerDetector,
)


class _SpyClassifier(PieceClassifier):
    """Records the grid it was handed (to check the side grid is passed)."""

    def __init__(self, board: BoardState) -> None:
        self._board = board
        self.received_grid: object = None

    def classify(self, frames: CameraFrames, grid: object) -> BoardState:
        self.received_grid = grid
        return self._board


def test_cannot_instantiate_abstract_interface() -> None:
    with pytest.raises(TypeError):
        BoardPerception()  # type: ignore[abstract]


def test_metadata_perception_implements_interface() -> None:
    assert issubclass(MetadataBoardPerception, BoardPerception)
    assert isinstance(MetadataBoardPerception(BoardState.empty()), BoardPerception)


def test_metadata_perception_returns_supplied_board_and_grid() -> None:
    board = BoardState.standard_starting_position()
    grid = GroundedGrid(regions={Square("e", 1): ImageRegion(0.0, 0.0, 10.0, 10.0)})
    result = MetadataBoardPerception(board, grid).perceive(frames=None)
    assert isinstance(result, PerceivedBoard)
    assert result.board_state is board
    assert result.grid is grid
    assert result.source == OCCUPANCY_SOURCE_METADATA


def test_metadata_perception_ignores_frames() -> None:
    board = BoardState.from_map(
        {Square("b", 1): Piece(PieceType.KNIGHT, PieceColor.WHITE)}
    )
    perception = MetadataBoardPerception(board)
    a = perception.perceive(frames={"observation.images.overhead": "imgA"})
    b = perception.perceive()
    assert a.board_state == b.board_state == board


def test_composed_perception_grounds_and_classifies() -> None:
    # Axis-aligned overhead board: a1 at (0,0), 80px squares -> h8 at (640,640).
    corners = BoardCorners(
        a1=(0.0, 0.0), h1=(640.0, 0.0), h8=(640.0, 640.0), a8=(0.0, 640.0)
    )
    grounder = CornerSquareGrounder(FixedCornerDetector(corners))
    board = BoardState.standard_starting_position()
    perception = ComposedBoardPerception(grounder, MetadataPieceClassifier(board))

    result = perception.perceive(
        {
            "observation.images.overhead": "overhead-frame",
            "observation.images.side": "side-frame",
        }
    )
    assert result.source == OCCUPANCY_SOURCE_PERCEPTION
    assert result.board_state == board

    regions = result.grid.regions
    assert regions is not None
    a1 = regions[Square("a", 1)]
    assert (a1.x_min, a1.y_min, a1.x_max, a1.y_max) == pytest.approx(
        (0.0, 0.0, 80.0, 80.0)
    )


def test_composed_perception_requires_grounding_frame() -> None:
    corners = BoardCorners(
        a1=(0.0, 0.0), h1=(8.0, 0.0), h8=(8.0, 8.0), a8=(0.0, 8.0)
    )
    perception = ComposedBoardPerception(
        CornerSquareGrounder(FixedCornerDetector(corners)),
        MetadataPieceClassifier(BoardState.empty()),
    )
    with pytest.raises(KeyError):
        perception.perceive({"observation.images.side": "only-side"})


def test_composed_perception_grounds_both_cameras_and_feeds_side_grid() -> None:
    overhead = BoardCorners((0.0, 0.0), (640.0, 0.0), (640.0, 640.0), (0.0, 640.0))
    side = BoardCorners((0.0, 0.0), (800.0, 0.0), (800.0, 800.0), (0.0, 800.0))
    board = BoardState.standard_starting_position()
    spy = _SpyClassifier(board)
    perception = ComposedBoardPerception(
        CornerSquareGrounder(FixedCornerDetector(overhead)),
        spy,
        piece_grounder=CornerSquareGrounder(FixedCornerDetector(side)),
    )

    result = perception.perceive(
        {"observation.images.overhead": "o", "observation.images.side": "s"}
    )
    assert result.grids is not None
    assert set(result.grids) == {
        "observation.images.overhead",
        "observation.images.side",
    }
    # the classifier is handed the SIDE grid (100px squares), not the overhead one
    assert spy.received_grid is result.grids["observation.images.side"]
    assert isinstance(spy.received_grid, GroundedGrid)
    side_regions = spy.received_grid.regions
    assert side_regions is not None
    assert side_regions[Square("a", 1)].x_max == pytest.approx(100.0)
    # the primary grid stays the overhead one (80px squares)
    overhead_regions = result.grid.regions
    assert overhead_regions is not None
    assert overhead_regions[Square("a", 1)].x_max == pytest.approx(80.0)


def test_cross_check_occupancy_agreement_and_mismatch() -> None:
    start = BoardState.standard_starting_position()
    a = MetadataBoardPerception(start).perceive()
    b = MetadataBoardPerception(start).perceive()
    assert cross_check_occupancy(a, b) == []

    empty_e1 = MetadataBoardPerception(start.without_piece(Square("e", 1))).perceive()
    assert any("e1" in problem for problem in cross_check_occupancy(a, empty_e1))

    wrong_e1 = MetadataBoardPerception(
        start.with_piece(Square("e", 1), Piece(PieceType.QUEEN, PieceColor.WHITE))
    ).perceive()
    assert any(
        "e1" in problem and "mismatch" in problem
        for problem in cross_check_occupancy(a, wrong_e1)
    )
