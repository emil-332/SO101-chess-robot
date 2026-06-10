"""Tests for the synthetic board renderer + colour classifier."""

import numpy as np

from chess_robot.chess.board_state import BoardState, PieceColor, PieceType, Square
from chess_robot.eval.perception_metrics import PerceptionSample, evaluate_perception
from chess_robot.perception.board_perception import (
    DEFAULT_GROUNDING_CAMERA,
    ComposedBoardPerception,
)
from chess_robot.perception.board_renderer import (
    PIECE_RGB,
    ColorPieceClassifier,
    render_board,
)
from chess_robot.perception.square_grounding import (
    BoardCorners,
    CornerSquareGrounder,
    FixedCornerDetector,
    grid_from_corners,
)

_IMAGE_SIZE = (256, 256)
_OVERHEAD = BoardCorners((8.0, 8.0), (248.0, 8.0), (248.0, 248.0), (8.0, 248.0))


def _perception(corners: BoardCorners) -> ComposedBoardPerception:
    return ComposedBoardPerception(
        CornerSquareGrounder(FixedCornerDetector(corners)), ColorPieceClassifier()
    )


def _sample(
    board: BoardState,
    corners: BoardCorners,
    *,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> PerceptionSample:
    rendered = render_board(
        board, corners, image_size=_IMAGE_SIZE, noise_std=noise_std, rng=rng
    )
    return PerceptionSample(
        ground_truth_board=board,
        frames={DEFAULT_GROUNDING_CAMERA: rendered.image},
        ground_truth_grid=grid_from_corners(corners),
    )


def test_render_shape_and_piece_colour_present() -> None:
    board = BoardState.standard_starting_position()
    rendered = render_board(board, _OVERHEAD, image_size=_IMAGE_SIZE)
    assert rendered.image.shape == (256, 256, 3)
    assert rendered.image.dtype == np.uint8
    # the white king's colour is drawn somewhere
    king_rgb = np.array(PIECE_RGB[(PieceType.KING, PieceColor.WHITE)])
    assert bool(np.any(np.all(rendered.image == king_rgb, axis=-1)))


def test_clean_overhead_classifies_perfectly() -> None:
    board = BoardState.standard_starting_position()
    frames = {
        DEFAULT_GROUNDING_CAMERA: render_board(
            board, _OVERHEAD, image_size=_IMAGE_SIZE
        ).image
    }
    result = _perception(_OVERHEAD).perceive(frames)
    assert result.board_state == board  # exact read of the rendered board


def test_clean_overhead_metrics_are_perfect() -> None:
    board = BoardState.standard_starting_position()
    report = evaluate_perception(_perception(_OVERHEAD), [_sample(board, _OVERHEAD)])
    assert report.occupancy_accuracy == 1.0
    assert report.piece_classification_accuracy == 1.0
    assert report.square_grounding_accuracy == 1.0


def test_heavy_noise_degrades_occupancy() -> None:
    board = BoardState.standard_starting_position()
    rng = np.random.default_rng(0)
    sample = _sample(board, _OVERHEAD, noise_std=200.0, rng=rng)
    report = evaluate_perception(_perception(_OVERHEAD), [sample])
    assert report.occupancy_accuracy is not None
    assert report.occupancy_accuracy < 1.0  # noise corrupts the colour reading


def test_empty_square_reads_as_empty() -> None:
    board = BoardState.from_fen("8/8/8/8/4P3/8/8/8")  # lone pawn on e4
    result = _perception(_OVERHEAD).perceive(
        {
            DEFAULT_GROUNDING_CAMERA: render_board(
                board, _OVERHEAD, image_size=_IMAGE_SIZE
            ).image
        }
    )
    assert result.board_state.is_occupied(Square("e", 4))
    assert not result.board_state.is_occupied(Square("d", 4))
    assert len(result.board_state) == 1
