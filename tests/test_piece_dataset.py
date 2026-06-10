"""Tests for the torch-free piece-classifier dataset / preprocessing layer."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from chess_robot.chess.board_mapper import GroundedGrid
from chess_robot.chess.board_state import BoardState, Square
from chess_robot.perception.board_renderer import render_board
from chess_robot.perception.piece_dataset import (
    OCCUPANCY_CLASSES,
    PIECE_CLASSES,
    PieceCropDataset,
    board_from_piece_predictions,
    build_dataset_from_boards,
    index_to_piece,
    label_for_piece,
    load_dataset,
    normalize_batch,
    occupancy_inference_crops,
    occupied_from_predictions,
    piece_class_index,
    piece_from_label,
    piece_inference_crops,
    random_board,
    resize_crop,
    save_dataset,
    train_val_split,
)
from chess_robot.perception.square_grounding import BoardCorners, grid_from_corners

IMAGE_SIZE = (240, 240)
OCC_SIZE = (100, 100)
PIECE_SIZE = (144, 96)


def _corners() -> BoardCorners:
    height, width = IMAGE_SIZE
    mx = 0.08 * width
    my = 0.08 * height
    return BoardCorners(
        a1=(mx, height - my),
        h1=(width - mx, height - my),
        h8=(width - mx, my),
        a8=(mx, my),
    )


def _render(board: BoardState) -> tuple[np.ndarray, GroundedGrid]:
    corners = _corners()
    rendered = render_board(board, corners, image_size=IMAGE_SIZE)
    return rendered.image, grid_from_corners(corners)


def test_piece_classes_are_twelve_distinct() -> None:
    assert len(PIECE_CLASSES) == 12
    assert len(set(PIECE_CLASSES)) == 12
    assert OCCUPANCY_CLASSES == ("empty", "occupied")


def test_piece_label_round_trips() -> None:
    for index in range(len(PIECE_CLASSES)):
        piece = index_to_piece(index)
        label = label_for_piece(piece)
        assert piece_class_index(label) == index
        assert piece_from_label(label) == piece


def test_resize_crop_shape_and_dtype() -> None:
    crop = (np.random.default_rng(0).random((50, 30, 3)) * 255).astype(np.uint8)
    out = resize_crop(crop, OCC_SIZE)
    assert out.shape == (100, 100, 3)
    assert out.dtype == np.uint8


def test_resize_crop_handles_grayscale() -> None:
    crop = np.zeros((10, 10), dtype=np.uint8)
    out = resize_crop(crop, (20, 20))
    assert out.shape == (20, 20, 3)


def test_normalize_batch_layout_and_values() -> None:
    crops = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    out = normalize_batch(crops)
    assert out.shape == (2, 3, 8, 8)
    assert out.dtype == np.float32
    # all-zero pixels => -mean/std per channel
    assert np.isclose(out[0, 0, 0, 0], -0.485 / 0.229, atol=1e-4)


def test_build_dataset_counts_match_board() -> None:
    start = BoardState.standard_starting_position()
    empty = BoardState.empty()
    boards = [(*_render(start), start), (*_render(empty), empty)]
    dataset = build_dataset_from_boards(
        boards,
        occupancy_size=OCC_SIZE,
        occupancy_top_pad=0.3,
        piece_size=PIECE_SIZE,
        piece_top_pad=1.0,
    )
    # 64 occupancy crops per board, both boards fully grounded.
    assert len(dataset.occupancy_labels) == 128
    assert int(dataset.occupancy_labels.sum()) == 32  # start has 32 pieces, empty has 0
    # one piece crop per occupied square (32 + 0).
    assert len(dataset.piece_labels) == 32
    assert dataset.piece_images.shape[1:] == (144, 96, 3)
    assert dataset.piece_labels.min() >= 0
    assert dataset.piece_labels.max() < len(PIECE_CLASSES)


def test_occupancy_and_piece_predictions_reconstruct_board() -> None:
    board = BoardState.from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR")
    squares = [Square.from_index(i) for i in range(64)]
    occupancy_pred = [1 if board.is_occupied(sq) else 0 for sq in squares]
    occupied = occupied_from_predictions(squares, occupancy_pred)
    assert occupied == board.occupied_squares()
    piece_pred: list[int] = []
    for square in occupied:
        piece = board.piece_at(square)
        assert piece is not None
        piece_pred.append(piece_class_index(label_for_piece(piece)))
    reconstructed = board_from_piece_predictions(occupied, piece_pred)
    assert reconstructed == board


def test_inference_crops_shapes() -> None:
    board = BoardState.standard_starting_position()
    image, grid = _render(board)
    squares, occ_crops = occupancy_inference_crops(image, grid, size=OCC_SIZE, top_pad_ratio=0.3)
    assert occ_crops.shape == (len(squares), 100, 100, 3)
    assert len(squares) == 64
    occupied = board.occupied_squares()
    piece_crops = piece_inference_crops(image, grid, occupied, size=PIECE_SIZE, top_pad_ratio=1.0)
    assert piece_crops.shape == (len(occupied), 144, 96, 3)


def test_save_load_round_trip(tmp_path: Path) -> None:
    start = BoardState.standard_starting_position()
    dataset = build_dataset_from_boards(
        [(*_render(start), start)],
        occupancy_size=OCC_SIZE,
        occupancy_top_pad=0.3,
        piece_size=PIECE_SIZE,
        piece_top_pad=1.0,
    )
    path = tmp_path / "piece.npz"
    save_dataset(path, dataset)
    loaded = load_dataset(path)
    assert isinstance(loaded, PieceCropDataset)
    assert np.array_equal(loaded.occupancy_images, dataset.occupancy_images)
    assert np.array_equal(loaded.piece_labels, dataset.piece_labels)
    assert loaded.piece_classes == PIECE_CLASSES
    assert loaded.occupancy_classes == OCCUPANCY_CLASSES


def test_train_val_split_partitions() -> None:
    images = np.zeros((20, 4, 4, 3), dtype=np.uint8)
    labels = np.arange(20, dtype=np.int64)
    train_x, train_y, val_x, val_y = train_val_split(images, labels, val_fraction=0.25, seed=1)
    assert len(train_y) == 15
    assert len(val_y) == 5
    assert set(train_y.tolist()) | set(val_y.tolist()) == set(range(20))


def test_random_board_fill_extremes() -> None:
    rng = np.random.default_rng(3)
    assert len(random_board(rng, fill_probability=1.0)) == 64
    assert len(random_board(rng, fill_probability=0.0)) == 0
