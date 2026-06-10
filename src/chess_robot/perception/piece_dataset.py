"""Crop extraction, labels, and numpy preprocessing for the piece classifier.

Torch-free on purpose so it runs on the laptop (dataset prep, dry-runs, tests)
and is shared by both the torch and the ONNX inference paths. Responsibilities:

- Stable class orderings for the two stages (occupancy: empty/occupied; piece:
  the 12 colour+type identities) and label <-> :class:`Piece` conversions.
- Turn a known board (image + grounded grid + :class:`BoardState`) into labelled
  square crops, auto-labelled from the position (no per-piece hand annotation).
- Fixed-size resizing (numpy bilinear) and ImageNet normalization, applied
  identically at train and inference time.
- Reconstruct a :class:`BoardState` from the two stages' class predictions.
- Save/load a prepared crop dataset as a single ``.npz`` artifact.

Geometry comes from the grounded grid (see ``square_grounding``); occupancy uses
a near-square crop, piece identity uses a taller crop (``top_pad_ratio``) that
captures the piece rising above its square. See ``docs/perception_piece_cnn.md``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from chess_robot.chess.board_mapper import GroundedGrid
from chess_robot.chess.board_state import (
    BoardState,
    Piece,
    PieceColor,
    PieceType,
    Square,
)
from chess_robot.perception.camera_utils import extract_square_crops

# --- Class orderings (stable; do not reorder silently) ------------------------

OCCUPANCY_CLASSES: tuple[str, ...] = ("empty", "occupied")

_PIECE_TYPE_ORDER: tuple[PieceType, ...] = (
    PieceType.PAWN,
    PieceType.KNIGHT,
    PieceType.BISHOP,
    PieceType.ROOK,
    PieceType.QUEEN,
    PieceType.KING,
)
_COLOR_ORDER: tuple[PieceColor, ...] = (PieceColor.WHITE, PieceColor.BLACK)

# 12 labels "{colour}_{type}" — matches camera_utils.square_label's non-empty form.
PIECE_CLASSES: tuple[str, ...] = tuple(
    f"{color.value}_{piece_type.value}"
    for color in _COLOR_ORDER
    for piece_type in _PIECE_TYPE_ORDER
)

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


def label_for_piece(piece: Piece) -> str:
    """The 12-class piece label for a piece, e.g. ``"white_knight"``."""
    return f"{piece.color.value}_{piece.piece_type.value}"


def piece_class_index(label: str) -> int:
    """Index of a piece label in :data:`PIECE_CLASSES`."""
    return PIECE_CLASSES.index(label)


def piece_from_label(label: str) -> Piece:
    """Inverse of :func:`label_for_piece` (``"black_rook"`` -> :class:`Piece`)."""
    color_str, type_str = label.split("_", 1)
    return Piece(PieceType(type_str), PieceColor(color_str))


def index_to_piece(index: int) -> Piece:
    """:class:`Piece` for a piece-stage class index."""
    return piece_from_label(PIECE_CLASSES[index])


# --- Resizing / normalization (numpy, no torch / PIL) -------------------------


def _ensure_rgb(crop: np.ndarray) -> np.ndarray:
    if crop.ndim == 2:
        return np.repeat(crop[:, :, None], 3, axis=2)
    if crop.shape[2] == 1:
        return np.repeat(crop, 3, axis=2)
    if crop.shape[2] >= 3:
        return crop[:, :, :3]
    raise ValueError(f"unexpected crop shape {crop.shape}")


def resize_crop(crop: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Bilinearly resize an ``HxWx3`` crop to ``size`` = ``(height, width)`` uint8."""
    rgb = _ensure_rgb(crop)
    out_h, out_w = size
    in_h, in_w = rgb.shape[0], rgb.shape[1]
    if in_h == out_h and in_w == out_w:
        return rgb.astype(np.uint8)
    src = rgb.astype(np.float32)
    ys = np.clip((np.arange(out_h) + 0.5) * in_h / out_h - 0.5, 0, in_h - 1)
    xs = np.clip((np.arange(out_w) + 0.5) * in_w / out_w - 0.5, 0, in_w - 1)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, in_h - 1)
    x1 = np.minimum(x0 + 1, in_w - 1)
    wy = (ys - y0)[:, None, None]
    wx = (xs - x0)[None, :, None]
    top = src[y0][:, x0] * (1 - wx) + src[y0][:, x1] * wx
    bottom = src[y1][:, x0] * (1 - wx) + src[y1][:, x1] * wx
    out = top * (1 - wy) + bottom * wy
    return np.clip(out, 0, 255).astype(np.uint8)


def normalize_batch(
    crops: np.ndarray,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> np.ndarray:
    """``N x H x W x 3`` uint8 -> ``N x 3 x H x W`` float32, ImageNet-normalized.

    The single normalization used by both the torch and ONNX inference paths so
    they stay byte-for-byte consistent.
    """
    array = crops.astype(np.float32) / 255.0
    mean_arr = np.asarray(mean, dtype=np.float32)
    std_arr = np.asarray(std, dtype=np.float32)
    array = (array - mean_arr) / std_arr
    return np.transpose(array, (0, 3, 1, 2)).copy()


# --- Labelled crops from a known board ----------------------------------------


def board_to_occupancy_samples(
    image: np.ndarray,
    grid: GroundedGrid,
    board: BoardState,
    *,
    size: tuple[int, int],
    top_pad_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-square crops + occupancy labels (0 empty, 1 occupied) for every square."""
    crops = extract_square_crops(image, grid, top_pad_ratio=top_pad_ratio)
    images: list[np.ndarray] = []
    labels: list[int] = []
    for square, crop in crops.items():
        images.append(resize_crop(crop, size))
        labels.append(1 if board.is_occupied(square) else 0)
    return _stack(images, size), np.asarray(labels, dtype=np.int64)


def board_to_piece_samples(
    image: np.ndarray,
    grid: GroundedGrid,
    board: BoardState,
    *,
    size: tuple[int, int],
    top_pad_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Taller crops + 12-class piece labels for the occupied squares only."""
    crops = extract_square_crops(image, grid, top_pad_ratio=top_pad_ratio)
    images: list[np.ndarray] = []
    labels: list[int] = []
    for square, crop in crops.items():
        piece = board.piece_at(square)
        if piece is None:
            continue
        images.append(resize_crop(crop, size))
        labels.append(piece_class_index(label_for_piece(piece)))
    return _stack(images, size), np.asarray(labels, dtype=np.int64)


def _stack(images: Sequence[np.ndarray], size: tuple[int, int]) -> np.ndarray:
    if not images:
        return np.empty((0, size[0], size[1], 3), dtype=np.uint8)
    return np.stack(images, axis=0).astype(np.uint8)


# --- Inference crops (no labels) ----------------------------------------------


def occupancy_inference_crops(
    image: np.ndarray,
    grid: GroundedGrid,
    *,
    size: tuple[int, int],
    top_pad_ratio: float,
) -> tuple[list[Square], np.ndarray]:
    """Square crops for occupancy inference; returns the square order and the batch.

    Squares whose crop is degenerate (clipped to zero area) are omitted and are
    therefore treated as empty downstream.
    """
    crops = extract_square_crops(image, grid, top_pad_ratio=top_pad_ratio)
    squares = list(crops.keys())
    batch = _stack([resize_crop(crops[square], size) for square in squares], size)
    return squares, batch


def piece_inference_crops(
    image: np.ndarray,
    grid: GroundedGrid,
    squares: Sequence[Square],
    *,
    size: tuple[int, int],
    top_pad_ratio: float,
) -> np.ndarray:
    """Taller crops for the given (occupied) squares, in order."""
    crops = extract_square_crops(image, grid, top_pad_ratio=top_pad_ratio)
    images: list[np.ndarray] = []
    for square in squares:
        crop = crops.get(square)
        if crop is None:
            images.append(np.zeros((size[0], size[1], 3), dtype=np.uint8))
        else:
            images.append(resize_crop(crop, size))
    return _stack(images, size)


# --- Predictions -> board -----------------------------------------------------


def occupied_from_predictions(
    squares: Sequence[Square], occupancy_pred: Sequence[int]
) -> list[Square]:
    """Squares the occupancy stage marked ``"occupied"``."""
    return [
        square
        for square, pred in zip(squares, occupancy_pred, strict=True)
        if OCCUPANCY_CLASSES[int(pred)] == "occupied"
    ]


def board_from_piece_predictions(
    squares: Sequence[Square], piece_pred: Sequence[int]
) -> BoardState:
    """Assemble a :class:`BoardState` from per-square piece-class predictions."""
    pieces = {
        square: index_to_piece(int(pred))
        for square, pred in zip(squares, piece_pred, strict=True)
    }
    return BoardState.from_map(pieces)


# --- Dataset artifact ---------------------------------------------------------


@dataclass
class PieceCropDataset:
    """A prepared two-stage crop dataset (in memory or loaded from ``.npz``)."""

    occupancy_images: np.ndarray  # N x H x W x 3 uint8
    occupancy_labels: np.ndarray  # N int64 (0/1)
    piece_images: np.ndarray  # M x H x W x 3 uint8
    piece_labels: np.ndarray  # M int64 (0..11)
    occupancy_classes: tuple[str, ...] = OCCUPANCY_CLASSES
    piece_classes: tuple[str, ...] = PIECE_CLASSES

    def summary(self) -> str:
        return (
            f"occupancy: {len(self.occupancy_labels)} crops "
            f"(occupied {int(self.occupancy_labels.sum())}), "
            f"piece: {len(self.piece_labels)} crops, "
            f"input occ {self.occupancy_images.shape[1:3]} "
            f"piece {self.piece_images.shape[1:3]}"
        )


def build_dataset_from_boards(
    boards: Iterable[tuple[np.ndarray, GroundedGrid, BoardState]],
    *,
    occupancy_size: tuple[int, int],
    occupancy_top_pad: float,
    piece_size: tuple[int, int],
    piece_top_pad: float,
) -> PieceCropDataset:
    """Aggregate labelled crops over many ``(image, grid, board)`` triples."""
    occ_images: list[np.ndarray] = []
    occ_labels: list[np.ndarray] = []
    piece_images: list[np.ndarray] = []
    piece_labels: list[np.ndarray] = []
    for image, grid, board in boards:
        oi, ol = board_to_occupancy_samples(
            image, grid, board, size=occupancy_size, top_pad_ratio=occupancy_top_pad
        )
        pi, pl = board_to_piece_samples(
            image, grid, board, size=piece_size, top_pad_ratio=piece_top_pad
        )
        occ_images.append(oi)
        occ_labels.append(ol)
        piece_images.append(pi)
        piece_labels.append(pl)
    return PieceCropDataset(
        occupancy_images=_concat(occ_images, occupancy_size),
        occupancy_labels=_concat_labels(occ_labels),
        piece_images=_concat(piece_images, piece_size),
        piece_labels=_concat_labels(piece_labels),
    )


def _concat(arrays: Sequence[np.ndarray], size: tuple[int, int]) -> np.ndarray:
    if not arrays:
        return np.empty((0, size[0], size[1], 3), dtype=np.uint8)
    return np.concatenate(arrays, axis=0)


def _concat_labels(arrays: Sequence[np.ndarray]) -> np.ndarray:
    if not arrays:
        return np.empty((0,), dtype=np.int64)
    return np.concatenate(arrays, axis=0)


def save_dataset(path: str | Path, dataset: PieceCropDataset) -> None:
    """Persist a :class:`PieceCropDataset` to a single compressed ``.npz``."""
    np.savez_compressed(
        path,
        occupancy_images=dataset.occupancy_images,
        occupancy_labels=dataset.occupancy_labels,
        piece_images=dataset.piece_images,
        piece_labels=dataset.piece_labels,
        occupancy_classes=np.asarray(dataset.occupancy_classes),
        piece_classes=np.asarray(dataset.piece_classes),
    )


def load_dataset(path: str | Path) -> PieceCropDataset:
    """Load a :class:`PieceCropDataset` saved by :func:`save_dataset`."""
    with np.load(path, allow_pickle=False) as data:
        return PieceCropDataset(
            occupancy_images=data["occupancy_images"],
            occupancy_labels=data["occupancy_labels"],
            piece_images=data["piece_images"],
            piece_labels=data["piece_labels"],
            occupancy_classes=tuple(str(c) for c in data["occupancy_classes"]),
            piece_classes=tuple(str(c) for c in data["piece_classes"]),
        )


def train_val_split(
    images: np.ndarray, labels: np.ndarray, *, val_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle and split arrays into ``(train_x, train_y, val_x, val_y)``."""
    count = len(labels)
    rng = np.random.default_rng(seed)
    order = rng.permutation(count)
    n_val = int(round(count * val_fraction))
    val_idx = order[:n_val]
    train_idx = order[n_val:]
    return images[train_idx], labels[train_idx], images[val_idx], labels[val_idx]


# --- Random positions (for the renderer-based synthetic source / tests) -------


def random_board(rng: np.random.Generator, *, fill_probability: float = 0.4) -> BoardState:
    """A random board: each square independently filled with a random piece."""
    pieces: dict[Square, Piece] = {}
    for index in range(64):
        if rng.random() < fill_probability:
            piece_type = _PIECE_TYPE_ORDER[int(rng.integers(len(_PIECE_TYPE_ORDER)))]
            color = _COLOR_ORDER[int(rng.integers(len(_COLOR_ORDER)))]
            pieces[Square.from_index(index)] = Piece(piece_type, color)
    return BoardState.from_map(pieces)
