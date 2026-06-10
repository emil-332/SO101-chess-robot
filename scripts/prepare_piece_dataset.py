"""Build a two-stage piece-classifier crop dataset (``.npz``) from labelled boards.

Crops are auto-labelled from the known position (no per-piece hand annotation):
each (image, board-corners, FEN) yields 64 occupancy crops and one piece crop per
occupied square. Crop sizes/pads come from the piece-CNN config so the dataset
matches the model input.

Sources:

* ``renderer`` — random boards rendered with the synthetic renderer. Runs on the
  laptop with no extra data; validates the full prep -> train plumbing. The flat
  renderer does not teach real piece *appearance*, so use it for plumbing only.
* ``manifest`` — a JSON list of real/rendered photos with corners + FEN. This is
  the path for the chesscog synthetic set and, later, our own captured boards.
  Each entry: ``{"image": "rel/path.npy|png", "corners": {"a1": [x, y], "h1": ...,
  "h8": ..., "a8": ...}, "fen": "<placement>"}``.

    # laptop: tiny synthetic set to exercise the pipeline
    python scripts/prepare_piece_dataset.py --source renderer --num-boards 40 \
        --out datasets/piece_smoke.npz
    # real/rendered photos listed in a manifest
    python scripts/prepare_piece_dataset.py --source manifest \
        --manifest labels.json --images-root images/ --out datasets/piece.npz
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from chess_robot.chess.board_mapper import GroundedGrid
from chess_robot.chess.board_state import BoardState
from chess_robot.perception.board_renderer import render_board
from chess_robot.perception.camera_utils import corners_from_points
from chess_robot.perception.piece_cnn_config import load_two_stage_config
from chess_robot.perception.piece_dataset import (
    PieceCropDataset,
    build_dataset_from_boards,
    random_board,
    save_dataset,
)
from chess_robot.perception.square_grounding import BoardCorners, grid_from_corners

Board = tuple[np.ndarray, GroundedGrid, BoardState]


def _canonical_corners(image_size: tuple[int, int], margin_ratio: float) -> BoardCorners:
    height, width = image_size
    mx = margin_ratio * width
    my = margin_ratio * height
    return BoardCorners(
        a1=(mx, height - my),
        h1=(width - mx, height - my),
        h8=(width - mx, my),
        a8=(mx, my),
    )


def _jittered_corners(
    base: BoardCorners, rng: np.random.Generator, jitter: float
) -> BoardCorners:
    if jitter <= 0:
        return base

    def shift(point: tuple[float, float]) -> tuple[float, float]:
        return (
            point[0] + float(rng.uniform(-jitter, jitter)),
            point[1] + float(rng.uniform(-jitter, jitter)),
        )

    return BoardCorners(
        a1=shift(base.a1), h1=shift(base.h1), h8=shift(base.h8), a8=shift(base.a8)
    )


def boards_from_renderer(
    num_boards: int,
    image_size: tuple[int, int],
    *,
    seed: int,
    noise_std: float,
    jitter: float,
    fill_probability: float,
) -> Iterator[Board]:
    """Yield ``(image, grid, board)`` for random rendered positions."""
    rng = np.random.default_rng(seed)
    base = _canonical_corners(image_size, margin_ratio=0.08)
    for _ in range(num_boards):
        board = random_board(rng, fill_probability=fill_probability)
        corners = _jittered_corners(base, rng, jitter)
        rendered = render_board(board, corners, image_size=image_size, noise_std=noise_std, rng=rng)
        yield rendered.image, grid_from_corners(corners), board


def _load_image(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    from PIL import Image  # lazy: only needed for image-file manifests

    return np.asarray(Image.open(path).convert("RGB"))


def boards_from_manifest(manifest_path: Path, images_root: Path) -> Iterator[Board]:
    """Yield ``(image, grid, board)`` from a JSON manifest of labelled photos."""
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in entries:
        image = _load_image(images_root / entry["image"])
        corner_map = entry["corners"]
        corners = corners_from_points(
            [tuple(corner_map[name]) for name in ("a1", "h1", "h8", "a8")]
        )
        board = BoardState.from_fen(entry["fen"])
        yield image, grid_from_corners(corners), board


def build(args: argparse.Namespace) -> PieceCropDataset:
    config = load_two_stage_config(args.config)
    if args.source == "renderer":
        boards: Iterator[Board] = boards_from_renderer(
            args.num_boards,
            (args.image_size[0], args.image_size[1]),
            seed=args.seed,
            noise_std=args.noise_std,
            jitter=args.jitter,
            fill_probability=args.fill_probability,
        )
    else:
        boards = boards_from_manifest(Path(args.manifest), Path(args.images_root))
    return build_dataset_from_boards(
        boards,
        occupancy_size=config.occupancy.input_size,
        occupancy_top_pad=config.occupancy.top_pad_ratio,
        piece_size=config.piece.input_size,
        piece_top_pad=config.piece.top_pad_ratio,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare a piece-classifier crop dataset.")
    parser.add_argument("--config", type=Path, default=Path("configs/perception/piece_cnn.yaml"))
    parser.add_argument("--source", choices=("renderer", "manifest"), default="renderer")
    parser.add_argument("--out", type=Path, required=True, help="output .npz path")
    # renderer source
    parser.add_argument("--num-boards", type=int, default=200)
    parser.add_argument("--image-size", type=int, nargs=2, default=(240, 240), metavar=("H", "W"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise-std", type=float, default=6.0)
    parser.add_argument("--jitter", type=float, default=4.0, help="corner jitter in pixels")
    parser.add_argument("--fill-probability", type=float, default=0.4)
    # manifest source
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--images-root", type=Path, default=Path("."))
    args = parser.parse_args(argv)

    if args.source == "manifest" and args.manifest is None:
        parser.error("--manifest is required when --source manifest")

    dataset = build(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_dataset(args.out, dataset)
    print(f"wrote {args.out}")
    print("  " + dataset.summary())


if __name__ == "__main__":
    main()
