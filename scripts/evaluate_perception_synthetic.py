"""Evaluate the perception loop on synthetic rendered boards (Tier B).

Renders boards under several conditions (overhead vs oblique, clean vs noisy),
runs the grounding + crop + colour-classifier pipeline through
``evaluate_perception``, and prints a metrics table. Produces *real* perception
numbers with no GPU/hardware, and shows how accuracy degrades under oblique views
(AABB crops overlap) and noise.

Caveat: piece colours are a controlled stand-in, not photorealistic pieces, and
ground-truth corners are used (so grounding is trivially 1.00). This validates the
geometry/crop/eval pipeline and robustness — not the learned CNN's appearance
generalization.

    python scripts/evaluate_perception_synthetic.py
"""

from __future__ import annotations

import argparse

import numpy as np

from chess_robot.chess.board_state import BoardState, Square
from chess_robot.eval.perception_metrics import (
    PerceptionReport,
    PerceptionSample,
    evaluate_perception,
)
from chess_robot.perception.board_perception import (
    DEFAULT_GROUNDING_CAMERA,
    ComposedBoardPerception,
)
from chess_robot.perception.board_renderer import (
    DEFAULT_PALETTE,
    ColorPieceClassifier,
    Palette,
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
_OBLIQUE = BoardCorners((20.0, 240.0), (236.0, 240.0), (190.0, 40.0), (66.0, 40.0))
# Steep oblique: far (top) edge strongly compressed -> tiny far squares, heavy
# AABB-crop overlap (the failure regime for axis-aligned crops).
_OBLIQUE_STEEP = BoardCorners(
    (20.0, 240.0), (236.0, 240.0), (146.0, 26.0), (110.0, 26.0)
)
_CAPTURE_TARGETS = (Square("a", 8), Square("e", 4))


def _boards() -> list[BoardState]:
    return [
        BoardState.standard_starting_position(),
        BoardState.from_fen("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R"),
        BoardState.from_fen("8/8/3k4/8/4P3/8/8/4K3"),
    ]


def _evaluate(
    corners: BoardCorners,
    *,
    palette: Palette,
    noise_std: float,
    rng: np.random.Generator,
) -> PerceptionReport:
    perception = ComposedBoardPerception(
        CornerSquareGrounder(FixedCornerDetector(corners)), ColorPieceClassifier()
    )
    grid = grid_from_corners(corners)
    samples = [
        PerceptionSample(
            ground_truth_board=board,
            frames={
                DEFAULT_GROUNDING_CAMERA: render_board(
                    board,
                    corners,
                    image_size=_IMAGE_SIZE,
                    palette=palette,
                    noise_std=noise_std,
                    rng=rng,
                ).image
            },
            ground_truth_grid=grid,
            capture_targets=_CAPTURE_TARGETS,
        )
        for board in _boards()
    ]
    return evaluate_perception(perception, samples)


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic perception evaluation.")
    parser.add_argument("--noise", type=float, default=40.0)
    parser.add_argument("--heavy-noise", type=float, default=150.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    conditions = (
        ("overhead-clean", _OVERHEAD, 0.0),
        (f"overhead-noisy(std={args.noise:g})", _OVERHEAD, args.noise),
        (f"overhead-heavy(std={args.heavy_noise:g})", _OVERHEAD, args.heavy_noise),
        ("oblique-clean", _OBLIQUE, 0.0),
        ("oblique-steep-clean", _OBLIQUE_STEEP, 0.0),
        (f"oblique-steep-noisy(std={args.noise:g})", _OBLIQUE_STEEP, args.noise),
    )

    header = f"{'condition':26} {'occ':>5} {'piece':>6} {'capt':>5} {'grnd':>5}  n"
    print(header)
    print("-" * len(header))
    for name, corners, noise in conditions:
        report = _evaluate(corners, palette=DEFAULT_PALETTE, noise_std=noise, rng=rng)
        print(
            f"{name:26} "
            f"{_fmt(report.occupancy_accuracy):>5} "
            f"{_fmt(report.piece_classification_accuracy):>6} "
            f"{_fmt(report.capture_detection_accuracy):>5} "
            f"{_fmt(report.square_grounding_accuracy):>5}  "
            f"{report.num_samples}"
        )
    print(
        "\nNote: piece colours are a controlled stand-in (not photorealistic); "
        "grounding=1.00\nbecause ground-truth corners are used (no detector tested)."
    )


if __name__ == "__main__":
    main()
