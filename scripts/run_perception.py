"""Run the perception pipeline on camera frames and print the board state.

Loads `configs/perception/perception.yaml`, builds the trained piece classifier +
grounding, reads the given frames, and prints the perceived position as FEN. This
is the perception entry point for the SO-101 pipeline: the resulting BoardState
feeds the deterministic move resolver.

Needs the `perception` extra (onnxruntime) plus the trained model weights under
models/ and the board corners calibrated in the config (see
docs/perception_piece_cnn.md). Until the physical board is calibrated, the
pipeline refuses to run rather than guess.

    python scripts/run_perception.py \
        --overhead frames/overhead.png --side frames/side.png

    # cross-check the reading against a known position (e.g. the start)
    python scripts/run_perception.py --side frames/side.png \
        --metadata-fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR
"""

from __future__ import annotations

import argparse
from pathlib import Path

from chess_robot.chess.board_state import BoardState
from chess_robot.perception.board_perception import (
    MetadataBoardPerception,
    cross_check_occupancy,
)
from chess_robot.perception.pipeline import (
    build_board_perception,
    load_frame,
    load_perception_config,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run perception on camera frames.")
    parser.add_argument("--config", type=Path, default=Path("configs/perception/perception.yaml"))
    parser.add_argument("--overhead", type=Path, default=None, help="overhead frame (.png/.npy)")
    parser.add_argument("--side", type=Path, default=None, help="side/oblique frame (.png/.npy)")
    parser.add_argument(
        "--metadata-fen", default=None, help="known position to cross-check the reading against"
    )
    args = parser.parse_args(argv)

    config = load_perception_config(args.config)
    frames = {}
    if args.side is not None:
        frames[config.piece_camera] = load_frame(args.side)
    if args.overhead is not None:
        frames[config.grounding_camera] = load_frame(args.overhead)
    if not frames:
        raise SystemExit("provide at least --side (the piece camera frame)")

    perception = build_board_perception(config)
    result = perception.perceive(frames)
    fen = result.board_state.to_fen()
    print(f"perceived ({result.source}): {fen}")
    print(f"  occupied squares: {len(result.board_state)}")

    if args.metadata_fen is not None:
        reference = MetadataBoardPerception(BoardState.from_fen(args.metadata_fen))
        problems = cross_check_occupancy(result, reference.perceive(frames))
        if problems:
            print(f"  cross-check vs metadata: {len(problems)} disagreement(s)")
            for problem in problems:
                print(f"    {problem}")
        else:
            print("  cross-check vs metadata: full agreement")


if __name__ == "__main__":
    main()
