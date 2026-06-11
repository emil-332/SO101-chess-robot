"""Collect teleoperated demonstrations into a LeRobotDataset.

the offline ``--dry-run`` path validates the dataset structure and
the perception preprocessing without any hardware or LeRobot.

    python scripts/collect_demos.py --dry-run \
        --config configs/dataset/collect_chess_demos.yaml \
        --instruction "move knight from b1 to c3"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from chess_robot.chess.board_state import BoardState
from chess_robot.chess.move_resolver import MoveResolver, OffBoardLocation
from chess_robot.data.lerobot_dataset import (
    ChessDemoRecorder,
    DatasetConfig,
    load_dataset_config,
)
from chess_robot.perception.board_perception import MetadataBoardPerception


def _blank_frames(config: DatasetConfig) -> dict[str, np.ndarray]:
    frame = np.zeros(
        (config.image_height, config.image_width, config.image_channels), dtype=np.uint8
    )
    return {camera: frame.copy() for camera in config.cameras}


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect chess demos.")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/dataset/collect_chess_demos.yaml")
    )
    parser.add_argument("--instruction", default="move knight from b1 to c3")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate structure + preprocessing offline (no hardware/LeRobot).",
    )
    args = parser.parse_args()

    config = load_dataset_config(args.config)
    if not args.dry_run:
        raise SystemExit(
            "Real recording is wired at   2.2; run with --dry-run for now."
        )

    # Offline dry run: metadata occupancy on the standard start position.
    perception = MetadataBoardPerception(BoardState.standard_starting_position())
    resolver = MoveResolver(OffBoardLocation(config.off_board_location or "capture_tray"))
    recorder = ChessDemoRecorder(config, perception, resolver)

    report = recorder.dry_run(_blank_frames(config), args.instruction)
    print("dry-run:", "OK" if report.ok else "FAILED")
    print(f"  repo_id:        {config.repo_id}")
    print(f"  episodes:       {report.num_episodes}")
    print(f"  tensor columns: {sorted(report.features)}")
    print(f"  metadata cols:  {list(report.metadata_fields)}")
    if report.sample_record_problems:
        for problem in report.sample_record_problems:
            print(f"  problem: {problem}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
