"""Create a tiny mock LeRobotDataset for the Option A pi0.5 smoke run.

Runs on the cloud GPU box (any machine with LeRobot installed). Builds a few
episodes of synthetic frames + random state/action with a real instruction, in
the LeRobot dataset format, so ``lerobot-train`` can consume it. The content is
**mock** — this validates the LeRobot/pi0.5 integration, not learning.

    python scripts/make_smoke_dataset.py --repo-id local/chess_smoke

If the installed LeRobot version differs (e.g. ``add_frame(frame, task=...)`` or
``consolidate()`` instead of ``finalize()``), adjust the marked lines — surfacing
those differences is exactly the point of the smoke run.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from chess_robot.data.lerobot_dataset import load_dataset_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a mock LeRobotDataset.")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/dataset/collect_chess_demos.yaml")
    )
    parser.add_argument("--repo-id", default="local/chess_smoke")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    config = load_dataset_config(args.config)

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:  # pragma: no cover - cloud-only
        raise SystemExit(
            "LeRobot not installed; run on the cloud box (see docs/cloud_smoke_test.md)."
        ) from exc

    features: dict[str, Any] = {}
    for camera in config.cameras:
        features[camera] = {
            # "image" (PNG per frame) avoids any ffmpeg dependency for a tiny
            # smoke dataset; switch to "video" for real recordings.
            "dtype": "image",
            "shape": (config.image_height, config.image_width, config.image_channels),
            "names": ["height", "width", "channel"],
        }
    features["observation.state"] = {
        "dtype": "float32",
        "shape": (config.state_dim,),
        "names": ["state"],
    }
    features["action"] = {
        "dtype": "float32",
        "shape": (config.action_dim,),
        "names": ["action"],
    }

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id, fps=args.fps, features=features, robot_type="so101"
    )

    rng = np.random.default_rng(0)
    instruction = "move knight from b1 to c3"
    for _ in range(args.episodes):
        for _frame in range(args.frames):
            frame: dict[str, Any] = {
                camera: rng.integers(
                    0,
                    255,
                    (config.image_height, config.image_width, config.image_channels),
                    dtype=np.uint8,
                )
                for camera in config.cameras
            }
            frame["observation.state"] = rng.standard_normal(config.state_dim).astype(
                np.float32
            )
            frame["action"] = rng.standard_normal(config.action_dim).astype(np.float32)
            # LeRobot 0.5.x: the task is a key in the frame dict (no `task=` kwarg).
            frame["task"] = instruction
            dataset.add_frame(frame)
        dataset.save_episode()
    dataset.finalize()  # older LeRobot: dataset.consolidate()

    print(f"created {args.episodes} episodes x {args.frames} frames")
    print(f"repo_id: {args.repo_id}")
    print(f"root:    {dataset.root}")


if __name__ == "__main__":
    main()
