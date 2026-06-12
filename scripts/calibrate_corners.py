"""Calibrate board corners into the perception pipeline's calibration file.

Click the four board corners (a1, h1, h8, a8) on a captured frame per camera and
write them to the lab-specific calibration file that
`configs/perception/perception.yaml` references (gitignored, per board). After
this, `scripts/run_perception.py` can ground and read the real board.

    pip install -e ".[tools]"     # matplotlib for the click UI
    python scripts/calibrate_corners.py \
        --frame side=frames/side.png --frame overhead=frames/overhead.png

The piece (side) camera is required for the classifier; overhead is optional
(square highlighting only).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from chess_robot.perception.camera_utils import click_corners, corners_from_points
from chess_robot.perception.pipeline import write_calibration_file
from chess_robot.perception.square_grounding import BoardCorners


def _parse_frame(spec: str) -> tuple[str, str]:
    name, _, image = spec.partition("=")
    if not image:
        raise SystemExit(f"--frame expects NAME=IMAGE, got {spec!r}")
    if name not in ("overhead", "side"):
        raise SystemExit(f"camera must be 'overhead' or 'side', got {name!r}")
    return name, image


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Calibrate board corners.")
    parser.add_argument(
        "--frame", action="append", required=True, metavar="NAME=IMAGE",
        help="camera name and image, e.g. side=frames/side.png (repeatable)",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("configs/perception/calibration.local.yaml")
    )
    args = parser.parse_args(argv)

    corners: dict[str, BoardCorners] = {}
    for spec in args.frame:
        name, image = _parse_frame(spec)
        print(f"click 4 corners (a1, h1, h8, a8) on the {name} image...")
        corners[name] = corners_from_points(click_corners(image))

    write_calibration_file(args.out, overhead=corners.get("overhead"), side=corners.get("side"))
    print(f"wrote {args.out}")
    print(f"  calibrated cameras: {sorted(corners)}")


if __name__ == "__main__":
    main()
