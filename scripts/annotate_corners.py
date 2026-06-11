"""Annotate board corners for the perception corner-detection dataset.

Interactive tool: show an image, click the four board
corners in order **a1, h1, h8, a8**, and append a record to a JSONL manifest.
These records train / calibrate the square-grounding corner detector.

Requires the optional GUI deps: ``pip install -e ".[tools]"``. Run on the laptop
(needs a display):

    python scripts/annotate_corners.py --image board.jpg --camera overhead \
        --board-type woodA --manifest perception_data/corners/manifest.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chess_robot.perception.camera_utils import corners_from_points
from chess_robot.perception.square_grounding import BoardCorners, Point


def corner_record(
    image: str, camera: str, board_type: str, corners: BoardCorners
) -> dict[str, object]:
    """Build a JSONL manifest record for one annotated image."""
    return {
        "image": image,
        "camera": camera,
        "board_type": board_type,
        "corners": {
            "a1": list(corners.a1),
            "h1": list(corners.h1),
            "h8": list(corners.h8),
            "a8": list(corners.a8),
        },
    }


def append_jsonl(manifest: Path, record: dict[str, object]) -> None:
    """Append one record as a JSON line, creating the manifest dir if needed."""
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _click_corners(image_path: str) -> list[Point]:
    try:
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional GUI extra
        raise SystemExit(
            "matplotlib is required for annotation: pip install -e '.[tools]'"
        ) from exc

    plt.imshow(mpimg.imread(image_path))
    plt.title("Click corners in order: a1, h1, h8, a8")
    points = plt.ginput(4, timeout=0)
    plt.close()
    return [(float(x), float(y)) for x, y in points]


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate board corners.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--camera", required=True, choices=["overhead", "side"])
    parser.add_argument("--board-type", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()

    corners = corners_from_points(_click_corners(args.image))
    append_jsonl(
        args.manifest, corner_record(args.image, args.camera, args.board_type, corners)
    )
    print(f"Appended corners for {args.image} to {args.manifest}")


if __name__ == "__main__":
    main()
