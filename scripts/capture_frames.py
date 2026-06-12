"""Capture frames from the lab cameras to disk (laptop, `camera` extra).

The foundation for working with the real board: grab overhead/side frames for
corner calibration, few-shot piece photos, and the perception eval set. Discover
camera indices with ``lerobot-find-cameras opencv``, then:

    pip install -e ".[camera]"
    python scripts/capture_frames.py --camera overhead=0 --camera side=2 \
        --out-dir frames

Saved as RGB PNGs (``<name>.png``, or ``<name>_NN.png`` with ``--count`` > 1),
which load_frame / the pipeline read directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def save_frame(image: np.ndarray, path: Path) -> None:
    """Write an RGB ``HxWx3`` uint8 frame to ``path`` as PNG."""
    from PIL import Image  # lazy: camera extra

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def grab_frame(index: int, *, warmup: int = 5) -> np.ndarray:
    """Capture one RGB frame from camera ``index`` (discarding ``warmup`` frames)."""
    import cv2  # lazy: camera extra

    capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        raise SystemExit(f"could not open camera index {index} (check the index)")
    try:
        ok, frame = False, None
        for _ in range(max(1, warmup)):  # let auto-exposure settle
            ok, frame = capture.read()
        if not ok or frame is None:
            raise SystemExit(f"could not read from camera index {index}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()


def _parse_camera(spec: str) -> tuple[str, int]:
    name, _, index = spec.partition("=")
    if not name or not index:
        raise SystemExit(f"--camera expects NAME=INDEX, got {spec!r}")
    return name, int(index)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Capture frames from lab cameras.")
    parser.add_argument(
        "--camera", action="append", required=True, metavar="NAME=INDEX",
        help="camera name and index, e.g. overhead=0 (repeatable)",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("frames"))
    parser.add_argument("--count", type=int, default=1, help="frames per camera")
    parser.add_argument("--warmup", type=int, default=5, help="frames to discard first")
    args = parser.parse_args(argv)

    for spec in args.camera:
        name, index = _parse_camera(spec)
        for i in range(args.count):
            image = grab_frame(index, warmup=args.warmup)
            suffix = "" if args.count == 1 else f"_{i:02d}"
            out = args.out_dir / f"{name}{suffix}.png"
            save_frame(image, out)
            print(f"wrote {out}  ({image.shape[1]}x{image.shape[0]})")


if __name__ == "__main__":
    main()
