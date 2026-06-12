"""Build a piece-dataset manifest from captured photos + board calibration.

Turns a folder of board photos into the ``{image, corners, fen}`` manifest that
``prepare_piece_dataset.py --source manifest`` consumes. Corners come from the
calibration file (one camera), so you do not re-click per photo.

Two modes:

* Few-shot fine-tune: many photos of the **same** position (e.g. the start), one
  shared FEN. Use ``--fen <PLACEMENT>``.
* Eval set: varied positions, one FEN per photo. Use ``--fen-map labels.json``
  (a ``{"side_03.png": "<placement>", ...}`` map).

    python scripts/build_piece_manifest.py --images-dir photos/start \
        --camera side --calibration configs/perception/calibration.local.yaml \
        --fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR --out manifests/ours.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def load_calibration_corners(path: Path, camera: str) -> dict[str, Any]:
    """Read the ``{a1,h1,h8,a8}`` corner map for ``camera`` from a calibration file."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    block = data.get("calibration", {}) if isinstance(data, dict) else {}
    corners = block.get(camera) if isinstance(block, dict) else None
    if not corners:
        raise SystemExit(f"no '{camera}' calibration in {path}; run calibrate_corners.py")
    return corners


def build_entries(
    image_names: list[str],
    corners: dict[str, Any],
    *,
    fen: str | None,
    fen_map: dict[str, str] | None,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for name in image_names:
        if fen_map is not None:
            if name not in fen_map:
                raise SystemExit(f"no FEN for {name!r} in --fen-map")
            position = fen_map[name]
        else:
            assert fen is not None
            position = fen
        entries.append({"image": name, "corners": corners, "fen": position})
    return entries


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a piece-dataset manifest.")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="*.png", help="glob for images (default *.png)")
    parser.add_argument("--camera", default="side", choices=("side", "overhead"))
    parser.add_argument(
        "--calibration", type=Path, default=Path("configs/perception/calibration.local.yaml")
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fen", default=None, help="one FEN placement for all images")
    group.add_argument("--fen-map", type=Path, default=None, help="JSON {image: fen} map")
    parser.add_argument("--out", type=Path, required=True, help="output manifest .json")
    args = parser.parse_args(argv)

    corners = load_calibration_corners(args.calibration, args.camera)
    image_names = sorted(p.name for p in args.images_dir.glob(args.pattern))
    if not image_names:
        raise SystemExit(f"no images matching {args.pattern!r} in {args.images_dir}")
    fen_map = json.loads(args.fen_map.read_text(encoding="utf-8")) if args.fen_map else None

    entries = build_entries(image_names, corners, fen=args.fen, fen_map=fen_map)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(entries), encoding="utf-8")
    print(f"wrote {args.out} ({len(entries)} entries, images-root={args.images_dir})")


if __name__ == "__main__":
    main()
