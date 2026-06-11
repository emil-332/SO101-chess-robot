"""Convert a chesscog rendered-dataset split into our piece-dataset manifest.

Reads chesscog's per-image JSON (``corners``, ``fen``, ``white_turn``), maps the
corners to our a1/h1/h8/a8 convention (orientation handled by ``white_turn``; see
``chesscog_adapter``), and writes a manifest consumed by
``scripts/prepare_piece_dataset.py --source manifest``.

Get the data from OSF (DOI 10.17605/OSF.IO/XF3KA). The train zip is split; merge
with ``zip -s 0 train.zip --out train_full.zip`` before unzipping.

    # convert one split
    python scripts/chesscog_to_manifest.py --chesscog-dir chesscog/train \
        --out manifests/train.json
    # then build the crop dataset (images-root = the chesscog split dir)
    python scripts/prepare_piece_dataset.py --source manifest \
        --manifest manifests/train.json --images-root chesscog/train \
        --out datasets/piece_train.npz

Use ``--preview N`` first: it writes N images with the a1/h1/h8/a8 squares
outlined in distinct colours so you can confirm orientation (a1 should sit on the
white queenside-rook corner) before spending GPU time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from chess_robot.chess.board_state import BoardState, Square
from chess_robot.perception.camera_utils import (
    extract_square_crops,
    highlight_squares,
    square_label,
)
from chess_robot.perception.chesscog_adapter import manifest_entry
from chess_robot.perception.square_grounding import BoardCorners, grid_from_corners

# Distinct outline colours for the four labelled corners in --preview.
_PREVIEW_COLORS = {
    "a1": (0, 255, 0),  # green
    "h1": (0, 128, 255),  # blue
    "h8": (255, 128, 0),  # orange
    "a8": (255, 0, 0),  # red
}


def build_manifest(chesscog_dir: Path, limit: int | None = None) -> list[dict]:
    """Manifest entries for every ``*.json`` with a sibling ``*.png`` under the dir."""
    entries: list[dict] = []
    skipped = 0
    for json_path in sorted(chesscog_dir.rglob("*.json")):
        if limit is not None and len(entries) >= limit:
            break
        image_path = json_path.with_suffix(".png")
        if not image_path.exists():
            skipped += 1
            continue
        data = json.loads(json_path.read_text(encoding="utf-8"))
        relative = image_path.relative_to(chesscog_dir).as_posix()
        try:
            entries.append(
                manifest_entry(relative, data["corners"], bool(data["white_turn"]), data["fen"])
            )
        except (KeyError, ValueError) as error:
            skipped += 1
            print(f"  skip {json_path.name}: {error}")
    if skipped:
        print(f"  skipped {skipped} record(s)")
    return entries


def _corners_from_entry(entry: dict) -> BoardCorners:
    corner_map = entry["corners"]
    return BoardCorners(
        a1=tuple(corner_map["a1"]),
        h1=tuple(corner_map["h1"]),
        h8=tuple(corner_map["h8"]),
        a8=tuple(corner_map["a8"]),
    )


def _write_preview(chesscog_dir: Path, entries: list[dict], count: int, preview_dir: Path) -> None:
    from PIL import Image  # lazy: preview only

    preview_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for entry in entries[:count]:
        stem = Path(entry["image"]).stem
        image = np.asarray(Image.open(chesscog_dir / entry["image"]).convert("RGB"))
        grid = grid_from_corners(_corners_from_entry(entry))
        # 1) full board with the four corner squares outlined (geometry/orientation)
        annotated = image
        for name, color in _PREVIEW_COLORS.items():
            annotated = highlight_squares(annotated, grid, [Square.from_name(name)], color=color)
        Image.fromarray(annotated).save(preview_dir / f"{stem}_board.png")
        # 2) labelled crops of occupied squares: each must show the named piece
        board = BoardState.from_fen(entry["fen"])
        crops = extract_square_crops(image, grid, top_pad_ratio=1.0)
        for square in board.occupied_squares()[:8]:
            crop = crops.get(square)
            if crop is not None:
                label = square_label(board, square)
                Image.fromarray(crop).save(preview_dir / f"{stem}_{square.name}_{label}.png")
        written += 1
    print(f"  wrote previews for {written} image(s) to {preview_dir}")
    print("  *_board.png: corner outlines a1=green h1=blue h8=orange a8=red")
    print("  *_<square>_<piece>.png: each crop must show the named piece (label correctness)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert chesscog labels to our manifest.")
    parser.add_argument("--chesscog-dir", type=Path, required=True, help="a chesscog split dir")
    parser.add_argument("--out", type=Path, required=True, help="output manifest .json")
    parser.add_argument("--limit", type=int, default=None, help="cap the number of records")
    parser.add_argument("--preview", type=int, default=0, help="write N orientation-check images")
    parser.add_argument("--preview-dir", type=Path, default=Path("manifests/preview"))
    args = parser.parse_args(argv)

    entries = build_manifest(args.chesscog_dir, args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(entries), encoding="utf-8")
    print(f"wrote {args.out} ({len(entries)} entries)")
    if args.preview > 0:
        _write_preview(args.chesscog_dir, entries, args.preview, args.preview_dir)


if __name__ == "__main__":
    main()
