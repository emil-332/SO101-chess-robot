"""Tests for the chesscog -> manifest adapter (corner ordering + orientation)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np

from chess_robot.perception.chesscog_adapter import (
    labeled_corners,
    manifest_entry,
    sort_corner_points,
)

# Axis-aligned board (image coords, y down): TL=(0,0) TR=(10,0) BR=(10,10) BL=(0,10).
_TL = (0.0, 0.0)
_TR = (10.0, 0.0)
_BR = (10.0, 10.0)
_BL = (0.0, 10.0)
_SHUFFLED = [_BR, _TL, _BL, _TR]


def test_sort_corner_points_orders_tl_tr_br_bl() -> None:
    ordered = sort_corner_points(_SHUFFLED)
    assert np.allclose(ordered, [_TL, _TR, _BR, _BL])


def test_sort_corner_points_rejects_wrong_shape() -> None:
    try:
        sort_corner_points([[0, 0], [1, 1]])
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-(4,2) input")


def test_labeled_corners_white() -> None:
    corners = labeled_corners(_SHUFFLED, white_turn=True)
    assert corners.a1 == _BL
    assert corners.h1 == _BR
    assert corners.h8 == _TR
    assert corners.a8 == _TL


def test_labeled_corners_black() -> None:
    corners = labeled_corners(_SHUFFLED, white_turn=False)
    assert corners.a1 == _TR
    assert corners.h1 == _TL
    assert corners.h8 == _BL
    assert corners.a8 == _BR


def test_manifest_entry_shape() -> None:
    entry = manifest_entry(
        "img/0001.png", _SHUFFLED, white_turn=True, fen="8/8/8/8/8/8/8/8 w - - 0 1"
    )
    assert entry["image"] == "img/0001.png"
    assert set(entry["corners"]) == {"a1", "h1", "h8", "a8"}
    assert entry["corners"]["a1"] == [0.0, 10.0]
    assert entry["fen"].startswith("8/8")


def _load_converter() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "chesscog_to_manifest.py"
    spec = importlib.util.spec_from_file_location("chesscog_to_manifest", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_record(directory: Path, stem: str, *, with_png: bool) -> None:
    record = {
        "corners": [list(_BR), list(_TL), list(_BL), list(_TR)],
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "white_turn": True,
    }
    (directory / f"{stem}.json").write_text(json.dumps(record), encoding="utf-8")
    if with_png:
        (directory / f"{stem}.png").write_bytes(b"")


def test_build_manifest_skips_records_without_image(tmp_path: Path) -> None:
    _write_record(tmp_path, "a", with_png=True)
    _write_record(tmp_path, "b", with_png=True)
    _write_record(tmp_path, "c", with_png=False)  # no image -> skipped
    converter = _load_converter()
    entries = converter.build_manifest(tmp_path)
    assert len(entries) == 2
    assert {entry["image"] for entry in entries} == {"a.png", "b.png"}
    assert set(entries[0]["corners"]) == {"a1", "h1", "h8", "a8"}
