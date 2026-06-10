"""Torch-free coverage of the piece-CNN scripts: renderer prep + train dry-run."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from chess_robot.perception.piece_dataset import load_dataset

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_renderer_then_train_dry_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "piece_smoke.npz"
    prepare = _load_script("prepare_piece_dataset")
    prepare.main(
        [
            "--source", "renderer",
            "--num-boards", "6",
            "--image-size", "160", "160",
            "--jitter", "0",
            "--out", str(out),
        ]
    )
    assert out.exists()
    dataset = load_dataset(out)
    assert len(dataset.occupancy_labels) == 6 * 64  # all squares grounded with no jitter

    train = _load_script("train_piece_cnn")
    train.main(["--data", str(out), "--dry-run", "--smoke"])
    captured = capsys.readouterr()
    assert "piece-CNN training plan" in captured.out
    assert "occupancy" in captured.out
