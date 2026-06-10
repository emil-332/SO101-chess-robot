"""Tests for the two-stage piece-classifier config loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from chess_robot.perception.piece_cnn_config import (
    StageConfig,
    load_two_stage_config,
    smoke_two_stage_config,
)

_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "perception" / "piece_cnn.yaml"
)


def test_config_loads_both_stages() -> None:
    config = load_two_stage_config(_CONFIG_PATH)
    assert config.occupancy.backbone == "resnet18"
    assert config.occupancy.input_size == (100, 100)
    assert config.occupancy.top_pad_ratio == 0.3
    assert config.piece.backbone == "resnet34"
    assert config.piece.input_size == (144, 96)
    assert config.piece.top_pad_ratio == 1.0
    assert config.pretrained is True
    assert config.camera == "observation.images.side"
    assert config.head_epochs == 1
    assert config.full_epochs == 3
    assert config.head_lr == 0.001
    assert config.full_lr == 0.0001


def test_smoke_config_is_cheap() -> None:
    config = smoke_two_stage_config(load_two_stage_config(_CONFIG_PATH))
    assert config.head_epochs == 1
    assert config.full_epochs == 1
    assert config.batch_size <= 8


def test_stage_config_rejects_unknown_backbone() -> None:
    with pytest.raises(ValueError, match="unsupported backbone"):
        StageConfig(backbone="vgg16", input_size=(100, 100), top_pad_ratio=0.3)


def test_stage_config_rejects_bad_geometry() -> None:
    with pytest.raises(ValueError):
        StageConfig(backbone="resnet18", input_size=(0, 100), top_pad_ratio=0.3)
    with pytest.raises(ValueError):
        StageConfig(backbone="resnet18", input_size=(100, 100), top_pad_ratio=-1.0)
