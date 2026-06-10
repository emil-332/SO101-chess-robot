"""Tests for the torch piece-CNN training/inference path.

Skipped where torch/torchvision are absent (the laptop dev env); they run on the
cloud GPU box where the perception-train extra is installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

import torch  # noqa: E402

from chess_robot.chess.board_mapper import GroundedGrid  # noqa: E402
from chess_robot.chess.board_state import BoardState  # noqa: E402
from chess_robot.perception import piece_cnn  # noqa: E402
from chess_robot.perception.board_renderer import render_board  # noqa: E402
from chess_robot.perception.piece_cnn_config import StageConfig, TwoStageConfig  # noqa: E402
from chess_robot.perception.piece_dataset import PieceCropDataset  # noqa: E402
from chess_robot.perception.square_grounding import BoardCorners, grid_from_corners  # noqa: E402

IMAGE_SIZE = (200, 200)


def _tiny_config() -> TwoStageConfig:
    stage = StageConfig(backbone="mobilenet_v3_small", input_size=(64, 64), top_pad_ratio=0.3)
    piece_stage = StageConfig(backbone="mobilenet_v3_small", input_size=(64, 64), top_pad_ratio=1.0)
    return TwoStageConfig(
        occupancy=stage,
        piece=piece_stage,
        pretrained=False,
        camera="observation.images.side",
        batch_size=4,
        head_epochs=1,
        full_epochs=1,
        head_lr=1e-3,
        full_lr=1e-4,
        weight_decay=1e-4,
        seed=0,
        device="cpu",
        val_fraction=0.25,
    )


def _tiny_dataset() -> PieceCropDataset:
    rng = np.random.default_rng(0)
    occ_images = (rng.random((12, 64, 64, 3)) * 255).astype(np.uint8)
    occ_labels = np.array([0, 1] * 6, dtype=np.int64)
    piece_images = (rng.random((12, 64, 64, 3)) * 255).astype(np.uint8)
    piece_labels = (np.arange(12) % 12).astype(np.int64)
    return PieceCropDataset(occ_images, occ_labels, piece_images, piece_labels)


def _render_board() -> tuple[np.ndarray, GroundedGrid]:
    height, width = IMAGE_SIZE
    mx, my = 0.08 * width, 0.08 * height
    corners = BoardCorners(
        a1=(mx, height - my), h1=(width - mx, height - my), h8=(width - mx, my), a8=(mx, my)
    )
    rendered = render_board(BoardState.standard_starting_position(), corners, image_size=IMAGE_SIZE)
    return rendered.image, grid_from_corners(corners)


def test_build_model_forward_shape() -> None:
    model, head = piece_cnn.build_model("resnet18", num_classes=2, pretrained=False)
    output = model(torch.randn(2, 3, 64, 64))
    assert output.shape == (2, 2)
    assert head.out_features == 2


def test_array_dataset_item() -> None:
    images = np.zeros((3, 64, 64, 3), dtype=np.uint8)
    labels = np.array([0, 1, 0], dtype=np.int64)
    dataset = piece_cnn._ArrayDataset(images, labels, train=True, seed=0)
    image, label = dataset[0]
    assert image.shape == (3, 64, 64)
    assert label in (0, 1)


def test_run_training_both_stages(tmp_path: Path) -> None:
    config = _tiny_config()
    results = piece_cnn.run_training(
        config, _tiny_dataset(), stage="both", out_dir=tmp_path, export=False
    )
    assert set(results) == {"occupancy", "piece"}
    for result in results.values():
        assert 0.0 <= result["val_accuracy"] <= 1.0
        assert (tmp_path / "occupancy.pt").exists()
        assert (tmp_path / "piece.pt").exists()


def test_classifier_returns_board(tmp_path: Path) -> None:
    config = _tiny_config()
    piece_cnn.run_training(config, _tiny_dataset(), stage="both", out_dir=tmp_path, export=False)
    classifier = piece_cnn.TorchTwoStageClassifier.from_checkpoints(
        config, tmp_path / "occupancy.pt", tmp_path / "piece.pt", device="cpu"
    )
    image, grid = _render_board()
    board = classifier.classify({config.camera: image}, grid)
    assert isinstance(board, BoardState)
    assert len(board) <= 64


def test_export_onnx(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    model, _ = piece_cnn.build_model("mobilenet_v3_small", num_classes=2, pretrained=False)
    path = tmp_path / "occupancy.onnx"
    piece_cnn.export_onnx(model, path, (64, 64), "cpu")
    assert path.exists()


def test_unsupported_backbone_raises() -> None:
    with pytest.raises(ValueError, match="unsupported backbone"):
        piece_cnn.build_model("vgg16", num_classes=2, pretrained=False)
