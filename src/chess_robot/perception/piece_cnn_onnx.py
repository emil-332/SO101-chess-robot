"""Laptop-side two-stage piece classifier using ONNX Runtime (no torch).

Loads the two ONNX stage models exported by :mod:`piece_cnn` and runs them with
onnxruntime on the laptop iGPU/CPU (the compute split keeps torch on the cloud).
Crop extraction, normalization, and prediction-to-board logic are reused from the
torch-free :mod:`piece_dataset`, so this path preprocesses identically to training.
Implements :class:`PieceClassifier`, so it drops into ``ComposedBoardPerception``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from chess_robot.chess.board_mapper import GroundedGrid
from chess_robot.chess.board_state import BoardState
from chess_robot.perception.board_perception import CameraFrames
from chess_robot.perception.piece_cnn_config import TwoStageConfig
from chess_robot.perception.piece_dataset import (
    board_from_piece_predictions,
    normalize_batch,
    occupancy_inference_crops,
    occupied_from_predictions,
    piece_inference_crops,
)
from chess_robot.perception.piece_locator import PieceClassifier


def _run_session(session: Any, crops: np.ndarray) -> np.ndarray:
    if len(crops) == 0:
        return np.empty((0,), dtype=np.int64)
    inputs = {session.get_inputs()[0].name: normalize_batch(crops)}
    logits = session.run(None, inputs)[0]
    return np.asarray(logits).argmax(axis=1).astype(np.int64)


class OnnxTwoStageClassifier(PieceClassifier):
    """Run the two exported ONNX stages on the laptop."""

    def __init__(
        self,
        occupancy_session: Any,
        piece_session: Any,
        config: TwoStageConfig,
    ) -> None:
        self._occupancy_session = occupancy_session
        self._piece_session = piece_session
        self._config = config

    @classmethod
    def from_paths(
        cls,
        config: TwoStageConfig,
        occupancy_onnx: str | Path,
        piece_onnx: str | Path,
        *,
        providers: list[str] | None = None,
    ) -> OnnxTwoStageClassifier:
        import onnxruntime as ort

        session_providers = providers or ["CPUExecutionProvider"]
        occupancy_session = ort.InferenceSession(str(occupancy_onnx), providers=session_providers)
        piece_session = ort.InferenceSession(str(piece_onnx), providers=session_providers)
        return cls(occupancy_session, piece_session, config)

    def classify(self, frames: CameraFrames, grid: GroundedGrid) -> BoardState:
        image = np.asarray(frames[self._config.camera])
        squares, occupancy_crops = occupancy_inference_crops(
            image,
            grid,
            size=self._config.occupancy.input_size,
            top_pad_ratio=self._config.occupancy.top_pad_ratio,
        )
        if not squares:
            return BoardState.empty()
        occupancy_pred = _run_session(self._occupancy_session, occupancy_crops)
        occupied = occupied_from_predictions(squares, occupancy_pred.tolist())
        if not occupied:
            return BoardState.empty()
        piece_crops = piece_inference_crops(
            image,
            grid,
            occupied,
            size=self._config.piece.input_size,
            top_pad_ratio=self._config.piece.top_pad_ratio,
        )
        piece_pred = _run_session(self._piece_session, piece_crops)
        return board_from_piece_predictions(occupied, piece_pred.tolist())
