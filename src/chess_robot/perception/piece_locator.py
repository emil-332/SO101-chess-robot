"""Piece localization / classification: per-square occupancy and identity.

Chosen approach (see ``docs/architecture.md``): a lightweight **per-square CNN**
(MobileNetV3 / EfficientNet-class) classifies each grounded square as empty or a
piece (type + colour) from the oblique/side camera. Piece classification is
**few-shot** — fine-tuned on ~2 starting-position photos per new board. That
model (trained on the cloud GPU, ONNX/OpenVINO on the laptop) is a
:class:`PieceClassifier` implementation and is the 1b.2 follow-up (needs a
dataset + a ``perception`` extra).

This module defines the interface and :class:`MetadataPieceClassifier`, the
metadata-sourced bootstrap/cross-check that returns a supplied occupancy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

from chess_robot.chess.board_mapper import GroundedGrid
from chess_robot.chess.board_state import BoardState

# Map of stable camera key -> raw frame (e.g. observation.images.side).
CameraFrames = Mapping[str, object]


class PieceClassifier(ABC):
    """Classify each grounded square into occupancy + piece identity.

    Takes the camera frames and the grounded grid (square -> region) and returns
    a :class:`BoardState`. The few-shot per-square CNN is the target
    implementation; it reads its configured (side/oblique) camera from ``frames``.
    """

    @abstractmethod
    def classify(self, frames: CameraFrames, grid: GroundedGrid) -> BoardState:
        raise NotImplementedError


class MetadataPieceClassifier(PieceClassifier):
    """Bootstrap/cross-check: return a supplied occupancy, ignoring the frames.

    Mirrors the metadata-first path so a composed perception pipeline can run
    deterministically before (or as a fallback to) the learned classifier.
    """

    def __init__(self, board_state: BoardState) -> None:
        self._board_state = board_state

    def classify(
        self, frames: CameraFrames | None = None, grid: GroundedGrid | None = None
    ) -> BoardState:
        return self._board_state
