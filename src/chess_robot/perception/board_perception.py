"""Board perception interface: raw image -> grid + occupancy + piece identity.

Perception is preprocessing (see ``docs/architecture.md``): it converts raw
camera frames into a structured board state BEFORE anything reaches pi0.5. Board
geometry generalizes **zero-shot**; piece classification is **few-shot** (~2
photos per new board). It always runs OUTSIDE the pi0.5 backbone.

This module defines the interface (:class:`BoardPerception`), a metadata-supplied
bootstrap (:class:`MetadataBoardPerception`), and the composed learned pipeline
(:class:`ComposedBoardPerception`). Occupancy may come from metadata (bootstrap /
fallback) or the perception model (target); both produce the same
:class:`PerceivedBoard`, so they are interchangeable and cross-checkable
(:func:`cross_check_occupancy`). ``PerceivedBoard.source`` records which produced
it. Grounding is **per camera** (``PerceivedBoard.grids``): the overhead grid
grounds/​highlights squares; the side/oblique grid is what the piece classifier
crops from.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from chess_robot.chess.board_mapper import GroundedGrid
from chess_robot.chess.board_state import NUM_SQUARES, BoardState, Square
from chess_robot.perception.piece_locator import PieceClassifier
from chess_robot.perception.square_grounding import SquareGrounder

# A raw camera frame (e.g. a numpy HxWxC array). Typed loosely: the metadata
# bootstrap ignores it and image backends vary; the learned model narrows it.
Image = Any
# Map of stable camera key -> frame (observation.images.overhead / .side / .wrist).
CameraFrames = Mapping[str, Image]

OCCUPANCY_SOURCE_METADATA = "metadata"
OCCUPANCY_SOURCE_PERCEPTION = "perception"

DEFAULT_GROUNDING_CAMERA = "observation.images.overhead"
DEFAULT_PIECE_CAMERA = "observation.images.side"


@dataclass(frozen=True)
class PerceivedBoard:
    """Structured perception output for one set of camera frames.

    - ``board_state``: per-square occupancy and piece identity.
    - ``grid``: the primary (grounding-camera) grid — used for highlighting.
    - ``source``: which path produced occupancy (``"metadata"`` vs ``"perception"``).
    - ``grids``: per-camera grids when available (e.g. overhead + side), so each
      policy-facing camera can be highlighted with its own geometry and the
      classifier can crop from the side camera's grid.
    """

    board_state: BoardState
    grid: GroundedGrid
    source: str
    grids: Mapping[str, GroundedGrid] | None = None


class BoardPerception(ABC):
    """Interface: raw camera frames -> a structured :class:`PerceivedBoard`.

    Implementations run as preprocessing, OUTSIDE the pi0.5 backbone.
    """

    @abstractmethod
    def perceive(self, frames: CameraFrames) -> PerceivedBoard:
        """Read the camera frames into a :class:`PerceivedBoard`.

        ``frames`` maps stable camera keys (e.g. ``observation.images.overhead``,
        ``observation.images.side``) to raw frames.
        """
        raise NotImplementedError


class MetadataBoardPerception(BoardPerception):
    """Bootstrap/fallback perception sourced from supplied metadata.

    Returns occupancy/identity from a given :class:`BoardState` (not from the
    image), so downstream manipulation work is not blocked on the learned model.
    The ``frames`` argument is accepted for interface compatibility and ignored.
    A grounded ``grid`` may be supplied (e.g. from calibration) so highlighting
    still works; otherwise the grid is empty.
    """

    def __init__(
        self, board_state: BoardState, grid: GroundedGrid | None = None
    ) -> None:
        self._board_state = board_state
        self._grid = grid if grid is not None else GroundedGrid()

    def perceive(self, frames: CameraFrames | None = None) -> PerceivedBoard:
        return PerceivedBoard(
            board_state=self._board_state,
            grid=self._grid,
            source=OCCUPANCY_SOURCE_METADATA,
            grids={DEFAULT_GROUNDING_CAMERA: self._grid},
        )


class ComposedBoardPerception(BoardPerception):
    """Learned perception: a :class:`SquareGrounder` + a :class:`PieceClassifier`.

    The overhead camera is grounded for square highlighting; if a ``piece_grounder``
    is supplied the side/oblique camera is grounded too and **that** grid is passed
    to the classifier (which crops piece images from the side view). This is the
    target structure for the trained YOLO-corner detector + per-square CNN; with
    the bootstrap components it also runs deterministically today.
    """

    def __init__(
        self,
        grounder: SquareGrounder,
        classifier: PieceClassifier,
        *,
        grounding_camera: str = DEFAULT_GROUNDING_CAMERA,
        piece_grounder: SquareGrounder | None = None,
        piece_camera: str = DEFAULT_PIECE_CAMERA,
    ) -> None:
        self._grounder = grounder
        self._classifier = classifier
        self._grounding_camera = grounding_camera
        self._piece_grounder = piece_grounder
        self._piece_camera = piece_camera

    def perceive(self, frames: CameraFrames) -> PerceivedBoard:
        if self._grounding_camera not in frames:
            raise KeyError(
                f"missing grounding camera frame {self._grounding_camera!r}"
            )
        grounding_grid = self._grounder.ground(frames[self._grounding_camera])
        grids: dict[str, GroundedGrid] = {self._grounding_camera: grounding_grid}

        classify_grid = grounding_grid
        if self._piece_grounder is not None:
            if self._piece_camera not in frames:
                raise KeyError(
                    f"missing piece camera frame {self._piece_camera!r}"
                )
            piece_grid = self._piece_grounder.ground(frames[self._piece_camera])
            grids[self._piece_camera] = piece_grid
            classify_grid = piece_grid

        board_state = self._classifier.classify(frames, classify_grid)
        return PerceivedBoard(
            board_state=board_state,
            grid=grounding_grid,
            source=OCCUPANCY_SOURCE_PERCEPTION,
            grids=grids,
        )


def cross_check_occupancy(a: PerceivedBoard, b: PerceivedBoard) -> list[str]:
    """Per-square disagreements between two perceptions (e.g. metadata vs model).

    Empty list == they agree. Use to validate the learned model against the
    metadata path (``docs/architecture.md``: the two sources must be cross-checkable).
    """
    problems: list[str] = []
    for index in range(NUM_SQUARES):
        square = Square.from_index(index)
        piece_a = a.board_state.piece_at(square)
        piece_b = b.board_state.piece_at(square)
        if (piece_a is None) != (piece_b is None):
            problems.append(
                f"{square.name}: occupancy mismatch "
                f"({a.source}={'occupied' if piece_a else 'empty'} vs "
                f"{b.source}={'occupied' if piece_b else 'empty'})"
            )
        elif (
            piece_a is not None
            and piece_b is not None
            and piece_a.piece_type is not piece_b.piece_type
        ):
            problems.append(
                f"{square.name}: piece mismatch "
                f"({a.source}={piece_a.piece_type.value} vs "
                f"{b.source}={piece_b.piece_type.value})"
            )
    return problems
