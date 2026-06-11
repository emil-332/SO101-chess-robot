"""LeRobotDataset episode schema definitions (board_state, capture/submove fields).

Defines the structured metadata/label schema recorded alongside each episode (see
``docs/data_collection.md``). The observation/action tensors live in the LeRobot
row under the stable key constants below and are presence-checked by
``validation.find_missing_required_fields``; :class:`EpisodeRecord` captures the
structured move metadata and labels that the resolver and evaluation depend on.

"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from chess_robot.chess.board_state import BoardState, PieceType
from chess_robot.chess.move_resolver import SubmoveRole

# --- Stable LeRobot row keys (never rename silently; see CLAUDE.md) -----------
INSTRUCTION_KEY = "instruction"
OBSERVATION_STATE_KEY = "observation.state"
ACTION_KEY = "action"
TIMESTAMP_KEY = "timestamp"
EPISODE_INDEX_KEY = "episode_index"
CAMERA_KEY_PREFIX = "observation.images."
CAMERA_KEYS: tuple[str, ...] = (
    "observation.images.overhead",
    "observation.images.side",
    "observation.images.wrist",
)

# Scalar keys that must be present in a supervised training row (a camera image,
# matched by CAMERA_KEY_PREFIX, is additionally required — see validation).
REQUIRED_FIELDS: tuple[str, ...] = (
    INSTRUCTION_KEY,
    OBSERVATION_STATE_KEY,
    ACTION_KEY,
    TIMESTAMP_KEY,
    EPISODE_INDEX_KEY,
    "piece_type",
    "start_square",
    "target_square",
)

# Strongly recommended: evaluation and the deterministic resolver rely on these.
RECOMMENDED_FIELDS: tuple[str, ...] = (
    "board_state",
    "is_capture",
    "submove_index",
    "submove_role",
    "captured_piece_type",
)


class FailureType(str, Enum):
    """Stable failure-type labels (extended for the full-board / capture setting).

    The dataset's source of truth; keep names stable.
    """

    BAD_GRASP = "bad_grasp"
    MISSED_PIECE = "missed_piece"
    DROPPED_PIECE = "dropped_piece"
    WRONG_SQUARE = "wrong_square"
    WRONG_PIECE = "wrong_piece"
    RELEASE_FAILURE = "release_failure"
    COLLISION = "collision"
    UNSAFE_MOTION = "unsafe_motion"
    CAPTURE_REMOVAL_FAILURE = "capture_removal_failure"
    TIMEOUT = "timeout"
    PERCEPTION_ERROR = "perception_error"
    CAMERA_FAILURE = "camera_failure"
    UNKNOWN = "unknown"


@dataclass
class EpisodeRecord:
    """Structured per-episode/per-submove metadata and labels.

    Defaults describe a single non-capturing move (``is_capture=False``,
    ``submove_index=0``, ``submove_role=MOVE``). ``start_square`` / ``target_square``
    are algebraic strings (e.g. ``"b1"``); ``piece_type`` is the *instructed*
    piece, while a captured piece is recorded in ``captured_piece_type``.
    """

    instruction: str
    piece_type: PieceType
    start_square: str
    target_square: str
    is_capture: bool = False
    submove_index: int = 0
    submove_role: SubmoveRole = SubmoveRole.MOVE
    captured_piece_type: PieceType | None = None
    board_state: BoardState | None = None
    success_label: bool | None = None
    failure_type: FailureType | None = None
    intervention_flag: bool = False
    corrected_action: Any | None = None
    reward: float | None = None


@dataclass
class Feedback:
    """Feedback object for RL/HIL rollouts (see ``docs/data_collection.md``).

    ``failure_type`` is tightened from ``str`` to :class:`FailureType` for
    consistency with the dataset schema.
    """

    success_label: bool | None
    scalar_reward: float | None
    intervention_flag: bool
    corrected_action: Any | None
    safety_violation_flag: bool
    failure_type: FailureType | None
    submove_index: int = 0
    preference_label: Any | None = None
