"""Rollout logger (observations, actions, rewards, interventions, submoves).

Append-only JSONL logger for autonomous / RL / HIL rollouts. Stored **separately**
from supervised demonstrations (  ): pass an autonomous-rollout directory.
Each episode is one JSON line with its steps. Per step it records the submove
index/role, observation state, base/residual/final actions (+ residual norm),
reward, intervention/corrected action, safety-violation flag, failure type, and
the board state (FEN); per episode it records success/failure labels.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from chess_robot.chess.board_state import BoardState


def _to_list(values: Sequence[float] | None) -> list[float] | None:
    return None if values is None else [float(v) for v in values]


def _enum_value(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


@dataclass
class RolloutStep:
    """One logged control step within a rollout."""

    step_index: int
    submove_index: int
    submove_role: str | None
    timestamp: float
    state: list[float] | None = None
    base_action: list[float] | None = None
    residual_action: list[float] | None = None
    final_action: list[float] | None = None
    residual_norm: float | None = None
    reward: float | None = None
    intervention_flag: bool = False
    corrected_action: list[float] | None = None
    safety_violation_flag: bool = False
    failure_type: str | None = None
    board_state_fen: str | None = None


@dataclass
class RolloutEpisode:
    """One logged rollout episode."""

    episode_index: int
    instruction: str
    piece_type: str | None = None
    start_square: str | None = None
    target_square: str | None = None
    is_capture: bool = False
    success_label: bool | None = None
    failure_type: str | None = None
    steps: list[RolloutStep] = field(default_factory=list)


class RolloutLogger:
    """Buffer a rollout episode in memory and flush it as one JSONL record."""

    def __init__(
        self, output_dir: str | Path, *, filename: str = "rollouts.jsonl"
    ) -> None:
        self._path = Path(output_dir) / filename
        self._episode: RolloutEpisode | None = None
        self._step_index = 0

    @property
    def path(self) -> Path:
        return self._path

    def start_episode(
        self,
        episode_index: int,
        instruction: str,
        *,
        piece_type: object = None,
        start_square: str | None = None,
        target_square: str | None = None,
        is_capture: bool = False,
    ) -> None:
        self._episode = RolloutEpisode(
            episode_index=episode_index,
            instruction=instruction,
            piece_type=_enum_value(piece_type),
            start_square=start_square,
            target_square=target_square,
            is_capture=is_capture,
        )
        self._step_index = 0

    def log_step(
        self,
        *,
        submove_index: int,
        submove_role: object = None,
        state: Sequence[float] | None = None,
        base_action: Sequence[float] | None = None,
        residual_action: Sequence[float] | None = None,
        final_action: Sequence[float] | None = None,
        residual_norm: float | None = None,
        reward: float | None = None,
        intervention_flag: bool = False,
        corrected_action: Sequence[float] | None = None,
        safety_violation_flag: bool = False,
        failure_type: object = None,
        board_state: BoardState | None = None,
        timestamp: float | None = None,
    ) -> None:
        if self._episode is None:
            raise RuntimeError("log_step called before start_episode")
        residual = _to_list(residual_action)
        if residual_norm is None and residual is not None:
            residual_norm = math.sqrt(sum(value * value for value in residual))
        self._episode.steps.append(
            RolloutStep(
                step_index=self._step_index,
                submove_index=submove_index,
                submove_role=_enum_value(submove_role),
                timestamp=time.time() if timestamp is None else timestamp,
                state=_to_list(state),
                base_action=_to_list(base_action),
                residual_action=residual,
                final_action=_to_list(final_action),
                residual_norm=residual_norm,
                reward=reward,
                intervention_flag=intervention_flag,
                corrected_action=_to_list(corrected_action),
                safety_violation_flag=safety_violation_flag,
                failure_type=_enum_value(failure_type),
                board_state_fen=None if board_state is None else board_state.to_fen(),
            )
        )
        self._step_index += 1

    def end_episode(
        self, *, success_label: bool | None = None, failure_type: object = None
    ) -> RolloutEpisode:
        if self._episode is None:
            raise RuntimeError("end_episode called before start_episode")
        self._episode.success_label = success_label
        if failure_type is not None:
            self._episode.failure_type = _enum_value(failure_type)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(self._episode)) + "\n")
        finished = self._episode
        self._episode = None
        return finished


def read_rollouts(path: str | Path) -> list[dict[str, Any]]:
    """Read rollout episode records from a JSONL file."""
    text = Path(path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]
