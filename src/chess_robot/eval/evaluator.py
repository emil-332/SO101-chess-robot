"""Evaluation harness / runner (  4.1).

Loads the eval config, maps logged rollouts (from ``utils.logging.RolloutLogger``)
into :class:`EvalEpisode` s, and produces a :class:`MetricReport`. Heavy eval jobs
may run on the cloud GPU, but this orchestration is laptop-runnable.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from chess_robot.data.schema import FailureType
from chess_robot.eval.metrics import EvalEpisode, MetricReport, evaluate_episodes
from chess_robot.utils.logging import read_rollouts


@dataclass(frozen=True)
class EvalConfig:
    """Parsed evaluation config (see configs/eval/chess_eval.yaml)."""

    num_episodes: int | None
    occupancy_source: str
    compare: tuple[str, ...]


def load_eval_config(path: str | Path) -> EvalConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    section = raw.get("eval", {}) if isinstance(raw, Mapping) else {}
    num_episodes = section.get("num_episodes")
    compare = section.get("compare") or []
    return EvalConfig(
        num_episodes=num_episodes if isinstance(num_episodes, int) else None,
        occupancy_source=str(section.get("occupancy_source", "metadata")),
        compare=tuple(str(name) for name in compare),
    )


def _failure_type(value: object) -> FailureType | None:
    if value is None or value == "":
        return None
    return FailureType(value)


def eval_episode_from_rollout(record: Mapping[str, Any]) -> EvalEpisode:
    """Map a rollout episode record (from RolloutLogger) into an EvalEpisode.

    Fields not present in the rollout log (grasp/target error/…) stay ``None`` and
    are excluded from their metric. ``success_label`` of None counts as not-success.
    """
    steps = record.get("steps") or []
    residual_norms = [
        s["residual_norm"] for s in steps if s.get("residual_norm") is not None
    ]
    return EvalEpisode(
        success=bool(record.get("success_label")),
        failure_type=_failure_type(record.get("failure_type")),
        is_capture=bool(record.get("is_capture", False)),
        intervention_count=sum(1 for s in steps if s.get("intervention_flag")),
        safety_violation=any(s.get("safety_violation_flag") for s in steps),
        mean_residual_norm=(
            sum(residual_norms) / len(residual_norms) if residual_norms else None
        ),
    )


def evaluate_rollout_file(path: str | Path) -> MetricReport:
    """Read a rollouts JSONL file and aggregate it into a MetricReport."""
    return evaluate_episodes(
        eval_episode_from_rollout(record) for record in read_rollouts(path)
    )
