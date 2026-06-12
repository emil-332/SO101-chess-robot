"""Action representation for the SO-101: the action-vector layout.

The policy and the safety layer speak a flat action vector. This module pins what
each index means (which entries are arm joints, which is the gripper) so the
safety layer can apply per-joint and gripper-range checks. The SO-101 default is
5 arm joints + gripper; the layout is config-driven (`configs/robot/so101.yaml`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# SO-101 motors in action-vector order (last entry is the gripper).
SO101_JOINTS: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


@dataclass(frozen=True)
class ActionSchema:
    """The action-vector layout: joint names (in order) and the gripper index."""

    joint_names: tuple[str, ...]
    gripper_index: int

    def __post_init__(self) -> None:
        if not self.joint_names:
            raise ValueError("joint_names must be non-empty")
        if not 0 <= self.gripper_index < len(self.joint_names):
            raise ValueError(
                f"gripper_index {self.gripper_index} out of range for "
                f"{len(self.joint_names)} joints"
            )

    @property
    def action_dim(self) -> int:
        return len(self.joint_names)

    @property
    def arm_joint_indices(self) -> tuple[int, ...]:
        """Action indices that are arm joints (everything but the gripper)."""
        return tuple(i for i in range(self.action_dim) if i != self.gripper_index)


def so101_action_schema() -> ActionSchema:
    """The default SO-101 layout (5 arm joints + gripper last)."""
    return ActionSchema(joint_names=SO101_JOINTS, gripper_index=SO101_JOINTS.index("gripper"))


def load_action_schema(path: str | Path) -> ActionSchema:
    """Load the action layout from a robot config (configs/robot/so101.yaml)."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    robot = raw.get("robot", {}) if isinstance(raw, Mapping) else {}
    joints = tuple(str(j) for j in robot.get("joints", SO101_JOINTS))
    gripper = str(robot.get("gripper_joint", "gripper"))
    if gripper not in joints:
        raise ValueError(f"gripper_joint {gripper!r} not in joints {joints}")
    return ActionSchema(joint_names=joints, gripper_index=joints.index(gripper))


def to_action_vector(action: Sequence[float] | np.ndarray) -> np.ndarray:
    """Coerce an action to a float32 numpy vector."""
    return np.asarray(action, dtype=np.float32).reshape(-1)


def action_to_motor_dict(action: Sequence[float], schema: ActionSchema) -> dict[str, Any]:
    """Map a flat action vector to a ``{joint: value}`` command dict."""
    values = to_action_vector(action)
    if values.shape != (schema.action_dim,):
        raise ValueError(f"action dim {values.shape[0]} != {schema.action_dim}")
    return {name: float(value) for name, value in zip(schema.joint_names, values, strict=True)}
