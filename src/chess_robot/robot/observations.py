"""Package robot state + camera frames into the observation dict.

Produces the raw observation with the project's stable keys
(``observation.state`` + ``observation.images.<name>``) that the dataset recorder
and the policy consume. Perception preprocessing (square highlighting) happens
downstream in the observation builder, not here; this is the raw robot reading.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from chess_robot.robot.actions import ActionSchema

STATE_KEY = "observation.state"


def build_robot_observation(
    state: Sequence[float] | np.ndarray,
    frames: Mapping[str, Any],
    *,
    schema: ActionSchema,
) -> dict[str, Any]:
    """Assemble ``{observation.state, observation.images.*}`` from a robot reading.

    ``state`` is the joint-position vector (length = ``schema.action_dim``);
    ``frames`` maps stable camera keys to raw frames.
    """
    state_array = np.asarray(state, dtype=np.float32).reshape(-1)
    if state_array.shape != (schema.action_dim,):
        raise ValueError(
            f"state dim {state_array.shape[0]} != schema action_dim {schema.action_dim}"
        )
    observation: dict[str, Any] = {STATE_KEY: state_array}
    for key, frame in frames.items():
        observation[key] = np.asarray(frame)
    return observation
