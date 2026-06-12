"""Teleoperator (leader arm): produces the actions that drive the follower.

The recording loop reads an action from the teleoperator each step and sends it
to the follower (through the safety layer). :class:`MockTeleoperator` scripts
actions for tests/dry-runs; :class:`SO101Leader` is the real leader via LeRobot
(lazy import, verified on the lab machine — lines marked ``VERIFY(lab)``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np

from chess_robot.robot.actions import ActionSchema, to_action_vector


class Teleoperator(ABC):
    """Leader-arm contract: connect, then produce one action vector per step."""

    def __init__(self, schema: ActionSchema) -> None:
        self._schema = schema

    @property
    def schema(self) -> ActionSchema:
        return self._schema

    @property
    @abstractmethod
    def is_connected(self) -> bool: ...

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def get_action(self) -> np.ndarray: ...


class MockTeleoperator(Teleoperator):
    """Scripted teleoperator for tests/dry-runs (cycles a list, or yields zeros)."""

    def __init__(
        self, schema: ActionSchema, *, actions: Sequence[Sequence[float]] | None = None
    ) -> None:
        super().__init__(schema)
        self._connected = False
        self._actions = [to_action_vector(a) for a in actions] if actions else None
        self._step = 0

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def get_action(self) -> np.ndarray:
        if self._actions:
            action = self._actions[self._step % len(self._actions)]
        else:
            action = np.zeros(self._schema.action_dim, dtype=np.float32)
        self._step += 1
        return action


class SO101Leader(Teleoperator):
    """Real SO-101 leader via LeRobot. Lazy import; ``VERIFY(lab)`` lines."""

    def __init__(self, schema: ActionSchema, *, leader_port: str) -> None:
        super().__init__(schema)
        self._leader_port = leader_port
        self._teleop: Any = None

    @property
    def is_connected(self) -> bool:
        return self._teleop is not None

    def connect(self) -> None:
        # VERIFY(lab): LeRobot SO-101 leader API (lerobot >= 0.5).
        from lerobot.teleoperators.so101_leader import SO101Leader as _Leader
        from lerobot.teleoperators.so101_leader import SO101LeaderConfig

        self._teleop = _Leader(SO101LeaderConfig(port=self._leader_port))
        self._teleop.connect()

    def disconnect(self) -> None:
        if self._teleop is not None:
            self._teleop.disconnect()
            self._teleop = None

    def get_action(self) -> np.ndarray:
        if self._teleop is None:
            raise RuntimeError("get_action before connect()")
        raw = self._teleop.get_action()  # VERIFY(lab): {joint.pos: value}
        return to_action_vector([float(raw[f"{joint}.pos"]) for joint in self._schema.joint_names])
