"""SO-101 robot interface: safety-routed actions, observations, mock + real.

Every action goes through the safety layer by construction: :meth:`send_action`
calls :meth:`SafetyLayer.enforce` before anything reaches the hardware, so a
caller cannot bypass it. :class:`MockRobot` runs the whole contract with no
hardware (tests, dry-runs). :class:`SO101Robot` is the real arm via LeRobot; it
**refuses to connect until the safety limits are configured** (the hardware gate)
and its LeRobot calls are lazily imported and verified on the lab machine against
the installed LeRobot — adjust the lines marked ``VERIFY(lab)`` if the API
differs (the same way the pi0.5 CLI flags were verified).

Keep real-robot execution separate from learning code: this module imports
neither torch nor a policy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from chess_robot.robot.actions import ActionSchema, action_to_motor_dict
from chess_robot.robot.observations import build_robot_observation
from chess_robot.safety.safety_layer import SafetyLayer


class RobotNotReadyError(RuntimeError):
    """Raised when a real-robot operation is attempted before it is safe/connected."""


class RobotInterface(ABC):
    """Common robot contract; routes every action through the safety layer."""

    def __init__(self, safety: SafetyLayer, schema: ActionSchema) -> None:
        self._safety = safety
        self._schema = schema

    @property
    def safety(self) -> SafetyLayer:
        return self._safety

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
    def read_observation(self) -> dict[str, Any]: ...

    @abstractmethod
    def home(self) -> None: ...

    @abstractmethod
    def _send_raw(self, action: Sequence[float]) -> None:
        """Send an already-safety-checked action to the hardware."""

    def send_action(
        self,
        action: Sequence[float],
        *,
        observation_age_s: float | None = None,
        camera_ok: bool = True,
        elapsed_s: float | None = None,
        previous_action: Sequence[float] | None = None,
        dt_s: float | None = None,
    ) -> Sequence[float]:
        """Safety-check ``action`` then send it; raises if unsafe (fail-closed)."""
        safe = self._safety.enforce(
            action,
            observation_age_s=observation_age_s,
            camera_ok=camera_ok,
            robot_connected=self.is_connected,
            elapsed_s=elapsed_s,
            previous_action=previous_action,
            dt_s=dt_s,
        )
        self._send_raw(safe)
        return safe


class MockRobot(RobotInterface):
    """In-memory robot for tests / dry-runs: records actions, no hardware."""

    def __init__(
        self,
        safety: SafetyLayer,
        schema: ActionSchema,
        *,
        frames: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(safety, schema)
        self._connected = False
        self._frames = dict(frames or {})
        self._state = np.zeros(schema.action_dim, dtype=np.float32)
        self.sent_actions: list[list[float]] = []

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def home(self) -> None:
        self._state = np.zeros(self._schema.action_dim, dtype=np.float32)

    def read_observation(self) -> dict[str, Any]:
        return build_robot_observation(self._state, self._frames, schema=self._schema)

    def _send_raw(self, action: Sequence[float]) -> None:
        self.sent_actions.append([float(a) for a in action])
        self._state = np.asarray(action, dtype=np.float32).reshape(-1)


class SO101Robot(RobotInterface):
    """Real SO-101 follower via LeRobot. Connect is gated on safety readiness.

    The LeRobot calls (lazy import) are verified on the lab machine; lines marked
    ``VERIFY(lab)`` may need adjusting to the installed LeRobot's API.
    """

    def __init__(
        self,
        safety: SafetyLayer,
        schema: ActionSchema,
        *,
        follower_port: str,
        cameras: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(safety, schema)
        self._follower_port = follower_port
        self._cameras = dict(cameras or {})
        self._robot: Any = None

    @property
    def is_connected(self) -> bool:
        return self._robot is not None and bool(getattr(self._robot, "is_connected", True))

    def connect(self) -> None:
        problems = self._safety.hardware_readiness_problems()
        if problems:
            raise RobotNotReadyError(
                "refusing to connect to the SO-101: safety not ready -> "
                + "; ".join(problems)
                + " (fill configs/safety/default_limits.yaml and review)"
            )
        # VERIFY(lab): LeRobot SO-101 follower API (lerobot >= 0.5).
        from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

        config = SO101FollowerConfig(port=self._follower_port, cameras=self._cameras)
        self._robot = SO101Follower(config)
        self._robot.connect()

    def disconnect(self) -> None:
        if self._robot is not None:
            self._robot.disconnect()
            self._robot = None

    def home(self) -> None:
        # The home/rest pose is rig-specific; set it on the lab machine.
        raise NotImplementedError("SO101Robot.home is configured on the lab machine")

    def read_observation(self) -> dict[str, Any]:
        if self._robot is None:
            raise RobotNotReadyError("read_observation before connect()")
        raw = self._robot.get_observation()  # VERIFY(lab): observation key format
        state = [float(raw[f"{joint}.pos"]) for joint in self._schema.joint_names]
        frames = {f"observation.images.{name}": raw[name] for name in self._cameras}
        return build_robot_observation(state, frames, schema=self._schema)

    def _send_raw(self, action: Sequence[float]) -> None:
        if self._robot is None:
            raise RobotNotReadyError("send_action before connect()")
        # VERIFY(lab): LeRobot expects a {motor.pos: value} action dict.
        motor = action_to_motor_dict(action, self._schema)
        command = {f"{name}.pos": value for name, value in motor.items()}
        self._robot.send_action(command)
