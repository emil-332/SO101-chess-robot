"""Run a policy on the SO-101 (laptop client to a remote policy server).

offline smoke test of the inference client. With no ``--server-url``
it uses a MockPi05Policy and a mock observation (blank frames + zero state on the
standard start), runs the full preprocessing chain (parse -> perceive -> resolve
-> highlight), gets a base action, and passes it through the safety layer. With
``--server-url`` it queries a remote pi0.5 server instead. Real-robot execution
(reading live observations, sending actions to the SO-101) is added later.

    python scripts/run_policy.py --instruction "move knight from b1 to c3"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from chess_robot.chess.board_state import BoardState
from chess_robot.chess.move_resolver import MoveResolver, OffBoardLocation
from chess_robot.data.lerobot_dataset import load_dataset_config, preprocess_observation
from chess_robot.perception.board_perception import MetadataBoardPerception
from chess_robot.policies.pi05_policy import MockPi05Policy, Pi05Policy, RemotePi05Policy
from chess_robot.safety.limits import load_limits
from chess_robot.safety.safety_layer import SafetyLayer


def _build_policy(server_url: str | None, action_dim: int) -> Pi05Policy:
    if server_url:
        return RemotePi05Policy(server_url, action_dim)
    return MockPi05Policy(action_dim)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a pi0.5 policy (client).")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/dataset/collect_chess_demos.yaml")
    )
    parser.add_argument(
        "--safety-config",
        type=Path,
        default=Path("configs/safety/default_limits.yaml"),
    )
    parser.add_argument("--instruction", default="move knight from b1 to c3")
    parser.add_argument(
        "--server-url", default=None, help="remote pi0.5 server (default: mock policy)"
    )
    args = parser.parse_args()

    config = load_dataset_config(args.config)
    frames = {
        camera: np.zeros(
            (config.image_height, config.image_width, config.image_channels),
            dtype=np.uint8,
        )
        for camera in config.cameras
    }
    perception = MetadataBoardPerception(BoardState.standard_starting_position())
    resolver = MoveResolver(OffBoardLocation(config.off_board_location or "capture_tray"))
    preprocessed = preprocess_observation(
        frames, args.instruction, perception=perception, resolver=resolver
    )
    observation = dict(preprocessed.images)
    observation["observation.state"] = np.zeros(config.state_dim, dtype=np.float32)

    policy = _build_policy(args.server_url, config.action_dim)
    action = policy.select_action(observation, args.instruction)

    safety = SafetyLayer(
        load_limits(args.safety_config), expected_action_dim=config.action_dim
    )
    result = safety.check_action(action.tolist())

    print(f"policy:        {type(policy).__name__}")
    print(f"instruction:   {args.instruction}")
    print(f"base action:   {action.tolist()}")
    print(f"safety ok:     {result.ok}")
    if result.violations:
        print(f"  violations:  {list(result.violations)}")
    print(f"hardware ready: {safety.is_hardware_ready()}")


if __name__ == "__main__":
    main()
