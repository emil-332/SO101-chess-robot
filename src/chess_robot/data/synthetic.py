"""Synthetic data + end-to-end offline pipeline verification (Tier A).

Generates schema-correct **mock** data (no hardware, no LeRobot, no cloud) and
runs it through every offline stage of the pipeline to verify the plumbing works:
perception -> move resolution (incl. captures) -> observation preprocessing ->
mock policy -> safety -> rollout logging -> evaluation, plus the dataset feature
schema and the pi0.5 train command.

This proves the code *runs end to end*; it does **not** prove the policy *learns*
(random frames/actions teach nothing — that needs real teleoperation data). See
the Tier A/B/C discussion in the project notes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from chess_robot.chess.board_mapper import GroundedGrid
from chess_robot.chess.board_state import BoardState, Square
from chess_robot.chess.move_resolver import MoveResolver, OffBoardLocation, SubmoveRole
from chess_robot.data.lerobot_dataset import (
    ChessDemoRecorder,
    DatasetConfig,
    PreprocessedObservation,
    build_observation,
    dataset_features,
    load_dataset_config,
    plan_move,
    to_lerobot_features,
)
from chess_robot.data.schema import FailureType
from chess_robot.data.validation import validate_episode_record
from chess_robot.eval.evaluator import evaluate_rollout_file
from chess_robot.eval.perception_metrics import PerceptionSample, evaluate_perception
from chess_robot.perception.board_perception import (
    OCCUPANCY_SOURCE_METADATA,
    ComposedBoardPerception,
    MetadataBoardPerception,
)
from chess_robot.perception.piece_locator import MetadataPieceClassifier
from chess_robot.perception.square_grounding import (
    BoardCorners,
    CornerSquareGrounder,
    FixedCornerDetector,
    grid_from_corners,
)
from chess_robot.policies.pi05_policy import (
    MockPi05Policy,
    build_train_command,
    load_pi05_train_config,
    unfilled_placeholders,
)
from chess_robot.safety.limits import load_limits
from chess_robot.safety.safety_layer import SafetyLayer
from chess_robot.utils.logging import RolloutLogger

# Moves on the standard start position: two non-captures + one capture (a8 held).
_DEMO_INSTRUCTIONS = (
    "move knight from b1 to c3",
    "move pawn from e2 to e4",
    "move rook from a1 to a8",
)


def synthetic_frames(config: DatasetConfig) -> dict[str, np.ndarray]:
    """A blank frame per configured camera, of the configured shape."""
    frame = np.zeros(
        (config.image_height, config.image_width, config.image_channels), dtype=np.uint8
    )
    return {camera: frame.copy() for camera in config.cameras}


def calibration_corners(config: DatasetConfig) -> BoardCorners:
    """Frame-spanning calibration corners (a convex quad inside the image)."""
    margin = 10.0
    width = float(config.image_width)
    height = float(config.image_height)
    return BoardCorners(
        a1=(margin, margin),
        h1=(width - margin, margin),
        h8=(width - margin, height - margin),
        a8=(margin, height - margin),
    )


def generate_demo_observations(
    config: DatasetConfig,
    *,
    frames: dict[str, np.ndarray],
    grid: GroundedGrid,
    resolver: MoveResolver,
) -> list[PreprocessedObservation]:
    """Preprocessed observations for the demo moves (capture yields 2 submoves)."""
    board = BoardState.standard_starting_position()
    perception = MetadataBoardPerception(board, grid)
    observations: list[PreprocessedObservation] = []
    for instruction in _DEMO_INSTRUCTIONS:
        plan = plan_move(frames, instruction, perception=perception, resolver=resolver)
        for index in range(len(plan.resolved.submoves)):
            observations.append(build_observation(frames, plan, index))
    return observations


def write_synthetic_rollouts(
    output_dir: str | Path, *, filename: str = "synthetic_rollouts.jsonl"
) -> Path:
    """Write a small, varied set of synthetic rollouts via the RolloutLogger."""
    logger = RolloutLogger(output_dir, filename=filename)

    logger.start_episode(0, "move knight from b1 to c3")
    logger.log_step(submove_index=0, submove_role=SubmoveRole.MOVE, final_action=[0.0], reward=1.0)
    logger.end_episode(success_label=True)

    logger.start_episode(1, "move pawn from e2 to e4")
    logger.log_step(submove_index=0, submove_role=SubmoveRole.MOVE)
    logger.end_episode(success_label=False, failure_type=FailureType.WRONG_SQUARE)

    logger.start_episode(2, "move rook from a1 to a8", is_capture=True)
    logger.log_step(submove_index=0, submove_role=SubmoveRole.REMOVE, residual_action=[0.1, 0.2])
    logger.log_step(submove_index=1, submove_role=SubmoveRole.PLACE, residual_action=[0.3, 0.4])
    logger.end_episode(success_label=True)

    logger.start_episode(3, "move bishop from c1 to f4")
    logger.log_step(
        submove_index=0,
        submove_role=SubmoveRole.MOVE,
        intervention_flag=True,
        safety_violation_flag=True,
    )
    logger.end_episode(success_label=False, failure_type=FailureType.BAD_GRASP)
    return logger.path


def synthetic_perception_samples(
    *, board: BoardState, grid: GroundedGrid, frames: dict[str, np.ndarray]
) -> list[PerceptionSample]:
    """Labeled perception samples (one held-out) sharing one known board."""
    return [
        PerceptionSample(
            ground_truth_board=board,
            frames=frames,
            ground_truth_grid=grid,
            board_type="synthetic_a",
            capture_targets=(Square("e", 4),),
        ),
        PerceptionSample(
            ground_truth_board=board,
            frames=frames,
            ground_truth_grid=grid,
            board_type="synthetic_b",
            held_out=True,
            capture_targets=(Square("e", 1),),
        ),
    ]


@dataclass
class StageResult:
    """Outcome of one verification stage."""

    name: str
    ok: bool
    detail: str


@dataclass
class PipelineVerification:
    """Aggregated result of the offline pipeline verification."""

    stages: list[StageResult]

    @property
    def ok(self) -> bool:
        return all(stage.ok for stage in self.stages)

    @property
    def num_passed(self) -> int:
        return sum(1 for stage in self.stages if stage.ok)


def _stage(name: str, run: Callable[[], str]) -> StageResult:
    try:
        return StageResult(name=name, ok=True, detail=run())
    except Exception as exc:  # diagnostic harness: record any failure, keep going
        return StageResult(name=name, ok=False, detail=f"{type(exc).__name__}: {exc}")


def _policy_observation(
    images: dict[str, np.ndarray], state_dim: int
) -> dict[str, np.ndarray]:
    observation = dict(images)
    observation["observation.state"] = np.zeros(state_dim, dtype=np.float32)
    return observation


def run_pipeline_verification(
    *,
    output_dir: str | Path,
    dataset_config_path: str | Path = "configs/dataset/collect_chess_demos.yaml",
    safety_config_path: str | Path = "configs/safety/default_limits.yaml",
    pi05_config_path: str | Path = "configs/policy/pi05.yaml",
) -> PipelineVerification:
    """Run mock data through every offline stage and report pass/fail per stage."""
    config = load_dataset_config(dataset_config_path)
    corners = calibration_corners(config)
    grid = grid_from_corners(corners)
    frames = synthetic_frames(config)
    start = BoardState.standard_starting_position()
    metadata_perception = MetadataBoardPerception(start, grid)
    resolver = MoveResolver(
        OffBoardLocation(config.off_board_location or "capture_tray")
    )
    mock_policy = MockPi05Policy(config.action_dim)
    safety = SafetyLayer(
        load_limits(safety_config_path), expected_action_dim=config.action_dim
    )
    output = Path(output_dir)
    stages: list[StageResult] = []

    def perception_metadata() -> str:
        result = metadata_perception.perceive(frames)
        assert result.board_state == start
        assert result.source == OCCUPANCY_SOURCE_METADATA
        return "metadata perception returns the supplied board"

    def perception_composed() -> str:
        composed = ComposedBoardPerception(
            CornerSquareGrounder(FixedCornerDetector(corners)),
            MetadataPieceClassifier(start),
            piece_grounder=CornerSquareGrounder(FixedCornerDetector(corners)),
        )
        out = composed.perceive(frames)
        assert out.grids is not None
        assert config.cameras[0] in out.grids
        return f"composed perception grounded {len(out.grids)} cameras"

    def move_resolution() -> str:
        plan = plan_move(
            frames,
            "move rook from a1 to a8",
            perception=metadata_perception,
            resolver=resolver,
        )
        assert plan.resolved.is_capture
        roles = [submove.role for submove in plan.resolved.submoves]
        assert roles == [SubmoveRole.REMOVE, SubmoveRole.PLACE]
        return "capture resolves to ordered remove + place"

    def preprocess_and_validate() -> str:
        observations = generate_demo_observations(
            config, frames=frames, grid=grid, resolver=resolver
        )
        problems: list[str] = []
        for observation in observations:
            problems += validate_episode_record(observation.record)
        assert problems == [], f"record validation problems: {problems}"
        assert any(
            o.record.submove_role is SubmoveRole.PLACE for o in observations
        ), "capture PLACE submove missing"
        return f"{len(observations)} preprocessed observations, all schema-valid"

    def dataset_structure() -> str:
        features = dataset_features(config)
        assert "action" in features and "observation.state" in features
        assert to_lerobot_features(config)
        report = ChessDemoRecorder(config, metadata_perception, resolver).dry_run(
            frames, "move knight from b1 to c3"
        )
        assert report.ok
        return f"{len(features)} tensor columns; dry-run structure valid"

    def policy_action() -> str:
        observation = _policy_observation(frames, config.state_dim)
        action = mock_policy.select_action(observation, "move knight from b1 to c3")
        assert action.shape == (config.action_dim,)
        return f"mock policy returns an action of dim {config.action_dim}"

    def safety_check() -> str:
        safe = [0.0] * config.action_dim
        assert safety.check_action(safe).ok
        bad = list(safe)
        bad[0] = float("nan")
        assert not safety.check_action(bad).ok
        return f"safe action passes, NaN blocked, hardware_ready={safety.is_hardware_ready()}"

    def end_to_end_episode() -> str:
        logger = RolloutLogger(output, filename="verify_e2e.jsonl")
        plan = plan_move(
            frames,
            "move rook from a1 to a8",
            perception=metadata_perception,
            resolver=resolver,
        )
        logger.start_episode(0, plan.instruction, is_capture=plan.resolved.is_capture)
        for index, submove in enumerate(plan.resolved.submoves):
            pre = build_observation(frames, plan, index)
            observation = _policy_observation(pre.images, config.state_dim)
            action = mock_policy.select_action(observation, plan.instruction)
            result = safety.check_action(action.tolist())
            logger.log_step(
                submove_index=submove.index,
                submove_role=submove.role,
                final_action=action.tolist(),
                safety_violation_flag=not result.ok,
                board_state=pre.record.board_state,
            )
        logger.end_episode(success_label=True)
        report = evaluate_rollout_file(logger.path)
        assert report.num_episodes == 1
        return "full capture episode: perceive->resolve->preprocess->policy->safety->log->eval"

    def rollout_eval() -> str:
        path = write_synthetic_rollouts(output, filename="verify_rollouts.jsonl")
        report = evaluate_rollout_file(path)
        assert report.num_episodes >= 1
        assert report.success_rate is not None
        return f"{report.num_episodes} rollouts evaluated (success={report.success_rate})"

    def perception_eval() -> str:
        samples = synthetic_perception_samples(board=start, grid=grid, frames=frames)
        report = evaluate_perception(MetadataBoardPerception(start, grid), samples)
        assert report.occupancy_accuracy == 1.0
        return f"{report.num_samples} samples, occupancy={report.occupancy_accuracy}"

    def train_command() -> str:
        train_config = load_pi05_train_config(pi05_config_path)
        command = build_train_command(train_config)
        assert command[0] == "lerobot-train"
        missing = unfilled_placeholders(train_config)
        return f"train command built ({len(command)} args); unfilled placeholders: {missing}"

    for name, run in (
        ("perception_metadata", perception_metadata),
        ("perception_composed_grids", perception_composed),
        ("move_resolution_capture", move_resolution),
        ("preprocess_and_validate", preprocess_and_validate),
        ("dataset_structure", dataset_structure),
        ("policy_action", policy_action),
        ("safety_check", safety_check),
        ("end_to_end_episode", end_to_end_episode),
        ("rollout_logging_and_eval", rollout_eval),
        ("perception_eval", perception_eval),
        ("train_command", train_command),
    ):
        stages.append(_stage(name, run))

    return PipelineVerification(stages=stages)
