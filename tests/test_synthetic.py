"""Tests for synthetic data generation + offline pipeline verification"""

from pathlib import Path

from chess_robot.chess.board_state import BoardState
from chess_robot.chess.move_resolver import MoveResolver, OffBoardLocation, SubmoveRole
from chess_robot.data.lerobot_dataset import load_dataset_config
from chess_robot.data.synthetic import (
    calibration_corners,
    generate_demo_observations,
    run_pipeline_verification,
    synthetic_frames,
    write_synthetic_rollouts,
)
from chess_robot.data.validation import validate_episode_record
from chess_robot.eval.evaluator import evaluate_rollout_file
from chess_robot.perception.square_grounding import grid_from_corners

_DATASET_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "dataset"
    / "collect_chess_demos.yaml"
)
_SAFETY_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "safety" / "default_limits.yaml"
)
_PI05_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "policy" / "pi05.yaml"
)


def test_generate_demo_observations_are_schema_valid_with_capture() -> None:
    config = load_dataset_config(_DATASET_CONFIG)
    grid = grid_from_corners(calibration_corners(config))
    observations = generate_demo_observations(
        config,
        frames=synthetic_frames(config),
        grid=grid,
        resolver=MoveResolver(OffBoardLocation("tray")),
    )
    # two non-capture moves (1 submove each) + one capture (2 submoves) = 4
    assert len(observations) == 4
    for observation in observations:
        assert validate_episode_record(observation.record) == []
    roles = {o.record.submove_role for o in observations}
    assert SubmoveRole.REMOVE in roles and SubmoveRole.PLACE in roles


def test_write_synthetic_rollouts_can_be_evaluated(tmp_path: Path) -> None:
    path = write_synthetic_rollouts(tmp_path)
    report = evaluate_rollout_file(path)
    assert report.num_episodes == 4
    assert report.success_rate is not None
    assert report.num_capture_episodes == 1


def test_run_pipeline_verification_all_stages_pass(tmp_path: Path) -> None:
    report = run_pipeline_verification(
        output_dir=tmp_path,
        dataset_config_path=_DATASET_CONFIG,
        safety_config_path=_SAFETY_CONFIG,
        pi05_config_path=_PI05_CONFIG,
    )
    failed = [stage.name for stage in report.stages if not stage.ok]
    assert report.ok, f"failed stages: {failed} -> {[s.detail for s in report.stages]}"
    assert report.num_passed == len(report.stages)
    assert len(report.stages) >= 10


def test_calibration_corners_inside_frame() -> None:
    config = load_dataset_config(_DATASET_CONFIG)
    corners = calibration_corners(config)
    # grounding the calibration corners produces a full 64-square grid
    regions = grid_from_corners(corners).regions
    assert regions is not None
    assert len(regions) == 64


def test_synthetic_frames_match_config_shape() -> None:
    config = load_dataset_config(_DATASET_CONFIG)
    frames = synthetic_frames(config)
    assert set(frames) == set(config.cameras)
    overhead = frames["observation.images.overhead"]
    assert overhead.shape == (
        config.image_height,
        config.image_width,
        config.image_channels,
    )
    # BoardState import kept meaningful: the demo board is the standard start
    assert len(BoardState.standard_starting_position()) == 32
