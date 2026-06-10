"""Tests for the LeRobotDataset collection wrapper"""

from pathlib import Path

import numpy as np
import pytest

from chess_robot.chess.board_mapper import BoardPoint
from chess_robot.chess.board_state import BoardState, Square
from chess_robot.chess.move_resolver import MoveResolver, OffBoardLocation
from chess_robot.data.lerobot_dataset import (
    ChessDemoRecorder,
    DatasetConfig,
    build_observation,
    dataset_features,
    load_dataset_config,
    plan_move,
    preprocess_observation,
    to_lerobot_features,
)
from chess_robot.perception.board_perception import (
    OCCUPANCY_SOURCE_METADATA,
    MetadataBoardPerception,
)
from chess_robot.perception.square_grounding import BoardCorners, grid_from_corners

_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "dataset"
    / "collect_chess_demos.yaml"
)
_OVERHEAD = BoardCorners(a1=(0.0, 0.0), h1=(640.0, 0.0), h8=(640.0, 640.0), a8=(0.0, 640.0))


def _config() -> DatasetConfig:
    return load_dataset_config(_CONFIG_PATH)


def _perception() -> MetadataBoardPerception:
    # Grid present so highlighting actually draws.
    return MetadataBoardPerception(
        BoardState.standard_starting_position(), grid_from_corners(_OVERHEAD)
    )


def _frames(config: DatasetConfig) -> dict[str, np.ndarray]:
    frame = np.zeros(
        (config.image_height, config.image_width, config.image_channels), dtype=np.uint8
    )
    return {camera: frame.copy() for camera in config.cameras}


def test_config_loads_cameras_and_dims() -> None:
    config = _config()
    assert config.cameras == (
        "observation.images.overhead",
        "observation.images.side",
    )
    assert config.highlight_camera == "observation.images.overhead"
    assert config.image_channels == 3
    assert config.state_dim > 0
    assert config.action_dim > 0


def test_dataset_features_has_cameras_state_action() -> None:
    config = _config()
    features = dataset_features(config)
    for camera in config.cameras:
        assert features[camera].dtype == "video"
        assert features[camera].shape == (
            config.image_height,
            config.image_width,
            config.image_channels,
        )
    assert features["observation.state"].shape == (config.state_dim,)
    assert features["action"].shape == (config.action_dim,)


def test_to_lerobot_features_format() -> None:
    spec = to_lerobot_features(_config())
    action = spec["action"]
    assert action["dtype"] == "float32"
    assert isinstance(action["shape"], list)


def test_preprocess_observation_highlights_and_builds_record() -> None:
    config = _config()
    resolver = MoveResolver(OffBoardLocation("tray", BoardPoint(0.0, 0.0)))
    frames = _frames(config)

    result = preprocess_observation(
        frames,
        "move knight from b1 to c3",
        perception=_perception(),
        resolver=resolver,
    )

    # record reflects the move
    assert result.record.piece_type.value == "knight"
    assert result.record.start_square == "b1"
    assert result.record.target_square == "c3"
    assert result.record.is_capture is False
    assert result.record.submove_index == 0
    assert result.record.board_state is not None
    assert result.record.board_state.is_occupied(Square("b", 1))

    # overhead frame was highlighted (changed); side frame untouched
    assert not np.array_equal(
        result.images["observation.images.overhead"],
        frames["observation.images.overhead"],
    )
    assert np.array_equal(
        result.images["observation.images.side"],
        frames["observation.images.side"],
    )


def test_capture_builds_both_submoves_from_one_plan() -> None:
    # a1 = white rook, a8 = black rook on the start position -> "Rxa8" is a capture.
    # Resolve ONCE; building submove 1 (PLACE) must not re-resolve (which would
    # see the target empty after removal and raise IndexError).
    config = _config()
    perception = _perception()
    resolver = MoveResolver(OffBoardLocation("tray"))
    plan = plan_move(
        _frames(config),
        "move rook from a1 to a8",
        perception=perception,
        resolver=resolver,
    )
    assert plan.resolved.is_capture is True
    assert len(plan.resolved.submoves) == 2

    remove = build_observation(_frames(config), plan, 0)
    place = build_observation(_frames(config), plan, 1)
    assert remove.record.submove_role.value == "remove"
    assert place.record.submove_role.value == "place"
    assert place.record.is_capture is True


def test_preprocess_observation_rejects_bad_submove_index() -> None:
    config = _config()
    resolver = MoveResolver(OffBoardLocation("tray"))
    with pytest.raises(IndexError):
        preprocess_observation(
            _frames(config),
            "move knight from b1 to c3",
            perception=_perception(),
            resolver=resolver,
            submove_index=5,
        )


def test_dry_run_reports_valid_empty_structure() -> None:
    config = _config()
    recorder = ChessDemoRecorder(
        config, _perception(), MoveResolver(OffBoardLocation("tray"))
    )
    report = recorder.dry_run(_frames(config), "move knight from b1 to c3")
    assert report.ok
    assert report.sample_record_problems == []
    assert report.num_episodes == 0
    assert "action" in report.features
    assert "board_state" in report.metadata_fields
    assert "is_capture" in report.metadata_fields


def test_metadata_perception_source_is_recorded() -> None:
    result = _perception().perceive({})
    assert result.source == OCCUPANCY_SOURCE_METADATA


def test_record_episode_deferred_to_2_2() -> None:
    recorder = ChessDemoRecorder(
        _config(), _perception(), MoveResolver(OffBoardLocation("tray"))
    )
    with pytest.raises(NotImplementedError):
        recorder.record_episode()
