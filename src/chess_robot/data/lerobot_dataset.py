"""LeRobotDataset collection wrapper with perception preprocessing.

Records teleoperated chess demonstrations into a LeRobotDataset, applying the
**same perception preprocessing at record time as at inference time** so the
stored observations match what the policy will later see: the relevant squares
are highlighted on the frame, and the structured move metadata (board state,
capture/submove fields) is stored alongside.

LeRobot is an optional/lazy dependency (installed via the ``train`` extra on the
recording machine). This module never imports it at top level: the config,
feature schema, preprocessing, and :meth:`ChessDemoRecorder.dry_run` all run
without LeRobot or a robot.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from chess_robot.chess.board_mapper import GroundedGrid
from chess_robot.chess.command_parser import parse_command
from chess_robot.chess.move_resolver import MoveResolver, ResolvedMove
from chess_robot.data.schema import EpisodeRecord
from chess_robot.data.validation import validate_episode_record
from chess_robot.perception.board_perception import (
    DEFAULT_GROUNDING_CAMERA,
    BoardPerception,
    PerceivedBoard,
)
from chess_robot.perception.camera_utils import highlight_squares

logger = logging.getLogger(__name__)

# Per-frame metadata columns stored alongside the tensors (instruction is the
# LeRobot "task"). Derived from EpisodeRecord so the two cannot drift.
METADATA_FIELDS: tuple[str, ...] = tuple(
    f.name for f in dataclasses.fields(EpisodeRecord) if f.name != "instruction"
)


@dataclass(frozen=True)
class FeatureSpec:
    """A LeRobot-agnostic description of one dataset column."""

    dtype: str
    shape: tuple[int, ...]
    names: tuple[str, ...] | None = None


@dataclass(frozen=True)
class DatasetConfig:
    """Parsed dataset-collection config (see configs/dataset/collect_chess_demos.yaml)."""

    repo_id: str
    root: str
    fps: int
    stage: int
    successful_demos_only: bool
    occupancy_source: str
    cameras: tuple[str, ...]
    image_height: int
    image_width: int
    image_channels: int
    state_dim: int
    action_dim: int
    highlight_camera: str
    off_board_location: str


def load_dataset_config(path: str | Path) -> DatasetConfig:
    """Load a :class:`DatasetConfig` from a YAML config file."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    section = raw.get("dataset", {}) if isinstance(raw, Mapping) else {}
    features = section.get("features")
    features = features if isinstance(features, Mapping) else {}
    image = features.get("image")
    image = image if isinstance(image, Mapping) else {}
    cameras = section.get("cameras") or []
    return DatasetConfig(
        repo_id=str(section.get("repo_id", "")),
        root=str(section.get("root", "")),
        fps=int(section.get("fps", 30)),
        stage=int(section.get("stage", 1)),
        successful_demos_only=bool(section.get("successful_demos_only", True)),
        occupancy_source=str(section.get("occupancy_source", "metadata")),
        cameras=tuple(str(camera) for camera in cameras),
        image_height=int(image.get("height", 480)),
        image_width=int(image.get("width", 640)),
        image_channels=int(image.get("channels", 3)),
        state_dim=int(features.get("state_dim", 6)),
        action_dim=int(features.get("action_dim", 6)),
        highlight_camera=str(
            section.get("highlight_camera", "observation.images.overhead")
        ),
        off_board_location=str(section.get("off_board_location", "")),
    )


def dataset_features(config: DatasetConfig) -> dict[str, FeatureSpec]:
    """The tensor columns of the dataset (cameras + state + action)."""
    features: dict[str, FeatureSpec] = {}
    for camera in config.cameras:
        features[camera] = FeatureSpec(
            dtype="video",
            shape=(config.image_height, config.image_width, config.image_channels),
            names=("height", "width", "channel"),
        )
    features["observation.state"] = FeatureSpec("float32", (config.state_dim,), ("state",))
    features["action"] = FeatureSpec("float32", (config.action_dim,), ("action",))
    return features


def to_lerobot_features(config: DatasetConfig) -> dict[str, dict[str, object]]:
    """Translate :func:`dataset_features` to LeRobot's ``features`` dict format."""
    return {
        name: {
            "dtype": spec.dtype,
            "shape": list(spec.shape),
            "names": list(spec.names) if spec.names is not None else None,
        }
        for name, spec in dataset_features(config).items()
    }


@dataclass
class PreprocessedObservation:
    """A recording-/inference-time observation after perception preprocessing."""

    images: dict[str, np.ndarray]
    instruction: str
    record: EpisodeRecord


@dataclass
class MovePlan:
    """A move resolved **once**, reused to build each submove's observation.

    Re-resolving per submove is wrong: after the REMOVE submove the target square
    is no longer occupied, so the resolver would no longer see a capture (and a
    PLACE submove index would be out of range). Resolve once while the target is
    still occupied, then build each submove from this fixed plan.
    """

    instruction: str
    perceived: PerceivedBoard
    resolved: ResolvedMove


def plan_move(
    frames: Mapping[str, np.ndarray],
    instruction: str,
    *,
    perception: BoardPerception,
    resolver: MoveResolver,
) -> MovePlan:
    """Perceive + resolve the move ONCE (call at the start of the move)."""
    parsed = parse_command(instruction)
    perceived = perception.perceive(frames)
    resolved = resolver.resolve(parsed, perceived.board_state)
    return MovePlan(instruction=instruction, perceived=perceived, resolved=resolved)


def _camera_grids(perceived: PerceivedBoard) -> dict[str, GroundedGrid]:
    if perceived.grids:
        return dict(perceived.grids)
    return {DEFAULT_GROUNDING_CAMERA: perceived.grid}


def build_observation(
    frames: Mapping[str, np.ndarray],
    plan: MovePlan,
    submove_index: int = 0,
    *,
    highlight_color: tuple[int, int, int] = (255, 0, 0),
) -> PreprocessedObservation:
    """Build one submove's preprocessed observation from a fixed :class:`MovePlan`.

    Highlights the submove's squares on **every grounded camera** (using each
    camera's own grid) and builds the structured :class:`EpisodeRecord`. Warns if
    highlighting was expected but no camera is grounded (so the no-grid case is
    never silent).
    """
    resolved = plan.resolved
    if not 0 <= submove_index < len(resolved.submoves):
        raise IndexError(
            f"submove_index {submove_index} out of range "
            f"(move has {len(resolved.submoves)} submoves)"
        )
    submove = resolved.submoves[submove_index]

    camera_grids = _camera_grids(plan.perceived)
    images = dict(frames)
    highlighted_any = False
    for camera, frame in list(images.items()):
        grid = camera_grids.get(camera)
        if grid is None or not grid.regions:
            continue
        images[camera] = highlight_squares(
            frame, grid, submove.highlighted_squares, color=highlight_color
        )
        highlighted_any = True
    if submove.highlighted_squares and not highlighted_any:
        logger.warning(
            "observation highlighting skipped: no grounded grid for any camera "
            "(square markers not drawn) - supply a grounded grid/perception"
        )

    record = EpisodeRecord(
        instruction=plan.instruction,
        piece_type=resolved.piece_type,
        start_square=resolved.start_square.name,
        target_square=resolved.target_square.name,
        is_capture=resolved.is_capture,
        submove_index=submove.index,
        submove_role=submove.role,
        captured_piece_type=resolved.captured_piece_type,
        board_state=plan.perceived.board_state,
    )
    return PreprocessedObservation(
        images=images, instruction=plan.instruction, record=record
    )


def preprocess_observation(
    frames: Mapping[str, np.ndarray],
    instruction: str,
    *,
    perception: BoardPerception,
    resolver: MoveResolver,
    submove_index: int = 0,
    highlight_color: tuple[int, int, int] = (255, 0, 0),
) -> PreprocessedObservation:
    """Resolve + build one (sub)move observation in a single call.

    Convenience for the single-segment / first-submove case. For a multi-submove
    **capture**, call :func:`plan_move` once and then :func:`build_observation`
    per submove — do **not** re-resolve mid-sequence (see :class:`MovePlan`).
    """
    plan = plan_move(frames, instruction, perception=perception, resolver=resolver)
    return build_observation(
        frames, plan, submove_index, highlight_color=highlight_color
    )


@dataclass
class DryRunReport:
    """Result of a no-hardware, no-LeRobot dataset structure check."""

    features: dict[str, FeatureSpec]
    lerobot_features: dict[str, dict[str, object]]
    metadata_fields: tuple[str, ...]
    sample_record_problems: list[str]
    num_episodes: int = 0

    @property
    def ok(self) -> bool:
        return not self.sample_record_problems


class ChessDemoRecorder:
    """Wrap perception preprocessing + (eventually) LeRobot recording."""

    def __init__(
        self,
        config: DatasetConfig,
        perception: BoardPerception,
        resolver: MoveResolver,
    ) -> None:
        self._config = config
        self._perception = perception
        self._resolver = resolver

    @property
    def config(self) -> DatasetConfig:
        return self._config

    def dry_run(
        self,
        frames: Mapping[str, np.ndarray],
        instruction: str,
        *,
        submove_index: int = 0,
    ) -> DryRunReport:
        """Validate the dataset structure + one preprocessed observation offline.

        Builds the feature schema and a sample preprocessed record (no LeRobot, no
        robot, no files) and validates the record against the dataset schema.
        """
        sample = preprocess_observation(
            frames,
            instruction,
            perception=self._perception,
            resolver=self._resolver,
            submove_index=submove_index,
        )
        return DryRunReport(
            features=dataset_features(self._config),
            lerobot_features=to_lerobot_features(self._config),
            metadata_fields=METADATA_FIELDS,
            sample_record_problems=validate_episode_record(sample.record),
            num_episodes=0,
        )

    def record_episode(self) -> None:
        """Record one teleoperated episode into a LeRobotDataset.

        Wired later (requires a pinned LeRobot + a connected SO-101).
        """
        raise NotImplementedError(
            "Real LeRobot recording is wired later. Use dry_run() for the offline check."
        )