"""Tests for the pi0.5 fine-tuning config + command builder"""

from dataclasses import replace
from pathlib import Path

from chess_robot.policies.pi05_policy import (
    build_train_command,
    load_pi05_train_config,
    smoke_config,
    unfilled_placeholders,
)

_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "policy" / "pi05.yaml"
)


def test_config_loads_training_block() -> None:
    config = load_pi05_train_config(_CONFIG_PATH)
    assert config.policy_type == "pi05"
    assert config.pretrained == "lerobot/pi05_base"
    assert config.device == "cuda"
    assert config.dtype == "bfloat16"
    assert config.steps == 3000
    assert config.batch_size == 32
    assert config.seed == 1000
    assert config.train_expert_only is False


def test_build_train_command_includes_verified_flags() -> None:
    config = load_pi05_train_config(_CONFIG_PATH)
    command = build_train_command(config)
    assert command[0] == "lerobot-train"
    joined = " ".join(command)
    assert "--policy.type=pi05" in joined
    assert "--policy.pretrained_path=lerobot/pi05_base" in joined
    assert "--policy.device=cuda" in joined
    assert "--steps=3000" in joined
    assert "--batch_size=32" in joined
    assert "--policy.train_expert_only=false" in joined
    assert "--wandb.enable=true" in joined
    assert f"--dataset.repo_id={config.dataset_repo_id}" in joined
    assert f"--output_dir={config.output_dir}" in joined


def test_smoke_config_is_cheap_and_low_memory() -> None:
    config = smoke_config(load_pi05_train_config(_CONFIG_PATH))
    assert config.steps == 2
    assert config.batch_size == 1
    assert config.train_expert_only is True
    assert config.compile_model is False
    assert config.wandb_enable is False
    assert config.push_to_hub is False
    assert config.normalization_mapping  # fresh-dataset normalization override set
    joined = " ".join(build_train_command(config))
    assert "--policy.train_expert_only=true" in joined
    assert "--wandb.enable=false" in joined
    assert "--policy.push_to_hub=false" in joined
    assert "--policy.normalization_mapping=" in joined


def test_dataset_root_flag_only_when_set() -> None:
    config = load_pi05_train_config(_CONFIG_PATH)
    assert "--dataset.root=" not in " ".join(build_train_command(config))
    with_root = replace(config, dataset_root="/data/smoke")
    assert "--dataset.root=/data/smoke" in " ".join(build_train_command(with_root))


def test_unfilled_placeholders_detected() -> None:
    missing = unfilled_placeholders(load_pi05_train_config(_CONFIG_PATH))
    # pretrained is a concrete model id now; the repo/output placeholders remain
    assert set(missing) == {"dataset_repo_id", "output_dir", "policy_repo_id"}


def test_filled_config_has_no_placeholders() -> None:
    config = replace(
        load_pi05_train_config(_CONFIG_PATH),
        dataset_repo_id="local/chess_smoke",
        output_dir="./outputs/smoke",
        policy_repo_id="local/chess_pi05_smoke",
    )
    assert unfilled_placeholders(config) == []
