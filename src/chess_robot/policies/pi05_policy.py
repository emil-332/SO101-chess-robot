"""pi0.5 policy — supervised fine-tuning config + command builder (  3.1).

Fine-tuning runs on the cloud GPU (vast.ai; no local CUDA) **through LeRobot**.
Rather than depend on a specific LeRobot Python training API, we drive its
documented ``lerobot-train`` CLI: :func:`build_train_command` turns a
:class:`Pi05TrainConfig` into the command argv. The exact flag names are verified
against the pinned LeRobot during the cloud run (see ``docs/training_pi05.md``).

The pi0.5 **inference wrapper** (`Pi05Policy`, the laptop client to a remote
policy server) lands at   3.3.
"""

from __future__ import annotations

import json
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Normalization override for a freshly-created dataset without quantile stats
# (per the LeRobot pi0.5 guide: https://huggingface.co/docs/lerobot/pi05).
NORMALIZATION_MEAN_STD = '{"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}'


@dataclass(frozen=True)
class Pi05TrainConfig:
    """Parsed pi0.5 fine-tuning config (see configs/policy/pi05.yaml).

    Flag mapping follows the LeRobot pi0.5 guide (verified 2026-06-08).
    """

    policy_type: str
    pretrained: str  # --policy.pretrained_path (e.g. lerobot/pi05_base)
    dataset_repo_id: str
    dataset_root: str  # local dataset root ("" => omit, resolve from the Hub)
    output_dir: str
    job_name: str
    policy_repo_id: str  # --policy.repo_id (where a trained policy would be pushed)
    device: str
    dtype: str
    steps: int
    batch_size: int
    seed: int
    train_expert_only: bool
    gradient_checkpointing: bool
    compile_model: bool
    wandb_enable: bool
    normalization_mapping: str  # "" => omit
    push_to_hub: bool  # push the trained policy to the Hub (needs a real namespace)


def load_pi05_train_config(path: str | Path) -> Pi05TrainConfig:
    """Load a :class:`Pi05TrainConfig` from a YAML config file."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    policy = raw.get("policy", {}) if isinstance(raw, Mapping) else {}
    training = policy.get("training")
    training = training if isinstance(training, Mapping) else {}
    return Pi05TrainConfig(
        policy_type=str(policy.get("type", "pi05")),
        pretrained=str(policy.get("pretrained", "lerobot/pi05_base")),
        dataset_repo_id=str(policy.get("dataset_repo_id", "")),
        dataset_root=str(policy.get("dataset_root", "")),
        output_dir=str(policy.get("output_dir", "")),
        job_name=str(policy.get("job_name", "chess_pi05")),
        policy_repo_id=str(policy.get("policy_repo_id", "")),
        device=str(policy.get("device", "cuda")),
        dtype=str(policy.get("dtype", "bfloat16")),
        steps=int(training.get("steps", 3000)),
        batch_size=int(training.get("batch_size", 32)),
        seed=int(training.get("seed", 1000)),
        train_expert_only=bool(training.get("train_expert_only", False)),
        gradient_checkpointing=bool(training.get("gradient_checkpointing", True)),
        compile_model=bool(training.get("compile_model", True)),
        wandb_enable=bool(training.get("wandb_enable", True)),
        normalization_mapping=str(training.get("normalization_mapping", "")),
        push_to_hub=bool(policy.get("push_to_hub", False)),
    )


def _flag_bool(value: bool) -> str:
    return "true" if value else "false"


def build_train_command(config: Pi05TrainConfig) -> list[str]:
    """Build the ``lerobot-train`` argv for a pi0.5 fine-tuning run (cloud GPU).

    Flags follow the LeRobot pi0.5 guide. Runs on the cloud GPU.
    """
    command = ["lerobot-train", f"--dataset.repo_id={config.dataset_repo_id}"]
    if config.dataset_root:
        command.append(f"--dataset.root={config.dataset_root}")
    command += [
        f"--policy.type={config.policy_type}",
        f"--policy.pretrained_path={config.pretrained}",
        f"--policy.device={config.device}",
        f"--policy.dtype={config.dtype}",
        f"--policy.train_expert_only={_flag_bool(config.train_expert_only)}",
        f"--policy.gradient_checkpointing={_flag_bool(config.gradient_checkpointing)}",
        f"--policy.compile_model={_flag_bool(config.compile_model)}",
        f"--output_dir={config.output_dir}",
        f"--job_name={config.job_name}",
        f"--policy.repo_id={config.policy_repo_id}",
        f"--batch_size={config.batch_size}",
        f"--steps={config.steps}",
        f"--seed={config.seed}",
        f"--wandb.enable={_flag_bool(config.wandb_enable)}",
        f"--policy.push_to_hub={_flag_bool(config.push_to_hub)}",
    ]
    if config.normalization_mapping:
        command.append(f"--policy.normalization_mapping={config.normalization_mapping}")
    return command


def smoke_config(config: Pi05TrainConfig) -> Pi05TrainConfig:
    """A cheap ~2-step smoke profile: low memory, no wandb, fresh-dataset norm.

    Freezes the VLM (``train_expert_only``), tiny batch/steps, no compile, and the
    MEAN_STD normalization override so a freshly-created (un-quantiled) dataset
    trains. Used to confirm the LeRobot pi0.5 integration runs for ~$1 of GPU.
    """
    return replace(
        config,
        steps=2,
        batch_size=1,
        train_expert_only=True,
        gradient_checkpointing=True,
        compile_model=False,
        wandb_enable=False,
        push_to_hub=False,
        normalization_mapping=config.normalization_mapping or NORMALIZATION_MEAN_STD,
    )


def unfilled_placeholders(config: Pi05TrainConfig) -> list[str]:
    """Config fields still holding ``<PLACEHOLDER>`` values (must be filled)."""
    candidates = {
        "dataset_repo_id": config.dataset_repo_id,
        "output_dir": config.output_dir,
        "policy_repo_id": config.policy_repo_id,
    }
    return [name for name, value in candidates.items() if "<" in value and ">" in value]


# --- Inference (  3.3): laptop client returning a base action ---------

Observation = Mapping[str, Any]
STATE_KEY = "observation.state"


class Pi05Policy(ABC):
    """Inference interface: a preprocessed observation + instruction -> action."""

    @abstractmethod
    def select_action(self, observation: Observation, instruction: str) -> np.ndarray:
        raise NotImplementedError

    def reset(self) -> None:
        """Clear any per-episode internal state (e.g. action-chunk buffers)."""
        return None


class MockPi05Policy(Pi05Policy):
    """Deterministic stand-in returning a fixed action (tests / offline smoke)."""

    def __init__(self, action_dim: int, *, fill: float = 0.0) -> None:
        self._action_dim = action_dim
        self._fill = fill

    def select_action(self, observation: Observation, instruction: str) -> np.ndarray:
        return np.full(self._action_dim, self._fill, dtype=np.float32)


def _state_list(observation: Observation) -> list[float] | None:
    state = observation.get(STATE_KEY)
    if state is None:
        return None
    return [float(value) for value in np.asarray(state).reshape(-1)]


class RemotePi05Policy(Pi05Policy):
    """Laptop client to a remote pi0.5 policy server (JSON over HTTP).

    pi0.5 is served on the cloud GPU; this client sends the instruction and
    observation and receives the base action. Image transport is pluggable via
    ``image_encoder`` (default: images are not sent — set an encoder, e.g. base64
    PNG, matching the deployed server). The exact server contract is finalized
    when the policy server is stood up (  3.2/3.3, cloud).
    """

    def __init__(
        self,
        url: str,
        action_dim: int,
        *,
        timeout: float = 5.0,
        image_encoder: Callable[[np.ndarray], Any] | None = None,
    ) -> None:
        self._url = url
        self._action_dim = action_dim
        self._timeout = timeout
        self._image_encoder = image_encoder

    def build_payload(self, observation: Observation, instruction: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "instruction": instruction,
            "state": _state_list(observation),
        }
        if self._image_encoder is not None:
            payload["images"] = {
                key: self._image_encoder(np.asarray(value))
                for key, value in observation.items()
                if key.startswith("observation.images.")
            }
        return payload

    def parse_action(self, body: Mapping[str, Any]) -> np.ndarray:
        action = np.asarray(body["action"], dtype=np.float32)
        if action.shape != (self._action_dim,):
            raise ValueError(
                f"server returned action shape {action.shape}, "
                f"expected ({self._action_dim},)"
            )
        return action

    def select_action(self, observation: Observation, instruction: str) -> np.ndarray:
        data = json.dumps(self.build_payload(observation, instruction)).encode("utf-8")
        request = urllib.request.Request(
            self._url, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=self._timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        return self.parse_action(body)
