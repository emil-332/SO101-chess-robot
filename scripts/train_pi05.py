"""Fine-tune pi0.5 via LeRobot (cloud GPU).

Builds and runs the ``lerobot-train`` command from ``configs/policy/pi05.yaml``
(flags verified against the LeRobot pi0.5 guide). ``--dry-run`` validates the
config and prints the command without LeRobot or a GPU (laptop-safe). ``--smoke``
applies a cheap ~2-step low-memory profile (Option A). The real run must execute
on the **cloud GPU** (vast.ai) with LeRobot installed (see docs/cloud_smoke_test.md).

    # laptop: validate the command
    python scripts/train_pi05.py --dry-run
    python scripts/train_pi05.py --smoke --dry-run
    # cloud GPU: ~$1 smoke on a local mock dataset, then the full run
    python scripts/train_pi05.py --smoke \
        --dataset-repo-id local/chess_smoke --output-dir ./outputs/smoke \
        --policy-repo-id local/chess_pi05_smoke
"""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import replace
from pathlib import Path

from chess_robot.policies.pi05_policy import (
    build_train_command,
    load_pi05_train_config,
    smoke_config,
    unfilled_placeholders,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune pi0.5 via LeRobot.")
    parser.add_argument("--config", type=Path, default=Path("configs/policy/pi05.yaml"))
    parser.add_argument("--dry-run", action="store_true", help="validate + print only")
    parser.add_argument(
        "--smoke", action="store_true", help="cheap ~2-step low-memory smoke profile"
    )
    parser.add_argument("--steps", type=int, default=None, help="override steps")
    parser.add_argument("--dataset-repo-id", default=None, help="override dataset repo id")
    parser.add_argument("--dataset-root", default=None, help="override local dataset root")
    parser.add_argument("--output-dir", default=None, help="override output dir")
    parser.add_argument("--policy-repo-id", default=None, help="override policy repo id")
    args = parser.parse_args()

    config = load_pi05_train_config(args.config)
    if args.dataset_repo_id is not None:
        config = replace(config, dataset_repo_id=args.dataset_repo_id)
    if args.dataset_root is not None:
        config = replace(config, dataset_root=args.dataset_root)
    if args.output_dir is not None:
        config = replace(config, output_dir=args.output_dir)
    if args.policy_repo_id is not None:
        config = replace(config, policy_repo_id=args.policy_repo_id)
    if args.smoke:
        config = smoke_config(config)
    if args.steps is not None:
        config = replace(config, steps=args.steps)

    command = build_train_command(config)
    placeholders = unfilled_placeholders(config)

    if args.dry_run:
        print("dry-run: pi0.5 fine-tune" + (" [smoke]" if args.smoke else ""))
        print("  command:", " ".join(command))
        print(f"  dataset: {config.dataset_repo_id} (root: {config.dataset_root or '-'})")
        print(f"  output:  {config.output_dir}")
        print(f"  steps:   {config.steps}  batch: {config.batch_size}  dtype: {config.dtype}")
        if placeholders:
            print("  WARNING unfilled placeholders:", ", ".join(placeholders))
        return

    if placeholders:
        raise SystemExit(
            f"Fill config placeholders before training: {', '.join(placeholders)}"
        )
    try:
        raise SystemExit(subprocess.call(command))
    except FileNotFoundError as exc:
        raise SystemExit(
            "lerobot-train not found; run on the cloud GPU with LeRobot installed "
            "(see docs/cloud_smoke_test.md)."
        ) from exc


if __name__ == "__main__":
    main()
