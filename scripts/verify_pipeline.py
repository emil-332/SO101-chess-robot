"""Verify the offline pipeline end-to-end with synthetic data (Tier A).

Runs mock data through every offline stage (perception -> resolve -> preprocess ->
mock policy -> safety -> rollout logging -> evaluation, plus dataset schema and the
pi0.5 train command) and prints a per-stage PASS/FAIL table. No hardware, no
LeRobot, no cloud. Exits non-zero if any stage fails.

    python scripts/verify_pipeline.py
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from chess_robot.data.synthetic import run_pipeline_verification


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the offline pipeline.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=Path("configs/dataset/collect_chess_demos.yaml"),
    )
    parser.add_argument(
        "--safety-config",
        type=Path,
        default=Path("configs/safety/default_limits.yaml"),
    )
    parser.add_argument(
        "--pi05-config", type=Path, default=Path("configs/policy/pi05.yaml")
    )
    args = parser.parse_args()

    output_dir = args.output_dir or Path(tempfile.mkdtemp(prefix="verify_pipeline_"))
    report = run_pipeline_verification(
        output_dir=output_dir,
        dataset_config_path=args.dataset_config,
        safety_config_path=args.safety_config,
        pi05_config_path=args.pi05_config,
    )

    print("Pipeline verification (synthetic data, no hardware):")
    for stage in report.stages:
        status = "PASS" if stage.ok else "FAIL"
        print(f"  [{status}] {stage.name}: {stage.detail}")
    print(f"{report.num_passed}/{len(report.stages)} stages passed")

    if not report.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
