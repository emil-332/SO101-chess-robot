"""Evaluate a policy and report metrics

Reads a rollouts JSONL file (written by ``utils.logging.RolloutLogger``), computes
the manipulation/capture/residual metrics, and prints a report. With ``--baseline``
it also prints the baseline report and the success-rate improvement over it.

    python scripts/evaluate_policy.py --rollouts runs/residual.jsonl \
        --baseline runs/base.jsonl
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from chess_robot.eval.evaluator import evaluate_rollout_file, load_eval_config
from chess_robot.eval.metrics import MetricReport, improvement_over_base


def _print_report(title: str, report: MetricReport) -> None:
    print(f"== {title} ==")
    for key, value in asdict(report).items():
        print(f"  {key}: {'n/a' if value is None else value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a policy from rollouts.")
    parser.add_argument("--rollouts", required=True, type=Path)
    parser.add_argument(
        "--baseline", type=Path, default=None, help="baseline rollouts to compare"
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/eval/chess_eval.yaml")
    )
    args = parser.parse_args()

    if args.config.exists():
        config = load_eval_config(args.config)
        print(
            f"eval config: occupancy_source={config.occupancy_source}, "
            f"compare={list(config.compare)}"
        )

    report = evaluate_rollout_file(args.rollouts)
    _print_report(args.rollouts.name, report)

    if args.baseline is not None:
        base = evaluate_rollout_file(args.baseline)
        _print_report(f"{args.baseline.name} (baseline)", base)
        delta = improvement_over_base(base, report)
        print(f"improvement_over_base (success_rate): {'n/a' if delta is None else delta}")


if __name__ == "__main__":
    main()
