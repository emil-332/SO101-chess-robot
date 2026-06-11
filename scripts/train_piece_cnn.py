"""Train the two-stage piece classifier (occupancy + piece) on the cloud GPU.

Reads ``configs/perception/piece_cnn.yaml`` and a prepared crop dataset
(``scripts/prepare_piece_dataset.py``), trains the requested stage(s), and writes
checkpoints + ONNX models. ``--dry-run`` validates the config/dataset and prints
the plan without importing torch (laptop-safe). ``--smoke`` shrinks to 1 epoch per
phase for a cheap end-to-end check. The real run needs torch + torchvision (the
cloud GPU); see ``docs/perception_piece_cnn.md``.

    # laptop: validate the plan
    python scripts/train_piece_cnn.py --data datasets/piece_smoke.npz --dry-run
    # cloud GPU: train both stages, export ONNX for the laptop
    python scripts/train_piece_cnn.py --data datasets/piece.npz \
        --out-dir outputs/piece_cnn --stage both
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from chess_robot.perception.piece_cnn_config import (
    load_two_stage_config,
    smoke_two_stage_config,
)
from chess_robot.perception.piece_dataset import PieceCropDataset, load_dataset


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the two-stage piece classifier.")
    parser.add_argument("--config", type=Path, default=Path("configs/perception/piece_cnn.yaml"))
    parser.add_argument("--data", type=Path, default=None, help="prepared crop dataset (.npz)")
    parser.add_argument(
        "--val-data", type=Path, default=None, help="separate held-out crop dataset (.npz)"
    )
    parser.add_argument("--stage", choices=("both", "occupancy", "piece"), default="both")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/piece_cnn"))
    parser.add_argument(
        "--init-dir",
        type=Path,
        default=None,
        help="warm-start from {occupancy,piece}.pt here (few-shot fine-tune base)",
    )
    parser.add_argument("--smoke", action="store_true", help="1 epoch per phase, tiny batch")
    parser.add_argument("--dry-run", action="store_true", help="validate + print plan only")
    parser.add_argument("--device", default=None, help="override config device (cpu/cuda)")
    parser.add_argument("--no-export", action="store_true", help="skip ONNX export")
    args = parser.parse_args(argv)

    config = load_two_stage_config(args.config)
    if args.smoke:
        config = smoke_two_stage_config(config)
    if args.device is not None:
        config = replace(config, device=args.device)

    dataset: PieceCropDataset | None = None
    if args.data is not None and args.data.exists():
        dataset = load_dataset(args.data)
    val_dataset: PieceCropDataset | None = None
    if args.val_data is not None and args.val_data.exists():
        val_dataset = load_dataset(args.val_data)

    print("piece-CNN training plan" + (" [smoke]" if args.smoke else ""))
    print(f"  stages:    {args.stage}")
    print(
        f"  occupancy: {config.occupancy.backbone} "
        f"input {config.occupancy.input_size} pad {config.occupancy.top_pad_ratio}"
    )
    print(
        f"  piece:     {config.piece.backbone} "
        f"input {config.piece.input_size} pad {config.piece.top_pad_ratio}"
    )
    print(
        f"  training:  device={config.device} batch={config.batch_size} "
        f"head_epochs={config.head_epochs} full_epochs={config.full_epochs}"
    )
    print(f"  data:      {args.data if args.data else '-'}")
    if dataset is not None:
        print("  " + dataset.summary())
    eval_source = f"held-out {args.val_data}" if val_dataset is not None else "internal split"
    print(f"  eval:      {eval_source}")
    print(f"  init:      {args.init_dir if args.init_dir else 'ImageNet weights'}")

    if args.dry_run:
        return

    if dataset is None:
        if args.data is not None:
            raise SystemExit(
                f"dataset not found at {args.data}; build it first with prepare_piece_dataset.py"
            )
        raise SystemExit(
            "--data is required for a real run (build it with prepare_piece_dataset.py)"
        )

    # Lazy import so --dry-run stays torch-free (laptop-safe).
    from chess_robot.perception import piece_cnn

    results = piece_cnn.run_training(
        config,
        dataset,
        stage=args.stage,
        out_dir=args.out_dir,
        export=not args.no_export,
        val_dataset=val_dataset,
        init_dir=args.init_dir,
    )
    for name, result in results.items():
        print(
            f"  {name}: val_accuracy={result['val_accuracy']:.4f} "
            f"checkpoint={result['checkpoint']} onnx={result['onnx']}"
        )


if __name__ == "__main__":
    main()
