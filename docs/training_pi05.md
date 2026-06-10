# Training pi0.5

LeRobot pi0.5 guide: https://huggingface.co/docs/lerobot/pi05

## Goal

```text
teleoperated demonstrations -> fine-tuned pi0.5 policy
```

The fine-tuned pi0.5 policy becomes the **base policy** for both later
approaches (residual RL and HIL-RL).

## Sequencing rule

Do **not** implement RL until the supervised policy can produce partially
useful real-robot behavior.

## Minimum expected supervised policy before moving to RL

```text
can approach the piece
can grasp sometimes
can move toward target square
can release without obviously unsafe motions
```

## Cloud GPU (vast.ai)

Training runs on a rented **vast.ai** GPU instance.

```text
provider:     vast.ai (per-hour rented instance; access via the project email)
GPU:          RTX 4090 or RTX 5090 — most cost-effective for this fine-tuning
budget:       ~$100 in credits — pick a cost-effective GPU and keep runs bounded
preinstalled: CUDA + PyTorch (use the image's torch; do not pin torch ourselves)
```

Per-instance setup (on the vast.ai box, not the laptop):

```bash
# CUDA + PyTorch already present on the instance image.
pip install -e ".[train]"   # installs LeRobot (pi0.5) on top of the image torch
```

Budget discipline: prefer the cheaper sufficient GPU, do short dry-run/1-step
checks before long runs, and stop idle instances. The `train` extra deliberately
does not pin torch so it reuses the instance's preinstalled build.

## Notes

* Training runs on the cloud GPU (no useful local CUDA). The laptop runs the
  inference client if a remote policy server is used.
* Use only successful demonstrations for supervised fine-tuning (see
  `data_collection.md`).
* Keep camera setup, board, and robot base fixed during data collection used
  for supervised training.
* Configs live in `configs/policy/pi05.yaml`; the training script is
  `scripts/train_pi05.py`. Do not hard-code checkpoint paths or dataset repo
  IDs — use config + placeholders.
