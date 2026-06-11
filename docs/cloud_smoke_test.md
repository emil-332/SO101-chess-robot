# Cloud smoke test — pi0.5 fine-tuning (Option A)

Goal: confirm the **LeRobot ↔ pi0.5 ↔ our-data-format** training integration
actually runs, cheaply (~$1, minutes), on a vast.ai GPU — **before** investing in
real data collection. This is the correct (and only viable) way to "see training
run": pi0.5 is a ~3B-param VLA and **cannot** train on the laptop (32 GB RAM is a
hard memory wall), so the smoke run is GPU-only.

It validates that the integration runs on mock data. It does **not** train a
useful policy (mock frames/actions teach nothing).

Verified against the LeRobot pi0.5 guide: <https://huggingface.co/docs/lerobot/pi05>.

> **✓ Ran successfully (2026-06-09)** on an RTX 4090 with **lerobot 0.5.2**, torch
> 2.11.0, CUDA 13, extras `pi,dataset,training`. pi0.5 (4B params, 693M trainable
> with `train_expert_only`) loaded, trained 2 steps, and wrote a checkpoint —
> integration confirmed. Gotchas hit (all fixed in the scripts): `hf auth login`
> (not `huggingface-cli`); install `[pi,dataset,training]`; accept the gated
> `google/paligemma-3b-pt-224`; `--policy.push_to_hub=false`; remove a stale
> `output_dir` between runs.

## Credentials (never commit these)

Set as environment variables **on the box** — do not paste them into chat or git.

| Variable | Needed? | Why |
|---|---|---|
| `HF_TOKEN` | **Yes** | Download `lerobot/pi05_base` from Hugging Face. First, on the HF website, **accept any gated licenses** the base model pulls in (Gemma / PaliGemma) with the same account. |
| `WANDB_API_KEY` | No | The smoke disables W&B (`--wandb.enable=false`). |
| `VAST_API_KEY` | Optional | Only if you provision/manage the instance via the vast.ai CLI instead of the web UI. |

`.env.example` lists these; copy it to `.env` (gitignored) locally, or `export`
them on the box.

## Steps

1. **Rent an instance** on vast.ai: an **RTX 4090 or 5090** (24–32 GB), an image
   with recent **CUDA + PyTorch preinstalled**, ~30 GB disk. Cost ≈ $0.3–0.6/hr;
   the smoke takes a few minutes ⇒ well under $1.
2. **SSH in**, clone this repo, `cd` into it.
3. `export HF_TOKEN=hf_...` (your read token).
4. `bash scripts/cloud_setup.sh` — installs LeRobot `[pi]` + this project, logs in
   to HF, builds a tiny mock dataset, and runs a ~2-step smoke fine-tune.

## What success looks like

- `make_smoke_dataset.py` prints the dataset root and episode count.
- `lerobot-train` loads `lerobot/pi05_base`, prints a couple of training steps with
  a (finite) loss, and writes a checkpoint under `./outputs/pi05_smoke`.

If that happens, the integration is verified: our `DatasetConfig` features, the
`lerobot-train` flags (`scripts/train_pi05.py --smoke`), and pi0.5 loading all
work together. **Pin the exact LeRobot version** that worked into `pyproject.toml`
afterward.

## Smoke profile (what `--smoke` sets)

`steps=2`, `batch_size=1`, `--policy.train_expert_only=true` (freeze the VLM →
much less memory), `--policy.gradient_checkpointing=true`, `--policy.dtype=bfloat16`,
`--policy.compile_model=false` (faster startup), `--wandb.enable=false`, and the
`--policy.normalization_mapping` MEAN_STD override (so a freshly-created,
un-quantiled dataset trains without the quantile-conversion step).

## Troubleshooting (the failures the smoke is meant to surface)

- **403 GatedRepoError**: pi0.5's backbone is PaliGemma; its tokenizer loads from
  the **gated** `google/paligemma-3b-pt-224`. `pi05_base` itself is not gated, so
  the download succeeds and then the tokenizer 403s. Fix: accept the license at
  <https://huggingface.co/google/paligemma-3b-pt-224> with the token's account
  (usually instant), then re-run. If another `google/gemma-*` repo 403s, accept
  that one too. (A 403 means access, not auth — the token is fine.)
- **CUDA OOM**: the smoke already freezes the VLM; also try a smaller image size in
  `configs/dataset/collect_chess_demos.yaml`, or a bigger GPU.
- **`add_frame()` / `finalize()` signature differs**: LeRobot's dataset API moves
  between versions — adjust the two marked lines in `make_smoke_dataset.py`
  (`task=` kwarg, or `consolidate()`), and note the working version.
- **`lerobot-train` flag rejected**: compare against `lerobot-train --help` on the
  box and update `build_train_command` in `policies/pi05_policy.py`.
- **torch reinstalled / version clash**: prefer a vast.ai image whose torch already
  satisfies LeRobot; if `pip install -e "./lerobot[pi]"` changes torch, that's a
  finding — record the working combination.
