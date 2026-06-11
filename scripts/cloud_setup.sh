#!/usr/bin/env bash
# Option A: pi0.5 fine-tuning smoke test on a vast.ai GPU box.
#
# Run this ON the rented instance (RTX 4090/5090; CUDA + PyTorch preinstalled),
# from the repo root. Requires HF_TOKEN in the environment (a Hugging Face read
# token). NEVER commit the token. See docs/cloud_smoke_test.md.
#
#   export HF_TOKEN=hf_...        # do NOT paste into git
#   bash scripts/cloud_setup.sh
set -euo pipefail

: "${HF_TOKEN:?Set HF_TOKEN (Hugging Face read token) before running}"

# 1. LeRobot with pi0.5 support (from source — the verified path).
if [ ! -d lerobot ]; then
  git clone https://github.com/huggingface/lerobot.git
fi
pip install -e "./lerobot[pi,dataset,training]"

# 2. This project (provides the dataset generator + train wrapper). Also put src
#    on PYTHONPATH directly — editable installs don't always expose subpackages.
pip install -e .
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

# 3. Hugging Face auth (downloads lerobot/pi05_base; accept any gated licenses
#    such as Gemma/PaliGemma on the HF model pages first). Newer huggingface-hub
#    replaced `huggingface-cli login` with `hf auth login`; HF_TOKEN in the env is
#    also picked up automatically, so this is best-effort.
hf auth login --token "$HF_TOKEN" || echo "hf auth login skipped; relying on HF_TOKEN env var"

# 4. Build a tiny local mock dataset.
python scripts/make_smoke_dataset.py --repo-id local/chess_smoke

# 5. Cheap ~2-step smoke fine-tune (low memory: action expert only, bf16, batch 1).
python scripts/train_pi05.py --smoke \
  --dataset-repo-id local/chess_smoke \
  --output-dir ./outputs/pi05_smoke \
  --policy-repo-id local/chess_pi05_smoke

echo
echo "Smoke run finished. A checkpoint under ./outputs/pi05_smoke means the"
echo "LeRobot + pi0.5 training integration works on our data format."
