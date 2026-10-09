#!/usr/bin/env bash
#
# Datacore Gemma 4 inference service — vllm/gemma-26ba4b-single on one RTX 3090
# (24 GB, TP=1) with Datacore's serving settings. The engine, compose, patches and
# chat template are the upstream slug's; this only pins its launch knobs and hands
# off to switch.sh. Setup and request usage: docs/DATACORE_GEMMA.md.
#
# Usage:
#   bash scripts/launch-datacore-gemma.sh            # switch + wait until ready
#   bash scripts/launch-datacore-gemma.sh --no-wait  # any switch.sh flag passes through
#   VLLM_ENABLE_LORA=true bash scripts/launch-datacore-gemma.sh  # + Indonesia LoRA adapter
#
# Serves the OpenAI-compatible API on the slug's default port (8040) as model
# gemma-4-26b-a4b-awq, plus poi-ai-gemma-silver-v1 when the LoRA adapter is enabled.

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# External MTP drafter (google/gemma-4-26B-A4B-it-assistant) at depth 4.
export SPEC=on
export SPEC_N=4
export MAX_MODEL_LEN=12500
export GPU_MEMORY_UTILIZATION=0.94
export MAX_NUM_SEQS=8
export MAX_NUM_BATCHED_TOKENS=4096
export TP=1

# Optional LoRA adapter, off by default. The path is inside the container, under the
# model-cache mount (MODEL_DIR on the host).
export VLLM_ENABLE_LORA="${VLLM_ENABLE_LORA:-false}"
export VLLM_LORA_ADAPTER_NAME="${VLLM_LORA_ADAPTER_NAME:-poi-ai-gemma-silver-v1}"
export VLLM_LORA_ADAPTER_PATH="${VLLM_LORA_ADAPTER_PATH:-/root/.cache/huggingface/poi-ai-gemma-silver-v1}"
export VLLM_MAX_LORA_RANK="${VLLM_MAX_LORA_RANK:-32}"

exec bash "${ROOT_DIR}/scripts/switch.sh" "$@" vllm/gemma-26ba4b-single
