#!/bin/bash
# /mnt/sdcard/opt/llama/llama-embedding-start.sh
# Startup wrapper for EmbeddingGemma-2 Multimodal Embedding Server (systemd).
set -euo pipefail

ENV_FILE="${1:-/mnt/sdcard/opt/llama/llama-embedding.env}"
if [ -f "$ENV_FILE" ]; then
  # shellcheck source=/dev/null
  source "$ENV_FILE"
fi

BIN="${LLAMA_BIN:-/home/triph/juicer/momonga-llama.cpp/build-volta/bin/llama-server}"
if [ ! -x "$BIN" ]; then
  BIN=/home/triph/juicer/momonga-llama.cpp/build/bin/llama-server
fi

if [ ! -x "$BIN" ]; then
  echo "Error: llama-server executable not found at $BIN" >&2
  exit 1
fi

MODEL="${MODEL_PATH:-/mnt/cache/huggingface/hub/models--unsloth--embeddinggemma-2-GGUF/snapshots/031f0d4b35536f69ab3509d4893c923264fcf253/embeddinggemma-2-Q8_0.gguf}"
if [ ! -f "$MODEL" ]; then
  MODEL=/mnt/ssd/cache/huggingface/hub/models--unsloth--embeddinggemma-2-GGUF/snapshots/031f0d4b35536f69ab3509d4893c923264fcf253/embeddinggemma-2-Q8_0.gguf
fi

if [ ! -f "$MODEL" ]; then
  echo "Error: Model not found at $MODEL" >&2
  exit 1
fi

MMPROJ_ARGS=()
MMPROJ="${MMPROJ_PATH:-/mnt/cache/huggingface/hub/models--unsloth--embeddinggemma-2-GGUF/snapshots/031f0d4b35536f69ab3509d4893c923264fcf253/mmproj-F16.gguf}"
if [ ! -f "$MMPROJ" ]; then
  MMPROJ=/mnt/ssd/cache/huggingface/hub/models--unsloth--embeddinggemma-2-GGUF/snapshots/031f0d4b35536f69ab3509d4893c923264fcf253/mmproj-F16.gguf
fi
if [ -f "$MMPROJ" ] && [ "$MMPROJ" != "none" ]; then
  MMPROJ_ARGS=(--mmproj "$MMPROJ")
fi

# Ensure CUDA graphs are enabled: presence of GGML_CUDA_DISABLE_GRAPHS disables them
unset GGML_CUDA_DISABLE_GRAPHS || true

exec "$BIN" \
  -m "$MODEL" \
  "${MMPROJ_ARGS[@]}" \
  --alias "${MODEL_ALIAS:-embeddinggemma-2}" \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-8086}" \
  -c "${CTX_SIZE:-8192}" \
  -b "${BATCH:-4096}" \
  -ub "${UBATCH:-4096}" \
  -fa "${FLASH_ATTN:-on}" \
  -ngl "${N_GPU_LAYERS:-99}" \
  ${EXTRA_ARGS:---embedding}
