#!/bin/bash
# /mnt/sdcard/opt/llama/llama-rerank-start.sh
# Startup wrapper for zerank-2 Generative Causal Reranker Server (systemd).
set -euo pipefail

ENV_FILE="${1:-/mnt/sdcard/opt/llama/llama-rerank.env}"
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

MODEL="${MODEL_PATH:-/mnt/cache/huggingface/hub/models--mradermacher--zerank-2-GGUF/snapshots/6886178557c58b375fcb79c8751673b97baeb633/zerank-2.Q8_0.gguf}"
if [ ! -f "$MODEL" ]; then
  MODEL=/mnt/cache/huggingface/hub/models--mradermacher--zerank-2-GGUF/blobs/7b9ba05a0509151c911582a4d62b14003f6a4fafa0e7ccdf572c7598cde1c100
fi
if [ ! -f "$MODEL" ]; then
  MODEL=/mnt/ssd/cache/huggingface/hub/models--mradermacher--zerank-2-GGUF/snapshots/6886178557c58b375fcb79c8751673b97baeb633/zerank-2.Q8_0.gguf
fi

if [ ! -f "$MODEL" ]; then
  echo "Error: Model not found at $MODEL" >&2
  exit 1
fi

# Ensure CUDA graphs are enabled: presence of GGML_CUDA_DISABLE_GRAPHS disables them
unset GGML_CUDA_DISABLE_GRAPHS || true

exec "$BIN" \
  -m "$MODEL" \
  --alias "${MODEL_ALIAS:-zerank-2}" \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-8012}" \
  -c "${CTX_SIZE:-8192}" \
  -np "${PARALLEL:-4}" \
  -b "${BATCH:-2048}" \
  -ub "${UBATCH:-1024}" \
  -fa "${FLASH_ATTN:-on}" \
  -ngl "${N_GPU_LAYERS:-99}" \
  ${EXTRA_ARGS:---reranking}
