#!/bin/bash
# ==============================================================================
# Script Name: run-deepseek-v4-9b.sh
# Description: Launch verified best configuration for DeepSeek-V4-Pro 9B Q8_0 Reasoning Model.
#
# Hardware: Jetson AGX Xavier (sm_72 Volta, 30 GB unified LPDDR4x memory)
# Architecture: Qwen 3.5 9B Dense Architecture
# Model: DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic-Q8_0.gguf (9.79 GB)
# Default Port: 8084
# Default Context: 16384 (16k context, leaves ~16 GB free RAM)
#
# Key Features & Optimizations:
#   - Full Q8_0 uncompressed quantization for zero-loss reasoning precision
#   - Full GPU offloading (-ngl 99) for all layers into unified memory
#   - Q8_0 KV cache precision (-ctk q8_0 -ctv q8_0)
#   - Host flag -bs (--backend-sampling) offloading sampling to GPU
#   - Decoupled physical microbatch (-ub 1024 -b 2048)
#
# Usage:
#   ./run-deepseek-v4-9b.sh              # Run with default 16k context on port 8084
#   ./run-deepseek-v4-9b.sh 32768        # Run with 32k context
#   PORT=8094 ./run-deepseek-v4-9b.sh   # Override port
#   WARMUP=0 ./run-deepseek-v4-9b.sh    # Skip warmup request
#
# Example Query:
#   curl -X POST http://127.0.0.1:8084/completion \
#     -H "Content-Type: application/json" \
#     -d '{"prompt": "Prove that the square root of 2 is irrational:", "n_predict": 128}'
# ==============================================================================
set -euo pipefail

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  sed -n "2,/^# ======/p" "$0" | sed -e "s/^# //" -e "s/^#$//"
  exit 0
fi

CTX=${1:-16384}
[ $# -gt 0 ] && shift

PORT=${PORT:-8084}
HOST=${HOST:-"0.0.0.0"}
WARMUP=${WARMUP:-1}
UB=${UB:-1024}
B=${B:-2048}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
BIN="${BIN:-$REPO_DIR/build-volta/bin/llama-server}"
if [ ! -x "$BIN" ]; then
  BIN=/home/triph/juicer/momonga-llama.cpp/build-volta/bin/llama-server
fi

if [ ! -x "$BIN" ]; then
  echo "[run-deepseek-v4-9b.sh] Error: llama-server executable not found at: $BIN" >&2
  exit 1
fi

MODEL=${MODEL:-"/mnt/ssd/models/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic-Q8_0.gguf"}
if [ ! -f "$MODEL" ]; then
  MODEL="/home/triph/models/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic-Q8_0.gguf"
fi

if [ ! -f "$MODEL" ]; then
  echo "[run-deepseek-v4-9b.sh] Error: Model file not found: $MODEL" >&2
  exit 1
fi

echo "================================================================="
echo " Starting DeepSeek-V4-Pro 9B Q8_0 Reasoning Server"
echo " Binary:   $BIN"
echo " Model:    $MODEL"
echo " Port:     $PORT"
echo " Context:  $CTX"
echo "================================================================="

# Ensure CUDA graphs are enabled: presence of GGML_CUDA_DISABLE_GRAPHS disables them
env -u GGML_CUDA_DISABLE_GRAPHS "$BIN" \
  -m "$MODEL" \
  -c "$CTX" -fa on -ngl 99 \
  -ctk q8_0 -ctv q8_0 \
  -ub "$UB" -b "$B" \
  -bs \
  --alias "DeepSeek V4 Pro 9B Q8_0" \
  --port "$PORT" \
  --host "$HOST" "$@" &
SRV_PID=$!

cleanup() {
  if kill -0 "$SRV_PID" 2>/dev/null; then
    echo "[run-deepseek-v4-9b.sh] Shutting down llama-server (PID $SRV_PID)..."
    kill -TERM "$SRV_PID" 2>/dev/null || true
    wait "$SRV_PID" 2>/dev/null || true
  fi
}
trap cleanup INT TERM EXIT

if [ "$WARMUP" -eq 1 ]; then
  echo "[run-deepseek-v4-9b.sh] Waiting for server to become ready on http://${HOST}:${PORT}..."
  for _ in $(seq 1 120); do
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
      echo "[run-deepseek-v4-9b.sh] Error: server process exited unexpectedly during startup." >&2
      exit 1
    fi
    if curl -s "http://127.0.0.1:${PORT}/health" | grep -q 'ok'; then
      break
    fi
    sleep 1
  done

  echo "[run-deepseek-v4-9b.sh] Server ready. Running startup graph warm-up..."
  curl -s "http://127.0.0.1:${PORT}/completion" \
    -H "Content-Type: application/json" \
    -d '{"prompt":"warmup","n_predict":1}' > /dev/null 2>&1 || true
  echo "[run-deepseek-v4-9b.sh] Warm-up complete. Server ready for user requests."
fi

wait "$SRV_PID"
