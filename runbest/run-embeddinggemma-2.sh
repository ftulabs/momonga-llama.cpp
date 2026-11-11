#!/bin/bash
# ==============================================================================
# Script Name: run-embeddinggemma-2.sh
# Description: Launch verified best configuration for EmbeddingGemma-2 Multimodal Embedding Server.
#
# Hardware: Jetson AGX Xavier (sm_72 Volta, 30 GB unified LPDDR4x memory)
# Architecture: EmbeddingGemma-2 with MMProj Vision Companion
# Model: embeddinggemma-2-Q8_0.gguf (310 MB)
# MMProj: mmproj-F16.gguf (843 MB Vision Projector)
# Default Port: 8086
# Default Context: 8192
#
# Key Features & Optimizations:
#   - High-throughput dedicated embedding mode (--embedding)
#   - Multimodal support: embeds text queries and image inputs into unified vector space
#   - Full GPU offloading (-ngl 99)
#   - Wide physical microbatch (-ub 4096 -b 4096) avoiding 1120 vision token clipping
#   - OpenAI-compatible /v1/embeddings endpoint
#
# Usage:
#   ./run-embeddinggemma-2.sh              # Run on default port 8086
#   ./run-embeddinggemma-2.sh 16384        # Override context size
#   PORT=8096 ./run-embeddinggemma-2.sh   # Override port
#   WARMUP=0 ./run-embeddinggemma-2.sh    # Skip warmup request
#
# Example Query:
#   curl -X POST http://127.0.0.1:8086/v1/embeddings \
#     -H "Content-Type: application/json" \
#     -d '{"input": "Autonomous edge robotics on Jetson Xavier"}'
# ==============================================================================
set -euo pipefail

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  sed -n "2,/^# ======/p" "$0" | sed -e "s/^# //" -e "s/^#$//"
  exit 0
fi

CTX=${1:-8192}
[ $# -gt 0 ] && shift

PORT=${PORT:-8086}
HOST=${HOST:-"0.0.0.0"}
WARMUP=${WARMUP:-1}
UB=${UB:-4096}
B=${B:-4096}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
BIN="${BIN:-$REPO_DIR/build-volta/bin/llama-server}"
if [ ! -x "$BIN" ]; then
  BIN=/home/triph/juicer/momonga-llama.cpp/build-volta/bin/llama-server
fi

if [ ! -x "$BIN" ]; then
  echo "[run-embeddinggemma-2.sh] Error: llama-server executable not found at: $BIN" >&2
  exit 1
fi

MODEL=${MODEL:-"/mnt/cache/huggingface/hub/models--unsloth--embeddinggemma-2-GGUF/snapshots/031f0d4b35536f69ab3509d4893c923264fcf253/embeddinggemma-2-Q8_0.gguf"}
if [ ! -f "$MODEL" ]; then
  MODEL="/mnt/ssd/cache/huggingface/hub/models--unsloth--embeddinggemma-2-GGUF/snapshots/031f0d4b35536f69ab3509d4893c923264fcf253/embeddinggemma-2-Q8_0.gguf"
fi

if [ ! -f "$MODEL" ]; then
  echo "[run-embeddinggemma-2.sh] Error: Model file not found: $MODEL" >&2
  exit 1
fi

MMPROJ_ARG=()
MMPROJ=${MMPROJ:-"/mnt/cache/huggingface/hub/models--unsloth--embeddinggemma-2-GGUF/snapshots/031f0d4b35536f69ab3509d4893c923264fcf253/mmproj-F16.gguf"}
if [ ! -f "$MMPROJ" ]; then
  MMPROJ="/mnt/ssd/cache/huggingface/hub/models--unsloth--embeddinggemma-2-GGUF/snapshots/031f0d4b35536f69ab3509d4893c923264fcf253/mmproj-F16.gguf"
fi
if [ "$MMPROJ" != "none" ] && [ -n "$MMPROJ" ] && [ -f "$MMPROJ" ]; then
  MMPROJ_ARG=(--mmproj "$MMPROJ")
fi

echo "================================================================="
echo " Starting EmbeddingGemma-2 Multimodal Embedding Server"
echo " Binary:   $BIN"
echo " Model:    $MODEL"
echo " Port:     $PORT"
echo " Context:  $CTX"
echo " MMProj:   ${MMPROJ:-none}"
echo "================================================================="

# Ensure CUDA graphs are enabled: presence of GGML_CUDA_DISABLE_GRAPHS disables them
env -u GGML_CUDA_DISABLE_GRAPHS "$BIN" \
  -m "$MODEL" \
  --embedding \
  -fa on \
  -ngl 99 \
  -c "$CTX" \
  -ub "$UB" \
  -b "$B" \
  "${MMPROJ_ARG[@]}" \
  --port "$PORT" \
  --host "$HOST" "$@" &
SRV_PID=$!

cleanup() {
  if kill -0 "$SRV_PID" 2>/dev/null; then
    echo "[run-embeddinggemma-2.sh] Shutting down llama-server (PID $SRV_PID)..."
    kill -TERM "$SRV_PID" 2>/dev/null || true
    wait "$SRV_PID" 2>/dev/null || true
  fi
}
trap cleanup INT TERM EXIT

if [ "$WARMUP" -eq 1 ]; then
  echo "[run-embeddinggemma-2.sh] Waiting for server to become ready on http://${HOST}:${PORT}..."
  for _ in $(seq 1 120); do
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
      echo "[run-embeddinggemma-2.sh] Error: server process exited unexpectedly during startup." >&2
      exit 1
    fi
    if curl -s "http://127.0.0.1:${PORT}/health" | grep -q 'ok'; then
      break
    fi
    sleep 1
  done

  echo "[run-embeddinggemma-2.sh] Server ready. Running startup warm-up embedding query..."
  curl -s "http://127.0.0.1:${PORT}/v1/embeddings" \
    -H "Content-Type: application/json" \
    -d '{"input":"warmup"}' > /dev/null 2>&1 || true
  echo "[run-embeddinggemma-2.sh] Warm-up complete. Embedding server ready for user requests."
fi

wait "$SRV_PID"
