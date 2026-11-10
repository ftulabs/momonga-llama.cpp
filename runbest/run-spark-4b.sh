#!/bin/bash
# ==============================================================================
# Script Name: run-spark-4b.sh
# Description: Launch verified best configuration for Spark-X2.6 4B Edge Model.
#
# Hardware: Jetson AGX Xavier (sm_72 Volta, 30 GB unified LPDDR4x memory)
# Architecture: Spark 4.0B Dense Architecture
# Model: Spark-X2.6-4B-Q8_0.gguf (4.38 GB; fallback to Q4_0 2.41 GB)
# Default Port: 8085
# Default Context: 16384 (supports up to 32768)
#
# Key Features & Optimizations:
#   - Full GPU offloading (-ngl 99) for blazing fast edge inference
#   - Sub-80ms per token decode latency
#   - Low memory footprint (~5 GB total including KV cache)
#   - Decoupled physical microbatch (-ub 1024 -b 2048)
#   - Host flag -bs (--backend-sampling) offloading sampling to GPU
#
# Usage:
#   ./run-spark-4b.sh              # Run with default 16k context on port 8085
#   ./run-spark-4b.sh 32768        # Run with 32k context
#   PORT=8095 ./run-spark-4b.sh   # Override port
#   WARMUP=0 ./run-spark-4b.sh    # Skip warmup request
#
# Example Query:
#   curl -X POST http://127.0.0.1:8085/completion \
#     -H "Content-Type: application/json" \
#     -d '{"prompt": "Hello world in Rust:", "n_predict": 64}'
# ==============================================================================
set -euo pipefail

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  sed -n "2,/^# ======/p" "$0" | sed -e "s/^# //" -e "s/^#$//"
  exit 0
fi

CTX=${1:-16384}
[ $# -gt 0 ] && shift

PORT=${PORT:-8085}
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
  echo "[run-spark-4b.sh] Error: llama-server executable not found at: $BIN" >&2
  exit 1
fi

MODEL=${MODEL:-"/mnt/ssd/models/Spark-X2.6-4B-Q8_0.gguf"}
if [ ! -f "$MODEL" ]; then
  MODEL="/home/triph/models/Spark-X2.6-4B-Q8_0.gguf"
fi
if [ ! -f "$MODEL" ]; then
  MODEL="/mnt/ssd/models/Spark-X2.6-4B-Q4_0.gguf"
fi
if [ ! -f "$MODEL" ]; then
  MODEL="/home/triph/models/Spark-X2.6-4B-Q4_0.gguf"
fi

if [ ! -f "$MODEL" ]; then
  echo "[run-spark-4b.sh] Error: Model file not found: $MODEL" >&2
  exit 1
fi

echo "================================================================="
echo " Starting Spark-X2.6 4B Edge Model Server"
echo " Binary:   $BIN"
echo " Model:    $MODEL"
echo " Port:     $PORT"
echo " Context:  $CTX"
echo "================================================================="

# Ensure CUDA graphs are enabled: presence of GGML_CUDA_DISABLE_GRAPHS disables them
env -u GGML_CUDA_DISABLE_GRAPHS "$BIN" \
  -m "$MODEL" \
  -c "$CTX" -fa on -ngl 99 \
  -ub "$UB" -b "$B" \
  -bs \
  --alias "Spark X2.6 4B" \
  --port "$PORT" \
  --host "$HOST" "$@" &
SRV_PID=$!

cleanup() {
  if kill -0 "$SRV_PID" 2>/dev/null; then
    echo "[run-spark-4b.sh] Shutting down llama-server (PID $SRV_PID)..."
    kill -TERM "$SRV_PID" 2>/dev/null || true
    wait "$SRV_PID" 2>/dev/null || true
  fi
}
trap cleanup INT TERM EXIT

if [ "$WARMUP" -eq 1 ]; then
  echo "[run-spark-4b.sh] Waiting for server to become ready on http://${HOST}:${PORT}..."
  for _ in $(seq 1 120); do
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
      echo "[run-spark-4b.sh] Error: server process exited unexpectedly during startup." >&2
      exit 1
    fi
    if curl -s "http://127.0.0.1:${PORT}/health" | grep -q 'ok'; then
      break
    fi
    sleep 1
  done

  echo "[run-spark-4b.sh] Server ready. Running startup graph warm-up..."
  curl -s "http://127.0.0.1:${PORT}/completion" \
    -H "Content-Type: application/json" \
    -d '{"prompt":"warmup","n_predict":1}' > /dev/null 2>&1 || true
  echo "[run-spark-4b.sh] Warm-up complete. Server ready for user requests."
fi

wait "$SRV_PID"
