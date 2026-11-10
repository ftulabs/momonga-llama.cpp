#!/bin/bash
# ==============================================================================
# Script Name: run-zerank-2.sh
# Description: Launch verified best configuration for zerank-2 Reranker on Jetson AGX Xavier.
#
# Hardware: Jetson AGX Xavier (sm_72 Volta, 30 GB unified LPDDR4x memory)
# Architecture: Qwen3 4.0B Causal Generative Reranker
# Model: zerank-2.Q8_0.gguf (4.28 GB)
# Default Port: 8012
# Default Context: 8192 (Pinned KV pool to avoid unpinned 48 GB OOM)
#
# Key Features & Optimizations:
#   - Target token 9454 ("Yes") zero-copy row view via ggml_view_2d
#   - Eliminates 151,936x full vocabulary projection (~400 MB DRAM read saved per pass)
#   - Flash Attention with Volta sm_72 Tensor Core MMA instructions enabled
#   - ZeroEntropy ChatML query/document auto-formatting
#   - Calibrated sigmoid probability scoring: sigma(logit / 5.0)
#   - Throughput: 177.87 ms/doc, 223.7 tok/s, 7.80 Tokens/Joule (ServerBench)
#
# Usage:
#   ./run-zerank-2.sh              # Run with default context (8192) on port 8012
#   ./run-zerank-2.sh 16384        # Override context size
#   PORT=8020 ./run-zerank-2.sh    # Override listening port
#   WARMUP=0 ./run-zerank-2.sh     # Skip warmup request
#
# Example Query:
#   curl -X POST http://127.0.0.1:8012/rerank \
#     -H "Content-Type: application/json" \
#     -d '{"query": "NVIDIA Jetson Xavier", "documents": ["Tegra SoC with Volta GPU", "Ancient Egyptian history"]}'
# ==============================================================================
set -euo pipefail

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  sed -n "2,/^# ======/p" "$0" | sed -e "s/^# //" -e "s/^#$//"
  exit 0
fi

CTX=${1:-8192}
[ $# -gt 0 ] && shift

PORT=${PORT:-8012}
HOST=${HOST:-"0.0.0.0"}
WARMUP=${WARMUP:-1}
NP=${NP:-4}
UB=${UB:-1024}
B=${B:-2048}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
BIN="${BIN:-$REPO_DIR/build-volta/bin/llama-server}"
if [ ! -x "$BIN" ]; then
  BIN=/home/triph/juicer/momonga-llama.cpp/build-volta/bin/llama-server
fi

if [ ! -x "$BIN" ]; then
  echo "[run-zerank-2.sh] Error: llama-server executable not found at: $BIN" >&2
  exit 1
fi

MODEL=${MODEL:-"/mnt/cache/huggingface/hub/models--mradermacher--zerank-2-GGUF/snapshots/6886178557c58b375fcb79c8751673b97baeb633/zerank-2.Q8_0.gguf"}
if [ ! -f "$MODEL" ]; then
  MODEL="/mnt/cache/huggingface/hub/models--mradermacher--zerank-2-GGUF/blobs/7b9ba05a0509151c911582a4d62b14003f6a4fafa0e7ccdf572c7598cde1c100"
fi
if [ ! -f "$MODEL" ]; then
  MODEL="/mnt/ssd/cache/huggingface/hub/models--mradermacher--zerank-2-GGUF/snapshots/6886178557c58b375fcb79c8751673b97baeb633/zerank-2.Q8_0.gguf"
fi

if [ ! -f "$MODEL" ]; then
  echo "[run-zerank-2.sh] Error: Model file not found: $MODEL" >&2
  exit 1
fi

echo "================================================================="
echo " Starting zerank-2 Generative Reranker Server"
echo " Binary:  $BIN"
echo " Model:   $MODEL"
echo " Port:    $PORT"
echo " Context: $CTX"
echo " Slots:   $NP"
echo "================================================================="

# Ensure CUDA graphs are enabled: presence of GGML_CUDA_DISABLE_GRAPHS disables them
env -u GGML_CUDA_DISABLE_GRAPHS "$BIN" \
  -m "$MODEL" \
  --reranking \
  -fa on \
  -ngl 99 \
  -c "$CTX" \
  -np "$NP" \
  -ub "$UB" \
  -b "$B" \
  --port "$PORT" \
  --host "$HOST" "$@" &
SRV_PID=$!

cleanup() {
  if kill -0 "$SRV_PID" 2>/dev/null; then
    echo "[run-zerank-2.sh] Shutting down llama-server (PID $SRV_PID)..."
    kill -TERM "$SRV_PID" 2>/dev/null || true
    wait "$SRV_PID" 2>/dev/null || true
  fi
}
trap cleanup INT TERM EXIT

if [ "$WARMUP" -eq 1 ]; then
  echo "[run-zerank-2.sh] Waiting for server to become ready on http://${HOST}:${PORT}..."
  for _ in $(seq 1 120); do
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
      echo "[run-zerank-2.sh] Error: server process exited unexpectedly during startup." >&2
      exit 1
    fi
    if curl -s "http://127.0.0.1:${PORT}/health" | grep -q 'ok'; then
      break
    fi
    sleep 1
  done

  echo "[run-zerank-2.sh] Server ready. Running startup graph warm-up query..."
  curl -s "http://127.0.0.1:${PORT}/rerank" \
    -H "Content-Type: application/json" \
    -d '{"query":"warmup query","documents":["doc a","doc b"]}' > /dev/null 2>&1 || true
  echo "[run-zerank-2.sh] Warm-up complete. Reranker ready for requests on port $PORT."
fi

wait "$SRV_PID"
