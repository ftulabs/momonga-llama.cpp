#!/bin/bash
# ==============================================================================
# Script Name: run-bge-reranker-v2-m3.sh
# Description: Launch verified best configuration for bge-reranker-v2-m3 on Jetson AGX Xavier.
#
# Hardware: Jetson AGX Xavier (sm_72 Volta, 30 GB unified LPDDR4x memory)
# Architecture: XLM-RoBERTa 568M Encoder Reranker (cls_out)
# Model: bge-reranker-v2-m3-Q8_0.gguf (636 MB)
# Default Port: 8012
# Default Context: 8192 (full query+doc context; np=1, b=8192, ub=8192)
#
# Key Features & Notes:
#   - Encoder models require n_ubatch >= n_tokens; UB=8192 and NP=1 prevent
#     n_ubatch assertion failures and preserve full 8192 context per pair
#   - Q8_0 preserves ranking calibration and maximizes sm_72 prefill throughput
#   - Measured 12-doc / 10,688-token rerank: 2.83s (3,776 tok/s) vs 17.0s on zerank-2
#
# Usage:
#   ./run-bge-reranker-v2-m3.sh              # Run with default context (8192) on port 8012
#   PORT=8020 ./run-bge-reranker-v2-m3.sh    # Override listening port
#   WARMUP=0 ./run-bge-reranker-v2-m3.sh     # Skip warmup request
#
# Example Query:
#   curl -X POST http://127.0.0.1:8012/v1/rerank \
#     -H "Content-Type: application/json" \
#     -d '{"model": "bge-reranker-v2-m3", "query": "NVIDIA Jetson Xavier", "documents": ["Tegra SoC with Volta GPU", "Ancient Egyptian history"]}'
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
NP=${NP:-1}
UB=${UB:-8192}
B=${B:-8192}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
BIN="${BIN:-$REPO_DIR/build-volta/bin/llama-server}"
if [ ! -x "$BIN" ]; then
  BIN=/home/triph/juicer/momonga-llama.cpp/build-volta/bin/llama-server
fi

if [ ! -x "$BIN" ]; then
  echo "[run-bge-reranker-v2-m3.sh] Error: llama-server executable not found at: $BIN" >&2
  exit 1
fi

MODEL=${MODEL:-"/mnt/cache/huggingface/hub/models--gpustack--bge-reranker-v2-m3-GGUF/snapshots/3093af03b1a635e67b084b1d8c03c5f5e020fd05/bge-reranker-v2-m3-Q8_0.gguf"}
if [ ! -f "$MODEL" ]; then
  MODEL="/mnt/ssd/cache/huggingface/hub/models--gpustack--bge-reranker-v2-m3-GGUF/snapshots/3093af03b1a635e67b084b1d8c03c5f5e020fd05/bge-reranker-v2-m3-Q8_0.gguf"
fi

if [ ! -f "$MODEL" ]; then
  echo "[run-bge-reranker-v2-m3.sh] Error: Model file not found: $MODEL" >&2
  exit 1
fi

echo "================================================================="
echo " Starting bge-reranker-v2-m3 Encoder Reranker Server"
echo " Binary:  $BIN"
echo " Model:   $MODEL"
echo " Port:    $PORT"
echo " Context: $CTX"
echo " Slots:   $NP"
echo "================================================================="

# Ensure CUDA graphs are enabled: presence of GGML_CUDA_DISABLE_GRAPHS disables them
env -u GGML_CUDA_DISABLE_GRAPHS "$BIN" \
  -m "$MODEL" \
  --alias bge-reranker-v2-m3 \
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
    echo "[run-bge-reranker-v2-m3.sh] Shutting down llama-server (PID $SRV_PID)..."
    kill -TERM "$SRV_PID" 2>/dev/null || true
    wait "$SRV_PID" 2>/dev/null || true
  fi
}
trap cleanup INT TERM EXIT

if [ "$WARMUP" -eq 1 ]; then
  echo "[run-bge-reranker-v2-m3.sh] Waiting for server to become ready on http://${HOST}:${PORT}..."
  for _ in $(seq 1 120); do
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
      echo "[run-bge-reranker-v2-m3.sh] Error: server process exited unexpectedly during startup." >&2
      exit 1
    fi
    if curl -s "http://127.0.0.1:${PORT}/health" | grep -q 'ok'; then
      break
    fi
    sleep 1
  done

  echo "[run-bge-reranker-v2-m3.sh] Server ready. Running startup warm-up query..."
  curl -s "http://127.0.0.1:${PORT}/v1/rerank" \
    -H "Content-Type: application/json" \
    -d '{"model":"bge-reranker-v2-m3","query":"warmup query","documents":["doc a","doc b"]}' > /dev/null 2>&1 || true
  echo "[run-bge-reranker-v2-m3.sh] Warm-up complete. Reranker ready for requests on port $PORT."
fi

wait "$SRV_PID"
