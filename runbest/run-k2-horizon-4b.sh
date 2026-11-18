#!/bin/bash
# ==============================================================================
# Script Name: run-k2-horizon-4b.sh
# Description: Launch verified best configuration for K2-Horizon 4B Q4_K_M on Xavier.
#
# Hardware: Jetson AGX Xavier (sm_72 Volta, 30 GB unified LPDDR4x memory)
# Architecture: K2-Horizon Dense 5.06B (36 layers, GQA 32/8, grouped RMSNorm, 250k vocab)
# Model: K2-Horizon-4B-Q4_K_M.gguf (2.93 GiB)
# Default Port: 8087
# Default Context: 32768 (-ctk f16 -ctv f16; ~7.5 GiB total footprint)
#
# Key Features & Optimizations:
#   - Grouped RMSNorm + weight multiply fusion in src/models/k2-horizon.cpp + RMS_NORM+MUL+ADD fusion
#   - Vectorized 4-warp dequantize_block_q4_K<half> + direct src1 cache + f32x4 GLU + compact RoPE
#   - Volta sm_72 deep-K cublasGemmEx ALGO4/5/6 + PR #29796 redundant sync skip + PR #30792 coalesced q8_0 KV tile loader
#   - Volta sm_72 dense Q4_K MMVQ (unfused, nwarps=1 rpb=4 -> 97.7 GB/s), Q6_K small_k (64 regs), q8_1 cache
#   - Decode: 30.61 tok/s standalone (+34.8% vs 22.71 baseline), 29.60 tok/s server median (-t 2 -tb 2 -bs)
#   - Prefill: 771.66 tok/s at pp2048 (-ub 2048 -b 2048, +44.8%), 729.59 tok/s at pp512 (-ub 1024, +35.8%)
#   - Native f16 KV cache (-ctk f16 -ctv f16): strictly faster than q8_0 at all depths
#     (17.55 vs 14.06 tok/s at d=16384; q8_0 KV boosted +30.9% from 10.74 via coalesced fattn-tile)
#   - Host flag -bs (--backend-sampling) offloads 250k-vocab sampling to GPU
#
# Usage:
#   ./run-k2-horizon-4b.sh              # Run with 32k context on port 8087
#   ./run-k2-horizon-4b.sh 65536        # Run with 64k context
#   PORT=8097 ./run-k2-horizon-4b.sh    # Override port
#   WARMUP=0 ./run-k2-horizon-4b.sh     # Skip warmup request
#
# Example Query:
#   curl -X POST http://127.0.0.1:8087/completion \
#     -H "Content-Type: application/json" \
#     -d '{"prompt": "def quicksort(arr):", "n_predict": 64, "temperature": 0.0}'
# ==============================================================================
set -euo pipefail

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  sed -n "2,/^# ======/p" "$0" | sed -e "s/^# //" -e "s/^#$//"
  exit 0
fi

CTX=${1:-32768}
[ $# -gt 0 ] && shift

PORT=${PORT:-8087}
HOST=${HOST:-"0.0.0.0"}
WARMUP=${WARMUP:-1}
UB=${UB:-2048}
B=${B:-2048}
PAR=${PAR:-1}
THREADS=${THREADS:-2}
TB=${TB:-2}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
BIN="${BIN:-$REPO_DIR/build-volta/bin/llama-server}"
if [ ! -x "$BIN" ]; then
  BIN=/home/triph/juicer/momonga-llama.cpp/build-volta/bin/llama-server
fi

if [ ! -x "$BIN" ]; then
  echo "[run-k2-horizon-4b.sh] Error: llama-server executable not found at: $BIN" >&2
  exit 1
fi

MODEL=${MODEL:-"/mnt/ssd/models/K2-Horizon-4B-Q4_K_M.gguf"}
if [ ! -f "$MODEL" ]; then
  MODEL="/home/triph/models/K2-Horizon-4B-Q4_K_M.gguf"
fi

if [ ! -f "$MODEL" ]; then
  echo "[run-k2-horizon-4b.sh] Error: Model file not found: $MODEL" >&2
  exit 1
fi

echo "================================================================="
echo " Starting K2-Horizon 4B Q4_K_M Server"
echo " Binary:   $BIN"
echo " Model:    $MODEL"
echo " Port:     $PORT"
echo " Context:  $CTX (KV f16)"
echo "================================================================="

# Ensure CUDA graphs are enabled: presence of GGML_CUDA_DISABLE_GRAPHS disables them
env -u GGML_CUDA_DISABLE_GRAPHS "$BIN" \
  -m "$MODEL" \
  -c "$CTX" -fa on -ngl 99 \
  -t "$THREADS" -tb "$TB" \
  -ctk f16 -ctv f16 \
  -ub "$UB" -b "$B" \
  --parallel "$PAR" --poll 0 \
  -bs \
  --alias "K2-Horizon 4B" \
  --port "$PORT" \
  --host "$HOST" "$@" &
SRV_PID=$!

cleanup() {
  if kill -0 "$SRV_PID" 2>/dev/null; then
    echo "[run-k2-horizon-4b.sh] Shutting down llama-server (PID $SRV_PID)..."
    kill -TERM "$SRV_PID" 2>/dev/null || true
    wait "$SRV_PID" 2>/dev/null || true
  fi
}
trap cleanup INT TERM EXIT

if [ "$WARMUP" -eq 1 ]; then
  echo "[run-k2-horizon-4b.sh] Waiting for server to become ready on http://${HOST}:${PORT}..."
  for _ in $(seq 1 120); do
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
      echo "[run-k2-horizon-4b.sh] Error: server process exited unexpectedly during startup." >&2
      exit 1
    fi
    if curl -s "http://127.0.0.1:${PORT}/health" | grep -q 'ok'; then
      break
    fi
    sleep 1
  done

  echo "[run-k2-horizon-4b.sh] Server ready. Running startup graph warm-up..."
  curl -s "http://127.0.0.1:${PORT}/completion" \
    -H "Content-Type: application/json" \
    -d '{"prompt":"warmup","n_predict":1}' > /dev/null 2>&1 || true
  echo "[run-k2-horizon-4b.sh] Warm-up complete. Server ready for user requests."
fi

wait "$SRV_PID"
