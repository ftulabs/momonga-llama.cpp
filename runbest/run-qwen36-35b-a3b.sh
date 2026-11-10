#!/bin/bash
# ==============================================================================
# Script Name: run-qwen36-35b-a3b.sh
# Description: Launch verified configuration for Qwen 3.6 35B A3B with LoRA and MMProj.
#
# Hardware: Jetson AGX Xavier (sm_72 Volta, 30 GB unified LPDDR4x memory)
# Architecture: Qwen 3.6 35B A3B (MoE 35B hybrid, 11 Attention + 30 DeltaNet)
# Model: qwen36-35b-a3b-tg-g3-IQ4_XS.gguf (17.2 GB)
# LoRA: qwen36-35b-tg-g3-lora.gguf (7.9 MB adapter)
# MMProj: qwen36-35b-mmproj-BF16.gguf (861 MB Vision Projector)
# Default Port: 8081
# Default Context: 32768
#
# Key Features & Optimizations:
#   - TokenGate auto-spanning with FlashAttention approximation (--tokengate-fa)
#   - MoE MMQ prefill enablement
#   - LoRA fine-tuning adapter loaded
#   - Multimodal Vision Projector enabled
#   - Throughput: ~34.5 tok/s decode, 2,000+ tok/s long-context prefill
#
# Usage:
#   ./run-qwen36-35b-a3b.sh              # Run with default 32k context on port 8081
#   ./run-qwen36-35b-a3b.sh 65536        # Run with 64k context
#   PORT=8091 ./run-qwen36-35b-a3b.sh   # Override port
#   WARMUP=0 ./run-qwen36-35b-a3b.sh    # Skip warmup request
#
# Example Query:
#   curl -X POST http://127.0.0.1:8081/completion \
#     -H "Content-Type: application/json" \
#     -d '{"prompt": "Explain quantum computing briefly:", "n_predict": 100}'
# ==============================================================================
set -euo pipefail

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  sed -n "2,/^# ======/p" "$0" | sed -e "s/^# //" -e "s/^#$//"
  exit 0
fi

CTX=${1:-32768}
[ $# -gt 0 ] && shift

PORT=${PORT:-8081}
HOST=${HOST:-"0.0.0.0"}
WARMUP=${WARMUP:-1}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
BIN="${BIN:-$REPO_DIR/build-volta/bin/llama-server}"
if [ ! -x "$BIN" ]; then
  BIN=/home/triph/juicer/momonga-llama.cpp/build-volta/bin/llama-server
fi

if [ ! -x "$BIN" ]; then
  echo "[run-qwen36-35b-a3b.sh] Error: llama-server executable not found at: $BIN" >&2
  exit 1
fi

MODEL=${MODEL:-"/mnt/ssd/models/qwen36-35b-a3b-tg-g3-IQ4_XS.gguf"}
if [ ! -f "$MODEL" ]; then
  MODEL="/home/triph/models/qwen36-35b-a3b-tg-g3-IQ4_XS.gguf"
fi

if [ ! -f "$MODEL" ]; then
  echo "[run-qwen36-35b-a3b.sh] Error: Model file not found: $MODEL" >&2
  exit 1
fi

LORA_ARG=()
LORA=${LORA:-"/mnt/ssd/models/qwen36-35b-tg-g3-lora.gguf"}
if [ ! -f "$LORA" ]; then
  LORA="/home/triph/models/qwen36-35b-tg-g3-lora.gguf"
fi
if [ -f "$LORA" ]; then
  LORA_ARG=(--lora "$LORA")
fi

MMPROJ_ARG=()
MMPROJ=${MMPROJ:-"/mnt/ssd/models/qwen36-35b-mmproj-BF16.gguf"}
if [ ! -f "$MMPROJ" ]; then
  MMPROJ="/home/triph/models/qwen36-35b-mmproj-BF16.gguf"
fi
if [ "$MMPROJ" != "none" ] && [ -n "$MMPROJ" ] && [ -f "$MMPROJ" ]; then
  MMPROJ_ARG=(--mmproj "$MMPROJ")
fi

echo "================================================================="
echo " Starting Qwen 3.6 35B A3B MoE Server"
echo " Binary:   $BIN"
echo " Model:    $MODEL"
echo " Port:     $PORT"
echo " Context:  $CTX"
echo " LoRA:     ${LORA:-none}"
echo " MMProj:   ${MMPROJ:-none}"
echo "================================================================="

# Ensure CUDA graphs are enabled: presence of GGML_CUDA_DISABLE_GRAPHS disables them
env -u GGML_CUDA_DISABLE_GRAPHS "$BIN" \
  -m "$MODEL" \
  -c "$CTX" -fa on -ngl 99 -ncmoe 0 -t 2 -tb 2 \
  -ctk q8_0 -ctv q8_0 -ub 32 -b 32 --poll 0 --parallel 1 \
  -bs \
  --tokengate --tokengate-k 2000 --tokengate-dilate 2 --tokengate-min-span 1024 --tokengate-fa \
  "${LORA_ARG[@]}" "${MMPROJ_ARG[@]}" --load-mode none \
  --alias "Qwen 3.6 35B A3B" \
  --port "$PORT" \
  --host "$HOST" "$@" &
SRV_PID=$!

cleanup() {
  if kill -0 "$SRV_PID" 2>/dev/null; then
    echo "[run-qwen36-35b-a3b.sh] Shutting down llama-server (PID $SRV_PID)..."
    kill -TERM "$SRV_PID" 2>/dev/null || true
    wait "$SRV_PID" 2>/dev/null || true
  fi
}
trap cleanup INT TERM EXIT

if [ "$WARMUP" -eq 1 ]; then
  echo "[run-qwen36-35b-a3b.sh] Waiting for server to become ready on http://${HOST}:${PORT}..."
  for _ in $(seq 1 120); do
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
      echo "[run-qwen36-35b-a3b.sh] Error: server process exited unexpectedly during startup." >&2
      exit 1
    fi
    if curl -s "http://127.0.0.1:${PORT}/health" | grep -q 'ok'; then
      break
    fi
    sleep 1
  done

  echo "[run-qwen36-35b-a3b.sh] Server ready. Running startup graph warm-up..."
  curl -s "http://127.0.0.1:${PORT}/completion" \
    -H "Content-Type: application/json" \
    -d '{"prompt":"warmup","n_predict":1}' > /dev/null 2>&1 || true
  echo "[run-qwen36-35b-a3b.sh] Warm-up complete. Server ready for user requests."
fi

wait "$SRV_PID"
