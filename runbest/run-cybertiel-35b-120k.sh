#!/bin/bash
# ==============================================================================
# Script Name: run-cybertiel-35b-120k.sh
# Description: Launch 120k ultra-long context production chat server for Cyber-Tiel-35B-A3B.
#
# Hardware: Jetson AGX Xavier (sm_72 Volta, 30 GB unified LPDDR4x memory)
# Architecture: Cyber-Tiel-35B-A3B (MoE 35B hybrid, 11 Attention + 30 DeltaNet)
# Model: Cyber-Tiel-35B-A3B-Q4KM-tg.gguf (20.47 GB)
# LoRA: Cyber-Tiel-35B-A3B-tg-lora.gguf (7.9 MB adapter)
# MMProj: mmproj-Ornith-1.5-35B-BF16.gguf (861 MB Vision Projector)
# Default Port: 8800 (standard systemd service port)
# Default Context: 120000 (120,000 tokens)
#
# Key Features & Optimizations:
#   - Full 120,000 token context window on single Jetson Xavier with 5.6 GB RAM headroom
#   - Q8_0 KV cache precision (-ctk q8_0 -ctv q8_0) for lossless long-range attention
#   - MoE expert GPU offloading (-ncmoe 0, -ngl 99)
#   - Decoupled prefill ubatch (-ub 1024, -b 2048) delivering 215+ tok/s prefill
#   - Flat steady-state autoregressive decode at 27.86 - 28.7 tok/s
#   - Multimodal vision support via /v1/chat/completions with image inputs
#
# Usage:
#   ./run-cybertiel-35b-120k.sh              # Run with 120k context on port 8800
#   ./run-cybertiel-35b-120k.sh 32768        # Run with shorter 32k context
#   PORT=8082 ./run-cybertiel-35b-120k.sh   # Override port
#   WARMUP=0 ./run-cybertiel-35b-120k.sh    # Skip warmup request
#
# Example Query:
#   curl -X POST http://127.0.0.1:8800/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{"messages": [{"role": "user", "content": "Analyze system telemetry data"}], "max_tokens": 128}'
# ==============================================================================
set -euo pipefail

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  sed -n "2,/^# ======/p" "$0" | sed -e "s/^# //" -e "s/^#$//"
  exit 0
fi

CTX=${1:-120000}
[ $# -gt 0 ] && shift

PORT=${PORT:-8800}
HOST=${HOST:-"0.0.0.0"}
WARMUP=${WARMUP:-1}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
BIN="${BIN:-$REPO_DIR/build-volta/bin/llama-server}"
if [ ! -x "$BIN" ]; then
  BIN=/home/triph/juicer/momonga-llama.cpp/build-volta/bin/llama-server
fi

if [ ! -x "$BIN" ]; then
  echo "[run-cybertiel-35b-120k.sh] Error: llama-server executable not found at: $BIN" >&2
  exit 1
fi

MODEL=${MODEL:-"/mnt/ssd/models/Cyber-Tiel-35B-A3B-Q4KM-tg.gguf"}
if [ ! -f "$MODEL" ]; then
  MODEL="/home/triph/models/Cyber-Tiel-35B-A3B-Q4KM-tg.gguf"
fi

if [ ! -f "$MODEL" ]; then
  echo "[run-cybertiel-35b-120k.sh] Error: Model file not found: $MODEL" >&2
  exit 1
fi

LORA_ARG=()
LORA=${LORA:-"/mnt/ssd/models/Cyber-Tiel-35B-A3B-tg-lora.gguf"}
if [ ! -f "$LORA" ]; then
  LORA="/home/triph/models/Cyber-Tiel-35B-A3B-tg-lora.gguf"
fi
if [ -f "$LORA" ]; then
  LORA_ARG=(--lora "$LORA")
fi

MMPROJ_ARG=()
MMPROJ=${MMPROJ:-"/mnt/ssd/models/mmproj-Ornith-1.5-35B-BF16.gguf"}
if [ ! -f "$MMPROJ" ]; then
  MMPROJ="/home/triph/models/mmproj-Ornith-1.5-35B-BF16.gguf"
fi
if [ "$MMPROJ" != "none" ] && [ -n "$MMPROJ" ] && [ -f "$MMPROJ" ]; then
  MMPROJ_ARG=(--mmproj "$MMPROJ")
fi

echo "================================================================="
echo " Starting Cyber-Tiel-35B-A3B 120k Production Chat Server"
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
  -ctk q8_0 -ctv q8_0 -ub 1024 -b 2048 \
  --poll 0 --parallel 1 -bs --load-mode none \
  "${LORA_ARG[@]}" "${MMPROJ_ARG[@]}" \
  --alias "Cyber Tiel 35B A3B 120k" \
  --port "$PORT" \
  --host "$HOST" "$@" &
SRV_PID=$!

cleanup() {
  if kill -0 "$SRV_PID" 2>/dev/null; then
    echo "[run-cybertiel-35b-120k.sh] Shutting down llama-server (PID $SRV_PID)..."
    kill -TERM "$SRV_PID" 2>/dev/null || true
    wait "$SRV_PID" 2>/dev/null || true
  fi
}
trap cleanup INT TERM EXIT

if [ "$WARMUP" -eq 1 ]; then
  echo "[run-cybertiel-35b-120k.sh] Waiting for server to become ready on http://${HOST}:${PORT} (120k context init)..."
  for _ in $(seq 1 180); do
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
      echo "[run-cybertiel-35b-120k.sh] Error: server process exited unexpectedly during startup." >&2
      exit 1
    fi
    if curl -s "http://127.0.0.1:${PORT}/health" | grep -q 'ok'; then
      break
    fi
    sleep 1
  done

  echo "[run-cybertiel-35b-120k.sh] Server ready. Running startup graph warm-up..."
  curl -s "http://127.0.0.1:${PORT}/completion" \
    -H "Content-Type: application/json" \
    -d '{"prompt":"warmup","n_predict":1}' > /dev/null 2>&1 || true
  echo "[run-cybertiel-35b-120k.sh] Warm-up complete. Server ready for user requests."
fi

wait "$SRV_PID"
