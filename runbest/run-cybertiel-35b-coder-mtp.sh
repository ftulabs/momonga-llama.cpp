#!/bin/bash
# ==============================================================================
# Script Name: run-cybertiel-35b-coder-mtp.sh
# Description: Launch verified best configuration for Cyber-Tiel-Coder-35B-A3B with MTP and TokenGate.
#
# Hardware: Jetson AGX Xavier (sm_72 Volta, 30 GB unified LPDDR4x memory)
# Architecture: Cyber-Tiel-Coder-35B-A3B (MoE 35B with 41 layers, 11 Attention + 30 DeltaNet)
# Model: Cyber-Tiel-Coder-35B-A3B-tg.gguf (18.12 GB)
# Default Port: 8080
# Default Context: 32768 (Fast init; supports up to 262,144 context with 4.4 GB free headroom)
#
# Key Features & Optimizations:
#   - Speculative Multi-Token Prediction (MTP) depth 6 with KV-only catch-up (ba3e503)
#   - Trimmed draft vocabulary LM head (6ec6635, 32k top draft tokens)
#   - TokenGate auto-spanning with FlashAttention approximation (--tokengate-fa)
#   - MoE MMQ prefill enablement with average tokens-per-expert tile geometry
#   - Host flag -bs (--backend-sampling) offloading sampler to GPU
#   - Steady-state decode: 47.5 - 48.4 tok/s (median 43.97 tok/s, NPRED=600)
#   - Long-context prefill: 2,070 tok/s at 9.6k tokens (TTFT 4.79s vs 53.3s baseline, 11x faster)
#
# Usage:
#   ./run-cybertiel-35b-coder-mtp.sh              # Default 32k context on port 8080
#   ./run-cybertiel-35b-coder-mtp.sh 262144       # Full 262k long context
#   PORT=8090 ./run-cybertiel-35b-coder-mtp.sh   # Override port
#   WARMUP=0 ./run-cybertiel-35b-coder-mtp.sh    # Skip warmup request
#
# Example Query:
#   curl -X POST http://127.0.0.1:8080/completion \
#     -H "Content-Type: application/json" \
#     -d '{"prompt": "Write a Python function to compute Fibonacci numbers:", "n_predict": 128}'
# ==============================================================================
set -euo pipefail

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  sed -n "2,/^# ======/p" "$0" | sed -e "s/^# //" -e "s/^#$//"
  exit 0
fi

CTX=${1:-32768}
[ $# -gt 0 ] && shift

PORT=${PORT:-8080}
HOST=${HOST:-"0.0.0.0"}
WARMUP=${WARMUP:-1}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
BIN="${BIN:-$REPO_DIR/build-volta/bin/llama-server}"
if [ ! -x "$BIN" ]; then
  BIN=/home/triph/juicer/momonga-llama.cpp/build-volta/bin/llama-server
fi

if [ ! -x "$BIN" ]; then
  echo "[run-cybertiel-35b-coder-mtp.sh] Error: llama-server executable not found at: $BIN" >&2
  exit 1
fi

MODEL=${MODEL:-"/mnt/ssd/models/Cyber-Tiel-Coder-35B-A3B-tg.gguf"}
if [ ! -f "$MODEL" ]; then
  MODEL="/home/triph/models/Cyber-Tiel-Coder-35B-A3B-tg.gguf"
fi
if [ ! -f "$MODEL" ]; then
  MODEL="/home/triph/models/Cyber-Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS.gguf"
fi

if [ ! -f "$MODEL" ]; then
  echo "[run-cybertiel-35b-coder-mtp.sh] Error: Model file not found: $MODEL" >&2
  exit 1
fi

VOCAB_ARG=()
DRAFT_VOCAB=${DRAFT_VOCAB:-"/mnt/ssd/models/draft-vocab-32k.json"}
if [ ! -f "$DRAFT_VOCAB" ]; then
  DRAFT_VOCAB="/home/triph/models/draft-vocab-32k.json"
fi
if [ -f "$DRAFT_VOCAB" ]; then
  VOCAB_ARG=(--spec-draft-vocab "$DRAFT_VOCAB")
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
echo " Starting Cyber-Tiel-Coder-35B-A3B MTP Server"
echo " Binary:   $BIN"
echo " Model:    $MODEL"
echo " Port:     $PORT"
echo " Context:  $CTX"
echo " Draft:    MTP Depth 6 (Vocab: ${DRAFT_VOCAB:-none})"
echo " MMProj:   ${MMPROJ:-none}"
echo "================================================================="

# Ensure CUDA graphs are enabled: presence of GGML_CUDA_DISABLE_GRAPHS disables them
env -u GGML_CUDA_DISABLE_GRAPHS "$BIN" \
  -m "$MODEL" \
  -c "$CTX" -fa on -ngl 99 -ncmoe 0 -t 2 -tb 2 \
  -ctk q8_0 -ctv q8_0 -ub 32 -b 32 --poll 0 --parallel 1 \
  -bs \
  --tokengate --tokengate-k 2000 --tokengate-dilate 2 --tokengate-min-span 1024 --tokengate-fa \
  --spec-type draft-mtp --spec-draft-n-max 6 --spec-draft-n-min 0 \
  --spec-draft-p-min 0.5 --spec-draft-p-split 0.1 \
  "${VOCAB_ARG[@]}" "${MMPROJ_ARG[@]}" --load-mode none \
  --port "$PORT" \
  --host "$HOST" "$@" &
SRV_PID=$!

cleanup() {
  if kill -0 "$SRV_PID" 2>/dev/null; then
    echo "[run-cybertiel-35b-coder-mtp.sh] Shutting down llama-server (PID $SRV_PID)..."
    kill -TERM "$SRV_PID" 2>/dev/null || true
    wait "$SRV_PID" 2>/dev/null || true
  fi
}
trap cleanup INT TERM EXIT

if [ "$WARMUP" -eq 1 ]; then
  echo "[run-cybertiel-35b-coder-mtp.sh] Waiting for server to become ready on http://${HOST}:${PORT}..."
  for _ in $(seq 1 120); do
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
      echo "[run-cybertiel-35b-coder-mtp.sh] Error: server process exited unexpectedly during startup." >&2
      exit 1
    fi
    if curl -s "http://127.0.0.1:${PORT}/health" | grep -q 'ok'; then
      break
    fi
    sleep 1
  done

  echo "[run-cybertiel-35b-coder-mtp.sh] Server ready. Running startup graph warm-up..."
  curl -s "http://127.0.0.1:${PORT}/completion" \
    -H "Content-Type: application/json" \
    -d '{"prompt":"warmup","n_predict":1}' > /dev/null 2>&1 || true
  echo "[run-cybertiel-35b-coder-mtp.sh] Warm-up complete. Server ready for user requests."
fi

wait "$SRV_PID"
