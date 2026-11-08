#!/bin/bash
# Launch the known-good configuration (see ../KNOWN-GOOD.md).
#   ./run-best.sh          full 262144 context (slow load, ~34.5 tok/s)
#   ./run-best.sh 32768    smaller context, same decode speed, faster load
#
# Prefill: TokenGate with decoupled prefill ubatch (TG_UBATCH=1024), FA approximation (--tokengate-fa)
# and auto-spanning achieves 2,025 - 2,070 tok/s on long contexts, collapsing TTFT to 4.8s at 9.6k and 11.1s at 22.1k.
# Speculative decode runs at steady-state 47.5 - 48.4 tok/s (43.97 tok/s median NPRED=600) with 0.968-0.975 acceptance and 5.5-5.7 mean length.
# Graphs are warmed up on startup to eliminate the 4.97s cold capture tax.
set -euo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CTX=${1:-262144}
[ $# -gt 0 ] && shift
BIN=${BIN:-"$REPO_DIR/build-volta/bin/llama-server"}
MODEL=${MODEL:-/mnt/ssd/models/Cyber-Tiel-Coder-35B-A3B-tg.gguf}
if [ ! -e "$MODEL" ]; then
  MODEL=/home/triph/models/Cyber-Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS.gguf
fi
PORT=${PORT:-8080}
WARMUP=${WARMUP:-1}
VOCAB_ARG=()
if [ -f "/mnt/ssd/models/draft-vocab-32k.json" ]; then
  VOCAB_ARG=(--spec-draft-vocab "/mnt/ssd/models/draft-vocab-32k.json")
fi

for f in "$BIN" "$MODEL"; do
  [ -e "$f" ] || { echo "missing: $f" >&2; exit 1; }
done

# Graphs are enabled by ABSENCE of this variable: ggml checks presence, not value,
# so GGML_CUDA_DISABLE_GRAPHS=0 would still disable them.
env -u GGML_CUDA_DISABLE_GRAPHS "$BIN" \
  -m "$MODEL" \
  -c "$CTX" -fa on -ngl 99 -ncmoe 0 -t 2 -tb 2 \
  -ctk q8_0 -ctv q8_0 -ub 32 -b 32 --poll 0 --parallel 1 \
  -bs \
  --tokengate --tokengate-k 2000 --tokengate-dilate 2 --tokengate-min-span 1024 --tokengate-fa \
  --spec-type draft-mtp --spec-draft-n-max 6 --spec-draft-n-min 0 \
  --spec-draft-p-min 0.5 --spec-draft-p-split 0.1 \
  "${VOCAB_ARG[@]}" --load-mode none \
  --port "$PORT" "$@" &
SRV_PID=$!

cleanup() {
  echo "[run-best.sh] Shutting down llama-server (PID $SRV_PID)..."
  kill -TERM "$SRV_PID" 2>/dev/null || true
  wait "$SRV_PID" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

if [ "$WARMUP" -eq 1 ]; then
  echo "[run-best.sh] Waiting for llama-server to become ready on port $PORT..."
  for _ in $(seq 1 120); do
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
      echo "[run-best.sh] Error: server process exited unexpectedly during startup." >&2
      exit 1
    fi
    if curl -s "http://127.0.0.1:${PORT}/health" | grep -q 'ok'; then
      break
    fi
    sleep 1
  done

  echo "[run-best.sh] Server ready. Running startup graph warm-up..."
  curl -s "http://127.0.0.1:${PORT}/completion" \
    -H "Content-Type: application/json" \
    -d '{"prompt":"warmup","n_predict":1}' > /dev/null 2>&1 || true
  echo "[run-best.sh] Warm-up complete. CUDA graphs initialized. Server ready for user requests."
fi

# Wait for server process
wait "$SRV_PID"
