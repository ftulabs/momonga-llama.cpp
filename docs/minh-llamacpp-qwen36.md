# Minh's llama.cpp Model Running Configuration (Qwen3.6-35B-A3B)

Saved configuration for the background llama.cpp service before it was stopped on 2026-11-01.

---

## 1. Service Overview

- **Service Name:** `llama-server.service`
- **Unit File:** `/etc/systemd/system/llama-server.service`
- **Owner / User:** `minh`
- **Host / Port:** `http://0.0.0.0:8081`
- **Status:** Inactive (stopped via `sudo systemctl stop llama-server.service`)
- **Binary:** `/mnt/sdcard/opt/llama/bin/llama-server-tgc`
- **Model:** `Qwen3.6-35B-A3B` (`/mnt/sdcard/models/tokengate/qwen36-35b-a3b-tg-g3-IQ4_XS.gguf`)
- **Multimodal Projector:** `/mnt/sdcard/models/mmproj-BF16.gguf`

---

## 2. Full Running Command

When active, the server ran with the following exact arguments:

```bash
/mnt/sdcard/opt/llama/bin/llama-server-tgc \
  -m /mnt/sdcard/models/tokengate/qwen36-35b-a3b-tg-g3-IQ4_XS.gguf \
  --mmproj /mnt/sdcard/models/mmproj-BF16.gguf \
  --alias Qwen3.6-35B-A3B \
  --host 0.0.0.0 \
  --port 8081 \
  --jinja \
  --metrics \
  --load-mode none \
  -c 131072 \
  -ngl 99 \
  --parallel 1 \
  -b 4096 \
  -ub 1024 \
  -fa on \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --temp 0.6 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0.0 \
  --spec-type draft-mtp \
  --spec-draft-n-max 2 \
  --cache-reuse 256 \
  --reasoning off \
  --tokengate \
  --tokengate-lora /mnt/sdcard/models/tokengate/qwen36-35b-tg-g3-lora.gguf \
  --tokengate-k 2000 \
  -cram 1024 \
  --ctx-checkpoints 8
```

---

## 3. Parameter Breakdown

| Flag / Parameter | Value | Description |
| :--- | :--- | :--- |
| `-m` | `/mnt/sdcard/models/tokengate/qwen36-35b-a3b-tg-g3-IQ4_XS.gguf` | Main model weights (TokenGate G3 gate version, IQ4_XS quant) |
| `--mmproj` | `/mnt/sdcard/models/mmproj-BF16.gguf` | Vision / multimodal projector file |
| `--alias` | `Qwen3.6-35B-A3B` | Model alias name for API endpoints |
| `--host` | `0.0.0.0` | Listen IP address |
| `--port` | `8081` | Listen port |
| `--jinja` | *flag present* | Enable Jinja chat template rendering |
| `--metrics` | *flag present* | Expose Prometheus metrics endpoint |
| `--load-mode` | `none` | Equivalent to upstream `--no-mmap` |
| `-c` | `131072` | Context size (128k tokens) |
| `-ngl` | `99` | Number of GPU offloaded layers |
| `--parallel` | `1` | Number of parallel request slots |
| `-b` | `4096` | Logical batch size |
| `-ub` | `1024` | Physical micro-batch size |
| `-fa` | `on` | Flash Attention enabled |
| `--cache-type-k` | `f16` | Key KV-cache precision (required for sm_72) |
| `--cache-type-v` | `f16` | Value KV-cache precision |
| `--temp` | `0.6` | Sampling temperature |
| `--top-p` | `0.95` | Top-p sampling |
| `--top-k` | `20` | Top-k sampling |
| `--min-p` | `0.0` | Minimum probability cutoff |
| `--spec-type` | `draft-mtp` | Speculative decoding type (Multi-Token Prediction draft) |
| `--spec-draft-n-max` | `2` | Maximum speculative draft tokens |
| `--cache-reuse` | `256` | KV cache reuse chunk size |
| `--reasoning` | `off` | Disable default reasoning prefix / token preservation |
| `--tokengate` | *flag present* | Enable TokenGate context compression |
| `--tokengate-lora` | `/mnt/sdcard/models/tokengate/qwen36-35b-tg-g3-lora.gguf` | TokenGate LoRA weights |
| `--tokengate-k` | `2000` | TokenGate gate layer k parameter |
| `-cram` | `1024` | Context RAM reservation |
| `--ctx-checkpoints` | `8` | Number of context checkpoints |

---

## 4. Configuration Files

### `/etc/systemd/system/llama-server.service`

```ini
[Unit]
Description=llama.cpp server - Qwen3.6-35B-A3B MTP (Xavier)
After=network-online.target mnt-sdcard.mount
Wants=network-online.target
RequiresMountsFor=/mnt/sdcard
StartLimitIntervalSec=600
StartLimitBurst=5

[Service]
Type=simple
User=minh
EnvironmentFile=/mnt/sdcard/opt/llama/llama-server.env
ExecStart=/mnt/sdcard/opt/llama/llama-server-start.sh
Restart=on-failure
RestartSec=15

[Install]
WantedBy=multi-user.target
```

### `/mnt/sdcard/opt/llama/llama-server.env`

```bash
# Xavier AGX (MomongaOS) llama-server config - lives on the SD card so it survives momonga-ota slot switches.
# 2026-10-27: upstream llama.cpp de5122d + TokenGate (~/src/llama.cpp-tokengate, results/tokengate_server_20261027 in
# the cluster repo), sm_72, CUDA 11.4, NO_VMM; build ~/src/build-tg-srv2.sh (Ubuntu 22.04 container, gcc-10).
# Short chats decode as fast as upstream (12-19% faster than the turboquant build); a document between <context>
# markers of >= 3072 tokens is pruned by the G3 gate (k = 2000): TTFT 4.9x faster at 16k, 7.3x at 32k.
# Rollback: llama-server.env.bak-20261027-turboquant (the start script defaults to the turboquant binary and flags).
LLAMA_BIN=/mnt/sdcard/opt/llama/bin/llama-server-tgc
# upstream renamed --no-mmap
LOAD_ARGS=--load-mode none
# the prod GGUF plus the G3 gate (21 tensors)
MODEL_PATH=/mnt/sdcard/models/tokengate/qwen36-35b-a3b-tg-g3-IQ4_XS.gguf
MMPROJ_PATH=/mnt/sdcard/models/mmproj-BF16.gguf
MODEL_ALIAS=Qwen3.6-35B-A3B
HOST=0.0.0.0
PORT=8081
CTX_SIZE=131072
PARALLEL=1
BATCH=4096
UBATCH=1024
FLASH_ATTN=on
# K must stay f16 on sm_72 (q8_0 K + FA gave garbage in the turboquant fork); the gated layers need a non-quantized V
CACHE_TYPE_K=f16
CACHE_TYPE_V=f16
SPEC_TYPE=draft-mtp
# MTP draft 2 beat 3 and 4 (2026-10-24 sweep)
SPEC_DRAFT_N_MAX=2
# per request: "tokengate": false | {"k": N, "fa": true} (fa = flash attention in the gated layers, ~10% faster decode,
# NIAH 10/10 but not yet through the full eval)
EXTRA_ARGS=--cache-reuse 256 --reasoning off --tokengate --tokengate-lora /mnt/sdcard/models/tokengate/qwen36-35b-tg-g3-lora.gguf --tokengate-k 2000 -cram 1024 --ctx-checkpoints 8
```

### `/mnt/sdcard/opt/llama/llama-server-start.sh`

```bash
#!/bin/sh
# Xavier llama-server wrapper (systemd). Config: /mnt/sdcard/opt/llama/llama-server.env
# LLAMA_BIN and LOAD_ARGS default to the turboquant build and its flags (llama-server.env.bak-20261027-turboquant).
set -eu
BIN=${LLAMA_BIN:-/mnt/sdcard/opt/llama/bin/llama-server}
VLM_ARGS=""
[ -n "${MMPROJ_PATH:-}" ] && [ -f "$MMPROJ_PATH" ] && VLM_ARGS="--mmproj $MMPROJ_PATH"
SPEC_ARGS=""
[ "${SPEC_TYPE:-none}" != "none" ] && SPEC_ARGS="--spec-type $SPEC_TYPE --spec-draft-n-max ${SPEC_DRAFT_N_MAX:-3}"
# shellcheck disable=SC2086  # word-splitting of *_ARGS is intended
exec "$BIN" -m "$MODEL_PATH" $VLM_ARGS --alias "$MODEL_ALIAS" \
  --host "$HOST" --port "$PORT" --jinja --metrics ${LOAD_ARGS:---no-mmap} \
  -c "$CTX_SIZE" -ngl 99 --parallel "$PARALLEL" -b "$BATCH" -ub "$UBATCH" \
  -fa "$FLASH_ATTN" --cache-type-k "$CACHE_TYPE_K" --cache-type-v "$CACHE_TYPE_V" \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 \
  $SPEC_ARGS ${EXTRA_ARGS:-}
```

---

## 5. Controlling the Service

To start the service again:
```bash
sudo systemctl start llama-server.service
```

To view service logs:
```bash
journalctl -u llama-server.service -f
```

To stop the service:
```bash
sudo systemctl stop llama-server.service
```
