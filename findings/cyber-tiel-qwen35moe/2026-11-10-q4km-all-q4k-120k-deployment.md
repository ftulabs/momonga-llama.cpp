# Deployment: Q4KM-all-q4k at 120k Context with Vision Projector (28.7 tok/s Decode)

**Date:** 2026-11-10  
**Model:** `Q4KM-all-q4k.gguf` (20.46 GB, NVMe SSD) + `mmproj-Ornith-1.5-35B-BF16.gguf` (861 MB)  
**Hardware:** NVIDIA Jetson AGX Xavier (sm_72 Volta, 30,991 MiB unified memory)  
**Service:** `llama-server.service` on port 8800  
**Status:** Adopted in production service. Rock-solid 28.7 tok/s decode, zero speculation fragility, 5.6 GB free RAM headroom.  

---

## 1. Executive Summary

Following real-world interactive chat sessions where MTP draft acceptance fluctuated between 32% (under client temperature 0.7 and tool calling) and 65% (under TokenGate attention modulation), producing 13.5 to 19.5 tok/s, we transitioned the primary server deployment to the high-throughput, non-speculative re-quantized profile (`Q4KM-all-q4k.gguf`).

When allocating full 262k context alongside the 20.46 GB model, 861 MB vision projector, and wide compute buffers, total memory exceeded Xavier's 30,991 MiB limit, causing an OOM reset. Sizing context to **120,000 tokens** (`-c 120000`) with wide batching (`-b 2048 -ub 1024`) provides **5,626 MiB (5.6 GB) of safe free RAM headroom** with zero NvMap errors.

---

## 2. Verified Performance Metrics

- **Steady-State Decode:** **28.28 – 28.70 tok/s** (34.85 ms per token).
- **Temperature Invariance:** Bit-identically **28.69 tok/s** at `temp 0.0` and **28.70 tok/s** at `temp 0.7`.
- **Memory Footprint:** 25,364 MiB resident out of 30,991 MiB (**5,626 MiB available headroom**).
- **Graph Reallocations:** **0** same size / **0** changed size.
- **Multimodal Support:** `mmproj-Ornith-1.5-35B-BF16.gguf` verified loaded and active.
- **Context Length:** 120,064 allocated tokens with Q8_0 KV cache.

---

## 3. Production Configuration (`/mnt/sdcard/opt/llama/llama-server.env`)

```bash
LLAMA_BIN=/home/triph/juicer/momonga-llama.cpp/build-volta/bin/llama-server
LOAD_ARGS=--load-mode none

MODEL_PATH=/mnt/ssd/models/Q4KM-all-q4k.gguf
MMPROJ_PATH=/mnt/ssd/models/mmproj-Ornith-1.5-35B-BF16.gguf
MODEL_ALIAS=Q4KM-all-q4k

HOST=0.0.0.0
PORT=8800

CTX_SIZE=120000
PARALLEL=1
BATCH=2048
UBATCH=1024

FLASH_ATTN=on
CACHE_TYPE_K=q8_0
CACHE_TYPE_V=q8_0

SPEC_TYPE=none
SPEC_DRAFT_N_MAX=0

EXTRA_ARGS=-ncmoe 0 -t 2 -tb 2 --poll 0 -bs
```
