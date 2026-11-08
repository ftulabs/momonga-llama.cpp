# 262k Context Allocation, BeeLlama KV Comparison, and CUDA 11.4 CUB ARGSORT

**Date:** 2026-11-08
**Target:** Jetson AGX Xavier (sm_72 Volta, 30,991 MiB unified memory, CUDA 11.4)
**Model:** `Cyber-Tiel-Coder-35B-A3B-tg.gguf` (17.3 GB)
**Prompt:** `bench/prompt2.txt` (NPRED=400)

---

## 1. Context & Motivation

The user requested:
1. Investigation of how BeeLlama (`Anbeeld/beellama.cpp`) achieved 261k (262,144) context with KV cache and how to configure this setup to run 261k context reliably.
2. Exploration of fork PRs, tips, and tricks for further performance enhancements (TTFT, prefill, decode).

---

## 2. 262,144 Context Feasibility: Hybrid vs Full-Attention Models

### Why Standard Full-Attention Models OOM at 262k
On standard 32-to-40-layer full-attention models (such as MiniCPM-V-4.7):
- At `CTX=262144` with F16 KV cache: 40 layers * 262,144 * 4,096 * 2 bytes = **85.9 GB**.
- At Q8_0 KV cache: **42.9 GB**.
- At Q4_0 KV cache: **21.5 GB** ($21.5 \text{ GB KV} + 18 \text{ GB model} = 39.5 \text{ GB} > 31 \text{ GB}$).
On a 32 GB Jetson Xavier, full-attention models OOMed at 262k under both Q8_0 and Q4_0. This necessitated BeeLlama's **KVarN** (low-bit variance-normalized quantization to 2-4 bits) and **KV Cache Precision Tail** (`--kv-tail-tokens`).

### Why Cyber-Tiel-Coder Fits Easily at 262k
Cyber-Tiel-Coder-35B-A3B is a hybrid DeltaNet architecture:
- 30 layers are recurrent Gated DeltaNet (linear attention). Recurrent state size is fixed and independent of context length.
- Only **11 of 41 layers** maintain a full KV cache.
- At `CTX=262144` with `-ctk q8_0 -ctv q8_0`:
  - KV cache size is only **8,448 MiB (8.25 GiB)**.
  - Model weights occupy **17,250 MiB**.
  - Total resident RAM: **25,698 MiB (25.1 GB)** out of 30,991 MiB.
  - **Free RAM headroom:** **5,293 MiB (5.17 GB)**!

---

## 3. KV Cache Quantization Benchmark at 262,144 Context

All runs executed at `CTX=262144`, `DRAFT=6`, `NPRED=400` on `prompt2.txt`:

| Arm | KV K / V | Used RAM | Free RAM | Draft Accept | Mean Draft Len | TG tok/s (Reps) | Median TG |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **test-262k-decode** | **q8_0 / q8_0** | 26,162 MiB | **4.83 GB** | **0.97546** | **5.61** | 42.04, 43.04, 43.08 | **43.04 tok/s** |
| test-262k-k8v4 | q8_0 / q4_0 | 23,336 MiB | 7.65 GB | 0.68000 | 2.61 | 24.20, 24.77, 24.83 | 24.77 tok/s |
| test-262k-q4kv | q4_0 / q4_0 | 22,686 MiB | 8.30 GB | 0.63732 | 2.21 | 22.70, 23.21, 23.25 | 23.21 tok/s |

### Critical Finding on KV Cache Quantization
Without BeeLlama's precision tailing (`--kv-tail-tokens`), compressing the KV cache to `q4_0` causes **draft acceptance to collapse from 0.975 to 0.637** and mean draft length to drop from 5.61 to 2.21 tokens. As a result, decode speed collapses by **-46%** (43.04 -> 23.21 tok/s).

Because Cyber-Tiel-Coder only has 11 attention layers, `q8_0-q8_0` fits within memory with 4.8 GB of headroom, avoiding any degradation in draft acceptance or speed. **Q8_0 KV is the optimal configuration for 262k context on this model.**

---

## 4. CUDA 11.4 CUB ARGSORT Hardware Acceleration

### The Problem
Upstream `ggml-cuda/common.cuh` gated `GGML_CUDA_USE_CUB` behind:
```cpp
#if !defined(GGML_USE_HIP) && (defined(GGML_USE_MUSA) || CUDART_VERSION >= 11070)
#    define GGML_CUDA_USE_CUB
#endif
```
On CUDA 11.4 (`CUDART_VERSION == 11040`), CUB was disabled even though `/usr/local/cuda-11.4/include/cub` was installed. Consequently, `GGML_OP_ARGSORT` failed device support checks on device `CUDA0` and top-p sampling fell back to CPU with warnings:
`W llama_sampler_backend_support: device 'CUDA0' does not have support for op ARGSORT needed for sampler 'top-p'`.

### The Fix
1. Updated `common.cuh` to check `CUDART_VERSION >= 11040`.
2. Updated `argsort.cu` to use `DeviceSegmentedRadixSort` when `CUDART_VERSION < 11070` (since `DeviceSegmentedSort` was only introduced in CUDA 11.7, whereas RadixSort is available across all versions).
3. Committed and pushed in commit `1f33695`.

### 5-Rep Verification at CTX=262144 (`cub-262k-opt`)
- Reps: 41.73, 42.88, 42.78, 42.95, 42.95 tok/s
- **Median TG:** **42.88 tok/s**
- **Draft Acceptance:** **0.97546**
- **Mean Draft Length:** **5.61**
- **ARGSORT Warning:** **0 occurrences** (hardware GPU radix sort enabled).
