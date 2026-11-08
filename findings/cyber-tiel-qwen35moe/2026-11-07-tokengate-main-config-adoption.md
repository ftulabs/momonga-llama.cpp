# Adoption of TokenGate into Main Configuration: 36.9 tok/s Decode Preserved, 3.5x-7.8x TTFT Acceleration

**Date:** 2026-11-07  
**Model:** `Cyber-Tiel-Coder-35B-A3B-tg.gguf` (17.3 GB, NVMe SSD)  
**Hardware:** NVIDIA Jetson AGX Xavier (sm_72 Volta, 30,991 MiB unified memory)  
**Status:** Adopted into main best configuration (`KNOWN-GOOD.md`, `bench/run-best.sh`).  

---

## 1. Executive Summary

TokenGate (arXiv 2610.26754) is now formally adopted into the main best configuration for `Cyber-Tiel-Coder-35B-A3B`.
By resolving the V-cache quantization constraint and decoupling LoRA requirements, TokenGate delivers up to **7.8x TTFT speedup on long contexts** while maintaining **exact 36.9 tok/s steady-state decode throughput** and **0.96657 draft acceptance** on standard interactive prompts.

### Key Headline Metrics

| Workload | Baseline (No TokenGate) | TokenGate K=2000:2 | TokenGate K=500:2 | Gain |
| :--- | :--- | :--- | :--- | :--- |
| **Decode (prompt2.txt, 400 tok)** | 36.86 tok/s (0.96657 / 5.68) | 36.89 tok/s (0.96657 / 5.68) | 37.00 tok/s (0.96657 / 5.68) | **0% penalty (identical)** |
| **Short Prefill (<1024 tok)** | 82.98 tok/s | 82.98 tok/s | 82.98 tok/s | **0% penalty (identical)** |
| **8.6k Doc TTFT (prompt-8k.txt)** | ~115 s (75 tok/s) | 33.0 s (261.8 tok/s) | **16.5 s (525.2 tok/s)** | **3.48x - 7.00x faster** |
| **16k Doc TTFT (niah_xav.jsonl)** | 82.6 s | 19.5 s | **10.5 s** | **4.24x - 7.85x faster** |
| **32k Doc TTFT (niah_xav.jsonl)** | 199.5 s (~3.3 min) | 30.9 s | **21.6 s** | **6.46x - 9.23x faster** |
| **Needle Retrieval Accuracy** | 100% | 100% | 100% | **100% Exact Match** |

---

## 2. Technical Investigation & Bottleneck Resolutions

### 2.1 The LoRA Decoupling Discovery
Earlier tests with `--tokengate-lora` caused MTP draft acceptance to collapse from 0.96657 down to 0.56597 and decode to halve (36.8 -> 18.8 tok/s) because `server-context.cpp` globally scaled the LoRA adapter on all requests.
In `task-1285`, we evaluated TokenGate completely without LoRA on `Cyber-Tiel-Coder-35B-A3B-tg.gguf` over `niah_xav.jsonl`.
Result: TokenGate achieved **100% exact needle retrieval** across all K settings (K=2000, K=500, K=100) without any LoRA adapter. LoRA was discarded.

### 2.2 Quantized V-Cache Compatibility (Commit `058ac72`)
`llama-context.cpp` originally rejected quantized V cache:
```cpp
} else if (ggml_is_quantized(params.type_v)) {
    why = "the gated layers need a non-quantized V cache";
}
```
When forced to run with `-ctv f16` or `q8_0-f16`, FlashAttention lacked compiled vector kernels for asymmetric K/V types, and dequantization caused draft acceptance to collapse to 0.641.
We patched `src/llama-context.cpp` to allow quantized V-cache when FlashAttention is active by defaulting to the FlashAttention head-mean mask form (`cparams.tg_fa_mean = true`).
Because FlashAttention is natively compiled for `q8_0-q8_0`, this allowed TokenGate to run directly on `-ctk q8_0 -ctv q8_0`, fully preserving the factory hidden state precision and MTP acceptance.

### 2.3 Backend Sampling Compatibility (Commit `bf3e047`)
`tools/server/server-context.cpp` contained an unnecessary check:
```cpp
} else if (task.params.sampling.backend_sampling) {
    why = "backend sampling";
}
```
Which silently disabled TokenGate whenever `-bs` was passed. Since TokenGate prefill skips logit sampling until the final prompt token, backend sampling is fully compatible. Removing this check enabled TokenGate to function seamlessly with `-bs`.

---

## 3. Adoption Verification

1. **Decode Stability:**
   Measured over 5 consecutive reps with `NPRED=400` on `bench/prompt2.txt`:
   - Reps 2-5: 36.98, 37.33, 36.80, 36.67 tok/s (Median: **36.89 tok/s**).
   - Draft acceptance: **0.96657**, mean draft length: **5.68**.
   - Graphs reused: 100%, 0 graph reallocations.

2. **Long-Document Verification:**
   Document prompt with 8,635 tokens (`bench/prompt-8k.txt`) with an embedded needle evaluated through `llama-server`:
   - Pruned document: 8,613 tokens -> 2,000 kept (K=2000) or 500 kept (K=500).
   - TTFT: **33.0 s** (K=2000) and **16.5 s** (K=500).
   - Generation: correctly retrieved the vault passkey.
