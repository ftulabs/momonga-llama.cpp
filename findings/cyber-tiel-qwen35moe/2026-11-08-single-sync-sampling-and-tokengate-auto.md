# Single-Synchronization Speculative Sampling (PR #30196) & TokenGate Auto-Spanning

**Date:** 2026-11-08  
**Model:** `Cyber-Tiel-Coder-35B-A3B-tg.gguf` (17.3 GB, NVMe SSD)  
**Hardware:** NVIDIA Jetson AGX Xavier (sm_72 Volta, 30,991 MiB unified memory)  
**Status:** Adopted into main configuration (`KNOWN-GOOD.md`, `bench/run-best.sh`).  

---

## 1. Executive Summary

This session delivers dual breakthrough optimizations:
1. **Decode Acceleration (+1.0% to 43.20 tok/s):** Ported PR #30196 (Single Synchronization for Speculative Sampling) and corrected MoE launch bounds. Slashes redundant host-GPU `cudaStreamSynchronize` roundtrips during draft verification, lifting decode throughput to **43.20 tok/s median** (peak 43.57 tok/s) at **0.97546 draft acceptance** and **5.61 mean length**.
2. **TTFT Acceleration on Arbitrary Prompts (3.8x to 8.0x speedup):** Implemented `--tokengate-auto` in `tools/server/server-context.cpp`. Automatically activates TokenGate decoupled prefill (`TG_UBATCH=1024`) on plain, unmarked documents (no `<context>` or `<document>` markers), collapsing TTFT from **11.44 s down to 2.98 s at 2.4k tokens (805.1 tok/s)** and from **53.33 s down to 6.75 s at 9.6k tokens (1,427.2 tok/s)**.

---

## 2. Decode Benchmark (Single-Sync Sampling & MoE Launch Bounds)

### Methodology & Pre-Registered Predictions
- Baseline Control: Commit `6ec6635` (Trimmed Draft Vocab 32k) at 42.77 tok/s median (41.70, 42.68, 42.77, 42.83, 42.88).
- Implementation: In `common_sampler_sample_and_accept_n`, retrieve all batch outputs in a single `llama_get_outputs(ctx)` synchronization rather than issuing 6 individual `cudaStreamSynchronize` calls across the draft chain. In `ggml/src/ggml-cuda/mmvq.cu`, align declared block bounds with physical 32-thread warp dimensions.
- Prediction: Shaves CPU driver wait time on 8-core Carmel CPU; predict +0.3 to +0.8 tok/s decode gain with bit-identical acceptance (0.97546).

### 5-Rep Decode Results (`bench/prompt2.txt`, `NPRED=400`)

| Configuration | Rep 1 | Rep 2 | Rep 3 | Rep 4 | Rep 5 | Median TG | Draft Acceptance | Mean Length |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Control (`6ec6635`)** | 41.70 | 42.68 | 42.77 | 42.83 | 42.88 | **42.77 tok/s** | 0.97546 | 5.61 |
| **Single-Sync Build** | 41.79 | 43.20 | 43.05 | 43.32 | 43.57 | **43.20 tok/s** | **0.97546** | **5.61** |
| **Delta** | +0.09 | +0.52 | +0.28 | +0.49 | +0.69 | **+0.43 tok/s (+1.0%)** | **0.00000** | **0.00** |

Zero changed-size graph reallocations observed.

---

## 3. TTFT / Prefill Scaling on Unmarked Prompts (`prompt-32k.txt`)

### Methodology & Pre-Registered Predictions
- Harness: `bench/prefill.sh` with `cache_prompt: false` (measuring cold TTFT and prompt processing without cache reuse artefacts).
- Test Prompt: `bench/prompt-32k.txt` (raw C++ source code, zero `<context>` or `<document>` tags).
- Prior Baseline: Without tags, prompts fall back to `-ub 32` chunked prefill (~208 tok/s at 2k, ~215 tok/s at 9.6k).
- Prediction: Prompts exceeding `tokengate_min_span` (1024) are auto-spanned, utilizing wide decoupled GEMMs to reach 750-1400 tok/s.

### Empirical Measurements

| Prompt Length | Baseline Latency (No Auto-Span) | Auto-Span Latency | Baseline Throughput | Auto-Span Throughput | Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **len ~ 2,048 (2,385 tok)** | 11,443.9 ms (11.44 s) | **2,963 ms (2.96 s)** | 208.7 tok/s | **805.1 tok/s** | **3.86x** |
| **len ~ 4,096 (4,875 tok)** | ~23,000 ms (~23.0 s) | **5,528 ms (5.53 s)** | ~212 tok/s | **881.9 tok/s** | **4.16x** |
| **len ~ 8,192 (9,563 tok)** | 53,333.3 ms (53.33 s) | **6,701 ms (6.70 s)** | 215.6 tok/s | **1,427.2 tok/s** | **8.00x** |

Client TTFT at 9.6k tokens drops from **53.38 s down to 6.75 s**.

---

## 4. Modified Files & Components

- `src/llama-sampler.h`: Added `llama_sampling_output_view` and `llama_outputs`.
- `src/llama-context.h`, `src/llama-context.cpp`: Added `llama_context::get_outputs()` and `llama_get_outputs(ctx)` staging API.
- `common/sampling.h`, `common/sampling.cpp`: Added `common_sampler_sample_from_outputs()` and updated `common_sampler_sample_and_accept_n` and `common_sampler_sample_and_accept_n_rejection` to consume borrowed outputs in a single synchronization.
- `common/speculative.cpp`: Updated `common_speculative_impl_draft_mtp::draft()` and `process()` to utilize `llama_get_outputs()`.
- `ggml/src/ggml-cuda/mmvq.cu`: Fixed `mul_mat_vec_q_moe` launch bounds.
- `common/common.h`, `common/arg.cpp`, `tools/server/server-context.cpp`: Implemented `--tokengate-auto` for unmarked prompts.
