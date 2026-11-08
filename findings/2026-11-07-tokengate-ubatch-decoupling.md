# TokenGate Prefill Ubatch Decoupling: 2.25x Prefill Speedup (TTFT 32.5s -> 15.2s)

**Date:** 2026-11-07  
**Model:** `Cyber-Tiel-Coder-35B-A3B-tg.gguf` (17.3 GB, NVMe SSD)  
**Hardware:** NVIDIA Jetson AGX Xavier (sm_72 Volta, 30,991 MiB unified memory)  
**Commit:** `afb1072` on `momonga-llama.cpp`  
**Status:** Adopted. TokenGate prefill ubatch decoupled to 512 while preserving `-b 32 -ub 32` for standard prefill and decode.  

---

## 1. Executive Summary

Previously, TokenGate document prefill inherited `cparams.n_batch` and `cparams.n_ubatch` from the server launch arguments (`-b 32 -ub 32`). On an 8.6k prompt (8,635 tokens), Phase 1 (`LLAMA_TG_PHASE_BOTTOM`, layers 0..3) executed 270 iterations of 32-token batches, and Phase 2 (`LLAMA_TG_PHASE_TOP`, layers 4..40 over 2,000 kept tokens) executed 63 iterations of 32-token batches. This bottlenecked Volta sm_72 tensor cores at narrow batch-32 GEMM throughput, resulting in a total prefill latency of 32.42 s (client TTFT 32.54 s).

Attempting to raise `-b 2048` globally caused decode throughput to collapse (-48%) because `server-context.cpp` submitted short prompts as single large batches, altering numerical boundaries and decoupling the MTP drafter (`findings/2026-11-07-logical-batch-b2048-negative.md`).

In this work, we decoupled the micro-batch size exclusively within `tokengate_prefill_ext()`, setting `n_batch = get_tg_ubatch()` (default 512, configurable via `TG_UBATCH`) during TokenGate prefill phases while strictly keeping `-b 32 -ub 32` for all standard prefill and speculative decode passes.

### Results
- **Document Prefill Latency (8,635 tokens, K=2000, W=2):** Dropped from **32,424.9 ms down to 14,392.1 ms (2.25x speedup)**.
- **Client TTFT:** Dropped from **32.54 s down to 15.25 s (53% latency cut)**.
- **Needle Retrieval Accuracy:** **100% exact match** (`BLUE-VELVET-99`).
- **Speculative Decode Throughput (`bench/prompt2.txt`, NPRED=400, 5 reps):** **36.96 tok/s** (vs 36.89 control).
- **MTP Draft Acceptance:** **0.96951** (vs 0.96657 control), mean length **5.68**, with **0 changed-size graph reallocations**.

---

## 2. Experimental Data

### Long Context (8,635 Tokens, `bench/prompt-8k.txt`) TTFT & Prefill Scaling

Server configuration:
`-c 32768 -fa on -ngl 99 -ncmoe 0 -t 2 -tb 2 -ctk q8_0 -ctv q8_0 -ub 32 -b 32 --poll 0 --parallel 1 -bs --tokengate --tokengate-k 2000 --tokengate-dilate 2 --tokengate-min-span 1024 --spec-type draft-mtp --spec-draft-n-max 6 --spec-draft-n-min 0 --spec-draft-p-min 0.5 --spec-draft-p-split 0.1 --load-mode none`

| Configuration | Document Prefill Latency | Client TTFT | Prompt Processing Speed | Passkey Retrieval Result |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (ubatch=32, K=2000, W=2)** | 32,424.9 ms | 32.54 s | 266 tok/s | 100% Exact Match (`BLUE-VELVET-99`) |
| **Decoupled (ubatch=512, K=2000, W=2)** | **14,392.1 ms** | **15.25 s** | **580 tok/s** | **100% Exact Match (`BLUE-VELVET-99`)** |
| **Decoupled (ubatch=512, K=1000, W=4)** | **9,407.7 ms** | **10.33 s** | **844 tok/s** | Retains `BLUE-VELV-99` |
| **Decoupled (ubatch=512, K=500, W=6)** | **7,313.5 ms** | **13.15 s** | **662 tok/s** | Retains `BLUE-VELVET` |

### Decode Stability Gate on `bench/prompt2.txt` (`NPRED=400`, 5 reps)

| Metric | Baseline Control | Decoupled Ubatch Build | Delta |
| :--- | :--- | :--- | :--- |
| **Median TG** | 36.89 tok/s | **36.96 tok/s** | +0.19% (within noise floor) |
| **Draft Acceptance** | 0.96657 | **0.96951** | +0.00294 |
| **Mean Draft Length** | 5.68 | **5.68** | 0.00 |
| **Graph Reallocations (changed size)** | 0 | **0** | Clean |

---

## 3. Architecture & Engine Changes

1. **Decoupled Batch Width in `src/llama-context.cpp`:**
   - In `tokengate_prefill_ext()`, Phase 1 loop iterates in steps of `n_batch = get_tg_ubatch()` (512) instead of `cparams.n_batch` (32).
   - Phase 2 loop iterates in steps of `n_top = get_tg_ubatch()` (512).
   - In `decode()`, relaxed `GGML_ASSERT(n_tokens_all <= cparams.n_batch)` to `n_batch_max = std::max(cparams.n_batch, get_tg_ubatch())` when `tg_prefill_phase != LLAMA_TG_PHASE_NONE`.
   - In `decode()`, set `n_ubatch = (uint32_t) get_tg_ubatch()` during prefill phases without clamping down to batch remainder, ensuring `n_ubatch > n_keep_tail` is always preserved.

2. **Batch Extractor Sizing in `src/llama-batch.cpp`:**
   - In `llama_batch_ext::llama_batch_ext(ctx)`, updated `n_tokens_max` to `std::max(llama_n_batch(ctx), get_tg_ubatch_size())` when `tokengate` is enabled, allowing 512 tokens to be added to the batch without tripping the batch size limit.

3. **Output & Scheduler Worst-Case Sizing:**
   - In `output_reserve()`, sized `n_batch` and `output_ids` to `get_tg_ubatch()` when in prefill phase, accommodating `embd_nextn` extraction for MTP state initialization.
   - In `sched_reserve()` and `memory_update()`, sized `n_tokens` to `std::max(cparams.n_ubatch, get_tg_ubatch())` when `cparams.tokengate` is enabled. This ensures compute activation buffers on sm_72 are sized up front, preventing memory reallocations during runtime.

---

## 4. Conclusion & Next Steps

Decoupling the TokenGate prefill micro-batch size successfully eliminated the primary latency bottleneck of long-context document prefill on the Jetson AGX Xavier, cutting TTFT in half without sacrificing decode speed or MTP draft acceptance.

Next steps for TTFT optimization:
- Phase 3: Evaluate quantized V cache (`-ctv q4_0` or `q8_0`) with FlashAttention mean approximation (`cparams.tg_fa_mean`).
- Phase 4: Benchmark L2 cache pinning and kernel tuning for `mul_mat_vec_q_moe` during prefill.
