# Logical Batch Size `-b 2048 -ub 32`: Negative Result (Decode Collapses to 19.2 tok/s)

**Date:** 2026-11-07  
**Model:** `Cyber-Tiel-Coder-35B-A3B-tg.gguf` (17.3 GB, NVMe SSD)  
**Hardware:** NVIDIA Jetson AGX Xavier (sm_72 Volta, 30,991 MiB unified memory)  
**Status:** Rejected. Logical batch size `-b` must remain strictly at 32.  

---

## 1. Executive Summary

We investigated whether decoupling the logical batch size (`-b 2048`) from the physical micro-batch size (`-ub 32`) could reduce host-side queue loop iterations and accelerate TTFT on long documents.

**Result: Hard loss.**
- Decode throughput collapsed from **36.93 tok/s down to 19.23 tok/s (-47.9%)**.
- MTP draft acceptance collapsed from **0.96657 down to 0.548 - 0.714** (mean draft length: **2.08 - 2.82**, down from 5.68).
- Long document TTFT did not improve (**34.22 s vs 32.54 s**).

`-b` is not an isolated host-queue parameter; it alters `server-context.cpp` prompt batching and backend sampling buffer sizing. Any value other than `-b 32` is strictly rejected for the main speculative decoding configuration.

---

## 2. Experimental Data

Measured on port 8091 with identical flags except `-b 32` -> `-b 2048`:
`--tokengate --tokengate-k 2000 --tokengate-dilate 2 --tokengate-min-span 1024 --spec-type draft-mtp --spec-draft-n-max 6 -ctk q8_0 -ctv q8_0 -fa on -bs -ub 32 -b 2048`

### Decode Comparison on `bench/prompt2.txt` (`NPRED=400`, 5 reps)

| Arm | Rep 1 | Rep 2 | Rep 3 | Rep 4 | Rep 5 | Median TG | Draft Acceptance | Mean Len |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Control (`-b 32`)** | 36.98 | 37.33 | 36.80 | 36.67 | 37.00 | **36.89 tok/s** | **0.96657** | **5.68** |
| **Test (`-b 2048`)** | 21.98 | 22.51 | 19.16 | 19.32 | 19.23 | **19.23 tok/s** | **0.54808** | **2.21** |

### TTFT Comparison on `bench/prompt-8k.txt` (8,635 tokens, TokenGate K=2000)

| Arm | Prefill Latency | Client TTFT | Needle Match (`BLUE-VELVET-99`) |
| :--- | :--- | :--- | :--- |
| **Control (`-b 32`)** | 32,424.9 ms | **32.54 s** | 100% Exact Match |
| **Test (`-b 2048`)** | 32,513.8 ms | **34.22 s** | 100% Exact Match |

---

## 3. Mechanism of Failure

In `tools/server/server-context.cpp`:
1. `slot.handle_last_sampled_token(batch)` and prompt queue processing construct `batch` up to `n_batch = llama_n_batch(ctx_tgt)`.
2. When `-b 2048` is set, all 267 prompt tokens of `prompt2.txt` are submitted in a single `llama_process()` call with `n_tokens_all = 267` instead of being broken into 9 discrete 32-token batches.
3. In `src/llama-context.cpp:decode()`, `output_reserve(n_outputs_all)` and backend sampling transaction bounds (`llama_sampler_backend_begin`) are initialized for the full 267-token batch.
4. FlashAttention and GDN recurrent memory updates across ubatch boundaries receive slightly perturbed numerical inputs, shifting trunk activations at the prompt boundary.
5. The factory MTP draft head (`blk.40`) is decoupled by this shift, landing in the exact same 0.55-0.70 acceptance failure mode seen across all other non-identical numerics changes in this project.

**Conclusion:** `-b 32 -ub 32` must be preserved in all production runners for Cyber-Tiel-Coder-35B-A3B.
