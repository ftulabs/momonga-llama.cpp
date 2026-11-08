# TokenGate Cross-Model Transfer: Force-Installing Qwen3.6-35B-A3B Gate into Cyber-Tiel-Coder-35B-A3B

**Date:** 2026-11-07  
**Model:** Cyber-Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS  
**Augmented Model:** `/mnt/ssd/models/Cyber-Tiel-Coder-35B-A3B-tg.gguf` (17.3 GB)  
**Donor Gate:** `g3_gate.npz` (trained on Qwen3.6-35B-A3B, step 3002, gate layer 3)  
**Paired LoRA:** `/mnt/sdcard/models/tokengate/qwen36-35b-tg-g3-lora.gguf` (8.19 MB)  
**Evaluation:** NIAH 16k & 32k (`niah_xav.jsonl`) on Jetson AGX Xavier (Volta sm_72)  

---

## 1. Executive Summary

TokenGate (arXiv 2610.26754) trained on `Qwen/Qwen3.6-35B-A3B` was successfully force-installed into
`Cyber-Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS.gguf` using `convert_tokengate_to_gguf.py`.

Because both models share the identical underlying `qwen35moe` backbone geometry (41 blocks,
hidden size 2048, 16 full-attention heads, full attention interval = 4), the 21 gate tensors
(4.2 MB) and the paired LoRA adapter (`qwen36-35b-tg-g3-lora.gguf`) transferred with 100% architectural
and numerical compatibility.

### Headline Results

- **16k Context (16,043 state tokens):**
  - Baseline Full TTFT: **82.4 - 83.2 s** (1.00x)
  - TokenGate K=2000:2 TTFT: **19.0 - 19.6 s** (**4.22x - 4.33x speedup**, 100% needle match)
  - TokenGate K=500:2 TTFT: **10.6 - 10.7 s** (**7.76x - 7.77x speedup**, 100% needle match)
  - TokenGate K=100:2 TTFT: **8.4 - 8.5 s** (**9.70x - 9.79x speedup**, 100% needle match)

- **32k Context (32,108 state tokens):**
  - Baseline Full TTFT: **199.5 s** (3.3 minutes, 1.00x)
  - TokenGate K=2000:2 TTFT: **30.9 s** (**6.46x speedup**, 100% needle match)
  - TokenGate K=500:2 TTFT: **21.6 s** (**9.23x speedup**, 100% needle match)
  - TokenGate K=100:2 TTFT: **19.2 s** (**10.36x speedup**, 100% needle match)

Needle retrieval accuracy was **100% across all tested rows and configurations**.

---

## 2. Bug Fix in momonga-llama.cpp (Commit 91b037d)

During initial testing of `llama-tokengate`, the binary suffered a segmentation fault (`SIGSEGV`)
at token position 16082 on row 0.

### Root Cause
In `commit d7c71f5`, TokenGate was ported into `momonga-llama.cpp`, and helper `n_embd_out_ctx()`
was introduced to account for `LLAMA_CONTEXT_TYPE_TG_BOTTOM`, where each token outputs
`[a_j | raw score]` (width `n_embd + 1` = 2049 floats).

However, callers in `src/llama-context.cpp` (`get_embeddings_ith`, `get_embeddings_nextn_ith`,
`decode`, `output_reserve`, and `output_reorder`) were not updated and continued to use
`model.hparams.n_embd_out()` (= 2048). As a result:
1. `output_reserve()` allocated only `2048 * n_outputs_max` floats.
2. `decode()` copied into `embd.data` at stride 2048 instead of 2049.
3. Accessing `out[i * 2049 + 2048]` in `tokengate.cpp` at token 16082 ran out of buffer bounds.

### Fix
All callers in `src/llama-context.cpp` were updated to use `n_embd_out_ctx()`, resolving the buffer
underrun and segmentation fault completely (commit `91b037d`).

---

## 3. Evaluation Data Table

Full results recorded in `/mnt/ssd/models/cyber_tiel_tg_eval_rows012.jsonl`:

| Row | State Tokens | Config | Kept Tokens | TTFT (ms) | Speedup | Phase 1 (ms) | Phase 2 (ms) | Decode tok/s | Match | Answer / Text |
|---|---|---|---|---|---|---|---|---|---|---|
| **0** | 16,043 | full | 16,090 | 82,632.2 | 1.00x | 82,632.2 | 0.0 | 15.1 | **yes** | Eat a sandwich and sit in Dolores Park on a sunny day. |
| 0 | 16,043 | 2000:2 | 2,047 | 19,598.3 | **4.22x** | 7,983.9 | 11,613.2 | 14.3 | **yes** | eat a sandwich and sit in Dolores Park on a sunny day |
| 0 | 16,043 | 500:2 | 547 | 10,637.5 | **7.77x** | 7,404.9 | 3,232.1 | 15.8 | **yes** | eat a sandwich and sit in Dolores Park on a sunny day |
| 0 | 16,043 | 100:2 | 147 | 8,522.2 | **9.70x** | 7,464.5 | 1,057.2 | 15.9 | **yes** | eat a sandwich and sit in Dolores Park on a sunny day |
| **1** | 15,977 | full | 16,023 | 82,366.3 | 1.00x | 82,366.3 | 0.0 | 16.6 | **yes** | The best thing to do in Regina is watch the glassblowers... |
| 1 | 15,977 | 2000:2 | 2,046 | 19,040.8 | **4.33x** | 7,353.2 | 11,686.8 | 14.7 | **yes** | watch the glassblowers work by the tidal pools... |
| 1 | 15,977 | 500:2 | 546 | 10,610.4 | **7.76x** | 7,339.8 | 3,270.2 | 15.9 | **yes** | watch the glassblowers work by the tidal pools... |
| 1 | 15,977 | 100:2 | 146 | 8,416.3 | **9.79x** | 7,354.1 | 1,061.9 | 16.3 | **yes** | watch the glassblowers work by the tidal pools... |
| **2** | 32,108 | full | 32,155 | 199,454.4 | 1.00x | 199,454.4 | 0.0 | 15.1 | **yes** | Eat a sandwich and sit in Dolores Park on a sunny day. |
| 2 | 32,108 | 2000:2 | 2,047 | 30,862.0 | **6.46x** | 19,291.8 | 11,568.8 | 14.1 | **yes** | eat a sandwich and sit in Dolores Park on a sunny day |
| 2 | 32,108 | 500:2 | 547 | 21,615.3 | **9.23x** | 18,381.5 | 3,233.0 | 15.2 | **yes** | eat a sandwich and sit in Dolores Park on a sunny day |
| 2 | 32,108 | 100:2 | 147 | 19,246.4 | **10.36x** | 18,199.1 | 1,046.5 | 15.7 | **yes** | eat a sandwich and sit in Dolores Park on a sunny day |
