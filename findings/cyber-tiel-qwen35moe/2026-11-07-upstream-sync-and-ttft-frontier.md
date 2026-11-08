# 2026-11-07 Upstream Sync (127 commits) & TTFT Optimization Frontier

## 1. What was done

1. **Rebase on Upstream Master:**
   - Upstream remote `ggml-org/llama.cpp` was updated by 127 commits (`2cbca38..f7591a6`, up to tag `b12190` / `v0.9.0-dev`).
   - Fork `momonga-llama.cpp/master` rebased cleanly onto `upstream/master` `f7591a6`.
   - All fork-local patches preserved:
     - `ggml-cuda/mmvq.cu`: Volta sm_72 `MUL_MAT_ID` dispatch fix, launch bounds, dense rows-per-block, `small_k` trigger.
     - `ggml-cuda/concat.cu`: non-contiguous concatenation launch geometry.
     - `ggml-cuda/mmq.cu`: tokens-per-expert tile width rule.
     - `src/llama-batch.cpp`, `src/llama-context.cpp`: TokenGate decoupled prefill ubatch (`TG_UBATCH=1024`).
     - `tools/server/`: backend sampling (`-bs`) and TokenGate server endpoints.
     - Fork-owned docs (`README.md`, `AGENTS.md`, `JETSON-XAVIER.md`) retained.

2. **Decode Stability Gate Verified:**
   - Binary: `momonga-llama.cpp/build-volta/bin/llama-server` (commit `af4b758`).
   - Model: `/mnt/ssd/models/Cyber-Tiel-Coder-35B-A3B-tg.gguf` (17.3 GB, NVMe SSD).
   - Prompt: `bench/prompt2.txt`, `NPRED=400`, 5 reps.
   - Results:
     - Rep 1: 36.24 tok/s (cold graph capture)
     - Rep 2: 37.09 tok/s
     - Rep 3: 37.34 tok/s
     - Rep 4: 37.34 tok/s
     - Rep 5: 37.34 tok/s
     - **Median TG: 37.34 tok/s** (exceeds 36.96 tok/s baseline).
     - **Draft acceptance: 0.96657** (bit-identical).
     - **Mean draft length: 5.68** (bit-identical).
     - **Graph reallocations: 0 same-size / 0 changed-size** across all generation steps.

3. **TTFT Optimization Frontier Benchmarked (8,648 tokens on `bench/prompt-8k.txt`):**
   - Decoupled `TG_UBATCH=1024` set as default in `src/llama-context.cpp`.
   - Automated 1-token graph warm-up integrated into `bench/run-best.sh` (eliminates 4.97 s first-request tax).
   - Sweep over TokenGate parameters $K/W$:
     - $K=2000, W=2$: Document prefill **12,745 ms (678.5 tok/s)**, Client TTFT **12.87 s** (vs 32.54 s cold baseline, **2.53x speedup, 60.4% latency cut**), 100% exact needle retrieval (`BLUE-VELVET-99`).
     - $K=1200, W=3$: Document prefill **11,470 ms (754.0 tok/s)**, Client TTFT **11.59 s** (**2.81x speedup**), needle retrieved.
     - $K=1000, W=4$: Document prefill **10,474 ms (825.7 tok/s)**, Client TTFT **10.60 s** (**3.07x speedup**), needle retrieved.

---

## 2. Benchmark Summary Table

| Configuration | Rebase Base | Decode TG (tok/s) | Acceptance | Mean Len | 8.6k Prefill | 8.6k Client TTFT |
|---|---|---|---|---|---|---|
| Stock Q4_K_M (pre-MTP) | `2cbca38` | 22.25 | - | 1.00 | 110.2 s (78 tok/s) | ~111 s |
| Factory MTP IQ4_XS | `2cbca38` | 34.30 | 0.96657 | 5.68 | 108.5 s (79 tok/s) | ~110 s |
| MTP + MMVQ + `-bs` (pre-TG) | `2cbca38` | 36.93 | 0.96657 | 5.68 | 101.0 s (85 tok/s) | ~102 s |
| TokenGate coupled (`-ub 32`) | `afb1072~` | 36.96 | 0.96951 | 5.68 | 32.42 s (266 tok/s) | 32.54 s |
| **TokenGate decoupled (`TG_UBATCH=1024`) $K=2000, W=2$** | **`f7591a6`** | **37.34** | **0.96657** | **5.68** | **12.75 s (678 tok/s)** | **12.87 s** |
| **TokenGate decoupled (`TG_UBATCH=1024`) $K=1200, W=3$** | **`f7591a6`** | **37.34** | **0.96657** | **5.68** | **11.47 s (754 tok/s)** | **11.59 s** |
| **TokenGate decoupled (`TG_UBATCH=1024`) $K=1000, W=4$** | **`f7591a6`** | **37.34** | **0.96657** | **5.68** | **10.47 s (826 tok/s)** | **10.60 s** |

---

## 3. Key Conclusions

1. **Rebase Integrity:** Upstream's 127 commits introduced refactoring of `model` into `llm_graph_context` and removed `sections` from RoPE, but preserved all tensor core kernels and math. Merging them resulted in a minor throughput improvement (36.96 -> 37.34 tok/s).
2. **TTFT Frontier:** By decoupling prefill ubatch to 1024 and adding automated startup warm-up, client TTFT on 8.6k documents reached **10.60-12.87 seconds**, a **2.5x-3.1x speedup** over un-decoupled TokenGate and an **8.5x-9.5x speedup** over standard `-ub 32` prefill (102 s).
3. **Pushed & Up to Date:** `momonga-llama.cpp` on branch `master` is fully synchronized with `origin/master`.
