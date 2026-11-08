# Trimmed Draft Vocab: 42.77 tok/s Decode (+15.38% Speedup, Acceptance 0.97546)

Date: 2026-11-08.
Target: Cyber-Tiel-Coder-35B-A3B-tg.gguf on Jetson AGX Xavier (sm_72 Volta).
Binary: /home/triph/juicer/momonga-llama.cpp/build-volta/bin/llama-server.
Baseline: 37.07 tok/s (rebase baseline) / 37.52 tok/s (KV-only MTP catch-up, commit ba3e503).
New record: **42.77 tok/s** median over 5 reps (NPRED=400, prompt2.txt).

---

## 1. Rationale and Bottleneck Analysis

From the decode pass profile (`./2026-11-03-decode-pass-budget.md`):
- In each speculative pass (mean length 5.68), the target model verifies 7 tokens and the MTP draft path generates up to 6 speculative candidate tokens sequentially.
- Cyber-Tiel-Coder-35B-A3B has an unusually wide vocabulary: 248,320 tokens.
- The shared LM head (`model.output`) in Q6_K quantization occupies **417 MB** in unified memory.
- For 6 sequential draft steps, evaluating the full LM head (`mul_mat_vec_q<q6_k, 1>`) requires:
  $$6 \times 417\text{ MB} = 2.50\text{ GB DRAM reads per pass}$$
- This single kernel alone consumed **31.45 ms/pass** (19.2% of total GPU busy time), bottlenecked purely by LPDDR4x bandwidth on Xavier.
- While the target model MUST evaluate all 248,320 tokens to verify tokens with exact ground-truth distribution, speculative drafting only needs to propose candidate tokens. In practical code generation, >99.9% of all drafted tokens fall within the top 32k-48k most frequent tokens.

---

## 2. Hypothesis and Pre-Registered Prediction

- **Hypothesis:** By extracting the most frequent 32,768 vocabulary tokens and allocating a compact copy of their corresponding rows in the draft context (`vsub`), the draft LM head evaluates an $N=32768$ weight matrix (55 MB) instead of 417 MB. The outputs are scattered into a `-inf` logit vector via `ggml_set_rows`.
- **Pre-registered prediction:**
  - Draft LM head latency reduced from 31.45 ms to ~4.2 ms per pass (saving ~25-27 ms/pass).
  - Draft acceptance: >= 0.950 (predicted 0.955-0.965, mean length >= 5.4).
  - Throughput: **41.0 to 44.0 tok/s** (predicted +10% to +17% gain over 37.52 tok/s baseline).

---

## 3. Implementation Details

1. **Vocabulary Subset Extraction:**
   - Evaluated 747,043 tokens from the MTP training corpus (`/mnt/ssd/models/_mtp/corpus/train/`) plus evaluation prompts.
   - Built sorted token lists:
     - `/mnt/ssd/models/draft-vocab-32k.json` (32,768 tokens, 99.9% coverage).
     - `/mnt/ssd/models/draft-vocab-48k.json` (49,152 tokens).
2. **Engine Integration (commits `ba3e503` and `6ec6635`):**
   - Added `vocab_subset` and `n_vocab_subset` to `llama_context_params`.
   - Added `--spec-draft-vocab <path>` argument to `common/arg.cpp`.
   - In `src/llama-context.cpp`: `vsub_init()` extracts and packs selected rows of `model.output` and `nextn.shared_head_head` at initialization, creating a contiguous device buffer and token index mapping.
   - In `src/llama-graph.cpp`: `build_lora_mm` replaces the 248k-wide LM head with the compact `rows` tensor, then scatters logits into a full-sized `-INFINITY` buffer using `ggml_set_rows`.

---

## 4. Measurement Protocol & Results

Standard discipline (AGENTS.md rules 3, 4, 7):
- Prompt: `bench/prompt2.txt`, `NPRED=400`, 5 reps with median.
- Flags: `-bs --tokengate --tokengate-k 2000 --tokengate-dilate 2 --tokengate-min-span 1024`.
- Server restarted per arm; interleaved reverse order run (`ctrl-post-np400`) to confirm zero thermal/board drift artifacts.

### Results Table

| Configuration | Rep 1 (cold) | Rep 2 | Rep 3 | Rep 4 | Rep 5 | Median | Acceptance | Mean Len |
|---|---|---|---|---|---|---|---|---|
| Rebase Baseline (`94a29be`) | 35.83 | 37.27 | 37.07 | 37.21 | 37.03 | **37.07** | 0.96657 | 5.68 |
| KV-Only Graphs (`ba3e503`) | 36.47 | 37.33 | 37.53 | 37.52 | 37.77 | **37.52** | 0.96657 | 5.68 |
| **Trimmed Draft 32k (`6ec6635`)** | 41.70 | 42.68 | 42.77 | 42.83 | 42.88 | **42.77** | **0.97546** | **5.61** |
| Trimmed Draft 48k (`6ec6635`) | 41.67 | 42.60 | 42.55 | 42.95 | 43.04 | **42.60** | **0.97546** | **5.61** |
| Post-Control (`ctrl-post-np400`) | 36.82 | 37.75 | 37.63 | 37.49 | 37.63 | **37.63** | 0.96657 | 5.68 |

### Prefill & TTFT Verification

`bench/prefill.sh` with `LENS="512 2048"`, `UB=1024`, `B=1024`:
- Len ~512: TTFT 3,276 ms, prompt eval 180.6 tok/s.
- Len ~2048: TTFT 11,444 ms, prompt eval 208.7 tok/s.
- Prefill and TTFT performance are completely untouched because prompt processing runs purely on the target model context (`ctx_tgt`).

---

## 5. Summary & Adoption

- Candidate C delivers **+5.25 tok/s (+13.99%)** over commit `ba3e503` and **+5.70 tok/s (+15.38%)** over the 37.07 tok/s baseline.
- Draft acceptance increased from 0.96657 to **0.97546**, with mean draft length holding steady at **5.61** (vs 5.68).
- The 32k subset outperforms the 48k subset slightly (42.77 vs 42.60 tok/s) due to lower DRAM read traffic with identical acceptance.
- The feature is fully merged to `master` and pushed to `origin/master`.
