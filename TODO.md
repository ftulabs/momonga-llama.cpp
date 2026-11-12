# TODO

Current best on varied text: **43.97 tok/s decode (steady-state tg_3s: 47.5 - 48.4 tok/s)**,
(Cyber-Tiel 35B-A3B MTP IQ4_XS on `/mnt/ssd/models/Cyber-Tiel-Coder-35B-A3B-tg.gguf`,
`--spec-draft-n-max 6 --spec-draft-p-min 0.50 -ub 32 -b 32 -ctk q8_0 -ctv q8_0 -bs --spec-draft-vocab /mnt/ssd/models/draft-vocab-32k.json`,
fork `master` rebased on `upstream/master` `f7591a6`, plus GPU draft sampling offload (`common/speculative.cpp`), sm_72 Volta MMVQ patches,
64-register launch bounds on the IQ3_S MoE kernel, dense rows-per-block/carveout/`small_k`,
`concat_non_cont` geometry, decoupled TokenGate prefill ubatch (`TG_UBATCH=1024`) with auto-spanning (`--tokengate-auto`), FlashAttention (`--tokengate-fa`), MoE MMQ prefill enablement, automated startup warm-up,
KV-only MTP draft catch-up (`ba3e503`), trimmed draft vocabulary LM head (`6ec6635`), corrected MoE bounds, and single-sync sampling (PR #30196)).
Document prefill runs at **2,025 - 2,070 tok/s** on long contexts (TTFT **4.79 s** at 9,562 tokens, **10.92 s** at 22,127 tokens; 1,715 tok/s with 100% exact needle retrieval).
Full 262,144 context verified with 4.4 GB RAM headroom.
Acceptance **0.968 - 0.975**, mean length **5.53 - 5.75**, and **0 graph reallocs**.

**Decode is at the floor; prefill is the open front (2026-11-04).** Every remaining decode item is
under 1% and the record has not moved since `-bs`. Prompt processing, measured properly for the
first time today, was **71-79 tok/s** - only 2x decode per token, ~5% of the int8 dp4a ceiling -
so TTFT at a 9,563-token prompt was **134.6 s**. **It is 79-86 tok/s and 120.2 s as of H3 (below).** Baseline, code audit and ranked hypotheses:
[`findings/cyber-tiel-qwen35moe/2026-11-04-prefill-starting-point.md`](findings/cyber-tiel-qwen35moe/2026-11-04-prefill-starting-point.md).
The campaign is the section immediately below.

Exact launch command and expected numbers: [`KNOWN-GOOD.md`](KNOWN-GOOD.md). Run it with
`bench/run-best.sh`.

**If MTP compatibility is dropped, the whole width family reopens (2026-11-05, corrected 16:45).**
With `--spec-type none` the ubatch no longer touches decode (18.23-18.35 across 32-8192, inside the
A/A floor) and becomes free prefill. **The width that measures reproducibly is 2048, not 3072**:
at 2,828 tokens `-ub 2048` reads 243-250 tok/s and repeats to 0.2-2.9%, while 3072 reads 165.6 /
172.4 / 200.4 / 218.9 over four identical loads (32%) and 4096 reads 159.2 / 273.4 (72%). The
profile is now **27.86 tok/s decode / 215.8 tok/s prefill / 53.3 s TTFT at 11,484 tokens** with
the all-q4k file, against 83.8 / 137.1 s at `-ub 32` under MTP. Four requant steps since the
Q4_K_M file, each measured A/B/B/A: 22.25 -> 27.86 tok/s (+25.2%) for +1.67% perplexity. Record and launch lines:
[`findings/cyber-tiel-qwen35moe/2026-11-05-mtp-free-campaign.md`](findings/cyber-tiel-qwen35moe/2026-11-05-mtp-free-campaign.md) sections
18-23 and `KNOWN-GOOD.md` "The MTP-free profile". A width must not exceed the prompt length (see
"Closed - Do not re-test").

---

## Dedicated Systemd Background Services (Completed 2026-11-12)

- **Target:** Host `embeddinggemma-2` and `zerank-2` simultaneously as dedicated, resilient background `systemd` services on Jetson AGX Xavier.
- **Status:** **Completed, Enabled & Verified Live**.
- **Services Deployed:**
  - `llama-embedding.service`: EmbeddingGemma-2 Q8_0 + Vision mmproj on port `8086` (`/v1/embeddings`), 1.0 GB RSS memory, ~31 ms query latency.
  - `llama-rerank.service`: zerank-2 Q8_0 Causal Generative Reranker on port `8012` (`/rerank`), 2.9 GB RSS memory, 177 ms/doc latency.
  - Co-residency verified: Both running simultaneously with under 4 GB combined RSS (~7.3 GB VRAM), leaving 22+ GB free RAM headroom.
  - Files: Unit definitions in `/etc/systemd/system/`, scripts and env configs in `/mnt/sdcard/opt/llama/` and tracked in `momonga-llama.cpp/systemd/`.

---

## Unified `runbest/` Production Suite (Completed 2026-11-10)

- **Target:** Create a turnkey, unified `runbest/` directory containing self-contained, hardware-tuned launch scripts for every verified model and workload profile on the Jetson AGX Xavier.
- **Status:** **Completed & Verified in Production**.
- **Delivered Runners (`runbest/`):**
  - `run-zerank-2.sh`: zerank-2 Q8_0 causal generative reranker (Port 8012, 177.87 ms/doc, 7.80 tok/J).
  - `run-cybertiel-35b-coder-mtp.sh`: Cyber-Tiel-Coder-35B-A3B MoE MTP Depth 6 + TokenGate + Vision (Port 8080, 48 tok/s decode, 2,070 tok/s prefill).
  - `run-cybertiel-35b-120k.sh`: Cyber-Tiel-35B-A3B-Q4KM 120,000-token context production chat server with LoRA + Vision (Port 8800, 28 tok/s flat decode).
  - `run-qwen36-35b-a3b.sh`: Qwen 3.6 35B A3B base MoE model with LoRA + Vision (Port 8081).
  - `run-qwen38-27b.sh`: Qwen 3.8 27B Dense Coder (64 layers, non-MoE, Port 8083).
  - `run-deepseek-v4-9b.sh`: DeepSeek-V4-Pro 9B Q8_0 High-Precision Reasoning (Port 8084).
  - `run-spark-4b.sh`: Spark-X2.6 4B Edge Model (Port 8085, <80ms decode latency, <5 GB VRAM).
  - `run-embeddinggemma-2.sh`: EmbeddingGemma-2 Multimodal Vector Embeddings (Port 8086).
  - `list-models.sh`: Quick CLI model reference and port matrix.
  - `README.md`: Exhaustive operational documentation and curl verification commands.

---

## zerank-2 Reranker Optimization (Completed 2026-11-10)

- **Target:** `zeroentropy/zerank-2-reranker` (4.0B dense Qwen3 architecture, Q8_0 GGUF).
- **Status:** **Completed & Verified in Production**.
- **Results:**
  - **177.87 ms median doc latency** (1,780 ms per 10-document query).
  - **223.7 tokens/sec**, **5.61 docs/sec**.
  - **100.0% Top-1 accuracy** on cross-domain multi-document evaluation.
  - **Edge Economic Objective:** 28.73 W, 7.80 Tokens/Joule, 0.195 Docs/Joule (ServerBench).
- **Key Changes Implemented:**
  - Fixed bogus softmax score bug (~1e-31) in `src/llama-graph.cpp` when `cls_out` is absent.
  - Sliced target token 9454 (`"Yes"`) row view via `ggml_view_2d` from `tok_embd`, eliminating 160,000x LM head compute/buffer waste (~400 MB DRAM read and 640 MiB buffer saved per pass).
  - Broadened sm_72 Volta equality checks across `common.cuh`, `fattn.cu`, `mmvq.cu`, and `ggml-cuda.cu`, unlocking Volta HMMA Tensor Core FlashAttention.
  - Integrated official ZeroEntropy ChatML template auto-formatting and sigmoid probability calibration ($\sigma(\text{logit}/5.0)$).
  - Removed arbitrary `n_batch = n_ubatch` clamping for causal rerankers in `server.cpp` and `common.cpp`.
  - Pinned `-c 8192` to avoid 48 GB OOM on Xavier.
  - Script: `bench/run-rerank.sh`.

---

## Prefill / TTFT Marathon (active, 2026-11-04)

**Score to beat. Updated 2026-11-04 by the adopted MoE tile-width change** (H3, below): prefill,
cache off, `bench/prefill.sh` protocol. The old line - 7,491 / 31,468 / 134,638 ms at
78.8 / 75.8 / 71.0 tok/s - is kept here as the pre-change baseline every H3 arm was measured
against.

| prompt | TTFT (now) | tok/s now | tok/s before H3 |
|---|---|---|---|
| 590 tokens | 6,776 ms | 87.1 | 78.8 |
| 2,385 tokens | 28,307 ms | 84.3 | 75.8 |
| 9,563 tokens | 120,231 ms | 79.5 | 71.0 |

Measured at the known-good decode config (`-ub 32 -b 32`, `-fa on`, q8_0 KV, MTP on). Reproducible
to 0.14% over 3 reps **within one server load**.

**Read the absolute level as a range, not a number (2026-11-05).** With the same binary and flags,
n_prompt 9,563 has measured **92.0, 79.5 and 76.5 tok/s** in different server loads, i.e. TTFT
**104-125 s**. Storage (a cold page cache costs 5.1%), GPU/EMC clock (pinned at 1,377 MHz, MAXN),
host load and the machine's pure-load DRAM rate (93.1 GB/s, identical in the 92.0 and 76.5 states)
have all been eliminated, so the drift is inside the served process - most likely where the 18 GB
model allocation physically lands - and it is not a lever this project can pull. **Only
same-server-load, interleaved A/B deltas are quotable, and a prefill delta below the session floor
(0.56% at 9.5k, 1.6% at 2,385, 2.1% at 590) is not evidence.** See
`findings/cyber-tiel-qwen35moe/2026-11-05-prefill-page-cache.md` section 3c. Harness: `bench/prefill.sh` - `cache_prompt:false` on every rep, `n_predict 1`,
client-side TTFT, `prompt_n` recorded, rep 1 reported separately. **Do not use `bench/one.sh` for
prefill**: its PP column is cache-contaminated (AGENTS.md rule 4), and every prefill number in this
repo before 2026-11-04 came from that path or from a single cold first request.

### Ranked, with predictions written down before the run

| # | Experiment | Prediction | Cost | Status |
|---|---|---|---|---|
| H1 | `-ub`/`-b` 32 -> 64 / 96 / 128 / 256 / 512. At 32 columns MMQ has almost no width to amortise tile loads, and a 9.5k prompt is 299 launches. | predicted 150-280 tok/s (2-4x) and TTFT 35-65 s at 9.5k | 1 load + 3 lengths per arm, ~10-15 min per arm | **prefill at 512: 162 / 186 / 170 tok/s = 2.05x / 2.45x / 2.39x, TTFT at 9.5k 134.6 s -> 56.3 s**, inside the predicted band. **Decode at 512 is a hard loss: 36.86 -> ~20.0 tok/s, acceptance 0.96657 -> 0.65882, mean len 5.68 -> 2.25.** So 512 is a TTFT-profile choice, not the launch line. **The width curve is now measured on four points and it is a direct trade:** prefill 1.00x / 1.36-1.48x / 1.80-1.93x / 2.05-2.45x at ub 32 / 128 / 256 / 512, against acceptance 0.96657 / 0.92258 / 0.57655 / 0.65882. Prefill is smooth in width; acceptance is not - and 256 being *worse* than 512 means a discrete decision is being lost, not a kernel degrading. `ub128` is also unstable rep to rep (18.7-30.8 tok/s). **Closed on all six widths, and 64 is the clearest negative:** +4-7% prefill (inside the drift floor) for -10% decode and acceptance 0.96657 -> 0.91592, so strictly dominated. 96 reads 0.68122, 64 reads 0.91592. **The configuration stays at `-ub 32 -b 32`.** Acceptance is not monotonic in width (0.96657 / 0.91592 / 0.68122 / 0.92258 / 0.57655 / 0.65882), which points at MMQ's exact-match `J` specialization rather than at width itself - see H3 |
| H2 | Per-kernel prefill budget via `nsys` on one prefill (`ncu` cannot fit alongside the model). Split MMQ vs `gated_delta_net` vs FA vs cuBLAS. | if GDN is >= 40% of prefill, H1's gain saturates below its prediction and H4 becomes the only large lever. Falsified if MMQ dominates | one traced run | **DONE 2026-11-04, and the prediction above is the falsified branch: MMQ dominates, not GDN.** 590-token prefill, 6,569.8 ms of kernel time at the production `-ub 32`: **`mul_mat_q` 5,786.4 ms = 88.1%**, `gated_delta_net` 214.0 = 3.3%, ssm_conv 78.6 = 1.2%, cuBLAS f32 67.2 = 1.0%, FlashAttention 60.7 = 0.9%, concat+cpy+fwht 100.9 = 1.5%, quantize 58.9 = 0.9%, dense MMVQ 53.7 = 0.8%. MMQ by type: **IQ3_S 3,473 = 52.9%**, IQ4_XS 1,508 = 23.0%, Q8_0 815 = 12.4%, Q6_K 175 = 2.7%, Q3_K 106 = 1.6%. At `-ub 512` (2,836.3 ms) the mix is entirely different: MMQ 17.2%, dense MMVQ 26.5%, cuBLAS INT8 14.2%, dequant 9.3%, GDN 5.1%, FA 3.1%. Mechanism: `MMQ_DP4A_MAX_BATCH_SIZE = 64` and `should_use_mmq()` returns `n_tokens < 64` when fp16 MMA exists (`cc >= VOLTA` is true on sm_72), so **the MoE is one fused MMQ launch per tensor per layer below 64 tokens and a per-expert loop above it**. Full detail in `findings/cyber-tiel-qwen35moe/2026-11-04-prefill-budget.md` |
| H3 | **DONE 2026-11-04, and it is the campaign's win: the MoE tile width.** At `-ub 32` each of 256 experts receives about one token of 32, but `J_best` is chosen from the token count, so every MoE launch runs a 32-column tile and 31/32 of its dp4a work is discarded (`vec_dot` has no column bound; only the write-back honours `tile_y_max_j`). The `ncols_opt = ceil(ne12*n_expert_used/ne02)` rule the code already applies on RDNA3/RDNA4 now runs on NVIDIA, giving J=8: **87.1 / 84.3 / 79.5 tok/s pooled over every mode-1 run against a control's 79.4 / 75.5 / 71.0** at 590 / 2,385 / 9,563 tokens (the first arm read 84.1 / 83.7 / 79.6), TTFT 134.6 s -> 120.2 s, with acceptance **0.96657** and mean len **5.68** identical line for line. The pre-registered +33-58% prediction was wrong in magnitude and in mechanism: 4x less dot work bought 12.6%, because the IQ3_S gate/up launch reads at only ~57 GB/s against this board's 109.7-112 GB/s ceiling (gate and up are separate launches of 115 MB each, not one fused launch of 231 MB), so neither the dot nor the bandwidth was the limit. Two negatives in the same file: a 100% shared-memory carveout is inert at 9.5k (+0.4%), and narrowing the tile for IQ3_S only is **1.49% slower** at 9.5k in both orders against a 0.56% A/A floor. Full detail, including the retired '4x weight re-read' theory (`jt*J >= col_diff` returns before loading) and the session's prefill floor, in `findings/cyber-tiel-qwen35moe/2026-11-04-prefill-moe-tile-width.md` | ~~+5-15% on the `J=32` config~~ - delivered +9 to +12% prefill with decode bit-identical | 1 hook + rebuild + the standard both-orders protocol with `DECODE_REPS=5` | **adopted, default-on in `mmq.cu`** (`GGML_CUDA_MMQ_MOE_TILE_AVG=0` disables it) |
| H4 | Chunked GDN for Volta. `gated-delta-net-mma.cu` holds a chunked delta-rule algorithm gated to Ampere+ **and its tiles are bf16/Ampere-only**, so this is a rewrite of the schedule (Volta has no bf16 MMA; sm_70 `mma` is m8n8k4), not a gate flip. The 30 GDN layers currently run a token-serial loop. | ~~the only lever that could reach 1000+ tok/s prefill~~ - **retracted 2026-11-04: that was an unmeasured assumption. H2 puts the entire delta-net block at 393 ms of a 6,570 ms prefill (6.0%), and the core `gated_delta_net` kernel at 214 ms (3.3%). Making it infinitely fast buys +6% prefill.** Days-to-weeks of schedule work for that ceiling | high | **closed as a campaign lever**; do not start without a different reason |
| H5 | FA prefill schedule (upstream issue #28037 split-D). Volta already takes the MMA-F16 kernel in prefill (`fattn.cu:704-712`), and only 11 of 41 layers attend. | ~~+3-8% prefill at best~~ - **measured 0.9% of prefill at `-ub 32` (60.7 ms of 6,570), so the absolute ceiling is +0.9%, not +3-8%** | medium | **closed** |

| H6 | **Intermediate ubatch widths, 33-63.** The width sweep measured 32/64/96/128/256/512 and closed, but 64 is also the `should_use_mmq()` threshold (`n_tokens < 64`), so every width below it keeps the MoE in the same fused-MMQ engine while amortising the expert weight read over more tokens per ubatch. The fused MoE launch reads all 256 experts' weights per ubatch at `-ub 32` where each expert gets about one token, so traffic per token should scale as 1/ub: MoE is 67.7% of prefill, so the prediction is `0.677/ratio + 0.323`, i.e. **+17% at 40, +29% at 48, +51% at 63**. | predicted +8-20% prefill at ub 48 (86-102 tok/s at 9.5k against a same-session control of 79-86) with acceptance **bit-identical at 0.96657 / mean len 5.68**; falsified if either moves | 4 interleaved arms in both orders, ~45 min | **DONE 2026-11-05 - closed, no free point exists.** `-ub 48` reads **87.0 tok/s at 11,484 tokens against a control's 78.4 (+11.0%)**, TTFT 146.6 s -> 132.0 s, and costs **acceptance 0.96657 -> 0.57955, mean len 5.68 -> 2.10, decode 36.76 -> 18.70 tok/s (-49%)**, reproduced exactly in both orders. The prediction that acceptance would stay bit-identical because the MMQ engine does not change is **falsified** - chunking changes the hidden state the f32 router sees, hence the top-8 - and the rest of the range is no better: gate scans read **36 = 0.88779 / 4.09** and **40 = 0.70755 / 2.99**, non-monotone against 64's 0.91592. What it did deliver is a validated traffic model, `distinct_experts(ub)/ub`, which predicted the +11% inside the corrected +10-16% band. Detail in `findings/cyber-tiel-qwen35moe/2026-11-05-prefill-ubatch-intermediate.md` |
| H7 | **TokenGate sparse prefill cross-model transfer (2026-11-07).** Force-installed G3 gate (trained on Qwen3.6-35B-A3B) and paired LoRA into Cyber-Tiel-Coder-35B-A3B. Fixed stride/buffer bug in `llama-context.cpp` (commit `91b037d`). | predicted 2.5-4x TTFT speedup on long context | 16k & 32k NIAH eval (`niah_xav.jsonl`) | **DONE 2026-11-07 - verified, 4.2x-6.5x speedup at K=2000, 7.8x-9.2x at K=500, 9.7x-10.4x at K=100 with 100% needle retrieval accuracy.** At 16k context (16,043 tokens), TTFT drops from 82.6 s (baseline) to 19.6 s (K=2000:2, 4.22x), 10.6 s (K=500:2, 7.77x), and 8.5 s (K=100:2, 9.70x). At 32k context (32,108 tokens), TTFT drops from 199.5 s (~3.3 min baseline) to 30.9 s (K=2000:2, 6.46x), 21.6 s (K=500:2, 9.23x), and 19.2 s (K=100:2, 10.36x). All arms achieved exact needle match (`match: yes`). Detail in `findings/cyber-tiel-qwen35moe/2026-11-07-tokengate-force-install-results.md` |
| H8 | **TokenGate Adoption into Main Best Config & Decode Preservation (2026-11-07).** Adopt TokenGate into main best config (`KNOWN-GOOD.md`, `bench/run-best.sh`). Added quantized V-cache support (`q8_0`) with FlashAttention `fa_mean` (commit `058ac72`) and fixed backend sampling `-bs` (commit `bf3e047`). | Target: Preserve 36.9 tok/s decode and 0.96657 acceptance on standard prompts while retaining 3.5x-7.8x TTFT speedup on long documents | 5 reps on `prompt2.txt` (NPRED=400) + 8.6k doc evaluation | **ADOPTED 2026-11-07: Verified 36.89 tok/s decode (reps 2-5: 36.98, 37.33, 36.80, 36.67), 0.96657 acceptance, 5.68 mean length, 0 graph reallocs.** Long document TTFT on 8.6k prompt achieves 33.0 s (K=2000) and 16.5 s (K=500) vs 115 s baseline. Model is `/mnt/ssd/models/Cyber-Tiel-Coder-35B-A3B-tg.gguf`. Detail in `findings/cyber-tiel-qwen35moe/2026-11-07-tokengate-main-config-adoption.md` |
| H9 | **Logical batch size decoupling `-b 2048 -ub 32` (2026-11-07).** Attempted to decouple `-b` from `-ub` to reduce host queue loop overhead. | predicted 1.2-1.5 s TTFT reduction at identical decode | 5 reps decode on `prompt2.txt` + 8.6k TTFT | **CLOSED 2026-11-07, negative result: -47.9% decode.** Draft acceptance collapsed from 0.96657 down to **0.548**, decode fell 36.9 -> **19.23 tok/s**, and 8.6k TTFT did not improve (34.22 s vs 32.54 s). `-b` controls prompt chunking in `server-context.cpp`, shifting numerical boundaries before MTP. Configuration must strictly remain `-b 32 -ub 32`. Detail in `findings/cyber-tiel-qwen35moe/2026-11-07-logical-batch-b2048-negative.md` |
| H10 | **TokenGate prefill ubatch decoupling to 512 (2026-11-07).** Decouple TokenGate bottom and top prefill phases to 512-token ubatches (`TG_UBATCH=512`, commit `afb1072`) while keeping `-b 32 -ub 32` for standard prefill and decode. | predicted 2-3x TTFT reduction at identical decode (36.9 tok/s, 0.96657 acc) | 5 reps decode on `prompt2.txt` + 8.6k prompt TTFT | **ADOPTED 2026-11-07: 2.25x prefill speedup, TTFT 32.5 s -> 15.2 s at 8.6k tokens.** Document prefill latency dropped 32,424.9 ms -> **14,392.1 ms** (580 tok/s), Client TTFT dropped 32.54 s -> **15.25 s**, with **100% needle retrieval accuracy** (`BLUE-VELVET-99`) at $K=2000, W=2$. Speculative decode throughput preserved at **36.96 tok/s** with **0.96951 acceptance** and **5.68 mean length** (0 changed-size graph reallocs). Detail in `findings/cyber-tiel-qwen35moe/2026-11-07-tokengate-ubatch-decoupling.md` |
| H11 | **Upstream sync (127 commits) & TTFT Optimization Frontier (`TG_UBATCH=1024`, warm-up) (2026-11-07).** Rebased on upstream master `f7591a6`. Set default `TG_UBATCH=1024` (commit `af4b758`) and implemented automated 1-token startup warm-up in `bench/run-best.sh`. | Target: verify rebase passes decode gate and reach sub-13s TTFT on 8.6k prompt | Decode gate on `prompt2.txt` (NPRED=400, 5 reps) + 8.6k prompt sweep | **ADOPTED 2026-11-07: Decode gate passed at 37.34 tok/s (0.96657 acc, 5.68 mean len, 0 reallocs). 8.6k prompt prefill: K=2000, W=2 TTFT 12.87 s (678 tok/s, 100% exact needle); K=1200, W=3 TTFT 11.59 s; K=1000, W=4 TTFT 10.60 s (826 tok/s, 3.07x speedup over un-decoupled TG). First-request 4.97s cold capture tax eliminated.** Detail in `findings/cyber-tiel-qwen35moe/2026-11-07-upstream-sync-and-ttft-frontier.md` |
| H12 | **KV-Only Graphs for MTP Draft Catch-Up (2026-11-08).** Skip attention and MoE FFN computation during MTP draft catch-up when `n_outputs == 0`, storing only K and V. (Commit `ba3e503`). | predicted +0.3-0.5 tok/s (+1%) with bit-identical acceptance | 5 reps decode on `prompt2.txt` (NPRED=400) | **ADOPTED 2026-11-08: +1.21% decode speedup (37.07 -> 37.52 tok/s).** Measured reps: 36.47, 37.33, 37.53, 37.52, 37.77. 100% bit-exact draft acceptance (0.96657) and mean length (5.68). Merged and pushed to `origin/master`. |
| H13 | **Reduced Draft Vocabulary LM Head (2026-11-08).** Trim MTP draft LM head evaluation to top 32k tokens (`--spec-draft-vocab /mnt/ssd/models/draft-vocab-32k.json`), slashing 6 sequential draft LM head DRAM reads from 2.50 GB to 330 MB per pass. Target verifies full 248k vocab. (Commit `6ec6635`). | predicted 41.0-44.0 tok/s (+10-17%) with acceptance >= 0.950 | 5 reps decode on `prompt2.txt` (NPRED=400) + prefill verification + post-control | **ADOPTED 2026-11-08: +13.99% decode speedup (37.52 -> 42.77 tok/s).** Measured reps: 41.70, 42.68, 42.77, 42.83, 42.88. Draft acceptance increased 0.96657 -> **0.97546**, mean length **5.61**. Interleaved post-control re-test confirmed baseline at 37.63 tok/s. Total speedup over rebase baseline: **+15.38% (37.07 -> 42.77 tok/s)**. Detail in `findings/cyber-tiel-qwen35moe/2026-11-08-trimmed-draft-vocab.md`. |

### TTFT levers that are not kernels

- **First-request graph capture: 5.21 s**, measured (12,703 ms vs 7,491 ms warm at 590 tokens).
  Nothing today avoids it; a warm-up request at load time would move it out of the user's TTFT.
- **Model load from emmc: ~5-6 min** at 88 MB/s with `--load-mode none`. An NVMe is being
  installed; do not spend board time on this until it lands.
- **`cache_prompt: true` for follow-up turns.** A chat turn after a cached prefix has a TTFT of
  hundreds of ms, not 135 s. That is the case most UIs present, and it is already the default;
  the 135 s figure is the uncached case.

### Rules for this campaign

1. Every arm reports **prefill at 3 lengths, decode tok/s, acceptance and mean len**. A prefill
   win that costs decode is not a win - the decode score is 36.93 at 0.96657 / 5.68.
2. Prefill is reproducible to 0.14%, so 3 reps is enough and the both-orders protocol of rule 7 is
   not the binding constraint here. Spend the time on arms.
3. Do not change weights. Any quant change still breaks draft acceptance (AGENTS.md section 7),
   and that constraint did not get weaker because the metric changed.
4. Record the `prompt_n` the server reports; the fixture is ~4 chars/token, not exactly 4.

---

## What Has Been Discovered & Proven (Measured Realities)

1. **Hardware is fully saturated, not idle:**
   - Xavier LPDDR4x sequential GPU read ceiling measured at **109.7 - 112.0 GB/s** (`bench/bwtest.cu`).
   - Decode moves **35 - 39 GB/s** (~32-35% of achievable bandwidth).
   - Running two concurrent streams (`--parallel 2`) drops aggregate throughput by 21% (31.20 -> 24.64 tok/s).
   - System is pinned at maximum clocks: 8 Carmel CPU cores at 2.26 GHz, GPU at 1377 MHz (99% load), EMC memory controller at 2133 MHz (`nvpmodel` MAXN). No frequency or power headroom remains.
   - The bottleneck is latency, not DRAM bus saturation - and not pure instruction throughput
     either. Counters (2026-11-03) put IQ4_XS at 84% of the ceiling and IQ3_S at 51%, with
     `long_scoreboard` the top stall in both. See item 2.

2. **The core bottleneck: IQ3_S decode is slower per byte than every other type (label corrected 2026-11-03 - not ALU-bound, but latency-bound at low occupancy):**
   - *Correction:* "ALU-bound" was inferred from timing alone. `ncu` counters on the same kernel
     put `long_scoreboard` (waiting on a load) at 3.05 warps per issue-active cycle against
     `math_pipe_throttle` at 1.16, and show occupancy at 34.9% with a 37.5% cap set by 79
     registers/thread (IQ4_XS: 60 regs, 50% cap, 46.2% achieved). Real DRAM read is 56.7 GB/s
     for IQ3_S against 92.7 for IQ4_XS, i.e. 51% vs 84% of the 111 GB/s ceiling. The numbers
     below are `opbench` end-to-end and still stand; only the diagnosis changed. See
     `findings/cyber-tiel-qwen35moe/2026-11-03-counters-on-xavier.md`.
   - In `Cyber-Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS.gguf`, `ffn_moe_gate` and `ffn_moe_up` are IQ3_S across all 41 layers (82 ops/pass).
   - Measured with `opbench` at 6 tokens (256 experts, 8 used, K=2048, N=512):
     - **IQ3_S:** 20.62 MB read, **0.580 ms**, **37.3 GB/s**.
     - **IQ4_XS:** 25.50 MB read, **0.388 ms**, **68.9 GB/s**.
     - **Q4_0:** 27.00 MB read, **0.418 ms**, **67.8 GB/s**.
     - **Q8_0:** 51.00 MB read, **0.658 ms**, **81.3 GB/s**.
   - IQ3_S reads **19% fewer bytes** than IQ4_XS and takes **49% longer**.
   - Mechanism: For every 32 activations, `vec_dot_iq3_s_q8_1` performs 8 dependent gathers into `iq3s_grid[512]` in `__device__` memory plus sign fixups. This table tax accounts for **~15.7 ms of the 164.7 ms pass (9.5%)**.

3. **The MoE batch cliff was the dispatch table (SOLVED 2026-11-03):**
   - Mainline checked `cc == GGML_CUDA_CC_VOLTA` (700); Xavier is 720 (sm_72), silently falling into `pascal_older` caps (IQ3_S cap 4, IQ4_XS cap 5). Batches >=5 fell onto MMQ (2.5x slower).
   - Fixed in `mmvq.cu` by putting token on `blockIdx.z` and checking arch as a range.
   - Allowed scaling draft depth from `n-max 3` to `n-max 6`, unlocking **31.50 -> 34.49 tok/s** (+9.5%).

4. **Post-hoc GGUF requantization is closed (Three failed builds):**
   - Requantizing gate/up experts to IQ4_XS sped up isolated plain decode (+5%), but collapsed MTP draft acceptance from a bit-stable 0.98940 down to 0.64-0.93.
   - Cause: The pre-trained MTP head (block 40) is calibrated against base-model activations (blocks 0-39). Altering base weights decouples the drafter. Speculative throughput fell from 31.5 to 25.8-30.5 tok/s.

5. **Server runtime configuration space is exhausted:**
   - 28 flag configurations swept across batch sizes, KV quantizations, threads, speculative samplers, SWA, and residency modes. All non-harmful flags sit within +-1% noise.

---

## Active Frontiers & Potential Speed Increases (The Path Beyond 34.5 tok/s)

**The pass budget is now measured, not inferred** (`findings/cyber-tiel-qwen35moe/2026-11-03-decode-pass-budget.md`,
nsys on a real server with graphs on). One pass = 41 main layers + 6 MTP drafts + 7 lm_head
calls, 163.4 ms busy for 4.89 tokens: MoE experts 63.2 ms (38.7%), **lm_head 41.2 ms (25.2%)**,
dense q8_0 attention/ssm 32 ms (19.6%), GDN 6.4 ms, router 3.2 ms. `output.weight` is q6_k and
417 MB, and it is read **7 times per pass**, so the lm_head alone moves 2.9 GB - more than the
entire main forward reads. Any frontier that does not touch the MoE experts or the lm_head is
working on at most a fifth of the pass.

**Corrected 2026-11-03** (`findings/cyber-tiel-qwen35moe/2026-11-03-mmwq-launch-params.md`): a per-signature query
over the *whole* trace divides by the pass count and mixes in prompt processing, which made
`mul_mat_q` (MMQ, prefill only) look like the expert matmuls. Measured strictly between two
lm_head verify launches, i.e. one real decode pass, **there is no `mul_mat_q` in it** - the
experts run `mul_mat_vec_q_moe`, as the dispatch patch intended. The authoritative decode
budget is MoE **61.7 ms (40%)**, lm_head **41.1 ms (27%)**, dense q8_0 **29.9 ms (19.4%)**, then
`gated_delta_net_cuda` 6.44, f32-via-cuBLAS 4.06, `concat_non_cont` 2.49, `cpy_scalar` 2.09,
`quantize_q8_1` 1.68, embeddings 1.66. Compute any future per-kernel budget per pass, never
per trace.

Target: **40+ tok/s** (requires pass latency <= 142 ms at unchanged acceptance, >=14% reduction).

| Frontier | Area | Mechanism | Expected Pass Latency | Projected Speed | Status / Requirements |
|---|---|---|---|---|---|
| **Frontier 1** | CUDA Kernel | Stage `iq3s_grid` (2 KB) in Shared Memory per block in `mul_mat_vec_q_moe` | ~148 - 150 ms (-12 to -15 ms) | ~38.0 - 38.5 tok/s (+8-10%), not reachable | **CLOSED 2026-11-03, negative.** Upper-bound probe (`iq3s_grid[0]`, reverted): 356.13 -> 258.69 us and 21.63 -> 13.38 MB DRAM, but it is contaminated - `qs[]` exists only to index the grid, so pinning the index makes 64 of the block's 116 bytes dead and the compiler deletes them. Nothing done to the gather can reach that number. Infeasible on its own terms too: 2 KB/block x 32 one-warp blocks/SM = 64 KB, so the carveout would have to halve L1 in a kernel whose L1 hit rate is why it is fast. See `findings/cyber-tiel-qwen35moe/2026-11-03-occupancy-on-iq3s.md` |
| **Frontier 2** | Memory Layout | MoE layout linearisation (Upstream PR #31047) | ~142 - 146 ms (-12-14%) | **~39.0 - 40.5 tok/s** (+13-17%) | **CLOSED 2026-11-04, not applicable to decode.** PR #31047 is `CUDA: linearize MoE data layout for MMQ/MMF` and touches `mmid.cu` plus the MMQ/MMF kernels only; its own body claims 2-4% from the new layout and the rest from smaller tiles, i.e. it is a *prefill* change. Decode never enters that path: `ggml_cuda_mul_mat_id` hands `dst->ne[2] <= get_mmvq_mmid_max_batch(type, cc)` to `ggml_cuda_mul_mat_vec_q`, and after the 2026-11-03 sm_72 dispatch fix that returns `MMVQ_MAX_BATCH_SIZE` (8) on Xavier, so all 6-7 column MoE work is `mul_mat_vec_q_moe`. The overhead the PR removes does not exist on our path either - per-launch time is linear in tokens with a 0.006 ms intercept, and there is no per-expert tile-assignment or fixup kernel to elide. The PR is also still **open** upstream (created 2026-10-23, not merged), so it is not a stable cherry-pick. The earlier cautions stand and were the clue: expert slices are already contiguous 3.6 MB forward reads, and IQ3_S reaches 56.7 GB/s where IQ4_XS reaches 92.7 on the same shape, so the gap is inside the IQ3_S dot |
| **Frontier 3** | Speculation | Adaptive MTP draft depth (Upstream PR #27210) | Variable (~120 - 170 ms) | **~36.0 - 38.0 tok/s** (+5-10% on varied text) | **CLOSED 2026-11-04, the region it targets is measured negative here.** #27210 is a counting state machine (climb after N clean verifies, drop on accumulated miss pressure) whose own author says `--spec-draft-n-max` must be **>= 7**, preferably 8+, and suggests 12; the gain comes from climbing *above* the fixed depth on easy text. On this board the marginal column costs 15-17 ms (10-11% of the pass) and depth 7 already measures **34.11 vs 34.49 at depth 6** (-1.1%) with depth 8 untested because it is strictly worse in the same direction. Mean len is already 5.68 of a 6 ceiling, so there is at most 5.6% of tokens left above it, against a 10% latency step for each extra column. The drop half is equally dead: `--spec-draft-p-min 0.5` already truncates low-confidence chains, and the floor of 3 the PR defaults to is worth about -15% on its own. Effort was 2-3 days of porting a 15-file, 521-line open PR onto this base for a measured-negative region. **Do not port it** |
| **Frontier 4** | Model Weights | Native BF16-to-IQ4_XS quant + co-trained MTP draft head | ~120 - 125 ms (-25%) | **~45.0 - 47.0 tok/s** (+30-35%) | Blocked: requires BF16 source checkpoint + draft head training |
| **Frontier 5** | Attention / Prefill | SM70 GQA pairing (PR #31525) & FA 256/256 config (P2b) | Prefill TTFT -15-25% | Marginal on decode (<1%), +20% TTFT | Open: prefill / long-context only |
| **Frontier 6** | Model Weights | `output.weight` q6_k -> q5_k (417 MB -> 350 MB, -16.2%) | -- | -- | **CLOSED 2026-11-03, -44.2%.** 34.82 -> 19.43 tok/s, acceptance **0.96657 -> 0.63603**, mean_len 5.68 -> 2.20. The file was verified first (753 tensors, only `output.weight` retyped, 8 tensors hashed byte-identical), so the quant is fine - the **drafter** is the constraint. This is the fourth distinct route to acceptance 0.63-0.72, which establishes that the MTP head needs numerics *identical* to the target: the same rule kills q4_k, `--override-tensor`, and uniform requants. Only a **bit-preserving** repack survives. See `findings/cyber-tiel-qwen35moe/2026-11-03-lmhead-q5k.md` |
| **Frontier 9** | CUDA Kernel | Dense `mul_mat_vec_q`: rows per block, shared-memory carveout, `small_k` trigger | ~168 -> ~155 ms | **36.56 tok/s, +4.3%** | **CLOSED 2026-11-03, WIN.** 35.07 -> 36.25 -> 36.56, acceptance 0.96657 and mean len 5.68 unchanged throughout. The same 4-rows lever applied at ncols 1 took the 1-column lm_head from 4.776 to 4.552 ms and added another +0.85% (predicted 36.5-36.6, measured 36.56). L1 load sectors -24% at the 1-column lm_head and -38% at the 7-column verify, at unchanged occupancy - the dense path was not at a hardware wall, it was reading `y` once per output row and paying for a 16.38 KB default carveout. Costs +3.4% on the n=512 shape (grid 128 blocks is too small for 4 rows). See `findings/cyber-tiel-qwen35moe/2026-11-03-dense-mmvq-rows-per-block.md` |
| **Frontier 7** | CUDA Kernel | `rows_per_cuda_block` for the dense MMVQ kernels: the 2 KB activation vector `y` is re-read once per output row, which at 7 columns is *more* L1 traffic than the weights | - | **-46%, no gain** | **CLOSED 2026-11-03.** Tested: `rpb` 2 -> 1 at ncols 5-8 took the lm_head 7-column shape from 9.57 to 13.95 ms. Halving rpb doubles the activation re-read, and occupancy is not the binding constraint at that shape - L1 throughput is. `rpb=4` is worse (shared reduction buffer 3584 B/block caps residency at 4). See `findings/cyber-tiel-qwen35moe/2026-11-03-mmwq-launch-params.md` |
| **Frontier 8** | CUDA Kernel | Raise the f32 MMVF threshold `ne11 <= 3` -> 8 so the 7-column router leaves cuBLAS | ~161.5 ms | -- | **CLOSED 2026-11-03, -45.6%.** 34.82 -> 18.93 tok/s (5 reps, medians). Not a speed regression: draft acceptance 0.96657 -> 0.67717 median (0.590 / 0.933 / 0.675 / 0.677 / 0.677) and mean len 5.68 -> 2.40. The f32 matmul's summation order changes the top-8-of-256 router selection, which decouples the MTP drafter. **Do not change MoE router numerics** - this also rules out `GGML_CUDA_CUBLAS_COMPUTE_TYPE`, TF32 settings and `--override-tensor` on `ffn_gate_inp`. See `findings/cyber-tiel-qwen35moe/2026-11-03-mmwq-launch-params.md` section 5 |

**Read the projections above against the counters, not the other way round.** Two of the rows
state mechanisms that a later measurement contradicts, and they have not been re-derived:

- **Frontier 1** assumes the `iq3s_grid` gather latency can be staged away. The `ncu` counters
  say the IQ3_S expert kernel is at 51% of the DRAM ceiling with `long_scoreboard` as its top
  stall, so the gather is real - but it is also an *L1 throughput* problem (895 sectors per block
  against 49 of payload). Shared memory moves the load off L1TEX but does not remove the
  traffic, so +8-10% is not supported by any measurement yet.
- **Frontier 2's premise is already disproven.** It says scattered expert reads "cap memory
  efficiency at 35-41 GB/s"; `AGENTS.md` section 7 records scattered vs contiguous reads at
  **108 vs 112 GB/s, about 4%**. Linearising the layout cannot pay 13-17% if the gather costs 4%.
- **Frontier 4** is the only one that reaches 45 tok/s, and it needs a BF16 checkpoint and
  draft-head training, i.e. it is not a kernel change at all.

The honest position after the 2026-11-03 counters: **every remaining lever is 2-3%, and the two
biggest buckets (MoE experts at 51% of ceiling, lm_head at 80 GB/s on 83% of its calls) are close
to the hardware limit for these quant types.** 40 tok/s needs a smaller or cheaper weight format,
not better launch parameters.

*Revision, later the same day (Frontier 9).* That paragraph was too final. The dense path was
issuing **15.6x more L1 sectors than it needed** and running on a shared-memory carveout nobody
chose, and fixing that was worth +3.4% - more than any single window above. The lesson is that
"L1-bound" describes where the time goes, not whether the traffic is necessary. Two levers of the
*Revision 2, same day.* The `rows_per_cuda_block` lever at ncols 1 is now **done** as well: it
added 36.25 -> 36.56. That leaves **one** bit-preserving lever of this kind, the
**sector-aligned padded block layout** - except that one was measured and **closed on 2026-11-03**:
padding q8_0 34 -> 64 B and q6_K 210 -> 224 B does cut L1 load sectors 37% (127.2M -> 80.7M on
the 1-column q6_K lm_head, DRAM unchanged) but doubles registers 48 -> 95 and halves occupancy,
for 18-26% slower kernels, and it costs +1.83 GB of file. See
`findings/cyber-tiel-qwen35moe/2026-11-03-padded-block-layout-probe.md`. With that closed there is no remaining change
that cannot move a number, aside from the `rows_per_block` value at ncols 2-4, which this workload never runs (the
batch is always 1 or 7). Everything that touches weights or the router numerics stays closed at
-44% to -46%.

---

## Round 2 Frontiers (2026-11-03, ranked fastest-first)

Input: a second review of the pass. Its numbers first, because three of them are not what the
trace says.

**Corrections to the supplied 155.4 ms budget.** The measured one is 159.1 ms busy / 170.6 ms
wall (`findings/cyber-tiel-qwen35moe/2026-11-03-moe-share-in-situ.md`):

| Supplied | Measured | Note |
|---|---|---|
| MoE 61.7 ms (39.7%) | 63.98 (40.1%) | agrees |
| lm_head 32.5 (20.9%) | 37.72 (23.7%) | six 1-column draft calls 28.84, verify 8.88 |
| dense Q8_0 17.0 | 24.82 (+ lm_head above) | both are `mul_mat_vec_q` |
| GDN 10.8 | 7.52 `gated_delta_net` + 2.55 concat + 2.10 `cpy_scalar` | close enough |
| router/embeddings/norms/quantize ~18 | 3.26 cuBLAS + 3.79 quantize/copy + 8.54 misc | the supplied figure double-counts |
| graph launch / GPU idle ~15 (10.0%) | not readable from a trace | an unprofiled pass measures 162 ms against 163.4 ms busy, so inter-pass gaps are a tracing artefact |

The MoE and the lm_head are both *larger* shares than the document assumed, and the 15 ms
"idle" is not a target - it is not there when you do not trace.

**Correction to Target 1 (multi-warp MoE blocks).** The 50% occupancy wall is the register file,
not the 32-blocks-per-SM limit. `mul_mat_vec_q_moe` spends 64 registers/thread at one warp per
block: 32 warps * 32 threads * 64 regs = 65536 = the whole register file. Packing four warps into
a block gives 8 blocks * 4 warps resident = the same 32 warps = the same 50%. Multi-warp blocks
buy nothing unless registers drop below 64 or k is split across warps, and a k-split changes the
summation order - the change class that took the router from cuBLAS to `mul_mat_vec_f` and cost
-45.6% through acceptance. Ranked last of the kernel targets; any attempt is measured on
acceptance as well as kernel time.

| Rank | Frontier | Expected | P(win) | Effort | Why this rank |
|---|---|---|---|---|---|
| ~~**R1**~~ | ~~GDN glue: `concat_non_cont` launch geometry~~ | ~~+0.9-1.4 tok/s~~ | - | - | **DONE 2026-11-04, +0.29%.** 83.2 -> 46.5 us per launch, 2.505 -> 1.398 ms/pass, 36.525 -> 36.630 pooled over 18 reps in both orders. See `findings/cyber-tiel-qwen35moe/2026-11-04-gdn-concat-geometry.md` |
| **R1b** | GDN glue, the rest: 210 `cpy_scalar` rollback snapshots (2.10 ms) + a shared-memory tiled concat (~1.1 ms of traced time) | +0.2-0.5 tok/s | 0.5 | 1-2 days | needs a fused op or a new state layout - no single `ggml_cpy` view can express the reversed slot order. **New concrete route (2026-11-04):** the concat kernel already reads exactly the data the 7 snapshots copy (old state + new tokens), so a fused "concat + write the 7 rollback slots" kernel removes 686 KB/layer of re-reads (~1 ms/pass) *and* the 210 launches with their 201 stream syncs per pass, keeping the values bit-identical. That is a kernel plus a graph restructure in `delta-net-base.cpp`, not a one-line view trick |
| ~~**R2**~~ | ~~1-column q6_K lm_head occupancy~~ | ~~+0.4-1.3~~ | - | - | **CLOSED 2026-11-04 by `ncu`, not by a benchmark: the kernel is not occupancy-limited in any way a launch parameter fixes.** At the model geometry it already reads at **78.95% of the memory SOL** (397.85 MB in 4.54 ms, 87.6 GB/s) with register-limited occupancy of 31.25% (theoretical) / 29.28% (achieved) and the top stall `lg_throttle` at 37.5% of issue cycles - warps are queued on the L1 *accepting* load instructions, not waiting for data (`long_scoreboard` is not the lead). The traffic lever is already spent (rows per block = 4 at one column, the +0.85% change) and the next one (8 rows) spills; the padded block layout was measured 2026-11-03 at 18-26% slower. Absolute headroom is 4.54 -> 3.59 ms, and reaching it needs a lower-traffic block layout, not more warps. See `findings/cyber-tiel-qwen35moe/2026-11-04-lmhead-1col-occupancy.md` |
| ~~**R3**~~ | ~~gate `rows_per_cuda_block` on `nrows_x > 512`~~ | ~~+0.1~~ | - | - | **CLOSED 2026-11-04 as below the measurement floor, not as a failure.** The dense `mul_mat_vec_q` launches in a decode pass, measured from the trace (gridX 128 = n=512, one launch per pass each): q8_0 7-col at gridX 2048 is 11.19 ms/pass and 8828 us for the q6_K verify, but the shapes the gate would protect are gridX 512 (6.86 ms), 1024 (4.27), 128 (2.54), and the 3.4% regression is on the n=512 one - **0.09 ms/pass, 0.06%**. Five reps cannot resolve that, and the change needs a second template parameter (rpb is `constexpr` inside the kernel). Not worth a build cycle |
| **R4** | 7-column q6_K verify: 8.88 ms, 47 GB/s, 25% occupancy, 128 regs | +0.3-0.8 | 0.4 | 1 day | same shape as R2 but only 5.6% of the pass, and R2's `ncu` read says the same lever class is not the constraint |
| **R5** | MoE multi-warp blocks (their Target 1) | +0-1.2 | 0.25 | 1-2 days | register file caps it; k-split risks acceptance |
| **R6** | adaptive MTP depth (their Target 3, PR #27210) | +0-2.0 | 0.2 | 2-3 days | port of a newer PR onto a 2022 base |
| ~~**R7**~~ | ~~`GGML_CUDA_GRAPH_CHUNK_SIZE`~~ | ~~+0-2%~~ | - | - | **CLOSED 2026-11-04 as a refuted positive.** The one-way scan (5 reps) read 36.29 at the 512 default against 36.71 at 128 (+1.16%), which looked like a mechanism; the interleaved both-orders run (5 reps x2 per value) reads **36.56 (512) / 36.61 (128) / 36.55 (64)** - a 0.11% spread, i.e. nothing. Acceptance and mean len bit-identical throughout. Also established that captures are *not* per-pass (364 capture-pairs over 91 pass boundaries, clustered, so decode replays a cached graph and there is no first-chunk recording to save). The whole-graph endpoint (0 and 4096) read -1.3% in the one-way scan and is not confirmed; do not set it to 0. The 7.14 ms/pass of idle is therefore still unattributed and is not a chunking artifact - a future attempt needs a CPU-side profile of the speculative loop, not a kernel trace. That profile was run 2026-11-04 and the answer is R8. See `findings/cyber-tiel-qwen35moe/2026-11-04-graph-chunk-size.md` |
| **R8** | Host side | The speculative loop's CPU cost: 41.7 ms of host work per 156.5 ms pass, of which ~7.8 ms is exposed as GPU idle | +1.04% taken, +0-0.9 left | 0.5 | partly done | **Measured 2026-11-04, the profile the earlier audits asked for.** `perf stat` task-clock 2937 ms per 11.21 s request (0.262 CPUs) = 41.7 ms/pass at ~6 ms per graph launch; `tegrastats GR3D_FREQ` median 95% confirms ~7.8 ms/pass of real idle, matching the nsys 7.14 ms. Three host clusters: the full 248320-wide logits row scanned and copied on the host per sampler call (18%), per-call graph fusion search + scheduler split (~9%, paid 7x per pass), allocator/mutex noise (~5%). **First lever taken: `-bs`, +1.04%** (36.550 -> 36.930, 16 reps per arm, both orders, acceptance and mean len identical); it is now in `bench/run-best.sh` and `KNOWN-GOOD.md`. **Still open: the per-call graph preparation cluster**, which has no flag and no cache - `ggml_cuda_can_fuse`/`try_fuse`/`ggml_can_fuse_subgraph_ext` plus `ggml_backend_sched_split_graph` re-run on every one of the 7 launches in a pass for an identical graph structure. That needs a change in `ggml-backend.cpp`/`ggml-cuda.cu` (key a cached split/fusion on the graph's tensor topology), not a knob. **Re-profiled with `-bs` on 2026-11-04:** instructions fall 57.5 -> 37.4 M per pass (-34%) and `common_sampler_sample` leaves the symbol table entirely, which is the mechanism confirmed; the graph-prep cluster is now ~11% of host cycles (4.5 ms of the ~42 ms/pass) and the honest value of the remainder is **<0.5%**, so it stays open and unstarted. The follow-up also produced a measurement rule: compare **instructions or cycles**, never task-clock milliseconds, across sessions - `-bs` cut instructions 34% while milliseconds did not move, because the implied core clock read 1.14-1.22 GHz against 1.385 GHz. See `findings/cyber-tiel-qwen35moe/2026-11-04-cpu-side-of-a-pass.md` and `findings/cyber-tiel-qwen35moe/2026-11-04-backend-sampling.md` |

Additions not in the supplied list, for the record: `quantize_q8_1` runs once per MMVQ op
(~2 ms/pass) and could be folded into its producer; `k_get_rows_float_vec` is 43.5 us per launch
for a 7-token gather (latency-bound, ~1.3 ms/pass); deduplicating the ~5 of 56 expert slices that
repeat inside one verify pass would save ~9% of the MoE traffic (~5.8 ms) but needs a device-side
sort. None of these rank above R1.

**R1 landed 2026-11-04 (+0.29%).** What it turned out to be: `concat_non_cont` was launched with
one 256-thread block per destination row while only `ne0`=10 threads had work, so 2.1M threads
moved 81,920 elements; packing rows into `blockDim.y` gives 83.2 -> 46.5 us and 2.505 -> 1.398
ms/pass at bit-identical acceptance. The `cpy_scalar` 210 launches are the 7 rollback snapshots
per GDN layer that `n_rs_seq = draft.n_max` requires; `n_rs_seq` cannot be lowered (partial
acceptance is the normal case, so every pass would take the checkpoint path) and the slots cannot
be one `ggml_cpy` because slot `j` must hold the state `j` tokens back while the source window
advances forward. That is now R1b. Also unmeasured: ~9.0 ms of a traced pass is not kernel time,
concentrated in one 3.57 ms gap between two `rms_norm` launches and six 481 us gaps between
`get_rows` and `rms_norm` (one per MTP draft step) - that shape says CPU work between graph
launches, but gaps in a traced run are not trustworthy, so it needs an untraced method first.

---

### Detailed Breakdown of Active Frontiers

### Frontier 1: Stage `iq3s_grid` in Shared Memory (Target: ~38 tok/s)
- **Bottleneck:** `ffn_moe_gate` and `ffn_moe_up` are IQ3_S, executing 82 times per forward pass. At 6 tokens, IQ3_S runs at 37.3 GB/s (0.580 ms) vs IQ4_XS at 68.9 GB/s (0.388 ms). *Updated 2026-11-03 after the occupancy fix:* the same pair now reads 0.467 ms (46.4 GB/s) against 0.409 (65.4) on `opbench`, and 352.58 us against 284.51 us kernel-only under `ncu`, where real DRAM read is 61.3 GB/s against 94.0 (55% vs 85% of the 111 GB/s ceiling).
- **Counter evidence (2026-11-03, `ncu`, kernel-only):** `long_scoreboard`, i.e. waiting on a
  load, is the top stall on IQ3_S at 3.05 warps per issue-active cycle, ahead of
  `math_pipe_throttle` at 1.16 - the gather latency this frontier targets is real and dominant.
  Occupancy was 34.9% against a 37.5% cap, because 79 registers/thread capped it at 24 one-warp
  blocks per SM. *That part is fixed:* Frontier 1a took it to 64 registers and 47.9% occupancy,
  which recovered 7.6% of kernel time and 2.25% end to end, so what remains here is the gather
  latency itself. See `findings/cyber-tiel-qwen35moe/2026-11-03-counters-on-xavier.md` and
  `findings/cyber-tiel-qwen35moe/2026-11-03-occupancy-on-iq3s.md`.
- **Mechanism:** `iq3s_grid` currently resides in `__device__` global memory (`static const __device__ uint32_t iq3s_grid[512]`). Threads in a warp make divergent lookups, stressing the L1 data cache.
- **Hypothesis:** Staging the 2 KB table into `__shared__ uint32_t s_grid[512]` inside `mul_mat_vec_q_moe` allows lookups to hit 32-bank Shared Memory without L1 line contention. 32 threads cooperatively load 512 values in 16 iterations (`512 / 32 = 16` coalesced 32-bit loads) at block entry, followed by `__syncwarp()`.
- **Prediction (recorded 2026-11-03, before the run):** If IQ3_S throughput reaches ~55-60 GB/s, gate/up drops from 0.580 ms to ~0.40 ms. 82 ops x 0.18 ms savings = ~14.7 ms cut from the 164.7 ms pass. At mean_len 5.68, throughput reaches ~38.0 tok/s. Measured against the counter baseline the same arithmetic is 0.382 -> ~0.233 ms per kernel, 82 ops, ~12.2 ms of a 164.7 ms pass, i.e. ~34.3 -> ~37.0 tok/s. A partial win scales linearly: shared-memory staging that only halves the stall moves nothing, because the kernel is also register-capped.
- **Falsifier:** If the 16-load staging overhead per block outweighs bank-conflict savings on 1-warp blocks (producing <1% delta), the cost is not table gather latency. Second falsifier, added from the counters: if `long_scoreboard` falls but time does not, the ceiling was the 37.5% occupancy cap, and the next step is registers (79 -> 64), not the table.

### Frontier 1a: raise IQ3_S occupancy with `__launch_bounds__` (tested first, 2026-11-03)
- **Why before Frontier 1:** the counters say the differentiator between IQ3_S and IQ4_XS is not
  L1 gather traffic - the two kernels issue 10.99M and 11.02M L1 load sectors, within 0.2% of
  each other - it is occupancy (34.9% vs 46.2%) and instruction count (10.41M vs 6.40M).
- **Change:** `mul_mat_vec_q_moe` is declared
  `__launch_bounds__(get_mmvq_mmid_max_batch_for_device<type>()*warp_size, 1)`, a 256-thread
  budget for a kernel that is launched with 32. The `minBlocksPerMultiprocessor` parameter is
  therefore inert and the compiler spends 79 registers. Declaring the real block size and 32
  blocks per SM caps IQ3_S at 64 registers and lifts the architectural cap from 37.5% to 50%.
- **Result (2026-11-03): WIN. 34.30 -> 35.07 tok/s (+2.25%), acceptance unchanged at 0.96657.**
  Registers 79 -> 64, occupancy 34.9% -> 47.9% (cap 37.5% -> 50%), kernel 381.66 -> 352.58 us
  (-7.6%), no spilling (instruction count +0.5%). The prediction below overshot: -7.6% on the
  kernel against -13% to -21%, +2.25% end to end against +3%. See
  `findings/cyber-tiel-qwen35moe/2026-11-03-occupancy-on-iq3s.md`. 50% is the ceiling for a one-warp block, so
  further occupancy work needs multi-warp blocks - a different change, not yet tried.
- **Prediction, written before the run:** IQ3_S kernel time falls from 381.66 us into the
  300-330 us band (-13% to -21%) and `ncu` reports 32 blocks per SM with `long_scoreboard`
  unchanged or lower. At 82 ops per pass, -60 us each is -4.9 ms of a 164.7 ms pass, ~+3% tok/s
  (34.3 -> ~35.3). If the kernel drops by less than 5%, or registers spill and it regresses,
  occupancy is not the lever and Frontier 1's shared-memory staging is the remaining option.
- **Scope guard:** applied to the MoE kernel only, and only when `type == GGML_TYPE_IQ3_S`, since
  a blanket 64-register budget would spill Q6_K, which the `ffn_down_exps` path uses.

### Frontier 2: MoE Layout Linearisation (PR #31047, Target: ~40 tok/s)
- **Bottleneck:** Reading 8 scattered experts across 256 experts incurs access penalties that cap memory efficiency at 35-41 GB/s, far below the 110 GB/s sequential hardware ceiling.
- **Mechanism:** PR #31047 linearises expert weight buffers in memory to improve spatial locality and coalescing during multi-token verify passes.
- **Prediction:** Increasing effective memory bandwidth to ~55 GB/s reduces expert matmul time by ~12-15% end-to-end, moving pass latency to ~142 ms and throughput to ~40 tok/s.

### Frontier 3: Adaptive MTP Draft Depth (PR #27210, Target: ~36-38 tok/s on varied prompts)
- **Bottleneck:** Fixed `n-max 6` pays the full 164.7 ms pass cost even when draft confidence drops. `n-max 7` degraded performance (34.49 -> 34.11 tok/s) because the 7th column cost ~17.4 ms while average token return fell.
- **Mechanism:** Upstream PR #27210 adjusts draft depth dynamically based on acceptance streaks (expanding to 8-10 on high confidence, contracting to 3-4 on low confidence).
- **Prediction:** Prevents pass-latency blowups on entropy spikes while capturing longer draft chains during high-confidence sequences, raising effective throughput by 5-10% on variable prompts.

### Frontier 6: `output.weight` is 25.2% of the pass - CLOSED 2026-11-03, -44.2%

**Measured and rejected.** Q6_K -> Q5_K, 417 MB -> 350 MB: 34.82 -> **19.43 tok/s**, acceptance
**0.96657 -> 0.63603**, mean_len 5.68 -> 2.20. The scope note below asked for per-rep
acceptance and a text comparison rather than an argument, and it was right to: the argument
that "target and drafter share this tensor so they move together" is **false in practice**.
Sharing the tensor does not make them agree once its values move; it makes them disagree,
because the drafter's token and the target's verify token are chosen from perturbed logits.
Do not retry with q4_k, q5_0 or `--override-tensor`. A bit-preserving repack is the only
weight-adjacent change still standing. See `findings/cyber-tiel-qwen35moe/2026-11-03-lmhead-q5k.md`.
- **Why this is now the biggest single lever.** The measured budget
  (`findings/cyber-tiel-qwen35moe/2026-11-03-decode-pass-budget.md`) puts lm_head at 41.2 ms of a 163.4 ms pass. The
  tensor is 417 MB at q6_k and is read **seven times per pass** - once at 7 columns for the
  verify, then six times at 1 column for the MTP drafts - so it alone moves 2.9 GB, more than the
  entire 41-layer main forward. It already runs at 75.9 GB/s, the best utilisation of any kernel
  in the run, so the only thing left to cut is the bytes.
- **Honest caveat on the +7%.** That projection assumes a classic 4-bit type also sustains
  ~76 GB/s. The project's own measurements say classic 4/5-bit types run at 65-78 GB/s at batch 5,
  so this is plausible but not established for this shape at 1 and 7 columns. Treat 37.5 tok/s as
  the optimistic end; the pessimistic end is much closer to zero and it could lose.
- **Scope note on "the requant avenue is closed".** That closure (`HANDOFF.md` section 2) is
  about **expert** numerics: changing blocks 0-39's expert weights decouples the pre-trained MTP
  head from the target. `output.weight` is a different case *in principle* - the drafter's
  `shared_head_norm` projects through this same tensor, so target and drafter would move together
  - but that is an argument, not a measurement, and the acceptance spread on a requantized file is
  the trap that produced three wasted builds. If this is attempted, it needs per-rep acceptance
  on all 5 reps plus a text comparison, not one number.
- **Where the 41 ms actually splits (measured 2026-11-03, `srv-decode.sqlite`).** Six 1-column
  draft GEMVs at 5.21 ms each = **33.8 ms/pass at 80.0 GB/s**, against a 111 GB/s ceiling; the
  verify call(s) add ~9.3 ms/pass. So 83% of the lm_head cost is drafting, and it is already
  running at 72% of the memory ceiling. **There is no launch-parameter headroom in the
  1-column shape.** The 7-column call is different - see below - but it is the smaller half.
- **The 7-column call degrades, and only for q6_k.** Same tensor, same bytes: 1 column 5210 us
  (80.0 GB/s), 2 columns 6060 us, 4 columns 9318 us, 7 columns 9728 us (42.9 GB/s). The dense
  q8_0 kernel does not do this (60.8 GB/s at 7 columns against 63.8 at 1), which is why the old
  "+10% for a 7-column GEMV" figure from `opbench` was right for q8_0 and wrong for this shape.
  **Measured mechanism (ncu, this shape):** it is *not* ALU-bound - `math_pipe_throttle` is 0.35,
  the lowest stall, and the SM is at 63.5%. It is bound by L1 global-load throughput
  (`lg_throttle` 0.85, L1/TEX 84.2%): DRAM and L2 are identical to the 1-column case at 417/418 MB,
  but L1 sectors are 2.21x because the 2 KB activation `y` is re-read once per output row. It also
  spends 128 registers against 48 and runs at 25% occupancy, from the same
  `__launch_bounds__(nwarps*warp_size, 1)` defect the MoE kernel had. Ceiling for fixing all of it
  is the verify call's 4.5 ms/pass = **+1.0 tok/s**, and an occupancy fix also needs a shared-
  memory carveout raise. This is Frontier 7 below.

### Frontier 4: Native IQ4_XS Base Quant + Co-Trained MTP Drafter (Target: ~45-47 tok/s)
- **Why post-hoc requant failed:** Requantizing the existing GGUF altered base weights (blocks 0-39), causing the pre-trained MTP head (block 40) to lose alignment (acceptance 0.989 -> 0.64).
- **The native path:** Quantizing directly from the BF16 source checkpoint to uniform `IQ4_XS`, and fine-tuning the MTP draft head on those exact quantized activations.
- **Prediction:** All expert matmuls run at 68.9 GB/s while maintaining >=0.97 draft acceptance. 6-token pass latency drops to ~120 ms, unlocking ~45-47 tok/s.

---

## How to Measure (Read Before Running Any Experiment)

- **Use `bench/prompt2.txt`.** `bench/one.sh` defaults to `prompt.txt`, which drops draft acceptance from 0.989 to 0.71 and makes the known-good config look like 22 tok/s. Every authoritative number is `prompt2.txt`.
- **400 generated tokens, 5 reps, median.** 128-token runs are for coarse ranking only.
- **Graphs enabled:** Env var `GGML_CUDA_DISABLE_GRAPHS` must be ABSENT (`env -u`). Setting `=0` disables graphs.
- **Memory settling:** One server load per config, wait for unified memory to drop below 26 GB before the next run.
- **Baseline to beat:** **36.93 tok/s decode** = 36.63 kernel (median of 18, both orders, 2026-11-04)
  + `-bs` (36.550 -> 36.930, 16 warm reps per arm in both orders). 36.63 is 36.56 plus the
  `concat_non_cont` geometry; 36.56 is 36.25 plus the same rows-per-block lever at 1 column; 35.07
  before the dense `mul_mat_vec_q` work; 34.30 before the 64-register MoE launch bounds; 31.50
  before the dispatch patch. Older rows in this file that say 34.49 / 34.30 / 31.50 are the same
  harness at earlier stages of the local commits, not different configurations. Acceptance must
  read **0.96657** and mean len **5.68** on every arm; any other value means the arm moved numbers,
  not kernels, and a throughput number from it is not comparable.
- **Prefill uses `bench/prefill.sh`, never `bench/one.sh`** (its PP column is cache-contaminated).
  No generated tokens; 3 lengths (590 / 2,385 / 9,563 tokens); 3 reps, rep 1 dropped; median.
  Reproducible to 0.14%. Every prefill arm reports decode and acceptance too, because `-ub` and
  `-b` are speculation parameters - pass `DECODE_REPS=5` and the harness measures decode first, in
  the same server load, appending to `results.csv` / `spec.csv` in `one.sh`'s format.
- **`results.csv` has no `n_predict` column**, and `NPRED` is worth 17% of TG (rule 4). Put
  `np400` in `NAME` or the row is unrankable later.

---

## Closed - Do Not Re-test

Full measurements in `findings/` and `AGENTS.md`.

- **Naive Q4_0 KV Cache without precision tailing at 262k context - closed 2026-11-08, hard negative.**
  Testing `-ctk q4_0 -ctv q4_0` at `CTX=262144` freed 3.5 GB of RAM (22.7 GB used vs 26.2 GB), but
  collapsed MTP draft acceptance from 0.97546 down to **0.63732** and mean draft length from 5.61 to **2.21**.
  Speculative decode throughput collapsed from 43.08 down to **23.21 tok/s (-46%)**. Asymmetric `-ctk q8_0 -ctv q4_0`
  also degraded acceptance to **0.68000** and throughput to **24.77 tok/s**. Naive 4-bit KV quantization
  destroys draft verification logits. Because Cyber-Tiel-Coder only has 11 full attention layers, `q8_0` KV cache
  takes only 8.25 GB at full 262,144 context and leaves 4.8 GB free RAM with full 43+ tok/s speed. Q8_0 KV is the optimal
  choice on this model. See `findings/cyber-tiel-qwen35moe/2026-11-08-262k-context-and-cub-argsort.md`.

- **CUB on CUDA 11.4 (sm_72) for ARGSORT / top-p - resolved 2026-11-08.**
  CUDA 11.4 CUB was disabled by upstream `CUDART_VERSION >= 11070` checks, causing `GGML_OP_ARGSORT`
  to fail device support checks and fall back to CPU. Patched `common.cuh` (`CUDART_VERSION >= 11040`)
  and `argsort.cu` to use `DeviceSegmentedRadixSort`. Eliminates all backend warnings and enables native
  GPU radix sorting. Commit `1f33695`.

- **MTP Phase 1 Retraining (dense projection and attention with frozen MoE) - closed 2026-11-07, negative result.**
  Retraining the 35.66M dense/norm parameters of `blk.40` on 3,000,519 tokens of activations dumped from
  `Q4KM-all-q4k.gguf` achieved strong teacher-forced metrics (val CE 0.8005, argmax agreement 80.88%),
  but deploying the spliced file (`Q4KM-all-q4k-retrained.gguf`) measured draft acceptance **0.49206**
  (reps 1/3/5 0.492-0.614, reps 2/4 0.401-0.423, mean len 2.08) and throughput **20.91 tok/s** (median
  reps 2-5). This failed to recover over the 0.64583 factory baseline and is **24.9% slower** than plain
  decode (27.86 tok/s). Freezing the 1.826B MoE expert parameters while updating front-end dense layers
  distorts representations entering the MoE router, and teacher-forced single-step loss on code does not
  prevent compounding error accumulation across 6-step autoregressive drafting on prose. Retraining dense
  layers alone cannot rescue MTP on a requantized trunk. See `findings/cyber-tiel-qwen35moe/2026-11-07-mtp-phase1-deployment-results.md`.


- **The two ways an MTP reference silently measures the wrong thing - closed 2026-11-06.** Both
  are in `bench/mtpdump.cpp --head-logits` and both were hit in one afternoon. (1) A context
  created with default `llama_context_params` runs the **trunk**, so its "head logits" are the
  target's own logits and agree with the label **100.00%**. `ctx_type = LLAMA_CONTEXT_TYPE_MTP`
  is what selects `LLM_GRAPH_TYPE_DECODER_MTP` (`llama-context.cpp:115`). A reference that agrees
  perfectly is a bug report, not a result. (2) `llama-model.cpp:3981` sets `mtp |= TENSOR_SKIP`
  unless `ml.load_mtp`, which comes from `llama_model_params.load_mtp`, so `blk.40` is not loaded
  at all and the MTP graph asserts on `nextn.eh_proj` - set it only for the `--head-logits` path,
  since it costs ~0.45 GB of the 31 GB the extractor already runs close to. It is inert for the
  trunk: rebuilt with the flag, the extractor reproduces `dumps/shard-000000.mtp` byte for byte.
  Related fact that has to be re-read rather than assumed every time: `QWEN35MOE` returns
  **IMROPE**, but `ops.cpp:6424` pairs dims half-split exactly as NEOX and `is_imrope` only
  re-picks theta per section, so with all three axes sharing one position (text) it *is* partial
  NeoX rope over 64 of 256 dims. See `findings/cyber-tiel-qwen35moe/2026-11-06-mtp-head-reference.md`.

- **Requantizing from the original BF16 to rescue MTP - closed 2026-11-05.** Tested with
  `UD-Q4_K_XL` (22.749 GB): a single pass from the 69 GB BF16 base with the author's imatrix,
  never retyped, dense attention/GDN and `output.weight` at **q8_0**, the same Q3_K head block,
  and **ppl 1.9142** - identical to `Q4_K_M` and 1.67% better than our `all-q4k`. MTP on it reads
  acceptance **0.534 / 0.585 / 0.534 / 0.585**, mean len 2.09, TG **19.53** against **22.014**
  plain (`-bs`, 5 reps), i.e. **-11.3%**: a net cost, and *worse* than the double-quantized file's
  0.64583 / -16.2%. Only the shipped pairing pays (0.96657, 36.93 against 22.25, +66%). Acceptance
  is therefore not a function of quantization error, bit width or how cleanly the file was derived
  from BF16 - a better target does not help, and `Q4_K_M` is not "close enough", it is exactly the
  file the head's argmaxes track. Do not download the 69 GB BF16 for this purpose; only a head
  retrained against the deployed file can recover the drafter. See
  `findings/cyber-tiel-qwen35moe/2026-11-05-mtp-finetune-and-bf16-requant.md` sections 6a-6c.

- **"Acceptance falls because the head reads a perturbed trunk" - closed 2026-11-06, refuted by
  measurement.** The retrained-head plan was founded on it, so it was tested directly with
  `bench/mtpdump.cpp` (2,000 positions of `prompt-32k.txt`, `--batch 8`, bit-deterministic A/A).
  Against `Q4_K_M`, `Q4_K_XL`'s hidden states are **five times closer** than `all-q4k`'s
  (cos 0.9933 / 0.46% median direction error against 0.9643 / 3.11%, argmax agreement 98.10%
  against 94.80%) and its acceptance is **worse** (0.585 against 0.64583). Distance in `h` does not
  rank the three files, and XL's `h` is as close to the reference as a *batch width change* is
  (11.67% relative RMS against 12.63%, width 8 vs 64). Do not rebuild a mechanism on `h` distance,
  do not quote a cosine as a predictor, and do not assume a small perturbation is a small loss.
  What remains is that the head has three inputs and the files differ in all three; the only one
  whose type family changes in both requants is the shared `output.weight`, which is the next
  experiment. See `findings/cyber-tiel-qwen35moe/2026-11-05-mtp-finetune-and-bf16-requant.md` section 8.

- **Restoring the shipped `output.weight` into a retyped file to rescue MTP - closed 2026-11-06,
  +2.7 points of acceptance and -3.5% throughput.** The file is `Q4KM-all-q4k-lmheadq6k.gguf`
  (20.592 GB): all-q4k's trunk with `output.weight` replaced by a verbatim copy of `Q4_K_M`'s q6_k
  tensor, verified 752/752 other tensors byte-identical and the replaced one matching the donor's
  sha256. A/B/B/A, 5 reps, `NPRED=400`: acceptance **0.64583 -> 0.67286**, mean len 2.10 -> 2.39,
  TG 23.347 -> 22.531 (**-3.5%**, A/A floor 0.12%). The prediction was 0.85-0.95; it missed. The
  mechanism is the pass: q6_k's 417 MB lm_head is read 7 times per pass against q4_k's 286 MB, so
  +4.2% relative acceptance cannot pay for an 18% longer pass. Do not retry this or any other
  single-tensor restoration of a shared tensor; acceptance is a joint property of the file, no
  scalar predicts it, and only a retrained head can recover the drafter. See
  `findings/cyber-tiel-qwen35moe/2026-11-05-mtp-finetune-and-bf16-requant.md` section 9.

- **Re-attaching MTP to a retyped file - closed 2026-11-05, -16.2%.** `--spec-type draft-mtp
  -n-max 6 -p-min 0.5` on `Q4KM-all-q4k.gguf` loads and drafts normally, because every requant
  pinned blk.40 back to its own types and nothing is missing. But acceptance is **0.64583** with
  mean len **2.10**, so the pass-accounting goes 35.9 ms/token plain against 89.9 ms per MTP pass
  over 2.10 accepted tokens = 42.8 ms, and decode is **23.351 tok/s against 27.86 plain**. The
  drafter does not need the target to be better, only for the target's argmax to be predictable
  from the same weights it read; retyping moves both and moves them apart. Every requant win in
  this file is therefore mutually exclusive with MTP, not additive to it. Do not try another
  draft depth or `p-min` on a retyped file - the chain is 2.10 long at depth 6, so there is
  nothing for a parameter to recover.

- **Expert blocks below q4_k - closed 2026-11-05, -8.9%.** `ffn_gate_exps` / `ffn_up_exps` /
  `ffn_down_exps` q4_k -> q3_k on the all-q4k file (117 tensors, 20.461 -> 16.290 GB) reads
  **27.944 -> 25.456 tok/s** over an interleaved A/B/B/A. The file is 20% smaller and a token
  reads 8.6% fewer expert bytes, and decode is 9% slower, because q3_k's 110-byte block costs more
  per byte issued than q4_k's 144-byte block saves. This is "type beats bits" on the decode path
  with no drafter: the four requant steps that paid all stayed on q4_k or above (yield 0.76-0.96x
  of their byte share); the first step that changed the dot's type family collected less than
  zero. No quality gate was run - the speed ends it. Do not test IQ3_S, IQ2 or any 3-bit expert
  variant on this board; the 2026-10 type table already showed IQ3_S at 22 GB/s and IQ2_S at 17.

- **Any ubatch above 2048 as a *measurement* setting - closed 2026-11-05.** Not for correctness:
  the same file and flags read differently every load. At 2,828 tokens, `-ub 3072` measured
  218.9 / 172.4 / 200.4 / 165.6 tok/s over four consecutive server loads (32% spread) and
  `-ub 4096` measured 159.2 / 273.4 (72%), while `-ub 1024` gave 228.2 / 226.8 (0.6%) and
  `-ub 2048` 250.2 / 243.1 (2.9%). The variable is the compute buffer the loader reserves, which
  grows with ubatch (`-ub 2048` asks for 3897 MiB on a 22.5 GB file). Consequence: the
  2026-11-04/05 prefill table (215.0 at 3072 for IQ4_XS, 173.3 for Q4_K_M, 225.5 for dense-q5k)
  is superseded, and the real ranking at 2048 is that **all three files are equal within 2.8%**.
- **A prefill comparison that spans two server loads without an in-load repeat - closed
  2026-11-05.** Writing the 22 GB dense-q5k file left the box reading wide-ubatch prefill ~60%
  slow for about ten minutes: the same file read 175.5 then 281.0 in adjacent positions. That
  produced a "+30% prefill from a retype" claim which vanished when interleaved (the two files are
  equal to 0.1-2.5% at 11,484 tokens). Repeat inside every load; the 0.56-2.1% floor previously
  quoted is a within-session figure and does not cover this.
- **Retyping the dense attention/GDN projections q8_0 -> q5_k - closed 2026-11-05, adopted for
  short prompts.** 1.36 GB of 2.46 GB per token, 112 tensors, 22.03 GB file. Decode 22.25 ->
  24.01 (+8.5%, interleaved both orders, A/A arms 0.09% apart), prefill unchanged, quality +0.67%
  perplexity on a 30k-token code corpus (paired t = 8.3). It *loses* 2.8% under the DFlash
  sidecar's 4-column verify (22.53 against 23.18) because q8_0's dot is cheaper than q5_k's per
  element at wide columns - do not generalise a byte saving into a column-independent win.
- **Retyping `output.weight` q6_k -> q5_k - closed 2026-11-05, adopted.** 417 -> 349 MB, 20% of
  the bytes a plain token reads, 1 tensor of 753 retyped and 8 sampled others byte-identical.
  Decode 24.16 -> 24.88 (+3.16% pooled over A/B/B/A, A/A floor 0.78%), prefill unchanged at this
  width (214.3 against 220.7 tok/s, inside the 2.9% `-ub 2048` floor), quality **+0.065%
  perplexity** (paired t = 10.4, 59/60 chunks worse). The prediction was +3.0-4.2% decode
  (hit, low edge) and +0.3-1.0% perplexity (wrong by 5-15x: the logits projection is far less
  sensitive than assumed). Under DFlash's wider verify columns it **loses**: 22.42 on 8.8k of
  code against the parent file's 22.53 and `Q4_K_M`'s 23.18, with acceptance 0.88925 -> 0.85849
  and mean len 3.40 -> 3.27. The DFlash profile stays on `Q4_K_M`. Two byte savings in a row have
  now inverted at wide columns - do not carry a plain-decode win over to DFlash without measuring
  it on the sidecar rig.
- **IQ4_XS as the MTP-free baseline - closed 2026-11-05.** At `-ub 2048` it is the worst of the
  three files on both axes: 18.30 decode, 215.7 prefill, 1.9408 PPL against Q4_K_M's 1.9142. It
  remains the MTP configuration's file; nothing about the MTP profile changes.
- **An ubatch wider than the prompt - closed 2026-11-05.** With no drafter, ubatch width is free to
  choose, so the obvious next move is "make it huge". It is wrong past the prompt length. The
  clean case is a 2,828-token prompt: `-ub 4096` and `-ub 8192` both fit it in exactly one chunk,
  so the compute is identical, yet they read **267.1 and 166.1 tok/s** - a 38% loss from the
  buffer the graph allocates. At 11,484 tokens `-ub 8192` reads 95.4 tok/s against 214.6 at
  `-ub 2048`. The cause was not diagnosed (memory headroom is the candidate), so the rule is
  empirical: **choose the width nearest below the prompt, default 2048, and never above it.**
  See `findings/cyber-tiel-qwen35moe/2026-11-05-mtp-free-campaign.md` section 6.
- **DFlash deeper than 3 drafts - closed 2026-11-05.** On 8.8k of code at `-ub 2048`, depth 3
  reads 22.97 tok/s, depth 5 21.09, depth 8 15.87 - acceptance falls 0.787 -> 0.592 -> 0.520
  while mean len rises 3.35 -> 3.95 -> 5.12, and a verify pass is linear in tokens in this MoE
  (0.221 ms/token x 41 layers), so the columns cost more than the extra accepted tokens return.
  Same verdict as the MTP head, for the same reason.
- **Quoting a DFlash number without its prompt - not an experiment, a trap.** The 267-token
  `prompt2.txt` gives the Ornith draft **0.31118** acceptance (it loses to plain decode); the same
  draft at `-ub 2048` on 8.8k of C++ source gives **0.78652** and beats plain decode by 40.7%.
  The drafter was trained on code. Any future DFlash test must state the context it ran on.

- **Pinning a faster MMQ config at wide `-ub` (the original H3 shape) - closed 2026-11-04.** Above
  64 tokens the MoE does not use MMQ at all (`MMQ_DP4A_MAX_BATCH_SIZE = 64`,
  `should_use_mmq()` returns `n_tokens < 64` on any card with fp16 MMA), so there is no `J`
  specialization to tune at wide widths. The 2.4x at `-ub 512` *is* the engine switch. The tile
  width only has anything to bite on at `-ub 32`, and there it was worth +12% (H3, above).
- **MoE shared-memory carveout at 100% (`GGML_CUDA_MMQ_CARVEOUT=100`) - closed 2026-11-04.**
  Predicted +15-30% prefill from two resident blocks per SM; measured **+0.4% at 9,563 tokens**
  and +1.6% at 2,385, i.e. inside the protocol floor, and its 6.9% at 590 tokens did not
  reproduce in a later session. The MoE launch reads at 57-59 GB/s of the 110 GB/s ceiling but is
  latency-bound rather than bandwidth-bound, and two resident blocks did not help - so extra
  residency is not the lever. Hook left in `mmq.cuh` as an opt-in experiment switch.
- **Narrow MoE tile for IQ3_S only (`GGML_CUDA_MMQ_MOE_TILE_AVG=2`) - closed 2026-11-04.**
  **-1.49%** at 9,563 tokens in both orders (122,136 / 122,146 ms against 120,019 / 120,689 for
  the narrow-everywhere mode), against a 0.56% A/A floor at that length. The theory behind it - that
  the narrow tile makes each expert's weights be re-read once per column tile - is wrong: the MoE
  branch returns before loading when `jt*J >= col_diff` (`mmq.cuh:1085-1088`), so the extra column
  tiles are free. IQ4_XS wants the narrow tile too.
- **Chasing the IQ4_XS down-projection's 59 GB/s (against 92.7 GB/s for the same type in the
  decode kernel) - closed 2026-11-04.** It is the one large kernel in the prefill budget that is
  not near a hardware limit, but it prefers the narrow tile (previous entry), so what is left is
  its dot, and the dot cannot change without moving bits. Do not spend another arm on its launch
  geometry.

- **Any ubatch width above 32 - closed 2026-11-05, measured at 36 / 40 / 48.** The width curve is
  not a slope and not monotone: acceptance **0.96657 at 32, 0.88779 at 36, 0.70755 at 40, 0.57955
  at 48**, then 0.91592 at 64, 0.68122 at 96, 0.92258 at 128. Every point outside 32 breaks the
  drafter, and 48 - which is inside the same fused-MMQ engine as 32, because
  `should_use_mmq()`'s threshold is 64 - is **worse than 64**, so the loss is not the engine
  switch: chunking changes the prompt's hidden state, and the f32 router's top-8 of 256 is a
  discrete choice on it (AGENTS.md section 7). What the range does pay is prefill - **+11.0% at
  48** (87.0 vs 78.4 tok/s at 11,484 tokens) against -49% decode, a worse exchange than `-ub 512`
  - so there is no free point in 33-63 and no reason to scan widths again. The one reusable
  result is the traffic model: prefill time per token scales as `distinct_experts(ub)/ub`, which
  is why width pays at all and why `-ub 512` is 2.4x. See
  `findings/cyber-tiel-qwen35moe/2026-11-05-prefill-ubatch-intermediate.md`.
- **MMQ prefill tile geometry - `I`, `nthreads`, `J` below 8, shared-memory carveout - closed
  2026-11-04/05, all inert.** The MoE prefill launch (`mul_mat_q`, 88.1% of prefill at `-ub 32`)
  resolves 56.8-59.0 GB/s against a 92 GB/s machine ceiling, which invited a residency fix: a
  pure-load probe measures bandwidth linear in resident warps, 47 / 72 / 85 / 92 GB/s at 8 / 16 /
  24 / 32 warps, while the launch has 16 resident warps. Every lever aimed at that was measured
  and did nothing. `I` 128 -> 64 (17,696 B shared, 16 -> 20 resident warps, `J=8`) reads **-0.11 /
  -0.03 / -0.28%** at 590 / 2,385 / 9,563 tokens over four arms in both orders, acceptance and
  mean len identical per rep - because at 16 warps the kernel already saturates its own LSU path
  (dequant + shared store + two `__syncthreads()` per K iteration, `mmq.cuh:956-991`) and extra
  warps only queue on it. A 100% carveout was already inert (+0.4% at 9.5k). `J` cannot go below 8
  without a different accumulator layout, and the per-type narrow tile is -1.49%. **The remaining
  MoE lever is the dp4a tile algorithm itself, not a parameter** - and with ~1 useful column per
  expert out of the 8 the tile computes, that is a kernel rewrite whose payoff is capped by the
  same LSU path. See `findings/cyber-tiel-qwen35moe/2026-11-04-prefill-57gbs-probe.md` and
  `findings/cyber-tiel-qwen35moe/2026-11-04-prefill-mmq-tile-i64.md`.
- **The "18 GB working set defeats the TLB, so prefill cannot reach the probe's bandwidth" theory -
  falsified 2026-11-04.** A scattered-chunk probe over a 0.115 / 1 / 4 / 18 GB extent reads 90.6 /
  89.9 / 91.7 / **92.1** GB/s, flat in working-set size, with a contiguous control at 101.6-102.7.
  So the MoE's 57 GB/s is not the memory system failing at this size. Do not rebuild this argument.
- **Cold page cache as a TTFT lever - closed 2026-11-05, it costs 5.1%.** Predicted 250-320 s
  (18.12 GB at 88.1 MB/s); measured **109,301.7 ms vs 103,979.0, i.e. 88.55 vs 92.02 tok/s, a 5.1%
  cost**, because `--load-mode none` still copies every tensor into backend buffers at load and
  prefill never re-reads the file. A load-time warm-up request is therefore not worth adding for
  this reason (it costs 305 ms in this build and buys nothing measurable); the 5.21 s graph capture
  that H2 measured does not reproduce here either (305 ms warm-up).
- **Root-causing the 15.7% absolute-prefill drift - closed 2026-11-05, do not spend more time on
  it.** Eliminated by measurement: storage (5.1%), the binary (interleaved A/new/new/A,
  76.52 / 75.90 / 76.16 / 75.63 at n_prompt 9,563), the flags, the GPU and EMC clock (`devfreq`
  min = max = 1,377,000,000, `nvpmodel` MAXN), host contention (load average 0.6, nothing above
  1.8% CPU) and the machine's achievable DRAM rate (`tlbstream 18 20` reads 93.2 / 92.7 / 93.1 /
  93.1 GB/s in the 76.5 tok/s state, the same values as in the 92.0 tok/s state). The surviving
  candidate - physical placement of the 18 GB allocation, hence how the GPU's scattered expert
  reads map to DRAM - is not visible with any tool on this board (`tegrastats` reports no EMC
  clock, `gv11b` has no `dram__` counters). **Publish a range and quote same-session deltas.**
  See `findings/cyber-tiel-qwen35moe/2026-11-05-prefill-page-cache.md` section 3c.
- **Chunked GDN for Volta (H4) as a prefill campaign lever - closed 2026-11-04 by measurement, not
  by judgement.** H2's per-kernel trace of a 590-token prefill at the production `-ub 32` puts
  `gated_delta_net` at **214.0 ms of 6,569.8 ms (3.3%)**, and the whole delta-net block including
  `ssm_conv`, `concat_non_cont`, `cpy_scalar` and the `fwht` calls at **393 ms (6.0%)**. An
  infinitely fast GDN therefore buys +6% prefill. The "1000+ tok/s" figure that made this the top
  frontier was never measured - it was an assumption about how much of the time 30 of 41 layers
  must be taking, and it is wrong by an order of magnitude: 30 layers run a token-serial loop but
  they are cheap ops on narrow tensors, while 88.1% of the time is `mul_mat_q` on the expert
  weights. The bf16/Ampere-only tiles still make it a schedule rewrite (Volta `mma` is m8n8k4), so
  it is days-to-weeks of work for a 6% ceiling that is not even all reachable. Do not start it
  expecting a large win. See `findings/cyber-tiel-qwen35moe/2026-11-04-prefill-budget.md`.

- **FlashAttention prefill schedule (H5) - closed 2026-11-04, ceiling 0.9%.** Measured 60.7 ms of
  6,569.8 ms at `-ub 32`, 87.1 ms of 2,836.3 ms at `-ub 512`. The old "+3-8% prefill at best"
  estimate was ~4-9x too high; only 11 of 41 layers attend and Volta already takes the MMA-F16
  kernel (`fattn.cu:704-712`). A split-D schedule cannot pay for itself at this share.

- **"Pin a faster MMQ config at wide `J`" as the route to H1's 2.4x - closed 2026-11-04, the
  premise does not hold.** The width sweep's non-monotonic acceptance was read as evidence that
  MMQ's exact-match `J == J_` dispatch was choosing different specializations at different widths.
  H2's trace shows MMQ's share *falls* from 88.1% at `-ub 32` to 17.2% at `-ub 512`: the J values
  seen are 16/24/32/40/48/64 and there is no wide-`J` MMQ launch at all, because
  `MMQ_DP4A_MAX_BATCH_SIZE = 64` and `should_use_mmq()` returns `n_tokens < 64` on any card with
  fp16 MMA (`cc >= VOLTA` is true on sm_72). Above 64 tokens the MoE is computed by a per-expert
  loop (<=8 tokens MMVQ, 9-63 MMQ, >=64 dequant+cuBLAS), so the 2.4x *is* the engine switch and no
  configuration of the MMQ kernel can reproduce it at unchanged acceptance. Tuning MMQ is still
  worth doing - it owns 88% of prefill - but at `J=32`, not wide `J`. See H3.

- **A wider ubatch in the decode launch line, all widths 64-2048, closed 2026-11-04 on six measured
  arms.** Prefill rises smoothly with `-ub` (1.00x / 1.06x / 1.28x / 1.45x / 1.91x / 2.4x vs ub 32)
  and draft acceptance falls off a cliff that is **not monotonic** (0.96657 / 0.91592 / 0.68122 /
  0.92258 / 0.57655 / 0.65882 at ub 32 / 64 / 96 / 128 / 256 / 512). Mechanism: `-ub` is the
  compute batch, so it sets the matmul column count during prompt processing, and the width decides
  which kernel computes the prompt and which arithmetic populates the KV/SSM cache. **Corrected
  2026-11-04 by H2:** this entry used to attribute the width effect to MMQ's exact-match `J == J_`
  config dispatch. That is only half right, and the larger half is the *engine*: above 64 tokens
  `should_use_mmq()` returns false and the MoE leaves MMQ for a per-expert loop, so at wide widths
  a different kernel computes the expert matmuls entirely, not the same kernel with a different
  `J` (`findings/cyber-tiel-qwen35moe/2026-11-04-prefill-budget.md`). The MTP drafter then starts its six-step chain from
  those numbers and its argmax stops matching the target. `ub 64` proves no width is free:
  +4-7% prefill (inside the 0.71% A/A drift floor) for -10% decode. `-ub 32 -b 32` stays. Do not
  sweep width again - and note the H3 question it points at is now "can the `J=32` MMQ config that
  owns 88% of prefill be made faster while staying bit-preserving", not "which width to pick".
  **Also: `-b 512 -ub 32` does not even fit** - it dies at graph
  capture with `NvMapMemAllocInternalTagged: error 12` / `CUDA error: out of memory`, so `-b` is not
  a lever on its own and its numerics were never tested.

- **`--poll 50` and `-t 1 -tb 1` (server flags), re-tested 2026-11-04.** Both are flat on top of
  the adopted `-bs` config: `--poll 50` **+0.15%** (36.970 -> 37.025), `-t 1 -tb 1` **-0.04%**
  (36.970 -> 36.955), pooled n=8 warm reps per configuration from one interleaved pass in each
  order (`bench/sweep-flagab.sh`, `ORDER="base poll t1 t1 poll base"`, `NPRED=400`). Acceptance
  0.96657 and mean len 5.68 in all six arms. Single-order, `--poll 50` read **+0.60%**, so the
  protocol that produced the original 28-config verdict would have adopted it. Two reasons not to
  retry these: the mechanism is tested - `-bs` already removed the host sampler work and only
  ~7.8 ms/pass of host CPU is exposed, so a different wait/thread-pool arrangement has nothing
  left to recover - and the board's own positional drift measured in this run is **0.71% between
  two identical `base` arms** and +0.95% from first position to last, both larger than either
  delta. Any further flag claim at this size needs the interleaved design and must beat that
  floor. See `findings/cyber-tiel-qwen35moe/2026-11-04-flag-retest-poll-t1.md`.
- **`GGML_CUDA_MOE_RPB=4` (MoE kernel rows per block), and by extension any MoE launch-geometry
  change at the model level.** Re-tested post-fix 2026-11-03: the kernel really is faster -
  355.14 -> 340.54 us at 6 columns and 416.19 -> 401.12 at 7, L1 load sectors -32%, registers
  and occupancy unchanged at 64 / 48% - but the model is neutral across four 5-rep runs in both
  orders (36.66 / 36.25 one way, 36.45 / 36.25 the other) with acceptance and mean len identical.
  8 rows is +15% on the kernel. **Mechanism established 2026-11-03 by re-tracing both arms
  adjacent in time:** the MoE is 40.1% of a 159.1 ms pass, so the share was never
  over-attributed, but its in-situ time does not move either - 63.98 vs 63.95 ms/pass with the
  grid verified changed (256/1024 -> 128/512) and the launch count identical. The cost is exactly
  linear in token count (0.1392 ms/token at gate/up, 0.0818 at down, intercept ~0.006 ms) because
  each token routes to its own 8 experts, so the kernel is pure DRAM streaming at 44-52 GB/s and
  an L1 sector cut has nothing to work on. A single-launch `opbench`/`ncu` number on this kernel
  is not evidence of anything at the model level. See
  `findings/cyber-tiel-qwen35moe/2026-11-03-moe-share-in-situ.md`.
- **Any single-order A/B claiming less than ~1%.** Same experiment produced the first controlled
  ordering test in this project: the arm run **second** read 0.55-1.1% lower in both directions.
  Pooled medians or it is not a result. See `AGENTS.md` measurement rule 7.

- **Making any per-pass weight smaller: `output.weight` q6_k -> q5_k, and by extension q4_k,
  `--override-tensor`, and uniform requants.** Measured 2026-11-03: the tensor is 417 MB read 7x
  per pass, so -16.2% bytes looked like +4.3%. It delivered **34.82 -> 19.43 tok/s (-44.2%)** with
  acceptance **0.96657 -> 0.63603** and mean_len 5.68 -> 2.20. The artefact was verified before
  measuring (753 tensors, only this one retyped, 8 tensors hashed byte-identical, requant cosine
  0.99938), so the quantizer is not at fault. **The MTP drafter needs numerics identical to the
  target, not close.** Four unrelated perturbations now land at acceptance 0.63-0.72 - the three
  expert requants, the f32 router change (0.67717), the deleted jusko fork (0.678), and this
  (0.636) - while the control reproduces 0.96657 bit-for-bit on every run ever logged. A change
  either touches no numbers or it is a different model as far as the drafter is concerned, so
  "small enough to be safe" is not a valid argument here. The only surviving weight-adjacent
  avenue is a **bit-preserving** repack, and the padded block layout was measured on 2026-11-03
  and closed (L1 sectors -37% but registers 48 -> 95 and occupancy 57% -> 29%, 18-26% slower).
  See `findings/cyber-tiel-qwen35moe/2026-11-03-padded-block-layout-probe.md` and
  `findings/cyber-tiel-qwen35moe/2026-11-03-lmhead-q5k.md`.

- **`--spec-draft-p-min` / `--spec-draft-p-split` / draft depth as a lever: swept at the operating
  point, nothing left.** Re-tuned `p-min` at depth 6 (2026-11-03) because it had only ever been
  swept at depth 3 on the pre-patch build: **0.40 = 18.55 tok/s (-46.7%)**, 0.60 = 33.67 (-3.3%),
  0.70 = 34.01 (-2.3%) against a 34.82 control, so **0.50 stays** and the surface is a cliff below
  and a slope above. `p-split` 0.05/0.10/0.20 is neutral. Depth is 6 (7 turns down at 34.11).
  Do not nudge `p-min` hoping for a percent: -46.7% is one step down, and any change that lowers
  effective drafter confidence (a requant, a perturbed router, a different accumulation order)
  pushes the same way into the cliff. See `findings/cyber-tiel-qwen35moe/2026-11-03-pmin-at-depth6.md`.
- **Reading acceptance or `mean_len` as evidence of a win.** Both are conditional on a draft having
  been made, so passes where the gate skips drafting are excluded entirely. At `p-min 0.70`
  acceptance reads **0.99677** and mean_len **5.83**, both better than the control's 0.96657 / 5.68,
  while throughput is 2.3% *lower*. Throughput is the only trustworthy metric; a rise in acceptance
  with no rise in tok/s means the gate is skipping work, not that drafting improved.

- **Changing the numerics of the MoE router (`ffn_gate_inp`), by any means.** Measured 2026-11-03 by
  raising the f32 MMVF threshold `ne11 <= 3` -> 8 so the 7-column router leaves cuBLAS: 34.82 ->
  **18.93 tok/s (-45.6%)**, 5 reps. The kernel was not slower - the graphs captured fine
  (`reallocs 0`) - the drafter stopped agreeing: acceptance 0.96657 -> **0.67717** median
  (0.590 / 0.933 / 0.675 / 0.677 / 0.677), mean len 5.68 -> 2.40. Mechanism: swapping cuBLAS
  SGEMM for `mul_mat_vec_f` changes the summation order, the router logits move in the last bits,
  and the top-8-of-256 expert selection flips wherever two experts are near-tied. Routing
  quantises a continuous score into a discrete choice, so it is maximally sensitive to
  perturbation. This closes the whole family with one measurement: `GGML_CUDA_CUBLAS_COMPUTE_TYPE`,
  TF32/FP32-accumulation flags, `--override-tensor` on `ffn_gate_inp`, and any router kernel rewrite
  all perturb routing the same way. Note that 0.67717 is the **same acceptance the deleted Volta
  fork produced (0.678)** - these are not random scatter, they are specific alternative routing
  attractors, which is why the unmodified control reproduces 0.96657 bit-for-bit every run.
  See `findings/cyber-tiel-qwen35moe/2026-11-03-mmwq-launch-params.md` section 5.

- **Dense q8_0 MMVQ (the 7-column attention/ssm matmuls): loop reordering, per-type register preloading of the weight slice, and raising occupancy.** All inert or irrelevant, mechanism now understood. The kernel is L1-bound (L1/TEX 82.94%, L2 52.11%) at 15.6x L1 sector amplification: 8,767,457 sectors requested for 563,305 unique, with DRAM clean at 18.03 MB for 17.00 MB of weights. 80 registers/thread cap occupancy at 37.5%, which is real but is not the wall. Column-count decomposition (tok=1: 4,515,237 sectors / 261.95 us; tok=7: 8,768,877 / 288.38 us) gives 3.81M sectors for the weights - constant in column count - and 709k per column for activations, so nvcc already holds the weight slice across columns and the 94.23% L1 hit rate was intra-struct sector reuse, not re-reads. An explicit preload compiled, ran, and changed nothing (294.69 -> 294.56 us, sectors identical); it was reverted and the rebuild is bit-identical to the baseline. The real cause is the 34-byte `block_q8_0` stride: no block is sector-aligned and the overlapping 8-byte loads straddle about twice the minimum sectors. Fixing that needs a padded (sector-aligned) block layout and a repacked GGUF, worth only ~+2% for q8_0 alone. See `findings/cyber-tiel-qwen35moe/2026-11-03-decode-pass-budget.md`.
- **Rebasing onto newer upstream master:** the tree was rebased from `dde879b` onto `2cbca38` (31 upstream commits, 2026-11-03) with the patch unchanged. 34.30 tok/s against 34.49, acceptance identical at 0.96657 / mean len 5.68. The gap is inside the noise of 5 reps, so upstream base version is not a lever and is not worth re-testing. What would change that: if upstream touches `mmvq.cu` or `ggml-cuda.cu`, the patch has to be re-verified because the code is no longer identical. See `findings/cyber-tiel-qwen35moe/2026-11-03-fork-rebase-reverification.md`.
- **Post-hoc GGUF gate/up requants:** Three builds tested (`gateupIQ4XS`, `-anchored`, `-draftkeep`). All degraded throughput (25.76 - 30.50 vs 31.41 control) and collapsed draft acceptance (0.64 - 0.93 vs 0.989 control). Altering target model weights breaks the MTP head.
- **Static draft depth > 6:** `n-max 7` measured at 34.11 tok/s (vs 34.49 at `n-max 6`). Marginal token costs ~17.4 ms; deeper drafting turns down without adaptive depth.
- **Launch geometry knobs on MoE kernel:**
  - `rows_per_block=4` was +3% in opbench but -0.7% at the model level (34.01 vs 34.26 tok/s). Default stays 2.
  - Column-splitting into chunks of <=4 columns: 0.842 stock vs 0.849 ms.
  - Warp padding in block token dimension: no change.
- **Forcing MMQ for MoE:** `GGML_CUDA_FORCE_MMQ=ON` or `GGML_CUDA_MMID_MAX_BATCH=0` is 2x slower at batch 1 and 2.5x slower at batch 5-6 (0.999 vs 0.479 ms). First-gen Volta tensor cores waste tile capacity on skinny matrices.
- **The Volta fork route:** `jusko-llama-volta-qwen3flash` / `mistrjirka` measured at 18.61 tok/s (vs 31.26 control); draft acceptance collapsed to 0.678.
- **f16 KV and non-q8_0 KV:** f16 KV measured at 17.74 tok/s (-43%). q8_0 KV is the optimal ceiling on sm_72.
- **Runtime server flags:** Swept 28 configurations (`-ub`, `-b`, `--poll`, `--parallel 2`, `--backend-sampling`, `--swa-full`, `--kv-unified`, draft KV quants, priority, load modes). All non-harmful options land inside +-1% noise - **but that verdict is retracted for `--backend-sampling`, which was the +1.04% winner when re-tested with both orders on 2026-11-04.** The sweep was single-order, 3 reps, on the 31.5 tok/s build, and AGENTS.md rule 7 says that cannot resolve 1%; so the other 27 arms were **unsettled, not closed**. Two of them have since been re-tested properly and are now closed as negatives: **`--poll 50` reads +0.15% and `-t 1 -tb 1` reads -0.04%** (pooled n=8 per config in both orders, acceptance and mean len identical; single-order `--poll` looked like +0.60%, so the old protocol would have banked it). That leaves **25 arms still unsettled** - a flag from this list may be re-tested with `bench/one.sh` interleaved both ways (`bench/sweep-flagab.sh` is the template) and should be, before anything else is built. See `findings/cyber-tiel-qwen35moe/2026-11-04-backend-sampling.md` and `findings/cyber-tiel-qwen35moe/2026-11-04-flag-retest-poll-t1.md`.
- **Theoretical proposals from CRITICISM.md:**
  - Memory wall (we use 35 GB/s of 110 GB/s measured limit).
  - Pinned memory / zero-copy to bypass PCIe (Xavier is an integrated SoC with unified DRAM; `cudaMalloc` already resides in system memory).
  - `cp.async` / async prefetch (Ampere sm_80+ feature; does not exist on Volta sm_72).
  - Caching weights in Shared Memory (SM has 96 KB; weights are hundreds of MB and streamed once per token).
