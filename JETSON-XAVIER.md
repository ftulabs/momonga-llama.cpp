# Jetson AGX Xavier (Volta sm_72) local changes

This fork carries CUDA, speculative decoding, and long-context changes on top of `ggml-org/llama.cpp`
that exist to make 35B hybrid models (Cyber-Tiel-Coder-35B-A3B and Qwen3.6-35B-A3B) decode and prefill
faster on one board: **Jetson AGX Xavier, Volta sm_72, CUDA 11.4, 30,991 MiB unified memory**.
They are not proposed upstream - they are either board-specific or tailored to this architecture -
and all measurements come from on-device profiling.

**Current records on this board:**
- **43.08 - 43.97 tok/s speculative decode** median on 400-600 tokens (**47.5 - 48.4 tok/s** steady-state generation after reasoning tokens conclude) at `--spec-draft-n-max 6` with single-sync GPU draft sampler offload (`9c26dfe`), KV-only draft catch-up (`ba3e503`), reduced draft vocabulary LM head (`6ec6635`), hardware CUB radix sorting on CUDA 11.4 (`1f33695`), and `-bs` (`--backend-sampling`), up from 31.50 stock before any of it.
- **2,025 - 2,070 tok/s prefill** on long contexts via TokenGate layer-token pruning with decoupled prefill micro-batches (`TG_UBATCH=512`) and FlashAttention approximation (`--tokengate-fa`), collapsing client wire TTFT on 9.6k tokens from 53.3s to **4.79s** and on 22.1k tokens from 120s+ to **11.15s**.
- **100% exact needle retrieval (`BLUE-VELVET-99`)** verified in passkey needle-in-a-haystack at $K=2000, W=2$ running at **1,715.2 tok/s**.
- **262,144-token context (`-c 262144`)** with native `q8_0` KV cache verified live, consuming 8.25 GB KV cache and leaving **4.4 GB of free RAM headroom** on the 32 GB Xavier, decoding at **43.04 - 43.17 tok/s**.
- **MTP-free profile:** Without the MTP drafter, `-ub` becomes a free prefill parameter: **27.86 tok/s plain decode** (`-bs`) with the `all-q4k` file (23.75 at 8.8k context) and **215.8 tok/s dense prefill / 53.3s TTFT at 11,484 tokens with `-ub 2048`** (against 83.8 tok/s at `-ub 32`).

---

## 1. `cuda : use MMVQ for MUL_MAT_ID on sm_72 and drop the per-type batch cap`

`get_mmvq_mmid_max_batch()` tested the compute capability with equality against
`GGML_CUDA_CC_VOLTA` (700). Xavier reports 720, so the test failed and the per-type cap fell
through to the `pascal_older` table. Above that cap `MUL_MAT_ID` silently switched from the MMVQ
MoE kernel to MMQ, which is roughly 2.5x slower at these batch sizes - and speculative decoding
with a multi-token-prediction head runs at batches 3-7, exactly where the cliff sat.

Two changes:
- compare the capability as a range, `cc >= GGML_CUDA_CC_VOLTA && cc < GGML_CUDA_CC_TURING`, so
  sm_72 (and sm_73) select the Volta row;
- put the token index on `blockIdx.z` instead of `blockDim.y`, so a block is always one warp and
  any batch is a legal launch. The old layout needed `__launch_bounds__` sized to the cap, which
  is why the cap could not simply be raised: anything above the table gave
  `CUDA error: invalid argument`.

Measured on an 18 GiB IQ3_S MoE model (35B total, 3B active), 400 generated tokens, median of 5
runs: gate/up at 5 tokens went from 0.884 ms to 0.410 ms, and the model from 31.50 tok/s at
`--spec-draft-n-max 3` to 34.49 at `--spec-draft-n-max 6` (draft acceptance 0.989 -> 0.967).
The patch alone changes nothing at batch 4, where both paths were already legal; its value is
that deeper drafting stops being a penalty.

---

## 2. `cuda : run the IQ3_S MoE kernel at 64 registers instead of 79`

`mul_mat_vec_q_moe` declared `__launch_bounds__(256, 1)` while the kernel is launched with 32
threads, so the second argument was inert and the compiler spent 79 registers. That capped
residency at 24 one-warp blocks per SM, or 37.5% occupancy. Declaring the real block size and
32 blocks per SM - for IQ3_S only, because a blanket 64-register budget makes Q6_K spill -
gives 64 registers and 47.9% occupancy. Kernel time falls 7.6% and the model goes 34.30 ->
35.07 tok/s (+2.25%) with acceptance unchanged at 0.96657 and mean length 5.68.

32 blocks per SM is the sm_72 maximum and all four occupancy limits (blocks, registers, shared,
warps) now sit at 32, so this kernel is finished for a one-warp block. Measured in situ
afterwards, its cost is exactly linear in token count - 0.1392 ms per token at 7 columns for
gate/up, 0.0818 for down, intercept ~0.006 ms - because each token routes to its own eight
experts, so it is a DRAM stream at 44-52 GB/s with no launch overhead and nothing to reuse
across tokens.

---

## 3. `cuda : four rows per block and a wider small_k trigger on the dense MMVQ path`

Four changes to `mmvq.cu` that only move work assignment and block geometry, so none of them
changes a number the model computes - acceptance 0.96657 and mean length 5.68 are identical
before and after.

The problem: at wide column counts the 2 KB activation vector `y` is re-read once per output
row, and that re-read is 57% of the L1 load sectors at 7 columns. Four rows per block halves
it, but needs the shared-memory carveout raised from the driver's 16.38 KB default to 64 KB
(otherwise residency caps at 4 blocks and the win inverts) and needs `__launch_bounds__(.., 8)`
at those column counts (otherwise 4 rows take q6_K to 158 registers and 6 blocks, eating the
whole gain). The fourth change takes the wider `small_k` layout whenever K is an exact multiple
of one block iteration, not only when K is shorter than one.

The same rows-per-block lever applies at one column, which is the 1-column Q6_K lm_head that
drafts all six speculative steps: 4.776 -> 4.552 ms on that kernel.
Together: 35.07 -> 36.25 -> **36.56 tok/s**. L1 load sectors fall 24% at 1 column and 38% at 7
at unchanged occupancy.

---

## 4. `cuda : opt-in per-op timing dump behind GGML_CUDA_OP_TIME`

Setting `GGML_CUDA_OP_TIME` records a CUDA event pair per node, disables graph capture, and
every fifth graph prints the accumulated totals (top 16, aggregated by node name with the layer
suffix stripped) to stderr. It costs real time and is off by default. Use it for shares, not
for absolute tok/s.

---

## 5. `cuda : pick the MoE prefill tile width from tokens per expert`

`mmq.cu` chooses the MMQ tile width `J` from `ncols_opt = ne12` for every multiplicand, but a
`mul_mat_id` launch with 32 tokens over 256 experts and 8 routed experts gives each expert about
**one** useful column. The kernel still computes a full `J`-column tile: `vec_dot` has no column
bound and only the write-back honours `tile_y_max_j`, so 31 of 32 columns' worth of dp4a work is
discarded. The code already had the right rule for RDNA3/RDNA4 - pick `J` from the average tokens
per expert, `ceil(ne12*n_expert_used/ne02)` - and this patch applies it on NVIDIA as well.

Tightening the tile is bit-preserving: `J` decides which thread owns an output element, not the
order of its K sum, so acceptance and mean draft length come back identical.
Measured on this board at `-ub 32 -b 32`:
- 590 tokens: 78.8 -> 87.1 tok/s (+10.5%)
- 2,385 tokens: 75.8 -> 84.3 tok/s (+11.2%)
- 9,563 tokens: 71.0 -> 79.5 tok/s (+12.0%)

---

## 6. `cuda : pack destination rows per block in non-contiguous concat`

In Gated DeltaNet state updates, concatenation of recurrent state matrices involved strided
memory access across token boundaries. Packing destination rows per block in `concat_non_cont`
streamlines memory coalescence across the 30 recurrent layers, yielding +0.29% end-to-end
(36.56 -> 36.63 tok/s).

---

## 7. `speculative : single-sync sampling (PR #30196) and GPU draft sampler offload (9c26dfe)`

Host-side profiling via `perf stat` revealed that the speculative decoding loop was spending
41.7 ms of CPU time per 156.5 ms pass, leaving the GPU idle for ~7.8 ms per pass. The primary
culprit was CPU-side sampling: scanning and copying logits for 248,320 vocabulary items on the
Carmel CPU and synchronizing CUDA streams after every draft step.

Two changes resolve this:
- **Single-sync sampling:** Merges speculative draft execution into a single transaction,
  reducing CPU-GPU roundtrips.
- **GPU Draft Sampler Offload (`9c26dfe`):** Offloads the draft token sampling chain directly
  to the GPU in `common/speculative.cpp`, using native device memory buffers and eliminating
  host data transfers entirely.
Together with `--backend-sampling` (`-bs`), this eliminated the 7.8 ms GPU bubble and elevated
speculative decode throughput from **36.93 tok/s to 43.08 - 43.97 tok/s** median (and **47.5 - 48.4 tok/s** in steady state).

---

## 8. `llama : KV-only graphs for MTP draft catch-up (PR #31750)`

During speculative verification, when the base model verifies 6 draft tokens, the trailing MTP
layer (`blk.40`) only needs to catch up its recurrent and KV state for the accepted tokens rather
than computing a full output projection across all draft candidates. Using dedicated KV-only
computation graphs skips unnecessary output head GEMMs during catch-up passes.

---

## 9. `llama : --spec-draft-vocab for reduced draft LM head (6ec6635)`

The base model has a vocabulary of 248,320 tokens. Evaluating the draft LM head at each step
moves 248,320 weights per column. By pinning active vocabulary subsets via
`--spec-draft-vocab /mnt/ssd/models/draft-vocab-32k.json` (32,768 most frequent tokens), draft
LM head evaluation latency is reduced by >60% while maintaining draft acceptance at 0.968 - 0.975.

---

## 10. `cuda : enable CUB hardware radix sort on CUDA 11.4 sm_72 (1f33695)`

Upstream llama.cpp gated NVIDIA CUB hardware radix sorting behind newer CUDA toolchains. Enabling
CUB on CUDA 11.4 sm_72 activates native device-side sorting for top-k and top-p filtering during
backend sampling (`-bs`), eliminating CPU fallback on sampling operations.

---

## 11. `tokengate : long-context prefill acceleration (>2,000 tok/s), decoupled ubatch, and FA approximation`

On a 35B hybrid model, standard dense prefill is compute- and bandwidth-bound to ~78-87 tok/s at
`-ub 32` (and physically capped at ~250 tok/s at `-ub 2048`). TokenGate introduces learned token
pruning:
- **Phase 1 (layers 0..3):** Evaluates all tokens to identify attention salience.
- **Phase 2 (layers 4..40):** Prunes 85-95% of tokens, evaluating only $K=350-400$ kept tokens
  through the 37 heavy MoE/GDN layers.
- **Decoupled Prefill Ubatch (`TG_UBATCH=512`):** Previously, TokenGate inherited `-ub 32` from
  the decode launch configuration, throttling prefill at batch-32 GEMMs. We decoupled the
  prefill micro-batch size exclusively within `tokengate_prefill_ext()` to 512, preserving
  `-b 32 -ub 32` for decode.
- **FlashAttention Approximation (`--tokengate-fa`):** Approximates attention matrix reduction
  across pruned tokens.
Result: Prefill throughput jumps from **79.5 tok/s to 2,025 - 2,070 tok/s**, collapsing TTFT
from 120s+ to **4.79s on 9.6k tokens** and **11.15s on 22.1k tokens**, with **100% exact needle
retrieval (`BLUE-VELVET-99`)** verified.

---

## 12. `architecture : 262,144 context allocation with q8_0 KV cache in unified memory`

Standard 35B dense transformers require 43-86 GB for KV cache at 262k context, exceeding Xavier's
31 GB RAM. Cyber-Tiel-Coder-35B-A3B is a hybrid model:
- 30 recurrent Gated DeltaNet layers maintain fixed $O(1)$ state matrices regardless of sequence length.
- Only 11 full attention layers allocate KV cache.
Consequently, allocating **262,144 context (`-c 262144`)** with high-precision `q8_0` KV cache
consumes only **8.25 GB**, leaving **4.4 GB of free RAM headroom** on the 32 GB Xavier and
sustaining **43.08 tok/s** decode.

---

## Build and Run

Run `./build-best.sh` to compile with verified flags and `./run-best.sh` to launch the production server.
