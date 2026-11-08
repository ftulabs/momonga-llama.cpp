# Forks, patches and unmerged kernels: what is worth having

Date: 2026-11-02. Hardware: Jetson AGX Xavier (GV11B, sm_72, CUDA 11.4.315, 30,991 MiB
unified LPDDR4x). Model: `Cyber-Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS.gguf`, 18.1 GB.

The question was whether a fork, a custom edit, or an unmerged/experimental kernel can
beat mainline `dde879b` on this model and board. This file records what was measured,
what was read in source, and what the remaining path actually is.

## The model, from its own header

| key | value |
| --- | --- |
| architecture | `qwen35moe` |
| blocks | 41 |
| experts / used | 256 / 8 |
| expert FF | 512 (shared FF 512) |
| embedding | 2048 |
| head count / kv heads | 16 / 2 (GQA 8) |
| key length / value length | 256 / 256 |
| nextn (MTP) layers | 1 |
| context | 262144 |

The D=256 / GQA-8 geometry matters: it is the exact shape the Volta forks were built
around, and it is the shape mainline does *not* have a tuned Volta flash-attention config
for (see below).

## Measured: the Volta fork loses badly

`jusko-llama-volta-qwen3flash` (a fork of `mistrjirka/llama.cpp` `v100-optimized`) was
already on disk and already built for sm_72 (`build-cuda`, `CMAKE_CUDA_ARCHITECTURES=72`,
CUDA 11.4). Same prompts, same flags, 3 reps of 400 tokens, `bench/prompt2.txt`.

| Config | TG tok/s (3 reps) | Median | Acceptance | mean len |
| --- | --- | --- | --- | --- |
| `v00` mainline control, q8_0 KV | 31.19 / 31.51 / 31.26 | **31.26** | 0.98940 | 3.69 |
| `v01` Volta fork, q8_0 KV | 18.32 / 18.61 / 18.68 | **18.61** | 0.67820 | 2.43 |
| `v03` mainline, **f16 KV** | 17.86 / 17.72 / 17.74 | **17.74** | 0.56911 | 1.98 |
| `v05` mainline, `-c 4096`, q8_0 | 31.23 / 31.67 / 31.62 | **31.62** | 0.98940 | 3.69 |

Two results stand out.

**The fork is 40% slower** (18.61 against 31.26). It is not a small regression that a flag
can fix: its draft acceptance collapses from 0.989 to 0.678, so it is producing different
numbers, not just doing the same work more slowly.

This reproduces an earlier, independent measurement rather than standing alone. On
2026-10-14 the same fork was A/B'd against BeeLlama at MTP depth 1
(`2026-10-14-builds-quants-and-models.md`): TG 21.36 against 24.42 (-12.5%), acceptance
0.7162 against 0.9389. Two sessions, two MTP depths, two different baselines, and the fork
loses TG and acceptance both times. The acceptance penalty is a property of the fork on this
model, not an artefact of one prompt.

**f16 KV is 43% slower than q8_0 KV** (17.74 against 31.26), with acceptance collapsing the
same way (0.569). Both failures share a signature, which suggests one mechanism: they route
decode attention onto the Volta tensor-core MMA path. q8_0 KV is not a type that path
handles natively, so it falls back to the TILE kernel; f16 KV and the fork's widened KV both
land on MMA. On this chip, for this shape, TILE is much faster in decode.

That reading is consistent with the fork's own documentation, which describes its edge as
*prefill*: "Prefill uses cuBLAS `TENSOR_OP` + FA MMA-F16 when Q is wide; decode FA on this
arch lands on TILE". Its published Ornith-1.5-35B-A3B row says the same thing in numbers:
**TG 59.26 -> 59.39 (neutral)**, PP 485.49 -> 580.36 (+19.5%). The fork spends its effort
where our benchmark does not.

**Context allocation does not affect decode.** `-c 4096` and `-c 32768` are identical
(31.62 vs 31.26). That is expected once you notice both runs only ever fill ~750 KV
entries; it is a check on the harness, not a result about long context.

## Read in source, not assumed

- **Mainline already enables Volta MMA.** `ggml/src/ggml-cuda/common.cuh:296` defines
  `VOLTA_MMA_AVAILABLE` for `__CUDA_ARCH__ == GGML_CUDA_CC_VOLTA`. This is upstream, not a
  fork feature. An earlier note in this project implied otherwise; that was wrong.
- **Mainline has no tuned Volta config for D=256.**
  `ggml_cuda_fattn_mma_get_config_volta()` (`fattn-mma-f16.cuh`) has cases for 320/256,
  512/512 and 576/512, then falls through to
  `ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols)` under a literal
  `// TODO tune specifically for Volta`. Our model is 256/256, so it takes the Ampere
  fallback. The jusko fork adds
  `CONFIG_CASE(256, 256, 32, 128, 2, 32, 128, 128, 64, 1, false)` and a `64` variant.
  This is a real gap in mainline, and it is a genuine cherry-pick candidate - but the
  config table is keyed on `ncols`, i.e. the width of Q. Decode has a narrow Q and does not
  use the MMA kernel at all, so this buys **prefill**, not tok/s.
- **The MoE is already on the right kernel.** `get_mmvq_mmid_max_batch()` returns
  `MMVQ_MAX_BATCH_SIZE` (8) for Volta (`mmvq.cu:443-444`), so `MUL_MAT_ID` uses MMVQ up to
  batch 8. Our verify batch is ~4.7. A recent fork PR tunes exactly this threshold and
  claims gains "below batch 6" - mainline is already above that bar. Nothing to take.
- **`GGML_CUDA_VOLTA_FORCE_MMQ=moe`** forces MMQ for routed MoE. On Volta mainline already
  selects MMQ whenever `!fp16_mma_hardware_available`, so for our batch this knob mostly
  shifts *prefill*, not decode.

## The outside landscape

| Project | What it is | Verdict for us |
| --- | --- | --- |
| `mistrjirka/llama.cpp` `v100-optimized`, `jackjusko/jusko-llama-volta-qwen3flash` | Deep Volta/Turing fork: FA MMA, 256x256 tiling, Qwen3.8 routing, NVLink/P2P, `--moe-cache` | **Measured 40% slower.** Its own Ornith row shows TG neutral. Its wins are PP and multi-GPU. |
| `fishlikeX/sm70-attn` (and `Flo5k5`, `jezza323` mirrors) | Ports 1Cat-vLLM's D256 split-D FlashAttention to Volta; SplitKV3; DFlash2; **sm_70 IQ4_XS dequant kernels** | Best technical match on paper, but **requires CUDA >= 12.x**; this board is pinned to 11.4. Targets dense Qwen3.8-27B. Its reported decode gain is at 174k context (26.0 vs 19.6), a regime we are not in. |
| `ikawrakow/ik_llama.cpp` | Performance fork, SOTA quants, strong CPU path; `docs/pascal-volta.md` | Volta support is real but partial, and **it does not use CUDA graphs below Ampere**. Our decode depends on graphs. Likely a net loss. |
| `Anbeeld/beellama.cpp` | KVarN KV quantization, precision tail, adaptive draft | Solves memory and KV quality, which are not our constraint. Adaptive draft is ruled out by our own marginal-cost data. |
| `homeofe/llama.cpp` | Scales across GPUs for models partly outside VRAM | Explicitly 1.0x once the model fits. Ours fits. |
| upstream issue #28037 | "Analysis: Volta (SM70/SM72) Flash Attention Optimization Paths", measured on **this exact board** with a Qwen 35B-A3B MoE | Open, no patch. Its own conclusion: the remaining gap is architectural, and its proposed split-D kernel is a **prefill** optimisation. |

## What this implies

The fork route is closed for decode on this model. Every project with real Volta work is
optimising attention and prefill, because that is where mainline is weak on Volta; our
measured bottleneck is the routed-MoE GEMV, where mainline is already on MMVQ + dp4a and
already above the batch thresholds those forks tune.

That does not mean the exploration was wasted. It converts a vague "maybe a fork is
faster" into three specific candidates, none of which is a drop-in:

1. **Prefill only, small, safe:** add the missing `256/256` Volta FA config. Bounded work,
   helps TTFT on long prompts, will not move tok/s.
2. **Decode, real work:** the Expert GEMV itself. This is now the only lever with 1.4x
   potential, and no existing fork supplies it for IQ4_XS on sm_72.
3. **Decode, cheap to test:** requantize to a lower bpw. Bytes per token falls
   proportionally, and decode is close enough to the memory-system limit for that to
   translate almost directly into speed. Costs quality; needs a BF16 source, since
   quantizing our IQ4_XS again would compound error.

## Subagent research: what survived checking

Two research passes were run. Several of their strongest claims did not survive verification
against the source, and the two that failed were both ranked first, so they are recorded here
explicitly:

- **`GGML_CUDA_GRAPHS_PRE_VOLTA` does not exist.** A claim that upstream ships an env knob to
  enable CUDA graphs below Ampere. `grep -rn PRE_VOLTA ggml/src/ggml-cuda/` returns nothing.
  The live graph knobs are `GGML_CUDA_DISABLE_GRAPHS`, `GGML_CUDA_GRAPH_CHUNK_SIZE` and
  `GGML_CUDA_GRAPH_OPT`.
- **"MMVQ re-reads each weight row once per token, so verify costs ~4.7x the traffic" is
  wrong.** In `mul_mat_vec_q` the column loop sits inside the K-block loop, so the same weight
  block is re-read back-to-back and hits cache, not DRAM. For the MoE path the kernel's own
  comment settles it: *"Expert rows are gathered per token, so a wider block adds reduction
  work without reuse"* (`mmvq.cu`, `should_halve_iters`). There is no cross-token weight reuse
  to recover, because two tokens rarely draw the same expert - with 256 experts, 8 used and a
  batch of ~4.7, the union is ~38 of ~40 draws. A row-shared small-batch GEMM would therefore
  save single-digit percent, not 4.7x.
- **PR #31268 (`MMVQ_PARAMETERS_ORIN`) is real but not for us.** It whitelists nine types;
  IQ4_XS is explicitly excluded because it gained only **1.4%** on a full model. Its mechanism
  (rows-per-block = nwarps on the single-column path) is nonetheless interesting, because
  mainline currently *disables* that path for IQ4_XS on Volta: in `should_use_small_k`,
  Volta fails `is_nvidia_turing_plus` and IQ4_XS is in `iq_slow_other`, so `use = false`
  whenever `ncols_dst == 1`. Flipping that is a small, testable experiment on plain decode.
- **`MMVQ_PARAMETERS_TURING` is shared by sm_70/72/75** (verified,
  `get_device_table_id`). Volta tuning from V100 does reach this board in principle, though
  none of it is tested on GV11B.
- **EAGLE-3 drafts for Qwen3.5-35B-A3B do exist** -
  `jiapingW/Qwen3.5-35B-A3B-Eagle3-Specforge` is real, 0.2B params, safetensors, trained on
  the *base* model. Its published accept length is **3.35** (training set) and **2.93**
  (ShareGPT). Our MTP already gets mean len **3.69** on this workload, and the draft is for the
  base model rather than this finetune. Low expected value.

## Open correction: the 12% bandwidth figure

`2026-11-02-bandwidth-and-flag-audit.md` computes decode bandwidth from one read of the active
weights per pass (~1.6 GB, ~13 GB/s). If expert rows are gathered per (token, expert) pair, a
pass with batch 4.7 reads ~38 expert matrices, not 8, so the true figure is several times
higher - on the order of 3-4 GB per pass. That does not overturn the two-stream result, but it
does mean "12% of peak" understates how hard the memory system is working. Not yet measured
directly; `tegrastats` reports `EMC_FREQ 0%` on this board and there is no usable DRAM counter
without `ncu`/`nsys`, both absent.

## Reproducing

`bench/sweep-fork.sh` runs the fork A/B (`v00`-`v04`), `bench/sweep-fork2.sh` the f16 and
short-context follow-ups. Both use `bench/one.sh` and append to `bench/results.csv`.

Note for anyone re-running these: `bench/one.sh` defaults to `prompt.txt`, which gives
acceptance ~0.71 and ~22 tok/s even on the known-good config. All historical numbers, and
every number above, use `prompt2.txt`. Comparing across the two is meaningless.
