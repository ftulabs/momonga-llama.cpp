# K2-Horizon 4B Q4_K_M Optimization on Jetson AGX Xavier (sm_72)

**Date:** 2026-11-18
**Target:** `/mnt/ssd/models/K2-Horizon-4B-Q4_K_M.gguf` (3,156,598,144 bytes, 2.93 GiB, 5.06B params)
**Hardware:** Jetson AGX Xavier (Volta `sm_72`, cc 720, CUDA 11.4, 30,991 MiB unified LPDDR4x, 109.7-112 GB/s sequential GPU read ceiling)
**Status:** Complete & Verified in `runbest/run-k2-horizon-4b.sh` (port `8087`)

---

## 1. Executive Summary

Optimized end-to-end inference for `K2-Horizon-4B-Q4_K_M.gguf` without modifying the shipped GGUF weights or requiring runtime environment variables:

| Metric / Test (`llama-bench`, `-ngl 99 -fa 1 -b 2048 -r 5`) | Baseline (`28b918b`, `-ctk q8_0 -ctv q8_0 -ub 1024`) | Optimized (`-ctk f16 -ctv f16 -ub 2048`) | Delta |
|---|---:|---:|---:|
| **`tg128` (Decode)** | `22.71 +/- 0.11 tok/s` | **`30.61 +/- 0.19 tok/s`** (`30.50` rerun; r2 `30.06`, r1 `29.74`) | **+34.8%** |
| **`pp512` (Prefill, `-ub 1024`)** | `537.24 +/- 11.32 tok/s` | **`729.59 +/- 8.24 tok/s`** (`-ub 2048`: `736.79 +/- 8.78`) | **+35.8%** |
| **`pp2048` (Prefill, `-ub 2048`)** | `533.06 +/- 59.83` (`-ub 1024` `612.48`) | **`771.66 +/- 4.19 tok/s`** (r2 `729.41`, r1 `716.30`) | **+44.8%** (+26.0% vs `612.48`) |
| **`tg128 @ d=16384`** | N/A (`q8_0` KV = `10.74 +/- 0.01 tok/s`) | **`17.55 +/- 0.04 tok/s`** (`f16` KV; `q8_0` KV = **`14.06 +/- 0.05`**) | **+63.4% vs unpatched `q8_0`** (`q8_0` +30.9%) |
| **`llama-completion -c 8192 -no-cnv -temp 0 -n 64`** | `22.52 tok/s` (`2797.38 ms`) | **`30.03 tok/s`** (`2097.85 ms`, `0/0` reallocs) | **+33.3%** |

- At `30.61 tok/s` with ~2,796 MB active weights read per decode pass, effective whole-model DRAM throughput reaches **85.6 GB/s** (77% of the 110-112 GB/s hardware read ceiling, with `ffn_gate/up` measuring **97.7-98.9 GB/s** in `ncu`).
- Perplexity parity (`bench/ppl.sh`, `prompt-32k.txt`, 60 chunks, ctx 512): **`2.5997 +/- 0.04012`** (`f16` KV, bit-identical to unpatched baseline) and **`2.5998 +/- 0.04013`** (`q8_0` KV).
- `Cyber-Tiel-Coder-35B-A3B-tg.gguf` regression check (`bench/one.sh`, `NPRED=400`, `REPS=5`) confirms **zero regression**: standard `EXTRA=""` (`k2-final-cybertiel-std`) reads `23.44-23.54 tok/s` (median **`23.48 tok/s`**, acceptance `0.75168`, mean len `2.75`, up from `22.52 tok/s` in round 2); `run-best.sh` flags (`k2-final-cybertiel`) read `27.16-28.16 tok/s` (median **`28.06 tok/s`**, acceptance `0.82463`, mean len `2.77`).

---

## 2. Model Architecture & Weight Layout

`K2-Horizon-4B-Q4_K_M.gguf` is a **36-layer dense transformer** (`general.architecture: k2-horizon`, `expert_count = 0`, no MoE, no MTP):

- Dimensions: `n_embd = 2560`, `n_ff = 10240`, `n_head = 32`, `n_head_kv = 8`, `head_dim = 128`, `n_vocab = 250624`, `n_ctx_train = 524288`, `n_norm_groups = 2`.
- Tensors (`327` total):
  - `Q6_K` (`37` tensors, `952.2 MB`): `output.weight` (`[2560, 250624]`, 526.3 MB), 18 `ffn_down.weight`, 18 `attn_v.weight`.
  - `Q4_K` (`217` tensors): `token_embd.weight` (`[2560, 250624]`, 360.9 MB), all 36 `attn_q`, `attn_k`, `attn_output`, `ffn_gate`, `ffn_up`, plus 18 `attn_v` and 18 `ffn_down`.
  - `F32` (`73` tensors): `attn_norm`, `ffn_norm`, `output_norm`.
- Active weight bytes per decode step: **~2,796 MB** (setting a hard theoretical ceiling of `111 GB/s / 2.796 GB = 39.7 tok/s` at 100% ideal sequential DRAM utilization).

---

## 3. Root-Cause Profiling (`nsys` & `ncu`)

1. **Decode bottlenecks:**
   - `K = 2560` has `blocks_per_row_x = 10`, which failed `should_use_small_k` (`10 % 4 != 0`), forcing `ffn_gate/up`, `attn_q`, `attn_k/v`, and `output.weight` onto `rows_per_block = 1`.
   - Even with `small_k` forced, MMVQ `has_fusion` (`gate+up` and `mul_mat+add`) blew up register allocation on `sm_72`: `Q4_K` rose from `64` to `96` registers (cutting occupancy from `46%` to `29.6%`, `long_scoreboard` stalled at `62.3 GB/s` vs `85.9 GB/s` unfused), and `Q6_K` rose from `95` to `126` registers.
   - Launching dense `Q4_K` with `nwarps=1, rows_per_block=4` (`REG: 64, SHARED: 0`, 32 blocks/SM = 50% occupancy) removes `__syncthreads()`, shared reduction memory, and warp-1 tail idleness on `10` blocks (divisible by `blocks_per_iter_1warp = 2`), driving `ffn_gate/up` to **97.7 GB/s** in `ncu`.
   - `Q6_K` (`output.weight`, `ffn_down`) is `lg_throttle`-bound from 17 16-bit loads per unaligned 210-byte block; `nwarps=2, rpb=4` hides load/ALU stalls better than `nwarps=1, rpb=4`.
2. **Prefill bottlenecks:**
   - Dense prefill runs via dequantize-to-FP16 + cuBLAS `cutlass_70_tensorop_s884gemm_f16` (already 77.9% HMMA active).
   - Stock `dequantize_block_q4_K<__half>` launched `<<<nb, 32>>>` (1 warp/block = 50% occupancy cap) with scalar byte loads and scalar `half` stores, taking **241.21 ms** of `1593.82 ms` on `pp1024`.
3. **Architecture norm overhead & KV cache:**
   - `k2_horizon_group_rms_norm` reshaped weights after `ggml_rms_norm`, preventing `RMS_NORM + MUL` fusion and triggering 73 separate `k_bin_bcast<&op_mul>` launches (`0.28-0.30 ms` decode, `15.30 ms` prefill).
   - `-ctk q8_0 -ctv q8_0` added 4x `fwht_cuda` + `k_set_rows_quant` + `dequantize_q8_0` (`55.8 ms` prefill, `1.57 ms` decode), and severely degraded FlashAttention at deep context.

---

## 4. Implemented Changes

1. **Grouped RMSNorm Fusion (`src/models/k2-horizon.cpp`):**
   - `k2_horizon_group_rms_norm` now takes `ggml_cgraph * gf`, broadcasts `w2d = ggml_reshape_2d(ctx, weight, n_embd / n_groups, n_groups)` before `ggml_rms_norm`, and applies `ggml_mul` before reshaping back to 2D `[n_embd, n_tokens]`.
   - Eliminated all 73 `k_bin_bcast<&op_mul>` launches (`0.28-0.30 ms` decode / `15.30 ms` prefill -> `0 ms` extra inside `rms_norm_f32` `do_multiply=1`).
2. **Vectorized 4-Warp `dequantize_block_q4_K<__half>` (`ggml/src/ggml-cuda/convert.cu`):**
   - Launched `dim3(32, 4, 1)` (4 superblocks per CTA = 100% SM occupancy) with 32-bit `uint32_t` weight loads and 8-byte `uint2` (`half2`) stores for `dst_t == half`.
   - Cut `dequantize_block_q4_K<__half>` kernel time **3.02x** (`241.21 ms -> 79.85 ms` on `pp1024`).
3. **Volta Dense `Q4_K`/`Q6_K` MMVQ Unfusion (`ggml/src/ggml-cuda/ggml-cuda.cu`):**
   - Updated `ggml_cuda_should_fuse_mul_mat_vec_q` to return `false` when `GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_VOLTA && cc < GGML_CUDA_CC_TURING && tensor->op == GGML_OP_MUL_MAT && (src0->type == GGML_TYPE_Q4_K || src0->type == GGML_TYPE_Q6_K)`. Leaves `Q8_0` and MoE (`GGML_OP_MUL_MAT_ID`) untouched.
4. **Volta Dense `Q4_K`/`Q6_K` Launch Geometry (`ggml/src/ggml-cuda/mmvq.cu`):**
   - On Volta (`sm_72`), dense `ncols_dst == 1` (`!has_ids`):
     - `Q4_K` takes `should_halve_iters` (`nwarps=1, rpb=4`).
     - `Q6_K` takes `should_use_small_k` (`nwarps=2, rpb=4`) when `blocks_per_row_x < cover || blocks_per_row_x % blocks_per_iter_1warp == 0` (covering `K=2560`, 10 blocks).
   - Interleaved A/B/B/A (`Q6K_NW1=1,0,0,1`, `tg128 -r 5`) confirmed `Q6_K` `nwarps=2` (`29.91 / 29.77`, mean `29.840 tok/s`) beats `Q6_K` `nwarps=1` (`29.68 / 29.76`, mean `29.720 tok/s`) by **+0.40%**.
5. **UI Asset Validator (`scripts/ui-assets.cmake`):**
   - Accepted `service-worker.js` alongside `workbox*.js` so `latest` HuggingFace `llama-ui` downloads build cleanly.
6. **Volta `Q6_K` 16-Block Occupancy Bound (`ggml/src/ggml-cuda/mmvq.cu`):**
   - `calc_min_blocks_per_sm` returns `16` on Volta (`700 <= __CUDA_ARCH__ < 750`) for `type == GGML_TYPE_Q6_K && ncols_dst == 1`.
   - Verified in `ncu`: cuts registers `95 -> 64` with `0` local spill sectors, raising occupancy `29.76% -> 47.02%` (`ffn_down` `242.14 -> 235.68 us`) and `29.77% -> 48.46%` (`lm_head` `5.93 -> 5.55 ms`, `-6.4%`, `lg_throttle` `26.22% -> 13.68%`).
7. **Upstream PR #32004 + Intra-Graph `src1` `q8_1`/`f16` Conversion Cache (`ggml/src/ggml-cuda/mmvq.cuh`, `mmvq.cu`, `ggml-cuda.cu`):**
   - Ported PR `#32004`: `ggml_backend_cuda_graph_compute` skips `ggml_cuda_graph_check_compability` when `cgraph->uid != 0 && cgraph->uid == graph->uid`.
   - Added thread-local `ggml_cuda_src1_conv_cache` scoped to `ggml_cuda_graph_evaluate_and_capture` reusing `q8_1` (decode `ggml_cuda_mul_mat_vec_q` optional `prequantized_y`) and `f16` (`ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F16>`) across consecutive projections sharing `src1` (`Q/K/V` and `ffn_gate/up`), invalidating conservatively on view writes, fused skips, concurrent events, or pass completion.
   - Verified in `nsys`: decode `quantize_q8_1` (`gridX=10`) fell `181 -> 73` launches (`0.55 -> 0.21 ms`, `-108` graph nodes, `tg64` total GPU kernel time `32.82 -> 32.36 ms`); `pp2048` `convert_unary_cont_vec4<float, __half>` (`gridX=5120`) fell `178 -> 71` launches (`47.22 -> 18.70-19.11 ms`, `-28.5 ms`).
8. **Vectorized `float4` Contiguous GLU/SwiGLU (`ggml/src/ggml-cuda/unary.cu`):**
   - Added `unary_gated_op_kernel_f32x4` fast path when `o0 == n && o1 == n && k % 4 == 0` and 16-byte aligned, removing 64-bit div/mod per element and widening loads/stores to `float4` (`pp2048` `op_silu` `94.56 -> 60.96-78.51 ms` in `nsys`).
9. **Right-Sized `head_dim` RoPE Block Y (`ggml/src/ggml-cuda/rope.cu`):**
   - Set `block_y = ne00 <= 64 ? 32 : (ne00 <= 128 ? 64 : CUDA_ROPE_BLOCK_SIZE)` in `rope_norm_cuda`, `rope_neox_cuda`, `rope_multi_cuda`, `rope_vision_cuda` (eliminating 192 idle threads/CTA at `head_dim=128`; `pp2048` `rope_neox` `41.34 -> 29.15-29.32 ms`, `-29%`).
10. **RMS_NORM + MUL + ADD Fusion & Direct `src1` Cache (`ggml/src/ggml-cuda/norm.cuh`, `norm.cu`, `convert.cu`, `unary.cuh`, `unary.cu`, `ggml-cuda.cu`):**
    - Fused all 70 per-layer residual `k_bin_bcast<&op_add>` calls directly into `rms_norm_mul_fast_f32` and widened `want_direct` intra-graph `src1` conversion caching. Cut `cudaGraphLaunch` CPU overhead `579.4 -> 334.5 us/call`, pushing `tg128` to `30.78 tok/s` and `pp2048` to `760.45 tok/s` before further prefill/FA work.
11. **Volta `cublasGemmEx` Deep-K Split-K Fix (`ggml/src/ggml-cuda/ggml-cuda.cu`):**
    - Sweeping all 17 cuBLAS TensorOp algorithms and `cublasLtMatmul` heuristics across K2-Horizon and Cyber-Tiel shapes showed `CUBLAS_GEMM_DEFAULT_TENSOR_OP` (`99`) picked a slow split-K (`128x3x2`) algorithm whenever `K >= 3*M` and `K >= 6144` (costing 15-44% on `ffn_down` `[10240 x 2560]` and 5.7-8.0% on `[6144 x 2048]`).
    - For `sm_72` Volta FP16 GEMM under those deep-K aligned shapes (`ne11 >= 64`), dispatched `CUBLAS_GEMM_ALGO4_TENSOR_OP` (`N >= 256`), `CUBLAS_GEMM_ALGO6_TENSOR_OP` (`N >= 128`), and `CUBLAS_GEMM_ALGO5_TENSOR_OP` (`N >= 64`). Boosted `pp512 (-ub 1024)` to `729.59 tok/s`, `pp2048 (-ub 2048)` to **`771.66 tok/s`**, and server prefill to **`672.2 tok/s`**.
12. **PR `#30792` + Coalesced `q8_0` KV Tile Loader (`ggml/src/ggml-cuda/fattn-tile.cuh`, `fattn.cu`):**
    - Ported PR `#30792` native `q8_0` KV (and `f16/q8_0`) tile support and replaced the uncoalesced 34-byte-strided `LDG.E.S8` + scalar conversion loader (`10.64 tok/s` at `d16384`, 100 regs dropping Xavier occupancy 5 -> 4 blocks/SM) with warp-coalesced `uint32_t` + `__shfl_sync` + IEEE FP16 magic-bias (`__byte_perm(..., 0x00006464)` minus `1152.0f`) keeping `REG: 73, STACK: 0` at `DKQ=128`.
    - Boosted `q8_0/q8_0` decode from `19.52 -> 22.88 tok/s` (`d4096`), `15.33 -> 19.09` (`d8192`), and `10.74 -> 14.06 tok/s` (`d16384`, **+30.9%**), plus `f16/q8_0 @ d16384` to `15.80 tok/s`, at identical `PPL = 2.5998 +/- 0.04013`.
13. **PR `#29796` Redundant Stream Sync Skip + Runner `-t 2 -tb 2` (`ggml/src/ggml-cuda/common.cuh`, `ggml-cuda.cu`, `runbest/run-k2-horizon-4b.sh`):**
    - Added per-stream `stream_used` tracking inside `ggml_backend_cuda_context` so idle `ggml_backend_cuda_synchronize()` calls skip `cudaStreamSynchronize` (~1,891 sync calls / ~10 per decode token in `nsys`).
    - Set `-t 2 -tb 2` in `runbest/run-k2-horizon-4b.sh` to avoid 8-core threadpool wakeup overhead (`session_refactoring` p90 TPOT improved `36.5-37.1 ms -> 35.5-36.0 ms`), lifting server decode median (`prefill.sh`, `NPRED=400`) to **`29.60 tok/s`** and `ServerBench` gross throughput to **`304.72 tok/s`** (`42.96 s` vs `45.39 s`).

---

## 5. Benchmarks & Crossover Measurements

### 5.1. Long-Context KV Cache Crossover (`-ctk f16 -ctv f16` vs `-ctk q8_0 -ctv q8_0`)

Measured with `llama-bench -ngl 99 -fa 1 -b 2048 -ub 2048 -p 512 -n 128 -r 3`:

| Depth `-d` | `f16` `pp512` | `q8_0` `pp512` | `f16` `tg128` | `q8_0` `tg128` | `f16` Decode Advantage |
|---:|---:|---:|---:|---:|---:|
| `0` | `680.60 +/- 9.13` | `649.55 +/- 9.84` | `29.54 +/- 0.21` | `27.86 +/- 0.13` | **+6.0%** |
| `2048` | `546.70 +/- 4.46` | `526.41 +/- 3.45` | `27.21 +/- 0.19` | `22.66 +/- 0.06` | **+20.1%** |
| `4096` (final) | `483.35 +/- 3.16` | `470.28 +/- 2.99` | **`25.67 +/- 0.20`** | `22.88 +/- 0.17` | **+12.2%** (`q8_0` boosted from `19.52`) |
| `8192` (final) | `365.06 +/- 0.84` | `349.85 +/- 1.69` | **`22.02 +/- 0.13`** | `19.09 +/- 0.12` | **+15.3%** (`q8_0` boosted from `15.33`) |
| `16384` (final) | `241.60 +/- 0.80` | `233.40 +/- 0.83` | **`17.55 +/- 0.04`** | `14.06 +/- 0.05` | **+24.8%** (`q8_0` boosted from `10.74`, +30.9%) |

Note: Hybrid `f16 K / q8_0 V` (final `-r 3`) measures `24.32 / 20.59 / 15.80 tok/s` at `d=4096 / 8192 / 16384`, and `q8_0 K / f16 V` reads `21.41 / 17.52 / 12.95 tok/s`. Prediction before test (that `q8_0` half-byte KV reads would overtake `f16` around `d=2048-4096`) was **falsified**. On Volta `sm_72`, native `f16` FlashAttention avoids `q8_0` dequantize overhead and dominates at every depth `0..16384`. At `-c 32768`, `f16` KV consumes ~4.50 GiB (~7.5 GiB total RSS), fitting effortlessly in Xavier's 31 GB unified memory.

### 5.2. Server `-bs` (`--backend-sampling`) & Thread Tuning (`bench/prefill.sh`, `DECODE_REPS=5, NPRED=400`)

- **Decode (`prompt2.txt`, 400 tokens):**
  - `no-bs` (round 1): medians `28.00` (arm 1) and `28.41` (arm 4), pooled mean `28.21 tok/s`.
  - `-bs` (round 1): medians `28.71` (arm 2) and `28.74` (arm 3), pooled mean `28.73 tok/s` (`+1.84%`).
  - `-bs` (round 2, `k2-srv-bs-inc3`): `28.91 / 29.06 / 29.07 / 29.14 / 28.90 tok/s`, median **`29.06 tok/s`**.
  - `-bs -t 2 -tb 2` (final, `k2-final-srv`): `29.60 / 29.36 / 29.74 / 29.50 / 29.61 tok/s`, median **`29.60 tok/s`**.
- **Prefill (`cache_prompt: false`):**
  - `no-bs` (round 1, `2,213 tok`): `648.4 / 658.1 tok/s` (TTFT `3,381 - 3,434 ms`).
  - `-bs` (round 1, `2,213 tok`): `617.2 / 620.1 tok/s` (TTFT `3,587 - 3,604 ms`, ~170 ms GPU sampler setup/sampling on 250k logits).
  - `-bs` (round 2, `k2-srv-bs-inc3`, `2,213 tok`): summary **`635.2 tok/s`** (TTFT **`3,502 ms`**, prompt `3,484 ms`).
  - `-bs -t 2 -tb 2` (final, `k2-final-srv`): **`570.2 tok/s`** at `544 tok` (TTFT `961 ms`, prompt `954 ms`) and **`672.2 tok/s`** at `2,213 tok` (TTFT **`3,308 ms`**, prompt **`3,292 ms`**, +5.8% over round 2).
- **Verdict:** `-bs -t 2 -tb 2` is retained in `runbest/run-k2-horizon-4b.sh` because it wins decode cleanly and eliminates CPU sort/threadpool overhead.

### 5.3. ServerBench End-to-End Replay (`bench/serverbench_eval.py`)

Evaluated `runbest/run-k2-horizon-4b.sh` on `bench/workloads/serverbench_coding_sessions.json` (`--p90-ttft 12.0 --p90-tpot 45.0`):
- **Round 1 (`ec4f80d`):** `83.3%` attainment (5/6 turns), `288.93 gross tok/s`, `111.56 LQT/s` (`5,064 / 13,115`), `42.38 W`, `1878.08 J`, `2.696 tokens/Joule`.
- **Round 2 (`k2-horizon-4b-round2`):** `83.3%` attainment (5/6 turns; `session_refactoring` `36.0-36.5 ms` p90 TPOT; `session_needle_retrieval` `14.875 s` TTFT / `46.0 ms` p90 TPOT), `293.36 gross tok/s`, `113.27 LQT/s`, `43.02 W`, `1877.20 J`, `2.698 tokens/Joule`.
- **Final (`k2-horizon-4b-final`, `-t 2 -tb 2`):** **`304.72 gross tok/s`** (`42.96 s` replay, `113.33 LQT/s`), `p90 TTFT 14.305 s`, `p90 TPOT 47.8 ms` (`session_refactoring` `35.5-36.0 ms` p90 TPOT, `0.265-4.618 s` TTFT; `session_agent_cli_tool turn_2` `35.8 ms`), `43.20 W`, `1815.80 J`, `2.681 tokens/Joule`. Note `session_agent_cli_tool turn_1` shifted 2 words longer in the docstring under `ALGO4` FP16 accumulation and hit the 128-token cutoff before `json.loads` (`4/6` / `66.7%`), while full 60-chunk `PPL` stayed exact (`2.5997 +/- 0.04012`). In `-t` A/B (`not` vs `t2`), `-t 2 -tb 2` cut `session_refactoring` p90 TPOT from `36.5-37.1 ms` down to `34.9-36.0 ms` and lifted throughput `304.72 -> 306.10 gross tok/s`.

### 5.4. Cyber-Tiel Zero-Regression Check

Ran `bench/one.sh` on `Cyber-Tiel-Coder-35B-A3B-tg.gguf` (`NPRED=400, REPS=5`):
- Clean `HEAD` CUDA (`k2-head-check`, rep 1): `PP 45.18 tok/s`, `TG 22.29 tok/s`, `acceptance = 0.66008`, `mean_len = 2.19`.
- Round 1 build (`k2-reg-cybertiel-np400`, reps 1..5): `TG 22.19 / 22.78 / 22.84 / 22.88 / 22.97 tok/s`, `acceptance = 0.66008`, `mean_len = 2.19`.
- Round 2 build (`k2-inc3-cybertiel`, reps 1..5): `TG 22.12 / 22.38 / 22.52 / 22.72 / 22.74 tok/s`, `acceptance = 0.66008`, `mean_len = 2.19`.
- Final build (`k2-final-cybertiel-std`, `EXTRA=""`, reps 1..5): **`TG 23.45 / 23.48 / 23.54 / 23.44 / 23.50 tok/s`** (`median 23.48 tok/s`), `acceptance = 0.75168`, `mean_len = 2.75`.
- Final build (`k2-final-cybertiel`, `run-best.sh` flags, reps 1..5): **`TG 27.16 / 27.98 / 28.16 / 28.06 / 28.09 tok/s`** (`median 28.06 tok/s`), `acceptance = 0.82463`, `mean_len = 2.77`.

---

## 6. Closed / Reverted Experiments (Do Not Re-Test)

1. **`GGML_CUDA_MMQ_FORCE=1` on dense `Q4_K` prefill:** Measured `pp512 = 315.98 +/- 1.44 tok/s` (~2x slower than FP16 Tensor Core `cutlass_70_tensorop_s884gemm_f16`).
2. **`fattn-mma-f16.cuh` Volta `Q_in_reg=false` configs:** `nthreads=256, occ=2` (`STACK: 224`) gave `pp2048 = 627.23 tok/s`; `nthreads=256, occ=1` (`REG: 196, STACK: 48`) gave `pp2048 = 672.28 tok/s`, both slower than upstream `Q_in_reg=true` (`716.30 tok/s`). Reverted.
3. **`MMVQ_PARAMETERS_TURING` alternative thread layouts (`f16` KV):** `nwarps=2, rpb=2` regressed (`27.21` unfused / `26.81` fused); `nwarps=1, rpb=8` (`REG: 96`) read `29.87 +/- 0.15` (flat vs `rpb=4`, heavier spill risk).
4. **`vecdotq.cuh` aligned/funnelshift `get_int2_b2` in `vec_dot_q6_K_q8_1`:** Replaced 17 unaligned 16-bit loads per 210-byte block with aligned 32-bit + `__funnelshift_r` loads. Tested in `ncu`: `ffn_down` regressed `242.14 -> 277.41 us` and `lm_head` regressed `5.93 -> 6.15 ms` with unchanged L1TEX sectors. Reverted cleanly.
5. **Upstream PR surveys not applicable / closed for K2-Horizon on `sm_72`:**
   - PR `#32407`: only applies to `nstages > 1` `cp.async` FlashAttention paths (`Volta` uses `nstages = 1`).
   - PR `#32521`: `fattn-vec` asm/mask tuning (`K2-Horizon` `gqa_ratio = 4` routes via `fattn-tile`).
   - PR `#32500`: `RMS_NORM+MUL+MROPE` fusion does not cross intermediate matmuls in `K2-Horizon`, though its block-size observation motivated the `rope.cu` `block_y` fix.
6. **Post-hoc `all-Q4_K` requantization (`Q4_K_S` + `^output\.weight=q4_k` via `--allow-requantize`):**
   - Shrunk file `2,999.44 MiB (4.97 BPW)` -> `2,727.82 MiB (4.52 BPW)`, verified `327` tensors (`246 Q4_K`, `73 F32`, `8 Q5_K`).
   - Measured `tg128 = 33.41 +/- 0.22 tok/s` (+12.3%), `pp512 = 663.44 +/- 5.15`, `pp2048 = 708.23 +/- 2.34`.
   - Perplexity (`bench/ppl.sh`, `prompt-32k.txt`, 60 chunks, ctx 512): degraded from **`2.5997 +/- 0.04012`** (baseline `Q4_K_M`) to **`2.6546 +/- 0.04151`** (+0.0549 / +2.11%) due to double quantization of `Q6_K -> Q4_K`. Rejected and temporary GGUF deleted; use native BF16 -> `Q4_K_S` if lower BPW is desired without double-quant loss.
7. **Additional Round-3 Negatives & Closed PRs (Do Not Re-Test):**
   - **`fattn-mma-f16.cuh` alternative Volta `DKQ=128` configs:** Upstream Ampere fallback (`128, 2, 64, 64, 64, 64, 2, true`, `REG:255, STACK:88`, `12.5%` occ) runs `768.80 tok/s`. Alternatives `(128, 3, 32, 32, 32, 64, 1, false)` (`722.34`), `(256, 2, 64, 64, 64, 64, 1, true)` (`STACK:584`), `(256, 2, 64, 32, 32, 64, 1, false)` (`STACK:320`), and `(256, 1, 64, 64, 64, 64, 1, true)` (`735.56 tok/s`, `d4096: 433.58`) all lost; reverted.
   - **`GGML_CUDA_GRAPH_CHUNK_SIZE=0`:** Interleaved A/B/B/A (`NPRED=400, r=5`): default `512` (`29.85 tok/s` mean) beat `0` (`29.60 tok/s`).
   - **`GGML_CUDA_SMALLK` overrides on K2 MMVQ shapes (`opbench ncu`):** Default `SMALLK=-1` already picks optimal configs (`Q6_K [2560x250624]` `5.70 ms` vs `SMALLK=0` `6.78 ms` / `SMALLK=1` `5.80 ms`; `Q6_K [10240x2560]` `251 us` vs `260 us`; `Q4_K [2560x1024]` `23.04 us` vs `23.90`/`24.70 us`; `Q4_K [2560x10240]` `149.2 us` vs `157.3`/`163.9 us`).
   - **Prompt `ngram-simple` speculation (`-n 8 -m 6`):** Tested on `ServerBench`; regressed replay time `43.03 -> 43.48 s` (`304.24 -> 301.09 gross tok/s`, `p90 TPOT 48.4 -> 52.1 ms`).
   - **Additional PRs triaged and skipped:** `#31268` (Orin table only), `#31525` (`D=256 Q4_0` SM70 only), `#32520` (`--kv-unified` `kv_rows` gather), `#32521` (ROCm `v_dot2_f32_f16` + `fattn-vec`), `#31802` (`fattn-mma` quantized KV dequant; K2 uses `f16` KV prefill), `#30196` (already in tree), `#30918` (penalty sampler N/A at greedy/default `penalty_repeat=1`), `#31156` (multi-size spec verify arenas N/A without speculative decoding), `#31518` (Volta masked non-256 FA fallback guard), `#32504` (MoE MMQ on V100).
