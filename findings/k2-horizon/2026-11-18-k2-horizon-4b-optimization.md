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
| **`tg128` (Decode)** | `22.71 +/- 0.11 tok/s` | **`29.74 +/- 0.23 tok/s`** | **+31.0%** |
| **`pp512` (Prefill, `-ub 1024`)** | `537.24 +/- 11.32 tok/s` | **`680.62 +/- 8.23 tok/s`** | **+26.7%** |
| **`pp2048` (Prefill, `-ub 2048`)** | `533.06 +/- 59.83` (`-ub 1024` `612.48`) | **`716.30 +/- 2.70 tok/s`** | **+34.4%** (+17.0% vs `612.48`) |
| **`tg128 @ d=16384`** | N/A (`q8_0` KV = `10.74 +/- 0.01 tok/s`) | **`17.31 +/- 0.08 tok/s`** (`f16` KV) | **+61.2% vs `q8_0` KV** |
| **`llama-completion -no-cnv -temp 0 -n 64`** | `22.52 tok/s` (`2797.38 ms`) | **`29.68 tok/s`** (`2122.99 ms`, `0/0` reallocs) | **+31.8%** |

- At `29.74 tok/s` with ~2,796 MB active weights read per decode pass, effective whole-model DRAM throughput reaches **83.2 GB/s** (75% of the 110-112 GB/s hardware read ceiling, with `ffn_gate/up` measuring **97.7 GB/s** in `ncu`).
- `Cyber-Tiel-Coder-35B-A3B-tg.gguf` regression check confirms **zero regression** (bit-identical `0.66008` draft acceptance / `2.19` mean len between clean `HEAD` CUDA and patched build on `prompt2.txt`).

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

---

## 5. Benchmarks & Crossover Measurements

### 5.1. Long-Context KV Cache Crossover (`-ctk f16 -ctv f16` vs `-ctk q8_0 -ctv q8_0`)

Measured with `llama-bench -ngl 99 -fa 1 -b 2048 -ub 2048 -p 512 -n 128 -r 3`:

| Depth `-d` | `f16` `pp512` | `q8_0` `pp512` | `f16` `tg128` | `q8_0` `tg128` | `f16` Decode Advantage |
|---:|---:|---:|---:|---:|---:|
| `0` | `680.60 +/- 9.13` | `649.55 +/- 9.84` | `29.54 +/- 0.21` | `27.86 +/- 0.13` | **+6.0%** |
| `2048` | `546.70 +/- 4.46` | `526.41 +/- 3.45` | `27.21 +/- 0.19` | `22.66 +/- 0.06` | **+20.1%** |
| `4096` | `457.71 +/- 3.74` | `448.16 +/- 4.45` | `24.89 +/- 0.15` | `19.52 +/- 0.03` | **+27.5%** |
| `8192` | `349.46 +/- 3.17` | `340.61 +/- 2.23` | `21.63 +/- 0.05` | `15.33 +/- 0.01` | **+41.1%** |
| `16384` | `233.32 +/- 1.37` | `229.20 +/- 1.34` | `17.31 +/- 0.08` | `10.74 +/- 0.01` | **+61.2%** |

Prediction before test (that `q8_0` half-byte KV reads would overtake `f16` around `d=2048-4096`) was **falsified**. On Volta `sm_72`, native `f16` FlashAttention avoids `q8_0` dequantize overhead inside `fattn-vec` and dominates at every depth `0..16384`. At `-c 32768`, `f16` KV consumes ~4.50 GiB (~7.5 GiB total RSS), fitting effortlessly in Xavier's 31 GB unified memory.

### 5.2. Server `-bs` (`--backend-sampling`) A/B/B/A (`bench/prefill.sh`, `DECODE_REPS=5, NPRED=400, LENS=2048`)

- **Decode (`prompt2.txt`, 400 tokens):**
  - `no-bs`: medians `28.00` (arm 1) and `28.41` (arm 4), pooled mean `28.21 tok/s`.
  - `-bs`: medians `28.71` (arm 2) and `28.74` (arm 3), pooled mean **`28.73 tok/s`** (**+1.84%**).
- **Prefill (`2,213 tokens`, `cache_prompt: false`):**
  - `no-bs`: `648.4 / 658.1 tok/s` (TTFT `3,381 - 3,434 ms`).
  - `-bs`: `617.2 / 620.1 tok/s` (TTFT `3,587 - 3,604 ms`, ~170 ms GPU sampler setup/sampling on 250k logits).
- **Verdict:** `-bs` is retained in `runbest/run-k2-horizon-4b.sh` because it wins decode by `+1.84%` (breaking even in latency around ~265 generated tokens and preventing 250k CPU sort overhead under stochastic `top-k`/`top-p` chat sampling).

### 5.3. ServerBench End-to-End Replay (`bench/serverbench_eval.py`)

Evaluated `runbest/run-k2-horizon-4b.sh` on `bench/workloads/serverbench_coding_sessions.json` (`--p90-ttft 12.0 --p90-tpot 45.0 --tokengate-k 0`):
- **Attainment:** `83.33%` (5/6 turns qualified; `session_refactoring` 37.2-38.2 ms p90 TPOT, 0.25-4.78 s TTFT; `session_agent_cli_tool` 35.2-35.8 ms p90 TPOT, 0.42-0.49 s TTFT; 8,002-token `session_needle_retrieval` read 14.852 s TTFT and 47.9 ms p90 TPOT).
- **Throughput & Power (`tegrastats`):** `288.93 gross tok/s`, `111.56 LQT/s` (`5,064 / 13,115` billable tokens), `42.38 W` mean power, `1878.08 J` consumed, **2.696 LQT/Joule** (`6.98` gross tok/J).

### 5.4. Cyber-Tiel Zero-Regression Check

Ran `bench/one.sh` on `Cyber-Tiel-Coder-35B-A3B-tg.gguf` (`NPRED=400, REPS=5`):
- Clean `HEAD` CUDA (`k2-head-check`, rep 1): `PP 45.18 tok/s`, `TG 22.29 tok/s`, `acceptance = 0.66008`, `mean_len = 2.19`.
- Patched build (`k2-reg-cybertiel-np400`, reps 1..5): `TG 22.19 / 22.78 / 22.84 / 22.88 / 22.97 tok/s`, `acceptance = 0.66008`, `mean_len = 2.19` (bit-identical acceptance/length).

---

## 6. Closed / Reverted Experiments (Do Not Re-Test)

1. **`GGML_CUDA_MMQ_FORCE=1` on dense `Q4_K` prefill:** Measured `pp512 = 315.98 +/- 1.44 tok/s` (~2x slower than FP16 Tensor Core `cutlass_70_tensorop_s884gemm_f16`).
2. **`fattn-mma-f16.cuh` Volta `Q_in_reg=false` configs:** `nthreads=256, occ=2` (`STACK: 224`) gave `pp2048 = 627.23 tok/s`; `nthreads=256, occ=1` (`REG: 196, STACK: 48`) gave `pp2048 = 672.28 tok/s`, both slower than upstream `Q_in_reg=true` (`716.30 tok/s`). Reverted.
3. **`MMVQ_PARAMETERS_TURING` alternative thread layouts (`f16` KV):** `nwarps=2, rpb=2` regressed (`27.21` unfused / `26.81` fused); `nwarps=1, rpb=8` (`REG: 96`) read `29.87 +/- 0.15` (flat vs `rpb=4`, heavier spill risk).
4. **Post-hoc `all-Q4_K` requantization (`Q4_K_S` + `^output\.weight=q4_k` via `--allow-requantize`):**
   - Shrunk file `2,999.44 MiB (4.97 BPW)` -> `2,727.82 MiB (4.52 BPW)`, verified `327` tensors (`246 Q4_K`, `73 F32`, `8 Q5_K`).
   - Measured `tg128 = 33.41 +/- 0.22 tok/s` (+12.3%), `pp512 = 663.44 +/- 5.15`, `pp2048 = 708.23 +/- 2.34`.
   - Perplexity (`bench/ppl.sh`, `prompt-32k.txt`, 60 chunks, ctx 512): degraded from **`2.5997 +/- 0.04012`** (baseline `Q4_K_M`) to **`2.6546 +/- 0.04151`** (+0.0549 / +2.11%) due to double quantization of `Q6_K -> Q4_K`. Rejected and temporary GGUF deleted; use native BF16 -> `Q4_K_S` if lower BPW is desired without double-quant loss.
