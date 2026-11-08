# The MoE batch cliff is the dispatch table, not the kernel

Date: 2026-11-03. Hardware: AGX Xavier, sm_72, CUDA 11.4.
Prior write-ups treated "IQ3_S falls at 5 columns, IQ4_XS at 6" as a hardware pathology
of `MUL_MAT_ID` at small batch. It is not. It is llama.cpp choosing MMQ instead of MMVQ,
because of two separate bugs that both bite exactly at 4-5 columns.

## 1. sm_72 fails an exact-equality architecture test

`get_mmvq_mmid_max_batch(type, cc)` in `mmvq.cu`:

```
if (cc == GGML_CUDA_CC_PASCAL || cc == GGML_CUDA_CC_VOLTA || cc >= GGML_CUDA_CC_ADA_LOVELACE) {
    return MMVQ_MAX_BATCH_SIZE;   // 8
}
if (cc >= GGML_CUDA_CC_TURING) { return ..._turing_plus(type); }
return ..._pascal_older(type);
```

`GGML_CUDA_CC_VOLTA` is **700** (sm_70) and `cc` is computed as `100*major + 10*minor`,
so Xavier (sm_72) is **720**. The equality test fails, `cc >= TURING` (750) fails, and sm_72
silently lands in the `pascal_older` table:

| type | pascal_older cap |
|---|---|
| IQ3_S | 4 |
| Q6_K | 4 |
| Q4_K / Q5_K / IQ4_XS | 5 |
| Q8_0 | 4 |

The comment on that branch says "P100, Volta, Ada Lovelace, and Blackwell always use MMVQ
for MUL_MAT_ID". sm_72 never gets there.

## 2. Above the cap, MMQ is 2.5x slower - measured

`bench/opbench`, gate/up shape (K=2048, n=512, 256 experts, 8 per token), IQ3_S, 30 reps,
median of the loop, `GGML_CUDA_DISABLE_GRAPHS=1`:

| tokens | default dispatch | MMQ forced (`GGML_CUDA_MMID_MAX_BATCH=0`) |
|---|---|---|
| 1 | 0.189 ms (19.0 GB/s) | 0.367 ms (9.8 GB/s) |
| 2 | 0.211 (34.1) | 0.457 (15.8) |
| 3 | 0.285 (37.9) | 0.599 (18.1) |
| 4 | 0.347 (41.6) | 0.736 (19.6) |
| 5 | **0.887 (20.3)** | 0.870 (20.7) |
| 6 | **1.003 (21.6)** | 0.983 (22.0) |

Rows 5 and 6 in the default column are not a kernel degrading. They are MMQ, and they match
the forced-MMQ column to within 2%. The "cliff" is the switch point, and the switch point is
the `pascal_older` table value of 4 for IQ3_S.

The `[mmid-time]` instrumentation in the same binary confirms it directly: it prints for
tokens 1-4 and never for 5-6, because `ggml_cuda_mul_mat_vec_q` is not entered at those
batch sizes at all.

## 3. Why the table exists: the kernel cannot legally launch above the cap

`mul_mat_vec_q_moe` is compiled with
`__launch_bounds__(get_mmvq_mmid_max_batch_for_device<type>()*warp_size, 1)`
and launched with `blockDim = (warp_size, ncols_dst)` - one warp per token. For IQ3_S that
is a 128-thread bound. Forcing the cap up with 5 tokens gives a hard CUDA failure:

```
GGML_CUDA_MMID_MAX_BATCH=4 -> tok5 0.913 ms (MMQ)
GGML_CUDA_MMID_MAX_BATCH=5 -> CUDA error: invalid argument
GGML_CUDA_MMID_MAX_BATCH=6 -> CUDA error: invalid argument
```

`invalid argument` is the launch failing, not the kernel computing something wrong.

## 4. Fix: one warp per block, token on blockIdx.z

Moving the token from `blockDim.y` to `blockIdx.z` (`tok_in_z` template parameter,
`token_idx = tok_in_z ? blockIdx.z : threadIdx.y`) makes the block one warp at every batch
size, so the `__launch_bounds__` bound stops being a constraint. With the cap raised:

| tokens | default | tok_in_z + cap 8 | speedup |
|---|---|---|---|
| 4 | 0.342 ms (42.2 GB/s) | 0.342 (42.2) | 1.00x |
| 5 | 0.884 (20.3) | **0.410 (44.0)** | **2.15x** |
| 6 | 0.999 (21.6) | **0.479 (45.2)** | **2.09x** |

Bandwidth now scales linearly with tokens instead of halving, and 45 GB/s is the same
efficiency the op reaches at batch 4 - so this is the expected shape of the curve, recovered.

## Correction to an earlier result

`./2026-11-02-cliff-is-not-launch-geometry.md` reported that tok_in_z "leaves the
cliff unchanged" and used that to conclude the pathology was not block-level. That test was
**inert**: the env hook changed the launch geometry, but MMVQ was never selected at 5 and 6
tokens, so the modified kernel never ran. The negative result is withdrawn. The `[tokz]
ACTIVE ncols_dst=5` line that was cited as proof the hook fired came from a different op in
the same process, not from the op under test.

## The kernel change is numerically exact

`bench/opbench` with `OPBENCH_FILL` (weights quantized from a seeded pseudo-random matrix, so
the output is not identically zero), Q6_K, K=512, n=2048, 256 experts, 8 per token, fixed ids:

| comparison | max abs difference |
|---|---|
| tok_in_z vs stock launch, 4 tokens | **0.0** (bitwise identical, hash 0333e6ea) |
| 4-token run vs 5-token run, first 4 tokens | **0.0** |
| MMQ vs MMVQ, 5 tokens | 0.0306 (mean output magnitude 2.92, so ~1%) |

Adding a column does not perturb the other tokens, and the new layout reproduces the old one
exactly wherever both were legal. The ~1% MMQ/MMVQ gap is two different valid accumulation
orders, and it is what moves draft acceptance in the model runs below.

## What it is worth at the model level

`bench/one.sh`, ctx 32768, `-ub 32 -b 32`, `p-min 0.5`, 5 reps, 128 generated tokens, median
TG. Acceptance and mean_len are the final request's figures from the server log.

**Correction, same day: this table was measured on `bench/prompt.txt`, not the `prompt2.txt`
that AGENTS.md designates and that all historical sweeps used.** `one.sh` defaults to
`prompt.txt` and that default was not overridden. The direction of every result below survives -
control vs fix on the same prompt is a valid A/B - but the absolute numbers are not comparable
to the rest of the repo. On `prompt2.txt` the same binary reads 31.50 baseline and 33.90 at
nmax 5 (see `2026-11-03-draft-depth-on-the-standard-prompt.md`). The "nmax 5, fix" conclusion is
unchanged; the "+7% over baseline" is now measured as +7.6% on prompt2 as well.

| config | tok/s | accept | mean_len | pass |
|---|---|---|---|---|
| nmax 3 (baseline, 4 col) | 28.00 | 0.80556 | 3.35 | 119.6 ms |
| nmax 4, control (5 col) | 29.37 | 0.95146 | 4.38 | 149.1 ms |
| nmax 4, fix | 28.63 | 0.83333 | 3.73 | 130.3 ms |
| nmax 5, control (6 col) | 24.90 | 0.78512 | 4.17 | 167.5 ms |
| **nmax 5, fix (6 col)** | **29.95** | 0.82203 | 4.59 | 153.3 ms |
| nmax 7, fix (8 col) | 28.98 | 0.77692 | 5.59 | 192.9 ms |

Three things follow.

1. **At nmax 3 the fix is inert.** Four columns already used MMVQ for both expert groups
   (gate/up are IQ3_S, cap 4; down is IQ4_XS, cap 5), so the change is a no-op there. Measured
   acceptance and mean_len are identical to the digit between the fix and control arms.
2. **At nmax 4 the pass gets 12% faster and the result does not improve.** 149.1 -> 130.3 ms is
   exactly what the microbenchmark promised for gate/up alone, but mean_len fell 4.38 -> 3.73
   as the MMQ/MMVQ arithmetic shift moved the draft chain, and it nets out to -2.5%. This is
   the AGENTS.md rule about judging weight-path changes by throughput, not by op time.
3. **At nmax 5 the fix is a clear win.** 24.90 -> 29.95 tok/s, +20% over the control and +7%
   over the nmax 3 baseline, with a slightly *better* mean_len (4.59 vs 4.17). Deeper is worse:
   nmax 7 reaches mean_len 5.59 but the pass grows to 192.9 ms.

So the best configuration found is nmax 5 with the fix, 29.95 tok/s against 28.00 for the
previous best in the same harness. That is a real gain and the first config change in this
campaign to beat the baseline, but it is far from 40 tok/s: at mean_len 4.59, 40 tok/s needs a
115 ms pass and the best measured is 153 ms.

Caveat on acceptance: it is one prompt and one figure per config. It is exactly reproducible
for a fixed config (the control repeats 0.95146 on every request) but moves several points
between configs from arithmetic alone, so nmax 4's apparent sensitivity is chain luck, not a
quality regression.

