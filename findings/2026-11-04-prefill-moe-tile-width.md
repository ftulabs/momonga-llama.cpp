# H3: the prefill MoE tile is 32 columns wide and 31 of them are thrown away

Date: 2026-11-04. Status: in progress. H2 established that `mul_mat_q` is **88.1% of prefill** at the
production `-ub 32` (`findings/2026-11-04-prefill-budget.md`), split IQ3_S 52.9% / IQ4_XS 23.0% /
Q8_0 12.4%. This file is the attack on that 88%.

## 1. Prediction, written before the first run

**The claim.** At `-ub 32` a `MUL_MAT_ID` prefill launch has `ncols_max = ne12 = 32` tokens, so
`ggml_cuda_mmq_get_J_best()` picks **J = 32**. But the 32 tokens route over 256 experts
(`n_expert_used = 8`), so an expert receives `32*8/256 = 1` token on average. The kernel computes a
full `J`-wide tile for every expert anyway: the per-tile column bound `tile_y_max_j` guards only the
**write-back**, never the `vec_dot` inner loop (`for (j0 = 0; j0 < J; j0 += nwarps)`,
`mmq-vec-dot.cuh:459` for the IQ3_S family). So 31 of every 32 columns of dp4a work are computed and
then discarded.

**The arithmetic that says this is the bottleneck, not bandwidth.** One IQ3_S gate/up launch
(measured 2,325 us, grid 1024, block 32x8, `n=1440` launches, 3,348 ms total):

- experts actually reached = 256 * (1 - (1 - 1/256)^256) = 162 of 256 (the rest return at
  `if (jt*J >= col_diff)`), each with 512 rows x 2048 K x 32 columns of dot product
  = 5.4e9 MACs.
- 1440 launches in 3.348 s = 2.33e12 MAC/s.
- the dp4a ceiling is 8 SM x 64 lanes x 1.38 GHz x 4 MACs = **2.83e12 MAC/s**, so the kernel is at
  **83% of the ALU peak**.
- the weight side of the same launch is 115 MB, i.e. 1.04 ms at the measured 111 GB/s ceiling -
  *less* than the 2.33 ms it takes. It is not bandwidth-bound.

**The change.** Pick the MoE tile width from the average tokens per expert instead of from the
token count. The code already does exactly this for RDNA3/RDNA4 (`mmq.cu:243-247`):

```cpp
// Each expert only sees ne12*n_expert_used/ne02 tokens on average.
if (GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4_M(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) {
    ncols_opt = (ne12*n_expert_used + ne02 - 1) / ne02;
}
```

`(32*8 + 255)/256 = 1` -> `ntiles_x = 1` at J = 8 -> **J = 8**, the smallest specialization the
switch has. dp4a work falls 4x, its floor to 0.38 ms/launch, which puts the 1.04 ms memory floor in
charge.

**Bit-preservation, argued from the source.** In the dp4a path each output element `(i, j)` gets one
accumulator slot, `sum[(j0/nwarps)*(I/warp_size) + i0/warp_size]`, where `j0` is the multiple of
`nwarps` that contains `j`. For a *fixed* `j` that slot is fixed regardless of `J`, and the
accumulation order over K is the `kb0`/`k01` loop, which `J` does not touch. `J` changes **which
thread** owns an output, not the order of its sum. Therefore J = 32 -> J = 8 must be bit-identical,
and decode cannot move: decode does not use MMQ at all. Expected signature is therefore the control
exactly - acceptance **0.96657**, mean len **5.68**, decode 36.9-37.0 tok/s.

**Numbers I predict, before measuring:**

| quantity | now | predicted after |
|---|---|---|
| IQ3_S gate/up per launch | 2,325 us | 1.1-1.3 ms |
| MoE MMQ total (IQ3_S + IQ4_XS) | 4,803 ms | 2.4-2.9 s |
| prefill kernel total, 590 tokens | 6,570 ms | 4.2-4.7 s |
| prefill tok/s at ~590 tokens | 78.8 | **105-120** |
| TTFT at 9,563 tokens | 134.6 s | 90-105 s |
| acceptance / mean len / decode | 0.96657 / 5.68 / 36.9 | unchanged |

**Falsification.** If prefill moves by less than 5%, the ALU analysis above is wrong and the MoE
MMQ is limited by something else (occupancy: shared memory is 46.5 KB/block against a 64 KB
carveout, so only one 8-warp block is resident per SM; that is arm 2, below). If acceptance is not
exactly 0.96657, the bit-preservation argument is wrong and the change is dead regardless of speed.

**Arm 2, if arm 1 is not enough.** Raise the shared-memory carveout to 100% so two 46.5 KB blocks
fit per SM (Xavier allows 96 KB/SM, 64 KB/block; IQ3_S J=32 uses 118 registers so 2 x 256 threads
needs 60,416 of 65,536 registers - it already fits). Purely a residency change, bit-preserving.

**Arm 3, the wide-ubatch question from H2.** `should_use_mmq()` returns `n_tokens < 64` on any card
with fp16 MMA, so at `-ub 512` the MoE leaves MMQ for a per-expert loop. If that engine switch is
the *only* reason acceptance collapses at 512 (0.96657 -> 0.65882), forcing MMQ at 512 would keep
the 2.4x prefill *and* restore acceptance, which is the campaign's whole prize. H2's trace says the
dense matmuls also switch engine at 64 tokens (cuBLAS f32/INT8 is 14.2% of ub-512 prefill), so the
prediction is that forcing MMQ on the MoE alone will **not** restore acceptance; forcing it
everywhere might, unless stream-k's K split (present at the narrow shapes, absent at the wide ones)
moves bits. Both are measured, not argued.

Both arms go behind env hooks so one rebuild serves the whole sweep.

## 2. Results: arm 1 wins, and the prediction was wrong about why

All arms: `-ub 32 -b 32`, `-ctk/ctv q8_0`, `cache_prompt: false`, one server load per arm, decode
measured first in that same load (`DECODE_REPS=5`, `NPRED=400`, prompt `bench/prompt2.txt`).
Prefill numbers are the median of reps 2-3 (rep 1 is the first-call cold rep); within-arm spread
across reps is 0.1-0.3%, which is the reproducibility `bench/prefill.sh` already claims.

| arm | ms/token at 590 | 2,385 | 9,563 | tok/s at 590 | 2,385 | 9,563 | TTFT @9.5k |
|---|---|---|---|---|---|---|---|
| `h3-ctl-np400` (control) | 12.665 | 13.265 | 14.076 | 79.4 | 75.5 | 71.0 | 134,718 ms |
| `h3-tileavg-np400` (J from tokens/expert) | 11.888 | 11.943 | 12.561 | 84.1 | 83.7 | 79.6 | 120,161 ms |
| delta | **-6.1%** | **-10.0%** | **-10.8%** | **+5.9%** | **+10.9%** | **+12.1%** | **-10.8%** |

Decode in the same loads (median of 5): control 36.50, arm 1 36.41 - a 0.25% difference, well
inside the 0.71% A/A floor of rule 7, and MMQ is not on the decode path at all (`should_use_mmq`
is false at 1-7 columns). Acceptance **0.96657** and mean len **5.68** in both arms, with the
per-rep acceptance lines identical line for line, including the accepted/generated counts
(`0.96657 ... 318/329`). The bit-preservation argument in section 1 is therefore confirmed
empirically, not just by reading the code.

The change is one condition in `mmq.cu`: the "average tokens per expert" heuristic that already
exists for RDNA3/RDNA4 (`ncols_opt = ceil(ne12*n_expert_used/ne02)`) is now reachable on NVIDIA
behind `GGML_CUDA_MMQ_MOE_TILE_AVG=1`. At 32 tokens and 8 routed experts over 256 experts that
reads 1, so `J_best` becomes 8 instead of 32.

### What the size of the win says

Predicted +33-58% prefill, delivered +12%. The prediction assumed the discarded dp4a work was
83% of the MoE's cost, so cutting it 4x had to dominate. It did not, and the arithmetic in
section 1 was wrong because it treated the kernel as ALU-bound on a dp4a throughput that the
kernel never reaches. A 4x cut in dot work producing a 12% total gain means the dot phase was a
small share of the pass and the rest was waiting - which is what the geometry says: one 256-thread
block resident per SM (46.5 KB at J=32, 42.9 KB at J=8, both against the 64 KB carveout), a k-loop
whose body is `load tile_x -> __syncthreads -> vec_dot -> __syncthreads`, and nothing else
resident to overlap the LPDDR4x latency with. That re-diagnosis is section 2a, written before arm
2, and it reframes the next question as occupancy rather than arithmetic.

Wall-clock effect: TTFT on a 9,563-token prompt goes **134.7 s -> 120.2 s**. That is the largest
single prefill improvement measured in this project so far, and it costs nothing on decode.

## 2a. Arm 2 prediction, written before that run

Arm 1's first look at ~9.5k tokens reads **80.7 tok/s against the control's 71.0 (+13.7%)**, not the
+33-58% the ALU arithmetic above predicted. That is a real gain and it is bit-identical (the
acceptance log lines match the control line for line, including the per-rep `318/329` counts), but
it falsifies the *diagnosis*: if the MoE kernel had been 83% ALU-saturated, cutting its dp4a work
4x would have moved it far more than this. A 13-17% gain instead says the dot phase was a minor
share and the kernel is **latency-bound on the `tile_x` load**, which fits the geometry: one
256-thread block is resident per SM (shared memory 46.5 KB at J=32 / 42.9 KB at J=8 against a 64 KB
carveout), the k-loop is `load -> __syncthreads -> vec_dot -> __syncthreads`, and with a single
resident block there is nothing to overlap the roughly 600 ns LPDDR4x latency with. 648 working
blocks over 8 SMs = 81 sequential waves, each paying 8 load phases.

**Prediction for arm 2 (`GGML_CUDA_MMQ_CARVEOUT=100`, stacked on arm 1):** two blocks per SM
(2 x 42.9 KB = 86 KB of the 96 KB Xavier allows) lets one block's load phase overlap the other's
dot phase, so the MoE MMQ should fall another **20-45%**, prefill should gain **+15-30%** on top of
arm 1 (i.e. roughly 95-110 tok/s at 590 tokens against the control's 79.4), and acceptance/mean
len/decode must stay bit-identical again. Falsified if prefill moves less than ~5%, which would
mean the driver was already granting a larger carveout than 64 KB, or that the carveout attribute
is ignored on Tegra.

## 2b. Arm 2 result: the prediction is falsified at the length that matters

`h3-tileavg-cv100-np400` = arm 1 + `GGML_CUDA_MMQ_CARVEOUT=100`.

| arm | ms/token at 590 | 2,385 | 9,563 | tok/s 590 | 2,385 | 9,563 | TTFT @9.5k |
|---|---|---|---|---|---|---|---|
| arm 1 (J=8 everywhere) | 11.888 | 11.943 | 12.561 | 84.1 | 83.7 | 79.6 | 120,161 ms |
| arm 2 (J=8 + carveout 100) | 11.107 | 11.751 | 12.513 | 89.8 | 85.0 | 79.9 | 119,732 ms |
| delta vs arm 1 | -6.6% | -1.6% | **-0.4%** | +6.9% | +1.6% | +0.4% | -0.4% |

Acceptance 0.96657 / mean len 5.68 again, acceptance lines identical to both earlier arms. Decode
median 36.74 against arm 1's 36.41 - the later position reads *faster* here, which is a warning
that this session's drift is not one-directional.

**The prediction said +15-30% at 590 tokens and 95-110 tok/s; the run delivered 89.8 tok/s
(+6.9% over arm 1) and only +0.4% at 9,563 tokens.** At the length that dominates TTFT the
carveout does nothing, so it is not adopted: 120.2 s -> 119.7 s is inside this board's drift.

Two candidate explanations, and the 590 number cannot separate them: (a) the carveout attribute
is ignored or already effectively granted on Tegra, so residency stayed at one block per SM; or
(b) residency did increase and the kernel is not latency-bound after all (section 2c). The
length dependence is itself evidence for drift rather than a real effect - the MoE runs one
launch per layer *per 32-token chunk*, so its cost per chunk is identical at every prompt
length, and a real MoE gain would show up equally at 590 and at 9,563. It does not. The 590-token
gain is therefore treated as unresolved and not adopted.

## 2c. The traced budget at J=8, and what it re-diagnoses

`bench/prof-prefill.sh` now takes `EXTRA_ENV` so a hook can survive `sudo -n env`'s reset
(`EXTRA_ENV="GGML_CUDA_MMQ_MOE_TILE_AVG=1" LEN=512 UB=32 ./prof-prefill.sh`). One traced
590-token prefill at J=8, 6,700.7 ms of kernel time:

| kernel | gridX | calls | total ms | share |
|---|---|---|---|---|
| `mul_mat_q` IQ3_S MoE gate/up | 4096 | 1440 | 2926.30 | 43.7% |
| `mul_mat_q` IQ4_XS MoE down | 16384 | 666 | 1610.56 | 24.0% |
| `mul_mat_q` dense (512 cols) | 64 | 738 | 341.94 | 5.1% |
| `gated_delta_net` | 32 | 630 | 221.42 | 3.3% |
| `mul_mat_q` dense (16) | 16 | 1494 | 207.31 | 3.1% |
| `mul_mat_q` (16384, other) | 16384 | 54 | 130.44 | 1.9% |
| `mul_mat_q` (32) | 32 | 540 | 130.09 | 1.9% |
| `mul_mat_q` (2048) | 2048 | 80 | 119.29 | 1.8% |
| `mul_mat_q` (8) | 8 | 1872 | 91.90 | 1.4% |
| `ssm_conv_f32` | 1 | 630 | 78.71 | 1.2% |
| `flash_attn_tile` | 8 | 187 | 48.38 | 0.7% |

`mul_mat_q` as a family is still 85.2% of prefill; the two MoE launches are 67.7% of it. The
geometry confirms the hook is live: the IQ3_S launch is `gridX 4096 = ceil(32/8) column tiles x
4 row tiles x 256 experts`, where the control had 1024 (one column tile).

**These totals must not be differenced against the H2 trace** - the same-binary cross-session
drift AGENTS.md warns about is 3-4% and the two traces were taken an hour apart. What can be read
from a single trace is the ratio between kernels inside it:

- The IQ3_S gate/up launch moves 256 experts x 2 tensors x 512 x 2048 at ~3.4 bpw = ~231 MB in
  2.03 ms = **~114 GB/s**, at or above the 109.7-112 GB/s measured ceiling for this board. [WRONG:
  corrected in section 2g - the launch is one projection, 115 MB, i.e. 57 GB/s. The conclusion two
  sentences later, that the dot was not the binding limit, survives; the "at the ceiling" part does
  not.] That
  is why cutting its dp4a work 4x bought only 12.6%: at J=32 the same launch reads ~231 MB in
  2.33 ms, so it was already at ~99 GB/s. The dot work was being hidden behind a saturated DRAM
  stream, not adding to it. **The gate/up MoE is finished in the launch-parameter space.**
- The IQ4_XS down launch moves 256 x 2048 x 512 at ~4.25 bpw = ~142 MB in 2.42 ms = **~59 GB/s**,
  roughly half of what the same type reaches in the decode kernel (92.7 GB/s). It is the only
  large kernel in this budget that is *not* near a hardware limit.
- The narrow tile is free in traffic terms, and the reason is a source detail worth recording: the
  MoE branch of `mul_mat_q` computes an expert's column range from `expert_bounds`
  (`mmq.cuh:1073-1110`) and **returns before loading anything** when `jt*J >= col_diff`. With
  ~1 routed column per expert, the extra column tiles that J=8 creates exit immediately, so the
  weight bytes moved are unchanged and only the dot shrinks. That is consistent with the 12.6%
  gain being the dot's true share of the launch.

## 2d. Arm 3 prediction, written before that run

`GGML_CUDA_MMQ_MOE_TILE_AVG=2` applies the narrow tile only to `GGML_TYPE_IQ3_S` and leaves the
IQ4_XS down projection on the wide tile (the dense launches are unaffected either way - the hook
is inside the `ids != nullptr` branch, so it only ever touches `mul_mat_id`).

**Prediction:** the down projection recovers most of the 59 -> 92.7 GB/s gap, taking its 24.0%
share to roughly 15-17%, so prefill gains **+4-7% on top of arm 1** (TTFT at 9,563 tokens from
120.2 s to ~113-115 s), with acceptance 0.96657 and mean len 5.68 bit-identical again, since J
still only changes which thread owns an output element. Falsified if prefill moves less than ~2%,
which would mean the down projection's 59 GB/s is set by its dot and not by the column-tile
re-read.

Protocol: because this is a single-digit-percent claim on a board with a 0.71% A/A floor and
+0.95% per-position drift, the two modes are run interleaved in both orders in one session -
`tile_avg=2`, `tile_avg=1`, `tile_avg=1`, `tile_avg=2` - and the pooled medians compared. The two
`tile_avg=1` runs double as this session's A/A floor.

### Addendum, 17:14, before any result of this arm was read

The prediction above rests on "J=8 re-reads each expert's weight tile 4x, which is why IQ4_XS fell
to 59 GB/s". A source read an hour after writing it kills that: the MoE branch returns before
loading when `jt*J >= col_diff` (`mmq.cuh:1085-1088`), so the extra column tiles cost nothing and
no type pays a re-read. The 59 GB/s figure is therefore **unexplained**, and the honest expectation
reverses: the dot is 4x smaller at J=8 for *every* type, so if IQ4_XS's dot is a share of its
launch at all, J=8 should help it too and **mode 2 should be slower than mode 1**, not +4-7%
faster. Both predictions stay on the record; the arm is running either way, and it also gives this
session's A/A floor from its two `tile_avg=1` runs.

## 2e. Arm 3 result: mode 2 loses where it matters, and the addendum was right

Four runs, interleaved in both orders in one session, one server load each, all at
`-ub 32 -b 32` with `cache_prompt: false`, `DECODE_REPS=5`, `REPS=3`, prompt tokens 590 / 2,385 /
9,563 as always. `mode 1` = narrow tile for every MoE type; `mode 2` = narrow tile for IQ3_S only,
IQ4_XS down keeps `J=32`.

| position | arm | ms/tok 590 | 2,385 | 9,563 | TTFT @9.5k |
|---|---|---|---|---|---|
| 1 | mode 2 | 11.14 | 11.93 | 12.77 | 122,136 ms |
| 2 | mode 1 | 11.36 | 11.80 | 12.55 | 120,019 ms |
| 3 | mode 1 | 11.64 | 11.95 | 12.62 | 120,689 ms |
| 4 | mode 2 | 11.02 | 11.93 | 12.77 | 122,146 ms |

Both orders agree, and they agree tightly - the two mode-2 runs are within **0.01%** of each other
at 9.5k (122,136 / 122,146) and the two mode-1 runs within 0.56% (120,019 / 120,689). This is the
**A/A floor of the prefill protocol in this session**: 0.56% at 9.5k, 1.6% at 2,385, 2.1% at 590.

- At **9,563 tokens, mode 2 is 1.49% slower** (122,141 vs 120,354 median). That is above the
  session's own 0.56% floor at that length and it reproduces in both positions, so the wide tile
  genuinely costs more on the IQ4_XS down projection. The section-2d prediction (+4-7% for mode 2)
  is falsified, exactly as the 17:14 addendum argued it would be - the narrow tile is free in
  traffic terms and the down projection benefits from it like everything else.
- At 2,385 tokens the two modes are indistinguishable (11.93 vs 11.88/11.95, inside the floor).
- At 590 tokens mode 2 reads 4.5% faster (11.02-11.14 vs 11.36-11.64), which is about twice the
  2.1% floor at that length. I do not have a mechanism for it and it is worth 0.2-0.35 s of
  TTFT, so it is recorded as an unexplained short-prompt effect and not chased.
- **Decode, acceptance and mean length are unaffected by all four arms**: acceptance **0.96657**
  and mean len **5.68** in every one of them, and the acceptance log lines are identical to the
  control's. Decode medians 36.30-36.45 against the 36.63 kernel-only record (these runs do not
  pass `-bs`) - i.e. no decode change, which is expected since MMQ is not on the decode path.
  That is now four independent binaries/configurations from this file confirming bit-preservation
  empirically: J changes which thread owns an output element, not the order of its K sum.

## 2f. Adopted default, verified with no environment variables

The hook's default was flipped to mode 1 in `mmq.cu` and rebuilt, then measured once more with no
`GGML_*` variables set at all (`h3-adopted-np400`), which is the run the shipped binary gets:

| prompt | TTFT | tok/s | tok/s before H3 |
|---|---|---|---|
| 590 tokens | 6,619 / 6,634 ms | 89.0 | 78.8 |
| 2,385 tokens | 28,154 / 28,265 ms | 84.5 | 75.8 |
| 9,563 tokens | 120,247 / 120,436 ms | 79.5 | 71.0 |
| decode | median 36.57 | acceptance 0.96657, mean len 5.68 | 36.63 kernel-only record |

So the default path is the measured path, and decode is where it always was.

**Adopted: mode 1**, pooled over every mode-1 run in both sessions and in both orders (four
measurements at each length):

| prompt | TTFT now (pooled median) | tok/s now | tok/s before | change |
|---|---|---|---|---|
| 590 tokens | 6,776 ms | 87.1 | 78.8 | +10.5% |
| 2,385 tokens | 28,307 ms | 84.3 | 75.8 | +11.2% |
| 9,563 tokens | 120,231 ms | 79.5 | 71.0 | +12.0% |

TTFT on a 9,563-token prompt: **134.6 s -> 120.2 s (-10.7%)**. The four mode-1 9.5k measurements
across two sessions span 120,019-120,689 ms, i.e. 0.56% - the same as the A/A floor quoted above -
while the pre-H3 control read 134,608 ms in the same protocol. The change is 10x the floor.

### The narrow tile also changes where the remaining time sits, in one line

The re-traced budget (section 2c) puts the two MoE launches at 67.7% of prefill, and the IQ3_S
gate/up launch at **57 GB/s of DRAM read, half of the 109.7-112 GB/s ceiling on this board (see
section 2g, which corrects an earlier 114 GB/s figure)**.
That is why the 4x cut in dp4a work bought 12.6% and not 4x: neither the dot nor the bandwidth was
saturated, so removing dot work removed only the part of the launch that was not already hidden
behind load latency. The IQ4_XS down launch reads at ~59 GB/s too, and the arm that gave it back
its wide tile made things 1.5% worse, so its limit is not its launch geometry either. Nothing
further is queued on this front: what is left is the same wall decode hit, that at a bit-preserving
32-token ubatch each expert's weights are streamed for about one token of work, and the launch runs
at half of peak because one useful column leaves nothing to overlap the next tile's load with.

## 2g. Correction to 2c: the MoE is at 57 GB/s, not at the DRAM ceiling

Section 2c said the IQ3_S gate/up launch "moves ~231 MB in 2.03 ms = ~114 GB/s, at or above the
109.7-112 GB/s ceiling". That is wrong, and the trace says so once the demangled kernel name is
parsed for its template arguments (`(ggml_type)N, (int)J, (bool)fallback, (ggml_prec)N,
(bool)has_fusion`) instead of being truncated:

| type | J | grid | calls | calls per 32-token chunk | per launch | total | share of MMQ |
|---|---|---|---|---|---|---|---|
| IQ3_S | 8 | 4096 | 1440 | **78.1** | 2.032 ms | 2926 ms | 50.9% |
| IQ4_XS | 8 | 16384 | 666 | **36.1** | 2.418 ms | 1611 ms | 28.0% |
| Q8_0 | 32 | 64 | 738 | 40.0 | 0.463 ms | 342 ms | 5.9% |
| Q8_0 | 32 | 16 | 1494 | 81.0 | 0.139 ms | 207 ms | 3.6% |

**`has_fusion` is 0 in every launch in the trace.** The MoE is not fused: 78 launches per chunk is
41 layers x 2, one for `gate` and one for `up`, each moving one projection of 512 x 2048 weights
over 256 experts at IQ3_S's 0.4297 B/weight = **115.4 MB**, and 36 per chunk is 41 layers x 1 for
the down projection (142.6 MB at IQ4_XS's 0.531 B/weight). So:

- IQ3_S gate or up: 115.4 MB / 2.032 ms = **56.8 GB/s**
- IQ4_XS down: 142.6 MB / 2.418 ms = **59.0 GB/s**
- the MoE per 32-token chunk: 259 ms for 15.3 GB = **59 GB/s**
- the whole prefill: 364 ms per chunk for the ~18 GB model = **~49 GB/s**

Against this board's measured 109.7-112 GB/s ceiling, **prefill runs at 45-54% of DRAM bandwidth,
not at it.** The consequence for the section-1/2a diagnosis is that neither "ALU-bound" nor
"bandwidth-bound" is right: the MoE launch is at half of peak with one useful column per expert,
which is a latency/issue-bound regime, and the 4x cut in dp4a work bought 12.6% because the dot
was neither the limit nor free.

What this does *not* change: the adopted change, its numbers, and the rejection of the carveout
and the per-type tile. What it changes is the size of the remaining headroom and who should attack
it. Even a perfect kernel that read at the full 112 GB/s would spend 139 ms per 32-token chunk in
the MoE instead of 259, which is **~195 tok/s and a 49 s TTFT floor at 9,563 tokens** - better
than today's 79.5 / 120.2 s by 2.4x, but still nowhere near interactive, because the configuration
re-reads the model's 18 GB once per 32 tokens.
