# Where the decode time actually goes

**Date:** 2026-11-02
**Config under test:** 31.26 tok/s (draft 3, `p-min 0.50`, `-ub 32 -b 32`, q8_0 KV,
qwen35moe IQ4_XS, mainline `dde879b`)
**Tools added:** `bench/gatherbw.cu`, `bench/opbench.cpp`

Until this file, every statement about the decode bottleneck was inferred from end-to-end
tok/s plus a hand-written flags sweep. `nsys` and `ncu` are not available on this board and
`tegrastats` reports `EMC_FREQ 0%`, so there was no way to see the memory system. This file
replaces inference with two measurements: a bandwidth microbenchmark for the real access
pattern, and a per-op profile that calls the actual ggml CUDA kernels at this model's exact
geometry.

The headline: **the model is not bandwidth-starved and not flag-starved. It loses its time
inside specific matmul kernels, and the loss is concentrated in one quant type.**

> **Status: the experiment has been run, and the prediction in this file was wrong by 10x.**
> `Cyber-Tiel-35B-A3B-gateupIQ4XS.gguf` measured **30.5 tok/s against 31.2 for the control** -
> 2% slower, not 48 tok/s. The cause is a batch-size cliff in the isolated op, explained in
> "The requant, measured". The per-op numbers below are all still reproducible and the type
> ranking still holds; what was wrong was extrapolating one comparison point (batch 5) to a
> model that runs at batch 4.

## The model is a hybrid, not a plain transformer

Parsed from the tensor table, which no previous file had done at this granularity:

| Group | Type | Total | Count |
|---|---|---|---|
| `ffn_down_exps` | IQ4_XS | 5032 MB | 41 |
| `ffn_gate_exps` | IQ3_S | 4400 MB | 41 |
| `ffn_up_exps` | IQ3_S | 4400 MB | 41 |
| `ffn_down_exps` | Q6_K / Q3_K | 630 / 330 MB | 41 |
| `token_embd` | Q8_0 | 515 MB | 1 |
| `attn_qkv` | Q8_0 | 510 MB | **30** |
| `output` | Q6_K | 398 MB | 1 |
| `attn_gate` | Q8_0 | 255 MB | 30 |
| `ssm_out` | Q8_0 | 255 MB | 30 |
| `attn_q` | Q8_0 | 187 MB | **11** |
| `attn_output` | Q8_0 | 93.5 MB | 11 |
| `attn_k`, `attn_v` | Q8_0 | 11.7 MB each | 11 |
| `ssm_conv1d`, `ssm_a`, `ssm_dt`, `ssm_alpha`, `ssm_beta` | F32 | small | 30 |

Two facts correct earlier assumptions:

- **Only 11 of 41 layers are full attention.** The other 30 are linear-attention (SSM) layers
  carrying `attn_qkv`, `attn_gate`, `ssm_out` and the `ssm_*` tensors. Anything to do with
  FlashAttention configuration can at most affect 27% of the layers, and only in prefill.
  The `-ctk/-ctv` KV flags likewise only quantize 11 layers, which is why KV-type changes
  moved nothing on their own.
- **Expert gate and up are IQ3_S, not IQ4_XS.** The model name says IQ4_XS, but that applies
  to `down` only. Two thirds of expert bytes are IQ3_S, which turns out to be the slowest
  quant in the whole model on this GPU.

## The scattered gather is not the problem

`bench/gatherbw.cu` builds real 557 KB expert matrices (IQ4_XS 256-weight blocks, 136 B each)
in a 512-matrix pool, selects them pseudo-randomly, and reads them with two shapes: one block
per matrix, and an mmvq-like row-slice layout. Both are compared against a contiguous read of
the same byte count.

| Experts | Bytes | expert-block | row-slice rpb=8 | contiguous |
|---|---|---|---|---|
| 8 | 4.2 MB | 88.0 GB/s | 99.2 GB/s | 105.1 GB/s |
| 16 | 8.4 MB | 104.9 | 105.0 | 110.2 |
| **29** | **15.4 MB** | **108.0** | **108.1** | **112.0** |
| 64 | 33.5 MB | 109.6 | 110.3 | 113.1 |
| 256 | 134 MB | 109.7 | 109.0 | 112.4 |

At the working-set size that matters (29 experts, one layer's worth of gate matrices),
scattering costs **4 GB/s, about 4%**. There is no scatter penalty to recover. The 109.7 GB/s
sequential ceiling from the earlier audit stands, and the gather can reach it too.

## Per-op profile, real kernels, real shapes

`bench/opbench.cpp` links the shipped `libggml-cuda.so` and calls `ggml_mul_mat` /
`ggml_mul_mat_id` at the model's exact tensor shapes, with scattered (not repeated) expert
ids, timing steady-state graph execution.

### Batch 1 (plain decode)

| Op | Weights | Time | Effective |
|---|---|---|---|
| `ffn_gate_exps` | IQ3_S 2048x512x256 | 0.200 ms | 18.0 GB/s |
| `ffn_up_exps` | IQ3_S 2048x512x256 | 0.164 ms | 21.9 GB/s |
| `ffn_down_exps` | IQ4_XS 512x2048x256 | 0.354 ms | **12.6 GB/s** |
| `ffn_down_exps` | Q6_K 512x2048x256 | 0.208 ms | 33.1 GB/s |
| `ffn_down_exps` | Q3_K 512x2048x256 | 0.257 ms | 14.0 GB/s |
| `ffn_gate_shexp` | Q8_0 2048x512 | 0.075 ms | 14.8 GB/s |
| `ffn_gate_inp` | F32 2048x256 | 0.058 ms | 35.9 GB/s |
| `attn_qkv` (x30) | Q8_0 2048x8192 | 0.301 ms | 59.3 GB/s |
| `attn_q` (x11) | Q8_0 2048x8192 | 0.301 ms | 59.2 GB/s |
| `ssm_out` (x30) | Q8_0 4096x2048 | 0.150 ms | 59.6 GB/s |
| `attn_gate` (x30) | Q8_0 2048x4096 | 0.184 ms | 48.4 GB/s |
| `output` logits | Q6_K 2048x248320 | 5.255 ms | 79.4 GB/s |

### Batch 5 (the speculative verify pass)

| Op | Time | Effective | vs batch 1 |
|---|---|---|---|
| `ffn_gate_exps` IQ3_S | 0.823 ms | 21.9 GB/s | 4.1x time for 5x tokens |
| `ffn_up_exps` IQ3_S | 0.823 ms | 21.9 GB/s | 5.0x |
| `ffn_down_exps` IQ4_XS | 0.490 ms | 45.8 GB/s | 1.4x |
| `ffn_down_exps` Q6_K | 0.948 ms | 36.3 GB/s | 4.6x |
| `ffn_down_exps` Q3_K | 0.934 ms | 19.3 GB/s | 3.6x |
| `attn_qkv` | 0.285 ms | 62.5 GB/s | **0.95x** |
| `ssm_out` | 0.174 ms | 51.3 GB/s | 1.2x |
| `output` logits | 8.042 ms | 51.9 GB/s | 1.5x |

## The budget reconciles end to end

Per verify pass, using the per-layer mix of `down` types (35 layers IQ4_XS, 3 Q6_K, 3 Q3_K):

| Component | Per layer | x41 layers |
|---|---|---|
| MoE gate + up + down | 2.20 ms | **90.3 ms** |
| attn/ssm projections | 0.61 ms | 25.0 ms |
| shared expert | 0.16 ms | 6.6 ms |
| router (`ffn_gate_inp`) | 0.12 ms | 4.8 ms |
| logits head | - | 8.0 ms |
| **total** | | **134.7 ms** |

Measured pass latency is about 118 ms (31.26 tok/s at 3.69 accepted tokens per pass), so the
profile accounts for the whole pass to within the difference between batch 5 and the true
batch of 4.7. At batch 1 the same sum is 61.5 ms against a measured 60 ms plain-decode pass.
**The model of where time goes is now closed.**

Of that 118 ms, **MoE expert matmuls are 90 ms, and 68 ms of that is gate+up alone.** Nothing
else on the list is worth attacking first.

## Why IQ3_S is the floor

Both dispatch paths converge on the same number:

| Path | IQ3_S at batch 5 | IQ4_XS at batch 5 |
|---|---|---|
| MMVQ (default on Volta) | 21.3 GB/s | 45.8 GB/s |
| MMQ (`GGML_CUDA_MMID_MAX_BATCH=0`) | 21.6 GB/s | 24.7 GB/s |

Forcing the tensor-core MMQ path is 2x **worse** at batch 1 across every expert type, and at
batch 5 it is neutral for IQ3_S and 2x worse for IQ4_XS. That closes the "use the Volta MMA
tensor cores for MoE" idea, and it also explains why the jusko fork's
`GGML_CUDA_VOLTA_FORCE_MMQ=moe` knob did not rescue it.

So the limit is neither the matmul kernel nor the memory system. It is in the **dequant work
both kernels share**.

### Two hypotheses tested and rejected

*Shape.* IQ3_S at K=2048 x 512 rows against the transposed K=512 x 2048 rows: 22.3 vs
19.3 GB/s. No shape effect. For every other type the *deep-K* shape is the faster one
(IQ4_XS 71.0 vs 46.7, Q6_K 44.4 vs 36.7, Q8_0 52.4 vs 45.3), so shape cannot explain IQ3_S
being slow.

*The lookup table.* IQ3_S gathers from `iq3s_grid`, a 512-entry
`static const __device__ uint32_t[512]` - 2 KB in **global** memory, addressed by
data-dependent indices that differ across every lane. That looked like the obvious cost, so
it was priced directly: a build-time probe replaced the two table loads with register
arithmetic (`GGML_IQ3S_PROBE_NOGRID`, numerics deliberately wrong). Result: **21.7 GB/s
against 21.4. Nothing.** The gather is not the bottleneck, and staging the table into shared
memory would buy nothing.

### What the type landscape actually shows

Same shape (K=2048, n=512), batch 5, one measurement per type:

| Type | Time | Effective |
|---|---|---|
| IQ4_XS | **0.317 ms** | **70.4 GB/s** |
| Q4_0 | 0.343 | 68.8 |
| Q4_1 | 0.357 | 73.3 |
| IQ4_NL | 0.362 | 65.1 |
| Q5_1 | 0.401 | 78.5 |
| Q5_0 | 0.419 | 68.8 |
| IQ2_S | 0.787 | 17.1 |
| MXFP4 | 0.791 | 28.2 |
| IQ3_S | 0.821 | 22.0 |

The split is not by bits per weight. Every classic 4/5-bit type and both IQ4 variants run at
65-78 GB/s. The **low-bit grid families - IQ2_S, IQ3_S, MXFP4 - collapse to 17-28 GB/s**,
roughly a 3x penalty for the same weight count. IQ3_S carries two thirds of this model's
expert bytes, which is the entire problem.

## What this changes

- **The 12% bandwidth figure from the flag audit is retired.** With exact tensor bytes counted,
  a batch-1 pass reads 2.34 GB (1.48 GB dense weights, 0.40 GB logits head, 0.46 GB of 8-of-256
  experts) and takes 60 ms, so plain decode runs at **39 GB/s**. A verify pass reads 4.18 GB
  (the dense weights and logits once, plus ~2.3 GB of experts for 40 token-expert pairs) and
  takes 118 ms, so it runs at **35 GB/s**. The real figure is **32-35% of peak**, not 12%.
  The headroom is real but it is roughly 3x, not 8x.
- **45-60 tok/s is still reachable, but only through the MoE dequant path.** Halving gate+up
  time alone takes the pass from 118 to ~84 ms, or about 44 tok/s. The configuration space
  cannot do it: the GPU is at 99% load and 28 flag configurations changed nothing.
- **FlashAttention tuning is worth much less than the TODO assumed.** Only 11 of 41 layers
  attend, and decode does not use the MMA path at all, so a 256/256 Volta config buys prefill
  only.
- **The `small_k` exclusion for IQ4_XS only affects batch 1.** `should_use_small_k` gates on
  `ncols_dst == 1`, and the verify pass runs at `ncols_dst` up to 5 where the path is already
  active - which is exactly why IQ4_XS goes from 12.6 to 45.8 GB/s between batch 1 and 5. It
  can help plain decode, not the headline number.

## The fix this points to: change the gate/up expert type

Measured at the gate/up shape (K=2048, n=512), batch 5, per 40 token-expert pairs:

| Type | Time | Effective |
|---|---|---|
| IQ3_S (today) | 0.810 ms | 22.3 GB/s |
| IQ4_XS | **0.314 ms** | **71.0 GB/s** |
| Q6_K | 0.774 | 44.4 |
| Q8_0 | 0.851 | 52.4 |

Substituting IQ4_XS for gate+up was predicted to take 1.62 -> 0.63 ms per layer, about
**41 ms off a 118 ms pass, or ~48 tok/s**.

**That prediction was wrong.** The measured result is 30.5 tok/s against a 31.2 control. The
error is in the batch-5 comparison point, not in the ranking. The table above compares IQ3_S
at 0.81 ms to IQ4_XS at 0.31 ms, but 0.81 ms is a cliff artifact.

## The requant, measured

### Batch sweep at the gate/up shape (K=2048, n=512, 8 of 256 experts, rerouted ids)

| tokens | IQ3_S | IQ4_XS | gain |
|---|---|---|---|
| 1 | 0.190 ms | 0.163 ms | 1.17x |
| 2 | 0.212 | 0.184 | 1.15x |
| 3 | 0.270 | 0.226 | 1.19x |
| **4** | **0.340** | **0.277** | **1.23x** |
| 5 | 0.870 | 0.322 | 2.70x |
| 6 | 0.999 | 0.892 | 1.12x |

The cliff is per type: IQ3_S falls off it between 4 and 5 tokens (0.340 -> 0.870 ms, 25% more
work for 2.6x the time), IQ4_XS between 5 and 6. **The speculative verify pass runs at batch
4**, which is the good regime for both - so the real gate/up gain is 1.23x on that op, worth
roughly 4-5% end to end, not 2.6x and not 50%.

### End-to-end

| Measurement | control (IQ3_S gate/up) | requant (IQ4_XS gate/up) |
|---|---|---|
| `llama-bench` tg64, no speculation | 16.89 +/- 0.13 t/s | **17.73 +/- 0.11 t/s** (+5.0%) |
| speculative, best 2 of 3 reps | 31.43 / 31.36 t/s | 30.29 / 30.73 t/s |
| draft acceptance | 0.98940 | **0.97037** |
| mean accepted length | 3.69 | **3.54** |
| pass latency (mean_len / tok/s) | 118 ms | 116 ms |

Plain decode gains the predicted ~5%. The speculative number loses 2%, because rebuilding the
experts perturbed the numerics and cost 1.9 points of draft acceptance - and this model's
throughput is `mean_len / pass`, so 4% fewer accepted tokens per pass cancels a 2% faster pass.
Part of that acceptance loss is the unintended `attn_output` Q8_0 -> Q6_K (see traps below),
which touches 11 layers, so an anchored rebuild would recover some of it.

### Draft depth: the cliff sets the optimum

The cliff is real in the end-to-end numbers, not just in the isolated op. Control model,
`prompt2.txt`, 3 reps, best two:

| `nmax` | TG tok/s | acceptance | mean_len | pass latency |
|---|---|---|---|---|
| **3** | **31.40** | 0.9894 | 3.69 | 118 ms |
| 4 | 28.69 | 0.9901 | 4.49 | 157 ms |
| 5 | 28.20 | 0.9936 | 5.20 | 184 ms |

Acceptance stays at ~0.99 as depth rises, so the drafts are good - the pass just gets
expensive. Marginal cost per extra accepted token: **48 ms at `nmax 4`**, 39 ms at `nmax 5`.
It has to be under ~32 ms to beat `nmax 3`, which is why depth 3 wins.

Decompose the 38 ms that `nmax 3 -> 4` costs: gate/up +22 ms (the cliff), down +6 ms (IQ4_XS
has the same, milder cliff), logits head +2.7 ms. **About 31 of the 38 ms is batch-size
cliffs.** Remove them and a marginal token costs ~8 ms, at which point `nmax 4` is worth
roughly 4.5 tokens per ~125 ms pass - the 38-40 tok/s range. That is the quantified case for
attacking the cliff, and the first payoff estimate for kernel work that is not a hunch.

Caveat on novelty: the end-to-end part of this table reproduces an item already closed in
`TODO.md` - deeper drafting was swept up to `n-max 12` before and loses 9-25%, with a 37-45 ms
marginal token against a 32 ms average. What is new here is only the decomposition that puts
~31 of the 38 ms on batch-size cliffs, and therefore an estimate of what fixing them buys. Do
not re-run the `nmax` sweep without a kernel change.

The requant model did *not* escape it: at `nmax 4` it scored 30.08 tok/s and its acceptance
collapsed to 0.642 (mean_len 2.20), so `p-min 0.5` cut drafting short and the pass shrank to
73 ms without producing more tokens. `nmax 5` was 27.77. Deeper drafting with IQ4_XS gate/up
is a dead end.

Measurement caveat: one rep in three on the requant model collapses to 19-20 tok/s (PP falls
from ~44 to ~10.7 at the same time). Control never showed it. Comparisons at this size need
5 reps and a median, not 3 reps and a best-of.

### The cliff is not the warp count

`calc_nwarps()` in `mmvq.cu` halves warps at `ncols_dst >= 5` for the GENERIC table, which Volta
uses, so that looked like the cause. It is not: forcing 4 warps for 5-8 tokens left IQ3_S at
0.877 ms against 0.870 stock (noise), and IQ4_XS unchanged at 0.331. Reverted.

The real mechanism is the template: the kernel is
`template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k, bool halve_iters>`, so
each token count is a **separate instantiation with its own register allocation**. Scaling
against the 1-token time shows the cutover plainly - both types amortize sub-linearly through
4 tokens, then one falls back to near-linear cost, at a different column count per type:

| tokens | IQ3_S | ratio to 1 token | IQ4_XS | ratio |
|---|---|---|---|---|
| 1 | 0.190 | 1.0x | 0.163 | 1.0x |
| 4 | 0.346 | 1.8x | 0.279 | 1.7x |
| 5 | 0.877 | **4.6x** | 0.331 | 2.0x |
| 6 | 1.018 | 5.4x | 0.889 | **5.5x** |

Next diagnostic, not yet run: `cuobjdump --dump-resource-usage` on `libggml-cuda.so` for the
`iq3_s` ncols=4 and ncols=5 instantiations. If ncols=5 shows spills or a register count that
drops occupancy, the cause is confirmed and the fix is per-type `nwarps`/launch-bounds tuning
in the same style as the existing `MMVQ_PARAMETERS_RDNA4` table, not a kernel rewrite.

*That diagnostic was run, and it was the wrong question.* Results, `cuobjdump -res-usage` on
`build-volta/bin/libggml-cuda.so`:

| ncols | IQ3_S (type 21) regs | IQ4_XS (type 23) regs |
|---|---|---|
| 1 | 56 | 50 |
| 4 | 121 | 110 |
| 5 | 128 | 127 |
| 6 | 147 | 146 |
| 8 | 166 | 153 |

`STACK:0 LOCAL:0` on all of them: **no spills**, and a smooth register curve through 5. But
those symbols are `mul_mat_vec_q` - the dense `mul_mat` kernel - and `MUL_MAT_ID` does not use
them. The MoE kernel is `mul_mat_vec_q_moe`, templated `<type, rows_per_block, has_fusion>`
**with `ncols_dst` as a runtime argument**, launched as `dim3 block_dims(warp_size, ncols_dst)`
(mmvq.cu:892 and :1100). There is no per-column instantiation on the MoE path at all, so the
template explanation is wrong and the register table above is evidence about a different
kernel.

What survives: the cliff is real and reproducible at `MUL_MAT_ID` (it shows in four separate
runs and, end to end, in the draft-depth sweep), it is type-specific (IQ3_S at 5 columns,
IQ4_XS at 6), and it is not warps, not registers, and not shared memory (`SHARED:0` on the MoE
kernel). Remaining candidates: the block shape `(warp_size, ncols_dst)` changing warp-per-column
occupancy at particular column counts, or an effect of the expert-id distribution rather than
the kernel. Neither has a cheap test left - there is no profiler on this board - so this should
be treated as an open question, not a lead with a known fix.

### What this leaves standing

- The **type ranking** at batch 5 and above is real and reproducible, cliff included.
- "**Fewer bits is slower on Volta**" is true in the batch >= 5 regime. At batch 1-4 the
  penalty is 1.15-1.23x, not 3x. Both statements are needed; the earlier file only had the
  stronger one.
- The batch-5 cliff is now the most interesting object in this file: a 2.6x discontinuity for
  25% more work means a dispatch or occupancy threshold in the MMVQ kernel, and it is worth
  finding because the model sits one token below it.
- **The budget reconciliation above is weaker than it looked.** It closes at batch 5, but the
  real pass runs at batch 4, where gate+up costs 0.68 ms/layer (28 ms/pass), not 68 ms. The
  per-op attribution of the 118 ms pass needs redoing at the real batch before it is trusted
  again.
- The requant file is not worth keeping as the daily driver: +2.2 GB for a 2% loss.

### The model was built, and the recipe has two traps

`Cyber-Tiel-35B-A3B-gateupIQ4XS.gguf`, 20,333,150,528 bytes (20.3 GB vs 18.1 GB source):

| Tensor | Before | After |
|---|---|---|
| `blk.*.ffn_gate_exps` (41) | IQ3_S, 110 MB | IQ4_XS, 136 MB |
| `blk.*.ffn_up_exps` (41) | IQ3_S, 110 MB | IQ4_XS, 136 MB |
| everything else | unchanged | unchanged (except one, below) |

Trap 1: `llama-quantize ... COPY --tensor-type X=iq4_xs` **does nothing**. In COPY mode the
default type is not quantized, so the override block in `src/llama-quant.cpp` (guarded by
`ggml_is_quantized(default_type)`) is skipped and you get a byte-identical duplicate. The
first attempt produced 18,119,934,272 bytes - exactly the source size - and was deleted.

Trap 2: override patterns are `std::regex_search`, not exact matches, and `.` is a wildcard.
My generated map contained `output.weight=q6_k`, which also matches `attn_output.weight`
as a substring, so **all 11 `attn_output` tensors became Q6_K instead of Q8_0** (93.5 MB ->
72.2 MB). Anchor patterns (`^`/`$`) or omit the dotted form. This is a small second variable
in the A/B - `attn_output` is only ~1.9 ms of the 118 ms pass - but it must be disclosed, and
a rerun should isolate it.

Working recipe:

```sh
# one line per tensor: "<full name>=<current type>", gate/up forced to iq4_xs, anchored
llama-quantize --allow-requantize --tensor-type-file /tmp/tensor_types.txt \
    <src.gguf> <dst.gguf> IQ4_XS
```

Quality: requantizing IQ3_S -> IQ4_XS is a round trip through the IQ3_S approximation, so
accuracy is bounded by today's IQ3_S, not better than it. If this wins, repeat from the
**Q4_K_M** source (~22.5 GB, already on disk) for a real quality gain at the same speed.

## Reproducing

```sh
# gather vs contiguous bandwidth at the real working-set size
/usr/local/cuda-11.4/bin/nvcc -O3 -arch=sm_72 -ccbin /usr/bin/g++-10.3.0 \
    -Xcompiler=-isystem,/usr/include/c++/10.3.0 \
    -Xcompiler=-isystem,/usr/include/c++/10.3.0/aarch64-poky-linux \
    bench/gatherbw.cu -o bench/gatherbw
bench/gatherbw 200

# per-op profile of the shipped kernels
bench/build-opbench.sh
bench/opbench 20

# same, with the expert ids frozen between reps (cache-warm, the old behaviour)
OPBENCH_ROTATE=0 bench/opbench 20

# MMQ vs MMVQ for MUL_MAT_ID (requires the GGML_CUDA_MMID_MAX_BATCH override)
GGML_CUDA_MMID_MAX_BATCH=0 bench/opbench 15
```

`opbench` reroutes the expert ids before every timed rep by default. The first version uploaded
ids once, so 20 reps re-read the same ~20 MB of expert matrices and L2 served them. Measured
effect of the fix: IQ3_S 21.8 -> 21.1 GB/s, IQ4_XS 72.5 -> 67.3 GB/s. Small, and it does not
change any ranking - but it removes a real objection to the numbers, so the default is the
honest one.

Tools added by this file, all in `bench/`:

| File | Purpose |
|---|---|
| `gatherbw.cu` | scattered vs contiguous read bandwidth at real expert-matrix size |
| `opbench.cpp` | per-op timing against the shipped `libggml-cuda.so` at the model's exact shapes |
| `build-opbench.sh` | builds opbench against `build-volta` |
| `sweep-gateup.sh` | A/B of the requantized model against the control |
| `../tools/repack_gate_up.py` | merges `ffn_gate_exps` + `ffn_up_exps` into `ffn_gate_up_exps` (built, not yet used - merge was only 8%) |

Note on the source tree: `ggml/src/ggml-cuda/mmvq.cu` is **modified but uncommitted** - it has
two experiment hooks, `GGML_CUDA_MMID_MAX_BATCH` (forces MMQ for `MUL_MAT_ID` when set to 0)
and `GGML_CUDA_SMALLK` (forces the small-k path on/off). Both default to the stock `-1`, so
behaviour is unchanged unless the env var is set. `git checkout` that file to return to
pristine mainline.
