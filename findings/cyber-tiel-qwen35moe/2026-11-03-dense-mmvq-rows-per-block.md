# The dense MMVQ path: rows per block, the shared-memory carveout, and the small_k trigger

2026-11-03. Three changes to `mmvq.cu`, all of which only move work assignment and block
geometry, so none of them changes a number the model computes. Together: **35.07 -> 36.25 tok/s
(+3.4%)**, acceptance **0.96657 and mean length 5.68 unchanged**, which is the check that
nothing numerical moved.

This is the first win on the dense `mul_mat_vec_q` path since the batch-cliff dispatch fix. The
earlier conclusion that "the type lever is worth only ~5% end to end and the dense kernels are
L1-bound, so there is nothing left" was right about the mechanism and wrong about the
conclusion: the L1 traffic was reducible, just not by the levers that had been tried.

## Where the time is (measured, not inferred)

A per-pass query of an nsys trace of a real `llama-server` decode, restricted to the kernels
between two consecutive 7-column lm_head verify launches (129 clean passes, median span
173.8 ms, 152.7 ms of kernel time):

| kernel | ms/pass | share | calls/pass | shape |
|---|---:|---:|---:|---|
| `mul_mat_vec_q_moe` (all) | 62.96 | 41.2% | | IQ3_S experts |
| `mul_mat_vec_q` q6_K 1-col | **28.09** | 18.4% | 5.38 | grid 248320, the MTP drafts |
| `mul_mat_vec_q` q8_0 7-col | **18.72** | 12.3% | | grids 4096 / 1024 / 2048 / 256 |
| `mul_mat_vec_q` q6_K 7-col | **6.95** | 4.6% | 0.71 | grid 124160, the verify |
| `gated_delta_net_cuda` | 6.18 | 4.0% | | |
| `volta_sgemm` (f32 router) | 3.21 | 2.1% | | cuBLAS, numerics off limits |
| `concat_non_cont` + `cpy_scalar` | 4.57 | 3.0% | 250 | GDN state movement |
| `quantize_q8_1` | 1.67 | 1.1% | 300 | |
| everything else | ~9 | 6% | | flash_attn, embeddings, norms, topk |

The four dense rows are 53.8 ms of a 152.7 ms pass - 35%. Two of them, the 1-column lm_head and
the 7-column shapes, are what this change attacks.

**A trap avoided while doing this:** the same trace shows 40-95 ms of idle time per pass. That
is nsys CUPTI instrumentation, not a real bubble - the traced run measures 28.9 tok/s against
34.8 untraced. Do not read idle time off a traced run. Kernel durations are still valid; the
gaps between them are not.

## The mechanism: `y` is re-read once per output row

`mul_mat_vec_q`'s inner loop (mmvq.cu, around line 795) is

```cpp
for (int kbx = ...; kbx < blocks_per_row_x; kbx += blocks_per_iter) {
    for (int j = 0; j < ncols_dst; ++j) {
        for (int i = 0; i < rows_per_cuda_block; ++i) {
            tmp[j][i] += vec_dot_q_cuda(vx, &y[j*stride_col_y + kby], ...);
```

`y` does not depend on `i`, so the compiler holds it across the row loop - but it is re-read once
per *block*, and `rows_per_cuda_block` is how many rows a block covers. At 7 columns the
activation side dominates: a column-count decomposition of the dense q8_0 shape at
K=2048/n=8192 gives 3.81M L1 load sectors for the weights (constant in the column count) plus
709k per column for activations, so at 7 columns activations are 4.96M of 8.77M sectors - 57%.

Halving the re-read was already known to be worth trying. It failed when tried in isolation,
because the cross-warp reduction buffer is
`__shared__ float tmp_shared[nwarps-1][ncols_dst][rows_per_cuda_block][warp_size]` floats:
at 7 columns and 4 rows that is 3.58 KB per block, and the driver picks a **16.38 KB**
shared-memory carveout by default, which caps residency at 4 blocks. The earlier attempt
measured +46% and was reverted (`./2026-11-03-mmwq-launch-params.md` section 3). The
conclusion recorded then - "occupancy is not the binding constraint at this shape, L1 throughput
is, so a change that trades L1 traffic for occupancy loses" - was half right: the trade was
forced by the carveout, not by the layout.

## The three changes

1. **`calc_rows_per_block`: 4 rows at `ncols_dst` 5..8 on the Turing table** (was 2). The table
   is shared with every other device, so this is gated on `table_id == MMVQ_PARAMETERS_TURING`
   and leaves the GENERIC/GCN/RDNA paths alone.

2. **Shared-memory carveout raised to 64 KB** (`cudaFuncAttributePreferredSharedMemoryCarveout`,
   50% of the 128 KB L1/shared) for `ncols_dst >= 5`, set once per instantiation in
   `mul_mat_vec_q_switch_fusion`. Without it the 3.58 KB buffer keeps residency at 4 blocks.
   The cost is 64 KB of L1 instead of ~111 KB; measured net positive.

3. **`__launch_bounds__(.., 8)` at `ncols_dst >= 5`.** With 4 rows the kernel needs 4 x 7 = 28
   live accumulators, and nvcc spent 158 registers, cutting residency to 6 blocks - which ate
   almost the whole sector win (9.81 -> 9.71 ms only). Asking for 8 blocks pins it at exactly
   128 registers and restores the 8 blocks the default allocation gave, keeping the sector win:
   9.71 -> 8.62 ms. Same class of defect as the `mul_mat_vec_q_moe` one, same fix.

4. **`should_use_small_k` takes the wider layout when K is an exact multiple of one block
   iteration**, not only when K is shorter than one (`blocks_per_row_x % cover == 0`). On sm_72
   q6_K at K=2048 that is the 1-column lm_head, where `rows_per_block` becomes `nwarps` = 2 and
   the q8_1 loads are shared by two rows. Every thread stays busy, so nothing is idled by it.

All four preserve the per-row k-partition and therefore the summation order: for a given row the
same lanes accumulate the same blocks in the same sequence. That is why acceptance is identical
to the last digit rather than merely close.

## Measured

opbench, 20 reps, medians. The A/B swapped the shared library in `build-volta/bin` so both
variants ran on one machine state; the sector counts and kernel times are `ncu` with the same
binary swap.

| shape | | before | after | delta | L1 sectors |
|---|---|---:|---:|---:|---|
| q6_K n=248320 tok=1 (drafts) | opbench | 5.289 ms | 4.857 ms | **-8.2%** | 127.2M -> 96.3M (-24%) |
| q6_K n=248320 tok=7 (verify) | opbench | 9.783 ms | 8.760 ms | **-10.5%** | 281.3M -> 174.0M (-38%) |
| q8_0 n=8192 tok=7 | opbench | 0.387 ms | 0.350 ms | **-9.6%** | 8.77M -> 5.79M (-34%) |
| q8_0 n=4096 tok=7 | opbench | 0.253 ms | 0.237 ms | -6.3% | |
| q8_0 n=1024 tok=7 | opbench | 0.145 ms | 0.139 ms | -4.1% | |
| q8_0 n=256 tok=7 | opbench | 0.088 ms | 0.091 ms | **+3.4%** | grid is only 128 blocks |

Kernel-only under `ncu` for the verify shape: 9.81 -> 8.62 ms, 128 registers, occupancy limit
8 blocks, achieved 22.4% - the same occupancy as before, with 38% fewer L1 sectors. That is the
whole point: the win is traffic, not parallelism.

The `n=256` regression is expected and accepted. With 4 rows per block the grid halves, and at
n=512 rows there are only 64 blocks for 8 SMs, so the tail is starved. It costs 0.15 ms of a
174 ms pass. `calc_rows_per_block` is a `constexpr` on type, column count and device table and
does not see `nrows_x`, so gating on grid size would need a new template parameter.

## End to end

`prompt2.txt`, 400 generated tokens, one server load, 5 reps, medians:

```
dense-rpb4 rep1: TG 35.57 tok/s
dense-rpb4 rep2: TG 36.31 tok/s
dense-rpb4 rep3: TG 36.55 tok/s
dense-rpb4 rep4: TG 36.25 tok/s
dense-rpb4 rep5: TG 36.13 tok/s
median 36.25   (previous best 35.07)
acceptance 0.96657   mean_len 5.68      <- both unchanged
```

Prediction, written before the run: 35.9 tok/s and unchanged acceptance. The acceptance half was
exactly right; the throughput half undershot slightly (+3.4% against +2.5% predicted).

## What this says about the remaining headroom

- The dense path was **not** at a hardware wall. It was issuing 15.6x more L1 sectors than it
  needed and paying for a default carveout nobody chose. Its four kernels are still 50 ms of the
  pass, so a sector-aligned (padded) block layout - the q8_0 block is 34 bytes and the q6_K
  block 210, so no block ever starts on a sector boundary - remains the next real lever there.
- `rows_per_cuda_block` at ncols 1 is 1 for every type except those that already take `small_k`.
  The 1-column lm_head now takes it; the q8_0 shapes at ncols 1 already did.
- Unchanged and still closed: anything that moves numbers (`--override-tensor`, requants, router
  numerics) costs 44-46% via draft acceptance, and the MoE kernel is hard-capped at 32 blocks/SM
  by all four occupancy limits at once (measured: blocks 32, registers 32, shared 32, warps 64).

## Addendum: four rows per block at 1 column too (same day, +0.85%)

The lever above was applied at ncols 5-8 only, because those were the shapes that had been
profiled. The 1-column lm_head is the second-biggest kernel in the pass at 28.09 ms, so it got
the same treatment in `calc_rows_per_block` case 1:

```cpp
case 1:
    if (small_k) {
        // Same argument as ncols 5-8: y is re-read once per row, so more rows per
        // block is less activation traffic.
        return table_id == MMVQ_PARAMETERS_TURING ? 2*nwarps : nwarps;
    }
```

It only takes effect with `small_k`, which the 1-column q6_K lm_head now selects after the
wider trigger above; the q8_0 shapes at ncols 1 already used 1 row, so this moves them too.

Direct A/B by swapping only `libggml-cuda.so.0`, 20 reps each:

| shape | 2 rows (`36.25` build) | 4 rows | delta |
|---|---:|---:|---:|
| q6_k n=248320 tok=1 (lm_head) | 4.776 ms | **4.552 ms** | -4.7% |
| q8_0 n=8192 | 0.271 | 0.272 | flat |
| q8_0 n=2048 | 0.155 | 0.152 | -2.0% |
| q8_0 n=1024 | 0.123 | 0.127 | **+3.3%** |
| q4_0 n=8192 | 0.195 | 0.189 | -3.1% |

End to end, same harness, 5 reps:

```
dense-rpb4c1 rep1: TG 35.67 tok/s
dense-rpb4c1 rep2: TG 36.61 tok/s
dense-rpb4c1 rep3: TG 36.69 tok/s
dense-rpb4c1 rep4: TG 36.56 tok/s
dense-rpb4c1 rep5: TG 36.42 tok/s
median 36.56   (previous best 36.25)
acceptance 0.96657   mean_len 5.68      <- both unchanged
```

Prediction, written before the run: **36.5-36.6 tok/s** from -4.7% on a 28.09 ms kernel on a
156.7 ms pass, with acceptance and mean_len identical. Both halves landed: the measured pass
time is 155.4 ms against 156.7 (-1.33 ms, predicted -1.32). This is the first prediction in
this project to come out inside its stated band, and it did so because the microbenchmark A/B
and the model-level share were both already measured rather than assumed.

The +3.3% on q8_0 n=1024 is real and is accepted: that kernel is about 1 ms/pass, so it costs
0.03 ms against the 1.33 ms won.

**Why this one worked where the earlier `rows_per_block = 4` attempt failed.** The first attempt
measured +46% in `opbench` and -0.7% end to end because it doubled the row count without raising
the shared-memory carveout, so residency fell and the win inverted. The carveout and the
`__launch_bounds__` are part of the change, not tuning on top of it. Any future rpb increase
needs both in the same commit.
