# Rows-per-block sweep - prediction recorded before running

Written 2026-11-02, before `GGML_CUDA_MMID_RPB` was measured. Same purpose as
[2026-11-02-anchored-prediction.md](2026-11-02-anchored-prediction.md): a prediction on disk
cannot be rationalised afterwards.

## Why this knob

The MoE kernel `mul_mat_vec_q_moe` hardcodes `rows_per_block = 2` with the comment "2 gives best
perf based on tuning". That tuning was done on other hardware. On this board:

- the verify pass at `ncols_dst=4` is the headline path, and gate/up at 4 tokens costs 0.341 ms
  for 13.75 MB, i.e. 42 GB/s against a 110 GB/s ceiling;
- one block reads 2 rows x 2048 elements, about 2 KB, so each of 32 lanes issues only a handful
  of loads before the block ends;
- upstream evidence that this knob is real: Orin PR #31268 wins up to +38% on Jetson by changing
  rows/nwarps per block, and ik_llama PR #2798 lists rows-per-block switching as one of two
  causes of an MMVQ batch cliff.

Rows per block is the number of independent accumulator chains each thread holds, so raising it
is the direct way to put more loads in flight from a latency-bound kernel.

## Prediction

| rows/block | predicted gate/up IQ3_S at 4 tokens |
|---|---|
| 1 | worse, ~0.40-0.45 ms |
| 2 (stock) | 0.341 ms |
| 4 | **best, ~0.24-0.29 ms (15-30% faster)** |
| 8 | similar or slightly worse than 4 - too few blocks (512 total) to fill 8 SMs |

Also predicted: raising rows/block **softens but does not remove the 5-token cliff**, because
the cliff is a step in the token dimension and this knob does not change the token dimension.

Stated plainly so it can fail: I expect a win of **15-30% on the gate/up verify time**, not more.
That alone moves the pass from 118 ms to roughly 105-110 ms, which is ~34 tok/s, not 45. Getting
to 45-60 needs the cliff gone as well.

**Result, 2026-11-02: the prediction was wrong.** RPB 1/2/4/8 measured 0.345/0.352/0.319/0.331
ms at tok=4 (IQ3_S), with the stock figure itself varying 0.333-0.352 across runs, so the best
cell is inside noise. The cliff is unchanged at every RPB. See
[2026-11-02-cliff-is-not-launch-geometry.md](2026-11-02-cliff-is-not-launch-geometry.md).
