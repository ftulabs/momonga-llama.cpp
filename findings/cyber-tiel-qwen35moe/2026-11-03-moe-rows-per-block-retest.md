# MoE rows-per-block retested post-fix: the kernel wins 4%, the model does not - and the ordering does

2026-11-03. The MoE expert kernel was closed on the argument that its occupancy is at the
architectural ceiling, so the only thing left in it was instruction count. That argument was
incomplete: the same lever that paid 4.3% on the dense path - more output rows per block, so the
activation row is re-read less often - had never been retested after the 64-register launch
bounds. Retested here. **The kernel improves 4%, the model does not move, and the exercise
turned up a harness artefact worth more than the kernel change.**

## Kernel level (ncu, the model's exact geometry)

`OPBENCH_SINGLE=1 OPBENCH_TYPE=iq3_s OPBENCH_NE=256 OPBENCH_NU=8`, K=2048, n=512, 256 experts:

| tok | rows/block | kernel | L1 load sectors | DRAM | registers | occupancy |
|---|---:|---:|---:|---:|---:|---:|
| 6 | 2 | 355.14 us | 10,993,873 | 21.63 MB | 64 | 47.79% |
| 6 | **4** | **340.54 us** | **7,450,637 (-32%)** | 21.97 MB | 64 | 47.97% |
| 6 | 8 | 408.22 us | 5,672,833 (-48%) | 22.82 MB | 64 | 47.24% |
| 7 | 2 | 416.19 us | 12,827,261 | 24.91 MB | 64 | 47.85% |
| 7 | **4** | **401.12 us** | **8,685,238 (-32%)** | 25.40 MB | 64 | 48.09% |

Four rows per block is -4.1% at 6 columns and -3.6% at 7, at identical registers and occupancy,
with DRAM up 1.7% because the output rows are streamed in a different order. Eight rows keeps
cutting sectors (-48%) but turns the time around (+15% against 2), so 4 is the knee. The
occupancy limits ncu reports at 4 rows are blocks 32, registers 32, shared 32, warps 64 - the
same four-at-32 as before, so this changes nothing about the ceiling, it removes traffic under it.

This is the clean version of the trade the sector-aligned block layout failed to make: a 32%
sector cut with no register or occupancy cost.

## Model level (the same binary, env var only)

`prompt2.txt`, 400 generated tokens, 5 reps, `GGML_CUDA_MOE_RPB` selects the arm:

| order | arm | reps (tok/s) | median |
|---|---|---:|---:|
| run 1st | 2 | 35.93 36.66 37.09 36.73 36.66 | **36.66** |
| run 2nd | 4 | 35.67 36.11 36.57 36.35 36.25 | **36.25** |
| run 1st | 4 | 35.92 36.52 36.82 36.39 36.45 | **36.45** |
| run 2nd | 2 | 35.69 36.39 36.89 36.22 36.25 | **36.25** |

Acceptance 0.96657 and mean length 5.68 in all four runs, so nothing numerical moved.

**The arm that ran first won both times**, by 1.1% and by 0.55%. Pooled: 2 rows 36.46 mean,
4 rows 36.35 mean - the change is inside the noise, and the apparent sign is an artefact of run
order.

## The finding that matters: run order is worth ~1%

This is the first properly controlled A/B in the project, and it shows that a single-order
comparison can flip the sign of an effect of about 0.5-1%. A long benchmarking session on this
board drifts: the arm benchmarked second reads 0.5-1.1% lower than the same arm benchmarked
first. Every previous A/B in this repo was single-order. That does not invalidate the large ones
(+3.4%, +2.25%, +0.85% are all larger than the effect), and the +0.85% one was measured with the
loser first, i.e. it is conservative. It does mean anything at or below ~1% needs both orders, or
it is not a result.

The practical rule: **for any effect under 2%, run each arm once in each order and compare the
pooled medians, not one pair.**

## Verdict

**Not adopted.** `GGML_CUDA_MOE_RPB` keeps its default of 2, and the kernel-level -4% is recorded
here as a win that does not reach the model. Something in the pass absorbs it - the honest
reading is that the MoE kernel's share of a real decode is smaller than the 62.96 ms
(41%) the nsys budget assigned it, or that the budget over-attributes to it because a traced
pass is not the same pass. Re-measuring the share on the current build with nsys is the thing to
do before spending more effort inside this kernel, not another launch-parameter sweep.

One more thing this rules out: "the kernel is 4% faster" is not a result on its own, in either
direction. The prediction for this experiment was 37.0-37.2 tok/s and it did not happen, in
either arm.

## Addendum (same day): the open question is answered, and neither guess was right

The verdict above left two candidates: the budget over-attributes time to the MoE, or a traced
pass is not the same pass. Both are now measured - see
[the in-situ finding](2026-11-03-moe-share-in-situ.md). The budget is right (the MoE is 40.1% of
a 159.1 ms pass), and the kernel time in situ does not move either: 63.98 ms/pass at 2 rows
against 63.95 at 4, with the launch grid verified changed and the launch count identical. The
apparent 4% that a first pair of traces showed was session drift on this board, which is worth
about 4% in DRAM-bound kernels between sessions. The mechanism is that the model's MoE cost is
exactly linear in tokens (each token routes to its own 8 experts, so nothing is reused), so the
kernel is pure DRAM streaming at 44-52 GB/s and an L1 sector cut has nothing to work on.

## Artefacts

- `/tmp/pred-moe-rpb4.txt` - the prediction, written before the run
- `bench/results.csv` rows `moe-rpb2`, `moe-rpb4`, `moe5-rpb2`, `moe5-rpb4`, `moe5b-rpb4`,
  `moe5b-rpb2`; `bench/spec.csv` same
- `/tmp/ncu-moe-ab.sh`, `/tmp/ncu-moe-ab1.sh` - the kernel measurements
