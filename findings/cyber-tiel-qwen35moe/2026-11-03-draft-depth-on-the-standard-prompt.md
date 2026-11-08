# Draft depth on the standard prompt: prediction first

Date: 2026-11-03. Hardware: AGX Xavier, sm_72, CUDA 11.4. Model IQ4_XS, MTP draft.
Prompt `bench/prompt2.txt`, `NPRED=400`, `REPS=5`, `UB=32 B=32 --spec-draft-p-min 0.5`.

## What is already measured (this file exists so the prediction below cannot be rationalised)

The dispatch-table fix (`mmvq.cu`, sm_72 arch range + tokens on `blockIdx.z`) is in the tree.
On `prompt2.txt`, the prompt every historical sweep used:

| cfg | reps (tok/s) | median | acceptance | mean_len |
|---|---|---|---|---|
| nmax 3 | 31.22 31.74 31.50 31.81 31.07 | 31.50 | 0.98940 | 3.69 |
| nmax 5 | 33.51 34.01 33.94 33.86 (4 reps) | 33.90 | 0.97476 | 5.07 |

nmax3 reproduces the documented 31.2-31.5 baseline exactly, which confirms `prompt2.txt` is the
right prompt and that the earlier 23 tok/s readings were `prompt.txt`, not a regression.

**mean_len is 5.07 at `nmax 5`** - the chain is running to its cap, and acceptance is 0.97. That is
the opposite of the situation the old "nmax 3 is optimal, deeper loses 9-25%" result was measured
in (acceptance 0.72-0.78). Deeper drafting is worth re-testing in *this* regime.

Derived pass latency: nmax3 = 3.69/31.50 = 117.1 ms; nmax5 = 5.07/33.90 = 149.6 ms.
Two extra draft slots cost 32.5 ms, i.e. ~16.3 ms per slot, linear so far.

## Prediction (written before the run)

If per-slot cost stays ~16 ms and acceptance holds near 0.97:

- **nmax 6**: mean_len ~5.9, pass ~166 ms, **~35.5 tok/s**
- **nmax 7**: mean_len ~6.6, pass ~182 ms, **~36.3 tok/s**
- **nmax 9**: mean_len ~8.0, pass ~215 ms, **~37.2 tok/s**, but this crosses
  `MMVQ_MAX_BATCH_SIZE` (8) and falls back to MMQ for the verify pass, so I expect it to come in
  *below* nmax 7 if the batch-10 MMQ penalty is real.

**Falsifier**: if nmax 7 comes in at or below nmax 5, the per-slot cost is superlinear (batch
growth in the verify pass outweighs the longer chain) and the old "nmax 3" conclusion was
approximately right for a different reason than stated.

Consequence to check if deeper wins: the `nmax 3` setting in `KNOWN-GOOD.md` and
`bench/run-best.sh` is then stale and has to change.

## Result: nmax 6 wins, the prediction was too optimistic

| cfg | reps (tok/s) | median | acceptance | mean_len | pass |
|---|---|---|---|---|---|
| nmax 5 | 33.51 34.01 33.94 33.86 | 33.90 | 0.97476 | 5.07 | 149.6 ms |
| **nmax 6** | 34.10 34.36 34.64 34.49 34.49 | **34.49** | 0.96657 | 5.68 | 164.7 ms |
| nmax 7 | 34.03 34.03 28.94 34.85 34.11 | 34.11 | 0.95562 | 6.21 | 182.1 ms |

Predicted 35.5 for nmax 6 and 36.3 for nmax 7; measured 34.49 and 34.11. The error is in the
per-slot cost, assumed flat at 16.3 ms: from nmax 5 to 6 it is 15.1 ms, from 6 to 7 it is 17.4 ms.
mean_len grows 5.07 -> 5.68 -> 6.21 per two slots, i.e. the chain does not extend as fast as
acceptance alone implies, so throughput flattens at ~34.5 and then turns down.

nmax 7's 28.94 rep is the known intermittent collapse (one rep in three on the larger files per
AGENTS.md). Median is used, so it does not change the ranking.

**nmax 6 at 34.49 tok/s is the best configuration measured on the standard prompt**, against
31.50 for the nmax 3 baseline, +9.5%. The old "nmax 3 is optimal, deeper loses 9-25%" is
correct for the low-acceptance `prompt.txt` regime (0.72-0.78) and wrong for this one (0.97).

## Where the remaining time goes

`GGML_CUDA_OP_TIME=1` (per-op CUDA events, graphs off, host overhead inflates the absolute ms
so only the ranking is trustworthy) puts the three expert matmuls at roughly 45-68% of the pass,
dense `MUL_MAT` ~17%, and the rest in GET_ROWS, the router logits matmul and DeltaNet outputs.

opbench at the real shapes, 256 experts / 8 used, gate/up IQ3_S, median of 50:

| rows_per_block | tok 5 | tok 6 | tok 7 |
|---|---|---|---|
| 1 | 0.527 ms | 0.596 ms | 0.691 ms |
| 2 (upstream default) | 0.480 ms | 0.545 ms | 0.619 ms |
| **4** | **0.446 ms** | **0.533 ms** | **0.594 ms** |
| 8 | 0.484 ms | - | - |

Harness: `GGML_CUDA_MOE_RPB=<n>` (new env hook in `mul_mat_vec_q_moe_launch`). rows_per_block=4
is 7% faster at 5 tokens and 2-4% at 6-7.

**Prediction:** +1 to +3% TG, so 5 reps and a 2% bar before it counts.

**Result: it is a loss, and the prediction was wrong.** nmax 6, `prompt2.txt`, 400 tokens, 5 reps:

| rows_per_block | reps (tok/s) | median | acceptance | mean_len |
|---|---|---|---|---|
| **2 (default)** | 33.38 34.30 34.52 34.23 34.26 | **34.26** | 0.96657 | 5.68 |
| 4 | 33.61 34.18 34.24 34.01 33.99 | 34.01 | 0.96657 | 5.68 |

-0.7%. Acceptance and mean_len are identical to the digit, so the arithmetic is identical and the
loss is in the pass. This is AGENTS.md rule 2 happening again: the kernel is 3% faster in
isolation and the model is slower, because the kernel is not the whole pass and the block count
and wave shapes that change with it are not modelled by opbench.

rows_per_block stays at 2. The env hook is kept so the test is reproducible, default unchanged.
`rows_per_block=8` was killed after ~20 minutes without completing a single tok-6 case: treat it
as unverified, not as merely slower.

Note the expert matmul runs at only 40-42 GB/s against a 110 GB/s ceiling, and IQ3_S is the
ALU-heavy type (AGENTS.md: classic 4/5-bit types run at 65-78 GB/s where IQ3_S collapses). The
per-expert read is theoretical anyway: at mean_len 5.7 the draft tokens share most experts.
Kernel-level gains beyond a few percent would need the IQ3_S decode cost itself addressed, which
means a requant, and requants have a poor record here (predicted 48 tok/s, delivered 30.5).
