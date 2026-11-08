# The MUL_MAT_ID batch cliff is not launch geometry - three negative results

> **Correction, 2026-11-03.** The tok_in_z experiment in this file ("moving the token into
> blockIdx.z leaves the cliff unchanged") was inert and its conclusion is withdrawn: MMVQ was
> never selected at 5-6 tokens, so the modified kernel never ran. The cliff turned out to be
> the dispatch table plus an sm_72 arch check - see
> [2026-11-03-cliff-is-the-dispatch-table.md](2026-11-03-cliff-is-the-dispatch-table.md), where
> token-in-z plus a raised cap makes 5 tokens 2.15x faster.

Written 2026-11-02. `TODO.md` left the batch-size cliff as the one open lever, with an estimate
that removing it was worth "roughly 4.5 tokens per 125 ms pass, the 38-40 tok/s range". Three
experiments below close the launch-geometry line completely and put a firmer number on what the
cliff is actually worth.

Baseline being explained, `bench/opbench 20`, gate/up shape (K=2048, n=512 rows, 8 of 256
experts), ms:

| tokens | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| IQ3_S | 0.180 | 0.208 | 0.264 | 0.328 | **0.842** | 0.978 |
| IQ4_XS | 0.159 | 0.178 | 0.230 | 0.272 | 0.327 | **0.859** |

Marginal cost of each extra column, IQ3_S: 0.028, 0.056, 0.064, **0.514**, 0.136. Columns 2-4
cost 0.03-0.06 ms each. The 5th costs 0.51. The 6th drops back to 0.14.

## Test 1: rows per block. Prediction was wrong.

`mul_mat_vec_q_moe_launch` hardcodes `rows_per_block = 2`. Upstream tuned that on other
hardware; llama.cpp PR #31268 reports +38% on Orin from changing rows and warps per block. On
Volta a block covers only ~2 KB of weights, so more rows per warp looked like it should hide
more latency. The prediction, written to disk first
(`./2026-11-02-rpb-prediction.md`), was **15-30% faster gate/up verify time**, with the
cliff softened but not removed.

Hook `GGML_CUDA_MMID_RPB` in place, gate/up, ms, IQ3_S / IQ4_XS:

| RPB | tok=4 | tok=5 | tok=6 |
|---|---|---|---|
| 1 | 0.345 / 0.288 | 0.868 / 0.341 | 0.998 / 0.884 |
| 2 (stock) | 0.352 / 0.281 | 0.866 / 0.334 | 0.991 / 0.872 |
| 4 | 0.319 / 0.275 | 0.857 / 0.331 | 1.011 / 0.887 |
| 8 | 0.331 / 0.279 | 0.862 / 0.346 | 0.982 / 0.868 |

Flat. The best cell is RPB=4 at tok=4, 0.319 against 0.352, about 9% - and the stock figure
itself ranges 0.333-0.352 across runs, so that is inside noise. **The 15-30% prediction did not
happen**, and the cliff is untouched at every rows-per-block. Register use rises steeply with
RPB (fused IQ3_S: 71 / 87 / 118 / 218 registers for RPB 1 / 2 / 4 / 8) which offsets any gain.
The kernel is not short of per-thread memory-level parallelism. Hook reverted.

## Test 2: split the multi-column launch into <=4 columns. Nothing.

ik_llama PR #2798 fixes the same class of cliff on GCN/CDNA by splitting MMVQ calls into
launches of at most 4 columns, "as is already done for Pascal above 12". This is the one
structural fix in the upstream record that had not been tried here. Hook `GGML_CUDA_MMID_COLS`
chunks the MoE launch over the token dimension, shifting only the token-strided bases
(activations, ids, dst), which is safe because the shared-expert column overwrites `dst` with
`fusion.shared_dst` and never reads the shifted pointer.

Full curve, IQ3_S gate/up, ms:

| COLS | tok=1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| 0 (stock) | 0.180 | 0.208 | 0.264 | 0.328 | 0.842 | 0.978 |
| 1 | 0.178 | 0.219 | 0.273 | 0.342 | **0.849** | 1.006 |
| 2 | 0.178 | 0.205 | 0.271 | 0.324 | **0.832** | 0.963 |
| 4 | 0.178 | 0.207 | 0.269 | 0.329 | **0.854** | 0.981 |

Five separate 1-column launches cost the same as one 5-column launch (0.849 vs 0.842). The
split was verified to be real, not an inert env var, by a debug print: `ncols_dst=5 launches=2
cols_step=4`.

This is the important result. A 1-column launch costs 0.18 ms on its own. Yet the 5th column
costs 0.51 ms even when it is issued as its own launch with its own data. So the cliff is not
"one launch with 5 columns is slow" - it is "a 5th column of this gather is slow, whoever
issues it".

## Test 3: serialize the chunk launches. Worst of the three.

If the sub-launches were overlapping in the stream and thereby corrupting each other's memory
behaviour, forcing order should fix it. `GGML_CUDA_MMID_SYNC=1` inserts
`cudaStreamSynchronize` between chunks (graphs disabled for the measurement, since a sync is
illegal inside capture). IQ3_S gate/up at 5 tokens: **0.858 unsynchronised vs 0.907
synchronised**. Serialisation costs the sum of five standalone launches, as it should, and is
1.5x slower than stock. Inter-launch concurrency is not the mechanism.

## What this establishes

The 5th column costs ~0.51 ms in every configuration tried: same launch, separate launch,
synchronised separate launch. The cliff is a property of the gather itself, at a type-specific
column count (IQ3_S at 5, IQ4_XS at 6), and it is now excluded as:

- warp count (`calc_nwarps` forced, 2026-11-02)
- register spills (`cuobjdump -res-usage`: `STACK:0 LOCAL:0`)
- template specialization on `ncols_dst` (the MoE kernel takes it at runtime)
- the expert-id distribution or scatter pattern (2026-11-02-mmid-cliff-probes.md, probe 1)
- the block's warp count at 5 tokens (same file, probe 2)
- rows per block (this file, test 1)
- launch decomposition, and inter-launch concurrency (this file, tests 2 and 3)

It is also not the MMVQ-vs-MMQ choice: forcing MMQ at batch 5 is neutral for IQ3_S
(`2026-11-02-per-op-profile.md`), so two unrelated kernels with different launch shapes both
land at ~20 GB/s once past the threshold.

Note that the *dense* ops in the same harness scale normally from batch 1 to batch 5
(`attn_qkv` 0.301 -> 0.292, `attn_output` 0.145 -> 0.172, `output` 5.241 -> 8.091), so the
harness is not inflating the batch-5 column. The pathology is specific to the expert gather.

## Consequence: what a perfect fix would actually be worth

Measured end to end, control model, `prompt2.txt`, 32768 ctx, `results.csv` / `spec.csv`:

| config | acceptance | mean_len | tok/s | implied pass |
|---|---|---|---|---|
| `--spec-draft-n-max 3` | 0.98940 | 3.69 | 31.6 | 116.5 ms |
| `--spec-draft-n-max 5` | 0.99361 | 5.20 | 28.2 | 185.0 ms |

Two extra columns cost +68.5 ms and buy +1.51 accepted tokens. Acceptance does not decay with
draft depth on this prompt - the drafter is close to always right - so the entire penalty is
pass latency.

Cliff-free gate/up at 6 columns, extrapolated from the measured 1-4 column slope of
0.047 ms/column, is ~0.43 ms against 0.978 measured: a 0.55 ms tax per op, about 33 ms per pass
across 30 layers, plus a similar tax on `ffn_down_exps`. Removing all of it gives a 6-column
pass of roughly 135-150 ms, i.e. **35-38 tok/s**. `TODO.md`'s 38-40 was the optimistic end of
that range.

For scale: 45 tok/s at mean_len 5.20 needs a 6-column pass of 115 ms - the same as today's
4-column pass. 60 tok/s needs 87 ms. That is a 1.6-2.1x cut in a pass that is 76% MoE expert
matmul, and no launch or scheduling change in 28 flag configurations and seven kernel
hypotheses has moved it. The cliff is worth chasing for ~35-38 tok/s, not for 45-60.

## Reproducing

```sh
cd bench
./opbench 20                                                        # stock curve
GGML_CUDA_MMID_RPB=4 ./opbench 20                                   # needs the RPB hook re-added
GGML_CUDA_MMID_COLS=1 ./opbench 20                                  # needs the COLS hook re-added
GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_MMID_COLS=1 GGML_CUDA_MMID_SYNC=1 ./opbench 20
```

All three hooks were reverted. `git diff ggml/src/ggml-cuda/mmvq.cu` shows only the two
pre-existing experiment hooks, 21 inserted lines, 0 removed.
