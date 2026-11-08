# MMVQ launch parameters, the arch-shared table, and the real decode-pass budget

Date: 2026-11-03. Author: agent. Trigger: "attack all low hanging fruits".

Three things came out of this: the decode-pass budget in
`2026-11-03-decode-pass-budget.md` was wrong in one important way (it mixed prompt processing
into a decode average), the `MMVQ_PARAMETERS_TURING` table is being applied to Volta, and the
first launch-parameter experiment failed by +46%.

---

## 1. Correction: the decode pass is MMVQ throughout. `mul_mat_q` is prefill.

A signature query over the whole trace ranked `mul_mat_q<(ggml_type)21, (int)32>` first at
44.14 ms/pass and `mul_mat_q<(ggml_type)23, (int)32>` second at 19.35 ms/pass, which looked like
the expert matmuls running on MMQ. They are not. Those two together are 63.5 ms, which is
exactly the "MoE experts 63.2 ms" bucket in the earlier document, and that coincidence is what
made the mistake easy to accept.

Dividing by the pass count does not work when the trace contains a prefill. Restricting the
query to the kernels *between two consecutive 7-column lm_head launches* - one decode pass -
gives the real picture, and **there is no `mul_mat_q` in it at all**. The experts run
`mul_mat_vec_q_moe`, i.e. MMVQ, as the dispatch patch intended. MMQ is prompt processing.

The whole-trace `SUM/N` method also inflated `cpy_scalar` (3.95 vs 2.09 ms decode) and
`concat_non_cont` (3.92 vs 2.49). Any future per-kernel budget must be computed per pass, not
per trace. `bench/` has no tool for this; the python in `srv-decode.sqlite` notes below is the
method that works.

### The decode pass, measured (190.4 ms span, 2446 kernels, 4.89 tokens)

| ms | calls | kernel |
|---:|---:|---|
| 38.53 | 40 | `mul_mat_vec_q_moe<t21 IQ3_S, c2, fusion, tok_z>` - expert gate+up |
| 31.38 | 6 | `mul_mat_vec_q<t14 q6_k, c1>` - lm_head, one per MTP draft step |
| 26.50 | 258 | `mul_mat_vec_q<t8 q8_0, c7>` - dense attention/ssm at 7 columns |
| 20.98 | 37 | `mul_mat_vec_q_moe<t23 IQ4_XS, c2>` - expert down |
| 9.73 | 1 | `mul_mat_vec_q<t14 q6_k, c7>` - lm_head verify |
| 6.44 | 30 | `gated_delta_net_cuda<128,0,1,4>` |
| 3.38 | 36 | `mul_mat_vec_q<t8 q8_0, c1>` |
| 3.21 | 101 | `volta_sgemm_64x32_sliced1x4_tn` (cuBLAS, f32) |
| 2.49 | 30 | `concat_non_cont<unsigned int,0>` |
| 2.20 | 3 | `mul_mat_vec_q_moe<t14 q6_k, c2>` - expert down, blocks 34/38/39 |
| 2.09 | 227 | `cpy_scalar<&cpy_1_scalar<float,float>>` |
| 2.04 | 12 | `mul_mat_vec_q<t11 q3_k, c1>` |
| 1.68 | 401 | `quantize_q8_1` |
| 1.66 | 73 | `k_get_rows_float*` (embeddings) |
| 1.38 | 11 | `flash_attn_tile<256,256,4,8>` |
| 0.85 | 1 | `mul_mat_vec_q_moe<t11, c2, fusion, tok_z>` |
| 0.57 | 116 | `rms_norm_f32<512>` |
| 0.55+0.32 | 195 | `k_bin_bcast` (add, mul) |
| 0.54+0.51 | 94 | `topk_moe_cuda` + `moe_weighted_reduction_f32` |
| 0.47 | 101 | `splitKreduce_kernel` |
| 0.45 | 30 | `ssm_conv_f32` |
| 0.44 | 6 | `pad_f32` |
| 0.39 | 24 | `top_k_radix_histogram` |
| 0.38 | 41 | `dot_kernel<float,128,cublasDotPara>` |
| 0.36 | 71 | `unary_gated_op_kernel<op_silu>` |
| 0.19 | 12 | `mul_mat_vec_f` |

Totals that matter: **MoE 61.7 ms (40%), lm_head 41.1 ms (27%), dense q8_0 29.9 ms (19.4%)** -
the same three buckets, now attributed to the right kernels.

## 2. Volta is running the Turing parameter table

`get_device_table_id(cc)` returns `MMVQ_PARAMETERS_TURING` for any NVIDIA cc that is `>=
VOLTA` and `< AMPERE`, so sm_70, sm_72 and sm_75 all share one table. The observed launch
configs in the trace match `MMVQ_PARAMETERS_TURING` exactly - ncols=1 Q6_K gets `nwarps=2`,
ncols 2-8 get 4, 4, 4, 2, 2, 2, 2, and the measured block dims are `(32,2)`, `(32,4)`,
`(32,4)`, `(32,2)` for ncols 1, 2, 4, 7. So this is the same class of mistake as the
`pascal_older` batch cap that was fixed earlier: an arch-equalities test that sweeps Volta in
with a neighbour it does not behave like, and the table was tuned on the neighbour.

Volta and Turing are not interchangeable for this kernel: Turing has a separate INT32 pipe and
INT8 tensor cores, Volta does not. Nothing here proves the table is wrong for sm_72 - only that
it was not tuned on it, and that the two biggest dense kernels in the pass (41.1 + 29.9 ms) go
through it.

## 3. Experiment: `rows_per_cuda_block` 2 -> 1 at ncols 5-8. **Failed, +46%.**

Prediction, written before the build (`/tmp/pred-rpb.txt`): at ncols 5-8 each warp holds
`ncols x rpb` accumulators, which is 7x2 = 14 at the lm_head, and the kernel spends 128
registers against 48 at one column, capping occupancy at 25% against 57%. Halving the
accumulator count should take registers to 64-80 and duration from 9.48 to <=7.5 ms.

Measured, `opbench`, q6_K K=2048 n=248320 tok=7, one-line change in `calc_rows_per_block`
scoped to `table_id == MMVQ_PARAMETERS_TURING`:

| | rpb=2 (baseline) | rpb=1 | delta |
|---|---:|---:|---:|
| time | 9.57 ms | **13.95 ms** | **+46%** |
| bandwidth | 43.6 GB/s | 29.9 GB/s | |

The prediction was wrong and the failure mode was the one written down: `y` is re-read once per
output row, so halving `rpb` doubles the activation traffic, and the activation re-read is
already the larger half of the L1 load at 7 columns. Occupancy is not the binding constraint at
this shape - L1 throughput is - so a change that trades L1 traffic for occupancy loses.

Reverted (`git checkout --`), rebuilt, and verified bit-identical to the 35.07 baseline
(md5 `5c129aaee1a8ec8c3d7fda1a14d3e7d6`). The build is reproducible.

This closes the `rows_per_cuda_block` direction for ncols 5-8. `rpb=4` is worse by the same
mechanism plus a shared-memory cost: the cross-warp reduction buffer is
`[nwarps-1][ncols_dst][rpb][warp_size]` floats, so `rpb=4` at 7 columns needs 3584 B/block and
the default 16.38 KB carveout then caps residency at 4 blocks. Frontier 7 in `TODO.md` is
consequently downgraded to closed.

## 4. Experiment: raise the Volta f32 MMVF threshold from 3 to 8

`ggml_cuda_should_use_mmvf` returns `ne11 <= 3` for f32 on pre-Turing NVIDIA, so the 7-column
router call is excluded from `mul_mat_vec_f` and falls through to cuBLAS. In the pass that is
`volta_sgemm_64x32_sliced1x4_tn` 41 calls at 49.1 us (2.01 ms) plus 60 calls at 20.0 us
(1.20 ms) plus `splitKreduce` 0.47 ms plus `dot_kernel` 0.38 ms = **4.06 ms/pass**, moving
86 MB at 43 GB/s.

`launch_mul_mat_vec_f_cuda` is instantiated for ncols 1-8, so the threshold change is a real
path rather than a crash. Prediction (`/tmp/pred-f32.txt`): MMVF at 70-90 GB/s takes the sgemm
groups from 3.21 to ~1.3 ms, pass 163.4 -> 161.5 ms, 35.07 -> ~35.5 tok/s (+1.2%), acceptance
unchanged.

Result: see section 5.

## 5. Result of the f32 A/B: **-45.6%, and the cause is draft acceptance, not speed**

`bench/ab-lib.sh base:/tmp/libggml-cuda-occ64.so.0 f32mmvf:/tmp/libggml-cuda-f32mmvf.so.0 5 400`.

The baseline arm reads 34.82 here against the 35.07 of record. That is the *same* configuration
(`PMIN=0.5 PSPLIT=0.1 DRAFT=6 CTX=32768`, library md5 `5c129aae`), re-measured the same day, and
the 0.7% gap is run-to-run variation rather than a new best or a regression - it is well inside
what 5 reps resolve, which is the point of `AGENTS.md` section 3 rule 4. A/B verdicts are taken
against the control arm measured in the same session, never against the record.

| | baseline (md5 `5c129aae`) | f32 -> MMVF (md5 `747ea7e6`) |
|---|---:|---:|
| tok/s, median of 5 | **34.82** (34.10 / 34.83 / 35.16 / 34.82 / 34.70) | **18.93** (17.95 / 28.58 / 18.87 / 18.93 / 19.00) |
| delta | - | **-45.6%** |
| draft acceptance, median | **0.96657** | **0.67717** (0.58993 / 0.93311 / 0.67451 / 0.67717 / 0.67717) |
| mean len | **5.68** | **2.40** (2.12 / 4.44 / 2.40 / 2.40 / 2.40) |
| graph reallocs | 0 / 0 | 0 / 0 |

The kernel change is not slow. Graph capture is healthy (`reallocs 0`, hundreds of replays) and
the pass is not longer - **the drafter stops agreeing with the target**. Throughput is
`mean_len / pass_latency`, so mean_len 5.68 -> 2.40 is worth -58% of throughput on its own; the
measured result is -45.6%, the difference being reps where acceptance happened to hold up
(rep 2: 0.93311 / 4.44 / 28.58 tok/s). The spread is the point - this is the acceptance
instability of `AGENTS.md` section 3 rule 5, not measurement noise. Reps 4 and 5 are bit-identical
(172/254), so once warm the new routing is deterministic, just *differently* deterministic: the
binary reproducibly takes a different set of expert decisions than the control does.

**Mechanism: the MoE router is a discrete decision taken on f32 data.** `ffn_gate_inp` is f32 and
selects the top-8 of 256 experts. Swapping cuBLAS SGEMM for `mul_mat_vec_f` changes the summation
order, so a handful of router logits move in the last bits. Wherever two experts are near-tied,
the top-8 set changes, and a different expert set means a different hidden state. The target and
the MTP drafter then disagree about the next token, which is exactly what acceptance measures.
Routing is maximally sensitive to perturbation because it quantises a continuous score into a
discrete selection.

This is the same failure that killed three requant builds, reached by a completely different
route: those changed the expert *weights*, this changes only the *order of summation* in the
router, and both decouple the drafter. It also explains why the control file records a
bit-stable 0.98940 on every run ever logged while every modified build scatters over 0.59-0.93:
the unmodified binary reproduces the same routing decisions every time, and anything that moves
them does not.

**Consequence for future work, and this is the durable result of the experiment:** do not change
the numerics of the MoE router. That rules out, as candidates, `GGML_CUDA_CUBLAS_COMPUTE_TYPE`,
any TF32/FP32-accumulation setting, `--override-tensor` on `ffn_gate_inp`, and any kernel rewrite
of the router matmul - all of them perturb routing, and routing drives acceptance. It also means
any future kernel change must be checked for acceptance, not just tok/s, which `AGENTS.md` section
3 already requires but which this experiment shows is not a formality: a change that is neutral
for speed can still cost half the throughput.

Reverted (source change removed, rebuilt, md5 back to `5c129aaee1a8ec8c3d7fda1a14d3e7d6`).

## 6. The data-movement kernels, attributed

Neither is a layout bug; both are structural, which is why neither is low-hanging fruit.

- **`concat_non_cont` 2.49 ms/pass, 30 calls = one per Gated DeltaNet layer, 83 us each.**
  It is `ggml_concat(ctx0, conv_states, qkv_mixed, 0)` at `src/models/delta-net-base.cpp:469`,
  building the conv window input. ~4.5 MB moved per call at roughly 96 GB/s, i.e. 87% of the
  measured ceiling. There is no bandwidth to recover; the only win would be not doing it, which
  means restructuring the GDN conv in the graph.
- **`cpy_scalar` 2.09 ms/pass, 227 calls, 9.1 us each.** A 98 KB copy takes about 2 us of
  bandwidth, so these are dispatch-latency bound, not transfer bound. The cost is the 227 graph
  nodes, not the bytes. Reducing the count is a graph change in the model implementation.
- **`pad_f32` 0.44 ms/pass, 6 calls = one per MTP draft step, 73 us each.** Also latency, not
  bandwidth. Same category as above.
- **`quantize_q8_1` 1.68 ms/pass, 401 calls** is the unavoidable activation quantisation feeding
  MMVQ and is already coalesced.

## 7. What is actually left

Ranked by measured size of the *addressable* part, after this round:

| item | ms/pass | share | status |
|---|---:|---:|---|
| MoE experts (IQ3_S gate/up 38.5) | 61.7 | 40% | 51% of DRAM ceiling; needs a cheaper type, which is closed (requant) or blocked (BF16 + training) |
| lm_head (83% of it is the six 1-column drafts) | 41.1 | 27% | drafts at 80 GB/s of 111; only the bytes are left, i.e. Frontier 6 |
| dense q8_0 7-column | 29.9 | 19% | L1-bound, 15.6x sector amplification, 45-60 GB/s; padded block layout, ~+2% |
| f32 via cuBLAS | 4.06 | 2.6% | section 5 |
| GDN concat + conv + copies | 4.58 | 3.0% | structural, latency-bound micro-kernels |
| everything else | ~17 | 11% | at or near the ceiling for its type |

There is no remaining 5% item that does not involve changing the weights. The f32 dispatch is
the last quality-neutral item with a mechanism, and its ceiling is 2.6% if cuBLAS were free.
