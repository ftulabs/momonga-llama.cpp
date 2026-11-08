# The MoE kernel is 40% of the pass and moves 3.1 GB of it - and rows-per-block is invisible in situ

2026-11-03. The MoE rows-per-block retest ended with an open question: the kernel measured 4%
faster standalone but the model did not move, so either the pass budget over-attributed time to
the MoE or the lever was not reaching the model. This closes both halves. **The share is real
(40.1% of kernel time) and the lever is real in the launch grid but produces no change in situ
(63.98 vs 63.95 ms/pass).** The reason is that the MoE kernel's entire cost is per-token DRAM
movement of expert slices, and the L1 sector cut that rows-per-block buys does not touch it.

## The pass budget on the current build (36.56 tok/s configuration)

nsys, graphs on, `bench/prof-nsys.sh`, one pass defined as the kernels between two consecutive
7-column q6_K lm_head verify launches - 92 passes in the trace, all decode, no prompt
processing. Zero-fill is required when taking a median per signature: a signature that appears
in only some passes otherwise reports its median over the passes it appears in, which inflated
`mul_mat_q` to 1.5 s/pass and the MoE to 2.7 s/pass in two earlier drafts of this table.

| category | ms/pass | share |
|---|---:|---:|
| MoE expert mmvq | 63.98 | 40.1% |
| lm_head q6_K, 1 column (6 draft steps) | 28.84 | 18.1% |
| dense q8_0, 7 columns (verify) | 24.82 | 15.6% |
| lm_head q6_K, 7 columns (verify) | 8.88 | 5.6% |
| misc (concat, scale, get_rows, bcast, ...) | 8.54 | 5.4% |
| GDN / ssm | 7.52 | 4.7% |
| quantize / copy | 3.79 | 2.4% |
| cuBLAS (f32 router) | 3.26 | 2.0% |
| dense q8_0, 1 column | 2.79 | 1.7% |
| attention | 2.40 | 1.5% |
| dense Q3_K, 1 column | 2.11 | 1.3% |
| router top-k | 1.28 | 0.8% |
| rms_norm | 1.20 | 0.7% |
| **kernel busy** | **159.1** | |
| **pass wall** | **170.6** | |

The 11.5 ms of non-kernel time per pass (6.7%) is graph launch and gaps. The three big rows are
the three big levers, and all three have now been worked: the dense path paid +4.3%, the lm_head
1-column draft call is at 80 GB/s, and the MoE is the one that did not move.

### Implied DRAM per pass: about 8 GB, not 4.18

Three kernels have known-exact traffic, and each is consistent with its measured time:

- lm_head `output.weight` is q6_k and 417 MB: 7 reads per pass (6 drafts at 1 column plus the
  verify) = **2.9 GB in 37.7 ms = 77 GB/s**
- the dense q8_0 tensors total 2.07 GB and are read once per verify forward = **2.07 GB in
  24.8 ms = 83 GB/s**
- the MoE reads 8 experts x 3 matrices x 451 KB per token = 10.8 MB per token per layer, and
  the verify runs 7 tokens over 41 layers = **3.1 GB in 64.0 ms = 48 GB/s**

Total ~8.1 GB in 170.6 ms wall = **47 GB/s, 43% of the 111 GB/s ceiling**. The older "2.34 GB
batch-1 / 4.18 GB verify pass" figures described only the 41-layer forward and omitted the six
1-column lm_head draft calls, which alone are 2.5 GB. Neither number was wrong for what it
measured; the pass total is the one that matters for speculation because the drafter's lm_head
reads are 31% of all the traffic in the pass.

## The MoE rows-per-block lever, measured in situ

`GGML_CUDA_MOE_RPB` is a runtime hook, so both arms are the same binary. Two traces, adjacent in
time, same prompt, same flags, 92 passes each:

| arm | MoE launches/pass | MoE gridX | MoE ms/pass | lm_head c1 ms/pass | pass busy | pass wall |
|---|---:|---|---:|---:|---:|---:|
| rpb=2 (default) | 82 | 256 / 1024 | **63.98** | 28.84 | 159.1 | 170.6 |
| rpb=4 | 82 | 128 / 512 | **63.95** | 27.98 | 158.0 | 169.2 |

The grid did change (256 -> 128 and 1024 -> 512 on every launch, 5886 of each shape in both
traces), so the hook was live. Launch counts are identical, so this is not a work-count
artefact. The MoE kernel time is unchanged to 0.05%.

An earlier pair of traces, an hour apart, read 66.40 ms (rpb=2) against 63.95 ms (rpb=4) and
looked like a -3.7% win. **That was session drift.** Re-tracing the default arm afterwards
returned 63.98. The board drifts about 4% between profiling sessions in DRAM-bound kernels, so
a cross-session kernel-time comparison - even of the same binary - can invent a result. Both
arms of a kernel A/B must be traced adjacent in time.

The drift is also visible inside one session: the lm_head 1-column kernel, which
`GGML_CUDA_MOE_RPB` cannot touch, differs by 2.9% between the two adjacent traces above. That
kernel and the MoE kernel are the two most DRAM-bound in the pass; the compute-bound kernels
(attention, cuBLAS, quantize, router, rms_norm) agree to 0-1% between the same two traces. So
the residual is a memory-clock/locality difference, not the change under test.

## Why the standalone win does not exist in the model

Per-launch MoE time in situ, binned by token count (the prompt phase supplies the intermediate
counts; the model's steady state is 7):

| tokens | gate/up, gridX 256 | ms/token | down, gridX 1024 | ms/token |
|---:|---:|---:|---:|---:|
| 2 | 0.286 ms | 0.1431 | 0.169 ms | 0.0846 |
| 3 | 0.423 | 0.1411 | 0.250 | 0.0833 |
| 4 | 0.559 | 0.1398 | 0.330 | 0.0825 |
| 5 | 0.697 | 0.1393 | 0.410 | 0.0821 |
| 6 | 0.837 | 0.1394 | 0.492 | 0.0820 |
| 7 | 0.974 | 0.1392 | 0.572 | 0.0818 |

The cost is **exactly linear in token count with an intercept of about 0.006 ms**. Every core
costs one full pass over its own expert slices, which is right: `channel_x` is
`ids[channel_dst + token_idx*ids_stride]`, so each token selects its own 8 experts and the 7
token-slices of a launch read 7 different 3.6 MB slices. There is no cross-token reuse to find -
not in the kernel, and not by moving tokens from `blockIdx.z` to `threadIdx.y`. That reframes
the earlier "the activation vector is re-read once per output row" result: in this kernel the
per-row re-reads are noise against per-token expert streaming.

At 3.6 MB per 0.0696 ms (gate/up, per matrix) and 3.6 MB per 0.0818 ms (down) the kernel is
running at **52 and 44 GB/s** of the 111 GB/s ceiling, with no launch overhead component at all.

`opbench`/`ncu` at the same shapes do not measure this regime:

- gate/up (K=2048, n=512, IQ3_S, 8 of 256 experts, 7 tokens): 402.66 us at rpb=4 against 414.02
  at rpb=2, i.e. **-2.7%**, and DRAM 24.91 MB - the probe streams per token like the model.
- down (K=512, n=2048): 195.94 us at rpb=4 against 218.43 at rpb=2, i.e. **-10.3%** - and DRAM
  only 6.1 MB for 7 tokens, i.e. the probe re-reads one expert slice for all 7 tokens. That is
  not what the model does, so the -10.3% is measured on the wrong traffic pattern.

**Prediction, written before the run: "rpb=4 will be slower on the down shape, +2% to +8%,
because K=512 gives a row only two IQ3_S blocks of work." It was -10.3% faster.** The
prediction was wrong because it reasoned about instruction-level parallelism and the probe
turned out to be dominated by a fixed, shared-slice cost that the model does not have.

## What this means for the MoE

- It is 40% of the pass and it is a pure streaming kernel at 44-52 GB/s. The lever that remains
  is achieved DRAM efficiency - more bytes in flight per warp - not launch geometry, not L1
  sectors, not occupancy. Rows-per-block changed L1 sectors 32% and the model not at all;
  registers/occupancy was already taken by the 64-register launch bounds (+2.25%).
- `gatherbw` measured 108 GB/s on a scattered read of a real expert matrix, so the pattern can
  reach the ceiling, but the same counters that showed IQ4_XS at 92.7 GB/s showed IQ3_S at 56.7
  on an identical shape. The gap is inside the IQ3_S dot - dequantisation work and the load
  latency it exposes, 1.63x the warped instructions of IQ4_XS - and the two levers that have been
  tried (registers, L1 sectors via rows per block) moved occupancy and sectors without moving the
  model. Changing the expert type is a weight change and is closed by the acceptance rule.
- Any future MoE experiment must be measured in situ. A single-launch `ncu` number on this
  kernel is not evidence: it has now produced a 4% and a 10% win that the model did not see.
- Do not attempt to make tokens share weight loads. It is not a missed optimisation, it is
  impossible: different tokens route to different experts.

## Verdict

`GGML_CUDA_MOE_RPB` stays at its default of 2, the budget's 40% attribution stands, and the MoE
kernel's remaining 2x is a DRAM-efficiency problem with a very small per-warp footprint. No
change to the shipped configuration, no change to 36.56 tok/s.

## Artefacts

- `/tmp/srv-d.sqlite` (default arm, re-traced), `/tmp/srv-r4.sqlite` (rpb=4 arm), both 92 passes
- `/tmp/pred-moe-down.txt`, `/tmp/pred-moe-insitu.txt`, `/tmp/pred-moe-insitu2.txt` - the
  predictions, written before each run
- `/tmp/prof-moe4.sh` - the rpb=4 profiling variant of `bench/prof-nsys.sh`
