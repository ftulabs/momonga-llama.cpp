# Where a decode pass actually goes (nsys, real server, graphs on)

Date 2026-11-03. First profile of a real `llama-server` decode run with CUDA graphs left
enabled. This replaces `GGML_CUDA_OP_TIME` for the purpose of splitting a pass, because that
hook perturbs the timeline and only covers nodes it is patched into.

## How it was captured

`/tmp/profile-server.sh` wraps `llama-server` in `sudo nsys profile --trace=cuda`. Root is
required for CUPTI on this board. Graphs are left on (`env -u GGML_CUDA_DISABLE_GRAPHS`).
Flags are identical to `bench/run-best.sh`: `-c 32768 -fa on -ngl 99 -ncmoe 0 -t 2 -tb 2
-ctk q8_0 -ctv q8_0 -ub 32 -b 32`, `--spec-type draft-mtp --spec-draft-n-max 6
--spec-draft-n-min 0 --spec-draft-p-min 0.5 --spec-draft-p-split 0.1`.
Model `Cyber-Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS.gguf`. One `/completion` of 200 tokens,
`cache_prompt: false`, `temperature: 0`. Artifact `/tmp/srv-decode.nsys-rep` (24 MB).

**Flush to do first.** The teardown must kill only `llama-server` and let `nsys` exit on its
own. `pkill -x nsys` makes nsys die before it writes the report; the first attempt lost the
whole capture that way. A non-zero exit from `wait` is normal and does not mean the capture
failed - check for the `.nsys-rep`.

`nsys stats --report gpukernsum` works on-device as of feed release `2026.11.03-2`.

## What the run did

```
prompt eval time =  12455.91 ms / 267 tokens (21.44 t/s)   <- profiler-starved, ignore
       eval time =   8550.11 ms / 200 tokens (23.27 t/s, 42.97 ms/token)
draft acceptance = 0.97368 (148 accepted / 152 generated), mean len = 4.89
graphs reused = 31, graph reallocs = 0
```

`23.27` is not comparable to the 35.07 record - nsys adds ~17% to the pass. Mean length is
4.89 here against 5.68 on the bench harness with the same flags, so passes are not the same
unit of work. **Use the per-pass kernel shares, not the tok/s.**

## Structure of one pass

Delimited by the 7-column Q6_K lm_head launch, which happens exactly once per pass. In steady
state (11 consecutive passes 180-200 ms apart):

```
[main forward, 41 layers, 7 columns] -> [lm_head 7 cols]
  -> 6 x ( [MTP draft layer, 1 column] -> [lm_head 1 col] )
```

Measured: **189.5 ms wall, 163.4 ms busy per pass, 26.1 ms idle (13.8%)**. Unprofiled the
pass is 5.68 / 35.07 = 162 ms, i.e. the same as the profiled *busy* time - so the idle is a
tracing artifact (CPU launch starvation), not a real gap, and killing launch overhead is not
a lever. Kernel time itself is not inflated.

## Per-pass kernel budget, 163.4 ms busy

| ms/pass | % busy | kernel | what it is |
|---:|---:|---|---|
| 38.56 | 23.6% | `mul_mat_vec_q_moe<iq3_s,2,1,1>` | routed gate+up, fused |
| 31.45 | 19.2% | `mul_mat_vec_q<q6_k,1>` | **lm_head, 6 MTP drafts** |
| 26.47 | 16.2% | `mul_mat_vec_q<q8_0,7>` | dense attn/ssm weights, main forward |
| 20.96 | 12.8% | `mul_mat_vec_q_moe<iq4_xs,2,0,1>` | routed down |
| 9.74 | 6.0% | `mul_mat_vec_q<q6_k,7>` | lm_head, main forward |
| 6.43 | 3.9% | `gated_delta_net_cuda<128,0,1,4>` | GDN recurrence |
| 3.23 | 2.0% | `volta_sgemm_64x32_sliced1x4_tn` | f32 router matmul, cuBLAS |
| 2.50 | 1.5% | `mul_mat_vec_q<q8_0,1>` | dense attn, 6 drafts |
| 2.48 | 1.5% | `concat_non_cont` | |
| 2.21 | 1.4% | `mul_mat_vec_q_moe<q6_k,2,0,1>` | 3 layers only |
| 2.09 | 1.3% | `cpy_scalar` | 285 launches/pass |
| 1.69 | 1.0% | `quantize_q8_1` | 464 launches/pass |
| 1.37 | 0.8% | `flash_attn_tile<256,256,4,8>` | |
| 1.32 | 0.8% | `k_get_rows_float_vec` | |
| 1.18+0.85 | 1.2% | `mul_mat_vec_q<q3_k,1>` | 3 layers |
| 0.83+0.68 | 0.9% | `mul_mat_vec_q_moe<q3_k,2>` | 3 layers |
| ~6 | ~4% | 30 more kernels, each <0.6 | rms_norm, rope, fwht, topk, moe_reduce, set_rows, pad, conv |

Grouped: **MoE experts 63.2 ms (38.7%), lm_head 41.2 ms (25.2%), dense attn/ssm 32 ms
(19.6%), GDN 6.4 ms, router 3.2 ms, everything else ~15 ms.**

The tensor types came from `tools/gguf_tensors.py dump`, not from the filename. The file says
`UD-IQ4_XS` but the routed experts are `iq3_s` for gate/up and `iq4_xs` for down, all
attention/ssm/dense weights are `q8_0`, `output.weight` is `q6_k`, and 3 layers carry `q3_k`.
Do not assume the quant name describes the tensors.

## The two facts that matter

1. **`output.weight` is read 7 times per pass.** 2048 x 248320 q6_k is 417 MB. Once at 7
   columns (9.74 ms) and six times at 1 column (5.24 ms each) because every MTP draft step
   needs a full-vocab projection. That is 2.9 GB of DRAM traffic per pass for 4.89 tokens -
   more than the entire main forward reads (~2.1 GB). The 1-column call runs at
   **79.6 GB/s against the 110 GB/s ceiling**, the best-utilised kernel in the run.
2. **The dense `q8_0` 7-column matmuls run at 53.5 GB/s.** 1278 MB of `q8_0` attention/ssm
   weights plus 137 MB of shared-expert weights per main forward, in 26.47 ms. The same
   tensor at 1 column in a draft forward runs at ~85 GB/s. Same bytes, 1.6x the bandwidth,
   different `ncols_dst`. That gap is quality-neutral headroom: at 90 GB/s the main forward
   dense matmuls would cost 15.7 ms instead of 26.5, worth **+6.6%**.

`volta_sgemm` deserves a note: the router `ffn_gate_inp` is f32 (2048x256 = 2.1 MB/layer,
86 MB/pass) and costs 3.23 ms = **26 GB/s** through cuBLAS. That is the worst utilisation in
the run, but the absolute headroom is only ~2.3 ms/pass.

## Next experiment, prediction written down first

Target: the `q8_0` 7-column MMVQ kernel, 26.47 ms/pass.

**Prediction: `ncu` will show it is occupancy-limited by registers, not bandwidth-limited, and
that its DRAM read is close to 1278 MB with little amplification.** Reasoning: the 1-column
path reaches 85 GB/s on the same bytes and the same gather-free access pattern, and the only
difference is 7 accumulators live at once instead of 1. If that holds, the fix is a
multi-warp block or a register budget, the same shape of fix that won on the MoE kernel.
Predicted achievable: 26.47 -> 17-19 ms/pass, i.e. **+4.5% to +6%**, with acceptance unchanged.

If instead `ncu` shows heavy L1/L2 amplification on the 7-column path, the prediction was
wrong and the cause is the access pattern, not occupancy - recorded before the measurement
so this cannot be rationalised after the fact.

## Result: prediction half right, and the real cause is not the column loop

The experiment above was run. **`ncu` on a 19 GB model OOMs the unified memory**
(`NvMapMemAllocInternalTagged: error 12`, then `CUBLAS_STATUS_NOT_INITIALIZED`), so the kernel
was measured in `opbench` at the model's exact geometry instead: `K=2048, n=8192, 7 tokens`,
q8_0, `OPBENCH_NE=1`. `bench/opbench.cpp` gained an `OPBENCH_NE`/`OPBENCH_NU` override for
this, because single mode previously forced 256 experts and therefore always built
`mul_mat_vec_q_moe`, never the dense kernel.

Also note: **plain `opbench` timing is unusable for kernels this small.** It reports
`cpu` (enqueue-only) 0.377 ms against 0.378 ms total, so the loop is CPU-bound on launch and
the number is the launch rate, not the kernel. All figures below are `gpu__time_duration.sum`
from `ncu`.

### What the prediction got right

- Register-limited occupancy: **80 registers/thread**, block limit 12 blocks of 64 threads,
  theoretical occupancy **37.5%**, achieved 34.64%. The same signature as the IQ3_S MoE kernel.
- DRAM is clean: **563,305 sectors = 18.03 MB** for 17.00 MB of weights. The gather is not
  the problem and no amplification happens at DRAM.

### What it got wrong

The kernel is **not** DRAM-bound and occupancy was not the lever. It is **L1-bound**:

| metric | value |
|---|---|
| L1/TEX cache throughput | **82.94%** |
| L2 cache throughput | 52.11% |
| Compute (SM) | 61.92% |
| L1 load sectors requested | **8,767,457** |
| Unique sectors (DRAM) | 563,305 |
| Amplification | **15.6x** |

A multi-warp block or a register budget would have done nothing, because the wall is L1
sectors, not warps in flight.

### The follow-up hypothesis, and its refutation

I read the 94.23% L1 hit rate as the 7-column loop re-reading the weight block once per
column, and implemented a register preload for q8_0 (`mmvq_x_q8_0`, `mmvq_load_x_q8_0`,
`mmvq_dot_x_q8_0`, called from a new `if constexpr (type == GGML_TYPE_Q8_0 && !has_fusion)`
branch in `mul_mat_vec_q`, with the row loop moved outside the column loop).

**Measured inert, and reverted.** Duration 294.69 -> 294.56 us and sectors 8,767,457 ->
8,768,854, i.e. unchanged. A first, cheaper attempt (loop reordering alone, on the theory that
nvcc could then CSE the loads) was also measured inert: 8,771,537 sectors.

Two runs at different column counts give the reason:

| columns | L1 sectors | duration | DRAM sectors |
|---:|---:|---:|---:|
| 1 | 4,515,237 | 261.95 us | 557,323 |
| 7 | 8,768,877 | 288.38 us | 563,403 |

Solving `A + B = 4.52M` and `A + 7B = 8.77M` gives **B = 709k sectors per column (the
activations) and A = 3.81M for the weights, constant in the column count.** nvcc already holds
the weight slice live across the column loop. The 94.23% L1 hit rate was **sector reuse inside
the 34-byte q8_0 struct**, not column re-reads - the hit-rate reading was simply wrong.

Two more facts fall out of that table, both of which matter more than the aborted fix:

- **The 7-column verify pass is nearly free.** It costs 288 us against 262 us at 1 column, +10%
  for 7x the tokens. Verify width is not costing throughput.
- The weight side alone is **3.81M sectors for 563k unique ones, 6.8x**. `block_q8_0` is 34
  bytes (`half d` + 32 int8), so at a 34-byte stride no block is sector-aligned, and
  `get_int_b2` issues overlapping 8-byte loads from two threads per block. Each warp
  instruction straddles roughly twice the minimum number of sectors.

## What this changes about the remaining headroom

Every hot kernel in the pass is limited by the same geometry, not by occupancy or DRAM:

| kernel | ms/pass | utilisation of the 110 GB/s ceiling | L1 note |
|---|---:|---:|---|
| q8_0 dense 7-col | 26.47 | 55 GB/s | 15.6x sector amplification, 8.1x of it intrinsic |
| q6_k lm_head 1-col | 31.45 | 75.9 GB/s | 69.6% L1 vs 68.0% L2, **90.6% L1 hit with 2.4% L2 hit** |
| iq3_s gate/up | 38.56 | 51% of ceiling | 895 L1 sectors per block against 49 of payload |

The general fix is a **sector-aligned padded quant block layout**: pad q8_0 34 -> 64 bytes and
q6_K 210 -> 256, so a block occupies whole sectors and one thread can read it with aligned
vector loads. That is quality-neutral by construction, because padding carries no information.
It is also a project, not a one-liner: it needs a repacked GGUF plus matching load paths in
every affected kernel.

Ceiling for q8_0 alone, from the numbers above: the weight share of L1 traffic is 43%, and
perfect packing would cut it about 3.3x, taking total sectors 8.77M -> ~6.1M, roughly -30% of
L1 work. At 83% L1 utilisation that is about 288 -> 250 us and 26.47 -> ~23 ms/pass, **+2% end
to end**. Not worth the risk on its own. It only pays if the same repack covers the MoE
kernels and the lm_head, whose amplification is larger (18x measured on IQ3_S), which is a
different and much bigger piece of work.

Do not re-attempt the following: loop reordering, per-type register preloading of the weight
slice, or raising occupancy on the dense q8_0 MMVQ kernel. All three are measured inert or
irrelevant, and the mechanism is understood.

## Addendum: the lm_head split by column count (2026-11-03, same trace)

The summary above treated lm_head as one bucket at 41.2 ms. Splitting the same
`CUPTI_ACTIVITY_KIND_KERNEL` rows by signature and launch shape gives a different picture, and
it **falsifies a claim in `AGENTS.md` section 7**.

The 7-column figure in the table above was extrapolated from `opbench` on the dense q8_0 shape.
On the q6_k lm_head the column count is expensive:

| ncols | grid | warps | calls | us/call | bytes | GB/s |
|---:|---|---:|---:|---:|---:|---:|
| 1 (draft) | 248320 | 2 | 201 | 5210 | 417.1 MB | **80.0** |
| 2 | 124160 | 4 | 16 | 6060 | 417.1 MB | 68.8 |
| 4 | 124160 | 4 | 3 | 9318 | 417.1 MB | 44.8 |
| 7 (verify) | 124160 | 2 | 22 | 9728 | 417.1 MB | 42.9 |

Every call reads the whole 417 MB `output.weight` once; only the column count changes. Per pass
(31-32 passes in the trace window):

- six draft GEMVs at 1 column: **32.7-33.8 ms/pass at 80.0 GB/s** - 72% of the 111 GB/s ceiling
- verify call(s) at 2/4/7 columns: **~9 ms/pass**

So **83% of the lm_head cost is the drafts, and they are already close to the memory ceiling.**
That half has no launch-parameter headroom. What is wrong is the verify call, which loses more
than half its bandwidth.

The dense q8_0 kernel does **not** do it - 60.8 GB/s at 7 columns against 63.8 at 1 column -
which is why the "+10% for a 7-column GEMV" measurement was correct for the shape it was taken
on and wrong here. The guess below was that q6_k, being 7x the dot work for the *same* weight
bytes, crosses into arithmetic-bound territory (q6_k unpacks scale, min, 4-byte `ql` and 2-byte
`qh` groups per 16 weights) while q8_0, one int8 MAD per element, stays memory-bound. **The
counters in the result section show that guess was wrong**: `math_pipe_throttle` is the lowest
of the three stalls.

### Prediction, written before the measurement

`ncu` on `mul_mat_vec_q<q6_k, 7>` at K=2048, n=248320 (opbench single mode, `OPBENCH_NE=1`):

- DRAM read near the observed 43 GB/s, i.e. **well below** the 111 GB/s ceiling
- L1 and L2 below saturation (unlike the q8_0 diagnosis, where L1/TEX was 82.9%)
- top stall `math_pipe_throttle` or a high issued-instruction rate, **not** `long_scoreboard`
- achieved occupancy comparable to the 1-column instantiation

If that holds, the 7-column call is ALU-bound and no launch-parameter change will help it; the
only lever left in this bucket is the number of bytes (Frontier 6). If `long_scoreboard`
dominates instead, the prediction is wrong and the kernel is memory-latency-bound, which would
make a launch-config search worthwhile.

**Result: prediction WRONG on both counts, and the mechanism is better than the guess.**

`ncu`, same op, same geometry, `--launch-count 1`, both column counts:

| | 1 column | 7 columns |
|---|---:|---:|
| duration | 5.17 ms | 9.48 ms |
| DRAM read (`lts__d_sectors_fill_sysmem` x 32) | 417.2 MB | 418.2 MB |
| L2 -> L1 (`lts__t_sectors_srcunit_tex_op_read` x 32) | 419.8 MB | 418.6 MB |
| **L1 global-load sectors** | **127,155,108** | **281,265,619** |
| L1 requests | 16,885,760 | 31,784,960 |
| sectors per request | 7.53 | 8.85 |
| L1/TEX throughput | 70.1% | 84.2% |
| L2 throughput | 69.0% | 40.0% |
| Compute (SM) | 55.8% | 63.5% |
| registers/thread | **48** | **128** |
| theoretical / achieved occupancy | 62.5% / 57.4% | **25% / 22.4%** |
| top stall | - | `lg_throttle` 0.85, `long_scoreboard` 0.78, `math_pipe_throttle` 0.35 |

Two things are now certain and they kill my hypothesis:

1. **It is not ALU-bound.** `math_pipe_throttle` is 0.35, the *lowest* of the three stalls, and
   the SM is at 63.5% while L1 is at 84.2%. The top stall is `lg_throttle` - the L1/LSU issue
   queue is full. The kernel is bound by **L1 global-load throughput**.
2. **DRAM and L2 are completely clean and identical at both column counts.** 417 MB from DRAM,
   418 MB from L2, in both cases: the weight matrix is read exactly once and nothing is
   re-fetched from L2. So the extra 154M L1 sectors at 7 columns are all satisfied inside L1.

Where the 2.21x L1 traffic comes from is arithmetic, not speculation. The weight side is
~111M sectors in both cases (248,320 rows x 2048 elements x 6.5625 bits, in 210-byte blocks that
span 7.5 sectors per warp request, i.e. 8.5x amplification on the weights alone). The remainder
is the activation vector: `y` is 2 KB and is re-read **once per output row**, because
`rows_per_cuda_block` is 1 or 2 - 540 MB of L1 traffic for a 2 KB vector at 1 column, and roughly
10x that at 7. So **at 7 columns the activation re-reads are larger than the weight traffic.**

The register cliff is separate and also measured: `mul_mat_vec_q` is declared
`__launch_bounds__(nwarps*warp_size, 1)` - the same "only one block per SM is required" mistake
that `mul_mat_vec_q_moe` had (see `AGENTS.md` section 7). Here it spends 128 registers at 7
columns against 48 at 1, which caps occupancy at 8 blocks and 25%. The block also needs 1.79 KB
of static shared memory for the staged `y`, and the default 16.38 KB carveout caps it at 9
blocks, so cutting registers alone buys 8 -> 9 blocks and nothing more. A useful occupancy fix
has to raise the shared-memory carveout at the same time.

### What that means for the remaining headroom

- **The 1-column draft shape is finished.** 80.0 GB/s against an 111 GB/s ceiling, 57% occupancy,
  DRAM and L2 clean. Six of these per pass at 32.7-33.8 ms/pass is the price of drafting and it is
  not improvable by launch parameters.
- **The 7-column verify call has at most ~4.5 ms/pass in it** (9.48 -> 5.2 ms at best, if it
  could reach the 1-column bandwidth), which is 2.8% of the pass, **+1.0 tok/s**. Getting it needs
  both a launch-bounds change and a shared-memory carveout change, against a compiler that is at
  128 registers for a reason, and the reward assumes L1 sector traffic falls with it.
- **The larger prize is the activation re-read**, and it is shared by every MMVQ kernel:
  `rows_per_cuda_block` controls how often a 2 KB vector is re-read for 248,320 output rows.
  Raising it cuts L1 sectors proportionally and is quality-neutral. But `GGML_CUDA_MOE_RPB=4`
  already showed the trap here - +3% in `opbench`, -0.7% at the model level
  (`TODO.md`, "Closed - do not re-test") - so this must be measured end to end, not in `opbench`.


One side observation that matters for Frontier 3 (adaptive draft depth): the verify width is not
fixed. Across ~31 passes the trace contains 22 lm_head calls at 7 columns, 16 at 2 and 3 at 4, so
the server already varies how many draft columns it verifies from pass to pass. Any adaptive-depth
work starts from a system that is partly adaptive already, and its baseline is this distribution,
not a constant `n-max 6`.

## Addendum (same day): re-measured on the 36.56 build

The budget in this file is from the 34.x build and from a pass definition that later work
corrected twice. It is re-measured in
[the in-situ finding](2026-11-03-moe-share-in-situ.md) on the current configuration: 159.1 ms
busy, 170.6 ms wall, MoE 63.98 ms (40.1%), lm_head 37.72 ms (23.7%), dense q8_0 27.6 ms (17.3%).
The shares are unchanged within a point, which is the useful part - four local optimisations
later, the pass still has the same three levers in the same proportions.

Two numbers in this file are superseded. The "2.34 GB batch-1 / 4.18 GB verify pass" traffic
figures describe only the 41-layer forward; a full speculation pass also runs six 1-column
lm_head calls, which are 2.5 GB on their own, so the pass moves about **8.1 GB** (MoE 3.1,
lm_head 2.9, dense q8_0 2.07) in 170.6 ms = 47 GB/s, 43% of the ceiling. And the MoE rows-per-block
trap noted in the last bullet is now explained: the kernel is pure per-token DRAM streaming, so
an L1 sector cut cannot reach it.
