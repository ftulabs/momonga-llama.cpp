# Hardware-wall audit: where the criticism is right and where it is not

Date: 2026-11-03. Prompted by `CRITICISM.md`, an outside review arguing that the project is at
the physical LPDDR4x wall and should stop tuning flags in favour of native Volta kernels.

That review is directionally right about the conclusion (kernel work is the only lever left) and
wrong about almost every supporting fact. This file records which of its claims are testable,
what the machine actually measures, and what the one real target is.

## The premise that decides everything: are we at the wall?

The review says the board "maxes out at 136.5 GB/s" and that "if your math loop is already
sitting at 30+ tokens per second for a 35B model, you are already approaching maximum physical
efficiency".

136.5 GB/s is the theoretical LPDDR4x figure. The measured sequential GPU read ceiling on this
board is **109.7-112 GB/s** (`./2026-11-02-bandwidth-and-flag-audit.md`), and decode does
not come close to it:

| measurement | GB/s | share of 110 |
|---|---|---|
| batch-1 pass, 2.34 GB in ~60 ms | 39 | 35% |
| verify pass, 4.18 GB in ~118 ms | 35 | 32% |
| MoE expert matmul at 6 tokens, opbench | 37-41 | 34-37% |

So the machine is not at the wall. It is at a third of it, which is a fact this repo has
published since 2026-11-02 (the "12% of bandwidth" claim was retracted upward to 32-35%).

## The real limiter: IQ3_S decode is ALU-bound

**Corrected later the same day - see
[2026-11-03-counters-on-xavier.md](2026-11-03-counters-on-xavier.md).** The timing below is
sound and still stands. The *label* is wrong. Hardware counters on the same kernel show
`long_scoreboard` (waiting on a load) as the top stall at 3.05 warps per issue-active cycle,
ahead of `math_pipe_throttle` at 1.16, so the kernel is memory-latency-bound, not ALU-bound. It
is stuck at 34.9% occupancy because 79 registers/thread cap it at 37.5%, which is why it reaches
only 51% of the DRAM ceiling. "ALU-bound" implied no kernel change could help; the occupancy
number implies one might.

New measurement today. opbench, gate/up shape (256 experts, 8 used, K=2048, n=512), 6 tokens,
30 reps, median. Same kernel, same launch geometry, only the weight type changes:

| type | bytes read | time | GB/s |
|---|---|---|---|
| **IQ3_S** (what gate/up actually uses) | 20.62 MB | **0.580 ms** | **37.3** |
| IQ4_XS | 25.50 MB | 0.388 ms | 68.9 |
| Q6_K | 39.38 MB | 0.599 ms | 68.9 |
| Q4_0 | 27.00 MB | 0.418 ms | 67.8 |
| Q8_0 | 51.00 MB | 0.658 ms | 81.3 |

IQ3_S reads **19% fewer bytes** than IQ4_XS and takes **49% longer**. Every classic type reaches
~68-81 GB/s; IQ3_S sits at 37. That is not a bandwidth effect and no amount of prefetching,
tiling or memory-mapping can fix it: the `iq3s_grid` lookup path costs roughly twice per byte
what a plain 4-bit or 8-bit decode costs.

This also explains, in one number, why the flag space looked exhausted. Flags change who runs
the op; they cannot change what the op costs.

## What that is worth if it were fixed

`ffn_moe_gate` and `ffn_moe_up` run once per block, 41 blocks, at IQ3_S. At 6 tokens they cost
0.580 ms each where the same shape at IQ4_XS costs 0.388.

- 82 ops x 0.192 ms = **~15.7 ms of the 164.7 ms pass**, i.e. 9.5% of the pass, is the
  IQ3_S-versus-a-cheap-type decode penalty alone.
- If an IQ3_S kernel reached 60 GB/s, gate/up would take ~0.36 ms and the pass would fall to
  ~147 ms, which at mean_len 5.68 is **~38.6 tok/s**. That is the path to the 40 target.

Tempting, and already closed: **do not requant gate/up to IQ4_XS.** It was built three times
(`TODO.md`, "Requant avenue: CLOSED"). The target model's numerics move, the MTP draft head is
calibrated against the original weights, acceptance collapses from a bit-stable 0.98940 to
0.64-0.93, and throughput is `mean_len / pass`, so the loss swamps the kernel gain. Best build:
30.5 tok/s against a 31.2 control. The third build left block 40 at q3_k to protect the draft
head and it changed nothing - block 40's quant type did not move a single draft decision.

The lever is therefore the **IQ3_S vec_dot itself**, not the model file.

## The rest of the review, claim by claim

| claim | verdict |
|---|---|
| "LPDDR4x maxes at 136.5 GB/s, you are near maximum efficiency" | Theory vs measured 110, and we use a third of it. The problem is ALU, not the bus. |
| "Force-merge the Qwen 3.6 Next-N MTP patch" | Already implemented and tuned. `--spec-type draft-mtp` is the entire difference between 16.9 and 34.5 tok/s. Depth is now 6, not 3 (`2026-11-03-draft-depth-on-the-standard-prompt.md`). |
| "Write hardcoded WMMA tensor-core kernels for Volta" | This is what MMQ already is, and it is measured: forcing `MUL_MAT_ID` onto MMQ is 2x worse at batch 1 and 2.5x worse at 5-6 columns. The GEMM is 6-7 rows tall; a 16x16x16 tile wastes more than half its rows and the quantized operand still has to be decoded first. |
| "Patch ggml-alloc for zero-copy pinned memory past the PCIe boundary" | There is no PCIe boundary. Xavier is an iGPU; `cudaMalloc` already returns the same DRAM the model lives in, and `GGML_CUDA_NO_VMM=ON` is already set. A pinned zero-copy path would be strictly slower. |
| "Async prefetch with `cudaMemcpyAsync` / warp-level async copies" | `cp.async` is an sm_80 feature (PTX 7.0). sm_72 does not have it, so the suggestion cannot be implemented as written. CUDA graphs already remove the launch gaps that prefetching would hide. |
| "Cache the layer weights in shared memory" | A layer is hundreds of MB across experts; SMem is 96 KB per SM on Xavier. Tiling only helps when a tile is re-read, and each expert row is read once per token. |
| "Audit for coalescing; uncoalesced reads lose 80%" | Already measured: scattered vs contiguous expert gathers run at 108 vs 112 GB/s, a 4% difference. Coalescing is not the issue. |
| "Stop tweaking configs, go after .cu files" | Agreed, and it is what happened: the batch cliff turned out to be `mmvq.cu`, and fixing it moved the record from 31.5 to 34.5. The next target is named above. |

## Testable claim in the review, and its answer

The review's audit prompt asks: are you using shared memory, `__syncthreads()`, or `wmma::` in
your kernels? Answer: the MoE kernel uses none of them, by design - one warp per block, no
shared-memory reduction, no MMA. Adding shared memory would not help because there is no reuse
to exploit; adding WMMA is MMQ, which is measured worse at this batch size. The kernel is short
of bandwidth because IQ3_S decode is expensive, and the fix has to reduce instructions per
element, not add staging.

## Why the obvious kernel fix does not exist

The first plan was an int8 `__dp4a` dot for IQ3_S, on the theory that the fast types reach
68-81 GB/s because they use one and IQ3_S does not. **That is wrong: IQ3_S already uses it.**
`get_vec_dot_q_cuda()` in `mmvq.cu` maps `GGML_TYPE_IQ3_S` to `vec_dot_iq3_s_q8_1`, which is a
dp4a path against a q8_1 activation, the same shape as every other type.

What IQ3_S does that the cheap types do not: for every 32 activations it performs **8 dependent
lookups into `iq3s_grid`**, a 512-entry `__device__` table in `ggml-common.h`, then the
`__vcmpne4`/`__vsub4` sign fixups. Q4_0 does none of this - it shifts and masks. The table is
2 KB and lives in global memory, so each lookup is a divergent gather through L1. That is the
entire 37-vs-69 GB/s gap, and it is a property of the 3-bit format, not a missing optimization.

Could the grid go to shared memory? The Vulkan backend does exactly that, but it has real
workgroups. Here a block is one warp handling 2-4 rows, so there is no reuse to amortise a 2 KB
staging cost against. A faithful test would be to stage the grid only when a block handles many
rows, which is the `rows_per_block` knob - and `rows_per_block=4` was measured 0.7% **slower**
end to end today (`2026-11-03-draft-depth-on-the-standard-prompt.md`).

So on this board, with this file, the IQ3_S cost is not reachable from the kernel side. The one
thing that would remove it is a different on-disk type for gate/up, and that is closed by the
acceptance collapse above. That is the honest ceiling of 34.5 tok/s at `n-max 6`: 9.5% above the
previous record, and 14% short of 40.
