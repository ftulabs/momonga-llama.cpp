# Two cheap probes at the MUL_MAT_ID batch cliff: both negative

Written 2026-11-02. `TODO.md` listed two surviving explanations for the type-specific
batch-size cliff at `MUL_MAT_ID`: the **expert-id distribution**, or the **block shape
`(warp_size, ncols_dst)` / warp-per-column occupancy**. Both were testable without new
instrumentation. Both are now ruled out, or close to it.

Baseline being explained, `bench/opbench 20`, gate/up shape (K=2048, n=512, 8 of 256 experts):

| tokens | IQ3_S | IQ3_S / 1-token | IQ4_XS | IQ4_XS / 1-token |
|---|---|---|---|---|
| 1 | 0.194 ms | 1.0x | 0.171 ms | 1.0x |
| 4 | 0.341 ms | 1.8x | 0.281 ms | 1.6x |
| 5 | **0.867 ms** | **4.5x** | 0.332 ms | 1.9x |
| 6 | 0.997 ms | 5.1x | **0.885 ms** | **5.2x** |

## Probe 1: the expert-id distribution. RULED OUT.

`opbench` reroutes the expert ids before every timed rep by default (`OPBENCH_ROTATE=1`).
Setting `OPBENCH_ROTATE=0` freezes them, so every rep re-reads the same ~20 MB of expert
matrices and L2 serves them. If the cliff were an effect of *which* experts get selected, or of
the scatter pattern, freezing the ids should change it.

| transition | ids rotated | ids frozen | ratio |
|---|---|---|---|
| IQ3_S 4 -> 5 | 0.341 -> 0.874 (2.56x) | 0.321 -> 0.809 (2.52x) | same |
| IQ4_XS 5 -> 6 | 0.331 -> 0.892 (2.69x) | 0.308 -> 0.789 (2.56x) | same |

Frozen ids are uniformly ~7% faster (the expected cache-warm effect), and the cliff is
**unchanged in size and position**. The expert-id distribution is not the mechanism.

## Probe 2: the block's token dimension. NEGATIVE, and not fully captured.

The MoE kernel is launched `dim3 block_dims(warp_size, ncols_dst)` - one warp per token - with
`if (token_idx >= ncols_dst) return;` already in the kernel (`mmvq.cu`, `mul_mat_vec_q_moe`).
So the block can be padded to a fixed warp count with no change in work: the extra warps exit
immediately. That isolates block shape from everything else.

Hook added: `GGML_CUDA_MMID_PAD_BLOCK=<n>` in `mul_mat_vec_q_moe_launch` sets
`block_dims(warp_size, max(ncols_dst, n))`. `MMVQ_MAX_BATCH_SIZE` is 8, so 8 is legal under the
kernel's `__launch_bounds__`.

Result: **padding IQ3_S to 8 warps at 5 tokens gave 0.867 ms against 0.878 ms stock - no
change.** Making the block 8 warps wide with 5 active does not relieve the cliff, so "the block
happens to have 5 warps" is not the mechanism.

The full padded curve could not be captured: with the hook active, `opbench` aborts with
`CUDA error: invalid argument` at `ggml_cuda_kernel_launch`, after the padded gate/up
measurement and before the batch sweep. Stock `opbench` completes 73 measurements with no error
on the same build, so it is specific to the changed launch geometry, most likely CUDA graph node
reuse with altered block dimensions rather than the kernel itself. **The hook was reverted**;
`git diff ggml/src/ggml-cuda/mmvq.cu` now shows only the two pre-existing hooks
(`GGML_CUDA_MMID_MAX_BATCH`, `GGML_CUDA_SMALLK`).

Treat probe 2 as a negative data point at 5 tokens, not as a complete curve. Anyone repeating it
should expect the graph-capture error and either disable graphs for the measurement
(`ggml_backend_cuda_graph_compute` is in the backtrace) or capture each batch size separately.

## What is left

The cliff is real, reproducible, type-specific, and now known not to be:

- warp count (`calc_nwarps` forced, 2026-11-02)
- register spills (`cuobjdump -res-usage`: `STACK:0 LOCAL:0`)
- template specialization on `ncols_dst` (the MoE kernel takes it at runtime)
- the expert-id distribution or scatter pattern (this file, probe 1)
- the block's warp count / occupancy shape at 5 tokens (this file, probe 2)

Every cheap hypothesis is now gone. What remains needs instrumentation this board does not ship
- no `nsys`, no `ncu` - so the next step is a hand-built probe (a kernel variant that records
its own timing or occupancy) rather than another launch-parameter tweak. `TODO.md` still
estimates ~38-40 tok/s if the cliff were removed, but that estimate has survived two failed
escape routes and should be discounted.

## Reproducing

```sh
cd bench
./opbench 20                     # ids rotated (default)
OPBENCH_ROTATE=0 ./opbench 20    # ids frozen
```
