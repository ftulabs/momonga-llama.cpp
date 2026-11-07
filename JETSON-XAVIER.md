# Jetson AGX Xavier (Volta sm_72) local changes

This fork carries CUDA changes on top of `ggml-org/llama.cpp` that exist to make one model decode
faster on one board: Jetson AGX Xavier, Volta sm_72, CUDA 11.4. They are not proposed upstream -
they are either board-specific or debug scaffolding - and the numbers below come from a single
machine. Current record on this board: **36.63 tok/s** at `--spec-draft-n-max 6`, **36.93** with
`-bs` (`--backend-sampling`, a host-side +1.04%), up from 31.50 before any of it. The last kernel
change in the ladder is the non-contiguous `concat` launch geometry (non-kernel, no numerics),
worth +0.29%; the kernel changes below take it from 31.50 to 36.56.
Without the MTP drafter the same build reads **27.86 tok/s plain decode** with the
`all-q4k` file (23.75 at 8.8k of context) and, because `-ub` no longer touches decode,
**215.8 tok/s prefill / 53.3 s TTFT at 11,484 tokens with `-ub 2048`** - against 83.8 and 137.1 s
at the MTP configuration's `-ub 32`. The DFlash sidecar at depth 3 reads 23.18 on 8.8k of code. `-ub 2048`
is the widest width that measures reproducibly; 3072 and 4096 scatter by 32-72% over identical
loads. That is a separate profile, not a replacement: re-attaching MTP to the retyped file loads
and drafts but drops acceptance to 0.64583 (mean len 2.10), reading 23.35 tok/s against 27.86
plain.

    git log --oneline upstream/master..HEAD

## 1. `cuda : use MMVQ for MUL_MAT_ID on sm_72 and drop the per-type batch cap`

`get_mmvq_mmid_max_batch()` tested the compute capability with equality against
`GGML_CUDA_CC_VOLTA` (700). Xavier reports 720, so the test failed and the per-type cap fell
through to the `pascal_older` table. Above that cap `MUL_MAT_ID` silently switched from the MMVQ
MoE kernel to MMQ, which is roughly 2.5x slower at these batch sizes - and speculative decoding
with a multi-token-prediction head runs at batches 3-7, exactly where the cliff sat.

Two changes:

- compare the capability as a range, `cc >= GGML_CUDA_CC_VOLTA && cc < GGML_CUDA_CC_TURING`, so
  sm_72 (and sm_73) select the Volta row;
- put the token index on `blockIdx.z` instead of `blockDim.y`, so a block is always one warp and
  any batch is a legal launch. The old layout needed `__launch_bounds__` sized to the cap, which
  is why the cap could not simply be raised: anything above the table gave
  `CUDA error: invalid argument`.

Measured on an 18 GiB IQ3_S MoE model (35B total, 3B active), 400 generated tokens, median of 5
runs: gate/up at 5 tokens went from 0.884 ms to 0.410 ms, and the model from 31.50 tok/s at
`--spec-draft-n-max 3` to 34.49 at `--spec-draft-n-max 6` (draft acceptance 0.989 -> 0.967).
The patch alone changes nothing at batch 4, where both paths were already legal; its value is
that deeper drafting stops being a penalty.

Re-measured on this base (`2cbca38`) after the rebase, same harness, 5 reps: 34.30 tok/s
(33.34 / 34.14 / 34.36 / 34.30 / 34.30), acceptance 0.96657, mean length 5.68. That is 0.5%
below the 34.49 measured on `dde879b`, which 5 reps cannot resolve from noise.

The same file also reads `GGML_CUDA_MOE_RPB` (rows per block in the MoE kernel) and
`GGML_CUDA_SMALLK`, both experiment hooks that default to the upstream behaviour. The
rows-per-block hook is for the MoE kernel only; the dense-path row count below is compiled in
and is not switchable.

## 2. `cuda : run the IQ3_S MoE kernel at 64 registers instead of 79`

`mul_mat_vec_q_moe` declared `__launch_bounds__(256, 1)` while the kernel is launched with 32
threads, so the second argument was inert and the compiler spent 79 registers. That capped
residency at 24 one-warp blocks per SM, or 37.5% occupancy. Declaring the real block size and
32 blocks per SM - for IQ3_S only, because a blanket 64-register budget makes Q6_K spill -
gives 64 registers and 47.9% occupancy. Kernel time falls 7.6% and the model goes 34.30 ->
35.07 tok/s (+2.25%) with acceptance unchanged at 0.96657 and mean length 5.68.

32 blocks per SM is the sm_72 maximum and all four occupancy limits (blocks, registers, shared,
warps) now sit at 32, so this kernel is finished for a one-warp block. Measured in situ
afterwards, its cost is exactly linear in token count - 0.1392 ms per token at 7 columns for
gate/up, 0.0818 for down, intercept ~0.006 ms - because each token routes to its own eight
experts, so it is a DRAM stream at 44-52 GB/s with no launch overhead and nothing to reuse
across tokens. Launch-geometry levers in this kernel (including `GGML_CUDA_MOE_RPB`) measure
changes in L1 sectors but none in the model; the remaining headroom is memory-level parallelism
per warp, not occupancy.

## 3. `cuda : four rows per block and a wider small_k trigger on the dense MMVQ path`

Four changes to `mmvq.cu` that only move work assignment and block geometry, so none of them
changes a number the model computes - acceptance 0.96657 and mean length 5.68 are identical
before and after, which is the check.

The problem: at wide column counts the 2 KB activation vector `y` is re-read once per output
row, and that re-read is 57% of the L1 load sectors at 7 columns. Four rows per block halves
it, but needs the shared-memory carveout raised from the driver's 16.38 KB default to 64 KB
(otherwise residency caps at 4 blocks and the win inverts, which is what an earlier attempt
measured as +46% in a microbenchmark and -0.7% at the model) and needs
`__launch_bounds__(.., 8)` at those column counts (otherwise 4 rows take q6_K to 158 registers
and 6 blocks, eating the whole gain). The fourth change takes the wider `small_k` layout
whenever K is an exact multiple of one block iteration, not only when K is shorter than one.

The same rows-per-block lever applies at one column, which is the 1-column Q6_K lm_head that
drafts all six speculative steps and is 18.4% of a pass: 4.776 -> 4.552 ms on that kernel.
Together: 35.07 -> 36.25 -> **36.56 tok/s**. L1 load sectors fall 24% at 1 column and 38% at 7
at unchanged occupancy, so this is a traffic win rather than a parallelism win.

It regresses the n=512 shape by 3.4% (64 blocks leave no room for 4 rows) and the dense q8_0
n=1024 shape by 3.3% at 1 column; both are a fraction of a millisecond per pass.

## 4. `cuda : opt-in per-op timing dump behind GGML_CUDA_OP_TIME`

`nsys` and `ncu` are not part of the stock image, so attributing decode time to individual nodes
otherwise means timing them by hand. Setting `GGML_CUDA_OP_TIME` records a CUDA event pair per
node, disables graph capture, and every fifth graph prints the accumulated totals (top 16,
aggregated by node name with the layer suffix stripped) to stderr.

It costs real time and is off by default. Use it for shares, not for absolute tok/s.

If `nsys`/`ncu` are installed (`nsight-systems-cli` and `nsight-compute-cli` in the
`momonga-extra` RPM feed, headers from `cuda-cupti-dev`), prefer them: both profile
graph-launched kernels, which this hook cannot do because it disables the graphs. CUPTI requires
root on this board - unprivileged calls return `CUPTI_ERROR_INSUFFICIENT_PRIVILEGES`.

## 5. `cuda : pick the MoE prefill tile width from tokens per expert`

`mmq.cu` chooses the MMQ tile width `J` from `ncols_opt = ne12` for every multiplicand, but a
`mul_mat_id` launch with 32 tokens over 256 experts and 8 routed experts gives each expert about
**one** useful column. The kernel still computes a full `J`-column tile: `vec_dot` has no column
bound and only the write-back honours `tile_y_max_j`, so 31 of 32 columns' worth of dp4a work is
discarded. The code already had the right rule for RDNA3/RDNA4 - pick `J` from the average tokens
per expert, `ceil(ne12*n_expert_used/ne02)` - and this patch applies it on NVIDIA as well.

Tightening the tile is bit-preserving: `J` decides which thread owns an output element, not the
order of its K sum, so acceptance and mean draft length come back identical. It is on by default;
`GGML_CUDA_MMQ_MOE_TILE_AVG=0` restores the old choice and `=2` is a negative control (IQ3_S only).

Measured on this board, `-ub 32 -b 32`, prompt not cached, three lengths, one server load per arm
with decode measured in the same load:

| prompt | before | after | TTFT before | TTFT after |
|---|---|---|---|---|
| 590 tokens | 78.8 tok/s | 87.1 | 7,491 ms | 6,776 ms |
| 2,385 tokens | 75.8 | 84.3 | 31,468 ms | 28,307 ms |
| 9,563 tokens | 71.0 | 79.5 | 134,638 ms | 120,231 ms |

Decode is untouched (36.4-36.5 tok/s without `-bs`, acceptance 0.96657, mean length 5.68) because
MMQ is not on the decode path. The gain is only 12% and not the 4x the removed work would suggest:
the launch reads at only about 57 GB/s against this board's 110 GB/s ceiling - `gate` and `up` are
separate 115 MB launches, not one fused one - so neither the dot nor the bandwidth was saturated.

## Build

Run `./build-best.sh` to configure and build all production targets (`llama-server`, `llama-cli`, `llama-quantize`, `llama-tokengate`) with verified flags, or configure manually:

```sh
cmake -S . -B build-volta -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=72 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.4/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10.3.0 \
  -DCMAKE_CUDA_FLAGS="-Xcompiler=-isystem,/usr/include/c++/10.3.0,-isystem,/usr/include/c++/10.3.0/aarch64-poky-linux,-isystem,/usr/include/cuda-compat-workarounds" \
  -DGGML_CUDA_NO_VMM=ON -DGGML_CUDA_NO_PEER_COPY=ON \
  -DGGML_CUDA_FA=ON -DGGML_CUDA_GRAPHS=ON -DGGML_NATIVE=ON \
  -DGGML_LTO=OFF -DLLAMA_BUILD_TESTS=OFF
```

`CMAKE_CUDA_HOST_COMPILER` and the `-isystem` flags are both required: nvcc as shipped with CUDA
11.4 rejects the system gcc on this image and cannot find the C++ headers without them. Do not
pass `CMAKE_CUDA_FLAGS` alone on the command line from a build that already has these flags - it
replaces them.
