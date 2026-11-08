# Does 57 GB/s survive at model-scale working sets? (2026-11-04, probe)

Status: prediction recorded before the run. No engine change.

## 1. Why this probe exists

The prefill MoE is 67.7% of prefill at `-ub 32` and it reads at **56.8-59.0 GB/s** against a
ceiling of **109.7-112 GB/s** (findings/2026-11-04-prefill-moe-tile-width.md, section 2g).
Every launch-parameter lever tried against that gap has been inert or negative:

| lever | result |
|---|---|
| tile width J 8 -> 32 (4x the dp4a work, identical bytes) | **-1.49%** prefill |
| shared-memory carveout 100% (doubles SMEM residency if it was 1 block) | **+0.4%** |
| rows per block (decode-side, same kernel family) | neutral |
| wrap B (ub 64/96/128/256/512) | leaves the kernel family entirely, costs acceptance |

Two levers point in opposite directions and both are inert: 4x the ALU costs 1.5%, and doubling
the resident blocks costs 0.4%. That leaves a third explanation that has never been tested:
**the 110 GB/s ceiling was measured at a 143 MB working set, and streaming an 18 GB model may not
reach it.** `bench/gatherbw.cu` allocates a pool of `N_SLOTS 512 x 557056 B = 285 MB` and reads at
most 143 MB of it, repeatedly, so any page-table or DRAM-page effect that only appears when the
whole 18 GB is touched is invisible to it. Every bandwidth number in this repo -- 109.7-112
sequential, 108 scattered, 92.7 for IQ4_XS, 84% of ceiling -- comes from that test.

## 2. Prediction, written before the run

**Predicted: the chunked-read rate falls monotonically with the working-set extent and reads
55-65 GB/s at 18 GB**, i.e. it reproduces the model's own number from a kernel that does no
arithmetic at all. Reasoning: the model's MoE launch moves 115.4 MB in 2.032 ms with 16 resident
256-thread blocks per... no, 2 blocks per SM (34.6 KB of SMEM each against a 96 KB per-SM budget),
which is 128 warps in flight across the part -- ample for MLP -- and yet it lands at 52% of a
ceiling measured on a working set 130x smaller.

**If the prediction holds**, the ub-32 prefill regime is memory-system-limited at model scale and
the honest statement is: no launch parameter, tile width, occupancy target or dot rewrite will
move it, because a pure-load kernel of the same shape does not beat it. The only remaining lever
on prefill is bytes per token, which means `-ub` -- a numerics trade, not a free win.

**If it fails and the rate holds >= 90 GB/s at 18 GB**, then the memory system is not the limit,
the MoE kernel is leaving ~40% on the table in its own tile-load schedule, and the next target is
the `load_tiles -> __syncthreads -> vec_dot -> __syncthreads` serialisation inside
`mul_mat_q_process_tile` (mmq.cuh:923-995), not the memory system.

## 3. Procedure

`bench/tlbstream.cu` (new):

- one allocation of 18 GB, `cudaMemset`-touched so the pages exist;
- kernel A `k_chunk`: grid 1024 x 256 threads -- exactly the MoE's useful block count and block
  size -- each block reading one contiguous **110 KB** chunk (128 rows x 880 B, the IQ3_S gate
  matrix row tile) from a pseudo-random offset inside the used extent, 16-byte vector loads only,
  no arithmetic. The offset table is re-seeded every rep so successive repetitions walk different
  parts of the extent;
- kernel B `k_seq`: contiguous read of the same 115 MB per launch as the control;
- used extent swept over 115 MB / 1 GB / 4 GB / 18 GB, 20 reps each, median of 3 trials.

## 4. Result: the prediction was wrong. The working set is not the limit.

`bench/tlbstream 18 20`, Xavier, no model resident:

| used extent | chunked, MoE shape | contiguous |
|---|---|---|
| 0.115 GB | 90.6 GB/s (1.19 ms) | 101.8 GB/s (1.05 ms) |
| 1.000 GB | 89.9 (1.19) | 101.6 (1.06) |
| 4.000 GB | 91.7 (1.17) | 101.7 (1.06) |
| **18.000 GB** | **92.1 (1.17)** | **102.7 (1.05)** |

The rate is flat in working-set size to within 2.5% across a 157x extent sweep. There is no
page-table or DRAM-page penalty at model scale, and the 109.7-112 GB/s ceiling from `gatherbw` is
real for this chunk geometry too (92.1 GB/s for 1024 scattered 110 KB chunks, 102.7 contiguous).
The MoE's 56.8 GB/s is **38% below what the memory system delivers for its own access shape**, so
it is the kernel, not the platform.

This is the second retracted explanation for the same number this week (the first was "at the DRAM
read ceiling", which came from assuming a fused gate/up launch). Recorded plainly: I predicted
55-65 GB/s at 18 GB and measured 92.1.

## 5. The confound in that probe, and the measurement that resolves it

`k_chunk` has no shared memory and no dot, so it ran at many more warps per SM than `mul_mat_q`
can ever have. `cuobjdump -res-usage` on `libggml-cuda.so` for this model's launch
(`IQ3_S`, `J=8`, `fallback=false`) gives **87 registers/thread**; 256 threads x 87 = 22,272
registers, and `mmq_get_nbytes_shared` gives **34,584 B** of dynamic SMEM. Against 65,536
registers and 96 KB of SMEM per SM that is **2 blocks per SM = 16 warps**, and 3 blocks would need
103.8 KB. So the MoE kernel runs at 16 warps while the probe ran at 30+.

`bench/tlbstream-occ.cu` repeats the probe with dynamic SMEM used purely as a residency control,
at the identical access shape, 18 GB working set, median of 3 x 20 reps:

| warps/SM (256 thr) | GB/s | warps/SM (128 thr) | GB/s |
|---|---|---|---|
| 48+ (96 blocks) | 90.0 | 48+ | 93.7 |
| 32 (12 blocks) | 93.8 | 32 | 91.5 |
| 24 (6 blocks) | 93.6 | 24 | 85.6 |
| 16 (4 blocks) | 90.8 | 16 | 73.3 |
| 12 (3 blocks) | 85.1 | 12 | 62.1 |
| 8 (2 blocks) | 71.8 | 8 | 46.8 |

Reading the curve by **warps per SM**, which is what the scheduler can overlap: 8 -> 46.8-47,
16 -> 71.8-73.3, 24 -> 85.1-85.6, 32 -> 91.5-93.8, and it saturates at ~92.

So the MoE kernel's ceiling at its resident 16 warps is ~72 GB/s, and it achieves 56.8, which is
79% of that. Two separate gaps:

1. **Residency: 16 warps where the memory system wants 32.** The dp4a tile costs 4.3 KB of SMEM per
   warp (65 ints per row x 16 rows per warp), which caps any shape at 96/4.3 = 22 warps. Going to
   the 22-warp cap is worth an estimated +19% on the pure-load rate (72 -> 85 at 24 warps).
2. **Efficiency inside the block: 79% of its own ceiling**, which is the
   `load_tiles -> __syncthreads -> vec_dot -> __syncthreads` serialisation at
   mmq.cuh:956-991, two barriers per K iteration with no double buffering.

Next lever, recorded before it is built: halve the block tile. `I` only decides which rows a block
owns, so `I=64` with 128 threads gives 17.6 KB per block and 5 blocks resident -- 20 warps instead
of 16, plus 3x more block-level concurrency -- and is bit-preserving. See
`findings/2026-11-04-prefill-mmq-tile-i64.md`.

## 6. Reproducing

```sh
cd bench
NVCC=/usr/local/cuda-11.4/bin/nvcc
$NVCC -O3 -ccbin /usr/bin/g++-10.3.0 \
  -D__Float32x4_t=int -D__Float64x2_t=int -D__SVFloat32_t=int -D__SVFloat64_t=int -D__SVBool_t=int \
  -Xcompiler=-isystem,/usr/include/c++/10.3.0 \
  -Xcompiler=-isystem,/usr/include/c++/10.3.0/aarch64-poky-linux \
  '--generate-code=arch=compute_72,code=[compute_72,sm_72]' tlbstream.cu -o tlbstream
./tlbstream 18 20
./tlbstream-occ 18 20
```

The five `-D__*x*_t=int` defines are new: nvcc 11.4 fails on this host's newer glibc with
`identifier "__Float32x4_t" is undefined` from `/usr/include/bits/math-vector-64.h`. They are
dummy scalar stand-ins for the ARM vector typedefs, which this code never calls. Unbuffered stdout
matters: a device-side fault in a probe with buffered printf loses every line printed before it.

