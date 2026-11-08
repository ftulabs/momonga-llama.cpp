# MMQ tile geometry for Volta: I=64 with 128 threads (2026-11-04)

Status: prediction recorded before the build and the run.

## 1. Where this comes from

`findings/2026-11-04-prefill-57gbs-probe.md` measured, with a pure-load kernel at the prefill MoE's
own access shape and working set, that read bandwidth is governed by **warps per SM**:

| warps/SM | 8 | 16 | 24 | 32+ |
|---|---|---|---|---|
| GB/s | 46.8-47 | 71.8-73.3 | 85.1-85.6 | 91.5-93.8 |

The MoE launch (`IQ3_S`, `J=8`, `fallback=false`) spends **87 registers/thread and 34,584 B of
dynamic SMEM**, so it runs at 2 blocks x 8 warps = **16 warps per SM** and is capped near
72 GB/s; it achieves 56.8, i.e. 79% of that. Three blocks would need 103.8 KB against 96 KB.

The dp4a tile costs 65 ints per row and 16 rows per warp = 4.3 KB of SMEM per warp, so no tile
shape can exceed 96/4.3 = 22 warps. The block tile therefore has to shrink for residency to rise.

## 2. The change

New `ggml/src/ggml-cuda/mmq-config-volta.cuh`, selected for NVIDIA GPUs whose highest compiled
arch is below Turing (i.e. this sm_72 build) at both the host and device dispatch sites in
`mmq.cuh`. For `IQ3_S` and `IQ4_XS` at `J=8` it returns

```
nthreads 128, occupancy 5, I 64, J 8, SRAM_LAYOUT_Q8_0, K_vram MMQ_ITER_K, stream_k true
```

instead of the Ampere row's `nthreads 256, occupancy 1, I 128`. Everything else -- every other
type, every other J, all dense launches -- delegates to the Ampere table unchanged.

- SMEM per block: `65*64*4 + ids 32 + pad(y 288 -> 512)` = **17,696 B**, so 5 blocks per SM by
  SMEM (88.5 KB), 5 by registers (87 x 128 = 11,136; 65,536/11,136 = 5.88).
- `__launch_bounds__(128, 5)` caps registers at 102, above the current 87, so the compiler should
  not spill; if `cuobjdump -res-usage` shows more than 87 registers or any STACK bytes, the arm is
  void and gets rebuilt.
- **Bit-preserving by construction:** `I` decides which rows of src0 a block owns and nothing else.
  A row's K sum is one thread's serial accumulation either way, and `write_back` per row is
  unchanged. The cheap check that nothing moved numerically is acceptance **0.96657** and
  mean len **5.68**, which must come back identical.

## 3. Prediction, written before the run

- Warps/SM goes 16 -> 20 (5 blocks x 4 warps).
- Pure-load rate at 20 warps interpolates to **76-79 GB/s** (16 -> 72, 24 -> 85).
- Kernel efficiency stays at its measured 79% of the pure-load ceiling, so the MoE launches read
  **60-63 GB/s** against today's 56.8-59.0, i.e. **+5 to +10% on the launch**.
- The MoE is 67.7% of prefill at `-ub 32`, so **prefill: +3.5 to +7%**. At 9,563 tokens that is
  79.5 -> **82-85 tok/s** and TTFT 120.2 -> **113-117 s**. At 590 and 2,385 tokens the same
  relative gain, 87.1 -> 90-93 and 84.3 -> 87-90.
- Decode must be **unchanged bit for bit** (36.9 tok/s, acceptance 0.96657, mean len 5.68). Decode
  is MMVQ, and this table is only read by `mul_mat_q`, so any movement at all is a bug in the change
  and not a trade to weigh.

Written down because the last prediction about this number was wrong by 3x in the other direction:
the tile-width change was predicted at +33-58% and delivered +9-12%, and the working-set probe
predicted 55-65 GB/s and measured 92.1. This one is deliberately narrow, and it is falsified if
prefill moves less than 2%, which is the floor measured at 9.5k with six arms.

## 4. Protocol

Two binaries, both built from this tree, alternating in both orders within one session so the
board's drift cannot flip the sign (the A/A floor for decode is 0.71% and per-position drift was
+0.95% in the 2026-11-04 flag re-test; the prefill floor at 9.5k is 0.56%):

- **A** = `build-volta` as it stands (Ampere table, I=128).
- **B** = same tree with `mmq-config-volta.cuh` in place (I=64).

Both are copied out so the arm can be re-run without a rebuild. Each arm, with
`DECODE_REPS=5`, `LENS="590 2385 9563"`, `UB=32`, 3 reps:

```
NAME=i128-a DECODE_REPS=5 LENS="590 2385 9563" ./prefill.sh   (A)
NAME=i64-a  DECODE_REPS=5 LENS="590 2385 9563" ./prefill.sh   (B)
NAME=i64-b  DECODE_REPS=5 LENS="590 2385 9563" ./prefill.sh   (B)
NAME=i128-b DECODE_REPS=5 LENS="590 2385 9563" ./prefill.sh   (A)
```

Acceptance and mean length from `bench/results.csv` are checked before any timing is looked at.

## 5. Result: falsified. The tile geometry is a wash, and the residency lever is closed.

`bench/ab-i64.sh`, 2026-11-05 00:07-00:59, four arms in order A B B A, `bench/prefill-i64.csv`,
3 reps per length, medians:

| len_req (n_prompt) | A: I=128 | B: I=64 | delta |
|---|---|---|---|
| 590 (703) | 97.20 (arm 1) | 97.80 (arm 2) | |
| | 104.05 (arm 4) | 103.22 (arm 3) | |
| | **pooled 100.62** | **pooled 100.51** | **-0.11%** |
| 2385 (2828) | 97.09 (arm 1) | 97.80 (arm 2) | |
| | 98.88 (arm 4) | 98.12 (arm 3) | |
| | **pooled 97.99** | **pooled 97.96** | **-0.03%** |
| 9563 (11484) | 90.90 (arm 1) | 90.83 (arm 2) | |
| | 90.38 (arm 4) | 89.94 (arm 3) | |
| | **pooled 90.64** | **pooled 90.39** | **-0.28%** |

Prediction was +3.5 to +7% and prefill did not move at any length: -0.11%, -0.03%, -0.28%, all
inside the protocol floor. **Not adopted.** `mmq-config-volta.cuh` is kept in the tree but the two
rows at J=8 are removed so the Volta path is currently identical to the Ampere table; the file
stays as the hook for the next attempt at this kernel.

Decode is untouched, as designed and as required: `bench/results-i64.csv` medians are 36.47/35.88
(A) and 36.53/36.49 (B) tok/s with acceptance **0.96657** and mean len **5.68** in every one of
the 20 reps, the same five per-rep acceptance values (0.96657, 0.97248, 0.97554, 0.97248,
0.96657) in every arm. The change is bit-preserving, exactly as argument said it would be.

### Why the warps curve did not transfer

The probe measured a *pure-load* kernel: 16 warps 72 GB/s, 20-24 warps 85. The real kernel gains
nothing from 16 -> 20 warps, so its limiter is not memory-level parallelism. Per byte of DRAM read
it also does a dequantise, a shared-memory store of the expanded int8 tile (14 KB read -> 33 KB at
I=128) and two barriers per K iteration, all of which scale with warps on the same LSU/SMEM path.
This is the same signature as the decode-side dp4a kernels, where `lg_throttle` - the L1 refusing
load instructions - is the top stall while occupancy has headroom (findings/2026-11-04-lmhead-1col-occupancy.md).
Adding warps to a pipe that is already issue-saturated buys nothing, which is what 16 -> 20 warps
delivered: 0.

**Closed:** I/J/nthreads/carveout are all measured inert on this launch. What is left in the MoE
is the dp4a tile algorithm itself - dequant-to-SMEM + two barriers per K step, with 8 columns
computed where the model uses 1 - and that is a kernel rewrite, not a parameter.

## 6. Side result that matters more: the same binary reads 92.0 tok/s tonight and 79.5 in the record

Arm A is a byte-identical copy of the binary that produced the recorded baseline. Re-measured at
the record's own point (`LENS=8192`, i.e. n_prompt 9563), tonight it reads **91.82 / 92.09 tok/s
(103.9-104.2 s TTFT)** against the recorded **79.53 / 79.53 / 79.40 (120.3 s)** in
`bench/prefill.csv` (`h3-adopted-np400`). Same flags, same prompt, same prompt length, same
session-local reproducibility (+-0.2%): **a 15.7% between-session difference with no code
difference.** At 590 and 2,385 tokens the same comparison is +12% and +15%.

Decode went the other way over the same interval: 35.88-36.53 tok/s now against 36.93 recorded, so
this is not a board clock that got faster.

Hypothesis, written before testing: `prefill.sh` runs `--load-mode none`, i.e. the 18.12 GB model
is mmap'd and paged in lazily, and prefill streams all of it every 32 tokens. If the file's pages
are not resident, every chunk faults weights back from the SD card; 15% eviction at ~180 MB/s is
about 15 s of a 120 s prefill, which is the size of the gap. The 2026-11-04 session may have been
running with the boot `llama-server.service` resident (24.5 GB), which cannot be reconciled with
an 18 GB working set in 30.9 GB of RAM, so the page cache thrashed on every rep - and indeed all
three recorded reps read 79.4-79.5, not a cold first rep followed by fast ones.

Test: drop the page cache and measure one rep (see
`findings/2026-11-05-prefill-page-cache.md`). If it reproduces ~79 tok/s the mechanism is
confirmed and the practical consequence is large: the documented TTFT depends on whether the
weights are resident, and a warm-up request or an eager load moves that cost out of the user's
first token.
