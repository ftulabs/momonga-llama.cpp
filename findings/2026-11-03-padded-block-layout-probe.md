# Sector-aligned block layout: measured, and not pursued

2026-11-03. A negative result with a confound on it, written up so nobody re-derives it. The
remaining bit-preserving lever in this project was a **sector-aligned padded block layout**:
`block_q8_0` is 34 bytes and `block_q6_K` 210, so no block ever starts on a 32-byte sector
boundary, and both kernels show heavy L1 sector amplification (15.6x on the dense 7-column q8_0
shape, 55.8 sectors per 210-byte block on the q6_K lm_head).

## What was done

Rather than build a new ggml type and repack the GGUF to test the premise, `block_q8_0` was
padded 34 -> 64 bytes and `block_q6_K` 210 -> 224 by adding a pad array inside the struct.
`sizeof` propagates through `ggml_type_traits` and `ggml_cuda_type_traits`, so every kernel and
every host-side size calculation picks up the new stride with no other change. 224 = 7 sectors
and 64 = 2, so every block starts exactly on a sector boundary. `opbench` synthesizes its own
tensors, so this measures the kernel; it breaks real GGUF loading, and it was reverted.

## Result

`opbench`, K=2048, one rep set each:

| shape | 34/210-byte blocks | 64/224-byte blocks | delta |
|---|---:|---:|---:|
| q6_k n=248320 tok=1 | 4.552 ms | 5.738 ms | **+26%** |
| q8_0 n=8192 tok=1 | 0.272 ms | 0.322 ms | **+18%** |
| q8_0 n=8192 tok=7 | - | 0.442 ms | - |

The prediction, written before the run, was 3.9-4.3 ms for the first row (-25% to -35% on L1
sectors). It was wrong, and it was wrong in a way the counters explain.

`ncu` on the padded q6_K 1-column kernel against the numbers already measured on the baseline:

| | baseline | padded |
|---|---:|---:|
| kernel time | 4.55 ms (opbench) | 4.83 ms (ncu, inflated) |
| L1 load sectors | 127,155,108 | **80,738,354 (-37%)** |
| L1 load requests | - | 12,416,000 |
| DRAM read | 417 MB | 417 MB (identical) |
| registers/thread | 48 | **95** |
| occupancy | 57% | **29%** |

Two things, and only one of them is about the layout.

1. **The sector hypothesis was partly right**: aligning the blocks cuts L1 load sectors by 37%
   at 1 column. DRAM traffic is unchanged, which is what alignment should do - it redistributes
   requests, it does not change the bytes that must be fetched.
2. **But the probe is confounded**: the padded struct doubles the register count, 48 -> 95, and
   halves occupancy, 57% -> 29%. The compiler's addressing for a 224-byte stride is not the same
   code as for 210. So this measurement cannot say whether the 37% sector cut would have paid if
   occupancy had been held constant, and it is not evidence that sectors are irrelevant.

## Why it stops here anyway

- The two effects are opposed and neither is large. Even a clean 37% sector cut on a kernel that
  is 80 GB/s of a 111 GB/s ceiling has to be worth less than the implementation.
- The register regression would have to be solved as part of any real version. This is the same
  trap the dense `rows_per_block` work hit twice: a traffic win that arrives with an occupancy
  loss measures as nothing, and the fix (`__launch_bounds__`) is another experiment on top.
- The file cost is real and asymmetric. From the model's own tensor inventory: Q8_0 is 2.07 GB
  in 259 tensors and would become 3.90 GB (+1.83 GB, +88%, because 34 of 64 bytes are padding
  that is never read but is always fetched by the sector); Q6_K is 1.08 GB in 4 tensors and would
  become 1.15 GB (+0.07 GB). The model would go from 18.11 GB to 20.01 GB on a 31 GB unified
  memory budget.
- It needs a new ggml type, a CUDA traits specialization, kernel instantiations, a repack path
  and a loader, none of which upstream would take, for a lever the earlier analysis had already
  estimated at ~+2% for q8_0 alone.

**Closed.** If someone wants to revisit it, the clean experiment is the same padded struct with
`__launch_bounds__` pinned so the padded build lands at the same register count and occupancy as
the baseline; without that control the measurement cannot separate the sector win from the
occupancy loss.

## Addendum: the same probe also shows what padding costs in DRAM

For q8_0 the padded tensor is 32.00 MB against 17.00 MB of weights, and the kernel reads 34 of
every 64 bytes - but it fetches 2 sectors per block, so it touches all 32 MB. Padding a type
whose used bytes do not fill the sector pair doubles its DRAM read. q6_K wastes only 6.7%
(210 of 224) which is why its DRAM is unchanged. Any future padding should target a stride whose
used bytes occupy whole sectors - which for these two types means changing the block contents,
not adding a gap.
