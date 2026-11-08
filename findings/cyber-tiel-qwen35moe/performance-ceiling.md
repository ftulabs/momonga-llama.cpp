# Performance ceiling: why 45-60 tok/s needs kernel work

**Date:** 2026-11-01
**Config measured:** the best setup, 31.52 tok/s (draft 3, `p-min 0.5`, `-ub 32 -b 32`, q8_0 KV)

## Telemetry during sustained decode

Sampled every 200 ms from `/sys/devices/gpu.0/load`,
`/sys/devices/platform/17000000.gv11b/devfreq/17000000.gv11b/cur_freq`, and (via sudo)
`/sys/kernel/debug/bpmp/debug/clk/emc/rate`, during a 400-token decode:

| Metric | Value |
|---|---|
| GPU load | **median 992 / 1000 (~99%)**, max 997 |
| GPU clock | 1377 MHz, pinned at maximum |
| EMC clock | 2133 MHz, pinned at maximum |

Conclusions:

- The GPU is **saturated**, not idle waiting on launch overhead. CUDA graphs are confirmed
  working (`graphs reused`, `realloc = 0`).
- Clocks are already at maximum. `nvpmodel` is not installed on this system, so there is no clock
  lever left. `jetson_clocks` did not measurably change throughput when tried.

## The arithmetic of the remaining gap

At 31.52 tok/s with mean accepted length 3.69, the engine runs **8.5 forward passes/s**, about
**118 ms per pass**, each pass carrying roughly 4.7 tokens (1 anchor + 3.69 accepted).

A plain batch-1 pass costs about 60 ms (16.64 tok/s with no speculation).

To reach 45 tok/s at the same tokens per pass the engine needs 11.9 passes/s, i.e. **~84 ms per
pass** - a **1.4x kernel-level improvement** with no change in acceptance.

## It is not a bandwidth problem for this model

Cyber-Tiel has ~3B active parameters at 4.25 bpw, so roughly **1.6 GB of weights per pass**.
At 8.5 passes/s that is only ~14 GB/s against a ~137 GB/s peak, while the SMs are 99% busy.

So the MoE decode path is **latency- and access-pattern-bound**, not bandwidth-bound: expert
gathering issues small scattered reads and the SMs end up waiting rather than streaming.

The contrast with a dense model on the same board confirms this. Spark 4B Q8_0 reads 4.37 GB per
token at 16.46 tok/s = **~72 GB/s (52% of peak)** - a genuinely bandwidth-bound workload, and it
gets much closer to the roofline. See
[2026-11-01-model-benchmarks.md](2026-11-01-model-benchmarks.md).

## Roofline caveats

Earlier documents in this project quoted two idealized figures. Both are illustrative only:

- **86 tok/s** - 3B active params x 4.25 bits, divided into nominal bandwidth.
- **69.4 tok/s** - 1.975 GB/pass at 137 GB/s nominal peak.

Both assume expert slices are accessed independently and that *all* peak bandwidth serves
weights. They ignore KV/SSM traffic, routing, kernel efficiency, draft and verification costs.
Measured reality is that bandwidth utilisation on the MoE is far below peak precisely because the
access pattern is scattered. **60 tok/s was never established by dividing RAM bandwidth by active
parameter count.**

## What would actually close the gap

All of this is `ggml-cuda` kernel work, not configuration:

1. **MoE expert gather path** - the index sort and dispatch. Fewer, larger, better-coalesced
   loads; avoid re-reading expert pointers per token.
2. **`mul_mat_vec_q` for IQ4_XS on sm_72** - Volta has no `cp.async` and no `ldmatrix`, so it
   falls back to older paths. IQ4_XS dequant is lookup-table based, unlike K-quants.
3. **A dedicated batch-about-5 path** - at depth 3 the verification batch is ~4.7 tokens, which
   is neither batch-1 decode nor prefill, so it hits neither tuned kernel.
4. **Reducing per-pass launch count** under CUDA graphs.

Returning to configuration: the GPU is at 99% utilization at maximum clocks with graphs active.
There is no flag, cache type, thread count, or batch size left that changes the fundamental cost
per pass. **45-60 tok/s is not reachable by configuration on this board.**

## Update 2026-11-02: the roofline argument is now a measurement

The arithmetic above used the nominal 137 GB/s and an assumed 1.6 GB/pass. Both have since been
measured directly, and the conclusion holds more strongly than it did on paper:

- A `sm_72` GPU read kernel on this board reaches **109.7 GB/s**, not 137.
- Decode effective traffic is **~13 GB/s, about 12% of that ceiling**, with SMs at 99% busy.
- Two concurrent streams produce **24.64 tok/s aggregate against 31.20 for a single stream**.
  Adding a second stream makes the machine slower, which is impossible if decode were already
  maxing the memory system.

So the "not bandwidth-bound" claim no longer rests on dividing by a spec sheet. See
[2026-11-02-bandwidth-and-flag-audit.md](2026-11-02-bandwidth-and-flag-audit.md), which also
closes out the remaining untested flags and upstream PRs.
