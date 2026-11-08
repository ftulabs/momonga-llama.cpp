# R2: the 1-column q6_K lm_head is at 79% of DRAM, not at an occupancy wall (2026-11-04)

Question (TODO R2): the 1-column `q6_K` lm_head - the kernel that drafts all six MTP steps -
was 28.05 ms of a 156.8 ms pass at 89.4 GB/s against a ~110 GB/s ceiling, at 48-95 registers
per thread with `calc_min_blocks_per_sm(ncols_dst) { return ncols_dst >= 5 ? 8 : 1; }`, i.e.
nothing pins registers below 5 columns. The cheap hypothesis was that the same lever class that
paid +3.4% on the 7-column path (rows per block, carveout, launch bounds) would pay here too.

## What ncu says

Measured at the model's exact geometry via `opbench` single mode (K=2048, n=248320, tok=1,
`OPBENCH_TYPE=q6_k OPBENCH_NE=1`), `ncu --kernel-name-base demangled -k regex:mul_mat_vec_q
-c 1`, on kernel `mul_mat_vec_q<(ggml_type)14, (int)1, (bool)0, (bool)1, (bool)0>`:

| metric | value |
|---|---|
| duration | 4.54 ms (plain opbench timing 4.667 ms, 89.4 GB/s) |
| SM throughput | 37.73% |
| memory throughput (SOL) | **78.95%** |
| DRAM read | 397.85 MB (the tensor is 417 MB; 95% of it) |
| L1/TEX throughput | 52.04% |
| L1 hit rate | 85.65% |
| L2 hit rate | 1.65% |
| warps per issued instruction | 12.52 cycles, of which **4.7 (37.5%) on `lg_throttle`** (LSU issue queue full) |
| theoretical occupancy | 31.25% (20 warps = 10 blocks x 2 warps) |
| achieved occupancy | 29.28% (18.74 warps/SM) |
| block limit: registers / SM / shared | 10 / 32 / 16 |

So the kernel *is* register-limited (10 blocks) and unpinned at one column, exactly as R2
predicted - and that is not the thing to fix:

- It already reads at 78.95% of the memory SOL. The absolute headroom is 4.54 -> 3.59 ms, and
  a DRAM-streaming GEMV at 79% of peak with 29% occupancy is close to the practical shape of
  this kernel on Volta.
- The dominant stall is `lg_throttle`, not `long_scoreboard`. Warps are not waiting for data to
  arrive; they are waiting for the L1 to accept load instructions. The kernel is already
  over-submitting the load pipe - the remaining L1 traffic is the 34-byte `block_q6_K` stride
  (`l1tex` sectors per request) plus the activation re-read. Raising occupancy adds more
  warps issuing more loads into a queue that is already full: it would raise the L1 hit rate
  and the throttle count, not the DRAM rate.
- The two levers that would cut that traffic are both already spent or measured inert here:
  rows per block is 4 at one column (the `small_k` path, the +0.85% change of 2026-11-03), and
  8 rows needs 8x1 accumulators plus addressing at 64 threads, which spills.

## Verdict

Closed as a target. The 28.05 ms is 79% of DRAM with the L1 issue rate as the binding
constraint; a 16-block register pin at one column would be tuning the constraint that is
already saturated. Not built, not benchmarked at the model, because the kernel needs a
traffic reduction - which is a different type's block layout, i.e. out of bounds
(`./2026-11-03-padded-block-layout-probe.md` measured the padded layout: -37% L1
sectors, 18-26% slower kernels, +1.83 GB of file).

The forward-looking statement from the 2026-11-03 budget remains the honest one: 83% of the
lm_head cost is drafting, the 1-column calls already run at 80-89 GB/s, and there is no
launch-parameter headroom in them.

## The same counters downgrade R4

The 7-column verify call (8.83 ms/pass) shows the same signature from the 2026-11-03 trace:
L1/TEX 84.2% against SM 63.5%, `lg_throttle` 0.85 as the top stall with `math_pipe_throttle` at
0.35, DRAM and L2 traffic identical to the 1-column case (417/418 MB - the weights are read
exactly once), 128 registers and 25%/22.4% occupancy. It is the same kernel in the same
L1-issue-bound regime, so nothing in the launch-parameter space moves it either; the only
candidate left is the activation re-read, and rows per block = 4 is already in. R4 stays on the
list only as "needs a traffic change", not as a quick probe.

## Pass-level audit of the non-kernel time

Done in the same session against the same trace (78 decode passes, a pass delimited by the
7-column q6_K lm_head launch), because R2's 18% share is often quoted against a pass whose
"idle" was never measured:

| quantity | per decode pass |
|---|---|
| span | 165.80 ms |
| kernel busy | 156.78 ms |
| GPU idle in gaps > 50 us | **7.14 ms in 7 gaps** |
| kernels | 2446, **all graph-launched** (no eager kernel at all) |
| `cudaGraphLaunch` | 15 calls, median 930 us of CPU each |
| `cudaStreamSynchronize` | 201 calls, median 2.4 us (p99 5.8 ms, max 114 ms) |
| `cudaMemcpyAsync` | 72 calls, median 11.4 us |
| gap after the last draft step | 3.7 ms, then six gaps of 0.4-1.3 ms, one per MTP draft step |

The six per-draft gaps are the host-side sampling loop (a graph launch, a D2H of the logits row
and a stream sync per step), which is structural to `draft-mtp` with host-side sampling. The
3.7 ms gap is CPU-side submission of some kind. This was tested as R7 with
`GGML_CUDA_GRAPH_CHUNK_SIZE` and **the chunking explanation did not survive both orders**
(512 vs 128 vs 64 = 36.56 / 36.61 / 36.55, a 0.11% spread; see
`2026-11-04-graph-chunk-size.md`). That trace also shows captures are not per-pass - decode
replays cached graphs - so the idle is not first-chunk recording and the 7.14 ms is still
unattributed.

**Corrected 2026-11-04 (host-side profile, `2026-11-04-cpu-side-of-a-pass.md`): the 7.14 ms is
real, not a tracing artefact, and it is host work.** `tegrastats GR3D_FREQ` - a hardware counter
with no CUPTI in it - reads median 95% over a decode, i.e. ~7.8 ms of idle in a 156.5 ms pass,
which agrees with the traced 7.14 ms. The cause is 41.7 ms/pass of host CPU (0.262 CPUs), almost
all of it overlapped with GPU execution except where a sync forces the hand-off: the full
248320-wide logits row is scanned and copied on the host per sampler call, and the graph fusion
search plus scheduler split is paid on every one of the 7 launches in a pass. The six per-draft
gaps listed above are that hand-off, not graph chunking.

Caveat that has to travel with this table: kernel busy under a trace is inflated (CUPTI adds
per-kernel overhead), so "busy + idle" does not decompose an untraced pass. An untraced pass at
this build is ~155 ms wall, and the traced span is 165.8 ms; the difference is the tracing
overhead, not a recoverable 10 ms.
