# R7: `GGML_CUDA_GRAPH_CHUNK_SIZE` is not a lever - a 1.16% one-way result that did not survive both orders (2026-11-04)

Outcome first: **no change adopted.** The interleaved both-orders run reads 36.56 (512, default),
36.61 (128) and 36.55 (64) tok/s - a spread of 0.11%, which is noise. The +1.16% the one-way scan
suggested was drift between arms, exactly the error AGENTS.md rule 7 exists to prevent.

## Why it was tried

R2's pass audit (`2026-11-04-lmhead-1col-occupancy.md`) found 7.14 ms of GPU idle per decode pass
in 7 gaps > 50 us, while all 2446 kernels in the pass are graph-launched and nothing else runs on
the device. Upstream records each ggml graph into CUDA-graph chunks of
`GGML_CUDA_GRAPH_CHUNK_SIZE` nodes (default 512) and launches a chunk as soon as it is recorded:

```cpp
if (use_cuda_graph && chunk_size > 0 && i >= chunk_end && !is_concurrent_event_active) {
    ggml_cuda_graph_launch_chunk(cuda_ctx, graph, n_chunks++);
    CUDA_CHECK(cudaStreamBeginCapture(cuda_ctx->stream(), cudaStreamCaptureModeRelaxed));
    chunk_end = i + chunk_size;
}
```

If the pass idle were the recording of the first chunk, a smaller chunk should start the device
sooner. The knob is numerics-neutral by construction: same kernels, same order, same values.

## One-way scan (5 reps per arm, `bench/prompt2.txt`, 400 generated tokens)

| `GGML_CUDA_GRAPH_CHUNK_SIZE` | TG tok/s | median | vs 512 |
|---|---|---|---|
| 512 (default, control) | 35.82, 36.25, 36.67, 36.29, 36.40 | 36.29 | - |
| 128 | 35.63, 36.65, 37.05, 36.71, 36.83 | 36.71 | **+1.16%** |
| 0 (whole ggml graph as one CUDA graph) | 34.44, 35.81, 35.98, 35.54, 35.77 | 35.77 | -1.43% |
| 4096 (same as 0 for a 2446-node graph) | 34.58, 36.00, 36.10, 35.73, 35.83 | 35.83 | -1.27% |

Acceptance 0.96657 and mean length 5.68 in every arm, which is the check that the knob moved only
submission timing. The shape of the curve (smaller better, whole-graph worst) looked like a
mechanism, and +1.16% was above the ~0.5% noise floor of 5 reps, so it was taken to confirmation.

## Interleaved confirmation - refuted

`bench/sweep-chunk3.sh`, arms in the order `512 128 64 64 128 512`, 5 reps each, same server
binary, same prompt, acceptance 0.96657 and mean len 5.68 in all six arms:

| chunk size | reps (10 pooled, two runs) | median |
|---|---|---|
| 512 (default) | 35.66 36.55 36.88 36.51 36.58 / 35.62 36.55 37.12 36.68 36.63 | **36.56** |
| 128 | 35.45 36.63 37.05 36.69 36.73 / 35.50 36.58 36.87 36.55 36.43 | **36.61** (+0.11%) |
| 64 | 35.52 36.36 37.06 36.54 36.35 / 35.15 36.67 36.84 36.63 36.57 | **36.55** (-0.03%) |

Both arms of the earlier "+1.16%" pair were single runs in one session, and the control was the
first arm - which rule 7 says is the *favoured* position, so the sign of the one-way result was
not even consistent with the known drift direction. Ten reps per value in both orders put all
three within 0.11% of each other.

The whole-graph endpoint (0 and 4096, -1.3% each in the one-way scan) was not confirmed in both
orders either, so it is not established. It is recorded here only as "do not set it to 0 on this
board in the hope of a win" - the two arms that tried it were slower, and the mechanism for a
regression (the device idles through the whole graph's recording) is plausible.

## What this says about the pass idle

The fix does not exist in this knob, and the mechanism behind the 7.14 ms was not established:

- Captures are not per-pass. The trace has 364 capture-pairs over 91 pass boundaries and they are
  clustered (81 in the first, then groups of 9-10 on scattered passes), so most decode passes
  replay a cached graph without recording anything. A replay-path knob cannot then be worth 1.2%
  through "first-chunk recording", and the measurement agrees: it is not.
- All 2446 kernels per pass are graph nodes (`gridId != 0`), so the gap is not eager launches.
- `cudaGraphLaunch` is called 15 times per pass with a median API duration of 930 us, and
  `cudaStreamSynchronize` 201 times with a median of 2.4 us. Both are CPU-side and neither maps
  cleanly onto the gaps.

The remaining candidate for the pass idle is the host-side speculative loop itself (graph launch,
D2H of the logits row and a stream synchronize per draft step, six per pass), which chunking
cannot touch. That is where a future attempt would have to go, and it needs a real CPU-side
profile rather than a kernel trace.

**Followed up the same day and confirmed:** `2026-11-04-cpu-side-of-a-pass.md` ran that CPU-side
profile. The idle is real (tegrastats `GR3D_FREQ` median 95%, which carries no tracing inflation)
and it is host work: `perf stat` gives 2937 ms of task-clock for an 11.21 s request, i.e.
**41.7 ms of host CPU per 156.5 ms pass**, of which ~7.8 ms is exposed as GPU idle - matching the
traced 7.14 ms. So this finding's conclusion holds and the "unattributed" part is closed: the
cause is per-call sampler and graph-preparation work on the CPU, not graph chunking.

One anchor so nobody chases the 7.14 ms as if it were all recoverable: the record configuration
is 36.63 tok/s with mean length 5.68, so a pass is **155.1 ms of wall time**, against a **traced
busy of 156.8 ms** and a traced span of 165.8 ms. Kernel time under a trace is inflated (CUPTI
adds per-kernel overhead), so the true idle is below 7.14 ms by an unmeasured amount - the trace
cannot separate the two, and any attempt on this needs `perf`-style CPU attribution of
`llama-server`, not another kernel trace.

## Harness trap found on the way

The first attempt at this scan used `NPRED=128` and returned 30.1 tok/s with `mean len = 4.22` for
the same build that reads 36.3 / 5.68 at `NPRED=400`, at visually identical acceptance. The draft
chain is cut short near the end of the token budget, so a short run measures a shallower draft,
and the difference (17%) is far larger than anything currently being chased. Recorded in
`AGENTS.md` section 3 with `ab-lib.sh`'s default of 400.
