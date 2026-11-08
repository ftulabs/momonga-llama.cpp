# The CPU side of a decode pass: 42 ms of host work per 156 ms pass (2026-11-04)

Status: **negative on the idle, positive on the accounting.** The 7.14 ms/pass of GPU idle that
`./2026-11-04-lmhead-1col-occupancy.md` could not attribute is real but small; the
surprise is how much host work a pass costs and where it sits. This is the CPU-side profile that
the earlier nsys-only audit said was needed.

## What was asked

The 2026-11-04 pass audit summed the gaps between graph-launched kernels in an nsys trace at
7.14 ms per 165.8 ms pass. Two explanations were open: real GPU idle, or CUPTI span inflation.
The audit could not separate them, and both candidate fixes (graph chunking, kernel work) had
already been measured to do nothing.

## Prediction, written down first

`/tmp/pred-idle-cpu.txt`: non-wait host work on the critical path would be **1-3 ms/pass**
(0.07-0.21 s over a 400-token request), so the 7.14 ms would be tracing artefact and the item
would close. Falsifier at **>= 5 ms/pass**.

**The prediction was wrong by an order of magnitude.** Measured host time is **41.7 ms/pass**.

## Method

Known-good config, 32768 ctx, `--spec-draft-n-max 6`, `--spec-draft-p-min 0.5`. One warm-up
request (`n_predict 64`, builds and captures the graphs, 21.5 tok/s) then one measured request
(`n_predict 400`, `cache_prompt`, prompt `bench/prompt2.txt`), so `prompt_n = 4` and PP is not
part of the figure - see AGENTS.md section 3 rule 4.

* `perf record -F 199 -p <server main thread>` for the measured request only, then
  `perf report --sort symbol`.
* `perf stat -p <pid> -e task-clock,cycles,instructions` around a second measured request.
* `tegrastats --interval 250` alongside it for `GR3D_FREQ`, which is a hardware utilisation
  counter and does not carry CUPTI's inflation.

## Numbers

```
perf stat: task-clock 2937.37 ms   cycles 4.069 G (1.385 GHz effective)   insn 4.047 G   IPC 0.99
request:   11.211 s wall, 400 tokens, 36.21 tok/s, acceptance 0.96657
           11.211 / (400 / 5.68) = 70.4 passes  ->  156.5 ms/pass wall
task-clock  2937 ms / 70.4 passes = 41.7 ms of host CPU per pass = 0.262 CPUs
```

So a pass is not one serialised GPU burst with a little host bookkeeping: the host thread is
executing for over a quarter of the pass's wall time. 41.7 ms / 7 decode calls per pass
(1 verify at 7 columns + 6 draft steps at 1 column) = **~6 ms of host work per graph launch**.

Top of the profile (self time, 606 samples, ~50% covered by resolvable symbols):

| share | symbol |
|---|---|
| 8.33% | `common_sampler_sample` |
| 6.87% | `__memcpy_generic` |
| 2.94% | `ggml_cuda_can_fuse` |
| 2.83% | (unresolved `0x268d8`) |
| 2.35% | `llama_token_data_array_partial_sort_inplace` |
| 2.16% | `ggml_backend_sched_split_graph` |
| 2.13% | [k] kernel address (a syscall) |
| 1.72% | `ggml_cuda_try_fuse` |
| 1.55% | `pthread_mutex_lock` |
| 1.44% | `_int_free` |
| 1.41% | `ggml_can_fuse_subgraph_ext` |
| 1.34% | `__memset_zva64` |
| 1.12% | `ggml_backend_cuda_graph_compute` |
| 1.09% | `ggml_cuda_match_moe_weighted_reduction` |

Three clusters, not one:

1. **The host sampler reads the full logits row.** `common_sampler_sample` plus
   `partial_sort_inplace` is 10.7%, and the `__memcpy_generic` 6.9% is the logits/embedding
   readback that feeds it. `output.weight` produces a 248320-wide row; the target-side sampler
   gathers all of it over to the host and top-k's it there, per row, per pass.
2. **Per-call graph preparation.** `ggml_cuda_can_fuse` + `try_fuse` + `ggml_can_fuse_subgraph_ext`
   + its PLT stub is ~7%, and `ggml_backend_sched_split_graph` is another 2.2%. This is the
   host-side fusion search and scheduler split, paid on every one of the 7 calls in a pass even
   though the graph structure is identical each time.
3. **Allocator and locking noise**: `_int_free`, `__memset_zva64`, `pthread_mutex_lock/unlock`
   total ~5%.

## GPU utilisation, measured without CUPTI

`GR3D_FREQ` over the decode, 250 ms samples, 45 of them: **median 95%, mean 80%, p10 42%,
min 0%**. The mean is dragged down by the interval around request start and teardown; the
decode itself sits at 95%.

95% of 156.5 ms leaves **~7.8 ms/pass of exposed idle**, which agrees with the nsys gap figure
of 7.14 ms to within the error of both methods. So the earlier reading was right after all -
there is roughly 7-8 ms/pass of GPU idle - and the CPU profile explains what the GPU is waiting
for: not a shortage of work to launch, but host work (sampler, readback, fusion search) that
sits between two launches on a strictly serialised dependency chain.

## What this changes

* The idle is **real and bounded at ~5%** (7.8 / 156.5). Removing all of it is +1.9 tok/s, and
  that is an upper bound on this whole line.
* It is not launch *count*. Every kernel in the pass is already a graph node; the lever is host
  work per launch, and the two largest blocks are addressable without touching numerics:
  the sampler chain and the per-call fusion/split.
* `--backend-sampling` (`-bs`) is the fork's existing switch for the first block: it runs the
  chain on the GPU and reads back only the candidates. A/B result is in
  `./2026-11-04-backend-sampling.md`.

## Caveats

* `perf record -p` samples the main thread only. CUDA's driver threads and the HTTP thread are
  not in the 2937 ms; that is deliberate, because only the main thread is on the pass's critical
  path.
* The 606 samples at 199 Hz are ~50% symbol-resolved. The unresolved addresses are in the CUDA
  runtime and in shared objects perf could not map; they are not counted in the three clusters
  above, so the clusters are shares *of the resolvable half*, i.e. lower bounds on absolute cost.

---

# Follow-up, same day: the same profile with `-bs` on (the adopted config)

`-bs` was adopted on the strength of `./2026-11-04-backend-sampling.md` (+1.04%), and this
profile was taken **before** it, so it no longer describes the record configuration. Re-run with
`-bs`, same method (warm-up 64 then 400, `perf stat -p` and `perf record -F 499 -p` on the server
main thread).

Prediction, written down first in `/tmp/r8/pred-bs-cpu.txt`: task-clock per pass **falls from
41.7 to 33-37 ms (-10 to -20%)**, `common_sampler_sample` drops off the top of the symbol table,
and the graph-preparation cluster becomes the largest single target. Falsifier: within 3% of
41.7 ms.

## Result: the work fell 34%, the milliseconds did not - and the milliseconds are the wrong metric

| metric | without `-bs` | with `-bs` | delta |
|---|---|---|---|
| instructions per request | 4.047 G | **2.64-2.67 G** | **-34%** |
| cycles per request | 4.069 G | 3.46-3.58 G | -14% |
| task-clock per request | 2937 ms | 2825-3137 ms | ~0% |
| implied effective clock (cycles / task-clock) | 1.385 GHz | 1.14-1.22 GHz | -12 to -18% |
| IPC | 0.99 | 0.74-0.77 | -25% |
| passes (400 tokens at mean len 5.68) | 70.42 | 70.42 | - |
| **instructions per pass** | **57.5 M** | **37.4-37.9 M** | **-34%** |
| tok/s during the profiled request | 36.21 | 36.51 / 36.03 | - |

**The prediction was right about the work and wrong about the metric.** Instructions fell 34% -
more than the predicted 10-20% - but task-clock in milliseconds did not fall at all, because the
implied core clock read 1.14-1.22 GHz in these runs against 1.385 GHz in the first one. The
strict falsifier ("task-clock within 3% of 41.7 ms/pass") therefore fires on the letter and not
on the substance, and that is worth stating plainly rather than choosing the metric after the
fact.

The consequence is a rule: **for host-side CPU work, compare instructions (or cycles), never
milliseconds, across sessions.** 2.9 s of CPU per 11.2 s request is 26% of wall either way, and a
DVFS or thermal difference between sessions is larger than every effect this project now chases.
`scaling_cur_freq` is useless here - it reads a constant 2265600 kHz - so the clock has to be
inferred from `cycles / task-clock`, which is exactly why the two counters must be collected
together.

## What is left, from the same trace with `-bs` on (shares of resolvable cycles)

| symbol | share | what it is |
|---|---|---|
| `ggml_can_fuse_subgraph_ext` (+ `@plt`) | 3.55% | fusion pattern match, per node |
| `ggml_backend_sched_split_graph` | 3.36% | recomputed per graph compute, 7x per pass |
| `__memcpy_generic` | 3.23% | logits readback and buffer copies |
| `__memset_zva64` + `__memset_generic` | 4.09% | buffer clears |
| `ggml_cuda_can_fuse` | 2.40% | fusion pattern match |
| `ggml_cuda_try_fuse` | 1.66% | fusion pattern match |
| `_int_malloc` + `malloc` + `__libc_calloc` + `free`/`cfree` | ~5.7% | allocator churn |
| `ggml_visit_parents_graph` | 1.54% | graph construction |
| `__printf_buffer` + `__strlen_generic` | 1.85% | log formatting |
| `ggml_gallocr_alloc_graph` | 0.83% | per-graph allocation |
| `ggml_cuda_graph_evaluate_and_capture` | 0.82% | graph capture bookkeeping |
| `ggml_backend_cuda_graph_optimize` | 0.62% | - |

`common_sampler_sample` (8.33% before) and `llama_token_data_array_partial_sort_inplace` (2.35%)
are **gone from the table**: `-bs` removed exactly the cluster it was aimed at. The graph
preparation cluster is now the largest coherent target at **~11% of host cycles, i.e. ~4.5 ms of
the ~42 ms per pass**, paid seven times per pass for a graph whose structure does not change.

## Bound on the remainder

Host CPU is 26% of wall and only ~7.8 ms/pass of it is exposed, so the conversion from removed
host work to wall time is roughly the ratio the `-bs` result shows: -34% of instructions bought
+1.04% of wall. An 11% cut of what is left is then worth **well under 0.5%**, and it needs a
split/fusion cache keyed on graph topology in `ggml-backend.cpp` / `ggml-cuda.cu` - a change that
silently alters which kernels run if it is ever wrong. That is the honest state of R8: one lever
taken (+1.04%), a hard ceiling of +1.9% for the whole line (7.8 ms/pass of idle), and the
remaining ~0.3-0.5% behind an invasive change. It stays open in `TODO.md` and unstarted.
