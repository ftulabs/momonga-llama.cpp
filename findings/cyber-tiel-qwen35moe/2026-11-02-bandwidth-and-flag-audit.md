# Decode bandwidth ceiling and the remaining flag audit

**Date:** 2026-11-02
**Config under test:** 31.63 tok/s control (draft 3, `p-min 0.50`, `-ub 32 -b 32`, q8_0 KV,
qwen35moe IQ4_XS, mainline `dde879b`)

This file answers three questions: whether every llama.cpp flag was actually tried, what
upstream has that we do not, and whether decode is memory-bandwidth-maxed.

## The model, measured rather than assumed

Parsed from the GGUF header of `Cyber-Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS.gguf`:

| Key | Value |
|---|---|
| `general.architecture` | `qwen35moe` |
| `block_count` | 41 |
| `embedding_length` | 2048 |
| `expert_count` | 256 |
| `expert_used_count` | 8 |
| `expert_feed_forward_length` | 512 |
| `nextn_predict_layers` | 1 |
| `context_length` | 262144 |

Two structural facts that close off whole branches of flags:

- There is **no `sliding_window` key**. `--swa-full` therefore has nothing to un-SWA on this
  model, and the SWA-related KV flags cannot help.
- The 256-expert MoE runs 8 experts per token. Expert selection is per-layer and data
  dependent, which is why the access pattern is scattered (see below).

## 1. Is decode memory-bandwidth-maxed? No.

### What the board actually delivers

Nominal peak for AGX Xavier LPDDR4x is 137 GB/s. That number is not available to a real
kernel, so both figures below were measured on this board rather than quoted.

| Path | Measured | Repeat | Source |
|---|---|---|---|
| GPU global read, `sm_72`, 512 MB buffer | **109.7 GB/s** | **110.2 GB/s** | `bench/bwtest.cu` |
| CPU STREAM read (float, 8 accumulators), 256 MB | **~54 GB/s** | **56.2 GB/s** | `bench/stream.c` |

So the achievable GPU ceiling is **~110 GB/s**, 80% of spec. Both figures were re-measured on an
idle machine after the sweeps and agreed to within 0.5%. The first CPU version of the STREAM
test reported 33.2 GB/s; that was FP64-bound (Carmel's FP64 units are scalar) and was rewritten
with float and eight independent accumulators. The CPU number matters only to rule out CPU-side
staging as an alternative.

### How much of it decode uses

Cyber-Tiel activates roughly 3B of its 35B parameters per token (8 of 256 experts), which at
4.25 bpw is about **1.6 GB of weight traffic per forward pass**. At 8.5 passes/s that is
**~13 GB/s - around 12% of the 109.7 GB/s the board delivers**.

| Workload | Effective GB/s | % of measured ceiling | Bound by |
|---|---|---|---|
| Cyber-Tiel, with MTP speculation | ~13 | **12%** | latency / access pattern |
| Cyber-Tiel, plain decode (no spec) | ~28 | 26% | latency / access pattern |
| Spark 4B Q4_0 | 58.2 | 53% | bandwidth |
| Spark 4B Q8_0 | 69.6 | 63% | bandwidth |

The contrast with Spark on the same board is the control: a dense 4B model that reads every
weight per token climbs to 53-63% of the achievable ceiling. The MoE does not, because it only
touches 3% of its experts and does so through scattered gathers.

### Direct proof: a second stream makes things worse

A roofline argument can always be argued with. This experiment cannot. If decode were
bandwidth-saturated, running two concurrent streams would cost almost nothing per stream,
because the expensive part (streaming weights) would already be paid. Measured
(`bench/conc.sh`, identical config, 400 tokens each):

| Streams | Per-stream tok/s | Aggregate tok/s |
|---|---|---|
| 1 | 31.20 | **31.20** |
| 2 | 12.92 + 11.72 | **24.64** |

(Repeated independently after the sweeps: 31.18 single, then 12.90 + 11.71 = **24.61**, a
0.1% reproduction.)

Aggregate throughput **fell 21%** when a second stream was added. There is no spare bandwidth
being left on the table by request serialisation; adding work makes the machine slower. Decode
is bound by per-pass latency and the MoE access pattern, not by DRAM.

Corroborating telemetry from the earlier ceiling measurement: GR3D load 99%, GPU pinned at
1377 MHz, EMC pinned at 2133 MHz, CPU 6%. The SMs are busy and stalled on scattered expert
reads, not idle and not waiting on memory.

### Consequence for the 45-60 tok/s target

Because it is not bandwidth-bound, the target cannot be reached by moving fewer bytes (smaller
quant, shorter context) - that path was already measured and loses more in acceptance than it
gains in traffic. It requires fewer milliseconds per pass: **~84 ms instead of ~118 ms**, a 1.4x
kernel-level improvement at unchanged acceptance. That is `TODO.md` section G, and it is the
only route left.

## 2. Was every flag tried?

`llama-server --help` was dumped in full (753 lines) and audited flag by flag. The earlier
sweeps had covered quantization, KV type, ubatch/batch, draft depth, `p-min`, `p-split 0.10`,
build flags, context depth and thread counts. The audit found these never tested, now measured
(`bench/sweep-newlevers.sh`, ports 8801-8809):

| Config | Median tok/s | vs control | Acceptance | Verdict |
|---|---|---|---|---|
| n00 control | 31.63 | - | 0.9894 / 3.69 | baseline reproduced |
| n01 `--backend-sampling` | 31.48 | -0.5% | 0.9894 / 3.69 | neutral; sampling was never the bottleneck |
| n02 `--spec-draft-n-ubatch 1` | 30.47 | **-3.7%** | 0.9894 / 3.69 | worst of the set, serialises the draft |
| n03 `--spec-draft-n-ubatch 4` | 31.18 | -1.4% | 0.9894 / 3.69 | regression |
| n04 `--spec-draft-p-split 0.05` | 31.45 | -0.6% | 0.9894 / 3.69 | neutral |
| n05 `--spec-draft-p-split 0.20` | 31.58 | -0.2% | 0.9894 / 3.69 | neutral, best non-control |
| n06 `--spec-draft-sampling probabilistic` | 31.37 | -0.8% | 0.9894 / 3.69 | no gain; greedy drafting already optimal |
| n07 `--swa-full` | 31.25 | -1.2% | 0.9894 / 3.69 | expected no-op, confirmed |
| n08 `--kv-unified` | 31.44 | -0.6% | 0.9894 / 3.69 | neutral |

**Nothing beat the control.** The full spread across nine configurations is 30.47-31.63, i.e.
everything that is not actively harmful sits inside the +-1% run-to-run noise. Acceptance is
bit-identical (0.9894 / 3.69) in every row, which proves these are pure scheduling and kernel
effects rather than trajectory effects.

`--spec-draft-p-split` deserves a note: only the default 0.10 had ever been used, so 0.05 and
0.20 were swept for the first time here. Both are flat, which means the default is not a
mis-set knob and does not need further exploration.

### Flags ruled out by argument, not by test

| Flag | Why it cannot help |
|---|---|
| `--swa-full` | model has no sliding-window layers (no `sliding_window` key in GGUF) |
| `--kv-unified` | single sequence; nothing to share |
| `--no-kv-offload` | KV cache is a rounding error next to 1.6 GB of weights per pass |
| `--lookup-cache-static/dynamic` | superseded by `ngram-cache` in `--spec-type`; tested as a speculator instead |
| `-ot` / `--override-tensor` | one GPU, unified memory; there is no second memory tier to move tensors to |
| `--n-cpu-moe` | CPU reaches ~54 GB/s vs the GPU's 109.7; offloading is strictly slower, and it is not bandwidth-bound anyway |
| `--fit`, `--lazy-mode` | load-time behaviour only, no effect on decode |
| `--spec-draft-threads/prio/poll` | draft context is one layer on the GPU; host-side knobs are noise |

## 3. Deeper drafting cannot win: the marginal token costs more than the average

`--spec-type` accepts a comma-separated list, so MTP had never been combined with an n-gram
speculator. Both combinations draft deeper (the n-gram speculator goes to 8 tokens), and both
lose. The reason is arithmetic, not tuning (`bench/sweep-newlevers2.sh`):

| Config | Reps (tok/s) | Median | vs control | Acceptance / mean len | Pass time |
|---|---|---|---|---|---|
| x00 control | 31.13 / 31.57 / 31.22 | 31.22 | - | 0.9894 / 3.69 | **118 ms** |
| x01 `draft-mtp,ngram-mod` | 34.07 / 20.04 / 29.66 | 29.66 | -5% | 0.8715 / **5.52** | **186 ms** |
| x02 `draft-mtp,ngram-cache` | 17.41 / 26.86 / 34.39 | 26.86 | **-14%** | 0.9227 / **8.21** | **306 ms** |
| x03 draft KV q8_0 (`-ctkd/-ctvd`) | 30.96 / 31.70 / 31.51 | 31.51 | +0.9% | 0.9894 / 3.70 | 117 ms |
| x04 draft KV q4_0 | 31.05 / 31.25 / 31.35 | 31.25 | +0.1% | 0.9929 / 3.69 | 118 ms |
| x05 `--spec-draft-prio 2` | 30.81 / 31.58 / 31.39 | 31.39 | +0.5% | 0.9894 / 3.69 | 118 ms |
| x06 `--spec-draft-poll 1` | 31.00 / 31.48 / 31.35 | 31.35 | +0.4% | 0.9894 / 3.69 | 118 ms |

Dividing pass time by mean length gives the cost of each verified token:

| Step | Tokens added | Pass time added | Marginal cost |
|---|---|---|---|
| control baseline | 3.69 | 118 ms | **32.0 ms/token average** |
| x01 (3.69 -> 5.52) | +1.83 | +68 ms | **37.1 ms/token** |
| x02 (5.52 -> 8.21) | +2.69 | +120 ms | **44.5 ms/token** |

**The marginal token costs more than the average token**, and the cost accelerates. Every extra
draft position therefore *lowers* throughput, which is why the hybrid speculators lose despite
carrying 50-120% more tokens per pass. It also explains, from the other direction, the older
result that `n-max 4` is no better than `n-max 3`: they are the same trade at the break-even
point.

The mechanism is the kernel gap, not the speculator. At depth 3 the verification batch is ~4
tokens; the n-gram hybrids push it to 9, and there is no tuned kernel for a batch of 9 - too big
for the batch-1 decode path and far too small for prefill kernels. Making the verification batch
cheap is worth 1.5x on its own even with today's acceptance.

The variance is worth recording too: x01 ranged 20.04-34.07 and x02 17.41-34.39. Long variable
passes are not just slower on average, they are unstable, which matters for interactive use.

**Caveat on the two hybrids specifically.** This harness re-sends the same prompt for all three
reps, and an n-gram speculator builds its index from the context, so its hit rate is inflated by
repetition and the medians above cannot cleanly rank `ngram-mod` or `ngram-cache`. That
limitation was already known when these were first observed hitting 33-34 tok/s on individual
reps. The honest statement is that the hybrids are *unranked* on this harness, not that they are
proven worse. The finding that does not depend on them is the fixed-depth one below: real MTP
drafting at depth 6-12, with no n-gram index involved, loses monotonically - and at
`n-max 8, p-min 0.85` the target accepts **every** drafted token and it is still slower.

### Perfect acceptance makes it slower, not faster

Round 3 tested the remaining shape of the idea: a deeper `--spec-draft-n-max` gated by a high
`--spec-draft-p-min`. This is how MTP drafting already works internally - the nextn head exposes
a per-position confidence and the draft loop breaks when it falls below `p-min`
(`common/speculative.cpp:1427-1433`) - so a high gate is a poor man's version of upstream's
`draft-mtp-adaptive` (#27210) using flags we already have.

| Config | Median tok/s | vs control | Acceptance | mean len | Pass time |
|---|---|---|---|---|---|
| y00 control | 31.26 | - | 0.9894 | 3.69 | 118 ms |
| y03 `GGML_CUDA_FORCE_CUBLAS=1` | 31.52 | +0.8% | 0.9894 | 3.69 | 117 ms |
| y04 `n-max 6, p-min 0.70` | 28.25 | -9.6% | 0.9841 | 5.83 | 206 ms |
| y05 `n-max 8, p-min 0.85` | 27.12 | **-13.3%** | **1.0000** | 6.81 | 251 ms |
| y06 `n-max 8, p-min 0.95` | 23.69 | **-24.2%** | **1.0000** | 5.15 | 217 ms |
| y07 `n-max 12, p-min 0.95` | 23.52 | -24.8% | 0.9922 | 5.18 | 220 ms |

y05 is the decisive row. Acceptance is **exactly 1.00000** - every single drafted token was
verified by the target model, the theoretical best a speculator can do - and it yields 6.81
tokens per pass. It is still **13% slower** than the control, because a pass costs 251 ms
instead of 118 ms.

That settles the speculation question. Acceptance is not a lever any more: it is already 0.9894
at depth 3, and pushing it to 1.0 costs more in pass time than it returns. There is nothing left
to win by drafting better, and #27210 (adaptive depth) cannot help a machine whose marginal
token costs 37-43 ms against a 32 ms average.

`GGML_CUDA_FORCE_CUBLAS=1` is worth one line: it is neutral, where `GGML_CUDA_FORCE_MMQ=ON` was
measured at -14%. The default dispatch chooses correctly for this model, so there is no build-time
flag left either.

### A dead end worth recording: `--spec-synth-len`

`--spec-synth-len` looked like an ideal controlled experiment: it replaces verification with a
synthetic acceptance draw while still running the same target forward pass and the same drafting,
so per-pass compute should be unchanged and only `mean len` should move. It does not work as
advertised here. Requesting `--spec-synth-len 3.69` produced a measured `mean len = 3.40` (not
3.69) and cost 7% (28.95 vs 31.22 tok/s), while acceptance fell to 0.8878. The synthetic path
adds its own overhead and does not reproduce the requested length at depth 3, so it cannot be
used as a control. Configurations built on it were dropped rather than reported.

### Residency: nothing is being paged out

One hypothesis survived the flag audit: with a 22.5 GB model in 31 GB of unified RAM, the kernel
could be reclaiming model pages between passes, so each pass would silently re-read weights from
storage. `--mlock` and `--no-mmap` no longer exist in this build - they are now
`-lm/--load-mode {auto,none,mmap,mlock,mmap+mlock,dio}` - and they settle the question
(`bench/sweep-newlevers4.sh`):

| Config | Median tok/s | vs control |
|---|---|---|
| z00 control (default: mmap, no mlock) | 31.38 | - |
| z01 `-lm mlock` | 31.38 | 0.0% |
| z02 `-lm mmap+mlock` | 31.51 | +0.4% |
| z03 `-lm dio` | 31.68 | +1.0% |
| z04 `-lm none` | 31.30 | -0.3% |

Pinning the entire model resident changes nothing, so no reclamation is happening and the
weights are not being re-read from storage. The `+1.0%` for `dio` is within the +-1% noise band
and has no mechanism (it affects the load path, not decode), so it is recorded as noise rather
than as a finding.

### Bottom line on flags

Four sweeps, 28 completed configurations (plus two that failed on removed flags), all at
400 generated tokens on the varied prompt, three reps each:

- **Nothing beat the control.** Everything that is not actively harmful lands inside +-1%.
- The best non-control result is `-lm dio` at +1.0%, which is noise with no mechanism.
- Every real regression comes from drafting deeper (down to -25%). The two n-gram hybrids are
  the exception to counting as clean data here: they are unranked by this harness (see caveat
  above), so the deep-draft regressions that matter are the fixed-depth ones.
- Acceptance is identical (0.9894 / 3.69) in all the neutral and negative cases except where
  deeper drafting deliberately changes it, so these are overhead effects, not trajectory effects.

Config space is exhausted. Combined with the bandwidth measurement, the remaining 1.4x to 45
tok/s is kernel work only.

## 4. What upstream has that we do not

Our checkout is `dde879b` (2026-11-01). Master is one commit ahead on an unrelated change, so
updating the tree yields nothing. The useful work is in open PRs and was verified against the
tree rather than assumed:

**Already present in our build** (confirmed by reading the source): the MoE fusions #31188
(`topk-moe.cu` has the fused top-k path) and #31288 (`mmq.cu` "Fused gate/up/GLU",
`ggml_cuda_should_fuse_mul_mat`). `GGML_CUDA_FORCE_MMQ` was already measured at **-14%** decode
and is not worth revisiting.

**Not applicable, verified:**

- **#31415 / #31302** "MTP graph reallocations when switching inputs". The reproduction requires
  an **image** request via mtmd; text-only servers are unaffected. Confirmed on our own logs:
  `graph reallocs = 0 / 0` in `print_timing`, and zero `sched_graph_realloc` lines.
- **#29143** d2t draft-vocabulary trim. Only helps a sidecar or FastMTP head trained with a
  reduced draft vocabulary. Our embedded nextn head uses the full 262144 vocabulary.
- **#31294** skip `p-min` for probabilistic drafts. Only matters with
  `--spec-draft-sampling probabilistic`, which loses to greedy here.

**Worth trying, config-level once patched:**

- **#27210** `--spec-type draft-mtp-adaptive`, a counting state machine that raises draft depth
  after runs of full acceptance and drops it after misses, recommended with
  `--spec-draft-n-max 12` and floor 3. Our acceptance is 0.9894, so most passes already verify
  all three drafts and depth is nearly fixed-optimal; the expected gain is small but it is the
  only remaining config-shaped lever. The patch is +521/-38 and currently conflicts with master.

**Kernel-level candidates, the real route:**

- **#31525** SM70 GQA pairing for long-KV decode (directly targets Volta attention).
- **#31047** MoE data layout linearisation - addresses the scattered-gather pattern that causes
  the 12% bandwidth utilisation measured above.
- **#30141** save/restore MTP hidden state.

Because our bottleneck is now measured to be the MoE access pattern and per-pass latency, #31047
and #31525 are the two that actually attack it.

## Reproducing

```
bench/bwtest.cu    GPU read/copy/memcpy bandwidth, built for sm_72
bench/stream.c     CPU STREAM (float, 8 accumulators)
bench/conc.sh      single vs N concurrent streams against a live server
bench/sweep-newlevers.sh   the flag audit sweep (ports 8801-8809)
bench/sweep-newlevers2.sh  hybrid speculators, draft KV, draft scheduling (8901-8910)
bench/sweep-newlevers3.sh  deep draft + confidence gate, FORCE_CUBLAS (9001-9008)
bench/sweep-newlevers4.sh  load-mode / residency (9101-9105)
```
