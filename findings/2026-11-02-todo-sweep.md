# 2026-11-02 - sweeping the TODO list: quantization, KV, batching, speculation, FA quants, context depth

Machine: Jetson AGX Xavier, sm_72, mainline `dde879b`.
Model: Cyber-Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS unless stated otherwise.

## Bottom line

**Nothing beat the standing best of ~31.5 tok/s.** The optimum from
[2026-11-01-decode-speed-tuning.md](2026-11-01-decode-speed-tuning.md) survives every
combination tried in this pass:

```
--spec-draft-n-max 3 --spec-draft-p-min 0.50 --spec-draft-p-split 0.1
-ub 32 -b 32 -fa on -ctk q8_0 -ctv q8_0 -t 2 -tb 2
```

Several long-standing open questions did get answered, most of them negatively, and one
of them overturns an earlier conclusion (the `p-min` optimum is a plateau, not a sharp
point). The two genuinely new results are:

- **Q4_K_M is 6% slower per pass than IQ4_XS** and also suppresses draft acceptance. The
  i-quant dequant path is not the bottleneck it was suspected of being.
- **Spark 4B gains 52% from Q8_0 to Q4_0** (15.55 to 23.71 tok/s), and the speedup fits a
  two-term model that predicts where the remaining headroom is.

## Read this before comparing any two rows

Speculative decoding makes tok/s a **bad** metric for comparing configurations that
change the model's numerics. Any change to KV type, ubatch, or quantization perturbs the
greedy token trajectory, which changes draft acceptance, which changes tok/s - with the
kernel speed held constant.

Worked example from the KV sweep: `q8_0/q8_0` and `f16/f16` produce *identical* output
(acceptance 0.98940 in both), and their speeds land within 0.3% of each other. Meanwhile
`f16/q8_0` produces a different trajectory with acceptance 0.520 and appears 45% slower,
even though most of that gap is acceptance, not kernel time.

The kernel-time metric used throughout is therefore **pass latency**:

```
pass_ms = (mean_len + 1) / tok_per_sec * 1000
```

`mean_len` and `acceptance` come from the server's `draft acceptance` / `mean len`
lines. One pass verifies `mean_len` drafted tokens and emits one bonus token, so this
divides out how many tokens each pass happened to win. It is approximate - a pass that
verifies more tokens does cost somewhat more - but it separates real kernel regressions
(10-15%) from acceptance noise.

All rows: one server load, 3 requests of 400 generated tokens, `bench/prompt2.txt`,
greedy. Median of the 3.

## B. KV cache type

| KV type | tok/s (median) | pass ms | FA path |
|---|---|---|---|
| q8_0 / q8_0 | 31.53 | 149 | native |
| f16 / f16 | 31.63 | 148 | native |
| q4_1 / q4_1 | 31.67 | 150 | f16 convert |
| q5_1 / q5_1 | 24.18 | 165 | f16 convert |
| q8_0 / f16 | 18.11 | 168 | f16 convert |
| f16 / q8_0 | 17.42 | 170 | f16 convert |

Two things were learned here.

**Mismatched K and V types are a trap.** llama.cpp only compiles a fixed list of K/V
pairs (`GGML_CUDA_FA_QUANTS`, default `q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16`). Any
other pair falls back to converting both to f16 on every attention call, and logs:

```
no FlashAttention vector kernel compiled for K/V types f16-q8_0,
converting K and V to f16 instead (slow)
```

That conversion costs ~13% of pass latency and, because it changes the numerics, also
collapses draft acceptance. Mixed KV is never worth it.

**Quantizing KV buys nothing on this model.** It is hybrid - only a minority of blocks
are full attention - so the KV cache is small to begin with, and on Volta the quantized
attention kernels are slower than the f16 path anyway. `q8_0` and `f16` tie at 148-149 ms;
every other type is 150-170 ms. Keep q8_0 (it halves KV memory versus f16 at zero
measurable speed cost).

This supersedes the old note that "q4_0 KV measured at -9%": the effect was never about
memory bandwidth.

## C. Batching and scheduling

| Config | tok/s (median) | pass ms |
|---|---|---|
| cache-reuse 256 | 31.75 | 148 |
| ctx-checkpoints 8 | 31.55 | 149 |
| **baseline (ub 32)** | **31.53** | **149** |
| poll 1 | 31.37 | 150 |
| parallel 2 | 31.22 | 150 |
| ub 64 | 30.13 | 154 |
| ub 128 | 24.29 | 160 |
| ub 256 | 18.90 | 158 |
| -b 2048 | 27.17 | 164 |

- **ub 32 is confirmed optimal.** Larger ubatch degrades both pass latency and, sharply,
  draft acceptance (0.989 at ub 32, 0.962 at 64, 0.828 at 128, 0.586 at 256). The
  acceptance collapse is systematic rather than random, which suggests the MTP draft head
  is itself evaluated less favourably at larger ubatch - worth a look upstream, but the
  operational conclusion is unambiguous.
- **`-b 2048` costs 10% of decode** even though decode never fills the batch. It only
  affects the compute-buffer allocation. Not worth it: prefill on this board is slow
  regardless, and prefill is not the target.
- **cache-reuse, ctx-checkpoints, poll 1, parallel 2 are all neutral** (within 0.9%).
  They do not harm decode, so `--cache-reuse 256` is safe to keep for prompt-reuse
  workloads, but it is not a speed lever.

## D. Speculation

| Config | tok/s (median) | pass ms | acceptance | mean len |
|---|---|---|---|---|
| p-min 0.50, depth 3 (best) | 31.53 | 149 | 0.9894 | 3.69 |
| p-min 0.45 | 31.51 | 148 | 0.9757 | 3.65 |
| p-min 0.48 | 31.08 | 151 | 0.9894 | 3.68 |
| p-min 0.52 | 29.31 | 156 | 0.9639 | 3.57 |
| n-min 1 | 31.32 | 150 | 0.9894 | 3.69 |
| depth 4 | 28.89 | 190 | 0.9901 | 4.49 |
| ngram-cache | 24.05 | 299 | 0.6545 | 6.20 |
| mtp + ngram-mod (n-max 8) | 29.79 | 219 | 0.8715 | 5.52 |
| mtp + ngram-cache | 26.67 | 345 | 0.9227 | 8.21 |
| ngram-mod (n-max 8) | 17.23 | 545 | 0.9457 | 8.39 |
| ngram-map-k4v | 16.54 | - | - | - |
| ngram-map-k | 16.52 | - | - | - |
| *no speculation* | *16.64* | - | - | - |

**The p-min optimum is a plateau, not a sharp point.** 0.45, 0.48 and 0.50 all land
within 1.5% of each other; only 0.52 and above clearly lose. The old claim that 0.50 is a
sharp optimum should be read as "anything in 0.45-0.50 is fine". Lower p-min slightly
reduces per-pass cost (fewer drafts generated) and slightly reduces acceptance; the two
almost exactly cancel.

**`--spec-draft-n-min` does nothing here** - acceptance and output are identical to
n-min 0. The p-min gate dominates it.

**Depth 4 is a clear loss**: acceptance rises (0.990) but pass latency jumps 28% because
each pass verifies ~5.5 tokens instead of ~4.7. Depth 3 stays optimal.

**Pure n-gram speculation does not work on prose.** The map-k variants give exactly
no-speculation speed (16.5 vs 16.64), because varied text has almost no literal repeats
to match. `ngram-cache` reaches 24 tok/s only by verifying ~7 tokens per 299 ms pass.

**The n-gram + MTP hybrids are unusable as measurements**: individual reps ranged from
17.3 to 34.2 tok/s within one configuration. The n-gram index grows every time the same
prompt is re-sent, so later reps find far more matches than the first. That is a
benchmark artifact of reusing one prompt - in a real session the prefix is not repeated -
and it is why these rows sit in the table with a warning rather than a ranking.

A caveat discovered along the way: `--spec-type ngram-mod` **aborts the server** with
`batch must be large enough to hold the sampled and draft tokens`
(`server-context.cpp:650`) at the default `--spec-ngram-mod-n-max 64` with `-ub 32`.
Bound it with `--spec-ngram-mod-n-max 8` to get any data at all.

## A. Quantization

The upstream repo does publish a proper imatrix Q4_K_M, so this was measured on the real
file rather than a re-quantized one.

| Model | tok/s (median) | pass ms | acceptance | mean len |
|---|---|---|---|---|
| IQ4_XS (current) | 31.53 | 149 | 0.9894 | 3.69 |
| Q4_K_M | 19.58 | 158 | 0.6524 | 2.09 |
| Q4_K_M, p-min 0.45 | 20.38 | - | 0.7443 | 2.72 |

**K-quants are slower than i-quants here, by about 6% per pass** (158 ms vs 149 ms), and
the Q4_K_M file is 22.5 GB against 18.1 GB. So the hypothesis that IQ4_XS's lookup-table
dequant on sm_72 was the hidden cost is **refuted** - IQ4_XS is the faster kernel.

The larger apparent gap (19.6 vs 31.5 tok/s) is mostly acceptance, not kernels: Q4_K_M's
different numerics send the trajectory somewhere the MTP head predicts much less well
(0.652 vs 0.989). The same caveat applies to any future quantization comparison made with
speculation on.

`q4_0` and `q8_0` of Cyber-Tiel were not measured: a Q4_0 would have to be re-quantized
from IQ4_XS (double quantization, invalid quality comparison) and a Q8 is 2x the bytes
and cannot be faster.

## E. Build flags

A second build, `build-volta-fa`, compiled the same flags plus seven FA K/V pairs:

```
-DGGML_CUDA_FA_QUANTS="q4_0-q4_0;q4_1-q4_1;q5_0-q5_0;q5_1-q5_1;q8_0-q8_0;f16-f16;bf16-bf16"
```

| Build | KV | tok/s | pass ms |
|---|---|---|---|
| build-volta (old) | q8_0/q8_0 | 31.53 | 149 |
| build-volta-fa (control) | q8_0/q8_0 | 31.62 | 148 |
| build-volta-fa | q4_1/q4_1 **native** | 18.80 | 173 |
| build-volta (old) | q4_1/q4_1 *converted* | 31.67 | 150 |
| build-volta-fa | q5_1/q5_1 **native** | 20.81 | 165 |
| build-volta (old) | q5_1/q5_1 *converted* | 24.18 | 165 |

**Compiling the extra kernels made q4_1 slower, not faster** - 173 ms native against
150 ms for convert-then-f16. The q8_0 and f16 kernels are fine (148 ms); the q4_1/q5_1
quantized attention kernels are simply poor on Volta. The default build's FA list is
correct and there is no reason to deviate from it.

Also settled: `GGML_CUDA_FA_QUANTS` is a *list* with a sane default, not an on/off
switch. The TODO item "set it ON to get quantized-KV kernels" was already satisfied for
q8_0 and f16 - those kernels ship by default.

## F. Runtime and OS

The `momonga-extra` GPG failure was **transient**; the repo works. Its signature verifies
cleanly against the installed key (`9449ABF1...BE274FDAFE86B911E0BAC3DB5B2AA4FE9449ABF1`),
so nothing was disabled to make it work. Installed from it:

- `tegra-tools-tegrastats` (35.6.4) - `/usr/bin/tegrastats`
- `python3-jetson-stats` (jtop 7.2.2) - was already present as a dependency

`bench/prof-tegra.sh` now samples tegrastats through one 400-token decode. Over 65
samples at 200 ms:

| Metric | Value |
|---|---|
| GR3D (GPU) load | 89.9% mean, 99% sustained during decode |
| CPU load (all 8 threads) | 6.0% mean, 22.2% peak |
| GPU temperature | 67.5 C mean, 69.0 C peak |
| RAM in use | 20.4 GB of 30.3 GB |

This confirms the earlier sysfs finding with a proper tool: **the GPU is saturated and the
CPU is idle**. There is no thermal throttling (69 C is far below the ~85 C limit) and no
headroom to configure away. `nvpmodel` remains uninstalled; clocks are already pinned at
GPU 1377 MHz / EMC 2133 MHz.

## H. Context depth (first real measurement)

A long, *varied* prompt was built by concatenating repo source files, so this does not
suffer from the repeated-text inflation that plagues the normal benchmark prompt.

| Context depth | tok/s (median) | pass ms | acceptance |
|---|---|---|---|
| ~0.5 K (prompt2.txt) | 31.5 | 149 | 0.9894 |
| ~8 K | 24.2 | 173 | 0.8457 |
| ~16 K | 25.3 | 188 | 0.9575 |
| ~42 K | **OOM** | - | - |

- **Pass latency grows monotonically: 149 -> 173 -> 188 ms** (+26% at 16K). Decode does
  get slower with context, as expected, but the effect is far milder than on a
  dense-attention model because only a minority of blocks are full attention.
- **tok/s is not monotonic with depth** (8K looks worse than 16K) purely because
  acceptance differs between trajectories. Read pass latency instead.
- **The 42K run ran out of memory during prefill** (`CUDA error: out of memory`) at
  `-c 49152`, despite the prompt fitting in the slot. This is the same wall that killed
  the old 64K and 128K attempts: it is not a context-length limit, it is unified memory
  being exhausted during a long prefill, most likely by the f16 conversion buffers the
  FlashAttention path allocates on Volta. Practical ceiling on this board with the 18 GB
  model is somewhere between 16 K and 42 K; the 262144 slot is advertised, not reachable.

## I. Spark-X2.6-4B: Q8_0 vs Q4_0

`llama-quantize --allow-requantize Spark-X2.6-4B-Q8_0.gguf ... Q4_0` produced a 2.29 GB
file (4.67 bpw) from the 4.16 GB Q8_0 in 98 s.

| Config | tok/s (median) |
|---|---|
| Spark Q8_0, ub 32 | 15.55 |
| Spark **Q4_0**, ub 32 | **23.68** |
| Spark Q4_0, ub 256 | 23.68 |
| Spark Q4_0, `-t 4` | 23.88 |

**+52%.** This model has no MTP head, so it is the honest unassisted ceiling, and it now
beats the 35B MoE's unassisted 16.64 tok/s while being 8x smaller. (The MoE still wins
overall at 31.5 once speculation is on.)

Fitting a two-term cost model to the two points:

```
t_per_token = 15.1 ms fixed + bytes / 91 GB/s

Q8_0: 15.1 + 4475/91 = 64.2 ms  -> 15.6 tok/s
Q4_0: 15.1 + 2458/91 = 42.1 ms  -> 23.7 tok/s
```

So effective streaming bandwidth is **~91 GB/s** (against a ~137 GB/s nominal peak), and
there is a **~15 ms per-token fixed cost**. At Q4_0, 64% of the time is streaming and 36%
is that fixed overhead - which is why halving the bytes only bought 1.52x rather than 2x.
Extrapolating, a Q2_K-class Spark (~1.3 GB) would land near 29 ms -> ~34 tok/s, with
sharply diminishing returns beyond that. Getting past ~34 tok/s on this model needs the
fixed per-token cost reduced (fewer kernel launches, CUDA graphs actually taking effect),
not a smaller file.

## Still open

Section G of the TODO list - the kernel work - is untouched and remains the only route to
45-60 tok/s:

- MoE expert gather path (index sort and dispatch are latency-bound)
- `mul_mat_vec_q` for IQ4_XS on sm_72 specifically
- a dedicated batch-about-5 verification path, since at depth 3 the batch is ~4.7 tokens
  and is neither batch-1 nor prefill
- reducing per-pass launch count under CUDA graphs

Everything measurable by configuration is now measured.
