# The prefill / TTFT starting point: 71-79 tok/s, and why this is a separate front

**Date:** 2026-11-04
**Purpose:** baseline and code audit for a prefill/TTFT campaign. Decode is at 36.9 tok/s and
the remaining items there are sub-1%; **prompt processing runs at only 2x decode throughput per
token on the inherited `-ub 32` setting (5x once it is raised, section 5), and it had never been
measured without the prefix cache spoiling it.**

## 1. Why the existing harness could not see this

`bench/one.sh` sends `"cache_prompt": true`. Reps 2+ therefore evaluate only the tokens past the
cached prefix, and the `prompt_per_second` it prints is an artefact - this is already recorded as
AGENTS.md rule 4 (a "44 -> 10.7 tok/s collapse" that was really 4 tokens per rep). Every prefill
number in this repo before today came from that path or from a single cold first request, which
mixes graph capture into the measurement.

`bench/prefill.sh` is the replacement:

- `cache_prompt: false` on every rep, so each one re-evaluates the whole prompt;
- `n_predict: 1`, `ignore_eos`, `temperature 0`, and `stream: true`, with the **client** clocking
  the first streamed chunk, so TTFT includes HTTP, tokenisation and scheduling;
- prompts built deterministically from `prompt-32k.txt` at ~4 chars/token, with the server's own
  `prompt_n` recorded so the fixture's approximation never enters the result;
- rep 1 reported separately, because it is not the same measurement as the rest;
- every run appends to `bench/prefill.csv` (`name,ctx,ub,b,rep,len_req,ttft_ms,prompt_ms,prompt_tps,n_prompt,pred_ms,client_ms`), so arms stay comparable across sessions.

## 2. Baseline

`NAME=p0-base CTX=40960 UB=32 B=32 LENS="512 2048 8192" REPS=3`, known-good binary and model,
`-fa on -ngl 99 -ncmoe 0 -ctk/-ctv q8_0`, MTP draft enabled as in production.

| requested | `prompt_n` | TTFT (median) | prefill ms | prefill tok/s | ms/token |
|---|---|---|---|---|---|
| 512 | 590 | **7,491 ms** | 7,491 | **78.8** | 12.7 |
| 2048 | 2,385 | **31,468 ms** | 31,468 | **75.8** | 13.2 |
| 8192 | 9,563 | **134,638 ms** | 134,638 | **71.0** | 14.1 |

Medians of reps 2-3 (rep 1 carries graph capture); the 9,563 row is the median of three reps
within 0.14%. `CTX=40960` so a 9.5k prompt fits with room to generate; this does not affect
prefill cost materially and decode was not measured at this context.

Two things to read from this:

- **TTFT at a 9.5k-token prompt is 135 seconds.** That is the number the campaign is against.
- **Prefill is only ~2x decode per token.** Decode moves 5.68 tokens per 154 ms pass = 27.1
  ms/token; prefill is 13-14 ms/token. Taking ~2B active parameters at 2 FLOP each, a prefill
  token is ~4 GFLOP, so 71 tok/s is roughly **0.28 TFLOPS effective** against the board's
  **1.43 TFLOPS fp32** and **~5.7 TOPS int8 dp4a** - about 20% of fp32 and 5% of dp4a. It is not
  memory-bound (110 GB/s x 14 ms would be 1.5 GB/token, which this model does not have) and not
  launch-bound. It is doing roughly the right arithmetic very inefficiently. The FLOP basis is an
  estimate, so treat the *ratio to peak* as the claim, not the absolute. Section 5 bounds what is
  actually reachable.

Prefill also measures far more reproducibly than decode: the three reps at 9,563 tokens are
134,793 / 134,770 / 134,605 ms, **0.14% spread**, where decode needs both orders to resolve 1%.
Three reps per length is enough here, and the A/A discipline of rule 7 is not the limiting
factor - so the campaign can spend its time on arms rather than on repeats.

**First-request cost is separate and also large:** rep 1 at 590 tokens took 12,703 ms against
7,491 ms warm, so **5.21 s of the first request is graph capture**, not prefill. Any TTFT figure
that starts at process launch is load (~5-6 min from emmc) + capture (~5 s) + prefill.

## 3. The code audit: which prefill paths exclude Volta

Decode is MMVQ throughout and MMQ is prompt processing (established 2026-11-03,
`findings/2026-11-03-mmwq-launch-params.md`). The prefill paths are a different set of kernels,
and **two of the three have an architecture gate that either excludes sm_72 or feeds it a table
tuned for another chip** - the same shape as the two bugs already fixed in this fork.

**a. Gated DeltaNet has no tensor-core path on Volta - 30 of 41 layers.**
`ggml/src/ggml-cuda/gated-delta-net-mma.cu:388-410`: the MMA path requires `n_tokens >= 64`,
`H_k == 16`, and then `ampere_mma_available(cc)` **plus** cc in {GA10X, ADA_LOVELACE, BLACKWELL,
DGX_SPARK}. On Xavier `select_gdn_mma_path` returns `gdn_path::ar`, so every GDN layer runs the
fallback kernel in `gated_delta_net.cu:75`, which is `for (int t = 0; t < n_tokens; t++)` - a
serial recurrence over the prompt inside each launch. With `-ub 32` that is one launch per 32
tokens per layer, each doing 32 dependent steps.
**Enabling this is not a gate flip.** The path's tiles are Ampere MMA
(`tile<16, 8, float>` against `nv_bfloat162` operands, `gated-delta-net-mma.cu:62-70`): Volta has
no bf16 tensor cores at all, and PTX `mma` on sm_70 is m8n8k4 only, so a Volta version needs its
own schedule. The *algorithm* in that file (chunked delta rule, `C`-token blocks, `O(C^2)`
inverse of the lower-triangular correction, then an MMA solve) is portable; the kernels are not.

**b. MMQ runs the Ampere parameter table on Volta.**
`ggml/src/ggml-cuda/mmq.cuh:261-266`: `ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA`
returns `ggml_cuda_mmq_get_config_ampere(...)`. sm_72 therefore takes tile shapes, `nthreads`,
`occupancy`, `J`, `K_vram`, `stream_k` and SRAM layout chosen for Ampere. Identical in kind to
`get_device_table_id` handing sm_70/72/75 the Turing MMVQ table, which was worth +3.4% and +0.85%
once corrected - **except that this one is on the prefill path and has never been touched.**

**c. FlashAttention already uses tensor cores in prefill; it is not the anomaly.**
`fattn.cu:704-712`: for `volta_mma_available(cc)` the selector returns `BEST_FATTN_KERNEL_MMA_F16`
whenever `Q->ne[1] * gqa_ratio_eff > 16`, which prefill always satisfies and decode (1 query)
never does. This is the code behind the old note "decode does not use the MMA path at all, so
this buys prefill". Only 11 of 41 layers attend, so FA - including upstream issue #28037's
proposed split-D kernel - is capped at 27% of layers.

## 4. Ranked hypotheses

Ranked by (expected gain x confidence) / cost. Predictions were written down before the runs, per
rule 1; each is falsifiable. H1 has since been measured (section 5); the rest have not.

| # | Hypothesis | Prediction | Cost |
|---|---|---|---|
| H1 | `-ub`/`-b` 32 -> 256-512. At 32 columns the MMQ matmuls have almost no width to amortise tile loads, and there are 299 launches per 9.5k prompt. | prefill **150-280 tok/s** (2-4x), TTFT at 9.5k from 135 s to **35-65 s**. Decode must be re-checked at the same time: `-ub` is also the speculation rollback unit, and the 2026-11-02 audit found values other than 32 worse for decode pre-dispatch-patch | 1 load + 3 lengths per arm, ~10 min per arm, 4 arms |
| H2 | GDN serial recurrence is the dominant prefill kernel (30 layers, token-serial, no tensor cores). | if true, H1's gain saturates well below the prediction and the trace shows `gated_delta_net` >= 40% of prefill time. Falsified if MMQ dominates | one nsys run on a 2048-token prefill |
| H3 | MMQ on the Ampere table leaves double digits on the table for sm_72, as the Turing MMVQ table did. | +5-15% prefill from a Volta config table, after H1 makes the columns wide enough for `J` to matter | needs H1 first, then a sweep over `J`, `nthreads`, `occupancy`, SRAM layout in `mmq-config-*.cuh` |
| H4 | Volta chunked GDN (port the algorithm, not the kernels). | the only lever that could return **1000+ tok/s** prefill, because it replaces a serial 14 ms/token recurrence with an `O(C^2)` blocked solve on tensor cores. Highest ceiling, highest effort, and it is a numerics change unless the chunked form is bit-matched - which it cannot be, so it must be re-derived, not ported for speed alone | days-to-weeks |
| H5 | FA prefill schedule (upstream #28037 split-D). | bounded at 27% of layers; **+3-8%** prefill at best | medium |

**H2 is turn-key, and its protocol is different from decode's.** There is no lm_head verify launch
to bracket a prefill pass with, so instead make the trace *be* one ubatch: send a prompt of exactly
one ubatch (512 tokens at `-ub 512`) with `cache_prompt:false`, and after the first request every
kernel in the trace belongs to that single prefill. Per-signature sums then give the prefill budget
for 512 tokens directly; divide by 512 for per-token cost. Base it on `bench/prof-nsys.sh` (nsys
under `sudo`, `--trace=cuda`, report name `gpukernsum`), and expect `mul_mat_q` (MMQ),
`gated_delta_net`, `flash_attn`/`fattn` and `quantize_q8_1` to be the four candidates.

**TTFT levers that are not kernels**, worth stating because they are larger than several kernel
items and cost nothing:

- first-request graph capture, **5.21 s** measured above;
- model load from emmc, ~5-6 min at 88 MB/s (`--load-mode none` today; the user is installing an
  NVMe, which removes this entirely);
- `cache_prompt: true` for the follow-up-turn case - a chat turn after a cached prefix has a
  TTFT of a few hundred ms rather than 135 s, and that is the case most UIs actually present.

## 5. H1 measured: `-ub/-b` 512 is a 2.4x prefill win

Same harness, same binary, same model, only `-ub 32 -b 32` -> `-ub 512 -b 512`:

| prompt | ub 32 | ub 512 | speedup |
|---|---|---|---|
| 590 tokens | 7,491 ms / 78.8 tok/s | **3,647 ms / 162.1 tok/s** | **2.05x** |
| 2,385 tokens | 31,468 ms / 75.8 | **12,852 ms / 185.8** | **2.45x** |
| 9,563 tokens | 134,638 ms / 71.0 | **56,342 ms / 169.9** | **2.39x** |

**TTFT at a 9,563-token prompt: 134.6 s -> 56.3 s.** The prediction was 150-280 tok/s and
35-65 s at 9.5k; the measurement lands inside both bands, at the low-middle of the throughput one.

Three things follow.

1. **`-ub 32` was costing more than half of prefill throughput and nobody could see it**, because
   every prefill number in this repo came from a harness that served the prompt from cache. It is
   also very likely the real explanation for the old decode-sweep note "`-ub`/`-b` other than 32 is
   worse" - that verdict was reached single-order, 3 reps, pre-dispatch-patch, and never looked at
   prefill at all. Whether decode is affected at 512 is being measured now; **do not adopt it
   until that lands**, because `-ub` is also the speculation rollback unit.
2. **The gain saturates rather than scales.** 2.05x at 590 tokens, 2.45x at 2,385, 2.39x at 9,563 -
   and per-token cost at 512 goes 6.2 / 5.4 / 5.9 ms, so a wider ubatch does not remove the
   length dependence, which is the KV cache growing and the attention term with it. That is FA's
   territory, 11 of 41 layers, capped by H5.
3. **At 512 the arithmetic rate is about 0.73 TFLOPS - half of the board's 1.43 TFLOPS fp32 peak,
   and ~13% of its ~5.7 TOPS dp4a peak.** Those two numbers bound the argument in opposite
   directions and both matter: against *fp32* peak there is only ~2x left, while against *dp4a*
   there is more - but MMQ's dequantisation runs in the same kernels, and nobody has measured how
   much of this kernel's time is dequant versus dot on sm_72. That is exactly what H3 (the Ampere
   MMQ table handed to Volta) and H2 (the per-kernel budget) are for. What is already clear is
   that the remaining gain is not a knob: it is dp4a efficiency or removing GDN's token-serial
   recurrence.

## 6. What the campaign must not do

- Do not tune prefill on `bench/one.sh`: its PP column is cache-contaminated by construction.
- Do not accept a prefill win that moves decode. The production score is 36.9 tok/s at
  acceptance 0.96657 / mean len 5.68, and `-ub`/`-b` are both prefill and speculation
  parameters. Every arm reports decode, acceptance and mean len alongside prefill.
- Do not compare prefill numbers across sessions without the same fixture: `prompt_n` is
  recorded because the char/token ratio of the fixture is not exactly 4.

## 7. Decode at `-ub 512`: it is not a trade-off, it is a different model

Measured `bench/one.sh`, `NPRED=400`, 5 reps, same binary and model, only `-ub/-b` differing:

| arm | warm TG (tok/s) | median | acceptance | mean len |
|---|---|---|---|---|
| `d2-base` ub 32 | 36.75 / 37.20 / 36.84 / 36.88 | 36.86 | 0.96657 | 5.68 |
| `d2-ub512` | 20.02 / 19.25 / 22.53 / 20.07 / 19.74 | ~20.0 | **0.65882** | **2.25** |

**-46% decode, and the acceptance lands in the same 0.63-0.72 band as every other numerics
perturbation in this project** (0.6524 requant, 0.63603 lm_head Q5_K, 0.67717 f32 router,
0.678 jusko fork). That is the signature of moved numbers, not of a scheduling change, and it is
consistent with throughput = `mean_len / pass_latency`: mean len 5.68 -> 2.25 alone accounts for
the whole loss. Note rep 1 is low too (20.02), so this is not a warm-up artefact.

Mechanism, as a hypothesis with a test: `-ub` is the *compute* batch, so it changes how a prompt
is chunked, and three things ride on that chunking - the GDN recurrent state is accumulated in f32
over ubatch boundaries (`gated_delta_net.cu:75` loops `t < n_tokens` inside one call), FlashAttention
picks its tile from the query count per call (`fattn.cu:704-712`), and the MoE router's discrete
top-8 of 256 experts is computed at a different column count. Any of the three moves the prompt's
hidden states in the last bits, and the MTP drafter needs numerics *identical* to the target, not
close. `-b` (logical batch) does not change what is computed per ubatch, so it should be free.

**Predictions, written before the run (rule 1):**

1. `-ub 32 -b 512` reads 36.8-36.9 tok/s with acceptance **0.96657** and mean len **5.68**,
   bit-identical to the control, and gives no prefill gain. If instead it collapses, the mechanism
   is wrong and `-b` matters too.
2. `-ub 128` and `-ub 256` **both collapse into 0.63-0.72 acceptance**, because the chunking
   changes for any ubatch size, not just at a threshold.
3. If 2 holds, `-ub` is a **binary workload-profile knob, not a tuning knob**: 32 for interactive
   decode, 256-512 for TTFT-critical long prompts, nothing in between. The marathon then has to
   buy prefill throughput *inside* a ub-32 ubatch (H2/H3/H4) rather than by widening it.

Arms are run with the new `DECODE_REPS` phase of `bench/prefill.sh`, which measures decode in the
same server load as the prefill reps, so every prefill arm from here on carries its own decode and
acceptance check (campaign rule 2) without paying a second 5-6 minute emmc load.

### 7.1 Arms as they land (2026-11-04)

`bench/sweep-ubwidth.sh` and `bench/sweep-ub64.sh`, `DECODE_REPS=5` at `NPRED=400` and 3 prefill
reps per length, all in one server load per arm.

| arm | decode TG (tok/s) | acceptance | mean len | prefill 590 / 2,385 / 9,563 |
|---|---|---|---|---|
| `d2-base` ub 32 | 36.75 / 37.20 / 36.84 / 36.88 | 0.96657 | 5.68 | 78.8 / 75.8 / 71.0 |
| `d2-ub512` | 20.02 / 19.25 / 22.53 / 20.07 / 19.74 | 0.65882 | 2.25 | 162.1 / 185.8 / 169.9 |
| `ub128` | 28.77 / 19.97 / 21.65 / 18.69 / 30.81 | **0.92258** | **4.45** | **107.5 / 112.1 / 103.0** |
| `ub256` | 20.00 / 20.02 / 20.00 / 20.02 / 19.42 | **0.57655** | **2.22** | **142 / 146.6 / 135.5** |
| `b512-ub32` | **did not run** | - | - | (prefill unchanged by construction) |
| `ub64` | 32.58 / 32.87 / 33.57 / 33.15 / 33.58 | **0.91592** | **5.12** | 81.9 / 80.3 / 75.8 |
| `ub96` | 19.10 / 18.73 / 19.36 / 18.80 / 19.29 | **0.68122** | **2.24** | 100.3 / 97.9 / 90.5 |

**`ub128` is the interesting arm and it is not usable.** Prefill is a real **1.36x / 1.48x / 1.45x**
- so the gain is not a 512-specific cliff, it scales with width - but acceptance falls to 0.92258
and, more damning, **TG sways 18.69 to 30.81 across five identical reps**. Per `AGENTS.md`, a
configuration at or above its optimum gives a bit-deterministic draft chain with identical
acceptance and generated counts across reps; swaying counts mean the run is outside the intended
regime. A 1.45x prefill for a configuration whose decode varies by 65% rep to rep is not a
candidate for the launch line.

Two numbers worth keeping from it: acceptance 0.92258 / mean len 4.45 predicts roughly
0.92258 x 4.45/5.68 = 72% of control throughput if the pass time were unchanged, and the measured
medians are ~21.65/36.86 = 59%, so part of the loss is also a longer pass - the mean len drop alone
does not account for it. The 512 arm, by contrast, is almost exactly its mean-len ratio
(2.25/5.68 = 40% against 20.0/36.86 = 54%), which is a reminder that `mean_len / pass_latency` has
a pass term that also moves.

**The width curve, now solid on four points.** Prefill rises monotonically with ubatch width and
decode falls monotonically with it:

| ub | prefill at 590 / 2,385 / 9,563 tok/s | speedup vs 32 | acceptance | mean len |
|---:|---|---|---|---|
| 32 | 78.8 / 75.8 / 71.0 | 1.00x | 0.96657 | 5.68 |
| 128 | 107.5 / 112.1 / 103.0 | 1.36x / 1.48x / 1.45x | 0.92258 | 4.45 |
| 256 | 142 / 146.6 / 135.5 | 1.80x / 1.93x / 1.91x | 0.57655 | 2.22 |
| 512 | 162.1 / 185.8 / 169.9 | 2.05x / 2.45x / 2.39x | 0.65882 | 2.25 |

The prefill side is a smooth function of width with no threshold anywhere; the acceptance side is
not smooth, and both 256 and 512 sit near the bottom of the observed band (0.576 and 0.659) rather
than degrading gradually. That is the shape of a discrete decision being lost - the drafter's
argmax diverging - not of a kernel getting slower. So the practical conclusion is not "find the
right width": it is that **prefill width and draft acceptance trade directly against each other,
and the only acceptable point is the one where acceptance is still bit-identical**, which for now
is 32. Whether 64 keeps it is the last open question (historical draft-3: 0.96167).

**`-ub 32 -b 512` did not produce a number, and the reason matters.** The server died during the
first request's graph capture with `NvMapMemAllocInternalTagged: error 12` and
`CUDA error: out of memory` (`/tmp/server-b512-ub32.log`), so prediction 1 above - bit-identical to
control - is **untested, not confirmed**. What it does establish is narrower and still useful: a
logical batch of 512 with a 32-column compute batch does not fit on this board alongside the model,
so `-b` cannot be raised on its own as a free lever regardless of what it would do to numerics. The
question stays open only in the sense that nobody should spend another load on it: the lever is
unusable at 512 either way, and raising `-b` without `-ub` buys no prefill throughput by
construction, because prefill works in ubatches.

Do not re-run that arm as written. If the mechanism ever needs settling, use `-b 128 -ub 32`
instead - inside the memory that is actually available.

**`ub64` decomposes cleanly and still is not free.** Decode 33.2 tok/s median against 36.86,
acceptance 0.91592, mean len 5.12. The arithmetic is the whole story: 5.12/5.68 = 0.901 and
33.2/36.86 = 0.901 - identical to three digits - so at 64 columns the pass latency is unchanged and
the entire loss is drafting. That is a different failure from 128 and 512, where the measured
throughput falls faster than the mean-len ratio, i.e. the pass itself also gets longer. Useful
because it says the mechanism starts as pure acceptance damage and only becomes a scheduling
problem at wider ubatches.

So the frontier is: **+X% prefill always costs acceptance, from the very first step.** 64 buys
whatever its prefill turns out to be (pending) for -10% decode and 0.96657 -> 0.91592. There is no
width at which prefill improves and decode does not, which is the finding that reorients the
campaign from configuration to kernels.

### 7.2 The width sweep is closed: six points, and the trade is not monotonic

All arms complete, same binary, same model, `NPRED=400`, 5 decode reps and 3 prefill reps per
length, one server load each.

| ub | prefill 590 / 2,385 / 9,563 tok/s | speedup vs 32 | acceptance | mean len | decode TG |
|---:|---|---|---|---|---|
| 32 | 78.8 / 75.8 / 71.0 | 1.00x | **0.96657** | **5.68** | 36.86 |
| 64 | 81.9 / 80.3 / 75.8 | 1.04x / 1.06x / 1.07x | 0.91592 | 5.12 | 33.2 |
| 96 | 100.3 / 97.9 / 90.5 | 1.27x / 1.29x / 1.27x | **0.68122** | 2.24 | 19.1 |
| 128 | 107.5 / 112.1 / 103.0 | 1.36x / 1.48x / 1.45x | 0.92258 | 4.45 | ~21.7 |
| 256 | 142 / 146.6 / 135.5 | 1.80x / 1.93x / 1.91x | **0.57655** | 2.22 | ~20.0 |
| 512 | 162.1 / 185.8 / 169.9 | 2.05x / 2.45x / 2.39x | 0.65882 | 2.25 | ~20.0 |

**Three things this settles.**

1. **There is no free width, and 64 is the clearest negative.** 64 buys +4-7% prefill - inside the
   session's own drift floor, so effectively nothing - and pays 0.96657 -> 0.91592 for it, which is
   -10% decode. Strictly dominated. **The configuration stays at 32.**
2. **Acceptance is not monotonic in width**: 0.96657, 0.91592, 0.68122, 0.92258, 0.57655, 0.65882
   at 32/64/96/128/256/512. 96 is worse than both its neighbours, and 256 is worse than 512. A
   mechanism that merely "perturbs the prompt state a bit more with width" would be monotonic; this
   is not. It is a property of *which kernel* runs at each width, not of the width.
3. **That reframes H3, and it is the campaign's best shot.** In the Ampere MMQ table
   (`mmq-config-ampere.cuh`) the config is dispatched by an exact `J == (J_)` match, and in prefill
   `J` *is* the ubatch width, so every width above selects its own specialization (lines 104-119 for
   `q8_0`, 295-... for `iq3_s`). One source-level detail is worth a look: `J` = 8, 16, 32, 64 and
   128 have both a `fallback=true` and a `fallback=false` entry, while 24, 40, 48, 80, **96** and
   112 exist only with `fallback=false` - and the function's tail returns the sentinel
   `GGML_TYPE_COUNT` config. Width 96 is the arm with the catastrophic acceptance. That is a lead,
   not a conclusion: what the caller does with the sentinel is unread.

   The prize if it pays: the 2.4x prefill at 512 is already measured and already reproducible; what
   is missing is a J-specialization that computes the same bits as the control. **A Volta MMQ table
   that pins a numerics-safe config at wide J would unlock 134.6 s -> 56 s TTFT at unchanged
   acceptance**, which is worth far more than the +5-15% H3 was originally scoped for. Rank it
   above any further width sweep; the width sweep is finished.
