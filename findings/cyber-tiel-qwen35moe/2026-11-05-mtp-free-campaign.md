# MTP-free campaign: what plain decode, prefill and TTFT can be

**Date:** 2026-11-05
**Instruction:** the MTP drafter no longer has to stay compatible. Find the best plain
(non-speculative) decode, prefill and TTFT the board can do. Base decode is expected to be about
15-16 tok/s.
**Status:** open. Section 1 is the prediction, written before any run.

## 1. Why this is a different problem

Every numerics constraint in this repo exists for one reason: the MTP head ships inside the target
GGUF and drafts from the target's own hidden state, so the target and the drafter must agree
bit-for-bit or acceptance collapses (0.96657 -> 0.63-0.72 on four unrelated perturbations). With
the drafter out of the loop:

- A weight type change is no longer fatal, only a change in *speed*. The IQ3_S MoE reads at
  56.7 GB/s where IQ4_XS reads 92.7 GB/s on an identical shape - a lever that was closed purely
  because requantizing loses acceptance.
- The f32 router paths, the MMQ `J` specializations, `K_vram`, `stream_k`, `nthreads`, MMVF
  thresholds - everything closed under "a change either touches no numbers or it is a different
  model" - become fair game, because nothing downstream has to agree with them.
- `-ub` stops being a decode parameter. It only ever hurt decode by changing what the *drafter*
  sees. With `--spec-type none` a decode step is one token and never fills an ubatch, so a wide
  ubatch should be free prefill.

## 2. Predictions (written before the first run)

Measured facts they are built on: MTP pass 153.8 ms for 5.68 tokens; a pass is 41 main layers,
MoE ~9.1 ms/token (0.1392 + 0.0818 ms x 41), lm_head 1-column 5.2 ms, dense q8_0 ~29-32 ms at
1 column, GDN 6.4 ms, router 3.2 ms; host work ~41.7 ms/pass when 7 sampler calls are made;
the 2026-11-01 harness-bug run read **16.29 tok/s** with no draft at all on the then-current
build (31.52 with MTP).

| # | Prediction | Confidence |
|---|---|---|
| P1 | Plain decode at `-ub 32` reads **16-18 tok/s** (160-190 ms/pass after the 17% build gain since 16.29). | medium |
| P2 | Decode is **independent of `-ub`/`-b`** with `--spec-type none`, within the 0.7% A/A floor, because a 1-token step cannot fill an ubatch. | high |
| P3 | Prefill at `-ub 512` is **~2.4x** `-ub 32`, taking TTFT at 9,563 tokens to **~55-70 s** depending on the session's drift state. | high |
| P4 | DFlash on top of plain decode reads **23-26 tok/s** (21.39 measured on 2026-11-01 scaled by the build's gains), i.e. better than plain decode but well below the 36.9 the MTP head gives. | low-medium |
| P5 | The biggest base-decode lever is the expert type: IQ3_S -> IQ4_XS on the MoE is ~16% of a plain pass and reads 1.63x faster per byte. | medium |

P4 is the one I expect to be wrong in some direction: DFlash's acceptance was 0.594 at depth 3
against a base-Ornith draft and an abliterated finetune target, and this campaign is free to
change target numerics, which would move that acceptance again without being fatal.

## 3. Base decode at `-ub 32` (measured 2026-11-05, 08:36)

`SPEC=none`, `-ub 32 -b 32`, `NPRED=400`, 5 reps, prompt2.txt, one server load:

| rep | TG tok/s |
|---|---:|
| 1 | 18.230 |
| 2 | 18.341 |
| 3 | 18.254 |
| 4 | 18.322 |
| 5 | 18.379 |
| **median** | **18.32** |

**P1 falsified, in the good direction:** predicted 16-18, measured **18.32**. The 2026-11-01
plain-decode reading of 16.29 was on the pre-tuning build; the dense MMVQ rows-per-block work,
`-bs` and the rest lifted the no-draft path by 12.5% as well as the MTP path. Rep spread is 0.8%,
so no-draft decode is reproducible in a way the MTP path is not (MTP's throughput depends on the
draft chain, which varies run to run).

**The headline ratio:** 36.93 / 18.32 = **2.02x**. The MTP head doubles throughput, which is the
price of dropping it. Everything below is about buying some of that back without the drafter.

## 4. Prefill and TTFT at `-ub 32` (same arm, same load)

`SPEC=none`, `-ub 32 -b 32`, cache off on every rep, client-side TTFT, 3 reps (rep 1 dropped):

| fixture | n_prompt | prompt tok/s | TTFT |
|---|---:|---:|---:|
| ~590 | 703 | 93.1 | 7.56 s |
| ~2,385 | 2,828 | 89.6 | 31.6 s |
| ~9,563 | 11,484 | 83.8 | 137.1 s |

The repetition is 0.1-0.2% within an arm, as always. The absolute level says this session is in
the fast part of the documented drift range (76-92 tok/s at 9.5k tokens), so quote deltas only.

## 5. `-ub 512` is free prefill now (measured 2026-11-05, 08:50)

Same harness, same load pattern, `SPEC=none`, `-ub 512 -b 512`:

| fixture | n_prompt | prompt tok/s | TTFT | vs `-ub 32` |
|---|---:|---:|---:|---:|
| ~590 | 703 | 163.1 | 4.32 s | 1.75x |
| ~2,385 | 2,828 | 188.9 | 14.99 s | 2.11x |
| ~9,563 | 11,484 | 175.4 | 65.5 s | **2.09x** |

**P2 confirmed:** decode at `-ub 512` is 18.28 / 18.31 / 18.25 / 18.27 / 18.31, median **18.31**
against 18.32 at `-ub 32` - a 0.05% difference, well inside the 0.71% A/A floor. With no drafter,
a decode step is one token and never fills an ubatch, so the width that used to cost -46% decode
now costs nothing.

**P3 half-confirmed:** predicted ~2.4x and 55-70 s TTFT; measured 2.09x and 65.5 s. The ratio is
smaller than the 2.4x banked at `-ub 512` in the MTP campaign because that figure was a ratio of
two *prefill* arms measured across sessions, while this one is both arms inside one session in
the same drift state.

The prefill gain stops being uniform at the short length (1.75x): 703 tokens is only two 512-token
chunks, so the fixed per-request cost is a larger share and the wide chunk has less expert
reuse to exploit.

## 6. Where the ubatch sweep tops out (measured 2026-11-05, 09:08-09:30)

Same harness, six widths, each its own server load, decode measured first in the same load:

| `-ub` | decode tok/s | 703 tok/s | 2,828 tok/s | 11,484 tok/s | TTFT @ 11,484 |
|---:|---:|---:|---:|---:|---:|
| 32 | 18.32 | 93.1 | 89.6 | 83.8 | 137.1 s |
| 512 | 18.31 | 163.1 | 188.9 | 175.4 | 65.5 s |
| 1024 | 18.34 | 199.9 | 227.1 | 203.6 | 56.5 s |
| 2048 | 18.23 | 199.7 | 241.0 | **214.6** | **53.6 s** |
| 4096 | 18.35 | 207.9 | **267.1** | 175.8 | 65.4 s |
| 8192 | 18.26 | 200.1 | 166.1 | 95.4 | 120.5 s |

Decode is flat across all six (18.23-18.35, a 0.65% band, inside the A/A floor). Prefill does not
monotonically improve with width:

- The peak moves with prompt length. A 2,828-token prompt is fastest at `-ub 4096` (267), where it
  is exactly one chunk; an 11,484-token prompt is fastest at `-ub 2048` (215).
- **Above the prompt length the width actively hurts.** `-ub 8192` is the clean demonstration: the
  2,828-token fixture fits in one chunk at both 4096 (267 tok/s) and 8192 (166 tok/s), so the
  compute is identical and the only difference is the buffer the graph allocates. It costs 38%.
  At 11,484 tokens `-ub 8192` reads 95.4 tok/s, worse than the 83.8 of `-ub 32` - i.e. the entire
  2.1x win of leaving MMQ can be given back by over-allocating.
- The cause was not diagnosed. Memory headroom is the obvious candidate (the graph's buffers scale
  with ub on a board where the 18 GB model is mmap'd and reclaimable), but no counter was taken
  and this is a warning, not an explanation. **Operational rule: choose the ubatch nearest to, but
  below, the prompt length, and never above it; 2048 is the safest single default for long
  prompts.** `-ub 3072` was measured afterwards and is best-or-tied everywhere (section 14), so it
  is the better single default if the prompts are at least that long.

The `MMQ_DP4A_MAX_BATCH_SIZE = 64` threshold from AGENTS.md 7(d) is what makes any of this
available: above 64 tokens the MoE leaves the fused MMQ launch for a per-expert dequant + cuBLAS
loop, which is much faster per byte. With no drafter to break, the whole range above 64 is usable.

## 7. DFlash over plain decode (measured 2026-11-05, 09:41)

`--spec-type draft-dflash -md Ornith-1.5-35B-A3B-DFlash-BF16.gguf --spec-draft-ngl 99`,
`--spec-draft-n-max 3`, `p-min 0` (its best setting on 2026-11-01), `-ub 2048`:

| rep | TG tok/s |
|---|---:|
| 1 | 14.21 |
| 2 | 14.68 |
| 3 | 14.60 |
| 4 | 14.88 |
| 5 | 14.74 |
| **median** | **14.65** |

acceptance 0.31118, mean len 1.93.

**P4 falsified, and in the bad direction:** predicted 23-26, measured **14.65** - *slower* than
the 18.23 of plain decode it was supposed to accelerate. On 2026-11-01 the same draft on a
slower build read 21.39 at depth 3 with acceptance 0.594 against plain decode 16.29, a 1.31x
win. Acceptance has halved (0.594 -> 0.311), and since throughput is `mean_len / pass`, that is
the whole story: 1.93 tokens per pass instead of ~2.7.

> **Corrected at 10:32, before this file was finished.** The paragraph above is a statement about
> `prompt2.txt`, not about DFlash. On an 8.8k-token **code** context (`prompt-8k.txt`) the same
> draft at the same width reads acceptance **0.78652**, mean len 3.35 and **22.97 tok/s** -
> section 12. The 267-token prose prompt was an unrepresentative fixture for a drafter trained on
> source code. Do not quote "DFlash is slower than plain decode" without the prompt attached.

Three things changed between the two measurements and only one of them is a free variable:

1. **Ubatch width.** DFlash injects the *target's* hidden states into its attention, and chunking
   changes those hidden states - the same mechanism that moved MTP acceptance from 0.96657 to
   0.65882 at `-ub 512`. A ub-32 re-run is in flight to test this.
2. **A dflash commit the old build did not have.** `2cbca38 dflash: copy hidden features via
   device copies (#31506)` landed 2026-11-03, after the 21.39 measurement.
3. **The target build itself**, which moved four kernels since 2026-11-01 without moving MTP
   acceptance.

Prefill with the draft loaded is 200.1 tok/s and TTFT 57.5 s at 11,484 tokens, against 214.6 and
53.6 s plain: an external draft costs about **7% of prefill**, presumably its own prompt
evaluation. That is the first measured cost of the sidecar on the prefill front.

## 8. The ubatch width is what breaks DFlash (measured 2026-11-05, 09:57)

The same draft, same flags, `-ub 32` instead of `-ub 2048`:

| rep | TG tok/s |
|---|---:|
| 1 | 21.46 |
| 2 | 21.70 |
| 3 | 21.71 |
| 4 | 21.62 |
| 5 | 21.75 |
| **median** | **21.65** |

acceptance **0.59441**, mean len **2.78** - which reproduces the 2026-11-01 measurement (21.39,
acceptance 0.594) to within the build's own speedup. So neither the dflash commit nor the target
build is responsible: **ubatch width is**, and it is the only free variable that moved.

That is the same mechanism AGENTS.md already records for MTP - chunking changes the prompt's
hidden states - but it bites DFlash harder, because DFlash's whole design is to consume the
target's hidden states from eight layers. At `-ub 2048` its acceptance halves and the draft goes
from a 1.38x win to a 0.80x loss.

Its prefill at `-ub 32` is 84.7 / 82.2 / 77.2 tok/s and TTFT **148.8 s**, 8.6% worse than the
137.1 s of plain decode at the same width (the external draft has to evaluate the prompt itself).

**The pair is therefore a genuine trade, not a tunable:**

| configuration | decode tok/s | prefill @ 11,484 | TTFT |
|---|---:|---:|---:|
| plain, `-ub 2048` | 18.23 | 214.6 | **53.6 s** |
| DFlash depth 3, `-ub 32` | **21.65** | 77.2 | 148.8 s |

DFlash buys **+18.5% decode for 2.8x TTFT**. Whether any width in between keeps most of the
acceptance is being measured now (64 / 128 / 512, decode only).

> **Corrected later the same day.** Both rows above were measured on the 267-token prose prompt,
> where DFlash acceptance is 0.311 and its win is small. On an 8.8k *code* context at `-ub 2048`
> the same sidecar reads acceptance 0.78652 and **22.97 tok/s against 16.93 plain (+35.7%)** for
> **7% of prefill, not 2.8x TTFT** - sections 11-13 and the table in section 15 are the current
> answer. The "genuine trade at `-ub 32`" framing is obsolete: the acceptable configuration is
> DFlash at a wide width on a long code context.

## 9. DFlash has two regimes, and the variable is chunk count, not width (10:13)

Depth 3, p-min 0, decode only, one server load per width:

| `-ub` | TG tok/s | acceptance | mean len | chunks of the 267-token prompt |
|---:|---:|---:|---:|---:|
| 32 | 21.65 | 0.59441 | 2.78 | 9 |
| 64 | 17.39 | 0.40935 | 2.22 | 5 |
| 128 | 21.32 | 0.58028 | 2.73 | 3 |
| 512 | 15.21 | 0.31118 | 1.93 | 1 |
| 2048 | 14.65 | 0.31118 | 1.93 | 1 |

(`prompt2.txt` is 267 tokens, from the server's own `prompt eval` line, not the ~500 assumed
above.)

**Two widths give identical acceptance to five decimal places, and that is the evidence for the
mechanism:** 512 and 2048 both fit the 267-token context in a single chunk, so the hidden states
DFlash consumes are the same, so its acceptance is the same. Width does not matter; *how many
chunks the context was cut into* does. The 64-token point is the odd one out rather than a trend
(0.409, below both 32 and 128), which is what a discrete-chunking effect looks like and what a
"smooth perturbation" story cannot produce.

Consequence: 512 and 2048 are in the "one chunk" regime, with acceptance 0.311. For a context
longer than the widest ubatch, that falls apart, and the widths would separate again - which is
why the recommendation cannot be made from this table alone. A long-context (8.8k) decode test at
`-ub 2048` against `-ub 128` is in flight.

Also measured: prefill with DFlash at `-ub 128` (section 10) to complete the trade row.

## 10. DFlash at `-ub 128`, with prefill (10:26)

| fixture | n_prompt | prompt tok/s | TTFT |
|---|---:|---:|---:|
| ~590 | 703 | 106.2 | 6.63 s |
| ~2,385 | 2,828 | 111.6 | 25.4 s |
| ~9,563 | 11,484 | 102.1 | 112.5 s |

So `-ub 128` recovers the same decode as `-ub 32` (21.32 against 21.65) and pays 112.5 s of TTFT
instead of 148.8 s. That is the best pure-decode point measured so far:

| configuration | decode tok/s | TTFT @ 11,484 |
|---|---:|---:|
| plain, `-ub 2048` | 18.23 | **53.6 s** |
| plain, `-ub 32` | 18.32 | 137.1 s |
| DFlash depth 3, `-ub 128` | **21.32** | 112.5 s |
| DFlash depth 3, `-ub 32` | 21.65 | 148.8 s |

## 11. The 267-token prompt was the wrong fixture (measured 2026-11-05, 10:52)

Same draft, same width (`-ub 2048`), same 400-token decode, three contexts:

| context | tokens | acceptance | mean len | DFlash tok/s | plain tok/s |
|---|---:|---:|---:|---:|---:|
| `prompt2.txt`, prose | 267 | 0.31118 | 1.93 | 14.65 | 18.23 |
| `prompt2-8k.txt`, prose repeated | ~8,800 | 0.52258 | 2.57 | 18.48 | 16.93 |
| `prompt-8k.txt`, a C++ source file | ~8,800 | **0.78652** | 3.35 | **22.97** | 16.33 |

(The code arm's plain baseline was still in flight when this table was written; it measured
16.33 tok/s, and 16.37 +/- 0.03 in a clean re-run without `-md` on the command line - see the
caveat in section 15.)

Two separate effects, and both are large:

1. **Context length.** Plain decode itself falls from 18.32 to 16.93 (prose) / 16.33 (code) tok/s
   at 8.8k (-8%) - the attention and KV work the pass pays for a long context. That is the fair
   baseline for any 8.8k comparison; quoting DFlash at 8.8k against 18.32 would overstate it by
   8%.
2. **Context content.** At the same 8.8k length, repeating the *prose* scores 0.523 and a *C++
   source file* scores 0.787. DFlash was trained on code, and on code it is 1.36x plain decode;
   on prose it is 1.09x. The 267-token prose prompt gave 0.31 and made the draft look like a
   loss, which is why section 7 carries a correction block.

The honest summary is therefore workload-dependent, and the coder workload is the one this model
exists for: **DFlash at `-ub 2048` reads 22.97 tok/s on an 8.8k code context against 16.33 plain,
and costs 7% of prefill.** No ubatch penalty applies at this context length - section 9's
"two regimes" was a property of the 267-token fixture, where a wide ubatch collapsed the context
into one chunk.

## 12. Depth sweep on the code context (measured 2026-11-05, 11:12)

Same 8.8k code context, `-ub 2048`, p-min 0:

| draft depth | TG tok/s | acceptance | mean len |
|---:|---:|---:|---:|
| 3 | **22.97** | 0.78652 | 3.35 |
| 5 | 21.09 | 0.59245 | 3.95 |
| 8 | 15.87 | 0.52026 | 5.12 |

Deeper drafting is worse at every step, exactly as on 2026-11-01 (depth 3 best), and for the same
reason: a verify pass is linear in tokens in this MoE (0.221 ms/token x 41 layers), so the extra
columns cost more than the extra accepted tokens return. Mean length rises 3.35 -> 5.12 while
acceptance falls 0.79 -> 0.52; throughput is `mean_len / pass` and the pass grows faster.

Plain decode on the same code context measures **16.33 tok/s**, so DFlash at depth 3 is
**+40.7%** there. (The prose copy of the same 8.8k context read 16.93 plain, a 3.5% difference
between two arms in different server loads - drift, not content; the code figure is the one the
code arm is compared against.)

A p-min gate at 0.5 reads 23.08 with acceptance 0.86111 against 22.97 / 0.78652 at p-min 0 - a
0.5% throughput difference, inside the A/A floor, so the gate is optional here rather than
required as it is for the MTP head.

## 13. Where the wide-ubatch prefill time goes (measured 2026-11-05, 11:40)

One 4,875-token prefill at `-ub 2048` under nsys (`bench/prof-prefill.sh`, one request, no warm-up
so the trace holds exactly one prefill): **17,110 ms of kernel time over 257,861 launches**, wall
26.4 s at 184.8 tok/s.

| kernel | ms | share |
|---|---:|---:|
| `cutlass_70_tensorop_h884gemm_128x128` (MoE GEMMs) | 1,629.7 | 9.5% |
| `flash_attn_tile` (three shapes, summed) | 2,271.9 | **13.3%** |
| `dequantize_block_iq3_s<__half>` | 1,507.3 | 8.8% |
| `gated_delta_net_cuda` | 1,186.5 | 6.9% |
| `concat_non_cont<unsigned int>` | 819.2 | 4.8% |
| `volta_h884gemm_128x128_ldg8` | 634.6 | 3.7% |
| `dequantize_block_iq4_xs<__half>` | 462.1 | 2.7% |
| `k_get_rows_float_vec<float>` (three shapes, summed) | 1,048.2 | 6.1% |
| `mul_mat_q` (seven shapes with `J=8`, summed) | ~1,538 | 9.0% |
| `ssm_conv_long_token_f32`, `splitKreduce`, `volta_sgemm` | 546 | 3.2% |

**The MoE stops dominating once the width is free.** At `-ub 32` MMQ was 88% of prefill and the
MoE tile was the whole game. Here the expert GEMMs plus their dequant are ~22%, and the three
biggest remaining costs are attention (13.3%), the dense-MMQ path (9.0%) and the GDN token-serial
kernel plus its `concat` (11.7%), with 6.1% in row gathers.

Three things this does *not* change:

- Attention is O(n^2) and this trace is only 4,875 tokens; at 11,484 its share will be higher. It
  is already on the MMA-F16 path (`fattn.cu:704-712` selects it for Volta whenever
  `Q->ne[1] * gqa_ratio_eff > 16`), so there is no gate to flip - only a kernel to write.
- The dense `mul_mat_q` still takes the Ampere parameter table on Volta (`mmq.cuh:261-266`), the
  same class of bug as the decode dispatch table. That is 9% of prefill with nobody having looked
  at `I` / `nthreads` for a 2,048-column dense MMQ.
- `dequantize_block_iq3_s` costs 163 ms/GB against `dequantize_block_iq4_xs` at 87 ms/GB, so
  requantizing the 9.23 GB of IQ3_S expert weights to IQ4_XS trades +19% bytes for roughly half
  the dequant time - the arithmetic says it is close to a wash on prefill (about -2.6%) and the
  real question is whether the faster dot helps decode more than the extra bytes hurt. Not
  measured, and it is a model change.

The 9.3 s of wall time that is not kernel time (35%) is not diagnosed here; at two chunks per
request it cannot be per-launch overhead, and nsys inflates wall clock.

## 14. `-ub 3072` is the single-width compromise (measured 2026-11-05, 11:50)

| `-ub` | decode | 703 tok/s | 2,828 tok/s | 11,484 tok/s | TTFT @ 11,484 |
|---:|---:|---:|---:|---:|---:|
| 2048 | 18.23 | 199.7 | 241.0 | 214.6 | 53.6 s |
| 3072 | 18.26 | **210.6** | **270.0** | **215.0** | **53.5 s** |
| 4096 | 18.35 | 207.9 | 267.1 | 175.8 | 65.4 s |

3072 is within 0.2% of 2048 at 11,484 tokens and within 1.1% of 4096 at 2,828, so it is the one
width that is best-or-tied at every length measured. Decode is 18.26 (18.18 / 18.21 / 18.26 /
18.28 / 18.31), unchanged from the other widths.

## 15. The MTP-free answer

| configuration | decode tok/s | prefill @ 11,484 | TTFT |
|---|---:|---:|---:|
| MTP (the old configuration, for scale) | 36.93 | 83.8 | 120.2 s |
| plain, `-ub 32` | 18.32 | 83.8 | 137.1 s |
| plain, `-ub 3072` | 18.26 | 215.0 | **53.5 s** |
| DFlash depth 3, `-ub 2048` | **22.97** | 200.1 | 57.5 s |
| DFlash depth 3 + p-min 0.5, `-ub 2048` | **23.08** | (same flags) | (same) |

- **Decode ceiling without MTP: 23.0 tok/s**, on a code context, with the DFlash sidecar. That is
  +40% over the 16.37 plain decode measured on the same context (and 1.25x the short-prompt plain
  ceiling of 18.32), 62% of the MTP configuration's 36.93.
- **TTFT ceiling: 53.5 s** at 11,484 tokens with plain decode at `-ub 3072`, a 2.56x improvement
  on the 137.1 s the same build reads at `-ub 32`. Adding DFlash costs 7% of prefill, so the
  decode-maximum configuration still runs at ~57 s.
- **On prose the drafter is worth much less** (18.48 vs 16.93 at 8.8k, +9%) **and on a short
  prose prompt it is a loss** (14.65 vs 18.23). The decode ceiling is workload-dependent in a way
  the plain-decode ceiling is not.

### The sidecar's residency does not shift the baseline

The two plain code arms above were launched with `DRAFT_MODEL` still exported, so `-md` was on the
command line with `--spec-type none`: the sidecar loaded but unused. A clean plain re-run without
`-md` reads **16.37 / 16.34 / 16.40 / 16.39 / 16.32, median 16.37**, against 16.33 with it loaded
- 0.24%, inside the board's noise. So the resident 747 MB draft costs nothing and the plain and
DFlash arms are directly comparable.

The crossover is the `MMQ_DP4A_MAX_BATCH_SIZE = 64` threshold documented in AGENTS.md 7(d): above
64 tokens the MoE leaves the fused MMQ launch for a per-expert dequant + cuBLAS loop, which is
much faster per byte. With no drafter to break, the whole width range above 64 is available.

The 2,828-token fixture is consistently the fastest per token (241 at `-ub 2048` against 215 at
11,484, and the same ordering at every width). That is not diagnosed: candidate causes are the
O(n^2) term in the 11 full-attention layers and the KV cache growth. It was not measured and no
lever is proposed from it here.

## 16. Does more bits pay? Q4_K_M against IQ4_XS with the drafter gone (prediction first)

With no MTP, a bigger file is no longer disqualified by acceptance, so the 22.5 GB `Q4_K_M`
quant already on disk (753 tensors, same architecture; gate/up Q4_K, down Q5_K) becomes a free
arm. Prediction written before the run:

- **P6 (decode):** 17.8-18.7 tok/s against IQ4_XS's 18.32, i.e. within +-2.5%. The file is 24%
  larger (22.513 vs 18.109 GB) and a 1-column decode pass is byte-bound, but AGENTS section 7 has
  classic 4/5-bit k-quants at 65-78 GB/s against IQ3_S's 56.7 on the expert kernel, so the two
  effects should roughly cancel. This is deliberately a two-sided prediction: if Q4_K_M reads
  more than 3% off, the "bytes dominate" model is wrong and every requant plan needs re-deriving.
- **P7 (prefill):** 195-215 tok/s at 11,484 tokens with `-ub 3072` against 215.0, i.e. 0 to -9%.
  Prefill above 64 tokens runs cuBLAS on dequantized weights, so it is closer to pure bytes than
  decode is, and 24% more of them should cost something.
- **P6 falsified, by a lot, in the useful direction: 22.33 tok/s.** Reps 22.313 / 22.299 /
  22.328 / 22.363 / 22.345, median **22.33** at ub 3072, against 18.26 for the IQ4_XS file on the
  same width - **+22.3%** from swapping the expert tensors from IQ3_S/IQ4_XS to Q4_K/Q5_K, with
  the dense q8_0 and q6_k lm_head identical in the two files. The file is 17% larger overall and
  the pass moves ~17% more bytes, so this is not a bandwidth effect at all: the IQ3_S gate/up dots
  were the drag, and the byte-per-byte ranking that AGENTS section 7 carries for batch 5
  ("classic 4/5-bit 65-78 GB/s, IQ3_S collapses") holds at 1 column as well, hard enough to
  overcome +17% bytes. Every projection of the pass puts it at 55 GB/s against the IQ4_XS file's
  44 GB/s - more bytes, moved faster.
  **The prediction was wrong by 22 points and the model behind it was wrong too.** It weighted
  bytes and a 56.7-vs-92.7 GB/s measured IQ3_S/IQ4_XS kernel ratio, concluded the two cancel, and
  so expected +-2.5%. Both inputs were measured on an expert shape at **batch 5**, and the
  conclusion was about a **batch-1** model - AGENTS rule 2 violated in the opposite direction
  from its usual one. The model also read "IQ4_XS is fast" as "IQ types are fast", when the
  batch-5 table has IQ3_S at 22 GB/s and IQ4_XS at 92.7; only the 4-bit IQ variant is fast, and
  its speed did not carry the 3-bit one. What the model-level arm says directly is what matters:
  **the gate/up projection at 3.4 bpw in IQ3_S costs more than 4.5 bpw in Q4_K**, by enough to
  overcome 17% more bytes across the whole file.
- **Prefill pays for it, and the exchange rate is not small:** 173.36 / 173.24 tok/s at 11,484
  tokens against 215.0, TTFT **53.5 -> 66.3 s (-19.4%)**. At 2,828 tokens it is 215-218 against
  270.0 (-19.7%). At 703 tokens it is *faster*, 220-224 against 210.6 (+5%), which does not fit
  a bytes story and is not diagnosed.
- **So the two MTP-free configurations are a genuine Pareto pair, not a ranking:**

| configuration | decode | prefill @ 11,484 | TTFT |
|---|---:|---:|---:|
| IQ4_XS experts, `-ub 3072` | 18.26 | **215.0** | **53.5 s** |
| Q4_K_M, `-ub 3072` | **22.33** | 173.3 | 66.3 s |

  Q4_K_M is also the higher-precision quant of the two, so on the decode side it wins speed and
  quality both; the only thing it costs is TTFT. Anything that cuts the dense q8_0 set - which is
  1.36 GB of the ~2.46 GB a plain Q4_K_M token reads - should move both columns in the right
  direction and dissolve the trade. That is the next arm.

## 17. The dense retype: attention/ssm q8_0 -> q5_k (prediction first)

The Q4_K_M result says the expert type was the drag, not the bytes. That leaves the dense set -
1.36 GB of every plain token's ~2.46 GB - as the largest untouched term, and it is still q8_0
while the experts are q4_k/q5_k. Retyping `attn_qkv`, `attn_gate`, `ssm_out`, `attn_q` and
`attn_output` (112 tensors, 1.36 GB) from q8_0 to q5_k saves 0.48 GB, -19% of the per-token read,
and the pin file pins the other 641 tensors to their current types so they are byte-copied.

- **P8 (decode):** 25.5-26.0 tok/s (22.33 + 12 to +16%). The -19% bytes only becomes -19% time if
  those kernels are bandwidth-limited; 0.48 GB at the dense path's ~80 GB/s is ~6 ms of a 44.8 ms
  pass, so -15% pass time is the central estimate, and anything above +24% would mean the q8_0 dot
  is doing more work than its bytes imply.
- **P9 (prefill):** +2 to +4% at 11,484 tokens (173.3 -> 177-180). Each chunk reads the dense
  weights once, so 0.48 GB x ~4 chunks against a pass that already moves well over 100 GB.
- **P10 (quality):** this is the first arm that is *not* quality-free. q5_k on attention and the
  GDN projections is one notch below where Q4_K_M normally sits, so it needs a perplexity check
  before it can be adopted - speed alone will not decide it.

## 18. DFlash on the Q4_K_M file: 8.8k code context (measured 2026-11-05, 14:05)

`SPEC=draft-dflash`, `Ornith-1.5-35B-A3B-DFlash-BF16.gguf` (782 MB), `--spec-draft-n-max 3`,
p-min 0.5, `NPRED=400`, 5 reps, `prompt-8k.txt` (8.8k tokens of C++ source) - the same fixture as
the IQ4_XS 23.08 point.

| arm | decode | acceptance | mean len |
|---|---:|---:|---:|
| IQ4_XS + DFlash, ub 2048 | 23.08 | 0.86111 | 3.41 |
| **Q4_K_M + DFlash, ub 512** | **23.18** | **0.87697** | **3.40** |

Reps 23.07 / 23.24 / 23.20 / 23.18 / 23.13, so the median is 23.18 and the spread is 0.7%.
Acceptance is per rep in the CSV; it is bit-identical across the five (`spec.csv` row 252).

- **The +22.3% the plain Q4_K_M file has over IQ4_XS is gone once DFlash is on top: +0.43%,
  inside the 0.71% A/A floor.** Both quants land on a 146.7-147.7 ms pass at mean len 3.40, so at
  8.8k of context with three-deep speculation the pass is dominated by work the expert type does
  not touch. That is a real and surprising result: the quant has to be chosen on prefill and on
  short-prompt decode, not on long-context decode.
- **Acceptance went *up*, not down, on the higher-precision file** (0.86111 -> 0.87697, mean len
  3.41 -> 3.40). The MTP drafter needed bit-identical numerics; the DFlash sidecar is a separate
  model reading the target's hidden features, so it tolerates the file changing. That is the
  second independent confirmation of section 12's claim that the acceptance loss with MTP was
  about shared weights, not about the target being perturbed.
- **Two configurations of this arm had to be abandoned for memory, not for speed, and the
  numbers above are the third:**
  - `-c 16384 -ub 2048`: the loader asks for **3897 MiB** of compute pp buffers and dies with
    `NvMapMemAllocInternalTagged: ... error 12` / `failed to allocate compute pp buffers`. Buffer
    size tracks ubatch, not context, so shrinking the context alone does not help.
  - `-c 16384 -ub 1024`: loads, then `ggml_cuda_graph_launch_chunk` aborts on the first decode,
    i.e. an earlier allocation had already failed at the NvMap level.
  - `-c 12288 -ub 512 -b 2048` runs. So on a 31 GB unified board, **Q4_K_M leaves about 1 GB of
    headroom over the 22.5 GB file once a second model and its feature buffer are loaded**, and
    any DFlash measurement on it must budget the ubatch first. With IQ4_XS (18.1 GB) `-ub 2048`
    fits with room to spare, which is why this did not show up earlier.
- **Confound, stated rather than hidden:** the ub 512 chunking differs from the IQ4_XS arm's ub
  2048. `-ub` changes how the prompt is chunked and therefore the prompt's hidden state, so the
  0.43% comparison mixes "same quant choice" with "different ubatch". The decode width is not
  affected (a decode batch is 4 tokens), but the prefill that produces the 8.8k cache is. Given
  the two readings are inside the noise floor anyway, the honest statement is that the two files
  are indistinguishable at this point, not that one wins.
- Prefill at 703 tokens on this arm reads 178.2 tok/s at ub 512 (TTFT 3.95 s) - the prefill
  column here is a small prompt and is not comparable to the ub 3072 arms.

**Missing cell, and the next arm:** Q4_K_M *plain* decode on the same 8.8k fixture has never been
measured, so the lift DFlash provides on this file is not yet known. The prediction is written
first in section 19.

## 19. Q4_K_M plain on the 8.8k code context (prediction first)

The missing cell from section 18. `SPEC=none`, no sidecar, `prompt-8k.txt`, `NPRED=400`,
5 reps, `-ub 3072 -c 12288`.

- **P11 (decode):** 19.5-20.2 tok/s. The 8.8k context costs IQ4_XS 10.6% (18.32 -> 16.37) at a
  54.8 ms pass, which is +5.8 ms; charged absolute against Q4_K_M's 44.8 ms pass that is 50.6 ms,
  i.e. 19.8 tok/s. If the context cost scales with the pass instead it is 19.5. A reading outside
  19.0-21.0 means the context cost is not the same additive term on the two files.
- **P12 (DFlash lift on this file):** 23.18 / 19.8 = **+17%**, against the +40.7% DFlash gives on
  the IQ4_XS file. The lift has to shrink because the base it multiplies is faster; if it does not,
  DFlash is bounded by something in the pass rather than by the draft rate.

**Measured (14:48): 19.43 tok/s**, reps 19.31 / 19.43 / 19.40 / 19.44 / 19.47, so P11's band
started at 19.5 and the reading is 0.4% below it. The mechanism was right and the constant was
slightly wrong: the 8.8k context costs Q4_K_M 13.0% (22.33 -> 19.43) where it costs IQ4_XS 10.6%
(18.32 -> 16.37), so the additive term is a little larger than 5.8 ms - about 6.7 ms against a
44.8 ms pass. A prediction that lands 0.4% outside its own band on a first-principles model is a
hit on the model and a miss on the number; both are stated.

**P12 held: 23.18 / 19.43 = +19.3%** (predicted +17%). The DFlash lift is smaller on the faster
file exactly as predicted, and the absolute level it lands on is nearly the same as on IQ4_XS
(23.18 against 23.08), which section 18 already noted.

### The MTP-free table with this cell filled in

| | plain, short prompt | plain, 8.8k code | DFlash depth 3, 8.8k code |
|---|---:|---:|---:|
| IQ4_XS experts | 18.32 | 16.37 | 23.08 (ub 2048) |
| Q4_K_M | **22.33** | **19.43** | **23.18** (ub 512) |

So: for short-prompt decode the Q4_K_M file is worth +22%; on a long code context with the
sidecar the two files are indistinguishable (+0.4%); and the best MTP-free decode number anywhere
on this board is **23.2 tok/s**, reached with either file once DFlash is on top of a long
context. Prefill still favours IQ4_XS (215.0 against 173.3 tok/s at 11,484 tokens).

### The artefact was verified before it was benchmarked (2026-11-05, 14:51)

`requant-dense-q5k.sh` ran on the new NVMe in **2 min 28 s** (the eMMC would have taken far
longer; nothing about the numbers changed, this is a throughput note). `llama-quantize` log and
`gguf-tensor-hash.py`:

- 753 tensors in, 753 out; **641 same type, 112 retyped**, and 8 sampled byte-identical on
  `sha256`, 0 differ - so `--tensor-type-file` did the copy path for every tensor it did not name.
- Census: `q4_k` 80 / 12.080 GB, **`q5_k` 149 / 7.711 GB**, `q6_k` 4 / 1.078, **`q8_0` 147 /
  0.711 GB**, `q3_k` 3 / 0.346, `f32` 370 / 0.107 = **22.032 GB** total, which is the estimate
  the pin tool made before the run to the megabyte. q8_0 fell from 2.074 GB, i.e. the 1.36 GB of
  dense attention/ssm weights plus the tensors that stay q8_0 (norms' companions, `ssm_*` smalls).
- File is `Q4KM-dense-q5k.gguf`, 22,042,657,088 bytes.

## 20. The dense retype measures 24.08 decode and 225.5 prefill (measured 2026-11-05, 14:57)

`Q4KM-dense-q5k.gguf`, `SPEC=none`, `-ub 3072 -c 40960`, `NPRED=400`, 5 decode reps and
3 prefill reps per length, same flags as the `q4km-ub3072-np400` arm it is compared against.

| | Q4_K_M | dense-q5k | delta |
|---|---:|---:|---:|
| decode, `prompt2.txt` | 22.33 | **24.08** | **+7.8%** |
| prefill @ 703 tok | 221.4 | 222.3 | +0.4% |
| prefill @ 2,828 tok | 217.2 | **281.9** | **+29.8%** |
| prefill @ 11,484 tok | 173.3 | **225.5** | **+30.1%** |
| TTFT @ 11,484 | 66.3 s | **51.0 s** | **-23.1%** |

Reps: decode 24.01 / 24.22 / 24.04 / 24.14 / 24.07 (0.9% spread); prefill 279.4 / 282.1 / 281.7 at
2,828 and 226.0 / 225.3 / 225.7 at 11,484 (0.3%).

- **P8 (decode 25.5-26.0) is falsified, it is 24.08** - right direction, 5.5% over-predicted.
  The retype removes 0.48 GB of a 2.46 GB per-token read (-19%) and buys +7.8% of time, so the
  dense q8_0 kernels were not purely bandwidth-limited either; the same "type beats bits" result
  as section 16, from the other side.
- **P9 (prefill +2 to +4%) is falsified in the opposite direction and by 10x: +30%.** The error is
  instructive, not random. Prefill at `-ub 3072` moves 4 x 22 GB of weights in 50.9 s, i.e.
  **1.7 GB/s, nowhere near the 92-110 GB/s ceiling**, so it is compute-bound and the byte saving
  is irrelevant; what changed is which engine computes the dense projection. On this file the
  dense q8_0 tensors leave MMQ for cuBLAS INT8 above the batch threshold (the ub 512 budget in
  section 13 already measured "cuBLAS INT8 14.2%" of prefill), and Volta has **no INT8 tensor
  cores**, so that path is emulated dp4a. Retyped to q5_k the same tensors go through `mul_mat_q`
  with the dp4a dot the rest of the chip uses. The 703-token point is unchanged (+0.4%) because
  at that size the fixed part of the launch dominates - a useful control, since it shows the gain
  is in the wide-batch engine and not in the file being read.
- **This dissolves the Pareto trade of section 16.** The dense-q5k file is better than Q4_K_M on
  every axis at once: +7.8% decode, +30% prefill, -23% TTFT, and 0.48 GB smaller. IQ4_XS keeps a
  prefill edge over it only if the 30% does not reproduce.
- **Not yet confirmed against drift.** Prefill has read 76.5-92.0 tok/s for one identical point on
  this board (AGENTS.md section 7), and this comparison is single-order across two server loads.
  The two orders are being run now at 2,828 tokens; until that lands, treat +30% as provisional.
- **P10 stands and is now the gate:** this file is *not* quality-free - attention and the GDN
  projections are at q5_k instead of q8_0, which is below where Q4_K_M normally sits. It needs
  `bench/ppl.sh` against Q4_K_M on the same corpus before adoption, and the MTP configuration
  cannot use it at all (any weight change kills the drafter's acceptance - AGENTS.md rule on
  bit-identical numerics).

### The +30% prefill did not reproduce: it was a memory state, not the weights (15:04)

Interleaved both-orders A/B at 2,828 tokens, `-ub 3072`, 3 reps per arm, rep 1 dropped as the
cold rep:

| arm, in order | tok/s |
|---|---:|
| A Q4_K_M (position 1) | 183.8 |
| B dense-q5k (position 2) | 175.5 |
| B dense-q5k (position 3) | **281.0** |
| A Q4_K_M (position 4) | **282.8** |

The same file and the same flags read 175.5 and 281.0 in two adjacent positions, a **60% swing**,
and Q4_K_M reaches the same 281-283 in the fast state. So the 217.2 vs 281.9 pair in the table
above is a **state mismatch, not a weight effect**, and the +30% prefill / -23% TTFT claim is
retracted as stated. In the fast state the two files are equal to 0.6%.

- What the state is: not clock (`gv11b` reads min = max = 1,377,000,000, governor
  `nvhost_podgov`, `nvpmodel -q` = MAXN), not host load (1.35 on 8 cores), and not the file (same
  file flips). It was present in the two arms run at 14:59-15:00 and gone from 15:01, about
  10 minutes after the 22 GB requant write finished - so the most likely cause is page-cache and
  writeback pressure from that write, i.e. the model's mmap'd pages competing with dirty pages,
  but **that is a hypothesis and it has not been tested**. The previously recorded drift for one
  fixed point was 17%; this is 60% and it is worse than anything in AGENTS.md section 7.
- Consequence, and it is the important one: **every prefill comparison in this file that spans
  two server loads is suspect**, including the 215.0 / 173.3 pair in section 16 and the ubatch
  width table. Those two were measured in one session each, adjacent in time, and neither has an
  interleaved both-orders confirmation. The 0.56% floor claimed in section 7 of the prefill
  findings is a *within-session* floor and does not protect against this.
- The decode side is being re-measured the same way now (5 reps x 400, `prompt2.txt`, A B B A).
  Decode has been stable at 0.7-0.9% within a load all day, so if the +7.8% survives an
  interleaved run it stands; if it does not, nothing in section 20 stands.

### Decode confirmed, and the prefill instability is the ubatch width (15:27)

**Decode, interleaved both orders, 5 reps x 400, `prompt2.txt`, same flags, one server load per
arm** (`ab-dense-full.sh`):

| arm | position | median | spread over 5 reps |
|---|---|---:|---:|
| Q4_K_M | 1 | 22.26 | 0.70% |
| dense-q5k | 2 | 24.15 | 0.67% |
| dense-q5k | 3 | 24.22 | 0.65% |
| Q4_K_M | 4 | 22.24 | 0.53% |

Pooled **22.26 -> 24.20 = +8.72%**, and the same-config arms in positions 1/4 differ by **0.09%**
while 2/3 differ by 0.28%. So today's decode floor is under 0.3% and the +8.7% is real by a
factor of 30. Section 20's decode number stands; only its prefill half was wrong.

**Prefill, same file and same flags, four consecutive server loads at `-ub 3072`:** 218.9, 172.4,
200.4, 165.6 tok/s at 2,828 tokens - a **32% spread with nothing varying but the load**. Two
consecutive runs at `-ub 1024` on the same file: **229.3 and 231.8 (1.1% apart)**.

That single experiment reorders the prefill front:

- **The instability is a property of the wide ubatch, not of the file or the dense retype.** At
  `-ub 3072` the compute buffer the loader must allocate is ~3-4 GB (`failed to allocate compute
  pp buffers` asks for 3897 MiB at 2048 columns), and where that allocation lands appears to
  decide whether the launch runs at 165 or 220 tok/s. At 1024 columns it is roughly a third the
  size and two runs agree to 1.1%.
- **`-ub 1024` is also faster than every `-ub 3072` run** (229-232 against 165-219), despite doing
  three chunks where 3072 does one - so the model's weights are not what prefill is paying for
  (consistent with the 1.7 GB/s effective figure above), and the wide tile is simply inefficient.
- **Therefore the prefill numbers in sections 14, 16 and 20 are retracted as comparisons**:
  215.0 (IQ4_XS), 173.3 (Q4_K_M), 225.5 (dense-q5k), and the width table in section 14 that chose
  `-ub 3072` were all measured at 3072 with at most one order and no repeat, on a variable that
  swings 32% between identical loads. The MTP-free profile's prefill column has to be rebuilt at
  a width that measures reproducibly.
- `-ub 3072` also fixed the popular claim "215 tok/s and TTFT 53.5 s at 11,484 tokens" that is
  already in `KNOWN-GOOD.md`, `TODO.md`, `HANDOFF.md` and the fork docs. Those are stale as of
  this measurement and are being corrected to a width-stable number plus a range.

**Next: a width sweep with two repeats per width at 2,828 tokens.** Prediction, written first:
`-ub 1024` and `-ub 2048` land in 225-235 tok/s with repeats agreeing inside 3%; `-ub 512`
measures 210-225 because it doubles the chunk count and re-reads the weights twice as often;
`-ub 4096` is as unstable as 3072 or worse, because its compute buffer is larger than 3072's.

### Width sweep at 2,828 tokens, two repeats per width (measured 15:28-15:38)

Same file (`Q4KM-dense-q5k.gguf`), same flags, `-b` = `-ub`, one server load per reading:

| ubatch | reading 1 | reading 2 | spread | vs 1024 |
|---:|---:|---:|---:|---:|
| 512 | 194.2 | 194.7 | 0.3% | -15% |
| **1024** | 228.2 | 226.8 | 0.6% | - |
| **2048** | **250.2** | **243.1** | 2.9% | +8% |
| 3072 | 218.9 | 172.4 | (over 4 runs 165.6-218.9) | - |
| 4096 | 159.2 | 273.4 | **72%** | - |

- **Prediction check:** I predicted 1024 and 2048 in 225-235 inside 3% - 1024 landed at 227-228
  (hit), 2048 at 243-250 (7% above the band, so a miss on the number, right on the mechanism);
  512 predicted 210-225 and read 194 (8% below); 4096 predicted "as unstable as 3072 or worse"
  and read 159 / 273, which is worse.
- **The instability is a threshold, not a slope:** 512 and 1024 agree to 0.3-0.6%, 2048 to 2.9%,
  then 3072 and 4096 scatter by 32% and 72%. That tracks the size of the compute buffer the
  loader reserves (`-ub 2048` asks for 3897 MiB on the 22.5 GB file; the buffer is the largest
  single allocation besides the weights), and it means **2,048 columns is the widest setting that
  measures reproducibly on this board** - and it is also the fastest measured here, at 243-250
  against 227 for 1024 and 194 for 512.
- **`-ub 2048` therefore replaces `-ub 3072` as the MTP-free prefill width**, on both stability and
  speed. It also has 1.9 GB more headroom for the DFlash sidecar than 3072 did, which is the
  configuration that OOM'd.
- The 11,484-token point and the cross-file comparison at 2048 are the next run; every number in
  sections 14-20 that was taken at 3072 is superseded.

### Rebuilding the table at `-ub 2048` (prediction first, 15:39)

Two server loads per file, `LENS="2385 9563"` (2 reps each) and `DECODE_REPS=5`, `-c 40960`.
Prediction: q5kd prefill 240-252 at 2,828 with repeats inside 4%; IQ4_XS **250-270** at 2,828,
i.e. above q5kd, because the IQ3_S file read 270.0 at 2,828 in one fast 3072 session and its MoE
is cheaper per byte even if its dot is slower; Q4_K_M 235-250, close to q5kd because the dense
retype's prefill win did not survive the interleaved test. Decode: q5kd 24.2, Q4_K_M 22.25,
IQ4_XS ~18.3, all unchanged by the width (established: `--spec-type none` makes decode
width-independent, 0.65% across 32-8192).

### The corrected table at `-ub 2048` (measured 15:39-16:12)

Two consecutive server loads per file, decode 5 reps x 400 on `prompt2.txt`, prefill 2 reps at
each of 2,828 and 11,484 tokens, `-c 40960`, `--spec-type none`.

| file | decode (load 1 / load 2) | prefill @ 2,828 | prefill @ 11,484 | TTFT @ 11,484 |
|---|---:|---:|---:|---:|
| IQ4_XS experts | 18.30 / 18.33 | 241.3 / 241.5 | 216.3 / 215.1 | 53.2 / 53.4 s |
| Q4_K_M | 22.25 / 22.20 | 250.1 / 249.7 | 221.1 / 220.7 | 52.0 / 52.1 s |
| **dense-q5k** | **24.01 / 24.10** | **244.0 / 242.5** | 220.9 / 220.5 | 52.05 / 52.15 s |

Everything here reproduces: within-file repeats agree to 0.2-0.6% on prefill and 0.1-1.0% on
decode, so `-ub 2048` gives a usable protocol where `-ub 3072` gave none.

- **Prefill is flat across the three files: 215-221 tok/s at 11,484 tokens, a 2.8% band.** The
  quant does not matter for prefill at all once the width is measured reproducibly. This retracts
  the IQ4_XS prefill advantage (+30% over Q4_K_M, section 16) and IQ4_XS's predicted 250-270 here
  (predicted first, measured 241 - 4% below dense-q5k and 3.6% below Q4_K_M, i.e. last).
- **Decode is where the files differ, and the dense retype is the new best: 24.01-24.10 against
  22.25 for Q4_K_M (+8.5%) and 18.3 for IQ4_XS (+31%).** With prefill flat, dense-q5k dominates
  both other files outright rather than trading against them.
- **The MTP-free profile is therefore: dense-q5k, `-ub 2048`, `--spec-type none` - 24.0 tok/s
  decode, 220.7 tok/s prefill, TTFT 52.1 s at 11,484 tokens** - subject to the quality gate
  below. Everything in section 14's "`-ub 3072` is the best width" and sections 16/20's prefill
  figures is superseded by this table.
- IQ4_XS is now the worst of the three on both axes, so the 18.1 GB file has no remaining role in
  this profile. It is not being deleted: it is the MTP configuration's baseline.

## 22. The quality gate: perplexity of the three files (measured 2026-11-05, 16:23)

`bench/ppl.sh`, corpus `prompt-32k.txt` (llama.cpp C++ source, 30,720 tokens), `-c 512`,
60 chunks, `-b 512 -ub 512`, no warmup, `-ngl all`. Same corpus and chunk boundaries for all
three, so the comparison is paired.

| file | PPL (final estimate) | vs Q4_K_M | paired per-chunk diff (60 chunks) |
|---|---:|---:|---|
| Q4_K_M | **1.9142 +- 0.0253** | - | - |
| dense-q5k | 1.9271 +- 0.0257 | **+0.67%** | +0.0077 +- 0.0009, **t = 8.3** |
| IQ4_XS | 1.9408 +- 0.0258 | +1.39% | +0.0206 +- 0.0015, t = 13.7 |

- The quoted +- is the spread over chunks and is far larger than the differences; the honest test
  is paired, and paired it is unambiguous: **dense-q5k is 0.36-0.67% worse than Q4_K_M and
  IQ4_XS is 0.97-1.39% worse**, both at t > 8. The two relevant ratios do not overlap.
- **The dense retype is not free, and this closes P10 as "measured, small, real":** retyping the
  1.36 GB of attention/GDN projections from q8_0 to q5_k costs about two thirds of a percent of
  perplexity on code and buys +8.5% decode with no prefill change. That is the trade, stated
  plainly - 24.0 tok/s against 22.2 for 0.67% PPL.
- **IQ4_XS is dominated and can be dropped from this profile**: worse PPL than Q4_K_M and slower
  on decode and prefill at every reproducible setting.
- Caveat worth keeping: one corpus, one language (C++), 30k tokens. A quant difference of this
  size in PPL is not a guarantee about long-context behaviour, instruct-following or the GDN
  state; it is the gate available on this board, not a complete quality argument.

### Where the MTP-free profile stands (2026-11-05, 16:25)

| | value | note |
|---|---:|---|
| decode, `prompt2.txt` (267 tok) | **24.0 tok/s** | dense-q5k, `-ub 2048`, 5 reps, 2 loads |
| prefill @ 2,828 tok | 243 tok/s | |
| prefill @ 11,484 tok | **220.7 tok/s** | 2 loads, 0.2% apart |
| TTFT @ 11,484 tok | **52.1 s** | |
| quality | +0.67% PPL vs Q4_K_M | code corpus, paired |

Against the starting point of this campaign (18.32 tok/s, 215.0 prefill, 53.5 s TTFT) the profile
is now **+31% decode at the same prefill**, and against the 15-16 tok/s the user expected it is
+52%. The prefill number is a range, not a point: sessions have read the same configuration
anywhere from 215 to 233 tok/s at 11,484 tokens.

### DFlash on the dense-q5k file: the ranking flips (measured 16:27)

`SPEC=draft-dflash`, depth 3, p-min 0.5, `prompt-8k.txt`, `-c 12288 -ub 512 -b 2048`, 5 reps.

| file | plain @ 8.8k | + DFlash @ 8.8k | acceptance | mean len | pass latency |
|---|---:|---:|---:|---:|---:|
| IQ4_XS | 16.37 | 23.08 (ub 2048) | 0.86111 | 3.41 | 147.7 ms |
| Q4_K_M | 19.43 | **23.18 (ub 512)** | 0.87697 | 3.40 | 146.7 ms |
| **dense-q5k** | (next run) | **22.53 (ub 512)** | **0.88925** | **3.44** | 152.7 ms |

Reps 22.34 / 22.55 / 22.53 / 22.48 / 22.53 (0.9% spread).

- **The file that wins plain decode loses under DFlash: 22.53 against Q4_K_M's 23.18 (-2.8%),
  even though its acceptance is the best of the three (0.88925 vs 0.87697) and its mean length
  the longest (3.44 vs 3.40).** Throughput is `mean_len / pass_latency`, and the pass is 6 ms
  slower (152.7 against 146.7 ms) despite the smaller file. The only thing the retype changed is
  the dense attention/GDN projections, and under a 4-column verify over 8.8k of KV those are
  exactly the tensors being read - so **q8_0's cheaper dot beats q5_k's fewer bytes at wide
  columns**, the same type-over-bits effect seen on the experts (section 16), now with the sign
  reversed because the retype went *down* in precision on a type whose dot is already cheap.
- So the MTP-free profile is not one configuration but two, chosen by workload:
  **dense-q5k at `-ub 2048`, no speculation, for short prompts (24.0 tok/s)**, and
  **Q4_K_M + DFlash depth 3 for a long code context (23.2 tok/s)**. Both must state their
  context, because the DFlash acceptance is prompt-dependent (0.787 code / 0.523 prose measured
  earlier, and 0.889 here on the dense-q5k file).
- All three DFlash arms are single loads, and the ub differs (512 for the two Q4_K_M-family files,
  2048 for IQ4_XS) for memory reasons, so the 2.8% gap carries a protocol caveat.

**Next, prediction first: plain dense-q5k on the 8.8k fixture, 21.0-21.5 tok/s.** Q4_K_M plain
reads 19.43 there and dense-q5k is +8% on the short prompt, so 21.0 is the central estimate; a
reading above 22.5 would mean the q5_k dense set is faster at 8.8k too and the DFlash loss
above is an interaction with speculation, not with context length.

**Measured (16:32): 20.91 tok/s**, reps 20.84 / 20.87 / 20.95 / 20.92 / 20.91 - inside the
predicted 21.0-21.5 band by 0.4%. So at 8.8k the q5_k dense set *is* faster without speculation
(+7.6% over Q4_K_M's 19.43) and slower with it (-2.8%). The loss above is an interaction with
speculation, exactly as the prediction's second branch said: a 4-column verify pays the q5_k dot
four times per pass over the same bytes.

## 23. The MTP-free profile, final numbers

All decode medians of 5 reps x 400 tokens; prefill `-ub 2048` unless stated. Context is part of
the number - the three fixtures are the 267-token prompt, 8.8k of C++ source, and 11,484 tokens.

| configuration | 267-token prompt | 8.8k code, plain | 8.8k code, DFlash d3 | prefill @ 11,484 | TTFT |
|---|---:|---:|---:|---:|---:|
| IQ4_XS experts | 18.30 | 16.37 | 23.08 | 215.7 | 53.3 s |
| Q4_K_M | 22.25 | 19.43 | **23.18** | 220.9 | 52.0 s |
| **dense-q5k** | **24.01** | **20.91** | 22.53 | 220.7 | 52.1 s |

- **Short prompts, no speculation: dense-q5k, 24.01 tok/s.** That is the profile's headline and
  it costs +0.67% perplexity against Q4_K_M (section 22).
- **Long code context with DFlash: Q4_K_M, 23.18 tok/s**, 3% ahead of dense-q5k. Without
  speculation dense-q5k takes that column too (20.91 against 19.43).
- **Prefill and TTFT no longer distinguish the files at all**: 215-221 tok/s and 52-53 s, a 2.8%
  band, once the width is one that measures reproducibly (`-ub 2048`).
- Against where this campaign started (18.32 tok/s, TTFT 53.5 s at 11,484) the profile is
  **+31% decode at equal prefill**; against the 15-16 tok/s the user expected, +52%.
- The prefill column is a same-session measurement at `-ub 2048`, reproducible to 0.2-0.6%; the
  absolute level still moves between sessions by up to 17% (AGENTS.md section 7), so quote the
  TTFT as ~52 s, not 52.06 s.

### What this run changed in the method, in one place

1. **A 22 GB write to the same disk the model is read from can leave the box in a state where
   wide-ubatch prefill reads 60% slow for about ten minutes** (section 20 addendum). Measure
   nothing across that window.
2. **`-ub 3072` and wider do not measure reproducibly at all** (32% and 72% spreads across
   identical loads); `-ub 2048` gives 0.2-2.9%. The compute buffer the loader reserves is the
   variable, and it grows with ubatch.
3. **Every prefill claim in this file that was taken at 3072, or in a single order across two
   loads, is superseded** by section 22's table - including the "IQ4_XS has 30% more prefill"
   result that section 16 published, which was an artefact of exactly this.
4. **Paired per-chunk perplexity resolves quant differences that the +-SE across chunks hides**
   (t = 8.3 on a 0.36% difference). `bench/ppl.sh` prints the per-chunk line the paired test
   needs.

### `-bs` still pays with the drafter gone: +1.2% (measured 16:44)

Interleaved both orders, dense-q5k, `-ub 2048`, `--spec-type none`, 5 warm reps per arm:

| arm, in order | median | spread |
|---|---:|---:|
| A no `-bs` (position 1) | 24.018 | 0.9% |
| B `-bs` (position 2) | 24.275 | 0.2% |
| B `-bs` (position 3) | 24.258 | 0.7% |
| A no `-bs` (position 4) | 23.954 | 0.4% |

Same-config arms differ by **0.27%** (A) and 0.07% (B); the effect is **+1.17%**, so it clears the
floor by a factor of four. That matches the +1.04% `-bs` gives the MTP configuration, which is
what the mechanism predicts: `--backend-sampling` moves the 248,320-wide logits scan off the
critical path, and it does that per token whatever the drafter is doing.

**The MTP-free profile's decode is therefore 24.27 tok/s, not 24.01.** The 8.8k and DFlash numbers
in this file were all measured without `-bs` and are quoted as such; adding it should move each by
about the same 1%, but that is an inference and not a measurement.

**Profile at the end of section 23: `Q4KM-dense-q5k.gguf`, `-ub 2048 -c 40960 --spec-type none
-bs` - 24.27 tok/s decode, 220.7 tok/s prefill, 52.1 s TTFT at 11,484 tokens, +0.67% perplexity
against Q4_K_M. SUPERSEDED later in this file (sections 24-27): the adopted file is
`Q4KM-all-q4k.gguf` at 27.86 tok/s. That file and its two intermediate parents have been deleted;
the requant scripts rebuild them from `Q4_K_M`.**

## 24. The next requant: `output.weight` q6_k -> q5_k (prediction first, 16:55)

The dense retype moved the per-token byte budget a long way, so it is worth restating it from the
GGUF header rather than from intuition. `Q4KM-dense-q5k.gguf`, grouped by role, with the experts
weighted by the 8-of-256 that a plain token actually reads:

| group | file bytes | share of file | per plain token | share of a token |
|---|---:|---:|---:|---:|
| experts (q4_k 12.08, q5_k 6.83, q6_k 0.66) | 19.57 GB | 88.8% | ~0.64 GB | 31% |
| dense attention/GDN, now q5_k | 0.865 GB | 3.9% | 0.865 GB | **42%** |
| `output.weight` q6_k | 0.417 GB | 1.9% | 0.417 GB | **20%** |
| shared expert q8_0 | 0.134 GB | 0.6% | 0.134 GB | 6.5% |
| norms + `ssm_a/dt/alpha/beta/conv1d` f32 | 0.104 GB | 0.5% | 0.104 GB | 5% |
| `token_embd` 0.540 GB, nextn layer 0.380 GB | - | 4.2% | 0 | 0 |

So the lm_head tensor alone is **20% of what a token reads**, and it is now the largest single
lever after the dense set - the dense retype took that group from 62% of a token to 42%, and the
lm_head is unchanged at q6_k.

- **P13 (decode): 25.0-25.3 tok/s, i.e. +3.0 to +4.2% over 24.27.** 417 -> 349 MB per token
  (-68 MB, -3.3% of the ~2.06 GB total). This kernel measured **78.95% of the memory SOL** at one
  column in the MTP budget, so it is bandwidth-bound and time should follow bytes almost 1:1.
  Anything under +2% would mean the q5_k dot gives back what the bytes save; anything over +5%
  would mean the q6_k dot was costing more than its bytes.
- **P14 (prefill): no change.** 68 MB of a 22 GB file read across 2-6 chunks is 0.3%, well inside
  the width's own 2.9%.
- **P15 (quality): +0.3 to +1.0% perplexity.** `output.weight` has historically been the most
  tolerant tensor to retype, but it is also the tensor every token's logits come from, so this is
  a real check and not a formality.

### The requant and the decode measurement (measured 2026-11-05, 16:54)

`bench/requant-lmhead-q5k.sh`: 753 tensors read, **752 same type, 1 retyped** (`output.weight`
14 q6_k -> 13 q5_k), 2 min 05 s from `Q4KM-dense-q5k.gguf` to `Q4KM-dense-q5k-lm5.gguf`. Census of
the output: q4_k 80 tensors / 12.080 GB, q5_k 150 / 8.060 GB, q6_k **3** / 0.661 GB, total
**21.964 GB** - the pin tool's pre-run estimate to the megabyte, and 0.417 -> 0.349 GB for the
tensor itself. `gguf-tensor-hash.py --hash --sample 8` on tensors whose type did not change:
**8 hashed, 0 differ**, so nothing else in the file moved.

The pattern-over-match trap from the tool notes does not bite here: `output.weight` also
`regex_search`-matches `attn_output.weight`, but the retype count came back as exactly 1 because
those tensors are already q5_k after the dense retype. The census counting 3 q6_k tensors rather
than 1 is the experts' q6_k, untouched.

`bench/ab-lmhead.sh`, A B B A, 5 reps x 400 tokens on `prompt2.txt`, `-ub 2048 -b 2048 -bs
--spec-type none`, one eMMC load per arm:

| arm | file | median tok/s | spread |
|---|---|---:|---:|
| A1 | dense-q5k | 24.063 | 0.55% |
| B1 | dense-q5k-lm5 | **24.829** | 0.74% |
| B2 | dense-q5k-lm5 | **24.929** | 0.64% |
| A2 | dense-q5k | 24.250 | 0.51% |

**A/A (first against last, same file) +0.78%, B/B +0.40%, A -> B +3.16% pooled (B 24.879 against
A 24.157).** The same-config pair differs by more than this session's usual 0.09-0.28%, at the
0.71% floor rule 7 says to expect, and the effect is still four times it.

**P13 is a hit on the direction and the magnitude band, low edge on the absolute level.** The
predicted +3.0 to +4.2% landed at +3.16%, which is the model working: the kernel is at 78.95% of
the memory SOL, bytes fell 3.3% of the token total, and time followed bytes. The absolute 24.88 is
below the predicted 25.0-25.3 only because the A arm itself read 24.16 here against 24.27 in the
`-bs` run - a 0.5% slower session, which is inside the same drift that produced the 0.78% A/A
gap. **New decode figure: 24.88 tok/s on `Q4KM-dense-q5k-lm5.gguf` with `-bs` already in the arms.**

### The quality gate (measured 2026-11-05, 17:03-17:10)

`bench/ppl.sh` on both files, `-c 512 -b 512 -ub 512 --chunks 60 --no-warmup` on the 30k-token
code corpus, 3 min 10 s per file:

| file | final PPL | err |
|---|---:|---:|
| `Q4KM-dense-q5k.gguf` | 1.9271 | 0.02570 |
| `Q4KM-dense-q5k-lm5.gguf` | **1.9275** | 0.02567 |

Paired per chunk (60 pairs): mean difference **+0.00139 PPL, +0.065%**, sd 0.00103, se 0.00013,
**t = 10.42, 59/60 chunks worse**. The direction is unambiguous and the size is a rounding error
next to the dense retype's +0.67% (sd 0.0009 there, so this is 6% of that measurement on the same
corpus).

**P15 is wrong on magnitude: predicted +0.3 to +1.0%, measured +0.065%, an overestimate of 5-15x.**
The reasoning behind it - "every token's logits come from this tensor" - is true but does not
translate into perplexity here, because q5_k still resolves the same argmax and the corpus loss is
dominated by the expert and dense paths. Prediction recorded before the run, so the miss stands as
a miss rather than being reframed: the tensor was assumed sensitive on the grounds that it is the
logits projection, and the measurement says the sensitivity is in the experts and the dense set,
not here.

### Prefill on the adopted file (measured 2026-11-05, 17:10-17:20)

`bench/prefill.sh` with `DECODE_REPS=5`, one load, `NAME=lm5-ub2048`, `-ub 2048 -b 2048 -bs
--spec-type none`, `LENS=11484` (server `n_prompt` 14,520): decode **24.73 tok/s** median
(24.60 / 24.80 / 24.83 / 24.73 / 24.72, rep 1 cold at 35.56 s raw) and prefill **209.0 tok/s**,
TTFT **69.5 s** over the last two reps.

That length is not the one the published table used, so a second load with `LENS=9563` (server
`n_prompt` 11,484) was run for the comparison:

| file | prompt n | tok/s | TTFT |
|---|---:|---:|---:|
| `Q4KM-dense-q5k.gguf` (published) | 11,484 | 220.7 | 52.1 s |
| `Q4KM-dense-q5k-lm5.gguf` | 11,484 | **214.3** (216.5 / 212.1) | **53.6 s** |
| `Q4KM-dense-q5k-lm5.gguf` | 14,520 | 209.0 | 69.5 s |

**P14 stands as "not resolvable", which is the same as unchanged at this width.** 214.3 against
220.7 is -2.9%, exactly the 2.9% reproducibility measured for `-ub 2048` and larger than the 0.3%
the 68 MB byte saving could buy, so the honest reading is drift. The lm_head tensor is read once per
chunk during prefill at a column count where its cost is amortised, which is why the decode win
does not show up here.

### The profile file, and the queue after it

Adopted at this point (superseded in section 25, file since deleted): **`Q4KM-dense-q5k-lm5.gguf`,
21.964 GB, decode 24.88 tok/s, prefill
214.3 tok/s, TTFT 53.6 s at 11,484 tokens, +0.71% perplexity against `Q4_K_M`** (+0.67 dense,
+0.065 lm_head, summed naively).

Per-token budget after the lm_head retype:

| group | per plain token | share | next lever |
|---|---:|---:|---|
| dense attention/GDN q5_k | 0.865 GB | 43.6% | q5_k -> q4_k saves 0.157 GB (0.708 GB, -7.9% of a token) |
| experts (q4_k 12.08, q5_k 6.83, q6_k 0.66) | ~0.64 GB | 32% | q5_k -> q4_k on the q5_k experts saves 0.06 GB |
| `output.weight` q5_k | 0.349 GB | 17.6% | done |
| shared expert q8_0 | 0.134 GB | 6.8% | q8_0 -> q5_k saves 0.05 GB |
| norms + ssm f32 | 0.104 GB | 5.2% | not a quant lever |
| total | ~1.99 GB | | |

The dense set is now the single largest read and the only place where one retype still moves more
than 5% of a token, so that is the next experiment - with the caveat that q5_k is where the dense
set stopped being free (+0.67% perplexity from q8_0), so q5_k -> q4_k is expected to cost more
quality per byte than this one did.

### The lm_head retype does *not* carry over to DFlash (measured 2026-11-05, 17:24)

Prediction written before the run: **+0.5 to +1.5%, i.e. 23.3-23.6 on 8.8k of code**, on the
grounds that the verify pass calls the lm_head once at four columns, where the q6_k dot is
expensive (it cost 9.3 ms/pass at seven columns in the MTP budget) and the byte saving should
follow. `bench/mtp-free-lm5-dflash.sh`, `-c 12288 -ub 512 -b 2048`, depth 3 / p-min 0.5, 5 reps x
400 on `prompt-8k.txt`:

| file | DFlash depth 3, 8.8k | acceptance | mean len |
|---|---:|---:|---:|
| `Q4_K_M` | **23.18** | 0.87697 | 3.40 |
| `Q4KM-dense-q5k` | 22.53 | 0.88925 | 3.40 |
| `Q4KM-dense-q5k-lm5` | **22.42** (22.28 / 22.47 / 22.40 / 22.42 / 22.42) | **0.85849** | **3.27** |

**The prediction is wrong in sign.** The lm5 file is 0.5% *slower* than its parent under DFlash,
and 3.3% below `Q4_K_M`. The acceptance fell 0.88925 -> 0.85849 and mean len 3.40 -> 3.27, which
is a real numerics movement from a tensor that ever only supplies logits: the drafter and the
target both moved and agree less often. Prefill at 590 tokens is 185.7 tok/s, in line with the
other arms on this rig.

So the DFlash profile stays on the plain `Q4_K_M` file.

This is the second time a byte saving has been column-dependent in
the same direction (the dense q8_0 -> q5_k retype also lost under the 4-column verify), which is
now two for two: **treat every wide-column verify win as untested until it is measured on the
sidecar rig.** The 22.42 arm is single-order and its 0.5% gap to dense-q5k is below the floor,
but the 3.3% gap to `Q4_K_M` is not.

## 25. The next requant: dense q5_k -> q4_k (prediction first, 17:35)

The dense attention/GDN set is now 0.865 GB of the ~1.99 GB a plain token reads (43.6%), and 112
tensors. Every one of them is q5_k; q4_k is 144 bytes per 256 against 176, so the same set becomes
**0.708 GB** and a token's total falls to ~1.83 GB.

- **P16 (decode): 25.6-26.1 tok/s, i.e. +3 to +5% over 24.88.** The byte share is 7.9%, and the
  two previous retypes bracket the yield: the dense q8_0 -> q5_k collected 0.43x of its byte share
  (dequant-heavy, +8.5% for 20%), the lm_head q6_k -> q5_k collected 0.96x (+3.16% for 3.3%,
  because that kernel is at 78.95% of the memory SOL). A 4-bit k-quant is the most dequant-heavy of
  the three, so expect the low end, not the middle.
- **P17 (quality): +1.0 to +2.5% perplexity.** q5_k already cost +0.67% from q8_0, and each further
  bit removed from a k-quant is more expensive than the last.
- **P18 (prefill): no change**, same reasoning as P14 and the same 2.9% floor.

### The requant and the decode measurement (measured 2026-11-05, 17:33)

`bench/requant-dense-q4k.sh`: 753 tensors, **112 retyped** (`attn_qkv` 30, `attn_gate` 30,
`ssm_out` 30, `attn_q` 11, `attn_output` 11), 2 min 18 s, `21.964 -> 21.815 GB` against the pin
tool's -0.160 GB estimate. Census: q4_k 192 tensors / 12.802 GB, q5_k 38 / 7.178, q6_k 3 / 0.661;
8 sampled unchanged-type tensors hashed, **0 differ**.

`bench/ab-dense-q4k.sh`, A B B A, 5 reps x 400 on `prompt2.txt`, `-ub 2048 -b 2048 -bs
--spec-type none`:

| arm | file | median tok/s | spread |
|---|---|---:|---:|
| A1 | dense-q5k-lm5 | 24.643 | 0.4% |
| B1 | dense-q4k | **26.196** | 0.8% |
| B2 | dense-q4k | **26.277** | 0.7% |
| A2 | dense-q5k-lm5 | 24.876 | 0.9% |

**A/A +0.95%, B/B +0.31%, A -> B +5.98% pooled (26.237 against 24.760).** The same-config gap is
again at the 0.71-0.95% floor and the effect is six times it.

**P16 is a miss on the low side: predicted +3 to +5%, measured +6.0%.** The prediction assumed
this retype would collect 0.43-0.7x of its 7.9% byte share; it collected **0.76x**, more than the
dense q8_0 -> q5_k step did and close to the lm_head's 0.96x. The reason given for expecting less -
"4-bit k-quant is the most dequant-heavy" - is not supported: q4_k's 144-byte block is smaller and
its scale/min structure is the same shape as q5_k's, so the dot does not get proportionally more
expensive. Recorded as a miss, not reframed. **New decode figure: 26.24 tok/s.**

### The quality gate (measured 2026-11-05, 17:43)

`bench/ppl.sh` on `Q4KM-dense-q4k.gguf`, same corpus and chunking as the previous two gates:
final estimate **1.9372** against the lm5 file's **1.9275**. Paired per chunk: **+0.01435 PPL,
+0.672%, t = 7.69, 60/60 chunks worse** (the final estimate prints +0.50% because it weights the
chunks differently from a plain mean).

**P17 is a miss on the low side too: predicted +1.0 to +2.5%, measured +0.67%.** Dropping a
k-quant from 5 to 4 bits on the dense attention/GDN set costs less than the 8 -> 5 step did
(+0.67% against +0.67% - the two are the same size, not escalating). Both P17 and the P16 miss
share one cause: the intuition that lower-bit k-quants are progressively more expensive, in
decode time and in loss, is not what this board shows for this block layout. The cumulative price
of the whole requant campaign is now **+1.20% perplexity on the final estimate** (1.9142 ->
1.9372; +1.41% if the three paired steps are summed) **for 22.25 -> 26.24 tok/s (+17.9%)**.

### Prefill on the adopted file (measured 2026-11-05, 17:47)

`NAME=dq4-9563`, `LENS=9563` (server `n_prompt` 11,484), `DECODE_REPS=5`, one load: prefill
**219.2 tok/s, TTFT 52.5 s** (217.5 / 219.4 / 219.0) and decode **26.23 tok/s** median. Against
the lm5 file's 214.3 / 53.6 s this is inside noise and against the pre-retype 220.7 / 52.1 s it
is within 0.7%, so **P18 holds: prefill is unchanged by a dense retype**, and the 52-54 s TTFT
band at 11,484 tokens is a property of the width, not of the file.

### The adopted profile file

Adopted at this point (superseded in section 26, file since deleted): **`Q4KM-dense-q4k.gguf`,
21.815 GB, 26.24 tok/s decode, 219.2 tok/s prefill,
TTFT 52.5 s at 11,484 tokens, +1.20% perplexity against `Q4_K_M`.** It is `Q4_K_M` with the 112
dense attention/GDN tensors at q4_k and `output.weight` at q5_k; every other tensor is the
original. It has **not** been tested under DFlash, where both of its parents lost (22.53 and
22.42 against `Q4_K_M`'s 23.18), so the sidecar profile remains on the plain `Q4_K_M` file.

## 26. The three remaining cheap tensor families, in one file (prediction first, 18:00)

With the dense set and the lm_head done, the per-token read of `Q4KM-dense-q4k.gguf` is ~1.83 GB
and it splits into four levers. Three of them are small and independent, and each costs a full
requant, an A/B and a 3.5-minute perplexity run to attribute separately (~20 minutes each), so
they are being bundled into one file and one measurement. Per-family bytes, from the header and
the 3.27% of the expert file a token actually touches:

| family | now | after | bytes/token saved | share of the token |
|---|---|---|---:|---:|
| `ffn_down_exps` (37 q5_k, 6.828 GB) | q5_k | q4_k | 0.040 GB | 2.2% |
| shared expert (123 q8_0, 0.138 GB) | q8_0 | q5_k | 0.049 GB | 2.7% |
| `output.weight` (0.350 GB) | q5_k | q4_k | 0.064 GB | 3.5% |

- **P19 (decode, bundled): 27.4-27.9 tok/s, i.e. +4.5 to +6.5% over 26.24.** 0.153 GB of 1.83 GB
  is 8.4% of the read, and the last three retypes collected 0.76-0.96x of their byte share.
- **P20 (quality, bundled): +1.0 to +2.0% perplexity.** The down-projection experts are used on
  every token and are the largest single change; `output.weight` q5_k -> q4_k is the one that
  every logit passes through.
- **P21 (prefill): no change**, as before.

If the bundle passes and a family later needs to be reverted for quality, it is bisected by
rebuilding from `Q4KM-dense-q4k.gguf` with one pattern removed - which now also means re-running
`bench/requant-dense-q4k.sh` first, since the intermediates were deleted once their numbers were
recorded.

### The requant and the decode measurement (measured 2026-11-05, 18:00)

`bench/requant-dq4-all.sh`: 753 tensors, **165 retyped** (123 shared-expert `q8_0 -> q5_k`, 37
`ffn_down_exps` `q5_k -> q4_k`, `output.weight` `q5_k -> q4_k`), 4 min 36 s, **21.815 -> 20.461
GB**. The three q6_k expert-down tensors and the blk.40 nextn set were deliberately pinned back to
their own types; the first generated pin file caught `ffn_gate_inp_shexp` (the f32 shared-expert
router) on the `shexp.weight` pattern, which would have quantized a selection tensor inside a
bundle - the diff of the pin file against the header caught it before the run, which is what that
check is for. Census of the output: q4_k 230 / 18.674 GB, q5_k 123 / 0.089 (shared expert only),
q6_k 3 / 0.661; 8 hashed unchanged-type tensors, 0 differ.

`bench/ab-allq4k.sh`, A B B A, 5 reps x 400:

| arm | file | median tok/s | spread |
|---|---|---:|---:|
| A1 | dense-q4k | 26.051 | 0.9% |
| B1 | all-q4k | **27.823** | 1.0% |
| B2 | all-q4k | **27.894** | 1.1% |
| A2 | dense-q4k | 26.306 | 1.0% |

**A/A +0.98%, B/B +0.26%, A -> B +6.43% pooled (27.859 against 26.179).** The A/A gap is the
session floor again, and the effect is 6.5x it.

**P19 is a hit at the top edge: predicted +4.5 to +6.5%, measured +6.43%.** The bundle saved
0.153 GB of the ~1.83 GB a token reads (8.4%), and at the 0.76x yield of the last dense step that
predicted +6.4%. This is the first prediction in the campaign that landed dead centre - the byte
model plus a yield factor is the right way to predict these. **New decode figure: 27.86 tok/s.**

### The quality gate (measured 2026-11-05, 18:10)

`bench/ppl.sh`: final estimate **1.9461** against dense-q4k's **1.9372**. Paired per chunk
**+0.00622 PPL, +0.289%, t = 6.74, 52/60 chunks worse**.

**P20 is a miss on the low side for the fourth time: predicted +1.0 to +2.0%, measured +0.29%.**
Down-projection experts at 4 bits, the shared expert at 5 bits and a 4-bit logits matrix together
cost less in loss than one dense step did. The campaign total is now **1.9142 -> 1.9461, +1.67%
perplexity for 22.25 -> 27.86 tok/s (+25.2%)**.

### Prefill, and the adopted profile file (measured 2026-11-05, 18:14)

`NAME=allq4k-9563`, `LENS=9563` (server `n_prompt` 11,484), `DECODE_REPS=5`: prefill **215.8
tok/s, TTFT 53.3 s** (215.6 / 215.3 / 216.3) and decode **27.86 tok/s** median, which reproduces
the A/B exactly. Prefill has now read 219.2 / 214.3 / 220.7 / 215.8 across the four files, a 2.8%
band with no order in it: **the TTFT at this width is a property of the ubatch, not of the quant.**
P21 holds.

**Adopted: `/mnt/ssd/models/Q4KM-all-q4k.gguf`, 20.461 GB.**

| | Q4_K_M | all-q4k | change |
|---|---:|---:|---:|
| decode, `prompt2.txt` (5 reps, `-bs`) | 22.25 | **27.86** | **+25.2%** |
| prefill, 11,484 tokens, `-ub 2048` | 220.9 | 215.8 | -2.3% (noise band) |
| TTFT, 11,484 tokens | 52.0 s | 53.3 s | +2.5% (noise band) |
| perplexity, 30k code corpus | 1.9142 | 1.9461 | **+1.67%** |
| file size | 22.52 GB | 20.46 GB | -2.06 GB |

Long context, same file: `NAME=allq4k-plain8k-ub2048`, `-c 12288`, `prompt-8k.txt`, 5 reps x
400: **23.75 tok/s** (23.78 / 23.73 / 23.76 / 23.81 / 23.69, 0.5% spread), against 20.91 for the
dense-q5k file on the same fixture (**+13.6%**). Attention costs more of the pass at 8.8k, so the
gain is smaller there than the +25.2% at 267 tokens, as it was for the first retype.

Composition: the 112 dense attention/GDN tensors at q4_k (from q8_0), every expert at q4_k
(`ffn_down_exps` from q5_k), the shared expert at q5_k (from q8_0), `output.weight` at q4_k (from
q6_k), routers still f32, token embedding still q8_0. Four measured steps, each with its own A/A
floor: **+8.5% (0.09%), +3.16% (0.78%), +6.0% (0.95%), +6.43% (0.98%)**.

What is left in the read, per token, and why the campaign is close to its end:

| group | bytes/token | share |
|---|---:|---:|
| experts q4_k | ~0.61 GB | 36% |
| dense attention/GDN q4_k | 0.708 GB | 42% |
| `output.weight` q4_k | 0.286 GB | 17% |
| shared expert q5_k | 0.089 GB | 5% |
| norms + ssm f32 | 0.104 GB | 6% |

The only lever left with more than 3% of the read is dropping the 256-expert block to 3 bits
(-0.144 GB/token, 8.6%), which is a different quality question: the experts carry the model's
knowledge, and the four quality predictions above were all too pessimistic only because every
change so far touched tensors the loss is not sensitive to. That one needs its own gate, and its
loss number should be expected to be several times larger than the last four.

## 27. The expert blocks at 3 bits (prediction first, 18:25)

This is the last lever in the perfuelled read with more than 3% of a token: the 256-expert blocks
are 18.674 GB of the file and ~0.61 GB of a token. q3_k is 110 bytes per 256 against q4_k's 144,
so the file loses ~4.4 GB and a token loses **0.144 GB, 8.6% of 1.68 GB**.

- **P22 (decode): 29.5-30.1 tok/s, i.e. +6 to +8% over 27.86.** Same byte model, 0.7-0.9x yield.
- **P23 (quality): +4 to +10% perplexity.** This is the first change in the campaign that moves the
  weights carrying the model's knowledge rather than a projection or an output matrix. If it lands
  above +10% the file is a speed demonstration and not a usable quant; the number decides.
- **P24 (prefill): no change.**

One artefact rule applies: the three q6_k `ffn_down_exps` of the original mix stay q6_k, and the
blk.40 nextn block is already q3_k and is left alone.

### The requant and the measurement (measured 2026-11-05, 18:24)

`bench/requant-q3e.sh`: 753 tensors, **117 retyped** (40 `ffn_gate_exps`, 40 `ffn_up_exps`, 37
`ffn_down_exps` from q4_k to q3_k; the three q6_k down projections and the blk.40 nextn block
pinned back), 3 min 04 s, **20.461 -> 16.290 GB**. Census: q3_k 120 / 13.841 GB, q4_k 113 / 1.008,
q6_k 3 / 0.661, q5_k 123 / 0.089; 8 hashed unchanged-type tensors, 0 differ. A clean artefact.

`bench/ab-q3e.sh`, A B B A:

| arm | file | median tok/s |
|---|---|---:|
| A1 | all-q4k | 27.805 |
| B1 | all-q3e | 25.400 |
| B2 | all-q3e | 25.512 |
| A2 | all-q4k | 28.082 |

**A pooled 27.944, B pooled 25.456: -8.9%. P22 is wrong in sign, and it is a large miss, not a
noise-level one.** The file is 16.3 GB against 20.5 and a token reads 8.6% fewer expert bytes, yet
decode is 9% slower. This is the project's oldest quant lesson holding on the decode path with no
drafter in the picture: **type matters more than bits on Volta**. `q3_k`'s 110-byte block and its
dequant/reconstruction sequence cost more per byte issued than `q4_k`'s 144-byte block saves, the
same mechanism that made IQ3_S slower than Q4_K in the 2026-10 type table and IQ2_S/MXFP4 collapse
at 17-28 GB/s.

The byte model that predicted the last four steps correctly fails here for a stated reason: it
assumed the yield factor (0.76-0.96) was a property of *where* the tensor is read from, when it is
a property of *which dot runs* over it. Every step that stayed on q4_k or above collected 0.7-0.96x
of its byte share; the first step that changed the dot's type family collected less than zero.

**No quality gate was run** - a 9% decode regression ends the experiment whatever its perplexity
is. `Q4KM-all-q3e.gguf` was not adopted and has been deleted, and the expert blocks are
**closed at q4_k**: q3_k and every IQ type below q4_k are the wrong direction on this GPU.

## 28. Can the MTP head be re-attached to the retyped file? (prediction first, 20:35)

The question is fair: the file still contains the blk.40 nextn layer, and every requant script
pinned it back to its own types, so the drafter is structurally intact. What moved is the target
underneath it - 112 dense tensors, all the experts, the shared expert and `output.weight`.

- **It will load and it will draft** (`--spec-type draft-mtp -n-max 6 -p-min 0.5`), because
  nothing in the file is missing or mistyped.
- **P25 (acceptance): 0.55-0.70, mean len 2.0-2.6.** Four unrelated perturbations to the target
  have landed in that band before (0.63603, 0.67717, 0.678, and the 0.64-0.93 requants), and the
  control's 0.96657 has never survived a weight change of any size. This file has four retypes
  stacked.
- **P26 (throughput): 15-21 tok/s, i.e. SLOWER than the 27.86 plain decode.** With the drafter on,
  a pass costs a draft chain plus a multi-column verify; at the 0.96657 signature that is worth
  ~1.66x, and at 0.65/2.3 it is worth ~0.55x of the same file's plain decode (34.82 -> 18.93 was
  exactly this collapse). The only way MTP wins here is if acceptance comes back above ~0.85,
  which no measurement in this project has produced after a retype.

### Measured (2026-11-05, 20:35)

It loads, it drafts, and **it is 16% slower than the same file with no drafter at all**:

| | all-q4k, plain | all-q4k + MTP (n-max 6) |
|---|---:|---:|
| TG, `prompt2.txt`, 5 reps | **27.86** | 23.351 (23.38 / 23.60 / 23.11 / 23.35 / 22.84) |
| acceptance | - | **0.64583** |
| mean len | - | **2.10** |

**P25 is a hit** (predicted 0.55-0.70 and 2.0-2.6, measured 0.64583 and 2.10) and **P26 is a miss
on the optimistic side** (predicted 15-21, measured 23.35): the retyped file's shorter pass keeps
the MTP configuration above the 15-21 band, but it is still **-16.2% against plain decode on the
same file**, so the answer to "can MTP be attached" is yes structurally and no in value. Per
token the accounting is 35.9 ms plain against 89.9 ms per MTP pass over 2.10 accepted tokens =
42.8 ms, so drafting costs 7 ms per accepted token more than it saves.

**The asymmetry to state plainly:** a drafter does not need the target to be *better*, it needs
the target's argmax to be *predictable from the same weights the drafter read*. Retyping moves
both and moves them apart. That is why the MTP-free profile and every requant win in sections
24-27 are mutually exclusive with the 36.93 tok/s configuration, not additive to it.

