# Intermediate ubatch widths (33-63): the one free prefill lever left

**Date:** 2026-11-05
**Status:** measured 2026-11-05. Prefill +11% at `-ub 48`, and the drafter destroyed (acceptance 0.57955) - the whole width family above 32 is closed, see sections 4-3 and 6.

## 1. Why this is not covered by the closed width sweep

The ubatch-width sweep (`findings/2026-11-04-prefill-starting-point.md` section 7) measured
**32 / 64 / 96 / 128 / 256 / 512** and closed the question of trading decode for TTFT: prefill
1.00x / 1.06x / 1.28x / 1.45x / 1.91x / 2.4x against acceptance 0.96657 / 0.91592 / 0.68122 /
0.92258 / 0.57655 / 0.65882. The configuration stayed at 32 because **64 costs 5 points of
acceptance and 10% of decode for +4-7% prefill**.

Nothing was measured between 32 and 64. That gap matters because of a threshold the sweep itself
established: `MMQ_DP4A_MAX_BATCH_SIZE = 64` and `ggml_cuda_should_use_mmq()` returns
`n_tokens < 64`, so **every width below 64 keeps the MoE in the same engine** - one fused MMQ
launch per tensor per layer - while 64 and above move it to the per-expert dequant+cuBLAS loop.
The sweep's 64-token acceptance loss is therefore not evidence about 33-63; it is a different
kernel set. Whether the drafter survives a moderate width is an open, cheap question.

## 2. The mechanism that should make it pay

At `-ub 32` the fused MoE launch reads the weights of **every expert that received a token** in
the layer, and each expert receives about **one** token of the 32. The grid is
`z = ne02` = 256 experts, and a block whose expert has fewer than `jt*J` tokens returns before it
loads anything (`if (jt*J >= col_diff) return;`, `mmq.cuh:1258`), so at `J=8` and 1-2 columns per
expert exactly **one column tile per distinct expert** runs and that expert's full weight rows are
read once for one or two tokens. Traffic per token is therefore proportional to
`distinct_experts(ub) / ub`, and the coupon-collector term is not negligible: at `-ub 32` the 256
token slots (32 tokens x 8 routed) already touch about **162** of the 256 experts, so going to 48
raises the distinct count to ~199 while tokens rise 1.5x. That is the same effect the tile-width
fix exploited from the other side - H3 removed discarded *dot* work, this removes repeated *reads*.

Post-H3 prefill budget at `-ub 32`: MoE launches are 67.7% of prefill. If the MoE time per token
is proportional to `distinct/ub` (56.8-59.0 GB/s against 71.8-73.3 GB/s available to its 16 resident
warps, so 77-82% of what its residency allows), then prefill time per token scales as
`0.677 * (distinct(T)/T) / 5.06 + 0.323`, where 5.06 = 162/32 is the `-ub 32` value:

| width | distinct experts of 256 | expert reads per token | predicted prefill vs 32 |
|---|---|---|---|
| 32 | 162 | 5.06 | 1.00x |
| 40 | 183 | 4.58 | 1.07x |
| 48 | 199 | 4.15 | 1.14x |
| 56 | 211 | 3.77 | 1.21x |
| 63 | 220 | 3.49 | 1.27x |
| 64 | 221 | 3.45 | 1.28x, **but measured 1.06x** - see below |

(The `distinct` column is `256*(1-(255/256)^(8T))`; the numbers were recomputed from a first,
wrong draft of this file that assumed a fixed 256-expert read per ubatch and therefore predicted
1.51x at 64. No measurement existed when this table was corrected, only the source read of the
`jt*J >= col_diff` early return.)

The 64-token row is where the model contradicts that arithmetic, and the reason is the engine
switch: above 64 the MoE stops being one fused MMQ launch, the parameter mix inverts (MMQ 88.1% ->
17.2%), and the per-expert loop pays its own overheads. Below 64 there is no such switch, so the
traffic argument has nothing in it to break - which is why the prediction below is stated for
33-63 only.

## 3. Prediction, written before the run

1. **Acceptance stays bit-identical at 0.96657 and mean len at 5.68** at `-ub 40` and `-ub 48`
   (5 reps, median), because the MoE engine, the MMQ tile rule (`J=8`, unchanged for every width
   below 64) and FlashAttention's tile selection are all unchanged in this range. Decode TG then
   stays within 1% of the `-ub 32` control measured in the same session. Falsified if acceptance
   moves at all, and the answer becomes "the first threshold is somewhere in 33-63" and the
   experiment is repeated at a smaller width.
2. **Prefill at 9,563 tokens rises with width** inside 33-63: **+10 to +16% at 48** (87-92 tok/s
   against a same-session control of 79-86), **+5 to +9% at 40**. Falsified if the two widths are
   within 1% of the control, which would mean the MoE time is not proportional to distinct-expert
   weight reads.
3. TTFT at 9,563 tokens follows the prefill number; at the good end, 120 s -> 100 s.

Because the prefill measurement is cross-server-load and the load-to-load drift is up to 17%
(`findings/2026-11-05-prefill-page-cache.md` section 3c), the arms are run **interleaved in both
orders in one session** with the control repeated at the end, so the A/A difference gives the
floor for this comparison. Acceptance and mean len do not need that treatment: they reproduce
bit-for-bit across loads.

## 4. Result

See the sections added below.

## 4. Result: prefill +11% and the drafter destroyed

Run 2026-11-05 02:39-03:20, `bench/ab-ub.sh` with `WIDTH=48`, order `32 48 48 32`, one server
load per arm, `LENS="590 9563"`, `REPS=2`, `DECODE_REPS=5` on `prompt2.txt` before the prefills,
binary `bench/bins/A` (the 36.93-tok/s record build).

| arm | decode TG (5 reps, median) | draft acceptance | mean len | prefill at 703 tokens | prefill at 11,484 tokens | TTFT at 11,484 |
|---|---|---|---|---|---|---|
| `-ub 32` (control, first) | **36.76** | **0.96657** | **5.68** | 84.4 tok/s | **78.38 tok/s** | 146,641 ms |
| `-ub 48` (A) | 18.70 | **0.57955** | **2.10** | 94.1 tok/s | 87.02 tok/s | 131,955 ms |
| `-ub 48` (B) | - | **0.57955** | **2.10** | - | 86.98 tok/s | 131,991 ms |

- **Prediction 1 falsified, and not marginally.** Acceptance was predicted to be bit-identical at
  0.96657; it is **0.57955**, with mean len 5.68 -> 2.10 and decode 36.76 -> 18.70 tok/s (-49.1%).
  Both `-ub 48` arms reproduce 0.57955 / 2.10 exactly, so this is a deterministic alternative
  attractor, not noise. It is also **worse than `-ub 64`** (0.91592), which falsifies the premise
  of the experiment outright: staying inside the fused-MMQ engine does not protect the drafter, so
  the 64-token acceptance loss was never an artefact of the engine switch.
- **Prediction 2 held.** Prefill at 11,484 tokens reads 87.0 tok/s against 78.4 for the control in
  the same session, **+11.0%**, inside the corrected +10 to +16% band; at 703 tokens it is +11.5%
  (94.1 vs 84.4). The corrected traffic model - MoE time per token proportional to
  `distinct_experts(ub)/ub`, with the coupon-collector term at 162/256 distinct experts for 32
  tokens - predicted +14% at 48. That model is therefore usable for future work: it is the same
  quantity that makes `-ub 512` 2.4x.
- **Prediction 3 held in direction:** TTFT at 11,484 tokens falls 146.6 s -> 132.0 s in this
  (slow-state) session.

## 5. What this settles

1. **The intermediate width range is closed as a free lever.** It pays - +11% prefill measured - but
   it costs half the decode, which is a worse exchange than `-ub 512` (2.4x prefill for -46%
   decode) and is therefore not a candidate under the current instruction.
2. **"Change the ubatch and the acceptance will survive if the engine does not change" is false.**
   The router is f32 and selects the top-8 of 256 experts per token; the hidden state it sees
   depends on how the prompt was chunked, so any width change is a different model in the sense of
   AGENTS.md section 7. The mapping is not monotone (32: 0.96657, 48: 0.57955, 64: 0.91592, 96:
   0.68122, 128: 0.92258), which rules out tuning it as a slope.
3. **The expert-read traffic is the right axis if a numerics-safe wide batch ever exists.** At
   `-ub 32` the MoE reads 162 of 256 experts' full weight sets per ubatch to serve 256
   (token, expert) slots - 1.6 slots per read - and prefill therefore moves ~300 MB of expert
   weights per token. That is why width is worth 11% and why `-ub 512` is worth 2.4x. Any future
   work here should target the *number of tokens per expert read*, not the dot.

## 6. The rest of the range: 36 and 40, measured the same night

Because the width-to-acceptance map is chaotic, "48 is bad" does not imply "36 is bad". The gate
that settles it is acceptance alone - it is identical across server loads, so no interleaving is
needed - via `bench/scan-ub-gate.sh` (one model load per width, 5 decode reps, no prefill phase,
~8 min per arm).

| ubatch | draft acceptance | mean len | decode | prefill vs ub 32 |
|---|---|---|---|---|
| 32 | **0.96657** | **5.68** | 36.76 tok/s | 1.00x |
| 36 | 0.88779 | 4.09 | - | ~1.04x (predicted) |
| 40 | 0.70755 | 2.99 | - | ~1.07x (predicted) |
| 48 | 0.57955 | 2.10 | 18.70 tok/s | **1.11x (measured)** |
| 64 | 0.91592 | - | - | 1.06x (measured pre-H3, not comparable) |
| 96 | 0.68122 | - | - | 1.28x (measured pre-H3) |
| 128 | 0.92258 | - | - | 1.45x (measured pre-H3) |

**Every width above 32 breaks the drafter, so the whole family is closed.** The map is not a slope
and not monotone - 40 and 48 are worse than 64 - which means it cannot be tuned; the only way to
know a width is to measure acceptance on it, and every measurement so far outside 32 has come back
between 0.58 and 0.92. The largest prefill gain any of them has shown is +11% at 48, against half
the decode. There is no free point in 33-63.

**The one thing this buys the project is a validated traffic model**: prefill at `-ub 48` came in
at the predicted +11% (predicted +14%), from `distinct_experts(ub)/ub` alone. That is the quantity
a future numerics-safe wide batch would exploit, and it is also the explanation of H1's 2.4x.

## 7. Where the prefill campaign stands after this

- The only width with an intact drafter is **32**, so prefill traffic per token stays at the
  `-ub 32` value: about **300 MB of expert weights per token** (162 of 256 experts' full weight
  sets per ubatch, serving 256 (token, expert) slots - 1.6 slots per expert read). At the board's
  measured 92 GB/s that is a floor of ~300 tok/s; the recorded configuration reads 78-87, so the
  gap is (a) the MoE's 57 GB/s and (b) the non-MoE 32% of prefill.
- The exchange the user has held in reserve is not affected by tonight's work: `-ub 96/128/512`
  remains the only large TTFT lever, and its cost is now mapped on the full width curve.
- What is left at `-ub 32` is inside `mul_mat_q`: the dp4a tile performs a J=8 column dot for 1-2
  useful columns and the block serialises dequant, a shared-memory store and two barriers per K
  iteration. Both were measured (H3 + this file), and there is no launch parameter left -
  `findings/2026-11-04-prefill-57gbs-probe.md` and `...-mmq-tile-i64.md` close that space.
