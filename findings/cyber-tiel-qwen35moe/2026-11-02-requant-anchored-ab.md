# The anchored requant A/B: measured, and the confound nobody controlled

Written 2026-11-02. The prediction was recorded in
[2026-11-02-anchored-prediction.md](2026-11-02-anchored-prediction.md) before the run, and it was
wrong. This file records what the A/B actually produced and the mechanism it exposed.

## Result

`bench/sweep-anchored.sh`, `prompt2.txt`, 400 generated tokens, 3 reps each, `-nmax 3
-p-min 0.50`, `-ub 32 -b 32`, q8_0 KV, identical server build.

| | control (gate/up IQ3_S) | anchored (gate/up IQ4_XS) |
|---|---|---|
| TG tok/s, rep 1/2/3 | 31.13 / 31.65 / 31.41 | **25.76 / 19.32 / 30.40** |
| TG median | **31.41** | **25.76** |
| draft acceptance, rep 1/2/3 | 0.98940 / 0.98944 / 0.98940 | **0.86364 / 0.63025 / 0.92527** |
| mean_len, rep 1/2/3 | 3.69 / 3.70 / 3.69 | 2.93 / 2.04 / 3.55 |
| prompt eval, first request (267 tok) | 44.00 tok/s | 34.46 tok/s |

The anchored build is slower on **every** rep, and its best rep (30.40) still loses to the
control's worst (31.13).

## The prediction was wrong

The recorded prediction was acceptance **0.983-0.986** ("partial recovery, not a full one") and
**~32.4 tok/s**. Actual: acceptance 0.925 at best, again 0.630 at worst, and 25.76 tok/s median.
Not a partial recovery - a further loss, and an unstable one.

## What the run actually exposed: two findings

### 1. The control is deterministic. The requantized files are not.

Every control run in `bench/spec.csv`, across every sweep and both draft depths, records
acceptance **0.98940** - to five decimal places, on repeated identical requests at temperature 0
with `cache_prompt`. The harness is deterministic.

The anchored model produced 0.864, then 0.630, then 0.925 on three identical requests to the same
server. `--spec-draft-p-min 0.5` cuts drafting short when the draft head is unsure; the requant
pushes the drafter into a regime where its confidence sits near that threshold, and the small
state difference between a fresh prefill (rep 1) and a prefix-cache hit (reps 2-3) is enough to
flip whole drafting decisions. The control is nowhere near that boundary, so it never varies.

Consequence for method: a single acceptance number for a requantized file is not a measurement of
a fixed property. Every rep needs one, and spread is itself a result.

### 2. The "anchored" build was not anchored against the part that matters: block 40 is the MTP layer

`qwen35moe.nextn_predict_layers = 1`, and `blk.40` carries `nextn.eh_proj`, `nextn.enorm`,
`nextn.hnorm`, `nextn.shared_head_norm` - it **is** the draft head. The original file gives it
different quant types from the other 40 blocks:

| tensor | blocks 0-39 | block 40 (MTP) |
|---|---|---|
| `ffn_gate_exps`, `ffn_up_exps` | iq3_s | **q3_k** |
| `ffn_down_exps` | q6_k | **q3_k** |

Both requants forced `ffn_gate_exps` and `ffn_up_exps` to `iq4_xs` in **all 41 blocks**, block 40
included. Verified by parsing both GGUFs (`tools/gguf_tensors.py diff`): exactly 82 type changes,
41 gate + 41 up, and two of those 82 are the draft layer's own expert weights.

So the earlier experiment did not isolate "expert numerics the draft head dislikes" in the main
model. It also rewrote the drafter's own experts from a type the model author chose specifically
for that layer. The `attn_output` collateral that the anchored rebuild removed was a second-order
effect; the first-order confound is block 40, and it was present in both builds.

This is consistent with the depth data already recorded, which shows the loss worsening as the
drafter is pushed harder:

| nmax | control acceptance | gate/up requant acceptance |
|---|---|---|
| 3 | 0.98940 | 0.97037 |
| 4 | 0.99010 | 0.64202 |
| 5 | 0.99361 | 0.68182 |

## The verify pass did not get faster

Pass latency cannot be compared directly across different `mean_len` - a pass that drafts fewer
tokens does less work - so the only clean comparison is throughput, and it is worse. But the
comparable prefill measurement (267 tokens, both models, first request) is **44.00 vs 34.46
tok/s**, 22% *slower* with IQ4_XS gate/up. One rep each, so indicative rather than established,
but it points the opposite way from the "IQ4_XS is 1.23x at batch 4" premise this experiment was
built on. Prefill runs MMQ, not the MMVQ path every quant-type microbenchmark in this repo was
taken on, so the type ranking measured there does not carry over.

## Verdict on the pre-registered rule

The rule said acceptance near 0.970 closes the avenue. Observed 0.630-0.925 - the rule's second
branch fires and then some. **As built, this requant is finished: not worth keeping, do not
rebuild the same file.**

But the mechanism it exposed is not closed, because the build was confounded. The next build,
running now, is identical except that block 40 keeps `q3_k` gate/up. Decision rule, fixed in
advance:

- acceptance >= 0.985 and stable across reps -> blocks 0-39 were innocent; the entire loss was
  the drafter's own weights. Then the IQ4_XS swap in the main model is viable and the question
  becomes whether it is worth +2.2 GB.
- acceptance still 0.86-0.93 -> the main-model expert swap itself costs acceptance, the
  `nmax 3` cliff is unmoved, and the requant avenue is closed for good.

## Follow-up build 3: block 40 kept at q3_k, and it did not help

`gateupIQ4XS-draftkeep.gguf` (20.30 GB) is identical to build 2 except that block 40 keeps its
original `q3_k` gate/up. Verified by type diff: exactly **80** changes instead of 82, and
`blk.40.ffn_gate_exps` / `ffn_up_exps` / `ffn_down_exps` all remain `q3_k`.

`sweep-draftkeep.sh`, 5 reps each, same params.

| rep | control TG | control acceptance | draftkeep TG | draftkeep acceptance | mean_len |
|---|---|---|---|---|---|
| 1 | 31.23 | 0.98940 | 18.71 | 0.64286 | 2.01 |
| 2 | 31.79 | 0.98944 | 30.22 | 0.92527 | 3.55 |
| 3 | 31.79 | 0.98940 | 28.55 | 0.90182 | 3.25 |
| 4 | 31.77 | 0.98944 | 29.97 | 0.90845 | 3.41 |
| 5 | 31.58 | 0.98940 | 19.05 | 0.65566 | 2.03 |
| **median** | **31.77** | **0.98940** | **29.97** | - | - |

**Block 40 was not the cause.** Keeping the MTP layer's own experts unchanged changed nothing
about the failure: acceptance still ranges 0.64-0.93 against a stable 0.989, and best rep still
loses to control's worst.

The clinching detail: draftkeep rep 2 recorded acceptance **0.92527 (260 accepted / 281
generated, mean_len 3.55)** - **identical, to five decimals, to the anchored build's rep 3**. The
two files differ only in block 40, so on that trajectory block 40's quant type made no difference
to a single draft decision. That is evidence that the MTP layer's gate/up type is nearly
irrelevant to drafting, and that the loss comes from blocks 0-39 - that is, from changing the
**target model's** expert numerics, which the draft head is calibrated against.

## Closed: the requant avenue

Both branches of the decision rule point the same way, and the second one fires:

- acceptance did not recover (0.64-0.93, unstable);
- throughput is worse on every rep; median 29.97 against a control median of 31.77 (-5.7%);
- the pass did not get faster;
- prefill got ~22% slower.

**Do not build another gate/up requant of this model.** The original hypothesis - "the draft head
dislikes the new expert numerics" - was right, and it is unavoidable: it is a property of
changing the target's expert weights at all, not of a fixable mistake in the tensor map.

## The unexplained part, stated as unexplained

Why is the loss *bistable* (0.90-0.93 on some reps, 0.64-0.66 on others) rather than uniform?
The best available explanation is positive feedback around `--spec-draft-p-min 0.5`: the drafter's
confidence sits near the threshold, the accepted-token count changes the next verify batch, and
the run locks into either a high- or low-acceptance trajectory. The control is at 0.989 and never
near that boundary, which is why it is bit-stable across every run ever recorded.

This is a hypothesis, not a measurement. Nobody has shown the mechanism, and there is no profiler
on this board. It did not need resolving to close the requant avenue, so it was not chased.

## Reproducing

```sh
cd bench && ./sweep-anchored.sh          # control vs gateupIQ4XS-anchored, ~16 min
cd bench && ./sweep-draftkeep.sh         # control vs gateupIQ4XS-draftkeep, 5 reps, ~20 min

# verify any requant artifact before benchmarking it (rule 6)
python3 tools/gguf_tensors.py diff <control.gguf> <requant.gguf>
```

Note: `sweep-anchored.sh` was created without the executable bit and had to be `chmod +x`'d
before it would run. It had never actually been executed when this file was written.
