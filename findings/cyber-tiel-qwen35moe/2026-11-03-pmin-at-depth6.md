# `--spec-draft-p-min` re-tuned at draft depth 6: 0.50 stays, and the low side is a cliff

Date: 2026-11-03. Machine: Jetson AGX Xavier, Volta sm_72, CUDA 11.4.
Library md5 `5c129aaee1a8ec8c3d7fda1a14d3e7d6` (the 35.07 baseline, dispatch patch + 64-register
IQ3_S MoE launch bounds). Model `Cyber-Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS.gguf`.
Harness `bench/one.sh`, `CTX=32768 DRAFT=6 PSPLIT=0.1 REPS=5 NPRED=400`, prompt `prompt2.txt`.
Control arm measured in the same session: **34.82 tok/s**, acceptance 0.96657, mean_len 5.68.

## 1. Why this was re-run

`--spec-draft-p-min` gates draft *depth per pass* on the drafter's top-1 probability
(`common/speculative.cpp:1468`). It had only ever been swept at **draft depth 3, on the
pre-dispatch-patch build** (`./2026-11-01-decode-speed-tuning.md`), where it was the single
largest lever found - worth more than every build flag combined. The dispatch patch moved the
optimal depth from 3 to 6 (`AGENTS.md` section 7), and that same finding says explicitly: *"the gate
and depth interact; do not tune one without the other."* So the 0.50 optimum rested on a
measurement taken at the wrong depth, on a build with a known dispatch defect. This run closes that.

## 2. Prediction, written before the run, and it was wrong

Recorded in `/tmp/pred-pmin.txt`. The reasoning was that mean_len 5.68 of a possible 6 means ~95%
of draft slots are used, so the gate must already be near-saturated and therefore nearly inert:

| `p-min` | predicted | measured | verdict |
|---|---:|---:|---|
| 0.40 | 34.2 (-1.7%) | **18.55 (-46.7%)** | **falsified**, 45 points off |
| 0.60 | 34.9 (+0.2%) | 33.67 (-3.3%) | falsified, right magnitude, wrong sign |
| 0.70 | 33.9 (-2.6%) | 34.01 (-2.3%) | close |

**The premise was wrong in a specific and reusable way.** `mean_len` is *conditional on a draft
having been made* - passes where the gate skips drafting entirely are excluded from it. So a high
mean_len says nothing about how often the gate binds. `./2026-11-01-decode-speed-tuning.md`
warns about exactly this ("acceptance and mean len are conditional... they hide passes that produce
no draft at all"), and this prediction walked into it anyway. The gate is *not* saturated at depth
6: at 0.40 the model generated 308-384 draft tokens per window against 310-326 at 0.50-0.70, so
lowering the gate admits a long tail that is then rejected.

## 3. Results

| `p-min` | tok/s, median of 5 | delta | all reps | acceptance (5 reps) | mean_len |
|---|---:|---:|---|---|---:|
| 0.40 | **18.55** | **-46.7%** | 18.55 / 18.48 / 18.33 / 18.93 / 27.86 | 0.572 / 0.503 / 0.514 / 0.584 / 0.816 | 2.26 |
| **0.50** | **34.82** | control | 34.10 / 34.83 / 35.16 / 34.82 / 34.70 | 0.96657 | 5.68 |
| 0.60 | 33.67 | -3.3% | 33.25 / 33.67 / 33.71 / 33.63 / 33.68 | 0.96012 (all 5 identical) | 5.67 |
| 0.70 | 34.01 | -2.3% | 33.68 / 34.01 / 34.00 / 34.13 / 34.15 | 0.99677 (all 5 identical) | 5.83 |

**0.50 remains the optimum. Both directions lose, and the surface is strongly asymmetric: -46.7%
below, -2.3% and -3.3% above.**

Two things in the table matter beyond the ranking:

- **At 0.60 and 0.70 the acceptance and generated-token counts are bit-identical across all five
  reps** (326 and 310 tokens respectively). The draft chain is fully deterministic at and above the
  gate optimum, which is why the control reproduces 0.96657 on every run ever logged. At 0.40 the
  counts move every rep (308-384) and acceptance scatters 0.50-0.82. Determinism is therefore a
  signature of "the gate is in its intended regime", and its absence is a warning sign.
- **0.70 shows the conditional-metric artifact in its purest form: acceptance 0.99677 and mean_len
  5.83, both *better* than the control's 0.96657 / 5.68, while throughput is 2.3% lower.** The gate
  is skipping weak passes outright, those passes are excluded from both statistics, and the ones
  that remain look excellent while plain decode drags the average down. Any future tuning that
  reports acceptance as evidence of a win must report throughput alongside it, and must be
  suspicious when acceptance improves and throughput does not.

## 4. Consequences

- `--spec-draft-p-min 0.50` is confirmed at the current depth on the current build. It is not a
  leftover from the depth-3 era; it is still the right value, now measured at the depth the model
  actually runs.
- The speculation flag surface is now swept at the operating point: `p-min` here, `p-split` at
  0.05/0.10/0.20 neutral (`./2026-11-02-bandwidth-and-flag-audit.md`), depth 3/5/6/7 with 6
  the optimum (`AGENTS.md` section 7). **Flags are not a lever.** The pass budget has to come down
  in kernel time or weight format.
- **The interval 0.40-0.50 was deliberately not fine-swept (0.45, 0.48), and neither was 0.55.**
  The reason is in the table: above the optimum the surface is *non-monotonic within a percent*
  (0.60 is 0.4% worse than 0.70), so 5 reps cannot rank points 0.05 apart - any "win" found there
  would be noise, which is exactly what `AGENTS.md` section 3 rule 4 excludes. On the low side the
  transition is a cliff rather than a gradient, and there is no mechanism by which 0.45 could beat
  0.50 that would not also have made 0.40 merely mildly worse instead of catastrophic. If someone
  does want the answer, a finer sweep needs more than 5 reps to be rankable, and it should report
  the generated-token count per rep, because that is the quantity the gate actually changes.
- The `-46.7%` at 0.40 is worth remembering as a trap: `p-min` is not a free parameter to nudge.
  Anything that lowers the effective confidence of the drafter - a different quant, a perturbed
  router, a build that changes accumulation order - pushes the same direction as lowering `p-min`,
  and the low side is a cliff, not a slope. That is the same fragility that produced -45.6% in
  `./2026-11-03-mmwq-launch-params.md` section 5.
