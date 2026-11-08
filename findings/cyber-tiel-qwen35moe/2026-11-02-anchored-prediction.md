# Anchored requant A/B - prediction recorded before running

Written 2026-11-02, before `bench/sweep-anchored.sh` was executed. This file exists so the
prediction below cannot be rationalised after the fact (AGENTS.md section 3, rule 1).

## Setup

| | gate/up experts | attn_output (11 tensors) | file |
|---|---|---|---|
| control | IQ3_S | Q8_0 | `Cyber-Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS.gguf` (18.12 GB) |
| first requant | IQ4_XS | **Q6_K (accident)** | `Cyber-Tiel-35B-A3B-gateupIQ4XS.gguf` (20.33 GB) |
| anchored requant | IQ4_XS | Q8_0 | `Cyber-Tiel-35B-A3B-gateupIQ4XS-anchored.gguf` (20.36 GB) |

Control reference: 31.4 tok/s, acceptance 0.98940, mean_len 3.69, pass 118 ms.
First requant: 30.5 tok/s, acceptance 0.97037, mean_len 3.54.

## Prediction

Draft acceptance: **0.983-0.986** - a partial recovery, not a full one. Reasoning: the
1.9-point drop has two candidate sources, and the attn_output collateral touches only 11 of
41 tensors while the gate/up swap touches 82 of the largest ones, including the shared expert.
I expect the expert numerics to be the larger contributor, so most but not all of the loss
should come back. I do **not** predict a clean return to 0.9894.

Throughput: **~32.4 tok/s, i.e. +3% against control**, off mean_len ~3.63 at a pass of
~112 ms. Even a full acceptance recovery to 0.989 would only give ~33.5 tok/s, because the
gate/up type change is worth 1.23x at batch 4 (the batch the verify pass actually runs),
not the 2.6x that exists only at batch 5.

Interpretation of the pre-registered decision rule, applied honestly:

- acceptance >= 0.989: attn_output was the whole loss; swap worth ~+5%; keep it.
- acceptance ~0.970: expert numerics; avenue closed.
- **acceptance in between (my prediction)**: neither branch of the rule was written for this
  outcome. The swap is then a small win at best and must be judged on measured tok/s, not on
  the recovery fraction. A 3% gain for +2.2 GB is not obviously worth keeping.

## Caveat

One rep in three on the larger requantized files is known to collapse to 19-20 tok/s with PP
falling from ~44 to ~10.7. The sweep runs 3 reps; if the median is within 2% of control the
result is not resolved and 5 reps are required before drawing a conclusion.
