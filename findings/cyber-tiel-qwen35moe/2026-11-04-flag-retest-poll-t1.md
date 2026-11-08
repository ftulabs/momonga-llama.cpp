# Re-testing two flags the 28-config sweep scored as noise: `--poll 50` and `-t 1 -tb 1`

**Date:** 2026-11-04
**Build:** `momonga-llama.cpp/build-volta/bin/llama-server`, unchanged during the run.
**Model:** `Cyber-Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS.gguf`, ctx 32768, `-ub 32 -b 32`,
`--spec-type draft-mtp -nmax 6 -n-min 0 -p-min 0.50 -p-split 0.10`, `NPRED 400`, KV `q8_0/q8_0`.
**Both arms of every pair include `-bs`**, so these are measured on top of the adopted config.
**Verdict: both are dead flat. Nothing adopted.**

## Prediction

**No prediction was written down before this run.** That is a failure against rule 1, and it
matters here because the result is negative: with no number on file beforehand, "I expected
nothing" cannot be distinguished after the fact from a rationalisation. Recording it so the next
agent knows this run's expectation is not evidence.

## Why these two arms

`./2026-11-04-backend-sampling.md` showed that a flag the 28-config sweep dismissed as
"inside +-1% noise" (`-bs`) was really **+1.04%** when re-tested with the interleaved both-orders
protocol. That sweep (and the earlier 2026-11-02 audit, which also scored `-bs` as neutral) ran
single-order, 3 reps, on the 31.5 tok/s build - a protocol that per rule 7 cannot resolve 1%.
That made its per-flag verdicts *unsettled rather than closed*.

`--poll 50` and `-t 1 -tb 1` were picked as the two most plausible survivors: `--poll` changes how
the server waits for GPU work (host-side idle, which is exactly where `-bs` won), and
`-t 1 -tb 1` cuts the CPU thread pool, which the host-side profile says is doing 41.7 ms of work
per pass on 8 Carmel cores.

## Protocol

`bench/sweep-flagab.sh`, `ORDER="base poll t1 t1 poll base"`, `REPS=5`, `NPRED=400`,
`BASE_EXTRA="-bs"`. Six arms, one interleaved pass in each direction. Rep 1 of every arm is the
post-load cold rep and is excluded, leaving **n=8 warm reps per configuration** (4 from each
order). All results in `bench/results.csv` under the `f1-` prefix.

## Result

| arm | position | warm examples | median |
|---|---|---|---|
| base | 1 | 36.85 37.20 36.63 36.70 | 36.775 |
| poll (`--poll 50`) | 2 | 37.01 37.39 36.92 36.98 | 36.995 |
| t1 (`-t 1 -tb 1`) | 3 | 37.07 37.33 36.80 36.77 | 36.935 |
| t1 (`-t 1 -tb 1`) | 4 | 36.95 37.07 36.73 36.96 | 36.955 |
| poll (`--poll 50`) | 5 | 37.04 37.26 37.00 37.04 | 37.040 |
| base | 6 | 37.01 37.76 37.06 36.93 | 37.035 |

Pooled, both orders, n=8 per configuration:

| config | median tok/s | vs base |
|---|---|---|
| base | 36.970 | - |
| `--poll 50` | 37.025 | **+0.055 (+0.15%)** |
| `-t 1 -tb 1` | 36.955 | **-0.015 (-0.04%)** |

Acceptance is **0.96657 and mean len 5.68 in all six arms**, identical to the control. That is the
expected result for flags that touch no numbers, and it confirms the arms are comparable.

## The drift this captures

The single-order reading of the same data would have been **+0.60%** for `--poll 50`
(base-1 at 36.775 against poll-2 at 36.995) - the same shape of false positive that `-bs` turned
out to be real and that graph chunk size 128 turned out *not* to be. Reversing the order collapses
it to +0.15%.

Two independent measures of how large that artefact is on this board today:

1. **Per-position means across all six positions:** 36.84 / 37.08 / 36.99 / 36.93 / 37.09 / 37.19.
   The board gained **+0.35 tok/s (+0.95%)** from the first position to the last over roughly
   three hours of wall clock.
2. **An A/A control falls out of the design.** Positions 1 and 6 are the *same configuration* and
   differ by 0.26 tok/s (**0.71%**). So the protocol's own floor today is about 0.7%: any
   single-order claim below that is not a measurement of the flag.

Both are larger than the 0.15% that `--poll 50` "won" by.

## What this closes

- `--poll 50` and `-t 1 -tb 1` are **closed as negatives**. They join `-bs` (real, +1.04%, adopted)
  as arms of the old sweep that have been re-measured properly; the other 25 are still unsettled,
  not closed.
- The stated purpose of `-t/-tb` was to cut host CPU. It does not help, which is consistent with
  the `-bs` profile: after `-bs` the host does 37.4 M instructions per pass and only ~7.8 ms of it
  is exposed as GPU idle, so the thread pool was not what was limiting.
- Nothing is added to `bench/run-best.sh`. The best configuration does not move.

## Note on the session baseline

The pooled base here is **36.970** against the **36.930** recorded for the same configuration
earlier today. Both arms include `-bs`, so the two are the same config; the 0.1% difference is
session drift of the size measured above and is **not** a new record. The documented figure
stays 36.93 until a change moves it by more than the drift.
