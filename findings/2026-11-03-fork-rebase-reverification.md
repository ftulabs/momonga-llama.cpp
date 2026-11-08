# Re-verification after the rebase onto fork master

**Date:** 2026-11-03
**Question:** the verified tree was `dde879b` + the local `mmvq.cu` patch. The directory was
renamed to `momonga-llama.cpp/` and connected to `ftulabs/momonga-llama.cpp`, whose `master` was
31 commits ahead. Does the known-good configuration still read the same number on the new base?

**Prediction, written before the run:** unchanged within noise, 34.3-34.5. The two patched files
(`ggml/src/ggml-cuda/mmvq.cu`, `ggml/src/ggml-cuda/ggml-cuda.cu`) are untouched by those 31
upstream commits, so the decode path should be bit-identical code. The only plausible mover was
something else upstream changed in the scheduler, but nothing in the diff touches the graph
split or the sampler.

**Setup:** fork `master` (`2cbca38`, upstream of 2026-11-03) plus the local commits, rebuilt
clean into `build-volta` with the recipe in `KNOWN-GOOD.md`. `bench/one.sh`, `NAME=rebase-nm6`,
`prompt2.txt`, `NPRED=400`, `REPS=5`, `UB=32 B=32`, `--spec-draft-n-max 6 --spec-draft-n-min 0
--spec-draft-p-min 0.5`.

**Result:**

| base | reps (tok/s) | median | acceptance | mean len |
|---|---|---|---|---|
| `dde879b` + patch | 34.10 / 34.36 / 34.64 / 34.49 / 34.49 | **34.49** | 0.96657 | 5.68 |
| `2cbca38` + patch | 33.34 / 34.14 / 34.36 / 34.30 / 34.30 | **34.30** | 0.96657 | 5.68 |

Acceptance and mean length are identical to five decimal places, which is the expected outcome
for an unchanged sampling path and is the stronger signal than the throughput figure.

The 0.55% throughput difference is **not** established as a regression. Five reps do not resolve
anything under 2% (AGENTS.md rule 4), rep 1 of the new run (33.34) is the low outlier that pulls
the median down, and the two runs were not interleaved, so board state (thermal, memory
placement) is uncontrolled between them. The honest statement is: the configuration reads
34.1-34.6 on either base and the base is not a lever.

Prompt eval in the same run (33.4 / 10.7 / 11.0 / 8.8 / 10.7) is the usual `cache_prompt`
artefact and carries no information; see AGENTS.md rule 4.

**Consequences:**

- No re-tuning is needed after the rebase. `KNOWN-GOOD.md` now quotes 34.3-34.5 and says why.
- The old verified binary is kept at `build-volta-dde879b` for comparison; it is not the default
  and nothing points at it.
- If a future rebase does touch `mmvq.cu` or `ggml-cuda.cu`, this re-verification does not
  cover it. Re-measure rather than assuming.
