# Decode-speed tuning: the artifacts, `p-min`, and the best config

**Date:** 2026-11-01
**Build:** mainline `dde879b` unless stated
**Model:** Cyber-Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS
**Result:** 22.7 -> **31.52 tok/s** on realistic varied text (+39%)

All numbers here are one server load per config, greedy sampling, EOS ignored, 400 generated
tokens, `bench/prompt2.txt` (six varied paragraphs), `-c 32768` unless stated.

## Two benchmark artifacts that inverted earlier conclusions

Both were discovered by accident and both had already produced wrong conclusions. They matter
more than any individual flag result.

### Artifact 1: repetitive prompts inflate MTP

The first harness prompt (`bench/prompt.txt`) was one paragraph repeated eight times. On it,
draft acceptance was 0.954 and mean accepted length 2.89. Replacing it with six varied paragraphs
of similar length dropped acceptance to 0.37-0.57 on short windows.

A repetitive prompt makes the model emit highly predictable text, which the MTP head predicts
almost perfectly. **Any MTP benchmark scored on exact repetition overstates throughput by
40-60%.** The previously reported 25.39 tok/s carried acceptance 0.9242, in the same range as the
repeated-paragraph case, so that figure is prompt-inflated.

This does not invalidate the earlier comparative sweeps (all arms shared a prompt), but the
absolute numbers are not realistic chat or coding throughput.

### Artifact 2: short measurement windows understate steady state

The same config measured 22.71 tok/s over 128 generated tokens and 26.45 tok/s over 400.
The 128-token window is dominated by per-request startup and covers only the harder opening
tokens - draft acceptance over the first 128 tokens is genuinely lower than over 400 on the same
deterministic output.

All headline numbers here are therefore measured over **400 generated tokens**. Short-window
numbers are kept only where useful for ranking.

## The dominant lever: `--spec-draft-p-min`

`common/speculative.cpp:1468` stops drafting as soon as the draft model's top-1 probability drops
below `p_min`. It gates draft *depth per pass*, not just acceptance after the fact. This single
parameter is worth more than every build flag tried combined.

Varied prompt, 400 tokens, draft 3, `-ub 32`:

| `--spec-draft-p-min` | TG tok/s | acceptance | mean len |
|---|---:|---:|---:|
| 0.40 | 24.64 | 0.833 | 2.97 |
| **0.50** | **31.52** | 0.989 | 3.69 |
| 0.55 | 29.15 | 0.970 | 3.48 |
| 0.60 | 30.75 | 0.972 | 3.68 |

The optimum at 0.50 is sharp and **not monotonic**: 0.40 admits many draft tokens whose
confidence sits in the 0.4-0.5 band, which are then verified at full cost and rejected. The MTP
confidence distribution appears bimodal.

**Metric warning:** reported `acceptance` and `mean len` are conditional on a draft having been
made, so they hide passes that produce no draft at all. At `p-min 0.9` acceptance reads 1.000
while throughput collapses to ~22 tok/s, because almost every pass skips drafting and falls back
to plain decode. **Throughput is the only trustworthy metric in this table.**

## Draft depth

At `p-min 0.50`, same conditions:

| draft max | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| TG tok/s | 26.45 (p-min 0) | 30.46 | **31.52** | 28.84 | 28.30 |
| mean len | 1.83 | 2.89 | 3.69 | 4.49 | 5.20 |

Depth 3 is the optimum: past it, verification cost grows faster than accepted tokens. This
ranking matched the short-window data even though the absolute numbers differed, so draft depth
was the one conclusion the artifacts did not invert.

Note also that at depth 1 the optimum `p-min` is lower (0.3 gives 25.45 in a 128-token window,
versus 22.71 at `p-min 0`). The gate and depth interact; do not tune one without the other.

## Steady-state results

Varied prompt, 400 tokens, `-ub 32 -b 32`, q8_0 KV:

| Config | TG tok/s |
|---|---:|
| no speculation (`--spec-type none`) | 16.64 |
| draft 1, p-min 0 | 26.45 |
| draft 2, p-min 0.5 | 30.46 |
| **draft 3, p-min 0.5** | **31.52** |
| draft 3, p-min 0.5, `-ub 1024` | 26.21 |
| draft 3, p-min 0.5, `-ub 16` | 28.35 |
| BeeLlama `58a1629`, draft 2, `-ub 1024` (128 tok) | 18.88 |
| BeeLlama `58a1629`, draft 1, `-ub 1024` (128 tok) | 21.08 |
| mainline, draft 1, `-ub 1024` (128 tok) | 22.27 |

Conclusions:

- **MTP with a tuned gate is worth +89%** over plain decode (31.52 vs 16.64). Speculation, not any
  build flag, is the performance story on this board.
- **UBatch 32 is a 20% win over 1024** at draft depth 3 (31.52 vs 26.21). Small ubatch matters much
  more once drafts are deep and few tokens are rejected. `-ub 16` is also worse (28.35), so 32 is
  a genuine optimum rather than "smaller is better".
- **Mainline beats BeeLlama by about 6%**, not the ~29% an early degenerate-prompt comparison
  suggested (22.27 vs 21.08 at matched settings). This resolves the open comparison.
- `--spec-draft-p-split` makes no difference at these depths (0.1/0.3/0.5 all within noise at
  31.5-31.6).
- Thread count is neutral (t2 26.43 vs t4 26.39 over 5 reps each).

## Negative results

- `GGML_CUDA_FORCE_MMQ=ON` build: **-14%** (27.27 vs 31.66 on the then-baseline). MMQ is a prefill
  win, not a decode win. Matches the external fork's own caution.
- `q4_0` KV instead of `q8_0`: **-9%**. q8_0 remains correct.
- `-ub 2048 -b 8192`: worse than `-ub 1024`, which is already worse than 32.
- `--spec-type draft-mtp,ngram-simple`: no measurable effect. No repeated content to exploit.
- `ngram-simple` alone: no gain on varied text.
- `f16` KV at 32K context: OOM (`NvMapMemAllocInternalTagged error 12`). Not retested.
- `p-min 0.7` at draft depth 1: 20.95, worse than `p-min 0.3` (25.45). Over-filtering forces
  fallback to plain decode.
- `p-min 0.9` at depth 2: 22.07 with acceptance reading 1.000 - almost no drafting happens.

## Best launch command

Validated at the full 262,144 slot: **31.2-31.5 tok/s**.

```bash
env -u GGML_CUDA_DISABLE_GRAPHS \
/home/triph/juicer/momonga-llama.cpp/build-volta/bin/llama-server \
  -m /home/triph/models/Cyber-Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS.gguf \
  -c 262144 -fa on -ngl 99 -ncmoe 0 -t 2 -tb 2 \
  -ctk q8_0 -ctv q8_0 -ub 32 -b 32 --poll 0 --parallel 1 \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-n-min 0 \
  --spec-draft-p-min 0.5 --spec-draft-p-split 0.1 --load-mode none
```

Verified end to end at `-c 262144` with q8_0 KV: **31.22 / 31.51 / 31.22** tok/s over three
400-token requests, acceptance 0.989, mean length 3.69. The full-size slot costs nothing versus
`-c 32768` (31.52), so there is no reason to shrink the context.

Set `GGML_CUDA_DISABLE_GRAPHS` to **nothing at all** to enable graphs; the check is on presence,
not value, so `=0` disables them.

## Harness

`/home/triph/juicer/bench/one.sh` loads a server, issues N measured requests, and appends to
`results.csv`. Key environment variables:

| Variable | Meaning |
|---|---|
| `PROMPTFILE` | use `prompt2.txt` (varied). `prompt.txt` is the degenerate repeated-prompt file, kept only to reproduce artifact 1 |
| `NPRED` | generated tokens per request; use 400 for steady state |
| `DRAFT` / `PMIN` / `PSPLIT` | MTP draft depth and gate |
| `UB` / `B` | ubatch / logical batch |
| `KVK` / `KVV` | cache types |
| `CTX`, `REPS`, `THREADS`, `TB`, `PORT` | context, repetitions, threads, port |
| `SPEC_TYPES` | e.g. `draft-mtp`, `none`, `draft-dflash` |
| `DRAFT_MODEL` / `DRAFT_NGL` | external draft model via `-md` and `--spec-draft-ngl` |

Companion scripts:

- `prof-best.sh` - samples `/sys/devices/gpu.0/load` and GPU/EMC clocks every 200 ms during a
  sustained 400-token decode.
- `sweep*.sh` - the sweeps referenced in these files.

**Operational lessons:**
- Every sweep sleeps 15 s (or waits for free memory) between arms. Back-to-back loads can hit
  `CUDA error: out of memory` from a server that has not released VRAM yet. This silently killed
  three arms across two sweeps (`p70-d2`, `p40-d3`, and all four `df-*-p50` retries), which then
  looked like "server died" rather than OOM.
- `pkill -f <script-name>` matches the killing shell's own command line. Use a character class
  such as `pkill -f "sweep[-]remain"`.
- `one.sh` echoes `$NAME CMD:` to stderr every run. This exists because a `-md` insertion
  silently failed once and the benchmark ran with no draft at all (see
  [2026-11-01-speculators.md](2026-11-01-speculators.md)).
