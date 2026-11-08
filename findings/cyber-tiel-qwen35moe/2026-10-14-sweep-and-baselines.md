# 125-point sweep, 25.39 tok/s candidate, and the 262K-slot baselines

**Date:** 2026-10-14
**Build:** BeeLlama `58a1629` (local `beellama.cpp/build-cuda`)
**Status:** historical. The 25.39 tok/s headline here is superseded - it was measured on a
repetitive prompt, which inflates MTP throughput. See
[2026-11-01-decode-speed-tuning.md](2026-11-01-decode-speed-tuning.md).

## Test environment at this date

- Device: Jetson AGX Xavier, Volta SM72, ~30,991 MiB available.
- Model: `/home/triph/models/Cyber-Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS.gguf`
  (fine-tune of Ornith-1.5-35B-A3B, includes an MTP head).
- Server: `/home/triph/juicer/beellama.cpp/build-cuda/bin/llama-server` (BeeLlama build 1,
  commit `58a1629`). CMake cache: CUDA on, `CMAKE_CUDA_ARCHITECTURES=72`,
  `GGML_CUDA_ENABLE_UNIFIED_MEMORY=ON`, FlashAttention and KVarN enabled, CUDA graphs enabled,
  `GGML_CUDA_FORCE_MMQ=OFF`.
- Power clocks set to maximum with `jetson_clocks`; this did not measurably improve throughput.

## Completed 125-point sweep

Results directory: `/tmp/opencode/xavier-mtp-262144-exhaustive-v2/`

- CSV `slot-screen.csv`, journal `slot-screen.csv.journal`, `run.log`, `screen.html`.
- Context wrapper in the worktree: `llama-server-fixed-ctx.sh`.
- The wrapper reserved the 262,144-token server slot while screening with short prompts.
  Optimizer prompt depth 512; measured prompts about 558 tokens.
- This is an **orthogonal-array screen, not a full Cartesian search**.
- A tmux supervisor resumed interrupted runs with `--resume --retry-crashed`. It logged
  `[supervisor] sweep complete: 125/125 rows` and exited.

### Best result and verification

**25.391 tok/s** (run 1). The optimizer's independent verification measured three extra times,
giving a four-measurement median of **25.4 tok/s** with **1% spread**.
Prompt processing was approximately 73-74 tok/s.

| Setting | Value |
|---|---|
| GPU layers / MoE CPU offload | `-ngl 99`, `-ncmoe 0` |
| MTP | enabled, `--spec-draft-n-max 1`, minimum draft 0 |
| KV cache | Q8_0 K/V |
| Threads | `-t 2 -tb 2` |
| Batch | `-ub 32 -b 32` |
| Polling | `--poll 0` |
| CUDA graphs | **actually disabled** - the run set `GGML_CUDA_DISABLE_GRAPHS=0`, which the runtime treats as "present" and therefore off |
| Repack / op offload / host buffers | repack off; standard op offload and host buffers |
| Load mode | `auto` |
| Slot / measured prompt | 262,144 reserved / about 558 prompt tokens |
| MTP stats in run 1 | draft acceptance 0.9242; coverage 0.4766 |

The optimizer's generated suggested command showed `-c 1024` because its own screening context
is short; the fixed-context wrapper is what reserved the 262K slot. Do not use the suggested
command alone as a full-context launch recipe.

### Other successful sweep points (single screening measurements)

Same short-prompt / 262K-reserved methodology; only run 1 received verification repeats.

| Run | tok/s | MTP max | KV | `-t/-tb` | `-ub/-b` | Graphs | Other notable settings |
|---|---:|---:|---|---|---|---|---|
| 28 | 23.93 | 1 | Q8_0 | 4/2 | 128/256 | on | poll 50; repack on; p-min .9; min fraction 1.0 |
| 105 | 24.68 | 3 | Q8_0 | 4/4 | 64/256 | on | poll 0; no-host; repack on; p-min 0; min fraction .5 |
| 9 | 25.13 | 2 | Q4_0 | 2/2 | 32/32 | off | poll 100; repack off; p-min .5; min fraction .5 |
| 78 | 25.29 | 3 | Q8_0 | 2/2 | 128/256 | off | poll 50; repack off; p-min .25; min fraction .5 |
| 1 | **25.39** | 1 | Q8_0 | 2/2 | 32/32 | off | poll 0; repack off; p-min 0; min fraction 0 |

`spec_p_split` varied across these points (.3/.3/.3/.1/.1 for runs 28/105/9/78/1);
`nkvo=0` and `ncmoe=0` for all five.

### Sweep status interpretation

- The 105 `SLOW` records include genuine measurements below the 23 tok/s cutoff **and** trials
  that timed out before any decode measurement. A recorded `tg_tps=0` in the latter is a
  placeholder, not a measured zero.
- 11 configurations OOMed, 4 errored.
- Some configurations reported "speculation did not run" when no response completed or
  llama.cpp drafted no tokens.
- The optimizer's predicted-optimal factor levels differ from the best complete configuration;
  the prediction must be benchmarked before it is treated as a winner.
- Being an orthogonal screen with many simultaneous factors, it does **not** isolate the causal
  effect of any one option (including Q4/Q8/KVarN or graphs).

## Earlier baseline results

### 24.77 tok/s earlier setup

Averaged three coding prompts, 16K slot:

- MTP max 1; Q8_0 KV; full target offload; `-t 4 -tb 2 -ub 64 -b 64`.
- `--poll 50`; CUDA graphs disabled.
- Predates the completed screen; not context-matched to the 262K-reserved tests.

### 24.68 tok/s screen candidate (run 105)

Initially the leading 262K-reserved screening result: MTP max 3 / min 1, Q8_0 KV,
`-t 4 -tb 4 -ub 64 -b 256`, poll 0, `--no-host`, repack on, graphs enabled, `ncmoe=0`.
Single short-prompt measurement.

### Short direct context/cache check

A direct test with a 262K slot and KVarN4 generated only 57 tokens at about 20.34 tok/s.
Too short to be reliable; not a controlled comparison.

## Long-context limitation

- All broad-sweep speed results reserved a 262,144-token slot but measured short prompts.
  They do **not** establish near-262K decode or prefill throughput.
- An earlier optimizer attempt at prompt depth 261,600 returned no useful measurements; both
  MTP horizon trials errored without speed results.

- A reliable next benchmark is a matched repeat of the 25.39 setup and the earlier 24.77 setup
  at identical large contexts, starting with a confirmed tokenized near-full prompt, and
  recording prefill and decode separately. Only after validating the prompt fits should depth be
  increased toward 262,144.
- Preserve the 23 tok/s cutoff, but report failures separately from true measured slow runs.

## 262K-slot retest of the earlier 24.77 tok/s setup

- Re-ran with the fixed-context wrapper forcing a **262,144-token slot**, short measured prompt
  (~558 tokens), 256 generated tokens.
- Configuration: MTP on, draft max 1/min 0, Q8_0 KV, `-ngl 99 -ncmoe 0 -t 4 -tb 2 -ub 64 -b 64`,
  `--poll 50`, repack and op-offload enabled, `--load-mode auto`, CUDA graphs disabled,
  `spec-p-min=0`, `spec-p-split=0.1`.
- Initial measurement: **25.72 tok/s** (PP 76.56, draft acceptance 0.9466, coverage 0.4844).
- Three verification measurements were requested but the run was terminated on request while
  starting the last server, so there is **no verified median or spread**.
- The optimizer did not clean up its last child server on SIGINT; that orphan was terminated
  and a process check confirmed nothing remained.
- Artifacts: `/tmp/opencode/xavier-mtp-262144-retest-24-77/hold-24-77.csv` and `.journal`.
- Runtime lesson: `--verify-picks 3` calls `drive_one` three more times, starting a fresh server
  each time. Model-file reads and startup dominated elapsed time. Prefer a harness that keeps
  one server loaded between measurements.

### Single-load three-repetition confirmation

- Same 24.77-era configuration, fixed 262,144-token slot, one server load, three measured
  requests in that session.
- Result: **25.52 tok/s** decode, PP 77.01, prompt 558 tokens, draft acceptance 0.9342,
  coverage 0.4805. Three measured requests took 84.17 s.
- Supports that the earlier speed holds with the 262K slot reserved: 25.52 vs the earlier
  24.77 average. Still a short-prompt measurement.
- Artifacts: `/tmp/opencode/xavier-mtp-262144-retest-24-77-singleload/hold-singleload.csv`,
  `report.html`.

## Deep-context sweep outcome (64K / 128K / 240K) - never completed

- Attempted with the 25.52 tok/s settings, one server launch, three reps per depth, 100%
  identical-prefix reuse, 262,144-token reserved slot.
- The first 64K-depth row ended as `TIMEOUT` after 3,600 s with no valid throughput. The CSV has
  `err_rate=0.25`, no measured prompt token count, and zero PP/TG. Those zeros are **not**
  measured speeds.
- The journal records the 128K row as next `TRY` but no result was saved; 240K was not completed.
- Resumed in tmux `xavier-mtp262k-depth` with `--resume --retry-crashed`, 14,400 s per
  config. The 128K row ran ~1 h 10 m with no measurement and was terminated on request.
- **Conclusion:** the deep-context sweep gives no evidence about 64K+ decode throughput. The
  64K prefill workload alone exceeded the one-hour per-config timeout.
- Results/journal: `/tmp/opencode/xavier-mtp-262k-depth-profile/depth-profile.csv` and
  `.journal`; supervisor `resume-depth-supervisor.sh`, log `resume.log`.

## Copy-paste launch command for the best verified setup at this date

The **25.39 tok/s** result used BeeLlama `58a1629`. It reserves a 262,144-token slot but the
measured prompt was ~558 tokens; it is not a near-full-context result.

```bash
env -u GGML_CUDA_DISABLE_GRAPHS \
/home/triph/juicer/beellama.cpp/build-cuda/bin/llama-server \
  -m /home/triph/models/Cyber-Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS.gguf \
  -c 262144 \
  -fa 1 \
  -ngl 99 \
  -ncmoe 0 \
  -t 2 \
  -tb 2 \
  -ctk q8_0 \
  -ctv q8_0 \
  -ub 32 \
  -b 32 \
  --poll 0 \
  --spec-type draft-mtp \
  --spec-draft-n-max 1 \
  --spec-draft-n-min 0 \
  --spec-draft-p-min 0.0 \
  --spec-draft-p-split 0.1 \
  --op-offload \
  --load-mode auto
```

Repacking was off (do not add `--repack`). Normal host buffers were used.
Source: `/tmp/opencode/xavier-mtp-262144-exhaustive-v2/slot-screen.csv`.

**Note:** the original entry wrote `GGML_CUDA_DISABLE_GRAPHS=0` here and called graphs enabled.
That is wrong - presence disables them. The command above uses `env -u` to genuinely enable
graphs, which is what current best practice requires.
