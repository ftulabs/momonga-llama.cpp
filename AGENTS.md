# AGENTS.md - notes for agents working in this fork

This is `ftulabs/momonga-llama.cpp`, a fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
that carries two CUDA changes for a Jetson AGX Xavier (Volta sm_72, CUDA 11.4). Read
[JETSON-XAVIER.md](JETSON-XAVIER.md) first for what the changes do and what they measured, and
[README.md](README.md) for the build recipe and the launch line.

The upstream contribution rules in
[ggml-org/llama.cpp AGENTS.md](https://github.com/ggml-org/llama.cpp/blob/master/AGENTS.md) are
for PRs against upstream and **do not apply here** - the upstream file says so itself. What
replaces them follows.

## Rules

1. **`origin` is this fork, `upstream` is `ggml-org/llama.cpp`.** Pushing to `origin` is normal
   and expected. **Never push to `upstream`, never open a PR against `ggml-org`, never send a
   patch anywhere else.** These changes are board-specific and will not be accepted.
2. **Commit only after the change is built and measured, not before.** A commit message must say
   what was measured: the number, the config it was measured in, and the base commit. A change
   that "should" be faster and has not been run does not get committed as an improvement.
3. **Verify a rebase instead of assuming it.** The other two commits are small and can be
   re-applied by hand, but the performance claims belong to the exact code that was measured. If
   upstream touches `ggml/src/ggml-cuda/mmvq.cu` or `ggml/src/ggml-cuda/ggml-cuda.cu`, rebuild
   and re-measure before repeating any number.
4. **ASCII only** in code, comments, commits and docs. No em dashes, no arrows, no curly quotes.
5. **Comments are 1-2 lines** and explain why, not what. Do not hard-wrap comments to a column.
6. **Do not benchmark with `GGML_CUDA_OP_TIME` set.** It disables CUDA graphs and costs real
   time; it is for attributing shares, not for tok/s.
7. **The experiment hooks stay off by default.** `GGML_CUDA_MOE_RPB`, `GGML_CUDA_MMID_MAX_BATCH`,
   `GGML_CUDA_SMALLK` and `GGML_CUDA_OP_TIME` must all be unset for the documented numbers to
   apply. If a hook is added, it defaults to upstream behaviour or it is not added.

## What has been measured (do not re-derive)

- **36.56 tok/s**, 400 generated tokens, greedy, median of 5: `--spec-draft-n-max 6
  --spec-draft-p-min 0.5 -ub 32 -b 32 -ctk q8_0 -ctv q8_0`, draft acceptance 0.96657, mean
  draft length 5.68. The best unpatched configuration is 31.50 at `n-max 3`; plain decode with
  no speculation is 16.9. The stack underneath: 35.07 from the MoE launch bounds, 34.30 without
  them, and the dense MMVQ commit on top for the last 4.3%.
- **The 9.5% comes from drafting deeper, not from the kernel being faster at a given batch.** The
  patch does nothing at batch 4, where both paths were already legal. It removes a cliff above
  4-5 columns that made every deeper config pay MMQ.
- **`n-max 7` is slower** than 6. **`rows_per_block = 4` on the dense path** is now the default
  and wins at the model level (+4.3% together with the carveout, the launch bounds and the
  wider `small_k` trigger). *Corrected:* the old note here said it loses 0.7% and that the
  default stays 2. That measurement was taken without the 64 KB carveout, which is part of the
  change and not tuning on top of it. The MoE kernel's own `GGML_CUDA_MOE_RPB` hook is still a
  model-level loss and still defaults to 2.
- **The board is not bandwidth-starved**: 32-35% of a measured 110 GB/s. *Corrected:* the IQ3_S
  expert matmul is not ALU-bound either - hardware counters put `long_scoreboard` (waiting on a
  load) ahead of `math_pipe_throttle`. It was occupancy-limited by a launch-bounds declaration
  that named a 256-thread block for a 32-thread kernel, and fixing that was worth +2.25%.
- Measurement discipline that produced those numbers: 5 reps and a median for anything under 2%,
  never extrapolate a microbenchmark across batch sizes, report draft acceptance with every
  weight or config change, and write a negative result down with the same care as a positive one.

## Build and run

Both are in [README.md](README.md). The short version: CMake with
`-DCMAKE_CUDA_ARCHITECTURES=72`, `-DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.4/bin/nvcc`,
`-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10.3.0`, the `-isystem` flags for the C++ headers, and
`-DGGML_CUDA_NO_VMM=ON`. nvcc as shipped here rejects the system gcc without them.

Running the server needs `env -u GGML_CUDA_DISABLE_GRAPHS`, because the code tests for the
variable's existence rather than its value.

## Where the rest of the context lives

The measurement notebooks are **not in this repository**. They live in the parent workspace on
the machine this fork was built on: `KNOWN-GOOD.md` (the validated configuration), `findings/`
(one file per experiment), `TODO.md` (including a "closed - do not re-test" list) and `bench/`
(the harness). Ask before assuming a number that appears in a commit message, because those files
are the source of truth and this file is only a summary.

## Maintenance

`README.md`, `AGENTS.md` and `JETSON-XAVIER.md` are fork-owned and will conflict when rebasing
onto upstream. Keep the fork version (`git checkout --ours README.md`). Everything else in the
tree is upstream and should be kept as upstream wrote it.
