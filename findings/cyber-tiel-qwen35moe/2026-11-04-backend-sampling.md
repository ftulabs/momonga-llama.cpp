# Backend sampling (`-bs`): the first host-side lever, worth about 1% (2026-11-04)

Status: **win, adopted: +1.04% pooled over 16 warm reps per arm in two opposite-order blocks.**
`--backend-sampling` moves the target sampler chain onto the GPU. The
host then reads back a candidate array instead of scanning and copying the full 248320-wide
logits row, which the 2026-11-04 host profile
([`2026-11-04-cpu-side-of-a-pass.md`](2026-11-04-cpu-side-of-a-pass.md)) measured at ~18% of
41.7 ms/pass of host CPU. Acceptance, mean draft length and the generated text are unchanged.

## Why this was tried

The profile found three host clusters in a 156.5 ms pass:

| cluster | share of host CPU |
|---|---|
| full logits row scanned + copied per sampler call (`common_sampler_sample`, `partial_sort_inplace`, `__memcpy_generic`) | ~18% |
| per-call graph preparation (`ggml_cuda_can_fuse`, `try_fuse`, `ggml_can_fuse_subgraph_ext`, `ggml_backend_sched_split_graph`) | ~9%, paid 7x per pass |
| allocator / mutex noise | ~5% |

The draft chain already used backend sampling (`common_params_speculative_draft::backend_sampling
= true`), but the *target* chain did not (`common_params_sampling::backend_sampling = false`), so
every pass still pulled 7 rows of 248320 floats to the host. `-bs` is the existing switch.

## Prediction, written down first

`bench/sweep-bs.sh`: +1-2% (36.6 -> 37.0-37.4), acceptance bit-identical at 0.96657, output text
identical. **Landed at +1.04%, inside the band, with identical acceptance and identical text.**

## Method

`bench/sweep-bs.sh`: `bench/one.sh` arms that differ only in `EXTRA="-bs"`, 5 reps each,
`NPRED=400`, ctx 32768, `-nmax 6 -p-min 0.5`, one server load per arm. Two interleaving blocks
with **opposite arm order** so that board drift cancels across the pooled medians:

* block 1 `ORDER="base bs bs base"` - base takes positions 1 and 4, `-bs` takes 2 and 3;
* block 2 `ORDER="bs base base bs"` - `-bs` takes 1 and 4, base takes 2 and 3.

Rep 1 of every arm is excluded: it is the cold rep after a model load and reads 35.0-35.8 in
every arm where reps 2-5 read 36.2-37.6.

## Results

Row names: `bs-base-*` / `bs-on-*` in block 1, `b2-*` in block 2.

| arm | block 1 warm median | block 2 warm median |
|---|---|---|
| base | 36.335 (pos 1) | 36.530 (pos 2) |
| base | 36.600 (pos 4) | 36.595 (pos 3) |
| `-bs` | 36.375 (pos 2) | 36.970 (pos 1) |
| `-bs` | 37.060 (pos 3) | 36.830 (pos 4) |

| block | base | `-bs` | delta |
|---|---|---|---|
| 1 (`base bs bs base`) | 36.505 | 36.825 | **+0.88%** |
| 2 (`bs base base bs`) | 36.550 | 36.930 | **+1.04%** |
| pooled, 16 warm reps per arm | 36.550 | 36.930 | **+0.380 tok/s (+1.04%)** |

The per-position means across all 36 measured reps are 36.59 / 36.48 / 36.45 / 36.50 for
positions 1-4, so within-block drift is 0.1-0.3% and no monotonic trend; with opposite orders in
the two blocks the arm/position assignment is balanced anyway. Acceptance is **0.96657** and mean
len **5.68** in every one of the 8 arms, i.e. nothing moved numerically.

Rep 1 is excluded from every arm (it is the cold rep after a load: 35.0-35.8 against 36.2-37.6
for reps 2-5). Including it changes nothing: the all-rep medians are 36.42 (base) and 36.85
(`-bs`).

## Adopted

`-bs` is in `bench/run-best.sh` and in the command above in `KNOWN-GOOD.md`. It is the first
change in this project that is neither a kernel nor a launch parameter - it removes host work
between launches rather than GPU work.

What is *not* yet done: the per-call graph preparation cluster (~9% of host CPU: `can_fuse`,
`try_fuse`, `ggml_backend_sched_split_graph`, paid 7 times per pass for an identical graph
structure) has no switch and no cache. That is the next host-side lever, and it would need a
change in `ggml-backend.cpp`/`ggml-cuda.cu` rather than a flag.

## Caveats

* `-bs` is marked experimental upstream. On this board the server logs
  `device 'CUDA0' does not have support for op ARGSORT needed for sampler 'top-p'`, so the chain
  is *partially* offloaded: the unsupported nodes stay on the host and operate on the reduced
  candidate set rather than the full distribution. With greedy sampling the result is identical
  (verified: byte-identical text, identical acceptance over 20 requests); for a chain where a
  host node needs the full distribution this could in principle differ, so re-verify output
  equality before relying on it outside benchmarking.
* This is a host-CPU change, not a kernel change: `libggml-cuda.so` is untouched.
