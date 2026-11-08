# IQ3_S MoE kernel: 64 registers instead of 79, +2.25% end to end

Date: 2026-11-03
Status: win, committed. New record 35.07 tok/s.

## What was changed

`mul_mat_vec_q_moe` in `ggml/src/ggml-cuda/mmvq.cu` is launched with one warp per block, but
its launch bounds declared a 256-thread block and one block per SM:

```cpp
__launch_bounds__(get_mmvq_mmid_max_batch_for_device<type>()*ggml_cuda_get_physical_warp_size(), 1)
```

The second parameter is inert at 1, and the first describes a block three times larger than the
one that actually runs. The compiler was therefore free to spend 79 registers per thread, which
capped residency at 24 one-warp blocks per SM - 37.5% of the 64-warp maximum, measured as 34.9%
achieved.

Declaring the real block size and asking for 32 blocks per SM caps IQ3_S at 64 registers and
raises the cap to 50%:

```cpp
__launch_bounds__(
        type == GGML_TYPE_IQ3_S ? ggml_cuda_get_physical_warp_size()
                                : get_mmvq_mmid_max_batch_for_device<type>()*ggml_cuda_get_physical_warp_size(),
        type == GGML_TYPE_IQ3_S ? 32 : 1)
```

The other types keep the previous declaration exactly, so their codegen is untouched. That
scoping is deliberate: a blanket 64-register budget would spill Q6_K, which `ffn_down_exps`
uses. 32 blocks per SM is also the hardware maximum on sm_72, so 50% occupancy is the ceiling
for a one-warp block - going higher needs multi-warp blocks, which is a different change.

## Prediction, recorded before the run, and what actually happened

Written into `TODO.md` before the build: kernel time falls from 381.66 us into the 300-330 us
band (-13% to -21%), and end to end reaches ~35.3 tok/s (+3%), falsified if the kernel drifts by
less than 5% or spills.

| | Predicted | Measured | |
|---|---|---|---|
| kernel time | 300-330 us | **352.58 us** | wrong, too optimistic |
| kernel delta | -13% to -21% | **-7.6%** | wrong |
| end to end | ~35.3 tok/s | **35.07 tok/s** | close, +2.25% |
| falsifier (kernel under -5%) | - | not triggered | |

The mechanism was right and the magnitude was overestimated. Occupancy went up by 37% and the
kernel got 7.6% faster, not 13-21%: the kernel is only partly latency-bound, and the remaining
1.63x instruction count against IQ4_XS (10.46M vs 6.40M warp instructions) does not move when
occupancy does.

## Kernel measurements

`bench/opbench`, `OPBENCH_SINGLE=1 OPBENCH_K=2048 OPBENCH_N=512 OPBENCH_TOK=6`, one replayed
launch under `sudo ncu`, and 5 opbench runs for the timing.

| | Control | 64 registers |
|---|---|---|
| registers per thread | 79 | **64** |
| occupancy limit (registers) | 24 blocks | **32 blocks** |
| architectural occupancy cap | 37.50% | **50.00%** |
| achieved occupancy | 34.93% | **47.85%** |
| kernel time | 381.66 us | **352.58 us** (-7.6%) |
| warp instructions | 10,407,936 | 10,457,088 (+0.5%) |
| DRAM read | 21.64 MB | 21.63 MB |
| achieved DRAM bandwidth | 56.7 GB/s | 61.3 GB/s (55% of ceiling) |
| `long_scoreboard` stall | 3.05 | 4.17 |
| `math_pipe_throttle` stall | 1.16 | 1.48 |

The +0.5% instruction count is the check for spill damage: a real spill would show up as a large
instruction increase and a slowdown, and neither happened.

The stall counts rise for both reasons, which is not a regression. They are warps *resident* per
issue-active cycle, so more resident warps means more warps counted waiting. IQ4_XS, unchanged
by this patch, measures 5.25 / 1.76 in the same run.

opbench end to end on the op, median of 5:

| | Control | 64 registers |
|---|---|---|
| IQ3_S, 6 tokens | 0.492 ms (43.9 GB/s) | **0.467 ms** (46.4 GB/s) |
| IQ4_XS control, 6 tokens | 0.401 ms | 0.409 ms (noise, untouched) |

## End to end

`bench/one.sh`, 32768 context, `-ub 32 -b 32`, `--spec-draft-n-max 6 --spec-draft-p-min 0.5`,
q8_0 KV, 5 reps of 400 tokens on `prompt2.txt`. The two binaries differ only in
`libggml-cuda.so.0`; md5 `9f3715d0...` is control, `5c129aae...` is the change.

| Config | TG tok/s per rep | Median |
|---|---|---|
| control | 33.10, 34.30, 34.47, 34.43, 34.08 | **34.30** |
| 64 registers | 33.94, 35.07, 35.29, 35.29, 34.95 | **35.07** |

+0.77 tok/s, **+2.25%**. Acceptance is identical at 0.96657 and mean length identical at 5.68,
so this is a pure speed change with no effect on what the draft head does. The control run also
reproduced the previously recorded 34.30 exactly, which is the check that the harness and the
prompt have not drifted.

Prompt processing in the table is not reported: `cache_prompt` serves the prefix from cache, so
reps 2+ evaluate only a few tokens and the PP column is an artefact (AGENTS.md, measurement
rule 4).

## What this rules out

The 64-register change is worth more in the kernel (-7.6%) than end to end (+2.25%), which is
consistent with the MoE gate/up pair being roughly 20-25% of pass latency. It also means the
remaining IQ3_S deficit - 61.3 GB/s against IQ4_XS at 94 GB/s - cannot be closed by occupancy
alone, because occupancy is now at the architectural ceiling for a one-warp block. The remaining
gap is the instruction count, which is the `iq3s_grid` decode path, i.e. Frontier 1.

Frontier 1's shared-memory staging should therefore be measured against this new baseline:
0.467 ms on opbench, 352.58 us under `ncu`, 35.07 tok/s. A repeat of the 0.382 -> 0.233 ms
prediction is no longer valid, because the 0.382 baseline is gone.

## Artifacts

- `git show <this commit>` - the launch-bounds change, 9 lines in `mmvq.cu`
- `/tmp/libggml-cuda-control.so.0` and `/tmp/libggml-cuda-occ64.so.0` - the two libraries
- `bench/results.csv` rows `occ-ctl` and `occ-64`, `bench/spec.csv` same

## Frontier 1 (stage `iq3s_grid` in shared memory): upper bound measured, then closed on feasibility

Same day, after the register fix. Two things were needed to decide this: an upper bound on what
the table can cost at all, and whether 2 KB per block is affordable.

**Upper-bound probe.** `vecdotq.cuh` computes `iq3s_grid[qs[l0] | ((qh << (8 - l0)) & 0x100)]`.
Replacing both lookups with `iq3s_grid[0]` and rebuilding gives, under `ncu` at the model's exact
MoE geometry (`OPBENCH_SINGLE=1 OPBENCH_NE=256 OPBENCH_NU=8 OPBENCH_TOK=6`):

| | kernel | L1 load sectors | DRAM read | registers | occupancy |
|---|---:|---:|---:|---:|---:|
| as measured (352-356 us) | 356.13 us | 10,992,952 | 21.63 MB | 64 | 47.86% |
| lookups removed | 258.69 us | 8,226,091 | 13.38 MB | 55 | 48.59% |

That is -27% kernel time and -8.25 MB of DRAM, which is far more than a 2 KB table can account
for. The probe is **contaminated**: in IQ3_S the `qs[]` bytes are used *only* as the grid index,
so pinning the index to 0 makes the `qs` loads dead and the compiler deletes them. 64 of the
116 bytes in `block_iq3_s` are `qs`. The measurement is therefore an upper bound on
"stop reading and decoding the qs array", not on "make the gather cheaper", and no optimization
of the gather itself can reach it.

**Feasibility.** `iq3s_grid` is 512 x uint32 = 2 KB. Staging it per block costs 2 KB of shared
memory, and the kernel now runs 32 one-warp blocks per SM (32 remain blocks, registers, shared
and warps - all four limits). 32 x 2 KB = 64 KB of the 128 KB unified L1/shared, so the carveout
would have to go to 64 KB and L1 would fall from ~111 KB to 64 KB. This kernel's L1 hit rate is
the reason it is fast at all; halving L1 to save a table gather whose upper bound is already
smaller than the qs bytes it cannot avoid reading is the wrong trade.

**Closed.** Not the mechanism (the qs decode is) and not affordable (2 KB/block at 32 blocks/SM).
The table stays in global memory. `iq3s_grid[0]` is reverted; the probe is not in any commit.
