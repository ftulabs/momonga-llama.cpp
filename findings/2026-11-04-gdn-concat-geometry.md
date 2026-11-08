# The GDN glue is 3.5 ms/pass, and only part of it is removable

2026-11-04. `concat_non_cont` + `cpy_scalar` in the Gated DeltaNet block cost **4.6 ms of a
166 ms pass** (2.8%), the largest bucket outside the three matmuls and the lm_head. This
records what they are, what a launch-geometry fix bought (**+0.29% pooled over 18 reps per
arm, both orders, acceptance bit-identical**), and why the rest of it cannot be removed
without a new fused op.

## What the two kernels are

Per decode pass, 30 GDN layers each run:

| kernel | launches/pass | us/launch | ms/pass | what it moves |
|---|---:|---:|---:|---|
| `concat_non_cont` | 30 | 83.2 | 2.505 | `conv_input` = `concat(conv_states[3, 8192], qkv_mixed^T[7, 8192], dim 0)`, 10x8192 f32 |
| `cpy_scalar` | 210 | 8.8 | 2.090 | 7 rollback snapshots x 3 rows x 8192 channels, into the conv state cache |
| `cpy_scalar` | 11 | | | the 11 attention layers' KV rows |
| `cpy_scalar` | 6 | | | small, 4096 elements |

The 210 = 30 x 7 because `n_rs_seq` is `draft.n_max` = 6 (`common/common.h`
`need_n_rs_seq()`), so `build_conv_state` writes K = 7 snapshots per layer
(`delta-net-base.cpp:501-518`), one `ggml_cpy` launch each.

## The concat was launching 25x more threads than it needed

`concat_non_cont` was launched with `grid = (dst->ne[1], ne[2], ne[3])` and 256 threads, and each
block then loops `i0 = threadIdx.x; i0 < ne0; i0 += blockDim.x`. For the GDN concat `ne0` is 10
(`d_conv-1 + n_tokens` = 3+7) and `ne1` is the 8192 channels, so the kernel launched
8192 blocks x 256 threads = 2.1M threads to move 81,920 elements - ten threads per block did
all the work. It also made the read pattern uncoalesced: within a block the ten active threads
read `qkv_mixed^T` at `i0*nb0` with `nb0` = 8192*4 = 32 KB, one 32-byte sector per 4-byte value.

The fix is in `ggml/src/ggml-cuda/concat.cu` only: flatten `(i1,i2,i3)` into a row index that is
packed into `blockDim.y` (`by = 256/bx`, `bx` = `ne0` rounded up to one warp, capped at 256), so
block covers `by` rows and the grid is `(n_rows + by - 1)/by`. Element-to-element copy, no
value touched. For this shape the launch becomes grid (1024,1,1) block (32,8,1).

Measured on adjacent traces (control and patched, six minutes apart, same harness, 78 decode
passes each - see rule 7 in `AGENTS.md`, do not compare these against older traces):

| | control | patched |
|---|---:|---:|
| `concat_non_cont` us/launch | 83.2 | **46.5** |
| `concat_non_cont` ms/pass | 2.505 | **1.398** |
| pass busy (sum of kernels) | 157.89 | **156.78** |
| pass span | 166.97 | **165.80** |
| `cpy_scalar` ms/pass (untouched) | 2.090 | 2.123 |

The pass busy falls by 1.11 ms, exactly the concat delta, and the span by 1.17 ms.

## Model level: +0.29%, and that is smaller than the trace implies

Two 9-rep A/B runs in both orders, one shared library swapped in place, everything else
identical:

| arm | order 1 median | order 2 median | pooled median (18 reps) | pooled mean |
|---|---:|---:|---:|---:|
| control | 36.58 | 36.41 | 36.525 | 36.448 |
| patched | 36.67 | 36.57 | **36.630** | **36.548** |
| delta | +0.25% | +0.44% | **+0.29%** | **+0.27%** |

Draft acceptance **0.96657** and mean length **5.68** in all four arms - bit-identical, as a copy
geometry change must be. That is the correctness check for this change.

**Why 0.3% and not the 0.67% the trace predicts.** A traced pass is longer than an untraced one
(166 ms here against 155.3 ms implied by 36.63 tok/s at mean_len 5.68), and the inflation falls
mostly on the small kernels: the 46 us concat is more sensitive to CUPTI per-launch cost than a
9 ms lm_head call is. So the untraced saving is smaller than the 1.11 ms the trace shows. The
sign is consistent across both orders and matches the trace; the magnitude is not the trace's.
This is the same lesson as the `rows_per_block` retest, in the other direction: a traced kernel
delta needs the model A/B before it is quoted as throughput.

## What is left, and why it is not free

- **The concat still costs 46.5 us for 328 KB** - 7 GB/s. The reads of the transposed
  `qkv_mixed` are still one sector per 4-byte value. A shared-memory tiled version (load 10x32
  tiles with `i1` contiguous, store rows contiguous) should reach ~10 us, worth another
  ~1.1 ms/pass of *traced* time, which by the above is maybe +0.2% of throughput. Not done.
- **The 210 rollback snapshots cannot be collapsed into one `ggml_cpy`.** Snapshot slot `j` must
  hold the state `j` tokens back (slot 0 = most recent; `gated_delta_net.cu:158-160`,
  `target_slot = n_tokens - 1 - t`), while the source window advances with the token index, so
  source and destination run in opposite directions - not expressible as a 4D view with positive
  strides. Collapsing them needs either a new fused op that writes all seven slots, or a state
  cache layout where the seven slots of one layer are contiguous. Both are real work for
  ~1.7 ms of traced time (1.0%).
- **`n_rs_seq` cannot be lowered to save the copies.** It is set to `draft.n_max` precisely so
  that a partial draft acceptance does not have to restore a checkpoint
  (`server-context.cpp:4961`); with mean length 5.68 of a 7-token verify, partial acceptance is
  the normal case, so every pass would take the checkpoint path instead.

## Also noticed, not measured

Inside a traced pass about **9.0 ms is not kernel time** (span 165.8 against busy 156.8), and it
is concentrated: one gap of 3.57 ms between two `rms_norm` launches, and six gaps of 481 us
between `get_rows` and `rms_norm` (the six MTP draft steps, i.e. one per graph re-entry). That
shape says CPU-side work between graph launches - sampling, acceptance, `seq_rm` - rather than
anything GPU. Per `findings/2026-11-03-dense-mmvq-rows-per-block.md`, gaps in a traced run are
not trustworthy as absolute numbers, so this is a lead and not a budget: it needs an untraced
method before it becomes a target.
