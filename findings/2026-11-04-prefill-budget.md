# H2: the per-kernel prefill budget - MMQ owns 88% of it, and GDN is 3%

**Date:** 2026-11-04
**Purpose:** decide where the 134.6 s of TTFT at 9,563 tokens actually goes. Until this was
measured, H3 (MMQ config) and H4 (chunked GDN) were guesses about which rebuild to spend board time
on. The answer reverses the ranking: **`mul_mat_q` is 88% of production prefill and
`gated_delta_net` is 3.3%**, so H4's ceiling is single-digit percent and H3 is the whole campaign.

Harness: `bench/prof-prefill.sh`. One request in the trace, `cache_prompt:false`, `n_predict 1`,
under `sudo nsys --trace=cuda`, exported to sqlite and summed per kernel signature by
`bench/nsys-kernels.py`. There is no lm_head verify launch to bracket a prefill pass with, so the
trace *is* the prefill; the first-request graph capture is host-side and does not enter the kernel
sums. Same prompt (590 tokens, `prompt_n` from the server) at both widths, so the two budgets are
directly comparable.

## 1. Prediction, recorded before the first run (rule 1)

Scaling from the measured decode pass (`gated_delta_net_cuda` 6.44 ms/pass over 30 launches at
7 tokens = ~30 us/token/layer) and from the known prefill rates (512 tokens at ~180 tok/s at
ub 512, ~78 tok/s at ub 32):

| bucket | predicted share at ub 512 | predicted share at ub 32 |
|---|---|---|
| `mul_mat_q` (MMQ: dense layers + MoE expert matmuls) | **55-68%** | **65-80%** |
| `gated_delta_net` (token-serial, 30 layers) | **13-18%** | 6-8% |
| FlashAttention (MMA-F16, 11 layers) | 4-8% | 3-6% |
| everything else (norm, quantize, copy, router, embedding) | 8-15% | 8-15% |

Two falsifiable statements attached:

1. **GDN's absolute time is invariant in ubatch width** (it is a serial loop over tokens inside one
   call, so 512 tokens cost the same whether they arrive in one launch or sixteen). Predicted
   **400-500 ms** for 512 tokens at both widths; if it differs by more than ~20% between the two
   traces, the loop is not the whole story and there is per-launch work on top.
2. **If `gated_delta_net` >= 40% of the budget, this falsifies the prediction above and H4 becomes
   the only large lever** - which is the outcome that would strand H3 as a small lever.

Written down before the run because a wrong forecast stated afterwards gets rationalised, and
because the whole point of H2 is to decide whether H3 is worth a rebuild (AGENTS.md rule 1).

## 2. Result: the prediction was wrong in both directions

Both traces are one 590-token prefill (`prompt_n` 590 in each), same binary, same model, only
`-ub`/`-b` differing. Kernel-busy totals check against the warm wall-clock prefill measured
earlier (7,491 ms at ub 32, 3,647 ms at ub 512), so ~88% and ~78% of the wall time is kernel time
and the sums below are the real budget.

| family | ub 32 ms | share | ub 512 ms | share |
|---|---:|---:|---:|---:|
| **`mul_mat_q` (MMQ)** | **5,786.4** | **88.1%** | 486.4 | 17.2% |
| **`gated_delta_net_cuda`** | **214.0** | **3.3%** | 145.5 | 5.1% |
| MMVQ dense (`mul_mat_vec_q`) | 53.7 | 0.8% | 752.5 | 26.5% |
| cuBLAS INT8 gemm (`h884gemm`) | 0.0 | 0.0% | 403.3 | 14.2% |
| `dequantize_*` (all types) | 3.6 | 0.1% | 263.0 | 9.3% |
| `ssm_conv_*` (GDN glue) | 78.6 | 1.2% | 26.7 | 0.9% |
| `concat_non_cont` + `cpy_scalar` + `fwht` (GDN glue) | 100.9 | 1.5% | 84.5 | 3.0% |
| `convert_unary` (f16<->f32 around cuBLAS) | 0.0 | 0.0% | 90.7 | 3.2% |
| `k_get_rows*` | 29.3 | 0.4% | 129.8 | 4.6% |
| **FlashAttention** | **60.7** | **0.9%** | 87.1 | 3.1% |
| `quantize_q8_1` + `quantize_mmq_q8_1` | 58.9 | 0.9% | 144.2 | 5.1% |
| cuBLAS FP32 gemm + splitK reduce | 67.2 | 1.0% | 58.3 | 2.1% |
| `rms_norm` / `unary` / `bin_bcast` / router / rest | ~110 | 1.7% | ~130 | 4.6% |
| **total kernel busy** | **6,569.8** | | **2,836.3** | |

`mul_mat_q` split by weight type, ub 32 (every launch is `J=32, fallback=false`):

| type | ms | share of prefill | what it is |
|---|---:|---:|---|
| IQ3_S | 3,473 | **52.9%** | MoE `ffn_moe_gate`/`up`, 41 layers |
| IQ4_XS | 1,508 | 23.0% | MoE `down` + dense attention/ssm |
| Q8_0 | 815 | 12.4% | dense q8_0 matmuls |
| Q6_K | 175 | 2.7% | `output.weight` (lm_head, 1 launch) |
| Q3_K | 106 | 1.6% | |

**Verdict on the prediction.** Two of the three headline numbers missed and one was right by
accident of a different mechanism:

- *MMQ at ub 32, predicted 65-80%*: **88.1%**, slightly above the band. Right in substance.
- *MMQ at ub 512, predicted 55-68%*: **17.2%**. Wrong by 3-4x, and for a reason worth knowing -
  MMQ is not *used* above 64 columns (section 3). The same work reappears as MMVQ dense (26.5%) +
  cuBLAS INT8 (14.2%) + dequant (9.3%) + convert (3.2%).
- *GDN at ub 512, predicted 13-18%*: **5.1%**. *At ub 32, predicted 6-8%*: **3.3%**. Both low.
- *FA, predicted 4-8%*: 3.1% at ub 512, 0.9% at ub 32. Low.
- *Prediction 1 (GDN absolute time invariant, 400-500 ms)*: **wrong by 2-3x and not invariant** -
  145.5 ms at ub 512 against 214.0 ms at ub 32. The serial loop is cheaper at wide width because
  there are 60 launches instead of 540: at ub 32 the per-launch cost is a real fraction of it, so
  GDN is one of the few things a *wider* ubatch actually helps.
- *Prediction 2 (the falsifier)*: **not triggered**. GDN is not >= 40% of anything; it is 3.3-5.5%.

## 3. Why the two widths use different engines - the 64-token threshold

The inversion is not a measurement artefact. It is one constant in `mmq.cu`:

- `mmq.cuh:9`: `#define MMQ_DP4A_MAX_BATCH_SIZE 64`
- `mmq.cu:450`: `ggml_cuda_should_use_mmq()` returns, for NVIDIA cards with fp16 MMA hardware,
  `ne11 < MMQ_DP4A_MAX_BATCH_SIZE`. `fp16_mma_hardware_available()` is true for `cc >= VOLTA`
  (`common.cuh:337`), so on sm_72 the test is simply **`n_tokens < 64`**.
- `ggml-cuda.cu:2160` (dense matmuls) and `:2189` (MUL_MAT_ID) both pass the **token count** as
  `ne11`.

So a ubatch of 32 tokens computes every quantized matmul on **MMQ**, and a ubatch of 64 or more
takes the MoE off MMQ entirely. Past that point `ggml_cuda_mul_mat_id` falls through to the
per-expert path: it loops over experts and launches one matmul per expert, each dispatched again by
its own token count (<= 8 -> MMVQ, 9-63 -> MMQ, >= 64 -> dequant + cuBLAS). The trace shows exactly
that mixture at ub 512 - ~24k small `mul_mat_q` / `mul_mat_vec_q` / `h884gemm` launches in the
508-token ubatch, against 2 fused MMQ launches per layer (gate, up) at ub 32.

Two structural facts fall out of the trace and matter for future work:

- **At `-ub 32` the MoE is one fused MMQ launch per tensor per layer** (`IQ3_S` launches: 1,440 for
  18 ubatches x 41 layers = 1.95 per layer, i.e. gate and up). It is not per-expert. That is the
  efficient shape and it is why production prefill is MMQ-dominated.
- **At `-ub 512` it is ~256 per-expert GEMMs per layer**, each too narrow to reach cuBLAS's
  efficient tile. This is the mechanism behind HANDOFF's observation that "width selects which
  kernel computes the prompt" - but it is a *dispatch threshold*, not a `J` specialization.

`J` was checked directly in the trace rather than inferred: at ub 32 every MMQ launch carries
`(int)32` with `fallback=false`; at ub 512 the MMQ launches that remain carry `(int)16/24/32/40/48/64`
and come from the per-expert path where each expert's token count is its own `J`. There is no
`J=512` launch anywhere, because at 512 columns MMQ is not selected at all.

## 4. What this does to the campaign

**H4 (chunked GDN) is not the lever, and its ceiling is now measured.** GDN's core kernel is
214 ms of a 6,570 ms prefill. Add every GDN adjunct - `ssm_conv` 78.6, `concat_non_cont` 49.3,
`cpy_scalar` 46.0, `fwht` 5.6 - and the whole delta-net block is **393 ms = 6.0%**. Porting the
chunked MMA schedule would have to make *all* of it free to buy 6%, and it attacks only the 214 ms
core, i.e. **3.3%**. The days-to-weeks estimate cannot be justified against that, and the
"1000+ tok/s" framing in `TODO.md`/`HANDOFF.md` was an assumption about GDN's share that nobody had
measured. It is retracted below.

**H5 (FA prefill schedule) is smaller still:** 0.9% at production width, 3.1% at ub 512, and that
is the whole of what 11 of 41 layers cost.

**H3 is the campaign, and its target changed.** The prefill budget is a single kernel family at a
single configuration: `mul_mat_q`, `J=32`, `fallback=false`, on the **Ampere** config table
(`mmq.cuh:261-266`), which for IQ3_S at every `J` is the same row -
`nthreads 256, occupancy 1, I 128, SRAM_LAYOUT_Q8_0, K_vram MMQ_ITER_K, stream_k true`. Nothing
there was chosen for an 8-SM, 64 KB-shared-memory Volta part, occupancy 1 gives at most 8 warps/SM,
and it is 88% of the time. That is a well-motivated and *small* tuning surface: one config per type
at one `J`.

**What H3 cannot do, and the premise it retires.** H3 as written in `HANDOFF.md`/`TODO.md` was "pin
a fast MMQ `J` specialization at wide `J` and get the measured 2.4x prefill at unchanged
acceptance". That is not reachable by this route, and the reason is in section 3: **at wide `J`,
MMQ is not running.** The 2.4x at ub 512 is exactly the engine change (fused MMQ -> per-expert
MMVQ/cuBLAS), so any attempt to keep the wide-width speed while restoring the control's numerics
has to make cuBLAS reproduce MMQ's bits, which is not a tiling question. Conversely, making the
ub-32 path 2.4x faster means making *MMQ itself* 2.4x faster, which is where the tuning work now
points - and the honest expectation there is the +5-15% H3 was originally scoped for, not 2.4x.

**Numerics guidance for that sweep, from the config struct (`mmq.cuh:167-181`) and the kernel:**
`I` (output-tile rows) and `occupancy` change which rows a block owns, not the order in which a
thread walks K, so they should be bit-preserving. `nthreads`, `K_vram` and `stream_k` change how
the K reduction is split and are the ones to expect a moved acceptance from - the same class as the
f32-router change that cost -45.6%. Every arm still reports acceptance and mean len per rep; the
control signature is 0.96657 / 5.68 and it does not bend.

## 5. TTFT arithmetic from the measured budget

At 590 tokens, ub 32: 6,570 ms of kernel time, 7,491 ms of TTFT. So each percent of prefill kernel
time is about 66 ms of TTFT, and:

| change | kernel time | TTFT at 590 tok | note |
|---|---:|---:|---|
| as measured | 6,570 ms | 7,491 ms | |
| GDN block entirely free | 6,176 | ~7,100 | H4's absolute ceiling, 6% |
| FA entirely free | 6,509 | ~7,430 | H5's absolute ceiling, 0.9% |
| MMQ 10% faster | 5,991 | ~6,910 | ~8% |
| MMQ 25% faster | 5,123 | ~6,040 | ~19% |

The other TTFT lever is not a kernel at all: **5.21 s of the first request after load is graph
capture** (12,703 ms against 7,491 ms warm at 590 tokens). At the 590-token prompt that is 41% of
TTFT and it is moved out of the user's path by a warm-up request at load time, which nobody has
tried. At 9.5k tokens it is 4% - the two levers matter at different prompt sizes.

## 6. Traps this trace set laid, for the next person

- **Group by kernel family, never by signature-plus-grid.** `flash_attn_tile` appears as `gridX=20`
  in one ubatch and `gridX=127` in the other; a `GROUP BY name, gridX` with a `LIMIT` silently
  dropped one of them and made FlashAttention vanish from a window it was 3.6% of.
- **`nsys-kernels.py` truncates at 58 characters**, which cuts off the `ggml_type` and `J` template
  arguments - the two fields that matter most here. Query the sqlite for the full `StringIds.value`
  when the type matters; the enum values are `8 Q8_0, 14 Q6_K, 21 IQ3_S, 23 IQ4_XS`.
- **`gridX` is not the column count in the sense the helper's docstring implies.** For MMQ it is
  `ntx` (column tiles) *or* the stream-k block count depending on branch, and for the per-expert
  path it is the destination-row tiling. Read `J` out of the signature instead.
- **Normalise every timestamp by `MIN(start)` before filtering.** The kernel timestamps are
  absolute nanoseconds (~2.1e17); comparing a normalised window against them returns zero rows
  silently and looks like "no kernels in this phase".
- A prefill trace at `-ub 32` is ~18 ubatches, and the last one is short (14 tokens here). It still
  uses MMQ, because the threshold is 64 and 14 < 64, so there is no engine mix to correct for.

See also `findings/2026-11-04-prefill-starting-point.md` for the baseline and harness, and
`findings/2026-11-03-decode-pass-budget.md` for the decode-side equivalent of this measurement.
