# Build A/Bs, quantization A/Bs, and other models

**Date:** 2026-10-14
**Build:** BeeLlama `58a1629` unless stated
**Status:** historical. The builds question was resolved on 2026-11-01 - see
[2026-11-01-decode-speed-tuning.md](2026-11-01-decode-speed-tuning.md).

## BeeLlama versus mainline llama.cpp (open question at this date)

- The completed sweep's executable was the local BeeLlama build, so **25.39 tok/s was BeeLlama
  performance**.
- At this point no matched mainline build had been benchmarked on this Xavier with this model or
  settings, so results could not say whether BeeLlama was faster than mainline.
- BeeLlama's public differentiators include KVarN KV quantization, precision tails, and extra
  low-bit cache types. These are primarily memory/context/quality tools; quantized KV can reduce
  traffic at long context but adds quantize/dequantize work. A speedup on unified-memory Xavier
  is not guaranteed.
- The sweep included Q8_0, Q4_0 and KVarN cache types but varied other factors simultaneously,
  so it cannot establish a causal cache-type ranking. The brief KVarN4 test is not reliable
  enough either.
- **Proper comparison:** build both for SM72 with comparable CUDA/compiler options, then
  alternate A/B/A/B using the same prompt, output length, context, model, MTP settings, cache
  type where possible, and clocks. Compare identical output tokens, report PP/TG separately.

**Resolved 2026-11-01:** mainline `dde879b` beats BeeLlama `58a1629` by about **6%** at matched
settings on varied text (22.27 vs 21.08 tok/s at draft 1, `-ub 1024`, 128-token window). The
roughly 29% gap an early degenerate-prompt comparison suggested was an artifact.

## Volta-focused llama.cpp fork research

Repository: [jackjusko/jusko-llama-volta-qwen3flash](https://github.com/jackjusko/jusko-llama-volta-qwen3flash);
local checkout `/home/triph/juicer/jusko-llama-volta-qwen3flash`.

- Builds on a Volta (SM70) / Turing (SM75) CUDA fork with further Qwen3.8-Flash-Next tuning.
  Published hardware is primarily Tesla V100, not Xavier SM72.
- Relevant overlap: its docs report Ornith-1.5-35B-A3B tests, directly relevant to this fine-tune.
- Its Volta 256x256 FlashAttention benchmark on V100 reports Ornith PP 485.49 -> 580.36 tok/s
  (+19.54%), while TG was 59.26 -> 59.39 (+0.21%). Mostly a prompt-processing improvement.
- A separate `GGML_CUDA_FORCE_MMQ=ON` Ornith test reports a large PP improvement (about +67-70%)
  and only about +0.8% TG on V100. The fork recommends a separate FORCE_MMQ build for routed MoE
  and cautions against using it globally for dense Qwen.
- Its +32.19% MTP TG result was on Qwen3.8-27B using specific Q5/Q6 and MTP3 kernels; not
  evidence of the same gain on this IQ4_XS Ornith fine-tune.
- Specialized Q5_K/Q6_K paths may not apply to IQ4_XS. CUDA version, kernel geometry, quant type,
  MTP architecture and SM72 behaviour all need validation.
- A separate llama.cpp issue on SM70/SM72 FlashAttention found custom tile changes were neutral
  to worse for decode (about 20.85 -> 19.68 tok/s pure decode) and suggested a future split-D
  WMMA kernel: [upstream issue #28037](https://github.com/ggml-org/llama.cpp/issues/28037).
- Another V100-specific fork reports +39.9% on 176K prompt processing but essentially flat decode,
  reinforcing that a prefill speedup does not imply faster generation:
  [sm70-attn benchmark](https://github.com/fishlikeX/sm70-attn/blob/main/README-sm70.md#performance).

**Verdict:** worth testing as a separate build after preserving the baseline; do not replace the
current server based on V100 numbers.

## Short A/B: BeeLlama vs `jusko-llama-volta-qwen3flash`

One server load per build; same model, MTP max 1, Q8_0 KV, `-ngl 99 -t 4 -tb 2 -ub 64 -b 64`,
poll 50, repack/op-offload on, auto load, graphs disabled. Both reserved a 262,144-token slot but
the actual prompt was 4,149 tokens. Three decode requests per build with shared-prefix reuse.
**Short-context A/B, not a 262K-filled-context benchmark.**

| Build | PP tok/s | TG tok/s | MTP acceptance | Draft coverage | Version |
|---|---:|---:|---:|---:|---|
| BeeLlama | 81.50 | **24.42** | 0.9389 | 0.4805 | 0.4.7-dev, `58a1629` |
| Jusko Volta fork | **82.13** | 21.36 | 0.7162 | 0.4141 | 0.4.1-dev, `d343373` |

Fork PP was effectively tied/slightly higher (+0.8%), TG about 12.5% lower. The fork also had
lower MTP draft acceptance/coverage, which likely contributed. Do not infer a general result from
one prompt; the builds are different upstream versions and the fork's headline MTP3 gains are for
Qwen3.8, not this Ornith fine-tune at MTP max 1.
Artifacts: `/tmp/opencode/xavier-bee-jusko-ab/{beellama,jusko}/` and `run.log`.

## Quantization A/B: Q4_K_M vs IQ4_XS

BeeLlama `58a1629`, separate server load each. 262,144-token slot reserved, actual 4,149-token
prompt; MTP max 1, Q8_0 KV, `-ngl 99 -t 4 -tb 2 -ub 64 -b 64`, poll 50, repack/op-offload on,
auto load, graphs disabled. Two reps, no verify reloads.

| Quant | PP tok/s | TG tok/s | MTP acceptance | Draft coverage |
|---|---:|---:|---:|---:|
| Q4_K_M | 83.32 | 24.04 | 0.8676 | 0.4609 |
| IQ4_XS | 81.79 | 24.39 | 0.9389 | 0.4805 |

Q4_K_M prefill ~1.9% faster; IQ4_XS decode ~1.5% faster. Differences are small - effectively
close absent repeated A/B rounds.
Artifacts: `/tmp/opencode/xavier-quant-ab/{q4_k_m,iq4_xs}/`.

## MiniCPM-V 4.7 quantization A/B: Q4_0 vs Q4_K_M

BeeLlama `58a1629`. One model load per quant: MTP off, 32K reserved slot, actual 4,149-token text
prompt, 256 generated tokens/request, two reps, Q8_0 KV, `-ngl 99 -t 4 -tb 2 -ub 64 -b 64`,
poll 50, repack/op-offload on, `-fa 1`, auto load, graphs disabled. Text only - no image
preprocessing or visual tokens measured.

| Quant | PP tok/s | TG tok/s |
|---|---:|---:|
| Q4_K_M | 89.63 | 19.68 |
| Q4_0 | 101.97 | **21.86** |

Q4_0 was ~13.7% faster at prefill and ~11.1% faster at decode. Initial result, no repeated A/B
rounds; does not establish quality or image-inference differences.
Artifacts: `/tmp/opencode/xavier-minicpm-quant-ab/{q4_k_m,q4_0}/`.

## MiniCPM-V DSpark/DFlash draft test

`/home/triph/models/MiniCPM-V-4.7-DSpark-BF16.gguf` as speculative draft for the Q4_0 target,
MTP off. A 9-point sweep varied draft GPU placement (0/2/5 layers) and draft KV type
(f16/q5_1/q4_0); 32K reserved slot, 4,149-token text prompt, two reps per config, no images.

| Draft layers on GPU | Draft KV | PP tok/s | TG tok/s |
|---:|---|---:|---:|
| 0 | f16 | 97.67 | 3.12 |
| 0 | q5_1 | 98.10 | 3.10 |
| 0 | q4_0 | 98.90 | 3.08 |
| 2 | f16 | 98.53 | 5.45 |
| 2 | q5_1 | 99.41 | 4.70 |
| 2 | q4_0 | 98.83 | 4.62 |
| 5 | f16 | 99.52 | **12.31** |
| 5 | q5_1 | 99.58 | 11.99 |
| 5 | q4_0 | 100.09 | 11.67 |

Best was all 5 draft layers on GPU with f16 draft KV: 12.31 tok/s decode - about **44% slower**
than the prior Q4_0 no-draft result (21.86) despite higher acceptance coverage. The DSpark draft
did not improve decode on Xavier in this setup.
Artifacts: `/tmp/opencode/xavier-minicpm-dflash/dspark-q4_0.csv`, `report.html`, `run.log`.

## MiniCPM-V Q4_K_M + DSpark Q4_K_M under Cyber-Tiel settings with KVarN4 KV

The 262K-slot attempt OOMed with Q8_0 KV, and with Q4_0 KV. KVarN4 succeeded.
Target `/home/triph/models/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf`, draft
`/home/triph/models/MiniCPM-V-4.7-DSpark-Q4_K_M.gguf`. Settings: MTP off, DFlash draft 5/5 layers
on GPU with f16 draft KV, 262,144-token slot, actual 4,149-token prompt, 2 reps,
`-ngl 99 -ncmoe 0 -t 2 -tb 2 -ub 32 -b 32 --poll 0`, FA on, graphs enabled, repack off,
KVarN4 K/V, op-offload on, auto load.

Result: **6.47 tok/s** decode, 18.70 tok/s prefill, draft acceptance 0.8265, coverage 0.707.
First successful measurement at the 262K reserved slot for this target+draft pair. Compare
cautiously with earlier MiniCPM runs; both slot and settings differed.
Artifacts: `/tmp/opencode/xavier-minicpm-q4km-dflash/cyberflags-kvarn4/`.

## Underdog-Saluki-27B IQ2-mix

Single pinned configuration (FFN placement `none`, no CPU FFN offload), two reps; 32K reserved
slot, actual 4,149-token prompt, 256 generated tokens/request, MTP off, Q8_0 KV, all 64 layers on
GPU, `-t 4 -tb 2 -ub 64 -b 64`, poll 50, repack/op-offload on, FA on, graphs disabled.

Result: **3.97 tok/s** decode, **17.09 tok/s** prefill. Took about 7m56s including warmup.
A dense 27B: IQ2-mix reduces storage, not per-token compute. Runs, but very slow interactively.
Artifacts: `/tmp/opencode/xavier-underdog-saluki-speed/single/`.

## MiniCPM-V Q4_K_M overnight 216K unified-KV search

Requested 2026-10-14. Target `/home/triph/models/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf`, draft
`/home/triph/models/MiniCPM-V-4.7-DSpark-Q4_K_M.gguf`, explicit unified KV, **221,184-token
reserved pool (216 x 1024)**, one slot, DFlash-family enabled, target >23 tok/s.

The checkpoint has a Markov head and `dflash.sample_from_anchor=true`, so `draft-dspark` is its
correct DFlash-family mode rather than the plain DFlash anchor interpretation. It has a trained
15-token block; longer draft lengths are not genuine trained capability.

Worktree: `/home/triph/.local/share/opencode/worktree/121db71f73e1b8f1a7a04b1dcac0e2dde2f4341e/minicpm-overnight-216k`.
Tmux session `xavier-minicpm-216k-overnight`. Artifacts
`/tmp/opencode/xavier-minicpm-216k-overnight-20261014/`.

One smoke test, then an eight-hour bounded resumable staged search: memory-fit/cache pairs; draft
lengths and probability cutoffs; 48-point balanced multi-factor screen; one-factor local
neighbours; multi-task confirmation; independent restart; a genuine near-full 220,160-token
prompt attempt. One server per configuration.

**Status:** setup run interrupted and resumed in the same tmux session; 0 completed main-search
configurations at restart. The first smoke load OOMed; the fit stage was to test alternative
cache pairs. Server verbosity was raised to capture context/unified-KV verification evidence.

Screen uses 512-token prompts, 128 output tokens, one sample; confirmations use 4,096-token
prompts, 256 outputs and three coding tasks with two reps each. Greedy sampling, EOS ignored to
force equal output lengths. Prefix reuse explicit; prefill from uncached warmups.

Supported runtime knobs include target/draft KV formats, asymmetric cache candidates,
target/draft batching and threads/polling, CUDA graphs/graph optimization, repack, op offload,
host/pinned buffers, DQ matvec/Q6_K/rows, MMQ tile limit, minimum op-offload batch, KVarN
materialization windows, exact KV tail, backend/draft sampling, draft GPU placement, checkpoint
count and load modes.

**Excluded by design:** no synthetic acceptance, n-gram prefix tricks, expert-count reduction,
context shrinkage or target requantization may qualify as a speed winner. The 23 tok/s criterion
requires all validation samples to clear it with real draft counters; lower results remain
diagnostic only.

Telemetry records temperatures, available RAM, and GPU/CPU frequency. Per-run memory/thermal
guards terminate only the harness-owned server; no privileged clock/power changes or system cache
flushes are made. Final results and an explicitly idealized weight-bandwidth roofline are appended
automatically by the harness. **>23 tok/s and a true maximum are goals, not promises; absent
successful validation the report must state that the requirement was not reached.**

**23 tok/s criterion: NOT established by independent validation.** No slower result is an
acceptable winner. A time-bounded staged search is not an exhaustive Cartesian enumeration or a
proof of a global maximum. OOM, errors, thermal/memory guards and timeouts are not throughput
measurements. Near-full-context measurements, when completed, are separate from merely reserving
the pool.

Artifacts: `summary.csv`, `results.jsonl`, per-run server logs/commands/props, `telemetry.jsonl`,
`roofline.json`.

**Illustrative non-speculative weight-streaming roofline:** 1.975 GB/pass at 137 GB/s nominal
peak -> **69.4 tok/s**. Assumes expert slices are accessed independently and all peak bandwidth
serves weights. Ignores KV/SSM traffic, routing, kernels, draft and verification costs. It is
neither achievable speed nor a bound on speculative output throughput. See
[performance-ceiling.md](performance-ceiling.md).
