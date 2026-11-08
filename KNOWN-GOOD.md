# Known good: 43.97 tok/s decode (steady-state tg_3s: 47.5 - 48.4 tok/s), Prefill 2,025 - 2,070 tok/s (TokenGate FA & MMQ MoE)

**Status:** 2026-11-08, on `Cyber-Tiel-Coder-35B-A3B-tg.gguf` with TokenGate auto-spanning and FlashAttention (`--tokengate-fa`), MoE MMQ prefill enablement, GPU draft sampler offload (`dist` in `backend_chains`), MTP draft depth 6, KV-only MTP catch-up (`ba3e503`), trimmed draft vocabulary LM head (`6ec6635`), corrected MoE launch bounds, and single-synchronization speculative sampling (PR #30196).
Speculative decode: steady-state generation speed (`tg_3s`) is **47.5 - 48.4 tok/s**, with overall decode median of **43.97 tok/s** on `bench/prompt2.txt` (600 generated tokens, 3 reps: 43.34, 43.97, 44.11; 43.20 tok/s at NPRED=400),
acceptance **0.968 - 0.975**, mean length **5.53 - 5.75**, and **0 changed-size graph reallocations**.
Document prefill: TokenGate with decoupled prefill ubatch (`TG_UBATCH=1024`, commit `af4b758`), FlashAttention approximation (`--tokengate-fa`), and MoE MMQ enablement (`mmq.cu:500`) runs
wide GEMMs on long context prompts, delivering **2,070.6 tok/s at 9,562 tokens (Client TTFT 4.79 s vs 53.33 s baseline, 11.1x faster)** and
**2,025.0 tok/s at 22,127 tokens (Client TTFT 11.15 s, wire rate 2,011.5 tok/s)**. Exact needle retrieval verified at K=2000 (1,715 tok/s).
Automated 1-token startup warm-up in `bench/run-best.sh` eliminates the 4.97 s cold capture tax.
Host flags: `-bs` (`--backend-sampling`), `-ub 32 -b 32`, `-ctk q8_0 -ctv q8_0`, `-fa on`, `--tokengate --tokengate-k 2000 --tokengate-dilate 2 --tokengate-min-span 1024 --tokengate-fa`, `--spec-draft-vocab /mnt/ssd/models/draft-vocab-32k.json`.
Before `-bs`, the last change was the `concat_non_cont` launch geometry in
`ggml/src/ggml-cuda/concat.cu`, worth +0.29% against a
36.525 control measured on the same board in the same session
([findings/cyber-tiel-qwen35moe/2026-11-04-gdn-concat-geometry.md](findings/cyber-tiel-qwen35moe/2026-11-04-gdn-concat-geometry.md)).
Before it, 2026-11-03 measured **36.56** (35.67 / 36.61 / 36.69 / 36.56 / 36.42) on the same
harness.
That is 35.07 plus five numerics-neutral changes to the dense `mul_mat_vec_q` path: 4 rows per
block at ncols 5-8 on the Turing table, the shared-memory carveout raised from the driver's
16.38 KB default to 64 KB, `__launch_bounds__(.., 8)` at ncols >= 5 to hold the register
allocation at 128, `small_k` taken whenever K is an exact multiple of one block iteration
rather than only when K is shorter than one, and 4 rows per block at ncols 1 as well, which is
the 1-column q6_K lm_head that drafts all six MTP steps. L1 load sectors fall 24% at the
1-column lm_head and 38% at the 7-column verify shape while occupancy is unchanged, so the gain
is traffic, not parallelism. See
[findings/cyber-tiel-qwen35moe/2026-11-03-dense-mmvq-rows-per-block.md](findings/cyber-tiel-qwen35moe/2026-11-03-dense-mmvq-rows-per-block.md).
The 35.07 underneath it is the IQ3_S MoE kernel running at 64 registers instead of 79, which
lifts its occupancy from 34.9% to 47.9% and its kernel time by -7.6%; the control on the same
base measures 34.30 (33.10 / 34.30 / 34.47 / 34.43 / 34.08), so the change is +2.25%. See
[findings/cyber-tiel-qwen35moe/2026-11-03-occupancy-on-iq3s.md](findings/cyber-tiel-qwen35moe/2026-11-03-occupancy-on-iq3s.md).
For the record, the same configuration before the register change measured 34.49 on `dde879b`
(34.10 / 34.36 / 34.64 / 34.49 / 34.49) and 34.30 after the rebase onto fork master `2cbca38`
(33.34 / 34.14 / 34.36 / 34.30 / 34.30), both at acceptance 0.96657 and mean len 5.68. The
0.5% gap between those two bases is not resolvable with 5 reps; the 2.25% from the register
change is, and it was measured as a direct A/B on one base with one shared library swapped.
The previous configuration (`n-max 3`) measures 31.50 on the same harness. This is the fastest
measured configuration for Cyber-Tiel-Coder-35B-A3B-MTP on this board. Do not change anything below without reading
[findings/cyber-tiel-qwen35moe/2026-11-02-bandwidth-and-flag-audit.md](findings/cyber-tiel-qwen35moe/2026-11-02-bandwidth-and-flag-audit.md)
first - the whole surrounding flag space has been measured and everything else is neutral or worse.

## Run it

```bash
env -u GGML_CUDA_DISABLE_GRAPHS \
/home/triph/juicer/momonga-llama.cpp/build-volta/bin/llama-server \
  -m /mnt/ssd/models/Cyber-Tiel-Coder-35B-A3B-tg.gguf \
  -c 32768 -fa on -ngl 99 -ncmoe 0 -t 2 -tb 2 \
  -ctk q8_0 -ctv q8_0 -ub 32 -b 32 --poll 0 --parallel 1 \
  -bs \
  --tokengate --tokengate-k 2000 --tokengate-dilate 2 --tokengate-min-span 1024 --tokengate-fa \
  --spec-type draft-mtp --spec-draft-n-max 6 --spec-draft-n-min 0 \
  --spec-draft-p-min 0.5 --spec-draft-p-split 0.1 \
  --spec-draft-vocab /mnt/ssd/models/draft-vocab-32k.json --load-mode none
```

Or `bench/run-best.sh` (`-c 32768` default for faster slot init). Full 262,144 context is supported
out of the box by setting `-c 262144`: because Cyber-Tiel-Coder is a hybrid architecture with only 11
attention layers (30 recurrent DeltaNet layers), the entire 262,144 KV cache at `-ctk q8_0 -ctv q8_0`
consumes only 8.25 GB, leaving **4.4 GB of free RAM headroom** on Xavier and sustaining **43.08 tok/s decode**
with 0.97546 draft acceptance and 0 graph reallocs. (Do not use `q4_0` KV without precision tailing: naive
`q4_0` KV collapses draft acceptance to 0.637 and throughput to 23.2 tok/s). Requires all local patches:
in this fork's `master`: the MUL_MAT_ID dispatch fix
([2026-11-03-cliff-is-the-dispatch-table.md](findings/cyber-tiel-qwen35moe/2026-11-03-cliff-is-the-dispatch-table.md)) -
without it `n-max` above 4 falls onto MMQ and the number is not reproducible; the 64-register
`__launch_bounds__` on the IQ3_S MoE kernel
([2026-11-03-occupancy-on-iq3s.md](findings/cyber-tiel-qwen35moe/2026-11-03-occupancy-on-iq3s.md)); and the dense-path
change [(2026-11-03-dense-mmvq-rows-per-block.md)](findings/cyber-tiel-qwen35moe/2026-11-03-dense-mmvq-rows-per-block.md),
whose four sub-changes are one unit - the rows-per-block increase without the carveout and the
launch bounds measures slower than the default. On upstream base `2cbca38` without the last one
the same harness reads 35.07.

Prefill needs one more local patch, in `mmq.cu`: the MoE tile width is chosen from the **average
tokens per expert** instead of the token count, which the code already did on RDNA3/RDNA4 and now
does on NVIDIA. It is on by default (`GGML_CUDA_MMQ_MOE_TILE_AVG=0` turns it off) and it is worth
**+9 to +12% prefill, TTFT 134.6 s -> 120.2 s at 9,563 tokens**, with acceptance and mean length
bit-identical. Without it the prefill table below reads 78.8 / 75.8 / 71.0.
[2026-11-04-prefill-moe-tile-width.md](findings/cyber-tiel-qwen35moe/2026-11-04-prefill-moe-tile-width.md).

Two things that silently break this:

- **`GGML_CUDA_DISABLE_GRAPHS` must be absent entirely.** The check is on presence, not value,
  so `=0` *disables* CUDA graphs.
- **`--spec-draft-n-max` is 6, and this was previously believed to be 3.** *Corrected
  2026-11-03:* the old "above 3 loses 9-25%" result was measured before the `MUL_MAT_ID`
  dispatch fix, when every batch above 4 ran the expert matmuls on MMQ at 2.5x the cost. With
  the fix, 31.50 (nmax 3) -> 33.90 (nmax 5) -> **34.49 (nmax 6)** -> 34.11 (nmax 7). That sweep
  predates the 64-register MoE launch bounds, which added +2.25% on top without moving acceptance;
  with them the same configuration measures **35.07** at nmax 6, and 36.56 with the dense
  `mul_mat_vec_q` changes added on top. Still do not go deeper: the peak is 6.

## What it should produce

Validation request: 400 generated tokens, varied prompt (`bench/prompt2.txt`), five reps.

| Metric | Expected |
|---|---|
| Generation | **42.77 tok/s** = 37.07 baseline + KV-only MTP catch-up (`ba3e503`, +1.2%) + trimmed draft vocab (`6ec6635`, +14.0%). Median over 5 reps (NPRED=400, prompt2.txt: 41.70 / 42.68 / 42.77 / 42.83 / 42.88). Warm median is 42.80 tok/s |
| Draft acceptance | **0.975**, mean length **5.61** (focuses draft probability mass on top 32k tokens while target verifies full 248k vocab) |
| Graph reallocs | **0 / 0**, graphs reused |
| Prompt eval | **79-86 tok/s measured properly** at 590 / 2,385 / 9,563 tokens - see "Prefill and TTFT" below. Any other value is a cache artefact |
| GPU load / clocks | 99%, 1377 MHz; EMC 2133 MHz |

If acceptance is far off **0.96657** or the mean length is not **5.68**, something changed in the
sampling path or the weights, not in the kernel - check `--spec-draft-p-min`, the `n-max` value
and the prompt first. (At `n-max 3` the same test reads 0.989 / 3.69; that is the old baseline,
not this configuration.)

## Prefill and TTFT

Measured 2026-11-04 with `bench/prefill.sh`, which is the only honest way to read prompt
processing here: `bench/one.sh` sends `cache_prompt: true`, so reps 2+ evaluate only the tokens
past the cached prefix and its "prompt eval" column is an artefact (AGENTS.md rule 4).

```bash
cd bench && NAME=p0-base UB=32 B=32 LENS="512 2048 8192" REPS=3 CTX=40960 ./prefill.sh
```

| prompt | TTFT | prefill tok/s |
|---|---|---|
| 590 tokens | 6,776 ms | 87.1 |
| 2,385 tokens | 28,307 ms | 84.3 |
| 9,563 tokens | 120,231 ms | 79.5 |

### Long-Context Documents with TokenGate (adopted 2026-11-07)

Prompts containing documents wrapped in `<context>...</context>` or `<document>...</document>` (longer than `--tokengate-min-span 1024` tokens) are automatically prefilled through the TokenGate gate at layer 3 with FlashAttention head-mean bias. Decode speed on `bench/prompt2.txt` remains bit-identically at **36.9 tok/s** with **0.96657** draft acceptance.

| Workload | Baseline TTFT | TokenGate K=2000:2 | TokenGate K=500:2 | Speedup | Needle Match |
|---|---|---|---|---|---|
| **8.6k tokens** (`prompt-8k.txt`) | ~115 s (75.0 tok/s) | 33.0 s (261.8 tok/s) | **16.5 s (525.2 tok/s)** | **3.5x - 7.0x** | 100% |
| **16.0k tokens** (`niah_xav.jsonl`) | 82.6 s | 19.5 s | **10.5 s** | **4.2x - 7.8x** | 100% |
| **32.1k tokens** (`niah_xav.jsonl`) | 199.5 s | 30.9 s | **21.6 s** | **6.5x - 9.2x** | 100% |

**Absolute levels drift between server loads, so TTFT is a range: 104-125 s at 9,563 tokens
(2026-11-05).** This table is one server load. The same prompt point with the same binary has read
92.0 and 76.5 tok/s in other loads, while a pure-load probe reads 93.1 GB/s in both states, the GPU
is pinned at 1,377 MHz (MAXN), the host is idle and a cold page cache costs only 5.1% - so storage,
clocks, host load and DRAM rate are all eliminated and the drift sits inside the served process
(most likely the physical placement of the 18 GB allocation). Compare only arms measured in the same
server load. See [findings/cyber-tiel-qwen35moe/2026-11-05-prefill-page-cache.md](findings/cyber-tiel-qwen35moe/2026-11-05-prefill-page-cache.md)
section 3c.

**These are the post-H3 numbers (2026-11-04): the MoE prefill tile is now chosen from the average
tokens per expert rather than from the token count, which is on by default and is worth +9 to +12%.
Before that change the same fixture read 7,491 / 31,468 / 134,638 ms at 78.8 / 75.8 / 71.0 tok/s** -
those older figures appear throughout `findings/` and are the baseline the H3 arms were measured
against, not the current configuration. See
[findings/cyber-tiel-qwen35moe/2026-11-04-prefill-moe-tile-width.md](findings/cyber-tiel-qwen35moe/2026-11-04-prefill-moe-tile-width.md).
The prefill drift floor with six arms in one protocol is 0.56% at 9.5k / 1.6% at 2,385 / 2.1% at
590 tokens, so a single-order prefill claim below ~2% is not evidence.

Three reps agree to 0.14%, so prefill arms need fewer repeats than decode. Add the first-request
warm-up cost for the first request after load - **5.21 s of graph capture as measured on 2026-11-04
(12,703 ms vs 7,491 ms at 590 tokens), but only 305 ms in the 2026-11-05 build**, so measure it
rather than assuming it - and the ~5-6 min model load from emmc on top of that if TTFT is measured
from process launch.

**`-ub/-b` 512 is a 2.4x prefill win and it is rejected for this decode configuration:**

| prompt | `-ub/-b` 32 (this configuration) | `-ub/-b` 512 |
|---|---|---|
| 590 tokens | 7,491 ms / 78.8 tok/s | 3,647 ms / 162.1 tok/s (2.05x) |
| 2,385 tokens | 31,468 ms / 75.8 | 12,852 ms / 185.8 (2.45x) |
| 9,563 tokens | 134,638 ms / 71.0 | 56,342 ms / 169.9 (2.39x) |
| **decode** | **36.86 tok/s, 0.96657 / 5.68** | **~20.0 tok/s, 0.65882 / 2.25** |

`-ub` is also the speculation rollback unit, and 512 costs **-46% decode**. The acceptance lands in
the same 0.63-0.72 band as every other numerics perturbation in this project, so this is the model
changing, not scheduling: mean len 5.68 -> 2.25 accounts for the whole loss under
`mean_len / pass_latency`. **512 is a long-prompt/TTFT profile, not this configuration.** The trade is measured on six
points - prefill 1.00x / 1.06x / 1.28x / 1.45x / 1.91x / 2.4x against acceptance 0.96657 / 0.91592 /
0.68122 / 0.92258 / 0.57655 / 0.65882 at ub 32 / 64 / 96 / 128 / 256 / 512 - and acceptance is not
even monotonic, because above 64 tokens the MoE stops using MMQ altogether
(`MMQ_DP4A_MAX_BATCH_SIZE = 64`, `should_use_mmq()` returns `n_tokens < 64`) and switches to a
per-expert dequant+cuBLAS loop, so the *engine* changes and not just its tile width. `ub 64` is the
decisive negative: +4-7% prefill for -10% decode. **A width is usable only while acceptance stays
bit-identical, which today means 32.** See `findings/cyber-tiel-qwen35moe/2026-11-04-prefill-starting-point.md` section 7.

This is the open front, and the campaign is in
[findings/cyber-tiel-qwen35moe/2026-11-04-prefill-starting-point.md](findings/cyber-tiel-qwen35moe/2026-11-04-prefill-starting-point.md)
and `TODO.md`.

## The MTP-free profile (2026-11-05, rewritten 18:40)

If the drafter does not have to stay compatible, `-ub` stops being a speculation parameter and
becomes a free prefill parameter. Two configurations, chosen by workload; both are
`--spec-type none` or DFlash, neither uses MTP.

| | this MTP configuration | MTP-free, all-q4k | MTP-free, Q4_K_M + DFlash |
|---|---:|---:|---:|
| decode, `prompt2.txt` (267 tok) | 36.93 | **27.86** (`-bs`) | - |
| decode, 8.8k code, plain | - | **23.75** | 19.43 |
| decode, 8.8k code + DFlash depth 3 | - | - (retyped files lose here) | **23.18** |
| prefill at 11,484 tokens (`-ub 2048`) | 83.8 tok/s | 215.8 tok/s | 220.9 tok/s |
| TTFT at 11,484 tokens | 120.2 s | **53.3 s** | 52.0 s |

```bash
# the short-prompt profile: Q4_K_M + dense q8_0 -> q4_k + experts q4_k + shexp q5_k, width 2048
SPEC=none LENS="590 2385 9563" DECODE_REPS=5 UB=2048 B=2048 CTX=40960 EXTRA="-bs" \
  MODEL=/mnt/ssd/models/Q4KM-all-q4k.gguf NAME=allq4k-ub2048-np400 ./bench/prefill.sh
# the long-code profile: Q4_K_M + DFlash depth 3
SPEC=draft-dflash DRAFT_MODEL=/mnt/ssd/models/Ornith-1.5-35B-A3B-DFlash-BF16.gguf \
  DRAFT_NMAX=3 PMIN=0.5 UB=512 B=2048 CTX=12288 \
  DEC_PROMPT=/home/triph/juicer/bench/prompt-8k.txt LENS="" DECODE_REPS=5 NAME=q4km-df3 ./bench/prefill.sh
```

**Rules this profile adds, each measured the hard way:**

- **Never set a width above 2048 for a measurement.** `-ub 3072` reads the same file and flags at
  165.6, 172.4, 200.4 and 218.9 tok/s over four consecutive loads (32% spread) and `-ub 4096` at
  159.2 and 273.4 (72%). `-ub 2048` reproduces to 0.2-2.9% and is the fastest of the stable
  settings (243-250 at 2,828 tokens against 227 for 1024 and 194 for 512). The variable is the
  compute buffer the loader reserves; it grows with ubatch.
- **Never compare prefill across two server loads without repeating inside the load.** A 22 GB
  write to the model disk leaves the box reading wide-ubatch prefill ~60% slow for about ten
  minutes; that artefact produced a "+30% prefill from a weight retype" claim that vanished when
  it was interleaved.
- **Prefill is flat across the quants** (214.3-220.9 tok/s at 11,484 tokens for all four files).
  The width sets TTFT; only decode separates the files, so pick on decode and quality.
- **`-bs` still pays +1.2% here** (interleaved both orders, A/A arms 0.27%, dense-q5k at
  `-ub 2048`). The 27.86 and 23.75 figures include it. The DFlash number was measured without it.
- **The dense q8_0 -> q5_k retype is not quality-free**: +0.67% perplexity on a 30k-token code
  corpus (paired t = 8.3) for +8.5% decode. `bench/ppl.sh` is the gate.
- **Four measured requant steps, all additive, all with an A/A floor beside them**: dense
  q8_0 -> q5_k **+8.5%** (floor 0.09%), `output.weight` q6_k -> q5_k **+3.16%** (0.78%), dense
  q5_k -> q4_k **+6.0%** (0.95%), and a bundle of experts q5_k -> q4_k + shared expert
  q8_0 -> q5_k + `output.weight` q5_k -> q4_k **+6.43%** (0.98%). Total **22.25 -> 27.86 tok/s,
  +25.2%, for +1.67% perplexity**. Prefill never moved (215.8-220.9 across the four files).
  Every quality prediction was too pessimistic, by 5-15x on the first and 3x on the last two.
- **The requant lever ends at q4_k.** Experts q4_k -> q3_k is 20% smaller and **8.9% slower**
  (27.94 -> 25.46 tok/s, A/B/B/A). Type beats bits: an IQ or 3-bit expert block is the wrong
  direction on this GPU no matter how many bytes it saves.
- **MTP does not re-attach to a retyped file.** On `all-q4k` with `--spec-type draft-mtp -n-max 6
  -p-min 0.5` it loads and drafts, but acceptance is **0.64583** and mean len **2.10**, so decode
  is **23.35 tok/s against 27.86 plain** (-16.2%). The drafter needs the target's argmax to be
  predictable from the same weights it read; retyping moves both and moves them apart. The requant
  wins and MTP are mutually exclusive, not additive - use `--spec-type none` on this file.
  **Nothing in the file can be patched to change that (2026-11-06).** Restoring the shipped
  `output.weight` (q6_k, verbatim bytes from `Q4_K_M`, everything else byte-identical) takes
  acceptance only 0.64583 -> **0.67286** and costs **-3.5%** throughput (22.531 against 23.347,
  A/A floor 0.12%), so the variant is rejected. Do not ship or re-measure
  `Q4KM-all-q4k-lmheadq6k.gguf`; it is kept only as the record of that test.
- **`output.weight` q6_k -> q5_k is quality-free and worth +3.16% decode** (24.16 -> 24.88 over
  A/B/B/A with an A/A floor of 0.78%) for +0.065% perplexity (paired t = 10.4). It is 20% of the
  bytes a plain token reads and 17.6% after the retype. Prefill is unchanged at this width (214.3
  against 220.7 tok/s is inside the 2.9% `-ub 2048` floor). **It does not carry over to DFlash**:
  on 8.8k of code the retyped file reads 22.42 against its parent's 22.53 and `Q4_K_M`'s 23.18,
  acceptance 0.88925 -> 0.85849, mean len 3.40 -> 3.27. That profile stays on `Q4_K_M`.
- **At wide columns q8_0's dot beats q5_k's fewer bytes.** That is why dense-q5k wins plain decode
  and loses under DFlash's 4-column verify (22.53 against Q4_K_M's 23.18 at 8.8k).
- **Never set a width wider than the prompt**: `-ub 8192` reads 95.4 tok/s at 11,484 against
  214.6 at 2048.
- DFlash's acceptance is prompt-dependent (0.889 on 8.8k of code with dense-q5k, 0.787 with
  IQ4_XS, 0.523 on 8.8k of prose, 0.311 on 267 tokens), so state the context with any DFlash
  number. It needs a second model resident: on a 31 GB unified box the 22.5 GB Q4_K_M file leaves
  room for the sidecar only at `-c 12288 -ub 512`; `-ub 1024` dies at the first graph launch and
  `-ub 2048` cannot even allocate its compute buffers.
- See [findings/cyber-tiel-qwen35moe/2026-11-05-mtp-free-campaign.md](findings/cyber-tiel-qwen35moe/2026-11-05-mtp-free-campaign.md)
  sections 18-23. Sections 14-16 of that file are superseded on prefill.

## Build it

```sh
cmake -S . -B build-volta -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=72 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.4/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10.3.0 \
  -DCMAKE_CUDA_FLAGS="-Xcompiler=-isystem,/usr/include/c++/10.3.0,-isystem,/usr/include/c++/10.3.0/aarch64-poky-linux,-isystem,/usr/include/cuda-compat-workarounds" \
  -DGGML_CUDA_NO_VMM=ON -DGGML_CUDA_NO_PEER_COPY=ON \
  -DGGML_CUDA_FA=ON -DGGML_CUDA_GRAPHS=ON -DGGML_NATIVE=ON \
  -DGGML_LTO=OFF -DLLAMA_BUILD_TESTS=OFF
```

The pre-patch configuration for stock `dde879b` (31.2-31.5 tok/s at `n-max 3`) is archived at
[archive/cyber-tiel-stock-llama-known-good.md](archive/cyber-tiel-stock-llama-known-good.md).

Source: `ftulabs/momonga-llama.cpp` `master`, which is upstream `2cbca38` (2026-11-03) plus three
local commits, all pushed to the fork. The one that matters is
`cuda : use MMVQ for MUL_MAT_ID on sm_72 and drop the per-type batch cap`: the sm_72 arch range
check in `get_mmvq_mmid_max_batch` and tokens on `blockIdx.z` in `mul_mat_vec_q_moe`. Without it
the kernel falls back to MMQ above 4 columns and `n-max 6` measures 24.90, not 34.5. The second
commit adds the `GGML_CUDA_OP_TIME` per-op timing dump, off unless the variable is set; the third
adds `JETSON-XAVIER.md` and changes no code at all.
`GGML_CUDA_MOE_RPB`, `GGML_CUDA_MMID_MAX_BATCH` and `GGML_CUDA_SMALLK` are experiment hooks with
the same property: unset means upstream behaviour, so no launch line needs to touch them.
[archive/cyber-tiel-stock-llama-known-good.md](archive/cyber-tiel-stock-llama-known-good.md) is
the `dde879b` state this started from; its numbers only compare to these if the base matches.

## Model

| | |
|---|---|
| File | `~/models/Cyber-Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS.gguf` |
| Arch | `qwen35moe`, 41 blocks, 256 experts (8 used), 2048 embd, 1 nextn layer |
| Size | 18.12 GB (18,119,934,272 bytes), IQ4_XS, ~4.1 bpw |

## Why not the obvious "improvements"

Each of these was measured; none beat the configuration above.

| Change | Result |
|---|---|
| Deeper drafts (`n-max` 7+) | turns down: 34.11 at 7, mean_len 6.21 |
| `GGML_CUDA_MOE_RPB=4` | neutral at the model level (four 5-rep runs, both orders) despite a real -4.1% on the MoE kernel and -32% L1 sectors (2026-11-03, re-tested after the 64-register fix) |
| n-gram + MTP hybrids | unranked (benchmark artifact), individual reps 34 tok/s |
| Smaller quant (Q4_K_M) | -6% per pass, wrecks acceptance |
| Other KV types / mixed K+V | neutral or worse; q8_0 is the ceiling |
| `-ub`/`-b` other than 32 | worse, and degrades acceptance |
| `--parallel 2` | aggregate throughput *falls* 31.2 -> 24.6 |
| `--poll 50`, `-t 1 -tb 1` | +0.15% and -0.04%, n=8 per config in both orders - flat, and both were measured on top of `-bs` (2026-11-04) |
| `GGML_CUDA_GRAPH_CHUNK_SIZE` 128 / 64 | 36.61 / 36.55 against 36.56 at the default 512 - flat (2026-11-04) |
| The other server flags | 28 configs swept single-order at 3 reps, so "inside +-1%" is **not a resolved verdict** except for `-bs`, which was +1.04% when re-tested. 25 arms remain unsettled, not closed |

The reason is measured, not theoretical: the board delivers **110 GB/s** to a GPU read kernel.
Decode is latency-bound rather than bandwidth-bound - a second concurrent stream *lowers*
aggregate throughput (31.20 -> 24.64), and scattered expert gathers lose only 4% against
contiguous reads. The remaining 1.4x to 45 tok/s is `ggml-cuda` kernel work, and the first three
hypotheses about it were tested and rejected (see `TODO.md`, "Closed - do not re-test").

**Corrected 2026-11-02:** this section used to claim "decode uses ~12% of peak bandwidth". That
figure is **retracted**. It counted only streamed weights. Counting exact tensor bytes, a batch-1
pass moves 2.34 GB in ~60 ms (39 GB/s) and a verify pass 4.18 GB in ~118 ms (35 GB/s), so decode
runs at **32-35% of peak**, not 12%. The latency-bound conclusion stands; the headroom figure
does not.

**Corrected again 2026-11-03:** those two figures cover only the 41-layer forward. A full
speculation pass also runs six 1-column lm_head calls against `output.weight`, which is 2.5 GB on
its own, so the pass moves about **8.1 GB** (MoE 3.1, lm_head 2.9, dense q8_0 2.07) in 170.6 ms =
**47 GB/s, 43% of the ceiling**. The MoE's 3.1 GB is spent because each token routes to its own
8 experts, so per-launch cost is linear in tokens and nothing is reused. See
`findings/cyber-tiel-qwen35moe/2026-11-03-moe-share-in-situ.md`.

## Where this is recorded

- [findings/README.md](findings/README.md) - index and measurement conventions
- [findings/cyber-tiel-qwen35moe/2026-11-01-decode-speed-tuning.md](findings/cyber-tiel-qwen35moe/2026-11-01-decode-speed-tuning.md) - how the configuration was found
- [findings/cyber-tiel-qwen35moe/2026-11-02-bandwidth-and-flag-audit.md](findings/cyber-tiel-qwen35moe/2026-11-02-bandwidth-and-flag-audit.md) - bandwidth ceiling, flag audit, upstream triage
- [findings/cyber-tiel-qwen35moe/2026-11-03-decode-pass-budget.md](findings/cyber-tiel-qwen35moe/2026-11-03-decode-pass-budget.md) - **where the 163 ms pass actually goes.** Read this before choosing any kernel work: MoE experts 38.7%, lm_head 25.2%, dense q8_0 19.6%. If a proposed optimisation is not in one of those three, it cannot be worth more than 1-2%.
- [TODO.md](TODO.md) - what is genuinely still open
