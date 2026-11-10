# Handoff

Rewritten 2026-11-04, updated 2026-11-08 for whoever picks this up next, human or agent.

**One line of state:** the adopted configuration runs on `Cyber-Tiel-Coder-35B-A3B-tg.gguf`
(17.3 GB NVMe SSD) rebased on `upstream/master` (`f7591a6`),
decoding at **43.97 tok/s** median (steady-state `tg_3s`: **47.5 - 48.4 tok/s**) with the MTP drafter (0.968 - 0.975 draft acceptance, 5.53 - 5.75 mean len, 0 reallocs)
via GPU draft sampling offload (`common/speculative.cpp`), single-sync sampling (PR #30196), KV-only MTP catch-up (`ba3e503`), reduced draft vocabulary LM head (`6ec6635`, `--spec-draft-vocab /mnt/ssd/models/draft-vocab-32k.json`),
CUB hardware radix sort on CUDA 11.4 (`1f33695`), and corrected MoE launch bounds, with **TokenGate auto-spanning (`--tokengate-auto`)**, MoE MMQ enablement (`mmq.cu:500`), and FlashAttention approximation (`--tokengate-fa`),
which drives prefill throughput to **2,025.0 tok/s at 22,127 tokens** (TTFT **10.92 s**) and **2,070.6 tok/s at 9,562 tokens** (TTFT **4.79 s**),
with **100% exact needle retrieval (`BLUE-VELVET-99`)** at K=2000 (1,715 tok/s).
Full **262,144-token context allocation** is verified and supported out of the box with `-ctk q8_0 -ctv q8_0` (only 11 attention layers grow with context, consuming 8.25 GB KV cache and leaving **4.4 GB of free RAM headroom** on the 32 GB Xavier, decoding at **43.04-43.17 tok/s**).
Automated 1-token startup graph warm-up is built into `bench/run-best.sh`, eliminating the 4.97 s first-request cold capture tax.
Standard prefill and decode strictly preserve `-b 32 -ub 32`. Branch `master` is fully synchronized with `origin/master`.

**A second, separate profile exists as of 2026-11-05: without MTP.** If the drafter may be
dropped, `-ub` becomes a free prefill parameter. Two configurations, both current:

- **Short prompts, no speculation: 27.86 tok/s decode (`-bs`), 215.8 tok/s prefill, 53.3 s TTFT** at
  11,484 tokens, with `Q4KM-all-q4k.gguf` (Q4_K_M with the dense attention/GDN set at q4_k,
  `output.weight` at q4_k, every expert at q4_k and the shared expert at q5_k; four measured
  steps, +25.2% decode for +1.67% perplexity, 20.461 GB) at
  `--spec-type none -ub 2048 -c 40960`. At 8.8k of context the same file reads **23.75 tok/s**
  (against 20.91 for dense-q5k, +13.6%).
- **Long code context with the DFlash sidecar at depth 3: 23.18 tok/s** on 8.8k of C++, with the
  plain Q4_K_M file at `-c 12288 -ub 512`. Dense-q5k reads 22.53 there, so the file that wins
  plain decode loses under a 4-column verify.
- **MTP cannot be re-attached to the adopted file.** `--spec-type draft-mtp -n-max 6 -p-min 0.5`
  loads and drafts (blk.40 is intact), but acceptance is **0.64583**, mean len **2.10** and decode
  **23.351 tok/s against 27.86 plain, -16.2%**. The two profiles are exclusive, not additive; the
  retyped file runs `--spec-type none`.

Prefill and TTFT no longer separate the three quant files at all (215-221 tok/s, 52-53 s), and
**`-ub 2048` is the widest width that measures reproducibly** - 3072 and 4096 scatter by 32-72%
over identical loads. Anything measured at 3072 for prefill before 2026-11-05 is superseded.
That is 65% of the MTP decode for a 2.3x shorter TTFT. Both profiles are current; neither
replaces the other.

**Answered 2026-11-06 01:34: no, the shared `output.weight` is not the driver.** `Q4KM-all-q4k.gguf`
with `Q4_K_M`'s q6_k `output.weight` swapped in (verbatim bytes, 752/752 other tensors verified
identical) reads acceptance **0.67286** against the control's 0.64583, and **-3.5%** throughput
(22.531 against 23.347 at an A/A floor of 0.12%) - the prediction was 0.85-0.95 and it missed. The
q6_k lm_head adds 0.9 GB to a pass that moves 8.1 GB and costs 18% of pass time. Combined with the
fact that `h` distance and argmax drift both fail to rank the three files, MTP on a retyped file is
not recoverable by file surgery, and a **retrained head is the only route left** - which needs the
extraction in the next paragraph and the T4 box (still unreachable from here).

**Extraction, ready and running: 8.99M tokens of corpus, 55 tok/s, 8.2 GB per 1M positions.**
`bench/make-mtp-corpus.py` builds 3,503 shards from the Python stdlib, both llama.cpp trees, a JS/TS
app and this repo's prose; `bench/run-mtpdump.sh` dumps `(h, token_in, argmax)` shard by shard into
`/mnt/ssd/models/_mtp/dumps/`, resumable at any shard boundary, `--batch 8` to match the 6-7-wide
verify pass. Nothing in the training path depends on a notification; the dump is on disk. The two shipped
profiles themselves are in `KNOWN-GOOD.md` ("The best configuration" and "The MTP-free profile") and
`findings/cyber-tiel-qwen35moe/2026-11-05-mtp-free-campaign.md` sections 18-23.

Read this file, then `AGENTS.md` (behaviours and traps), then `TODO.md` ("Closed - do not
re-test"), then `findings/`. `KNOWN-GOOD.md` is the configuration that is actually shipped.

Everything here is on-disk state. Nothing depends on a notification firing, and no background job
is needed to keep the project consistent.

---

## 1. Where things stand

**Machine:** Jetson AGX Xavier, Volta sm_72 (cc 720), CUDA 11.4 only, 30,991 MiB unified memory.
Measured sequential GPU read ceiling 109.7-112 GB/s. 8-core Carmel.

**Model:** `/mnt/ssd/models/Cyber-Tiel-Coder-35B-A3B-tg.gguf` (17.3 GB, NVMe SSD). Hybrid, not a plain
transformer: 41 blocks, 30 Gated DeltaNet (linear attention) + 11 full attention; 256 experts, 8
routed plus a shared expert; expert FF 512; hidden 2048; vocab 248320; one `nextn` MTP layer.
TokenGate G3 gate donor weights integrated at layer 3.
Only 11 of 41 layers attend, which is why attention tuning is worth little here.

**Best known decode configuration: 36.93 tok/s** = 36.63 kernel (median of 18 reps, both orders)
plus `-bs` (`--backend-sampling`, +1.04%, host-side, additive), with TokenGate active for document prefill.
Launch line:

```
--tokengate --tokengate-k 2000 --tokengate-dilate 2 --tokengate-min-span 1024 \
--spec-type draft-mtp --spec-draft-n-max 6 --spec-draft-n-min 0 \
--spec-draft-p-min 0.50 --spec-draft-p-split 0.1 \
-ub 32 -b 32 -ctk q8_0 -ctv q8_0 -fa on -bs -ngl 99 -ncmoe 0
```

plus the local `mmvq.cu` patches. Runner: `bench/run-best.sh`. Full launch line with every flag:
`KNOWN-GOOD.md`. Ladder, so a regression can be located: 36.63 -> 36.56 without the
`concat_non_cont` geometry -> 36.25 without the ncols-1 rows per block -> 35.07 without the dense
`mul_mat_vec_q` work -> 34.30 without the 64-register MoE launch bounds -> 31.50 before the
dispatch patch at `n-max 3`. Plain decode with no speculation is **27.86 tok/s** on `prompt2.txt` with the
all-q4k file and `-bs` (**23.75 on an 8.8k context**), and 23.18 with the DFlash sidecar on 8.8k of
code with the plain Q4_K_M file. The 18.32 / 16.37 figures this line used to carry were the same
files with the dense attention set still at q8_0.

**Acceptance 0.96657 and mean len 5.68 are the control signature.** They reproduce bit-for-bit on
every run ever logged. Any arm that reads something else moved numbers, not kernels, and its
throughput is not comparable (section 6).

---

## 2. The ubatch-width question: closed, and what it changed

The two sweeps finished on 2026-11-04. **The board is free, no sweeps are queued, and the width
lever is settled for the MTP configuration: it stays at `-ub 32 -b 32`.** (Without MTP the width
is free - see section 1. The two statements are about different configurations, not a
contradiction.) Full write-up in
`findings/cyber-tiel-qwen35moe/2026-11-04-prefill-starting-point.md` sections 7.1-7.2; raw rows in `bench/results.csv`,
`bench/spec.csv`, `bench/prefill.csv`.

| ub | prefill at 590 / 2,385 / 9,563 tok/s | speedup | acceptance | mean len | decode TG |
|---:|---|---|---|---|---|
| **32** | **78.8 / 75.8 / 71.0** | 1.00x | **0.96657** | **5.68** | **36.86** |
| 64 | 81.9 / 80.3 / 75.8 | 1.04-1.07x | 0.91592 | 5.12 | 33.2 |
| 96 | 100.3 / 97.9 / 90.5 | 1.27-1.29x | 0.68122 | 2.24 | 19.1 |
| 128 | 107.5 / 112.1 / 103.0 | 1.36-1.48x | 0.92258 | 4.45 | ~21.7 |
| 256 | 142 / 146.6 / 135.5 | 1.80-1.93x | 0.57655 | 2.22 | ~20.0 |
| 512 | 162.1 / 185.8 / 169.9 | 2.05-2.45x | 0.65882 | 2.25 | ~20.0 |

`ub 64` is the decisive arm: +4-7% prefill, which is inside the session's 0.71% A/A drift floor and
therefore no gain at all, for -10% decode. Strictly dominated, and every wider point is worse.

**Why this is more than a negative result.** Acceptance is *not monotonic* in width - 0.92 at 64,
0.68 at 96, 0.92 at 128, 0.58 at 256, 0.66 at 512. Smooth degradation would be monotonic; this is
not, so width is selecting *which kernel computes the prompt*, not merely perturbing it. That is
literally true, and H2 established the exact mechanism: `-ub` is the compute batch, so it is the
matmul column count during prompt processing, and crossing 64 tokens moves the MoE **off MMQ
entirely** onto a per-expert loop. The prompt's KV and GDN state come out of a different kernel and
the draft chain starts from those numbers.

**The 2.4x at 512 is an engine switch, not a config knob.** Earlier drafts of this file called a
numerics-safe wide-`J` MMQ config "the campaign's best shot, and it is H3", on the theory that MMQ
dispatches by exact `J == J_` and only the specialization needed pinning. H2 falsified that: MMQ is
88.1% of prefill at `-ub 32` and 17.2% at `-ub 512`, with no wide-`J` MMQ launch in the trace at
all, so there is nothing to pin. See section 4.

Two failure notes that should not be rediscovered:

- `-b 512 -ub 32` never ran: `NvMapMemAllocInternalTagged: error 12` / `CUDA error: out of memory`
  at graph capture. The `-b`-only prediction is **untested**, and `-b` is not a lever anyway -
  prefill works in ubatches.
- A stale `llama-server` that survives SIGTERM will hold 21 GB and stall the next arm's memory wait
  loop until `sudo kill -9`. Check `pgrep -x llama-server` after every kill.

## 3. Decode: closed as a front

Everything measured on 2026-11-04 after `-bs` was flat or negative: `--poll 50` +0.15%,
`-t 1 -tb 1` -0.04% (both n=8 per config, both orders), `GGML_CUDA_GRAPH_CHUNK_SIZE` 128/64 flat.
That flag run contained an **A/A control by construction** - the same config in the first and last
position - which read **0.71%**, and per-position means drifted +0.95% over the session. **On this
board no single-order difference below ~1% is evidence.**

What is left on decode is sub-1% each and none of it is cheap: conv rollback snapshot fusion
(~1.6 ms/pass, needs a new kernel), folding `quantize_q8_1` into the MMVQ dot (~1.7 ms), graph-prep
caching (<0.5%). The pass budget, measured in situ: one pass is 41 main layers + 6 MTP draft steps
+ 7 lm_head calls, 159.1 ms busy / 170.6 ms wall, MoE 63.98 ms (40.1%), lm_head 37.72 (23.7%),
dense q8_0 27.6 (17.3%), moving ~8.1 GB at 47 GB/s. The MoE kernel is a pure per-token DRAM stream
at 44-52 GB/s - launch-geometry levers in it are measurably inert at the model.

Do not reopen the kernel-geometry or quant space. `TODO.md` "Closed - do not re-test" has the
mechanism for each dead end.

---

## 4. The active front: prefill and TTFT

**Prefill runs at 79-86 tok/s, only 2x decode per token and ~5% of the int8 dp4a ceiling.** TTFT
at a 9,563-token prompt is **120.2 s in the recorded server load, and 104-125 s as a range**. It was
71-79 tok/s / 134.6 s until 2026-11-04, when the MoE tile width was fixed (H3, below, +9 to +12%);
before that every prefill number in this repo came from a harness that served the prompt from
cache.

**Absolute prefill levels are not stable between server loads (2026-11-05).** Same binary, same
flags, n_prompt 9,563: **92.0, 79.5 and 76.5 tok/s** across three server loads, while a pure-load
probe reads 93.1 GB/s in both the fastest and slowest state, the GPU is pinned at 1,377 MHz (MAXN),
the host is idle, and a cold page cache costs only 5.1%. Storage, binary, flags, clocks and DRAM
rate have each been eliminated; what is left is where the 18 GB model allocation lands, which is not
controllable here. **Quote prefill as a range, and compare only arms measured in the same server
load; a prefill delta below the session floor (0.56% at 9.5k, 1.6% at 2,385, 2.1% at 590) is not
evidence.** See `findings/cyber-tiel-qwen35moe/2026-11-05-prefill-page-cache.md` section 3c.

**What is left after that, honestly:** the MMQ tile geometry (`I`, `nthreads`, `J`, carveout) is
closed as inert - `I` 64 buys 4 more resident warps and measures -0.11 / -0.03 / -0.28% at the
three lengths, because the kernel already saturates its own LSU path at 16 warps. **The ubatch
width family is closed too (2026-11-05):** 33-63 was the last untested range, `-ub 48` pays
**+11.0% prefill** (87.0 vs 78.4 tok/s at 11,484 tokens) and destroys the drafter -
**acceptance 0.96657 -> 0.57955, mean len 5.68 -> 2.10, decode -49%** - and gate scans read
**36 = 0.88779** and **40 = 0.70755**, so every width above 32 breaks it and the map is not
monotone. The only remaining prefill lever that is not a rewrite is the `-ub` exchange at 96/128
(1.28x / 1.45x prefill for a decode loss), which the user has held in reserve.
See `findings/cyber-tiel-qwen35moe/2026-11-05-prefill-ubatch-intermediate.md`.

Harness: `bench/prefill.sh` - `cache_prompt:false` on every rep, `n_predict 1`, streams and clocks
the first chunk client side, records the server's `prompt_n`, drops rep 1. Reproducible to
**0.14%** over 3 reps, so prefill arms need fewer repeats than decode.

```bash
cd bench && NAME=p0-base UB=32 B=32 LENS="512 2048 8192" REPS=3 CTX=40960 ./prefill.sh
```

| prompt | TTFT now | tok/s now | TTFT before H3 | tok/s before H3 |
|---|---|---|---|---|
| 590 tokens | 6,776 ms | 87.1 | 7,491 ms | 78.8 |
| 2,385 tokens | 28,307 ms | 84.3 | 31,468 ms | 75.8 |
| 9,563 tokens | 120,231 ms | 79.5 | 134,638 ms | 71.0 |

The pre-H3 column is the baseline every arm in `bench/prefill.csv` was measured against; the
"now" column is the adopted configuration (median of two sessions, both orders at 9.5k:
120,019 / 120,689 ms, and 120,120 in the earlier session). Prefill's own drift floor, measured with
six arms in one protocol, is **0.56% at 9.5k, 1.6% at 2,385, 2.1% at 590 tokens** - a
single-order prefill claim below ~2% is not evidence.

`DECODE_REPS=5` added to the same script measures decode on `prompt2.txt` **in the same server
load**, before the prefills, and appends to `results.csv` / `spec.csv` in `one.sh`'s format. Use it
on every prefill arm: that is one emmc load per arm instead of two, and it enforces the rule below
by construction.

**The lever is known and it is not usable as a configuration: `-ub/-b` 32 -> 512 is 2.4x prefill
and -46% decode.** TTFT at 9.5k goes 134.6 s -> 56.3 s, and the numbers are real and reproducible -
they just cannot live in the same configuration as the decode record. The six-point curve is in
section 2. **The mechanism sentence in earlier drafts of this file said width "selects a different
MMQ `J` specialization". H2 corrected that:** at wide widths MMQ is not running on the MoE at all
(17.2% of prefill against 88.1%), because crossing 64 tokens switches the engine - so the *kernel*
changes, not just its `J`. See below.

**H2 is done and it reorders everything (2026-11-04).** One 590-token prefill traced at both
widths, kernel durations from hardware timestamps:

| family | `-ub 32` (6,569.8 ms) | `-ub 512` (2,836.3 ms) |
|---|---|---|
| `mul_mat_q` (MMQ) | **5,786.4 ms = 88.1%** | 486.4 = 17.2% |
| dense MMVQ | 53.7 = 0.8% | 752.5 = 26.5% |
| cuBLAS (f32 / INT8) | 67.2 = 1.0% | 403.3 = 14.2% |
| dequantize | - | 263.0 = 9.3% |
| `gated_delta_net` | 214.0 = **3.3%** | 145.5 = 5.1% |
| ssm_conv | 78.6 = 1.2% | - |
| FlashAttention | 60.7 = **0.9%** | 87.1 = 3.1% |
| concat + cpy + fwht | 100.9 = 1.5% | - |
| quantize | 58.9 = 0.9% | - |
| get_rows | - | 129.8 = 4.6% |

MMQ by weight type at `-ub 32`: **IQ3_S 3,473 ms = 52.9% of all prefill**, IQ4_XS 1,508 = 23.0%,
Q8_0 815 = 12.4%, Q6_K 175 = 2.7%, Q3_K 106 = 1.6%. Every launch is `J=32, fallback=false` on the
**Ampere** table.

- **H4 (chunked GDN) is closed as a campaign lever.** The entire delta-net block is 393 ms of 6,570
  = 6.0%; the core kernel is 3.3%. The "1000+ tok/s" premise was never measured and is retracted.
- **H5 (FA schedule) is closed.** Ceiling 0.9%.
- **H3 is answered, adopted, and it was the tile width (2026-11-04).** The original framing - pin
  a faster MMQ config at wide `J` - was falsified: above 64 tokens the MoE leaves MMQ entirely
  (`MMQ_DP4A_MAX_BATCH_SIZE = 64`, `should_use_mmq()` returns `n_tokens < 64` on any fp16-MMA card),
  so the 2.4x at `-ub 512` *is* the engine switch. But the `J=32` config that owns 88% of prefill
  had a different problem: at `-ub 32` each of 256 experts receives about one token of the 32, yet
  `J_best` comes from the token count, so every MoE launch runs a 32-column tile and 31/32 of its
  dp4a work is discarded (`vec_dot` has no column bound; only `write_back` honours
  `tile_y_max_j`). Applying the existing RDNA3/RDNA4 average-tokens-per-expert rule on NVIDIA
  (`ncols_opt = ceil(ne12*n_expert_used/ne02)` = 1, so `J = 8`) is **+9 to +12% prefill, TTFT
  134.6 s -> 120.2 s, with acceptance 0.96657 and mean len 5.68 bit-identical**. It is on by
  default in `mmq.cu`; `GGML_CUDA_MMQ_MOE_TILE_AVG=0` disables it. The 4x cut in dot work bought
  12.6%, not 4x, because the launch reads at only ~57 GB/s against this board's 111 GB/s ceiling
  (section 2g of the findings file corrects an earlier 114 GB/s claim: `gate` and `up` are two
  115 MB launches, not one fused launch), so neither the dot nor the bandwidth was saturated. Negatives in the same file: a 100%
  shared-memory carveout is inert at 9.5k (+0.4%), and narrowing the tile for IQ3_S only is 1.49%
  *slower* in both orders. What is left is the same wall decode hit - at a bit-preserving 32-token
  ubatch each expert's weights are streamed for about one token of work.

**Full detail in `findings/cyber-tiel-qwen35moe/2026-11-04-prefill-budget.md`**; the tool is `bench/prof-prefill.sh`
plus `bench/nsys-kernels.py`. Arithmetic at 590 tokens: **1% of prefill = 66 ms of TTFT**, the
GDN-free floor is ~7,100 ms, MMQ -10% is ~6,910 ms, MMQ -25% is ~6,040 ms.

**Non-kernel TTFT levers:** first-request graph capture costs **5.21 s** (measured) - a warm-up
request at load time would move it out of the user's TTFT and nobody has tried it; model load from
emmc is **~5-6 min** and the user is installing an NVMe (do not spend board time on it until it
lands); and `cache_prompt:true` gives follow-up turns a TTFT of hundreds of ms rather than 135 s,
which is the case most UIs present.

**Campaign rules:**

1. Every arm reports prefill at 3 lengths **and** decode tok/s, acceptance and mean len. The
   production score is 36.93 at 0.96657 / 5.68.
2. A prefill win that moves decode or acceptance is not a win.
3. Do not tune prefill with `bench/one.sh` - its prompt-processing column is cache-contaminated by
   construction (`AGENTS.md` rule 4).
4. Do not compare prefill numbers across sessions without the same fixture; `prompt_n` is recorded
   because the fixture's char/token ratio is not exactly 4.

---

## 5. Why a prefill change can cost decode: the mechanism

`-ub` is the **compute** batch, so it changes how a prompt is chunked, and three things ride on
that chunking: the GDN recurrent state is accumulated in f32 inside one call
(`gated_delta_net.cu:75` loops `t < n_tokens`), FlashAttention picks its tile from the query count
per call (`fattn.cu:704-712`), and the MoE router's discrete top-8-of-256 is evaluated at a
different column count. Any of them moves the prompt's hidden states in the last bits.

Why "last bits" is enough to cost 46%: the MTP drafter is **autoregressive over six steps**, so a
perturbation in its starting state compounds across the chain, while the target's verify pass does
not - the drafter's argmax stops coinciding with the target's and acceptance falls. That asymmetry
is the reason the project's tolerance is effectively zero, and it is the same mechanism behind the
requant, lm_head Q5_K, f32-router and jusko-fork failures, all of which land in a 0.63-0.72 band
while the untouched control reproduces 0.96657 bit-for-bit. (The compounding is inference, not a
direct measurement; the band is measured.)

**Consequence for this campaign: a prefill change is only free if it does not move the prompt's
numbers.** Prefer scheduling and launch-geometry changes that leave the arithmetic identical, and
measure acceptance on every arm rather than assuming prefill is "the other path".

---

## 6. The central constraint: the drafter needs identical numerics

Five independent perturbations now sit in the same 0.63-0.72 acceptance band - three expert
requants (0.64-0.93), the `output.weight` Q6_K -> Q5_K build (0.63603), the f32 router change
(0.67717) and the deleted jusko fork (0.678) - against a control that reproduces 0.96657 on every
run ever logged. **A change either touches no numbers or it is a different model.** Do not argue
that a quantization step is "small enough"; measure acceptance per rep and expect the worst.

That closes every non-bit-preserving weight change, all `--override-tensor` on per-pass tensors,
and the router's numerics (discrete top-8 selection is the most perturbation-sensitive number in
the model). A bit-preserving repack remains legal in principle, but the one candidate - padded
sector-aligned blocks - was measured: -37% L1 sectors, +90% registers, half the occupancy, 18-26%
slower kernels.

**The one remaining double-digit model-side lever is retraining the MTP head**, and it is a
research project: a drafter trained against the new weights would stop depending on bit-identical
numerics and could raise the ceiling above mean len 5.68-of-7. It needs a corpus of the target's
own hidden states (the Xavier can dump `h_nextn`; ~37 h of board time and 20 GB per 5M tokens) and
PyTorch work against the original full-precision MTP weights - llama.cpp's MTP layer is not set up
for training. Days to weeks, can fail. The T4 is the right machine for the training step
(section 11).

**State of that project on 2026-11-06 (this is the live thread).** Extraction completed cleanly
at 19:13:02 (rc=0) with **3,000,519 positions** banked across 1,190 shards (22.91 GB on
`/mnt/ssd/models/_mtp/dumps/`). The supervisor has exited and GPU VRAM is completely freed.
The fork's own head output is now a reference file: `mtpdump --head-logits` writes HLG1 and a
256-position reference on `shard-000000` reads **89.84%** head-vs-trunk argmax (56% at pos 0-31
rising to 100% past 96). Two traps are recorded in `TODO.md` under "Closed": a default context runs the
trunk and fakes 100% agreement, and `load_mtp` is required or `blk.40` is never loaded. `mtp-bundle/`
(1.34 GB) holds the head GGUF, the reference, its 256 input records, `mtp_head_torch.py`, the training
script `train-mtp-head.py`, and the format doc, for the user's laptop to relay because the Xavier
cannot reach `ml`.

The gate before any training was `mtp_head_torch.py --parity`. Evaluated on `ml` on 2026-11-07:
**98.83% argmax agreement (253/256)** and **mean logit cosine 0.999678** across 248,320 dimensions.
6 of 8 bands score 100.00% argmax agreement. The 3 mismatches were diagnosed as floating-point
tie-breaking on the 256-expert router at delta <= 0.0002 margins between Volta FP16 accumulation
and Turing FP32 dequantization (pos 177 delta is 8e-6). Non-divergence across all 256 positions
confirms mathematical parity within hardware precision limits. Gate authorized and Phase 1 training
cleared. See `findings/cyber-tiel-qwen35moe/2026-11-07-mtp-parity-gate-analysis.md`.

Phase 1 retraining (35.66M dense/norm parameters, MoE experts frozen) completed on `aorus-ts` (Tesla K80)
across 3,000,519 tokens (val CE 0.8005, argmax agreement 80.88%). The weights were exported to
`retrained_weights.npz`, relayed to Xavier, and spliced into `Q4KM-all-q4k-retrained.gguf` via
`bench/splice-retrained-mtp.py`. Verified byte-exact layout. Deployed benchmark measured **acceptance
0.49206** (mean len 2.08) and decode throughput **20.91 tok/s** (median reps 2-5), failing to improve
over the 0.64583 factory baseline and **24.9% slower** than plain decode (27.86 tok/s). Freezing 98%
of the block's capacity (1.826B MoE expert parameters) while adapting front-end dense layers distorts
representations entering the router, and teacher-forced single-step loss on code does not translate
to 6-step autoregressive drafting on prose. See `findings/cyber-tiel-qwen35moe/2026-11-07-mtp-phase1-deployment-results.md`.

---

## 7. Traps that will bite in the first hour

- **A second `llama-server` comes up at boot and eats 24.5 GB**, and `pgrep -x llama-server` does
  not find it because its `comm` is truncated to `llama-server-tg`. Stop it with
  `sudo systemctl stop llama-server` before any benchmark, and check `free -m`.
- **`free -m` used must be below 26 GB** before a new server can map 19 GB. `bench/one.sh` and
  `bench/prefill.sh` wait for that; a hand-written loop must too.
- **A server killed with SIGTERM can survive and hold 21 GB.** Check
  `pgrep -x llama-server` after every kill; `sudo kill -9 <pid>` if it lingers.
- **`bench/results.csv` does not record `n_predict`**, and `NPRED` is worth 17% of the TG number
  (128 reads 30.1 tok/s where 400 reads 36.3). Encode it in `NAME`.
- **`bench/one.sh`'s prompt-processing column is an artefact** (`cache_prompt:true`). Ignore PP
  from that harness entirely; TG and acceptance are the metrics.
- **One load takes 5-6 minutes from emmc** at ~88 MB/s. Budget it per arm, and prefer one load
  that measures several things (that is why `DECODE_REPS` exists).
- **`nsys` and `ncu` need root here**; unprivileged CUPTI returns
  `CUPTI_ERROR_INSUFFICIENT_PRIVILEGES`. Report name on this build is `gpukernsum`, not
  `cuda_gpu_kern_sum`. This nsys stores kernel names as `StringIds` ids, not text -
  `bench/nsys-kernels.py` already joins it.
- **`ncu` cannot run with the 19 GB model resident** (`NvMapMemAllocInternalTagged: error 12`).
  Profile single kernels in `bench/opbench` at the model's geometry instead.
- Build flags, the Volta `-ccbin` requirement, and the zsh glob trap are in `AGENTS.md`
  section 4. Do not set `CMAKE_CUDA_FLAGS` on the cmake command line.

---

## 8. Environment and build

- Build that counts: `momonga-llama.cpp/build-volta` (currently patched). `build-volta-dde879b`
  is the previous verified binary; `build-mmq` and `build-volta-fa` predate the source rename and
  fail until re-configured.
- Four experiment hooks are committed and all default to stock behaviour, so no launch line needs
  to unset them: `GGML_CUDA_MMID_MAX_BATCH` and `GGML_CUDA_SMALLK` (`mmvq.cu`),
  `GGML_CUDA_MOE_RPB` (MoE rows per block, default 2) and `GGML_CUDA_OP_TIME` (per-op dump,
  disables graphs).
  `git diff upstream/master -- ggml/src/ggml-cuda/` prints the complete local delta.
- CUDA graph capture is enabled by the **absence** of `GGML_CUDA_DISABLE_GRAPHS`; setting it to
  `0` does not enable anything. Every harness here uses `env -u`.
- Build with `-ccbin /usr/bin/g++-10.3.0` for nvcc, and pass the C++ include paths through
  `-Xcompiler=-isystem,...`. Authoritative flags live in
  `build-volta/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/flags.make`.

---

## 9. Git layout and standing constraints

| remote | url | role |
|---|---|---|
| `origin` | `https://github.com/ftulabs/momonga-llama.cpp` | the fork, default push target |
| `upstream` | `https://github.com/ggml-org/llama.cpp` | read-only reference, never push |

- `master` carries three local commits on top of upstream: the `MUL_MAT_ID`/sm_72 dispatch fix,
  the opt-in `GGML_CUDA_OP_TIME` dump, and the fork's own docs. Sync:
  `git fetch upstream && git rebase upstream/master && git push origin master`.
- **Never push to `upstream` and never open a PR against `ggml-org`** - the patches are
  Jetson-specific and some of it is debug scaffolding. Commit messages state what was measured and
  say the change is local to the fork. ASCII only in code and docs.
- `README.md`, `AGENTS.md` and `JETSON-XAVIER.md` inside the fork describe the fork and are
  fork-owned: keep them on rebase with `git checkout --ours <file>`.
- Commit after a change is verified to work. If the source moves, rebuild and re-measure rather
  than carrying old numbers over.
- Passwordless sudo exists and was authorised for telemetry and package installation. It was not
  authorised for anything broader; do not disable GPG verification to work around a broken repo.
- **Do not delete models without explicit confirmation** for the specific files. Delete a model
  only when a report exists showing it is useless (repo rule).
- **Ask before irreversible actions** (deleting, pushing, changing security settings); do not ask
  before ordinary measurements.

---

## 10. Disk and models

`/mnt/ssd/models` (symlinked as `/home/triph/models`) has about 101 GB free. On disk:

| file | size | keep? |
|---|---|---|
| `Cyber-Tiel-35B-A3B-Q4KM-tg.gguf` | 20.46 GB | **yes**, current active production model (Q4KM + Layer 3 TokenGate) |
| `Cyber-Tiel-Coder-35B-A3B-tg.gguf` | 18.12 GB | **yes**, IQ4_XS + TokenGate reference |
| `qwen36-35b-a3b-tg-g3-IQ4_XS.gguf` | 18.21 GB | **yes**, Qwen 3.6 reference and TokenGate donor |
| `Q4KM-all-q4k.gguf` | 20.46 GB | base un-grafted Q4KM (loaded in Unsloth Studio PID 140714) |
| `Ornith-1.5-35B-A3B-DFlash-BF16.gguf` | 0.78 GB | draft sidecar, tested and lost to MTP |
| `Spark-X2.6-4B-Q4_0.gguf`, `-Q8_0.gguf` | 6.8 GB | unrelated small model |

The old vanilla models (`Cyber-Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS.gguf` [18.12 GB] and `Cyber-Tiel-Coder-35B-A3B-MTP-UD-Q4_K_M.gguf` [22.52 GB]) were deleted on 2026-11-10, freeing 40.64 GB.

The three failed requant builds (61 GB) and the two comparison source trees (3.9 GB,
`beellama.cpp/`, `jusko-llama-volta-qwen3flash/`) were deleted with explicit confirmation on
2026-11-02 and 2026-11-03. `bench/sweep-fork.sh` no longer runs. Do not rebuild either tree - what
they offered is recorded in `findings/cyber-tiel-qwen35moe/2026-10-14-builds-quants-and-models.md` and
`findings/cyber-tiel-qwen35moe/2026-11-02-forks-and-kernels.md`.

---

## 11. The second rig: amd64 host + Tesla T4 (sm_75, 16 GB)

**What it cannot do:** hold the model (18.12 GB of weights before KV), so it cannot run
`bench/one.sh`, A/B a library, or validate any Xavier number. Different SM count, memory and
tensor-core generation, so its numbers must never enter `KNOWN-GOOD.md`. For prefill it is no
better: `gated-delta-net-mma.cu` gates on `ampere_mma_available` and Turing is sm_75, so a T4
takes the same scalar GDN path the Xavier does - **H4 needs an Ampere or newer card for the
reference, and the T4 can only host Volta-schedule development.**

**What it is good for:**

1. **Training the MTP head** (section 6), the one non-marginal lever left on the model side. The
   16 GB is not the blocker; generating the corpus on the Xavier is.
2. **`opbench` on the arch the shared MMVQ table was written for.** `get_device_table_id` puts
   sm_70/72/75 all on `MMVQ_PARAMETERS_TURING`, so a T4 answers whether that table is optimal for
   Turing - which bounds whether sm_72 needs its own table without touching the Xavier.
3. **`ncu` without the fight.** The Xavier cannot profile with the model resident; a T4 has
   dedicated VRAM.
4. **Fast build and quantize host** if a future experiment does need to rewrite a GGUF.

---

## 13. TokenGate cross-model transfer (2026-11-07)

- **Model:** `/mnt/ssd/models/Cyber-Tiel-Coder-35B-A3B-tg.gguf` (17.3 GB, 774 tensors).
- **Source:** Force-installed G3 TokenGate from Qwen3.6-35B-A3B (`/mnt/sdcard/home/minh/src/g3_gate.npz`, step 3002) using `momonga-llama.cpp/tools/tokengate/convert_tokengate_to_gguf.py`.
- **Paired LoRA adapter:** `/mnt/sdcard/models/tokengate/qwen36-35b-tg-g3-lora.gguf` (8.19 MB).
- **Fork fix (commit `91b037d` on branch `tokengate`):** Fixed embedding buffer sizing and extraction stride in `src/llama-context.cpp` by calling `n_embd_out_ctx()` instead of `hparams.n_embd_out()`. In `LLAMA_CONTEXT_TYPE_TG_BOTTOM`, tokens output `n_embd + 1` (2049) floats. Stride 2048 caused an out-of-bounds segfault in `llama-tokengate` at position 16082.
- **Performance on NIAH (`niah_xav.jsonl`):**
  - **16k context (16,043 tokens):** TTFT drops from **82.6 s** (baseline full prefill) to **19.6 s** at $K=2000:2$ (**4.22x speedup**), **10.6 s** at $K=500:2$ (**7.77x speedup**), and **8.5 s** at $K=100:2$ (**9.70x speedup**).
  - **32k context (32,108 tokens):** TTFT drops from **199.5 s** (~3.3 min baseline) to **30.9 s** at $K=2000:2$ (**6.46x speedup**), **21.6 s** at $K=500:2$ (**9.23x speedup**), and **19.2 s** at $K=100:2$ (**10.36x speedup**).
  - **Accuracy:** 100% exact needle retrieval match across all evaluated rows.

---

## 12. Map: which file answers which question

| question | file |
|---|---|
| What is the shipped configuration, and why these flags? | `KNOWN-GOOD.md` |
| How do I measure without fooling myself? | `AGENTS.md` section 3; `TODO.md` "How to Measure" |
| What is the open work, ranked, with predictions? | `TODO.md` "Prefill / TTFT Marathon" |
| What has already been tried and must not be retried? | `TODO.md` "Closed - Do Not Re-test" |
| Where does the time actually go in a decode pass? | `findings/cyber-tiel-qwen35moe/2026-11-03-decode-pass-budget.md`, `findings/cyber-tiel-qwen35moe/2026-11-03-moe-share-in-situ.md` |
| Baseline, code audit and H1-H5 for prefill | `findings/cyber-tiel-qwen35moe/2026-11-04-prefill-starting-point.md` |
| Where the time actually goes in a prefill (H2, the per-kernel budget) | `findings/cyber-tiel-qwen35moe/2026-11-04-prefill-budget.md` |
| Why weights cannot be requantized | `findings/cyber-tiel-qwen35moe/2026-11-02-requant-anchored-ab.md`, `findings/cyber-tiel-qwen35moe/2026-11-03-lmhead-q5k.md` |
| Environment traps, counters, profilers | `AGENTS.md` sections 4-5, `findings/cyber-tiel-qwen35moe/2026-11-03-counters-on-xavier.md` |
| Everything, indexed | `FINDINGS.md`, `findings/README.md` |
