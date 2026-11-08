# Can the MTP head be retrained, and does a fresh BF16 requant behave differently?

Started 2026-11-05 21:15, predictions recorded before any run.

## 1. The two questions

After section 28 of `2026-11-05-mtp-free-campaign.md` (MTP on the retyped `all-q4k` file: acceptance
0.96657 -> 0.64583, mean len 5.68 -> 2.10, decode 27.86 -> 23.351 tok/s, -16.2%) the fair
follow-up questions are:

1. Can the MTP head be retrained or fine-tuned so that it agrees with the retyped target?
2. Would requantizing from the original BF16 weights, instead of chaining from the shipped
   `Q4_K_M` file, behave better?

These are separable: the first is about who the drafter is trained to agree with, the second is
about how much the target moved in the first place. Everything below is either already measured,
or a prediction that is written down before the corresponding run.

## 2. What is already measured, and what it says about the mechanism

The asymmetry is now documented on both sides of the drafter/target pair.

**Drafter-side weight noise is nearly free.** The upstream model card for this exact file family
(`peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-GGUF-MTP`) states: "we swept head precision from Q8_0
down to Q2_K and found draft acceptance flat (~82%); the head therefore ships small (Q3_K,
quantized straight from the trained BF16 head)". Our own copy confirms the type: every
`blk.40.*_exps` tensor in `Q4KM-all-q4k.gguf` is **Q3_K** (type 11), and the whole nextn block is
844.6 M parameters, so the head ships as a 0.4 GB block in every tier.

So a head quantized from BF16 all the way down to Q3_K - and in the card's sweep, to Q2_K -
still drafts at full acceptance. Meanwhile every change we have made **on the target side**
collapses acceptance into the 0.63-0.72 band: `output.weight` q6_k -> q5_k alone (0.63603), the
f32 router summation-order change with no weight change at all (0.67717), the expert requants
(0.64-0.93), and the four-step `all-q4k` file (0.64583). The untouched control reproduces 0.96657
on every run ever logged.

That is a strong statement about the mechanism, and it points away from "quantization error is
bad" and towards "the target moved, and the head was trained against the target that no longer
exists". Corroborating evidence from the other direction: the DFlash sidecar, which is a
separately trained drafter and takes tokens rather than the target's hidden state, kept
acceptance at 0.889 -> 0.858 across the same retypes - a 3.5% relative loss where the MTP head
lost 33%.

## 3. The BF16 question, answered analytically before it is measured

`all-q4k` is not a single-pass quant of BF16. Its ancestry is:

| tensor family | BF16 -> shipped file | -> `all-q4k` | compounding steps |
|---|---|---|---|
| dense attention/GDN | BF16 -> q8_0 | q8_0 -> q5_k -> q4_k | 2 |
| experts | BF16 -> q5_k | q5_k -> q4_k | 1 |
| shared expert | BF16 -> q8_0 | q8_0 -> q5_k | 1 |
| `output.weight` | BF16 -> q6_k | q6_k -> q5_k -> q4_k | 2 |

A fresh BF16 -> `all-q4k`-recipe file removes those steps, so at the **same speed** (same types =
same bytes = same kernels) it is strictly better quality. The size of the gain is bounded by the
double-quantization share of the measured loss, and that share looks small: the intermediate
types were q8_0 (near-lossless), q6_k and q5_k, and the four steps together cost +1.67%
perplexity against the parent (1.9142 -> 1.9461), of which the one step that requantized
q5_k -> q4_k and q8_0 -> q5_k (the bundle) was only +0.29%. The bulk of the 1.67% is the type
*choices* (all experts at q4_k, `output.weight` at q4_k), not the chaining.

- **P27: a BF16-direct single-pass file will measure +0.3 to +0.9% perplexity better than
  `all-q4k`, i.e. 1.9461 -> 1.929-1.940, and its decode will be identical within noise because
  the types and byte counts are the same.** It will not move MTP acceptance by itself.
- **P28: quality is not what governs acceptance.** If it were, a 1.67%-perplexity move could not
  cost 33 points of acceptance while a Q8_0 -> Q2_K move in the head costs ~0. The dose-response
  is not linear in error and is not even monotone in bit width; it is a cliff on the target side.

## 4. The probe that separates "double quantization" from "any target drift"

Both predictions above are arguments, not measurements. There is one cheap experiment that turns
them into a measurement without downloading 69 GB of BF16: **run MTP against a file that was
quantized from BF16 in a single pass and never touched since.** `UD-Q4_K_XL` (22.7 GB, the card's
"start here" tier) is exactly that, and its MTP head is the same Q3_K block, so the target is the
only variable.

- If acceptance comes back near 0.96657, the double quantization is the culprit and the fix is
  mechanical: download the 69 GB BF16 base plus the author's 192 MB imatrix, quantize once to the
  `all-q4k` recipe, and expect the retype's +25.2% decode with the drafter on - roughly
  **27.86 x 1.66 = 46 tok/s**, which is the number the flag sweeps of 2026-11-04 could not reach.
- If acceptance lands back in 0.63-0.72, then no requant can preserve the drafter, and the only
  route is a retrained head (section 5).

- **P29 (recorded 21:20, before the download finished): `UD-Q4_K_XL` + MTP reads acceptance
  0.62-0.88, central estimate ~0.75, with P(>= 0.90) about 25%.** Reasoning: the router-ordering
  experiment collapsed acceptance with a change far smaller than a quant-tier difference, which
  says the target side is chaotic; but the control pair (head and target quantized together from
  the same BF16) survives at 0.96657, which says a *matched* derivation is tolerated. XL is a
  matched derivation of the same BF16, so it is a genuine coin-flip and the run is worth its
  half hour. Decode on XL alone: **20.5-22.5 tok/s**, since it is 22.7 GB against `all-q4k`'s
  20.46 GB.

## 5. What a retrained head actually costs

The fine-tune does **not** need the BF16 weights, because the target to agree with is the
deployed file. The standard self-distillation recipe (EAGLE-style) is:

1. **Extract** (target hidden state `h`, next-token target) pairs from `Q4KM-all-q4k.gguf` itself,
   via `llama_get_embeddings` on the deployed build. Prefill runs at 215 tok/s, so 10 M tokens is
   ~13 h of extraction; 1-2 M is enough for a first attempt (~1.5 h).
2. **Train** the 844.6 M-parameter nextn block in PyTorch against those pairs. The target is
   frozen and not needed in the trainer at all. A T4 (16 GB, sm_75) holds 0.85 B params in fp16
   plus optimizer state; at ~6 N flops/token the compute is ~1 h per 5 M tokens, or minutes per
   epoch at 1 M.
3. **Export** the trained tensors back into a GGUF. The tooling exists: `bench/requant-one-tensor.cpp`
   writes a tensor in place at a known offset, and `bench/build-lmhead-q5k.py` shows how to
   rebuild the header plus offsets for a tensor whose size changes.
4. **Gate**: acceptance per rep, 5 reps, target >= 0.90 with mean len >= 5.0. Everything else is
   secondary.

The known risks: `blk.40.ffn_gate_inp` is the top-8 router of the head and the head's own expert
weights are already Q3_K in the file we would start from, so the initialization is lossy unless
the BF16 base (Huihui-Ornith-1.5-35B-A3B-abliterated, 69 GB, per the model card) is downloaded
and its nextn block used as the starting point - which matches step 3 of section 4 and is the
reason to do both together if the probe says the drafter can be saved at all.

## 6. Results

### 6a. P29: a single-pass BF16 quant does NOT preserve acceptance (2026-11-05, 23:20)

`UD-Q4_K_XL` is the author's "start here" tier, 22.749 GB, quantized once from the 69 GB BF16
base with their imatrix, never retyped. Artefact check first: 753 tensors, tensor-name set
identical to `all-q4k`, all consecutive offsets consistent with the declared types, loads with no
missing or unknown tensors. (One oddity in the published file: 11,013,952 bytes of undeclared
data follow the last tensor, `blk.40.post_attention_norm.weight`, which ends at
22,738,866,688. It is real f32-looking data, it is present in the file on the HF hub as served
- `X-Linked-Size` matches our local size exactly - and it is harmless because llama.cpp reads
only declared offsets. Worth knowing, not worth acting on.)

Its tensor types are *higher* precision than `Q4_K_M` on the tensors that matter for this test:
dense attention/GDN **q8_0** (134 tensors), `output.weight` **q8_0** (against q6_k), 78 experts
at q4_k, 38 at q5_k, 4 at q6_k, and the same Q3_K + Q8_0 + f32 head block in `blk.40`.

With the same harness, the same flags (`--spec-type draft-mtp -n-max 6 -p-min 0.5`, `-ub 32`,
`-bs`) and 5 reps x 400 tokens on `prompt2.txt`:

| file | TG (5 reps) | acceptance per rep | mean len |
|---|---:|---|---:|
| `UD-Q4_K_M` (control, never retyped) | 36.93 | 0.96657 on every run ever logged | 5.68 |
| `Q4KM-all-q4k` (four chained retypes) | 23.351 | 0.64583 | 2.10 |
| `UD-Q4_K_XL` (single pass from BF16) | **19.53** (reps 2-5: 19.53 / 19.52 / 18.86 / 19.45; rep 1 18.79 cold) | **0.58113 / 0.58491 / 0.53358 / 0.58491** | **2.09** |

**P29 was a miss, and in the useful direction:** I predicted 0.62-0.88 with a central estimate of
0.75 and P(>= 0.90) about 25%; the measurement is 0.53-0.58, below the whole band. The
"fragile" hypothesis won, and it won harder than I gave it credit for.

The decisive part is not that XL also loses - it is that **XL loses more than the double-quantized
file while being the higher-fidelity quant of the same BF16**. `output.weight` is q8_0 instead of
q6_k, the whole dense set is q8_0, and the file is 2.29 GB larger; none of that buys a single
point of acceptance. Acceptance is not a function of quantization error, or of bit width, or of
how carefully the file was derived from BF16. It is a function of whether the target's numbers are
the ones the head was trained against, and Q4_K_M happens to be that file.

### 6b. What this closes

- **Requantizing from the original BF16 will not rescue the drafter** (P28 confirmed). The gap
  between 0.96657 and 0.64583 is not "our chain was sloppy"; a fresh single-pass file with more
  bits in every tensor that matters reads 0.585 on the same harness. A 69 GB BF16 download plus
  an imatrix plus a single-pass quant would give a better-looking file and the same dead drafter.
- **The quality question in section 3 stands on its own and is unaffected**: a BF16-direct pass to
  the `all-q4k` recipe is still strictly better at identical speed, and P27's +0.3 to +0.9%
  perplexity band is still the estimate. It is simply a quality lever, not an acceptance lever.
- **The only route left for MTP + a retyped target is a retrained head** (section 5). The
  XL arm also prices that work correctly: the head is not "almost right" on any other file, so
  the target it is trained against has to be the deployed file itself, and the gate is
  acceptance >= 0.90 with mean len >= 5.0, not perplexity.

### 6c. Artefact and quality check on XL

Both runs are clean, and they sharpen 6a rather than qualify it.

- **PPL = 1.9142 on the 60-chunk code corpus** (`bench/ppl.sh xl`), which is *identical* to the
  parent `Q4_K_M` (1.9142) and 1.67% better than our `all-q4k` (1.9461). So XL is a genuinely
  better-quality file than ours, with more bits in every tensor that matters - and still cannot
  draft.
- **Plain decode 22.014 tok/s** (5 reps: 21.98 / 22.04 / 21.98 / 21.95 / 22.01), against
  `Q4_K_M`'s 22.25 and `all-q4k`'s 27.86. The extra 0.25 GB costs 1.1% decode, as the type-vs-bits
  rule predicts.

That completes the comparison table:

| file | plain | + MTP n-max 6 | MTP effect | acceptance | mean len |
|---|---:|---:|---:|---:|---:|
| `UD-Q4_K_M` (shipped pairing) | 22.25 | **36.93** | **+66%** | 0.96657 | 5.68 |
| `Q4KM-all-q4k` (four retypes) | 27.86 | 23.351 | -16.2% | 0.64583 | 2.10 |
| `UD-Q4_K_XL` (single pass from BF16) | 22.014 | 19.53 | -11.3% | 0.534-0.585 | 2.09 |

**The drafter pays on exactly one file: the one it shipped with.** On both re-derived files it is
a net cost, and the higher-quality one (XL: same ppl as Q4_K_M, more bits) is *worse* for the
drafter than the double-quantized one. This is about as clean a refutation of "make the target
better and the drafter will cope" as the bench can produce.

## 7. TokenGate: what it is, and what it would be worth here (analysis only, no run)

Investigated 2026-11-05 on request. `arxiv.org/abs/2610.26754` (Patel, Ranjan, Stoica, Zaharia,
Guestrin - Stanford/Berkeley), code Apache-2.0 at `github.com/token-gate-llms/TokenGate`, trained
gates published only for Qwen3-1.7B and Qwen3.5-2B. It is not an off-the-shelf model; it is a
fine-tuning recipe plus a new attention mechanism.

**The mechanism, as published.** A small MLP gate sits inside an early layer L (L=1 for Qwen3-1.7B)
and scores every token from the post-attention residual. Only the top-K continue through the
layers above; pruned tokens create no KV entries there and are never forwarded again. Above the
gate, attention gets a per-key logit bias `softplus(gamma) * b_j` with a learnable head-wise gain,
which is what makes top-K differentiable end-to-end (the MoE-router trick). K is sampled
log-uniform in [1, N/2] per step during training, so one model covers a family of budgets. The
backbone is **not** frozen: they fine-tune with LoRA r=16 on q/v of all layers, jointly with the
gate, standard next-token CE, ~1.3x the SFT step time, 1.2-4 h per dataset on 4x A100 for a 1.7B
model. Decode stays compatible: layers below the gate see the full context, layers above see the
kept tokens plus generated ones.

**Why the headline 11.3x does not transfer to this board.** Their speedup is measured on one A100,
batch 1, bf16, on Qwen3-1.7B with NIAH-16k, and it comes from two things this model does not have:

1. **Quadratic attention compute.** Measured here (`findings/2026-11-04-prefill-budget.md`), at the
   production `-ub 32` FlashAttention is **0.9% of prefill** and the GDN core is 3.3%.
2. **KV-cache memory.** We run `-ctk q8_0 -ctv q8_0` over 11 attention layers at 40,960 context;
   it has never been the binding constraint on this box.

What our prefill actually is, is **88.1% MMQ** (MoE gate/up 52.9% + down 23.0% + q8_0 12.4%), and
that is per-token work: every retained token still routes through 8 of 256 experts. So the part of
TokenGate that would transfer is the boring part - **fewer tokens in 40 of 41 layers** - and its
ceiling is set by what fraction of prefill remains if K/N -> 0. From the same budget, that is
~93%, i.e. the theoretical ceiling here is ~14x, but the realistic operating point is K/N at which
code quality survives, which the paper puts at 3.0-3.8x for their models. On this file a 3x would
take TTFT at 11,484 tokens from **53.3 s to ~18 s**, and 4x to ~13 s.

**The compatibility risk is larger here than in the paper.** They demonstrate one hybrid
(Qwen3.5-2B) where the Gated DeltaNet layers "run unchanged on kept tokens". Our model is 30 GDN
layers to 11 attention layers - the majority of the stack is a *recurrent* scan, and dropping
tokens from a scan does not skip work, it changes the state trajectory: every pruned token's
contribution to the f32 recurrent state disappears. The paper's ratio is the opposite way round.
Nobody has published a gate on a Muon/GDN-heavy 256-expert MoE, and the release does not mention
MoE at all.

**Cost, in order of difficulty.**

1. **Fine-tune a 35B with LoRA on q/v of all layers at long context.** A T4 with 16 GB cannot hold
   this - a 3-bit/4-bit 35B is ~18-22 GB before activations, and their own recipe uses full
   contexts. This needs rented A100/H100 time, tens of GPU-hours minimum.
2. **Implement sparse prefill in llama.cpp.** A gate MLP and per-head gains are new GGUF tensors;
   the graph needs a row-gather at layer L (llama.cpp's graph builder has no sparse-token
   concept); the upper-layer KV cache must hold only the kept positions with an index map, and the
   decode path must attend to that subset. The biased-attention kernel (`softplus(gamma)*b_j`) is
   a new CUDA kernel on a card where we write every kernel by hand. This is weeks of fork work,
   not a patch, and it would have to survive the same numerics discipline as everything else here.
3. **Re-validate quality on code**, not NIAH: their own MultiNews example fails on names at K=256,
   and RepoBench is the only code-facing result. We would gate on `bench/ppl.sh` plus a real
   task, at a K/N that costs us the prefill win if it has to be generous.

**Verdict.** It is the only remaining lever that could move prefill by more than the 2-3% left in
kernel geometry, and it would be additive to the 2.4x already banked from `-ub 2048`. It is also a
multi-GPU fine-tune plus a llama.cpp sparse-prefill implementation with no prior art for a
35B recurrent-hybrid MoE, and the published 10x number is an attention-bound A100 figure that this
profile cannot reproduce - the honest expectation here is **2.5-4x prefill, TTFT 53 s -> 13-20 s**,
at high cost. If we ever take it on, the cheap first step is not the training: it is to answer
"what fraction of *our* prefill is per-token work at `-ub 2048`" from an nsys trace, because that
number is the ceiling and it is measurable in an hour.

The alternative that needs no training and no fork work is prompt-level compression: drop or
summarise low-value spans *before* the server sees them. It captures the same "fewer tokens"
benefit on this profile, is testable with the existing `bench/prefill.sh` plus `bench/ppl.sh`, and
costs a day instead of a month - at the price of being task-specific rather than learned.

## 8. The extractor, and the hypothesis it refuted (2026-11-05/06)

Section 5 priced a retrained head. This section builds the first half of it - turning a deployed
GGUF into `(h, token_in, argmax)` training tuples - and reports what the validation battery
measured, which is a refutation of the mechanism the plan was founded on.

### 8a. What `bench/mtpdump.cpp` does

The head graph is fixed by `src/models/qwen35moe.cpp` (`mtp_hnorm` / `mtp_enorm` / `mtp_concat` /
`mtp_eh_proj`): `h_norm = rmsnorm(h_trunk_before_final_norm, hnorm)`,
`e_norm = rmsnorm(emb(tok_in), enorm)`, concat -> `eh_proj` -> one attention+MoE block -> the
shared `output.weight`. So `(h, token_in)` is the complete input and the target's argmax is the
label. The extractor drives the fork's staging API (`llama_set_embeddings_nextn(ctx, true,
masked=false)` for one dense row per batch position, `llama_get_embeddings_nextn_ith`,
`llama_get_logits_ith`) and writes:

```
char magic[4]="MTP1"; u32 n_embd; u32 n_vocab; u32 n_tokens;
u32 token_in; u32 token_tgt; f32 h[n_embd]   (repeated)
```

8,200 bytes per position, so 1M positions is 8.2 GB. Build with `bench/build-mtpdump.sh` (it needs
the g++-10.3.0 C++ include paths, the same trap AGENTS.md records for nvcc). Memory is cleared per
corpus file, so each shard is an independent document and a run can be split anywhere.

### 8b. Validation battery

All runs on `prompt-32k.txt`, 2,000 positions, `--batch 8` unless stated. The A/A arm is the same
file extracted twice.

| arm | argmax agreement | cos mean | cos p5 | rel RMS of h |
|---|---|---|---|---|
| same file, two runs | 100.00% | 1.0000 | 1.0000 | 0.00% |
| `Q4_K_XL` vs `Q4_K_M` | 98.10% | 0.9933 | 0.9843 | 11.67% |
| batch 64 vs batch 8 (all-q4k) | 97.55% | 0.9921 | 0.9790 | 12.63% |
| `all-q4k` vs `Q4_K_M` | 94.80% | 0.9643 | 0.9297 | 26.81% |

Extraction is bit-deterministic (identical file, identical bytes), h is well scaled (per-element
RMS 2.40, per-row L2 norm 108.5, range -37.4 to 32.2), and the target's argmax equals the next
corpus token on 83.8% of positions - plausible for 32k of code. The batch-width row is a direct
confirmation of the AGENTS.md claim that `-ub` changes the prompt's hidden state: width 64 against
width 8 moves h by 12.6% relative RMS with no other change.

### 8c. The hypothesis, and why it is dead

The plan in section 5 rested on a story: a requant moves the trunk, the head reads an
out-of-distribution `h`, and acceptance collapses. That story predicts that **how far `h` moves
predicts how much acceptance falls.** It does not, and the XL arm is decisive:

| target | acceptance | mean len | decode tok/s | cos to Q4_K_M | argmax agreement |
|---|---|---|---|---|---|
| `Q4_K_M` (shipped) | 0.96657 | 5.68 | 36.93 | - | - |
| `Q4_K_XL` | 0.585 | 2.09 | 19.53 | **0.9933** | **98.10%** |
| `all-q4k` | 0.64583 | 2.10 | 23.35 | 0.9643 | 94.80% |

XL is five times closer to the reference than all-q4k in `h` direction error (0.46% median against
3.11%), four times closer at the 5th percentile (0.9843 against 0.9297), and drifts the argmax less
than half as often - and its acceptance is **worse** by 6 points. The prediction written down before
the XL arm ran was 0.62-0.88 central 0.75; it measured 0.585, below the whole band, and the reason
is not the one I gave.

Both quantities I thought were the mechanism fail to rank three files. So the "the head reads a
perturbed trunk state" account is retracted. XL's `h` is nearly as close to Q4_K_M's as a *batch
width change* is (11.67% against 12.63% relative RMS) while its acceptance is 0.585 - distance
below the noise of the extraction protocol itself buys nothing.

### 8d. What is left, and it is testable

The head does not consume `h` alone. It consumes three tensors, and the three files differ in all
three:

| tensor | `Q4_K_M` | `Q4_K_XL` | `all-q4k` |
|---|---|---|---|
| `blk.40.*` (the head, 20 tensors) | shipped | **byte-identical** | retyped |
| `token_embd.weight` | q8_0 | q8_0, different values | q8_0, different values |
| `output.weight` (shared logit projection) | **q6_k** | q8_0 | q4_k |

The head itself is byte-identical between Q4_K_M and XL (all 20 tensors hashed), so the XL pair is a
clean control: same drafter, different surroundings, closer `h`, worse acceptance. The one input
whose *type family* changes in both requants is `output.weight`, and that lever has already been
measured alone on the shipped file: `output.weight` q6_k -> q5_k, with 753 tensors verified and
eight hashed byte-identical, took acceptance 0.96657 -> 0.63603. Both requants therefore move a
tensor that is independently known to cost ~0.33 of acceptance, and neither can be said to have
failed because of its trunk.

**Prediction for the decisive test (written before the run).** Build a file whose trunk is
`all-q4k`'s retyped trunk but whose `output.weight` is a verbatim byte copy of `Q4_K_M`'s q6_k
tensor, and measure MTP acceptance on it. If the logit projection is the driver, acceptance
recovers to **0.85-0.95**; if the trunk `h` is the driver it stays at **0.64 +- 0.03**. Central
prediction **0.88**. `bench/build-lmhead-q5k.py` already does exactly the header/offset rewrite this
needs (one tensor replaced, every later offset moved, verbatim tail), so it is a minutes-long file
build, not a requant. It is queued behind the extraction run because both need the GPU.

This also decides whether retraining can work at all. It can - the head's last stage *is*
`output.weight`, so a head trained against the deployed file's own projection learns to match it -
but it means the extraction must come from the file that will serve, which is `all-q4k`, not
`Q4_K_M`.

### 8e. Production extraction

Corpus is `bench/make-mtp-corpus.py`: **3,503 shards, 8.99M tokens** assembled from the Python
3.12 stdlib, both llama.cpp trees, a 533 MB JS/TS app and this repo's own prose, deduplicated by
content hash, seeded-shuffle ordered, ~11 kB of text per shard, every 20th shard held out for
validation. Sources are what this box has; there is no network dataset in the loop. Shard sizes
are **measured with the real tokenizer, not estimated from characters**: `--count-only` reports a
range of 434 to 10,780 tokens for shards of nearly equal character length, so a character budget
is not a token bound and the run's `--ctx` (16,384) is set from the count file.

Width is `--batch 8`: the head consumes `h` from the verify pass, whose width is the number of
accepted tokens plus one, and the measured mean length is 5.68 of 6, so 7-8 is the operating width.
8b shows the width matters (12.6% at 64), so a mixture is the robustness option if the trained head
turns out narrow-sensitive - at a further full pass per width.

Cost, measured on the real corpus: **54.9 tok/s of decode, 8.2 GB per 1M positions**, and a model
load of **26.6 s per invocation** against a median shard of 48.2 s of decode. Loading once per
shard would waste 35% of the wall clock, so `bench/run-mtpdump.sh` passes 24 shards per call (2.3%
overhead) and `mtpdump --out-dir` writes one file per shard, `.tmp` then renamed, so a kill loses
at most the shard in flight. 1M positions is 5.1 h and 8.2 GB; the full corpus is 8.99M positions,
45 h and 74 GB, which does not fit alongside the models, so the run is budgeted in positions and is
stoppable at any shard boundary.

**Sequencing, decided after 8d and before either run.** The decisive `output.weight` test in 8d
costs a 2-minute file build plus one 40-minute A/B and can make the whole extraction unnecessary,
so it goes first and the extraction runs afterwards if it fails. The prediction is already
recorded above, so the order cannot bias the reading.

### 8f. The artefact for the decisive test, built and verified

`bench/replace-one-tensor.py` takes the source file that will be served, the donor file, and one
tensor name, and writes a fresh file whose header carries the donor's type and whose data section
is the source's bytes with that one tensor's byte range swapped for the donor's. Offsets after the
replaced tensor move by the size delta, and the file is never patched in place. For this test:

| | |
|---|---|
| source | `Q4KM-all-q4k.gguf`, 20.461 GB |
| donor | `Q4_K_M`, tensor `output.weight` q6_k 417,177,600 B (shipped bytes) |
| built | `Q4KM-all-q4k-lmheadq6k.gguf`, 20.592 GB, delta +131,112,960 B in 2:13 |

Verified with `bench/verify-one-tensor.py` before any benchmark (rule 8): 753 tensors,
**752 of 752 untouched tensors hash identically to the source**, `output.weight` is type 14
(q6_k) with the donor's sha256 `3f83314d6f37c538`, offsets contiguous and the file size equal to
the header's padded data length. That last check earned its place immediately - the first build
wrote the source's *original* header while moving the data, leaving a file 131,112,960 B longer
than its own tensor table described. The size check caught it in 90 s instead of after a 40-minute
A/B, and for the record the failure mode is invisible to a hash comparison alone because every
tensor is read at the offset the header gives.

**The swap barely moves the target.** Run through `mtpdump` on one 1,705-token shard, the
target's argmax agrees with the corpus token on **55.78%** of positions with the q4_k head and
**56.19%** with the q6_k head - 0.41 points, on text where the all-q4k-vs-`Q4_K_M` label drift is
5.2 points. So if the A/B recovers acceptance, it is not because the answer distribution moved
back - the trunk is still all-q4k and the labels it produces are the ones all-q4k already produced -
it can only be the drafter's own projection, which is the only thing in the head's path that
changed.

`bench/mtp-lmhead-ab.sh` runs the two files A/B/B/A, 5 reps at `NPRED=400`, `-bs`, n-max 6,
p-min 0.5. Prediction on the record above: **0.85-0.95 acceptance (central 0.88) and 33-36 tok/s**
if the shared projection is the driver, **0.64 +- 0.03 and ~23 tok/s** if the trunk is. 8e's
extraction is what happens if the second answer wins.

## 9. The shared `output.weight` is not the driver: 0.673, not 0.88 (2026-11-06)

The A/B/B/A ran at 01:25-01:34 on 2026-11-06. Both arms are `-bs`, n-max 6, p-min 0.5, 5 reps at
`NPRED=400` on `prompt2.txt`, `-ub 32`.

| arm | TG per rep (tok/s) | median | acceptance | mean len |
|---|---|---|---|---|
| A1 `all-q4k` (`output.weight` q4_k) | 23.361 / 23.658 / 23.041 / 23.364 / 23.004 | **23.361** | 0.64583 | 2.10 |
| B1 `+q6_k output.weight` | 23.230 / 22.950 / 22.492 / 20.833 / 22.514 | **22.514** | 0.67286 | 2.39 |
| B2 `+q6_k output.weight` | 23.193 / 22.898 / 22.548 / 20.819 / 22.451 | **22.548** | 0.67286 | 2.39 |
| A2 `all-q4k` | 23.467 / 23.648 / 23.038 / 23.334 / 22.885 | **23.334** | 0.64583 | 2.10 |

**The prediction was wrong, and not marginally.** I wrote 0.85-0.95 acceptance (central 0.88) and
33-36 tok/s if the shared projection was the driver, 0.64 +- 0.03 if the trunk was. The answer is
**0.67286**: +2.7 points, a little outside the "trunk" band but nowhere near recovery, and 15
points below the shipped file. So restoring the shipped `output.weight` byte-for-byte into a
retyped file recovers almost nothing, and the fourth candidate mechanism is dead too.

**It is also slower, so the file is rejected outright.** Pooled median 22.531 against 23.347
(**-3.5%**), with the session's A/A floor at **0.12%** (23.361 against 23.334, arms in both
orders - the tightest floor this project has recorded). The arithmetic is mean_len over pass
latency: `2.39/22.531` against `2.10/23.347`, so the 14% longer draft chain bought only 4.2%
relative acceptance while the pass itself got **18% slower** - that is the q6_k lm_head's 417 MB
read 7 times per pass (2.92 GB) against q4_k's 286 MB (2.00 GB), on a pass that moves about 8.1 GB.
A 0.9 GB byte addition costing 18% of a pass is worse than its share, which is the same
"type beats bits" asymmetry the draft-path kernels show in the other direction.

Both B arms also reproduce the occasional-rep collapse this project has seen on requantized files:
rep 4 reads 20.83 / 20.82 while reps 1-3 and 5 sit at 22.45-23.23. The median absorbs it; a
best-of-2 or a mean would not.

### What the campaign has now established, in one place

The head consumes three tensors and the requant changes all three. Each candidate driver has now
been tested or measured:

| candidate | test | result |
|---|---|---|
| trunk `h` distance to `Q4_K_M` | extract and compare (`mtpdump`) | **does not rank acceptance**: XL 0.9933 cosine / 0.585 acceptance, all-q4k 0.9643 / 0.64583 |
| target argmax drift | same | **does not rank acceptance**: XL 1.9% drift / 0.585, all-q4k 5.2% / 0.64583 |
| shared `output.weight` type | swap shipped q6_k bytes in, A/B/B/A | **+2.7 points only** (0.64583 -> 0.67286) and -3.5% throughput |
| everything else in the file | by elimination | not separable; the head is a 20-tensor block that is byte-identical to the shipped one, so the residual is the trunk's numbers |

So acceptance is a property of the *joint* configuration that no single scalar or single shared
tensor explains, and it is not recoverable by file surgery. **A retrained head is the only route
that remains**, which is what section 5 priced and section 8e's extraction exists for.

One positive measured number to carry forward: swapping only `output.weight` q4_k -> q6_k moves the
target's own argmax labels by 0.41 points on a 1,705-token code shard (55.78% -> 56.19% agreement
with the corpus token), while the all-q4k trunk retype moves them by 5.2 points. Label drift is a
trunk property, not a head property - which is why the label side of this experiment could not
have explained a recovery even if there had been one.
