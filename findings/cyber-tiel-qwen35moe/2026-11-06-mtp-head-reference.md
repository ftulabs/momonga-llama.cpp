# The fork's own MTP head output is now a reference file (2026-11-06)

## 1. Why

The drafter cannot be repaired by requantizing anything: four independent weight perturbations,
a f32 router change, a published higher-fidelity quant and a verbatim q6_k `output.weight`
restoration all failed to bring acceptance back above 0.673 (AGENTS.md section 7). The remaining
route is to retrain `blk.40`, and the retrained head has to be judged against something better
than "does the model run". That something is the fork's own MTP graph: dump the head's logits for
positions whose `h` is already on disk, then require a PyTorch reimplementation to reproduce them
before a single gradient step.

`mtpdump` gained `--head-logits FILE` for this. It runs a second context over the same positions
the `.mtp` dump records, feeds it the same `h` rows, and writes a reference file.

## 2. Two bugs, both of which produced a confident wrong answer

**(a) The first "reference" was the trunk compared with itself.** A context created with default
`llama_context_params` runs the normal decoder. Its head logits were therefore the target's own
logits and agreed with the label **100.00%** on 256 positions. Nothing about that number is a
success: it is the signature of a broken measurement. The fix is `ctx_type =
LLAMA_CONTEXT_TYPE_MTP`, which `llama-context.cpp:115` maps to `LLM_GRAPH_TYPE_DECODER_MTP`.

**(b) `GGML_ASSERT(layer.nextn.eh_proj)` on the next run.** The MTP graph is built for
`model.layers[hparams.n_layer()]`, but the loader skips those tensors:
`llama-model.cpp:3981` sets `mtp |= TENSOR_SKIP` unless `ml.load_mtp`, which comes from
`llama_model_params.load_mtp` (`common/common.cpp:1745` derives it from `--spec-type draft-mtp`).
Without it `blk.40` is not loaded at all and the assert fires. Fixed by setting
`mparams.load_mtp` only when `--head-logits` is in use, so the extractor path keeps exactly the
memory footprint that produced the 1.5M banked positions.

Also matched to the two runtime creation sites (`common/speculative.cpp:2757`,
`common/common.cpp:1287`): `ctx_other = ctx_tgt` and `n_rs_seq = 0`. Neither is required for
`qwen35moe` (`llama-context.cpp:247-263` only *requires* `ctx_other` for Gemma4Assistant, EAGLE3
and DFlash), but the runtime sets both and the reference should be produced the way production
produces it.

**Provenance check after the rebuild**: re-running `bench/mtpdump` on `shard-000000.txt` with the
new binary reproduces `dumps/shard-000000.mtp` **byte for byte**. `load_mtp` is inert for the
trunk - it loads weights the trunk graph never reads - so the banked dumps and the reference come
from the same function.

## 3. What the reference says

256 positions of `corpus/train/shard-000000.txt`, `--batch 8 --ctx 16384`, `all-q4k` file:

- head argmax vs trunk argmax: **230/256 = 89.84%**
- by position band: pos 0-31 **56.2%**, 32-63 87.5%, 64-95 84.4%, 96-127 **100.0%**,
  128-159 93.8%, 160-191 100.0%, 192-223 96.9%, 224-255 100.0%
- logits span -14.27 .. 34.22, median top1-top2 gap 4.77, so `f16` logits in the reference file
  lose nothing that matters

The rising curve is the expected shape and is also the reason this is a *single-step,
teacher-forced* number: after ~100 tokens of C++ the continuation is nearly forced, so both models
agree trivially. It is **not** comparable to the deployed acceptance of 0.64583, which is a chain
statistic over 6 draft steps on prose. Read it as "the head is live and sane", not as a target.

Position 0 is special and useful: the head's `h` input there is `pending_h`, which the extractor
initialises to zeros, so position 0 exercises the front end (embedding, `enorm`/`hnorm`,
`eh_proj`) with no hidden state at all. That is why it mismatches the trunk (head 11 against trunk
220) and why it is the cheap bisection point if parity fails.

## 4. Graph facts re-derived, for the trainer

Read from the source again rather than trusted from notes, because every one of these would
silently train a different function:

- `rope`: `llama_model_rope_type` returns **IMROPE** for `QWEN35MOE` (`llama-model.cpp:3700-3707`,
  the `QWEN3VL/QWEN35/QWEN35MOE/CLEF/QWEN4EXP` group). IMROPE is not a different pairing:
  `ops.cpp:6424` calls `rotate_pairs(n_dims, n_dims/2, ...)`, the same half-split as NEOX, and
  `is_imrope` only changes which position an index takes its theta from
  (`ggml_mrope_cache_init`, `ops.cpp:6204`). With all four positions equal - text - every branch
  returns the same theta, so it reduces exactly to partial NeoX rope over `n_rot = 64` of 256
  dims with `base = 1e7`. The PyTorch head's plain partial rope is correct.
- `concat = [e_norm, h_norm]` (`qwen35moe.cpp:600-608`), `eh_proj` 4096 -> 2048.
- `attn_q` is `n_embd_head * 2` wide per head and the layout is per head: `[q(256), gate(256)]`
  interleaved, gate taken at offset `n_embd_head` (`qwen35moe.cpp:625-640`), multiplied into the
  attention output *before* `attn_output`.
- MoE: `build_moe_ffn` softmaxes over all 256 experts, takes top-8 by argsort, gathers the
  probabilities, renormalises with the sum clamped at `6.103515625e-5`, and `expert_weights_scale`
  is `0.0` in this file so no extra scale is applied (`llama-graph.cpp:2268-2380`).
- The shared expert is `up`/`gate` parallel with SILU, and its sigmoid gate is a dot against
  `ffn_gate_inp_shexp` of the *MoE input*, not of the MoE output (`qwen35moe.cpp:700-730`).
- The head file has no `nextn.embed_tokens` and no `nextn.shared_head_head`, so the graph uses
  `token_embd` and `output` - both shared with the trunk and therefore frozen in training
  (`qwen35moe.cpp:577`). Confirmed by listing the 23-tensor head GGUF.
- The draft chain (`common/speculative.cpp:1810-1930`, `chain_heads == false` because
  `n_mtp_layers == 1`) feeds step 1 `pending_h` and `id_last`, then each later step **its own**
  `h` and its own sampled token. So the trainer's teacher-forced objective covers step 1 exactly
  and steps 2..6 only through the assumption that the head's hidden looks like the trunk's `h`.
  That assumption is testable and cheap to add to the loss (section 6).

## 5. Prediction, written down before the parity run

`bench/mtp_head_torch.py --parity` on the T4 must read **cosine min > 0.999, argmax agreement
100%** over the 256 positions. The reference's logits are `f16`, so cosine cannot be expected to
exceed ~0.99999 no matter how exact the graph is; a min below 0.999 is a real disagreement.

If it fails, the per-band table localises it: a failure that appears at position 0 and is flat is
the front end (embedding order, `enorm`/`hnorm`, `eh_proj`, or the MoE); a failure that grows with
position is the head's own KV and attention (masking, the `h` shift, or the gate).

## 6. Bundle and next step

`mtp-bundle/` (1.34 GB) holds `Q4KM-all-q4k.head.gguf` (23 tensors, sha256-identical to the served
file), `head-ref-256pos.hlg` (127 MB), `parity-input-256pos.mtp` (the 256 records the reference was
computed from), `mtp_head_torch.py` and `MTP-DUMP-FORMAT.md`. The Xavier cannot reach `ml` over
TCP (`AGENTS.md` section 4), so the user's laptop relays it.

After parity passes, the trainer needs three things this reference makes checkable:
training windows whose attention prefix is real (the head's KV covers the whole sequence), the
frozen `token_embd`/`output`, and - the one addition to the standard DeepSeek-style objective - a
consistency term pulling the head's own hidden at position `p` toward the trunk's `h` at `p`,
because that is what the chain feeds itself at steps 2..6.

## 7. Extraction state

Restarted 11:43 after being stopped for the reference run: 1,516,190 positions banked of
3,000,000, resuming at `shard-000590`. Supervisor `bench/supervise-mtpdump.sh` restarts on a
transient NvMap failure within ~1 minute (the 2026-11-06 08:50 abort cost 2h29m before it existed).
