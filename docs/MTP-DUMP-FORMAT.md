# MTP dump format and what the training box needs

## The dump (.mtp)

Produced by `bench/mtpdump.cpp` against the deployed GGUF on the Jetson AGX Xavier. Little-endian,
no padding between records.

    char[4]   magic = "MTP1"
    u32       n_embd    = 2048
    u32       n_vocab   = 248320
    u32       n_tokens  = number of records that follow
    then n_tokens records of 8200 bytes each, in sequence order:
        u32   token_in     the token at position i, fed to the head
        u32   token_tgt    the label: the deployed file's own argmax at position i, which is
                           the token it would emit at i+1 (see "What the label is" below)
        f32   h[2048]      trunk hidden state at position i, *before* the final norm

The file is complete iff `n_tokens` matches the record count implied by the file size:
`file_size == 16 + 8200 * n_tokens`. Dumps are written to `<name>.mtp.tmp` and renamed, so a
`.mtp` without its `.tmp` sibling is finished.

`h` comes from the Xavier's own kernels, which is the whole point: the head has to be trained on
the hidden states the deployed binary will actually give it.

## What the label is, and the one-position shift

Acceptance is "did the draft equal the token the trunk would have emitted", so the training label
is the deployed file's **own argmax** at position i, which is `token_tgt`. Training against the
corpus token instead optimizes a different objective (predicting human text rather than predicting
this file's greedy output) and would not raise acceptance; `mtpdump --no-logits` records the
corpus token if that variant is ever wanted as an auxiliary signal.

The head's input is **shifted by one position**, and this is easy to get wrong. In
`common/speculative.cpp:1720-1736` the MTP driver pairs each token with the target hidden state of
the *previous* row (`h_tgt + (k-1)*n_embd`, with `pending_h` carrying the last row across calls),
so at sequence position p the head consumes:

    h     = trunk hidden state at position p-1   (record i-1's `h`)
    token = the token at position p              (record i's `token_in`)
    label = the token at position p+1            (record i's `token_tgt`)

A training example is therefore spliced from **two adjacent records**: `h` from record i-1, token
and label from record i. Pairing `h_i` with `token_in_i` is off by one position, which trains the
head on a function the runtime never calls. Drop position 0 of each shard (no predecessor row);
at runtime a fresh sequence sees `pending_h` zeros there, so it is one position in ~2,600.

Records are written in sequence order within a shard (`pos = start + i`), so consecutive records
are consecutive positions and a trainer can rebuild the attention context the head sees at
runtime from a sliding window of them. Do not shuffle shards into an order that loses that.

## Runtime call shape (read from the source, not inferred)

`common_speculative_impl_draft_mtp::process()` decodes ctx_dft with the (h, token) pairs for every
position of every prompt ubatch and every verify batch, which fills the head's **own** KV cache
(`is_mem_shared` is false for qwen35moe; the head does not share the trunk's cache). `draft()`
then runs the head one position at a time, reading `pending_h` for the last accepted position.
So the head is a causal self-attention block over its own history: a trainer must run it over
consecutive positions, not per row.

`src/models/qwen35moe.cpp` fixes the rest of the block, and each of these is a place where a
reimplementation silently diverges:

    concat   = [ rmsnorm(emb(tok_in), nextn.enorm), rmsnorm(h, nextn.hnorm) ]   # emb FIRST
    x        = nextn.eh_proj @ concat                                          # 4096 -> 2048
    a        = rmsnorm(x, attn_norm)
    Q_full   = attn_q @ a                        # 16 heads x 512 = 8192 rows
    Q, gate  = Q_full[..., :256], Q_full[..., 256:]                            # 256-dim sink gate
    Q        = rmsnorm(Q, attn_q_norm);  K = rmsnorm(K, attn_k_norm)            # per-head, 256
    Q, K     = rope_multi(..., n_rot = 64 of 256, sections [11,11,10,0], base 1e7)
    out      = attn(Q, K, V) * sigmoid(gate) ;  x = attn_output @ out + x
    x        = x + MoE(rmsnorm(x, attn_post_norm))                             # 256 experts, top-8,
                                                                               # softmax gate, renorm,
                                                                               # SILU, + sigmoid-gated
                                                                               # shared expert
    logits   = output.weight @ rmsnorm(x, nextn.shared_head_norm)

`nextn.shared_head_norm` exists in these files, so `output_norm` is not the head's final norm.
`output.weight` is shared with the trunk, so it is frozen during training along with
`token_embd.weight`.


## What the head computes

Read from `src/models/qwen35moe.cpp` (line ~728):

    h_norm = rmsnorm(h, blk.40.nextn.hnorm)
    e_norm = rmsnorm(emb(token_in), blk.40.nextn.enorm)
    x      = concat(h_norm, e_norm)              -> 4096
    x      = blk.40.nextn.eh_proj @ x            -> 2048
    x      = attention(blk.40.attn_*)            one block, same as a trunk layer
    x      = MoE(x, blk.40.ffn_*)                top-8 of 256 + shared expert
    logits = blk.40.nextn.shared_head_norm(x) -> output.weight @ .

There is no `nextn.shared_head_head` tensor in these files, so the head shares the trunk's
`output.weight` for its logits. `output.weight` is therefore a head input as well as a trunk one,
and it is the tensor whose type family differs across the requant files.

## Files to transfer

    Q4KM-all-q4k.head.gguf   1.21 GB   23 tensors: all blk.40.*, token_embd.weight,
                                      output.weight, output_norm.weight, plus the verbatim
                                      metadata block including the tokenizer
    dumps/*.mtp              ~25 GB    the dataset, one file per corpus shard
    MTP-DUMP-FORMAT.md                 this file

Each tensor in the subset is byte-identical to the same tensor in the served
`Q4KM-all-q4k.gguf` (verified by sha256 over all 23), so dequantizing on the training box
reproduces the exact weights the board runs.

## Why extraction cannot move to the T4

The T4 has 16 GB of device memory and the served file is 20.5 GB. llama.cpp would offload part of
the trunk to the CPU, and a partly-CPU forward produces different `h` than the board's all-GPU
one - the same class of change as the `-ub` width effect, which moves `h` by 12.6% relative RMS
with nothing else altered. A head trained on those hidden states would be trained against a
different function than the one it is deployed against. Extraction stays on the Xavier; the T4
trains.
