# Speculators: DFlash sidecar and TokenGate

**Date:** 2026-11-01
**Build:** mainline `dde879b`
**Verdict:** neither improves on the model's own MTP head. MTP at
`--spec-draft-n-max 3 --spec-draft-p-min 0.5` remains best.

## DFlash sidecar: works, but slower than MTP

`/home/triph/models/Ornith-1.5-35B-A3B-DFlash-BF16.gguf` is a real DFlash draft:

| Property | Value |
|---|---|
| arch | `dflash` |
| blocks | 6 |
| embedding | 2048 |
| heads / KV heads | 32 / 8 |
| feed forward | 6144 |
| block size | 16 |
| `target_layers` | `[2, 7, 12, 17, 23, 28, 33, 38]` |
| sliding window | 4096, pattern `[T,T,T,T,T,F]` |
| base model | Ornith 1.5 35B A3B |
| file type | 32 (BF16), 747 MiB |

Mainline supports it via `src/models/dflash.cpp`. Wiring:

```sh
-md /home/triph/models/Ornith-1.5-35B-A3B-DFlash-BF16.gguf \
--spec-draft-ngl 99 --spec-type draft-dflash
```

`--spec-draft-ngl` (aliases `-ngld`, `--gpu-layers-draft`, `--n-gpu-layers-draft`) must be set,
otherwise the draft runs on CPU.

It loads and runs, but is consistently slower than the MTP head. Varied prompt, 400 tokens,
`-ub 32`:

| draft max | p-min 0 | p-min 0.5 | acceptance @ best |
|---|---:|---:|---:|
| 3 | **21.39** | 18.56 | 0.594 |
| 5 | 15.95 | 13.17 | 0.441 |
| 8 | 7.27 | 13.85 | 0.118 |
| 12 | 6.19 | 19.43 | 0.068 |
| 16 | 5.83 | - | 0.054 |

Best DFlash is **21.39 tok/s**, about **32% slower** than MTP at 31.52.

### Why it loses

1. **Acceptance collapses with depth** at `p-min 0` - 0.59 at depth 3 down to 0.054 at depth 16.
   The `p-min` gate patches this substantially (depth 12 goes 6.19 -> 19.43, a 3.1x recovery) but
   never enough to be competitive. Notably the gate helps DFlash and hurts it at the same depths
   as MTP (it hurts at depth 3 and 5, helps at 8 and 12), which confirms `p-min` is a general
   mechanism rather than an MTP quirk.
2. The draft was trained against **base Ornith 1.5**, while Cyber-Tiel is an abliterated finetune
   of it. The MTP head ships inside the target GGUF and was trained jointly with the finetune,
   which is why it predicts this model far better at every depth.

### Constraints found

- **MTP and DFlash cannot be combined.** With `-md` set, the draft context is built from the
  DFlash model, so `--spec-type draft-mtp,draft-dflash` fails with
  `context type MTP requested but model doesn't contain MTP layers`. The draft slot is
  single-occupancy. `common_speculative_init_from_params` chooses the draft model when
  `has_dft()` is true and only falls back to the target model for MTP when it is not.
- When only DFlash is enabled, the target's block 40 (the nextn/MTP head) is reported as unused
  and skipped. This is expected and harmless, and costs no throughput.

### Harness bug found during this work

`-md` support was initially added to `bench/one.sh` incorrectly: the `DRAFT_ARGS` array was
created but `"${DRAFT_ARGS[@]}"` was never inserted into the server command line, so the first
DFlash run silently ran with no draft at all and produced 16.29 tok/s (plain-decode speed).

Root cause: `has_dft()` is `!draft.mparams.empty()`, where `empty()` checks `get_name()`, which
returns `path`. The flag must actually reach the command line or nothing happens and no warning
is printed. `one.sh` now echoes `$NAME CMD:` to stderr on every run so a silently dropped flag
cannot recur.

## TokenGate: not available for this model

- TokenGate is **not upstream**. It is a fork at `/home/minh/src/llama.cpp-tokengate`
  (upstream `de5122d` plus the gate), shipped as a plain source snapshot with no `.git`, so there
  is no diff or branch to cherry-pick.
- It is already compiled: `/mnt/sdcard/opt/llama/bin/llama-server-tgc` and `llama-server-tg`
  accept `--tokengate` (a hidden argument, so it does not appear in `--help`). Mainline rejects it.
- The gate is a **trained architecture feature, not a runtime toggle**. It needs the
  `tg_gate` / `tg_up` / `tg_down` / `tg_qscale` / `tg_kbias` tensors plus the
  `qwen35moe.tokengate.gate_layer` metadata key; when the key is absent gating is disabled.
  Verified at runtime on Cyber-Tiel: `TokenGate disabled: the model has no gate`.
- The only gate artifacts on disk are trained for Qwen3.6-35B-A3B (`g3_gate` and `swa_gate`,
  `.npz` plus LoRA GGUF). The fork contains inference tooling only, not a gate trainer.

### Architectures are byte-identical apart from the gate

GGUF metadata confirms both Cyber-Tiel and the gated Qwen3.6 model are `qwen35moe` with:
41 blocks, 2048 embedding, 16 heads / 2 KV heads, 256 experts with 8 used, identical SSM dims
(conv 4, state 128, groups 16, rank 32, inner 4096), `full_attention_interval` 4, and 1 nextn
layer. Cyber-Tiel's base model is `Huihui Ornith 1.5 35B A3B Abliterated`.

Therefore a gate **transplant** is mechanically possible, but it is an unvalidated experiment
rather than an available checkpoint: the gate was trained on Qwen3.6 layer-3 activations, not on
an abliterated Ornith finetune.

### Why TokenGate would not help even if transplanted

- The fork refuses a quantized V cache, which conflicts with the q8_0 KV / 262K configuration
  that the fastest setup depends on.
- It requires a single default context type.
- The gate's purpose is compute reduction at long context, while the measured bottleneck here is
  decode kernel efficiency at short context. The expected speed win is small relative to the
  cost of losing q8_0 KV.

**Conclusion:** chasing TokenGate was dropped in favour of speed tuning, per instruction. No gate
checkpoint exists for Ornith/Cyber-Tiel and none can be trained with the available tooling.
