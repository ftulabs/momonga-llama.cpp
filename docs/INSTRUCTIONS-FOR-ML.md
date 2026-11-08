# MTP Head Retraining Guide for Agent on `ml`

This document provides complete, self-contained context and executable instructions for the
agent on the remote training box **`ml`** (Tesla T4, host `100.64.0.22`, user `tphuong`).

---

## 1. Mission Context & Objective

### What We Are Doing
We are retraining a replacement Multi-Token Prediction (MTP) draft head (`blk.40`) for
the model `Cyber-Tiel-Coder-35B-A3B` deployed on a Jetson AGX Xavier.

### Why We Are Retraining
1. **The Starting Point**: The base model shipped with an MTP draft head running speculatively
   at `--spec-draft-n-max 6 --spec-draft-p-min 0.50`, achieving **0.96657 draft acceptance**,
   **5.68 mean draft length**, and **36.93 tok/s** steady-state decode on varied text.
2. **The Base Requantization**: To improve plain decode speed and reduce memory pressure on the
   Xavier's unified memory, the trunk was retyped to `Q4KM-all-q4k.gguf` (20.46 GB, 27.86 tok/s
   plain decode, +25.2% faster than stock `Q4_K_M`).
3. **The Regression**: The shipped MTP head (`blk.40`) was pre-trained against activations of
   the original uncompressed trunk. Retyping the base model slightly shifted the internal hidden
   state geometry. Consequently, the shipped MTP head's draft acceptance collapsed from
   **0.96657 down to 0.64583** (mean draft length 5.68 -> 2.10), causing speculative decode to
   drop from 36.93 tok/s down to **23.35 tok/s** (slower than plain decode without speculation).
4. **The Goal**: Retrain `blk.40` against **3,000,519 extracted tokens of actual runtime activations**
   dumped from `Q4KM-all-q4k.gguf` on the Xavier's own GPU kernels. This will realign the drafter
   with the deployed file's representation, recovering draft acceptance to ~0.95+ and delivering
   **~37 to 40 tok/s**.

---

## 2. Hardware & Environment on `ml`

* **Host**: `ml` (Tailscale MagicDNS `ml.tailnet.ftu`, IP `100.64.0.22`, user `tphuong`).
* **GPU**: NVIDIA Tesla T4 (16 GB GDDR6 VRAM, Turing architecture, compute capability `sm_75`).
* **CRITICAL ARCHITECTURAL CONSTRAINT**: Turing `sm_75` has **NO BFLOAT16 tensor cores**.
  * **DO NOT USE `torch.bfloat16`**: On T4, `bfloat16` drops to slow software emulation.
  * **USE FP16 AMP**: Use `torch.float16` with `torch.cuda.amp.autocast()` and
    `torch.cuda.amp.GradScaler()`. The training script `train-mtp-head.py` is pre-configured
    for this.

---

## 3. Strict Non-Negotiable Project Rules

These rules come from `AGENTS.md` and are mandatory:
1. **ASCII Only**: Code, documentation, and commit messages must be strict ASCII.
2. **Write Predictions Down Before Running**: Record expected metrics before launching runs.
3. **Strict Gate Boundary**: **NO gradient steps may be run until the parity gate passes.**
   A head trained against a Python function that does not match the engine's exact graph
   optimizes a metric that will not deploy.
4. **Report Negative Results with Equal Prominence**: Document failures, losses, and spreads
   honestly.

---

## 4. Architecture & Mathematical Alignment (Where Traps Bite)

The model is a hybrid architecture with 41 blocks (layers 0..39 are the base trunk, layer 40
is the MTP drafter). Layer 40 consists of a dense projection, a causal attention block, and an
MoE block (256 routed experts + 1 shared expert).

### A. The One-Position Shift
In `common/speculative.cpp:1745-1749`, the MTP driver pairs each token with the target
hidden state of the *previous* row. At sequence position $p$:
* The head consumes:
  * `token` = `token_in[p]` (the token at position $p$)
  * `h` = trunk hidden state at position $p-1$ (zeros for position 0, `pending_h`)
* The target to predict is:
  * `token_tgt[p]` = the trunk's greedy argmax emitted after position $p$ (i.e., token at $p+1$).
* **Empirical Verification**: We tested this alignment on the 256-position reference file:
  * Current alignment (`head[i] == tgts[i]`): **89.84% agreement** (reaching 100% past pos 96).
  * Shifted by -1 (`head[i] == tgts[i-1]`): **1.57% agreement**.
  * Shifted by +1 (`head[i] == tgts[i+1]`): **1.18% agreement**.

### B. The Dual Training Objective
At training time:
$$\mathcal{L} = \mathcal{L}_{\text{CE}}(\text{logits}_{\text{head}}[p], \text{token\_tgt}[p]) + \lambda \cdot \mathcal{L}_{\text{MSE}}(h_{\text{head}}[p], h_{\text{trunk}}[p])$$
* **Why Cross-Entropy against trunk argmax**: Speculative acceptance means "did the draft
  token equal what the trunk would have greedily generated". We train against `token_tgt`
  (the trunk's greedy prediction), not human text continuation.
* **Why the Consistency MSE Loss ($\lambda = 0.1$)**: At runtime, MTP drafts 6 steps
  in a chain (`n-max 6`). Step 1 receives $h_{\text{trunk}}$, but steps 2..6 receive the
  head's **own** hidden state from the previous draft step. If $h_{\text{head}}$ drifts
  from $h_{\text{trunk}}$, multi-step draft acceptance collapses. The MSE term pulls
  $h_{\text{head}}[p]$ toward $h_{\text{trunk}}[p]$.

### C. Shared Weights (MUST REMAIN FROZEN)
`token_embd.weight` and `output.weight` are shared with the trunk:
* `output.weight` is 248,320 x 2048.
* They are shared references and must **never** receive gradients.

---

## 5. Files in Your Workspace (`~/mtp-bundle/`)

| File / Folder | Size | Role |
|---|---|---|
| `Q4KM-all-q4k.head.gguf` | 1.21 GB | The served file's `blk.40.*` plus `token_embd` and `output` (23 tensors). |
| `head-ref-256pos.hlg` | 122 MB | **Ground truth**: 248,320 `f16` logits per position dumped by Xavier's C++ MTP graph on 256 positions. |
| `parity-input-256pos.mtp` | 2.1 MB | Exact 256 input records (`token_in`, `token_tgt`, `h[2048]`) matching the reference. |
| `mtp_head_torch.py` | 16 KB | PyTorch head implementation, self-check, and parity gate test harness. |
| `train-mtp-head.py` | 23 KB | Complete training script with fp16 AMP, KV caching, MSE loss, and checkpointing. |
| `dumps/` | 22.91 GB | 1,190 shards (`shard-000000.mtp` to `shard-001189.mtp`), 3,000,519 positions. |
| `MTP-DUMP-FORMAT.md` | 6.8 KB | Specification of the binary `.mtp` and `.hlg` dump formats. |

---

## 6. Step-by-Step Instructions for the Agent on `ml`

### Step 1: Environment & Dependency Check
```bash
cd ~/mtp-bundle

# Verify GPU:
python3 -c "import torch; print(f'PyTorch {torch.__version__} on {torch.cuda.get_device_name(0)} (sm_{torch.cuda.get_device_capability(0)})')"

# Verify gguf library:
python3 -c "import gguf" || pip install gguf

# Run structural self-check:
python3 mtp_head_torch.py --gguf Q4KM-all-q4k.head.gguf --selfcheck
```
*Expected Selfcheck Output*:
* Shape: `(4, 248320)`
* Deterministic: `True`
* Trainable: `~1827.0 M`, Frozen: `~741.0 M` (all parameters loaded).

---

### Step 2: The Mandatory Parity Gate

Execute the parity verification:
```bash
python3 mtp_head_torch.py \
    --gguf Q4KM-all-q4k.head.gguf \
    --parity head-ref-256pos.hlg \
    --dump parity-input-256pos.mtp
```

### Pass Criteria:
1. **`argmax agreement: 100.0000%`** (256/256 positions).
2. **`logit cosine: min > 0.999000`** (and mean $> 0.9999$).

> [!WARNING]
> If parity fails, STOP. Do not modify thresholds to make it pass.
> Inspect the 8-band table:
> * Band 0 failure (flat across all positions): Front-end issue (`enorm`/`hnorm`, `eh_proj`, or MoE gating).
> * Growing failure with position: Attention/KV cache issue (masking, RoPE, or $h$ shift).

---

### Step 3: Run Phase 1 Training (`--freeze-moe`)

Once the parity gate passes, launch Phase 1 training:
```bash
python3 train-mtp-head.py \
    --gguf Q4KM-all-q4k.head.gguf \
    --dumps-dir ./dumps \
    --out-dir ./checkpoints \
    --freeze-moe \
    --epochs 1 \
    --lr 2e-4 \
    --batch-size 4 \
    --window-len 512 \
    --prefix-len 256 \
    --mse-weight 0.1 \
    --val-interval 500
```

#### Why `--freeze-moe` (Phase 1):
* Freezes the 1.8B MoE weights (`blk.40.ffn_*`) and trains only the **35.7M dense parameters**:
  * `blk.40.nextn.eh_proj.weight` (2048 x 4096)
  * `blk.40.attn_q.weight`, `attn_k.weight`, `attn_v.weight`, `attn_output.weight`
  * Layer norms: `enorm`, `hnorm`, `attn_norm`, `attn_q_norm`, `attn_k_norm`, `post_attention_norm`, `shared_head_norm`.
* **VRAM Efficiency**: Consumes only **~6 to 9 GB VRAM** on the T4 (comfortably fitting in 16 GB without risking out-of-memory errors).
* **Stability**: Preserves expert routing while realigning the linear projection and attention layers to the new trunk representation.

#### Monitoring During Training:
* Shards where `index % 20 == 19` (e.g. `shard-000019.mtp`) are automatically held out as validation.
* Checkpoints are automatically saved to `./checkpoints/best_checkpoint.pt`.
* Watch:
  * `loss_ce`: Should decrease steadily.
  * `val_acc`: Argmax agreement against trunk predictions should rise above 80-85%.
  * `val_cos`: Hidden state cosine similarity should remain high ($> 0.98$).

---

### Step 4: Completion & Reporting

When training finishes:
1. Print the final evaluation metrics from `./checkpoints/best_checkpoint.pt`:
   * Final validation CE loss
   * Final validation argmax agreement (`val_acc`)
   * Final validation hidden state cosine similarity (`val_cos`)
2. The output checkpoint contains:
   ```python
   {
       "model": {name: tensor_cpu for name, tensor in state_dict.items() if not shared},
       "step": step,
       "epoch": epoch,
       "val_ce": val_ce
   }
   ```
3. Report the completion and metrics so the checkpoint can be transferred back to Xavier for re-integration into `Q4KM-all-q4k.gguf` and live benchmark validation.
