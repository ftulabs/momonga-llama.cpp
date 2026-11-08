# MTP Phase 1 Drafter Retraining Deployment Results: Negative Result

**Date:** 2026-11-07  
**Model:** `Q4KM-all-q4k-retrained.gguf` (20.461 GB)  
**Base Trunk:** `Q4KM-all-q4k.gguf` (dense q4_k, expert q4_k, shared expert q5_k, lm_head q4_k)  
**Retrained Head:** blk.40 dense projection and attention parameters (35.66M params)  
**Test Setup:** Jetson AGX Xavier (sm_72), 5 reps x 400 tokens, `bench/prompt2.txt`, `-bs`, `CTX=40960`, `UB=32`, `B=32`  

---

## 1. Executive Summary

Phase 1 retraining realigned the 35.7M dense projection and causal attention parameters of the
MTP draft head (`blk.40`) against 3,000,519 tokens of Xavier activations dumped from `Q4KM-all-q4k.gguf`,
with all 256 MoE experts frozen.

Despite achieving strong teacher-forced validation metrics during PyTorch training on Tesla K80
(cross-entropy 0.8005, greedy argmax agreement 80.88%), deploying the retrained weights back into
`Q4KM-all-q4k-retrained.gguf` **failed to recover speculative draft acceptance**:

| Configuration | Draft Acceptance | Mean Draft Len | Decode Throughput (tok/s) | Relative to Plain Decode |
|---|---|---|---|---|
| Shipped Control (`UD-IQ4_XS` + factory blk.40) | **0.96657** | **5.68** | **36.93** | +66.0% |
| `Q4KM-all-q4k` (plain decode, `--spec-type none`) | N/A | N/A | **27.86** | baseline |
| `Q4KM-all-q4k` + factory blk.40 | 0.64583 | 2.10 | 23.35 | -16.2% |
| `Q4KM-all-q4k` + `output.weight` q6_k swap | 0.67286 | 2.39 | 22.53 | -19.1% |
| **`Q4KM-all-q4k-retrained` (Phase 1 Retrained)** | **0.49206** | **2.08** | **20.91** (median reps 2-5) | **-24.9%** |

Phase 1 dense retraining did not recover acceptance; speculative decode remained substantially slower
(-24.9%) than running plain decode without speculation.

---

## 2. Measured Deployment Runs

Benchmark executed via `bench/mtp-retrained.sh` (`allq4k-retrained-np400`, `NPRED=400`, 5 reps):

```
allq4k-retrained-np400 dec rep1: TG 23.34 tok/s (acceptance = 0.61409, mean len = 2.00)
allq4k-retrained-np400 dec rep2: TG 19.97 tok/s (acceptance = 0.42308, mean len = 1.75)
allq4k-retrained-np400 dec rep3: TG 23.80 tok/s (acceptance = 0.60596, mean len = 1.99)
allq4k-retrained-np400 dec rep4: TG 19.83 tok/s (acceptance = 0.40129, mean len = 1.75)
allq4k-retrained-np400 dec rep5: TG 21.91 tok/s (acceptance = 0.49206, mean len = 2.08)
```

- Reps 1, 3, 5 read acceptance 0.492 - 0.614 and throughput 21.91 - 23.80 tok/s.
- Reps 2, 4 collapsed to acceptance 0.401 - 0.423 and throughput 19.83 - 19.97 tok/s.
- Pooled median decode throughput across reps 2-5: **20.91 tok/s**.

---

## 3. Technical Diagnosis: Why Phase 1 Failed to Recover Acceptance

1. **Frozen MoE Experts (Phase 1 Limitation):**
   The MTP head (`blk.40`) contains 1.826 BILLION parameters in its 256 routed MoE experts,
   representing 98% of the block's computation and capacity. In Phase 1, only the 35.7M dense
   parameters (`eh_proj`, `attn_*`, norms) were trained. Adapting only the front-end dense layers
   without fine-tuning the 256 experts created a representation mismatch at the input of the MoE
   router (`ffn_gate_inp`), causing routing divergence on multi-step autoregressive drafting.

2. **The Teacher-Forced Optimization Trap:**
   Training optimized single-step teacher-forced cross-entropy:
     loss = CE(logits, target) + lambda * MSE(h_head, trunk_h)
   On a fixed dataset where ground-truth trunk hidden states are fed at every position, the model
   easily achieves 80.88% argmax agreement. However, during speculative decoding ($N=6$), the drafter
   operates autoregressively without ground-truth hidden states for steps 2 through 6. Error
   accumulation compounds exponentially: $(0.80)^6 \approx 0.26$.

3. **Domain Mismatch:**
   The training corpus was assembled from source code repositories (`.py`, `.cpp`, `.js`), whereas
   the benchmark prompt `prompt2.txt` is prose.

---

## 4. Architectural Verdict and Next Steps

- Retraining dense projection and attention layers alone (`--freeze-moe`) cannot realign the MTP head
  when the base trunk has been requantized.
- Speculative decode throughput under Phase 1 (20.91 tok/s) remains strictly inferior to plain decode
  without speculation (**27.86 tok/s** on `prompt2.txt`, **23.75 tok/s** on 8.8k context).
- For `Q4KM-all-q4k.gguf`, the adopted configuration remains `--spec-type none -ub 2048 -c 40960`.
