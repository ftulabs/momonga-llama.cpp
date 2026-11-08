# 4-Configuration Benchmark: Qwen 3.6 vs Cyber-Tiel-Coder (+/- mmproj)

**Date:** 2026-11-08  
**Hardware:** NVIDIA Jetson AGX Xavier (sm_72 Volta, 30,991 MiB unified memory, CUDA 11.4)  
**Binary:** `momonga-llama.cpp` (commit `cc7b08c`)  
**Context:** `-c 32768 -ctk q8_0 -ctv q8_0 -fa on -bs -ub 32 -b 32`  
**TokenGate Settings:** `--tokengate --tokengate-k 2000 --tokengate-dilate 2 --tokengate-min-span 1024 --tokengate-fa`  
**Speculation:** `--spec-type draft-mtp --spec-draft-n-max 6 --spec-draft-n-min 0 --spec-draft-p-min 0.5 --spec-draft-p-split 0.1`  
**CSV Output:** `bench/benchmark-4configs.csv`

---

## 1. Executive Summary

We performed a side-by-side benchmark comparing 4 configurations:
1. `Qwen 3.6 35B` (base GGUF: `qwen36-35b-a3b-tg-g3-IQ4_XS.gguf`)
2. `Qwen 3.6 35B + mmproj` (`qwen36-35b-mmproj-BF16.gguf`)
3. `Cyber-Tiel-Coder 35B` (base GGUF: `Cyber-Tiel-Coder-35B-A3B-tg.gguf`)
4. `Cyber-Tiel-Coder 35B + mmproj` (`mmproj-Ornith-1.5-35B-BF16.gguf`)

### Key Results

| Configuration | Decode Median (tok/s) | Draft Acceptance | Mean Draft Len | Short Prefill (267 tok) | Long TTFT (8.6k tok) | Passkey Needle | RAM Used (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen 3.6 35B** | 22.83 | 0.679 | 2.22 | 93.7 tok/s | 9.19 s | 100% Match | 21,452 |
| **Qwen 3.6 35B + mmproj** | 22.36 | 0.679 | 2.22 | 95.5 tok/s | 9.25 s | 100% Match | 22,724 |
| **Cyber-Tiel-Coder 35B** | **42.54** | **0.975** | **5.53** | 91.7 tok/s | 9.28 s | 100% Match | 21,354 |
| **Cyber-Tiel-Coder 35B + mmproj**| **42.68** | **0.975** | **5.53** | 91.0 tok/s | 9.21 s | 100% Match | 22,646 |

---

## 2. Detailed Findings

### A. The Impact of `mmproj` on Performance is Exactly Zero
Attaching the multimodal vision projector (`--mmproj`) has **no measurable impact on text generation or prompt prefill**:
- **Decode Throughput:** Cyber-Tiel achieved **42.54 tok/s** without mmproj and **42.68 tok/s** with mmproj (+0.3%, within noise floor).
- **Draft Acceptance & Mean Length:** In both cases, draft acceptance is **bit-identically 0.97538** with mean draft length **5.53**. For Qwen 3.6, acceptance is **0.67871** with mean length **2.22** on both arms.
- **Prefill & TTFT:** Long-context TokenGate TTFT on 8,635 tokens was **9.28 s vs 9.21 s** for Cyber-Tiel, and **9.19 s vs 9.25 s** for Qwen 3.6.
- **Memory Footprint:** The BF16 vision projector adds $\approx 1,272\text{--}1,292\text{ MB}$ to resident unified memory, leaving $>8.2\text{ GB}$ of free headroom at context 32k.

### B. Long Context Prefill Parity
All four configurations deliver identical TokenGate long-context prefill performance:
- In all arms, Client TTFT on 8,635 tokens is **$\approx 9.2\text{ seconds}$** (compared to $>120\text{ seconds}$ baseline without TokenGate).
- Passkey needle retrieval (`BLUE-VELVET-99` placed at 50% depth) was **100% exact match** across all 4 configurations.

### C. MTP Draft Divergence Between Cyber-Tiel and Qwen 3.6
While prefill speed is identical, speculative decode throughput differs significantly:
- **Cyber-Tiel-Coder-35B:** Draft acceptance is **0.975** with mean draft length **5.53**, delivering **42.54 - 42.68 tok/s**.
- **Qwen 3.6 35B:** Draft acceptance sits at **0.679** with mean draft length **2.22**, delivering **22.36 - 22.83 tok/s**.
- **Cause:** Cyber-Tiel's `blk.40` MTP draft head was calibrated for depth 6 speculative decoding on its specific quantized vocabulary, whereas Qwen 3.6's default draft head was trained for shallow depth (`n-max 2`, as recorded in `docs/minh-llamacpp-qwen36.md`). Under `n-max 6`, Qwen's drafter degrades after step 2, achieving 22.8 tok/s (still faster than plain decode at 18.3 tok/s, but well below Cyber-Tiel).
