# Momonga llama.cpp - Experimental Findings Index

This directory archives empirical findings, telemetry logs, profiling audits, and A/B benchmark
evaluations conducted on edge acceleration platforms (NVIDIA Jetson AGX Xavier, Volta sm_72).

Every finding here represents measured facts on real hardware, following the measurement discipline
defined in [AGENTS.md](../AGENTS.md) and synthesized in [OPTIMIZATION-GUIDE.md](../OPTIMIZATION-GUIDE.md).

---

## Architecture Directories

Findings are partitioned by model architecture and workload:

### 1. [`cyber-tiel-qwen35moe/`](cyber-tiel-qwen35moe/) (Goal Reached)
Complete engineering record (58 dated reports, October 14 - November 8, 2026) for the
**Cyber-Tiel-Coder-35B-A3B / Qwen3.5-35B-A3B** hybrid Recurrent-Attention MoE architecture.
- **Milestone Reached:**
  - **43.08 - 43.97 tok/s speculative decode** (and **47.5 - 48.4 tok/s** steady-state generation) at draft acceptance 0.968 - 0.975.
  - **2,025 - 2,070 tok/s long-context prefill** (TTFT 4.79 s at 9.6k, 11.15 s at 22.1k tokens) via decoupled TokenGate.
  - **262,144 context allocation (`-c 262144`)** with 4.4 GB free RAM headroom.
  - Multimodal vision projector integration (BF16 Ornith / Qwen-VL mmproj).
- **Index:** See [cyber-tiel-qwen35moe/README.md](cyber-tiel-qwen35moe/README.md) for the full catalog of 58 dated reports.

### 2. [`zerank-2/`](zerank-2/) (Active Production Reranker)
Complete engineering record for the **ZeroEntropy `zerank-2-reranker`** (4.0B dense Qwen3 causal cross-encoder).
- **Milestone Reached:**
  - **177.87 ms median document latency** (1,780 ms for 10-document batch).
  - **223.7 tokens/sec** end-to-end throughput across HTTP `/rerank`.
  - **100.0% Top-1 accuracy** on cross-domain multi-document ranking.
  - **Edge Economic Objective:** 28.73 W, 7.80 Tokens/Joule, 0.195 Docs/Joule (ServerBench).
  - Zero-copy view projection on target token 9454 (`"Yes"`), saving 160,000x LM head compute/buffer waste (~400 MB DRAM read and 640 MiB buffer per pass).
  - Unlocked Volta Tensor Core HMMA and GQA flash attention on sm_72 (cc 720).
- **Report:** [zerank-2/2026-11-10-zerank-2-causal-reranker-optimization.md](zerank-2/2026-11-10-zerank-2-causal-reranker-optimization.md).

### 3. [`k2-horizon/`](k2-horizon/) (Dense 4B/5B Volta Optimization Complete)
Engineering record for **K2-Horizon 4B Q4_K_M** (36-layer dense transformer, 5.06B params, grouped RMSNorm, 250k vocab):
- **Milestone Reached:**
  - **30.61 +/- 0.19 tok/s decode** (+34.8% vs 22.71 baseline; `29.60 tok/s` server `-t 2 -tb 2 -bs` median; `ffn_gate/up` at 97.7-98.9 GB/s, `Q6_K` `lm_head` cut `5.93 -> 5.55 ms` via `64`-reg `16` blocks/SM, `RMS_NORM+MUL+ADD` fusion, direct `src1` `q8_1` cache, PR `#29796` sync skip).
  - **771.66 +/- 4.19 tok/s prefill (`pp2048`)** (+44.8% vs 533.06 baseline; `729.59 +/- 8.24` `pp512` `-ub 1024`; `672.2 tok/s` server prefill) via 4-warp vectorized `dequantize_block_q4_K<half>`, Volta deep-K `cublasGemmEx ALGO4/5/6`, `src1` `f16` cache, `float4` contiguous GLU, and right-sized `rope.cu` `block_y`.
  - Native `f16` KV cache beats `q8_0` KV across all depths `0..16384` (**`17.55`** vs coalesced **`14.06 tok/s`** at 16k via PR `#30792`; exact `PPL = 2.5997 +/- 0.04012`).
  - **Report:** [k2-horizon/2026-11-18-k2-horizon-4b-optimization.md](k2-horizon/2026-11-18-k2-horizon-4b-optimization.md).

### 4. Future Architectural Campaigns
Future model investigations (e.g., dense architectures like Qwen 2.5 / 3.8 27B, DeepSeek MoE, Gemma, or multimodal models) should create their own subfolder under `findings/<model-architecture>/` following the template and guidelines in [OPTIMIZATION-GUIDE.md](../OPTIMIZATION-GUIDE.md).

---

## Cross-Model References

- [references.md](references.md): Upstream and external repository comparisons, build recipes, and toolchain versions.
- [../OPTIMIZATION-GUIDE.md](../OPTIMIZATION-GUIDE.md): The unified cross-architecture optimization playbook.
- [../KNOWN-GOOD.md](../KNOWN-GOOD.md): The verified production configuration.
- [../FINDINGS.md](../FINDINGS.md): Executive summary and chronological index.
- [../AGENTS.md](../AGENTS.md): Operational rules, measurement discipline, and tool traps.
