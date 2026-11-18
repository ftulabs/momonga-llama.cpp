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
  - **29.74 +/- 0.23 tok/s decode** (+31.0% vs 22.71 baseline; `ffn_gate/up` at 97.7 GB/s, 83.2 GB/s whole-model throughput).
  - **716.30 +/- 2.70 tok/s prefill (`pp2048`)** (+34.4% vs 533.06 baseline) via 4-warp vectorized `dequantize_block_q4_K<half>`.
  - Native `f16` KV cache strictly beats `q8_0` KV across all depths `0..16384` (up to **+61.2% decode at 16k**: `17.31` vs `10.74 tok/s`).
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
