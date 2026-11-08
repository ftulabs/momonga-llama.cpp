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

### 2. Future Architectural Campaigns
Future model investigations (e.g., dense architectures like Qwen 2.5 / 3.8 27B, DeepSeek MoE, Gemma, or multimodal models) should create their own subfolder under `findings/<model-architecture>/` following the template and guidelines in [OPTIMIZATION-GUIDE.md](../OPTIMIZATION-GUIDE.md).

---

## Cross-Model References

- [references.md](references.md): Upstream and external repository comparisons, build recipes, and toolchain versions.
- [../OPTIMIZATION-GUIDE.md](../OPTIMIZATION-GUIDE.md): The unified cross-architecture optimization playbook.
- [../KNOWN-GOOD.md](../KNOWN-GOOD.md): The verified production configuration.
- [../FINDINGS.md](../FINDINGS.md): Executive summary and chronological index.
- [../AGENTS.md](../AGENTS.md): Operational rules, measurement discipline, and tool traps.
