# Momonga llama.cpp - Cross-Architecture Optimization Playbook & Engineering Guide

**Target Audience:** Autonomous coding agents and performance engineers optimizing LLMs on resource-constrained embedded/edge accelerators (NVIDIA Jetson AGX Xavier sm_72, Orin sm_87, and similar Unified Memory SoC platforms) using `llama.cpp`.

**Author:** ftulabs / Momonga Project
**Date:** November 2026
**Status:** Validated on production hardware (Jetson AGX Xavier, Volta sm_72, 32 GB LPDDR4x, Carmel 8-core CPU).

---

## 1. Executive Overview & Hardware Realities

This playbook captures the methodology, kernel tuning recipes, telemetry techniques, and failure modes discovered while pushing the Cyber-Tiel / Qwen 3.5 MoE architecture from an unpatched baseline of **31.5 tok/s decode / 71 tok/s prefill** to **43.97 tok/s decode (48.4 tok/s steady state) / 2,070 tok/s prefill at 262,144 context**.

These principles are **architecture-agnostic** and apply equally to dense models (e.g. Qwen 2.5 / 3.8 27B, LLaMA 3, Mistral), hybrid linear-attention/SSM models (DeltaNet, Mamba), and Mixture-of-Experts (MoE).

### 1.1 The Unified Memory Architecture (UMA) Reality
On embedded SoCs (Jetson AGX Xavier, Orin, Apple Silicon):
- **CPU and GPU share the same physical DRAM bus** (e.g., 256-bit LPDDR4x @ 2133 MHz = 136.5 GB/s theoretical on Xavier).
- **There is no PCIe bus bottleneck.** Zero copy between host and device is physically possible, but cache coherency and system bus arbitration introduce real overhead.
- **The true hardware ceiling must be measured, not calculated.** On Xavier, a raw sequential read kernel measures **109.7 - 112.0 GB/s** (~82% of bus theory). Decode bandwidth will never exceed this ceiling.
- **DRAM latency and LSU queue depth are the primary walls.** Because embedded memory controllers have shallower queues than discrete server GPUs (H100/A100), memory latency hiding requires higher active warp occupancy.

---

## 2. The Seven Golden Rules of Measurement Discipline

Every single optimization trap encountered in this project came from violating one of these discipline rules. Treat them as non-negotiable.

### Rule 1: Pre-register predictions in writing before running
Before executing any benchmark or profiling run, write down the predicted throughput (tok/s), latency (ms), and draft acceptance. If a change fails or deviates, the failure is immediately obvious; otherwise, the engineer will rationalize the outcome post-hoc.

### Rule 2: The A/B Interleaved Both-Orders Protocol (`A-B-B-A`)
Board thermal drift, clock adjustments, and background OS memory compaction cause the Jetson board to drift by **0.5% to 1.1% over a 1-hour session**.
- A single-order A/B test (running all config A runs, then all config B runs) will routinely report a phantom 1% win or flip the sign of a real optimization.
- **Requirement:** For any effect under ~2%, run interleaved arms in both orders (e.g. `base -> test -> test -> base` or `A-B-B-A`).
- **Measure the noise floor with an A/A arm:** Run the identical baseline configuration at the beginning and end of the session. The difference between the two baseline arms is your protocol noise floor. Never claim an effect smaller than this floor.

### Rule 3: Speculative Acceptance is King
Speculative decoding throughput is governed by:
$$\text{Throughput} = \frac{\text{Mean Draft Length}}{\text{Pass Latency}}$$
A kernel modification or weight quantization that speeds up pass latency by 2% but degrades draft acceptance by 2 points (e.g. 0.97 -> 0.95) results in a **net slower model**. Never evaluate kernel speed or weight changes without reporting draft acceptance and mean draft length alongside tok/s.

### Rule 4: Standardize Request Lengths (`NPRED=400`)
At the end of a generation budget, the draft engine truncates speculative chains prematurely. A benchmark with `NPRED=128` measures an artificially truncated draft chain (e.g. mean length 4.22, 30.1 tok/s), whereas `NPRED=400` allows the draft engine to reach steady state (mean length 5.68, 36.3 tok/s) at identical acceptance. Always use `NPRED=400` for comparable decode runs.

### Rule 5: For CPU Host Work, Compare Instructions/Cycles, Never Task-Clock
Carmel / ARM CPU cores dynamically scale clock frequencies between 1.14 GHz and 1.38 GHz even under constant load.
- Measuring CPU runtime in task-clock milliseconds hides 12-18% speedups.
- Use Linux `perf stat` and compare `instructions` and `cycles`.

### Rule 6: Never Extrapolate Microbenchmarks Across Batch Sizes
A kernel that wins at batch size 5 or 7 may lose at batch size 1 or 4. Matrix-vector (`MUL_MAT_VEC_Q`) and matrix-matrix (`MUL_MAT_Q`) paths have distinct arithmetic intensity transitions. Always measure at the exact batch shape the model runs during decoding and prefill.

### Rule 7: Verify GGUF Artifacts Before Benchmarking
When modifying tensor formats or quantizing individual layers, parse the GGUF header and verify tensor types, counts, and byte offsets. A failed override produces a plausible file that silently measures the wrong baseline.

---

## 3. Profiling & Diagnostic Toolkit

### 3.1 Profiler Setup on Tegra / Jetson
- **Root access is required** for CUPTI activity profiling. Run profilers with `sudo`.
- **`nsys` (Nsight Systems):** Use `nsys profile --trace=cuda,nvtx,osrt`. On older Jetpack/momonga releases, use `gpukernsum` (not `cuda_gpu_kern_sum`) for CLI summaries.
- **`ncu` (Nsight Compute):** Do NOT profile `llama-server` directly with the full model loaded in memory; the NCU replay buffer will trigger an OOM (`NvMapMemAllocInternalTagged: error 12`). Profile kernels in isolation using standalone microbenchmarks (`opbench`) at the model's exact tensor geometry.
- **Hardware Counter Trap on Tegra (`gv11b`):**
  - There are NO `dram__` counters on Jetson AGX Xavier.
  - Read actual DRAM read traffic as:
    $$\text{DRAM Bytes} = \text{lts\_\_d\_sectors\_fill\_sysmem.sum} \times 32\text{ bytes}$$
  - The counter `lts__d_sectors_fill_device.sum` always reports 0 because GPU memory is system memory on Tegra.
- **Avoid CUPTI Idle Bubbles:** CUPTI activity tracing introduces 40-95 ms of synthetic idle delay per pass. Do not evaluate GPU bubbles from an nsys trace; evaluate kernel runtimes only.

### 3.2 Time Budget Decomposition
A decode pass must be partitioned into its fundamental components:
1. **MoE Expert Projections (`mul_mat_vec_q_moe`):** Usually 35-45% of pass time.
2. **Dense Attention & Linear-Attention / SSM Projections (`mul_mat_vec_q`):** Usually 20-30%.
3. **LM Head Projection (`output.weight`):** 15-25% of pass time (multiplied by the number of draft steps in speculative decoding).
4. **Non-GEMM Kernels (Norms, Activations, Concat, Copy):** 5-10%.
5. **Host CPU Graph Submission & Sampling:** 5-15% (visible as idle time between CUDA graph replays).

If an optimization does not target one of the top three buckets, its theoretical ceiling on end-to-end throughput is capped at <2%.

---

## 4. Kernel Optimization Recipes in `ggml-cuda`

### Recipe A: The Dispatch Table & Batch Cliff Hunt
**Problem:** In `llama.cpp`, the CUDA backend selects kernel implementations based on architecture capability (`cc`). In many upstream files, architecture checks test for exact equality (e.g. `cc == GGML_CUDA_CC_VOLTA` where `GGML_CUDA_CC_VOLTA = 700`).
- Jetson AGX Xavier is `sm_72` (`cc = 720`).
- An exact check fails, silently dropping sm_72 into the `pascal_older` dispatch table!
- **Symptom:** At batch size > 4 (e.g. during speculative verification), the dispatcher switches from fast `mul_mat_vec_q` (MMVQ) to slow `mul_mat_q` (MMQ), creating a 2.5x latency cliff.
- **Fix:** Ensure architecture checks use `>=` or properly include sm_72. Migrate batch handling to the grid Z-axis (`blockIdx.z` for token indices), allowing thread blocks to remain 1 warp wide regardless of batch size.

### Recipe B: Launch Bounds & Register Clamping
**Problem:** `nvcc` optimizes for instruction-level parallelism (ILP) by default, consuming registers liberally (e.g. 79 registers for a 32-thread block).
- On Volta/Turing SMs, this caps active block residency at 24 blocks per SM (37.5% occupancy), preventing latency hiding on memory-latency-bound quantized kernels (like `IQ3_S`).
- **Fix:** Explicitly declare launch bounds:
  ```cuda
  __launch_bounds__(32, 8) // or (32, 32)
  ```
- **Result:** Pins register usage to 64 registers, increasing occupancy from 34.9% to 47.9%, yielding an immediate **+2.25% end-to-end decode speedup**.

### Recipe C: Multi-Row GEMV & Activation Vector Reuse
**Problem:** In dense matrix-vector multiplications (`mul_mat_vec_q` in `mmvq.cu`), each CUDA block computes one row of the matrix. During multi-token speculative verification (5 to 8 tokens), the activation vector `y` must be re-read for every output row. At 7 tokens, activation re-reads account for 57% of all L1 cache sectors!
- **Fix:**
  1. Increase `rows_per_cuda_block` from 2 to 4 for column counts 5..8.
  2. Increase the shared memory carveout to 64 KB using `cudaFuncAttributePreferredSharedMemoryCarveout` (avoiding residency caps caused by larger accumulator buffers).
  3. Clamp launch bounds with `__launch_bounds__(..., 8)` to prevent register spills.
  4. Widen `should_use_small_k` to trigger multi-row layouts when K is an exact multiple of the block iteration.
- **Result:** Slashes L1 global load sectors by 38% and cuts kernel runtime by **8% to 10.5%**, providing a **+3.4% end-to-end speedup**.

### Recipe D: Glue & Memory Movement Fusion
**Problem:** Non-contiguous memory copies (e.g. `concat_non_cont` or state rollbacks in recurrent DeltaNet/SSM models) often launch grid configurations with poor thread utilization (e.g. launching 256 threads per block when only 10 threads have work).
- **Fix:** Pack rows into `blockDim.y` (e.g. grid `8192x1` -> `1024x1`, block `256x1` -> `32x8`).
- **Result:** Cuts kernel execution time in half (83 us -> 46 us per launch).

---

## 5. Speculative Decoding & Pipeline Offloading

### 5.1 Self-Speculation (MTP) vs Draft Models
- **Multi-Token Prediction (MTP):** Uses the model's own recurrent draft head to predict future tokens. Does not consume extra model memory, but requires **bit-for-bit numerical consistency**.
- **The MTP Sensitivity Rule:** Any change to weight quantizations (e.g. requantizing trunk layers to Q5_K) or router arithmetic (e.g. switching cuBLAS SGEMM to custom float dot products) alters near-tie expert selections in deep MoE layers. While perplexity appears unaffected, MTP draft acceptance **collapses from 0.97 down to 0.64**, reducing overall generation speed by 15-40%.
- **Rule:** Never modify weights or router summation order on an MTP-enabled model unless you retrain the draft head against the new representation.

### 5.2 Single-Sync GPU Sampler (`-bs`)
- **The Host CPU Bottleneck:** In standard llama.cpp, candidate logits are copied to the CPU for top-k, top-p, and softmax sampling. For models with large vocabularies (e.g. 248k tokens), CPU scanning consumes 41.7 ms of CPU time per pass, starving the GPU.
- **The Fix:** Enable backend sampling (`-bs` / `--backend-sampling`) and single-synchronization speculative sampling (PR #30196). Offload draft sampling entirely to the GPU using hardware-accelerated sorting (`cub::DeviceRadixSort`).
- **Result:** Slashes host CPU cycles by 34%, eliminating CPU-GPU pipeline bubbles and delivering a **+1.04% to +3.5% decode speedup**.

### 5.3 Reduced Draft Vocabulary Head
- During draft generation, projecting hidden states across the full vocabulary (248,320 tokens) is wasteful when drafting predominantly uses high-frequency vocabulary.
- Restricting the draft LM head to a top-32k vocabulary subset (`--spec-draft-vocab`) cuts draft head projection time by >50% without degrading acceptance on standard text.

---

## 6. Prefill, TTFT, and Deep Context Scaling

### 6.1 Ubatch Decoupling
- **The Dilemma:** Fast prefill requires wide logical micro-batches (`-ub 1024` or `-ub 2048`) to saturate tensor cores (achieving 200+ tok/s vs 70 tok/s at `-ub 32`). However, setting `-ub` globally causes speculative decode passes to evaluate short prompt prefixes as wide batches, destroying draft acceptance.
- **Solution:** Decouple prefill ubatch from speculative decode ubatch. TokenGate and modern llama.cpp schedulers support independent prefill batching (`TG_UBATCH=1024`).

### 6.2 TokenGate Dynamic Token Pruning
- For long contexts (>8k tokens), quadratic attention and deep MoE prefill dominate latency.
- **Strategy:**
  1. In shallow layers (layers 0..3), evaluate 100% of tokens densely to build strong contextual attention matrices.
  2. In deep layers (layers 4..40), prune 85-95% of intermediate tokens, routing only critical query tokens ($K=350-400$).
- **Result:** Explodes prefill throughput to **2,025 - 2,070 tok/s** (TTFT 4.79 s at 9.6k tokens) while maintaining 100% exact needle retrieval in long-context evaluations.

### 6.3 Deep Context on Hybrid Architectures
- Models combining linear recurrent layers (DeltaNet / SSM / Mamba) with sparse attention layers consume vastly less KV cache memory:
  - Recurrent layers store state in fixed-size registers ($O(1)$ memory per token).
  - Only the 11 attention layers allocate KV cache slots.
- **Outcome:** A 262,144 context window (`-c 262144`) consumes only **8.25 GB** in `Q8_0` KV cache on Xavier, leaving **4.4 GB of free RAM headroom** and sustaining full **43+ tok/s** decode.

---

## 7. Multi-Modal Vision Projectors

When integrating vision encoders (e.g. Ornith, Qwen-VL mmproj):
- **Always retain BF16 or FP16 for the vision projector.**
- Vision projector weights are tiny (800-900 MB), representing <4% of total model memory.
- Quantizing the vision projector to 4-bit or 8-bit degrades optical character recognition (OCR) and token embedding alignment, with zero measurable speed gain during language generation.

---

## 8. Step-by-Step Decision Tree for Optimizing a New Model

Follow this sequence when picking up a new model on edge hardware:

```
[1. Baseline Audit]
  |-- Measure peak DRAM read bandwidth with stream/probe kernel.
  |-- Run 5 reps of prompt2.txt (NPRED=400) to get baseline tok/s, TTFT, and acceptance.
  `-- Record hardware clocks and memory footprint.
        |
[2. Architecture & Graph Audit]
  |-- Identify layer composition: Dense vs MoE, Attention vs DeltaNet/SSM, Vocab size.
  `-- Check GGUF tensor types and verify all overrides applied.
        |
[3. Kernel Attribution (nsys)]
  |-- Trace one decode pass (between consecutive verify launches).
  |-- Rank kernels by time: MoE GEMV, Dense GEMV, LM Head, Glue.
  `-- Check Tegra counters (lts__d_sectors_fill_sysmem) to determine bandwidth vs latency bound.
        |
[4. Dispatch & Launch Bounds Check]
  |-- Check ggml-cuda dispatch tables for architecture equality bugs (cc == 700 vs 720).
  |-- Audit registers/thread with NCU. If registers > 64 and blockDim=32, clamp launch bounds.
  `-- Verify batch handling operates on grid Z-dimension (blockIdx.z).
        |
[5. Dense Multi-Row GEMV & Carveout]
  |-- If multi-column verify is >15% of pass, evaluate 4 rows/block in mmvq.cu.
  `-- Raise shared memory carveout to 64 KB to maintain active block residency.
        |
[6. Host & Sampler Pipeline Offloading]
  |-- Check perf stat: If host task-clock > 15 ms/pass, host CPU is stalling GPU.
  |-- Enable backend sampling (-bs) and GPU draft offloading.
  `-- If vocab > 150k, test reduced draft vocabulary head (--spec-draft-vocab).
        |
[7. Speculative Engine & Prefill Tuning]
  |-- Tune draft depth (n-max 5..7) and confidence threshold (p-min 0.4..0.6).
  |-- Decouple prefill ubatch (-ub 1024/2048) from decode.
  `-- For extreme long context, deploy dynamic sparsity (TokenGate).
```
