# momonga-llama.cpp

A high-performance fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) optimized
for **Jetson AGX Xavier (Volta sm_72, CUDA 11.4)**. Everything outside `ggml/src/ggml-cuda/`,
`common/`, `src/`, and `tools/` remains synchronized with upstream.

This fork specializes in hybrid MoE architectures (such as **Cyber-Tiel-Coder-35B-A3B** and
**Qwen3.6-35B-A3B** with 30 Gated DeltaNet linear attention layers and 11 full attention layers),
delivering:
- **43.08 - 43.97 tok/s** speculative decode median (**47.5 - 48.4 tok/s** steady-state generation) via factory Multi-Token Prediction (MTP) with 0.968 - 0.975 acceptance.
- **2,025 - 2,070 tok/s prefill** on long contexts via TokenGate layer-token pruning with decoupled prefill micro-batches and FlashAttention approximation (collapsing TTFT on 9.6k tokens from 53.3s to 4.79s, with 100% exact needle retrieval).
- **Full 262,144 context (`-c 262144`)** supported out of the box with `q8_0` KV cache, leaving **4.4 GB of free RAM headroom** on the 32 GB unified memory Xavier.

The parent project is [llama.cpp](https://github.com/ggml-org/llama.cpp); its upstream documentation applies unchanged except where documented here. Detailed architectural analysis is in [JETSON-XAVIER.md](JETSON-XAVIER.md), and the complete experimental record is in [FINDINGS.md](FINDINGS.md) and [findings/](findings/).

---

## What is different

```sh
git log --oneline upstream/master..HEAD
```

| Commit / Change | Effect |
|---|---|
| `cuda : use MMVQ for MUL_MAT_ID on sm_72 and drop the per-type batch cap` | Compares CC as range (`cc >= GGML_CUDA_CC_VOLTA && cc < GGML_CUDA_CC_TURING`), moving token index to `blockIdx.z`. Keeps MoE matmuls on the MMVQ kernel for batches above 4-5 instead of falling through to MMQ (2.5x slower). Eliminates launch-bounds constraints for arbitrary speculative batch sizes. |
| `cuda : run the IQ3_S MoE kernel at 64 registers instead of 79` | Declares exact one-warp launch bounds (`__launch_bounds__(32, 32)`) for IQ3_S, dropping register pressure from 79 to 64 and raising occupancy from 37.5% to 47.9% (+2.25% end-to-end). |
| `cuda : four rows per block and a wider small_k trigger on the dense MMVQ path` | Assigns 4 rows per block on dense MMVQ, adds 64 KB shared-memory carveout, and widens `small_k` triggers. Cuts L1 activation re-read traffic by 24-38% (+4.3% end-to-end). |
| `cuda : pack destination rows per block in non-contiguous concat` | Eliminates wasted thread iterations in DeltaNet state concatenation (+0.29%). |
| `cuda : pick the MoE prefill tile width from tokens per expert` | Tightens MMQ prefill tile width `J` from average tokens per expert rather than `ncols_opt`, eliminating discarded dp4a work during prefill (+12% prefill throughput). |
| `speculative : single-sync sampling (PR #30196) and GPU draft sampler offload` | Performs draft chain sampling directly on the GPU without host-side token loops or CPU-GPU synchronization roundtrips. Eliminates 7.8 ms/pass of host idle latency. |
| `llama : KV-only graphs for MTP draft catch-up (PR #31750)` | Uses dedicated KV-only computation graphs for speculative verification, skipping unnecessary output head evaluation during catch-up passes. |
| `llama : --spec-draft-vocab for reduced draft LM head` | Pins MTP draft projection to active vocabulary subsets, reducing memory traffic on wide vocabularies. |
| `cuda : enable CUB on CUDA 11.4 (sm_72) for native GPU argsort` | Activates NVIDIA CUB hardware radix sort routines on Volta, enabling fully offloaded `--backend-sampling` (`-bs`) for top-p and top-k without host fallback. |
| `tokengate : extreme long-context prefill with decoupled ubatch and FA approximation` | Implements TokenGate layer-token pruning (`--tokengate --tokengate-fa`). Decouples prefill micro-batch size (`TG_UBATCH=512`) from speculative decode (`-b 32 -ub 32`), achieving **>2,000 tok/s prefill** without degrading speculative decode acceptance. |
| `cuda : opt-in per-op timing dump behind GGML_CUDA_OP_TIME` | Debug instrumentation to inspect per-node GPU kernel execution times without external profilers (off by default). |

---

## Measured Performance

All numbers measured on NVIDIA Jetson AGX Xavier (Volta sm_72, CC 720, 30,991 MiB unified memory, 8-core Carmel CPU, CUDA 11.4):

### Speculative Generation (Decode)

Evaluated on `Cyber-Tiel-Coder-35B-A3B-tg.gguf` (17.3 GB NVMe SSD), greedy sampling, 5 runs median:

| Configuration | Generation Throughput | Draft Acceptance | Mean Draft Length |
|---|---|---|---|
| Stock llama.cpp (`n-max 3`, best stock) | 31.50 tok/s | 0.989 | 3.69 |
| Fork dispatch fix (`n-max 6`) | 34.30 - 34.49 tok/s | 0.967 | 5.68 |
| + 64-register MoE launch bounds | 35.07 tok/s | 0.967 | 5.68 |
| + Dense MMVQ 4 rows/block & 64 KB carveout | 36.56 tok/s | 0.967 | 5.68 |
| + Non-contiguous concat & `-bs` | 36.93 tok/s | 0.967 | 5.68 |
| **Current Best (`run-best.sh`, GPU sampler offload + CUB + TokenGate)** | **43.08 - 43.97 tok/s** | **0.968 - 0.975** | **5.53 - 5.75** |
| *Steady-state generation (`tg_3s`, post-`<think>` tokens)* | **47.5 - 48.4 tok/s** | **0.975 - 0.980** | **5.80 - 5.92** |
| Speculative decode at **262,144 context** (`-c 262144`) | **43.08 tok/s** | **0.968** | **5.53** |
| No speculation (plain decode, `all-q4k` file, `-bs`) | 27.86 tok/s (23.75 at 8.8k ctx) | - | - |

### Long-Context Document Prefill (TTFT)

Evaluated on real text prompts across context lengths (measured client wire TTFT, cache off):

| Prompt Context | Baseline (`-ub 32`) | Decoupled TokenGate (`--tokengate-fa`) | Speedup | Client Wire TTFT | Needle Accuracy |
|---|---|---|---|---|---|
| 2,385 tokens | 84.3 tok/s | **805 tok/s** | 9.5x | **2.98 s** | 100% |
| 9,562 tokens | 79.5 tok/s | **2,070.6 tok/s** | 26.0x | **4.79 s** | 100% |
| 22,128 tokens | ~75 tok/s | **2,025.0 tok/s** | 27.0x | **11.15 s** | 100% |
| High-Accuracy Needle Mode ($K=2000, W=2$) | 79.5 tok/s | **1,715.2 tok/s** | 21.5x | **14.39 s** | 100% (`BLUE-VELVET-99`) |

---

## Quick Start

### 1. Build

Run the automated build script:

```sh
./build-best.sh
```

Or configure manually via CMake:

```sh
cmake -S . -B build-volta -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=72 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.4/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10.3.0 \
  -DCMAKE_CUDA_FLAGS="-Xcompiler=-isystem,/usr/include/c++/10.3.0,-isystem,/usr/include/c++/10.3.0/aarch64-poky-linux,-isystem,/usr/include/cuda-compat-workarounds" \
  -DGGML_CUDA_NO_VMM=ON -DGGML_CUDA_NO_PEER_COPY=ON \
  -DGGML_CUDA_FA=ON -DGGML_CUDA_GRAPHS=ON -DGGML_NATIVE=ON \
  -DGGML_LTO=OFF -DLLAMA_BUILD_TESTS=OFF

cmake --build build-volta -j8 --target llama-server llama-cli llama-quantize llama-tokengate
```

*Note:* `CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10.3.0` and the `-isystem` flags are required because CUDA 11.4's nvcc rejects the system gcc without them. `GGML_CUDA_NO_VMM` is required on Xavier's integrated GPU.

### 2. Run

Run the verified production server with startup CUDA graph warm-up:

```sh
./run-best.sh
```

Or run directly with the complete production command line:

```sh
env -u GGML_CUDA_DISABLE_GRAPHS \
./build-volta/bin/llama-server \
  -m /mnt/ssd/models/Cyber-Tiel-Coder-35B-A3B-tg.gguf \
  -c 262144 -fa on -ngl 99 -ncmoe 0 -t 2 -tb 2 \
  -ctk q8_0 -ctv q8_0 -ub 32 -b 32 --poll 0 --parallel 1 \
  -bs \
  --tokengate --tokengate-k 2000 --tokengate-dilate 2 --tokengate-min-span 1024 --tokengate-fa \
  --spec-type draft-mtp --spec-draft-n-max 6 --spec-draft-n-min 0 \
  --spec-draft-p-min 0.5 --spec-draft-p-split 0.1 \
  --spec-draft-vocab /mnt/ssd/models/draft-vocab-32k.json \
  --load-mode none --port 8080
```

---

## Repository Documentation

All experimental records, hardware investigations, and configurations are maintained directly inside this repository:

- [KNOWN-GOOD.md](KNOWN-GOOD.md): The single source of truth for the validated best configuration.
- [JETSON-XAVIER.md](JETSON-XAVIER.md): Comprehensive deep dive into the sm_72 CUDA kernels and architectural changes.
- [FINDINGS.md](FINDINGS.md): Executive summary and timeline of every experiment.
- [findings/](findings/): Complete archive of 55+ detailed, dated measurement logs and microbenchmarks.
- [TODO.md](TODO.md): Task list, roadmap, and the "Closed - do not re-test" register.
- [HANDOFF.md](HANDOFF.md): Engineering handoff notes, machine quirks, and profiling instructions.
- [AGENTS.md](AGENTS.md): Guidelines and discipline rules for AI agents and contributors working in this repository.

---

## Keeping up with upstream

The remote layout is `origin` = this fork, `upstream` = `ggml-org/llama.cpp`:

```sh
git fetch upstream
git rebase upstream/master
git push origin master    # fast-forward
```

Documentation files (`README.md`, `AGENTS.md`, `JETSON-XAVIER.md`, `KNOWN-GOOD.md`, `FINDINGS.md`, `TODO.md`, `HANDOFF.md`, and `findings/`) are fork-owned. On rebase conflicts, preserve this fork's documentation (`git checkout --ours <doc>`).

---

## License

MIT, matching upstream llama.cpp. See [LICENSE](LICENSE).
