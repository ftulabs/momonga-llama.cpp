# Verified Best Run Scripts (Jetson AGX Xavier)

This directory (`runbest/`) contains turnkey, self-contained, hardware-tuned launch scripts for each model and task profile available on the NVIDIA Jetson AGX Xavier (30 GB unified memory, sm_72 Volta).

Each script embeds verified hardware flags discovered and validated through empirical profiling (A/B testing, nsys/ncu traces, ServerBench latency-qualified scoring).

---

## 1. Quick Reference & Port Matrix

| Script | Default Port | Model Arch | Quant / File | Task Profile | Headline Measured Performance |
|---|---|---|---|---|---|
| [`run-zerank-2.sh`](run-zerank-2.sh) | `8012` | Qwen3 4.0B Causal | Q8_0 (4.28 GB) | Generative Reranker | **177.87 ms/doc**, 223.7 tok/s, 7.80 tok/J |
| [`run-cybertiel-35b-coder-mtp.sh`](run-cybertiel-35b-coder-mtp.sh) | `8080` | MoE 35B A3B + Vision | IQ4_XS hybrid (18.1 GB) | Speculative MTP Coder | **43.97 tok/s** decode (48 tok/s tg_3s), 2,070 tok/s prefill |
| [`run-cybertiel-35b-120k.sh`](run-cybertiel-35b-120k.sh) | `8800` | MoE 35B A3B + LoRA + Vision | Q4_K_M (20.5 GB) | 120k Context Production Chat | **27.86 - 28.7 tok/s** flat decode, 215+ tok/s prefill |
| [`run-qwen36-35b-a3b.sh`](run-qwen36-35b-a3b.sh) | `8081` | Qwen 3.6 35B MoE + Vision | IQ4_XS (17.2 GB) | General MoE Chat | ~34.5 tok/s decode, TokenGate long-context |
| [`run-qwen38-27b.sh`](run-qwen38-27b.sh) | `8083` | Qwen 3.8 27B Dense | Q4_K_M (18.5 GB) | Dense Coder (64 layers) | Full GPU offloading, 0 graph reallocs |
| [`run-deepseek-v4-9b.sh`](run-deepseek-v4-9b.sh) | `8084` | Qwen 3.5 9B Dense | Q8_0 (9.79 GB) | High-Precision Reasoning | Full Q8_0 lossless weights, 16k context |
| [`run-spark-4b.sh`](run-spark-4b.sh) | `8085` | Spark 4.0B Dense | Q8_0 (4.38 GB) | Ultra-Fast Edge LLM | Sub-80ms decode latency, <5 GB VRAM |
| [`run-embeddinggemma-2.sh`](run-embeddinggemma-2.sh) | `8086` | EmbeddingGemma-2 + Vision | Q8_0 (310 MB) | Multimodal Embedding | Text + image vector search (/v1/embeddings) |

To list all available runners in terminal:
```bash
./runbest/list-models.sh
```

---

## 2. Common Usage Conventions

All scripts adhere to a uniform, robust convention:

1. **Context Override:** Pass desired context tokens as argument 1:
   ```bash
   ./runbest/run-cybertiel-35b-coder-mtp.sh 262144   # launch with 262k context
   ./runbest/run-zerank-2.sh 16384                 # launch reranker with 16k context
   ```
2. **Environment Overrides:**
   - `PORT`: Set listening port (e.g. `PORT=9000 ./runbest/run-spark-4b.sh`).
   - `HOST`: Set bind interface (default `0.0.0.0`).
   - `WARMUP`: `1` by default. Performs 1-request warm-up to eliminate cold CUDA graph capture overhead. Set `WARMUP=0` to skip.
   - `MODEL`: Override the path to the model GGUF file.
3. **Clean Shutdown:**
   - Press `Ctrl+C` or send `SIGTERM` / `SIGINT`. Traps guarantee clean server shutdown without leaving orphaned processes or leaked unified memory.
4. **CUDA Graphs:**
   - All scripts ensure graphs are enabled by unsetting `GGML_CUDA_DISABLE_GRAPHS` (as presence of the env var, regardless of value, disables graphs in ggml-cuda).

---

## 3. Detailed Model Profiles & Verification Commands

### 3.1. zerank-2 Generative Causal Reranker (`run-zerank-2.sh`)
- **Port:** `8012`
- **Context:** Pinned `-c 8192` (Prevents default auto-allocation of 48 GB KV pool).
- **Architecture:** Dense 4.0B Qwen3 base.
- **Hardware Enhancements:**
  - Zero-copy target token (`"Yes"`, id `9454`) row-view projection via `ggml_view_2d`.
  - Bypasses 151,936-element full vocabulary projection, saving ~400 MB DRAM read and 640 MiB buffer allocation per pass.
  - Volta sm_72 Tensor Core MMA instructions enabled (`VOLTA_MMA_AVAILABLE`).
  - Calibrated sigmoid probability scoring: $\sigma(\text{logit} / 5.0)$.
- **Test Request:**
  ```bash
  curl -s -X POST http://127.0.0.1:8012/rerank \
    -H "Content-Type: application/json" \
    -d '{
      "query": "What is the capital of France?",
      "documents": [
        "The capital of France is Paris.",
        "Berlin is the capital of Germany.",
        "Mount Everest is the highest mountain."
      ]
    }' | jq .
  ```

---

### 3.2. Cyber-Tiel-Coder-35B-A3B Speculative MTP (`run-cybertiel-35b-coder-mtp.sh`)
- **Port:** `8080`
- **Context:** `-c 32768` default (Full `262144` context uses only 8.25 GB KV cache due to hybrid 11 Attention + 30 DeltaNet architecture).
- **Hardware Enhancements:**
  - Multi-Token Prediction (MTP) depth 6 (`--spec-draft-n-max 6`).
  - KV-only MTP catch-up (`ba3e503`).
  - Trimmed draft vocabulary LM head (`6ec6635`, top 32k tokens).
  - TokenGate auto-spanning with FlashAttention approximation (`--tokengate-fa`).
  - Decoupled prefill ubatch (`TG_UBATCH=1024`).
- **Test Request:**
  ```bash
  curl -s -X POST http://127.0.0.1:8080/completion \
    -H "Content-Type: application/json" \
    -d '{
      "prompt": "def quick_sort(arr):",
      "n_predict": 128
    }' | jq -r .content
  ```

---

### 3.3. Cyber-Tiel-35B-A3B 120k Long Context (`run-cybertiel-35b-120k.sh`)
- **Port:** `8800`
- **Context:** `-c 120000` (120,000 tokens).
- **Hardware Enhancements:**
  - Q8_0 lossless KV cache (`-ctk q8_0 -ctv q8_0`).
  - Decoupled physical microbatch (`-ub 1024 -b 2048`).
  - MoE expert GPU offloading (`-ncmoe 0 -ngl 99`).
  - Vision projector (`mmproj-Ornith-1.5-35B-BF16.gguf`) and fine-tuning LoRA adapter active.
- **Test Request:**
  ```bash
  curl -s -X POST http://127.0.0.1:8800/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "Cyber Tiel 35B A3B 120k",
      "messages": [{"role": "user", "content": "Hello! What model are you?"}],
      "max_tokens": 64
    }' | jq -r '.choices[0].message.content'
  ```

---

### 3.4. Qwen 3.6 35B A3B Base MoE (`run-qwen36-35b-a3b.sh`)
- **Port:** `8081`
- **Context:** `-c 32768`.
- **Hardware Enhancements:**
  - TokenGate auto-spanning long-context prefill.
  - LoRA fine-tuning adapter and vision mmproj projector.
- **Test Request:**
  ```bash
  curl -s -X POST http://127.0.0.1:8081/completion \
    -H "Content-Type: application/json" \
    -d '{"prompt": "The primary color of the clear sky is", "n_predict": 32}' | jq -r .content
  ```

---

### 3.5. Qwen 3.8 27B Dense Coder (`run-qwen38-27b.sh`)
- **Port:** `8083`
- **Context:** `-c 16384`.
- **Hardware Enhancements:**
  - 64-layer dense architecture with full GPU offloading into 18.5 GB memory.
  - Host sampling offloaded to GPU (`-bs`).
- **Test Request:**
  ```bash
  curl -s -X POST http://127.0.0.1:8083/completion \
    -H "Content-Type: application/json" \
    -d '{"prompt": "def bubble_sort(arr):", "n_predict": 64}' | jq -r .content
  ```

---

### 3.6. DeepSeek-V4-Pro 9B Q8_0 (`run-deepseek-v4-9b.sh`)
- **Port:** `8084`
- **Context:** `-c 16384`.
- **Hardware Enhancements:**
  - Full Q8_0 uncompressed quantization for zero-loss reasoning precision.
  - 9.79 GB weights fit with ~16 GB free RAM headroom.
- **Test Request:**
  ```bash
  curl -s -X POST http://127.0.0.1:8084/completion \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Prove that the sum of two even numbers is always even:", "n_predict": 64}' | jq -r .content
  ```

---

### 3.7. Spark-X2.6 4B Edge LLM (`run-spark-4b.sh`)
- **Port:** `8085`
- **Context:** `-c 16384`.
- **Hardware Enhancements:**
  - Sub-80ms decode latency, ideal for ultra-low-latency real-time agents.
- **Test Request:**
  ```bash
  curl -s -X POST http://127.0.0.1:8085/completion \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Write a 3-line poem about robotics:", "n_predict": 64}' | jq -r .content
  ```

---

### 3.8. EmbeddingGemma-2 Multimodal Embedding (`run-embeddinggemma-2.sh`)
- **Port:** `8086`
- **Context:** `-c 4096`.
- **Hardware Enhancements:**
  - Dedicated `--embedding` mode with vision projection companion.
  - High-throughput batching (`-ub 2048 -b 2048`).
- **Test Request:**
  ```bash
  curl -s -X POST http://127.0.0.1:8086/v1/embeddings \
    -H "Content-Type: application/json" \
    -d '{"input": "Deep learning on NVIDIA Jetson Xavier"}' | jq .data[0].embedding[:5]
  ```

---

## 4. Operational Guidelines on Jetson Xavier

1. **System Memory:** The Jetson AGX Xavier has 30 GB of unified LPDDR4x memory shared between CPU and GPU. Only run one large model (>=18 GB) at a time.
2. **Background Services:**
   Check that background servers are stopped before launching a new benchmark or runner:
   ```bash
   sudo systemctl status llama-server
   ```
   If active:
   ```bash
   sudo systemctl stop llama-server
   ```
3. **Checking Running Servers:**
   ```bash
   ps aux | grep llama-server
   ```
