# zerank-2 Generative Causal Reranker: Architecture, CUDA Optimization, and Edge Evaluation on Jetson AGX Xavier

**Date:** 2026-11-10  
**Target:** `zeroentropy/zerank-2-reranker` (4.0B dense Qwen3 architecture, Q8_0 GGUF, 36 layers, `n_embd = 2560`)  
**Hardware:** NVIDIA Jetson AGX Xavier (sm_72 Volta / cc 720, CUDA 11.4, 30,991 MiB unified LPDDR4x)  
**Binary / Fork:** `ftulabs/momonga-llama.cpp` (`build-volta`)  

---

## 1. Executive Summary

Stock `llama.cpp` completely fails to serve modern generative causal cross-encoder rerankers like ZeroEntropy's `zerank-2`:
1. It defaults to BERT formatting (`[BOS] query [EOS] doc [EOS]`), causing semantic failure.
2. In graph pooling (`LLAMA_POOLING_TYPE_RANK`), when no classification head tensor (`cls_out`) exists, it executes a softmax across the entire raw 2560-dimensional hidden state and returns `embd[0]` (~`1e-31`), producing completely bogus relevance scores.
3. If evaluated via next-token prediction, it materializes the full 151,936-dimensional vocabulary projection matrix (`output.weight`), reading ~400 MB of DRAM and allocating 641 MiB of compute buffer per pass.
4. On NVIDIA Jetson AGX Xavier (`sm_72`), Volta Tensor Core instructions (`VOLTA_MMA_AVAILABLE`, `volta_mma_available`) and specialized FlashAttention GQA kernels were locked out due to exact equality checks (`cc == 700` failing on Xavier's `cc = 720`).

We engineered, verified, and benchmarked an end-to-end native implementation in `momonga-llama.cpp`:
- **Direct Next-Token Logit Projection:** Slices row `9454` (`"Yes"`) directly from `token_embd` / `output` weights via a zero-copy 2D view (`[2560, 1]`), bypassing 151,936 rows. This eliminates ~400 MB of weight DRAM reads per pass and drops head compute from milliseconds to sub-microsecond dot products.
- **sm_72 Volta Hardware Unlocking:** Fixed all CC equality gates across `common.cuh`, `fattn.cu`, `mmvq.cu`, and `ggml-cuda.cu`, activating Volta Tensor Core HMMA paths and specialized GQA flash attention.
- **Server Reranking Integration:** Auto-detects `zerank-2`, applies the official ZeroEntropy prompt template (`<|im_start|>system\n{query}<|im_end|>\n<|im_start|>user\n{document}<|im_end|>\n<|im_start|>assistant\n`), sets `task.index`, and applies canonical probability calibration:
  $$\text{score} = \sigma(\text{logit}[9454] / 5.0) = \frac{1}{1 + e^{-\text{logit}/5.0}}$$
- **Wide Logical Batching:** Removed arbitrary `n_batch = n_ubatch` clamping for causal rerankers in `server.cpp` and `common.cpp`.

### Measured Results on Jetson AGX Xavier (Q8_0 4.0B Model)
- **Top-1 Accuracy:** **100.0%** (12/12 test queries across Geography, CS, Geology, and Physics).
- **Latency:** **177.87 ms** median per document (1.78 s for a 10-document batch containing ~400 tokens).
- **Throughput:** **5.61 docs/sec** and **223.7 tokens/sec** end-to-end through HTTP `/rerank`.
- **Edge Economic Objective (arXiv:2611.06841):** **28.73 W** average power, **7.80 Tokens/Joule**, **0.195 Docs/Joule**.

---

## 2. Root Cause Analysis

### A. The Bogus Score Bug
In upstream `llama.cpp` (`src/llama-graph.cpp:4495`):
```cpp
case LLAMA_POOLING_TYPE_RANK:
    if (cls_out) {
        cur = ggml_mul_mat(ctx0, cls_out, cur);
        if (cls_out_b) {
            cur = ggml_add(ctx0, cur, cls_out_b);
        }
    }
    // softmax for qwen3 reranker
    if (arch == LLM_ARCH_QWEN3 || arch == LLM_ARCH_QWEN3VL) {
        cur = ggml_soft_max(ctx0, cur);
    }
```
`zerank-2` is a generative causal cross-encoder with tied embeddings. It has no `cls_out` tensor.
Consequently:
1. `cls_out` is `nullptr`, so `cur` remains the raw 2560-dimensional pooled hidden vector.
2. The Qwen3 check unconditionally executed `ggml_soft_max` over all 2560 dimensions of the hidden state.
3. In `tools/server/server-context.cpp:3033`, the server read `embd[0]`, which was the normalized softmax probability of element 0 of the hidden state (~`1e-31`).

### B. Prompt Formatting Fallback
In `tools/server/server-common.cpp:1995`, `format_prompt_rerank` queried `llama_model_chat_template(model, "rerank")`. Because `zerank-2` GGUF files contain only `tokenizer.chat_template` without the `.rerank` key, it fell back to BERT format (`[BOS] query [EOS] doc [EOS]`). `zerank-2` was trained specifically on ChatML tokens (`<|im_start|>` and `<|im_end|>`).

### C. sm_72 Volta Lockout
In `ggml/src/ggml-cuda/common.cuh`:
```cpp
#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ == GGML_CUDA_CC_VOLTA
#define VOLTA_MMA_AVAILABLE
#endif
```
When compiling for compute capability 7.2 (`-arch=compute_72`), nvcc defines `__CUDA_ARCH__ = 720`. Because `GGML_CUDA_CC_VOLTA = 700`, `__CUDA_ARCH__ == GGML_CUDA_CC_VOLTA` evaluated to false. As a result, `VOLTA_MMA_AVAILABLE` was never defined, locking Xavier out of Tensor Core FlashAttention (`fattn-mma-f16.cuh`) and MMF kernels. Similar bugs were present in `fattn.cu:244`, `mmvq.cu:382`, and `ggml-cuda.cu:1760`.

---

## 3. Implementation Details

### A. Target Token Logit View Projection (`src/llama-graph.cpp`)
In `llm_graph_context::build_pooling`, we implemented the targeted row view projection:
```cpp
} else if (hparams.rerank_token_id >= 0) {
    // Causal generative reranker: project using only target token row (9454 for "Yes")
    // Avoids 151,936x full vocab projection (~400 MB DRAM read and 640 MiB buffer per pass)
    ggml_tensor * head_w = model.output ? model.output : model.tok_embd;
    GGML_ASSERT(head_w != nullptr);
    const size_t offset = (size_t)hparams.rerank_token_id * head_w->nb[1];
    ggml_tensor * row_view = ggml_view_2d(ctx0, head_w, hparams.n_embd, 1, head_w->nb[1], offset);
    row_view->buffer = head_w->buffer;
    if (head_w->data) {
        row_view->data = (char *)head_w->data + offset;
    }
    cur = ggml_mul_mat(ctx0, row_view, cur);
}
```
This reduces the output dimension from `[151936, n_seqs]` to `[1, n_seqs]`. We also guarded `ggml_soft_max` with `cls_out != nullptr && hparams.n_cls_out > 1`.

### B. Hardware Architecture Range Gates (`ggml/src/ggml-cuda/`)
All Volta equality checks were broadened to include sm_72:
- `common.cuh`: `__CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA && __CUDA_ARCH__ < GGML_CUDA_CC_TURING`
- `common.cuh`: `volta_mma_available(cc)` returns true for `arch >= 700 && arch < 750`.
- `fattn.cu:244`: Enables Volta GQA switch for sm_72 (`gqa_ratio % 4 == 0`).
- `mmvq.cu:382`: Volta batch limits applied for sm_72.
- `ggml-cuda.cu:1760`: Direct F32 output without intermediate allocation enabled for sm_72.

### C. Server Chat Template & Probability Calibration
In `src/llama-model.cpp`:
- `load_hparams` detects `zerank` in `general.name` or `general.basename` and sets `hparams.rerank_token_id = 9454` and `hparams.n_cls_out = 1`.
- `llama_model_chat_template` returns `<|im_start|>system\n{query}<|im_end|>\n<|im_start|>user\n{document}<|im_end|>\n<|im_start|>assistant\n` when queried for `"rerank"`.
- `send_rerank` in `tools/server/server-context.cpp` computes calibrated probability:
  `res->score = 1.0f / (1.0f + expf(-raw_score / 5.0f));`
- `handle_rerank_impl` records `task.index = (int)i;` so responses correctly track document indices.

---

## 4. Empirical Evaluation

### A. Ground-Truth Numerical Verification
Verified using `/home/triph/juicer/momonga-llama.cpp/tools/test_zerank2_scoring.cpp` against the official Q8_0 model:

| Document | Target Text | Raw Logit | Calibrated Score | Rank |
|---|---|---|---|---|
| Doc 0 | "Paris is the capital of France." | +23.20 | 0.9904 | 1 (Match) |
| Doc 2 | "Madrid is the capital of Spain." | +3.18 | 0.6539 | 2 (Related) |
| Doc 1 | "The recipe for boiling pasta takes 10 minutes." | -0.25 | 0.4874 | 3 (Irrelevant) |

### B. Server Configuration Sweep (A/B Testing)
Evaluated across 3 rounds of 4 diverse multi-document queries (120 document evaluations total per configuration), with continuous real-time power sampling via `tegrastats`:

| Configuration | Concurrency (`-np`) | Microbatch (`-ub`) | Mean Req Latency | Median Doc Latency | Token Throughput | Power | Energy Eff. (Tokens/J) |
|---|---|---|---|---|---|---|---|
| **Best: np4-ub1024** | **4** | **1024** | **1,780.88 ms** | **177.87 ms/doc** | **223.68 tok/s** | **28.73 W** | **7.80 tok/J** |
| np2-ub1024 | 2 | 1024 | 1,785.92 ms | 178.03 ms/doc | 223.06 tok/s | 28.88 W | 7.73 tok/J |
| np4-ub2048 | 4 | 2048 | 1,786.81 ms | 178.21 ms/doc | 222.93 tok/s | 28.72 W | 7.77 tok/J |

### C. Operational KV Cache Pinning on Jetson
When launching `llama-server`, setting `-c 8192` is mandatory on models with large training contexts (`n_ctx_train = 40960`). Without `-c`, `server.cpp` auto-sizes the unified KV pool to `n_parallel * n_ctx_train = 8 * 40960 = 327,680` tokens, requiring 21.9 GB of VRAM for KV buffers alone and crashing with `NvMapMemAllocInternalTagged error 12 (OOM)`. Pinned `-c 8192` allocates only 1.2 GB of KV cache, leaving 25+ GB of free memory.

---

## 5. Verified Best Configuration

Reproduced by `bench/run-rerank.sh`:
```bash
./momonga-llama.cpp/build-volta/bin/llama-server \
  -m /mnt/ssd/cache/huggingface/hub/models--mradermacher--zerank-2-GGUF/blobs/7b9ba05a0509151c911582a4d62b14003f6a4fafa0e7ccdf572c7598cde1c100 \
  --reranking \
  -fa on \
  -ngl 99 \
  -c 8192 \
  -np 4 \
  -ub 1024 \
  -b 2048 \
  --port 8012 \
  --host 0.0.0.0
```
