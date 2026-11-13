#!/bin/bash
# ==============================================================================
# Script Name: list-models.sh
# Description: Print summary table of verified best model runners on Jetson AGX Xavier.
# ==============================================================================
cat << 'EOF'
====================================================================================================
 MOMONGA LLAMA.CPP - JETSON AGX XAVIER VERIFIED RUNNERS (sm_72 Volta, 30 GB Unified Memory)
====================================================================================================

Script                     Port  Context   Arch & Model                      Task / Key Feature
----------------------------------------------------------------------------------------------------
run-bge-reranker-v2-m3.sh  8012  8,192     XLM-RoBERTa 568M Q8_0             Encoder Reranker (2.83s / 10.7k tok, 3,776 tok/s)
run-zerank-2.sh            8012  8,192     Qwen3 4B Q8_0                     Causal Reranker (177ms/doc, 7.8 tok/J)
run-cybertiel-35b-coder-mtp.sh 8080 32,768 Cyber-Tiel-Coder-35B MoE + Vision Speculative MTP 6 (48 tok/s, 2070 tok/s prefill)
run-cybertiel-35b-120k.sh  8800  120,000   Cyber-Tiel-35B Q4KM + LoRA + Vis  120k Long Context Chat (28 tok/s flat)
run-qwen36-35b-a3b.sh      8081  32,768    Qwen 3.6 35B A3B + LoRA + Vision  Base MoE Coder with TokenGate
run-qwen38-27b.sh          8083  16,384    Qwen 3.8 27B Dense Coder Q4_K_M   Dense 27B Coder (64 layers, non-MoE)
run-deepseek-v4-9b.sh      8084  16,384    DeepSeek-V4-Pro 9B Q8_0           Dense 9B Zero-Loss Reasoning
run-spark-4b.sh            8085  16,384    Spark-X2.6 4B Q8_0                Ultra-Fast Edge Model (<80ms/tok)
run-embeddinggemma-2.sh    8086  8,192     EmbeddingGemma-2 Q8_0 + Vision    Multimodal Vector Embeddings

====================================================================================================
USAGE EXAMPLES:
  cd runbest
  ./run-bge-reranker-v2-m3.sh    # Start fast encoder reranker on port 8012
  ./run-zerank-2.sh              # Start causal reranker on port 8012
  ./run-cybertiel-35b-coder-mtp.sh # Start speculative MTP coding assistant on port 8080
  ./run-cybertiel-35b-120k.sh    # Start 120,000 token context production server on port 8800
  ./run-embeddinggemma-2.sh      # Start multimodal embedding server on port 8086

  Any script accepts a context length override as the first argument, e.g.:
    ./run-cybertiel-35b-coder-mtp.sh 262144

  Any script accepts environment overrides:
    PORT=9000 WARMUP=0 ./run-spark-4b.sh
====================================================================================================
EOF
