#include "llama.h"
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>

static float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

int main(int argc, char ** argv) {
    const char * model_path = "/mnt/ssd/cache/huggingface/hub/models--mradermacher--zerank-2-GGUF/blobs/7b9ba05a0509151c911582a4d62b14003f6a4fafa0e7ccdf572c7598cde1c100";
    if (argc > 1) {
        model_path = argv[1];
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;

    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 8192;
    cparams.n_batch = 2048;
    cparams.n_ubatch = 1024;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        return 1;
    }

    const std::string query = "What is the capital of France and what are its major landmarks?";
    const std::vector<std::string> docs = {
        "Paris is the capital of France. Major landmarks include the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral.",
        "The recipe for boiling pasta takes 10 minutes. First boil salted water in a large pot, then add pasta and stir occasionally.",
        "Madrid is the capital of Spain. Famous landmarks include the Royal Palace, Prado Museum, and Plaza Mayor in the center.",
        "Berlin is the capital of Germany, known for the Brandenburg Gate, Museum Island, and the historic Berlin Wall memorial.",
        "Rome is the capital of Italy, renowned worldwide for the Colosseum, Vatican City, St. Peter's Basilica, and the Pantheon.",
        "Tokyo is the capital of Japan, featuring landmarks such as the Tokyo Skytree, Tokyo Tower, Senso-ji temple, and Shibuya Crossing.",
        "London is the capital of the United Kingdom, home to Big Ben, the Tower of London, Buckingham Palace, and the British Museum.",
        "Baking sourdough bread requires flour, water, salt, and active sourdough starter. Fermentation takes several hours.",
        "Lyon is a major city in France known as the gastronomic capital of France, featuring the Basilica of Notre-Dame de Fourviere.",
        "Marseille is the second largest city in France, situated on the Mediterranean coast with the famous Old Port and Notre-Dame de la Garde."
    };

    const llama_token yes_token = 9454;
    const size_t N = docs.size();

    printf("Benchmarking %zu documents...\n", N);

    // Warm-up run
    {
        llama_batch_ext * b = llama_batch_ext_init(ctx);
        llama_token tok = 151643;
        llama_batch_ext_add_token(b, 0, tok);
        llama_pos pos = 0;
        llama_batch_ext_set_pos(b, 0, &pos);
        llama_batch_ext_set_output_logits(b, 0, true);
        llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, b);
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_batch_ext_free(b);
    }

    // -------------------------------------------------------------
    // Method 1: Sequential evaluation (baseline)
    // -------------------------------------------------------------
    {
        llama_batch_ext * batch = llama_batch_ext_init(ctx);
        int total_tokens = 0;
        int64_t t_start = ggml_time_us();

        for (size_t i = 0; i < N; i++) {
            std::string prompt = "<|im_start|>system\n" + query + "<|im_end|>\n"
                               + "<|im_start|>user\n" + docs[i] + "<|im_end|>\n"
                               + "<|im_start|>assistant\n";

            std::vector<llama_token> tokens(prompt.size() + 16);
            int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), tokens.size(), false, false);
            tokens.resize(n_tokens);
            total_tokens += n_tokens;

            llama_memory_clear(llama_get_memory(ctx), true);

            llama_batch_ext_clear(batch);
            for (int t = 0; t < n_tokens; ++t) {
                int32_t idx = llama_batch_ext_add_token(batch, 0, tokens[t]);
                llama_pos pos = t;
                llama_batch_ext_set_pos(batch, idx, &pos);
            }
            llama_batch_ext_set_output_logits(batch, n_tokens - 1, true);

            llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch);

            float * logits = llama_get_logits_ith(ctx, -1);
            float score = sigmoid(logits[yes_token] / 5.0f);
            if (i < 3) {
                printf("  [Sequential] Doc %zu score: %.4f (logit: %+.2f)\n", i, score, logits[yes_token]);
            }
        }

        int64_t t_end = ggml_time_us();
        double elapsed_ms = (t_end - t_start) / 1000.0;
        printf("[Method 1: Sequential] Total time: %.2f ms (%zu docs, %d tokens, %.1f tok/s, %.2f ms/doc)\n\n",
               elapsed_ms, N, total_tokens, total_tokens * 1e6 / (t_end - t_start), elapsed_ms / N);

        llama_batch_ext_free(batch);
    }

    // -------------------------------------------------------------
    // Method 2: Prefix-Cached / Reused Query KV Cache
    // -------------------------------------------------------------
    {
        // 1. Evaluate query prefix once into sequence 0
        std::string query_prefix = "<|im_start|>system\n" + query + "<|im_end|>\n";
        std::vector<llama_token> q_tokens(query_prefix.size() + 16);
        int n_q_tokens = llama_tokenize(vocab, query_prefix.c_str(), query_prefix.size(), q_tokens.data(), q_tokens.size(), false, false);
        q_tokens.resize(n_q_tokens);

        llama_memory_clear(llama_get_memory(ctx), true);

        llama_batch_ext * batch = llama_batch_ext_init(ctx);
        llama_batch_ext_clear(batch);
        for (int t = 0; t < n_q_tokens; ++t) {
            int32_t idx = llama_batch_ext_add_token(batch, 0, q_tokens[t]);
            llama_pos pos = t;
            llama_batch_ext_set_pos(batch, idx, &pos);
        }
        llama_batch_ext_set_output_logits(batch, n_q_tokens - 1, false);

        int64_t t_prefix_start = ggml_time_us();
        llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch);
        int64_t t_prefix_end = ggml_time_us();

        // 2. For each document, copy/fork sequence 0 prefix to sequence 1, evaluate doc suffix
        int total_doc_tokens = 0;
        int64_t t_docs_start = ggml_time_us();

        for (size_t i = 0; i < N; i++) {
            std::string doc_suffix = "<|im_start|>user\n" + docs[i] + "<|im_end|>\n<|im_start|>assistant\n";
            std::vector<llama_token> d_tokens(doc_suffix.size() + 16);
            int n_d_tokens = llama_tokenize(vocab, doc_suffix.c_str(), doc_suffix.size(), d_tokens.data(), d_tokens.size(), false, false);
            d_tokens.resize(n_d_tokens);
            total_doc_tokens += n_d_tokens;

            // Copy seq 0 KV cache to seq 1
            llama_memory_seq_cp(llama_get_memory(ctx), 0, 1, 0, n_q_tokens);

            llama_batch_ext_clear(batch);
            for (int t = 0; t < n_d_tokens; ++t) {
                int32_t idx = llama_batch_ext_add_token(batch, 1, d_tokens[t]);
                llama_pos pos = n_q_tokens + t;
                llama_batch_ext_set_pos(batch, idx, &pos);
            }
            llama_batch_ext_set_output_logits(batch, n_d_tokens - 1, true);

            llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch);

            float * logits = llama_get_logits_ith(ctx, -1);
            float score = sigmoid(logits[yes_token] / 5.0f);
            if (i < 3) {
                printf("  [Prefix-Shared] Doc %zu score: %.4f (logit: %+.2f)\n", i, score, logits[yes_token]);
            }

            // Remove seq 1 cache
            llama_memory_seq_rm(llama_get_memory(ctx), 1, 0, -1);
        }

        int64_t t_docs_end = ggml_time_us();
        double prefix_ms = (t_prefix_end - t_prefix_start) / 1000.0;
        double docs_ms = (t_docs_end - t_docs_start) / 1000.0;
        double total_ms = prefix_ms + docs_ms;
        int total_tokens = n_q_tokens + total_doc_tokens;

        printf("[Method 2: Prefix-Shared] Total time: %.2f ms (Prefix: %.2f ms, Docs: %.2f ms, %.2f ms/doc, Effective %.1f tok/s)\n\n",
               total_ms, prefix_ms, docs_ms, total_ms / N, (n_q_tokens * N + total_doc_tokens) * 1e6 / (t_docs_end - t_prefix_start));

        llama_batch_ext_free(batch);
    }

    // -------------------------------------------------------------
    // Method 3: Parallel Multi-Sequence Batched Prefill
    // -------------------------------------------------------------
    {
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_batch_ext * batch = llama_batch_ext_init(ctx);
        llama_batch_ext_clear(batch);

        int total_tokens = 0;
        std::vector<int> last_token_indices;

        for (size_t i = 0; i < N; i++) {
            std::string prompt = "<|im_start|>system\n" + query + "<|im_end|>\n"
                               + "<|im_start|>user\n" + docs[i] + "<|im_end|>\n"
                               + "<|im_start|>assistant\n";

            std::vector<llama_token> tokens(prompt.size() + 16);
            int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), tokens.size(), false, false);
            tokens.resize(n_tokens);
            total_tokens += n_tokens;

            for (int t = 0; t < n_tokens; ++t) {
                int32_t idx = llama_batch_ext_add_token(batch, (llama_seq_id) i, tokens[t]);
                llama_pos pos = t;
                llama_batch_ext_set_pos(batch, idx, &pos);
                if (t == n_tokens - 1) {
                    llama_batch_ext_set_output_logits(batch, idx, true);
                    last_token_indices.push_back(idx);
                }
            }
        }

        int64_t t_start = ggml_time_us();
        llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch);
        int64_t t_end = ggml_time_us();

        double elapsed_ms = (t_end - t_start) / 1000.0;
        for (size_t i = 0; i < std::min((size_t)3, N); i++) {
            float * logits = llama_get_logits_ith(ctx, last_token_indices[i]);
            float score = sigmoid(logits[yes_token] / 5.0f);
            printf("  [Batched] Doc %zu score: %.4f (logit: %+.2f)\n", i, score, logits[yes_token]);
        }

        printf("[Method 3: Batched Parallel] Total time: %.2f ms (%zu docs in single batch, %d tokens, %.1f tok/s, %.2f ms/doc)\n\n",
               elapsed_ms, N, total_tokens, total_tokens * 1e6 / (t_end - t_start), elapsed_ms / N);

        llama_batch_ext_free(batch);
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
