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
    cparams.n_ctx = 4096;
    cparams.n_batch = 2048;
    cparams.n_ubatch = 1024;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        return 1;
    }

    const std::string query = "What is the capital of France?";
    const std::vector<std::string> docs = {
        "Paris is the capital of France.",
        "The recipe for boiling pasta takes 10 minutes.",
        "Madrid is the capital of Spain."
    };

    const llama_token yes_token = 9454;
    llama_batch_ext * batch = llama_batch_ext_init(ctx);

    for (size_t i = 0; i < docs.size(); i++) {
        std::string prompt = "<|im_start|>system\n" + query + "<|im_end|>\n"
                           + "<|im_start|>user\n" + docs[i] + "<|im_end|>\n"
                           + "<|im_start|>assistant\n";

        std::vector<llama_token> tokens(prompt.size() + 16);
        int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), tokens.size(), false, false);
        tokens.resize(n_tokens);

        llama_memory_clear(llama_get_memory(ctx), true);

        llama_batch_ext_clear(batch);
        for (int t = 0; t < n_tokens; ++t) {
            int32_t idx = llama_batch_ext_add_token(batch, 0, tokens[t]);
            llama_pos pos = t;
            llama_batch_ext_set_pos(batch, idx, &pos);
        }
        llama_batch_ext_set_output_logits(batch, n_tokens - 1, true);

        int64_t t0 = ggml_time_us();
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) != 0) {
            fprintf(stderr, "Failed to process batch\n");
            return 1;
        }
        int64_t t1 = ggml_time_us();

        float * logits = llama_get_logits_ith(ctx, -1);
        float yes_logit = logits[yes_token];
        float score_calibrated = sigmoid(yes_logit / 5.0f);

        printf("========================================\n");
        printf("Doc %zu: '%s'\n", i, docs[i].c_str());
        printf("  Tokens: %d, Time: %.2f ms (%.1f tok/s)\n", n_tokens, (t1 - t0) / 1000.0, n_tokens * 1e6 / (t1 - t0));
        printf("  Raw Logit[Yes(9454)]: %+.4f\n", yes_logit);
        printf("  Calibrated Score (sigmoid(logit/5)): %.6f\n", score_calibrated);
    }
    printf("========================================\n");

    llama_batch_ext_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
