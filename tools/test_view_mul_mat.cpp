#include "llama.h"
#include "ggml.h"
#include "ggml-cuda.h"
#include <cstdio>
#include <vector>

int main() {
    llama_backend_init();

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        printf("CUDA backend init failed\n");
        return 1;
    }

    const int64_t n_embd = 2560;
    const int64_t n_vocab = 151936;
    const int64_t target_token = 9454;

    struct ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx = ggml_init(params);

    // Create a mock tok_embd tensor in Q8_0
    struct ggml_tensor * tok_embd = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, n_embd, n_vocab);
    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cuda_buffer_type(0));

    // Create activation tensor in F32: [n_embd, 10]
    struct ggml_tensor * act = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, 10);

    // Create row view of tok_embd: shape [n_embd, 1] at target_token
    struct ggml_tensor * row_view = ggml_view_2d(ctx, tok_embd, n_embd, 1, tok_embd->nb[1], target_token * tok_embd->nb[1]);
    row_view->buffer = tok_embd->buffer;

    // Matrix multiply: [n_embd, 1] x [n_embd, 1] -> [1, 1]
    struct ggml_tensor * out = ggml_mul_mat(ctx, row_view, act);

    ggml_backend_buffer_t buf_act = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cuda_buffer_type(0));

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    printf("Graph nodes: %d\n", ggml_graph_n_nodes(gf));
    printf("out shape: [%lld, %lld]\n", (long long)out->ne[0], (long long)out->ne[1]);

    enum ggml_status status = ggml_backend_graph_compute(backend, gf);
    printf("Compute status: %d (0 = success)\n", status);

    ggml_backend_buffer_free(buf_w);
    ggml_backend_buffer_free(buf_act);
    ggml_backend_free(backend);
    ggml_free(ctx);
    llama_backend_free();

    return status == GGML_STATUS_SUCCESS ? 0 : 1;
}
