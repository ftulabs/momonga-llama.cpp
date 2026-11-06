#include "models.h"

void llama_model_qwen3::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    uint32_t tg_layer = UINT32_MAX;
    if (ml.get_key(LLM_KV_TOKENGATE_GATE_LAYER, tg_layer, false)) {
        hparams.tg_gate_layer = (int32_t) tg_layer;
    }

    switch (hparams.n_layer()) {
        case 28: type = hparams.n_embd == 1024 ? LLM_TYPE_0_6B : LLM_TYPE_1_7B; break;
        case 36: type = hparams.n_embd == 2560 ? LLM_TYPE_4B : LLM_TYPE_8B; break;
        case 40: type = LLM_TYPE_14B; break;
        case 64: type = LLM_TYPE_32B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen3::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    // output rerank head
    cls_out = create_tensor(tn(LLM_TENSOR_CLS_OUT, "weight"), {n_embd, hparams.n_cls_out}, TENSOR_NOT_REQUIRED);

    // output embedding projection
    dense_2_out_layers   = create_tensor(tn(LLM_TENSOR_DENSE_2_OUT, "weight"), {n_embd, hparams.n_embd_out()}, TENSOR_NOT_REQUIRED);
    dense_2_out_layers_b = create_tensor(tn(LLM_TENSOR_DENSE_2_OUT, "bias"),   {hparams.n_embd_out()        }, TENSOR_NOT_REQUIRED);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_gqa, n_embd_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);

        // TokenGate: the scoring branch is a clone of the gate layer's MLP with an absorbed 1-wide readout
        const int32_t tg_layer = hparams.tg_gate_layer;
        if (tg_layer >= 0 && i == tg_layer) {
            layer.tg_gate = create_tensor(tn(LLM_TENSOR_TG_GATE, "weight", i), {n_embd, n_ff}, 0);
            layer.tg_up   = create_tensor(tn(LLM_TENSOR_TG_UP,   "weight", i), {n_embd, n_ff}, 0);
            layer.tg_down = create_tensor(tn(LLM_TENSOR_TG_DOWN, i),           {n_ff,   1},    0); // kept f32: it ranks the tokens
        }
        if (tg_layer >= 0 && i > tg_layer) {
            layer.tg_qscale = create_tensor(tn(LLM_TENSOR_TG_QSCALE, i), {1, n_head},    0);
            layer.tg_kbias  = create_tensor(tn(LLM_TENSOR_TG_KBIAS,  i), {1, 1, n_head}, 0);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen3::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(params);
}

llama_model_qwen3::graph::graph(const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * cur = build_backbone();

    // TokenGate bottom half: the output is [a_j | gate score], no lm_head
    if (cparams.ctx_type == LLAMA_CONTEXT_TYPE_TG_BOTTOM) {
        return;
    }

    // lm_head
    cur = build_lora_mm(model.output, cur, model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;
}

ggml_tensor * llama_model_qwen3::graph::build_backbone() {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // TokenGate (arXiv 2610.26754): the model runs as two contexts split at the gate layer L
    //   bottom: layers [0, L) + the attention half of L, outputs [a_j | score_j] for every token
    //   top:    the MLP half of L + layers (L, n_layer), over the kept tokens only, with the gated softmax
    //           softmax(gamma_h r_q q.k / sqrt(d) + kb_k c_h) in every layer above L
    const int32_t tg_layer  = hparams.tg_gate_layer;
    const bool    tg_bottom = cparams.ctx_type == LLAMA_CONTEXT_TYPE_TG_BOTTOM;
    const bool    tg_top    = cparams.ctx_type == LLAMA_CONTEXT_TYPE_TG_TOP;
    GGML_ASSERT(!(tg_bottom || tg_top) || tg_layer >= 0);

    const int il_begin = tg_top    ? tg_layer     : 0;
    const int il_end   = tg_bottom ? tg_layer + 1 : n_layer;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    ggml_tensor * tg_qr = nullptr; // [1, n_head, n_tokens] scaled-softmax ratio of each query
    ggml_tensor * tg_qoff = nullptr;
    ggml_tensor * tg_kb = nullptr; // [n_kv, 1, n_head]     gate bias of each key
    if (tg_top) {
        auto * inp_tg = build_inp_tokengate();
        tg_qr   = ggml_repeat_4d(ctx0, inp_tg->qr, 1, n_head, n_tokens, 1);
        tg_qoff = inp_tg->qoff;
        tg_kb = ggml_repeat_4d(ctx0, inp_tg->kb, inp_tg->kb->ne[0], 1, n_head, inp_tg->kb->ne[3]);
    }

    for (int il = il_begin; il < il_end; ++il) {
        res->t_layer_inp[il] = inpL;

        ggml_tensor * inpSA = inpL;

        // TokenGate top half: the attention of the gate layer ran in the bottom half
        if (tg_top && il == tg_layer) {
            cur = build_tg_ffn(inpL, il);
            inpL = cur;
            continue;
        }

        // norm
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            // compute Q and K and RoPE them
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);

            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);

            Qcur = build_rope(Qcur, inp_pos, il);

            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
            cb(Kcur, "Kcur_normed", il);

            Kcur = build_rope(Kcur, inp_pos, il);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            ggml_tensor * kq_b = nullptr;
            if (tg_top) {
                // per-(query, head) temperature, per-(key, head) gate bias
                // tg_kbias is stored premultiplied by sqrt(d): kq_b is scaled by kq_scale in the softmax
                Qcur = ggml_mul(ctx0, Qcur, ggml_add(ctx0, ggml_mul(ctx0, tg_qr, model.layers[il].tg_qscale), tg_qoff));
                cb(Qcur, "Qcur_tg", il);
                kq_b = ggml_mul(ctx0, tg_kb, model.layers[il].tg_kbias);
                cb(kq_b, "kq_b_tg", il);
            }

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, kq_b, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
        }
        if (il == il_end - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // TokenGate bottom half: stop at the attention residual a_j of the gate layer and score it
        if (tg_bottom && il == tg_layer) {
            const auto & layer = model.layers[il];

            ggml_tensor * z = build_norm(ffn_inp, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
            ggml_tensor * score = build_ffn(z,
                    layer.tg_up,   NULL, NULL,
                    layer.tg_gate, NULL, NULL,
                    layer.tg_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(score, "tg_score", il);

            cur = ggml_concat(ctx0, ffn_inp, score, 0);
            cb(cur, "tg_out", il);
            res->t_embd = cur;

            return cur;
        }

        cur = build_tg_ffn(ffn_inp, il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    return cur;
}

// the feed-forward half of a layer: ffn_inp + MLP(norm(ffn_inp))
ggml_tensor * llama_model_qwen3::graph::build_tg_ffn(ggml_tensor * ffn_inp, int il) {
    ggml_tensor * cur = build_norm(ffn_inp,
            model.layers[il].ffn_norm, NULL,
            LLM_NORM_RMS, il);
    cb(cur, "ffn_norm", il);

    cur = build_ffn(cur,
            model.layers[il].ffn_up,   NULL, model.layers[il].ffn_up_s,
            model.layers[il].ffn_gate, NULL, model.layers[il].ffn_gate_s,
            model.layers[il].ffn_down, NULL, model.layers[il].ffn_down_s,
            NULL,
            LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(cur, "ffn_out", il);

    cur = ggml_add(ctx0, cur, ffn_inp);

    cur = build_cvec(cur, il);
    cb(cur, "l_out", il);

    return cur;
}
