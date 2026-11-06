// TokenGate (arXiv 2610.26754) inference: gated prefill + greedy decode
//
// The model is split at the gate layer L into two contexts that share the weights:
//   bottom (LLAMA_CONTEXT_TYPE_TG_BOTTOM): layers [0, L] up to the attention residual a_j of L,
//          run over the whole prompt; returns [a_j | raw gate score] for every token
//   top    (LLAMA_CONTEXT_TYPE_TG_TOP):    the MLP of L and the layers above it, run over the kept tokens
//          only (all non-state tokens + the top-K state tokens), with the gated softmax
//
// Decode runs every new token through both halves: the layers <= L keep the full prompt in their KV cache,
// the layers above L only the kept tokens.
//
// input: JSONL, one prompt per line: {"ids": [...], "state_start": s, "state_end": e, "answer": "..."}
//   ids are the prompt token ids (chat template applied), [s, e) is the prunable state span; optional "id" is
//   copied to the output and optional "n_predict" caps that row's generation below -n
//
// -k takes K[:w] entries: w > 0 keeps each chosen token's +-w state neighbours at the same budget K, exactly as the
// reference's TG_KEEP_DILATE=w (default w: --dilate)
//
// --lora: the gate's paired adapter as a llama.cpp LoRA GGUF, applied to both halves: it starts at the gate layer,
// whose attention runs in the bottom half when that layer is a full-attention one (a gate on layer 3 of qwen35moe),
// and the reference trains and runs that part with the adapter on too. Embedding inputs carry explicit positions,
// so the top half passes the M-RoPE text form [p, p, p, 0]; a model with one position per token reads the first
// entry.

#include "llama.h"
#include "llama-cpp.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

struct sample {
    std::vector<llama_token> ids;
    int32_t state_start = 0;
    int32_t state_end   = 0;
    int32_t n_predict   = -1; // per-row cap, -1 = -n
    std::string answer;
    std::string id;
};

// one gated arm: K state tokens kept (-1 = all), selected with +-w dilation (0 = plain top-K)
struct arm {
    int32_t k;
    int32_t w;
};

struct params {
    std::string model;
    std::string samples;
    std::string out; // JSONL with one result per (sample, K)
    std::vector<arm> ks = { { -1, -1 } }; // -1 = keep every state token; w = -1 takes --dilate
    int32_t dilate    = 0;
    int32_t n_rows    = -1;
    int32_t n_predict = 32;
    int32_t n_gpu     = 99;
    int32_t n_ubatch  = 1024;
    int32_t n_ubatch_top = 256; // the gated layers materialize KQ [n_kv, n_ubatch, n_head] (no flash attention)
    int32_t n_threads = 0;
    bool    baseline  = false; // also time a plain full prefill
    int32_t baseline_rows = -1; // ... on the first N rows only (it is the slow arm)
    int32_t all_rows  = -1;     // the keep-all arm (K = -1, as slow as the baseline) on the first N rows only
    bool    no_gate   = false; // only the plain full prefill
    bool    kv_q8     = false; // q8_0 KV cache in every context
    std::string lora;          // the gate's paired adapter, on both halves
    float   lora_scale = 1.0f;
    bool    native     = false; // the gated arms through one context with the split memory (llama_tokengate_prefill)
    bool    native_plain = false; // also a plain prefill arm through that context ("nplain")
    bool    spec_check = false; // native decode in 3-token batches with 2 wrong drafts, rolled back each step
    bool    native_fa  = false; // native gated layers with flash attention and the head-mean key bias
    bool    native_doc = false; // native, as the server: the prefix decoded plain, the state span and the rest of the
                                // prompt through llama_tokengate_prefill_doc()
};

static void usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s -m model.gguf -f samples.jsonl [-k 100,500:2,-1] [--dilate W] [-n n_predict] [--rows N]\n"
        "          [-ngl N] [-ub N] [-ubt N] [-t N] [--baseline] [--no-gate] [--kv-q8] [--lora adapter.gguf]\n"
        "          [--lora-scale S] [--baseline-rows N] [--all-rows N] [--native] [--native-plain] [--spec-check] [--native-doc]\n"
        "          [-o results.jsonl]\n", argv0);
}

static bool parse_args(int argc, char ** argv, params & p) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for %s\n", a.c_str());
                exit(1);
            }
            return argv[++i];
        };
        if      (a == "-m")              { p.model     = next(); }
        else if (a == "-f")              { p.samples   = next(); }
        else if (a == "-o")              { p.out       = next(); }
        else if (a == "-n")              { p.n_predict = atoi(next()); }
        else if (a == "--rows")          { p.n_rows    = atoi(next()); }
        else if (a == "-ngl")            { p.n_gpu     = atoi(next()); }
        else if (a == "-ub")             { p.n_ubatch  = atoi(next()); }
        else if (a == "-ubt")            { p.n_ubatch_top = atoi(next()); }
        else if (a == "--kv-q8")         { p.kv_q8     = true; }
        else if (a == "--no-gate")       { p.no_gate   = true; p.baseline = true; }
        else if (a == "-t")              { p.n_threads = atoi(next()); }
        else if (a == "--baseline")      { p.baseline  = true; }
        else if (a == "--lora")          { p.lora      = next(); }
        else if (a == "--baseline-rows") { p.baseline_rows = atoi(next()); p.baseline = true; }
        else if (a == "--all-rows")      { p.all_rows  = atoi(next()); }
        else if (a == "--lora-scale")    { p.lora_scale = (float) atof(next()); }
        else if (a == "--dilate")        { p.dilate    = atoi(next()); }
        else if (a == "--native")        { p.native    = true; }
        else if (a == "--native-plain")  { p.native_plain = true; p.native = true; }
        else if (a == "--spec-check")    { p.spec_check = true; p.native = true; }
        else if (a == "--native-fa")     { p.native_fa = true; p.native = true; }
        else if (a == "--native-doc")    { p.native_doc = true; p.native = true; }
        else if (a == "-k") {
            p.ks.clear();
            std::string s = next();
            size_t pos = 0;
            while (pos <= s.size()) {
                const size_t c = s.find(',', pos);
                const std::string e = s.substr(pos, c - pos);
                const size_t colon = e.find(':');
                arm m = { atoi(e.substr(0, colon).c_str()), -1 };
                if (colon != std::string::npos) {
                    m.w = atoi(e.substr(colon + 1).c_str());
                }
                p.ks.push_back(m);
                if (c == std::string::npos) {
                    break;
                }
                pos = c + 1;
            }
        } else {
            usage(argv[0]);
            return false;
        }
    }
    if (p.model.empty() || p.samples.empty()) {
        usage(argv[0]);
        return false;
    }
    for (auto & m : p.ks) {
        if (m.w < 0) {
            m.w = p.dilate;
        }
    }
    return true;
}

static std::vector<sample> load_samples(const std::string & path, int32_t n_rows) {
    std::vector<sample> res;
    std::ifstream f(path);
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path.c_str());
        exit(1);
    }
    std::string line;
    while (std::getline(f, line) && (n_rows < 0 || (int32_t) res.size() < n_rows)) {
        if (line.empty()) {
            continue;
        }
        const json j = json::parse(line);
        sample s;
        s.ids         = j.at("ids").get<std::vector<llama_token>>();
        s.state_start = j.at("state_start").get<int32_t>();
        s.state_end   = j.at("state_end").get<int32_t>();
        s.n_predict   = j.value("n_predict", -1);
        s.answer      = j.value("answer", "");
        s.id          = j.value("id", "");
        if (s.state_start < 0 || s.state_end > (int32_t) s.ids.size() || s.state_start >= s.state_end) {
            fprintf(stderr, "invalid state span [%d, %d) for %zu ids\n", s.state_start, s.state_end, s.ids.size());
            exit(1);
        }
        res.push_back(std::move(s));
    }
    return res;
}

// scaled-softmax ratio of the kept slot m (1-based) at original position p: log(m)/log(p + 1), clamped to <= 1
static float ssmax_ratio(int64_t m, int64_t p) {
    const double lm = std::log(std::max<double>(m, 2.0));
    const double lp = std::log(std::max<double>(p + 1, 2.0));
    return (float) std::min(1.0, lm / lp);
}

// the kept state tokens (indices into b, ascending): the top k by score, or with w > 0 the reference's dilated
// selection (gate_forward._dilate_topk): ceil(k / (2w+1)) anchors by score plus their +-w windows, then trimmed by
// score (overlapping windows at the edges) or topped up with the next-best tokens (merged windows) to exactly k
static std::vector<int32_t> select_state(const std::vector<float> & b, int32_t k, int32_t w) {
    const int32_t n = (int32_t) b.size();
    auto top = [&](std::vector<int32_t> idx, int32_t m) {
        if (m < (int32_t) idx.size()) {
            std::nth_element(idx.begin(), idx.begin() + m, idx.end(),
                    [&](int32_t x, int32_t y) { return b[x] > b[y]; });
            idx.resize(m);
        }
        return idx;
    };
    std::vector<int32_t> all(n);
    std::iota(all.begin(), all.end(), 0);

    std::vector<int32_t> res;
    if (w <= 0 || k >= n) {
        res = top(all, std::min(k, n));
    } else {
        std::vector<char> in(n, 0);
        for (int32_t a : top(all, std::max(1, (k + 2*w) / (2*w + 1)))) {
            for (int32_t o = -w; o <= w; ++o) {
                const int32_t j = std::min(std::max(a + o, 0), n - 1);
                if (!in[j]) {
                    in[j] = 1;
                    res.push_back(j);
                }
            }
        }
        if ((int32_t) res.size() > k) {
            res = top(res, k);
        } else if ((int32_t) res.size() < k) {
            std::vector<int32_t> rest;
            for (int32_t j = 0; j < n; ++j) {
                if (!in[j]) {
                    rest.push_back(j);
                }
            }
            const auto extra = top(rest, std::min<int32_t>(k - (int32_t) res.size(), (int32_t) rest.size()));
            res.insert(res.end(), extra.begin(), extra.end());
        }
    }
    std::sort(res.begin(), res.end());
    return res;
}

static float model_meta_f32(const llama_model * model, const char * key, float def) {
    char buf[64];
    if (llama_model_meta_val_str(model, key, buf, sizeof(buf)) < 0) {
        return def;
    }
    return (float) atof(buf);
}

static llama_token argmax(const float * logits, int32_t n_vocab) {
    return (llama_token) (std::max_element(logits, logits + n_vocab) - logits);
}

static std::string detok(const llama_vocab * vocab, const std::vector<llama_token> & toks) {
    std::string res;
    for (auto t : toks) {
        char buf[256];
        const int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, false);
        if (n > 0) {
            res.append(buf, n);
        }
    }
    return res;
}

static std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}

static double now_ms() {
    return llama_time_us() / 1000.0;
}

struct result {
    int32_t k;
    int32_t w;
    int32_t n_prompt;
    int32_t n_state;
    int32_t n_kept;
    double  t_phase1_ms;
    double  t_gate_ms;
    double  t_phase2_ms;
    double  t_decode_ms;
    int32_t n_gen;
    std::string text;
    std::vector<llama_token> gen;    // generated token ids (exact comparisons between arms)
    std::vector<int32_t> kept_state; // kept state positions
};

struct tokengate_runner {
    llama_model   * model;
    llama_context * bot;
    llama_context * top;

    int32_t n_embd;
    int32_t n_vocab;
    float   bnorm_w; // gain of the sequence RMSNorm over the raw scores

    const llama_vocab * vocab;

    result run(const sample & s, arm a, int32_t n_predict) {
        const int32_t k = a.k;
        result r = {};
        r.k        = k;
        r.w        = a.w;
        r.n_prompt = (int32_t) s.ids.size();
        r.n_state  = s.state_end - s.state_start;

        llama_memory_clear(llama_get_memory(bot), true);
        llama_memory_clear(llama_get_memory(top), true);
        llama_tokengate_clear(top);

        const int32_t n_tok = r.n_prompt;
        const int32_t n_out = n_embd + 1; // [a_j | raw score] per token

        // phase 1: the bottom half over the whole prompt, in one batch so that the output buffer keeps
        // every row until phase 2 has read the kept ones
        const double t0 = now_ms();

        if (n_tok > (int32_t) llama_n_batch(bot)) {
            fprintf(stderr, "prompt of %d tokens exceeds the bottom batch size %u\n", n_tok, llama_n_batch(bot));
            exit(1);
        }
        {
            llama_batch_ext_ptr batch(llama_batch_ext_init(bot));
            for (int32_t i = 0; i < n_tok; ++i) {
                const int32_t idx = llama_batch_ext_add_token(batch.get(), 0, s.ids[i]);
                const llama_pos pos = i;
                llama_batch_ext_set_pos(batch.get(), idx, &pos);
                llama_batch_ext_set_output_embd(batch.get(), idx, true);
            }
            if (llama_process(bot, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
                fprintf(stderr, "phase 1 decode failed\n");
                exit(1);
            }
        }
        const float * out = llama_get_embeddings(bot); // [n_tok][n_out], batch order

        std::vector<float> raw(n_tok);
        for (int32_t i = 0; i < n_tok; ++i) {
            raw[i] = out[(size_t) i * n_out + n_embd];
        }

        const double t1 = now_ms();

        // gate: RMS-normalize the raw scores over the state span, keep the top-K state tokens
        const int32_t s0 = s.state_start;
        const int32_t s1 = s.state_end;
        const int32_t n_state = s1 - s0;

        double ss = 0.0;
        for (int32_t i = s0; i < s1; ++i) {
            ss += (double) raw[i] * raw[i];
        }
        const float inv_rms = (float) (1.0 / std::sqrt(ss / n_state + 1e-6));

        std::vector<float> bn(n_state);
        for (int32_t i = 0; i < n_state; ++i) {
            bn[i] = raw[s0 + i] * inv_rms * bnorm_w;
        }

        const int32_t n_keep = k < 0 ? n_state : std::max(1, std::min(k, n_state));

        const std::vector<int32_t> order = select_state(bn, n_keep, a.w);

        std::vector<llama_pos> kept;
        std::vector<float>     kept_kb;
        kept.reserve(s0 + n_keep + (n_tok - s1));
        for (int32_t i = 0; i < s0; ++i) {
            kept.push_back(i);
            kept_kb.push_back(0.0f);
        }
        for (int32_t j : order) {
            kept.push_back(s0 + j);
            kept_kb.push_back(bn[j]);
            r.kept_state.push_back(s0 + j);
        }
        for (int32_t i = s1; i < n_tok; ++i) {
            kept.push_back(i);
            kept_kb.push_back(0.0f);
        }
        const int32_t n_kept = (int32_t) kept.size();
        r.n_kept = n_kept;

        // the softmax form of the gated layers: per-slot ratio r, key bias = normalized score * r
        std::vector<float> qr(n_kept);
        for (int32_t m = 0; m < n_kept; ++m) {
            qr[m] = ssmax_ratio(m + 1, kept[m]);
            kept_kb[m] *= qr[m];
        }
        llama_tokengate_set(top, kept.data(), qr.data(), kept_kb.data(), n_kept);

        const double t2 = now_ms();

        // phase 2: the top half over the kept tokens
        llama_token tok;
        {
            const int32_t n_bat_top = (int32_t) llama_n_batch(top);
            llama_batch_ext_ptr batch(llama_batch_ext_init(top));
            for (int32_t i0 = 0; i0 < n_kept; i0 += n_bat_top) {
                const int32_t n = std::min(n_bat_top, n_kept - i0);
                llama_batch_ext_clear(batch.get());
                for (int32_t i = 0; i < n; ++i) {
                    const llama_pos p = kept[i0 + i];
                    const llama_pos pos[4] = { p, p, p, 0 };
                    const llama_embd e = { out + (size_t) p * n_out, 1, (size_t) n_embd };
                    const int32_t idx = llama_batch_ext_add_embd(batch.get(), 0, e);
                    llama_batch_ext_set_pos(batch.get(), idx, pos);
                    llama_batch_ext_set_output_logits(batch.get(), idx, i0 + i == n_kept - 1);
                }
                if (llama_process(top, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
                    fprintf(stderr, "phase 2 decode failed\n");
                    exit(1);
                }
            }
            tok = argmax(llama_get_logits_ith(top, -1), n_vocab);
        }

        const double t3 = now_ms();

        // decode: every new token through the bottom half (full KV) and the top half (kept KV)
        std::vector<llama_token> gen;
        {
            llama_batch_ext_ptr bb(llama_batch_ext_init(bot));
            llama_batch_ext_ptr bt(llama_batch_ext_init(top));
            for (int32_t step = 0; step < n_predict; ++step) {
                if (llama_vocab_is_eog(vocab, tok)) {
                    break;
                }
                gen.push_back(tok);
                if (step == n_predict - 1) {
                    break;
                }
                const llama_pos pos = n_tok + step;

                llama_batch_ext_clear(bb.get());
                int32_t idx = llama_batch_ext_add_token(bb.get(), 0, tok);
                llama_batch_ext_set_pos(bb.get(), idx, &pos);
                llama_batch_ext_set_output_embd(bb.get(), idx, true);
                if (llama_process(bot, LLAMA_PROCESS_TYPE_DECODE, bb.get()) != 0) {
                    fprintf(stderr, "decode (bottom) failed\n");
                    exit(1);
                }
                const float * h = llama_get_embeddings_ith(bot, 0);

                const float r_dec = ssmax_ratio(n_kept + 1 + step, pos);
                llama_tokengate_set(top, &pos, &r_dec, nullptr, 1);

                llama_batch_ext_clear(bt.get());
                const llama_embd e = { h, 1, (size_t) n_embd };
                const llama_pos pos4[4] = { pos, pos, pos, 0 };
                idx = llama_batch_ext_add_embd(bt.get(), 0, e);
                llama_batch_ext_set_pos(bt.get(), idx, pos4);
                llama_batch_ext_set_output_logits(bt.get(), idx, true);
                if (llama_process(top, LLAMA_PROCESS_TYPE_DECODE, bt.get()) != 0) {
                    fprintf(stderr, "decode (top) failed\n");
                    exit(1);
                }
                tok = argmax(llama_get_logits_ith(top, 0), n_vocab);
            }
        }

        const double t4 = now_ms();

        r.t_phase1_ms = t1 - t0;
        r.t_gate_ms   = t2 - t1;
        r.t_phase2_ms = t3 - t2;
        r.t_decode_ms = t4 - t3;
        r.n_gen       = (int32_t) gen.size();
        r.text        = detok(vocab, gen);
        r.gen         = gen;
        return r;
    }
};

// the gated arms through one context whose memory is split at the gate layer: llama_tokengate_prefill() over all but
// the last prompt token, then every further token (the last prompt token included) through both halves in one graph
struct native_runner {
    llama_context * ctx;

    int32_t n_vocab;
    bool    spec_check; // decode [tok, junk, junk] and roll the junk back, as a rejected 2-token draft
    bool    doc = false; // the server's way: plain prefix, the span and the rest through llama_tokengate_prefill_doc()

    const llama_vocab * vocab;

    // decode tok at pos (with 2 wrong drafts behind it under spec_check) and return the greedy next token
    llama_token step(llama_batch_ext * b, llama_token tok, llama_pos pos) {
        llama_batch_ext_clear(b);
        const int32_t n = spec_check ? 3 : 1;
        for (int32_t i = 0; i < n; ++i) {
            const llama_token t = i == 0 ? tok : (llama_token) ((tok + 7*i) % n_vocab);
            const llama_pos   p = pos + i;
            const int32_t idx = llama_batch_ext_add_token(b, 0, t);
            llama_batch_ext_set_pos(b, idx, &p);
            llama_batch_ext_set_output_logits(b, idx, true);
        }
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, b) != 0) {
            fprintf(stderr, "native decode failed at pos %d\n", pos);
            exit(1);
        }
        const llama_token next = argmax(llama_get_logits_ith(ctx, 0), n_vocab);
        if (spec_check && !llama_memory_seq_rm(llama_get_memory(ctx), 0, pos + 1, -1)) {
            fprintf(stderr, "native rollback failed at pos %d\n", pos);
            exit(1);
        }
        return next;
    }

    result run(const sample & s, arm a, int32_t n_predict) {
        result r = {};
        r.k        = a.k;
        r.w        = a.w;
        r.n_prompt = (int32_t) s.ids.size();
        r.n_state  = s.state_end - s.state_start;

        llama_memory_clear(llama_get_memory(ctx), true);
        llama_tokengate_reset(ctx);

        const int32_t n_tok  = r.n_prompt;
        const int32_t n_keep = a.k < 0 ? r.n_state : std::max(1, std::min(a.k, r.n_state));

        llama_batch_ext_ptr b(llama_batch_ext_init(ctx));

        const double t0 = now_ms();

        int32_t n_pre = 0; // prompt tokens before the last one, processed by the prefill
        if (a.k == -3) {
            // plain prefill through the split memory, all but the last prompt token
            const int32_t n_batch = (int32_t) llama_n_batch(ctx);
            for (int32_t i0 = 0; i0 < n_tok - 1; i0 += n_batch) {
                const int32_t n = std::min(n_batch, n_tok - 1 - i0);
                llama_batch_ext_clear(b.get());
                for (int32_t i = 0; i < n; ++i) {
                    const llama_pos pos = i0 + i;
                    const int32_t idx = llama_batch_ext_add_token(b.get(), 0, s.ids[i0 + i]);
                    llama_batch_ext_set_pos(b.get(), idx, &pos);
                    llama_batch_ext_set_output_logits(b.get(), idx, false);
                }
                if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, b.get()) != 0) {
                    fprintf(stderr, "native plain prefill failed\n");
                    exit(1);
                }
            }
            r.n_kept = n_tok;
        } else if (doc) {
            // the tokens [i0, i1) through llama_decode(): plain before the span, through both halves after it
            auto decode_range = [&](int32_t i0, int32_t i1) {
                const int32_t n_batch = (int32_t) llama_n_batch(ctx);
                for (int32_t j0 = i0; j0 < i1; j0 += n_batch) {
                    const int32_t n = std::min(n_batch, i1 - j0);
                    llama_batch_ext_clear(b.get());
                    for (int32_t i = 0; i < n; ++i) {
                        const llama_pos pos = j0 + i;
                        const int32_t idx = llama_batch_ext_add_token(b.get(), 0, s.ids[j0 + i]);
                        llama_batch_ext_set_pos(b.get(), idx, &pos);
                        llama_batch_ext_set_output_logits(b.get(), idx, false);
                    }
                    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, b.get()) != 0) {
                        fprintf(stderr, "native-doc decode failed at %d\n", j0);
                        exit(1);
                    }
                }
            };
            decode_range(0, s.state_start);
            if (llama_tokengate_prefill_doc(ctx, 0, s.ids.data() + s.state_start, n_tok - 1 - s.state_start,
                        s.state_end - s.state_start, n_keep, a.w) != 0) {
                fprintf(stderr, "llama_tokengate_prefill_doc failed\n");
                exit(1);
            }
            const int32_t     n_kept = llama_tokengate_n_kept(ctx);
            const llama_pos * kept   = llama_tokengate_kept(ctx);
            for (int32_t i = 0; i < n_kept; ++i) {
                if (kept[i] < s.state_end) {
                    r.kept_state.push_back(kept[i]);
                }
            }
            r.n_kept = s.state_start + n_kept + 1; // and the last prompt token
        } else {
            if (llama_tokengate_prefill(ctx, 0, s.ids.data(), n_tok - 1, s.state_start, s.state_end, n_keep, a.w) != 0) {
                fprintf(stderr, "llama_tokengate_prefill failed\n");
                exit(1);
            }
            const int32_t     n_kept = llama_tokengate_n_kept(ctx);
            const llama_pos * kept   = llama_tokengate_kept(ctx);
            for (int32_t i = 0; i < n_kept; ++i) {
                if (kept[i] >= s.state_start && kept[i] < s.state_end) {
                    r.kept_state.push_back(kept[i]);
                }
            }
            r.n_kept = n_kept + 1; // and the last prompt token
        }
        n_pre = n_tok - 1;

        const double t1 = now_ms();

        // the last prompt token gives the first generated token
        llama_token tok = step(b.get(), s.ids[n_pre], n_pre);

        const double t3 = now_ms();

        std::vector<llama_token> gen;
        for (int32_t i = 0; i < n_predict; ++i) {
            if (llama_vocab_is_eog(vocab, tok)) {
                break;
            }
            gen.push_back(tok);
            if (i == n_predict - 1) {
                break;
            }
            tok = step(b.get(), tok, n_tok + i);
        }

        const double t4 = now_ms();

        r.t_phase1_ms = t1 - t0; // the whole prefill
        r.t_gate_ms   = 0.0;
        r.t_phase2_ms = t3 - t1; // the last prompt token
        r.t_decode_ms = t4 - t3;
        r.n_gen       = (int32_t) gen.size();
        r.text        = detok(vocab, gen);
        r.gen         = gen;
        return r;
    }
};

// plain prefill of the whole prompt in a default context: the time-to-first-token reference
static result run_baseline(llama_context * ctx, const llama_vocab * vocab, const sample & s, int32_t n_predict) {
    result r = {};
    r.k        = -2;
    r.n_prompt = (int32_t) s.ids.size();
    r.n_state  = s.state_end - s.state_start;
    r.n_kept   = r.n_prompt;

    llama_memory_clear(llama_get_memory(ctx), true);

    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const int32_t n_tok   = r.n_prompt;
    const int32_t n_bat   = (int32_t) llama_n_batch(ctx);

    const double t0 = now_ms();
    llama_batch_ext_ptr batch(llama_batch_ext_init(ctx));
    for (int32_t i0 = 0; i0 < n_tok; i0 += n_bat) {
        const int32_t n = std::min(n_bat, n_tok - i0);
        llama_batch_ext_clear(batch.get());
        for (int32_t i = 0; i < n; ++i) {
            const int32_t idx = llama_batch_ext_add_token(batch.get(), 0, s.ids[i0 + i]);
            const llama_pos pos = i0 + i;
            llama_batch_ext_set_pos(batch.get(), idx, &pos);
            llama_batch_ext_set_output_logits(batch.get(), idx, i0 + i == n_tok - 1);
        }
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            fprintf(stderr, "baseline decode failed\n");
            exit(1);
        }
    }
    llama_token tok = argmax(llama_get_logits_ith(ctx, -1), n_vocab);
    const double t1 = now_ms();

    std::vector<llama_token> gen;
    for (int32_t step = 0; step < n_predict; ++step) {
        if (llama_vocab_is_eog(vocab, tok)) {
            break;
        }
        gen.push_back(tok);
        if (step == n_predict - 1) {
            break;
        }
        const llama_pos pos = n_tok + step;
        llama_batch_ext_clear(batch.get());
        const int32_t idx = llama_batch_ext_add_token(batch.get(), 0, tok);
        llama_batch_ext_set_pos(batch.get(), idx, &pos);
        llama_batch_ext_set_output_logits(batch.get(), idx, true);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            fprintf(stderr, "baseline decode failed\n");
            exit(1);
        }
        tok = argmax(llama_get_logits_ith(ctx, 0), n_vocab);
    }
    const double t2 = now_ms();

    r.t_phase1_ms = t1 - t0;
    r.t_decode_ms = t2 - t1;
    r.n_gen       = (int32_t) gen.size();
    r.text        = detok(vocab, gen);
    r.gen         = gen;
    return r;
}

static llama_context * make_ctx(llama_model * model, llama_context_type type, uint32_t n_ctx, const params & p,
        bool native = false) {
    llama_context_params cp = llama_context_default_params();
    cp.ctx_type        = type;
    cp.tokengate       = native;
    cp.n_rs_seq        = native && p.spec_check ? 2 : 0;
    cp.n_ctx           = n_ctx;
    // the bottom half takes the whole prompt as one batch (split into ubatches internally)
    cp.n_batch         = type == LLAMA_CONTEXT_TYPE_TG_BOTTOM || native ? n_ctx : p.n_ubatch;
    cp.n_ubatch        = type == LLAMA_CONTEXT_TYPE_TG_TOP ? std::min(p.n_ubatch_top, p.n_ubatch) : p.n_ubatch;
    cp.n_seq_max       = 1;
    cp.kv_unified      = true;
    cp.no_perf         = true;
    cp.embeddings      = type == LLAMA_CONTEXT_TYPE_TG_BOTTOM && !native;
    cp.pooling_type    = LLAMA_POOLING_TYPE_NONE;
    // the gated layers add a per-key bias to the logits, which the flash-attention path cannot take
    cp.flash_attn_type = type == LLAMA_CONTEXT_TYPE_TG_TOP ? LLAMA_FLASH_ATTN_TYPE_DISABLED : LLAMA_FLASH_ATTN_TYPE_AUTO;
    // a quantized V cache needs flash attention, which the gated layers cannot use; their cache is small anyway
    if (p.kv_q8 && type != LLAMA_CONTEXT_TYPE_TG_TOP && !native) {
        cp.type_k = GGML_TYPE_Q8_0;
        cp.type_v = GGML_TYPE_Q8_0;
    }
    if (p.n_threads > 0) {
        cp.n_threads       = p.n_threads;
        cp.n_threads_batch = p.n_threads;
    }
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "failed to create context of type %d\n", (int) type);
        exit(1);
    }
    return ctx;
}

static json to_json(const result & r, size_t row, const sample & s) {
    const bool ok = !s.answer.empty() && lower(r.text).find(lower(s.answer)) != std::string::npos;
    json j = {
        {"row", row},
        {"id", s.id},
        {"k", r.k},
        {"dilate", r.w},
        {"n_prompt", r.n_prompt},
        {"n_state", r.n_state},
        {"n_kept", r.n_kept},
        {"ttft_ms", r.t_phase1_ms + r.t_gate_ms + r.t_phase2_ms},
        {"phase1_ms", r.t_phase1_ms},
        {"gate_ms", r.t_gate_ms},
        {"phase2_ms", r.t_phase2_ms},
        {"decode_ms", r.t_decode_ms},
        {"n_gen", r.n_gen},
        {"match", ok},
        {"text", r.text},
        {"gen", r.gen},
    };
    if (!r.kept_state.empty() && r.k >= 0) {
        j["kept_state"] = r.kept_state;
    }
    return j;
}

int main(int argc, char ** argv) {
    params p;
    if (!parse_args(argc, argv, p)) {
        return 1;
    }

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = p.n_gpu;
    llama_model_ptr model(llama_model_load_from_file(p.model.c_str(), mp));
    if (!model) {
        fprintf(stderr, "failed to load %s\n", p.model.c_str());
        return 1;
    }
    if (llama_model_tokengate_layer(model.get()) < 0) {
        fprintf(stderr, "%s has no TokenGate gate (convert it with convert_tokengate_to_gguf.py)\n", p.model.c_str());
        return 1;
    }

    const auto samples = load_samples(p.samples, p.n_rows);

    size_t n_max = 0;
    for (const auto & s : samples) {
        n_max = std::max(n_max, s.ids.size());
    }
    const uint32_t n_ctx = GGML_PAD(n_max + p.n_predict, 256);

    // the top half holds only the kept tokens: size it by the largest K
    uint32_t n_ctx_top = 0;
    for (const auto & s : samples) {
        for (const arm & m : p.ks) {
            const int32_t k = m.k;
            const int32_t n_state = s.state_end - s.state_start;
            const int32_t n_keep  = k < 0 ? n_state : std::min(k, n_state);
            n_ctx_top = std::max<uint32_t>(n_ctx_top, s.ids.size() - n_state + n_keep + p.n_predict);
        }
    }
    n_ctx_top = GGML_PAD(n_ctx_top, 256);

    char buf[64];
    snprintf(buf, sizeof(buf), "%s", "general.architecture");
    char arch[64] = "qwen3";
    llama_model_meta_val_str(model.get(), buf, arch, sizeof(arch));
    const std::string key_bnorm = std::string(arch) + ".tokengate.bnorm_weight";

    tokengate_runner tg;
    tg.model   = model.get();
    tg.vocab   = llama_model_get_vocab(model.get());
    tg.n_embd  = llama_model_n_embd(model.get());
    tg.n_vocab = llama_vocab_n_tokens(tg.vocab);
    tg.bnorm_w = model_meta_f32(model.get(), key_bnorm.c_str(), 1.0f);
    tg.bot     = p.no_gate || p.native ? nullptr : make_ctx(model.get(), LLAMA_CONTEXT_TYPE_TG_BOTTOM, n_ctx,     p);
    tg.top     = p.no_gate || p.native ? nullptr : make_ctx(model.get(), LLAMA_CONTEXT_TYPE_TG_TOP,    n_ctx_top, p);

    native_runner nat = {};
    nat.ctx        = p.native && !p.no_gate ? make_ctx(model.get(), LLAMA_CONTEXT_TYPE_DEFAULT, n_ctx, p, true) : nullptr;
    nat.n_vocab    = tg.n_vocab;
    nat.spec_check = p.spec_check;
    nat.doc        = p.native_doc;
    nat.vocab      = tg.vocab;
    if (nat.ctx) {
        llama_tokengate_set_fa(nat.ctx, p.native_fa);
    }

    llama_context * base = p.baseline ? make_ctx(model.get(), LLAMA_CONTEXT_TYPE_DEFAULT, n_ctx, p) : nullptr;

    // the adapter goes on both halves (layers it does not cover are untouched); the baseline stays the plain model
    llama_adapter_lora * lora = nullptr;
    if (!p.lora.empty() && (tg.top || nat.ctx)) {
        lora = llama_adapter_lora_init(model.get(), p.lora.c_str());
        if (!lora) {
            fprintf(stderr, "failed to load the adapter %s\n", p.lora.c_str());
            return 1;
        }
        float scale = p.lora_scale;
        if (nat.ctx) {
            // the gated sequences only: a plain arm in the same context runs the plain model
            llama_tokengate_set_adapter(nat.ctx, lora);
        }
        for (llama_context * ctx : { tg.bot, tg.top, nat.ctx }) {
            if (!ctx) {
                continue;
            }
            if (llama_set_adapters_lora(ctx, &lora, 1, &scale) != 0) {
                fprintf(stderr, "failed to apply the adapter %s\n", p.lora.c_str());
                return 1;
            }
        }
        fprintf(stderr, "tokengate: adapter %s (scale %.2f) on both halves\n", p.lora.c_str(), scale);
    }

    fprintf(stderr, "tokengate: gate layer %d, %zu samples, n_ctx %u / top %u, bnorm %.4f\n",
            llama_model_tokengate_layer(model.get()), samples.size(), n_ctx, n_ctx_top, tg.bnorm_w);

    FILE * fout = p.out.empty() ? nullptr : fopen(p.out.c_str(), "w");

    // warm-up: the first decode of each context pays for graph allocation and kernel loading; a 2k-token prefix
    // of the first prompt reaches every ubatch shape without paying for a long prefill
    if (!samples.empty()) {
        sample w = samples[0];
        if (w.ids.size() > 2048) {
            w.ids.resize(2048);
            w.state_end = std::min<int32_t>(w.state_end, 2000);
            w.state_start = std::min<int32_t>(w.state_start, w.state_end - 1);
        }
        if (nat.ctx) {
            nat.run(w, { std::min<int32_t>(16, w.state_end - w.state_start), 0 }, 2);
        } else if (!p.no_gate) {
            tg.run(w, { std::min<int32_t>(16, w.state_end - w.state_start), 0 }, 2);
        }
        if (base) {
            run_baseline(base, tg.vocab, w, 2);
        }
    }

    printf("%4s %7s %7s %7s %10s %9s %9s %9s %6s  %s\n",
            "row", "K", "n_state", "n_kept", "ttft_ms", "p1_ms", "p2_ms", "tok/s", "match", "text");

    for (size_t row = 0; row < samples.size(); ++row) {
        const auto & s = samples[row];
        const int32_t n_predict = s.n_predict > 0 ? std::min(s.n_predict, p.n_predict) : p.n_predict;
        std::vector<result> rs;
        if (base && (p.baseline_rows < 0 || (int32_t) row < p.baseline_rows)) {
            rs.push_back(run_baseline(base, tg.vocab, s, n_predict));
        }
        if (nat.ctx && p.native_plain) {
            rs.push_back(nat.run(s, { -3, 0 }, n_predict));
        }
        for (const arm & m : p.ks) {
            if (m.k < 0 && p.all_rows >= 0 && (int32_t) row >= p.all_rows) {
                continue;
            }
            if (nat.ctx) {
                rs.push_back(nat.run(s, m, n_predict));
            } else if (!p.no_gate) {
                rs.push_back(tg.run(s, m, n_predict));
            }
        }
        for (const auto & r : rs) {
            const json j = to_json(r, row, s);
            const double tps = r.n_gen > 1 ? (r.n_gen - 1) / (r.t_decode_ms / 1000.0) : 0.0;
            std::string text = r.text;
            std::replace(text.begin(), text.end(), '\n', ' ');
            if (text.size() > 70) {
                text = text.substr(0, 70) + "...";
            }
            const std::string k_str = r.k == -2 ? "full" : r.k == -3 ? "nplain" : r.k == -1 ? "all" :
                    std::to_string(r.k) + (r.w > 0 ? ":" + std::to_string(r.w) : "");
            printf("%4zu %7s %7d %7d %10.1f %9.1f %9.1f %9.1f %6s  %s\n",
                    row, k_str.c_str(),
                    r.n_state, r.n_kept, j["ttft_ms"].get<double>(), r.t_phase1_ms, r.t_phase2_ms, tps,
                    s.answer.empty() ? "-" : (j["match"].get<bool>() ? "yes" : "no"), text.c_str());
            fflush(stdout);
            if (fout) {
                // generated pieces can split a multi-byte character: keep the line valid JSON
                fprintf(fout, "%s\n", j.dump(-1, ' ', false, json::error_handler_t::replace).c_str());
                fflush(fout);
            }
        }
    }

    if (fout) {
        fclose(fout);
    }
    if (base) {
        llama_free(base);
    }
    if (tg.top) {
        llama_free(tg.top);
    }
    if (nat.ctx) {
        llama_free(nat.ctx);
    }
    if (lora) {
        llama_adapter_lora_free(lora);
    }
    if (tg.bot) {
        llama_free(tg.bot);
    }
    llama_backend_free();
    return 0;
}
