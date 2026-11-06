#include "llama-memory-hybrid-tg.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-context.h"
#include "llama-io.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <stdexcept>

//
// llama_memory_hybrid_tg
//

llama_memory_hybrid_tg::llama_memory_hybrid_tg(
        const llama_model & model,
                     bool   no_alloc,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr) :
    hparams(model.hparams),
    lo(new llama_memory_hybrid(
        model, no_alloc, type_k, type_v, v_trans, kv_size, 0, LLAMA_SWA_TYPE_NONE,
        type_r, type_s, rs_size, n_seq_max, n_rs_seq, offload, unified,
        [&](int32_t il) { return il <= hparams.tg_gate_layer && filter_attn(il); },
        [&](int32_t il) { return il <= hparams.tg_gate_layer && filter_recr(il); })),
    hi(new llama_memory_hybrid(
        model, no_alloc, type_k, type_v, v_trans, kv_size, 0, LLAMA_SWA_TYPE_NONE,
        type_r, type_s, rs_size, n_seq_max, n_rs_seq, offload, unified,
        [&](int32_t il) { return il >  hparams.tg_gate_layer && filter_attn(il); },
        [&](int32_t il) { return il >  hparams.tg_gate_layer && filter_recr(il); })) {
    GGML_ASSERT(hparams.tg_gate_layer >= 0);
}

llama_memory_context_ptr llama_memory_hybrid_tg::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    do {
        balloc.split_reset();

        // the same splits as llama_memory_hybrid, the halves see the same ubatches
        std::vector<llama_ubatch> ubatches;

        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                const bool unified = (lo->get_mem_attn()->get_n_stream() == 1);

                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                const uint32_t n_rs_seq = lo->get_mem_recr()->n_rs_seq;

                ubatch = balloc.split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        llama_kv_cache::slot_info_vec_t heads_lo;
        llama_kv_cache::slot_info_vec_t heads_hi;

        for (int h = 0; h < 2; ++h) {
            const bool use = h == 0 ? use_lo() : use_hi();
            if (!use) {
                continue;
            }

            llama_memory_hybrid * mem = h == 0 ? lo.get() : hi.get();

            // prepare the recurrent batches first
            if (!mem->get_mem_recr()->prepare(ubatches)) {
                LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches (%s half)\n", __func__, h == 0 ? "lo" : "hi");
                return std::make_unique<llama_memory_hybrid_tg_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
            }

            auto heads = mem->get_mem_attn()->prepare(ubatches);
            if (heads.empty()) {
                LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches (%s half)\n", __func__, h == 0 ? "lo" : "hi");
                return std::make_unique<llama_memory_hybrid_tg_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
            }

            (h == 0 ? heads_lo : heads_hi) = std::move(heads);
        }

        return std::make_unique<llama_memory_hybrid_tg_context>(
                this, std::move(heads_lo), std::move(heads_hi), std::move(ubatches), use_lo(), use_hi());
    } while(false);

    return std::make_unique<llama_memory_hybrid_tg_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid_tg::init_full() {
    return std::make_unique<llama_memory_hybrid_tg_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid_tg::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_tg_context>(this, lctx, optimize);
}

bool llama_memory_hybrid_tg::get_can_shift() const {
    return lo->get_can_shift() && hi->get_can_shift();
}

void llama_memory_hybrid_tg::clear(bool data) {
    lo->clear(data);
    hi->clear(data);

    if (tg_states) {
        for (auto & st : tg_states->seqs) {
            st.clear();
        }
    }
}

bool llama_memory_hybrid_tg::tg_can_truncate(llama_seq_id seq_id, llama_pos p0) const {
    if (!tg_states) {
        return true;
    }
    for (size_t s = 0; s < tg_states->seqs.size(); ++s) {
        if (seq_id >= 0 && (size_t) seq_id != s) {
            continue;
        }
        for (const auto & d : tg_states->seqs[s].docs) {
            if (d.beg < p0 && p0 < d.end) {
                return false;
            }
        }
    }
    return true;
}

bool llama_memory_hybrid_tg::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    const bool to_end = p1 < 0 || p1 == std::numeric_limits<llama_pos>::max();

    // a gated document stays whole (its kept set comes from the whole span), and only the end of a gated sequence
    // can be removed (its slots count the kept positions before each token)
    if (tg_states) {
        if (!to_end) {
            for (size_t s = 0; s < tg_states->seqs.size(); ++s) {
                if ((seq_id < 0 || (size_t) seq_id == s) && tg_states->seqs[s].active) {
                    LLAMA_LOG_ERROR("%s: seq_rm(%d, %d, %d) inside a TokenGate sequence\n", __func__, seq_id, p0, p1);
                    return false;
                }
            }
        } else if (p0 > 0 && !tg_can_truncate(seq_id, p0)) {
            LLAMA_LOG_ERROR("%s: seq_rm(%d, %d, %d) would cut a TokenGate document\n", __func__, seq_id, p0, p1);
            return false;
        }
    }

    // a range a half does not hold is a no-op there; the halves roll back together, they saw the same last ubatches
    if (!lo->seq_rm(seq_id, p0, p1)) {
        return false;
    }
    if (!hi->seq_rm(seq_id, p0, p1)) {
        LLAMA_LOG_ERROR("%s: seq_rm(%d, %d, %d) failed in the upper half only, the halves have diverged\n",
                __func__, seq_id, p0, p1);
        return false;
    }

    if (tg_states && to_end) {
        for (size_t s = 0; s < tg_states->seqs.size(); ++s) {
            if (seq_id >= 0 && (size_t) seq_id != s) {
                continue;
            }
            auto & st = tg_states->seqs[s];
            if (p0 <= 0) {
                st.clear();
            } else {
                GGML_ASSERT(st.truncate(p0));
            }
        }
    }

    return true;
}

void llama_memory_hybrid_tg::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    lo->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    hi->seq_cp(seq_id_src, seq_id_dst, p0, p1);

    // the destination takes the source's tables (a copy of the whole sequence, in practice)
    if (tg_states && seq_id_src != seq_id_dst &&
        seq_id_src >= 0 && (size_t) seq_id_src < tg_states->seqs.size() &&
        seq_id_dst >= 0 && (size_t) seq_id_dst < tg_states->seqs.size()) {
        tg_states->seqs[seq_id_dst] = tg_states->seqs[seq_id_src];
    }
}

void llama_memory_hybrid_tg::seq_keep(llama_seq_id seq_id) {
    lo->seq_keep(seq_id);
    hi->seq_keep(seq_id);

    if (tg_states) {
        for (size_t s = 0; s < tg_states->seqs.size(); ++s) {
            if ((llama_seq_id) s != seq_id) {
                tg_states->seqs[s].clear();
            }
        }
    }
}

void llama_memory_hybrid_tg::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    lo->seq_add(seq_id, p0, p1, shift);
    hi->seq_add(seq_id, p0, p1, shift);
}

void llama_memory_hybrid_tg::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    lo->seq_div(seq_id, p0, p1, d);
    hi->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_memory_hybrid_tg::seq_pos_min(llama_seq_id seq_id) const {
    if (!use_hi()) {
        return lo->seq_pos_min(seq_id);
    }
    if (!use_lo()) {
        return hi->seq_pos_min(seq_id);
    }
    return std::max(lo->seq_pos_min(seq_id), hi->seq_pos_min(seq_id));
}

llama_pos llama_memory_hybrid_tg::seq_pos_max(llama_seq_id seq_id) const {
    if (!use_hi()) {
        return lo->seq_pos_max(seq_id);
    }
    if (!use_lo()) {
        return hi->seq_pos_max(seq_id);
    }
    // the upper half of a gated sequence misses the pruned tail of its last document: the sequence ends where the
    // lower half does once it is past that document
    llama_pos hi_max = hi->seq_pos_max(seq_id);
    if (tg_states && seq_id >= 0 && (size_t) seq_id < tg_states->seqs.size() && tg_states->seqs[seq_id].active) {
        hi_max = std::max<llama_pos>(hi_max, tg_states->seqs[seq_id].n_prompt - 1);
    }
    return std::min(lo->seq_pos_max(seq_id), hi_max);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid_tg::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = lo->memory_breakdown();
    for (const auto & buft_size : hi->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    return mb;
}

void llama_memory_hybrid_tg::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    lo->state_write(io, seq_id, flags);
    hi->state_write(io, seq_id, flags);

    // a partial state (a checkpoint) restores the recurrent parts only: the tables follow the removal that comes
    // with it (seq_rm), the whole state carries them
    if (flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) {
        return;
    }

    const uint32_t n_seq = tg_states ? (uint32_t) tg_states->seqs.size() : 0;
    const uint32_t n     = seq_id < 0 ? n_seq : 1;
    io.write(&n, sizeof(n));
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t s = seq_id < 0 ? i : (uint32_t) seq_id;
        const std::vector<uint8_t> data = tg_states && s < n_seq ? tg_states->seqs[s].serialize() : llama_tg_state().serialize();
        const uint32_t size = (uint32_t) data.size();
        io.write(&size, sizeof(size));
        io.write(data.data(), size);
    }
}

void llama_memory_hybrid_tg::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    lo->state_read(io, seq_id, flags);

    try {
        hi->state_read(io, seq_id, flags);

        if (flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) {
            return;
        }

        uint32_t n = 0;
        io.read(&n, sizeof(n));
        if (seq_id >= 0 && n != 1) {
            throw std::runtime_error("TokenGate: expected the tables of one sequence");
        }
        // a whole-context state sets the tables of every sequence: those it does not hold are empty
        if (seq_id < 0 && tg_states) {
            for (auto & st : tg_states->seqs) {
                st.clear();
            }
        }
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t size = 0;
            io.read(&size, sizeof(size));
            // the tables of a sequence: two floats per position plus a few documents
            if (size > (64u << 20)) {
                throw std::runtime_error("TokenGate: invalid sequence tables");
            }
            std::vector<uint8_t> data(size);
            io.read(data.data(), size);

            llama_tg_state st;
            if (!st.deserialize(data.data(), data.size())) {
                throw std::runtime_error("TokenGate: invalid sequence tables");
            }
            const size_t s = seq_id >= 0 ? (size_t) seq_id : i;
            if (tg_states && s < tg_states->seqs.size()) {
                tg_states->seqs[s] = std::move(st);
            }
        }
    } catch (...) {
        // a failed restore leaves the sequence empty in both halves
        lo->seq_rm(seq_id, -1, -1);
        hi->seq_rm(seq_id, -1, -1);
        if (tg_states) {
            for (size_t s = 0; s < tg_states->seqs.size(); ++s) {
                if (seq_id < 0 || (size_t) seq_id == s) {
                    tg_states->seqs[s].clear();
                }
            }
        }

        throw;
    }
}

void llama_memory_hybrid_tg::set_phase(llama_tg_phase phase) {
    this->phase = phase;

    // [EXPERIMENTAL] TG_SCORE_WINDOW=N: the attention layers of the bottom half see the last N positions only while
    // the document is scored (the decode of later tokens still sees the whole prompt)
    static const uint32_t score_window = [] {
        const char * s = getenv("TG_SCORE_WINDOW");
        return s ? (uint32_t) std::max(0, atoi(s)) : 0u;
    }();
    if (score_window > 0 && lo->get_mem_attn()) {
        lo->get_mem_attn()->set_mask_window(phase == LLAMA_TG_PHASE_BOTTOM ? score_window : 0);
    }
}

llama_tg_phase llama_memory_hybrid_tg::get_phase() const {
    return phase;
}

llama_memory_hybrid * llama_memory_hybrid_tg::get_lo() const {
    return lo.get();
}

llama_memory_hybrid * llama_memory_hybrid_tg::get_hi() const {
    return hi.get();
}

void llama_memory_hybrid_tg::set_tg_states(llama_tg_states * states) {
    tg_states = states;
}

//
// llama_memory_hybrid_tg_context
//

llama_memory_hybrid_tg_context::llama_memory_hybrid_tg_context(llama_memory_status status) : status(status) {}

llama_memory_hybrid_tg_context::llama_memory_hybrid_tg_context(llama_memory_hybrid_tg * mem) :
    ctx_lo(mem->get_lo()->init_full()),
    ctx_hi(mem->get_hi()->init_full()),
    status(llama_memory_status_combine(ctx_lo->get_status(), ctx_hi->get_status())) {
}

llama_memory_hybrid_tg_context::llama_memory_hybrid_tg_context(
        llama_memory_hybrid_tg * mem,
                 llama_context * lctx,
                          bool   optimize) :
    ctx_lo(mem->get_lo()->init_update(lctx, optimize)),
    ctx_hi(mem->get_hi()->init_update(lctx, optimize)),
    status(llama_memory_status_combine(ctx_lo->get_status(), ctx_hi->get_status())) {
}

llama_memory_hybrid_tg_context::llama_memory_hybrid_tg_context(
          llama_memory_hybrid_tg * mem,
                  slot_info_vec_t   sinfos_lo,
                  slot_info_vec_t   sinfos_hi,
        std::vector<llama_ubatch>   ubatches,
                             bool   use_lo,
                             bool   use_hi) :
    ubatches(std::move(ubatches)),
    // note: here we copy the ubatches. not sure if this is ideal
    ctx_lo(use_lo ? new llama_memory_hybrid_context(mem->get_lo(), std::move(sinfos_lo), this->ubatches) : nullptr),
    ctx_hi(use_hi ? new llama_memory_hybrid_context(mem->get_hi(), std::move(sinfos_hi), this->ubatches) : nullptr),
    status(llama_memory_status_combine(
                ctx_lo ? ctx_lo->get_status() : LLAMA_MEMORY_STATUS_SUCCESS,
                ctx_hi ? ctx_hi->get_status() : LLAMA_MEMORY_STATUS_SUCCESS)) {
}

bool llama_memory_hybrid_tg_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    if (ctx_lo) {
        ctx_lo->next();
    }
    if (ctx_hi) {
        ctx_hi->next();
    }

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_memory_hybrid_tg_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    bool res = true;

    if (ctx_lo) {
        res = res & ctx_lo->apply();
    }
    if (ctx_hi) {
        res = res & ctx_hi->apply();
    }

    return res;
}

llama_memory_status llama_memory_hybrid_tg_context::get_status() const {
    return status;
}

const llama_ubatch & llama_memory_hybrid_tg_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return ubatches[i_next];
}

const llama_memory_hybrid_context * llama_memory_hybrid_tg_context::get_lo() const {
    return static_cast<const llama_memory_hybrid_context *>(ctx_lo.get());
}

const llama_memory_hybrid_context * llama_memory_hybrid_tg_context::get_hi() const {
    return static_cast<const llama_memory_hybrid_context *>(ctx_hi.get());
}
