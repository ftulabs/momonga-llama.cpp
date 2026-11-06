#pragma once

#include "llama-memory-hybrid.h"
#include "llama-tokengate.h"

#include <memory>
#include <vector>

//
// llama_memory_hybrid_tg
//

// TokenGate memory: the hybrid memory split at the gate layer L into two hybrid memories, lo for the layers [0, L]
// and hi for the layers above L. The phase routes the batches: the BOTTOM phase writes lo only (the whole prompt),
// the TOP phase hi only (the kept prompt tokens, so hi holds a pruned prompt), NONE and DECODE both.

class llama_memory_hybrid_tg : public llama_memory_i {
public:
    llama_memory_hybrid_tg(
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
                            /* layer filters of the whole model, split at the gate layer */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr);

    ~llama_memory_hybrid_tg() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    // the positions of the half (or halves) the current phase writes
    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0)       override;

    //
    // llama_memory_hybrid_tg specific API
    //

    void           set_phase(llama_tg_phase phase);
    llama_tg_phase get_phase() const;

    llama_memory_hybrid * get_lo() const;
    llama_memory_hybrid * get_hi() const;

    // the gated softmax tables of the sequences (owned by the context): they follow the sequences, so that a removal,
    // a copy or a state restore keeps them in step with what the halves hold
    void set_tg_states(llama_tg_states * states);

private:
    bool use_lo() const { return phase != LLAMA_TG_PHASE_TOP;    }
    bool use_hi() const { return phase != LLAMA_TG_PHASE_BOTTOM; }

    // the gated documents of seq_id (all sequences for seq_id < 0) end before p0 or start at or after it
    bool tg_can_truncate(llama_seq_id seq_id, llama_pos p0) const;

    const llama_hparams & hparams;

    llama_tg_phase phase = LLAMA_TG_PHASE_NONE;

    llama_tg_states * tg_states = nullptr;

    const std::unique_ptr<llama_memory_hybrid> lo;
    const std::unique_ptr<llama_memory_hybrid> hi;
};

class llama_memory_hybrid_tg_context : public llama_memory_context_i {
public:
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    // init failure
    explicit llama_memory_hybrid_tg_context(llama_memory_status status);

    // init full: both halves, whatever the phase
    explicit llama_memory_hybrid_tg_context(llama_memory_hybrid_tg * mem);

    // init update
    explicit llama_memory_hybrid_tg_context(
        llama_memory_hybrid_tg * mem,
                 llama_context * lctx,
                          bool   optimize);

    // init success: the halves with slots are the ones the phase writes
    llama_memory_hybrid_tg_context(
          llama_memory_hybrid_tg * mem,
                  slot_info_vec_t   sinfos_lo,
                  slot_info_vec_t   sinfos_hi,
        std::vector<llama_ubatch>   ubatches,
                             bool   use_lo,
                             bool   use_hi);

    ~llama_memory_hybrid_tg_context() = default;

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    //
    // llama_memory_hybrid_tg_context
    //

    // nullptr when the phase does not write that half
    const llama_memory_hybrid_context * get_lo() const;
    const llama_memory_hybrid_context * get_hi() const;

private:
    // the index of the next ubatch to process
    size_t i_next = 0;

    std::vector<llama_ubatch> ubatches;

    const llama_memory_context_ptr ctx_lo;
    const llama_memory_context_ptr ctx_hi;

    const llama_memory_status status;
};
