#pragma once

#include "llama.h"

#include <cstdint>
#include <vector>

// TokenGate (arXiv 2610.26754): a gate at layer L scores the prompt tokens on the attention residual of L, and the
// layers above L see only the prompt tokens it keeps. In a single context the memory is split at L
// (llama_memory_hybrid_tg) and a phase tells the graph and the memory which half runs:
enum llama_tg_phase {
    LLAMA_TG_PHASE_NONE   = 0, // plain model: both halves see every token
    LLAMA_TG_PHASE_BOTTOM = 1, // layers [0, L] up to the attention residual of L, outputs [a_j | gate score]
    LLAMA_TG_PHASE_TOP    = 2, // the MLP of L and the layers above it over the kept prompt tokens (a_j as input)
    LLAMA_TG_PHASE_DECODE = 3, // after a prefill: every new token through both halves, the top one gated
};

// a document of a sequence that went through the gate
struct llama_tg_doc {
    int32_t beg = 0; // the document span [beg, end), in positions
    int32_t end = 0;
    int32_t k   = 0; // the gate's budget and dilation
    int32_t w   = 0;

    int32_t n_kept_end = 0; // kept positions of the sequence before end
    int32_t fa         = 0; // the gated layers took the flash-attention (head-mean bias) form in its prefill
};

// the softmax form of the gated layers for one sequence, by token position
struct llama_tg_state {
    // the sequence is gated: its tokens take the gated softmax in the layers above the gate
    bool active = false;

    std::vector<float> qr; // scaled-softmax ratio of the query at each position (1 when unset)
    std::vector<float> kb; // gate bias of the key at each position (0 when unset)

    // set by a prefill: the positions from n_prompt on are all kept, each at the next kept slot
    int32_t n_prompt = 0;
    int32_t n_kept   = 0;

    // the documents of the sequence that went through the gate, in order; the last one sets n_prompt and n_kept
    std::vector<llama_tg_doc> docs;

    // scaled-softmax ratio of the query at position p
    float get_qr(llama_pos p) const;

    // gate bias of the key at position p
    float get_kb(llama_pos p) const {
        return p >= 0 && (size_t) p < kb.size() ? kb[p] : 0.0f;
    }

    // the kept positions of the sequence before p, for p >= n_prompt (all of them in a sequence that is not gated)
    int32_t n_kept_before(llama_pos p) const {
        return n_kept + (p - n_prompt);
    }

    // the positions from p on are removed: drop the documents there; false (and no change) if p cuts a document
    bool truncate(llama_pos p);

    // state save/restore of one sequence
    std::vector<uint8_t> serialize() const;
    bool deserialize(const uint8_t * data, size_t size);

    void clear() {
        active = false;
        qr.clear();
        kb.clear();
        n_prompt = 0;
        n_kept   = 0;
        docs.clear();
    }
};

// the states of all sequences of a context
struct llama_tg_states {
    std::vector<llama_tg_state> seqs; // by sequence id

    // the state of a gated sequence, nullptr if the sequence is not gated
    const llama_tg_state * get(llama_seq_id seq_id) const {
        return seq_id >= 0 && (size_t) seq_id < seqs.size() && seqs[seq_id].active ? &seqs[seq_id] : nullptr;
    }

    bool any_active() const {
        for (const auto & s : seqs) {
            if (s.active) {
                return true;
            }
        }
        return false;
    }

    // scaled-softmax ratio of a query of seq_id at position p (1 for a sequence that is not gated)
    float get_qr(llama_seq_id seq_id, llama_pos p) const {
        const auto * s = get(seq_id);
        return s ? s->get_qr(p) : 1.0f;
    }
};

// scaled-softmax ratio of the kept slot m (1-based) at original position p: log(m)/log(p + 1), clamped to <= 1
float llama_tg_ssmax_ratio(int64_t m, int64_t p);

// the kept state tokens (indices into b, ascending): the top k by score, or with w > 0 the reference's dilated
// selection (gate_forward._dilate_topk): ceil(k / (2w+1)) anchors by score plus their +-w windows, then trimmed by
// score (overlapping windows at the edges) or topped up with the next-best tokens (merged windows) to exactly k
std::vector<int32_t> llama_tg_select(const std::vector<float> & b, int32_t k, int32_t w);
