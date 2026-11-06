#include "llama-tokengate.h"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

float llama_tg_state::get_qr(llama_pos p) const {
    if (n_prompt > 0 && p >= n_prompt) {
        return llama_tg_ssmax_ratio((int64_t) n_kept + 1 + (p - n_prompt), p);
    }
    return p >= 0 && (size_t) p < qr.size() ? qr[p] : 1.0f;
}

bool llama_tg_state::truncate(llama_pos p) {
    if (docs.empty() || docs.back().end <= p) {
        return true; // the generated positions after the last document: nothing to change
    }

    p = std::max<llama_pos>(p, 0);

    size_t n_keep = 0;
    while (n_keep < docs.size() && docs[n_keep].end <= p) {
        ++n_keep;
    }
    if (n_keep < docs.size() && docs[n_keep].beg < p) {
        return false; // a document cannot be cut: its kept set comes from the whole span
    }

    docs.resize(n_keep);
    if (docs.empty()) {
        clear();
        return true;
    }

    // the positions after the last kept document are all kept again
    n_prompt = docs.back().end;
    n_kept   = docs.back().n_kept_end;
    qr.resize(n_prompt, 1.0f);
    kb.resize(n_prompt, 0.0f);

    return true;
}

// [u32 magic][u8 active][i32 n_prompt][i32 n_kept][u32 n_docs][docs][u32 n_tab][qr f32 x n_tab][kb f32 x n_tab]
static constexpr uint32_t LLAMA_TG_STATE_MAGIC = 0x32534754; // "TGS2"

std::vector<uint8_t> llama_tg_state::serialize() const {
    std::vector<uint8_t> out;
    auto put = [&](const void * src, size_t size) {
        const auto * s = (const uint8_t *) src;
        out.insert(out.end(), s, s + size);
    };

    const uint32_t magic  = LLAMA_TG_STATE_MAGIC;
    const uint8_t  act    = active ? 1 : 0;
    const uint32_t n_docs = (uint32_t) docs.size();
    const uint32_t n_tab  = (uint32_t) qr.size();
    GGML_ASSERT(kb.size() == qr.size());

    put(&magic, sizeof(magic));
    put(&act, sizeof(act));
    put(&n_prompt, sizeof(n_prompt));
    put(&n_kept, sizeof(n_kept));
    put(&n_docs, sizeof(n_docs));
    for (const auto & d : docs) {
        put(&d, sizeof(d));
    }
    put(&n_tab, sizeof(n_tab));
    put(qr.data(), n_tab * sizeof(float));
    put(kb.data(), n_tab * sizeof(float));

    return out;
}

bool llama_tg_state::deserialize(const uint8_t * data, size_t size) {
    size_t off = 0;
    auto get = [&](void * dst, size_t n) {
        if (off + n > size) {
            return false;
        }
        std::memcpy(dst, data + off, n);
        off += n;
        return true;
    };

    uint32_t magic  = 0;
    uint8_t  act    = 0;
    uint32_t n_docs = 0;
    uint32_t n_tab  = 0;

    llama_tg_state st;
    if (!get(&magic, sizeof(magic)) || magic != LLAMA_TG_STATE_MAGIC ||
        !get(&act, sizeof(act)) || !get(&st.n_prompt, sizeof(st.n_prompt)) || !get(&st.n_kept, sizeof(st.n_kept)) ||
        !get(&n_docs, sizeof(n_docs))) {
        return false;
    }
    st.active = act != 0;
    // the counts must fit the bytes left before anything is allocated
    if ((uint64_t) n_docs * sizeof(llama_tg_doc) > size - off) {
        return false;
    }
    st.docs.resize(n_docs);
    for (auto & d : st.docs) {
        if (!get(&d, sizeof(d))) {
            return false;
        }
    }
    if (!get(&n_tab, sizeof(n_tab)) || (uint64_t) n_tab * 2 * sizeof(float) != size - off) {
        return false;
    }
    st.qr.resize(n_tab);
    st.kb.resize(n_tab);
    if (!get(st.qr.data(), n_tab * sizeof(float)) || !get(st.kb.data(), n_tab * sizeof(float)) || off != size) {
        return false;
    }

    *this = std::move(st);
    return true;
}

float llama_tg_ssmax_ratio(int64_t m, int64_t p) {
    const double lm = std::log(std::max<double>(m, 2.0));
    const double lp = std::log(std::max<double>(p + 1, 2.0));
    return (float) std::min(1.0, lm / lp);
}

std::vector<int32_t> llama_tg_select(const std::vector<float> & b, int32_t k, int32_t w) {
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
