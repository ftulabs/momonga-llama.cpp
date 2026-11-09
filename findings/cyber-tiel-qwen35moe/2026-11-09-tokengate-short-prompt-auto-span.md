# TokenGate Short Prompt Auto-Spanning & Systemd Service Update

**Date:** 2026-11-09  
**Model:** `Cyber-Tiel-Coder-35B-A3B-tg.gguf` (17.3 GB, NVMe SSD)  
**Hardware:** NVIDIA Jetson AGX Xavier (sm_72 Volta, 30,991 MiB unified memory)  
**Commit:** `b2005c8` on `momonga-llama.cpp`  
**Status:** Adopted into main systemd service (`llama-server.service`) and fork master.  

---

## 1. Problem: Short Prompts (e.g. Search Snippets) Falling Back to 67 tok/s

Previously, `--tokengate-auto` in `tools/server/server-context.cpp` required `n_doc > k` before creating an auto-span:
```cpp
if (params_base.tokengate_auto && n_doc > k && (forced || n_doc >= params_base.tokengate_min_span))
```
When `k=2000`, any short prompt (e.g., Google search queries + snippets of 100 to 1,000 tokens) failed `n_doc > k`. Consequently:
- TokenGate skipped the prompt (`no document markers`).
- The prompt fell back to standard prefill at `-ub 32`, running at only **67.4 tok/s** (~7.5s latency for a 500-token search result).

## 2. Fix: Decoupled Prefill for All Prompts >= `tokengate_min_span`

In commit `b2005c8`:
1. Removed `n_doc > k` and `e - b > k` gating checks in `server-context.cpp`.
2. TokenGate auto-spans any prompt satisfying `n_doc >= params_base.tokengate_min_span`.
3. Set `--tokengate-min-span 32`.
4. When $N \le K$, `llama_tg_select` in `src/llama-tokengate.cpp` selects all $N$ tokens ($\min(k, N) = N$). **Zero tokens are pruned**, preserving 100% exact bit-level fidelity (including URLs, symbols, and code).
5. Prefill executes via TokenGate's decoupled wide GEMM engine (`TG_UBATCH=1024`) at full tensor core saturation.

## 3. Measured Results

Tested on a 206-token Google search snippet:
- **TokenGate Status:** `document [0, 205) -> 205 of 205 tokens kept (100% fidelity)`.
- **Warm Prompt Eval Latency:** Dropped from 7.5s down to **1.40s (144.5 tok/s cold-to-warm)**.
- **Prompt2 (267 tokens):** **173.2 tok/s**, prefilled in 1.54s.
- **Speculative Decode:** **32.4 - 44+ tok/s** with MTP and GPU backend sampling active (`--temp 0.0`).

## 4. Systemd Service Replacement

Updated `/etc/systemd/system/llama-server.service`, `/mnt/sdcard/opt/llama/llama-server.env`, and `/mnt/sdcard/opt/llama/llama-server-start.sh`:
- Service: `llama-server.service` enabled on boot.
- User: `triph`
- Port: `8080`
- Model: `/mnt/ssd/models/Cyber-Tiel-Coder-35B-A3B-tg.gguf`
- Drafter: MTP depth 6 (`--spec-draft-vocab /mnt/ssd/models/draft-vocab-32k.json`, `-bs`, `--temp 0.0`)
- TokenGate: `--tokengate --tokengate-auto --tokengate-min-span 32 --tokengate-k 2000 --tokengate-dilate 2 --tokengate-fa`
- Startup: Automatic warm-up via `ExecStartPost` eliminating cold graph capture tax.
