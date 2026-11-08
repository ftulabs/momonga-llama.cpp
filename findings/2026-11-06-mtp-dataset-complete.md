# MTP dataset extraction complete: 3,000,519 positions banked

**Date:** 2026-11-06
**Target:** 3,000,000 tokens of `(h, token_in, argmax)` tuples dumped from `Q4KM-all-q4k.gguf`
**Output path:** `/mnt/ssd/models/_mtp/dumps/`
**Supervisor:** `bench/supervise-mtpdump.sh`

---

## 1. Final extraction metrics

The extraction run completed cleanly on schedule:

- **Total tokens banked:** 3,000,519 positions (100.02% of 3,000,000 target).
- **Total shards:** 1,190 shards (`shard-000000.mtp` to `shard-001189.mtp`).
- **Disk usage:** 22.91 GB on NVMe SSD (`/mnt/ssd`).
- **Completion time:** 2026-11-06 19:12:46+07:00 (`rc=0`), supervisor exit at 19:13:02.
- **Sustained throughput:** ~55.0 to 59.0 tok/s across the entire corpus.
- **Trunk-vs-corpus argmax agreement:** Averaged ~79.0 - 80.5% throughout the run.
- **Hardware status:** All GPU memory completely released upon exit (24 GB available unified memory).

---

## 2. Dataset integrity and layout

Each shard follows the `MTP1` format specified in `bench/MTP-DUMP-FORMAT.md`:
- Magic: `MTP1` (4 bytes)
- Header: `dim = 2048`, `n_vocab = 248320`, `n_pos` (positions in shard)
- Per position:
  - `token_in`: uint32 (corpus token entering position p)
  - `target_argmax`: uint32 (trunk greedy argmax emitted at position p, label for p+1)
  - `h`: fp16[2048] (trunk hidden state at position p-1, zeros at position 0)

The dataset is verified intact and readable.

---

## 3. Training bundle readiness

The training bundle in `/home/triph/juicer/mtp-bundle/` contains:
1. `Q4KM-all-q4k.head.gguf`: 23 tensors (1.2 GB), SHA256 verified.
2. `head-ref-256pos.hlg`: 256-position ground-truth head logits reference (122 MB).
3. `parity-input-256pos.mtp`: 256 positions of exact input activations (2.1 MB).
4. `mtp_head_torch.py`: PyTorch reference implementation and parity tester.
5. `train-mtp-head.py`: Full trainer supporting fp16 AMP + `GradScaler`, KV caching, MSE consistency loss, and `--freeze-moe` Phase 1 mode.
6. `MTP-DUMP-FORMAT.md` & `README.md`.

---

## 4. Next gate

In accordance with `AGENTS.md` and `HANDOFF.md`, no training or further modification starts
until the parity gate passes on the remote T4 (`ml`):

```bash
python mtp_head_torch.py --gguf Q4KM-all-q4k.head.gguf --selfcheck
python mtp_head_torch.py --gguf Q4KM-all-q4k.head.gguf --parity head-ref-256pos.hlg --dump parity-input-256pos.mtp
```

Required gate criteria:
- **Cosine similarity min > 0.999** across all 256 positions.
- **Argmax agreement: 100.00%** (256/256).
