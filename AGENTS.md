# AGENTS.md - notes for agents working in this fork

This is `ftulabs/momonga-llama.cpp`, a fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
that carries CUDA, speculative decoding, and long-context optimizations for **Jetson AGX Xavier (Volta sm_72, CUDA 11.4)**.
Read [JETSON-XAVIER.md](JETSON-XAVIER.md) for detailed analysis of the kernel changes, [README.md](README.md) for the build recipe and launch command, and [KNOWN-GOOD.md](KNOWN-GOOD.md) for the verified production configuration.

The upstream contribution rules in
[ggml-org/llama.cpp AGENTS.md](https://github.com/ggml-org/llama.cpp/blob/master/AGENTS.md) are
for PRs against upstream and **do not apply here** - the upstream file says so itself. What
replaces them follows.

---

## Rules

1. **`origin` is this fork, `upstream` is `ggml-org/llama.cpp`.** Pushing to `origin` is normal
   and expected. **Never push to `upstream`, never open a PR against `ggml-org`, never send a
   patch anywhere else.** These changes are board-specific and will not be accepted upstream.
2. **Commit only after the change is built and measured, not before.** A commit message must state
   what was measured: the number, the config it was measured in, and the base commit. A change
   that "should" be faster and has not been run does not get committed as an improvement.
3. **Verify a rebase instead of assuming it.** If upstream touches `ggml/src/ggml-cuda/`,
   `common/speculative.cpp`, or `src/llama-context.cpp`, rebuild and re-measure before repeating
   any number.
4. **ASCII only** in code, comments, commits, and docs. No em dashes, no arrows, no curly quotes.
5. **Comments are 1-2 lines** and explain why, not what. Do not hard-wrap comments to a column.
6. **Do not benchmark with `GGML_CUDA_OP_TIME` set.** It disables CUDA graphs and costs real
   time; it is for attributing shares, not for tok/s.
7. **The experiment hooks stay off by default.** `GGML_CUDA_MOE_RPB`, `GGML_CUDA_MMID_MAX_BATCH`,
   `GGML_CUDA_SMALLK` and `GGML_CUDA_OP_TIME` must all be unset for the documented numbers to
   apply. If a hook is added, it defaults to upstream behaviour or it is not added.
8. **Update documentation with every finding or code change.** Whenever an experiment is completed
   or code is modified, write the finding in `findings/<model-architecture>/YYYY-MM-DD-<topic>.md`, update `FINDINGS.md`,
   `findings/README.md`, `OPTIMIZATION-GUIDE.md`, `TODO.md`, `KNOWN-GOOD.md`, `AGENTS.md`, `README.md`, and `JETSON-XAVIER.md`,
   and commit to `origin`. Stale documentation wastes expensive GPU cycles.
9. **Adopt ServerBench Rejection Hierarchy & Latency-Qualified Scoring (arXiv:2611.06841).**
   A microbenchmark can only reject a candidate; it can never accept one. Only an end-to-end scored scenario with independent correctness verification and SLA attainment (p90 TTFT and p90 TPOT) can accept a candidate. Throughput must be evaluated alongside energy efficiency (Tokens per Joule via `tegrastats` power rails) using `bench/serverbench_eval.py`.

---

## What has been measured (do not re-derive)

- **43.08 - 43.97 tok/s speculative decode** median on 400-600 tokens (and **47.5 - 48.4 tok/s** steady-state generation once `<think>` reasoning tags conclude), with draft acceptance 0.968 - 0.975 and mean draft length 5.53 - 5.75 on `Cyber-Tiel-Coder-35B-A3B-tg.gguf`. The stack underneath:
  - 31.50 stock unpatched `n-max 3` baseline.
  - 34.30 - 34.49 from the sm_72 MoE MMVQ dispatch patch (`blockIdx.z`).
  - 35.07 from the 64-register MoE launch bounds.
  - 36.56 from dense MMVQ 4 rows per block + 64 KB carveout + wider `small_k`.
  - 36.63 from non-contiguous concat geometry.
  - 36.93 from backend sampling (`-bs`).
  - **43.08 - 43.97** from single-sync sampling (PR #30196) and GPU draft sampler offload (`9c26dfe`), eliminating host sampling roundtrips and CPU idle overhead, paired with reduced draft LM head vocabulary (`--spec-draft-vocab`) and hardware CUB radix sorting on CUDA 11.4 (`1f33695`).
- **2,025 - 2,070 tok/s long-context prefill** via TokenGate (`--tokengate --tokengate-fa`):
  - In layers 0..3, dense evaluation runs at decoupled ubatch (`TG_UBATCH=512`).
  - In layers 4..40, 85-95% of tokens are pruned, evaluating only kept tokens ($K=350-400$).
  - Client wire TTFT collapses to **4.79 s at 9.6k tokens** and **11.15 s at 22.1k tokens**.
  - High-accuracy needle mode ($K=2000, W=2$) runs at **1,715 tok/s** with 100% exact needle retrieval (`BLUE-VELVET-99`) verified.
- **262,144 context allocation (`-c 262144`)** with `q8_0` KV cache verified live. Because the hybrid model has 30 recurrent DeltaNet layers ($O(1)$ memory per token) and only 11 attention layers, KV cache at 262k consumes only 8.25 GB, leaving **4.4 GB of free RAM headroom** on the 32 GB Xavier and sustaining **43.08 tok/s** decode.
- **Without the MTP drafter, `-ub` is free and prefill is 2.6x faster on dense models.** With `--spec-type none`, decode does not move with width: **215.8 tok/s prefill / 53.3 s TTFT at 11,484 tokens with `-ub 2048`**, against 83.8 tok/s at `-ub 32`. Plain decode is **27.86 tok/s** (`-bs`) with `Q4KM-all-q4k.gguf` (23.75 at 8.8k context). MTP cannot re-attach to the retyped file without retraining (acceptance drops to 0.64583, decode collapses to 23.35 tok/s).
- **The board is not bandwidth-starved**: a full speculation pass moves about 8.1 GB in 171 ms, which is 47 GB/s or 43% of the measured 110 GB/s peak sequential ceiling. In situ the MoE is measured as a pure DRAM stream at 44-52 GB/s whose cost is linear in tokens with no intercept.
- **Measurement discipline:** 5 reps and median for anything under 2%, pre-register predictions in writing before running experiments (Rule 1), evaluate draft acceptance alongside tok/s, use interleaved both-orders testing for flags, and report negative results with equal prominence.

---

## Build and Run

Run the automated scripts:

```sh
./build-best.sh
./run-best.sh
```

Or run manual CMake and server commands as documented in [README.md](README.md).

---

## Repository Documentation

All documentation and measurement records live **directly inside this repository**:

- [OPTIMIZATION-GUIDE.md](OPTIMIZATION-GUIDE.md): Cross-architecture optimization playbook, recipes, and decision tree for agents.
- [KNOWN-GOOD.md](KNOWN-GOOD.md): Validated best configuration and reproduction line.
- [FINDINGS.md](FINDINGS.md): Chronology and executive summary of all experiments.
- [findings/](findings/): Complete directory of experimental records organized by architecture.
  - [findings/cyber-tiel-qwen35moe/](findings/cyber-tiel-qwen35moe/): 58 dated reports on the completed Cyber-Tiel-Coder-35B-A3B / Qwen3.5-35B-A3B MTP campaign.
- [TODO.md](TODO.md): Task roadmap and closed-do-not-retest register.
- [HANDOFF.md](HANDOFF.md): Engineering handoff notes, machine telemetry, and profiling instructions.
- [JETSON-XAVIER.md](JETSON-XAVIER.md): Comprehensive deep dive into sm_72 CUDA kernels.

## Maintenance and Rebase

`README.md`, `AGENTS.md`, `OPTIMIZATION-GUIDE.md`, `JETSON-XAVIER.md`, `KNOWN-GOOD.md`, `FINDINGS.md`, `TODO.md`, `HANDOFF.md`, and `findings/` are fork-owned and will conflict when rebasing onto upstream. Keep the fork version (`git checkout --ours <file>`). Everything else in the tree stays aligned with upstream.
