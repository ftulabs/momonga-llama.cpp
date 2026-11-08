# References, artifacts, and non-comparable results

## Source repositories

| Repo | Notes |
|---|---|
| [BeeLlama.cpp](https://github.com/Anbeeld/beellama.cpp) | `58a1629`. KVarN KV, precision tails, extra low-bit cache types. **Local tree deleted 2026-11-03** (measured slower than mainline, see the 2026-10-14 and 2026-11-01 findings) |
| [jusko-llama-volta-qwen3flash](https://github.com/jackjusko/jusko-llama-volta-qwen3flash) | Volta SM70/Turing SM75 fork. Headline gains are V100 prefill. **Local tree deleted 2026-11-03**: measured 18.61 tok/s against a 31.26 control |
| [llama.cpp issue #28037](https://github.com/ggml-org/llama.cpp/issues/28037) | SM70/SM72 FlashAttention investigation; custom tile changes neutral-to-worse for decode, suggests split-D WMMA |
| [sm70-attn benchmark](https://github.com/fishlikeX/sm70-attn/blob/main/README-sm70.md#performance) | V100: +39.9% at 176K prompt processing, flat decode |
| TokenGate fork (local, no git) | `/home/minh/src/llama.cpp-tokengate`, upstream `de5122d` + gate. See [2026-11-01-speculators.md](2026-11-01-speculators.md) |

## Local builds

| Path | Build |
|---|---|
| `/home/triph/juicer/momonga-llama.cpp/build-volta` | fork `master` (`2cbca38` upstream + the MMVQ dispatch patch), the current best build |
| `/home/triph/juicer/momonga-llama.cpp/build-volta-dde879b` | the same patch on `dde879b`, previous verified build, kept as fallback |
| `/home/triph/juicer/momonga-llama.cpp/build-mmq` | `dde879b` + `GGML_CUDA_FORCE_MMQ=ON`, slower for decode; CMake cache still points at the pre-rename path |
| Cherry-picked prior trees | BeeLlama `58a1629` and `jusko-llama-volta-qwen3flash` were built here until 2026-11-03, then deleted; both measured worse than the patched fork |
| `/mnt/sdcard/opt/llama/bin/llama-server-tgc`, `llama-server-tg` | prebuilt TokenGate-enabled servers |

## Result artifacts from the 2026-10-14 sweeps

- Sweep: `/tmp/opencode/xavier-mtp-262144-exhaustive-v2/`
  (`slot-screen.csv`, `.journal`, `run.log`, `screen.html`)
- 24.77 retest: `/tmp/opencode/xavier-mtp-262144-retest-24-77/` (`hold-24-77.csv`, `.journal`)
- Single-load confirmation: `/tmp/opencode/xavier-mtp-262144-retest-24-77-singleload/`
  (`hold-singleload.csv`, `report.html`)
- Deep-context attempt: `/tmp/opencode/xavier-mtp-262k-depth-profile/`
  (`depth-profile.csv`, `.journal`, `resume-depth-supervisor.sh`, `resume.log`), tmux
  `xavier-mtp262k-depth`
- BeeLlama vs jusko A/B: `/tmp/opencode/xavier-bee-jusko-ab/{beellama,jusko}/`, `run.log`
- Quant A/B: `/tmp/opencode/xavier-quant-ab/{q4_k_m,iq4_xs}/`
- MiniCPM quant A/B: `/tmp/opencode/xavier-minicpm-quant-ab/{q4_k_m,q4_0}/`
- MiniCPM DSpark draft: `/tmp/opencode/xavier-minicpm-dflash/dspark-q4_0.csv`, `report.html`,
  `run.log`
- MiniCPM at 262K with KVarN4: `/tmp/opencode/xavier-minicpm-q4km-dflash/cyberflags-kvarn4/`
- Underdog-Saluki: `/tmp/opencode/xavier-underdog-saluki-speed/single/`
- MiniCPM overnight 216K: `/tmp/opencode/xavier-minicpm-216k-overnight-20261014/`
  (`summary.csv`, `results.jsonl`, `telemetry.jsonl`, `roofline.json`), tmux
  `xavier-minicpm-216k-overnight`

## Result artifacts from 2026-11-01

- `/home/triph/juicer/bench/results.csv` - every server-config measurement
- `/home/triph/juicer/bench/spec.csv` - acceptance / mean length per config
- `/home/triph/juicer/bench/prompt2.txt` - the varied prompt (correct one)
- `/home/triph/juicer/bench/prompt.txt` - degenerate repeated prompt, kept only to reproduce the
  artifact
- `/tmp/prof-best.txt` - raw GPU/EMC telemetry samples

## Worktree note

`/home/triph/.local/share/opencode/worktree/121db71f73e1b8f1a7a04b1dcac0e2dde2f4341e/xavier-speed-tuning`
has local uncommitted optimizer/wrapper/supervisor changes from the October experiments. Do not
discard them without checking ownership and status.

## Coworker's separate multi-GPU benchmark (NOT Xavier-comparable)

- A shared `nvidia-smi` screenshot showed six Tesla K80 GPU devices, each about 12 GiB, plus a
  separate 4 GiB GeForce device. A K80 card contains two GPUs, so six logical devices likely means
  three K80 cards.
- In llama.cpp, `-ts` / `--tensor-split` gives proportional model shares across selected GPUs; it
  is not a count of tensors. `-sm` selects layer/row/tensor split behaviour.
- The table looked like a before/after benchmark with decode improvements at 745 MHz fixed
  clocks, but hardware, runtime, model, quant and exact flags were not confirmed.
- **Its reported gains are not comparable to Xavier results.** Do not cite them alongside the
  numbers in these files.

## Voltron / other hardware claims to treat with caution

- A Volta-focused fork's Ornith results are V100 measurements. Its +19.54% was prefill, +0.21%
  was decode.
- Its `GGML_CUDA_FORCE_MMQ=ON` result was ~+67-70% prefill and ~+0.8% decode on V100. On this
  Xavier, FORCE_MMQ measured **-14% decode**. Direction of the prefill gain agrees; decode does
  not transfer.
