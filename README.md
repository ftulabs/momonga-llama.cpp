# momonga-llama.cpp

A fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) that carries CUDA changes
for one board: **Jetson AGX Xavier, Volta sm_72, CUDA 11.4**. Everything outside
`ggml/src/ggml-cuda/` is upstream, unmodified.

The parent project is [llama.cpp](https://github.com/ggml-org/llama.cpp); its documentation
(backends, tools, build options for every other platform) applies here unchanged except where
this file says otherwise. See [JETSON-XAVIER.md](JETSON-XAVIER.md) for the full reasoning behind
the patches.

## What is different

    git log --oneline upstream/master..HEAD

| commit | effect |
|---|---|
| `cuda : use MMVQ for MUL_MAT_ID on sm_72 and drop the per-type batch cap` | keeps the MoE matmul on the MMVQ kernel for batches above 4-5, where stock silently switches to MMQ and runs about 2.5x slower |
| `cuda : run the IQ3_S MoE kernel at 64 registers instead of 79` | the launch bounds declared a 256-thread block for a one-warp kernel, so the compiler spent 79 registers and residency was capped at 37.5%; 64 registers gives 47.9% occupancy, -7.6% kernel time, +2.25% end to end |
| `cuda : four rows per block and a wider small_k trigger on the dense MMVQ path` | the activation vector is re-read once per output row and that re-read is most of the L1 traffic at wide column counts; 4 rows per block, a 64 KB carveout, launch bounds at 8 blocks, a wider `small_k` trigger, and 4 rows at 1 column as well. Numerics-neutral. +4.3% end to end |
| `cuda : opt-in per-op timing dump behind GGML_CUDA_OP_TIME` | debug only, off unless the variable is set |
| `docs : ...` | fork-specific documentation: `JETSON-XAVIER.md`, this `README.md`, no code |

Why the first one is needed: `get_mmvq_mmid_max_batch()` compared the compute capability with
equality against `GGML_CUDA_CC_VOLTA` (700). Xavier reports 720, the comparison failed, and the
per-type cap fell through to the `pascal_older` table. Above the cap, `MUL_MAT_ID` switched to
MMQ. That matters because speculative decoding with the model's multi-token-prediction head runs
at batches 3-7, exactly where the cliff sat.

The cap could not simply be raised: the MoE kernel was compiled with `__launch_bounds__` derived
from the cap while launching `blockDim.y = n_tokens`, so anything above the table produced
`CUDA error: invalid argument`. The patch moves the token index to `blockIdx.z`, which makes a
block one warp at any batch, and compares the capability as a range instead.

## Measured result

One model, one board, 400 generated tokens, greedy, median of 5 runs:

| configuration | tok/s | draft acceptance | mean draft length |
|---|---|---|---|
| stock `n-max 3` (best before the patch) | 31.50 | 0.989 | 3.69 |
| this fork, `n-max 6`, dispatch patch only | 34.30-34.49 | 0.967 | 5.68 |
| this fork, `n-max 6`, plus the 64-register launch bounds | 35.07 | 0.967 | 5.68 |
| this fork, `n-max 6`, plus the dense MMVQ work | **36.56** | 0.967 | 5.68 |
| stock, no speculation | 16.9 | - | - |

The patch alone does nothing at batch 4, where both paths were already legal. Its value is that
drafting deeper stops being a penalty, which is where the 9.5% comes from. The range above is the
same build measured on two upstream bases (`2cbca38` and `dde879b`); the difference is inside
the noise of five reps. The 35.07 and 36.56 rows were each measured as a direct A/B on
`2cbca38`: the runs differ only in `libggml-cuda.so.0`, and every one of them reports
acceptance 0.96657 and mean length 5.68. That last point is the check that none of the
kernel changes moved a number the model computes - only when a weight file changes does
that pair move, and then it costs 44-46%.

## Build

On the Xavier, with CUDA 11.4 and the system gcc:

```sh
cmake -S . -B build-volta -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=72 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.4/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10.3.0 \
  -DCMAKE_CUDA_FLAGS="-Xcompiler=-isystem,/usr/include/c++/10.3.0,-isystem,/usr/include/c++/10.3.0/aarch64-poky-linux,-isystem,/usr/include/cuda-compat-workarounds" \
  -DGGML_CUDA_NO_VMM=ON -DGGML_CUDA_NO_PEER_COPY=ON \
  -DGGML_CUDA_FA=ON -DGGML_CUDA_GRAPHS=ON -DGGML_NATIVE=ON \
  -DGGML_LTO=OFF -DLLAMA_BUILD_TESTS=OFF

cmake --build build-volta -j8 --target llama-server
```

`CMAKE_CUDA_HOST_COMPILER` and the `-isystem` flags are both required on this image: nvcc as
shipped with CUDA 11.4 rejects the system gcc and cannot find the C++ headers without them.
`GGML_CUDA_NO_VMM` is required because Xavier is an integrated GPU without VMM support. For
builds on anything else, use the upstream [build guide](docs/build.md) and ignore this section.

## Run

```sh
env -u GGML_CUDA_DISABLE_GRAPHS \
build-volta/bin/llama-server \
  -m Cyber-Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS.gguf \
  -c 262144 -fa on -ngl 99 -ncmoe 0 -t 2 -tb 2 \
  -ctk q8_0 -ctv q8_0 -ub 32 -b 32 --poll 0 --parallel 1 \
  --spec-type draft-mtp --spec-draft-n-max 6 --spec-draft-n-min 0 \
  --spec-draft-p-min 0.5 --spec-draft-p-split 0.1 --load-mode none
```

Graphs are enabled by the *absence* of `GGML_CUDA_DISABLE_GRAPHS`: the code checks for the
variable's existence, not its value, so `GGML_CUDA_DISABLE_GRAPHS=0` still disables them.
`-t 2 -tb 2` and the q8_0 KV types are measured optima on this board, not defaults. The values
above are the fastest configuration found; the surrounding flag space was swept (batching,
polling, KV types, draft depth, parallelism, load mode) and everything else measured neutral or
worse on this board.

## Experiment hooks

All of these default to upstream behaviour; nothing needs to be set to get the results above.

| variable | effect |
|---|---|
| `GGML_CUDA_MOE_RPB` | rows per block in the MoE kernel, default 2. 4 rows cuts L1 sectors 32% and kernel time 4.1% standalone but is exactly neutral in situ (63.98 vs 63.95 ms per pass), because the model's MoE cost is a per-token DRAM stream that an L1 lever cannot reach; kept for reproducibility |
| `GGML_CUDA_MMID_MAX_BATCH` | forces the `MUL_MAT_ID` batch cap; 0 forces the MMQ path for A/B |
| `GGML_CUDA_SMALLK` | forces the wide rows-per-block layout on or off |
| `GGML_CUDA_OP_TIME` | per-op GPU timing dump to stderr, top 16 by time, every fifth graph. Disables CUDA graphs and costs real time - use it for shares, not for tok/s |

## Keeping up with upstream

`origin` is this fork, `upstream` is `ggml-org/llama.cpp`:

```sh
git fetch upstream
git rebase upstream/master
git push origin master    # fast-forward, no force needed
```

The rebase applies cleanly as long as upstream has not touched
`ggml/src/ggml-cuda/mmvq.cu` or `ggml/src/ggml-cuda/ggml-cuda.cu`. **If it has, re-measure
instead of assuming the numbers still hold** - the patches are small enough to re-apply by hand,
but the performance claims belong to the exact code that was measured. This `README.md` is
fork-owned and will conflict on rebase; keep this version (`git checkout --ours README.md`).

These changes are not proposed upstream. They exist for one board, one model and one
configuration, and the profiler is debug scaffolding rather than something a maintainer should
carry.

## License

MIT, as upstream. See [LICENSE](LICENSE). The upstream project's acknowledgements for bundled
single-header libraries apply unchanged; they are listed in the
[upstream README](https://github.com/ggml-org/llama.cpp#acknowledgements).
