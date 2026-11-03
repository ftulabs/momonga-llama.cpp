# Jetson AGX Xavier (Volta sm_72) local changes

This fork carries two commits on top of `ggml-org/llama.cpp` that exist to make one model decode
faster on one board: Jetson AGX Xavier, Volta sm_72, CUDA 11.4. They are not proposed upstream -
they are either board-specific or debug scaffolding - and the numbers below come from a single
machine.

    git log --oneline upstream/master..HEAD

## 1. `cuda : use MMVQ for MUL_MAT_ID on sm_72 and drop the per-type batch cap`

`get_mmvq_mmid_max_batch()` tested the compute capability with equality against
`GGML_CUDA_CC_VOLTA` (700). Xavier reports 720, so the test failed and the per-type cap fell
through to the `pascal_older` table. Above that cap `MUL_MAT_ID` silently switched from the MMVQ
MoE kernel to MMQ, which is roughly 2.5x slower at these batch sizes - and speculative decoding
with a multi-token-prediction head runs at batches 3-7, exactly where the cliff sat.

Two changes:

- compare the capability as a range, `cc >= GGML_CUDA_CC_VOLTA && cc < GGML_CUDA_CC_TURING`, so
  sm_72 (and sm_73) select the Volta row;
- put the token index on `blockIdx.z` instead of `blockDim.y`, so a block is always one warp and
  any batch is a legal launch. The old layout needed `__launch_bounds__` sized to the cap, which
  is why the cap could not simply be raised: anything above the table gave
  `CUDA error: invalid argument`.

Measured on an 18 GiB IQ3_S MoE model (35B total, 3B active), 400 generated tokens, median of 5
runs: gate/up at 5 tokens went from 0.884 ms to 0.410 ms, and the model from 31.50 tok/s at
`--spec-draft-n-max 3` to 34.49 at `--spec-draft-n-max 6` (draft acceptance 0.989 -> 0.967).
The patch alone changes nothing at batch 4, where both paths were already legal; its value is
that deeper drafting stops being a penalty.

Re-measured on this base (`2cbca38`) after the rebase, same harness, 5 reps: 34.30 tok/s
(33.34 / 34.14 / 34.36 / 34.30 / 34.30), acceptance 0.96657, mean length 5.68. That is 0.5%
below the 34.49 measured on `dde879b`, which 5 reps cannot resolve from noise.

The same file also reads `GGML_CUDA_MOE_RPB` (rows per block in the MoE kernel) and
`GGML_CUDA_SMALLK`, both experiment hooks that default to the upstream behaviour. The
rows-per-block one only reproduced a microbenchmark win that is a model-level loss, so the
default stays 2.

## 2. `cuda : opt-in per-op timing dump behind GGML_CUDA_OP_TIME`

`nsys`, `ncu` and `nvprof` are not available on this board's image, so attributing decode time to
individual nodes otherwise means timing them by hand. Setting `GGML_CUDA_OP_TIME` records a CUDA
event pair per node, disables graph capture, and every fifth graph prints the accumulated totals
(top 16, aggregated by node name with the layer suffix stripped) to stderr.

It costs real time and is off by default. Use it for shares, not for absolute tok/s.

## Build

```sh
cmake -S . -B build-volta -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=72 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.4/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10.3.0 \
  -DCMAKE_CUDA_FLAGS="-Xcompiler=-isystem,/usr/include/c++/10.3.0,-isystem,/usr/include/c++/10.3.0/aarch64-poky-linux,-isystem,/usr/include/cuda-compat-workarounds" \
  -DGGML_CUDA_NO_VMM=ON -DGGML_CUDA_NO_PEER_COPY=ON \
  -DGGML_CUDA_FA=ON -DGGML_CUDA_GRAPHS=ON -DGGML_NATIVE=ON \
  -DGGML_LTO=OFF -DLLAMA_BUILD_TESTS=OFF
```

`CMAKE_CUDA_HOST_COMPILER` and the `-isystem` flags are both required: nvcc as shipped with CUDA
11.4 rejects the system gcc on this image and cannot find the C++ headers without them. Do not
pass `CMAKE_CUDA_FLAGS` alone on the command line from a build that already has these flags - it
replaces them.
