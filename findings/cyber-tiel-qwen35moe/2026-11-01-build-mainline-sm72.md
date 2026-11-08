# Mainline llama.cpp SM72 build

**Date:** 2026-11-01
**Source:** `/home/triph/juicer/momonga-llama.cpp`, commit `dde879b` (ggml 0.28.0). The
directory was called `llama.cpp` at the time; paths here were rewritten after the 2026-11-03
rename.
**Build dir:** `/home/triph/juicer/momonga-llama.cpp/build-volta`

Produced the first working mainline build on this board and closed the BeeLlama-vs-mainline
question. Runtime reports `version: 0.8.0-dev (build 1, commit dde879b)`,
`built with GNU 13.5.0 for Linux aarch64`, and
`CUDA0: Xavier (30991 MiB, 24856 MiB free)`.

## Working build recipe

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

Build exits 0. Runtime reports `compute capability 7.2, VMM: no`.

## Toolchain constraints that actually gate this build

- The only usable toolkit is CUDA 11.4.315 at `/usr/local/cuda-11.4`. There is **no
  `/usr/local/cuda` symlink and no `nvcc` on `PATH`**, so both `CMAKE_CUDA_COMPILER` and
  `CUDAToolkit_ROOT` must be set explicitly.
- `/usr/local/cuda-12.2` exists but is a `compat` stub with no `bin/`.
- nvcc 11.4 rejects the host default gcc 13.5 (`gcc versions later than 11 are not supported`).
  `CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10.3.0` is mandatory.
- With that host compiler, the `-Xcompiler=-isystem,...` list is also mandatory; without it the
  first translation unit dies at `cuda_runtime.h -> <new>: No such file or directory`.
- `GGML_LTO=ON` **cannot link here**: CUDA host objects come from gcc 10.3 and everything else
  from gcc 13.5, and the LTO bytecode versions are incompatible
  (`generated with LTO version 9.3 instead of the expected 13.1`).
- `nvpmodel` is not installed; `jetson_clocks` is.

## CPU ISA flags: `dotprod` is NOT available on Carmel

`/proc/cpuinfo` flags are:
`fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm dcpop`
- there is no `asimddp`.

A `vdotq_s32` binary compiled with `-march=armv8.2-a+fp16+dotprod` exits **132 (SIGILL)** on this
board. Any advice recommending `+dotprod` for Xavier is wrong and produces a crashing binary.

`GGML_NATIVE=ON` is correct and safe: ggml probes features by *running* them and settles on
`-mcpu=generic+crc+lse+fp16+rdma+crypto+nodotprod+noi8mm+nosve`, reporting
`HAVE_DOTPROD - Failed` and `HAVE_FP16_VECTOR_ARITHMETIC - Success`.

Note `GGML_CPU_ARM_ARCH` is only read in the `else()` branch of `if (GGML_NATIVE)`, so setting
both silently ignores the arch string.

## Flag verdicts

| Flag | Verdict |
|---|---|
| `GGML_CUDA` | required |
| `CMAKE_CUDA_ARCHITECTURES=72` | required, matches GV11B |
| `GGML_CUDA_NO_VMM` | required; board reports `VMM: no` |
| `GGML_CUDA_NO_PEER_COPY` | required, single device |
| `GGML_CUDA_FA` | already ON by default |
| `GGML_CUDA_GRAPHS` | already ON by default |
| `GGML_NATIVE` | correct choice; do not add `+dotprod` |
| `GGML_LTO` | must be OFF here (gcc 10/13 split) |
| `GGML_CUDA_FORCE_MMQ` | builds, but **-14% decode**. Prefill tool only |
| `GGML_CUDA_FA_ALL_QUANTS` | deprecated in favour of `GGML_CUDA_FA_QUANTS`; unnecessary, since the default `f16-f16;q4_0-q4_0;q8_0-q8_0;bf16-bf16` already covers q8_0 KV |
| `GGML_CUDA_F16` | **does not exist** in current llama.cpp (zero occurrences) |
| `GGML_CPU_ARM_ARCH` | ignored when `GGML_NATIVE=ON` |

## Corrections to commonly repeated advice

An external recommendation (Gemini) for this board contained eight distinct flaws, each of which
was verified empirically:

1. Recommended `-march=armv8.2-a+fp16+dotprod` - **crashes** (SIGILL) on Carmel.
2. Put ISA flags in `CMAKE_C_CXX_FLAGS` instead of relying on `GGML_NATIVE` probing.
3. Set `CMAKE_CUDA_FLAGS` to a value that breaks the nvcc host include path.
4. Referenced `GGML_CUDA_F16`, which does not exist.
5. Used `GGML_CUDA_FA_ALL_QUANTS`, deprecated and unnecessary.
6. Recommended `GGML_LTO=ON`, which cannot link across the gcc 10/13 split.
7. Omitted the explicit `CMAKE_CUDA_COMPILER` despite no `nvcc` on `PATH`.
8. Omitted `GGML_CUDA_NO_VMM`, which this board needs.

Also note `--no-mmap` was renamed to `--load-mode none` in the runtime.

## Other build variants

- `/home/triph/juicer/momonga-llama.cpp/build-mmq` - same recipe plus `-DGGML_CUDA_FORCE_MMQ=ON`.
  Builds fine; measured slower for decode (see negative results in
  [2026-11-01-decode-speed-tuning.md](2026-11-01-decode-speed-tuning.md)).
- BeeLlama `58a1629` was built at `/home/triph/juicer/beellama.cpp/build-cuda` for comparison;
  that tree and the build were **deleted on 2026-11-03** once mainline had been measured about 6%
  faster and nothing pointed at it anymore.
