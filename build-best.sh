#!/bin/bash
# build-best.sh: Build momonga-llama.cpp in the verified best configuration for Jetson AGX Xavier (sm_72 Volta).
#
# Usage:
#   ./build-best.sh [build_dir]
#
# Defaults:
#   build_dir = ./build-volta
#   JOBS      = $(nproc)
set -euo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BUILD_DIR=${1:-"$REPO_DIR/build-volta"}
JOBS=${JOBS:-$(nproc)}

NVCC=/usr/local/cuda-11.4/bin/nvcc
HOST_CXX=/usr/bin/g++-10.3.0
CUDA_FLAGS="-Xcompiler=-isystem,/usr/include/c++/10.3.0,-isystem,/usr/include/c++/10.3.0/aarch64-poky-linux,-isystem,/usr/include/cuda-compat-workarounds"

echo "======================================================================"
echo "momonga-llama.cpp: Best Configuration Build for Jetson Xavier (sm_72)"
echo "======================================================================"
echo "Source dir:   $REPO_DIR"
echo "Build dir:    $BUILD_DIR"
echo "Build jobs:   $JOBS"
echo "NVCC:         $NVCC"
echo "Host CXX:     $HOST_CXX"
echo "======================================================================"

# Validate toolchain
for bin in "$NVCC" "$HOST_CXX"; do
    if [ ! -x "$bin" ]; then
        echo "Error: Required compiler not found or not executable: $bin" >&2
        exit 1
    fi
done

# Configure with verified flags
echo "[1/2] Configuring CMake..."
cmake -S "$REPO_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=72 \
  -DCMAKE_CUDA_COMPILER="$NVCC" \
  -DCMAKE_CUDA_HOST_COMPILER="$HOST_CXX" \
  -DCMAKE_CUDA_FLAGS="$CUDA_FLAGS" \
  -DGGML_CUDA_NO_VMM=ON \
  -DGGML_CUDA_NO_PEER_COPY=ON \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DGGML_NATIVE=ON \
  -DGGML_LTO=OFF \
  -DLLAMA_BUILD_TESTS=OFF

# Build core targets
echo "[2/2] Building targets (llama-server, llama-cli, llama-quantize, llama-tokengate)..."
cmake --build "$BUILD_DIR" -j"$JOBS" --target llama-server llama-cli llama-quantize llama-tokengate

# Verify targets
echo "======================================================================"
echo "Build verification:"
TARGETS=(llama-server llama-cli llama-quantize llama-tokengate)
ALL_OK=true
for tgt in "${TARGETS[@]}"; do
    bin_path="$BUILD_DIR/bin/$tgt"
    if [ -x "$bin_path" ]; then
        echo "  [OK] $bin_path"
    else
        echo "  [MISSING] $bin_path"
        ALL_OK=false
    fi
done

if [ "$ALL_OK" = true ]; then
    echo "======================================================================"
    echo "Build SUCCESSFUL. All production targets ready in $BUILD_DIR/bin/"
    echo "======================================================================"
else
    echo "======================================================================"
    echo "Build finished with missing binaries. Check the log above." >&2
    exit 1
fi
