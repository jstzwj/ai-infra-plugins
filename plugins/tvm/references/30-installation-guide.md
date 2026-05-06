# 30 — Installation Guide

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Ubuntu 18.04, macOS 10.14, Windows 10 | Ubuntu 20.04+, macOS 12+ |
| Compiler | GCC 7, Clang 6, MSVC 2019 | GCC 9+, Clang 10+ |
| CMake | 3.18 | 3.24+ |
| Python | 3.8 | 3.10+ |
| LLVM | 15+ (for CPU codegen) | 17+ |
| Git | 2.0+ | Latest |
| CUDA | 11.0 (optional) | 12.0+ |

---

## Building from Source

### Step 1: Clone Repository

```bash
git clone https://github.com/apache/tvm.git
cd tvm
git submodule update --init --recursive
```

### Step 2: Configure CMake

```bash
mkdir build
cp cmake/config.cmake build/
cd build
```

Edit `build/config.cmake` to enable features:

```cmake
# GPU Support
USE_CUDA=ON              # NVIDIA CUDA
USE_ROCM=OFF             # AMD ROCm
USE_METAL=OFF            # Apple Metal
USE_OPENCL=OFF           # OpenCL
USE_VULKAN=OFF           # Vulkan
USE_WEBGPU=OFF           # WebGPU
USE_HEXAGON=OFF          # Qualcomm Hexagon

# CPU Support
USE_LLVM=ON              # Auto-detect LLVM
# USE_LLVM=/path/to/llvm-config  # Or specify path

# Runtime Features
USE_RPC=ON               # RPC support
USE_GRAPH_EXECUTOR=ON    # Graph executor
USE_PROFILER=ON          # Profiling
USE_SORT=ON              # Sort operations
USE_RANDOM=ON            # Random operations
USE_MICRO=OFF            # MicroTVM

# Build Options
USE_CCACHE=ON            # ccache for faster rebuilds
HIDE_PRIVATE_SYMBOLS=ON  # Hide private symbols
SET_CAPABILITY_JIT=OFF   # JIT capability
```

### Step 3: Build

```bash
cmake ..
cmake --build . --parallel $(nproc)
```

### Step 4: Install Python Package

```bash
cd ../python

# Option A: pip editable install
pip install -e .

# Option B: Set PYTHONPATH
export TVM_HOME=/path/to/tvm
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH
```

### Step 5: Verify Installation

```python
import tvm
print(tvm.__version__)

# Check CUDA
import tvm.testing
if tvm.cuda(0).exist:
    print("CUDA available")
```

---

## Conda Environment Setup

```bash
# Create environment
conda create -n tvm python=3.10
conda activate tvm

# Install dependencies
conda install -c conda-forge \
    cmake \
    llvm \
    numpy \
    pytest \
    cython

# Build TVM
git clone https://github.com/apache/tvm.git
cd tvm && git submodule update --init --recursive
mkdir build && cd build
cp ../cmake/config.cmake .
cmake .. && cmake --build . --parallel $(nproc)

# Install Python
cd ../python && pip install -e .
```

---

## Docker Setup

### Pre-built Images
```bash
# CPU only
docker pull tlcpack/ci-cpu:latest

# GPU
docker pull tlcpack/ci-gpu:latest
```

### Custom Docker
```dockerfile
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    git cmake g++ python3 python3-pip \
    libssl-dev llvm-dev

RUN git clone https://github.com/apache/tvm.git /tvm
WORKDIR /tvm
RUN git submodule update --init --recursive
RUN mkdir build && cd build && \
    cp ../cmake/config.cmake . && \
    cmake .. && cmake --build . --parallel $(nproc)

ENV PYTHONPATH=/tvm/python
```

---

## Platform-Specific Notes

### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y gcc g++ cmake libssl-dev python3-dev
```

### macOS (Apple Silicon)
```bash
brew install cmake llvm
# Metal support is built-in on macOS
# Set LLVM path:
export LLVM_CONFIG=/opt/homebrew/opt/llvm/bin/llvm-config
```

### Windows
- Install Visual Studio Build Tools 2019+
- Use CMake GUI for configuration
- Consider using WSL2 for a Linux-like experience

### ROCm (AMD GPU)
```bash
# Install ROCm first (follow AMD docs)
# Then configure:
USE_ROCM=ON
ROCM_PATH=/opt/rocm
```

---

## Advanced Options

### ccache for Faster Rebuilds
```cmake
USE_CCACHE=ON
```
Requires ccache installed: `sudo apt install ccache`

### GTest Integration
```cmake
USE_GTEST=ON
# Or specify path:
USE_GTEST=/path/to/gtest
```

### Debug Builds
```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
```

### Static Linking
```cmake
USE_STATIC_LIB=ON
```

---

## Validation

### Basic Test
```python
import tvm
import tvm.testing
import numpy as np

# Test basic operation
a = tvm.nd.array(np.arange(10, dtype="float32"))
print(a.numpy())

# Test CUDA if available
if tvm.cuda(0).exist:
    a_gpu = tvm.nd.array(np.arange(10, dtype="float32"), device=tvm.cuda(0))
    print(f"CUDA device: {a_gpu.device}")
```

### Run Unit Tests
```bash
cd /path/to/tvm
python -m pytest tests/python/unittest/test_tvmscript.py -v
```

### Test Specific Backend
```bash
# CUDA tests
python -m pytest tests/python/unittest/test_cuda.py -v

# RPC tests
python -m pytest tests/python/unittest/test_rpc.py -v
```

---

## Common Issues

### LLVM Not Found
```
Error: Cannot find LLVM
```
**Solution**: Set `USE_LLVM=/path/to/llvm-config` or install LLVM development package.

### CUDA Not Found
```
Error: CUDA not found
```
**Solution**: Set `CUDA_PATH=/usr/local/cuda` and ensure nvcc is in PATH.

### Python Import Error
```
ModuleNotFoundError: No module named 'tvm'
```
**Solution**: Run `pip install -e .` from the `python/` directory or set `PYTHONPATH`.

### Build OOM
```
c++: fatal error: Killed program cc1plus
```
**Solution**: Reduce parallel build jobs: `cmake --build . --parallel 2`
