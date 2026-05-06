# FlashAttention: Installation and Setup Reference

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [FlashAttention-2 Installation](#flashattention-2-installation)
3. [FlashAttention-3 Installation (Hopper)](#flashattention-3-installation-hopper)
4. [FlashAttention-4 Installation (CuTeDSL)](#flashattention-4-installation-cutedsl)
5. [ROCm Installation](#rocm-installation)
6. [Build System Details](#build-system-details)
7. [Environment Variables](#environment-variables)
8. [Docker Setup](#docker-setup)
9. [Verifying Installation](#verifying-installation)
10. [Compatibility Matrix](#compatibility-matrix)
11. [Troubleshooting Installation Issues](#troubleshooting-installation-issues)

---

## Prerequisites

### Hardware Requirements

| Component | FA2 Minimum | FA2 Recommended | FA3 | FA4 |
|-----------|-------------|-----------------|-----|-----|
| GPU | Ampere (SM80+): A100, RTX 3090, RTX 4090 | A100/H100 | Hopper (SM90): H100/H800 | SM80+ (Hopper/Blackwell preferred) |
| GPU Memory | 16 GB+ | 40 GB+ (A100 40GB) | 80 GB (H100 SXM) | 16 GB+ |
| System RAM | 32 GB (compilation) | 64 GB+ | 64 GB+ | 32 GB+ |
| Disk Space | 10 GB | 20 GB+ | 20 GB+ | 10 GB+ |

### Software Requirements

| Software | FA2 Minimum | FA2 Recommended | FA3 | FA4 |
|----------|-------------|-----------------|-----|-----|
| CUDA Toolkit | 12.0 | 12.3+ | 12.3+ (12.8 recommended) | 12.x / 13.x |
| PyTorch | 2.2+ | 2.3+ | 2.4+ | 2.4+ |
| Python | 3.9+ | 3.10+ | 3.8+ | 3.9+ |
| GCC/G++ | 9+ | 11+ | 11+ | 11+ |
| ninja | Required | Latest | Required | N/A |
| packaging | Required | Latest | Required | Required |
| psutil | Required | Latest | Required | N/A |

### Additional FA4 Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| nvidia-cutlass-dsl | >= 4.4.1 | CuTeDSL kernel compilation |
| einops | Latest | Tensor manipulation |
| apache-tvm-ffi | Latest | FFI for kernel execution |
| quack-kernels | >= 0.4.0 | Compilation utilities |

---

## FlashAttention-2 Installation

### Method 1: pip Install (Recommended)

The simplest installation method. Pre-built wheels are downloaded when available.

```bash
pip install flash-attn --no-build-isolation
```

The `--no-build-isolation` flag is required because the build system needs
access to the already-installed PyTorch for version detection.

**What happens during installation:**
1. pip attempts to download a pre-built wheel matching your configuration
2. If no wheel is found, it falls back to building from source
3. Source build compiles CUDA kernels for all supported head dimensions and data types

**Estimated installation time:**
- Pre-built wheel: 10-30 seconds
- Source build with ninja (64-core machine): 3-5 minutes
- Source build without ninja: 30 minutes - 2 hours

### Method 2: pip Install from Source

```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
pip install . --no-build-isolation
```

### Method 3: pip Install with Constraints

If you have limited RAM (< 96 GB) or want faster compilation:

```bash
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

### Method 4: pip Install (Editable/Development)

```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
pip install -e . --no-build-isolation
```

### Selective Build (Disable Features)

To speed up compilation, you can disable unused features via environment
variables. These are read at build time and excluded from compilation.

```bash
# Disable backward pass (forward-only)
FLASHATTENTION_DISABLE_BACKWARD=1 pip install flash-attn --no-build-isolation

# Disable dropout
FLASHATTENTION_DISABLE_DROPOUT=1 pip install flash-attn --no-build-isolation

# Disable ALiBi support
FLASHATTENTION_DISABLE_ALIBI=1 pip install flash-attn --no-build-isolation

# Disable softcapping
FLASHATTENTION_DISABLE_SOFTCAP=1 pip install flash-attn --no-build-isolation

# Disable local (sliding window) attention
FLASHATTENTION_DISABLE_LOCAL=1 pip install flash-attn --no-build-isolation

# Disable uneven K support
FLASHATTENTION_DISABLE_UNEVEN_K=1 pip install flash-attn --no-build-isolation
```

### Target Specific CUDA Architectures

By default, FA2 builds for SM80, SM90, SM100, SM110, and SM120. To build for
only specific architectures:

```bash
# Build for A100 only
FLASH_ATTN_CUDA_ARCHS="80" pip install flash-attn --no-build-isolation

# Build for H100 only
FLASH_ATTN_CUDA_ARCHS="90" pip install flash-attn --no-build-isolation

# Build for A100 and H100
FLASH_ATTN_CUDA_ARCHS="80;90" pip install flash-attn --no-build-isolation
```

Architecture mapping:
| Value | Architecture | GPUs |
|-------|-------------|------|
| `80` | SM80 | A100, A800 |
| `90` | SM90 | H100, H800 |
| `100` | SM100 | B200, B100 |
| `110` | SM110 | Thor |
| `120` | SM120 | DGX Spark (GeForce) |

---

## FlashAttention-3 Installation (Hopper)

FA3 is installed from the `hopper/` subdirectory. It requires Hopper GPUs and
CUDA >= 12.3.

### Requirements

- **GPU**: H100 or H800 (SM90 required)
- **CUDA**: >= 12.3 (12.8 strongly recommended for best performance)
- **PyTorch**: >= 2.4

### Installation

```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper
python setup.py install
```

Or with pip:

```bash
cd flash-attention/hopper
pip install . --no-build-isolation
```

### Package Name

FA3 installs as `flash_attn_3`:
```python
import flash_attn_interface  # FA3 interface module
# or
import flash_attn_3._C        # Low-level CUDA bindings
```

### Selective Build (FA3)

FA3 supports granular feature toggles:

```bash
# Disable backward pass
FLASH_ATTENTION_DISABLE_BACKWARD=TRUE pip install . --no-build-isolation

# Disable specific head dimensions
FLASH_ATTENTION_DISABLE_HDIM64=TRUE pip install . --no-build-isolation
FLASH_ATTENTION_DISABLE_HDIM96=TRUE pip install . --no-build-isolation
FLASH_ATTENTION_DISABLE_HDIM128=TRUE pip install . --no-build-isolation
FLASH_ATTENTION_DISABLE_HDIM192=TRUE pip install . --no-build-isolation
FLASH_ATTENTION_DISABLE_HDIM256=TRUE pip install . --no-build-isolation

# Disable FP16
FLASH_ATTENTION_DISABLE_FP16=TRUE pip install . --no-build-isolation

# Disable FP8 forward
FLASH_ATTENTION_DISABLE_FP8=TRUE pip install . --no-build-isolation

# Disable SplitKV
FLASH_ATTENTION_DISABLE_SPLIT=TRUE pip install . --no-build-isolation

# Disable Paged KV
FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE pip install . --no-build-isolation

# Disable Append KV
FLASH_ATTENTION_DISABLE_APPENDKV=TRUE pip install . --no-build-isolation

# Disable local attention
FLASH_ATTENTION_DISABLE_LOCAL=TRUE pip install . --no-build-isolation

# Disable softcapping
FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE pip install . --no-build-isolation

# Disable Pack GQA
FLASH_ATTENTION_DISABLE_PACKGQA=TRUE pip install . --no-build-isolation

# Disable varlen support
FLASH_ATTENTION_DISABLE_VARLEN=TRUE pip install . --no-build-isolation

# Disable SM8x fallback
FLASH_ATTENTION_DISABLE_SM80=TRUE pip install . --no-build-isolation

# Enable V column-major layout
FLASH_ATTENTION_ENABLE_VCOLMAJOR=TRUE pip install . --no-build-isolation
```

### FA3 Toolchain Details

FA3 downloads specific CUDA toolchain versions for optimal performance:

- **nvcc**: CUDA 12.6.85 (for front-end compilation)
- **ptxas**: CUDA 12.8.93 (for best PTX assembly optimization)

This is handled automatically by the build system. The toolchain is cached at
`~/.flashattn/nvidia/`.

For CUDA >= 13.0, the system nvcc is used directly.

---

## FlashAttention-4 Installation (CuTeDSL)

FA4 is the newest generation, written in Python using CuTeDSL.

### Installation

```bash
pip install flash-attn-4
```

For CUDA 13 (recommended for Blackwell):

```bash
pip install "flash-attn-4[cu13]"
```

### Development Install

```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
pip install -e "flash_attn/cute[dev]"
```

### Dependencies

FA4 requires:
- `nvidia-cutlass-dsl >= 4.4.1`
- `torch`
- `einops`
- `apache-tvm-ffi`
- `quack-kernels >= 0.4.0`

### Co-installation with FA2

FA4 can co-exist with FA2 in the same environment. The `flash_attn` package
uses `pkgutil.extend_path` to allow both FA2 and FA4 to provide modules under
the `flash_attn` namespace.

```python
# FA2 API
from flash_attn import flash_attn_func as fa2_func

# FA4 API
from flash_attn.cute import flash_attn_func as fa4_func
```

---

## ROCm Installation

FlashAttention-2 supports AMD GPUs via two backends: Composable Kernel (CK)
and Triton.

### Prerequisites

- ROCm 6.0+
- PyTorch for ROCm (from https://pytorch.org/get-started/locally/)
- Supported GPU: MI200, MI250, MI300, MI355, RDNA 3, RDNA 4

### Composable Kernel (CK) Backend (Default)

```bash
cd flash-attention
pip install . --no-build-isolation
```

CK backend supports:
- MI200x, MI250x, MI300x, MI355x, RDNA 3/4
- FP16 and BF16
- Head dimensions up to 256
- RDNA 3: forward only (no backward)
- RDNA 4: backward only with `deterministic=False`

Target specific GPU architectures:

```bash
# For MI300
GPU_ARCHS="gfx942" pip install . --no-build-isolation

# For MI250
GPU_ARCHS="gfx90a" pip install . --no-build-isolation

# For auto-detection
GPU_ARCHS="native" pip install . --no-build-isolation
```

Valid architectures: `native`, `gfx90a`, `gfx942`, `gfx950`, `gfx1100`,
`gfx1101`, `gfx1102`, `gfx1150`, `gfx1151`, `gfx1200`, `gfx1201`

### Triton Backend

```bash
cd flash-attention
FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" pip install --no-build-isolation .
```

Triton backend features:
- FP16, BF16, FP32 datatypes
- Forward and backward passes
- Causal masking
- Variable sequence lengths
- MQA/GQA
- Dropout
- Rotary embeddings
- ALiBi
- Paged attention
- FP8 (via FA3 interface)

To use a specific aiter commit:

```bash
cd flash-attention
cd third_party/aiter && git fetch origin && git checkout <commit-sha> && cd ../..
FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" pip install --no-build-isolation .
```

### Triton Autotuning

For peak throughput with Triton backend:

```bash
FLASH_ATTENTION_TRITON_AMD_AUTOTUNE="TRUE" pytest tests/test_flash_attn_triton_amd.py
```

This incurs a one-time warmup cost to search for optimal settings.

Custom Triton config:

```bash
FLASH_ATTENTION_FWD_TRITON_AMD_CONFIG_JSON='{"BLOCK_M":128,"BLOCK_N":64,"waves_per_eu":1,"PRE_LOAD_V":false,"num_stages":1,"num_warps":8}'
```

---

## Build System Details

### FA2 Build System (setup.py)

The FA2 build system uses `setuptools` with `torch.utils.cpp_extension` for
CUDA extension compilation.

**Build pipeline:**

1. Detect CUDA/ROCm from PyTorch
2. Download pre-built wheel if available (via `CachedWheelsCommand`)
3. Fall back to source compilation:
   a. Initialize CUTLASS submodule (`csrc/cutlass`)
   b. Generate `-gencode` flags based on target architectures
   c. Compile all CUDA kernel instantiations (60+ .cu files)
   d. Link into `flash_attn_2_cuda` shared library

**Source files compiled (FA2):**

The following kernel files are compiled, covering all combinations of:
- Head dimension: 32, 64, 96, 128, 192, 256
- Data type: fp16, bf16
- Direction: forward, backward, forward_split (inference)
- Causal mode: causal, non-causal

```
csrc/flash_attn/src/flash_fwd_hdim{32,64,96,128,192,256}_{fp16,bf16}{,_causal}_sm80.cu
csrc/flash_attn/src/flash_bwd_hdim{32,64,96,128,192,256}_{fp16,bf16}{,_causal}_sm80.cu
csrc/flash_attn/src/flash_fwd_split_hdim{32,64,96,128,192,256}_{fp16,bf16}{,_causal}_sm80.cu
```

Total: ~72 CUDA source files.

### NVCC Compilation Flags

```bash
-O3                                    # Optimization level 3
-std=c++17                             # C++17 standard
-U__CUDA_NO_HALF_OPERATORS__           # Enable half operators
-U__CUDA_NO_HALF_CONVERSIONS__         # Enable half conversions
-U__CUDA_NO_HALF2_OPERATORS__          # Enable half2 operators
-U__CUDA_NO_BFLOAT16_CONVERSIONS__     # Enable bfloat16 conversions
--expt-relaxed-constexpr               # Relaxed constexpr
--expt-extended-lambda                 # Extended lambdas
--use_fast_math                        # Fast math approximations
--threads 4                            # NVCC thread count for compilation
```

### FA3 Build System (hopper/setup.py)

FA3 uses a monkey-patched `_write_ninja_file` to route CUDA files to
appropriate architecture targets:
- Files ending in `_sm90.cu` compile with `arch=compute_90a,code=sm_90a`
- Files ending in `_sm80.cu` compile with `arch=compute_80,code=sm_80`
- Files ending in `_sm100.cu` compile with `arch=compute_100a,code=sm_100a`
- Other files compile with both SM80 and SM90 support

Additional NVCC flags for FA3:
```bash
-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED  # Required for WGMMA shapes
-DCUTLASS_ENABLE_GDC_FOR_SM90            # For PDL (Pipeline Dispatch Language)
-DCUTLASS_DEBUG_TRACE_LEVEL=0            # Debug tracing (disable for release)
-DNDEBUG                                 # Important for performance
--resource-usage                         # Print register usage
-lineinfo                                # Source mapping for profilers
```

### FA4 Build System

FA4 uses CuTeDSL JIT compilation. There is no CUDA C++ build step at install
time. Kernels are compiled on first use (JIT) and cached.

**Build steps:**
1. Install Python dependencies
2. No CUDA compilation at install time
3. Kernels are JIT-compiled when first called
4. Compiled kernels are cached in memory (LRU) and optionally on disk

### MAX_JOBS and Compilation Parallelism

The `MAX_JOBS` environment variable controls parallel compilation jobs:

```bash
# Default: auto-calculated based on CPU cores and available RAM
# Manual override:
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

**Auto-calculation logic** (in `NinjaBuildExtension`):
1. `max_num_jobs_cores = cpu_count // 2`
2. `max_num_jobs_memory = available_ram_gb / (5 * nvcc_threads)`
3. `max_jobs = min(max_num_jobs_cores, max_num_jobs_memory)`

Each NVCC compilation job uses approximately 3-5 GB of RAM.

### NVCC_THREADS

Controls the number of threads NVCC uses for a single compilation unit:

```bash
NVCC_THREADS=4 pip install flash-attn --no-build-isolation  # Default
NVCC_THREADS=8 pip install flash-attn --no-build-isolation  # More threads per file
```

---

## Environment Variables

### FA2 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASH_ATTN_CUDA_ARCHS` | `80;90;100;110;120` | Target CUDA architectures (semicolon-separated) |
| `FLASH_ATTENTION_FORCE_BUILD` | `FALSE` | Force source build (skip pre-built wheel) |
| `FLASH_ATTENTION_SKIP_CUDA_BUILD` | `FALSE` | Skip CUDA compilation (for sdist) |
| `FLASH_ATTENTION_FORCE_CXX11_ABI` | `FALSE` | Force C++11 ABI |
| `MAX_JOBS` | Auto | Maximum parallel compilation jobs |
| `NVCC_THREADS` | `4` | Threads per NVCC invocation |
| `FLASH_ATTN_LOCAL_VERSION` | None | Append local version suffix |
| `BUILD_TARGET` | `auto` | Build target: `auto`, `cuda`, or `rocm` |
| `FLASH_ATTENTION_TRITON_AMD_ENABLE` | `FALSE` | Use Triton backend for ROCm |
| `OPT_DIM` | `32,64,128,256` | Head dimensions to optimize for CK |

### FA2 Feature Disable Flags (Build Time)

| Variable | Effect |
|----------|--------|
| `FLASHATTENTION_DISABLE_BACKWARD` | Disable backward pass |
| `FLASHATTENTION_DISABLE_DROPOUT` | Disable dropout support |
| `FLASHATTENTION_DISABLE_ALIBI` | Disable ALiBi support |
| `FLASHATTENTION_DISABLE_SOFTCAP` | Disable softcapping |
| `FLASHATTENTION_DISABLE_LOCAL` | Disable sliding window attention |
| `FLASHATTENTION_DISABLE_UNEVEN_K` | Disable uneven K |

### FA3 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASH_ATTENTION_FORCE_BUILD` | `FALSE` | Force source build |
| `FLASH_ATTENTION_SKIP_CUDA_BUILD` | `FALSE` | Skip CUDA compilation |
| `FLASH_ATTENTION_FORCE_CXX11_ABI` | `FALSE` | Force C++11 ABI |
| `FLASH_ATTENTION_DISABLE_BACKWARD` | `FALSE` | Disable backward pass |
| `FLASH_ATTENTION_DISABLE_SPLIT` | `FALSE` | Disable SplitKV |
| `FLASH_ATTENTION_DISABLE_PAGEDKV` | `FALSE` | Disable paged KV |
| `FLASH_ATTENTION_DISABLE_APPENDKV` | `FALSE` | Disable append KV |
| `FLASH_ATTENTION_DISABLE_LOCAL` | `FALSE` | Disable local attention |
| `FLASH_ATTENTION_DISABLE_SOFTCAP` | `FALSE` | Disable softcapping |
| `FLASH_ATTENTION_DISABLE_PACKGQA` | `FALSE` | Disable PackGQA |
| `FLASH_ATTENTION_DISABLE_FP16` | `FALSE` | Disable FP16 |
| `FLASH_ATTENTION_DISABLE_FP8` | `FALSE` | Disable FP8 forward |
| `FLASH_ATTENTION_DISABLE_VARLEN` | `FALSE` | Disable varlen |
| `FLASH_ATTENTION_DISABLE_CLUSTER` | `FALSE` | Disable cluster (CTA) |
| `FLASH_ATTENTION_DISABLE_HDIM64` | `FALSE` | Disable hdim=64 |
| `FLASH_ATTENTION_DISABLE_HDIM96` | `FALSE` | Disable hdim=96 |
| `FLASH_ATTENTION_DISABLE_HDIM128` | `FALSE` | Disable hdim=128 |
| `FLASH_ATTENTION_DISABLE_HDIM192` | `FALSE` | Disable hdim=192 |
| `FLASH_ATTENTION_DISABLE_HDIM256` | `FALSE` | Disable hdim=256 |
| `FLASH_ATTENTION_DISABLE_SM80` | `FALSE` | Disable SM8x fallback |
| `FLASH_ATTENTION_ENABLE_VCOLMAJOR` | `FALSE` | Enable V column-major |
| `FLASH_ATTENTION_DISABLE_HDIMDIFF64` | `FALSE` | Disable hdim=64 w/ diff V dim |
| `FLASH_ATTENTION_DISABLE_HDIMDIFF192` | `FALSE` | Disable hdim=192 w/ diff V dim |
| `FLASH_ATTENTION_TRITON_AMD_ENABLE` | `FALSE` | Use Triton ROCm backend |
| `FLASH_ATTENTION_OFFLINE_BUILD` | `FALSE` | Offline build (no downloads) |
| `FLASH_ATTENTION_HOME` | `~` | Cache directory |

### FA3 Runtime Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASH_ATTENTION_TRITON_AMD_ENABLE` | `FALSE` | Use Triton AMD backend at runtime |

### FA4 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASH_ATTENTION_ARCH` | Auto-detect | Override kernel architecture selection (e.g., `sm_90`) |
| `FLASH_ATTENTION_FAKE_TENSOR` | `0` | Use FakeTensorMode for compilation without GPU |
| `FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED` | `0` | Enable persistent disk cache |
| `CUTE_DSL_KEEP_PTX` | None | Keep intermediate PTX files |
| `CUTE_DSL_PTXAS_PATH` | None | Custom ptxas binary path |
| `CUTE_DSL_LINEINFO` | None | Add line info to PTX |
| `CUTE_CUBIN_PATH` | None | Dump CUBIN/SASS to path |

### FA4 Runtime Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `FA_LOG_LEVEL` | `0` | Logging level (0=none, 1=info, 2=debug) |

### ROCm Triton Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASH_ATTENTION_TRITON_AMD_ENABLE` | `FALSE` | Enable Triton backend |
| `FLASH_ATTENTION_TRITON_AMD_AUTOTUNE` | `FALSE` | Enable autotuning for best perf |
| `FLASH_ATTENTION_FWD_TRITON_AMD_CONFIG_JSON` | None | Custom Triton kernel config |

---

## Docker Setup

### NVIDIA Docker (FA2)

```dockerfile
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

WORKDIR /workspace

# Install build dependencies
RUN pip install ninja packaging psutil

# Install FlashAttention-2
RUN pip install flash-attn --no-build-isolation

# Verify
RUN python -c "import flash_attn; print(f'FlashAttention version: {flash_attn.__version__}')"
```

Build and run:
```bash
docker build -t flash-attn .
docker run --gpus all -it flash-attn
```

### NVIDIA Docker (FA3)

```dockerfile
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

WORKDIR /workspace

# Install build dependencies
RUN pip install ninja packaging

# Clone and install FA3
RUN git clone https://github.com/Dao-AILab/flash-attention.git && \
    cd flash-attention/hopper && \
    python setup.py install

# Verify
RUN python -c "import flash_attn_interface; print('FA3 installed successfully')"
```

### NVIDIA Docker (FA4)

```dockerfile
FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel

WORKDIR /workspace

# Install FA4
RUN pip install flash-attn-4

# Verify
RUN python -c "from flash_attn.cute import flash_attn_func; print('FA4 installed')"
```

### ROCm Docker (Triton Backend)

```dockerfile
FROM rocm/pytorch:latest

WORKDIR /workspace

# Build flash attention with triton backend
RUN git clone https://github.com/Dao-AILab/flash-attention && \
    cd flash-attention && \
    FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" pip install --no-build-isolation .

# Set environment variable to use triton backend
ENV FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
```

Build and run:
```bash
docker build -t flash-attn-triton .
docker run -it --network=host --user root --group-add video \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --ipc=host --shm-size 16G \
    --device=/dev/kfd --device=/dev/dri flash-attn-triton
```

### NGC Container (Recommended for NVIDIA)

The NVIDIA NGC PyTorch container includes all necessary build tools:

```bash
# Pull the latest NGC PyTorch container
docker pull nvcr.io/nvidia/pytorch:24.07-py3

# Run with GPU access
docker run --gpus all -it nvcr.io/nvidia/pytorch:24.07-py3

# Inside the container
pip install flash-attn --no-build-isolation
```

---

## Verifying Installation

### FA2 Verification

```python
import torch
import flash_attn
from flash_attn import flash_attn_func

print(f"FlashAttention version: {flash_attn.__version__}")

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")

# Simple test
batch, seqlen, heads, dim = 2, 128, 8, 64
q = torch.randn(batch, seqlen, heads, dim, device='cuda', dtype=torch.float16)
k = torch.randn(batch, seqlen, heads, dim, device='cuda', dtype=torch.float16)
v = torch.randn(batch, seqlen, heads, dim, device='cuda', dtype=torch.float16)

out = flash_attn_func(q, k, v)
print(f"Output shape: {out.shape}")  # Expected: (2, 128, 8, 64)
print(f"Output dtype: {out.dtype}")  # Expected: torch.float16
print("FA2 installation verified successfully!")
```

### FA3 Verification

```python
import torch

# FA3 is imported differently
import flash_attn_interface

print("FlashAttention-3 interface loaded")

# Simple test
batch, seqlen, heads, dim = 2, 128, 8, 128
q = torch.randn(batch, seqlen, heads, dim, device='cuda', dtype=torch.bfloat16)
k = torch.randn(batch, seqlen, heads, dim, device='cuda', dtype=torch.bfloat16)
v = torch.randn(batch, seqlen, heads, dim, device='cuda', dtype=torch.bfloat16)

out = flash_attn_interface.flash_attn_func(q, k, v)
print(f"Output shape: {out.shape}")
print("FA3 installation verified successfully!")
```

### FA4 Verification

```python
import torch
from flash_attn.cute import flash_attn_func

print("FlashAttention-4 (CuTeDSL) loaded")

# Simple test
batch, seqlen, heads, dim = 2, 128, 8, 128
q = torch.randn(batch, seqlen, heads, dim, device='cuda', dtype=torch.bfloat16)
k = torch.randn(batch, seqlen, heads, dim, device='cuda', dtype=torch.bfloat16)
v = torch.randn(batch, seqlen, heads, dim, device='cuda', dtype=torch.bfloat16)

out = flash_attn_func(q, k, v, causal=True)
print(f"Output shape: {out.shape}")
print("FA4 installation verified successfully!")
```

### Running Tests

FA2:
```bash
pytest -q -s tests/test_flash_attn.py
```

FA3:
```bash
cd hopper
export PYTHONPATH=$PWD
pytest -q -s test_flash_attn.py
```

FA4:
```bash
pytest tests/cute/test_flash_attn.py -x
```

FA4 fast two-pass testing (parallel compilation + cached execution):
```bash
# Pass 1: compile all kernels in parallel (no GPU needed)
FLASH_ATTENTION_FAKE_TENSOR=1 FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 \
    pytest -n 64 -x tests/cute/test_flash_attn.py

# Pass 2: run tests with cached kernels
FLASH_ATTENTION_FAKE_TENSOR=0 FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 \
    pytest -x tests/cute/test_flash_attn.py
```

---

## Compatibility Matrix

### GPU x CUDA x PyTorch (FA2)

| GPU | Architecture | CUDA 11.8 | CUDA 12.0 | CUDA 12.1 | CUDA 12.3 | CUDA 12.4+ |
|-----|-------------|-----------|-----------|-----------|-----------|------------|
| A100 | SM80 | PyTorch 2.2+ | Yes | Yes | Yes | Yes |
| A10G | SM86 | PyTorch 2.2+ | Yes | Yes | Yes | Yes |
| RTX 3090 | SM86 | PyTorch 2.2+ | Yes | Yes | Yes | Yes |
| RTX 4090 | SM89 | PyTorch 2.2+ | Yes | Yes | Yes | Yes |
| H100 | SM90 | N/A | Yes | Yes | Yes | Yes |
| B200 | SM100 | N/A | N/A | N/A | N/A | Yes (limited) |

### GPU x CUDA x PyTorch (FA3)

| GPU | Architecture | CUDA 12.3 | CUDA 12.4 | CUDA 12.6 | CUDA 12.8 | CUDA 13.0+ |
|-----|-------------|-----------|-----------|-----------|-----------|------------|
| H100 | SM90 | PyTorch 2.4+ | Yes | Yes | Recommended | Yes |
| H800 | SM90 | PyTorch 2.4+ | Yes | Yes | Recommended | Yes |

FA3 requires SM90 (Hopper) GPUs. It does not support Ampere or earlier.

### GPU x CUDA x PyTorch (FA4)

| GPU | Architecture | CUDA 12.x | CUDA 13.0+ |
|-----|-------------|-----------|------------|
| A100 | SM80 | Yes | Yes |
| H100 | SM90 | Yes | Yes |
| B200 | SM100 | Yes | Yes (recommended) |
| B100 | SM100 | Yes | Yes |
| Thor | SM110 | Limited | Yes |
| DGX Spark | SM120 | Limited | Yes |

### Data Type Support Matrix

| Data Type | FA2 | FA3 | FA4 |
|-----------|-----|-----|-----|
| FP16 | Yes | Yes | Yes |
| BF16 | Yes (SM80+) | Yes | Yes |
| FP32 | No | No | No |
| FP8 E4M3 | No | Yes (forward only) | Yes (SM100, forward only) |
| FP8 E5M2 | No | No | Yes (SM100, forward only) |
| INT8 | No | No | No |

---

## Troubleshooting Installation Issues

### Issue: ninja not found or not working

**Symptoms:**
```
RuntimeError: Ninja is not installed
```
or compilation takes 1-2 hours.

**Solution:**
```bash
pip uninstall -y ninja && pip install ninja
# Verify:
ninja --version && echo $?
```

### Issue: CUDA_HOME not set

**Symptoms:**
```
UserWarning: flash_attn was requested, but nvcc was not found.
```

**Solution:**
```bash
export CUDA_HOME=/usr/local/cuda
# Or use the NVIDIA NGC container where CUDA_HOME is pre-set
```

### Issue: Out of memory during compilation

**Symptoms:**
```
c++: fatal error: Killed program cc1plus
```

**Solution:**
```bash
MAX_JOBS=2 pip install flash-attn --no-build-isolation
```

Reduce `MAX_JOBS` based on available RAM. Each compilation job uses ~3-5 GB.

### Issue: PyTorch version mismatch

**Symptoms:**
```
RuntimeError: FlashAttention is only supported on CUDA 11.7 and above
```
or pre-built wheel not found.

**Solution:**
Ensure PyTorch is installed with CUDA support:
```bash
python -c "import torch; print(torch.version.cuda)"
# Should print a CUDA version, not None
```

### Issue: Incompatible CUDA toolkit version

**Symptoms:**
```
RuntimeError: FlashAttention-3 is only supported on CUDA 12.3 and above
```

**Solution:**
```bash
nvcc --version  # Check installed CUDA version
# Upgrade CUDA toolkit or use a compatible container
```

### Issue: ROCm build failure with CK

**Symptoms:**
```
AssertionError: csrc/composable_kernel is missing
```

**Solution:**
```bash
cd flash-attention
git submodule update --init csrc/composable_kernel
pip install . --no-build-isolation
```

### Issue: FA4 CuTeDSL import error

**Symptoms:**
```
ModuleNotFoundError: No module named 'cutlass'
```

**Solution:**
```bash
pip install nvidia-cutlass-dsl>=4.4.1
```

### Issue: FA4 compilation error on non-GPU machine

**Solution:**
Set the architecture override for CPU-only compilation:
```bash
FLASH_ATTENTION_ARCH=sm_80 CUTE_DSL_ARCH=sm_80 pip install flash-attn-4
```

### Issue: Pre-built wheel download fails

**Symptoms:**
```
urllib.error.HTTPError: HTTP Error 404: Not Found
Precompiled wheel not found. Building from source...
```

**Solution:**
This is normal behavior. The build system automatically falls back to source
compilation. No action needed unless source compilation also fails.

### Issue: Windows compilation

FlashAttention primarily targets Linux. Windows support is experimental since
v2.3.2. If you encounter Windows issues:

1. Use WSL2 (Windows Subsystem for Linux)
2. Use Docker with GPU passthrough
3. Try the Windows-specific build flags (handled automatically in setup.py)

### Issue: Multiple CUDA versions

If you have multiple CUDA installations:

```bash
# Check which CUDA PyTorch was built with
python -c "import torch; print(torch.version.cuda)"

# Set CUDA_HOME to match
export CUDA_HOME=/usr/local/cuda-12.3  # Match PyTorch's CUDA version
```

### Issue: FlashAttention and torch.compile

FA2 requires PyTorch >= 2.4 for `torch.compile` support (uses `custom_op` API).

```python
# Check if torch.compile support is available
import torch
print(torch.__version__)  # Must be >= 2.4.0
```

### Issue: "cannot import name 'flash_attn_2_cuda'"

**Symptoms:**
```
ImportError: cannot import name 'flash_attn_2_cuda'
```

**Causes and solutions:**

1. **Wrong PyTorch CUDA version**: The wheel was built for a different CUDA version
   ```bash
   pip uninstall flash-attn
   pip install flash-attn --no-build-isolation  # Force rebuild
   ```

2. **ABI mismatch**: The wheel was built with different C++ ABI
   ```bash
   FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn --no-build-isolation
   ```

3. **GPU architecture mismatch**: The wheel doesn't include your GPU arch
   ```bash
   FLASH_ATTN_CUDA_ARCHS="native" pip install flash-attn --no-build-isolation
   ```
