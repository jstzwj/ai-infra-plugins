# 02 - Installation and Setup

## Prerequisites

JAX requires:

- **Python**: 3.10 or later (3.11 and 3.12 recommended; 3.13 supported on recent releases)
- **Operating System**: Linux (primary), macOS (CPU and experimental Metal), Windows (CPU only)
- **pip**: 20.3+ (for manylinux wheels)

For GPU support, additional requirements apply (see GPU-specific sections below).

## Quick Install

The fastest way to get JAX running:

```bash
# CPU only (works everywhere)
pip install jax

# NVIDIA GPU with CUDA 12
pip install jax[cuda12]

# Google TPU
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

Verify the installation:

```python
import jax
import jax.numpy as jnp

print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")

# Quick test
x = jnp.ones(3)
y = jnp.dot(x, x)
y.block_until_ready()
print(f"Test computation: {y}")  # Should print 3.0
```

## CPU Installation

### pip

```bash
pip install --upgrade pip
pip install jax
```

The CPU-only installation includes the `jaxlib` package with XLA's CPU backend, LLVM-based code generation, and bundled OpenBLAS for linear algebra.

### Conda / Miniconda

```bash
conda install -c conda-forge jax
```

Or with mamba (faster dependency resolution):

```bash
mamba install -c conda-forge jax
```

### Verifying CPU Installation

```python
import jax

assert jax.default_backend() == "cpu"
devices = jax.devices("cpu")
print(f"CPU devices: {devices}")
# Output: [CpuDevice(id=0)]

# Check version
print(f"JAX: {jax.__version__}")       # 0.6.1
print(f"jaxlib: {jax.lib.version}")     # 0.6.1
```

## NVIDIA GPU Installation (CUDA 12)

### Requirements

| Component | Minimum Version | Recommended |
|-----------|----------------|-------------|
| NVIDIA GPU | Compute capability 5.2+ (Maxwell) | Ampere (8.0+) or Hopper (9.0+) |
| CUDA driver | 525.60.13+ | 550.x+ |
| CUDA toolkit | Not required (bundled in jaxlib) | N/A |
| cuDNN | Not required (bundled in jaxlib) | N/A |
| Python | 3.10+ | 3.11 or 3.12 |
| Linux | x86_64 or aarch64 | x86_64 |

**Note**: The `jax[cuda12]` package bundles the CUDA runtime, cuBLAS, cuDNN, and cuFFT. You do NOT need to install the CUDA toolkit separately. Only the NVIDIA driver is required.

### pip Install (Recommended)

```bash
pip install --upgrade pip
pip install jax[cuda12]
```

The `jax[cuda12]` extra installs:
- `jax` - Core JAX library
- `jaxlib` - C++ extension with CUDA 12 support
- `nvidia-cuda-nvcc` - CUDA compiler (nvcc)
- `nvidia-cuda-runtime` - CUDA runtime libraries
- `nvidia-cublas` - CUDA Basic Linear Algebra
- `nvidia-cudnn` - CUDA Deep Neural Network library
- `nvidia-cufft` - CUDA Fast Fourier Transform library
- `nvidia-cusolver` - CUDA solver library
- `nvidia-cusparse` - CUDA sparse matrix library

### Verifying GPU Installation

```python
import jax
import jax.numpy as jnp

# Check GPU availability
print(f"Default backend: {jax.default_backend()}")  # Should be "gpu"
print(f"GPU devices: {jax.devices('gpu')}")

# Device details
for device in jax.devices():
    print(f"  {device.id}: {device.device_kind}")
    print(f"  Memory: {device.memory_stats()}")

# Quick GPU computation
x = jnp.ones((1000, 1000))
y = jnp.dot(x, x)
y.block_until_ready()  # Wait for async GPU computation
print(f"GPU computation successful: {y[0, 0]}")  # 1000.0
```

### Multi-GPU Support

JAX automatically detects all available GPUs. No special configuration is needed for multi-GPU setups:

```python
import jax

print(f"Number of GPUs: {jax.device_count()}")
print(f"Local GPUs: {jax.local_device_count()}")
print(f"All devices: {jax.devices()}")

# Individual GPU access
for i, device in enumerate(jax.devices()):
    print(f"GPU {i}: {device.device_kind}")
```

For distributed multi-host training, see the distributed computing section.

### CUDA 12 Local Installation

If you have CUDA 12 installed locally and want JAX to use it instead of the bundled version:

```bash
pip install jax[cuda12_local]
```

This variant expects CUDA libraries to be available on `LD_LIBRARY_PATH` rather than installing them via pip.

### Troubleshooting NVIDIA GPU

**Error: "ExternalSetupError: jaxlib build requires CUDA"**

Make sure you installed the CUDA extras:
```bash
pip install jax[cuda12]  # Not just "pip install jax"
```

**Error: "CUDA version is insufficient"**

Update your NVIDIA driver:
```bash
nvidia-smi  # Check driver version and CUDA capability
# Driver version should be 525.60.13 or later for CUDA 12
```

**Error: "No GPU/TPU found, falling back to CPU"**

```bash
# Check that NVIDIA GPU is visible
nvidia-smi

# Check CUDA libraries
ldconfig -p | grep libcuda
ldconfig -p | grep libcublas

# Try setting the platform explicitly
export JAX_PLATFORMS=cuda
python -c "import jax; print(jax.devices())"
```

**Error: "RuntimeError: Unable to load cuDNN"**

```bash
# Reinstall with CUDA extras
pip install --force-reinstall jax[cuda12]
```

**Memory preallocation issues**

JAX preallocates 75% of GPU memory by default. To change this:

```python
# Option 1: In Python code (must be before importing JAX)
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"  # Use 50% of GPU memory

# Option 2: Via jax.config
import jax
jax.config.update("xla_python_client_mem_fraction", 0.5)

# Option 3: Preallocate specific amount
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "8G"  # 8 GB
```

## AMD GPU Installation (ROCm)

### Requirements

| Component | Minimum Version | Recommended |
|-----------|----------------|-------------|
| AMD GPU | MI200 series (gfx90a), MI300 series (gfx942) | MI300A/MI300X |
| ROCm | 6.0+ | 6.2+ |
| Linux | x86_64 | x86_64 |

### pip Install

```bash
pip install jax[rocm] -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

Or for nightly builds:

```bash
pip install jax[rocm] -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html
```

### Docker (Recommended for AMD)

```bash
docker pull rocm/jax-community:latest
```

### Verifying AMD GPU Installation

```python
import jax
print(f"Default backend: {jax.default_backend()}")  # Should be "gpu" or "rocm"
print(f"Devices: {jax.devices()}")
```

### Known Limitations

- Not all JAX operations are optimized for AMD GPUs
- Pallas GPU kernels may have limited support
- Performance may differ from NVIDIA GPUs
- Some XLA compiler optimizations may not be available

## Google TPU Installation

### TPU VM (Recommended)

On Google Cloud TPU VMs, install JAX with TPU support:

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

### Google Colab

In Google Colab, select "TPU" as the runtime type, then:

```python
import jax
# Colab TPU runtime already has JAX and libtpu installed
print(f"Devices: {jax.devices()}")
# Output: [TpuDevice(id=0), TpuDevice(id=1), ..., TpuDevice(id=7)]
```

### TPU Pod Setup

For multi-host TPU Pods, each TPU VM worker needs the same JAX version:

```bash
# Run on all TPU VM workers
pip install jax[tpu]==0.6.1 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

### Verifying TPU Installation

```python
import jax
import jax.numpy as jnp

print(f"Default backend: {jax.default_backend()}")  # "tpu"
print(f"TPU devices: {jax.devices('tpu')}")
print(f"Number of TPU chips: {jax.device_count()}")

# TPU topology
import jax._src.lib.tpu_driver as tpu_driver
# For TPU v4: 4 chips per device, each with 2 cores
```

## Apple Mac GPU (Experimental)

### Requirements

- Apple Silicon (M1, M2, M3, M4) or Apple GPU
- macOS 13.0+
- Experimental: API stability not guaranteed

### Installation

```bash
pip install jax-metal
```

Or:

```bash
pip install jax -f https://storage.googleapis.com/jax-releases/jax_releases.html
# Plus jax-metal plugin
pip install jax-metal
```

### Usage

```python
import jax
# Mac GPU should be auto-detected
print(f"Backend: {jax.default_backend()}")  # "METAL" or "cpu"

# Force Metal backend
import os
os.environ["JAX_PLATFORMS"] = "metal"
```

### Limitations

- Many operations are not yet optimized
- Performance is generally lower than NVIDIA GPU
- Not all JAX features are supported
- bfloat16 may have limited support

## Intel GPU (Experimental)

### Requirements

- Intel Data Center GPU Max (Ponte Vecchio) or Intel Arc GPU
- Intel oneAPI Base Toolkit

### Installation

```bash
pip install jax
pip install intel-extension-for-openxla
```

### Usage

```python
import os
os.environ["JAX_PLATFORMS"] = "xpu"  # Intel GPU backend

import jax
print(f"Backend: {jax.default_backend()}")
```

## Building from Source

Building JAX from source is rarely needed but may be required for:
- Development contributions
- Custom hardware backends
- Debugging JAX internals

### Building jaxlib

```bash
# Clone the JAX repository
git clone https://github.com/jax-ml/jax.git
cd jax

# Install build dependencies
pip install numpy cython pybind11

# Build jaxlib (CPU only)
python build/build.py

# Install the built wheel
pip install dist/jaxlib-*.whl

# Install jax (editable mode)
pip install -e .
```

### Building jaxlib with CUDA Support

```bash
# Set CUDA path
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH

# Build with CUDA
python build/build.py --enable_cuda

# Specify CUDA compute capability
python build/build.py --enable_cuda --cuda_compute_capability=sm_80
```

### Build Options

```bash
python build/build.py --help

# Common options:
--enable_cuda          # Build with CUDA support
--enable_rocm          # Build with ROCm support
--enable_tpu           # Build with TPU support
--cuda_compute_capability  # Target GPU architecture (sm_70, sm_80, sm_90, etc.)
--bazel_options        # Additional Bazel build options
--target_cpu_features  # CPU feature flags (avx2, avx512, etc.)
```

### Pre-release / Development Builds

```bash
# Nightly builds (latest development version)
pip install --pre jax jaxlib -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html

# Specific nightly date
pip install jax==0.6.1.dev20250501 jaxlib==0.6.1.dev20250501 \
    -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html
```

## Docker Containers

### Official JAX Docker Images

```bash
# CPU only
docker pull us-docker.pkg.dev/jax-ml/jax/jax:latest

# GPU (CUDA 12)
docker pull us-docker.pkg.dev/jax-ml/jax/jax-cuda12:latest

# Specific version
docker pull us-docker.pkg.dev/jax-ml/jax/jax-cuda12:0.6.1
```

### Running JAX Docker Containers

```bash
# CPU
docker run -it us-docker.pkg.dev/jax-ml/jax/jax:latest python -c "import jax; print(jax.devices())"

# GPU (requires nvidia-docker)
docker run --gpus all -it us-docker.pkg.dev/jax-ml/jax/jax-cuda12:latest \
    python -c "import jax; print(jax.devices())"

# Interactive session
docker run --gpus all -it us-docker.pkg.dev/jax-ml/jax/jax-cuda12:latest bash
```

### Custom Dockerfile

```dockerfile
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3-pip && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install jax[cuda12]

# Copy your application
COPY . /app
WORKDIR /app

CMD ["python3", "train.py"]
```

## Conda Installation

```bash
# Create a conda environment
conda create -n jax-env python=3.11 -y
conda activate jax-env

# Install JAX (CPU)
conda install -c conda-forge jax

# Install JAX with GPU support
conda install -c conda-forge jaxlib=*=*cuda*
```

**Note**: The conda-forge JAX packages may lag behind the PyPI releases. For the latest version, prefer pip installation.

## Nightly Builds

Nightly builds contain the latest development code and may include unreleased features:

```bash
# Latest nightly (CPU)
pip install --pre jax jaxlib -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html

# Latest nightly (CUDA 12)
pip install --pre jax jaxlib -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html
pip install jax[cuda12]
```

**Warning**: Nightly builds may be unstable. Use them for testing new features but not for production code.

## Platform Compatibility Table

| Platform | Architecture | Accelerator | Support Level | Python |
|----------|-------------|-------------|---------------|--------|
| Linux | x86_64 | CPU | Stable | 3.10-3.13 |
| Linux | x86_64 | NVIDIA GPU (CUDA 12) | Stable | 3.10-3.13 |
| Linux | x86_64 | AMD GPU (ROCm 6) | Stable | 3.10-3.13 |
| Linux | x86_64 | Google TPU | Stable | 3.10-3.13 |
| Linux | aarch64 | CPU | Stable | 3.10-3.13 |
| Linux | aarch64 | NVIDIA GPU (CUDA 12) | Beta | 3.10-3.13 |
| macOS | arm64 (Apple Silicon) | CPU | Stable | 3.10-3.13 |
| macOS | arm64 (Apple Silicon) | Metal GPU | Experimental | 3.10-3.13 |
| macOS | x86_64 (Intel) | CPU | Stable | 3.10-3.13 |
| Windows | x86_64 | CPU | Stable | 3.10-3.13 |
| Windows | x86_64 | NVIDIA GPU | Community | 3.10-3.13 |
| Linux | x86_64 | Intel GPU (XPU) | Experimental | 3.10-3.13 |

## Environment Variables for Configuration

JAX uses environment variables for configuration. These must typically be set **before** importing JAX.

### Core Configuration

```bash
# Force specific backend (comma-separated priority list)
export JAX_PLATFORMS=cpu          # Force CPU
export JAX_PLATFORMS=cuda,cpu     # Prefer CUDA, fallback to CPU
export JAX_PLATFORMS=tpu          # Force TPU

# Enable 64-bit floating point (disabled by default)
export JAX_ENABLE_X64=True

# Traceback filtering
export JAX_TRACEBACK_FILTERING=off       # Show full tracebacks
export JAX_TRACEBACK_FILTERING=clang     # Filter to JAX-specific frames (default)
export JAX_TRACEBACK_FILTERING=numba     # Numba-style filtering
export JAX_TRACEBACK_FILTERING=hide_frames  # Hide internal frames

# Disable JIT (for debugging)
export JAX_DISABLE_JIT=True
```

### Memory Configuration

```bash
# GPU memory preallocation fraction (default: 0.75)
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5

# GPU memory preallocation in bytes
export XLA_PYTHON_CLIENT_MEM_FRACTION="8589934592"  # 8 GB

# Enable CUDA memory allocator (alternative to BFC allocator)
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda

# Preallocate only what is needed (dynamic allocation)
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

### Distributed Computing

```bash
# Number of CPU devices to simulate (for testing multi-device code)
export XLA_FLAGS="--xla_force_host_platform_device_count=8"

# Distributed coordinator address
export JAX_COORDINATOR_IP=10.0.0.1
export JAX_COORDINATOR_PORT=6000

# Process ID for distributed JAX
export JAX_PROCESS_ID=0

# Number of total processes
export JAX_NUM_PROCESSES=4
```

### XLA Compiler Flags

```bash
# XLA compiler options
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
export XLA_FLAGS="--xla_gpu_autotune_level=4"

# Enable HLO dumping for debugging
export XLA_FLAGS="--xla_dump_to=/tmp/hlo_dumps"
export XLA_FLAGS="--xla_dump_hlo_as_text"
```

### Debugging

```bash
# Enable NaN checking during computation
export JAX_DEBUG_NANS=True

# Check invariants during tracing
export JAX_CHECK_TRACER_LEAKS=True

# Enable logging
export JAX_LOG_COMPILES=True          # Log each compilation
export JAX_LOG_SIZE_RETAINING_COMPILED_CACHE=True
export TF_CPP_MIN_LOG_LEVEL=0         # Verbose XLA logging
```

### Setting Environment Variables in Python

You can also set some configuration options in Python, but this must be done **before** importing JAX or at the very start of your script:

```python
# Method 1: os.environ (before import)
import os
os.environ["JAX_ENABLE_X64"] = "True"
os.environ["JAX_PLATFORMS"] = "cpu"

import jax

# Method 2: jax.config (after import, for some options)
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", False)

# Method 3: config file (~/.config/jax/config.py)
# JAX reads this file on startup if it exists
```

## Verification Steps

After installing JAX, run these verification checks to confirm everything is working:

### Basic Verification

```python
import jax
import jax.numpy as jnp

# 1. Version check
print(f"JAX version: {jax.__version__}")
assert jax.__version__.startswith("0.6"), f"Unexpected version: {jax.__version__}"

# 2. Backend check
backend = jax.default_backend()
print(f"Default backend: {backend}")

# 3. Device listing
devices = jax.devices()
print(f"Devices ({len(devices)}): {devices}")

# 4. Basic computation
x = jnp.array([1.0, 2.0, 3.0])
y = jnp.sum(x ** 2)
y.block_until_ready()
assert float(y) == 14.0
print(f"Basic computation: PASS")
```

### JIT Compilation Verification

```python
import jax

@jax.jit
def f(x):
    return x * 2 + 1

result = f(jnp.array([1.0, 2.0, 3.0]))
result.block_until_ready()
assert jnp.allclose(result, jnp.array([3.0, 5.0, 7.0]))
print("JIT compilation: PASS")
```

### Automatic Differentiation Verification

```python
import jax
import jax.numpy as jnp

def f(x):
    return jnp.sum(x ** 2)

grad_f = jax.grad(f)
x = jnp.array([1.0, 2.0, 3.0])
grads = grad_f(x)
expected = 2 * x
assert jnp.allclose(grads, expected)
print(f"Automatic differentiation: PASS (grad = {grads})")
```

### GPU Verification

```python
import jax
import jax.numpy as jnp

if jax.default_backend() == "gpu":
    # GPU computation
    x = jnp.ones((10000, 10000))
    y = jnp.dot(x, x)
    y.block_until_ready()
    print(f"GPU computation: PASS ({y[0, 0]})")

    # GPU memory info
    device = jax.devices()[0]
    stats = device.memory_stats()
    if stats:
        print(f"GPU memory used: {stats['bytes_in_use'] / 1e9:.2f} GB")
        print(f"GPU memory limit: {stats['bytes_limit'] / 1e9:.2f} GB")
else:
    print(f"No GPU found, using backend: {jax.default_backend()}")
```

### Multi-Device Verification

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

num_devices = jax.device_count()
print(f"Number of devices: {num_devices}")

if num_devices > 1:
    devices = jax.devices()
    mesh = Mesh(devices, ("devices",))
    sharding = NamedSharding(mesh, P("devices",))

    x = jnp.arange(num_devices * 4.0)
    x_sharded = jax.device_put(x, sharding)
    print(f"Sharded array: {x_sharded}")
    print(f"Sharding: {x_sharded.sharding}")
    print("Multi-device sharding: PASS")
else:
    print("Single device detected. Multi-device tests skipped.")
```

### Complete Verification Script

```python
"""
Complete JAX installation verification script.
Run: python verify_jax.py
"""
import sys

def verify():
    errors = []

    # 1. Import
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as e:
        print(f"FAIL: Cannot import JAX: {e}")
        sys.exit(1)

    print(f"JAX version: {jax.__version__}")
    print(f"jaxlib version: {jax.lib.version}")
    print(f"Default backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")
    print(f"Device count: {jax.device_count()}")

    # 2. Basic ops
    try:
        x = jnp.ones(10)
        assert x.sum() == 10.0
        print("[PASS] Basic array operations")
    except Exception as e:
        errors.append(f"Basic ops: {e}")
        print(f"[FAIL] Basic array operations: {e}")

    # 3. JIT
    try:
        @jax.jit
        def add_one(x):
            return x + 1.0
        result = add_one(jnp.zeros(5))
        assert jnp.allclose(result, jnp.ones(5))
        print("[PASS] JIT compilation")
    except Exception as e:
        errors.append(f"JIT: {e}")
        print(f"[FAIL] JIT compilation: {e}")

    # 4. grad
    try:
        grad_fn = jax.grad(lambda x: x ** 2)
        assert jnp.allclose(grad_fn(3.0), 6.0)
        print("[PASS] Automatic differentiation")
    except Exception as e:
        errors.append(f"grad: {e}")
        print(f"[FAIL] Automatic differentiation: {e}")

    # 5. vmap
    try:
        vmapped = jax.vmap(lambda x: x ** 2)
        result = vmapped(jnp.array([1.0, 2.0, 3.0]))
        assert jnp.allclose(result, jnp.array([1.0, 4.0, 9.0]))
        print("[PASS] vmap (auto-vectorization)")
    except Exception as e:
        errors.append(f"vmap: {e}")
        print(f"[FAIL] vmap: {e}")

    # 6. Pytree
    try:
        params = {"a": jnp.ones(3), "b": jnp.zeros(2)}
        leaves, treedef = jax.tree.flatten(params)
        restored = jax.tree.unflatten(treedef, leaves)
        assert set(restored.keys()) == {"a", "b"}
        print("[PASS] Pytree operations")
    except Exception as e:
        errors.append(f"pytree: {e}")
        print(f"[FAIL] Pytree operations: {e}")

    # 7. Random
    try:
        key = jax.random.PRNGKey(0)
        key, subkey = jax.random.split(key)
        x = jax.random.normal(subkey, (5,))
        assert x.shape == (5,)
        print("[PASS] Random number generation")
    except Exception as e:
        errors.append(f"random: {e}")
        print(f"[FAIL] Random number generation: {e}")

    # 8. GPU (if available)
    if jax.default_backend() == "gpu":
        try:
            x = jnp.ones((1000, 1000))
            y = jnp.dot(x, x)
            y.block_until_ready()
            assert y.shape == (1000, 1000)
            print("[PASS] GPU computation")
        except Exception as e:
            errors.append(f"GPU: {e}")
            print(f"[FAIL] GPU computation: {e}")
    else:
        print("[SKIP] GPU computation (no GPU detected)")

    # Summary
    print()
    if errors:
        print(f"FAILED: {len(errors)} error(s)")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED")

if __name__ == "__main__":
    verify()
```

## Common Installation Issues

### Issue: Version Mismatch Between jax and jaxlib

```bash
# Error: RuntimeError: jaxlib version is incompatible with jax version
# Fix: Upgrade both together
pip install --upgrade jax jaxlib
```

### Issue: NumPy Version Conflict

```bash
# Error: AttributeError: module 'numpy' has no attribute '...'
# Fix: Upgrade NumPy
pip install --upgrade numpy
```

### Issue: Import Error on WSL2

```bash
# Error: OSError: libcuda.so.1: cannot open shared object file
# Fix: Ensure NVIDIA GPU drivers are properly installed on Windows
# The CUDA libraries should be available through the Windows driver

# Verify CUDA is available:
ls /usr/lib/wsl/lib/libcuda.so*
```

### Issue: SSL Certificate Errors

```bash
# Error: SSL: CERTIFICATE_VERIFY_FAILED
# Fix: Install certifi or use --trusted-host
pip install jax --trusted-host pypi.org --trusted-host files.pythonhosted.org
```

### Issue: Slow First Compilation

The first time you call a `jax.jit`-compiled function, it will be slow because XLA compiles the computation graph. This is expected behavior:

```python
import jax
import jax.numpy as jnp
import time

@jax.jit
def f(x):
    return jnp.dot(x, x.T)

x = jnp.ones((1000, 1000))

# First call: compilation (slow)
start = time.time()
_ = f(x).block_until_ready()
print(f"First call (compile): {time.time() - start:.2f}s")

# Subsequent calls: cached (fast)
start = time.time()
_ = f(x).block_until_ready()
print(f"Second call (cached): {time.time() - start:.4f}s")
```

### Issue: GPU Out of Memory

```python
# JAX preallocates 75% of GPU memory by default
# Reduce the fraction:
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.25"  # 25% of GPU memory

# Or disable preallocation entirely (slower, but avoids OOM during initialization)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
```

## Installing JAX Ecosystem Libraries

After installing JAX, you may want to install commonly used ecosystem libraries:

```bash
# Neural network libraries
pip install flax                  # Linen module system
pip install optax                 # Optimizers (Adam, SGD, etc.)
pip install equinox               # Elegant NN via pytrees

# Utilities
pip install chex                  # Testing utilities
pip install orbax-checkpoint      # Checkpointing
pip install jaxtyping             # Type annotations for JAX arrays

# Probabilistic programming
pip install numpyro               # Probabilistic programming

# Scientific computing
pip install diffrax               # Differential equations
pip install lineax                # Linear solvers

# All together
pip install flax optax chex orbax-checkpoint jaxtyping
```

## Summary

| Use Case | Install Command |
|----------|----------------|
| CPU development | `pip install jax` |
| NVIDIA GPU (CUDA 12) | `pip install jax[cuda12]` |
| AMD GPU (ROCm) | `pip install jax[rocm] -f .../jax_releases.html` |
| Google TPU | `pip install jax[tpu] -f .../libtpu_releases.html` |
| Docker (GPU) | `docker pull us-docker.pkg.dev/jax-ml/jax/jax-cuda12` |
| Nightly (latest) | `pip install --pre jax jaxlib -f .../jax_nightly_releases.html` |
| Conda (CPU) | `conda install -c conda-forge jax` |
| From source | `python build/build.py && pip install dist/jaxlib-*.whl && pip install -e .` |
