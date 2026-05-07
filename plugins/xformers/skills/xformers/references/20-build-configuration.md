# 20 - Build Configuration

## Overview

xFormers uses a complex build system that supports both CUDA (NVIDIA) and ROCm (AMD) builds, with CUTLASS integration and PyTorch stable ABI for version-independent wheels.

**Source**: `setup.py`, `setup.cfg`, `pyproject.toml`

## Build System Files

### `setup.py`

Main build script with custom build logic:

**Custom classes:**
- `BuildExtensionWithExtraFiles` - Custom `build_ext` that generates `cpp_lib.json`
- `bdist_wheel_abi_none` - Creates ABI-independent wheels

**Build options:**
- CUDA architectures via `TORCH_CUDA_ARCH_LIST`
- ROCm support (detected automatically)
- CUTLASS integration
- Parallel build via `MAX_JOBS`
- Selective builds for CI/CD

### `setup.cfg`

Basic package metadata:
```
[metadata]
name = xformers
[options]
packages = find:
```

### `pyproject.toml`

Minimal configuration:
```toml
[build-system]
requires = ["setuptools", "torch"]
backend = "torch.utils.cpp_extension"
```

## Build Requirements

| Requirement | Version | Purpose |
|-------------|---------|---------|
| PyTorch | >= 2.10.0 | Must be pre-installed |
| CUDA Toolkit | 12.6+ | For NVIDIA builds |
| ROCm | 7.1+ | For AMD builds |
| ninja | latest | Faster builds (optional) |
| C++ compiler | GCC 7+ | Must match NVCC capabilities |

## Building from Source

### Basic Build

```bash
pip install ninja  # Optional but recommended
pip install -v --no-build-isolation -U .
```

### Specifying GPU Architecture

```bash
# For A100 only
TORCH_CUDA_ARCH_LIST="8.0" pip install -v --no-build-isolation .

# For multiple architectures
TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" pip install -v --no-build-isolation .
```

### Limiting Build Parallelism

```bash
# Useful for machines with limited RAM
MAX_JOBS=2 pip install -v --no-build-isolation .
```

### ROCm Build

```bash
# ROCm builds are detected automatically
pip install -v --no-build-isolation .
```

## Generated Files

### `cpp_lib.json`

Build metadata file generated during build:

```json
{
    "cuda_version": "12.6",
    "torch_version": "2.10.0",
    "python_version": "3.11",
    "build_env": {
        "TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9;9.0",
        "CUDA_HOME": "/usr/local/cuda"
    }
}
```

Loaded at runtime by `_cpp_lib.py` for diagnostics.

## PyTorch Stable ABI

Since xFormers 0.0.34, builds use PyTorch's stable ABI:

```python
# setup.py
class BuildExtensionWithExtraFiles(build_ext):
    def build_extensions(self):
        # Use stable ABI
        for ext in self.extensions:
            ext.extra_compile_args.append("-DTORCH_API=__attribute__((visibility(\"default\")))")
```

This means:
- Build once for PyTorch 2.10, works with 2.11, 2.12, etc.
- No more per-version wheel builds

## CUTLASS Integration

xFormers bundles CUTLASS in `third_party/cutlass/`:

```python
# setup.py
CUTLASS_INCLUDE = os.path.join("third_party", "cutlass", "include")
```

CUTLASS is used for:
- Attention GEMM kernels
- Sparse GEMM kernels
- Blackwell-optimized kernels

## CUDA Architecture Support

| Architecture | GPU | Compute Capability | Supported |
|-------------|-----|-------------------|-----------|
| Ampere | A100, A10 | 8.0 | Yes (minimum) |
| Ampere | A6000, A40 | 8.6 | Yes |
| Ada Lovelace | L40, L4 | 8.9 | Yes |
| Hopper | H100, H200 | 9.0 | Yes |
| Blackwell | B100, B200 | 10.0 | Yes |

**Note**: V100 (7.0) and older are no longer supported since xFormers 0.0.31.

## Troubleshooting

### Common Build Issues

1. **NVCC/CUDA mismatch**:
   ```bash
   module unload cuda; module load cuda/12.6
   ```

2. **GCC version**:
   ```bash
   # Check NVCC-supported GCC versions
   nvcc --version
   ```

3. **Out of memory during build**:
   ```bash
   MAX_JOBS=2 pip install -v --no-build-isolation .
   ```

4. **Long path names (Windows)**:
   ```bash
   git config --global core.longpaths true
   ```

5. **CUTLASS build errors**:
   ```bash
   # Ensure CUTLASS submodule is initialized
   git submodule update --init --recursive
   ```

### Checking Installation

```bash
python -m xformers.info
```

This shows:
- xFormers version
- PyTorch version
- CUDA version
- Available kernels
- Build configuration

## Binary Distribution

Pre-built wheels are distributed via PyTorch's package index:

```
https://download.pytorch.org/whl/cu126   # CUDA 12.6
https://download.pytorch.org/whl/cu128   # CUDA 12.8
https://download.pytorch.org/whl/cu130   # CUDA 13.0
https://download.pytorch.org/whl/rocm7.1 # ROCm 7.1
```

Since 0.0.31, wheels are Python-version agnostic (works with Python 3.9-3.13).
