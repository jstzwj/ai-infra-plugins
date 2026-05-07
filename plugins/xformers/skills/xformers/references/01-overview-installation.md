# 01 - Overview, Installation & Architecture

## What is xFormers?

xFormers is a PyTorch library from Facebook Research (Meta) providing optimized building blocks for Transformer models. It focuses on:

- **Customizable building blocks**: Independent, domain-agnostic components usable without boilerplate
- **Research first**: Bleeding-edge components not yet in mainstream PyTorch
- **Built for efficiency**: Custom CUDA/Triton kernels with automatic dispatch to the best available backend

## Installation

### Recommended (pip, Linux & Windows)

Requires PyTorch 2.10.0+.

```bash
# CUDA 12.6
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu126
# CUDA 12.8
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu128
# CUDA 13.0
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu130
# ROCm 7.1 (experimental, Linux only)
pip3 install -U xformers --index-url https://download.pytorch.org/whl/rocm7.1
```

### Development binaries

```bash
pip install --pre -U xformers
```

### From source

```bash
pip install ninja  # Optional, speeds up build
pip install -v --no-build-isolation -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
# This can take dozens of minutes
```

### Verify installation

```python
python -m xformers.info
```

This provides information on what kernels are built and available.

## Architecture Overview

### Package Structure

```
xformers/
├── __init__.py              # Package init, version, triton detection
├── _cpp_lib.py              # C++/CUDA extension loading
├── utils.py                 # Import utilities, benchmarking helpers
├── checkpoint.py            # Selective activation checkpointing
├── fwbw_overlap.py          # Forward-backward pass overlap
├── attn_bias_utils.py       # Attention bias utilities
├── ops/                     # Core operations module
│   ├── __init__.py          # Exports all operations
│   ├── common.py            # Operator registry infrastructure
│   ├── fmha/                # Flash Memory-Efficient Attention
│   │   ├── __init__.py      # FMHA API (re-exports from mslk)
│   │   ├── dispatch.py      # Operator dispatch logic
│   │   ├── attn_bias.py     # Attention bias classes
│   │   ├── common.py        # Common FMHA utilities
│   │   ├── flash.py         # Flash Attention v2 backend
│   │   ├── flash3.py        # Flash Attention v3 backend (H100+)
│   │   ├── cutlass.py       # CUTLASS backend
│   │   ├── cutlass_blackwell.py  # CUTLASS for Blackwell GPUs
│   │   ├── ck.py            # Composable Kernel backend (ROCm)
│   │   ├── ck_splitk.py     # Split-K CK backend
│   │   ├── triton_splitk.py # Triton split-K implementation
│   │   ├── merge_training.py # Attention merging for gradient checkpointing
│   │   ├── torch_attention_compat.py # PyTorch compatibility
│   │   └── _triton/         # Triton split-K kernels
│   ├── _triton/             # Triton kernel implementations
│   │   ├── rmsnorm_kernels.py
│   │   ├── rope_padded_kernels.py
│   │   ├── k_scaled_index_add.py
│   │   ├── k_index_select_cat.py
│   │   ├── tiled_matmul_kernels.py
│   │   └── matmul_perf_model.py
│   ├── swiglu_op.py         # SwiGLU activation
│   ├── rmsnorm.py           # RMS Normalization
│   ├── rope_padded.py       # RoPE with padded KV-cache
│   ├── sp24.py              # 2:4 structured sparsity
│   ├── sequence_parallel_fused_ops.py  # Fused sequence parallel ops
│   ├── seqpar.py            # Sequence parallel matmul
│   ├── modpar_layers.py     # Model parallel linear layers
│   ├── tiled_matmul.py      # Tiled matrix multiplication
│   ├── indexing.py          # Optimized indexing
│   ├── tree_attention.py    # Tree attention
│   ├── unbind.py            # Efficient tensor unbind/stack
│   ├── differentiable_collectives.py  # Differentiable distributed ops
│   └── common.py            # Base operator infrastructure
├── sparse/                  # Block-sparse tensor support
│   ├── blocksparse_tensor.py
│   └── utils.py
├── components/              # Legacy components
│   └── attention/
│       └── attention_patterns.py  # Attention pattern generators
├── profiler/                # Profiling utilities
│   ├── api.py               # Public API
│   ├── profiler.py          # Core profiler implementation
│   ├── profiler_dcgm.py     # DCGM profiling
│   ├── profiler_dcgm_impl.py
│   ├── profile_analyzer.py  # Profile analysis (MFU/HFU)
│   ├── device_limits.py     # Device limit detection
│   └── find_slowest.py      # Bottleneck identification
├── benchmarks/              # Benchmark scripts
├── triton/                  # Triton utilities
│   └── importing.py
├── flash_attn_3/            # Flash Attention 3 support
└── csrc/                    # C++/CUDA extensions
    ├── attention/            # Attention CUDA kernels
    │   ├── hip_fmha/         # ROCm FMHA kernels
    │   └── hip_decoder/      # ROCm decoder kernels
    ├── sparse24/            # 2:4 sparsity CUDA kernels
    └── pt_stable_utils.cu   # PyTorch stable ABI utilities
```

### Key Design Principles

1. **Automatic Backend Dispatch**: Operations automatically select the best available backend based on GPU architecture, tensor dtypes, and input shapes.

2. **Operator Registry**: All operators are registered through a central registry (`xformers/ops/common.py`) that provides `register_operator`, `get_operator`, and `get_xformers_operator`.

3. **C++ Library Loading**: The `_cpp_lib.py` module manages loading of compiled C++/CUDA extensions with proper error handling and platform-specific logic.

4. **Triton Integration**: Custom Triton kernels are conditionally loaded based on GPU capability (compute capability >= 8.0 required) and environment variables:
   - `XFORMERS_ENABLE_TRITON=1` to force-enable
   - `XFORMERS_FORCE_DISABLE_TRITON=1` to force-disable

5. **PyTorch Stable ABI**: xFormers 0.0.34+ uses PyTorch's stable API/ABI, meaning binary builds targeting PyTorch 2.10+ are compatible with any later version.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `XFORMERS_ENABLE_TRITON` | `"0"` | Force-enable Triton kernels |
| `XFORMERS_FORCE_DISABLE_TRITON` | `"0"` | Force-disable Triton kernels |
| `DISABLE_FUSED_SEQUENCE_PARALLEL` | `"0"` | Disable fused sequence parallel ops |
| `XFORMERS_CUSPARSELT_TUNE` | `"0"` | Enable cuSPARSELt algorithm tuning |
| `XFORMERS_TILED_MATMUL_ENABLE_TRITON` | `"1"` | Enable Triton tiled matmul kernel |
| `TORCH_CUDA_ARCH_LIST` | auto | Target GPU architectures for build |
| `MAX_JOBS` | auto | Limit ninja build parallelism |

### Dependencies

| Package | Version | Required | Purpose |
|---------|---------|----------|---------|
| PyTorch | >= 2.10.0 | Yes | Core framework |
| Triton | latest | Optional | Custom GPU kernels (A100+) |
| scipy | >= 1.9.0 | Optional | Optimal checkpointing (MILP) |
| CUTLASS | bundled | Build-time | Attention GEMM kernels |
| Flash Attention | external | Optional | Flash Attention backend |
| ninja | latest | Optional | Faster builds |

### Hardware Support Matrix

| GPU | Compute Capability | Supported Features |
|-----|-------------------|-------------------|
| V100 | 7.0 | No longer supported (since 0.0.31) |
| A100 | 8.0 | All features, Triton kernels |
| H100 | 9.0 | All features, Flash Attention 3 |
| B100 | 9.x | All features, Blackwell CUTLASS |
| AMD (ROCm) | N/A | CK backend, FMHA only |
