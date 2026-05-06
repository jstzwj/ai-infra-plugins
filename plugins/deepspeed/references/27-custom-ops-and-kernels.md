# DeepSpeed Custom Ops and CUDA Kernels Reference

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [OpBuilder Base Class](#opbuilder-base-class)
4. [All Custom Ops](#all-custom-ops)
5. [JIT Compilation](#jit-compilation)
6. [Pre-compiled Op Installation](#pre-compiled-op-installation)
7. [Build Environment Variables](#build-environment-variables)
8. [CUDA Architecture Selection](#cuda-architecture-selection)
9. [csrc Directory Structure](#csrc-directory-structure)
10. [Kernel Implementation Details](#kernel-implementation-details)
11. [Performance Characteristics](#performance-characteristics)
12. [Op Builder API for Custom Ops](#op-builder-api-for-custom-ops)
13. [Configuration Examples](#configuration-examples)
14. [Troubleshooting](#troubleshooting)

---

## Overview

DeepSpeed includes a comprehensive collection of highly optimized custom CUDA/C++ operations and kernels that provide significant performance improvements over standard PyTorch implementations. These custom ops cover:

- **Fused optimizers**: Adam, Lamb, Lion, Adagrad with multi-operator fusion to reduce kernel launch overhead and memory traffic
- **Transformer kernels**: Fused self-attention, feed-forward, and layer-norm operations
- **Sparse attention**: Memory-efficient sparse attention patterns for long sequences
- **Quantization**: INT8/INT4 weight and activation quantization kernels
- **Async I/O**: Non-blocking checkpoint and model state I/O
- **GDS (GPUDirect Storage)**: Direct GPU-NVMe data transfer
- **Inference kernels**: Specialized kernels for high-throughput inference serving
- **Ragged operations**: Variable-length sequence processing for inference V2

These operations are built using PyTorch's C++ extension mechanism (`torch.utils.cpp_extension`) with CUDA kernels written in `csrc/`. They can be either pre-compiled during installation or JIT-compiled at runtime.

### Why Custom Ops?

Standard PyTorch operations launch individual kernels for each sub-operation. For example, a single Adam optimizer step involves:
1. Compute gradient moment (1 kernel)
2. Update momentum (1 kernel)
3. Update variance (1 kernel)
4. Bias correction (1 kernel)
5. Weight update (1 kernel)

Each kernel launch has overhead (~5-10us), and each intermediate result must be written to and read from global memory. DeepSpeed's fused Adam combines all 5 operations into a single kernel, eliminating launch overhead and keeping intermediate results in registers/shared memory.

**Performance improvement**: Fused Adam is typically 2-5x faster than `torch.optim.Adam` for large models.

---

## Architecture

### Source Code Structure

```
deepspeed/
    ops/                           # Python wrapper modules
        op_builder/                # Build system for custom ops
            __init__.py
            builder.py             # OpBuilder base class
            all_ops.py             # Registry of all available ops
            utils.py               # Build utilities
            cuda_op_builder.py     # CUDA-specific builder
            cpu_op_builder.py      # CPU-specific builder
            
            # Individual op builders:
            fused_adam.py
            fused_lamb.py
            fused_lion.py
            cpu_adam.py
            cpu_adagrad.py
            cpu_lion.py
            transformer.py
            stochastic_transformer.py
            transformer_inference.py
            inference_cutlass_builder.py
            sparse_attn.py
            random_ltd.py
            async_io.py
            gds.py
            quantizer.py
            fp_quantizer.py
            spatial_inference.py
            evoformer_attn.py
            dc.py                  # DeepCompile ops
            ragged_utils.py
            ragged_ops.py
            
    csrc/                          # C++/CUDA source code
        adam/                      # Fused Adam kernel
        lamb/                      # Fused Lamb kernel
        lion/                      # Fused Lion kernel
        adagrad/                   # Fused Adagrad kernel
        transformer/               # Transformer layer kernels
        sparse_attention/          # Sparse attention kernels
        aio/                       # Async I/O kernels
        gds/                       # GPUDirect Storage
        quantization/              # Quantization kernels
        fp_quantizer/              # FP quantizer
        random_ltd/                # Random LTD kernels
        spatial/                   # Spatial inference kernels
        deepspeed4science/         # Evoformer attention
        xpu/                       # Intel XPU kernels
        cpu/                       # CPU-optimized kernels
        utils/                     # Shared utilities
        include/                   # Header files
```

### Op Lifecycle

```
1. Import Time
   |
   v
2. OpBuilder.check Availability()
   |-- Check if pre-compiled op is available
   |-- Check if JIT compilation is possible
   |
   v
3. Op Loading (one of):
   |-- import pre-compiled module (e.g., import deepspeed.ops.adam.fused_adam)
   |-- JIT compile via torch.utils.cpp_extension.load()
   |
   v
4. Op Usage
   |-- Called from Python as regular torch.autograd.Function
   |-- Forward: custom CUDA kernel
   |-- Backward: custom CUDA kernel (if differentiable)
```

---

## OpBuilder Base Class

The `OpBuilder` class in `deepspeed/op_builder/builder.py` is the abstract base for all custom op builders. It handles compilation configuration, source file discovery, and op loading.

### Class Definition

```python
# deepspeed/op_builder/builder.py
import os
import sys
from pathlib import Path

class OpBuilder:
    """Base class for building and loading DeepSpeed custom operations.
    
    Subclasses define the sources, compilation flags, and dependencies
    for specific custom ops. The builder handles both pre-compiled and
    JIT compilation workflows.
    """
    
    # Subclasses override these:
    BUILD_VAR = None        # Environment variable to control build (e.g., "DS_BUILD_FUSED_ADAM")
    NAME = None             # Canonical name (e.g., "fused_adam")
    
    def __init__(self, name=None):
        self.name = name or self.NAME
        self.jit_mode = False
        self.extra_build_args = {}
```

### Key Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `is_compatible` | `is_compatible(verbose=True) -> bool` | Check if this op can be built/loaded on current platform |
| `build` | `build() -> None` | Build and install the op as a Python extension |
| `load` | `load(verbose=True) -> module` | Load the op module (pre-compiled or JIT) |
| `sources` | `sources() -> list[str]` | Return list of C++/CUDA source file paths |
| `include_paths` | `include_paths() -> list[str]` | Return include directories for compilation |
| `nvcc_args` | `nvcc_args() -> list[str]` | Return NVCC-specific compilation flags |
| `cxx_args` | `cxx_args() -> list[str]` | Return C++ compiler flags |
| `libraries` | `libraries() -> list[str]` | Return libraries to link against |
| `extra_ldflags` | `extra_ldflags() -> list[str]` | Return additional linker flags |
| `absolute_name` | `absolute_name() -> str` | Return the Python module name for the compiled op |
| `check_so_version` | `check_so_version() -> bool` | Check compatibility of pre-compiled .so file |

### Core Methods Detail

#### `is_compatible()`

```python
def is_compatible(self, verbose=True):
    """Check if this op can be built on the current platform.
    
    Checks:
    1. CUDA toolkit availability (for CUDA ops)
    2. Python version compatibility
    3. PyTorch version compatibility
    4. Required libraries (e.g., cuBLAS, cuDNN)
    
    Args:
        verbose (bool): Print detailed compatibility info.
    
    Returns:
        bool: True if the op can be built/loaded.
    """
    # Check if pre-compiled version is available
    try:
        op_module = self.load(verbose=False)
        if op_module is not None:
            return True
    except ImportError:
        pass
    
    # Check if JIT compilation is possible
    if not torch.cuda.is_available():
        if verbose:
            print(f"[{self.NAME}] CUDA not available, cannot JIT compile")
        return False
    
    return True
```

#### `sources()`

```python
def sources(self):
    """Return list of source files to compile.
    
    Returns:
        list[str]: Paths relative to deepspeed/csrc/
    """
    return []
```

#### `nvcc_args()`

```python
def nvcc_args(self):
    """Return NVCC compilation flags.
    
    Default flags include architecture targeting and optimization.
    
    Returns:
        list[str]: NVCC flags
    """
    args = [
        '-O3',
        '--use_fast_math',
        '-U__CUDA_NO_HALF_OPERATORS__',
        '-U__CUDA_NO_HALF_CONVERSIONS__',
        '-U__CUDA_NO_BFLOAT16_OPERATORS__',
        '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
        '-U__CUDA_NO_BFLOAT162_OPERATORS__',
        '-U__CUDA_NO_BFLOAT162_CONVERSIONS__',
    ]
    return args
```

#### `cxx_args()`

```python
def cxx_args(self):
    """Return C++ compiler flags.
    
    Returns:
        list[str]: g++/clang++ flags
    """
    args = ['-O3', '-fPIC', '-std=c++17']
    if sys.platform == "win32":
        args += ['/DLL']
    return args
```

### CUDAOpBuilder

The `CUDAOpBuilder` extends `OpBuilder` with CUDA-specific logic:

```python
# deepspeed/op_builder/cuda_op_builder.py
class CUDAOpBuilder(OpBuilder):
    """Base class for CUDA-accelerated operations."""
    
    def cuda_version(self):
        """Return CUDA toolkit version."""
        return torch.version.cuda
    
    def compute_capability(self):
        """Return the compute capability of the current GPU."""
        if torch.cuda.is_available():
            return torch.cuda.get_device_capability()
        return (0, 0)
    
    def nvcc_args(self):
        args = super().nvcc_args()
        # Add architecture-specific flags
        arch_list = os.environ.get('TORCH_CUDA_ARCH_LIST', '')
        if not arch_list:
            cap = self.compute_capability()
            arch = f"{cap[0]}{cap[1]}"
            args.append(f'-gencode=arch=compute_{arch},code=sm_{arch}')
        return args
    
    def include_paths(self):
        paths = [
            torch.utils.cpp_extension.CUDA_HOME,
            os.path.join(torch.utils.cpp_extension.CUDA_HOME, 'include'),
        ]
        return paths
```

### CPUOpBuilder

```python
# deepspeed/op_builder/cpu_op_builder.py
class CPUOpBuilder(OpBuilder):
    """Base class for CPU-optimized operations.
    
    Uses SIMD instructions (AVX2, AVX-512) when available.
    """
    
    def cxx_args(self):
        args = ['-O3', '-fPIC', '-std=c++17']
        # Add SIMD flags based on CPU capability
        import cpuinfo
        flags = cpuinfo.get_cpu_info().get('flags', [])
        if 'avx512f' in flags:
            args.append('-mavx512f')
        elif 'avx2' in flags:
            args.append('-mavx2')
        return args
```

---

## All Custom Ops

### Fused Adam (`fused_adam`)

**Builder**: `deepspeed/op_builder/fused_adam.py`
**Source**: `deepspeed/csrc/adam/`
**Build Variable**: `DS_BUILD_FUSED_ADAM`

A fused implementation of the Adam optimizer that combines all update steps into a single CUDA kernel.

#### Python API

```python
from deepspeed.ops.adam import FusedAdam

optimizer = FusedAdam(
    model_params,              # Iterable of parameters or parameter groups
    lr=1e-3,                   # Learning rate
    bias_correction=True,      # Apply bias correction
    betas=(0.9, 0.999),        # Beta1 and beta2
    eps=1e-8,                  # Epsilon for numerical stability
    adam_w_mode=True,          # Use AdamW (decoupled weight decay)
    weight_decay=0.0,          # Weight decay coefficient
    amsgrad=False,             # Use AMSGrad variant
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `params` | Iterable | Required | Model parameters or parameter groups |
| `lr` | float | `1e-3` | Learning rate |
| `bias_correction` | bool | `True` | Apply bias correction to moments |
| `betas` | Tuple[float, float] | `(0.9, 0.999)` | Exponential decay rates for moment estimates |
| `eps` | float | `1e-8` | Numerical stability constant |
| `adam_w_mode` | bool | `True` | Use decoupled weight decay (AdamW) |
| `weight_decay` | float | `0.0` | Weight decay coefficient |
| `amsgrad` | bool | `False` | Use AMSGrad variant |

#### Source Files

```
csrc/adam/
    fused_adam_frontend.cpp    # Python binding (pybind11)
    fused_adam_kernel.cu       # CUDA kernel implementation
    fused_adam_kernel.h        # Kernel header
```

#### Kernel Details

The fused Adam CUDA kernel processes parameter elements in parallel:

```cuda
// Simplified kernel structure (csrc/adam/fused_adam_kernel.cu)
template <typename T, typename GRAD_T>
__global__ void fused_adam_kernel(
    T* __restrict__ params,        // [num_elements]
    GRAD_T* __restrict__ grads,    // [num_elements]
    T* __restrict__ exp_avg,       // [num_elements] momentum
    T* __restrict__ exp_avg_sq,    // [num_elements] variance
    T* __restrict__ max_exp_avg_sq,// [num_elements] (AMSGrad only)
    const float beta1,
    const float beta2,
    const float eps,
    const float step_size,
    const float decay,
    const bool amsgrad,
    const bool bias_correction,
    const int num_elements) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    
    // Load values
    float param_val = static_cast<float>(params[idx]);
    float grad_val = static_cast<float>(grads[idx]);
    float exp_avg_val = static_cast<float>(exp_avg[idx]);
    float exp_avg_sq_val = static_cast<float>(exp_avg_sq[idx]);
    
    // Apply weight decay
    if (decay != 0.0f) {
        param_val *= decay;
    }
    
    // Update momentum
    exp_avg_val = beta1 * exp_avg_val + (1.0f - beta1) * grad_val;
    
    // Update variance
    exp_avg_sq_val = beta2 * exp_avg_sq_val + (1.0f - beta2) * grad_val * grad_val;
    
    // Bias correction
    float bias_correction1 = 1.0f, bias_correction2 = 1.0f;
    if (bias_correction) {
        bias_correction1 = 1.0f - powf(beta1, step);
        bias_correction2 = 1.0f - powf(beta2, step);
    }
    
    // Compute update
    float denom = sqrtf(exp_avg_sq_val / bias_correction2) + eps;
    float step_val = step_size * (exp_avg_val / bias_correction1) / denom;
    
    // Update parameter
    params[idx] = static_cast<T>(param_val - step_val);
    
    // Store updated moments
    exp_avg[idx] = static_cast<T>(exp_avg_val);
    exp_avg_sq[idx] = static_cast<T>(exp_avg_sq_val);
}
```

#### Performance

| Model Size | PyTorch Adam | DeepSpeed Fused Adam | Speedup |
|-----------|-------------|---------------------|---------|
| 1.3B params | 45 ms/step | 12 ms/step | 3.75x |
| 7B params | 280 ms/step | 75 ms/step | 3.73x |
| 13B params | 520 ms/step | 145 ms/step | 3.59x |

---

### Fused Lamb (`fused_lamb`)

**Builder**: `deepspeed/op_builder/fused_lamb.py`
**Source**: `deepspeed/csrc/lamb/`
**Build Variable**: `DS_BUILD_FUSED_LAMB`

Fused implementation of the LAMB (Layer-wise Adaptive Moments) optimizer for large-batch training.

#### Python API

```python
from deepspeed.ops.lamb import FusedLamb

optimizer = FusedLamb(
    model_params,
    lr=1e-3,
    bias_correction=True,
    betas=(0.9, 0.999),
    eps=1e-6,
    weight_decay=0.01,
    adam_w_mode=True,
    max_grad_norm=1.0,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `params` | Iterable | Required | Model parameters |
| `lr` | float | `1e-3` | Learning rate |
| `bias_correction` | bool | `True` | Apply bias correction |
| `betas` | Tuple[float, float] | `(0.9, 0.999)` | Moment decay rates |
| `eps` | float | `1e-6` | Numerical stability |
| `weight_decay` | float | `0.01` | Weight decay |
| `adam_w_mode` | bool | `True` | AdamW style weight decay |
| `max_grad_norm` | float | `1.0` | Max gradient norm for clipping |

#### Source Files

```
csrc/lamb/
    fused_lamb_frontend.cpp     # Python binding
    fused_lamb_kernel.cu        # CUDA kernel
    fused_lamb_kernel.h         # Kernel header
```

#### Kernel Details

LAMB adds layer-wise trust ratio scaling on top of the Adam update:

```
trust_ratio = ||param|| / ||update||
final_update = trust_ratio * update
```

This enables stable training with very large batch sizes (64K+) by adaptively scaling the learning rate per layer.

---

### Fused Lion (`fused_lion`)

**Builder**: `deepspeed/op_builder/fused_lion.py`
**Source**: `deepspeed/csrc/lion/`
**Build Variable**: `DS_BUILD_FUSED_LION`

Fused implementation of the Lion (EvoLved Sign Momentum) optimizer, which uses the sign of the momentum for updates.

#### Python API

```python
from deepspeed.ops.lion import FusedLion

optimizer = FusedLion(
    model_params,
    lr=1e-4,
    betas=(0.9, 0.99),
    weight_decay=0.0,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `params` | Iterable | Required | Model parameters |
| `lr` | float | `1e-4` | Learning rate |
| `betas` | Tuple[float, float] | `(0.9, 0.99)` | Moment decay rates |
| `weight_decay` | float | `0.0` | Weight decay |

#### Key Difference from Adam

Lion replaces the expensive element-wise multiplication and square root with a simple sign operation:

```cuda
// Lion update (simplified)
update = sign(beta1 * momentum + (1 - beta1) * grad)
param = param - lr * update
momentum = beta2 * momentum + (1 - beta2) * grad
```

This makes Lion 2-3x more memory-efficient than Adam (no variance tensor needed) and faster per step.

---

### CPU Adam (`cpu_adam`)

**Builder**: `deepspeed/op_builder/cpu_adam.py`
**Source**: `deepspeed/csrc/cpu/`
**Build Variable**: `DS_BUILD_CPU_ADAM`

Highly optimized CPU implementation of Adam, using AVX2/AVX-512 SIMD instructions for ZeRO-Offload.

#### Python API

```python
from deepspeed.ops.cpu_adam import DeepSpeedCPUAdam

optimizer = DeepSpeedCPUAdam(
    model_params,
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.0,
    adamw_mode=True,
    fp32_param_states=False,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `params` | Iterable | Required | Model parameters |
| `lr` | float | `1e-3` | Learning rate |
| `betas` | Tuple[float, float] | `(0.9, 0.999)` | Moment decay rates |
| `eps` | float | `1e-8` | Numerical stability |
| `weight_decay` | float | `0.0` | Weight decay |
| `adamw_mode` | bool | `True` | AdamW mode |
| `fp32_param_states` | bool | `False` | Store optimizer states in FP32 |

#### SIMD Optimization

The CPU Adam kernel uses SIMD intrinsics for parallel element processing:

```cpp
// AVX-512 implementation (simplified)
__m512 beta1_vec = _mm512_set1_ps(beta1);
__m512 beta2_vec = _mm512_set1_ps(beta2);
__m512 one_minus_beta1 = _mm512_set1_ps(1.0f - beta1);
__m512 one_minus_beta2 = _mm512_set1_ps(1.0f - beta2);

for (int i = 0; i < n; i += 16) {
    __m512 grad = _mm512_loadu_ps(&grads[i]);
    __m512 exp_avg = _mm512_loadu_ps(&momentum[i]);
    __m512 exp_avg_sq = _mm512_loadu_ps(&variance[i]);
    
    exp_avg = _mm512_fmadd_ps(beta1_vec, exp_avg, 
                               _mm512_mul_ps(one_minus_beta1, grad));
    exp_avg_sq = _mm512_fmadd_ps(beta2_vec, exp_avg_sq,
                                  _mm512_mul_ps(one_minus_beta2, 
                                                _mm512_mul_ps(grad, grad)));
    
    _mm512_storeu_ps(&momentum[i], exp_avg);
    _mm512_storeu_ps(&variance[i], exp_avg_sq);
}
```

#### Performance

| Configuration | Throughput |
|--------------|-----------|
| PyTorch CPU Adam | ~20 GB/s |
| DeepSpeed CPU Adam (AVX2) | ~40 GB/s |
| DeepSpeed CPU Adam (AVX-512) | ~60 GB/s |

---

### CPU Adagrad (`cpu_adagrad`)

**Builder**: `deepspeed/op_builder/cpu_adagrad.py`
**Source**: `deepspeed/csrc/cpu/`
**Build Variable**: `DS_BUILD_CPU_ADAGRAD`

CPU-optimized Adagrad for offloading scenarios.

```python
from deepspeed.ops.cpu_adagrad import DeepSpeedCPUAdagrad

optimizer = DeepSpeedCPUAdagrad(
    model_params,
    lr=1e-2,
    eps=1e-8,
    weight_decay=0.0,
)
```

---

### CPU Lion (`cpu_lion`)

**Builder**: `deepspeed/op_builder/cpu_lion.py`
**Source**: `deepspeed/csrc/cpu/`
**Build Variable**: `DS_BUILD_CPU_LION`

CPU-optimized Lion optimizer.

```python
from deepspeed.ops.cpu_lion import DeepSpeedCPULion

optimizer = DeepSpeedCPULion(
    model_params,
    lr=1e-4,
    betas=(0.9, 0.99),
    weight_decay=0.0,
)
```

---

### Transformer (`transformer`)

**Builder**: `deepspeed/op_builder/transformer.py`
**Source**: `deepspeed/csrc/transformer/`
**Build Variable**: `DS_BUILD_TRANSFORMER`

Fused transformer layer implementation that combines attention, feed-forward, and layer normalization into optimized kernels.

#### Python API

```python
from deepspeed.ops.transformer import DeepSpeedTransformerLayer, DeepSpeedTransformerConfig

config = DeepSpeedTransformerConfig(
    batch_size=32,
    max_seq_length=2048,
    hidden_size=4096,
    heads=32,
    attn_dropout_ratio=0.1,
    hidden_dropout_ratio=0.1,
    num_hidden_layers=32,
    initializer_range=0.02,
    layer_norm_eps=1e-5,
    fp16=True,
    pre_layer_norm=True,
    local_rank=0,
)

transformer_layer = DeepSpeedTransformerLayer(config)
```

#### DeepSpeedTransformerConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | int | Required | Training batch size |
| `max_seq_length` | int | Required | Maximum sequence length |
| `hidden_size` | int | Required | Hidden dimension size |
| `heads` | int | Required | Number of attention heads |
| `attn_dropout_ratio` | float | `0.0` | Attention dropout probability |
| `hidden_dropout_ratio` | float | `0.0` | Feed-forward dropout probability |
| `num_hidden_layers` | int | Required | Total transformer layers |
| `initializer_range` | float | `0.02` | Weight initialization range |
| `layer_norm_eps` | float | `1e-5` | Layer norm epsilon |
| `fp16` | bool | `False` | Enable FP16 |
| `bf16` | bool | `False` | Enable BF16 |
| `pre_layer_norm` | bool | `True` | Pre-LN vs Post-LN |
| `local_rank` | int | `0` | Local GPU rank |
| `stochastic_mode` | bool | `False` | Stochastic depth mode |
| `rotary_embedding` | bool | `False` | Use rotary positional embeddings |
| `mlp_after_attn` | bool | `True` | Include MLP after attention |
| `mlp_type` | str | `"standard"` | MLP type: "standard" or "residual" |

#### Source Files

```
csrc/transformer/
    ds_transformer_cuda.cpp     # Main CUDA kernel dispatcher
    ds_transformer_runtime.cu   # Runtime implementation
    Kernels.h                   # Kernel declarations
    cubelin.cu                  # cuBLAS linear algebra
    layer_norm.cu               # Layer normalization kernel
    quantization.cu             # Quantization support
    System.h                    # System utilities
    custom_cuda_kernels.cu      # Additional CUDA kernels
```

#### Fused Operations

The transformer kernel fuses these operations into a single pass:

1. **Layer Norm** (pre-norm or post-norm)
2. **QKV Projection**: `x @ W_qkv` -> `[Q, K, V]`
3. **Self-Attention**: Scaled dot-product attention with softmax
4. **Output Projection**: `attn_out @ W_out`
5. **Residual Connection**: `output + residual`
6. **Feed-Forward Network**: Two linear layers with activation
7. **Final Layer Norm**
8. **Dropout** (training only)

---

### Stochastic Transformer (`stochastic_transformer`)

**Builder**: `deepspeed/op_builder/stochastic_transformer.py`
**Source**: `deepspeed/csrc/transformer/` (shared with transformer)
**Build Variable**: `DS_BUILD_STOCHASTIC_TRANSFORMER`

Extension of the fused transformer with stochastic depth (layer dropout) for training very deep models.

```python
config = DeepSpeedTransformerConfig(
    ...,
    stochastic_mode=True,
)

# Each layer has a survival probability
# p_l = 1 - (l / L) * (1 - p_L)
# where L is total layers, p_L is survival probability of last layer
```

---

### Sparse Attention (`sparse_attn`)

**Builder**: `deepspeed/op_builder/sparse_attn.py`
**Source**: `deepspeed/csrc/sparse_attention/`
**Build Variable**: `DS_BUILD_SPARSE_ATTN`

Memory-efficient sparse attention implementation that reduces the O(n^2) memory and compute of standard attention to O(n * sqrt(n)) for long sequences.

#### Python API

```python
from deepspeed.ops.sparse_attention import SparseSelfAttention, SparsityConfig

sparsity_config = SparsityConfig(
    num_heads=32,
    block=16,                    # Block size for sparse pattern
    different_layout_per_head=False,
    num_layouts=1,
)

sparse_attn = SparseSelfAttention(
    sparsity_config=sparsity_config,
    max_seq_length=16384,
    attn_dropout=0.1,
)
```

#### SparsityConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_heads` | int | Required | Number of attention heads |
| `block` | int | `16` | Block size for sparse blocking |
| `different_layout_per_head` | bool | `False` | Different sparsity pattern per head |
| `num_layouts` | int | `1` | Number of distinct sparsity layouts |
| `attention` | str | `"fixed"` | Attention type: "fixed", "bigbird", "bslongformer", "variable" |
| `horizontal_global_attention` | bool | `False` | Add horizontal global attention |
| `num_different_blocks` | int | `1` | Number of different block patterns |
| `num_random_blocks` | int | `0` | Number of random blocks |
| `num_local_blocks` | int | `4` | Number of local blocks |
| `num_global_blocks` | int | `1` | Number of global blocks |
| `attention_type` | str | `"uni"` | "uni" for unidirectional, "bi" for bidirectional |

#### Sparse Attention Patterns

1. **Fixed**: Static sparsity pattern with local + global + random blocks
2. **BigBird**: Google BigBird pattern (random + window + global)
3. **BSLongformer**: Longformer-style (sliding window + global)
4. **Variable**: Dynamic sparsity pattern that varies per layer

#### Source Files

```
csrc/sparse_attention/
    sparse_attention_cuda.cpp    # Python bindings
    sparse_attention_kernel.cu   # CUDA kernel
    sparse_attention_utils.h     # Utility functions
```

---

### Random LTD (`random_ltd`)

**Builder**: `deepspeed/op_builder/random_ltd.py`
**Source**: `deepspeed/csrc/random_ltd/`
**Build Variable**: `DS_BUILD_RANDOM_LTD`

Random Long-Term Dependency attention for training with mixed sparse/dense patterns.

```python
from deepspeed.ops.random_ltd import RandomLTDOp

random_ltd = RandomLTDOp(
    config,
    seq_len=2048,
)
```

---

### Async I/O (`async_io`)

**Builder**: `deepspeed/op_builder/async_io.py`
**Source**: `deepspeed/csrc/aio/`
**Build Variable**: `DS_BUILD_AIO`

Asynchronous I/O operations for non-blocking checkpoint saving/loading and ZeRO-Infinity NVMe offloading.

#### Python API

```python
from deepspeed.ops.aio import AsyncIOReader, AsyncIOWriter

# Async read
reader = AsyncIOReader(
    block_size=1048576,    # 1 MB block size
    queue_depth=8,         # Number of async operations
    single_submit=False,   # Batch submissions
    overlap_events=True,   # Overlap CPU/GPU transfers
)

# Async write
writer = AsyncIOWriter(
    block_size=1048576,
    queue_depth=8,
    single_submit=False,
    overlap_events=True,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `block_size` | int | `1048576` | I/O block size in bytes |
| `queue_depth` | int | `8` | Number of pending async operations |
| `single_submit` | bool | `False` | Submit one operation at a time |
| `overlap_events` | bool | `True` | Overlap CPU and GPU transfers |

#### Source Files

```
csrc/aio/
    aio_pybind.cpp              # Python bindings (pybind11)
    aio_common.h                # Common definitions
    libaio_context.h            # Linux libaio context
    libaio_context.cpp          # libaio implementation
    ds_aio.cpp                  # DeepSpeed async I/O
```

#### Supported I/O Backends

| Backend | Library | Platform | Description |
|---------|---------|----------|-------------|
| `libaio` | `libaio.so` | Linux | Linux native asynchronous I/O |
| `io_uring` | `liburing.so` | Linux 5.1+ | Modern Linux async I/O (faster) |
| `posix` | N/A | Any | POSIX AIO (fallback) |

---

### GDS - GPUDirect Storage (`gds`)

**Builder**: `deepspeed/op_builder/gds.py`
**Source**: `deepspeed/csrc/gds/`
**Build Variable**: `DS_BUILD_GDS`

GPUDirect Storage enables direct data transfer between GPU memory and NVMe storage, bypassing CPU memory entirely.

```python
from deepspeed.ops.gds import GdsFileLoader, GdsFileWriter

# Direct GPU-NVME load
loader = GdsFileLoader()
buffer = torch.empty(size, device='cuda')
loader.load(buffer, "/path/to/checkpoint")

# Direct GPU-NVME write
writer = GdsFileWriter()
writer.write(buffer, "/path/to/checkpoint")
```

---

### Quantizer (`quantizer`)

**Builder**: `deepspeed/op_builder/quantizer.py`
**Source**: `deepspeed/csrc/quantization/`
**Build Variable**: `DS_BUILD_QUANTIZER`

Quantization kernels for INT8 and INT4 weight/activation quantization.

```python
from deepspeed.ops.quantizer import quantize, dequantize

# INT8 quantization
quantized, scale, min_val = quantize(
    tensor,
    q_type="int8",
    q_groups=1,
)

# Dequantization
restored = dequantize(quantized, scale, min_val, q_type="int8")
```

#### Supported Quantization Types

| Type | Bits | Min Value | Max Value | Description |
|------|------|-----------|-----------|-------------|
| `int8` | 8 | -128 | 127 | Standard INT8 |
| `int4` | 4 | -8 | 7 | INT4 for extreme compression |
| `fp8_e4m3` | 8 | N/A | N/A | FP8 E4M3 (Hopper+) |
| `fp8_e5m2` | 8 | N/A | N/A | FP8 E5M2 (Hopper+) |

---

### FP Quantizer (`fp_quantizer`)

**Builder**: `deepspeed/op_builder/fp_quantizer.py`
**Source**: `deepspeed/csrc/fp_quantizer/`
**Build Variable**: `DS_BUILD_FP_QUANTIZER`

Floating-point quantization for FP8 and custom floating-point formats.

```python
from deepspeed.ops.fp_quantizer import FP_Quantizer

quantizer = FP_Quantizer()
quantized = quantizer.quantize(
    tensor,
    q_bits=8,
    q_mantissa_bits=3,
)
```

---

### Spatial Inference (`spatial_inference`)

**Builder**: `deepspeed/op_builder/spatial_inference.py`
**Source**: `deepspeed/csrc/spatial/`
**Build Variable**: `DS_BUILD_SPATIAL_INFERENCE`

Optimized kernels for spatial inference operations (image/video models).

```python
from deepspeed.ops.spatial_inference import SpatialInferenceOp

spatial_op = SpatialInferenceOp(config)
```

---

### Transformer Inference (`transformer_inference`)

**Builder**: `deepspeed/op_builder/transformer_inference.py`
**Source**: `deepspeed/csrc/transformer/inference/`
**Build Variable**: `DS_BUILD_TRANSFORMER_INFERENCE`

Specialized inference kernels for the DeepSpeed inference engine V1.

```python
from deepspeed.ops.transformer.inference import DeepSpeedInferenceOp

# These ops are typically used internally by the inference engine
# and not called directly by users
```

---

### Inference CUTLASS Builder (`inference_cutlass_builder`)

**Builder**: `deepspeed/op_builder/inference_cutlass_builder.py`
**Source**: `deepspeed/csrc/transformer/inference/cutlass/`
**Build Variable**: `DS_BUILD_CUTLASS`

NVIDIA CUTLASS-based GEMM kernels for high-performance inference with quantized weights.

```python
# Used internally for INT8/INT4 quantized inference GEMM
# Supports:
# - INT8 weight * INT8 activation -> INT32 accumulate
# - INT4 weight * INT8 activation -> INT32 accumulate
# - Mixed-precision GEMM with on-the-fly dequantization
```

---

### Inference Core Ops (`inference_core_ops`)

Builds core operations for the inference engine V2, including attention kernels and GEMM wrappers.

```python
# Used internally by inference V2 engine
# Includes:
# - Blocked KV cache operations
# - Ragged batch processing
# - Rotary embedding kernels
# - Fused attention kernels
```

---

### Evoformer Attention (`evoformer_attn`)

**Builder**: `deepspeed/op_builder/evoformer_attn.py`
**Source**: `deepspeed/csrc/deepspeed4science/`
**Build Variable**: Part of `DS_BUILD_OPS`

Specialized attention kernels for Evoformer architecture used in protein structure prediction (AlphaFold2-style).

```python
from deepspeed.ops.evoformer_attn import EvoformerAttentionOp

evo_attn = EvoformerAttentionOp(
    hidden_size=256,
    num_heads=32,
    pair_dim=128,
)
```

---

### DC - DeepCompile (`dc`)

**Builder**: `deepspeed/op_builder/dc.py`
**Source**: Part of `deepspeed/compile/`
**Build Variable**: Part of `DS_BUILD_OPS`

DeepCompile-specific operations for activation offloading and memory management.

```python
# Used internally by DeepCompile for:
# - Activation tensor free/offload
# - Symmetric memory operations
# - Double buffer management
```

---

### Ragged Utils and Ragged Ops

**Builder**: `deepspeed/op_builder/ragged_utils.py`, `deepspeed/op_builder/ragged_ops.py`
**Source**: `deepspeed/csrc/inference/`
**Build Variable**: `DS_BUILD_RAGGED_OPS`

Operations for handling variable-length (ragged) sequences in inference V2.

```python
from deepspeed.ops.ragged_ops import RaggedOps

# Used internally by inference V2 for:
# - Ragged batch construction
# - Variable-length attention
# - Ragged softmax
# - Ragged KV cache management
```

---

## JIT Compilation

DeepSpeed can compile custom ops at runtime using PyTorch's JIT compilation mechanism. This is useful when pre-compiled ops are not available or when the hardware/platform requires custom compilation.

### How JIT Compilation Works

```python
# deepspeed/op_builder/builder.py (simplified)

import torch.utils.cpp_extension as cpp_extension

class OpBuilder:
    def load(self, verbose=True):
        # Try pre-compiled first
        try:
            return importlib.import_module(self.absolute_name())
        except ImportError:
            pass
        
        # JIT compile
        return cpp_extension.load(
            name=self.name,
            sources=self.sources(),
            extra_include_paths=self.include_paths(),
            extra_cflags=self.cxx_args(),
            extra_cuda_cflags=self.nvcc_args(),
            extra_ldflags=self.extra_ldflags(),
            verbose=verbose,
        )
```

### JIT Compilation Flow

```
User imports DeepSpeed op
    |
    v
OpBuilder.load() called
    |
    v
Try import pre-compiled module
    |
    +-- Success --> Return module
    |
    +-- ImportError --> JIT compile
            |
            v
        torch.utils.cpp_extension.load()
            |
            v
        Create build directory (~/.cache/torch_extensions/)
            |
            v
        Generate setup.py / ninja.build
            |
            v
        Run nvcc + g++ compilation
            |
            v
        Load shared library
            |
            v
        Return module
```

### JIT Compilation Cache

JIT-compiled ops are cached in `~/.cache/torch_extensions/` (configurable via `TORCH_EXTENSIONS_DIR`):

```
~/.cache/torch_extensions/
    fused_adam/
        fused_adam.so         # Compiled shared library
        build.ninja           # Build configuration
```

### Controlling JIT Behavior

```python
# Force JIT recompilation
import torch.utils.cpp_extension
torch.utils.cpp_extension.CPP_EXTENSION_CACHE = None

# Change cache directory
import os
os.environ['TORCH_EXTENSIONS_DIR'] = '/path/to/cache'

# Verbose JIT output
import deepspeed
deepspeed.ops.op_builder.CUDAOpBuilder.VERBOSE = True
```

---

## Pre-compiled Op Installation

For production deployments, pre-compiling ops during installation avoids JIT overhead at runtime.

### Standard Installation with Ops

```bash
# Build all ops during installation
DS_BUILD_OPS=1 pip install deepspeed

# Build specific ops
DS_BUILD_FUSED_ADAM=1 \
DS_BUILD_FUSED_LAMB=1 \
DS_BUILD_CPU_ADAM=1 \
DS_BUILD_TRANSFORMER=1 \
DS_BUILD_AIO=1 \
pip install deepspeed
```

### Build from Source

```bash
git clone https://github.com/microsoft/DeepSpeed.git
cd DeepSpeed

# Build all ops
DS_BUILD_OPS=1 python setup.py build_ext

# Install
DS_BUILD_OPS=1 pip install .

# Or with specific ops
DS_BUILD_FUSED_ADAM=1 DS_BUILD_CPU_ADAM=1 pip install .
```

### Verify Op Installation

```python
import deepspeed
from deepspeed.ops.op_builder import get_op_status

# Check status of all ops
print(deepspeed.ops.op_builder.get_all_ops_status())

# Check specific op
from deepspeed.ops.op_builder import FusedAdamBuilder
builder = FusedAdamBuilder()
print(f"Fused Adam available: {builder.is_compatible()}")
print(f"Fused Adam loaded: {builder.load() is not None}")
```

### DeepSpeed Op Report

```bash
# Print comprehensive op status report
ds_report
```

Output example:
```
[2024-01-15 10:00:00,000] [INFO] [real_accelerator.py:191:get_accelerator] Setting DS_ACCELERATOR = 'cuda'
--------------------------------------------------
DeepSpeed C++/CUDA extension op report
--------------------------------------------------
NOTE: Ops not installed will be just-in-time (JIT) compiled at
      runtime if needed. Op compatibility means that your system
      meets the necessary requirements for JIT compilation.

op name   status
--------- ---------
fused_adam            [INSTALLED]
fused_lamb            [INSTALLED]
fused_lion            [INSTALLED]
cpu_adam              [INSTALLED]
cpu_adagrad           [INSTALLED]
cpu_lion              [INSTALLED]
transformer           [INSTALLED]
stochastic_transformer[INSTALLED]
sparse_attn           [INSTALLED]
random_ltd            [INSTALLED]
async_io              [INSTALLED]
gds                   [NOT INSTALLED]
quantizer             [INSTALLED]
fp_quantizer          [NOT INSTALLED]
spatial_inference     [NOT INSTALLED]
evoformer_attn        [NOT INSTALLED]
ragged_utils          [INSTALLED]
ragged_ops            [INSTALLED]
--------------------------------------------------
```

---

## Build Environment Variables

### Global Build Control

| Variable | Default | Description |
|----------|---------|-------------|
| `DS_BUILD_OPS` | `1` | Master switch for building all custom ops. Set to `0` to skip all ops. |
| `DS_BUILD_OPS_PATH` | Auto | Override path for op source files |

### Per-Op Build Variables

| Variable | Default | Op |
|----------|---------|-----|
| `DS_BUILD_FUSED_ADAM` | `1` | Fused Adam CUDA kernel |
| `DS_BUILD_FUSED_LAMB` | `1` | Fused Lamb CUDA kernel |
| `DS_BUILD_FUSED_LION` | `1` | Fused Lion CUDA kernel |
| `DS_BUILD_CPU_ADAM` | `1` | CPU Adam SIMD kernel |
| `DS_BUILD_CPU_ADAGRAD` | `1` | CPU Adagrad SIMD kernel |
| `DS_BUILD_CPU_LION` | `1` | CPU Lion SIMD kernel |
| `DS_BUILD_TRANSFORMER` | `1` | Fused transformer layer |
| `DS_BUILD_STOCHASTIC_TRANSFORMER` | `1` | Stochastic transformer |
| `DS_BUILD_TRANSFORMER_INFERENCE` | `1` | Transformer inference kernels |
| `DS_BUILD_SPARSE_ATTN` | `0` | Sparse attention (opt-in) |
| `DS_BUILD_RANDOM_LTD` | `1` | Random LTD attention |
| `DS_BUILD_AIO` | `1` | Async I/O |
| `DS_BUILD_GDS` | `0` | GPUDirect Storage (opt-in) |
| `DS_BUILD_QUANTIZER` | `1` | Quantization kernels |
| `DS_BUILD_FP_QUANTIZER` | `0` | FP quantizer (opt-in) |
| `DS_BUILD_SPATIAL_INFERENCE` | `1` | Spatial inference |
| `DS_BUILD_RAGGED_OPS` | `1` | Ragged inference ops |
| `DS_BUILD_UTILS` | `1` | Utility ops |
| `DS_BUILD_CUTLASS` | `0` | CUTLASS-based ops (opt-in) |
| `DS_BUILD_EVOFORMER_ATTN` | `0` | Evoformer attention (opt-in) |

### Build Dependency Environment Variables

| Variable | Description |
|----------|-------------|
| `TORCH_CUDA_ARCH_LIST` | CUDA architectures to target (e.g., `"8.0;8.6;9.0"`) |
| `CUDA_HOME` | Path to CUDA toolkit installation |
| `NCCL_HOME` | Path to NCCL installation |
| `TORCH_EXTENSIONS_DIR` | Path for JIT compilation cache |
| `DS_BUILD_TEST` | Build test executables (`0` or `1`) |
| `DS_BUILD_REF` | Build with debug symbols (`0` or `1`) |

---

## CUDA Architecture Selection

The `TORCH_CUDA_ARCH_LIST` environment variable controls which GPU architectures the CUDA kernels are compiled for. This significantly affects compilation time and runtime performance.

### Setting Architecture List

```bash
# Target specific architectures
export TORCH_CUDA_ARCH_LIST="8.0"        # A100 only
export TORCH_CUDA_ARCH_LIST="8.0;8.6"    # A100 + RTX 3090
export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"  # A100 + RTX 3090 + H100
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0"  # Full range

# Auto-detect (compiles for current GPU only)
export TORCH_CUDA_ARCH_LIST=""
```

### Architecture Impact on Build Time

| Architectures | Approximate Build Time |
|--------------|----------------------|
| Single (e.g., `"8.0"`) | 2-5 minutes |
| Two (e.g., `"8.0;8.6"`) | 4-10 minutes |
| All (5+) | 10-25 minutes |

### Performance Impact

Compiling for the exact target architecture enables:
- Use of architecture-specific instructions (e.g., Tensor Core layouts)
- Optimal register allocation for the target SM count
- Better instruction scheduling

Running code compiled for a different architecture triggers JIT recompilation for the current architecture.

---

## csrc Directory Structure

The `csrc/` directory contains all C++/CUDA source code for DeepSpeed's custom kernels.

```
csrc/
    includes/
        ds_kernel_utils.h        # Shared kernel utilities
        ds_cuda.h                # CUDA helper macros
        type_shim.h              # Type dispatch helpers
    
    adam/
        fused_adam_frontend.cpp  # pybind11 bindings for Adam
        fused_adam_kernel.cu     # Adam CUDA kernels
        fused_adam_kernel.h      # Adam kernel declarations
        
    lamb/
        fused_lamb_frontend.cpp  # pybind11 bindings for Lamb
        fused_lamb_kernel.cu     # Lamb CUDA kernels
        fused_lamb_kernel.h      # Lamb kernel declarations
        
    lion/
        fused_lion_frontend.cpp  # pybind11 bindings for Lion
        fused_lion_kernel.cu     # Lion CUDA kernels
        fused_lion_kernel.h      # Lion kernel declarations
        
    adagrad/
        fused_adagrad_frontend.cpp
        fused_adagrad_kernel.cu
        fused_adagrad_kernel.h
        
    transformer/
        ds_transformer_cuda.cpp          # Main transformer kernel
        ds_transformer_runtime.cu        # Runtime dispatch
        Kernels.h                         # Kernel declarations
        cubelin.cu                        # cuBLAS operations
        layer_norm.cu                     # Layer normalization
        quantization.cu                   # Transformer quantization
        System.h                          # System utilities
        custom_cuda_kernels.cu            # Additional kernels
        inference/
            cublas_wrappers.cu            # cuBLAS wrappers for inference
            gelu.h                        # GELU activation
            layer_norm.cu                 # Inference layer norm
            relu.h                        # ReLU activation
            convert.h                     # Type conversion
            moe.h                         # MoE inference
            performance_config.h          # Performance tuning
            pt_binding.cpp                # Python bindings
            cutlass/
                fmha/                     # CUTLASS flash-attention
                gemm/                     # CUTLASS GEMM kernels
                qgemm/                    # Quantized GEMM
                
    sparse_attention/
        sparse_attention_cuda.cpp         # Python bindings
        sparse_attention_kernel.cu        # Sparse attention kernel
        sparse_attention_utils.h          # Utility functions
        
    aio/
        aio_pybind.cpp                    # Python bindings
        aio_common.h                      # Common definitions
        libaio_context.h                  # libaio interface
        libaio_context.cpp                # libaio implementation
        ds_aio.cpp                        # DS async I/O core
        
    gds/
        gds_pybind.cpp                    # Python bindings
        gds_context.h                     # GDS context
        gds_kernel.cu                     # GDS kernel
        
    quantization/
        quantize_kernel.cu                # Quantization kernels
        dequantize_kernel.cu              # Dequantization kernels
        
    fp_quantizer/
        fp_quantize_kernel.cu             # FP quantization kernel
        
    random_ltd/
        random_ltd_kernel.cu              # Random LTD kernel
        
    spatial/
        spatial_kernel.cu                 # Spatial inference kernel
        
    deepspeed4science/
        evoformer_attn.cu                 # Evoformer attention
        
    xpu/
        xpu_adam.cpp                      # Intel XPU Adam
        xpu_lamb.cpp                      # Intel XPU Lamb
        
    cpu/
        cpu_adam.cpp                      # CPU Adam (AVX2/AVX-512)
        cpu_adagrad.cpp                   # CPU Adagrad
        cpu_lion.cpp                      # CPU Lion
        cpu_shared.h                      # Shared CPU utilities
        
    utils/
        ds_utils.cpp                      # General utilities
```

---

## Kernel Implementation Details

### Common Kernel Patterns

#### Template-Based Type Dispatch

All kernels use C++ template specialization to handle multiple data types:

```cuda
// Type dispatch pattern
template <typename T>
__global__ void kernel_impl(/* ... */);

// Specializations for:
// half (FP16)
// __nv_bfloat16 (BF16)
// float (FP32)
// __nv_fp8_e4m3 (FP8)
```

#### Grid-Stride Loop Pattern

Kernels use grid-stride loops to handle arbitrary tensor sizes efficiently:

```cuda
template <typename T>
__global__ void kernel(T* data, int N) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < N;
         idx += blockDim.x * gridDim.x) {
        // Process element idx
        data[idx] = /* ... */;
    }
}
```

#### Warp-Level Reduction

For reduction operations (layer norm, softmax), kernels use warp-level primitives:

```cuda
#include <cub/block/block_reduce.cuh>

__device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}
```

### Adam Kernel Details

The fused Adam kernel packs 5 operations into a single kernel launch:

1. Weight decay application
2. Momentum (first moment) update
3. Variance (second moment) update
4. Bias correction computation
5. Parameter update with adaptive learning rate

**Block size**: 256 threads per block
**Grid size**: `(N + 255) / 256` blocks
**Memory access**: Coalesced global memory reads/writes
**Register usage**: ~32 registers per thread (FP16), ~48 registers (FP32)

### Transformer Kernel Details

The fused transformer kernel is the most complex custom op. It processes the entire transformer layer in a series of fused CUDA calls:

1. **Layer Norm Kernel**: Mean/variance computation + normalization + scale/shift
2. **QKV GEMM**: Batched GEMM for Q, K, V projections using cuBLAS
3. **Flash Attention**: Tiled attention computation without materializing the full attention matrix
4. **Output GEMM**: Output projection
5. **MLP GEMMs**: Two GEMM operations for the feed-forward network
6. **Residual Addition**: Element-wise addition for residual connections

**cuBLAS Handle Management**: The kernel creates and caches cuBLAS handles to avoid repeated initialization overhead.

### Sparse Attention Kernel Details

The sparse attention kernel uses a block-sparse pattern:

```
Attention matrix (16x16 blocks):
[G] [G] [G] [G]     G = Global block
[L] [L] [  ] [  ]    L = Local block
[  ] [L] [L] [  ]    R = Random block
[R] [  ] [L] [L]     . = Empty (skipped)

For sequence length 16384 with block_size=16:
- Dense attention: 16384^2 = 268M elements
- Sparse attention: ~16384 * 4 * 16 = ~1M elements (256x reduction)
```

---

## Performance Characteristics

### Optimizer Performance Comparison

| Optimizer | Implementation | 7B Params/step | Memory | Notes |
|-----------|---------------|----------------|--------|-------|
| Adam | PyTorch `torch.optim.Adam` | ~280ms | 16 bytes/param | Baseline |
| Adam | DeepSpeed Fused Adam | ~75ms | 16 bytes/param | 3.7x faster |
| Adam | DeepSpeed CPU Adam | ~500ms (CPU) | 16 bytes/param | For offload |
| Lamb | DeepSpeed Fused Lamb | ~85ms | 16 bytes/param | Large batch |
| Lion | DeepSpeed Fused Lion | ~55ms | 8 bytes/param | 2x less memory |
| Adagrad | DeepSpeed CPU Adagrad | ~400ms (CPU) | 12 bytes/param | For offload |

### Transformer Layer Performance

| Configuration | PyTorch | DeepSpeed Fused | Speedup |
|--------------|---------|----------------|---------|
| BERT-Large (FP16) | 45ms | 18ms | 2.5x |
| GPT-3 175B (FP16) | 380ms | 160ms | 2.4x |
| LLaMA-7B (BF16) | 55ms | 25ms | 2.2x |

### Sparse Attention Performance

| Sequence Length | Dense Attention | Sparse Attention | Memory Reduction |
|----------------|-----------------|------------------|------------------|
| 4096 | 16 GB | 1 GB | 16x |
| 8192 | 64 GB | 2 GB | 32x |
| 16384 | 256 GB | 4 GB | 64x |
| 32768 | OOM | 8 GB | >100x |

### Async I/O Performance

| Operation | Synchronous | Async I/O | GDS (Direct) |
|-----------|------------|-----------|-------------|
| Checkpoint save (10 GB) | 12s | 2s | 1.5s |
| Checkpoint load (10 GB) | 10s | 1.8s | 1.2s |
| NVMe swap (1 GB) | 1.2s | 0.2s | 0.15s |

---

## Op Builder API for Custom Ops

### Creating a Custom Op Builder

```python
# my_op_builder.py
from deepspeed.op_builder.cuda_op_builder import CUDAOpBuilder

class MyCustomOpBuilder(CUDAOpBuilder):
    """Builder for a custom CUDA operation."""
    
    BUILD_VAR = "DS_BUILD_MY_CUSTOM_OP"
    NAME = "my_custom_op"
    
    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)
    
    def absolute_name(self):
        """Return the full Python module path."""
        return f"deepspeed.ops.my_custom_op.{self.NAME}"
    
    def sources(self):
        """Return list of source files."""
        return [
            "csrc/my_custom_op/binding.cpp",
            "csrc/my_custom_op/kernel.cu",
        ]
    
    def include_paths(self):
        """Return include directories."""
        return [
            "csrc/includes/",
            "csrc/my_custom_op/",
        ]
    
    def nvcc_args(self):
        """Return NVCC compilation flags."""
        return super().nvcc_args() + [
            "-DMY_CUSTOM_FLAG=1",
        ]
    
    def cxx_args(self):
        """Return C++ compilation flags."""
        return super().cxx_args() + [
            "-DMY_CUSTOM_CPP_FLAG",
        ]
    
    def extra_ldflags(self):
        """Return extra linker flags."""
        return ["-lmylib"] if self._library_available() else []
    
    def _library_available(self):
        import importlib
        try:
            importlib.import_module("mylib")
            return True
        except ImportError:
            return False
    
    def is_compatible(self, verbose=True):
        """Check compatibility."""
        if not super().is_compatible(verbose):
            return False
        # Add custom compatibility checks
        return True
```

### Writing the CUDA Kernel

```cpp
// csrc/my_custom_op/kernel.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename T>
__global__ void my_kernel_impl(
    const T* input,
    T* output,
    const int N,
    const float scale) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = static_cast<T>(static_cast<float>(input[idx]) * scale);
    }
}

// Dispatch function
torch::Tensor my_kernel_forward(
    torch::Tensor input,
    float scale) {
    
    auto output = torch::empty_like(input);
    int N = input.numel();
    
    const int block_size = 256;
    const int grid_size = (N + block_size - 1) / block_size;
    
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "my_kernel_forward", [&] {
            my_kernel_impl<scalar_t><<<grid_size, block_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                N, scale);
        });
    
    return output;
}
```

### Writing the Python Binding

```cpp
// csrc/my_custom_op/binding.cpp
#include <torch/extension.h>

// Forward declarations
torch::Tensor my_kernel_forward(torch::Tensor input, float scale);

// Bind to Python
PYBIND11_MODULE(my_custom_op, m) {
    m.def("forward", &my_kernel_forward, "My custom op forward");
}
```

### Registering with all_ops.py

```python
# deepspeed/op_builder/all_ops.py
from deepspeed.op_builder.my_custom_op import MyCustomOpBuilder

ALL_OPS = {
    # ... existing ops ...
    "my_custom_op": MyCustomOpBuilder,
}
```

---

## Configuration Examples

### Full Training Configuration with Custom Ops

```json
{
    "train_batch_size": 64,
    "gradient_accumulation_steps": 4,
    
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },
    
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "initial_scale_power": 16
    },
    
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true
    },
    
    "gradient_clipping": 1.0,
    
    "steps_per_print": 100,
    "wall_clock_breakdown": true
}
```

### Specifying Optimizer via DeepSpeed Config

```json
{
    "optimizer": {
        "type": "FusedAdam",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "bias_correction": true,
            "adam_w_mode": true,
            "weight_decay": 0.01
        }
    }
}
```

### Sparse Attention Configuration

```json
{
    "sparse_attention": {
        "mode": "sparse",
        "sparsity_config": {
            "attention": "fixed",
            "block": 16,
            "num_heads": 32,
            "num_local_blocks": 4,
            "num_global_blocks": 1,
            "num_random_blocks": 2,
            "attention_type": "uni"
        }
    }
}
```

### CPU Offload with CPU Adam

```json
{
    "optimizer": {
        "type": "CPUAdam",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "weight_decay": 0.01,
            "adamw_mode": true,
            "fp32_param_states": true
        }
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true,
            "buffer_count": 4,
            "fast_init": false
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true,
            "buffer_count": 4
        }
    }
}
```

### Async Checkpointing Configuration

```json
{
    "checkpoint": {
        "use_aio": true,
        "aio_config": {
            "block_size": 1048576,
            "queue_depth": 8,
            "single_submit": false,
            "overlap_events": true,
            "thread_count": 2
        }
    }
}
```

---

## Troubleshooting

### JIT Compilation Failure

**Symptom**: `RuntimeError: Error building extension '<op_name>'`

**Common causes and solutions**:

1. **Missing CUDA toolkit**:
   ```bash
   # Verify CUDA toolkit
   nvcc --version
   export CUDA_HOME=/usr/local/cuda
   ```

2. **Incompatible GCC version**:
   ```bash
   # CUDA requires specific GCC versions
   # CUDA 12.x requires GCC <= 12
   gcc --version
   ```

3. **Out of disk space for build cache**:
   ```bash
   # Check cache directory
   du -sh ~/.cache/torch_extensions/
   rm -rf ~/.cache/torch_extensions/
   ```

### Op Not Available at Runtime

**Symptom**: `ImportError: Cannot import deepspeed.ops.<op_name>`

**Solutions**:
1. Rebuild ops: `DS_BUILD_OPS=1 pip install -e . --no-build-isolation`
2. Check build logs for specific op failures
3. Use `ds_report` to check op availability

### CUDA Architecture Mismatch

**Symptom**: `RuntimeError: CUDA error: no kernel image is available for execution on the device`

**Solution**:
```bash
# Set architecture list to include your GPU
export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"

# Rebuild all ops
DS_BUILD_OPS=1 pip install -e . --force-reinstall --no-deps
```

### Memory Issues During Compilation

**Symptom**: `cicc` process killed during compilation (OOM)

**Solution**:
```bash
# Reduce parallel compilation jobs
export MAX_JOBS=4

# Or compile one op at a time
DS_BUILD_FUSED_ADAM=1 DS_BUILD_FUSED_LAMB=0 DS_BUILD_TRANSFORMER=0 pip install .
```

### Sparse Attention Build Failure

**Symptom**: `sparse_attn` fails to compile

**Solutions**:
1. Ensure Triton is installed: `pip install triton`
2. Skip sparse attention if not needed: `export DS_BUILD_SPARSE_ATTN=0`

### libaio Not Found

**Symptom**: `async_io` build fails with `cannot find -laio`

**Solution**:
```bash
# Install libaio development headers
sudo apt-get install libaio-dev     # Ubuntu/Debian
sudo yum install libaio-devel       # CentOS/RHEL
```

### GDS Build Failure

**Symptom**: `gds` build fails

**Solutions**:
1. Ensure CUDA 11.4+ and nvidia-fs driver installed
2. Skip GDS if not needed: `export DS_BUILD_GDS=0`

---

## Quick Reference: Op Builder API

```python
from deepspeed.ops.op_builder import all_ops

# List all available ops
for name, builder_cls in all_ops.ALL_OPS.items():
    builder = builder_cls()
    print(f"{name}: compatible={builder.is_compatible()}")

# Check specific op
from deepspeed.ops.op_builder.fused_adam import FusedAdamBuilder
builder = FusedAdamBuilder()
print(f"Sources: {builder.sources()}")
print(f"NVCC args: {builder.nvcc_args()}")
print(f"CXX args: {builder.cxx_args()}")
print(f"Absolute name: {builder.absolute_name()}")
print(f"Compatible: {builder.is_compatible()}")

# Load op module
module = builder.load()
print(f"Loaded: {module}")
```

---

## Op Dependency Graph

```
DeepSpeed Engine
    |
    +-- ZeRO Stage 1/2/3
    |       |
    |       +-- fused_adam / fused_lamb / fused_lion
    |       +-- cpu_adam / cpu_adagrad / cpu_lion (offload)
    |       +-- async_io (checkpoint/NVMe)
    |       +-- gds (GPUDirect Storage)
    |
    +-- Transformer Training
    |       |
    |       +-- transformer (fused layer)
    |       +-- stochastic_transformer
    |       +-- sparse_attn (long sequences)
    |       +-- random_ltd
    |
    +-- Inference V1
    |       |
    |       +-- transformer_inference
    |       +-- inference_cutlass_builder
    |       +-- quantizer
    |       +-- spatial_inference
    |
    +-- Inference V2
    |       |
    |       +-- ragged_utils
    |       +-- ragged_ops
    |       +-- inference_core_ops
    |       +-- quantizer
    |
    +-- DeepCompile
    |       |
    |       +-- dc (activation ops)
    |
    +-- DeepSpeed4Science
            |
            +-- evoformer_attn
```
