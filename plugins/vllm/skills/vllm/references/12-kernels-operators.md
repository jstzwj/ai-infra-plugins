# vLLM Kernels and Operators Reference

This document provides a comprehensive reference for vLLM's kernel dispatch system,
custom operators, fused MoE (Mixture of Experts) infrastructure, Flash Attention
interface, Triton kernels, Helion kernel integration, and platform-specific operator
implementations. Every function, class, method, parameter, and type is documented
with full signatures and descriptions.

---

## Table of Contents

1. [Kernel IR Op Dispatch System](#1-kernel-ir-op-dispatch-system)
   - [Kernel Module Initialization (`vllm/kernels/__init__.py`)](#kernel-module-initialization)
   - [vllm_c Backend (`vllm/kernels/vllm_c.py`)](#vllm_c-backend)
   - [AITER Backend (`vllm/kernels/aiter_ops.py`)](#aiter-backend)
   - [XPU Backend (`vllm/kernels/xpu_ops.py`)](#xpu-backend)
   - [Oink Backend (`vllm/kernels/oink_ops.py`)](#oink-backend)
2. [Custom Operations (`vllm/_custom_ops.py`)](#2-custom-operations)
   - [FP4 Quantization Ops](#fp4-quantization-ops)
   - [Paged Attention Ops](#paged-attention-ops)
   - [Position Encoding Ops](#position-encoding-ops)
   - [Layer Norm Ops](#layer-norm-ops)
   - [Repetition Penalty Ops](#repetition-penalty-ops)
   - [Fused Quant Layer Norm Ops](#fused-quant-layer-norm-ops)
   - [Quantization Ops](#quantization-ops)
     - [AWQ Ops](#awq-ops)
     - [GPTQ Ops](#gptq-ops)
     - [CUTLASS Scaled MM Ops](#cutlass-scaled-mm-ops)
     - [CUTLASS MoE MM Ops](#cutlass-moe-mm-ops)
     - [GPTQ Marlin Ops](#gptq-marlin-ops)
     - [AWQ Marlin Ops](#awq-marlin-ops)
     - [Marlin GEMM Ops](#marlin-gemm-ops)
     - [Machete Ops](#machete-ops)
     - [CUTLASS W4A8 Ops](#cutlass-w4a8-ops)
     - [FP8 Quantization Ops](#fp8-quantization-ops)
     - [INT8 Quantization Ops](#int8-quantization-ops)
     - [GGUF Ops](#gguf-ops)
     - [FP4 Expert Quantization Ops](#fp4-expert-quantization-ops)
     - [MXFP4 Expert Quantization Ops](#mxfp4-expert-quantization-ops)
     - [AllSpark Ops](#allspark-ops)
   - [Mamba Selective Scan Ops](#mamba-selective-scan-ops)
   - [ROCm Skinny GEMM Ops](#rocm-skinny-gemm-ops)
   - [MoE Dispatch Ops](#moe-dispatch-ops)
   - [Top-K Routing Ops](#top-k-routing-ops)
   - [Cache Ops](#cache-ops)
   - [Custom All-Reduce Ops](#custom-all-reduce-ops)
   - [CUDA Utils Ops](#cuda-utils-ops)
3. [Fused MoE Infrastructure](#3-fused-moe-infrastructure)
   - [MoE Activation (`activation.py`)](#moe-activation)
   - [MoE Config (`config.py`)](#moe-config)
   - [MoE Router (`router/fused_moe_router.py`)](#moe-router)
   - [FusedMoEMethodBase (`fused_moe_method_base.py`)](#fusedmoemethodbase)
   - [FusedMoEModularMethod (`fused_moe_modular_method.py`)](#fusedmoemodularmethod)
   - [Modular Kernel Framework (`modular_kernel.py`)](#modular-kernel-framework)
   - [FusedMoE Layer (`layer.py`)](#fusedmoe-layer)
   - [Module Exports (`__init__.py`)](#fused-moe-module-exports)
4. [Flash Attention Interface](#4-flash-attention-interface)
   - [Module Init (`vllm_flash_attn/__init__.py`)](#flash-attn-module-init)
   - [FA Interface (`flash_attn_interface.py`)](#flash-attn-interface)
5. [Triton Kernel Utilities](#5-triton-kernel-utilities)
   - [Triton Import Detection (`triton_utils/importing.py`)](#triton-import-detection)
   - [JIT Monitor (`triton_utils/jit_monitor.py`)](#jit-monitor)
   - [Triton Allocator (`triton_utils/allocation.py`)](#triton-allocator)
   - [Module Re-exports (`triton_utils/__init__.py`)](#triton-module-re-exports)
6. [Triton FP8 Quantization Kernel](#6-triton-fp8-quantization-kernel)
   - [QKV Padded FP8 Quant (`kernels/triton/qkv_padded_fp8_quant.py`)](#qkv-padded-fp8-quant)
7. [Helion Kernel Integration](#7-helion-kernel-integration)
   - [Helion Module Init (`kernels/helion/__init__.py`)](#helion-module-init)
   - [Kernel Registration (`kernels/helion/register.py`)](#helion-kernel-registration)
   - [Config Manager (`kernels/helion/config_manager.py`)](#helion-config-manager)
   - [GPU Name Utilities (`kernels/helion/utils.py`)](#helion-gpu-name-utils)
   - [SiLU Mul FP8 Op (`kernels/helion/ops/silu_mul_fp8.py`)](#helion-silu-mul-fp8)
8. [AITER (AMD ROCm) Operations](#8-aiter-amd-rocm-operations)
   - [`rocm_aiter_ops` Class](#rocm-aiter-ops-class)
9. [XPU (Intel) Operations](#9-xpu-intel-operations)
   - [`xpu_ops` Class](#xpu-ops-class)

---

## 1. Kernel IR Op Dispatch System

vLLM uses a kernel IR (Intermediate Representation) dispatch system to route operations
to the best available backend for the current platform. Each IR op can have multiple
implementations registered under different backend names (e.g., `vllm_c`, `aiter`,
`xpu_kernels`, `oink`). At dispatch time, the system selects the highest-priority
supported implementation based on the current platform and argument compatibility.

### Kernel Module Initialization

**File**: `vllm/kernels/__init__.py`

```python
from . import aiter_ops, oink_ops, vllm_c, xpu_ops
__all__ = ["vllm_c", "aiter_ops", "oink_ops", "xpu_ops"]
```

Importing this module triggers registration of all IR op implementations for the four
supported backends.

### vllm_c Backend

**File**: `vllm/kernels/vllm_c.py`

The `vllm_c` backend wraps C++/CUDA extension ops registered under `torch.ops._C`.
It is the default backend for all CUDA-alike platforms.

#### Constants

```python
CUDA_ALIKE: bool = current_platform.is_cuda_alike()
```
Whether the current platform supports CUDA-alike kernels.

```python
rms_no_var_size: Callable
```
Predicate: returns `True` when `variance_size is None` and `weight.dtype == x.dtype`
(or weight is None). Used as the `supports_args` guard for `rms_norm`.

```python
rms_add_no_var_size: Callable
```
Predicate: returns `True` when `variance_size is None` and `weight.dtype == x.dtype`
(or weight is None). Used as the `supports_args` guard for `fused_add_rms_norm`.

#### Functions

##### `rms_norm`

```python
@ir.ops.rms_norm.register_impl(
    "vllm_c", supports_args=rms_no_var_size, supported=CUDA_ALIKE
)
def rms_norm(
    x: Tensor,
    weight: Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> Tensor
```

Computes RMS normalization: `x / sqrt(mean(x^2) + epsilon) * weight`.

- If `weight` is `None`, a ones tensor is created internally.
- Asserts `variance_size is None`.
- Returns a new tensor with the same shape as `x`.

##### `fused_add_rms_norm`

```python
@ir.ops.fused_add_rms_norm.register_impl(
    "vllm_c",
    supports_args=rms_add_no_var_size,
    supported=CUDA_ALIKE,
    inplace=True,
)
def fused_add_rms_norm(
    x: Tensor,
    x_residual: Tensor,
    weight: Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> tuple[Tensor, Tensor]
```

Fused operation computing `x = x + x_residual` then RMS-normalizing the result.
Modifies `x` and `x_residual` in place (registered with `inplace=True`).

- Returns the modified `(x, x_residual)` tuple.

### AITER Backend

**File**: `vllm/kernels/aiter_ops.py`

The AITER backend wraps AMD's AITER (AMD Instinct Triton Extensions for ROCm)
library for ROCm GPU acceleration.

#### Constants

```python
AITER_SUPPORTED: bool = is_aiter_found()
```
Whether the `aiter` Python package is installed.

#### Module-Level Objects

```python
aiter_lib: Library = Library("vllm_aiter", "FRAGMENT")
```
A PyTorch `Library` in FRAGMENT mode that holds custom ops wrapping AITER operations.
These ops remain invisible to `torch.compile` even after lowering.

```python
direct_register_aiter_op: Callable
```
Partial application of `direct_register_custom_op` with `target_lib=aiter_lib`.
Used to register AITER custom ops concisely.

#### Helper Functions

##### `is_aiter_found`

```python
def is_aiter_found() -> bool
```

Checks whether the `aiter` package is importable via `importlib.util.find_spec`.

#### Guards

```python
rms_no_var_16bit_only: Callable
```
Predicate: returns `True` when `variance_size is None`, `x.dtype` is `float16` or
`bfloat16`, and `weight.dtype` matches `x.dtype` (or weight is None).

```python
rms_add_no_var_16bit_only: Callable
```
Same constraints as `rms_no_var_16bit_only` for the fused_add_rms_norm op.

#### IR Op Implementations

##### `rms_norm` (AITER)

```python
@ir.ops.rms_norm.register_impl(
    "aiter", supports_args=rms_no_var_16bit_only, supported=AITER_SUPPORTED
)
def rms_norm(
    x: Tensor, weight: Tensor | None, epsilon: float, variance_size: int | None = None
) -> Tensor
```

Dispatches to `torch.ops.vllm_aiter.rms_norm(x, weight, epsilon)`.

##### `fused_add_rms_norm` (AITER)

```python
@ir.ops.fused_add_rms_norm.register_impl(
    "aiter", supports_args=rms_add_no_var_16bit_only, supported=AITER_SUPPORTED
)
def fused_add_rms_norm(
    x: Tensor, x_residual: Tensor, weight: Tensor | None, epsilon: float,
    variance_size: int | None = None,
) -> tuple[Tensor, Tensor]
```

Dispatches to `torch.ops.vllm_aiter.fused_add_rms_norm(x, x_residual, weight, epsilon)`.

#### Internal Implementation Functions

##### `_rms_norm_impl`

```python
def _rms_norm_impl(x: Tensor, weight: Tensor, variance_epsilon: float) -> Tensor
```

Calls `aiter.rms_norm`. Reshapes input to 2D if it has more than 2 dimensions,
then restores the original shape.

##### `_rms_norm_fake`

```python
def _rms_norm_fake(x: Tensor, weight: Tensor, variance_epsilon: float) -> Tensor
```

Fake implementation for `torch.compile` tracing. Returns `torch.empty_like(x)`.

##### `_rocm_aiter_rmsnorm2d_fwd_with_add_impl`

```python
def _rocm_aiter_rmsnorm2d_fwd_with_add_impl(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
    variance_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]
```

Calls `aiter.rmsnorm2d_fwd_with_add` with separate output and residual output buffers.

##### `_rocm_aiter_rmsnorm2d_fwd_with_add_fake`

```python
def _rocm_aiter_rmsnorm2d_fwd_with_add_fake(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
    variance_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]
```

Fake implementation returning empty tensors matching input shapes.

### XPU Backend

**File**: `vllm/kernels/xpu_ops.py`

The XPU backend provides Intel GPU implementations using `torch.ops._C` (the same
namespace as CUDA, but gated on `is_xpu_kernels_found()`). The registered IR ops
and their signatures are identical to the `vllm_c` backend, differing only in the
platform guard.

#### Functions

##### `rms_norm` (XPU)

```python
@ir.ops.rms_norm.register_impl(
    "xpu_kernels", supports_args=rms_no_var_size, supported=XPU_SUPPORTED
)
def rms_norm(x: Tensor, weight: Tensor | None, epsilon: float,
             variance_size: int | None = None) -> Tensor
```

Same semantics as `vllm_c.rms_norm` but for Intel XPU platforms.

##### `fused_add_rms_norm` (XPU)

```python
@ir.ops.fused_add_rms_norm.register_impl(
    "xpu_kernels", supports_args=rms_add_no_var_size, supported=XPU_SUPPORTED,
    inplace=True,
)
def fused_add_rms_norm(x: Tensor, x_residual: Tensor, weight: Tensor | None,
                       epsilon: float, variance_size: int | None = None
                       ) -> tuple[Tensor, Tensor]
```

Same semantics as `vllm_c.fused_add_rms_norm` for Intel XPU platforms.

### Oink Backend

**File**: `vllm/kernels/oink_ops.py`

The Oink backend provides implementations from an external Oink plugin. Ops are
registered under the `torch.ops.oink` namespace. Oink is available only on devices
with compute capability >= 100 (Blackwell+).

#### Constants

```python
OINK_AVAILABLE: bool = current_platform.has_device_capability(100) and hasattr(torch.ops, "oink")
```

#### Helper Functions

##### `has_oink_op`

```python
def has_oink_op(name: str) -> bool
```

Returns `True` if `OINK_AVAILABLE` is `True` and the named op exists in
`torch.ops.oink`.

##### `_can_view_as_2d`

```python
def _can_view_as_2d(x: Tensor) -> bool
```

Returns `True` if `x.view(-1, x.shape[-1])` can be done as a view (no copy).
Checks that all leading dimensions are contiguous with respect to each other.

##### `_is_oink_stride_compatible_2d`

```python
def _is_oink_stride_compatible_2d(x_2d: Tensor) -> bool
```

Returns `True` if the 2D tensor meets Oink's pointer-path stride constraints:
- `stride(1) == 1` (contiguous in the last dimension)
- `stride(0)` divisible by 16 for float16/bfloat16, or 8 for float32

#### Guards

```python
oink_rms_supported: Callable
```
Predicate: `variance_size is None`, weight is not None, x is >= 2D, `x.dtype == weight.dtype`,
weight is contiguous, x can be viewed as 2D, and the 2D view meets Oink stride constraints.

```python
oink_add_rms_supported: Callable
```
Same as `oink_rms_supported` plus residual must match x in shape/dtype and also
be 2D-viewable with compatible strides.

#### IR Op Implementations

##### `rms_norm` (Oink)

```python
@ir.ops.rms_norm.register_impl(
    "oink", supports_args=oink_rms_supported, supported=has_oink_op("rmsnorm")
)
def rms_norm(x: Tensor, weight: Tensor | None, epsilon: float,
             variance_size: int | None = None) -> Tensor
```

Reshapes `x` to 2D, calls `torch.ops.oink.rmsnorm(x_2d, weight, epsilon)`, then
restores the original shape.

##### `fused_add_rms_norm` (Oink)

```python
@ir.ops.fused_add_rms_norm.register_impl(
    "oink", supports_args=oink_add_rms_supported,
    supported=has_oink_op("fused_add_rms_norm"), inplace=True,
)
def fused_add_rms_norm(x: Tensor, x_residual: Tensor, weight: Tensor | None,
                       epsilon: float, variance_size: int | None = None
                       ) -> tuple[Tensor, Tensor]
```

Reshapes both `x` and `x_residual` to 2D, calls
`torch.ops.oink.fused_add_rms_norm(x_2d, residual_2d, weight, epsilon)` in place,
then returns the original tensors.

---

## 2. Custom Operations

**File**: `vllm/_custom_ops.py`

This massive module (~2950+ lines) defines Python wrappers around all C++/CUDA
extension ops used throughout vLLM. It handles fake implementations for `torch.compile`
tracing, platform-specific dispatch, and out-variant memory management.

### FP4 Quantization Ops

#### `create_fp4_scale_tensor`

```python
def create_fp4_scale_tensor(
    m: int,
    n: int,
    device: torch.device,
    is_sf_swizzled_layout: bool,
) -> torch.Tensor
```

Allocates the output scale tensor for `scaled_fp4_quant`.

- When `is_sf_swizzled_layout=True`: pads rows to multiples of 128 and columns to
  multiples of 4, stores scales packed as int32 (every 4 float8_e4m3fn values).
  Shape: `(round_up(m, 128), round_up(n//16, 4) // 4)` with dtype `int32`.
- When `is_sf_swizzled_layout=False`: shape `(m, n // 16)` with dtype `uint8`.

#### `create_fp4_output_tensors`

```python
def create_fp4_output_tensors(
    m: int,
    n: int,
    device: torch.device,
    is_sf_swizzled_layout: bool,
) -> tuple[torch.Tensor, torch.Tensor]
```

Allocates both output tensors for `scaled_fp4_quant`:
- `output`: shape `(m, n // 2)`, dtype `uint8` (packed FP4 values).
- `output_scale`: allocated via `create_fp4_scale_tensor`.

#### `_scaled_fp4_quant_fake`

```python
@register_fake("_C::scaled_fp4_quant")
def _scaled_fp4_quant_fake(
    input: torch.Tensor,
    input_scale: torch.Tensor,
    is_sf_swizzled_layout: bool,
) -> tuple[torch.Tensor, torch.Tensor]
```

Fake impl for torch.compile. Calls `create_fp4_output_tensors`.

#### `_scaled_fp4_quant_out_fake`

```python
@register_fake("_C::scaled_fp4_quant.out")
def _scaled_fp4_quant_out_fake(
    input: torch.Tensor,
    input_scale: torch.Tensor,
    is_sf_swizzled_layout: bool,
    *,
    output: torch.Tensor,
    output_scale: torch.Tensor,
) -> None
```

Fake impl for the out-variant. Returns `None`.

### Paged Attention Ops

#### `paged_attention_v1`

```python
def paged_attention_v1(
    out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_seq_len: int,
    alibi_slopes: torch.Tensor | None,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    tp_rank: int = 0,
    blocksparse_local_blocks: int = 0,
    blocksparse_vert_stride: int = 0,
    blocksparse_block_size: int = 64,
    blocksparse_head_sliding_step: int = 0,
) -> None
```

Paged attention kernel v1. Direct wrapper around `torch.ops._C.paged_attention_v1`.

Parameters:
- `out`: Output tensor `[num_seqs, num_heads, head_size]`
- `query`: Query tensor `[num_seqs, num_heads, head_size]`
- `key_cache`: Paged key cache `[num_blocks, block_size, num_kv_heads, head_size]`
- `value_cache`: Paged value cache `[num_blocks, block_size, num_kv_heads, head_size]`
- `num_kv_heads`: Number of KV heads
- `scale`: Softmax scale factor
- `block_tables`: Block tables `[num_seqs, max_num_blocks_per_seq]`
- `seq_lens`: Sequence lengths `[num_seqs]`
- `block_size`: Number of tokens per block
- `max_seq_len`: Maximum sequence length
- `alibi_slopes`: Optional ALiBi slopes `[num_heads]` or `[num_seqs, num_heads]`
- `kv_cache_dtype`: KV cache data type string (e.g., `"auto"`, `"fp8"`)
- `k_scale`: Key scale factor for FP8 KV cache
- `v_scale`: Value scale factor for FP8 KV cache
- `tp_rank`: Tensor parallel rank (default 0)
- `blocksparse_local_blocks`: Number of local blocks for blocksparse attention
- `blocksparse_vert_stride`: Vertical stride for blocksparse
- `blocksparse_block_size`: Block size for blocksparse (default 64)
- `blocksparse_head_sliding_step`: Sliding step per head for blocksparse

#### `paged_attention_v2`

```python
def paged_attention_v2(
    out: torch.Tensor,
    exp_sum: torch.Tensor,
    max_logits: torch.Tensor,
    tmp_out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_seq_len: int,
    alibi_slopes: torch.Tensor | None,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    tp_rank: int = 0,
    blocksparse_local_blocks: int = 0,
    blocksparse_vert_stride: int = 0,
    blocksparse_block_size: int = 64,
    blocksparse_head_sliding_step: int = 0,
) -> None
```

Paged attention kernel v2 with separate reduction buffers. Same parameters as v1
plus intermediate reduction tensors:
- `exp_sum`: Log-sum-exp values `[num_seqs, num_heads]`
- `max_logits`: Max logits `[num_seqs, num_heads]`
- `tmp_out`: Temporary output `[num_seqs, num_heads, head_size]`

#### `paged_attention_rocm`

```python
def paged_attention_rocm(
    out: torch.Tensor,
    exp_sum: torch.Tensor,
    max_logits: torch.Tensor,
    tmp_out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor | None,
    block_size: int,
    max_seq_len: int,
    alibi_slopes: torch.Tensor | None,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    fp8_out_scale: torch.Tensor | None = None,
    mfma_type: str = "fp8" if envs.VLLM_ROCM_FP8_MFMA_PAGE_ATTN else "f16",
) -> None
```

ROCm-specific paged attention via `torch.ops._rocm_C.paged_attention`. Supports
MFMA (Matrix Fused Multiply-Add) types for FP8 matrix operations on MI300.

#### `mla_decode_kvcache_cpu`

```python
def mla_decode_kvcache_cpu(
    out: torch.Tensor,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
) -> None
```

MLA (Multi-head Latent Attention) decode from KV cache on CPU.

### Merge Attention States Ops

#### `merge_attn_states`

```python
def merge_attn_states(
    output: torch.Tensor,
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output_lse: torch.Tensor | None = None,
    prefill_tokens_with_context: int | None = None,
    output_scale: torch.Tensor | None = None,
) -> None
```

Merges prefix and suffix attention states. Used for chunked prefill where
attention is computed over prefix and suffix separately then combined.

### Index Conversion Ops

#### `convert_vertical_slash_indexes`

```python
def convert_vertical_slash_indexes(
    q_seqlens: torch.Tensor,       # [BATCH]
    kv_seqlens: torch.Tensor,      # [BATCH]
    vertical_indexes: torch.Tensor, # [BATCH, N_HEADS, NNZ_V]
    slash_indexes: torch.Tensor,    # [BATCH, N_HEADS, NNZ_S]
    context_size: int,
    block_size_M: int,
    block_size_N: int,
    causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
```

Converts vertical/slash sparse attention indexes to block-level representations.
Returns `(block_count, block_offset, column_count, column_index)`.

#### `convert_vertical_slash_indexes_mergehead`

```python
def convert_vertical_slash_indexes_mergehead(
    q_seqlens: torch.Tensor,
    kv_seqlens: torch.Tensor,
    vertical_indexes: torch.Tensor,
    slash_indexes: torch.Tensor,
    vertical_indices_count: torch.Tensor,
    slash_indices_count: torch.Tensor,
    context_size: int,
    block_size_M: int,
    block_size_N: int,
    causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
```

Same as `convert_vertical_slash_indexes` but supports per-head index counts,
allowing different heads to use different numbers of vertical/slash indices.

### Position Encoding Ops

#### `rotary_embedding`

```python
def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    rope_dim_offset: int = 0,
    inverse: bool = False,
) -> None
```

Applies Rotary Position Embedding (RoPE) to query and key tensors in place.

Parameters:
- `positions`: Position indices `[num_seqs]`
- `query`: Query tensor, modified in place
- `key`: Key tensor, modified in place (can be None for query-only)
- `head_size`: Dimension of each attention head
- `cos_sin_cache`: Pre-computed cos/sin values `[max_position, 2, head_size // 2]`
- `is_neox`: Whether to use Neox-style (interleaved) RoPE
- `rope_dim_offset`: Offset for RoPE dimension (default 0)
- `inverse`: Whether to apply inverse RoPE (default False)

### Layer Norm Ops

#### `rms_norm`

```python
def rms_norm(
    out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> None
```

Computes RMS normalization: `out = input / sqrt(mean(input^2) + eps) * weight`.

#### `fused_add_rms_norm`

```python
def fused_add_rms_norm(
    input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> None
```

Fused add + RMS norm: `input += residual; residual = input; input = rms_norm(input)`.
Both tensors are modified in place.

#### `fused_qk_norm_rope`

```python
def fused_qk_norm_rope(
    qkv: torch.Tensor,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    position_ids: torch.Tensor,
    forced_token_heads_per_warp: int = -1,
) -> None
```

Fused QK normalization + RoPE applied to a packed QKV tensor.
Applies RMS norm to Q and K separately, then applies RoPE.

### Repetition Penalty Ops

#### `apply_repetition_penalties`

```python
def apply_repetition_penalties(
    logits: torch.Tensor,
    prompt_mask: torch.Tensor,
    output_mask: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> None
```

Applies repetition penalties to logits in place. Dispatches to CUDA kernel if
on GPU with contiguous tensors, otherwise uses PyTorch implementation.

Parameters:
- `logits`: Shape `[num_seqs, vocab_size]`
- `prompt_mask`: Boolean mask for prompt tokens
- `output_mask`: Boolean mask for output tokens
- `repetition_penalties`: Shape `(num_seqs,)`

#### `apply_repetition_penalties_cuda`

```python
def apply_repetition_penalties_cuda(
    logits: torch.Tensor, prompt_mask: torch.Tensor,
    output_mask: torch.Tensor, repetition_penalties: torch.Tensor,
) -> None
```

CUDA implementation via `torch.ops._C.apply_repetition_penalties_`.

#### `apply_repetition_penalties_torch`

```python
def apply_repetition_penalties_torch(
    logits: torch.Tensor, prompt_mask: torch.Tensor,
    output_mask: torch.Tensor, repetition_penalties: torch.Tensor,
) -> None
```

Pure PyTorch fallback. Divides positive logits by penalty, multiplies negative
logits by penalty.

### Fused Quant Layer Norm Ops

#### `rms_norm_dynamic_per_token_quant`

```python
def rms_norm_dynamic_per_token_quant(
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    quant_dtype: torch.dtype,
    scale_ub: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]
```

Fused RMS norm + dynamic per-token quantization. Returns `(output, scales)` where
`output` has `quant_dtype` and `scales` has shape `(num_tokens, 1)` with float32.

#### `rms_norm_per_block_quant`

```python
def rms_norm_per_block_quant(
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    quant_dtype: torch.dtype,
    group_size: list[int],
    scale_ub: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    is_scale_transposed: bool = False,
    tma_alignment: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]
```

Fused RMS norm + per-block quantization for block-wise quantization schemes
like DeepSeek V3. Returns `(output, scales)`.

Parameters:
- `group_size`: `[group_m, group_n]` block dimensions for quantization
- `is_scale_transposed`: Whether to transpose the scales tensor
- `tma_alignment`: TMA descriptor alignment (0 or 4)

#### `silu_and_mul_per_block_quant`

```python
def silu_and_mul_per_block_quant(
    input: torch.Tensor,
    group_size: int,
    quant_dtype: torch.dtype,
    scale_ub: torch.Tensor | None = None,
    is_scale_transposed: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]
```

Fused SiLU-and-multiply + per-block quantization. Input has `[gate || up]` layout
with shape `(num_tokens, hidden * 2)`. Output has shape `(num_tokens, hidden)`.

### Quantization Ops

#### AWQ Ops

##### `awq_dequantize`

```python
def awq_dequantize(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    split_k_iters: int,
    thx: int,
    thy: int,
) -> torch.Tensor
```

Dequantizes AWQ-compressed weights. Dispatches to Triton implementation if
`VLLM_USE_TRITON_AWQ` is set.

##### `awq_gemm`

```python
def awq_gemm(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    split_k_iters: int,
) -> torch.Tensor
```

AWQ quantized GEMM. Dispatches to Triton implementation if `VLLM_USE_TRITON_AWQ`
is set.

#### GPTQ Ops

##### `gptq_gemm`

```python
def gptq_gemm(
    a: torch.Tensor,
    b_q_weight: torch.Tensor,
    b_gptq_qzeros: torch.Tensor,
    b_gptq_scales: torch.Tensor,
    b_g_idx: torch.Tensor,
    use_exllama: bool,
    use_v2_format: bool,
    bit: int,
) -> torch.Tensor
```

GPTQ quantized GEMM via `torch.ops._C.gptq_gemm`.

##### `gptq_shuffle`

```python
def gptq_shuffle(q_weight: torch.Tensor, q_perm: torch.Tensor, bit: int) -> None
```

Shuffles GPTQ weight tensor according to permutation, in place.

#### CUTLASS Scaled MM Ops

##### `cutlass_scaled_mm_supports_fp4`

```python
def cutlass_scaled_mm_supports_fp4(cuda_device_capability: int) -> bool
```

Checks if the CUDA device supports FP4 CUTLASS scaled matrix multiply.

##### `cutlass_scaled_fp4_mm`

```python
def cutlass_scaled_fp4_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    block_scale_a: torch.Tensor,
    block_scale_b: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor
```

FP4 scaled matrix multiply using CUTLASS. Returns output of shape `(m, n)`.

##### `cutlass_scaled_mm_supports_fp8`

```python
def cutlass_scaled_mm_supports_fp8(cuda_device_capability: int) -> bool
```

Checks if FP8 CUTLASS scaled MM is supported.

##### `cutlass_scaled_mm_supports_block_fp8`

```python
def cutlass_scaled_mm_supports_block_fp8(cuda_device_capability: int) -> bool
```

Checks if block-wise FP8 CUTLASS scaled MM is supported.

##### `cutlass_scaled_mm`

```python
def cutlass_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor
```

FP8 scaled matrix multiply: `output = (scale_a * a) @ (scale_b * b).to(out_dtype)`.

Supports extended group broadcasting for block-wise scaling:
- `scale_a.shape * [1, 128] == a.shape`
- `scale_b.shape * [128, 128] == b.shape`

On ROCm or non-CUTLASS-compatible sizes, falls back to `triton_scaled_mm`.

##### `cutlass_scaled_mm_azp`

```python
def cutlass_scaled_mm_azp(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    azp_adj: torch.Tensor,
    azp: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor
```

CUTLASS scaled MM with asymmetric zero points (AZP). The `azp_adj` is always
per-channel and includes the AZP in the per-tensor case. `azp` is per-token
only in the per-token case.

##### `cutlass_group_gemm_supported`

```python
def cutlass_group_gemm_supported(cuda_device_capability: int) -> bool
```

Returns `True` if CUTLASS group GEMM is supported (compute capability 90-109).

#### CUTLASS MoE MM Ops

##### `get_cutlass_moe_mm_data`

```python
def get_cutlass_moe_mm_data(
    topk_ids: torch.Tensor,
    expert_offsets: torch.Tensor,
    problem_sizes1: torch.Tensor,
    problem_sizes2: torch.Tensor,
    input_permutation: torch.Tensor,
    output_permutation: torch.Tensor,
    num_experts: int,
    n: int,
    k: int,
    blockscale_offsets: torch.Tensor | None = None,
    is_gated: bool = True,
) -> ...
```

Prepares data for CUTLASS grouped matrix multiplications in fused MoE. Computes
expert offsets, problem sizes, and permutations from the token-expert mapping.

##### `get_cutlass_moe_mm_problem_sizes_from_expert_offsets`

```python
def get_cutlass_moe_mm_problem_sizes_from_expert_offsets(
    expert_first_token_offset: torch.Tensor,
    problem_sizes1: torch.Tensor,
    problem_sizes2: torch.Tensor,
    n: int,
    k: int,
    swap_ab: bool,
) -> ...
```

Computes per-expert `(M, N, K)` problem sizes from expert token offsets.

##### `shuffle_rows`

```python
def shuffle_rows(input_tensor: torch.Tensor, dst2src_map: torch.Tensor) -> torch.Tensor
```

Shuffles and expands input tensor according to dst2src_map for MoE permutation.

##### `get_cutlass_batched_moe_mm_data`

```python
def get_cutlass_batched_moe_mm_data(
    expert_offsets: torch.Tensor,
    problem_sizes1: torch.Tensor,
    problem_sizes2: torch.Tensor,
    expert_num_tokens: torch.Tensor,
    num_local_experts: int,
    padded_m: int,
    n: int,
    k: int,
) -> ...
```

Prepares data for batched CUTLASS MoE. Takes per-expert token counts.

##### `cutlass_moe_mm`

```python
def cutlass_moe_mm(
    out_tensors: torch.Tensor,
    a_tensors: torch.Tensor,
    b_tensors: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    expert_offsets: torch.Tensor,
    problem_sizes: torch.Tensor,
    a_strides: torch.Tensor,
    b_strides: torch.Tensor,
    c_strides: torch.Tensor,
    per_act_token: bool,
    per_out_ch: bool,
) -> ...
```

Single grouped matrix multiplication for FP8 CUTLASS MoE.

##### `cutlass_fp4_moe_mm`

```python
def cutlass_fp4_moe_mm(
    out_tensors: torch.Tensor,
    a_tensors: torch.Tensor,
    b_tensors: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    alphas: torch.Tensor,
    problem_sizes: torch.Tensor,
    expert_offsets: torch.Tensor,
    sf_offsets: torch.Tensor,
) -> ...
```

NVFP4 block-scaled grouped GEMM for MoE via `torch.ops._C.cutlass_fp4_group_mm`.

##### `cutlass_mxfp4_moe_mm`

```python
def cutlass_mxfp4_moe_mm(
    out_tensors: torch.Tensor,
    a_tensors: torch.Tensor,
    b_tensors: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    problem_sizes: torch.Tensor,
    expert_offsets: torch.Tensor,
    sf_offsets: torch.Tensor,
) -> ...
```

MXFP4 block-scaled grouped GEMM for MoE. Uses E8M0 scale factors with 32-element
blocks via `torch.ops._C.cutlass_mxfp4_group_mm`.

##### `mxfp8_experts_quant`

```python
def mxfp8_experts_quant(
    input_tensor: torch.Tensor,
    problem_sizes: torch.Tensor,
    expert_offsets: torch.Tensor,
    blockscale_offsets: torch.Tensor,
    quant_output: torch.Tensor,
    scale_factor: torch.Tensor,
) -> None
```

MXFP8 per-expert quantization for MoE activations.

##### `cutlass_mxfp8_grouped_mm`

```python
def cutlass_mxfp8_grouped_mm(
    a_tensors: torch.Tensor,
    b_tensors: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    out_tensors: torch.Tensor,
    problem_sizes: torch.Tensor,
    expert_offsets: torch.Tensor,
    blockscale_offsets: torch.Tensor,
) -> None
```

MXFP8 grouped matrix multiply for MoE.

##### `cutlass_w4a8_moe_mm`

```python
def cutlass_w4a8_moe_mm(
    out_tensors: torch.Tensor,
    a_tensors: torch.Tensor,
    b_tensors: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    b_group_scales: torch.Tensor,
    b_group_size: int,
    expert_offsets: torch.Tensor,
    problem_sizes: torch.Tensor,
    a_strides: torch.Tensor,
    b_strides: torch.Tensor,
    c_strides: torch.Tensor,
    group_scale_strides: torch.Tensor,
    maybe_schedule: str | None = None,
) -> ...
```

CUTLASS grouped MM for W4A8 (INT4 weight, FP8 activation) quantization scheme.
Uses group-wise quantization with per-channel and per-token scaling in the epilogue.

#### GPTQ Marlin Ops

##### `gptq_marlin_repack`

```python
def gptq_marlin_repack(
    b_q_weight: torch.Tensor,
    perm: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
    is_a_8bit: bool = False,
) -> torch.Tensor
```

Repacks GPTQ weights into Marlin format for efficient GPU execution.

#### AWQ Marlin Ops

##### `awq_marlin_repack`

```python
def awq_marlin_repack(
    b_q_weight: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
    is_a_8bit: bool = False,
) -> torch.Tensor
```

Repacks AWQ weights into Marlin format.

##### `gptq_marlin_moe_repack`

```python
def gptq_marlin_moe_repack(
    b_q_weight: torch.Tensor,
    perm: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
    is_a_8bit: bool = False,
) -> torch.Tensor
```

Repacks GPTQ MoE weights into Marlin format. Loops over experts.

##### `awq_marlin_moe_repack`

```python
def awq_marlin_moe_repack(
    b_q_weight: torch.Tensor,
    perm: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
    is_a_8bit: bool = False,
) -> torch.Tensor
```

Repacks AWQ MoE weights into Marlin format. Loops over experts.

#### Marlin GEMM Ops

##### `marlin_int4_fp8_preprocess`

```python
def marlin_int4_fp8_preprocess(
    qweight: torch.Tensor,
    qzeros_or_none: torch.Tensor | None = None,
    inplace: bool = False,
) -> ...
```

Preprocesses INT4 FP8 quantized weights for Marlin GEMM.

##### `marlin_gemm`

```python
def marlin_gemm(
    a: torch.Tensor,
    c: torch.Tensor | None,
    b_q_weight: torch.Tensor,
    b_bias: torch.Tensor | None,
    b_scales: torch.Tensor,
    a_scales: torch.Tensor | None,
    global_scale: torch.Tensor | None,
    b_zeros: torch.Tensor | None,
    g_idx: torch.Tensor | None,
    perm: torch.Tensor | None,
    workspace: torch.Tensor,
    b_q_type: ScalarType,
    size_m: int,
    size_n: int,
    size_k: int,
    is_k_full: bool = True,
    use_atomic_add: bool = False,
    use_fp32_reduce: bool = False,
    is_zp_float: bool = False,
) -> torch.Tensor
```

General Marlin quantized GEMM supporting multiple quantization types.

Parameters:
- `a`: Input activations
- `c`: Optional output buffer
- `b_q_weight`: Quantized weight matrix
- `b_bias`: Optional bias
- `b_scales`: Weight scales
- `a_scales`: Optional activation scales (for W4A8)
- `global_scale`: Optional global scale (for NVFP4)
- `b_zeros`: Optional zero points
- `g_idx`: Optional group indices
- `perm`: Optional permutation
- `workspace`: Scratch workspace tensor
- `b_q_type`: `ScalarType` identifying the quantization scheme
- `size_m`, `size_n`, `size_k`: Matrix dimensions
- `is_k_full`: Whether K dimension is full
- `use_atomic_add`: Whether to use atomic adds for reduction
- `use_fp32_reduce`: Whether to use FP32 for reduction
- `is_zp_float`: Whether zero points are float type

#### Machete Ops

##### `machete_supported_schedules`

```python
def machete_supported_schedules(
    a_type: torch.dtype,
    b_type: ScalarType,
    group_scales_type: torch.dtype | None,
    group_zeros_type: torch.dtype | None = None,
    channel_scales_type: torch.dtype | None = None,
    token_scales_type: torch.dtype | None = None,
    out_type: torch.dtype | None = None,
) -> list[str]
```

Returns list of supported kernel schedules for the given types.

##### `machete_mm`

```python
def machete_mm(
    a: torch.Tensor,
    b_q: torch.Tensor,
    b_type: ScalarType,
    out_type: torch.dtype | None = None,
    b_group_scales: torch.Tensor | None = None,
    b_group_zeros: torch.Tensor | None = None,
    b_group_size: int | None = None,
    b_channel_scales: torch.Tensor | None = None,
    a_token_scales: torch.Tensor | None = None,
    schedule: str | None = None,
) -> torch.Tensor
```

Machete mixed-precision GEMM.

##### `machete_prepack_B`

```python
def machete_prepack_B(
    b_q_weight: torch.Tensor,
    a_type: torch.dtype,
    b_type: ScalarType,
    group_scales_type: torch.dtype | None,
) -> torch.Tensor
```

Prepacks weight matrix B for Machete GEMM.

#### CUTLASS W4A8 Ops

##### `cutlass_w4a8_mm`

```python
def cutlass_w4a8_mm(
    a: torch.Tensor,
    b_q: torch.Tensor,
    b_group_scales: torch.Tensor,
    b_group_size: int,
    b_channel_scales: torch.Tensor,
    a_token_scales: torch.Tensor,
    out_type: torch.dtype | None = None,
    maybe_schedule: str | None = None,
) -> torch.Tensor
```

CUTLASS W4A8 (INT4 weight, FP8 activation) matrix multiply.

##### `cutlass_pack_scale_fp8`

```python
def cutlass_pack_scale_fp8(scales: torch.Tensor) -> torch.Tensor
```

Packs FP8 scales for CUTLASS W4A8 format.

##### `cutlass_encode_and_reorder_int4b`

```python
def cutlass_encode_and_reorder_int4b(b: torch.Tensor) -> torch.Tensor
```

Encodes and reorders INT4 weight tensor for CUTLASS W4A8.

##### `cutlass_encode_and_reorder_int4b_grouped`

```python
def cutlass_encode_and_reorder_int4b_grouped(b_tensors: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]
```

Grouped version of INT4 encoding and reordering for MoE experts.

##### `permute_cols`

```python
def permute_cols(a: torch.Tensor, perm: torch.Tensor) -> torch.Tensor
```

Permutes columns of tensor `a` according to permutation `perm`.

#### FP8 Quantization Ops

##### `scaled_fp8_quant`

```python
def scaled_fp8_quant(
    input: torch.Tensor,
    scale: torch.Tensor | None = None,
    num_token_padding: int | None = None,
    scale_ub: torch.Tensor | None = None,
    use_per_token_if_dynamic: bool = False,
    output: torch.Tensor | None = None,
    group_shape: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]
```

Quantizes input tensor to FP8. Supports both static and dynamic quantization.

- **Static quantization**: Provide `scale` tensor. Supports per-tensor, per-channel,
  per-token, and group scaling via `group_shape`.
- **Dynamic quantization**: `scale=None`. If `use_per_token_if_dynamic=True`, uses
  per-token dynamic quantization; otherwise per-tensor.

Parameters:
- `input`: 2D tensor `[M, N]`
- `scale`: Optional scaling factor. Shapes: 0D/1-element for per-tensor, 1D with
  explicit `group_shape`, or 2D for group scaling
- `num_token_padding`: Optional padding for first dimension
- `scale_ub`: Optional upper bound for dynamic per-token scaling
- `use_per_token_if_dynamic`: Use per-token quantization in dynamic mode
- `output`: Optional pre-allocated output tensor
- `group_shape`: Optional `(group_m, group_n)` for static group quantization.
  Use -1 for full extent (e.g., `(-1, -1)` for per-tensor)

Returns `(output_fp8, scale)`.

#### INT8 Quantization Ops

##### `scaled_int8_quant`

```python
def scaled_int8_quant(
    input: torch.Tensor,
    scale: torch.Tensor | None = None,
    azp: torch.Tensor | None = None,
    symmetric: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]
```

Quantizes input to INT8.

- **Static**: Provide `scale` (and `azp` for asymmetric). `symmetric=True` means
  no zero point.
- **Dynamic**: `scale=None`. Per-token dynamic quantization. If `symmetric=False`,
  computes asymmetric zero points.

Returns `(output_int8, scale, azp)`.

#### GGUF Ops

##### `ggml_dequantize`

```python
def ggml_dequantize(
    W: torch.Tensor, quant_type: int, m: int, n: int, dtype: torch.dtype | None
) -> torch.Tensor
```

Dequantizes GGML quantized weights.

##### `ggml_mul_mat_vec_a8`

```python
def ggml_mul_mat_vec_a8(
    W: torch.Tensor, X: torch.Tensor, quant_type: int, row: int,
) -> torch.Tensor
```

GGML quantized matrix-vector multiply.

##### `ggml_mul_mat_a8`

```python
def ggml_mul_mat_a8(
    W: torch.Tensor, X: torch.Tensor, quant_type: int, row: int,
) -> torch.Tensor
```

GGML quantized matrix multiply.

##### `ggml_moe_a8`

```python
def ggml_moe_a8(
    X: torch.Tensor, W: torch.Tensor,
    sorted_token_ids: torch.Tensor, expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    quant_type: int, row: int, top_k: int, tokens: int,
) -> torch.Tensor
```

GGML quantized MoE matrix multiply.

##### `ggml_moe_a8_vec`

```python
def ggml_moe_a8_vec(
    X: torch.Tensor, W: torch.Tensor, topk_ids: torch.Tensor,
    top_k: int, quant_type: int, row: torch.SymInt, tokens: torch.SymInt,
) -> torch.Tensor
```

GGML quantized MoE vector multiply.

##### `ggml_moe_get_block_size`

```python
def ggml_moe_get_block_size(quant_type: int) -> int
```

Returns the block size for a given GGML quantization type.

#### FP4 Expert Quantization Ops

##### `scaled_fp4_quant`

```python
def scaled_fp4_quant(
    input: torch.Tensor,
    input_global_scale: torch.Tensor,
    is_sf_swizzled_layout: bool = True,
    backend: str = "none",
) -> tuple[torch.Tensor, torch.Tensor]
```

Quantizes input to NVFP4 with 16-element blocks. Returns packed FP4 values and
FP8-E4M3 scaling factors in swizzled layout.

Parameters:
- `input`: 1D or 2D tensor with last dim multiple of 16, dtype float16/bfloat16
- `input_global_scale`: Global scaling factor for the tensor
- `is_sf_swizzled_layout`: Whether to use 128x4 swizzled layout for scales
- `backend`: Backend hint. If `"trtllm"` and `m <= 32`, uses 8x4 layout

##### `scaled_fp4_experts_quant`

```python
def scaled_fp4_experts_quant(
    input_tensor: torch.Tensor,
    input_global_scale: torch.Tensor,
    expert_offsets: torch.Tensor,
    blockscale_offsets: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]
```

NVFP4 quantization for packed MoE expert inputs. Returns `(output, output_scales)`.

##### `silu_and_mul_scaled_fp4_experts_quant`

```python
def silu_and_mul_scaled_fp4_experts_quant(
    input_tensor: torch.Tensor,
    input_global_scale: torch.Tensor,
    expert_offsets: torch.Tensor,
    blockscale_offsets: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]
```

Fused SiLU+Mul+NVFP4 quantization for MoE intermediate activations. Input has
`[gate || up]` layout.

#### MXFP4 Expert Quantization Ops

##### `mxfp4_experts_quant`

```python
def mxfp4_experts_quant(
    input_tensor: torch.Tensor,
    expert_offsets: torch.Tensor,
    blockscale_offsets: torch.Tensor,
    n_experts: int,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]
```

MXFP4 quantization with 32-element blocks and E8M0 scale factors. No global scale.

##### `silu_and_mul_mxfp4_experts_quant`

```python
def silu_and_mul_mxfp4_experts_quant(
    input_tensor: torch.Tensor,
    expert_offsets: torch.Tensor,
    blockscale_offsets: torch.Tensor,
    n_experts: int,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]
```

Fused SiLU+Mul+MXFP4 quantization for MoE intermediate activations.

#### AllSpark Ops

##### `allspark_repack_weight`

```python
def allspark_repack_weight(
    qweight: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None = None,
    has_zp: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

Rearranges quantized weight, scale, and optional zero_point to N32K16 format for
Ampere W8A16 Fused GEMM.

Returns `(qweight_reorder, scale_reorder, zero_point_reorder)`.

##### `allspark_w8a16_gemm`

```python
def allspark_w8a16_gemm(
    a: torch.Tensor,
    b_qweight: torch.Tensor,
    b_scales: torch.Tensor,
    b_qzeros: torch.Tensor | None,
    n: int,
    group_size: int,
    sm_count: int,
    sm_version: int,
    CUBLAS_M_THRESHOLD: int,
    has_zp: bool,
    n32k16_reorder: bool,
) -> torch.Tensor
```

AllSpark W8A16 quantized GEMM.

### Mamba Selective Scan Ops

#### `selective_scan_fwd`

```python
def selective_scan_fwd(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D_: torch.Tensor | None,
    z_: torch.Tensor | None,
    delta_bias_: torch.Tensor | None,
    delta_softplus: bool,
    query_start_loc: torch.Tensor | None,
    cache_indices: torch.Tensor | None,
    has_initial_state: torch.Tensor | None,
    ssm_states: torch.Tensor,
    null_block_id: int,
    block_size: int = 1024,
    block_idx_first_scheduled_token: torch.Tensor | None = None,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
    cu_chunk_seqlen: torch.Tensor | None = None,
    last_chunk_indices: torch.Tensor | None = None,
) -> ...
```

Forward pass of selective scan for Mamba/SSM models. Supports variable-length
sequences, caching, chunked computation, and initial states.

### ROCm Skinny GEMM Ops

#### `LLMM1`

```python
def LLMM1(a: torch.Tensor, b: torch.Tensor, rows_per_block: int) -> torch.Tensor
```

ROCm skinny GEMM kernel LLMM1 via `torch.ops._rocm_C.LLMM1`.

#### `wvSplitK`

```python
def wvSplitK(
    a: torch.Tensor, b: torch.Tensor, cu_count: int, bias: torch.Tensor = None,
) -> torch.Tensor
```

ROCm SplitK GEMM via `torch.ops._rocm_C.wvSplitK`.

#### `wvSplitKrc`

```python
def wvSplitKrc(
    a: torch.Tensor, b: torch.Tensor, cu_count: int, bias: torch.Tensor = None,
) -> torch.Tensor
```

ROCm SplitK row-column GEMM via `torch.ops._rocm_C.wvSplitKrc`.

#### `wvSplitKQ`

```python
def wvSplitKQ(
    a: torch.Tensor,
    b: torch.Tensor,
    out_dtype: torch.dtype,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    cu_count: int,
    bias: torch.Tensor = None,
) -> torch.Tensor
```

ROCm quantized SplitK GEMM. Pre-allocates output and dispatches via
`torch.ops._rocm_C.wvSplitKQ`.

### MoE Dispatch Ops

#### `moe_sum`

```python
def moe_sum(input: torch.Tensor, output: torch.Tensor) -> None
```

Sums MoE expert outputs via `torch.ops._moe_C.moe_sum`.

#### `moe_align_block_size`

```python
def moe_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    experts_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    expert_map: torch.Tensor | None = None,
) -> None
```

Aligns token counts to block sizes for MoE kernels via `torch.ops._moe_C.moe_align_block_size`.

#### `batched_moe_align_block_size`

```python
def batched_moe_align_block_size(
    max_tokens_per_batch: int,
    block_size: int,
    expert_num_tokens: torch.Tensor,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> None
```

Batched version of block size alignment for batched MoE formats.

#### `moe_lora_align_block_size`

```python
def moe_lora_align_block_size(
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    num_experts: int,
    block_size: int,
    max_loras: int,
    max_num_tokens_padded: int,
    max_num_m_blocks: int,
    sorted_token_ids: torch.Tensor,
    experts_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    adapter_enabled: torch.Tensor,
    lora_ids: torch.Tensor,
    expert_map: torch.Tensor | None = None,
) -> None
```

Block size alignment for MoE with LoRA support via `torch.ops._moe_C.moe_lora_align_block_size`.

#### `moe_wna16_gemm`

```python
def moe_wna16_gemm(
    input: torch.Tensor,
    output: torch.Tensor,
    b_qweight: torch.Tensor,
    b_scales: torch.Tensor,
    b_qzeros: torch.Tensor | None,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor,
    experts_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    top_k: int,
    BLOCK_SIZE_M: int,
    BLOCK_SIZE_N: int,
    BLOCK_SIZE_K: int,
    bit: int,
) -> torch.Tensor
```

WNA16 (Weight-only quantized, Activation 16-bit) MoE GEMM. Only available on CUDA.

#### `dsv3_router_gemm`

```python
def dsv3_router_gemm(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor
```

Optimized router GEMM for DeepSeek V3 style models via `torch.ops._moe_C.dsv3_router_gemm`.

#### `moe_wna16_marlin_gemm`

```python
def moe_wna16_marlin_gemm(
    input: torch.Tensor,
    output: torch.Tensor | None,
    b_qweight: torch.Tensor,
    b_bias: torch.Tensor | None,
    b_scales: torch.Tensor,
    a_scales: torch.Tensor | None,
    global_scale: torch.Tensor | None,
    b_qzeros: torch.Tensor | None,
    g_idx: torch.Tensor | None,
    perm: torch.Tensor | None,
    workspace: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_past_padded: torch.Tensor,
    topk_weights: torch.Tensor,
    moe_block_size: int,
    top_k: int,
    mul_topk_weights: bool,
    b_q_type: ScalarType,
    size_m: int,
    size_n: int,
    size_k: int,
    is_k_full: bool,
    use_atomic_add: bool,
    use_fp32_reduce: bool,
    is_zp_float: bool,
    thread_k: int = -1,
    thread_n: int = -1,
    blocks_per_sm: int = -1,
) -> torch.Tensor
```

Marlin-based WNA16 MoE GEMM with per-expert quantized weights.

### Top-K Routing Ops

#### `topk_softmax`

```python
def topk_softmax(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    e_score_correction_bias: torch.Tensor | None = None,
) -> None
```

Fused softmax top-K routing via `torch.ops._moe_C.topk_softmax`. Computes
top-K experts with optional renormalization and e-score correction bias.

#### `topk_sigmoid`

```python
def topk_sigmoid(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    e_score_correction_bias: torch.Tensor | None = None,
) -> None
```

Fused sigmoid top-K routing via `torch.ops._moe_C.topk_sigmoid`.

#### `topk_hash_softplus_sqrt`

```python
def topk_hash_softplus_sqrt(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
    input_tokens: torch.Tensor | None = None,
    hash_indices_table: torch.Tensor | None = None,
) -> None
```

Hash-based top-K routing with softplus+sqrt activation, used for DeepSeek V4
style models.

#### `grouped_topk`

```python
def grouped_topk(
    scores: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    bias: torch.Tensor,
    scoring_func: int = 0,
) -> ...
```

Grouped top-K routing for DeepSeek V3 style models. Selects top-K groups first,
then top-K experts within groups.

Parameters:
- `scoring_func`: 0=none (no activation), 1=sigmoid

### Cache Ops

#### `reshape_and_cache`

```python
def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None
```

Reshapes and caches KV pairs into paged KV cache.

#### `reshape_and_cache_flash`

```python
def reshape_and_cache_flash(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None
```

Flash-attention-compatible KV cache reshape and store.

#### `concat_and_cache_mla`

```python
def concat_and_cache_mla(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    scale: torch.Tensor,
) -> None
```

Concatenates and caches MLA (Multi-head Latent Attention) KV pairs.

#### `concat_and_cache_mla_rope_fused`

```python
def concat_and_cache_mla_rope_fused(
    positions: torch.Tensor,
    q_pe: torch.Tensor,
    k_pe: torch.Tensor,
    kv_c: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    slot_mapping: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_cache_dtype: str,
    kv_cache_scale: torch.Tensor,
) -> None
```

Fused MLA RoPE application + KV cache store.

#### `swap_blocks`

```python
def swap_blocks(
    src: torch.Tensor,
    dst: torch.Tensor,
    block_size_in_bytes: int,
    block_mapping: torch.Tensor,
) -> None
```

Copies specific blocks between tensors. `block_mapping` has shape
`(num_blocks_to_copy, 2)` mapping source block indices to destination indices.

#### `swap_blocks_batch`

```python
def swap_blocks_batch(
    src_ptrs: torch.Tensor,
    dst_ptrs: torch.Tensor,
    sizes: torch.Tensor,
) -> None
```

Batch block copy using raw pointers. Uses `cuMemcpyBatchAsync` on CUDA 12.8+.

#### `convert_fp8`

```python
def convert_fp8(
    output: torch.Tensor, input: torch.Tensor, scale: float = 1.0, kv_dtype: str = "fp8"
) -> None
```

Converts tensor to FP8 format for KV cache storage.

#### `gather_and_maybe_dequant_cache`

```python
def gather_and_maybe_dequant_cache(
    src_cache: torch.Tensor,
    dst: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    token_to_seq: torch.Tensor,
    num_tokens: int,
    kv_cache_dtype: str,
    scale: torch.Tensor,
    seq_starts: torch.Tensor | None = None,
) -> None
```

Gathers KV cache blocks and optionally dequantizes from FP8.

#### `cp_gather_cache`

```python
def cp_gather_cache(
    src_cache: torch.Tensor,
    dst: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    batch_size: int,
    seq_starts: torch.Tensor | None = None,
) -> None
```

Gathers KV cache blocks for context parallelism.

#### `cp_gather_and_upconvert_fp8_kv_cache`

```python
def cp_gather_and_upconvert_fp8_kv_cache(
    src_cache: torch.Tensor,
    dst: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    workspace_starts: torch.Tensor,
    batch_size: int,
) -> None
```

Gathers and upconverts FP8 KV cache to BF16 for context parallelism.

#### `concat_mla_q`

```python
def concat_mla_q(
    ql_nope: torch.Tensor,
    q_pe: torch.Tensor,
    q_out: torch.Tensor,
) -> None
```

Concatenates query nope and RoPE components for MLA attention.

#### `indexer_k_quant_and_cache`

```python
def indexer_k_quant_and_cache(
    k: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    quant_block_size: int,
    kv_cache_dtype: str,
) -> None
```

Quantizes and caches K tensor with index-based lookup.

#### `cp_gather_indexer_k_quant_cache`

```python
def cp_gather_indexer_k_quant_cache(
    kv_cache: torch.Tensor,
    dst_k: torch.Tensor,
    dst_scale: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
) -> None
```

Gathers and dequantizes K cache for context parallelism.

### Custom All-Reduce Ops

#### `init_custom_ar`

```python
def init_custom_ar(
    ipc_tensors: list[torch.Tensor],
    rank_data: torch.Tensor,
    rank: int,
    fully_connected: bool,
) -> int
```

Initializes custom all-reduce. Returns a handle (int).

#### `all_reduce`

```python
def all_reduce(
    fa: int,
    inp: torch.Tensor,
    out: torch.Tensor,
    reg_buffer: int,
    reg_buffer_sz_bytes: int,
) -> None
```

Performs custom all-reduce using the initialized handle.

#### `dispose`

```python
def dispose(fa: int) -> None
```

Disposes of a custom all-reduce handle.

### CUDA Utils Ops

#### `get_device_attribute`

```python
def get_device_attribute(attribute: int, device: int) -> int
```

Gets a CUDA device attribute value.

#### `get_max_shared_memory_per_block_device_attribute`

```python
def get_max_shared_memory_per_block_device_attribute(device: int) -> int
```

Returns the maximum shared memory per block for a CUDA device.

---

## 3. Fused MoE Infrastructure

vLLM's fused MoE (Mixture of Experts) system provides high-performance implementations
for routing tokens to experts and computing expert outputs. The architecture follows
a modular design: Router -> Quantize/Dispatch -> Permute/Experts/Unpermute -> Combine.

### MoE Activation

**File**: `vllm/model_executor/layers/fused_moe/activation.py`

#### `MoEActivation` Enum

```python
class MoEActivation(Enum):
    SILU = "silu"
    GELU = "gelu"
    GELU_TANH = "gelu_tanh"
    RELU2 = "relu2"
    SWIGLUOAI = "swigluoai"
    SWIGLUSTEP = "swiglustep"
    SILU_NO_MUL = "silu_no_mul"
    GELU_NO_MUL = "gelu_no_mul"
    GELU_TANH_NO_MUL = "gelu_tanh_no_mul"
    RELU2_NO_MUL = "relu2_no_mul"
```

Enum representing activation functions used in MoE layers. Gated activations
(gate * activation(up)) expect input of shape `[..., 2*d]`. Non-gated activations
expect `[..., d]`.

**Properties:**

```python
@property
def is_gated(self) -> bool
```
Returns `True` if the activation expects gate*activation(up) pattern.

```python
@property
def custom_op_name(self) -> str
```
Maps to the CustomOp name in `vllm/model_executor/layers/activation.py`.

**Methods:**

```python
def without_mul(self) -> MoEActivation
```
Returns the non-gated variant of this activation.

```python
@classmethod
def from_str(cls, s: str) -> MoEActivation
```
Parses from string with backward compatibility aliases.

#### `activation_without_mul`

```python
def activation_without_mul(activation: str) -> str
```

Returns the non-gated variant name of an activation function string.

#### `apply_moe_activation`

```python
def apply_moe_activation(
    activation: MoEActivation,
    output: torch.Tensor,
    input: torch.Tensor,
) -> torch.Tensor
```

Applies the MoE activation function to input, writing to output.
Dispatches to the appropriate C++/Triton kernel based on the activation type.

### MoE Config

**File**: `vllm/model_executor/layers/fused_moe/config.py`

#### `RoutingMethodType` IntEnum

```python
class RoutingMethodType(IntEnum):
    Default = (0,)            # Softmax -> TopK
    Renormalize = (1,)        # TopK -> Softmax/Sigmoid
    DeepSeekV3 = (2,)         # Sigmoid -> Bias -> Top2 group -> Top4 group -> Top8
    Llama4 = (3,)             # Top1 -> Sigmoid
    RenormalizeNaive = (4,)   # Softmax/Sigmoid -> TopK -> Renormalize
    TopK = (5,)               # TopK (no softmax)
    SigmoidRenorm = (6,)      # Sigmoid -> TopK -> Renormalize
    MiniMax2 = (7,)           # Sigmoid + Bias -> TopK -> ScaledSumNormalize
    Unspecified = (8,)
    DeepseekV4 = (100,)       # sqrtsoftplus + Bias + Normalize
    Custom = (101,)
    Simulated = (102,)
```

Identifies the routing method for FlashInfer MoE integration.

#### `get_routing_method_type`

```python
def get_routing_method_type(
    scoring_func: str,
    top_k: int,
    renormalize: bool,
    num_expert_group: int | None,
    has_e_score_bias: bool,
) -> RoutingMethodType
```

Determines the routing method type from routing configuration parameters.

#### `FusedMoEQuantDesc` Dataclass

```python
@dataclass
class FusedMoEQuantDesc:
    dtype: torch.dtype | str | None = None
    shape: GroupShape | None = None
    scale: Union[torch.Tensor, "PrecisionConfig", None] = None
    alpha_or_gscale: torch.Tensor | None = None
    zp: torch.Tensor | None = None
    bias: torch.Tensor | None = None
```

Quantization descriptor for a single parameter (activation or weight) in a fused
MoE operation.

#### `FusedMoEQuantConfig` Dataclass

```python
@dataclass
class FusedMoEQuantConfig:
    _a1: FusedMoEQuantDesc
    _a2: FusedMoEQuantDesc
    _w1: FusedMoEQuantDesc
    _w2: FusedMoEQuantDesc
    is_nvfp4_scale_swizzled: bool = True
    gemm1_alpha: float | None = None
    gemm1_beta: float | None = None
    gemm1_clamp_limit: float | None = None
    mx_alignment: int = 0
```

Complete quantization configuration for a fused MoE operation. Contains four
`FusedMoEQuantDesc` descriptors for: gate-up activation, down-projection activation,
gate-up weight, and down-projection weight.

**Properties** (all read-only):

| Property | Type | Description |
|---|---|---|
| `quant_dtype` | `torch.dtype \| str \| None` | Activation quantization dtype |
| `weight_quant_dtype` | `torch.dtype \| str \| None` | Weight quantization dtype |
| `is_quantized` | `bool` | Whether any quantization is active |
| `is_per_act_token` | `bool` | Per-token activation quantization |
| `per_act_token_quant` | `bool` | Same as `is_per_act_token` |
| `per_out_ch_quant` | `bool` | Per-channel output quantization |
| `is_per_tensor` | `bool` | Per-tensor quantization |
| `block_shape` | `list[int] \| None` | Block shape for block quantization |
| `is_block_quantized` | `bool` | Whether block quantization is active |
| `a1_scale`, `a2_scale` | `Tensor \| None` | Activation scales |
| `w1_scale`, `w2_scale` | `Tensor \| None` | Weight scales |
| `w1_zp`, `w2_zp` | `Tensor \| None` | Weight zero points |
| `w1_bias`, `w2_bias` | `Tensor \| None` | Weight biases |
| `g1_alphas`, `g2_alphas` | `Tensor \| None` | Global scales / per-channel scales |
| `a1_gscale`, `a2_gscale` | `Tensor \| None` | Activation global scales |
| `use_fp8_w8a8` | `bool` | FP8 weights + FP8 activations |
| `use_int8_w8a8` | `bool` | INT8 weights + INT8 activations |
| `use_fp8_w8a16` | `bool` | FP8 weights + FP16/BF16 activations |
| `use_int4_w4a16` | `bool` | INT4 weights + FP16/BF16 activations |
| `use_nvfp4_w4a16` | `bool` | NVFP4 weights + FP16/BF16 activations |
| `use_mxfp4_w4a16` | `bool` | MXFP4 weights + FP16/BF16 activations |
| `use_mxfp4_w4a4` | `bool` | MXFP4 weights + MXFP4 activations |
| `use_mxfp4_w4a8` | `bool` | MXFP4 weights + FP8 activations |
| `use_nvfp4_w4a4` | `bool` | NVFP4 weights + NVFP4 activations |
| `ocp_mx_scheme` | `str \| None` | OCP MX quantization scheme |

**Methods:**

```python
def config_name(self, dtype: torch.dtype) -> str | None
```

Returns config filename string for auto-tuning lookup.

```python
def scale_shape(self, max_tokens: int, hidden_dim: int) -> tuple[int, int] | None
```

Returns the proper activation scale shape for this config.

```python
def batched_scale_shape(self, num_experts: int, max_tokens: int, hidden_dim: int
) -> tuple[int, int, int] | None
```

Returns the batched (per-expert) activation scale shape.

```python
@staticmethod
def make(
    quant_dtype: torch.dtype | str | None = None,
    per_act_token_quant: bool = False,
    per_out_ch_quant: bool = False,
    block_shape: list[int] | None = None,
    w1_scale: Union[torch.Tensor, "PrecisionConfig", None] = None,
    w2_scale: Union[torch.Tensor, "PrecisionConfig", None] = None,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    g1_alphas: torch.Tensor | None = None,
    g2_alphas: torch.Tensor | None = None,
    a1_gscale: torch.Tensor | None = None,
    a2_gscale: torch.Tensor | None = None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    w1_zp: torch.Tensor | None = None,
    w2_zp: torch.Tensor | None = None,
    weight_dtype: torch.dtype | str | None = None,
    is_nvfp4_scale_swizzled: bool = True,
    gemm1_alpha: float | None = None,
    gemm1_beta: float | None = None,
    gemm1_clamp_limit: float | None = None,
) -> FusedMoEQuantConfig
```

General builder function. Factory method that constructs the four
`FusedMoEQuantDesc` instances from flat parameters.

#### Quant Config Factory Functions

```python
def fp8_w8a8_moe_quant_config(...) -> FusedMoEQuantConfig
def int8_w8a8_moe_quant_config(...) -> FusedMoEQuantConfig
def gptq_marlin_moe_quant_config(...) -> FusedMoEQuantConfig
def mxfp4_w4a16_moe_quant_config(...) -> FusedMoEQuantConfig
def mxfp4_mxfp8_moe_quant_config(...) -> FusedMoEQuantConfig
def mxfp4_w4a8_moe_quant_config(...) -> FusedMoEQuantConfig
def ocp_mx_moe_quant_config(...) -> FusedMoEQuantConfig
def nvfp4_moe_quant_config(...) -> FusedMoEQuantConfig
def mxfp4_moe_quant_config(...) -> FusedMoEQuantConfig
def nvfp4_w4a16_moe_quant_config(...) -> FusedMoEQuantConfig
def int4_w4a16_moe_quant_config(...) -> FusedMoEQuantConfig
def fp8_w8a16_moe_quant_config(...) -> FusedMoEQuantConfig
def int8_w8a16_moe_quant_config(...) -> FusedMoEQuantConfig
def int4_w4afp8_moe_quant_config(...) -> FusedMoEQuantConfig
def awq_marlin_moe_quant_config(...) -> FusedMoEQuantConfig
def biased_moe_quant_config(...) -> FusedMoEQuantConfig
```

Each factory function creates a `FusedMoEQuantConfig` tailored for a specific
quantization scheme.

```python
FUSED_MOE_UNQUANTIZED_CONFIG: FusedMoEQuantConfig = FusedMoEQuantConfig.make()
```

Constant for an unquantized MoE operation.

#### `FusedMoEParallelConfig` Dataclass

```python
@dataclass
class FusedMoEParallelConfig:
    tp_size: int
    pcp_size: int
    dp_size: int
    ep_size: int
    tp_rank: int
    pcp_rank: int
    dp_rank: int
    ep_rank: int
    sp_size: int
    use_ep: bool
    all2all_backend: str
    enable_eplb: bool
```

MoE parallelization configuration. Determines how experts are distributed
across devices.

**Properties:**

| Property | Type | Description |
|---|---|---|
| `is_sequence_parallel` | `bool` | `sp_size > 1` |
| `use_all2all_kernels` | `bool` | `dp_size > 1 and use_ep` |
| `use_deepep_ht_kernels` | `bool` | DeepEP high-throughput backend |
| `use_deepep_ll_kernels` | `bool` | DeepEP low-latency backend |
| `use_fi_nvl_two_sided_kernels` | `bool` | FlashInfer NVLink two-sided |
| `use_fi_nvl_one_sided_kernels` | `bool` | FlashInfer NVLink one-sided |
| `use_batched_activation_format` | `bool` | Whether batched format is needed |
| `needs_round_robin_routing_tables` | `bool` | Round-robin routing required |
| `use_ag_rs_all2all_kernels` | `bool` | AllGather/ReduceScatter backend |
| `use_mori_kernels` | `bool` | Mori backend |
| `use_nixl_ep_kernels` | `bool` | NIXL EP backend |

**Methods:**

```python
@staticmethod
def flatten_tp_across_dp_and_pcp(
    tp_size: int, dp_size: int, dp_rank: int, pcp_size: int, pcp_rank: int,
) -> tuple[int, int]
```

Flattens TP across DP and PCP dimensions. Returns `(flatten_tp_size, flatten_tp_rank)`.

```python
@staticmethod
def make(
    tp_size_: int,
    pcp_size_: int,
    dp_size_: int,
    sp_size_: int,
    vllm_parallel_config: ParallelConfig,
) -> FusedMoEParallelConfig
```

Determines the MoE parallel configuration from parallelism parameters and the
global vLLM parallel config. Handles TP, DP, EP, and SP combinations.

```python
@classmethod
def make_no_parallel(cls) -> FusedMoEParallelConfig
```

Creates a no-parallelism config for testing.

#### `FusedMoEConfig` Dataclass

```python
@dataclass
class FusedMoEConfig:
    num_experts: int
    experts_per_token: int
    hidden_dim: int
    intermediate_size_per_partition: int
    num_local_experts: int
    num_logical_experts: int
    activation: MoEActivation
    device: torch.device | str
    routing_method: RoutingMethodType
    moe_parallel_config: FusedMoEParallelConfig
    in_dtype: torch.dtype
    router_logits_dtype: torch.dtype | None = None
    hidden_dim_unpadded: int | None = None
    intermediate_size_per_partition_unpadded: int | None = None
    moe_backend: MoEBackend = "auto"
    max_num_tokens: int = SchedulerConfig.DEFAULT_MAX_NUM_BATCHED_TOKENS_FOR_BATCHED_DP
    has_bias: bool = False
    is_act_and_mul: bool = True
    is_lora_enabled: bool = False
    disable_inplace: bool = True
```

Complete configuration for a fused MoE layer. Delegates parallel config properties
to `moe_parallel_config`.

**Properties** (all delegated to `moe_parallel_config`):

`tp_size`, `dp_size`, `pcp_size`, `ep_size`, `sp_size`, `tp_rank`, `dp_rank`,
`pcp_rank`, `ep_rank`, `use_ep`, `use_deepep_ht_kernels`, `use_deepep_ll_kernels`,
`use_mori_kernels`, `use_fi_nvl_two_sided_kernels`, `use_fi_nvl_one_sided_kernels`,
`use_ag_rs_all2all_kernels`, `use_nixl_ep_kernels`, `needs_round_robin_routing_tables`,
`is_sequence_parallel`.

### MoE Router

**File**: `vllm/model_executor/layers/fused_moe/router/fused_moe_router.py`

#### `FusedMoERouter` ABC

```python
class FusedMoERouter(ABC):
```

Abstract base class for MoE routing. Defines the interface for expert selection.

**Abstract Methods:**

```python
@abstractmethod
def set_capture_fn(
    self, capture_fn: Callable[[torch.Tensor], None] | None,
) -> None
```

Sets an optional capture function for router logits (used by EPLB).

```python
@property
@abstractmethod
def routing_method_type(self) -> RoutingMethodType
```

Returns the routing method type for this router.

```python
@abstractmethod
def select_experts(
    self,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    *,
    input_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]
```

Routes hidden states to top-K experts based on router logits.

Returns `(topk_weights, topk_ids)`.

### FusedMoEMethodBase

**File**: `vllm/model_executor/layers/fused_moe/fused_moe_method_base.py`

```python
class FusedMoEMethodBase(QuantizeMethodBase):
    def __init__(self, moe: FusedMoEConfig):
        super().__init__()
        self.moe: FusedMoEConfig = moe
        self.moe_quant_config: FusedMoEQuantConfig | None = None
        self.moe_kernel: mk.FusedMoEKernel | None = None
```

Base class for fused MoE methods. Inherits from `QuantizeMethodBase`.

**Properties:**

| Property | Type | Description |
|---|---|---|
| `supports_internal_mk` | `bool` | Whether MK interface migration is complete |
| `mk_owns_shared_expert` | `bool` | Whether MK owns shared expert computation |
| `topk_indices_dtype` | `torch.dtype \| None` | Expected dtype for top-K indices |
| `skip_forward_padding` | `bool` | Whether to skip padding in forward |
| `supports_eplb` | `bool` | Whether EPLB is supported |
| `method_name` | `str` | Class name of the method |
| `is_monolithic` | `bool` | Whether the method is monolithic (non-modular) |

**Abstract Methods:**

```python
@abstractmethod
def create_weights(
    self,
    layer: torch.nn.Module,
    num_experts: int,
    hidden_size: int,
    intermediate_size_per_partition: int,
    params_dtype: torch.dtype,
    **extra_weight_attrs,
) -> None
```

Creates weight parameters for the MoE layer.

```python
@abstractmethod
def get_fused_moe_quant_config(
    self, layer: torch.nn.Module,
) -> FusedMoEQuantConfig | None
```

Returns the quantization config for the MoE layer.

```python
@abstractmethod
def apply(
    self,
    layer: FusedMoE,
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
) -> torch.Tensor
```

Applies the MoE computation using pre-computed routing.

```python
@abstractmethod
def apply_monolithic(
    self,
    layer: FusedMoE,
    x: torch.Tensor,
    router_logits: torch.Tensor,
    input_ids: torch.Tensor | None = None,
) -> torch.Tensor
```

Monolithic forward combining routing and expert computation.

### FusedMoEModularMethod

**File**: `vllm/model_executor/layers/fused_moe/fused_moe_modular_method.py`

```python
@CustomOp.register("modular_fused_moe")
class FusedMoEModularMethod(FusedMoEMethodBase, CustomOp):
```

Modular MoE method combining a `FusedMoEPrepareAndFinalizeModular` with a
`FusedMoEExpertsModular` through a `FusedMoEKernel`.

**Constructor:**

```python
def __init__(
    self, old_quant_method: FusedMoEMethodBase, moe_kernel: FusedMoEKernel
):
```

Wraps an existing quant method with a modular kernel.

**Static Methods:**

```python
@staticmethod
def make(
    moe_layer: torch.nn.Module,
    old_quant_method: FusedMoEMethodBase,
    prepare_finalize: FusedMoEPrepareAndFinalizeModular,
    shared_experts: SharedExperts | None,
    inplace: bool = False,
) -> FusedMoEModularMethod
```

Factory method that creates a `FusedMoEModularMethod` by combining the
prepare/finalize module with the GEMM implementation selected by the old method.

**Methods:**

```python
def apply(
    self,
    layer: FusedMoE,
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
) -> torch.Tensor
```

Delegates to `self.moe_kernel.apply()`.

```python
def apply_monolithic(
    self,
    layer: FusedMoE,
    x: torch.Tensor,
    router_logits: torch.Tensor,
    input_ids: torch.Tensor | None = None,
) -> torch.Tensor
```

Combines routing and expert computation in a single call.

### Modular Kernel Framework

**File**: `vllm/model_executor/layers/fused_moe/modular_kernel.py`

#### `FusedMoEActivationFormat` Enum

```python
class FusedMoEActivationFormat(Enum):
    Standard = ("standard",)           # (num_tokens, hidden_dim)
    BatchedExperts = ("batched_experts",)  # (num_experts, max_tokens_per_expert, hidden_dim)
```

#### `ExpertTokensMetadata` Dataclass

```python
@dataclass
class ExpertTokensMetadata:
    expert_num_tokens: torch.Tensor
    expert_num_tokens_cpu: torch.Tensor | None

    @staticmethod
    def make_from_list(
        expert_num_tokens_list: list[int], device: str,
    ) -> ExpertTokensMetadata
```

Metadata about per-expert token counts, with both GPU and CPU copies.

#### `TopKWeightAndReduce` ABC

```python
class TopKWeightAndReduce(ABC):
    @abstractmethod
    def apply(
        self,
        output: torch.Tensor | None,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor
```

Abstract base class for applying top-K weights and reducing expert outputs.

#### `FusedMoEPrepareAndFinalizeModular` ABC

Abstract base class for MoE preparation (quantization, distribution) and
finalization (weight application, reduction).

#### `FusedMoEExpertsModular` ABC

Abstract base class for the core fused MoE computation (matmul + activation +
optional quant + matmul).

#### `FusedMoEKernel`

Combines a `FusedMoEPrepareAndFinalizeModular` with a `FusedMoEExpertsModular`
to provide the standard fused MoE kernel interface.

### FusedMoE Layer

**File**: `vllm/model_executor/layers/fused_moe/layer.py`

#### `FusedMoeWeightScaleSupported` Enum

```python
class FusedMoeWeightScaleSupported(Enum):
    TENSOR = "tensor"
    CHANNEL = "channel"
    GROUP = "group"
    BLOCK = "block"
```

#### `determine_expert_map`

```python
def determine_expert_map(
    ep_size: int,
    ep_rank: int,
    global_num_experts: int,
    expert_placement_strategy: ExpertPlacementStrategy = "linear",
    num_fused_shared_experts: int = 0,
    return_expert_mask: bool = False,
) -> tuple[int, torch.Tensor | None, torch.Tensor | None]
```

Computes expert-to-rank mapping for expert parallelism. Supports `linear` and
`round_robin` placement strategies.

Returns `(local_num_experts, expert_map, expert_mask)`.

#### `FusedMoE` Class

The main MoE layer class that combines routing, expert weight management, and
dispatch to the appropriate fused MoE method. Integrates with LoRA, EPLB, and
shared experts.

### Fused MoE Module Exports

**File**: `vllm/model_executor/layers/fused_moe/__init__.py`

**Config Functions:**

```python
@contextmanager
def override_config(config) -> Generator[None, None, None]
```

Context manager to temporarily override the global MoE config.

```python
def get_config() -> dict[str, Any] | None
```

Returns the current global MoE config.

**Exported Classes (always available):**

`FusedMoE`, `FusedMoERouter`, `FusedMoEConfig`, `FusedMoEMethodBase`,
`MoEActivation`, `UnquantizedFusedMoEMethod`, `FusedMoeWeightScaleSupported`,
`FusedMoEExpertsModular`, `FusedMoEActivationFormat`,
`FusedMoEPrepareAndFinalizeModular`, `GateLinear`, `RoutingMethodType`.

**Exported Classes (Triton only):**

`AiterExperts`, `fused_topk`, `fused_experts`, `GroupedTopk`,
`TritonExperts`, `TritonWNA16Experts`, `BatchedTritonExperts`,
`CutlassExpertsFp8`, `CutlassBatchedExpertsFp8`, `CutlassExpertsW4A8Fp8`,
`DeepGemmExperts`, `BatchedDeepGemmExperts`, `TritonOrDeepGemmExperts`,
`XPUExperts`, `XPUExpertsFp8`, `XPUExpertsMXFp4`.

---

## 4. Flash Attention Interface

### Flash Attn Module Init

**File**: `vllm/vllm_flash_attn/__init__.py`

Handles Flash Attention module initialization with symlink support for development.
When `VLLM_FLASH_ATTN_SRC_DIR` is set, the `cute/` directory becomes a symlink
to the real source tree, and a virtual `flash_attn` package is registered in
`sys.modules` to resolve `flash_attn.cute.*` imports.

**Exported Symbols:**

```python
FA2_AVAILABLE: bool
FA3_AVAILABLE: bool
fa_version_unsupported_reason: Callable
flash_attn_varlen_func: Callable
get_scheduler_metadata: Callable
is_fa_version_supported: Callable
```

Raises `ImportError` if neither FA2 nor FA3 is available.

### Flash Attn Interface

**File**: `vllm/vllm_flash_attn/flash_attn_interface.py`

Provides the Python interface to Flash Attention C++/CUDA extensions (_vllm_fa2_C
and _vllm_fa3_C).

#### Flash Attention Version Support

```python
DEFAULT_FA_VERSION: int = 3  # Default to FA3 if available
```

```python
def is_fa_version_supported(fa_version: int) -> bool
```

Returns whether the specified FA version is supported on the current platform.

```python
def fa_version_unsupported_reason(fa_version: int) -> str
```

Returns the reason a FA version is unsupported.

#### `get_scheduler_metadata`

```python
def get_scheduler_metadata(
    batch_size: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    num_heads_q: int,
    num_heads_k: int,
    headdim: int,
    q_dtype: torch.dtype,
    seqused_k: torch.Tensor | None = None,
    causal: bool = False,
    num_splits: int = 0,
    fa_version: int = DEFAULT_FA_VERSION,
) -> ...
```

Computes FA3 scheduler metadata for optimal tile sizes and split configuration.

#### `flash_attn_varlen_func`

```python
def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    cu_seqlens_k: torch.Tensor | None = None,
    max_seqlen_k: int = 0,
    seqused_k: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] | None = None,
    softcap: float = 0.0,
    alibi_slopes: torch.Tensor | None = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    return_softmax_lse: bool = False,
    scheduler_metadata: Any | None = None,
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    num_splits: int = 0,
    fa_version: int = DEFAULT_FA_VERSION,
    s_aux=None,
    cp_world_size=1,
    cp_rank=0,
    cp_tot_seqused_k=None,
) -> torch.Tensor | tuple[torch.Tensor, ...]
```

Variable-length Flash Attention function supporting FA2, FA3, and FA4.

Parameters:
- `q`: `(total_q, nheads, headdim)` query tensor
- `k`: `(total_k, nheads_k, headdim)` key tensor
- `v`: `(total_k, nheads_k, headdim)` value tensor
- `cu_seqlens_q`: `(batch_size + 1,)` cumulative query sequence lengths
- `max_seqlen_q`: Maximum query sequence length
- `cu_seqlens_k`: `(batch_size + 1,)` cumulative key sequence lengths
- `max_seqlen_k`: Maximum key sequence length
- `seqused_k`: Alternative to `cu_seqlens_k` for PagedAttention
- `block_table`: Block table for PagedAttention (requires `seqused_k`)
- `dropout_p`: Dropout probability (0.0 for inference)
- `softmax_scale`: QK^T scaling (default: `1/sqrt(headdim)`)
- `causal`: Whether to apply causal mask
- `window_size`: `(left, right)` sliding window, `(-1, -1)` for full attention
- `softcap`: Attention logit softcap (> 0 activates softcapping)
- `alibi_slopes`: `(nheads,)` or `(batch_size, nheads)` ALiBi bias slopes
- `deterministic`: Deterministic backward pass
- `return_attn_probs`: Whether to return attention probabilities (testing only)
- `return_softmax_lse`: Whether to return logsumexp values
- `scheduler_metadata`: Pre-computed FA3 scheduler metadata
- `q_descale`, `k_descale`, `v_descale`: Per-head descale tensors (FA3+)
- `num_splits`: Number of splits for FA3 parallel attention
- `fa_version`: 2, 3, or 4 (default: `DEFAULT_FA_VERSION`)
- `s_aux`: Auxiliary output tensor (FA3+)
- `cp_world_size`, `cp_rank`: Context parallelism configuration
- `cp_tot_seqused_k`: Total K sequence lengths for context parallelism

Returns:
- `out`: `(total_q, nheads, headdim)` attention output
- Optionally: `softmax_lse` `(nheads, total_q_seqlen)`

---

## 5. Triton Kernel Utilities

### Triton Import Detection

**File**: `vllm/triton_utils/importing.py`

#### `HAS_TRITON`

```python
HAS_TRITON: bool
```

Global flag indicating whether Triton is available. Set by checking:
1. `importlib.util.find_spec("triton")` succeeds
2. Triton driver is functional (not just a stub install)

#### `TritonPlaceholder`

```python
class TritonPlaceholder:
    def __getattr__(self, name):
        raise RuntimeError("Triton is not installed. ...")
```

Placeholder class used when Triton is not available. Raises `RuntimeError`
on any attribute access.

#### `TritonLanguagePlaceholder`

```python
class TritonLanguagePlaceholder:
    def __getattr__(self, name):
        raise RuntimeError("Triton language is not installed. ...")
```

Placeholder for `triton.language` when Triton is unavailable.

### JIT Monitor

**File**: `vllm/triton_utils/jit_monitor.py`

#### `activate`

```python
def activate() -> None
```

Activates JIT compilation monitoring. After warmup is complete, logs any
Triton autotuning cache misses and JIT compilations as warnings. This helps
detect performance regressions where new kernel shapes trigger JIT compilation
during serving.

### Triton Allocator

**File**: `vllm/triton_utils/allocation.py`

#### `set_triton_allocator`

```python
def set_triton_allocator(device: torch.device) -> None
```

Sets the Triton memory allocator to use PyTorch's memory manager. This ensures
Triton kernel allocations go through vLLM's memory management rather than
Triton's default allocator.

### Triton Module Re-exports

**File**: `vllm/triton_utils/__init__.py`

Re-exports `triton`, `tl` (triton.language), and `tldevice` (triton.language.extra)
with placeholder fallbacks when Triton is not installed. All modules in vLLM
should import Triton via `from vllm.triton_utils import triton, tl` to ensure
graceful degradation.

---

## 6. Triton FP8 Quantization Kernel

### QKV Padded FP8 Quant

**File**: `vllm/kernels/triton/qkv_padded_fp8_quant.py`

Provides Triton kernels for FP8 quantization with head dimension padding.

#### Triton Kernel: `_quantize_pad_fp8_kernel`

```python
@triton.jit
def _quantize_pad_fp8_kernel(
    # Pointers
    Y_ptr, Y_scale_ptr, X_ptr, scale_ptr,
    # Strides
    stride_yt, stride_ys, stride_xt, stride_xs,
    # Dimensions
    N, N_padded,
    # Block size
    BLOCK_N: tl.constexpr,
)
```

Triton JIT kernel that quantizes input to FP8 while padding the head dimension.
Each program instance handles one token's quantization with optional padding
to `N_padded`.

#### `quantize_fp8_pad_head_dim_triton`

```python
def quantize_fp8_pad_head_dim_triton(
    input: torch.Tensor,
    scale: torch.Tensor,
    head_dim_pad_size: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]
```

Quantizes input tensor to FP8 with optional head dimension padding using a
Triton kernel.

Parameters:
- `input`: Input tensor `[num_tokens, head_dim]`
- `scale`: Scale tensor `[num_tokens, 1]` (per-token)
- `head_dim_pad_size`: Number of padding elements to add to head dimension

Returns `(output, scale)` where `output` has shape
`[num_tokens, head_dim + head_dim_pad_size]`.

#### `quantize_fp8_maybe_pad_head_dim`

```python
def quantize_fp8_maybe_pad_head_dim(
    input: torch.Tensor,
    scale: torch.Tensor,
    head_dim_pad_size: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]
```

Dispatches between the Triton kernel and `scaled_fp8_quant` based on whether
padding is needed. Uses `scaled_fp8_quant` when no padding is required for
better performance.

---

## 7. Helion Kernel Integration

### Helion Module Init

**File**: `vllm/kernels/helion/__init__.py`

Exports the Helion kernel integration API:

```python
from .register import (
    ConfiguredHelionKernel,
    HelionKernelWrapper,
    register_kernel,
    infer_fake_impl,
    _REGISTERED_KERNELS,
)
from .config_manager import ConfigManager, ConfigSet
```

### Helion Kernel Registration

**File**: `vllm/kernels/helion/register.py`

#### `PresetConfigSearch`

Custom autotuner that searches through pre-tuned configurations rather than
exhaustively. Selects the best config from a preset list matching the current
input shapes and GPU platform.

#### `ConfiguredHelionKernel`

Wraps a Helion kernel bound to a specific platform and configuration. Provides
a `__call__` method that runs the kernel with the pre-tuned config.

#### `HelionKernelWrapper`

Wraps a registered Helion kernel with vLLM's CustomOp and HOP (Higher-Order
Protocol) support. Handles platform detection, config loading, and kernel
dispatch.

**Constructor:**

```python
def __init__(
    self,
    kernel_name: str,
    kernel_func: Callable,
    fake_impl: Callable,
    platforms: set[str] | None = None,
)
```

**Methods:**

```python
def __call__(self, *args, **kwargs) -> torch.Tensor
```

Dispatches to the configured Helion kernel for the current platform.

#### `register_kernel`

```python
def register_kernel(
    kernel_name: str,
    platforms: set[str] | None = None,
) -> Callable
```

Decorator that registers a Helion kernel with vLLM's kernel registry. The
decorated function should accept `(input_generator, config_picker, baseline_impl)`.

#### `infer_fake_impl`

```python
def infer_fake_impl(
    kernel_func: Callable,
) -> Callable
```

Automatically infers a fake (abstract) implementation from a Helion kernel
function for `torch.compile` tracing.

#### `_REGISTERED_KERNELS`

```python
_REGISTERED_KERNELS: dict[str, HelionKernelWrapper]
```

Global registry of all registered Helion kernels.

### Helion Config Manager

**File**: `vllm/kernels/helion/config_manager.py`

#### `ConfigSet`

```python
class ConfigSet:
    def __init__(self) -> None
    def add(self, gpu_name: str, signature: tuple, config: dict) -> None
    def get(self, gpu_name: str, signature: tuple) -> dict | None
```

In-memory collection of kernel configurations indexed by GPU name and input
signature.

#### `ConfigManager`

```python
class ConfigManager:
    _instance: ConfigManager | None = None  # Singleton

    @classmethod
    def get_instance(cls) -> ConfigManager

    def __init__(self) -> None
    def load_configs(self, kernel_name: str) -> ConfigSet
    def save_configs(self, kernel_name: str, configs: ConfigSet) -> None
```

Singleton manager for Helion kernel configurations. Stores configs in JSON files
per GPU platform in a `.helion_configs/` directory.

### Helion GPU Name Utils

**File**: `vllm/kernels/helion/utils.py`

#### `canonicalize_gpu_name`

```python
def canonicalize_gpu_name(name: str) -> str
```

Canonicalizes GPU names to a standard form. Maps common aliases:
- `NVIDIA H100 80GB HBM3` -> `H100`
- `NVIDIA H200` -> `H200`
- `NVIDIA A100-SXM4-80GB` -> `A100`
- `NVIDIA V100` -> `V100`
- `AMD Instinct MI300X` -> `MI300X`

#### `get_canonical_gpu_name`

```python
def get_canonical_gpu_name() -> str | None
```

Returns the canonical GPU name for the current device. Returns `None` if the
device is not a known GPU.

### Helion SiLU Mul FP8

**File**: `vllm/kernels/helion/ops/silu_mul_fp8.py`

Example Helion kernel implementing fused SiLU+Mul+FP8 quantization.

#### `silu_mul_fp8`

```python
@helion.kernel
def silu_mul_fp8(
    input: torch.Tensor,    # [M, 2K] gate || up
    scale_ub: torch.Tensor, # [1] scale upper bound
) -> tuple[torch.Tensor, torch.Tensor]  # (output, scale)
```

Fused kernel computing `silu(gate) * up` followed by FP8 quantization.

---

## 8. AITER (AMD ROCm) Operations

**File**: `vllm/_aiter_ops.py`

The `rocm_aiter_ops` class centralizes all AMD AITER (AMD Instinct Triton
Extensions for ROCm) operations. Individual operations can be enabled/disabled
through environment variables.

### `rocm_aiter_ops` Class

```python
class rocm_aiter_ops:
```

All methods are class methods. The class checks environment variables
(`VLLM_ROCM_USE_AITER`, `VLLM_ROCM_USE_AITER_LINEAR`, `VLLM_ROCM_USE_AITER_MOE`,
`VLLM_ROCM_USE_AITER_MLA`, etc.) to determine which operations to use.

#### MoE Operations

##### `fused_moe`

```python
@classmethod
def fused_moe(
    cls,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gelu_and_mul: bool = False,
    activation: str = "silu",
    expert_map: torch.Tensor | None = None,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    a1_gscale: torch.Tensor | None = None,
    w1_zp: torch.Tensor | None = None,
    w2_zp: torch.Tensor | None = None,
) -> torch.Tensor
```

AITER fused MoE kernel for ROCm.

##### `asm_moe_tkw1`

```python
@classmethod
def asm_moe_tkw1(
    cls,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    expert_map: torch.Tensor | None = None,
) -> torch.Tensor
```

Assembly-optimized MoE for token-K=1 cases.

#### Top-K Routing Operations

##### `topk_softmax`

```python
@classmethod
def topk_softmax(
    cls,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
) -> None
```

##### `biased_grouped_topk`

```python
@classmethod
def biased_grouped_topk(
    cls,
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    num_expert_group: int,
    topk_group: int,
    renormalize: bool = True,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]
```

##### `grouped_topk`

```python
@classmethod
def grouped_topk(
    cls,
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    num_expert_group: int,
    topk_group: int,
    renormalize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]
```

##### `fused_topk`

```python
@classmethod
def fused_topk(
    cls,
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]
```

#### MLA (Multi-head Latent Attention) Operations

##### `mla_decode_fwd`

```python
@classmethod
def mla_decode_fwd(
    cls,
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv_c_and_k_pe_cache: torch.Tensor,
    scale: float,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    y: torch.Tensor,
) -> None
```

AITER MLA decode forward pass.

#### GEMM Operations

##### `w8a8_gemm`

```python
@classmethod
def w8a8_gemm(
    cls,
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor | None = None,
    b_scale: torch.Tensor | None = None,
) -> torch.Tensor
```

W8A8 (FP8 weight, FP8 activation) GEMM for ROCm.

#### Quantization Operations

##### `per_tensor_quant`

```python
@classmethod
def per_tensor_quant(cls, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]
```

##### `per_token_quant`

```python
@classmethod
def per_token_quant(cls, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]
```

##### `group_fp8_quant`

```python
@classmethod
def group_fp8_quant(
    cls, x: torch.Tensor, group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]
```

#### Fused RMSNorm + Quant Operations

##### `rmsnorm_w8a8_bf16`

```python
@classmethod
def rmsnorm_w8a8_bf16(
    cls, x: torch.Tensor, weight: torch.Tensor, eps: float,
) -> tuple[torch.Tensor, torch.Tensor]
```

##### `rmsnorm_with_residual_w8a8_bf16`

```python
@classmethod
def rmsnorm_with_residual_w8a8_bf16(
    cls, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

#### All-Reduce Operations

##### `fused_allreduce_rmsnorm`

```python
@classmethod
def fused_allreduce_rmsnorm(
    cls,
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]
```

Fused all-reduce + RMS norm.

##### `fused_mla_dual_rms_norm`

```python
@classmethod
def fused_mla_dual_rms_norm(
    cls,
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]
```

#### Attention Operations

##### `flash_attn_varlen_func`

```python
@classmethod
def flash_attn_varlen_func(
    cls,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float | None = None,
    causal: bool = False,
) -> torch.Tensor
```

AITER Flash Attention for ROCm.

##### `pa_fwd_asm`

```python
@classmethod
def pa_fwd_asm(
    cls,
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    **kwargs,
) -> None
```

Assembly-optimized paged attention forward for ROCm.

---

## 9. XPU (Intel) Operations

**File**: `vllm/_xpu_ops.py`

The `xpu_ops` class provides Intel XPU GPU implementations of key operations,
dispatching to `vllm_xpu_kernels` and `torch.ops._C`.

### `xpu_ops` Class

```python
class xpu_ops:
```

All methods are class methods.

#### Attention Operations

##### `flash_attn_varlen_func`

```python
@classmethod
def flash_attn_varlen_func(
    cls,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    seqused_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] | None = None,
    alibi_slopes: torch.Tensor | None = None,
) -> torch.Tensor
```

Intel XPU Flash Attention via `vllm_xpu_kernels`.

##### `gdn_attention_core_xpu`

```python
@classmethod
def gdn_attention_core_xpu(
    cls,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    head_dim: int,
) -> torch.Tensor
```

GDN (Global Dependency Network) attention core for XPU.

#### Position Encoding Operations

##### `deepseek_scaling_rope`

```python
@classmethod
def deepseek_scaling_rope(
    cls,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    scaling_factor: float,
) -> None
```

DeepSeek-style scaling RoPE for XPU.

#### Sampling Operations

##### `topk_topp_sampler`

```python
@classmethod
def topk_topp_sampler(
    cls,
    logits: torch.Tensor,
    k: int,
    p: float | None,
    **kwargs,
) -> torch.Tensor
```

Top-K/Top-P sampling for XPU.

#### Quantization Operations

##### `mxfp8_quantize`

```python
@classmethod
def mxfp8_quantize(
    cls, input: torch.Tensor, scale: torch.Tensor,
) -> torch.Tensor
```

MXFP8 quantization for Intel XPU.

##### `mxfp4_quantize`

```python
@classmethod
def mxfp4_quantize(
    cls, input: torch.Tensor, scale: torch.Tensor,
) -> torch.Tensor
```

MXFP4 quantization for Intel XPU.

#### Cache Operations

##### `indexer_k_quant_and_cache`

```python
@classmethod
def indexer_k_quant_and_cache(
    cls,
    k: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    quant_block_size: int,
    kv_cache_dtype: str,
) -> None
```

Quantize and cache K for XPU.

##### `cp_gather_indexer_k_quant_cache`

```python
@classmethod
def cp_gather_indexer_k_quant_cache(
    cls,
    kv_cache: torch.Tensor,
    dst_k: torch.Tensor,
    dst_scale: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
) -> None
```

Gather and dequantize K cache for context parallelism on XPU.

#### Top-K Operations

##### `top_k_per_row_prefill`

```python
@classmethod
def top_k_per_row_prefill(
    cls,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]
```

Top-K per-row for prefill on XPU.

##### `top_k_per_row_decode`

```python
@classmethod
def top_k_per_row_decode(
    cls,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]
```

Top-K per-row for decode on XPU.

#### Fake GEMM Implementations

The class also provides fake (placeholder) implementations for operations that
require hardware-specific backends not available on all platforms:

```python
@classmethod
def fp8_gemm(cls, a, b, a_scale, b_scale, output_dtype, bias=None) -> torch.Tensor

@classmethod
def fp8_gemm_w8a16(cls, a, b, scale) -> torch.Tensor

@classmethod
def int4_gemm_w4a8(cls, a, b, a_scale, b_scale, scale) -> torch.Tensor

@classmethod
def int4_gemm_w4a16(cls, a, b, scale) -> torch.Tensor
```

These raise `NotImplementedError` or return placeholder tensors when the
XPU kernel backend is not available.

---

## Cross-Reference: Environment Variables

The following environment variables control kernel/operator selection:

| Variable | Default | Description |
|---|---|---|
| `VLLM_ROCM_USE_AITER` | `1` (if available) | Enable AITER ops on ROCm |
| `VLLM_ROCM_USE_AITER_LINEAR` | `1` | Use AITER for linear ops |
| `VLLM_ROCM_USE_AITER_MOE` | `1` | Use AITER for MoE ops |
| `VLLM_ROCM_USE_AITER_MLA` | `1` | Use AITER for MLA ops |
| `VLLM_ROCM_FP8_MFMA_PAGE_ATTN` | `False` | Use FP8 MFMA for paged attention on ROCm |
| `VLLM_USE_TRITON_AWQ` | `False` | Use Triton AWQ implementation |
| `VLLM_USE_OINK_OPS` | `False` | Enable Oink backend for IR ops |
| `VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE` | `256` | Max tokens per expert for FP4 MoE |

---

## Cross-Reference: C++ Extension Namespaces

| Namespace | Source | Description |
|---|---|---|
| `torch.ops._C` | `csrc/` | Core CUDA/C++ extensions |
| `torch.ops._rocm_C` | `csrc/rocm/` | ROCm-specific extensions |
| `torch.ops._moe_C` | `csrc/moe/` | MoE-specific extensions |
| `torch.ops._C_cache_ops` | `csrc/cache_hndl/` | KV cache operations |
| `torch.ops._C_cuda_utils` | `csrc/` | CUDA utility operations |
| `torch.ops._C_custom_ar` | `csrc/custom_all_reduce/` | Custom all-reduce |
| `torch.ops.vllm_aiter` | `vllm/kernels/aiter_ops.py` | AITER custom ops |
| `torch.ops.oink` | External plugin | Oink ops (Blackwell+) |
