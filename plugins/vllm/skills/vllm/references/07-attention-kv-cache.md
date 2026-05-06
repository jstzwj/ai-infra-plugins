# vLLM V1 Attention and KV Cache Reference

This document provides a comprehensive reference for the attention mechanisms and KV cache
management in vLLM's V1 architecture. It covers the attention backend abstraction, all
backend implementations, PagedAttention block management, prefix caching, KV cache
coordination, KV offloading, MLA (Multi-head Latent Attention), and all attention layer
implementations.

---

## Table of Contents

1. [Attention Backend Architecture](#1-attention-backend-architecture)
2. [Attention Type System](#2-attention-type-system)
3. [Attention Metadata](#3-attention-metadata)
4. [Attention Backend Registry](#4-attention-backend-registry)
5. [Attention Backend Implementations](#5-attention-backend-implementations)
6. [KV Cache Interface and Specs](#6-kv-cache-interface-and-specs)
7. [Block Pool and Block Management](#7-block-pool-and-block-management)
8. [KV Cache Manager](#8-kv-cache-manager)
9. [KV Cache Coordinator](#9-kv-cache-coordinator)
10. [Single-Type KV Cache Managers](#10-single-type-kv-cache-managers)
11. [Block Tables](#11-block-tables)
12. [Prefix Caching](#12-prefix-caching)
13. [KV Cache Utils](#13-kv-cache-utils)
14. [KV Cache Metrics](#14-kv-cache-metrics)
15. [Encoder Cache Manager](#15-encoder-cache-manager)
16. [KV Cache Offloading](#16-kv-cache-offloading)
17. [MLA (Multi-head Latent Attention)](#17-mla-multi-head-latent-attention)
18. [Attention Layer Implementations](#18-attention-layer-implementations)
19. [Attention Operations](#19-attention-operations)
20. [Sliding Window Attention](#20-sliding-window-attention)
21. [Cross-Attention for Encoder-Decoder](#21-cross-attention-for-encoder-decoder)
22. [Attention Selector](#22-attention-selector)
23. [KV Cache Layout](#23-kv-cache-layout)

---

## 1. Attention Backend Architecture

The attention system in vLLM V1 is built around an abstract base class hierarchy that
separates the concerns of backend selection, metadata building, and attention computation.

### AttentionBackend (Abstract Base Class)

**File:** `vllm/v1/attention/backend.py`

```python
class AttentionBackend(ABC):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list["CacheDType"]] = [
        "auto", "float16", "bfloat16",
    ]
    forward_includes_kv_cache_update: bool = True
```

#### Class Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `get_name` | `() -> str` | Returns the backend name identifier |
| `get_impl_cls` | `() -> type[AttentionImplBase]` | Returns the attention implementation class |
| `get_builder_cls` | `() -> type[AttentionMetadataBuilder]` | Returns the metadata builder class |
| `get_kv_cache_shape` | `(num_blocks: int, block_size: int, num_kv_heads: int, head_size: int, cache_dtype_str: str = "auto") -> tuple[int, ...]` | Returns the shape of the KV cache tensor |
| `get_kv_cache_block_dim` | `(block_size: int, num_kv_heads: int, head_size: int, cache_dtype_str: str = "auto") -> int` | Discovers which tensor dim is the block index |
| `get_kv_cache_stride_order` | `(include_num_layers_dimension: bool = False) -> tuple[int, ...]` | Gets the physical memory ordering of KV cache dimensions |
| `get_supported_head_sizes` | `() -> list[int]` | Returns supported head sizes (empty = all) |
| `supports_head_size` | `(head_size: int) -> bool` | Checks if head size is supported |
| `supports_dtype` | `(dtype: torch.dtype) -> bool` | Checks if dtype is supported |
| `supports_kv_cache_dtype` | `(kv_cache_dtype: CacheDType \| None) -> bool` | Checks if KV cache dtype is supported |
| `supports_block_size` | `(block_size: int \| None) -> bool` | Checks if block size is supported |
| `get_preferred_block_size` | `(default_block_size: int) -> int` | Gets the preferred block size |
| `is_mla` | `() -> bool` | Whether this is an MLA backend |
| `supports_sink` | `() -> bool` | Whether attention sinks are supported |
| `supports_alibi_sqrt` | `() -> bool` | Whether ALIBI sqrt is supported |
| `supports_mm_prefix` | `() -> bool` | Whether multimodal prefix is supported |
| `is_sparse` | `() -> bool` | Whether this is a sparse attention backend |
| `supports_per_head_quant_scales` | `() -> bool` | Whether per-head quant scales are supported |
| `supports_non_causal` | `() -> bool` | Whether non-causal (bidirectional) attention is supported |
| `supports_batch_invariance` | `() -> bool` | Whether batch invariant mode is supported |
| `supports_attn_type` | `(attn_type: str) -> bool` | Whether a given attention type is supported |
| `supports_compute_capability` | `(capability: DeviceCapability) -> bool` | Whether a compute capability is supported |
| `validate_configuration` | `(head_size, dtype, kv_cache_dtype, block_size, use_mla, has_sink, use_sparse, use_mm_prefix, use_per_head_quant_scales, device_capability, attn_type, use_non_causal=False, use_batch_invariant=False) -> list[str]` | Validates full configuration, returns list of invalid reasons |
| `get_required_kv_cache_layout` | `() -> KVCacheLayoutType \| None` | Returns required KV cache layout if any |
| `get_supported_kernel_block_sizes` | `() -> list[int \| MultipleOf]` | Returns supported kernel block sizes |
| `is_ssm` | `() -> bool` | Whether this is an SSM (state space model) backend |

#### validate_configuration

Full parameter signature:

```python
@classmethod
def validate_configuration(
    cls,
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: CacheDType | None,
    block_size: int | None,
    use_mla: bool,
    has_sink: bool,
    use_sparse: bool,
    use_mm_prefix: bool,
    use_per_head_quant_scales: bool,
    device_capability: DeviceCapability,
    attn_type: str,
    use_non_causal: bool = False,
    use_batch_invariant: bool = False,
) -> list[str]:
```

### AttentionImplBase (Abstract Base)

```python
class AttentionImplBase(ABC, Generic[T]):
    num_heads: int
    head_size: int
    scale: float
    can_return_lse_for_decode: bool = False
    supports_pcp: bool = False
    supports_mtp_with_cp_non_trivial_interleave_size: bool = False
    need_to_return_lse_for_decode: bool = False
    supports_quant_query_input: bool = False
    dcp_world_size: int
    dcp_rank: int
    pcp_world_size: int
    pcp_rank: int
    total_cp_world_size: int
    total_cp_rank: int
```

### AttentionImpl (Standard Implementation)

```python
class AttentionImpl(AttentionImplBase[T], Generic[T]):
    kv_cache_dtype: str

    @abstractmethod
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
    ) -> None: ...

    @abstractmethod
    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: T,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
```

### MLAAttentionImpl (MLA Implementation)

```python
class MLAAttentionImpl(AttentionImplBase[T], Generic[T]):
    @abstractmethod
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        qk_head_dim: int,
        v_head_dim: int,
        kv_b_proj: ColumnParallelLinear,
        indexer: object | None = None,
        q_pad_num_heads: int | None = None,
    ) -> None: ...

    @abstractmethod
    def forward_mha(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: T,
        k_scale: torch.Tensor,
        output: torch.Tensor,
    ) -> None: ...

    @abstractmethod
    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: T,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
```

### SparseMLAAttentionImpl

```python
class SparseMLAAttentionImpl(AttentionImplBase[T], Generic[T]):
    @abstractmethod
    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: T,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
```

---

## 2. Attention Type System

### AttentionType Enum

```python
class AttentionType(str, Enum):
    DECODER = "decoder"
    """Decoder attention between previous layer Q/K/V."""
    ENCODER = "encoder"
    """Encoder attention between previous layer Q/K/V for encoder-decoder."""
    ENCODER_ONLY = "encoder_only"
    """Encoder attention between previous layer Q/K/V."""
    ENCODER_DECODER = "encoder_decoder"
    """Attention between dec. Q and enc. K/V for encoder-decoder."""
```

### AttentionCGSupport Enum

```python
class AttentionCGSupport(Enum):
    ALWAYS = 3
    """Cudagraph always supported; supports mixed-prefill-decode"""
    UNIFORM_BATCH = 2
    """Cudagraph supported for batches with the same query lengths"""
    UNIFORM_SINGLE_TOKEN_DECODE = 1
    """Cudagraph supported for batches with query_len==1 decodes"""
    NEVER = 0
    """No cudagraph support"""
```

### MultipleOf Helper

```python
class MultipleOf:
    base: int
    def __init__(self, base: int):
        self.base = base
```

Used in `get_supported_kernel_block_sizes()` to indicate the block size must be a
multiple of `base`.

---

## 3. Attention Metadata

### CommonAttentionMetadata

**File:** `vllm/v1/attention/backend.py`

```python
@dataclass
class CommonAttentionMetadata:
    query_start_loc: torch.Tensor
    query_start_loc_cpu: torch.Tensor
    """(batch_size + 1,), start location of each request in query Tensor"""

    seq_lens: torch.Tensor
    """(batch_size,), number of computed tokens for each request"""

    num_reqs: int
    num_actual_tokens: int
    max_query_len: int
    max_seq_len: int

    block_table_tensor: torch.Tensor
    slot_mapping: torch.Tensor

    causal: bool = True
    logits_indices_padded: torch.Tensor | None = None
    num_logits_indices: int | None = None
    encoder_seq_lens: torch.Tensor | None = None
    encoder_seq_lens_cpu: np.ndarray | None = None
    dcp_local_seq_lens: torch.Tensor | None = None
    dcp_local_seq_lens_cpu: torch.Tensor | None = None
    positions: torch.Tensor | None = None
    is_prefilling: torch.Tensor | None = None
    seq_lens_cpu_upper_bound: torch.Tensor | None = None
```

#### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `batch_size` | `() -> int` | Returns batch size from seq_lens |
| `naive_query_lens` | `() -> torch.Tensor` | Assumes query ends where next starts |
| `replace` | `(**kwargs) -> CommonAttentionMetadata` | Creates a copy with replacements |
| `compute_num_computed_tokens` | `() -> torch.Tensor` | Computes on device: seq_lens - query_lens |
| `unpadded` | `(num_actual_tokens, num_actual_reqs) -> CommonAttentionMetadata` | Returns trimmed version |

### AttentionMetadataBuilder (Abstract)

```python
class AttentionMetadataBuilder(ABC, Generic[M]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER
    reorder_batch_threshold: int | None = None
    supports_update_block_table: bool = False
```

#### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(kv_cache_spec: AttentionSpec, layer_names: list[str], vllm_config: VllmConfig, device: torch.device)` | Initialize with spec and config |
| `get_cudagraph_support` | `(cls, vllm_config, kv_cache_spec) -> AttentionCGSupport` | Class method for CUDA graph support level |
| `build` | `(common_prefix_len: int, common_attn_metadata: CommonAttentionMetadata, fast_build: bool = False) -> M` | Central method to build attention metadata |
| `update_block_table` | `(metadata: M, blk_table: torch.Tensor, slot_mapping: torch.Tensor) -> M` | Update block table in existing metadata |
| `build_for_cudagraph_capture` | `(common_attn_metadata) -> M` | Build for CUDA graph capture |
| `build_for_drafting` | `(common_attn_metadata, draft_index: int) -> M` | Build for draft model |
| `use_cascade_attention` | `(common_prefix_len, query_lens, num_query_heads, num_kv_heads, use_alibi, use_sliding_window, use_local_attention, num_sms, dcp_world_size) -> bool` | Whether to use cascade attention |

---

## 4. Attention Backend Registry

**File:** `vllm/v1/attention/backends/registry.py`

### AttentionBackendEnum

All supported attention backends:

| Enum Member | Class Path | Description |
|-------------|-----------|-------------|
| `FLASH_ATTN` | `vllm.v1.attention.backends.flash_attn.FlashAttentionBackend` | FlashAttention 2/3 |
| `FLASH_ATTN_DIFFKV` | `vllm.v1.attention.backends.flash_attn_diffkv.FlashAttentionDiffKVBackend` | FlashAttention with different K/V |
| `TRITON_ATTN` | `vllm.v1.attention.backends.triton_attn.TritonAttentionBackend` | Pure Triton attention |
| `ROCM_ATTN` | `vllm.v1.attention.backends.rocm_attn.RocmAttentionBackend` | ROCm attention |
| `ROCM_AITER_MLA` | `vllm.v1.attention.backends.mla.rocm_aiter_mla.AiterMLABackend` | ROCm Aiter MLA |
| `ROCM_AITER_FA` | `vllm.v1.attention.backends.rocm_aiter_fa.AiterFlashAttentionBackend` | ROCm Aiter FlashAttention |
| `XPU_MLA_SPARSE` | `vllm.v1.attention.backends.mla.xpu_mla_sparse.XPUMLASparseBackend` | XPU MLA sparse |
| `FLASHINFER` | `vllm.v1.attention.backends.flashinfer.FlashInferBackend` | FlashInfer backend |
| `FLASHINFER_MLA` | `vllm.v1.attention.backends.mla.flashinfer_mla.FlashInferMLABackend` | FlashInfer MLA |
| `FLASHINFER_MLA_SPARSE` | `...flashinfer_mla_sparse.FlashInferMLASparseBackend` | FlashInfer MLA sparse |
| `TRITON_MLA` | `vllm.v1.attention.backends.mla.triton_mla.TritonMLABackend` | Triton MLA |
| `CUTLASS_MLA` | `vllm.v1.attention.backends.mla.cutlass_mla.CutlassMLABackend` | CUTLASS MLA |
| `FLASHMLA` | `vllm.v1.attention.backends.mla.flashmla.FlashMLABackend` | FlashMLA |
| `FLASH_ATTN_MLA` | `vllm.v1.attention.backends.mla.flashattn_mla.FlashAttnMLABackend` | FlashAttention MLA |
| `FLEX_ATTENTION` | `vllm.v1.attention.backends.flex_attention.FlexAttentionBackend` | FlexAttention (PyTorch) |
| `TREE_ATTN` | `vllm.v1.attention.backends.tree_attn.TreeAttentionBackend` | Tree attention |
| `ROCM_AITER_UNIFIED_ATTN` | `...rocm_aiter_unified_attn.RocmAiterUnifiedAttentionBackend` | ROCm unified |
| `CPU_ATTN` | `vllm.v1.attention.backends.cpu_attn.CPUAttentionBackend` | CPU attention |
| `TURBOQUANT` | `vllm.v1.attention.backends.turboquant_attn.TurboQuantAttentionBackend` | TurboQuant |
| `CUSTOM` | `None` | Custom third-party backend |

### MambaAttentionBackendEnum

| Enum Member | Class Path | Description |
|-------------|-----------|-------------|
| `MAMBA1` | `vllm.v1.attention.backends.mamba1_attn.Mamba1AttentionBackend` | Mamba 1 SSM |
| `MAMBA2` | `vllm.v1.attention.backends.mamba2_attn.Mamba2AttentionBackend` | Mamba 2 SSM |
| `SHORT_CONV` | `vllm.v1.attention.backends.short_conv_attn.ShortConvAttentionBackend` | Short convolution |
| `LINEAR` | `vllm.v1.attention.backends.linear_attn.LinearAttentionBackend` | Linear attention |
| `GDN_ATTN` | `vllm.v1.attention.backends.gdn_attn.GDNAttentionBackend` | GDN attention |

### register_backend

```python
def register_backend(
    backend: AttentionBackendEnum | MambaAttentionBackendEnum,
    class_path: str | None = None,
    is_mamba: bool = False,
) -> Callable[[type], type]:
```

Register or override a backend implementation. Can be used as a decorator:

```python
@register_backend(AttentionBackendEnum.FLASH_ATTN)
class MyCustomFlashAttn: ...
```

---

## 5. Attention Backend Implementations

### 5.1 FlashAttention Backend

**File:** `vllm/v1/attention/backends/flash_attn.py`

```python
class FlashAttentionBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    forward_includes_kv_cache_update: bool = False
```

Key characteristics:
- KV cache shape: `(2, num_blocks, block_size, num_kv_heads, head_size)`
- Supports NHD and HND layouts via `get_kv_cache_stride_order()`
- Block size must be a multiple of 16
- Supports head sizes up to 256 (512 with FA4)
- Supports all attention types (DECODER, ENCODER, ENCODER_ONLY, ENCODER_DECODER)
- Supports non-causal attention
- Supports batch invariance
- Supports per-head quant scales (FA3+)
- Supports attention sinks (FA3+)
- CUDA graph support: `AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE`

#### FlashAttentionMetadataBuilder

```python
class FlashAttentionMetadataBuilder(AttentionMetadataBuilder[FlashAttentionMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
```

#### FlashAttentionImpl

```python
class FlashAttentionImpl(AttentionImpl[FlashAttentionMetadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
    ): ...

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
```

### 5.2 FlashInfer Backend

**File:** `vllm/v1/attention/backends/flashinfer.py`

```python
class FlashInferBackend(AttentionBackend):
    forward_includes_kv_cache_update: bool = True
```

Key characteristics:
- Uses FlashInfer library for attention computation
- Supports cascade attention via `MultiLevelCascadeAttentionWrapper`
- Supports TRT-LLM Gen attention kernels
- Workspace buffer: `VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE`
- CUDA graph support: `AttentionCGSupport.ALWAYS`
- Supports FP8 and FP4 KV cache
- Supports batch invariance

### 5.3 Triton Attention Backend

**File:** `vllm/v1/attention/backends/triton_attn.py`

```python
class TritonAttentionBackend(AttentionBackend):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS
```

Key characteristics:
- Pure Triton kernel implementation
- No external dependencies
- Supports cascade attention
- Parallel softmax segmentation: `NUM_PAR_SOFTMAX_SEGMENTS = 16`
- Minimum 2D launch grid: `MIN_LAUNCH_GRID_SIZE_2D = 128`

#### TritonAttentionMetadata

```python
@dataclass
class TritonAttentionMetadata:
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    seq_threshold_3D: int
    num_par_softmax_segments: int
    softmax_segm_output: torch.Tensor
    softmax_segm_max: torch.Tensor
    softmax_segm_expsum: torch.Tensor
    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: torch.Tensor | None
    prefix_kv_lens: torch.Tensor | None
    suffix_kv_lens: torch.Tensor | None
    scheduler_metadata: torch.Tensor | None = None
    prefix_scheduler_metadata: torch.Tensor | None = None
    mm_prefix_range: dict[int, list[tuple[int, int]]] | None = None
```

### 5.4 ROCm Attention Backend

**File:** `vllm/v1/attention/backends/rocm_attn.py`

ROCm-specific attention backend for AMD GPUs.

### 5.5 CPU Attention Backend

**File:** `vllm/v1/attention/backends/cpu_attn.py`

CPU-based attention backend using PyTorch SDPA.

### 5.6 MLA Backends

MLA backends support the DeepSeek V2/V3/V4 style Multi-head Latent Attention.

| Backend | File | Description |
|---------|------|-------------|
| FlashInferMLABackend | `mla/flashinfer_mla.py` | FlashInfer-based MLA |
| FlashInferMLASparseBackend | `mla/flashinfer_mla_sparse.py` | FlashInfer MLA sparse |
| TritonMLABackend | `mla/triton_mla.py` | Triton MLA |
| CutlassMLABackend | `mla/cutlass_mla.py` | CUTLASS MLA |
| FlashMLABackend | `mla/flashmla.py` | FlashMLA |
| FlashMLASparseBackend | `mla/flashmla_sparse.py` | FlashMLA sparse |
| FlashAttnMLABackend | `mla/flashattn_mla.py` | FlashAttention MLA |
| AiterMLABackend | `mla/rocm_aiter_mla.py` | ROCm Aiter MLA |
| AiterTritonMLABackend | `mla/aiter_triton_mla.py` | ROCm Aiter Triton MLA |
| ROCMAiterMLASparseBackend | `mla/rocm_aiter_mla_sparse.py` | ROCm Aiter MLA sparse |
| XPUMLASparseBackend | `mla/xpu_mla_sparse.py` | XPU MLA sparse |

### 5.7 Other Backends

| Backend | File | Description |
|---------|------|-------------|
| FlashAttentionDiffKVBackend | `flash_attn_diffkv.py` | Different K/V dimensions |
| FlexAttentionBackend | `flex_attention.py` | PyTorch FlexAttention |
| TreeAttentionBackend | `tree_attn.py` | Tree attention for speculative decoding |
| TurboQuantAttentionBackend | `turboquant_attn.py` | TurboQuant attention |
| RocmAiterUnifiedAttentionBackend | `rocm_aiter_unified_attn.py` | ROCm unified |

### 5.8 Mamba/SSM Backends

| Backend | Description |
|---------|-------------|
| Mamba1AttentionBackend | Mamba 1 state space model |
| Mamba2AttentionBackend | Mamba 2 state space model |
| ShortConvAttentionBackend | Short convolution attention |
| LinearAttentionBackend | Linear attention |
| GDNAttentionBackend | Gated attention |

---

## 6. KV Cache Interface and Specs

**File:** `vllm/v1/kv_cache_interface.py`

### KVQuantMode

```python
class KVQuantMode(IntEnum):
    NONE = 0
    FP8_PER_TENSOR = 1
    INT8_PER_TOKEN_HEAD = 2
    FP8_PER_TOKEN_HEAD = 3
    NVFP4 = 4

    @property
    def is_per_token_head(self) -> bool: ...
    @property
    def is_nvfp4(self) -> bool: ...
```

### KVCacheSpec (Base)

```python
@dataclass(frozen=True)
class KVCacheSpec:
    block_size: int

    @property
    def page_size_bytes(self) -> int: ...
    @property
    def storage_block_size(self) -> int: ...
    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int: ...
    def copy_with_new_block_size(self, block_size: int) -> Self: ...
    @classmethod
    def merge(cls, specs: list[Self]) -> Self: ...
```

### AttentionSpec

```python
@dataclass(frozen=True, kw_only=True)
class AttentionSpec(KVCacheSpec):
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype
    kv_quant_mode: KVQuantMode = KVQuantMode.NONE
    page_size_padded: int | None = None

    @property
    def page_size_bytes(self) -> int: ...
    @property
    def real_page_size_bytes(self) -> int: ...
```

### FullAttentionSpec

```python
@dataclass(frozen=True, kw_only=True)
class FullAttentionSpec(AttentionSpec):
    head_size_v: int = None  # defaults to head_size
    sliding_window: int | None = None
    attention_chunk_size: int | None = None

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int: ...
    @classmethod
    def merge(cls, specs: list[Self]) -> Self: ...
```

### MLAAttentionSpec

```python
@dataclass(frozen=True, kw_only=True)
class MLAAttentionSpec(FullAttentionSpec):
    cache_dtype_str: str | None = None
    alignment: int | None = None
    compress_ratio: int = 1
    model_version: str | None = None

    @property
    def storage_block_size(self) -> int: ...
    @property
    def real_page_size_bytes(self) -> int: ...
```

### SlidingWindowSpec

```python
@dataclass(frozen=True, kw_only=True)
class SlidingWindowSpec(AttentionSpec):
    sliding_window: int
    head_size_v: int = None  # defaults to head_size

    def max_admission_blocks_per_request(
        self, max_num_batched_tokens: int, max_model_len: int
    ) -> int: ...
    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int: ...
```

### ChunkedLocalAttentionSpec

```python
@dataclass(frozen=True, kw_only=True)
class ChunkedLocalAttentionSpec(AttentionSpec):
    attention_chunk_size: int

    def max_admission_blocks_per_request(
        self, max_num_batched_tokens: int, max_model_len: int
    ) -> int: ...
    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int: ...
```

### SlidingWindowMLASpec

```python
@dataclass(frozen=True, kw_only=True)
class SlidingWindowMLASpec(SlidingWindowSpec):
    cache_dtype_str: str | None = None
    alignment: int | None = None
    compress_ratio: int = 1
    model_version: str | None = None
```

### MambaSpec

```python
@dataclass(frozen=True)
class MambaSpec(KVCacheSpec):
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[torch.dtype]
    page_size_padded: int | None = None
    mamba_type: str = "mamba2"
    mamba_cache_mode: str = "none"
    num_speculative_blocks: int = 0
```

### CrossAttentionSpec

```python
@dataclass(frozen=True)
class CrossAttentionSpec(AttentionSpec):
    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int: ...
```

### SinkFullAttentionSpec

```python
@dataclass(frozen=True)
class SinkFullAttentionSpec(FullAttentionSpec):
    sink_len: int | None = None
```

### TQFullAttentionSpec

```python
@dataclass(frozen=True, kw_only=True)
class TQFullAttentionSpec(FullAttentionSpec):
    tq_slot_size: int = 0
```

### UniformTypeKVCacheSpecs

```python
@dataclass(frozen=True)
class UniformTypeKVCacheSpecs(KVCacheSpec):
    kv_cache_specs: dict[str, KVCacheSpec]

    @property
    def page_size_bytes(self) -> int: ...
    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int: ...
    @classmethod
    def is_uniform_type(cls, kv_cache_specs: dict[str, KVCacheSpec]) -> bool: ...
    @classmethod
    def from_specs(cls, kv_cache_specs: dict[str, KVCacheSpec]) -> Self | None: ...
    def get_page_sizes(self) -> list[int]: ...
    def get_num_layer_tuples(self) -> int: ...
    def max_memory_usage_pages(self, vllm_config: VllmConfig) -> int: ...
```

### KVCacheTensor

```python
@dataclass
class KVCacheTensor:
    size: int                    # size in bytes
    shared_by: list[str]         # layer names sharing this tensor
```

### KVCacheGroupSpec

```python
@dataclass
class KVCacheGroupSpec:
    layer_names: list[str]
    kv_cache_spec: KVCacheSpec
    is_eagle_group: bool = False
```

### KVCacheConfig

```python
@dataclass
class KVCacheConfig:
    num_blocks: int
    kv_cache_tensors: list[KVCacheTensor]
    kv_cache_groups: list[KVCacheGroupSpec]

    @property
    def has_mamba_layers(self) -> bool: ...
    @property
    def needs_kv_cache_zeroing(self) -> bool: ...
```

---

## 7. Block Pool and Block Management

**File:** `vllm/v1/core/block_pool.py`

### KVCacheBlock

```python
@dataclass(slots=True)
class KVCacheBlock:
    block_id: int
    ref_cnt: int = 0
    _block_hash: BlockHashWithGroupId | None = None
    prev_free_block: KVCacheBlock | None = None
    next_free_block: KVCacheBlock | None = None
    is_null: bool = False

    @property
    def block_hash(self) -> BlockHashWithGroupId | None: ...
    @block_hash.setter
    def block_hash(self, block_hash: BlockHashWithGroupId): ...
    def reset_hash(self) -> None: ...
```

### FreeKVCacheBlockQueue

Doubly-linked list of free blocks, ordered by LRU eviction priority.

```python
class FreeKVCacheBlockQueue:
    def __init__(self, blocks: list[KVCacheBlock]) -> None: ...
    num_free_blocks: int

    def popleft(self) -> KVCacheBlock: ...
    def popleft_n(self, n: int) -> list[KVCacheBlock]: ...
    def remove(self, block: KVCacheBlock) -> None: ...
    def append(self, block: KVCacheBlock) -> None: ...
    def append_n(self, blocks: list[KVCacheBlock]) -> None: ...
    def get_all_free_blocks(self) -> list[KVCacheBlock]: ...
```

### BlockHashToBlockMap

```python
class BlockHashToBlockMap:
    def get_one_block(self, key: BlockHashWithGroupId) -> KVCacheBlock | None: ...
    def insert(self, key: BlockHashWithGroupId, block: KVCacheBlock) -> None: ...
    def pop(self, key: BlockHashWithGroupId, block_id: int) -> KVCacheBlock | None: ...
```

### BlockPool

```python
class BlockPool:
    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool,
        hash_block_size: int,
        enable_kv_cache_events: bool = False,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ): ...
```

#### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `get_cached_block` | `(block_hash: BlockHash, kv_cache_group_ids: list[int]) -> list[KVCacheBlock] \| None` | Get cached block by hash |
| `cache_full_blocks` | `(request, blocks, num_cached_blocks, num_full_blocks, block_size, kv_cache_group_id) -> None` | Cache full blocks for prefix caching |
| `get_new_blocks` | `(num_blocks: int) -> list[KVCacheBlock]` | Allocate new blocks from free pool |
| `touch` | `(blocks: Sequence[KVCacheBlock]) -> None` | Increase ref count for cache hit |
| `free_blocks` | `(ordered_blocks: Iterable[KVCacheBlock]) -> None` | Free blocks in eviction order |
| `evict_blocks` | `(block_ids: set[int]) -> None` | Evict blocks by ID from prefix cache |
| `reset_prefix_cache` | `() -> bool` | Reset entire prefix cache |
| `get_num_free_blocks` | `() -> int` | Get number of free blocks |
| `get_usage` | `() -> float` | Get KV cache usage (0.0-1.0) |
| `take_events` | `() -> list[KVCacheEvent]` | Atomically take all KV cache events |

---

## 8. KV Cache Manager

**File:** `vllm/v1/core/kv_cache_manager.py`

### KVCacheBlocks

```python
@dataclass
class KVCacheBlocks:
    blocks: tuple[Sequence[KVCacheBlock], ...]

    def __add__(self, other: KVCacheBlocks) -> KVCacheBlocks: ...
    def get_block_ids(self, allow_none: bool = False) -> tuple[list[int], ...] | None: ...
    def get_unhashed_block_ids(self) -> list[int]: ...
    def get_unhashed_block_ids_all_groups(self) -> list[list[int]]: ...
    def new_empty(self) -> KVCacheBlocks: ...
```

### KVCacheManager

```python
class KVCacheManager:
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        hash_block_size: int,
        max_num_batched_tokens: int | None = None,
        enable_caching: bool = True,
        use_eagle: bool = False,
        log_stats: bool = False,
        enable_kv_cache_events: bool = False,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ): ...
```

#### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `usage` | `property -> float` | KV cache usage ratio |
| `make_prefix_cache_stats` | `() -> PrefixCacheStats \| None` | Get and reset prefix cache stats |
| `get_computed_blocks` | `(request: Request) -> tuple[KVCacheBlocks, int]` | Get cached blocks for request |
| `allocate_slots` | `(request, num_new_tokens, num_new_computed_tokens=0, new_computed_blocks=None, num_lookahead_tokens=0, num_external_computed_tokens=0, delay_cache_blocks=False, num_encoder_tokens=0, full_sequence_must_fit=False) -> KVCacheBlocks \| None` | Allocate KV cache slots |
| `free` | `(request: Request) -> None` | Free request's blocks |
| `remove_skipped_blocks` | `(request_id, total_computed_tokens) -> None` | Remove blocks outside attention window |
| `evict_blocks` | `(block_ids: set[int]) -> None` | Evict blocks from prefix cache |
| `reset_prefix_cache` | `() -> bool` | Reset prefix cache |
| `get_num_common_prefix_blocks` | `(running_request_id) -> list[int]` | Count common prefix blocks |
| `take_events` | `() -> list[KVCacheEvent]` | Take KV cache events |
| `get_blocks` | `(request_id) -> KVCacheBlocks` | Get blocks for a request |
| `get_block_ids` | `(request_id) -> tuple[list[int], ...]` | Get block IDs for a request |
| `cache_blocks` | `(request, num_computed_tokens) -> None` | Cache blocks for request |
| `take_new_block_ids` | `() -> list[int]` | Drain and return new block IDs |
| `new_step_starts` | `() -> None` | Called when a new step starts |

#### allocate_slots Block Layout

```
----------------------------------------------------------------------
| < comp > | < new_comp > | < ext_comp >  | < new >  | < lookahead > |
----------------------------------------------------------------------
                          |   < to be computed >                     |
----------------------------------------------------------------------
              |            < to be allocated >                        |
----------------------------------------------------------------------
```

Abbreviations:
- `comp` = `request.num_computed_tokens`
- `new_comp` = `num_new_computed_tokens` = `len(new_computed_blocks) * block_size`
- `ext_comp` = `num_external_computed_tokens` (cached by connector)
- `new` = `num_new_tokens` (including unverified draft tokens)
- `lookahead` = `num_lookahead_tokens`

---

## 9. KV Cache Coordinator

**File:** `vllm/v1/core/kv_cache_coordinator.py`

### KVCacheCoordinator (Abstract)

```python
class KVCacheCoordinator(ABC):
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        max_num_batched_tokens: int,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        hash_block_size: int,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ): ...
```

#### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `get_num_blocks_to_allocate` | `(request_id, num_tokens, new_computed_blocks, num_encoder_tokens, total_computed_tokens, num_tokens_main_model, apply_admission_cap=False) -> int` | Calculate blocks to allocate |
| `allocate_new_computed_blocks` | `(request_id, new_computed_blocks, num_local_computed_tokens, num_external_computed_tokens) -> None` | Add new computed blocks |
| `allocate_new_blocks` | `(request_id, num_tokens, num_tokens_main_model, num_encoder_tokens=0) -> tuple[list[KVCacheBlock], ...]` | Allocate new blocks |
| `cache_blocks` | `(request, num_computed_tokens) -> None` | Cache blocks |
| `free` | `(request_id) -> None` | Free request blocks |
| `get_num_common_prefix_blocks` | `(running_request_id) -> list[int]` | Count common prefix blocks |
| `remove_skipped_blocks` | `(request_id, total_computed_tokens) -> None` | Remove skipped blocks |
| `find_longest_cache_hit` | `(block_hashes, max_cache_hit_length) -> tuple[tuple[list[KVCacheBlock], ...], int]` | Abstract: find longest cache hit |
| `new_step_starts` | `() -> None` | Called at step start |

### KVCacheCoordinatorNoPrefixCache

Used when prefix caching is disabled. Supports arbitrary numbers of KV cache groups
including 0 groups.

```python
class KVCacheCoordinatorNoPrefixCache(KVCacheCoordinator):
    def find_longest_cache_hit(...) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        return empty_blocks, 0
```

### UnitaryKVCacheCoordinator

For models with only one KV cache group (single attention type).

```python
class UnitaryKVCacheCoordinator(KVCacheCoordinator):
    def find_longest_cache_hit(
        self, block_hashes, max_cache_hit_length
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        # Left-to-right scan of block hashes
        ...
```

### HybridKVCacheCoordinator

For hybrid models with multiple KV cache types.

```python
class HybridKVCacheCoordinator(KVCacheCoordinator):
    def find_longest_cache_hit(
        self, block_hashes, max_cache_hit_length
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        # Iterative fixed-point algorithm across attention types
        ...
```

### get_kv_cache_coordinator

```python
def get_kv_cache_coordinator(
    kv_cache_config: KVCacheConfig,
    max_model_len: int,
    max_num_batched_tokens: int,
    use_eagle: bool,
    enable_caching: bool,
    enable_kv_cache_events: bool,
    dcp_world_size: int,
    pcp_world_size: int,
    hash_block_size: int,
    metrics_collector: KVCacheMetricsCollector | None = None,
) -> KVCacheCoordinator:
```

Returns:
- `KVCacheCoordinatorNoPrefixCache` if caching disabled
- `UnitaryKVCacheCoordinator` if single KV cache group
- `HybridKVCacheCoordinator` if multiple KV cache groups

---

## 10. Single-Type KV Cache Managers

**File:** `vllm/v1/core/single_type_kv_cache_manager.py`

### SingleTypeKVCacheManager (Abstract Base)

```python
class SingleTypeKVCacheManager(ABC):
    def __init__(
        self,
        kv_cache_spec: KVCacheSpec,
        block_pool: BlockPool,
        enable_caching: bool,
        kv_cache_group_id: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
        max_admission_blocks_per_request: int | None = None,
    ): ...
```

#### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `get_num_blocks_to_allocate` | `(request_id, num_tokens, new_computed_blocks, total_computed_tokens, num_tokens_main_model, apply_admission_cap=False) -> int` | Calculate blocks needed |
| `allocate_new_computed_blocks` | `(request_id, new_computed_blocks, num_local_computed_tokens, num_external_computed_tokens) -> None` | Handle prefix cache hits |
| `allocate_new_blocks` | `(request_id, num_tokens, num_tokens_main_model) -> list[KVCacheBlock]` | Allocate new blocks |
| `cache_blocks` | `(request, num_tokens) -> None` | Cache full blocks |
| `free` | `(request_id) -> None` | Free request blocks |
| `get_num_common_prefix_blocks` | `(running_request_id) -> int` | Count common prefix blocks |
| `find_longest_cache_hit` | `(cls, block_hashes, max_length, kv_cache_group_ids, block_pool, kv_cache_spec, use_eagle, alignment_tokens, dcp_world_size, pcp_world_size) -> tuple[list[KVCacheBlock], ...]` | Class method: find cache hit |
| `remove_skipped_blocks` | `(request_id, total_computed_tokens) -> None` | Remove blocks outside attention window |
| `get_num_skipped_tokens` | `(num_computed_tokens) -> int` | Default: 0 |
| `take_new_block_ids` | `() -> list[int]` | Drain new block IDs |
| `new_step_starts` | `() -> None` | Called at step start |

### FullAttentionManager

```python
class FullAttentionManager(SingleTypeKVCacheManager):
    # Never skips tokens (get_num_skipped_tokens returns 0)
    # Left-to-right cache hit scan
    # Supports EAGLE drop of last matched block
```

### SlidingWindowManager

```python
class SlidingWindowManager(SingleTypeKVCacheManager):
    def __init__(self, kv_cache_spec: SlidingWindowSpec, **kwargs) -> None: ...
    sliding_window: int

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        return max(0, num_computed_tokens - self.sliding_window + 1)

    # Right-to-left cache hit scan with contiguous block requirement
```

### ChunkedLocalAttentionManager

```python
class ChunkedLocalAttentionManager(SingleTypeKVCacheManager):
    def __init__(self, kv_cache_spec: ChunkedLocalAttentionSpec, **kwargs) -> None: ...
    attention_chunk_size: int

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        return (num_computed_tokens // self.attention_chunk_size) * self.attention_chunk_size
```

### MambaManager

```python
class MambaManager(SingleTypeKVCacheManager):
    def __init__(self, kv_cache_spec: MambaSpec, block_pool, **kwargs) -> None: ...
    cached_blocks_this_step: set[BlockHashWithGroupId]
    mamba_cache_mode: str
    num_speculative_blocks: int

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        return num_computed_tokens - 1

    # Supports "align" mode for prefix caching with Mamba
    # Last state block tracking for efficient state management
```

### CrossAttentionManager

```python
class CrossAttentionManager(SingleTypeKVCacheManager):
    # No prefix caching for cross-attention
    # No cache hit support
    # Used for encoder-decoder models
```

### SinkFullAttentionManager

```python
class SinkFullAttentionManager(FullAttentionManager):
    def __init__(
        self,
        kv_cache_spec: SinkFullAttentionSpec,
        block_pool: BlockPool,
        enable_caching: bool,
        kv_cache_group_id: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ): ...
    sink_blocks: list[KVCacheBlock]  # Permanently allocated sink blocks
```

### Spec-to-Manager Mapping

```python
spec_manager_map: dict[type[KVCacheSpec], type[SingleTypeKVCacheManager]] = {
    FullAttentionSpec: FullAttentionManager,
    TQFullAttentionSpec: FullAttentionManager,
    MLAAttentionSpec: FullAttentionManager,
    SlidingWindowSpec: SlidingWindowManager,
    SlidingWindowMLASpec: SlidingWindowManager,
    ChunkedLocalAttentionSpec: ChunkedLocalAttentionManager,
    MambaSpec: MambaManager,
    CrossAttentionSpec: CrossAttentionManager,
    SinkFullAttentionSpec: SinkFullAttentionManager,
}
```

---

## 11. Block Tables

**File:** `vllm/v1/worker/block_table.py`

### BlockTable

```python
class BlockTable:
    def __init__(
        self,
        block_size: int,
        max_num_reqs: int,
        max_num_blocks_per_req: int,
        max_num_batched_tokens: int,
        pin_memory: bool,
        device: torch.device,
        kernel_block_size: int,
        cp_kv_cache_interleave_size: int,
    ): ...
```

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `block_size` | `int` | Kernel block size |
| `blocks_per_kv_block` | `int` | Ratio of allocation to kernel blocks |
| `use_hybrid_blocks` | `bool` | Whether using hybrid (split) blocks |
| `block_table` | `CpuGpuBuffer` | 2D block table tensor (int32) |
| `num_blocks_per_row` | `np.ndarray` | Per-request block counts (int32) |
| `slot_mapping` | `CpuGpuBuffer` | Slot mapping tensor (int64) |
| `pcp_world_size` | `int` | Prefill context parallel world size |
| `pcp_rank` | `int` | Prefill context parallel rank |
| `dcp_world_size` | `int` | Decode context parallel world size |
| `dcp_rank` | `int` | Decode context parallel rank |

#### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `append_row` | `(block_ids: list[int], row_idx: int) -> None` | Append blocks to a row |
| `add_row` | `(block_ids: list[int], row_idx: int) -> None` | Set blocks for a row |
| `clear_row` | `(row_idx: int) -> None` | Clear a row |
| `move_row` | `(src: int, tgt: int) -> None` | Move row data |
| `swap_row` | `(src: int, tgt: int) -> None` | Swap two rows |
| `compute_slot_mapping` | `(num_reqs, query_start_loc, positions) -> None` | Compute slot mapping via Triton kernel |
| `commit_block_table` | `(num_reqs: int) -> None` | Copy block table to GPU |
| `clear` | `() -> None` | Clear all data |
| `get_device_tensor` | `(num_reqs: int) -> torch.Tensor` | Get GPU block table |
| `get_cpu_tensor` | `() -> torch.Tensor` | Get CPU block table |
| `get_numpy_array` | `() -> np.ndarray` | Get numpy block table |

#### Hybrid Block Support

When `kernel_block_size != block_size`, blocks are split:

```python
@staticmethod
def map_to_kernel_blocks(
    kv_manager_block_ids: np.ndarray,
    blocks_per_kv_block: int,
    kernel_block_arange: np.ndarray,
) -> np.ndarray:
    # e.g., block_id 0 with blocks_per_kv_block=2 -> [0, 1]
    #       block_id 1 with blocks_per_kv_block=2 -> [2, 3]
```

### MultiGroupBlockTable

```python
class MultiGroupBlockTable:
    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        pin_memory: bool,
        device: torch.device,
        block_sizes: list[int],
        kernel_block_sizes: list[int],
        max_num_blocks: list[int] | None = None,
        cp_kv_cache_interleave_size: int = 1,
    ): ...
    block_tables: list[BlockTable]

    def append_row(self, block_ids: tuple[list[int], ...], row_idx: int) -> None: ...
    def add_row(self, block_ids: tuple[list[int], ...], row_idx: int) -> None: ...
    def clear_row(self, row_idx: int) -> None: ...
    def move_row(self, src: int, tgt: int) -> None: ...
    def swap_row(self, src: int, tgt: int) -> None: ...
    def compute_slot_mapping(self, num_reqs, query_start_loc, positions) -> None: ...
    def commit_block_table(self, num_reqs: int) -> None: ...
    def clear(self) -> None: ...
    def __getitem__(self, idx: int) -> BlockTable: ...
```

---

## 12. Prefix Caching

Prefix caching allows reusing KV cache blocks across requests that share common prompt
prefixes. The system computes block hashes based on token content and uses them for
cache lookup.

### Block Hash Types

```python
BlockHash = NewType("BlockHash", bytes)
BlockHashWithGroupId = NewType("BlockHashWithGroupId", bytes)
ExternalBlockHash: TypeAlias = bytes | int
```

### Hash Functions

```python
def make_block_hash_with_group_id(block_hash: BlockHash, group_id: int) -> BlockHashWithGroupId: ...
def get_block_hash(key: BlockHashWithGroupId) -> BlockHash: ...
def get_group_id(key: BlockHashWithGroupId) -> int: ...
def maybe_convert_block_hash(hash_bytes: BlockHash) -> ExternalBlockHash: ...
def hash_block_tokens(
    hash_function: Callable[[Any], bytes],
    parent_block_hash: BlockHash | None,
    curr_block_token_ids: Sequence[int],
    extra_keys: tuple[Any, ...] | None = None,
) -> BlockHash: ...
```

### Extra Keys for Hash

```python
def generate_block_hash_extra_keys(
    request: Request, start_token_idx: int, end_token_idx: int, start_mm_idx: int
) -> tuple[tuple[Any, ...] | None, int]:
```

Extra keys include:
- Multimodal input hashes (mm_hash, start_offset)
- LoRA names
- Cache salt (first block only)
- Prompt embedding hashes

### Block Hash List

```python
BlockHashList = list[BlockHash] | BlockHashListWithBlockSize
```

### BlockHashListWithBlockSize

Converts block hashes from `hash_block_size` granularity to `target_block_size`:

```python
class BlockHashListWithBlockSize:
    def __init__(
        self,
        block_hashes: list[BlockHash],
        hash_block_size: int,
        target_block_size: int,
    ): ...

    def __len__(self) -> int: ...
    def __getitem__(self, idx) -> BlockHash | list[BlockHash]: ...
    def __iter__(self) -> Iterator[BlockHash]: ...
```

---

## 13. KV Cache Utils

**File:** `vllm/v1/core/kv_cache_utils.py`

### Block Size Resolution

```python
def resolve_kv_cache_block_sizes(
    kv_cache_config: KVCacheConfig,
    vllm_config: VllmConfig,
) -> tuple[int, int]:
    """Returns (scheduler_block_size, hash_block_size)."""
```

### Memory Estimation

```python
def max_memory_usage_bytes(
    vllm_config: VllmConfig, kv_cache_specs: Iterable[KVCacheSpec]
) -> int: ...

def estimate_max_model_len(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
    available_memory: int,
) -> int: ...

def check_enough_kv_cache_memory(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
    available_memory: int,
) -> None:
```

### KV Cache Group Management

```python
def create_kv_cache_group_specs(
    kv_cache_spec: dict[str, KVCacheSpec],
    grouped_layer_names: list[list[str]]
) -> list[KVCacheGroupSpec]: ...

def is_kv_cache_spec_uniform(kv_cache_spec: dict[str, KVCacheSpec]) -> bool: ...
def get_max_concurrency_for_kv_cache_config(
    vllm_config, kv_cache_config
) -> float: ...
def may_override_num_blocks(vllm_config: VllmConfig, num_blocks: int) -> int: ...
def get_num_blocks(vllm_config, num_layers, available_memory, page_size) -> int: ...
def get_uniform_page_size(kv_cache_specs) -> int: ...
def unify_kv_cache_spec_page_size(kv_cache_spec) -> dict[str, KVCacheSpec]: ...
```

### KV Cache Configuration Generation

```python
def get_kv_cache_configs(
    vllm_config: VllmConfig,
    kv_cache_specs: list[dict[str, KVCacheSpec]],
    available_memory: list[int],
) -> list[KVCacheConfig]: ...

def get_kv_cache_config_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig: ...

def generate_scheduler_kv_cache_config(
    kv_cache_configs: list[KVCacheConfig],
) -> KVCacheConfig: ...
```

### Hybrid Model Support

```python
def unify_hybrid_kv_cache_specs(kv_cache_spec: dict[str, KVCacheSpec]) -> None: ...
def group_and_unify_kv_cache_specs(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[UniformTypeKVCacheSpecs] | None: ...
```

---

## 14. KV Cache Metrics

**File:** `vllm/v1/core/kv_cache_metrics.py`

### BlockMetricsState

```python
class BlockMetricsState:
    birth_time_ns: int
    last_access_ns: int
    access_history: deque[int]  # maxlen=4

    def record_access(self) -> None: ...
    def get_lifetime_seconds(self) -> float: ...
    def get_idle_time_seconds(self) -> float: ...
    def get_reuse_gaps_seconds(self) -> list[float]: ...
```

### KVCacheMetricsCollector

```python
class KVCacheMetricsCollector:
    def __init__(self, sample_rate: float = 0.01): ...
    sample_rate: float
    block_metrics: dict[int, BlockMetricsState]

    def should_sample_block(self) -> bool: ...
    def on_block_allocated(self, block: KVCacheBlock) -> None: ...
    def on_block_accessed(self, block: KVCacheBlock) -> None: ...
    def on_block_evicted(self, block: KVCacheBlock) -> None: ...
    def reset(self) -> None: ...
    def drain_events(self) -> list[KVCacheEvictionEvent]: ...
```

---

## 15. Encoder Cache Manager

**File:** `vllm/v1/core/encoder_cache_manager.py`

### EncoderCacheManager

```python
class EncoderCacheManager:
    def __init__(self, cache_size: int): ...
    cache_size: int
    num_free_slots: int
    num_freeable_slots: int
    cached: dict[str, set[str]]    # mm_hash -> request IDs
    freeable: OrderedDict[str, int]  # mm_hash -> num_encoder_embeds
    freed: list[str]                # Evicted mm_hashes
```

#### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `reset` | `() -> None` | Clear all cache state |
| `check_and_update_cache` | `(request, input_id) -> bool` | Check if encoder output is cached |
| `can_allocate` | `(request, input_id, encoder_compute_budget, num_embeds_to_schedule) -> bool` | Check allocation capacity |
| `allocate` | `(request, input_id) -> None` | Reserve cache space |
| `get_cached_input_ids` | `(request) -> set[int]` | Get cached input IDs |
| `free_encoder_input` | `(request, input_id) -> None` | Free one input reference |
| `free` | `(request) -> None` | Free all request references |
| `get_freed_mm_hashes` | `() -> list[str]` | Get recently evicted hashes |

### compute_mm_encoder_budget

```python
def compute_mm_encoder_budget(
    scheduler_config: SchedulerConfig,
    mm_max_toks_per_item: Mapping[str, int],
) -> tuple[int, int]:
    """Returns (encoder_compute_budget, encoder_cache_size)."""
```

### EncoderDecoderCacheManager

```python
class EncoderDecoderCacheManager(EncoderCacheManager):
    allocated: list[str]
    to_free: list[str]
```

---

## 16. KV Cache Offloading

### Simple KV Offload

**Directory:** `vllm/v1/simple_kv_offload/`

| File | Description |
|------|-------------|
| `manager.py` | Simple KV offload manager |
| `worker.py` | Worker-side offload logic |
| `copy_backend.py` | Copy backend for offloading |
| `cuda_mem_ops.py` | CUDA memory operations |
| `metadata.py` | Offload metadata |

### Advanced KV Offload

**Directory:** `vllm/v1/kv_offload/`

| File | Description |
|------|-------------|
| `base.py` | Base offload interface |
| `factory.py` | Offload factory |
| `reuse_manager.py` | Block reuse management |
| `cpu/common.py` | CPU offload common |
| `cpu/gpu_worker.py` | GPU worker for CPU offload |
| `cpu/manager.py` | CPU offload manager |
| `cpu/shared_offload_region.py` | Shared offload region |
| `cpu/spec.py` | CPU offload spec |
| `cpu/policies/base.py` | Eviction policy base |
| `cpu/policies/lru.py` | LRU eviction policy |
| `cpu/policies/arc.py` | ARC eviction policy |
| `worker/worker.py` | Offload worker |

---

## 17. MLA (Multi-head Latent Attention)

### MLA Architecture

MLA compresses KV cache into a latent representation, reducing memory usage while
maintaining attention quality. It uses two computation paths:

1. **MHA (Multi-Head Attention) / Prefill Path** (`forward_mha`): Compute-friendly,
   decompresses latent KV into full K/V heads for standard MHA.

2. **MQA (Multi-Query Attention) / Decode Path** (`forward_mqa`): Data-movement
   friendly, operates directly on compressed latent representations.

### MLA Dimensions

| Symbol | Description | DeepSeek V3 Example |
|--------|-------------|-------------------|
| H | Hidden size | 7168 |
| Lq | Q latent dimension | 1536 |
| Lkv | KV latent dimension | 512 |
| P | NoPE dimension (no rope) | 128 |
| R | RoPE dimension | 64 |
| V | V head dimension | 128 |
| N | Number of attention heads | 128 |

### MLA Prefill Backend Registry

**File:** `vllm/v1/attention/backends/mla/prefill/`

| File | Description |
|------|-------------|
| `registry.py` | Prefill backend registry |
| `selector.py` | Backend selection logic |
| `base.py` | Base prefill implementation |
| `flash_attn.py` | FlashAttention prefill |
| `flashinfer.py` | FlashInfer prefill |
| `trtllm_ragged.py` | TRT-LLM ragged prefill |

### MLA Operations

**File:** `vllm/v1/attention/ops/flashmla.py`

FlashMLA operation wrappers for MLA attention computation.

### MLA Indexer

**File:** `vllm/v1/attention/backends/mla/indexer.py`

Indexer for sparse MLA attention, used in DeepSeek V4 with sliding window MLA.

### MLA Sparse Utils

**File:** `vllm/v1/attention/backends/mla/sparse_utils.py`

Utilities for sparse MLA attention computation.

### MLA Compressor Utils

**File:** `vllm/v1/attention/backends/mla/compressor_utils.py`

Utilities for KV cache compression in MLA models.

---

## 18. Attention Layer Implementations

**Directory:** `vllm/model_executor/layers/attention/`

### Attention (Standard)

**File:** `attention.py`

```python
class Attention(nn.Module, AttentionLayerBase):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        use_alibi_sqrt: bool | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        logits_soft_cap: float | None = None,
        per_layer_sliding_window: int | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        attn_backend: type[AttentionBackend] | None = None,
        head_size_v: int | None = None,
        **extra_impl_args,
    ) -> None: ...
```

Key attributes:
- `kv_cache_dtype`: KV cache data type string
- `kv_cache_torch_dtype`: Actual torch dtype for KV cache
- `calculate_kv_scales`: Whether to compute dynamic scales
- `attn_backend`: Selected attention backend class
- `impl`: Backend-specific attention implementation
- `sliding_window`: Sliding window size (if any)
- `num_heads`, `head_size`, `head_size_v`, `num_kv_heads`
- `query_quant`: Optional query quantization

#### Forward Method

```python
def forward(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output_shape: torch.Size | None = None,
) -> torch.Tensor: ...
```

The forward method:
1. Optionally calculates KV scales (for FP8 dynamic quantization)
2. Optionally quantizes queries
3. Reshapes Q/K/V to 3D (tokens, heads, head_dim)
4. Calls `unified_kv_cache_update` to store KV in cache
5. Calls `unified_attention_with_output` for attention computation

#### get_kv_cache_spec

```python
def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
    # Returns SlidingWindowSpec, FullAttentionSpec, TQFullAttentionSpec
    # based on configuration
```

### MLAAttention

**File:** `mla_attention.py`

Implements MLA attention for DeepSeek V2/V3/V4 models. Supports both MHA (prefill)
and MQA (decode) computation paths.

### CrossAttention

**File:** `cross_attention.py`

Implements cross-attention for encoder-decoder models (e.g., Whisper).

### EncoderOnlyAttention

**File:** `encoder_only_attention.py`

Attention for encoder-only models that do not use KV cache.

### StaticSinkAttention

**File:** `static_sink_attention.py`

Attention with static sink tokens that are always retained in KV cache.

### ChunkedLocalAttention

**File:** `chunked_local_attention.py`

Attention with chunked local windows (e.g., LLaMA 4).

### MmEncoderAttention

**File:** `mm_encoder_attention.py`

Multimodal encoder attention for vision-language models.

### KV Transfer Utils

**File:** `kv_transfer_utils.py`

```python
def maybe_transfer_kv_layer(func):
    """Decorator that handles KV transfer for disaggregated prefill."""
```

---

## 19. Attention Operations

**Directory:** `vllm/v1/attention/ops/`

| File | Description |
|------|-------------|
| `common.py` | Common attention ops (context parallel LSE aggregation) |
| `paged_attn.py` | Paged attention operation |
| `prefix_prefill.py` | Prefix prefill attention |
| `merge_attn_states.py` | Merge attention states (chunked prefill) |
| `flashmla.py` | FlashMLA operations |
| `chunked_prefill_paged_decode.py` | Chunked prefill with paged decode |
| `triton_prefill_attention.py` | Triton prefill attention kernel |
| `triton_decode_attention.py` | Triton decode attention kernel |
| `triton_unified_attention.py` | Triton unified attention |
| `triton_reshape_and_cache_flash.py` | Triton reshape and cache |
| `triton_attention_helpers.py` | Triton attention helper functions |
| `triton_merge_attn_states.py` | Triton merge attention states |
| `triton_turboquant_decode.py` | TurboQuant decode kernel |
| `triton_turboquant_store.py` | TurboQuant store kernel |
| `vit_attn_wrappers.py` | ViT attention wrappers |
| `dcp_alltoall.py` | Decode context parallel all-to-all |
| `xpu_mla_sparse.py` | XPU MLA sparse operations |
| `rocm_aiter_mla_sparse.py` | ROCm Aiter MLA sparse |

### DeepSeek V4 Ops

**Directory:** `vllm/v1/attention/ops/deepseek_v4_ops/`

| File | Description |
|------|-------------|
| `cache_utils.py` | Cache utilities |
| `fused_compress_quant_cache.py` | Fused compress and quantize cache |
| `fused_indexer_q.py` | Fused indexer for Q |
| `fused_inv_rope_fp8_quant.py` | Fused inverse RoPE with FP8 quant |
| `fused_qk_rmsnorm.py` | Fused QK RMS normalization |

---

## 20. Sliding Window Attention

Sliding window attention limits the attention window to a fixed number of recent tokens,
reducing both memory and compute.

### Management

- `SlidingWindowManager` handles block allocation with window-aware eviction
- `SlidingWindowSpec` defines the window size and block layout
- Blocks outside the window are freed and replaced with null blocks
- `get_num_skipped_tokens` returns `max(0, num_computed - sliding_window + 1)`

### Cache Hit Logic

For sliding window, cache hits require contiguous blocks within the window:
- Scans right-to-left for contiguous matching blocks
- Needs `cdiv(sliding_window - 1, block_size)` contiguous blocks
- Prefix blocks outside window are set to null

### Chunked Local Attention

`ChunkedLocalAttentionSpec` implements attention within fixed-size chunks:
- `attention_chunk_size` defines the chunk boundary
- Tokens before the current chunk are skipped
- `get_num_skipped_tokens` returns `(computed // chunk_size) * chunk_size`

---

## 21. Cross-Attention for Encoder-Decoder

### Cross-Attention KV Cache

- Uses `CrossAttentionSpec` and `CrossAttentionManager`
- No prefix caching (encoder outputs are request-specific)
- Static allocation based on encoder input length
- Separate block table from decoder attention

### Encoder Cache

- `EncoderCacheManager` manages encoder output caching
- Supports sharing encoder outputs across requests (via mm_hash)
- LRU eviction when cache is full
- Budget-based allocation with `compute_mm_encoder_budget()`

---

## 22. Attention Selector

**File:** `vllm/v1/attention/selector.py`

### AttentionSelectorConfig

```python
class AttentionSelectorConfig(NamedTuple):
    head_size: int
    dtype: torch.dtype
    kv_cache_dtype: CacheDType | None
    block_size: int | None
    use_mla: bool = False
    has_sink: bool = False
    use_sparse: bool = False
    use_mm_prefix: bool = False
    use_per_head_quant_scales: bool = False
    attn_type: str = AttentionType.DECODER
    use_non_causal: bool = False
    use_batch_invariant: bool = False
```

### get_attn_backend

```python
def get_attn_backend(
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str | None,
    use_mla: bool = False,
    has_sink: bool = False,
    use_sparse: bool = False,
    use_mm_prefix: bool = False,
    use_per_head_quant_scales: bool = False,
    attn_type: str | None = None,
    num_heads: int | None = None,
) -> type[AttentionBackend]:
```

Selects the appropriate attention backend based on model configuration, hardware
capabilities, and runtime parameters. The selection is cached for efficiency.

### get_mamba_attn_backend

```python
def get_mamba_attn_backend(
    mamba_type: str,
) -> type[AttentionBackend]:
```

Selects the Mamba/SSM attention backend by type string. Valid types:
`"mamba1"`, `"mamba2"`, `"short_conv"`, `"linear_attention"`, `"gdn_attention"`, `"custom"`.

---

## 23. KV Cache Layout

**File:** `vllm/v1/attention/backends/utils.py`

### KVCacheLayoutType

```python
KVCacheLayoutType = Literal["NHD", "HND"]
```

- **NHD**: Shape `(2, num_blocks, block_size, num_kv_heads, head_size)` - Block-size
  and head dimensions are contiguous
- **HND**: Shape `(2, num_blocks, num_kv_heads, block_size, head_size)` - Head
  and block-size dimensions are contiguous

### Layout Functions

```python
PAD_SLOT_ID = -1
NULL_BLOCK_ID = 0

def get_kv_cache_layout() -> KVCacheLayoutType: ...
def set_kv_cache_layout(cache_layout: KVCacheLayoutType | None) -> None: ...
def is_valid_kv_cache_layout(value: str) -> bool: ...
```

### PerLayerParameters

```python
@dataclass
class PerLayerParameters:
    window_left: int
    logits_soft_cap: float | None
    sm_scale: float
    has_sinks: bool = False
    has_same_window_lefts: bool | None = None
    has_same_all_params: bool | None = None
```

### Utility Functions

```python
def get_per_layer_parameters(
    vllm_config: VllmConfig,
    layer_names: list[str],
    cls_: type[AttentionImpl],
) -> dict[str, PerLayerParameters]: ...

def infer_global_hyperparameters(
    per_layer_params: dict[str, PerLayerParameters],
) -> PerLayerParameters: ...

def split_decodes_and_prefills(
    common_attn_metadata: CommonAttentionMetadata,
    decode_threshold: int,
) -> tuple[np.ndarray, np.ndarray]: ...

def get_dcp_local_seq_lens(
    seq_lens: torch.Tensor,
    dcp_world_size: int,
    dcp_rank: int,
) -> torch.Tensor: ...
```

---

## PagedAttention Algorithm Summary

vLLM's PagedAttention algorithm divides the KV cache into fixed-size blocks, similar to
virtual memory paging:

1. **Block Allocation**: KV cache is partitioned into blocks of `block_size` tokens each.
   Each block has a unique `block_id`.

2. **Block Tables**: Each request maintains a block table mapping virtual block indices
   to physical block IDs. This is stored as a 2D tensor `[num_requests, max_blocks_per_req]`.

3. **Slot Mapping**: A flat tensor maps each token position to its physical slot in the
   KV cache: `slot = block_id * block_size + offset_within_block`.

4. **Prefix Caching**: Full blocks are hashed based on their token content (and
   multimodal/LoRA metadata). Requests sharing prefixes can reuse cached blocks by
   matching block hashes.

5. **Reference Counting**: Each block tracks a `ref_cnt`. Blocks with `ref_cnt > 0`
   cannot be evicted. When `ref_cnt` drops to 0, the block enters the free queue.

6. **Eviction**: When the free queue is exhausted, cached blocks (those with a hash)
   are evicted in LRU order. Their hash is removed and they become available for reuse.

7. **Sliding Window**: For sliding window attention, blocks outside the window are
   freed and replaced with null blocks. Only the recent `sliding_window` tokens
   retain their KV cache.

8. **Hybrid Models**: Models with multiple attention types (e.g., full attention +
   sliding window) use separate block tables for each type, coordinated by
   `HybridKVCacheCoordinator`.

9. **Context Parallelism**: With decode context parallelism (DCP) or prefill context
   parallelism (PCP), the effective block size is multiplied by the world size, and
   slot mapping accounts for interleave patterns.
