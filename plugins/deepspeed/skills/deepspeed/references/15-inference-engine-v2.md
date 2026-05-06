# DeepSpeed Inference Engine V2 Reference

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [State Management](#state-management)
4. [Blocked KV Cache](#blocked-kv-cache)
5. [Ragged Batching](#ragged-batching)
6. [Configuration](#configuration)
7. [Model Implementations](#model-implementations)
8. [Module System](#module-system)
9. [Kernel Operations](#kernel-operations)
10. [Configuration Examples by Model](#configuration-examples-by-model)
11. [Performance Tuning](#performance-tuning)
12. [Code Examples](#code-examples)
13. [Troubleshooting](#troubleshooting)

---

## Overview

DeepSpeed Inference Engine V2 is a next-generation inference framework built from the ground up for high-throughput, low-latency serving of large language models. It introduces several fundamental advances over V1:

- **Ragged batching**: Native support for variable-length sequences without padding, eliminating wasted computation on pad tokens
- **Blocked KV cache**: Paged KV cache management using 6D tensor storage with linked-list block allocation, preventing memory fragmentation
- **Modular architecture**: Clean separation between model policies, module interfaces, and kernel implementations
- **Advanced attention**: Dense blocked attention with flash-style computation, rotary positional embeddings, and multi-query/grouped-query attention support
- **Quantized inference**: Built-in support for INT8/INT4 weight quantization with specialized GEMM kernels
- **MoE inference**: Full support for Mixture of Experts models with expert sharding

V2 is located in `deepspeed/inference/v2/` and is separate from the V1 engine in `deepspeed/inference/engine.py`.

---

## Architecture

### Directory Structure

```
deepspeed/inference/v2/
  __init__.py
  engine.py                    # InferenceEngineV2 main class
  config.py                    # RaggedInferenceEngineConfig
  state_manager.py             # DSStateManager: KV cache and sequence tracking
  blocked_kvcache.py           # BlockedKVCache: paged KV cache
  blocked_allocator.py         # BlockedAllocator: linked-list block manager
  ragged_batch.py              # Ragged batch data structures

  model_implementations/       # Model-specific policies and containers
    __init__.py
    inference_model_base.py    # DSInferenceModelBase
    transformer_base.py        # DSTransformerModelBase
    moe_base.py                # DSMOETransformerModelBase
    llama_v2/                  # LLaMA 2 implementation
      __init__.py
      policy.py               # Llama2Policy
      container.py            # Llama2TransformerContainer
    mistral/                   # Mistral implementation
    mixtral/                   # Mixtral MoE implementation
    falcon/                    # Falcon implementation
    phi/                       # Phi implementation
    phi3/                      # Phi-3 implementation
    opt/                       # OPT implementation
    qwen/                      # Qwen implementation
    qwen_v2/                   # Qwen2 implementation
    qwen_v2_moe/              # Qwen2-MoE implementation
    exaone4/                   # Exaone4 implementation
    sharding/                  # Parameter sharding for TP

  modules/                     # Reusable module interfaces and implementations
    __init__.py
    interfaces/               # Abstract interfaces
      __init__.py
      attention_base.py       # DSSelfAttentionBase
      linear_base.py          # DSLinearBase
      embedding_base.py       # DSEmbeddingBase
      moe_base.py             # DSMOEBase
      norm_base.py            # DSNormBase
    implementations/          # Concrete implementations
      __init__.py
      dense_blocked_attention.py   # DenseBlockedAttention
      blas_linear.py               # BLASLinear
      quantized_linear.py          # QuantizedLinear
      ragged_embedding.py          # RaggedEmbedding
      ragged_unembed.py            # RaggedUnembed
      attention/                   # Attention variants
      embedding/                   # Embedding variants
      linear/                      # Linear variants
      moe/                         # MoE variants
      post_norm/                   # Post-normalization
      pre_norm/                    # Pre-normalization
      unembed/                     # Unembedding layers

  kernels/                     # Low-level kernel implementations
    __init__.py
    core_ops/                 # Core CUDA operations
      bias_activations.py     # Fused bias + activation
      blas_kernels.py         # BLAS GEMM wrappers
      cuda_layer_norm.py      # CUDA Layer normalization
      cuda_linear.py          # CUDA linear operations
      cuda_rms_norm.py        # CUDA RMS normalization
      gated_activations.py    # SiLU, GeGLU activations
    cutlass_ops/              # CUTLASS-based operations
      mixed_gemm.py           # Mixed-precision GEMM
      moe_gemm.py             # MoE batched GEMM
      shared_resources.py     # Shared CUTLASS resources
    ragged_ops/               # Ragged tensor operations
      atom_builder.py         # Atom (micro-batch) builder
      blocked_flash.py        # Blocked flash attention
      embed.py                # Ragged embedding kernel
      linear_blocked_kv_rotary.py  # Linear + blocked KV + rotary
      logits_gather.py        # Logits gathering
      moe_gather.py           # MoE output gathering
      moe_scatter.py          # MoE input scattering
      ragged_helpers.py       # Utility helpers
      top_k_gating.py         # Top-k gating kernel
```

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Inference Engine V2                             │
│                                                                      │
│  ┌─────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│  │   Ragged Batch   │  │  DSStateManager  │  │  Model Policy    │   │
│  │   (variable len) │──│  (KV cache, seq  │──│  (per-model      │   │
│  │                  │  │   tracking)      │  │   logic)         │   │
│  └────────┬────────┘  └────────┬─────────┘  └────────┬─────────┘   │
│           │                    │                      │              │
│           └────────────────────┼──────────────────────┘              │
│                                │                                      │
│                    ┌───────────▼───────────┐                          │
│                    │  Module System         │                          │
│                    │  ┌──────┐ ┌──────┐    │                          │
│                    │  │ Attn │ │Linear│    │                          │
│                    │  └──────┘ └──────┘    │                          │
│                    │  ┌──────┐ ┌──────┐    │                          │
│                    │  │Embed │ │ MoE  │    │                          │
│                    │  └──────┘ └──────┘    │                          │
│                    │  ┌──────┐ ┌──────┐    │                          │
│                    │  │Norm  │ │Unemb │    │                          │
│                    │  └──────┘ └──────┘    │                          │
│                    └───────────┬───────────┘                          │
│                                │                                      │
│                    ┌───────────▼───────────┐                          │
│                    │  Kernel Operations     │                          │
│                    │  ┌──────────────────┐  │                          │
│                    │  │ core_ops         │  │                          │
│                    │  │ cutlass_ops      │  │                          │
│                    │  │ ragged_ops       │  │                          │
│                    │  └──────────────────┘  │                          │
│                    └───────────────────────┘                          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## State Management

### DSStateManager

The `DSStateManager` manages the KV cache allocation, sequence descriptor tracking, and block lifecycle for the inference engine. It is the central coordinator for the blocked KV cache.

```python
class DSStateManager:
    """Manages inference state: KV cache, sequence tracking, block allocation.

    Responsibilities:
    - Track active sequences and their metadata
    - Allocate and free KV cache blocks
    - Manage multiple KV cache groups (e.g., local + global attention)
    - Coordinate block allocation across attention layers

    Attributes:
        free_blocks: Set of available block IDs
        n_tracked_sequences: Number of currently active sequences
        n_kv_cache_groups: Number of KV cache groups
    """

    def __init__(
        self,
        max_sequences: int = 256,
        max_seq_length: int = 4096,
        block_size: int = 16,
        n_layers: int = 32,
        n_heads: int = 32,
        head_dim: int = 128,
        n_kv_heads: Optional[int] = None,
        dtype: torch.dtype = torch.float16,
        n_kv_cache_groups: int = 1,
    ):
        self.max_sequences = max_sequences
        self.max_seq_length = max_seq_length
        self.block_size = block_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads or n_heads
        self.dtype = dtype
        self.n_kv_cache_groups = n_kv_cache_groups

        # Compute total number of blocks needed
        self.max_blocks_per_seq = math.ceil(max_seq_length / block_size)
        total_blocks = max_sequences * self.max_blocks_per_seq

        # Initialize blocked KV cache
        self.kv_cache = BlockedKVCache(
            total_blocks=total_blocks,
            block_size=block_size,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=self.n_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            n_kv_cache_groups=n_kv_cache_groups,
        )

        # Block allocator
        self.allocator = BlockedAllocator(total_blocks)

        # Sequence tracking
        self.n_tracked_sequences = 0
        self.sequence_descriptors = {}

    def allocate_sequence(self, seq_id: int) -> "SequenceDescriptor":
        """Allocate resources for a new sequence.

        Args:
            seq_id: Unique sequence identifier.

        Returns:
            SequenceDescriptor with allocated block lists.
        """
        descriptor = SequenceDescriptor(
            seq_id=seq_id,
            block_size=self.block_size,
            max_blocks=self.max_blocks_per_seq,
        )
        self.sequence_descriptors[seq_id] = descriptor
        self.n_tracked_sequences += 1
        return descriptor

    def free_sequence(self, seq_id: int):
        """Free all resources for a completed sequence.

        Args:
            seq_id: Sequence to free.
        """
        descriptor = self.sequence_descriptors.pop(seq_id)
        self.allocator.free_blocks(descriptor.all_blocks)
        self.n_tracked_sequences -= 1

    def allocate_block(self, seq_id: int) -> int:
        """Allocate a new KV cache block for a sequence.

        Args:
            seq_id: The sequence needing more KV cache space.

        Returns:
            Block ID of the newly allocated block.
        """
        block_id = self.allocator.allocate_block()
        self.sequence_descriptors[seq_id].add_block(block_id)
        return block_id
```

### SequenceDescriptor

```python
class SequenceDescriptor:
    """Tracks the state of a single inference sequence.

    Attributes:
        seq_id: Unique sequence identifier.
        block_size: Number of tokens per KV cache block.
        prompt_tokens: Number of tokens in the initial prompt.
        generated_tokens: Number of tokens generated so far.
        all_blocks: List of allocated block IDs.
        is_prompt: Whether the sequence is in prompt processing phase.
    """

    def __init__(self, seq_id: int, block_size: int, max_blocks: int):
        self.seq_id = seq_id
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.prompt_tokens = 0
        self.generated_tokens = 0
        self.all_blocks = []

    @property
    def total_tokens(self):
        return self.prompt_tokens + self.generated_tokens

    @property
    def is_prompt(self):
        return self.generated_tokens == 0

    def add_block(self, block_id: int):
        self.all_blocks.append(block_id)
```

---

## Blocked KV Cache

### BlockedKVCache

The `BlockedKVCache` implements paged KV cache storage using a 6D tensor, avoiding memory fragmentation by allocating fixed-size blocks.

```python
class BlockedKVCache:
    """Paged KV cache using 6D tensor storage.

    Storage layout: [n_blocks, 2 (K+V), block_size, n_kv_heads, head_dim, n_kv_cache_groups]

    The 6D tensor allows for:
    - Efficient block-granularity allocation and deallocation
    - Support for multiple cache groups (local + global attention)
    - Direct indexing by block ID without scatter/gather overhead

    Args:
        total_blocks: Total number of KV cache blocks.
        block_size: Number of tokens per block (e.g., 16).
        n_layers: Number of transformer layers.
        n_heads: Number of attention heads.
        n_kv_heads: Number of key-value heads (for GQA/MQA).
        head_dim: Dimension per head.
        dtype: Data type for cache storage.
        n_kv_cache_groups: Number of KV cache groups.
    """

    def __init__(
        self,
        total_blocks: int,
        block_size: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        n_kv_cache_groups: int = 1,
    ):
        self.total_blocks = total_blocks
        self.block_size = block_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.n_kv_cache_groups = n_kv_cache_groups

        # Allocate 6D cache tensor
        # Shape: [total_blocks, 2, block_size, n_kv_heads, head_dim, n_kv_cache_groups]
        self.data = torch.zeros(
            total_blocks, 2, block_size, n_kv_heads, head_dim, n_kv_cache_groups,
            device=torch.cuda.current_device(),
            dtype=dtype,
        )

    def get_k_cache(self, layer_idx: int, block_ids: torch.Tensor) -> torch.Tensor:
        """Get K cache for specific blocks.

        Args:
            layer_idx: Transformer layer index.
            block_ids: Block IDs to retrieve [num_blocks].

        Returns:
            K cache tensor [num_blocks, block_size, n_kv_heads, head_dim].
        """
        return self.data[block_ids, 0]  # Index 0 = K

    def get_v_cache(self, layer_idx: int, block_ids: torch.Tensor) -> torch.Tensor:
        """Get V cache for specific blocks."""
        return self.data[block_ids, 1]  # Index 1 = V

    def update_cache(
        self,
        layer_idx: int,
        block_ids: torch.Tensor,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        positions: torch.Tensor,
    ):
        """Update KV cache with new key-value pairs.

        Args:
            layer_idx: Layer to update.
            block_ids: Target block IDs.
            k_new: New K values [num_tokens, n_kv_heads, head_dim].
            v_new: New V values [num_tokens, n_kv_heads, head_dim].
            positions: Token positions within blocks [num_tokens].
        """
        # Scatter new values into the correct positions
        self.data[block_ids, 0, positions] = k_new
        self.data[block_ids, 1, positions] = v_new
```

### 6D Tensor Layout

```
Dimension breakdown of the 6D KV cache tensor:

  [total_blocks,   2,   block_size,   n_kv_heads,   head_dim,   n_kv_cache_groups]
       |           |         |              |            |              |
       v           v         v              v            v              v
   Block ID    K or V   Token position   KV head    Head dim    Cache group
   (0..N-1)   (0,1)    (0..bs-1)      (0..Hkv-1) (0..D-1)   (0..G-1)

Example with:
  total_blocks=1024, block_size=16, n_kv_heads=8, head_dim=128, n_kv_cache_groups=2
  Total memory: 1024 * 2 * 16 * 8 * 128 * 2 * 2 bytes = 1 GB (bf16)
```

### BlockedAllocator

```python
class BlockedAllocator:
    """Linked-list based block allocator for KV cache management.

    Manages a pool of fixed-size blocks using a free list.
    Blocks are allocated sequentially and freed individually.

    Args:
        total_blocks: Total number of blocks to manage.
    """

    def __init__(self, total_blocks: int):
        self.total_blocks = total_blocks
        self.free_blocks = set(range(total_blocks))
        self.allocated_blocks = set()

    def allocate_block(self) -> int:
        """Allocate a free block.

        Returns:
            Block ID of the allocated block.

        Raises:
            RuntimeError: If no blocks are available.
        """
        if not self.free_blocks:
            raise RuntimeError(
                f"KV cache out of memory: all {self.total_blocks} blocks allocated. "
                f"Consider increasing max_sequences or decreasing max_seq_length."
            )
        block_id = self.free_blocks.pop()
        self.allocated_blocks.add(block_id)
        return block_id

    def free_blocks(self, block_ids: List[int]):
        """Free a list of blocks.

        Args:
            block_ids: Block IDs to release.
        """
        for block_id in block_ids:
            if block_id in self.allocated_blocks:
                self.allocated_blocks.discard(block_id)
                self.free_blocks.add(block_id)

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    @property
    def num_allocated_blocks(self) -> int:
        return len(self.allocated_blocks)
```

---

## Ragged Batching

Ragged batching enables efficient batching of sequences with different lengths without padding, eliminating wasted compute on pad tokens.

### Problem: Padded vs Ragged Batching

```
Padded Batching (V1):
  Sequence 1: [tok, tok, tok, tok, tok, PAD, PAD, PAD]   (3/8 wasted)
  Sequence 2: [tok, tok, tok, tok, tok, tok, tok, tok]   (0/8 wasted)
  Sequence 3: [tok, tok, PAD, PAD, PAD, PAD, PAD, PAD]   (6/8 wasted)
  Total waste: 9/24 = 37.5%

Ragged Batching (V2):
  All tokens: [tok, tok, tok, tok, tok, tok, tok, tok, tok, tok, tok, tok, tok, tok, tok]
              |--- Seq 1 (5) ---|--------- Seq 2 (8) ---------|-- Seq 3 (2) --|
  Total waste: 0%
```

### Ragged Batch Data Structure

```python
class RaggedBatch:
    """Represents a batch of variable-length sequences.

    Instead of a padded 2D tensor [batch, max_seq_len], sequences are
    stored as a flat 1D tensor with metadata to reconstruct individual
    sequence boundaries.

    Attributes:
        tokens: Flat tensor of all tokens [total_tokens].
        seq_ids: Sequence ID for each token [total_tokens].
        seq_start: Start index of each sequence [batch_size + 1].
        seq_lengths: Length of each sequence [batch_size].
        positions: Position within each sequence for each token [total_tokens].
    """

    def __init__(
        self,
        tokens: torch.Tensor,
        seq_ids: List[int],
        seq_lengths: List[int],
    ):
        total_tokens = tokens.numel()
        self.tokens = tokens.flatten()

        # Compute sequence boundaries
        self.seq_lengths = torch.tensor(seq_lengths, device=tokens.device)
        self.seq_start = torch.zeros(len(seq_ids) + 1, dtype=torch.int32, device=tokens.device)
        self.seq_start[1:] = torch.cumsum(self.seq_lengths, dim=0)

        # Compute positions
        self.positions = torch.arange(total_tokens, device=tokens.device)
        for i, length in enumerate(seq_lengths):
            start = self.seq_start[i]
            self.positions[start:start + length] = torch.arange(length, device=tokens.device)

        # Sequence IDs
        self.seq_ids = seq_ids

    @property
    def batch_size(self):
        return len(self.seq_ids)

    @property
    def total_tokens(self):
        return self.tokens.numel()
```

---

## Configuration

### RaggedInferenceEngineConfig

```python
@dataclass
class RaggedInferenceEngineConfig:
    """Configuration for Inference Engine V2.

    Args:
        model_name: Name/path of the model.
        dtype: Data type for inference (torch.float16 or torch.bfloat16).
        tensor_parallel: Tensor parallel configuration.
        max_batch_size: Maximum number of concurrent sequences.
        max_seq_length: Maximum sequence length (prompt + generation).
        max_new_tokens: Maximum tokens to generate per sequence.
        block_size: KV cache block size in tokens.
        enable_quantization: Whether to use weight quantization.
        quant_config: Quantization configuration.
        max_ragged_batch_size: Maximum tokens in a single ragged batch.
    """

    # Model settings
    model_name: Optional[str] = None
    dtype: torch.dtype = torch.float16

    # Tensor parallelism
    tensor_parallel: Optional[Dict] = None
    tp_size: int = 1

    # Batch settings
    max_batch_size: int = 32
    max_seq_length: int = 4096
    max_new_tokens: int = 1024

    # KV cache
    block_size: int = 16

    # Quantization
    enable_quantization: bool = False
    quant_config: Optional[Dict] = None

    # Ragged batching
    max_ragged_batch_size: int = 4096
```

### Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | `str` | `None` | Model name or path (HuggingFace format) |
| `dtype` | `torch.dtype` | `torch.float16` | Inference data type |
| `tensor_parallel` | `Dict` | `None` | TP configuration |
| `tp_size` | `int` | `1` | Tensor parallel degree |
| `max_batch_size` | `int` | `32` | Maximum concurrent sequences |
| `max_seq_length` | `int` | `4096` | Maximum total sequence length |
| `max_new_tokens` | `int` | `1024` | Maximum generated tokens per sequence |
| `block_size` | `int` | `16` | KV cache block size (tokens per block) |
| `enable_quantization` | `bool` | `False` | Enable weight quantization |
| `quant_config` | `Dict` | `None` | Quantization parameters |
| `max_ragged_batch_size` | `int` | `4096` | Maximum tokens per ragged batch |

---

## Model Implementations

### Base Classes

#### DSInferenceModelBase

```python
class DSInferenceModelBase(nn.Module):
    """Base class for all V2 inference model implementations.

    Provides common functionality for:
    - Model loading and weight initialization
    - Tensor parallel sharding
    - State manager integration
    - Forward pass orchestration
    """

    def __init__(self, config: RaggedInferenceEngineConfig):
        super().__init__()
        self.config = config
        self.state_manager: Optional[DSStateManager] = None

    def initialize_state_manager(self, model_config):
        """Create and initialize the DSStateManager based on model config."""
        self.state_manager = DSStateManager(
            max_sequences=self.config.max_batch_size,
            max_seq_length=self.config.max_seq_length,
            block_size=self.config.block_size,
            n_layers=model_config.num_hidden_layers,
            n_heads=model_config.num_attention_heads,
            head_dim=model_config.hidden_size // model_config.num_attention_heads,
            n_kv_heads=getattr(model_config, "num_key_value_heads", None),
            dtype=self.config.dtype,
        )

    def forward(self, batch: RaggedBatch) -> torch.Tensor:
        """Process a ragged batch through the model.

        Args:
            batch: Ragged batch of sequences.

        Returns:
            Logits tensor [total_tokens, vocab_size].
        """
        raise NotImplementedError
```

#### DSTransformerModelBase

```python
class DSTransformerModelBase(DSInferenceModelBase):
    """Base class for transformer-based inference models.

    Implements the common transformer forward pass:
    1. Embedding lookup
    2. Stack of transformer layers
    3. Final normalization
    4. LM head (unembedding)
    """

    def __init__(self, config: RaggedInferenceEngineConfig):
        super().__init__(config)
        self.embed_tokens: Optional[DSEmbeddingBase] = None
        self.layers: nn.ModuleList = nn.ModuleList()
        self.norm: Optional[DSNormBase] = None
        self.lm_head: Optional[DSLinearBase] = None

    def forward(self, batch: RaggedBatch) -> torch.Tensor:
        # Embedding
        hidden_states = self.embed_tokens(batch.tokens)

        # Transformer layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                batch=batch,
                kv_cache=self.state_manager.kv_cache,
            )

        # Final norm + LM head
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits
```

#### DSMOETransformerModelBase

```python
class DSMOETransformerModelBase(DSTransformerModelBase):
    """Base class for MoE transformer models.

    Extends DSTransformerModelBase with MoE-specific functionality:
    - Expert routing and dispatch
    - Expert parallel communication
    - Load balancing tracking
    """

    def __init__(self, config: RaggedInferenceEngineConfig):
        super().__init__(config)
        self.expert_parallel_size = config.tp_size  # EP uses same GPUs as TP
        self.num_experts = 0  # Set by subclass
        self.top_k = 2

    def forward(self, batch: RaggedBatch) -> torch.Tensor:
        hidden_states = self.embed_tokens(batch.tokens)

        for layer in self.layers:
            if hasattr(layer, 'is_moe') and layer.is_moe:
                # MoE layer: route tokens to experts
                hidden_states = self._moe_forward(layer, hidden_states, batch)
            else:
                hidden_states = layer(hidden_states, batch=batch,
                                      kv_cache=self.state_manager.kv_cache)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

    def _moe_forward(self, layer, hidden_states, batch):
        """Process MoE layer with expert routing."""
        # Gate computation
        gate_logits = layer.gate(hidden_states)
        top_k_weights, top_k_indices = gate_logits.topk(self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_weights, dim=-1)

        # Dispatch tokens to experts
        dispatched = self._dispatch_to_experts(
            hidden_states, top_k_indices, layer.experts
        )

        # Gather results
        output = self._gather_from_experts(
            dispatched, top_k_weights, top_k_indices
        )

        return output
```

### Supported Models

#### LLaMA 2

```python
# deepspeed/inference/v2/model_implementations/llama_v2/

class Llama2Policy:
    """Policy for LLaMA 2 model implementation.

    Defines the mapping from HuggingFace LlamaForCausalLM modules
    to DeepSpeed V2 optimized modules.

    Architecture:
    - RMSNorm (pre-norm)
    - Grouped Query Attention (GQA)
    - SwiGLU FFN
    - Rotary positional embeddings (RoPE)
    """

    def get_module_mapping(self):
        return {
            "model.embed_tokens": RaggedEmbedding,
            "model.layers.*.self_attn": DenseBlockedAttention,
            "model.layers.*.mlp.gate_proj": BLASLinear,
            "model.layers.*.mlp.up_proj": BLASLinear,
            "model.layers.*.mlp.down_proj": BLASLinear,
            "model.layers.*.input_layernorm": DSRMSNorm,
            "model.layers.*.post_attention_layernorm": DSRMSNorm,
            "model.norm": DSRMSNorm,
            "lm_head": RaggedUnembed,
        }


class Llama2TransformerContainer(DSTransformerModelBase):
    """Container for LLaMA 2 model with V2 optimizations.

    Handles:
    - GQA attention with configurable num_key_value_heads
    - SwiGLU FFN (gate + up + down projections)
    - RMSNorm instead of LayerNorm
    - RoPE embeddings
    """

    def __init__(self, config, hf_config):
        super().__init__(config)
        self.hidden_size = hf_config.hidden_size
        self.num_heads = hf_config.num_attention_heads
        self.num_kv_heads = hf_config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.intermediate_size = hf_config.intermediate_size
        self.num_layers = hf_config.num_hidden_layers
        self.vocab_size = hf_config.vocab_size
        self.rms_norm_eps = hf_config.rms_norm_eps
        self.rope_theta = getattr(hf_config, "rope_theta", 10000.0)
```

#### Mistral

```python
class MistralPolicy(Llama2Policy):
    """Policy for Mistral models.

    Similar to LLaMA 2 with differences:
    - Sliding window attention support
    - Different default RoPE theta
    """
    pass
```

#### Mixtral

```python
class MixtralPolicy(Llama2Policy):
    """Policy for Mixtral MoE models.

    Extends LLaMA 2 policy with MoE-specific mappings.
    """

    def get_module_mapping(self):
        mapping = super().get_module_mapping()
        mapping.update({
            "model.layers.*.block_sparse_moe.gate": TopKGating,
            "model.layers.*.block_sparse_moe.experts.*.w1": BLASLinear,
            "model.layers.*.block_sparse_moe.experts.*.w2": BLASLinear,
            "model.layers.*.block_sparse_moe.experts.*.w3": BLASLinear,
        })
        return mapping
```

#### Falcon

```python
class FalconPolicy:
    """Policy for Falcon models.

    Architecture:
    - LayerNorm (post-norm for Falcon-7B) or parallel attention+FFN
    - Multi-query attention (Falcon-7B) or GQA (Falcon-40B/180B)
    - GeLU activation
    """

    def get_module_mapping(self):
        return {
            "transformer.word_embeddings": RaggedEmbedding,
            "transformer.h.*.self_attention": DenseBlockedAttention,
            "transformer.h.*.mlp.dense_h_to_4h": BLASLinear,
            "transformer.h.*.mlp.dense_4h_to_h": BLASLinear,
            "transformer.h.*.input_layernorm": DSRMSNorm,
            "transformer.ln_f": DSRMSNorm,
            "lm_head": RaggedUnembed,
        }
```

#### Phi / Phi-3

```python
class Phi3Policy:
    """Policy for Phi-3 models.

    Architecture:
    - RMSNorm
    - GQA with dense attention
    - SwiGLU FFN
    - RoPE with long context support
    """

    def get_module_mapping(self):
        return {
            "model.embed_tokens": RaggedEmbedding,
            "model.layers.*.self_attn": DenseBlockedAttention,
            "model.layers.*.mlp.gate_up_proj": BLASLinear,  # Fused gate+up
            "model.layers.*.mlp.down_proj": BLASLinear,
            "model.layers.*.input_layernorm": DSRMSNorm,
            "model.layers.*.post_attention_layernorm": DSRMSNorm,
            "model.norm": DSRMSNorm,
            "lm_head": RaggedUnembed,
        }
```

#### OPT

```python
class OPTPolicy:
    """Policy for OPT models.

    Architecture:
    - LayerNorm (pre-norm)
    - Standard multi-head attention
    - ReLU activation in FFN
    - Learned position embeddings
    """

    def get_module_mapping(self):
        return {
            "model.decoder.embed_tokens": RaggedEmbedding,
            "model.decoder.layers.*.self_attn": DenseBlockedAttention,
            "model.decoder.layers.*.fc1": BLASLinear,
            "model.decoder.layers.*.fc2": BLASLinear,
            "model.decoder.layers.*.self_attn_layer_norm": DSLayerNorm,
            "model.decoder.layers.*.final_layer_norm": DSLayerNorm,
            "model.decoder.final_layer_norm": DSLayerNorm,
            "lm_head": RaggedUnembed,
        }
```

#### Qwen / Qwen2

```python
class Qwen2Policy(Llama2Policy):
    """Policy for Qwen2 models.

    Architecture similar to LLaMA 2:
    - RMSNorm
    - GQA
    - SwiGLU FFN
    - RoPE
    """

    pass


class Qwen2MoEPolicy(Qwen2Policy):
    """Policy for Qwen2-MoE models."""

    def get_module_mapping(self):
        mapping = super().get_module_mapping()
        mapping.update({
            "model.layers.*.mlp.gate": TopKGating,
            "model.layers.*.mlp.experts.*.gate_proj": BLASLinear,
            "model.layers.*.mlp.experts.*.up_proj": BLASLinear,
            "model.layers.*.mlp.experts.*.down_proj": BLASLinear,
            "model.layers.*.mlp.shared_expert.gate_proj": BLASLinear,
            "model.layers.*.mlp.shared_expert.up_proj": BLASLinear,
            "model.layers.*.mlp.shared_expert.down_proj": BLASLinear,
        })
        return mapping
```

#### Exaone4

```python
class Exaone4Policy:
    """Policy for Exaone4 models."""

    def get_module_mapping(self):
        return {
            "model.embed_tokens": RaggedEmbedding,
            "model.layers.*.self_attn": DenseBlockedAttention,
            "model.layers.*.mlp.gate_proj": BLASLinear,
            "model.layers.*.mlp.up_proj": BLASLinear,
            "model.layers.*.mlp.down_proj": BLASLinear,
            "model.layers.*.input_layernorm": DSRMSNorm,
            "model.layers.*.post_attention_layernorm": DSRMSNorm,
            "model.norm": DSRMSNorm,
            "lm_head": RaggedUnembed,
        }
```

### Sharding Module

The sharding module handles parameter distribution for tensor parallelism:

```python
# deepspeed/inference/v2/model_implementations/sharding/

class ParameterSharder:
    """Shards model parameters across TP ranks.

    Supports:
    - Column parallel: Split weight along output dimension
    - Row parallel: Split weight along input dimension
    - Replicated: Copy weight to all ranks
    """

    @staticmethod
    def shard_weight(
        weight: torch.Tensor,
        partition_type: str,  # "column", "row", "replicated"
        tp_rank: int,
        tp_size: int,
    ) -> torch.Tensor:
        if partition_type == "column":
            chunks = torch.chunk(weight, tp_size, dim=0)
            return chunks[tp_rank].contiguous()
        elif partition_type == "row":
            chunks = torch.chunk(weight, tp_size, dim=1)
            return chunks[tp_rank].contiguous()
        else:  # replicated
            return weight.clone()
```

---

## Module System

### Interfaces (Abstract Base Classes)

#### DSSelfAttentionBase

```python
class DSSelfAttentionBase(nn.Module, ABC):
    """Interface for self-attention modules in V2.

    All attention implementations must implement:
    - forward(): Process hidden states with KV cache
    - allocate_kv_cache(): Create KV cache blocks for new sequences
    """

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        batch: RaggedBatch,
        kv_cache: BlockedKVCache,
        layer_idx: int,
    ) -> torch.Tensor:
        """Forward pass with blocked KV cache.

        Args:
            hidden_states: [total_tokens, hidden_size]
            batch: Ragged batch metadata
            kv_cache: Blocked KV cache
            layer_idx: Current transformer layer index

        Returns:
            Output hidden states [total_tokens, hidden_size]
        """
        pass
```

#### DSLinearBase

```python
class DSLinearBase(nn.Module, ABC):
    """Interface for linear projection modules."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def weight(self) -> torch.Tensor:
        pass
```

#### DSEmbeddingBase

```python
class DSEmbeddingBase(nn.Module, ABC):
    """Interface for embedding modules."""

    @abstractmethod
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        pass
```

#### DSMOEBase

```python
class DSMOEBase(nn.Module, ABC):
    """Interface for MoE modules."""

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        batch: RaggedBatch,
    ) -> torch.Tensor:
        pass
```

#### DSNormBase

```python
class DSNormBase(nn.Module, ABC):
    """Interface for normalization modules."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
```

### Implementations

#### DenseBlockedAttention

```python
class DenseBlockedAttention(DSSelfAttentionBase):
    """Dense attention with blocked KV cache.

    Features:
    - Flash-style attention computation
    - Blocked KV cache for efficient memory management
    - Rotary positional embeddings
    - Grouped Query Attention (GQA) / Multi-Query Attention (MQA)
    - Support for both prompt (prefill) and generation (decode) phases

    Architecture:
      1. QKV projection
      2. Rotary position embedding application
      3. KV cache update (store new K, V in blocks)
      4. Flash attention computation
      5. Output projection
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position_embeddings: int = 4096,
        rope_theta: float = 10000.0,
        bias: bool = False,
        block_size: int = 16,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size

        # QKV projections
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=bias)

        # RoPE
        self.rotary_emb = RotaryEmbedding(
            dim=head_dim,
            base=rope_theta,
            max_position_embeddings=max_position_embeddings,
        )

    def forward(self, hidden_states, batch, kv_cache, layer_idx):
        total_tokens = hidden_states.shape[0]

        # QKV projection
        q = self.q_proj(hidden_states).view(total_tokens, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(total_tokens, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(total_tokens, self.num_kv_heads, self.head_dim)

        # Apply RoPE
        q, k = apply_rotary_pos_emb(q, k, batch.positions, self.rotary_emb)

        # Update KV cache (blocked)
        block_ids, positions_in_block = self._update_kv_cache(
            k, v, batch, kv_cache, layer_idx
        )

        # Retrieve full KV for attention (all blocks for each sequence)
        k_full, v_full = self._retrieve_kv(batch, kv_cache, layer_idx)

        # Flash attention
        attn_output = self._flash_attention(q, k_full, v_full, batch)

        # Output projection
        attn_output = attn_output.reshape(total_tokens, self.hidden_size)
        return self.o_proj(attn_output)
```

#### BLASLinear

```python
class BLASLinear(DSLinearBase):
    """Standard BLAS-based linear layer.

    Uses cuBLAS GEMM for matrix multiplication. This is the default
    linear implementation when quantization is not used.

    Supports tensor parallelism with column or row partitioning.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        partition_type: Optional[str] = None,  # "column" or "row"
        gather_output: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.partition_type = partition_type

        # Adjusted dimensions for TP
        if partition_type == "column":
            out_features = out_features // get_tp_size()
        elif partition_type == "row":
            in_features = in_features // get_tp_size()

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x):
        output = F.linear(x, self.weight, self.bias)
        if self.partition_type == "row" and self.gather_output:
            dist.all_reduce(output, group=get_tp_group())
        return output
```

#### QuantizedLinear

```python
class QuantizedLinear(DSLinearBase):
    """Quantized linear layer for INT8/INT4 inference.

    Uses CUTLASS mixed-precision GEMM kernels for efficient
    quantized matrix multiplication.

    Weight layout: quantized (INT8 or INT4)
    Activation layout: FP16 or BF16
    Compute: Mixed-precision GEMM with dequantization
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        quant_bits: int = 8,
        group_size: int = 64,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quant_bits = quant_bits
        self.group_size = group_size

        # Quantized weight storage
        if quant_bits == 8:
            self.register_buffer(
                "quant_weight",
                torch.empty(out_features, in_features, dtype=torch.int8),
            )
        elif quant_bits == 4:
            self.register_buffer(
                "quant_weight",
                torch.empty(out_features, in_features // 2, dtype=torch.uint8),
            )

        # Scale and zero-point for group quantization
        n_groups = (out_features * in_features) // group_size
        self.register_buffer("scale", torch.empty(n_groups, 1, dtype=torch.float16))
        self.register_buffer("zero_point", torch.empty(n_groups, 1, dtype=torch.float16))

    def forward(self, x):
        # Use CUTLASS mixed GEMM
        from deepspeed.inference.v2.kernels.cutlass_ops import mixed_gemm
        return mixed_gemm(x, self.quant_weight, self.scale, self.zero_point)
```

#### RaggedEmbedding

```python
class RaggedEmbedding(DSEmbeddingBase):
    """Embedding layer optimized for ragged (variable-length) batches.

    Handles variable-length token sequences without padding by
    working directly with flat token tensors.
    """

    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed flat token tensor.

        Args:
            input_ids: [total_tokens] flat tensor of token IDs.

        Returns:
            [total_tokens, hidden_size] embeddings.
        """
        return self.embedding(input_ids)
```

#### RaggedUnembed

```python
class RaggedUnembed(DSLinearBase):
    """Output projection (LM head) for ragged batches.

    Projects hidden states to vocabulary logits. Only computes
    logits for the last token of each sequence during generation.
    """

    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, hidden_size))

    def forward(self, hidden_states: torch.Tensor, last_token_only: bool = False) -> torch.Tensor:
        """Project to vocabulary.

        Args:
            hidden_states: [total_tokens, hidden_size]
            last_token_only: If True, only compute logits for last token per sequence.

        Returns:
            logits: [total_tokens or batch_size, vocab_size]
        """
        if last_token_only:
            # Only compute logits for generation tokens
            return F.linear(hidden_states, self.weight)
        return F.linear(hidden_states, self.weight)
```

---

## Kernel Operations

### Core Operations (core_ops)

#### bias_activations

```python
def fused_bias_gelu(input: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Fused bias addition + GeLU activation.

    Computes: gelu(input + bias) in a single kernel launch.

    Args:
        input: [*, hidden_size]
        bias: [hidden_size]

    Returns:
        Result tensor [*, hidden_size]
    """
    ...

def fused_bias_silu(input: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Fused bias addition + SiLU activation.

    Computes: silu(input + bias) in a single kernel launch.
    """
    ...
```

#### blas_kernels

```python
def cublas_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    transpose_A: bool = False,
    transpose_B: bool = False,
) -> torch.Tensor:
    """cuBLAS GEMM wrapper for optimized matrix multiplication.

    Supports transposed inputs for efficient weight access patterns.
    """
    ...
```

#### cuda_layer_norm

```python
def cuda_layer_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    epsilon: float = 1e-5,
) -> torch.Tensor:
    """CUDA-optimized Layer normalization.

    Computes: weight * (input - mean) / sqrt(var + eps) + bias
    """
    ...
```

#### cuda_rms_norm

```python
def cuda_rms_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-5,
) -> torch.Tensor:
    """CUDA-optimized RMS normalization.

    Computes: weight * input / sqrt(mean(input^2) + eps)
    """
    ...
```

#### gated_activations

```python
def fused_gated_silu(
    gate: torch.Tensor,
    up: torch.Tensor,
) -> torch.Tensor:
    """Fused gated SiLU (SwiGLU) activation.

    Computes: silu(gate) * up in a single kernel.
    Used in LLaMA/Mistral-style FFN layers.
    """
    ...
```

### CUTLASS Operations (cutlass_ops)

#### mixed_gemm

```python
def mixed_gemm(
    activation: torch.Tensor,     # FP16/BF16 [M, K]
    weight: torch.Tensor,         # INT8/INT4 [N, K]
    scale: torch.Tensor,          # FP16 [N_groups, 1]
    zero_point: torch.Tensor,     # FP16 [N_groups, 1]
    group_size: int = 64,
) -> torch.Tensor:
    """Mixed-precision GEMM using CUTLASS.

    Performs: activation @ dequant(weight) with fused dequantization.

    Args:
        activation: Input activations in FP16/BF16.
        weight: Quantized weights in INT8 or INT4.
        scale: Per-group scale factors.
        zero_point: Per-group zero points.
        group_size: Number of elements per quantization group.

    Returns:
        Output in FP16/BF16, shape [M, N].
    """
    ...
```

#### moe_gemm

```python
def moe_batched_gemm(
    activations: List[torch.Tensor],  # Per-expert input tensors
    weights: List[torch.Tensor],       # Per-expert weight tensors
) -> List[torch.Tensor]:
    """Batched GEMM for MoE expert computation.

    Performs multiple GEMM operations (one per expert) in a single
    batched kernel launch for improved GPU utilization.

    Args:
        activations: List of input tensors, one per expert.
        weights: List of weight tensors, one per expert.

    Returns:
        List of output tensors, one per expert.
    """
    ...
```

### Ragged Operations (ragged_ops)

#### atom_builder

```python
def build_atoms(
    tokens: torch.Tensor,
    seq_lengths: torch.Tensor,
    atom_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build atoms (micro-batches) from ragged token sequences.

    Atoms are fixed-size micro-batches used to improve GPU utilization
    when processing variable-length sequences.

    Args:
        tokens: [total_tokens] flat token tensor.
        seq_lengths: [batch_size] sequence lengths.
        atom_size: Number of tokens per atom.

    Returns:
        atoms: [n_atoms, atom_size] micro-batch tensor.
        atom_metadata: Atom-to-sequence mapping.
    """
    ...
```

#### blocked_flash

```python
def blocked_flash_attention(
    q: torch.Tensor,           # [total_tokens, num_heads, head_dim]
    k_cache: torch.Tensor,     # [n_blocks, block_size, num_kv_heads, head_dim]
    v_cache: torch.Tensor,     # [n_blocks, block_size, num_kv_heads, head_dim]
    block_ids: torch.Tensor,   # [total_tokens] block ID per token
    seq_starts: torch.Tensor,  # [batch_size + 1] sequence boundaries
    block_size: int = 16,
) -> torch.Tensor:
    """Flash attention with blocked KV cache.

    Computes attention over the full KV cache for each sequence,
    using the blocked memory layout for efficient access.

    Args:
        q: Query tensor (all tokens in batch).
        k_cache: K cache in blocked format.
        v_cache: V cache in blocked format.
        block_ids: Block ID assignment for each token.
        seq_starts: Sequence boundary indices.
        block_size: Tokens per KV cache block.

    Returns:
        Attention output [total_tokens, num_heads, head_dim].
    """
    ...
```

#### embed

```python
def ragged_embed(
    input_ids: torch.Tensor,
    embedding_weight: torch.Tensor,
) -> torch.Tensor:
    """Ragged embedding lookup.

    Embeds a flat tensor of token IDs without requiring
    padding to a fixed batch dimension.
    """
    ...
```

#### linear_blocked_kv_rotary

```python
def fused_linear_blocked_kv_rotary(
    hidden_states: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    positions: torch.Tensor,
    rotary_cos: torch.Tensor,
    rotary_sin: torch.Tensor,
    kv_cache: BlockedKVCache,
    block_ids: torch.Tensor,
    positions_in_block: torch.Tensor,
    layer_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused QKV projection + KV cache update + RoPE.

    Combines multiple operations into a single kernel:
    1. Q = hidden_states @ q_weight
    2. K = hidden_states @ k_weight
    3. V = hidden_states @ v_weight
    4. Apply RoPE to Q and K
    5. Store K, V in blocked KV cache

    This fusion reduces memory bandwidth and kernel launch overhead.
    """
    ...
```

#### logits_gather

```python
def gather_logits(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    seq_starts: torch.Tensor,
    last_token_only: bool = True,
    top_k: int = 0,
) -> torch.Tensor:
    """Gather logits for generation.

    When last_token_only=True, only computes logits for the last
    token of each sequence, avoiding unnecessary computation.

    When top_k > 0, only returns top-k logits for speculative
    decoding or constrained generation.
    """
    ...
```

#### moe_gather / moe_scatter

```python
def moe_scatter(
    hidden_states: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_weights: torch.Tensor,
    num_experts: int,
) -> List[torch.Tensor]:
    """Scatter tokens to their assigned experts.

    Args:
        hidden_states: [total_tokens, hidden_size]
        expert_indices: [total_tokens, top_k] expert assignments
        expert_weights: [total_tokens, top_k] gate weights
        num_experts: Total number of experts

    Returns:
        List of [expert_tokens, hidden_size] tensors, one per expert.
    """
    ...

def moe_gather(
    expert_outputs: List[torch.Tensor],
    expert_indices: torch.Tensor,
    expert_weights: torch.Tensor,
    total_tokens: int,
    hidden_size: int,
) -> torch.Tensor:
    """Gather expert outputs and combine with gate weights.

    Args:
        expert_outputs: List of [expert_tokens, hidden_size] outputs
        expert_indices: [total_tokens, top_k] expert assignments
        expert_weights: [total_tokens, top_k] gate weights
        total_tokens: Total number of tokens
        hidden_size: Hidden dimension

    Returns:
        Combined output [total_tokens, hidden_size]
    """
    ...
```

#### top_k_gating

```python
def top_k_gating(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    k: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Top-k gating kernel for MoE routing.

    Computes: top_k(softmax(hidden_states @ gate_weight))

    Args:
        hidden_states: [total_tokens, hidden_size]
        gate_weight: [hidden_size, num_experts]
        k: Number of experts to select per token

    Returns:
        weights: [total_tokens, k] gate weights (softmax probabilities)
        indices: [total_tokens, k] expert indices
    """
    ...
```

---

## Configuration Examples by Model

### LLaMA 2 (7B)

```python
config = RaggedInferenceEngineConfig(
    model_name="meta-llama/Llama-2-7b-hf",
    dtype=torch.float16,
    max_batch_size=32,
    max_seq_length=4096,
    block_size=16,
    max_ragged_batch_size=4096,
)
```

### LLaMA 2 (70B) with TP

```python
config = RaggedInferenceEngineConfig(
    model_name="meta-llama/Llama-2-70b-hf",
    dtype=torch.bfloat16,
    tp_size=4,
    max_batch_size=16,
    max_seq_length=4096,
    block_size=16,
    max_ragged_batch_size=2048,
)
```

### Mistral 7B

```python
config = RaggedInferenceEngineConfig(
    model_name="mistralai/Mistral-7B-v0.1",
    dtype=torch.float16,
    max_batch_size=32,
    max_seq_length=8192,
    block_size=16,
)
```

### Mixtral 8x7B with MoE

```python
config = RaggedInferenceEngineConfig(
    model_name="mistralai/Mixtral-8x7B-v0.1",
    dtype=torch.float16,
    tp_size=4,
    max_batch_size=8,
    max_seq_length=4096,
    block_size=16,
)
```

### Qwen2 72B with Quantization

```python
config = RaggedInferenceEngineConfig(
    model_name="Qwen/Qwen2-72B-Instruct",
    dtype=torch.float16,
    tp_size=4,
    max_batch_size=16,
    max_seq_length=8192,
    block_size=16,
    enable_quantization=True,
    quant_config={
        "weight_bits": 4,
        "group_size": 128,
    },
)
```

### Phi-3 Medium

```python
config = RaggedInferenceEngineConfig(
    model_name="microsoft/Phi-3-medium-4k-instruct",
    dtype=torch.bfloat16,
    max_batch_size=32,
    max_seq_length=4096,
    block_size=16,
)
```

---

## Performance Tuning

### Block Size Selection

| Block Size | KV Cache Overhead | Memory Fragmentation | Recommendation |
|-----------|-------------------|---------------------|----------------|
| 8 | Low | Minimal | Short sequences, low latency |
| 16 | Optimal | Low | Default for most workloads |
| 32 | Moderate | Low | Long sequences, high throughput |
| 64 | High | Some | Very long sequences (>16K) |

### Batch Size Tuning

| Model Size | GPU | Max Batch Size | Throughput (tok/s) | Latency (ms/token) |
|-----------|-----|---------------|-------------------|-------------------|
| 7B | 1x A100 | 64 | ~8000 | ~2 |
| 7B | 1x H100 | 128 | ~15000 | ~1.5 |
| 13B | 1x A100 | 32 | ~5000 | ~3 |
| 70B | 4x A100 | 16 | ~3000 | ~8 |
| 70B | 4x H100 | 32 | ~6000 | ~4 |

### Ragged Batch Size

The `max_ragged_batch_size` controls the maximum number of tokens in a single forward pass:

```python
# For high throughput (many short sequences):
config.max_ragged_batch_size = 8192

# For long sequences (few but long):
config.max_ragged_batch_size = 4096

# Memory estimate: max_ragged_batch_size * hidden_size * 4 * n_layers * 2 bytes
# For 7B model (hidden=4096, 32 layers): 8192 * 4096 * 4 * 32 * 2 = 32 GB
```

### Quantization Tuning

| Quantization | Memory Reduction | Accuracy Impact | Throughput Improvement |
|-------------|-----------------|----------------|----------------------|
| INT8 weight | 2x | < 0.1% | 1.2-1.5x |
| INT4 weight | 4x | 0.5-2% | 1.5-2x |
| INT4 + FP16 compute | 4x | 0.5-2% | 1.3-1.8x |

---

## Code Examples

### Example 1: Basic V2 Inference

```python
import torch
from deepspeed.inference.v2 import InferenceEngineV2, RaggedInferenceEngineConfig

config = RaggedInferenceEngineConfig(
    model_name="meta-llama/Llama-2-7b-hf",
    dtype=torch.float16,
    max_batch_size=32,
    max_seq_length=4096,
    block_size=16,
)

engine = InferenceEngineV2(config)

# Generate
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
prompt = "Deep learning is"
inputs = tokenizer(prompt, return_tensors="pt")
output_ids = engine.generate(inputs["input_ids"].cuda(), max_new_tokens=100)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

### Example 2: Multi-GPU TP Serving

```python
import torch.distributed as dist
from deepspeed.inference.v2 import InferenceEngineV2, RaggedInferenceEngineConfig

dist.init_processgroup("nccl")
local_rank = int(os.environ["LOCAL_RANK"])

config = RaggedInferenceEngineConfig(
    model_name="meta-llama/Llama-2-70b-hf",
    dtype=torch.bfloat16,
    tp_size=4,
    max_batch_size=16,
    max_seq_length=4096,
)

engine = InferenceEngineV2(config)

# Batch generation
prompts = ["Hello, how are you?", "Explain quantum computing.", "Write a poem."]
# ... tokenize and create ragged batch ...
output_ids = engine.generate(input_ids, max_new_tokens=200)
```

### Example 3: Quantized Inference

```python
config = RaggedInferenceEngineConfig(
    model_name="meta-llama/Llama-2-7b-hf",
    dtype=torch.float16,
    max_batch_size=64,
    enable_quantization=True,
    quant_config={
        "weight_bits": 4,
        "group_size": 128,
    },
)

engine = InferenceEngineV2(config)
```

### Example 4: Custom Model Integration

```python
from deepspeed.inference.v2.model_implementations import DSInferenceModelBase
from deepspeed.inference.v2.modules.implementations import (
    DenseBlockedAttention,
    BLASLinear,
    RaggedEmbedding,
    DSRMSNorm,
)

class CustomModelContainer(DSInferenceModelBase):
    """Custom model implementation for V2 engine."""

    def __init__(self, config, hf_config):
        super().__init__(config)
        self.embed = RaggedEmbedding(hf_config.vocab_size, hf_config.hidden_size)
        self.layers = nn.ModuleList([
            CustomLayer(hf_config) for _ in range(hf_config.num_hidden_layers)
        ])
        self.norm = DSRMSNorm(hf_config.hidden_size, hf_config.rms_norm_eps)
        self.lm_head = BLASLinear(hf_config.hidden_size, hf_config.vocab_size, bias=False)

    def forward(self, batch):
        h = self.embed(batch.tokens)
        for i, layer in enumerate(self.layers):
            h = layer(h, batch, self.state_manager.kv_cache, layer_idx=i)
        h = self.norm(h)
        return self.lm_head(h)
```

---

## Troubleshooting

### Common Issues

**1. KV cache OOM**

```
RuntimeError: KV cache out of memory: all 1024 blocks allocated.
```

Reduce `max_batch_size` or `max_seq_length`, or increase `block_size`:
```python
config.max_batch_size = 16  # Reduce concurrent sequences
config.max_seq_length = 2048  # Reduce max sequence length
```

**2. Model not supported**

```
ValueError: No V2 implementation found for model type CustomModel
```

Create a custom model policy and container class extending `DSInferenceModelBase`.

**3. TP size mismatch**

```
RuntimeError: TP size (4) does not match available GPUs (2)
```

Ensure the number of GPUs matches `tp_size`:
```bash
deepspeed --num_gpus=4 inference.py --tp_size 4
```

**4. Ragged batch size too small**

```
RuntimeError: Ragged batch overflow: total tokens (5120) exceeds max (4096)
```

Increase `max_ragged_batch_size`:
```python
config.max_ragged_batch_size = 8192
```

**5. Quantization kernel not available**

```
RuntimeError: CUTLASS mixed-precision GEMM not available for SM 70
```

INT4/INT8 quantization requires SM 75+ (Turing or later). Use FP16 instead on older GPUs.
