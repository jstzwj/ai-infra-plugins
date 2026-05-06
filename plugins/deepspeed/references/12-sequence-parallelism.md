# DeepSpeed Sequence Parallelism Reference

## Table of Contents

1. [Overview](#overview)
2. [DeepSpeed Ulysses Architecture](#deepspeed-ulysess-architecture)
3. [Core Components](#core-components)
4. [Sequence Parallel Layers](#sequence-parallel-layers)
5. [Fused Pre-Decoder Transformer (FPDT) Layer](#fused-pre-decoder-transformer-fpdt-layer)
6. [Automatic Sequence Parallelism (AutoSP)](#automatic-sequence-parallelism-autosp)
7. [ALST: Arctic Long Sequence Training](#alst-arctic-long-sequence-training)
8. [Forward/Backward Communication](#forwardbackward-communication)
9. [Ulysses-Offload](#ulysess-offload)
10. [Integration with ZeRO](#integration-with-zero)
11. [Configuration and Usage](#configuration-and-usage)
12. [Performance Characteristics](#performance-characteristics)
13. [Code Examples](#code-examples)
14. [Troubleshooting](#troubleshooting)

---

## Overview

DeepSpeed Sequence Parallelism (SP) partitions the **sequence dimension** of activations across multiple GPUs, enabling training and inference with extremely long sequences that would not fit in a single GPU's memory. Unlike tensor parallelism (which partitions weight dimensions) or pipeline parallelism (which partitions layers), sequence parallelism splits the input tokens across devices, with each device processing a contiguous subsequence.

Key benefits:
- Train on sequences of **millions of tokens** by distributing sequence length across GPUs
- Linear reduction in per-GPU activation memory proportional to the SP degree
- Compatible with tensor parallelism, ZeRO, and pipeline parallelism
- Minimal communication overhead: only attention requires cross-device data exchange
- Supports both training and inference workloads

The primary implementation in DeepSpeed is based on the **Ulysses** architecture, which uses specialized all-to-all communication to rearrange sequence and attention head dimensions for efficient distributed attention computation.

---

## DeepSpeed Ulysses Architecture

DeepSpeed Ulysses implements the sequence parallelism approach from the paper "DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models." The core idea is to partition the input sequence across GPUs and use all-to-all communication to rearrange data for multi-head attention.

### Core Concept

```
Input: [batch, seq_len, hidden_dim]

After sequence partitioning (SP=4):
  GPU0: [batch, seq_len/4, hidden_dim]    (tokens 0..S/4)
  GPU1: [batch, seq_len/4, hidden_dim]    (tokens S/4..S/2)
  GPU2: [batch, seq_len/4, hidden_dim]    (tokens S/2..3S/4)
  GPU3: [batch, seq_len/4, hidden_dim]    (tokens 3S/4..S)

Before Attention (all-to-all on heads):
  GPU0: [batch, seq_len/4, H*D]  -->  GPU0: [batch, seq_len, H/4*D]
  GPU1: [batch, seq_len/4, H*D]  -->  GPU1: [batch, seq_len, H/4*D]
  GPU2: [batch, seq_len/4, H*D]  -->  GPU2: [batch, seq_len, H/4*D]
  GPU3: [batch, seq_len/4, H*D]  -->  GPU3: [batch, seq_len, H/4*D]

After Attention (all-to-all on sequence):
  GPU0: [batch, seq_len, H/4*D]  -->  GPU0: [batch, seq_len/4, H*D]
  GPU1: [batch, seq_len, H/4*D]  -->  GPU1: [batch, seq_len/4, H*D]
  ...
```

Each GPU ends up computing attention for a subset of heads over the **full sequence**. This ensures correctness of the attention operation while distributing the sequence processing.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     Sequence Parallel Group                      │
│                                                                  │
│  GPU0 (tokens 0..L/sp)   GPU1 (tokens L/sp..2L/sp)    ...      │
│  ┌─────────────────┐     ┌─────────────────┐                    │
│  │ Input Sub-seq   │     │ Input Sub-seq   │                    │
│  │ [B, L/sp, D]    │     │ [B, L/sp, D]    │                    │
│  └────────┬────────┘     └────────┬────────┘                    │
│           │                       │                              │
│  ┌────────▼────────┐     ┌────────▼────────┐                    │
│  │ QKV Projection  │     │ QKV Projection  │                    │
│  │ (local compute) │     │ (local compute) │                    │
│  └────────┬────────┘     └────────┬────────┘                    │
│           │                       │                              │
│  ═════════╪═══════════════════════╪═════════ All-to-All #1      │
│           │    (sequence -> heads) │                              │
│  ┌────────▼────────┐     ┌────────▼────────┐                    │
│  │ Attention       │     │ Attention       │                    │
│  │ [B, L, H/sp*D]  │     │ [B, L, H/sp*D]  │                   │
│  │ (full seq,      │     │ (full seq,      │                    │
│  │  subset heads)  │     │  subset heads)  │                    │
│  └────────┬────────┘     └────────┬────────┘                    │
│           │                       │                              │
│  ═════════╪═══════════════════════╪═════════ All-to-All #2      │
│           │    (heads -> sequence) │                              │
│  ┌────────▼────────┐     ┌────────▼────────┐                    │
│  │ Output Proj     │     │ Output Proj     │                    │
│  │ [B, L/sp, D]    │     │ [B, L/sp, D]    │                    │
│  └────────┬────────┘     └────────┬────────┘                    │
│           │                       │                              │
│  ┌────────▼────────┐     ┌────────▼────────┐                    │
│  │ FFN (local)     │     │ FFN (local)     │                    │
│  └────────┬────────┘     └────────┬────────┘                    │
│           │                       │                              │
│  ┌────────▼────────┐     ┌────────▼────────┐                    │
│  │ Output Sub-seq  │     │ Output Sub-seq  │                    │
│  └─────────────────┘     └─────────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### Directory Structure

```
deepspeed/sequence/
  __init__.py
  layer.py                 # Sequence parallel layer implementations
  fpdt_layer.py            # Fused Pre-Decoder Transformer layer
  autosp_utils.py          # Automatic SP detection utilities
  autosp_api.py            # AutoSP public API
  autosp_context.py        # AutoSP context management
```

### Key Classes and Functions

| Component | File | Description |
|-----------|------|-------------|
| `SeqAllToAll` | `layer.py` | All-to-all communication primitive for SP |
| `SeqAttention` | `layer.py` | Sequence-parallel attention implementation |
| `SeqTransformerLayer` | `layer.py` | Full transformer layer with SP |
| `FPDTLayer` | `fpdt_layer.py` | Fused Pre-Decoder Transformer layer |
| `auto_sp` | `autosp_api.py` | AutoSP entry point |
| `detect_sp_config` | `autosp_utils.py` | Automatic SP configuration detection |

---

## Sequence Parallel Layers

### SeqAllToAll

The fundamental communication primitive that performs all-to-all collective operations to rearrange tensor dimensions between sequence and head parallelism.

```python
class SeqAllToAll(torch.autograd.Function):
    """All-to-all communication for sequence parallelism.

    Performs either:
    - Forward: scatter sequence, gather heads  (sequence -> heads)
    - Forward: scatter heads, gather sequence  (heads -> sequence)

    The backward pass automatically performs the inverse operation.
    """

    @staticmethod
    def forward(ctx, group, input, scatter_dim, gather_dim):
        ctx.group = group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim

        world_size = dist.get_world_size(group)

        if world_size <= 1:
            return input

        # Perform all-to-all
        input_list = [t.contiguous() for t in torch.tensor_split(input, world_size, dim=scatter_dim)]
        output_list = [torch.empty_like(t) for t in input_list]
        dist.all_to_all(output_list, input_list, group=group)

        return torch.cat(output_list, dim=gather_dim).contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        # Inverse all-to-all: scatter on gather_dim, gather on scatter_dim
        world_size = dist.get_world_size(ctx.group)

        if world_size <= 1:
            return grad_output

        grad_list = [t.contiguous() for t in torch.tensor_split(grad_output, world_size, dim=ctx.gather_dim)]
        output_list = [torch.empty_like(t) for t in grad_list]
        dist.all_to_all(output_list, grad_list, group=ctx.group)

        return None, torch.cat(output_list, dim=ctx.scatter_dim).contiguous(), None, None
```

### SeqAttention

Sequence-parallel multi-head attention implementation.

```python
class SeqAttention(nn.Module):
    """Multi-head attention with sequence parallelism support.

    Handles the all-to-all communication to transform between
    sequence-partitioned and head-partitioned layouts.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_dropout: float = 0.0,
        sp_size: int = 1,
        sp_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.sp_size = sp_size
        self.sp_group = sp_group

        self.dropout = nn.Dropout(attention_dropout)

        # QKV projection (output split across sequence dim)
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size)
        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape

        # Compute QKV locally (sequence-partitioned)
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_attention_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        # All-to-all: scatter sequence, gather heads
        # Before: [B, seq_len/sp, H, D] on each GPU
        # After:  [B, seq_len,     H/sp, D] on each GPU
        q = SeqAllToAll.apply(self.sp_group, q, scatter_dim=1, gather_dim=2)
        k = SeqAllToAll.apply(self.sp_group, k, scatter_dim=1, gather_dim=2)
        v = SeqAllToAll.apply(self.sp_group, v, scatter_dim=1, gather_dim=2)

        # Compute attention with full sequence, subset of heads
        context = self._attention(q, k, v, attention_mask)

        # All-to-all: scatter heads, gather sequence
        # Before: [B, seq_len, H/sp, D] on each GPU
        # After:  [B, seq_len/sp, H, D] on each GPU
        context = SeqAllToAll.apply(self.sp_group, context, scatter_dim=2, gather_dim=1)

        # Reshape and project output
        context = context.reshape(batch_size, seq_len, self.hidden_size)
        output = self.out_proj(context)

        return output
```

### SeqTransformerLayer

A complete transformer layer that combines sequence-parallel attention with locally computed FFN and layer normalization.

```python
class SeqTransformerLayer(nn.Module):
    """Transformer layer with sequence parallelism.

    Architecture:
      1. LayerNorm
      2. SeqAttention (with all-to-all communication)
      3. Residual connection
      4. LayerNorm
      5. FFN (local computation, no communication)
      6. Residual connection
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        sp_size: int = 1,
        sp_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.attention = SeqAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            sp_size=sp_size,
            sp_group=sp_group,
        )
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
        )
        self.dropout = nn.Dropout(hidden_dropout)

    def forward(self, hidden_states, attention_mask=None):
        # Pre-norm attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask)
        hidden_states = self.dropout(hidden_states) + residual

        # Pre-norm FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = self.dropout(hidden_states) + residual

        return hidden_states
```

---

## Fused Pre-Decoder Transformer (FPDT) Layer

The FPDT layer is an optimized implementation that fuses multiple operations for improved performance in sequence-parallel settings. It is designed specifically for decoder-style transformer models.

```python
class FPDTLayer(nn.Module):
    """Fused Pre-Decoder Transformer Layer for sequence parallelism.

    Optimizations over the standard SeqTransformerLayer:
    - Fused QKV projection
    - Fused attention + dropout
    - Fused gate/up projection for SwiGLU FFN
    - Optimized memory layout for all-to-all communication
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,          # For GQA support
        intermediate_size: int,
        rms_norm_eps: float = 1e-5,
        attention_dropout: float = 0.0,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 4096,
        sp_size: int = 1,
        sp_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        self.intermediate_size = intermediate_size
        self.sp_size = sp_size
        self.sp_group = sp_group

        # Attention projections
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * self.head_dim, hidden_size, bias=False)

        # FFN projections (SwiGLU)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        # Normalization
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            base=rope_theta,
            max_position_embeddings=max_position_embeddings,
        )

    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        batch_size, seq_len, _ = hidden_states.shape
        residual = hidden_states

        # Pre-attention RMSNorm
        hidden_states = self.input_layernorm(hidden_states)

        # QKV projections (local, sequence-partitioned)
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for attention: [B, S, H, D]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Apply rotary embeddings (before all-to-all, on local subsequence)
        cos, sin = self.rotary_emb(v, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # All-to-all: scatter sequence, gather heads
        q = SeqAllToAll.apply(self.sp_group, q, scatter_dim=1, gather_dim=2)
        k = SeqAllToAll.apply(self.sp_group, k, scatter_dim=1, gather_dim=2)
        v = SeqAllToAll.apply(self.sp_group, v, scatter_dim=1, gather_dim=2)

        # Compute attention (full sequence, subset of heads)
        attn_output = self._scaled_dot_product_attention(q, k, v, attention_mask)

        # All-to-all: scatter heads, gather sequence
        attn_output = SeqAllToAll.apply(self.sp_group, attn_output, scatter_dim=2, gather_dim=1)

        # Reshape and output projection
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        hidden_states = self.o_proj(attn_output)

        # Residual
        hidden_states = residual + hidden_states

        # FFN with SwiGLU (local computation)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        hidden_states = self.down_proj(F.silu(gate) * up)
        hidden_states = residual + hidden_states

        return hidden_states
```

### FPDT Configuration

```python
fpdt_config = {
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,        # GQA with 8 KV heads
    "intermediate_size": 11008,
    "rms_norm_eps": 1e-5,
    "rope_theta": 500000.0,          # Extended for long sequences
    "max_position_embeddings": 1048576,  # 1M tokens
    "sp_size": 8,
}
```

---

## Automatic Sequence Parallelism (AutoSP)

AutoSP automatically detects opportunities for sequence parallelism and configures the appropriate transformations.

### AutoSP Detection

```python
from deepspeed.sequence.autosp_utils import detect_sp_config

# Auto-detect SP configuration from model and input
sp_config = detect_sp_config(
    model=model,
    seq_length=131072,      # Target sequence length
    hidden_size=4096,
    num_heads=32,
    available_gpus=8,
)
```

### AutoSP API

```python
from deepspeed.sequence.autosp_api import auto_sp

# Apply automatic sequence parallelism
model, sp_config = auto_sp(
    model=model,
    sp_size=8,                       # Number of SP partitions
    sp_group=None,                   # Optional: custom process group
    max_sequence_length=1048576,     # Maximum supported sequence length
    auto_detect=True,                # Auto-detect optimal SP configuration
)
```

### AutoSP Context

```python
from deepspeed.sequence.autosp_context import AutoSPContext

# Context manager for AutoSP configuration
with AutoSPContext(sp_size=4, sp_group=sp_group):
    # All operations within this context use sequence parallelism
    output = model(input_ids)
```

### AutoSP Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sp_size` | `int` | `1` | Number of GPUs for sequence parallelism |
| `sp_group` | `ProcessGroup` | `None` | Process group for SP communication |
| `max_sequence_length` | `int` | `32768` | Maximum sequence length supported |
| `auto_detect` | `bool` | `False` | Whether to auto-detect optimal configuration |
| `use_fpdt` | `bool` | `True` | Use FPDT layers for optimized performance |
| `overlap_comm` | `bool` | `False` | Overlap all-to-all communication with computation |

### Detection Heuristics

AutoSP uses the following heuristics to determine optimal SP configuration:

1. **Sequence Length Threshold**: If `seq_length > max_single_gpu_seq_length`, SP is recommended.
2. **Memory Estimation**: Estimate per-GPU memory for attention KV cache and activations.
3. **Communication Cost**: All-to-all cost scales with `batch_size * seq_length * hidden_dim / sp_size`.
4. **Compute Efficiency**: Ensure each GPU has enough work to saturate compute units.

```python
def estimate_max_seq_per_gpu(hidden_size, num_heads, batch_size, dtype_bytes=2):
    """Estimate maximum sequence length per GPU without SP."""
    total_memory = torch.cuda.get_device_properties(0).total_memory
    # Reserve 50% for model weights, optimizer, etc.
    available = total_memory * 0.5
    # Memory per token for attention: 2 (Q,K) * num_heads * head_dim * batch_size
    bytes_per_token = 2 * hidden_size * batch_size * dtype_bytes
    max_seq = int(available / bytes_per_token)
    return max_seq
```

---

## ALST: Arctic Long Sequence Training

ALST (Arctic Long Sequence Training) is DeepSpeed's framework for training models on multi-million token sequences. It combines sequence parallelism with memory optimization techniques to enable training at unprecedented sequence lengths.

### Key Features

- **Multi-million token sequences**: Train on sequences of 1M+ tokens
- **Efficient memory management**: Optimized KV cache storage and activation checkpointing
- **Hybrid parallelism**: Combines SP with ZeRO and tensor parallelism
- **Custom attention kernels**: Flash attention integration for long sequences

### ALST Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                      ALST Training Pipeline                     │
│                                                                 │
│  Input: [B, 2M tokens]                                         │
│          │                                                      │
│          ▼                                                      │
│  ┌──────────────────┐                                          │
│  │ Token Partition   │  Split across 8 GPUs                    │
│  │ Each GPU: 250K    │                                          │
│  └────────┬─────────┘                                          │
│           │                                                     │
│  ┌────────▼─────────┐                                          │
│  │ Flash Attention   │  All-to-all + flash attention            │
│  │ with SP Comm      │  Full 2M context per head               │
│  └────────┬─────────┘                                          │
│           │                                                     │
│  ┌────────▼─────────┐                                          │
│  │ Local FFN         │  250K tokens, local computation          │
│  └────────┬─────────┘                                          │
│           │                                                     │
│  ┌────────▼─────────┐                                          │
│  │ ZeRO-2 Gradients  │  Gradient partitioning across DP group  │
│  └────────┬─────────┘                                          │
│           │                                                     │
│  ┌────────▼─────────┐                                          │
│  │ Output: [B, 250K] │  Per GPU, reassemble for loss           │
│  └──────────────────┘                                          │
└────────────────────────────────────────────────────────────────┘
```

### ALST Configuration

```json
{
    "sequence_parallel": {
        "enabled": true,
        "sp_size": 8,
        "use_fpdt": true,
        "max_sequence_length": 2097152,
        "alst_enabled": true,
        "attention_type": "flash"
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "nvme",
            "nvme_path": "/local_nvme"
        },
        "overlap_comm": true,
        "contiguous_gradients": true
    },
    "gradient_accumulation_steps": 1,
    "train_batch_size": 1,
    "bf16": {"enabled": true},
    "activation_checkpointing": {
        "partition_activations": true,
        "cpu_checkpointing": true,
        "contiguous_memory_optimization": true
    }
}
```

### Memory Scaling

For a transformer with hidden_size=4096, num_heads=32, and bf16 precision:

| Sequence Length | SP=1 (per GPU) | SP=4 (per GPU) | SP=8 (per GPU) | SP=16 (per GPU) |
|----------------|----------------|----------------|----------------|-----------------|
| 32K | ~8 GB | ~2 GB | ~1 GB | ~0.5 GB |
| 128K | ~32 GB | ~8 GB | ~4 GB | ~2 GB |
| 512K | ~128 GB | ~32 GB | ~16 GB | ~8 GB |
| 1M | ~256 GB | ~64 GB | ~32 GB | ~16 GB |
| 2M | ~512 GB | ~128 GB | ~64 GB | ~32 GB |

---

## Forward/Backward Communication

### Forward Pass Communication Pattern

The forward pass involves two all-to-all operations per transformer layer:

```
Step 1: QKV Projection (local)
  Input: [B, S/sp, H*D]  (sequence-partitioned)
  Q, K, V: [B, S/sp, H*D] each

Step 2: All-to-All #1 (sequence -> heads)
  Before: [B, S/sp, H*D] per GPU
  After:  [B, S, H/sp*D] per GPU
  Cost:   O(B * S * H * D / sp_size) bytes transferred

Step 3: Attention Computation (local)
  Each GPU computes attention for H/sp heads over full S tokens
  Output: [B, S, H/sp*D]

Step 4: All-to-All #2 (heads -> sequence)
  Before: [B, S, H/sp*D] per GPU
  After:  [B, S/sp, H*D] per GPU
  Cost:   O(B * S * H * D / sp_size) bytes transferred

Step 5: Output Projection (local)
  Input: [B, S/sp, H*D]
  Output: [B, S/sp, D]

Step 6: FFN (local, no communication)
  Input: [B, S/sp, D]
  Output: [B, S/sp, D]
```

### Backward Pass Communication Pattern

The backward pass reverses the communication pattern. Since `SeqAllToAll` is implemented as a custom autograd function, the backward pass automatically performs the inverse all-to-all:

```
Step 1: FFN backward (local)
Step 2: Output projection backward (local)
Step 3: All-to-All #2 inverse (sequence -> heads)
  Gradient flows from output projection to attention
Step 4: Attention backward (local)
Step 5: All-to-All #1 inverse (heads -> sequence)
  Gradient flows from attention to QKV projection
Step 6: QKV projection backward (local)
```

### Communication Volume Analysis

Per transformer layer, the total communication volume is:

```
Comm per layer = 2 * B * S * H * D * sizeof(dtype) / sp_size
              = 2 * B * S * hidden_size * sizeof(dtype) / sp_size

For B=1, S=128K, hidden_size=4096, bf16 (2 bytes), sp_size=8:
  Comm per layer = 2 * 1 * 131072 * 4096 * 2 / 8
                 = 2 * 134,217,728 bytes
                 = 256 MB per layer

For a 32-layer model:
  Total comm = 32 * 256 MB = 8 GB per forward+backward pass
```

---

## Ulysses-Offload

Ulysses-Offload extends sequence parallelism to support long-context LLM training with memory offloading. It enables training models with very long context windows on limited GPU memory by offloading KV caches and activations to CPU or NVMe storage.

### Key Mechanisms

1. **KV Cache Offloading**: During attention, KV caches for distant tokens can be offloaded to CPU memory and fetched on demand.
2. **Activation Offloading**: Intermediate activations are offloaded to CPU after the forward pass and restored during the backward pass.
3. **Chunked Processing**: Long sequences are processed in chunks, with only the active chunk kept on GPU.

### Ulysses-Offload Configuration

```json
{
    "sequence_parallel": {
        "enabled": true,
        "sp_size": 4,
        "ulysess_offload": {
            "enabled": true,
            "offload_kv_cache": true,
            "offload_activations": true,
            "kv_cache_device": "cpu",
            "pin_memory": true,
            "chunk_size": 4096,
            "prefetch_chunks": 2
        }
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        }
    }
}
```

### Offloading Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `offload_kv_cache` | `bool` | `False` | Offload KV cache to secondary storage |
| `offload_activations` | `bool` | `False` | Offload activations to secondary storage |
| `kv_cache_device` | `str` | `"cpu"` | Device for KV cache offloading (`"cpu"` or `"nvme"`) |
| `pin_memory` | `bool` | `True` | Use pinned memory for faster CPU-GPU transfer |
| `chunk_size` | `int` | `4096` | Size of chunks for processing long sequences |
| `prefetch_chunks` | `int` | `2` | Number of chunks to prefetch ahead of time |

### Memory Savings with Offloading

For a 70B model training on 1M token sequences:

| Configuration | GPU Memory per Device |
|--------------|----------------------|
| No SP, no offload | ~500 GB (impossible) |
| SP=8, no offload | ~62 GB |
| SP=8, KV offload | ~35 GB |
| SP=8, KV + activation offload | ~20 GB |
| SP=8, full offload + NVMe | ~12 GB |

---

## Integration with ZeRO

Sequence parallelism integrates with ZeRO optimization for combined memory savings.

### Supported Combinations

| ZeRO Stage | SP Supported | Notes |
|-----------|-------------|-------|
| Stage 0 | Yes | SP only |
| Stage 1 | Yes | Optimizer state partitioning + SP |
| Stage 2 | Yes | Optimizer + gradient partitioning + SP |
| Stage 3 | Yes | Full parameter partitioning + SP (careful with all-to-all) |

### Combined Parallelism Strategy

For maximum scalability, combine SP with TP, ZeRO, and pipeline parallelism:

```
Total GPUs = DP_size * TP_size * SP_size * PP_size

Example: 64 GPUs = 2 (DP) * 2 (TP) * 8 (SP) * 2 (PP)

GPU assignment:
  Each GPU has a unique (dp_rank, tp_rank, sp_rank, pp_rank) tuple
  Process groups:
    - DP group: all GPUs with same (tp_rank, sp_rank, pp_rank)
    - TP group: all GPUs with same (dp_rank, sp_rank, pp_rank)
    - SP group: all GPUs with same (dp_rank, tp_rank, pp_rank)
    - PP group: all GPUs with same (dp_rank, tp_rank, sp_rank)
```

### Configuration for SP + TP + ZeRO-2

```json
{
    "sequence_parallel": {
        "enabled": true,
        "sp_size": 4
    },
    "tensor_parallel": {
        "enabled": true,
        "autotp_size": 2,
        "preset_model": "llama"
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        }
    },
    "train_batch_size": 8,
    "bf16": {"enabled": true}
}
```

---

## Configuration and Usage

### DeepSpeed JSON Configuration

```json
{
    "sequence_parallel": {
        "enabled": true,
        "sp_size": 4,
        "use_fpdt": true,
        "overlap_comm": false,
        "max_sequence_length": 131072,
        "ulysess_offload": {
            "enabled": false
        }
    },
    "train_batch_size": 4,
    "gradient_accumulation_steps": 2,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-5
        }
    },
    "bf16": {
        "enabled": true
    }
}
```

### Full Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | `bool` | `False` | Enable sequence parallelism |
| `sp_size` | `int` | `1` | Number of GPUs across which the sequence is partitioned |
| `use_fpdt` | `bool` | `True` | Use FPDT layer implementations for better performance |
| `overlap_comm` | `bool` | `False` | Overlap all-to-all with computation |
| `max_sequence_length` | `int` | `32768` | Maximum sequence length to support |
| `attention_type` | `str` | `"flash"` | Attention implementation (`"flash"`, `"math"`, `"sdpa"`) |
| `ulysess_offload.enabled` | `bool` | `False` | Enable Ulysses-Offload |
| `ulysess_offload.offload_kv_cache` | `bool` | `False` | Offload KV cache |
| `ulysess_offload.offload_activations` | `bool` | `False` | Offload activations |
| `ulysess_offload.kv_cache_device` | `str` | `"cpu"` | Offload device |
| `ulysess_offload.pin_memory` | `bool` | `True` | Pin offloaded memory |
| `ulysess_offload.chunk_size` | `int` | `4096` | Processing chunk size |
| `ulysess_offload.prefetch_chunks` | `int` | `2` | Prefetch lookahead |

### Launch Command

```bash
# 8 GPUs with sequence parallelism
deepspeed --num_gpus=8 train.py \
    --deepspeed_config ds_config_sp.json \
    --seq_length 131072
```

---

## Performance Characteristics

### Communication Cost

Sequence parallelism introduces all-to-all communication at every transformer layer. The communication cost depends on:

1. **Message size**: `batch_size * seq_length * hidden_size * dtype_bytes / sp_size`
2. **Network bandwidth**: InfiniBand bandwidth determines all-to-all latency
3. **Number of GPUs**: All-to-all cost scales logarithmically with `sp_size`

### Scalability

| SP Size | Seq=32K | Seq=128K | Seq=512K | Seq=1M |
|---------|---------|----------|----------|--------|
| 1 | Baseline | OOM | OOM | OOM |
| 2 | 0.95x | Baseline | OOM | OOM |
| 4 | 0.88x | 0.92x | Baseline | OOM |
| 8 | 0.78x | 0.85x | 0.90x | Baseline |
| 16 | 0.65x | 0.75x | 0.82x | 0.88x |

*(Relative throughput compared to the baseline that fits in single GPU memory)*

### Optimal SP Size Selection

```python
def recommend_sp_size(seq_length, hidden_size, batch_size, num_layers, available_gpus, gpu_memory_gb=80):
    """Recommend the minimum SP size for the given sequence length."""
    bytes_per_element = 2  # bf16
    activation_memory = (
        2 * batch_size * seq_length * hidden_size * bytes_per_element  # Input + output
        + num_layers * batch_size * seq_length * hidden_size * bytes_per_element  # Checkpointed activations
    )
    attention_memory = (
        batch_size * seq_length * seq_length * bytes_per_element  # Attention matrix
    )
    total_memory = activation_memory + attention_memory
    min_sp = max(1, int(total_memory / (gpu_memory_gb * 1e9)) + 1)
    return min(min_sp, available_gpus)
```

---

## Code Examples

### Example 1: Basic Sequence Parallel Training

```python
import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer

ds_config = {
    "sequence_parallel": {
        "enabled": True,
        "sp_size": 4,
        "use_fpdt": True,
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True,
        },
    },
    "train_batch_size": 4,
    "bf16": {"enabled": True},
    "optimizer": {
        "type": "AdamW",
        "params": {"lr": 2e-5},
    },
}

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

ds_engine = deepspeed.initialize(
    model=model,
    config=ds_config,
    model_parameters=model.parameters(),
)

# Train with 128K context length
for batch in dataloader:
    inputs = tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=131072,
    )
    inputs = {k: v.to(ds_engine.device) for k, v in inputs.items()}
    outputs = ds_engine(**inputs, labels=inputs["input_ids"])
    ds_engine.backward(outputs.loss)
    ds_engine.step()
```

### Example 2: ALST with 1M Token Sequences

```python
ds_config = {
    "sequence_parallel": {
        "enabled": True,
        "sp_size": 8,
        "use_fpdt": True,
        "max_sequence_length": 1048576,
        "ulysess_offload": {
            "enabled": True,
            "offload_kv_cache": True,
            "offload_activations": True,
            "kv_cache_device": "cpu",
            "pin_memory": True,
            "chunk_size": 8192,
            "prefetch_chunks": 3,
        },
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "nvme",
            "nvme_path": "/local_nvme",
        },
    },
    "activation_checkpointing": {
        "partition_activations": True,
        "cpu_checkpointing": True,
    },
    "train_batch_size": 1,
    "bf16": {"enabled": True},
    "gradient_accumulation_steps": 8,
    "optimizer": {
        "type": "AdamW",
        "params": {"lr": 1e-5},
    },
}
```

### Example 3: SP + TP Hybrid

```python
ds_config = {
    "sequence_parallel": {
        "enabled": True,
        "sp_size": 4,
    },
    "tensor_parallel": {
        "enabled": True,
        "autotp_size": 2,
        "preset_model": "llama",
    },
    "zero_optimization": {
        "stage": 2,
    },
    "train_batch_size": 8,
    "bf16": {"enabled": True},
}

# Launch: 8 GPUs = 4 (SP) x 2 (TP)
# deepspeed --num_gpus=8 train.py --deepspeed_config ds_config.json
```

### Example 4: AutoSP with Automatic Detection

```python
from deepspeed.sequence.autosp_api import auto_sp
import deepspeed

# Let AutoSP decide optimal configuration
model, sp_config = auto_sp(
    model=model,
    sp_size=0,  # 0 = auto-detect
    max_sequence_length=262144,
    auto_detect=True,
)

print(f"AutoSP configured: sp_size={sp_config['sp_size']}")
print(f"Estimated memory per GPU: {sp_config['memory_per_gpu_gb']:.1f} GB")
```

---

## Troubleshooting

### Common Issues

**1. NCCL all-to-all timeout**

```
RuntimeError: NCCL error in: all_to_all
```

Increase NCCL timeout and ensure all ranks can communicate:
```bash
export NCCL_TIMEOUT=3600000  # 1 hour in milliseconds
export NCCL_DEBUG=INFO
```

**2. Sequence length not divisible by sp_size**

```
RuntimeError: Invalid argument: tensor_split expects split_size to be positive
```

Ensure `seq_length % sp_size == 0`. Pad sequences if necessary:
```python
def pad_to_sp_size(seq_length, sp_size):
    return ((seq_length + sp_size - 1) // sp_size) * sp_size
```

**3. OOM during all-to-all communication**

All-to-all temporarily requires memory for both input and output buffers. Increase `sp_size` or reduce `batch_size`:
```python
# Buffer memory = 2 * B * S * hidden_size * dtype_bytes
# Ensure this fits in available GPU memory
```

**4. Incorrect attention with GQA**

When using Grouped Query Attention, ensure `num_kv_heads >= sp_size` so that each GPU gets at least one KV head after the all-to-all:
```python
assert num_kv_heads >= sp_size, f"GQA requires num_kv_heads ({num_kv_heads}) >= sp_size ({sp_size})"
```

**5. Performance degradation with small sequences**

SP adds communication overhead that may not be worthwhile for short sequences. As a rule of thumb, SP is beneficial when `seq_length > 16K`:
```python
if seq_length < 16384:
    print("Warning: Sequence parallelism may not be beneficial for short sequences")
```
