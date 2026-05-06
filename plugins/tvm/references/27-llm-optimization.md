# Apache TVM — Chapter 27: LLM Optimization

This reference covers optimization techniques for Large Language Models (LLMs) in TVM. LLMs present unique challenges due to their autoregressive decoding nature, massive parameter counts, and memory-bound execution profiles. TVM provides specialized infrastructure for building, optimizing, and deploying LLMs efficiently.

---

## 27.1 LLM-Specific Challenges

### Memory-Bound Execution

LLM inference is dominated by memory bandwidth rather than compute throughput. During autoregressive decoding, each forward pass processes a single token, making the computation extremely memory-bound:

```
Arithmetic Intensity (Decode):
    FLOPs per byte = 2 * hidden_dim / bytes_per_param
    For LLaMA-7B (FP16): ~2 * 4096 / 2 = 4096 FLOPs/byte (ideal)
    Actual: far less due to KV cache reads, attention, etc.

The key bottleneck is reading model weights from memory for each token.
```

### KV Cache

The Key-Value (KV) cache stores previously computed key and value tensors from attention layers to avoid recomputation during autoregressive decoding:

```python
# Without KV cache (naive):
#   For each new token, recompute attention over ALL previous tokens
#   Cost per token: O(seq_len * hidden_dim)

# With KV cache:
#   For each new token, compute attention only over the new token
#   and attend to cached key/value tensors
#   Cost per token: O(1 * hidden_dim + seq_len * hidden_dim) for attention
#   But weight loading cost: O(total_params) — still dominant
```

### Autoregressive Decoding

The generation process follows this loop:

```
1. Prefill: Process the entire prompt (parallel, compute-heavy)
2. For each subsequent token:
   a. Compute new key/value from the current token
   b. Append to KV cache
   c. Compute attention over full KV cache
   d. Sample next token from logits
   e. Append to output sequence
```

---

## 27.2 Building LLaMA Architecture

### Relax NN Frontend for Transformer Blocks

TVM provides building blocks for constructing transformer architectures using the Relax NN frontend:

```python
import tvm
from tvm import relax
from tvm.relax.frontend import nn
import tvm.tir as T

class LlamaConfig:
    """Configuration for LLaMA model."""
    def __init__(
        self,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_attention_heads: int = 32,
        num_hidden_layers: int = 32,
        num_key_value_heads: int = 32,  # For GQA
        rms_norm_eps: float = 1e-6,
        vocab_size: int = 32000,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.head_dim = hidden_size // num_attention_heads
```

### RMSNorm

```python
class LlamaRMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(
            (hidden_size,), dtype="float32", name="rms_norm_weight"
        )
        self.eps = eps
        self.hidden_size = hidden_size

    def forward(self, x: nn.Tensor) -> nn.Tensor:
        # x: (batch, seq_len, hidden_size)
        # Compute RMS: sqrt(mean(x^2) + eps)
        variance = nn.ops.mean(
            x.astype("float32") ** 2, axis=-1, keepdim=True
        )
        x_normed = x * nn.ops.rsqrt(variance + self.eps)
        return (self.weight * x_normed).astype(x.dtype)
```

### RoPE Positional Encoding

Rotary Position Embedding (RoPE) encodes position information directly into the query and key vectors:

```python
class LlamaRotaryEmbedding(nn.Module):
    """Rotary Position Embedding."""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta

    def forward(
        self,
        q: nn.Tensor,  # (batch, num_heads, seq_len, head_dim)
        k: nn.Tensor,  # (batch, num_kv_heads, seq_len, head_dim)
        position_ids: nn.Tensor,  # (batch, seq_len)
    ) -> tuple:
        """Apply rotary embedding to query and key tensors."""
        # Compute rotation frequencies
        # inv_freq = 1.0 / (theta ^ (2i / head_dim))
        inv_freq = nn.ops.emit(
            # Build inverse frequency table
            # For each pair of dimensions (2i, 2i+1):
            #   freq = position / theta^(2i/d)
            #   q_rot[2i]   = q[2i] * cos(freq) - q[2i+1] * sin(freq)
            #   q_rot[2i+1] = q[2i] * sin(freq) + q[2i+1] * cos(freq)
            nn.ops.reshape(
                nn.ops.broadcast_to(
                    position_ids.unsqueeze(-1).unsqueeze(-1),
                    q.shape,
                ),
                q.shape,
            )
        )
        # Apply rotation via fused RoPE kernel
        q_embed = nn.ops.nn.rotary_embedding(q, position_ids, self.rope_theta)
        k_embed = nn.ops.nn.rotary_embedding(k, position_ids, self.rope_theta)
        return q_embed, k_embed
```

### Fused FFN (Feed-Forward Network)

LLaMA uses a SwiGLU-style FFN with gated activations:

```python
class LlamaMLP(nn.Module):
    """LLaMA MLP with SwiGLU activation."""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.intermediate_size = config.intermediate_size

    def forward(self, x: nn.Tensor) -> nn.Tensor:
        # SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))
        gate = self.gate_proj(x)        # (batch, seq, intermediate)
        up = self.up_proj(x)            # (batch, seq, intermediate)
        # Fused silu + multiply kernel
        return self.down_proj(nn.ops.silu(gate) * up)
```

### QKV Fusion for Attention

```python
class LlamaAttention(nn.Module):
    """Multi-head attention with QKV fusion."""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        # Fused QKV projection (or separate Q, K, V)
        self.qkv_proj = nn.Linear(
            config.hidden_size,
            (self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.rotary_emb = LlamaRotaryEmbedding(config)

    def forward(
        self,
        hidden_states: nn.Tensor,
        position_ids: nn.Tensor,
        kv_cache: tuple = None,
    ) -> tuple:
        batch, seq_len, _ = hidden_states.shape

        # Fused QKV projection
        qkv = self.qkv_proj(hidden_states)
        q, k, v = nn.ops.split(
            qkv,
            [
                self.num_heads * self.head_dim,
                (self.num_heads + self.num_kv_heads) * self.head_dim,
            ],
            axis=-1,
        )

        # Reshape for multi-head attention
        q = nn.ops.reshape(q, (batch, seq_len, self.num_heads, self.head_dim))
        k = nn.ops.reshape(k, (batch, seq_len, self.num_kv_heads, self.head_dim))
        v = nn.ops.reshape(v, (batch, seq_len, self.num_kv_heads, self.head_dim))

        # Transpose to (batch, heads, seq_len, head_dim)
        q = nn.ops.permute_dims(q, axes=[0, 2, 1, 3])
        k = nn.ops.permute_dims(k, axes=[0, 2, 1, 3])
        v = nn.ops.permute_dims(v, axes=[0, 2, 1, 3])

        # Apply rotary embedding
        q, k = self.rotary_emb(q, k, position_ids)

        # Update KV cache
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = nn.ops.concat([cached_k, k], axis=2)
            v = nn.ops.concat([cached_v, v], axis=2)

        # Attention: softmax(Q @ K^T / sqrt(d)) @ V
        # Use fused attention kernel
        attn_output = nn.ops.nn.attention(q, k, v)

        # Reshape and project output
        attn_output = nn.ops.reshape(
            nn.ops.permute_dims(attn_output, axes=[0, 2, 1, 3]),
            (batch, seq_len, self.hidden_size),
        )
        return self.o_proj(attn_output), (k, v)
```

### Full LLaMA Model

```python
class LlamaDecoderLayer(nn.Module):
    """Single transformer decoder layer."""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states, position_ids, kv_cache=None):
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, new_kv = self.self_attn(hidden_states, position_ids, kv_cache)
        hidden_states = residual + attn_output

        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, new_kv


class LlamaModel(nn.Module):
    """Complete LLaMA model."""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, position_ids, kv_caches=None):
        hidden_states = self.embed_tokens(input_ids)

        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            layer_kv = kv_caches[i] if kv_caches is not None else None
            hidden_states, new_kv = layer(hidden_states, position_ids, layer_kv)
            new_kv_caches.append(new_kv)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits, new_kv_caches
```

---

## 27.3 Paged KV Cache

### Efficient KV Cache Management

Traditional KV caches pre-allocate contiguous memory for the maximum sequence length, leading to significant waste due to fragmentation and over-provisioning. TVM implements **paged attention** inspired by the vLLM approach:

```
Traditional KV Cache:
+---------------------------+
|  pre-allocated block      |
|  (max_seq_len * layers *  |
|   heads * head_dim)       |
|  WASTED SPACE >>>         |
+---------------------------+

Paged KV Cache:
+----+----+----+----+----+
| pg | pg | pg | pg | .. |  (allocated on-demand)
+----+----+----+----+----+
  |    |    |
  v    v    v
 Each page stores KV data for a fixed number of tokens.
 Pages are allocated from a pool as the sequence grows.
```

### Page-Based Allocation

```python
from tvm import relax
import tvm

class PagedKVCacheConfig:
    """Configuration for paged KV cache."""
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        page_size: int = 16,       # tokens per page
        max_num_pages: int = 1024, # max pages per sequence
        dtype: str = "float16",
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.max_num_pages = max_num_pages
        self.dtype = dtype
```

### Attention with Paged KV Cache

TVM provides fused attention kernels that operate directly on the paged KV cache:

```python
import tvm
from tvm.script import ir as I, tir as T, relax as R

@I.ir_module
class PagedAttentionModule:
    @T.prim_func
    def paged_attention(
        # Query: (num_seqs, num_heads, head_dim)
        q: T.Buffer((T.int64(1), T.int64(32), T.int64(128)), "float16"),
        # Paged KV cache: (max_num_pages, 2, num_heads, page_size, head_dim)
        kv_cache: T.Buffer(
            (T.int64(4096), T.int64(2), T.int64(32), T.int64(16), T.int64(128)),
            "float16",
        ),
        # Page table: maps logical pages to physical pages
        page_table: T.Buffer((T.int64(1), T.int64(64)), "int32"),
        # Sequence lengths
        seq_lens: T.Buffer((T.int64(1),), "int32"),
        # Output: (num_seqs, num_heads, head_dim)
        output: T.Buffer((T.int64(1), T.int64(32), T.int64(128)), "float16"),
    ):
        # This kernel implements paged attention:
        # 1. Look up the page table to find physical pages
        # 2. Iterate over pages in the KV cache
        # 3. Compute attention scores for each page
        # 4. Aggregate using online softmax
        # 5. Write the attention output
        T.func_attr({"tir.is_scheduled": 1})
        for b in T.thread_binding(1, thread="blockIdx.y"):
            for h in T.thread_binding(32, thread="blockIdx.x"):
                seq_len = seq_lens[b]
                # Online softmax accumulation
                max_score = T.float16(-65504.0)
                sum_exp = T.float16(0.0)
                acc = T.Buffer((128,), "float32")

                # Iterate over pages
                for page_idx in range(T.ceildiv(seq_len, 16)):
                    physical_page = page_table[b, page_idx]
                    page_offset = page_idx * 16

                    for local_pos in range(16):
                        pos = page_offset + local_pos
                        if pos < seq_len:
                            # Compute Q @ K^T for this position
                            score = T.float16(0.0)
                            for d in range(128):
                                score += q[b, h, d] * kv_cache[
                                    physical_page, 0, h, local_pos, d
                                ]
                            score /= T.sqrt(T.float16(128.0))

                            # Online softmax update
                            new_max = T.max(max_score, score)
                            scale = T.exp(max_score - new_max)
                            sum_exp = sum_exp * scale + T.exp(score - new_max)
                            for d in range(128):
                                acc[d] = acc[d] * T.float32(scale) + T.float32(
                                    T.exp(score - new_max) * kv_cache[
                                        physical_page, 1, h, local_pos, d
                                    ]
                                )
                            max_score = new_max

                # Write output
                for d in range(128):
                    output[b, h, d] = T.float16(acc[d] / T.float32(sum_exp))
```

### KV Cache Operations

```python
# The paged KV cache provides these operations:
# 1. Attention with paged cache
# 2. Append new KV data to the cache
# 3. Copy pages between sequences (for beam search)
# 4. Clear pages for completed sequences

# In Relax, these are exposed as built-in operations:
#   R.nn.attention_with_kv_cache  - attention using paged cache
#   R.nn.kv_cache_append          - append new KV entries
#   R.nn.kv_cache_view            - read cached KV data
```

---

## 27.4 Attention Mechanisms

### Standard Multi-Head Attention

```python
# MHA: num_q_heads == num_kv_heads
# Each query head has its own key and value head
config_mha = LlamaConfig(
    num_attention_heads=32,
    num_key_value_heads=32,
)
# K, V shape: (batch, 32, seq_len, head_dim)
```

### Grouped Query Attention (GQA)

GQA shares key-value heads across multiple query heads, reducing KV cache memory:

```python
# GQA: num_q_heads > num_kv_heads
# Multiple query heads share the same key-value head
config_gqa = LlamaConfig(
    num_attention_heads=32,
    num_key_value_heads=8,  # 32/8 = 4 query heads per KV head
)
# K, V shape: (batch, 8, seq_len, head_dim) — 4x smaller cache

# GQA attention implementation
def grouped_query_attention(q, k, v, num_kv_heads):
    """Attention with GQA."""
    # q: (batch, num_q_heads, seq_len, head_dim)
    # k: (batch, num_kv_heads, seq_len, head_dim)
    # v: (batch, num_kv_heads, seq_len, head_dim)
    num_q_heads = q.shape[1]
    group_size = num_q_heads // num_kv_heads

    # Repeat K, V for each group
    # Or use batched GEMM with broadcast
    k_expanded = nn.ops.repeat(k, repeats=group_size, axis=1)
    v_expanded = nn.ops.repeat(v, repeats=group_size, axis=1)

    # Standard attention with expanded K, V
    return nn.ops.nn.attention(q, k_expanded, v_expanded)
```

### Fused Attention Operators

TVM provides fused attention operators that combine the QK^T, softmax, and V multiplication into a single kernel:

```python
# Fused attention in Relax
@R.function
def fused_attention(
    q: R.Tensor((1, 32, 1, 128), "float16"),
    k: R.Tensor((1, 32, 2048, 128), "float16"),
    v: R.Tensor((1, 32, 2048, 128), "float16"),
) -> R.Tensor((1, 32, 1, 128), "float16"):
    with R.dataflow():
        # Single fused kernel for:
        #   scores = q @ k^T / sqrt(d)
        #   weights = softmax(scores)
        #   output = weights @ v
        out = R.nn.attention(q, k, v)
        R.output(out)
    return out
```

### Flash Attention Integration

TVM integrates Flash Attention for memory-efficient attention computation:

```python
# Flash Attention reduces memory from O(n^2) to O(n)
# by tiling the computation and using online softmax

# Enable Flash Attention via target configuration
import tvm
from tvm import relax

target = tvm.target.Target(
    "cuda",
    host="llvm",
)

# The attention operator automatically uses Flash Attention
# when the target supports it and the shapes are compatible
```

---

## 27.5 opt_llm Pipeline

### Specialized Pipeline for LLM Workloads

TVM provides the `opt_llm` pipeline specifically optimized for LLM inference:

```python
from tvm import relax

# Apply the LLM optimization pipeline
mod = relax.get_pipeline("opt_llm")(mod)

# The opt_llm pipeline includes:
# 1. FuseOps — group LLM-specific patterns
# 2. FuseOpsByPattern — match external backend patterns (CUTLASS, etc.)
# 3. LegalizeOps — convert to TIR
# 4. FuseTIR — merge TIR kernels
# 5. DLight rules — apply specialized schedules for GEMV/decode
# 6. Memory planning — optimize buffer allocation
```

### FuseOps for LLM Patterns

The LLM pipeline applies fusion patterns specific to transformer workloads:

```python
# LLM-specific fusion patterns:
# 1. QKV projection fusion: Linear(Q), Linear(K), Linear(V) -> fused QKV
# 2. FFN fusion: gate_proj + silu + up_proj + multiply + down_proj
# 3. RMSNorm fusion: variance + rsqrt + multiply
# 4. Attention fusion: QK^T + softmax + V multiplication
# 5. Residual connection fusion: add + norm

# Example of LLM fusion configuration
from tvm.relax.dpl.pattern import wildcard, is_op

def ffn_fusion_pattern():
    """Pattern for fused SwiGLU FFN."""
    x = wildcard()
    gate_weight = wildcard()
    up_weight = wildcard()
    down_weight = wildcard()

    gate = is_op("relax.matmul")(x, gate_weight)
    silu_gate = is_op("relax.nn.silu")(gate)
    up = is_op("relax.matmul")(x, up_weight)
    mul = is_op("relax.multiply")(silu_gate, up)
    return is_op("relax.matmul")(mul, down_weight)
```

### DLight Rules for GEMV/Decode

DLight provides specialized scheduling rules for the memory-bound decode phase:

```python
from tvm.relax.dpl import DLightRules

# During decode, matmul becomes GEMV (matrix-vector multiplication)
# DLight generates optimized GEMV schedules:
# - Weight matrix is loaded in a tiled fashion
# - Vectorized loads for contiguous weight access
# - Thread-level reduction for the dot product

# For the prefill phase (compute-heavy), DLight uses:
# - Standard GEMM tiling
# - Shared memory for intermediate results
# - Tensor core utilization when available
```

---

## 27.6 Prefill vs Decode Phases

### Prefill Phase

The prefill phase processes the entire prompt in parallel. It is compute-heavy and benefits from standard GEMM optimization:

```python
# Prefill: process all prompt tokens at once
# Shape: (batch, prompt_len, hidden_dim)
# Matmul becomes: (batch, prompt_len, hidden_dim) @ (hidden_dim, hidden_dim)
# This is a standard GEMM with large M dimension

def prefill_forward(model, input_ids, position_ids):
    """Prefill: process entire prompt."""
    # input_ids: (batch, prompt_len)
    # This is compute-bound — use GEMM-optimized schedules
    logits, kv_caches = model(input_ids, position_ids)
    return logits, kv_caches

# TVM generates GEMM kernels for prefill:
# - Tiled GEMM with shared memory
# - Tensor core utilization (FP16/BF16)
# - Large tile sizes for maximum occupancy
```

### Decode Phase

The decode phase generates tokens one at a time. It is memory-bound and requires GEMV-optimized kernels:

```python
# Decode: generate one token at a time
# Shape: (batch, 1, hidden_dim)
# Matmul becomes: (batch, 1, hidden_dim) @ (hidden_dim, hidden_dim)
# This is a GEMV — memory bandwidth limited

def decode_step(model, input_id, position_id, kv_caches):
    """Decode: generate a single token."""
    # input_id: (batch, 1) — single token
    # This is memory-bound — use GEMV-optimized schedules
    logits, new_kv_caches = model(input_id, position_id, kv_caches)
    next_token = sample(logits)
    return next_token, new_kv_caches

# TVM generates GEMV kernels for decode:
# - Weight matrix loaded in tiles matching vector width
# - No shared memory needed (single row of output)
# - Vectorized loads for maximum bandwidth utilization
# - Warp-level reduction for the dot product
```

### Separate Scheduling for Each Phase

TVM can compile separate kernels for prefill and decode:

```python
from tvm import relax

# Build two versions of the model:
# 1. Prefill model (static prompt length or dynamic with symbolic vars)
prefill_mod = build_llm_prefill(model, prompt_len=128)

# 2. Decode model (single token input)
decode_mod = build_llm_decode(model)

# At inference time, use the appropriate model:
# - First call: prefill with the prompt
# - Subsequent calls: decode with cached KV

# Some deployments combine both into a single model with
# a dynamic shape dimension:
@R.function
def main(
    input_ids: R.Tensor((1, "seq_len"), "int64"),
    position_ids: R.Tensor((1, "seq_len"), "int64"),
) -> R.Tuple(
    R.Tensor((1, "seq_len", vocab_size), "float32"),
    R.Object,  # KV cache
):
    # The scheduler selects GEMM (seq_len > threshold)
    # or GEMV (seq_len == 1) optimization automatically
    ...
```

---

## 27.7 Quantization Support

### INT4/INT8 Quantization

TVM supports weight quantization to reduce memory bandwidth requirements during decode:

```python
import tvm
from tvm import relax

# INT4 weight-only quantization
# Weights are stored in INT4, dequantized on-the-fly during computation
mod = relax.transform.FuseOpsByPattern(
    patterns=[
        # Quantized matmul patterns
        ("quantized_matmul_int4", int4_matmul_pattern()),
    ]
)(mod)

# Apply quantization transformation
mod = relax.transform.FuseQuantize(
    dtype="int4",
    storage_dtype="int32",  # Pack 8 int4 values into one int32
    group_size=128,          # Quantization group size
    quantize_activation=False,  # Weight-only quantization
)(mod)
```

### Weight-Only Quantization

Weight-only quantization is the most common approach for LLMs since activation quantization can degrade accuracy:

```python
# Weight-only INT4 quantization
# Format: weights stored as INT4, scale factors stored in FP16
# At runtime: dequantize weights to FP16 before computation

@T.prim_func
def dequantize_int4_weight(
    packed_weights: T.Buffer((4096, 512), "int32"),  # Packed INT4
    scale: T.Buffer((4096, 128), "float16"),          # Per-group scale
    zeros: T.Buffer((4096, 128), "float16"),          # Per-group zero-point
    output: T.Buffer((4096, 4096), "float16"),
):
    # Each int32 holds 8 int4 values
    for i, j in T.grid(4096, 4096):
        with T.sblock("dequant"):
            vi, vj = T.axis.remap("SS", [i, j])
            packed_idx = j // 8
            shift = (j % 8) * 4
            raw = T.bitwise_and(
                T.shift_right(packed_weights[vi, packed_idx], shift),
                T.int32(0xF),
            )
            group_idx = j // 128  # group_size = 128
            output[vi, vj] = T.float16(raw) * scale[vi, group_idx] + zeros[vi, group_idx]
```

### SmoothQuant

SmoothQuant migrates quantization difficulty from weights to activations using a smoothing factor:

```python
# SmoothQuant: x * W = (x * diag(s)^-1) * (diag(s) * W)
# Smooth the activation, sharpen the weight
# Then quantize both to INT8

def apply_smoothquant(model, calibration_data, alpha=0.5):
    """Apply SmoothQuant to the model."""
    # 1. Collect activation statistics from calibration data
    # 2. Compute smoothing factor: s_j = max(|x_j|)^alpha / max(|w_j|)^(1-alpha)
    # 3. Absorb smoothing factor into weights
    # 4. Quantize smoothed weights to INT8
    # 5. At runtime, activations can be quantized to INT8 without loss

    for layer in model.layers:
        # Get activation statistics
        max_act = get_activation_max(layer, calibration_data)
        max_weight = get_weight_max(layer)

        # Compute smoothing factor
        smooth_factor = (max_act ** alpha) / (max_weight ** (1 - alpha))

        # Apply smoothing
        layer.self_attn.qkv_proj.weight *= smooth_factor.unsqueeze(0)
        layer.mlp.gate_proj.weight *= smooth_factor.unsqueeze(0)

    return model
```

---

## 27.8 Complete Inference Workflow

### Tokenizer Integration

```python
from transformers import AutoTokenizer

# Load tokenizer (external to TVM, typically from HuggingFace)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

def encode_prompt(prompt: str) -> list:
    """Tokenize a prompt."""
    return tokenizer.encode(prompt, return_tensors="np")

def decode_tokens(token_ids: list) -> str:
    """Decode token IDs back to text."""
    return tokenizer.decode(token_ids, skip_special_tokens=True)
```

### Prefill + Decode Loop

```python
import tvm
from tvm import relax
import numpy as np

class LLMInferenceEngine:
    """Complete LLM inference engine using TVM."""

    def __init__(self, model_path: str, device: tvm.runtime.Device):
        # Load compiled model
        self.mod = tvm.runtime.load_module(model_path)
        self.vm = relax.VirtualMachine(self.mod, device)
        self.device = device

        # Load parameters
        params = tvm.runtime.load_param_dict(
            open(model_path + ".params", "rb").read()
        )
        self.vm["set_constants"](params)

    def generate(
        self,
        prompt_tokens: list,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> list:
        """Generate tokens autoregressively."""
        all_tokens = list(prompt_tokens)
        kv_cache = None

        # Phase 1: Prefill
        input_ids = tvm.nd.array(
            np.array([prompt_tokens], dtype="int64"), self.device
        )
        position_ids = tvm.nd.array(
            np.arange(len(prompt_tokens), dtype="int64").reshape(1, -1),
            self.device,
        )
        logits, kv_cache = self.vm["prefill"](input_ids, position_ids)

        # Sample first new token
        next_token = self._sample(logits, temperature, top_p)
        all_tokens.append(next_token)

        # Phase 2: Decode
        for step in range(max_new_tokens - 1):
            input_ids = tvm.nd.array(
                np.array([[next_token]], dtype="int64"), self.device
            )
            position_id = len(all_tokens) - 1
            position_ids = tvm.nd.array(
                np.array([[position_id]], dtype="int64"), self.device
            )
            logits, kv_cache = self.vm["decode"](input_ids, position_ids, kv_cache)

            next_token = self._sample(logits, temperature, top_p)
            all_tokens.append(next_token)

            # Check for end-of-sequence token
            if next_token == tokenizer.eos_token_id:
                break

        return all_tokens

    def _sample(self, logits, temperature, top_p):
        """Sample a token from logits."""
        # Convert to numpy for sampling
        logits_np = logits.numpy()[0, -1, :]  # Last token logits

        # Apply temperature
        if temperature > 0:
            logits_np = logits_np / temperature

        # Apply top-p filtering
        if top_p < 1.0:
            sorted_indices = np.argsort(logits_np)[::-1]
            sorted_logits = logits_np[sorted_indices]
            probs = np.exp(sorted_logits) / np.sum(np.exp(sorted_logits))
            cumulative_probs = np.cumsum(probs)
            # Remove tokens with cumulative probability above threshold
            cutoff = sorted_indices[cumulative_probs > top_p]
            logits_np[cutoff] = -np.inf

        # Sample from distribution
        probs = np.exp(logits_np) / np.sum(np.exp(logits_np))
        return np.random.choice(len(probs), p=probs).item()
```

### Sampling Strategies

```python
def greedy_search(logits: np.ndarray) -> int:
    """Greedy decoding: always pick the most probable token."""
    return np.argmax(logits).item()

def temperature_sampling(logits: np.ndarray, temperature: float = 1.0) -> int:
    """Temperature-controlled sampling."""
    scaled = logits / temperature
    probs = np.exp(scaled) / np.sum(np.exp(scaled))
    return np.random.choice(len(probs), p=probs).item()

def top_k_sampling(logits: np.ndarray, k: int = 50) -> int:
    """Top-K sampling: only consider top K tokens."""
    top_k_indices = np.argpartition(logits, -k)[-k:]
    top_k_logits = logits[top_k_indices]
    probs = np.exp(top_k_logits) / np.sum(np.exp(top_k_logits))
    return top_k_indices[np.random.choice(k, p=probs)].item()

def top_p_sampling(logits: np.ndarray, p: float = 0.9) -> int:
    """Nucleus (top-p) sampling."""
    sorted_indices = np.argsort(logits)[::-1]
    sorted_logits = logits[sorted_indices]
    probs = np.exp(sorted_logits) / np.sum(np.exp(sorted_logits))
    cumulative = np.cumsum(probs)
    cutoff_idx = np.searchsorted(cumulative, p) + 1
    top_indices = sorted_indices[:cutoff_idx]
    top_probs = probs[:cutoff_idx]
    top_probs = top_probs / np.sum(top_probs)
    return top_indices[np.random.choice(len(top_indices), p=top_probs)].item()
```

---

## 27.9 Performance Optimization Tips

### Flash Attention Integration

Flash Attention reduces memory usage from O(n^2) to O(n) and improves wall-clock time by reducing HBM accesses:

```python
# Flash Attention is automatically used when:
# 1. The target is CUDA
# 2. The head_dim is <= 128
# 3. The data type is FP16 or BF16
# 4. The attention function is called with compatible shapes

# Verify Flash Attention is being used
import tvm
from tvm import relax

# Build with Flash Attention
target = tvm.target.Target("nvidia/nvidia-a100")
mod = relax.get_pipeline("opt_llm")(imported_mod)
exec = relax.build(mod, target=target)

# Flash Attention kernel characteristics:
# - Tiled Q, K, V loading from HBM
# - Online softmax (no materialization of full attention matrix)
# - O(n) memory instead of O(n^2)
```

### Tensor Parallelism

For models too large for a single GPU, TVM supports tensor parallelism via the Disco distributed runtime:

```python
import tvm
from tvm import relax
from tvm.runtime import disco

# Configure tensor parallelism
num_gpus = 4

# Create a device mesh for tensor parallelism
mesh = disco.DeviceMesh(
    range(num_gpus),  # Physical devices
    shape=(num_gpus,),  # 1D mesh for tensor parallelism
)

# The model is sharded across GPUs:
# - Weight matrices are split column-wise (or row-wise)
# - Attention heads are distributed across GPUs
# - All-reduce is used to synchronize partial results

# Import and optimize with sharding annotations
mod = from_exported_program(model)
mod = relax.get_pipeline("opt_llm")(mod)

# Annotate sharding
# Each linear layer is sharded across the device mesh
# Attention heads are partitioned across GPUs
```

### CUDA Graph Capture

CUDA graph capture eliminates kernel launch overhead by recording the entire execution graph and replaying it:

```python
# CUDA graph capture is particularly effective for the decode phase
# where kernel launch overhead is a significant fraction of total time

import tvm
from tvm import relax

# Build with CUDA graph support
target = tvm.target.Target("cuda")
exec = relax.build(mod, target=target)
vm = relax.VirtualMachine(exec, tvm.cuda(0))

# Capture execution graph
# TVM supports CUDA graph capture for the decode step
# which replays the entire decode computation as a single GPU command
graph = vm.capture_cuda_graph("decode", input_ids, position_ids, kv_cache)

# Replay the captured graph for subsequent tokens
for step in range(max_new_tokens):
    output, new_kv_cache = vm.run_cuda_graph(graph)
    next_token = sample(output)
    # Update inputs for next step
```

### Memory Optimization

```python
# Key memory optimizations for LLM inference:

# 1. Weight sharing: Tie embedding and LM head weights
#    Reduces memory by vocab_size * hidden_dim
config.tie_word_embeddings = True

# 2. KV cache dtype: Use FP8 or INT8 for KV cache
#    Reduces KV cache memory by 2x
kv_cache_dtype = "float8_e5m2"  # or "int8"

# 3. Continuous batching: Share KV cache pages across requests
#    Enabled by paged KV cache

# 4. Weight quantization: INT4 reduces weight memory by 4x
quantize_weights = True
weight_dtype = "int4"
group_size = 128

# 5. Activation checkpointing: Recompute activations during
#    forward pass instead of storing them (mainly for training)
```

### Batched Inference

```python
# For throughput-oriented serving, batch multiple sequences together

# Continuous batching: sequences can enter and exit the batch dynamically
class ContinuousBatcher:
    def __init__(self, engine, max_batch_size=32):
        self.engine = engine
        self.max_batch_size = max_batch_size
        self.active_sequences = []

    def add_request(self, prompt_tokens):
        """Add a new sequence to the batch."""
        if len(self.active_sequences) < self.max_batch_size:
            self.active_sequences.append({
                "tokens": prompt_tokens,
                "kv_cache": None,
                "phase": "prefill",
            })

    def step(self):
        """Run one step for all active sequences."""
        # Group sequences by phase
        prefill_seqs = [s for s in self.active_sequences if s["phase"] == "prefill"]
        decode_seqs = [s for s in self.active_sequences if s["phase"] == "decode"]

        # Run prefill for new sequences
        if prefill_seqs:
            for seq in prefill_seqs:
                logits, kv_cache = self.engine.prefill(seq["tokens"])
                seq["kv_cache"] = kv_cache
                seq["phase"] = "decode"
                seq["last_logits"] = logits

        # Run decode for all decode sequences (batched)
        if decode_seqs:
            # Concatenate inputs from all sequences
            # Use paged KV cache to handle different sequence lengths
            batch_inputs = np.stack([s["tokens"][-1:] for s in decode_seqs])
            # Run batched decode
            batch_logits, new_caches = self.engine.decode_batch(batch_inputs)

            # Update each sequence
            for i, seq in enumerate(decode_seqs):
                next_token = sample(batch_logits[i])
                seq["tokens"].append(next_token)
                seq["kv_cache"] = new_caches[i]
                if next_token == eos_token:
                    self.active_sequences.remove(seq)
```

---

## 27.10 Benchmarking LLM Performance

### Key Metrics

```python
# Throughput: tokens/second (aggregate across batch)
# Latency: time to first token (TTFT), time per output token (TPOT)
# Memory: peak GPU memory usage

import time
import numpy as np

def benchmark_llm(engine, prompt_lengths=[32, 128, 512], decode_lengths=[64, 128, 256]):
    """Benchmark LLM inference performance."""
    results = {}

    for prompt_len in prompt_lengths:
        for decode_len in decode_lengths:
            prompt = list(range(prompt_len))

            # Measure prefill time
            start = time.perf_counter()
            logits, kv_cache = engine.prefill(prompt)
            prefill_time = time.perf_counter() - start

            # Measure decode time
            decode_times = []
            for _ in range(decode_len):
                start = time.perf_counter()
                next_token = sample_token(logits)
                logits, kv_cache = engine.decode(next_token, kv_cache)
                decode_times.append(time.perf_counter() - start)

            total_time = prefill_time + sum(decode_times)
            throughput = decode_len / sum(decode_times)

            results[(prompt_len, decode_len)] = {
                "prefill_ms": prefill_time * 1000,
                "decode_ms_per_token": np.mean(decode_times) * 1000,
                "throughput_tok_s": throughput,
                "total_ms": total_time * 1000,
            }

    return results
```

---

## 27.11 Summary

| Technique | Phase | Benefit | Implementation |
|-----------|-------|---------|---------------|
| Fused QKV | Prefill/Decode | 3x fewer kernels | FuseOps |
| Fused FFN (SwiGLU) | Prefill/Decode | 2x fewer kernels | FuseOps |
| Paged KV cache | Decode | Efficient memory | Runtime |
| Flash Attention | Prefill | O(n) memory | Fused kernel |
| GEMV schedule | Decode | Max bandwidth | DLight |
| GEMM schedule | Prefill | Max FLOPS | DLight/MetaSchedule |
| INT4 quantization | Decode | 4x less memory | Quantize transform |
| Tensor parallelism | Both | Multi-GPU | Disco |
| CUDA graph | Decode | Less launch overhead | Runtime |
| Continuous batching | Serve | Higher throughput | Runtime |

TVM's LLM optimization stack covers the entire pipeline from model construction to deployment, with specialized optimizations for both the compute-heavy prefill phase and the memory-bound decode phase.
