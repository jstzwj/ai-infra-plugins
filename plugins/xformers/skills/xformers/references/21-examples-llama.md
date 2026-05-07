# 21 - LLaMA Inference Example

## Overview

xFormers includes a complete LLaMA inference example that demonstrates how to use the library's key features together: FMHA, RoPE, RMSNorm, SwiGLU, and heterogeneous batch inference.

**Source**: `examples/llama_inference/`

## File Structure

```
examples/llama_inference/
├── model.py          # LLaMA model implementation
├── generate.py       # Text generation script
├── tokenizer.py      # Tokenization utilities
├── sample_utils.py   # Sampling strategies
├── mp_utils.py       # Multi-processing utilities
├── stats.py          # Statistics collection
├── requirements.txt  # Dependencies
└── README.md         # Documentation
```

## Model Architecture

### `model.py`

Complete LLaMA model using xFormers components:

```python
import torch
import torch.nn as nn
import xformers.ops as xops
from xformers.ops import rms_norm, rope_padded

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None  # For GQA
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    max_batch_size: int = 32
    max_seq_len: int = 2048
```

### Key Components

#### Attention

```python
class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = ColumnParallelLinear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = ColumnParallelLinear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = ColumnParallelLinear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = RowParallelLinear(args.n_heads * self.head_dim, args.dim, bias=False)

    def forward(self, x, start_pos, freqs_cis, mask):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # ... RoPE and cache management
        # Uses memory_efficient_attention
        output = xops.memory_efficient_attention(xq, cache_k, cache_v, attn_bias=mask)
        return self.wo(output)
```

#### FeedForward (SwiGLU)

```python
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of, ffn_dim_multiplier):
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(dim, hidden_dim, bias=False)
        self.w2 = RowParallelLinear(hidden_dim, dim, bias=False)
        self.w3 = ColumnParallelLinear(dim, hidden_dim, bias=False)

    def forward(self, x):
        # SwiGLU: F.silu(w1(x)) * w3(x) -> w2
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

#### Transformer Block

```python
class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, start_pos, freqs_cis, mask):
        # Pre-norm architecture
        x = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x
```

#### Full Transformer

```python
class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        self.layers = torch.nn.ModuleList(
            [TransformerBlock(params) for _ in range(params.n_layers)]
        )
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(params.dim, params.vocab_size, bias=False)

    def forward(self, tokens, start_pos):
        h = self.tok_embeddings(tokens)
        freqs_cis = self.precompute_freqs_cis(...)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)

        h = self.norm(h)
        output = self.output(h)
        return output
```

## Heterogeneous Batch Inference

The example demonstrates how to handle batches with different sequence lengths:

```python
# Using rope_padded for heterogeneous batching
from xformers.ops import rope_padded
from xformers.ops.fmha.attn_bias import BlockDiagonalCausalWithOffsetPaddedKeysMask

# Multiple sequences of different lengths packed into one batch
attn_bias = BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
    q_seqlen=[1, 1, 1],        # One new token per sequence
    kv_seqlen=[100, 200, 50],   # Current cache lengths
).to("cuda")

# Apply RoPE and update caches in one fused operation
out_q = rope_padded(xq, xk, xv, cache_k, cache_v, attn_bias)

# Run attention on the full packed batch
output = xops.memory_efficient_attention(out_q, cache_k, cache_v, attn_bias=attn_bias)
```

## Generation Script

### `generate.py`

Main text generation script:

```python
from model import Transformer

def generate(model, prompts, max_gen_len, temperature=0.8, top_p=0.95):
    # Tokenize prompts
    prompt_tokens = [tokenizer.encode(p) for p in prompts]

    # Initialize KV caches
    cache_k = torch.zeros(...)
    cache_v = torch.zeros(...)

    # Prefill phase
    for pos in range(max_prompt_len):
        logits = model(tokens[:, pos], start_pos=pos)
        # Update caches

    # Decode phase
    for pos in range(max_prompt_len, total_len):
        logits = model(tokens[:, pos], start_pos=pos)
        next_token = sample(logits, temperature, top_p)
        tokens[:, pos] = next_token
```

### Sampling Strategies (`sample_utils.py`)

```python
def sample_top_p(logits, prob_threshold):
    """Nucleus sampling: sample from the smallest set of tokens
    whose cumulative probability exceeds prob_threshold."""
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with cumulative probability above threshold
    sorted_indices_to_remove = cumulative_probs - sorted_probs > prob_threshold
    sorted_probs[sorted_indices_to_remove] = 0
    sorted_probs /= sorted_probs.sum()

    # Sample from the filtered distribution
    next_token = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_indices.gather(-1, next_token)
```

## xFormers Features Demonstrated

| Feature | Usage |
|---------|-------|
| `memory_efficient_attention` | Main attention computation |
| `rope_padded` | RoPE with KV-cache management |
| `RMSNorm` | Layer normalization |
| SwiGLU | Feed-forward activation |
| `BlockDiagonalCausalWithOffsetPaddedKeysMask` | Heterogeneous batching |
| `ColumnParallelLinear` / `RowParallelLinear` | Model parallelism |

## Performance Tips

1. **Prefill vs Decode**: Use different attention backends:
   - Prefill: Flash Attention (parallel across sequence)
   - Decode: Split-K or CUTLASS (optimized for single query)

2. **KV-cache management**:
   ```python
   # Pre-allocate caches
   max_batch_size = 32
   max_seq_len = 4096
   cache_k = torch.zeros(n_layers, max_batch_size, max_seq_len, n_kv_heads, head_dim)
   cache_v = torch.zeros_like(cache_k)
   ```

3. **Batched inference**: Use `BlockDiagonalCausalWithOffsetPaddedKeysMask` to batch multiple sequences efficiently

4. **Paged attention**: For serving, use `PagedBlockDiagonalPaddedKeysMask` with page tables to manage KV-cache memory
