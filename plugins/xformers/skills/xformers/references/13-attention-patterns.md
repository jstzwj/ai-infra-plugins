# 13 - Attention Patterns

## Overview

xFormers provides generators for common attention patterns used in sparse and structured attention mechanisms. These patterns can be converted to block-sparse layouts for efficient computation.

**Source**: `xformers/components/attention/attention_patterns.py`

## Pattern Generators

### 1D Patterns

#### `local_1d_pattern`

```python
from xformers.components.attention.attention_patterns import local_1d_pattern

mask = local_1d_pattern(
    attn_size: int,    # Sequence length
    window_size: int,  # Window size (must be odd)
) -> torch.Tensor     # [attn_size, attn_size] bool
```

Creates a local attention window where each position attends to a fixed-size neighborhood. Window size is odd (counts self-attention + 2 wings).

```
window_size=5:
[1, 1, 1, 0, 0]  # Position 0 sees [0,1,2]
[1, 1, 1, 1, 0]  # Position 1 sees [0,1,2,3]
[0, 1, 1, 1, 1]  # Position 2 sees [1,2,3,4]
[0, 0, 1, 1, 1]  # Position 3 sees [2,3,4]
```

#### `causal_1d_pattern`

```python
from xformers.components.attention.attention_patterns import causal_1d_pattern

mask = causal_1d_pattern(
    attn_size: int,    # Sequence length
) -> torch.Tensor     # [attn_size, attn_size] bool
```

Standard causal (lower triangular) mask.

```
[1, 0, 0, 0, 0]
[1, 1, 0, 0, 0]
[1, 1, 1, 0, 0]
[1, 1, 1, 1, 0]
[1, 1, 1, 1, 1]
```

### 2D Patterns

#### `local_2d_pattern`

```python
from xformers.components.attention.attention_patterns import local_2d_pattern

mask = local_2d_pattern(
    H: int,           # Height
    W: int,           # Width
    distance: float,   # Maximum distance
    p: float = 2.0,   # Distance norm (L2 by default)
) -> torch.Tensor     # [H*W, H*W] bool
```

Local attention in 2D, where each spatial position attends to nearby positions within a given distance.

#### `axial_2d_pattern`

```python
from xformers.components.attention.attention_patterns import axial_2d_pattern

mask = axial_2d_pattern(H, W) -> torch.Tensor  # [H*W, H*W] bool
```

Axial attention: each position attends only to positions in the same row or column.

#### `swin_attention_pattern`

```python
from xformers.components.attention.attention_patterns import swin_attention_pattern

mask = swin_attention_pattern(
    H: int,            # Height (must be divisible by window_size)
    W: int,            # Width (must be divisible by window_size)
    window_size: int,  # Swin window size
    shift_size: int = 0,  # Window shift (0 to window_size-1)
) -> torch.Tensor     # [H*W, H*W] bool
```

Swin Transformer style windowed attention with optional shifted windows.

#### `dilated_2d_pattern`

```python
from xformers.components.attention.attention_patterns import dilated_2d_pattern

mask = dilated_2d_pattern(
    H: int,
    W: int,
    k: int = 2,  # Dilation factor (sample every k-th element)
) -> torch.Tensor
```

Dilated attention: samples 1 every k elements in the attention mask. Like downsampling, where every pixel attends to a downsampled version of the input.

### N-Dimensional Patterns

#### `local_nd_pattern`

```python
from xformers.components.attention.attention_patterns import local_nd_pattern

mask = local_nd_pattern(
    *sizes,       # Size along each dimension
    distance,     # Maximum distance
    p: float = 2.0,  # Distance norm
) -> torch.Tensor
```

Generic N-dimensional local attention.

#### `axial_nd_pattern`

```python
from xformers.components.attention.attention_patterns import axial_nd_pattern

mask = axial_nd_pattern(*sizes) -> torch.Tensor
```

N-dimensional axial attention (attend along each axis independently).

#### `local_nd_distance`

```python
from xformers.components.attention.attention_patterns import local_nd_distance

distances = local_nd_distance(
    *sizes,
    p: float = 2.0,
    weights = None,  # Per-dimension weights
) -> torch.Tensor    # Distance matrix
```

Computes pairwise distances between all positions in an N-dimensional grid.

#### `local_nd_gaussian_distribution`

```python
from xformers.components.attention.attention_patterns import local_nd_gaussian_distribution

probs = local_nd_gaussian_distribution(
    *sizes,
    sigma: float = 1.0,
) -> torch.Tensor    # Gaussian probability matrix
```

Computes Gaussian distribution over position distances.

### Special Patterns

#### `random_pattern`

```python
from xformers.components.attention.attention_patterns import random_pattern

mask = random_pattern(
    attn_size: int,    # Sequence length
    sparsity: float,   # Fraction of zeros (0 < sparsity < 1)
) -> torch.Tensor     # [attn_size, attn_size] bool
```

Random sparse attention pattern with given sparsity level.

#### `global_token_pattern`

```python
from xformers.components.attention.attention_patterns import global_token_pattern

mask = global_token_pattern(
    attention_query_mask: torch.Tensor,  # [seq_len] bool - which tokens are global
) -> torch.Tensor                        # [seq_len, seq_len] bool
```

Tokens marked as global attend to and are attended by all other tokens.

#### `alibi_pattern`

```python
from xformers.components.attention.attention_patterns import alibi_pattern

mask = alibi_pattern(
    threshold: float,        # Threshold for creating binary mask
    mask_shape: torch.Size,  # [heads, seq, seq]
) -> torch.Tensor
```

Creates attention pattern based on ALiBi (Attention with Linear Biases) positional encoding.

#### `random_pattern_from_probability_matrix`

```python
from xformers.components.attention.attention_patterns import random_pattern_from_probability_matrix

mask = random_pattern_from_probability_matrix(
    dist_matrix: torch.Tensor,  # Probability distribution over positions
    nnz: int,                    # Number of non-zero entries to select
) -> torch.Tensor
```

Samples a random sparse pattern from a probability distribution.

## Layout Conversion

### `pattern_to_layout`

```python
from xformers.components.attention.attention_patterns import pattern_to_layout

layout = pattern_to_layout(
    mask: torch.Tensor,    # [Heads, Seq, Seq] or [Seq, Seq] bool
    block_size: int,        # Block size for sparse computation
) -> torch.Tensor          # [Heads, Seq//block_size, Seq//block_size]
```

Converts a dense boolean mask to a block-level layout. Uses max pooling to determine which blocks contain any non-zero entries.

### `layout_to_pattern`

```python
from xformers.components.attention.attention_patterns import layout_to_pattern

pattern = layout_to_pattern(
    layout: torch.Tensor,  # [Heads, H_blocks, W_blocks]
    block_size: int,
) -> torch.Tensor          # [Heads, H_blocks*block_size, W_blocks*block_size]
```

Converts a block-level layout back to a dense pattern using Kronecker product.

### `block_sparsify_tensor`

```python
from xformers.components.attention.attention_patterns import block_sparsify_tensor

blocks = block_sparsify_tensor(
    x: torch.Tensor,      # [B, Heads, Seq, Seq]
    mask: torch.Tensor,    # Boolean mask
    block_size: int,
) -> torch.Tensor          # [B, nnz, block_size, block_size]
```

Extracts non-zero blocks from a dense tensor according to a mask.

## Usage Examples

### Local Attention for Vision Transformer

```python
from xformers.components.attention.attention_patterns import local_2d_pattern, pattern_to_layout

# 224x224 image with 16x16 patches = 14x14 = 196 tokens
H, W = 14, 14
block_size = 16

# Local attention with distance 2
pattern = local_2d_pattern(H, W, distance=2)
layout = pattern_to_layout(pattern, block_size)

# Use with BlockSparseTensor
from xformers.sparse import BlockSparseTensor
nnz = layout.sum().item()
values = torch.randn(1, nnz, block_size, block_size, device="cuda")
attn = BlockSparseTensor(values, layout)
```

### Swin Transformer Pattern

```python
from xformers.components.attention.attention_patterns import swin_attention_pattern

# 56x56 feature map, window size 7, shifted by 3
pattern = swin_attention_pattern(56, 56, window_size=7, shift_size=3)
```

### ALiBi Pattern

```python
from xformers.components.attention.attention_patterns import alibi_pattern

# 8 heads, 256 sequence length
mask = alibi_pattern(threshold=-2.0, mask_shape=torch.Size([8, 256, 256]))
```

### Combining Patterns

```python
from xformers.components.attention.attention_patterns import (
    causal_1d_pattern,
    local_1d_pattern,
    global_token_pattern,
)

# Causal + local window
causal = causal_1d_pattern(512)
local = local_1d_pattern(512, window_size=65)
combined = causal & local  # Causal local attention

# Add global tokens
global_mask = torch.zeros(512, dtype=torch.bool)
global_mask[0] = True  # First token is global
global_pattern = global_token_pattern(global_mask)
final = combined | global_pattern  # Causal local + global
```
