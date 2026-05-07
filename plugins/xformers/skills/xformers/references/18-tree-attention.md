# 18 - Tree Attention

## Overview

Tree attention is a mechanism for hierarchical/tree-structured attention, primarily used in speculative decoding where multiple candidate tokens are evaluated in parallel.

**Source**: `xformers/ops/tree_attention.py` (re-exports from `mslk.attention.fmha.tree_attention`)

## API Reference

### `tree_attention`

```python
from xformers.ops import tree_attention

output = tree_attention(
    # Parameters from the mslk package
    ...
) -> torch.Tensor
```

Computes attention for tree-structured sequences, enabling efficient evaluation of multiple speculative paths.

### `TreeAttnMetadata`

```python
from xformers.ops import TreeAttnMetadata
```

Metadata structure describing the tree attention layout, including the tree structure of candidate tokens.

### `construct_tree_choices`

```python
from xformers.ops import construct_tree_choices
```

Constructs the tree choice structure for speculative decoding.

### `construct_full_tree_choices`

```python
from xformers.ops import construct_full_tree_choices
```

Constructs a full tree of choices for complete tree attention.

### `get_full_tree_size`

```python
from xformers.ops import get_full_tree_size
```

Returns the total size of the full attention tree.

### `use_triton_splitk_for_prefix`

```python
from xformers.ops import use_triton_splitk_for_prefix
```

Determines whether to use the Triton Split-K kernel for prefix attention.

### `SplitKAutotune`

```python
from xformers.ops import SplitKAutotune
```

Autotuning configuration for the Split-K attention kernel.

## Use Case: Speculative Decoding

In speculative decoding, a small "draft" model proposes multiple candidate tokens. The main model then verifies these candidates in parallel using tree attention:

```
Draft model proposes:  ["The", " cat", " sat"]
                          ├── "The" ──┬── " cat" ──┬── " sat"
                          │           │            ├── " ran"
                          │           │            └── " lay"
                          │           └── " dog" ──┬── " barked"
                          │                        └── " slept"
                          └── "A" ──┬── " bird" ─── " flew"
                                    └── " fish" ── " swam"
```

Tree attention allows verifying all these paths in a single forward pass, where each path gets proper causal attention.

## Relationship to FMHA

Tree attention is built on top of the FMHA infrastructure:
- Uses the same attention bias system
- Dispatches to the same backends (Flash, CUTLASS, etc.)
- Reuses `BlockDiagonalMask` and related mask types

The `SplitKAutotune` determines the optimal split factor for the Triton Split-K kernel when processing tree-structured sequences.

## Integration Pattern

```python
import torch
from xformers.ops import (
    tree_attention,
    construct_tree_choices,
    TreeAttnMetadata,
)

# During speculative decoding:
# 1. Draft model proposes candidate tokens
# 2. Construct tree choices
choices = construct_tree_choices(draft_tokens)

# 3. Create metadata
metadata = TreeAttnMetadata(choices=choices, ...)

# 4. Run tree attention
output = tree_attention(q, k, v, metadata=metadata)

# 5. Verify candidates and select best path
```

## Performance Considerations

1. **Tree width vs depth**: Wider trees are more parallelizable but use more memory
2. **KV-cache**: Tree attention works with paged KV-caches for efficient memory management
3. **Split-K**: For large trees, the Split-K approach divides the attention computation across multiple thread blocks
