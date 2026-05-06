# 20 - Gradient Checkpointing (jax.checkpoint / jax.remat)

## Overview

`jax.checkpoint` (aliased as `jax.remat`) trades computation for memory by recomputing intermediate values during the backward pass instead of storing them. This enables training much larger models.

---

## 1. The Memory Problem

### Standard backward pass

```
Forward:  x → [layer1] → h1 → [layer2] → h2 → [layer3] → loss
                                    ↑ store    ↑ store
Backward: loss → grad_h2 → grad_h1
                      uses h2      uses h1
```

All intermediates `h1`, `h2` must be stored in memory for the backward pass.

### With checkpointing

```
Forward:  x → [layer1] → h1 → [layer2] → h2 → [layer3] → loss
Backward: loss → recompute h2 from h1 → grad_h2 → recompute h1 from x → grad_h1
```

Only checkpoints (e.g., `h1`) are stored. Others are recomputed on demand.

---

## 2. Basic Usage

### Decorator form

```python
import jax
import jax.numpy as jnp

@jax.checkpoint
def expensive_block(x, w):
    """This block's intermediates won't be stored."""
    for _ in range(10):
        x = jnp.tanh(x @ w)
    return x

# Or equivalently:
expensive_block = jax.remat(expensive_block)
```

### Wrapper form

```python
def f(x):
    x = expensive_block(x, w1)  # Checkpointed
    x = expensive_block(x, w2)  # Checkpointed
    return x

grads = jax.grad(loss)(params)
```

### Per-layer checkpointing

```python
def make_block(w1, w2, b1, b2):
    @jax.checkpoint
    def block(x):
        x = jnp.maximum(0, x @ w1 + b1)
        return x @ w2 + b2
    return block

blocks = [make_block(**p) for p in params]

def forward(x):
    for block in blocks:
        x = block(x)
    return x
```

---

## 3. Policy Functions

`jax.checkpoint` accepts a `policy` argument that controls which operations to rematerialize.

### Built-in policies

```python
from jax.checkpoint_policies import (
    everything_saveable,       # No checkpointing (default behavior)
    nothing_saveable,          # Recompute everything
    dots_saveable,             # Save dot products (large outputs)
    dots_with_no_batch_dims_saveable,
    checkpoint_dots,           # Checkpoint around dot products
    checkpoint_dots_with_no_batch_dims,
    save_any_names_here,       # Custom: save ops with given names
    save_from_both_policies,   # Intersection of two policies
    save_and_drop_extra,
)
```

### Using policies

```python
# Save dot products (they're expensive to recompute)
f_rematted = jax.remat(f, policy=jax.checkpoint_policies.dots_saveable)

# Recompute everything
f_rematted = jax.remat(f, policy=jax.checkpoint_policies.nothing_saveable)
```

### Custom policy

```python
def my_policy(prim_name, *args, **kwargs):
    """Return True to save, False to recompute."""
    # Save dot products and convolutions
    if prim_name in ('dot_general', 'conv_general_dilated'):
        return True
    # Recompute everything else
    return False

f_rematted = jax.remat(f, policy=my_policy)
```

### Policy function signature

```python
def policy(
    prim_name: str,       # Name of the primitive
    involved_devices: ..., # For distributed operations
    *args, **kwargs       # Other context
) -> bool:               # True = save, False = recompute
```

---

## 4. Prevent_c=False Mode

By default (`prevent_c=True`), remat prevents CSE (Common Subexpression Elimination) on rematerialized values. Setting `prevent_c=False` allows more aggressive optimization:

```python
f_rematted = jax.remat(f, prevent_c=False)
```

---

## 5. Differentiable remat

```python
@jax.remat
def block(x, w):
    x = jnp.sin(x @ w)
    x = jnp.cos(x)
    return x

# Works with grad
def loss(params, x, y):
    h = block(x, params)
    return jnp.mean((h - y) ** 2)

grads = jax.grad(loss)(params, x, y)
```

---

## 6. Nested Checkpointing

```python
@jax.checkpoint
def outer_block(x, params):
    @jax.checkpoint
    def inner_block(x, w):
        return jnp.relu(x @ w)

    x = inner_block(x, params['w1'])
    x = inner_block(x, params['w2'])
    return x
```

Memory savings: O(sqrt(n)) with nested checkpoints for n sequential blocks, vs O(n) without.

---

## 7. Selective Checkpointing in Transformers

```python
def transformer_block(x, params, key):
    # Checkpoint the attention computation
    @jax.checkpoint
    def attention(x, params):
        q = x @ params['wq']
        k = x @ params['wk']
        v = x @ params['wv']
        scores = q @ k.transpose(-1, -2) / jnp.sqrt(q.shape[-1])
        mask = jnp.triu(jnp.ones_like(scores), k=1) * (-1e9)
        attn = jax.nn.softmax(scores + mask, axis=-1)
        return attn @ v

    # Checkpoint the MLP
    @jax.checkpoint
    def mlp(x, params):
        h = jnp.gelu(x @ params['w1'] + params['b1'])
        return h @ params['w2'] + params['b2']

    # Save residuals, recompute attention/MLP
    h = x + attention(jax.nn.layer_norm(x), params['attn'])
    out = h + mlp(jax.nn.layer_norm(h), params['mlp'])
    return out
```

---

## 8. Memory Analysis

### Memory vs compute tradeoff

```python
import functools

# No checkpointing: O(n) memory, O(n) compute
def train_step_no_remat(params, x, y):
    ...

# Full checkpointing: O(1) memory, O(2n) compute
@functools.partial(jax.remat, policy=jax.checkpoint_policies.nothing_saveable)
def train_step_full_remat(params, x, y):
    ...

# Selective: O(sqrt(n)) memory, O(n + sqrt(n)) compute
@jax.remat
def train_step_selective(params, x, y):
    ...
```

### Checking memory usage

```python
# Profile memory
with jax.profiler.trace("/tmp/profile"):
    grads = jax.grad(loss)(params, x, y)

# Check device memory
print(jax.devices()[0].memory_stats())
```

---

## 9. API Reference

```python
jax.checkpoint(
    fun: Callable,
    *,
    policy: Callable | None = None,     # What to save vs recompute
    prevent_c: bool = True,             # Prevent CSE on remat'd values
    static_argnums: int | tuple = (),   # Static arguments
    boundary: bool = False,             # Boundary for DCE
    remat_mode: str = ...,
) -> Callable

jax.remat = jax.checkpoint  # Alias
```

### Policy reference

```python
jax.checkpoint_policies.everything_saveable       # No remat
jax.checkpoint_policies.nothing_saveable          # Full remat
jax.checkpoint_policies.dots_saveable              # Save matmuls
jax.checkpoint_policies.dots_with_no_batch_dims_saveable
jax.checkpoint_policies.checkpoint_dots            # Checkpoint around dots
jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims
jax.checkpoint_policies.save_any_names_here(names) # Custom name filter
jax.checkpoint_policies.save_from_both_policies(p1, p2)  # Union
jax.checkpoint_policies.drop_extra Cecil Saveable  # Save + drop extras
```
