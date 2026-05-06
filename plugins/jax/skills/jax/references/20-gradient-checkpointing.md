# Gradient Checkpointing (jax.checkpoint / jax.remat)

This document provides an exhaustive reference for JAX's gradient checkpointing system, also known as rematerialization. Gradient checkpointing trades compute for memory by recomputing intermediate values during the backward pass instead of storing them, enabling training of larger models.

---

## Table of Contents

1. [Overview](#1-overview)
2. [jax.checkpoint / jax.remat API](#2-jaxcheckpoint--jaxremat-api)
3. [Policy Functions](#3-policy-functions)
4. [everything_saveable](#4-everything_saveable)
5. [nothing_saveable](#5-nothing_saveable)
6. [dots_saveable](#6-dots_saveable)
7. [dots_with_no_batch_dims_saveable](#7-dots_with_no_batch_dims_saveable)
8. [save_anything_except_these_names](#8-save_anything_except_these_names)
9. [save_only_these_names](#9-save_only_these_names)
10. [save_from_both_policies](#10-save_from_both_policies)
11. [checkpoint_name](#11-checkpoint_name)
12. [Offloading Support](#12-offloading-support)
13. [saved_residuals and print_saved_residuals](#13-saved_residuals-and-print_saved_residuals)
14. [Memory Analysis Examples](#14-memory-analysis-examples)
15. [Composition with jit, grad, vmap](#15-composition-with-jit-grad-vmap)
16. [Complete Examples](#16-complete-examples)

---

## 1. Overview

During backpropagation, JAX must store intermediate values (residuals) computed during the forward pass so they can be used to compute gradients. For deep or wide networks, these residuals can consume a large amount of memory -- often more than the model parameters themselves.

**Gradient checkpointing** (rematerialization) addresses this by:
1. Not saving certain intermediate values during the forward pass.
2. Recomputing them on-the-fly during the backward pass when needed.

This trades compute time for memory, allowing you to train models that would otherwise not fit in GPU/TPU memory.

```
Without checkpointing:
  Forward:  save a, save b, save c, save d, save e
  Backward: use e, use d, use c, use b, use a
  Memory:   O(depth * width)  -- all intermediates stored

With checkpointing (every layer):
  Forward:  save a, recompute b, recompute c, recompute d, save e
  Backward: use e, recompute d, recompute c, recompute b, use a
  Memory:   O(sqrt(depth * width))  -- fewer intermediates stored
```

```python
import jax
import jax.numpy as jnp

# The problem: deep networks store many intermediates
def deep_network(params, x, num_layers=20):
    h = x
    intermediates = []  # All of these would be saved for backward!
    for w, b in params:
        h = jnp.dot(h, w) + b
        h = jax.nn.relu(h)
        intermediates.append(h)
    return h

# With checkpointing: wrap each layer
def deep_network_checkpointed(params, x, num_layers=20):
    h = x
    for w, b in params:
        # jax.checkpoint prevents saving h for backward
        # It recomputes h during the backward pass
        h = jax.checkpoint(lambda h, w=w, b=b: jax.nn.relu(jnp.dot(h, w) + b))(h)
    return h
```

---

## 2. jax.checkpoint / jax.remat API

`jax.checkpoint` and `jax.remat` are aliases for the same function. `jax.checkpoint` is the newer, preferred name.

### API Signature

```python
jax.checkpoint(
    fun,                           # Function to checkpoint
    policy=None,                   # Which residuals to save vs recompute
    prevent_cse=True,             # Prevent common subexpression elimination
    static_argnums=(),            # Arguments treated as static (compile-time constants)
    boundary=None,                # Boundary for rematerialization scope
    offload=None,                 # Offloading configuration (e.g., to CPU)
    out_shardings=None,           # Output shardings hint
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fun` | Callable | required | The function to wrap with checkpointing. |
| `policy` | Callable or None | `None` | A policy function that decides which operations' outputs to save. `None` means save nothing (recompute everything). |
| `prevent_cse` | bool | `True` | Prevent CSE within the checkpointed region, which can affect rematerialization behavior. |
| `static_argnums` | tuple | `()` | Indices of arguments that are static (trace-time constants). These are not treated as JAX arrays. |
| `boundary` | str or None | `None` | Controls the scope of rematerialization. |
| `offload` | bool or OffloadConfig | `None` | Whether to offload saved residuals to CPU memory. |

### Basic Usage

```python
import jax
import jax.numpy as jnp

# Without checkpointing
def layer(x, w, b):
    h = jnp.dot(x, w) + b
    return jax.nn.relu(h)

# With checkpointing
checkpointed_layer = jax.checkpoint(layer)

# Or use as decorator
@jax.checkpoint
def checkpointed_layer2(x, w, b):
    h = jnp.dot(x, w) + b
    return jax.nn.relu(h)

# jax.remat is an alias
remat_layer = jax.remat(layer)

# All three produce the same result
x = jnp.ones((32, 64))
w = jnp.ones((64, 128))
b = jnp.zeros(128)

result1 = layer(x, w, b)
result2 = checkpointed_layer(x, w, b)
result3 = remat_layer(x, w, b)

assert jnp.allclose(result1, result2)
assert jnp.allclose(result1, result3)
```

### Checkpointing a Multi-Layer Network

```python
import jax
import jax.numpy as jnp

def make_layers(key, input_dim, hidden_dim, output_dim, num_layers):
    params = []
    for i in range(num_layers):
        k1, k2, key = jax.random.split(key, 3)
        if i == 0:
            w = jax.random.normal(k1, (input_dim, hidden_dim)) * 0.01
        elif i == num_layers - 1:
            w = jax.random.normal(k1, (hidden_dim, output_dim)) * 0.01
        else:
            w = jax.random.normal(k1, (hidden_dim, hidden_dim)) * 0.01
        b = jnp.zeros(w.shape[-1])
        params.append((w, b))
    return params

@jax.checkpoint
def forward_one_layer(x, w, b):
    return jax.nn.relu(jnp.dot(x, w) + b)

def forward_no_checkpoint(params, x):
    h = x
    for w, b in params:
        h = jax.nn.relu(jnp.dot(h, w) + b)
    return h

def forward_full_checkpoint(params, x):
    h = x
    for w, b in params:
        h = forward_one_layer(h, w, b)  # Each layer is checkpointed
    return h

def forward_selective_checkpoint(params, x, checkpoint_every=2):
    h = x
    for i, (w, b) in enumerate(params):
        if i % checkpoint_every == 0:
            h = forward_one_layer(h, w, b)  # Checkpoint every N layers
        else:
            h = jax.nn.relu(jnp.dot(h, w) + b)  # Normal
    return h

# Compare
key = jax.random.key(0)
params = make_layers(key, 128, 256, 10, 8)
x = jnp.ones((64, 128))

out_no_cp = forward_no_checkpoint(params, x)
out_full_cp = forward_full_checkpoint(params, x)
assert jnp.allclose(out_no_cp, out_full_cp, atol=1e-5)
```

---

## 3. Policy Functions

Policy functions control which residuals (intermediate values) are saved during the forward pass and which are recomputed during the backward pass. They are callables that take an operation name and return `True` (save) or `False` (recompute).

### Policy Function Interface

```python
import jax.checkpoint_policies as cp

# A policy function has this signature:
# policy(call_type, name_jaxpr, eqn) -> bool | str
# Where:
#   - call_type: the type of the operation
#   - name_jaxpr: the JAX expression information
#   - eqn: the XLA equation (operation)
#
# Returns True to SAVE the residual, False to RECOMPUTE it
```

### Available Policy Functions

| Policy | Behavior |
|--------|----------|
| `everything_saveable` | Save all residuals (no checkpointing) |
| `nothing_saveable` | Recompute all residuals (maximum memory savings) |
| `dots_saveable` | Save dot products (matmuls), recompute everything else |
| `dots_with_no_batch_dims_saveable` | Save weight-only matmuls, recompute batched matmuls and everything else |
| `save_anything_except_these_names(names)` | Recompute named operations, save everything else |
| `save_only_these_names(names)` | Save named operations, recompute everything else |
| `save_from_both_policies(p1, p2)` | Save if either policy says to save |

### Using Policies

```python
import jax
import jax.numpy as jnp
import jax.checkpoint_policies as cp

# Default (no policy): recompute everything inside the checkpoint boundary
@jax.checkpoint
def fn_default(x, w):
    h = jnp.dot(x, w)
    return jax.nn.relu(h)

# Save everything (effectively no checkpointing)
@jax.checkpoint(policy=cp.everything_saveable)
def fn_save_all(x, w):
    h = jnp.dot(x, w)
    return jax.nn.relu(h)

# Save only matmuls
@jax.checkpoint(policy=cp.dots_saveable)
def fn_save_dots(x, w):
    h = jnp.dot(x, w)
    return jax.nn.relu(h)

# Save nothing (recompute everything)
@jax.checkpoint(policy=cp.nothing_saveable)
def fn_save_nothing(x, w):
    h = jnp.dot(x, w)
    return jax.nn.relu(h)

x = jnp.ones((32, 64))
w = jnp.ones((64, 32))
# All produce the same forward result, but different memory/compute tradeoffs
```

---

## 4. everything_saveable

`everything_saveable` saves all residuals. This effectively disables checkpointing inside the wrapped function, making it equivalent to not using `jax.checkpoint` at all.

```python
import jax
import jax.numpy as jnp
import jax.checkpoint_policies as cp

@jax.checkpoint(policy=cp.everything_saveable)
def layer(x, w, b):
    """All intermediates are saved for backward."""
    h = jnp.dot(x, w)
    h = h + b
    h = jax.nn.relu(h)
    h = jnp.dot(h, w.T)
    return h

# Memory behavior: same as without checkpoint
x = jnp.ones((32, 512))
w = jnp.ones((512, 256))
b = jnp.zeros(256)

# Useful for debugging or for nested checkpointing where
# the outer checkpoint handles rematerialization
def nested_example(params, x):
    # Outer checkpoint: only saves input to this block
    @jax.checkpoint
    def block(x):
        # Inner: save everything (don't recompute within the block)
        for w, b in params:
            x = jax.checkpoint(
                lambda x, w=w, b=b: jax.nn.relu(jnp.dot(x, w) + b),
                policy=cp.everything_saveable
            )(x)
        return x
    return block(x)
```

### When to Use everything_saveable

1. **Debugging:** Compare memory/compute with and without checkpointing.
2. **Nested checkpointing:** Outer checkpoint handles rematerialization of the block boundary; inner uses `everything_saveable`.
3. **Selective application:** Use in parts of the network where recomputation is more expensive than storage.

---

## 5. nothing_saveable

`nothing_saveable` recomputes all residuals during the backward pass. This provides maximum memory savings but at the cost of additional compute.

```python
import jax
import jax.numpy as jnp
import jax.checkpoint_policies as cp

@jax.checkpoint(policy=cp.nothing_saveable)
def expensive_layer(x, w, b):
    """All intermediates are recomputed during backward."""
    # Forward: compute but don't save anything
    h = jnp.dot(x, w)        # Recomputed during backward
    h = h + b                 # Recomputed during backward
    h = jax.nn.gelu(h)       # Recomputed during backward
    h = jnp.dot(h, w.T)      # Recomputed during backward
    h = h + b                 # Recomputed during backward
    return h

# This is the DEFAULT behavior when no policy is specified
@jax.checkpoint  # Equivalent to policy=cp.nothing_saveable
def same_as_default(x, w, b):
    h = jnp.dot(x, w) + b
    return jax.nn.gelu(h)

# Memory savings example
def compare_memory():
    key = jax.random.key(0)
    x = jnp.ones((128, 1024))
    w1 = jax.random.normal(key, (1024, 2048)) * 0.01
    w2 = jax.random.normal(key, (2048, 1024)) * 0.01

    # Without checkpointing
    def fn_no_ckpt(x):
        h = jnp.dot(x, w1)
        h = jax.nn.gelu(h)
        h = jnp.dot(h, w2)
        return jnp.sum(h)

    grad_no_ckpt = jax.jit(jax.grad(fn_no_ckpt))

    # With checkpointing (nothing saveable)
    @jax.checkpoint
    def gelu_block(x):
        h = jnp.dot(x, w1)
        h = jax.nn.gelu(h)
        h = jnp.dot(h, w2)
        return h

    def fn_ckpt(x):
        return jnp.sum(gelu_block(x))

    grad_ckpt = jax.jit(jax.grad(fn_ckpt))

    # Both produce the same gradients
    g1 = grad_no_ckpt(x)
    g2 = grad_ckpt(x)
    assert jnp.allclose(g1, g2, atol=1e-4)
```

### Cost Analysis: nothing_saveable

For a function with `N` layers, each with cost `C`:
- **Without checkpointing:** Memory = O(N * C), Compute = O(N * C)
- **With nothing_saveable:** Memory = O(sqrt(N) * C), Compute = O(N * sqrt(N) * C)

The compute overhead is roughly sqrt(N) extra forward passes.

---

## 6. dots_saveable

`dots_saveable` saves the outputs of dot product (matrix multiplication) operations but recomputes everything else. This is useful because matrix multiplications are typically the most expensive operations, and their outputs are relatively compact compared to element-wise operations.

```python
import jax
import jax.numpy as jnp
import jax.checkpoint_policies as cp

@jax.checkpoint(policy=cp.dots_saveable)
def transformer_ffn(x, w1, b1, w2, b2):
    """Feed-forward network with dot products saved."""
    # dot_general (matmul): SAVED
    h = jnp.dot(x, w1) + b1
    # Activation (element-wise): RECOMPUTED during backward
    h = jax.nn.gelu(h)
    # dot_general (matmul): SAVED
    h = jnp.dot(h, w2) + b2
    # More element-wise ops: RECOMPUTED
    h = jax.nn.dropout(h, rate=0.1)
    return h

x = jnp.ones((32, 512))
w1 = jnp.ones((512, 2048))
b1 = jnp.zeros(2048)
w2 = jnp.ones((2048, 512))
b2 = jnp.zeros(512)

result = transformer_ffn(x, w1, b1, w2, b2)
grad = jax.grad(lambda x: jnp.sum(transformer_ffn(x, w1, b1, w2, b2)))(x)
```

### Why dots_saveable Is Often the Best Policy

1. **Matmuls are the bottleneck:** Saving their outputs avoids expensive recomputation.
2. **Activations are cheap to recompute:** Element-wise operations like ReLU, GELU, LayerNorm are computationally cheap but can produce large output tensors.
3. **Good balance:** Provides meaningful memory savings without a large compute overhead.

```python
import jax
import jax.numpy as jnp
import jax.checkpoint_policies as cp

def make_transformer_block(dim, hidden_dim, num_heads):
    """Create a transformer block with dots_saveable checkpointing."""
    @jax.checkpoint(policy=cp.dots_saveable)
    def block(x, wq, wk, wv, wo, w1, b1, w2, b2):
        seq_len, _ = x.shape
        head_dim = dim // num_heads

        # Self-attention (matmuls are saved)
        q = jnp.dot(x, wq).reshape(seq_len, num_heads, head_dim)
        k = jnp.dot(x, wk).reshape(seq_len, num_heads, head_dim)
        v = jnp.dot(x, wv).reshape(seq_len, num_heads, head_dim)

        # Attention computation (element-wise: recomputed)
        scores = jnp.einsum('qhd,khd->hqk', q, k) / jnp.sqrt(head_dim)
        weights = jax.nn.softmax(scores, axis=-1)
        attn_out = jnp.einsum('hqk,khd->qhd', weights, v)
        attn_out = attn_out.reshape(seq_len, dim)
        proj = jnp.dot(attn_out, wo)

        # Residual connection + layer norm (recomputed)
        x = jax.nn.layer_norm(x + proj)

        # FFN (matmuls saved, activations recomputed)
        h = jnp.dot(x, w1) + b1
        h = jax.nn.gelu(h)
        h = jnp.dot(h, w2) + b2

        return jax.nn.layer_norm(x + h)

    return block
```

---

## 7. dots_with_no_batch_dims_saveable

`dots_with_no_batch_dims_saveable` is a more selective version of `dots_saveable`. It only saves matrix multiplications that have no batch dimensions -- typically weight-only matmuls (not batched matmuls). This saves memory for large-batch scenarios.

```python
import jax
import jax.numpy as jnp
import jax.checkpoint_policies as cp

@jax.checkpoint(policy=cp.dots_with_no_batch_dims_saveable)
def attention_layer(x, wq, wk, wv, wo):
    """Attention with only weight matmuls saved (not batched)."""
    seq_len, dim = x.shape

    # These are batched matmuls (have contracting batch dims):
    # q = jnp.dot(x, wq) -- actually this has no batch dims in the dot
    # So these ARE saved:
    q = jnp.dot(x, wq)
    k = jnp.dot(x, wk)
    v = jnp.dot(x, wv)

    # Batched dot product for attention scores:
    # This has batch dims, so NOT saved
    scores = jnp.einsum('sd,sd->s', q, k)  # Example batched op

    return jnp.dot(v, wo)

# The key distinction:
# - jnp.dot(x, w) where x is (batch, in), w is (in, out): no batch dims in dot -> SAVED
# - jnp.einsum('bsi,bsj->bij', a, b): has batch dims -> NOT SAVED
# - Convolution: has batch dims -> NOT SAVED
```

### When to Use dots_with_no_batch_dims_saveable

```python
import jax
import jax.numpy as jnp
import jax.checkpoint_policies as cp

# This policy is useful when batched operations produce very large outputs
# but weight-only operations produce smaller outputs

# Example: ViT (Vision Transformer)
@jax.checkpoint(policy=cp.dots_with_no_batch_dims_saveable)
def vit_layer(x, wq, wk, wv, wo, w1, w2, ln1_w, ln1_b, ln2_w, ln2_b):
    """ViT layer where only weight matmuls are saved."""
    batch, seq, dim = x.shape

    # Layer norm (recomputed -- cheap)
    x_norm = jax.nn.layer_norm(x, -1, (ln1_w, ln1_b))

    # Weight matmuls (saved -- these are not batched)
    q = jnp.dot(x_norm, wq)
    k = jnp.dot(x_norm, wk)
    v = jnp.dot(x_norm, wv)

    # Attention (batched operations -- recomputed)
    # einsum with batch dim is NOT saved
    q = q.reshape(batch, seq, 8, -1).transpose(0, 2, 1, 3)
    k = k.reshape(batch, seq, 8, -1).transpose(0, 2, 1, 3)
    v = v.reshape(batch, seq, 8, -1).transpose(0, 2, 1, 3)

    scores = jnp.matmul(q, k.transpose(0, 1, 3, 2))  # Batched: NOT saved
    weights = jax.nn.softmax(scores / jnp.sqrt(64))
    attn = jnp.matmul(weights, v)  # Batched: NOT saved

    attn = attn.transpose(0, 2, 1, 3).reshape(batch, seq, dim)
    proj = jnp.dot(attn, wo)  # Weight matmul: SAVED

    x = x + proj
    x_norm = jax.nn.layer_norm(x, -1, (ln2_w, ln2_b))

    h = jnp.dot(x_norm, w1)  # Weight matmul: SAVED
    h = jax.nn.gelu(h)       # Recomputed
    h = jnp.dot(h, w2)       # Weight matmul: SAVED

    return x + h
```

---

## 8. save_anything_except_these_names

`save_anything_except_these_names` saves all residuals except those produced by operations with the specified names. This lets you selectively recompute expensive-to-store but cheap-to-compute operations.

```python
import jax
import jax.numpy as jnp
import jax.checkpoint_policies as cp

# Recompute only activation functions, save everything else
@jax.checkpoint(policy=cp.save_anything_except_these_names("relu", "gelu"))
def layer_with_recomputed_activations(x, w, b):
    h = jnp.dot(x, w)  # SAVED
    h = h + b           # SAVED
    h = jax.nn.relu(h)  # RECOMPUTED (name contains "relu")
    return h

# Recompute only normalization operations
@jax.checkpoint(policy=cp.save_anything_except_these_names(
    "reduce_window",  # Used in layer norm
))
def layer_with_recomputed_norm(x, w, b, ln_w, ln_b):
    h = jax.nn.layer_norm(x, -1, (ln_w, ln_b))  # RECOMPUTED
    h = jnp.dot(h, w) + b                        # SAVED
    return h

# Multiple names to recompute
@jax.checkpoint(policy=cp.save_anything_except_these_names(
    "relu", "exp", "log", "sigmoid"
))
def custom_layer(x, w, b):
    h = jnp.dot(x, w) + b  # SAVED
    h = jax.nn.relu(h)     # RECOMPUTED
    h = jnp.exp(h)         # RECOMPUTED
    h = jnp.log(h + 1e-8)  # RECOMPUTED
    return h
```

### Finding Operation Names

```python
import jax
import jax.numpy as jnp

# To find the operation names in your function, inspect the HLO:
@jax.jit
def my_layer(x, w, b):
    h = jnp.dot(x, w) + b
    h = jax.nn.gelu(h)
    h = jnp.dot(h, w.T)
    return h

x = jnp.ones((32, 64))
w = jnp.ones((64, 32))
b = jnp.zeros(32)

# Lower and view the HLO
lowered = jax.jit(my_layer).lower(x, w, b)
hlo_text = lowered.as_text()

# Look for operation names in the HLO:
# "dot.{{.*}}" -> dot products
# "max" -> used in ReLU
# "exp" -> used in GELU
# "reduce_window" -> used in LayerNorm
# "convolve" -> convolutions
```

### Practical Example: Recompute Only Expensive-to-Store Operations

```python
import jax
import jax.numpy as jnp
import jax.checkpoint_policies as cp

# Large intermediate tensors from broadcasting are expensive to store
# but cheap to recompute
@jax.checkpoint(policy=cp.save_anything_except_these_names("broadcast_in_dim"))
def layer_with_selective_recompute(x, w, b):
    h = jnp.dot(x, w)   # SAVED
    h = h + b            # SAVED (broadcast is recomputed)
    h = jax.nn.relu(h)   # SAVED
    return h
```

---

## 9. save_only_these_names

`save_only_these_names` is the inverse of `save_anything_except_these_names`: it saves only the residuals from operations with the specified names and recomputes everything else.

```python
import jax
import jax.numpy as jnp
import jax.checkpoint_policies as cp

# Save only matmul outputs, recompute everything else
@jax.checkpoint(policy=cp.save_only_these_names("dot_general"))
def layer_save_matmuls(x, w1, b1, w2, b2):
    h = jnp.dot(x, w1)   # SAVED (dot_general)
    h = h + b1            # RECOMPUTED
    h = jax.nn.gelu(h)   # RECOMPUTED
    h = jnp.dot(h, w2)   # SAVED (dot_general)
    h = h + b2            # RECOMPUTED
    return h

# Save only specific expensive-to-recompute operations
@jax.checkpoint(policy=cp.save_only_these_names(
    "dot_general", "conv_general_dilated"
))
def conv_block(x, w_conv, w_linear, b):
    h = jax.lax.conv_general_dilated(
        x, w_conv,
        window_strides=(1, 1),
        padding='SAME',
        dimension_numbers=('NCHW', 'OIHW', 'NCHW')
    )  # SAVED (conv_general_dilated)
    h = jax.nn.relu(h)    # RECOMPUTED
    h = jnp.dot(h.flatten(), w_linear) + b  # SAVED (dot_general)
    return h
```

### Combining with Operation Inspection

```python
import jax
import jax.numpy as jnp
import jax.checkpoint_policies as cp

def print_operations(fn, *args):
    """Print the operations in a function to help choose save names."""
    lowered = jax.jit(fn).lower(*args)
    text = lowered.as_text()

    # Extract operation names
    import re
    ops = set()
    for line in text.split('\n'):
        # HLO instructions look like: %result = op_name(...)
        match = re.search(r'= (\w+(?:_\w+)*)\(', line)
        if match:
            ops.add(match.group(1))

    print("Operations found:")
    for op in sorted(ops):
        print(f"  - {op}")
    return ops

def my_layer(x, w, b):
    h = jnp.dot(x, w) + b
    h = jax.nn.layer_norm(h, -1)
    h = jax.nn.gelu(h)
    return h

ops = print_operations(my_layer, jnp.ones((32, 64)), jnp.ones((64, 32)), jnp.zeros(32))
```

---

## 10. save_from_both_policies

`save_from_both_policies` combines two policies: a residual is saved if **either** policy says to save it. This allows composing fine-grained policies.

```python
import jax
import jax.numpy as jnp
import jax.checkpoint_policies as cp

# Combine: save matmuls OR save operations named "custom_op"
combined_policy = cp.save_from_both_policies(
    cp.dots_saveable,
    cp.save_only_these_names("conv_general_dilated")
)

@jax.checkpoint(policy=combined_policy)
def mixed_layer(x, w, b, conv_w):
    # dot product: saved by dots_saveable
    h = jnp.dot(x, w) + b

    # convolution: saved by save_only_these_names
    h = jax.lax.conv_general_dilated(
        h.reshape(1, *h.shape, 1),
        conv_w,
        window_strides=(1, 1),
        padding='SAME',
        dimension_numbers=('NCHW', 'OIHW', 'NCHW')
    ).flatten()

    # element-wise: NOT saved by either policy -> RECOMPUTED
    h = jax.nn.gelu(h)
    return h
```

### Combining Multiple Policies

```python
import jax
import jax.numpy as jnp
import jax.checkpoint_policies as cp

# Save: matmuls AND convolutions AND specific named operations
# Use nested save_from_both_policies
full_policy = cp.save_from_both_policies(
    cp.dots_saveable,
    cp.save_from_both_policies(
        cp.save_only_these_names("conv_general_dilated"),
        cp.save_only_these_names("sort")
    )
)

# Alternatively, define a custom policy function
def my_custom_policy(*args, **kwargs):
    """Save matmuls, convolutions, and batch norms."""
    dots_result = cp.dots_saveable(*args, **kwargs)
    conv_result = cp.save_only_these_names("conv_general_dilated")(*args, **kwargs)
    # If either says save, we save
    if dots_result or conv_result:
        return True
    return False
```

### Practical Combined Policy

```python
import jax
import jax.numpy as jnp
import jax.checkpoint_policies as cp

# For a transformer model: save matmuls and layer norms, recompute activations
def transformer_policy(call_type, name_jaxpr, eqn):
    """Save matmuls and reduce operations (used in layernorm), recompute activations."""
    # Check dots policy
    if cp.dots_saveable(call_type, name_jaxpr, eqn):
        return True
    # Check for specific operation names
    name = str(eqn.primitive) if hasattr(eqn, 'primitive') else ''
    if 'reduce' in name:
        return True
    return False
```

---

## 11. checkpoint_name

`checkpoint_name` assigns a name to a specific intermediate value within a checkpointed function, allowing policy functions to refer to it by name.

### Basic Usage

```python
import jax
import jax.numpy as jnp
from jax.checkpoint_policies import checkpoint_name

@jax.checkpoint
def named_layer(x, w1, b1, w2, b2):
    # Name intermediate values for policy reference
    h = jnp.dot(x, w1) + b1
    h = checkpoint_name(jax.nn.gelu(h), "activation_1")
    h = jnp.dot(h, w2) + b2
    h = checkpoint_name(jax.nn.relu(h), "activation_2")
    return h

# Now use a policy that references these names
import jax.checkpoint_policies as cp

@jax.checkpoint(policy=cp.save_anything_except_these_names("activation_1", "activation_2"))
def named_layer_selective(x, w1, b1, w2, b2):
    h = jnp.dot(x, w1) + b1
    h = checkpoint_name(jax.nn.gelu(h), "activation_1")
    h = jnp.dot(h, w2) + b2
    h = checkpoint_name(jax.nn.relu(h), "activation_2")
    return h
```

### Named Checkpoints in a Transformer

```python
import jax
import jax.numpy as jnp
from jax.checkpoint_policies import checkpoint_name

@jax.checkpoint
def transformer_block(x, params):
    wq, wk, wv, wo = params["wq"], params["wk"], params["wv"], params["wo"]
    w1, b1, w2, b2 = params["w1"], params["b1"], params["w2"], params["b2"]

    # Attention
    q = checkpoint_name(jnp.dot(x, wq), "query_proj")
    k = checkpoint_name(jnp.dot(x, wk), "key_proj")
    v = checkpoint_name(jnp.dot(x, wv), "value_proj")

    scores = jnp.dot(q, k.T) / jnp.sqrt(q.shape[-1])
    weights = checkpoint_name(jax.nn.softmax(scores, axis=-1), "attn_weights")
    attn_out = jnp.dot(weights, v)
    proj = checkpoint_name(jnp.dot(attn_out, wo), "attn_proj")

    x = jax.nn.layer_norm(x + proj)

    # FFN
    h = checkpoint_name(jnp.dot(x, w1) + b1, "ffn_hidden")
    h = checkpoint_name(jax.nn.gelu(h), "ffn_activation")
    h = checkpoint_name(jnp.dot(h, w2) + b2, "ffn_output")

    return jax.nn.layer_norm(x + h)

# Now you can use targeted policies:
# Save all projections, recompute activations
policy = cp.save_anything_except_these_names("attn_weights", "ffn_activation")
```

---

## 12. Offloading Support

JAX supports offloading saved residuals to CPU memory (or other devices) during the forward pass and fetching them back during the backward pass. This is useful when GPU memory is limited but CPU memory is abundant.

### Basic Offloading

```python
import jax
import jax.numpy as jnp

# Enable offloading of residuals to CPU
@jax.checkpoint(offload=True)
def layer_with_offload(x, w, b):
    h = jnp.dot(x, w) + b
    h = jax.nn.gelu(h)
    return h

# The residuals from this layer are stored in CPU memory
# during the forward pass and transferred back during backward
```

### Offloading Configuration

```python
import jax
import jax.numpy as jnp

# Offloading with specific configuration
from jax._src.checkpoint import OffloadConfig

config = OffloadConfig(
    # Offload to CPU memory
    # The actual API may vary by JAX version
)

@jax.checkpoint(offload=True)
def offloaded_layer(x, w, b):
    h = jnp.dot(x, w) + b
    h = jax.nn.gelu(h)
    return h
```

### Offloading with Policy

```python
import jax
import jax.numpy as jnp
import jax.checkpoint_policies as cp

# Combine offloading with a policy:
# Save matmuls (to CPU), recompute activations
@jax.checkpoint(
    policy=cp.dots_saveable,
    offload=True
)
def layer_offload_policy(x, w1, b1, w2, b2):
    h = jnp.dot(x, w1) + b1  # Saved, offloaded to CPU
    h = jax.nn.gelu(h)        # Recomputed (not saved)
    h = jnp.dot(h, w2) + b2  # Saved, offloaded to CPU
    h = jax.nn.relu(h)        # Recomputed (not saved)
    return h
```

### Offloading in a Deep Network

```python
import jax
import jax.numpy as jnp

def deep_network_with_offload(params, x, offload_every=4):
    h = x
    for i, (w, b) in enumerate(params):
        if i % offload_every == 0:
            # Every N layers, checkpoint with offloading
            @jax.checkpoint(offload=True)
            def checkpointed_block(h, w=w, b=b):
                return jax.nn.relu(jnp.dot(h, w) + b)
            h = checkpointed_block(h)
        else:
            h = jax.nn.relu(jnp.dot(h, w) + b)
    return h

# This approach:
# 1. Reduces GPU memory by offloading to CPU
# 2. Reduces recomputation by only checkpointing every N layers
# 3. Has PCIe transfer overhead for CPU <-> GPU communication
```

---

## 13. saved_residuals and print_saved_residuals

JAX provides tools to inspect which residuals are actually saved, helping you verify that your policy is working as expected.

### saved_residuals

`jax.checkpoint_policies.saved_residuals` can be used to examine what a specific policy would save for a given function.

```python
import jax
import jax.numpy as jnp
import jax.checkpoint_policies as cp

@jax.checkpoint(policy=cp.dots_saveable)
def my_layer(x, w1, b1, w2, b2):
    h = jnp.dot(x, w1) + b1
    h = jax.nn.gelu(h)
    h = jnp.dot(h, w2) + b2
    return h

# After running with grad, JAX internally tracks saved residuals
x = jnp.ones((32, 64))
w1 = jnp.ones((64, 128))
b1 = jnp.zeros(128)
w2 = jnp.ones((128, 32))
b2 = jnp.zeros(32)

# Run forward + backward
grad_fn = jax.grad(lambda x: jnp.sum(my_layer(x, w1, b1, w2, b2)))
grad = grad_fn(x)
```

### print_saved_residuals

`print_saved_residuals` prints a detailed report of what residuals are saved vs recomputed for a checkpointed function. This is extremely useful for debugging policies.

```python
import jax
import jax.numpy as jnp
import jax.checkpoint_policies as cp
from jax.ad_checkpoint import print_saved_residuals

@jax.checkpoint(policy=cp.dots_saveable)
def my_layer(x, w1, b1, w2, b2):
    h = jnp.dot(x, w1) + b1
    h = jax.nn.gelu(h)
    h = jnp.dot(h, w2) + b2
    return h

x = jnp.ones((2, 4))
w1 = jnp.ones((4, 8))
b1 = jnp.zeros(8)
w2 = jnp.ones((8, 4))
b2 = jnp.zeros(4)

# Print what's saved vs recomputed
loss_fn = lambda x: jnp.sum(my_layer(x, w1, b1, w2, b2))
print_saved_residuals(loss_fn, x)
# Output shows which operations' outputs are saved and which are recomputed
```

### Inspecting Residuals for Different Policies

```python
import jax
import jax.numpy as jnp
import jax.checkpoint_policies as cp
from jax.ad_checkpoint import print_saved_residuals

def layer(x, w1, b1, w2, b2):
    h = jnp.dot(x, w1) + b1
    h = jax.nn.gelu(h)
    h = jnp.dot(h, w2) + b2
    return h

x = jnp.ones((2, 4))
w1 = jnp.ones((4, 8))
b1 = jnp.zeros(8)
w2 = jnp.ones((8, 4))
b2 = jnp.zeros(4)

print("=== Policy: nothing_saveable ===")
fn1 = lambda x: jnp.sum(jax.checkpoint(layer, policy=cp.nothing_saveable)(x, w1, b1, w2, b2))
print_saved_residuals(fn1, x)

print("\n=== Policy: dots_saveable ===")
fn2 = lambda x: jnp.sum(jax.checkpoint(layer, policy=cp.dots_saveable)(x, w1, b1, w2, b2))
print_saved_residuals(fn2, x)

print("\n=== Policy: everything_saveable ===")
fn3 = lambda x: jnp.sum(jax.checkpoint(layer, policy=cp.everything_saveable)(x, w1, b1, w2, b2))
print_saved_residuals(fn3, x)
```

---

## 14. Memory Analysis Examples

### Comparing Memory Usage Across Policies

```python
import jax
import jax.numpy as jnp
import jax.checkpoint_policies as cp
import gc

def measure_peak_memory(fn, *args):
    """Measure peak GPU memory usage of a function."""
    device = jax.devices()[0]

    # Clear memory
    gc.collect()
    result = fn(*args)
    if hasattr(result, 'block_until_ready'):
        result.block_until_ready()

    stats = device.memory_stats()
    return stats['peak_bytes_in_use'] if stats else 0

# Create a deep network
key = jax.random.key(0)
dim = 512
num_layers = 20
params = []
for i in range(num_layers):
    k1, k2, key = jax.random.split(key, 3)
    w = jax.random.normal(k1, (dim, dim)) * 0.01
    b = jnp.zeros(dim)
    params.append((w, b))

x = jnp.ones((64, dim))

# 1. No checkpointing
def forward_no_ckpt(params, x):
    h = x
    for w, b in params:
        h = jax.nn.gelu(jnp.dot(h, w) + b)
    return h

loss_no_ckpt = lambda x: jnp.sum(forward_no_ckpt(params, x))
grad_no_ckpt = jax.jit(jax.grad(loss_no_ckpt))
mem_no_ckpt = measure_peak_memory(grad_no_ckpt, x)

# 2. Full checkpointing (nothing_saveable)
@jax.checkpoint
def forward_full_ckpt(params, x):
    h = x
    for w, b in params:
        h = jax.nn.gelu(jnp.dot(h, w) + b)
    return h

loss_full_ckpt = lambda x: jnp.sum(forward_full_ckpt(params, x))
grad_full_ckpt = jax.jit(jax.grad(loss_full_ckpt))
mem_full_ckpt = measure_peak_memory(grad_full_ckpt, x)

# 3. Per-layer checkpointing
def forward_per_layer_ckpt(params, x):
    h = x
    for w, b in params:
        h = jax.checkpoint(
            lambda h, w=w, b=b: jax.nn.gelu(jnp.dot(h, w) + b)
        )(h)
    return h

loss_per_layer = lambda x: jnp.sum(forward_per_layer_ckpt(params, x))
grad_per_layer = jax.jit(jax.grad(loss_per_layer))
mem_per_layer = measure_peak_memory(grad_per_layer, x)

# 4. dots_saveable policy
def forward_dots_ckpt(params, x):
    h = x
    for w, b in params:
        h = jax.checkpoint(
            lambda h, w=w, b=b: jax.nn.gelu(jnp.dot(h, w) + b),
            policy=cp.dots_saveable
        )(h)
    return h

loss_dots = lambda x: jnp.sum(forward_dots_ckpt(params, x))
grad_dots = jax.jit(jax.grad(loss_dots))
mem_dots = measure_peak_memory(grad_dots, x)

print(f"No checkpointing:     {mem_no_ckpt / 1e9:.2f} GB")
print(f"Full checkpointing:   {mem_full_ckpt / 1e9:.2f} GB")
print(f"Per-layer checkpoint: {mem_per_layer / 1e9:.2f} GB")
print(f"dots_saveable policy: {mem_dots / 1e9:.2f} GB")
```

### Memory vs Compute Tradeoff Visualization

```python
import jax
import jax.numpy as jnp
import jax.checkpoint_policies as cp
import time

def benchmark_memory_and_time(grad_fn, x, warmup=3, runs=10):
    """Measure both memory usage and execution time."""
    device = jax.devices()[0]

    # Warmup
    for _ in range(warmup):
        result = grad_fn(x)
        result.block_until_ready()

    # Measure
    times = []
    peak_memories = []
    for _ in range(runs):
        start = time.perf_counter()
        result = grad_fn(x)
        result.block_until_ready()
        times.append(time.perf_counter() - start)

        stats = device.memory_stats()
        if stats:
            peak_memories.append(stats['peak_bytes_in_use'])

    import statistics
    return {
        "mean_time_ms": statistics.mean(times) * 1000,
        "median_time_ms": statistics.median(times) * 1000,
        "mean_peak_mem_gb": statistics.mean(peak_memories) / 1e9 if peak_memories else 0,
    }

# Run the benchmark for different configurations
key = jax.random.key(42)
dim = 256
x = jnp.ones((32, dim))

results = {}
for num_layers in [4, 8, 16, 32]:
    params = []
    k = key
    for _ in range(num_layers):
        k1, k2, k = jax.random.split(k, 3)
        params.append((jax.random.normal(k1, (dim, dim)) * 0.01, jnp.zeros(dim)))

    # No checkpointing
    def fwd_no_cp(params, x):
        h = x
        for w, b in params:
            h = jax.nn.gelu(jnp.dot(h, w) + b)
        return h

    grad_fn = jax.jit(jax.grad(lambda x: jnp.sum(fwd_no_cp(params, x))))
    results[f"no_cp_{num_layers}"] = benchmark_memory_and_time(grad_fn, x)

    # With per-layer checkpointing
    def fwd_cp(params, x):
        h = x
        for w, b in params:
            h = jax.checkpoint(lambda h, w=w, b=b: jax.nn.gelu(jnp.dot(h, w) + b))(h)
        return h

    grad_fn_cp = jax.jit(jax.grad(lambda x: jnp.sum(fwd_cp(params, x))))
    results[f"cp_{num_layers}"] = benchmark_memory_and_time(grad_fn_cp, x)

print(f"{'Config':<20} | {'Time (ms)':>10} | {'Memory (GB)':>12}")
print("-" * 50)
for name, r in results.items():
    print(f"{name:<20} | {r['median_time_ms']:>10.2f} | {r['mean_peak_mem_gb']:>12.3f}")
```

---

## 15. Composition with jit, grad, vmap

### Checkpointing with jit

```python
import jax
import jax.numpy as jnp

@jax.checkpoint
def layer(x, w, b):
    return jax.nn.relu(jnp.dot(x, w) + b)

# jit compiles the entire forward + backward graph including
# the rematerialization logic
@jax.jit
def train_step(params, x, y):
    def loss_fn(params):
        h = x
        for w, b in params:
            h = layer(h, w, b)
        return jnp.mean((h - y) ** 2)
    return jax.grad(loss_fn)(params)

key = jax.random.key(0)
params = [
    (jax.random.normal(key, (64, 32)) * 0.01, jnp.zeros(32)),
    (jax.random.normal(key, (32, 10)) * 0.01, jnp.zeros(10)),
]
x = jnp.ones((16, 64))
y = jnp.ones((16, 10))

grads = train_step(params, x, y)
```

### Checkpointing with vmap

```python
import jax
import jax.numpy as jnp

@jax.checkpoint
def layer(x, w, b):
    return jax.nn.relu(jnp.dot(x, w) + b)

# vmap over batch dimension
batched_layer = jax.vmap(layer, in_axes=(0, None, None))

x = jnp.ones((32, 64))
w = jnp.ones((64, 32))
b = jnp.zeros(32)

result = batched_layer(x, w, b)

# Per-example gradients with checkpointing
per_ex_grad = jax.vmap(
    jax.grad(lambda x, w, b: jnp.sum(layer(x, w, b))),
    in_axes=(0, None, None)
)
grads = per_ex_grad(x, w, b)
```

### Checkpointing with pmap

```python
import jax
import jax.numpy as jnp

@jax.checkpoint
def layer(x, w, b):
    return jax.nn.relu(jnp.dot(x, w) + b)

# pmap across devices with checkpointing
@jax.pmap
def batched_grad(params, x_batch, y_batch):
    def loss_fn(x, y):
        h = x
        for w, b in params:
            h = layer(h, w, b)
        return jnp.mean((h - y) ** 2)

    return jax.grad(loss_fn)(x_batch, y_batch)

# Usage with multiple devices (if available)
if jax.device_count() > 1:
    key = jax.random.key(0)
    params = [(jax.random.normal(key, (64, 32)) * 0.01, jnp.zeros(32))]

    n_devices = jax.device_count()
    x = jnp.ones((n_devices, 8, 64))  # (devices, batch_per_device, features)
    y = jnp.ones((n_devices, 8, 32))

    grads = batched_grad(params, x, y)
```

### Checkpointing inside scan

```python
import jax
import jax.numpy as jnp

# Using checkpoint with lax.scan for recurrent models
def rnn_with_checkpoint(params, x_sequence):
    @jax.checkpoint
    def step(h, x):
        w_h, w_x, b = params
        h_new = jnp.tanh(jnp.dot(h, w_h) + jnp.dot(x, w_x) + b)
        return h_new, h_new

    h0 = jnp.zeros(params[0].shape[0])
    final_h, all_h = jax.lax.scan(step, h0, x_sequence)
    return final_h

key = jax.random.key(0)
hidden_dim = 64
params = (
    jax.random.normal(key, (hidden_dim, hidden_dim)) * 0.01,
    jax.random.normal(key, (32, hidden_dim)) * 0.01,
    jnp.zeros(hidden_dim),
)

seq = jax.random.normal(key, (100, 32))
result = rnn_with_checkpoint(params, seq)
grad = jax.grad(lambda p: jnp.sum(rnn_with_checkpoint(p, seq)))(params)
```

---

## 16. Complete Examples

### Example 1: Training a Large MLP with Checkpointing

```python
import jax
import jax.numpy as jnp
import optax
import jax.checkpoint_policies as cp

# Architecture
input_dim = 784
hidden_dims = [2048, 2048, 2048, 2048, 1024, 1024, 512, 512]
output_dim = 10

# Initialize parameters
def init_params(key):
    params = []
    dims = [input_dim] + hidden_dims + [output_dim]
    for i in range(len(dims) - 1):
        k1, key = jax.random.split(key)
        w = jax.random.normal(k1, (dims[i], dims[i+1])) * 0.01
        b = jnp.zeros(dims[i+1])
        params.append((w, b))
    return params

# Forward with selective checkpointing
@jax.checkpoint(policy=cp.dots_saveable)
def forward_block(x, w1, b1, w2, b2):
    h = jax.nn.gelu(jnp.dot(x, w1) + b1)
    h = jnp.dot(h, w2) + b2
    return h

def forward(params, x):
    h = x
    # Process in blocks of 2 layers
    for i in range(0, len(params) - 1, 2):
        (w1, b1), (w2, b2) = params[i], params[i+1]
        h = forward_block(h, w1, b1, w2, b2)
        h = jax.nn.relu(h)
    # Final layer
    w, b = params[-1]
    return jnp.dot(h, w) + b

def loss_fn(params, x, y):
    logits = forward(params, x)
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))

# Training
key = jax.random.key(0)
params = init_params(key)
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# Training loop
for step in range(1000):
    key, k1, k2 = jax.random.split(key, 3)
    x = jax.random.normal(k1, (128, input_dim))
    y = jax.random.randint(k2, (128,), 0, output_dim)
    params, opt_state, loss = train_step(params, opt_state, x, y)
    if step % 100 == 0:
        print(f"Step {step}: loss = {loss:.4f}")
```

### Example 2: Transformer with Fine-Grained Checkpointing

```python
import jax
import jax.numpy as jnp
import jax.checkpoint_policies as cp
from jax.checkpoint_policies import checkpoint_name

@jax.checkpoint(policy=cp.dots_with_no_batch_dims_saveable)
def transformer_layer(x, params, num_heads=8):
    """Single transformer layer with selective checkpointing."""
    seq_len, dim = x.shape
    head_dim = dim // num_heads

    # Layer norm 1
    x_norm = jax.nn.layer_norm(x)

    # Self-attention
    q = jnp.dot(x_norm, params["wq"])
    k = jnp.dot(x_norm, params["wk"])
    v = jnp.dot(x_norm, params["wv"])

    q = q.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)
    k = k.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)
    v = v.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)

    scores = jnp.matmul(q, k.transpose(0, 2, 1)) / jnp.sqrt(head_dim)
    weights = jax.nn.softmax(scores, axis=-1)
    attn = jnp.matmul(weights, v)

    attn = attn.transpose(1, 0, 2).reshape(seq_len, dim)
    proj = jnp.dot(attn, params["wo"])
    x = x + proj

    # Layer norm 2 + FFN
    x_norm = jax.nn.layer_norm(x)
    h = jnp.dot(x_norm, params["w1"]) + params["b1"]
    h = jax.nn.gelu(h)
    h = jnp.dot(h, params["w2"]) + params["b2"]
    x = x + h

    return x

def transformer(params, x, num_layers=12):
    for i in range(num_layers):
        x = transformer_layer(x, params[f"layer_{i}"])
    return x

# Usage
key = jax.random.key(0)
seq_len, dim = 256, 512

params = {}
for i in range(12):
    k1, k2, k3, k4, k5, key = jax.random.split(key, 6)
    params[f"layer_{i}"] = {
        "wq": jax.random.normal(k1, (dim, dim)) * 0.02,
        "wk": jax.random.normal(k2, (dim, dim)) * 0.02,
        "wv": jax.random.normal(k3, (dim, dim)) * 0.02,
        "wo": jax.random.normal(k4, (dim, dim)) * 0.02,
        "w1": jax.random.normal(k5, (dim, dim * 4)) * 0.02,
        "b1": jnp.zeros(dim * 4),
        "w2": jax.random.normal(key, (dim * 4, dim)) * 0.02,
        "b2": jnp.zeros(dim),
    }

x = jnp.ones((seq_len, dim))
output = transformer(params, x)
loss = jnp.sum(output)
grads = jax.grad(lambda p: jnp.sum(transformer(p, x)))(params)
```

### Example 3: Memory-Efficient U-Net

```python
import jax
import jax.numpy as jnp
import jax.checkpoint_policies as cp

@jax.checkpoint(policy=cp.dots_saveable)
def conv_block(x, w, b):
    """Convolution + activation with checkpointed activations."""
    # Simple 1D conv for illustration
    h = jnp.dot(x, w) + b
    h = jax.nn.relu(h)
    return h

def unet_encoder(params, x, depth=4):
    """U-Net encoder with checkpointing at each level."""
    skips = []
    h = x
    for i in range(depth):
        h = conv_block(h, params[f"enc_{i}_1"], params[f"enc_b{i}_1"])
        h = conv_block(h, params[f"enc_{i}_2"], params[f"enc_b{i}_2"])
        skips.append(h)
        h = h[:, ::2]  # Downsample (stride-2)
    return h, skips

def unet_decoder(params, h, skips, depth=4):
    """U-Net decoder with checkpointing."""
    for i in reversed(range(depth)):
        h = jnp.concatenate([h[:, ::1], skips[i]], axis=-1)  # Upsample + skip
        h = conv_block(h, params[f"dec_{i}_1"], params[f"dec_b{i}_1"])
        h = conv_block(h, params[f"dec_{i}_2"], params[f"dec_b{i}_2"])
    return h

def unet(params, x):
    h, skips = unet_encoder(params, x)
    h = conv_block(h, params["bottleneck_w"], params["bottleneck_b"])
    h = unet_decoder(params, h, skips)
    return h

# Without checkpointing, all skip connections would be stored
# With checkpointing, skip connections are recomputed from the encoder inputs
```

---

## Summary

| Feature | Description | Memory Savings | Compute Cost |
|---------|-------------|---------------|--------------|
| `jax.checkpoint` (default) | Recompute all residuals | Maximum | Highest |
| `everything_saveable` | Save all residuals | None | Baseline |
| `nothing_saveable` | Recompute all residuals | Maximum | Highest |
| `dots_saveable` | Save matmuls, recompute activations | High | Moderate |
| `dots_with_no_batch_dims_saveable` | Save weight matmuls only | High | Moderate |
| `save_anything_except_these_names` | Selective recompute by name | Variable | Variable |
| `save_only_these_names` | Selective save by name | Variable | Variable |
| `save_from_both_policies` | Combine two policies | Variable | Variable |
| `checkpoint_name` | Name intermediates for policy targeting | N/A (metadata) | N/A |
| `offload=True` | Offload residuals to CPU | Frees GPU memory | PCIe transfer |
| `saved_residuals` | Inspect what is saved | N/A (debug) | N/A |
| `print_saved_residuals` | Print save/recompute report | N/A (debug) | N/A |

### Policy Selection Guide

| Scenario | Recommended Policy |
|----------|-------------------|
| Maximum memory savings, don't care about speed | `nothing_saveable` |
| Transformer training | `dots_saveable` or `dots_with_no_batch_dims_saveable` |
| CNN training | `dots_saveable` |
| Selective recompute of cheap ops | `save_anything_except_these_names` |
| Need precise control | `save_only_these_names` with `checkpoint_name` |
| GPU memory full, CPU available | `offload=True` with `dots_saveable` |
