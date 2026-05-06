# 32 - External Callbacks

## Overview

JAX provides several callback mechanisms for interacting with the outside world from within transformed code. These are essential for debugging, logging, and integration with non-JAX code.

---

## 1. Callback Types Overview

| Callback | Purpose | JIT-compatible | Differentiable | vmap-compatible |
|---|---|---|---|---|
| `jax.debug.print` | Print values | Yes | No | Yes |
| `jax.debug.callback` | Arbitrary Python | Yes | No | Yes |
| `jax.pure_callback` | Pure Python functions | Yes | No | Yes |
| `jax.experimental.io_callback` | Side-effecting I/O | Yes | No | Limited |

---

## 2. jax.debug.print

### Basic usage

```python
import jax
import jax.numpy as jnp

@jax.jit
def f(x):
    jax.debug.print("x = {}", x)
    y = x + 1
    jax.debug.print("y = {}", y)
    return y

f(jnp.array([1.0, 2.0, 3.0]))
# x = [1. 2. 3.]
# y = [2. 3. 4.]
```

### Multiple arguments

```python
jax.debug.print("x = {}, shape = {}", x, x.shape)
```

### Ordered printing

```python
# Guarantee order of output
jax.debug.print("step {}", i, ordered=True)
```

### Inside vmap

```python
@jax.vmap
def f(x):
    jax.debug.print("x = {}", x)
    return x + 1

f(jnp.arange(3))  # Prints 3 times, once per batch element
```

### Inside grad

```python
@jax.grad
def f(x):
    jax.debug.print("x = {}", x)  # Prints during forward pass only
    return jnp.sum(x ** 2)
```

---

## 3. jax.debug.callback

Execute arbitrary Python functions at runtime:

```python
def my_callback(x):
    """Arbitrary Python — runs on host."""
    print(f"Value: {x}")
    plt.plot(x)  # Can do plotting
    return None  # Must return None or ()

@jax.jit
def f(x):
    jax.debug.callback(my_callback, x)
    return x + 1
```

### With results

```python
# callback cannot return values to JAX
# Use pure_callback for that
```

---

## 4. jax.pure_callback

Call a pure Python function and return results to JAX:

```python
def my_numpy_fn(x):
    """Must be a pure function — no side effects."""
    return np.fft.fft(x)

@jax.jit
def f(x):
    result = jax.pure_callback(my_numpy_fn, jnp.ones_like(x, dtype=jnp.complex64), x)
    return jnp.abs(result)

f(jnp.ones(10))
```

### Signature

```python
jax.pure_callback(
    callback: Callable,        # Python function to call
    result_shape_dtypes: Any,  # Expected output shapes/dtypes (pytree)
    *args,                     # Arguments to pass
    vmap_method: str = 'sequential',  # How vmap handles this
    **kwargs
)
```

### vmap_method options

| Method | Behavior |
|---|---|
| `'sequential'` | Call callback once per batch element |
| `'expand_dims'` | Call once with extra batch dimension |

### Example: NumPy functions not in JAX

```python
def numpy_sort(x):
    return np.sort(x, axis=-1)

@jax.jit
def sort_with_numpy(x):
    return jax.pure_callback(
        numpy_sort,
        jax.ShapeDtypeStruct(x.shape, x.dtype),
        x,
        vmap_method='expand_dims'
    )
```

---

## 5. jax.experimental.io_callback

For side-effecting operations:

```python
from jax.experimental import io_callback

def write_to_file(x):
    with open('output.txt', 'a') as f:
        f.write(f"{x}\n")

@jax.jit
def f(x):
    io_callback(write_to_file, (), x)
    return x + 1
```

### Limitations

- Cannot return values to JAX
- May be called multiple times due to retries/recomputation
- Not compatible with all transformations
- Order not guaranteed

---

## 6. Callbacks with shard_map

```python
from jax.shard_map import shard_map

@shard_map(mesh, in_specs=P('x'), out_specs=P('x'))
def f(x):
    jax.debug.print("Local shard shape: {}", x.shape)
    return x + 1
```

---

## 7. Callbacks for Debugging Training

```python
@jax.jit
def train_step(params, x, y, step):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)

    # Debug print every 100 steps (use ordered for correct sequence)
    jax.debug.callback(
        lambda l, s: print(f"Step {s}: loss = {l}") if s % 100 == 0 else None,
        loss, step, ordered=True
    )

    return jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

---

## 8. Performance Considerations

### Callbacks are synchronous

```python
# Callbacks block the device pipeline
# Don't use in hot inner loops
@jax.jit
def f(x):
    for i in range(10000):
        x = x + 1
        # BAD: callback every iteration → very slow
        jax.debug.print("i={}, x={}", i, x)
    return x
```

### Minimize callback frequency

```python
# GOOD: Only callback at boundaries
@jax.jit
def train_step(params, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    # Only print loss, not per-iteration
    return jax.tree.map(lambda p, g: p - lr * g, params, grads), loss

# Print outside JIT
params, loss = train_step(params, x, y)
if step % 100 == 0:
    print(f"Loss: {loss}")
```

---

## 9. API Reference

```python
# Debug printing
jax.debug.print(fmt: str, *args, ordered: bool = False)

# Arbitrary callback (no return value)
jax.debug.callback(callback: Callable, *args, ordered: bool = False, **kwargs)

# Pure callback (with return value)
jax.pure_callback(
    callback: Callable,
    result_shape_dtypes: Any,
    *args,
    vmap_method: str = 'sequential',
    **kwargs
)

# I/O callback (experimental)
jax.experimental.io_callback(
    callback: Callable,
    result_shape: Any,
    *args,
    ordered: bool = False,
    **kwargs
)
```
