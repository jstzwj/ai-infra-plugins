# 33 - Building on JAX (Library Development)

## Overview

This chapter covers best practices and APIs for building libraries and frameworks on top of JAX. Topics include registering custom primitives, writing transformation rules, and the `jax.extend` API.

---

## 1. JAX Extension Points

### What you can extend

| Extension point | Module | Purpose |
|---|---|---|
| Custom primitives | `jax.core` | New operations |
| Pytree nodes | `jax.tree_util` | Custom container types |
| Random distributions | `jax.random` | Custom PRNG algorithms |
| XLA custom calls | `jax.ffi` | C/C++/CUDA integration |
| Interpretation rules | `jax.interpreters` | Custom trace behavior |

---

## 2. Custom Primitives

### Full example

```python
import jax
from jax import core
import jax.numpy as jnp

# 1. Define the primitive
my_mul_p = core.Primitive("my_mul")

# 2. Implementation rule (runs on concrete arrays)
@my_mul_p.def_impl
def my_mul_impl(x, y):
    return jnp.multiply(x, y)

# 3. Abstract evaluation rule (shape/dtype only)
@my_mul_p.def_abstract_eval
def my_mul_abstract(x_aval, y_aval):
    return core.ShapedArray(jnp.broadcast_shapes(x_aval.shape, y_aval.shape),
                            jnp.result_type(x_aval.dtype, y_aval.dtype))

# 4. User-facing function
def my_mul(x, y):
    return my_mul_p.bind(x, y)

# 5. JVP rule (for differentiation)
from jax.interpreters import ad

def my_mul_jvp(primals, tangents):
    x, y = primals
    dx, dy = tangents
    return my_mul(x, y), my_mul(x, dy) + my_mul(dx, y)

ad.primitive_jvps[my_mul_p] = my_mul_jvp

# 6. Batching rule (for vmap)
from jax.interpreters import batching

def my_mul_batch(args, dims):
    x, y = args
    x_dim, y_dim = dims
    x = batching.moveaxis(x, x_dim, 0) if x_dim is not None else x
    y = batching.moveaxis(y, y_dim, 0) if y_dim is not None else y
    return my_mul(x, y), 0

batching.primitive_batchers[my_mul_p] = my_mul_batch

# 7. XLA lowering rule
from jax.interpreters import mlir

def my_mul_lowering(ctx, x, y):
    return [mlir.ir_constants(x * y)]  # Simplified; real version uses HLO ops

mlir.register_lowering(my_mul_p, my_mul_lowering)
```

---

## 3. Registering Custom Pytree Nodes

```python
from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
class MyLinear:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def tree_flatten(self):
        """Return (children, aux_data)."""
        children = (self.weight, self.bias)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct from children and aux_data."""
        return cls(*children)

    def __call__(self, x):
        return x @ self.weight + self.bias
```

### With aux_data

```python
@register_pytree_node_class
class LayerNorm:
    def __init__(self, weight, bias, eps=1e-5):
        self.weight = weight
        self.bias = bias
        self.eps = eps

    def tree_flatten(self):
        children = (self.weight, self.bias)
        aux_data = {'eps': self.eps}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        weight, bias = children
        return cls(weight, bias, **aux_data)
```

### Using `register_pytree_node`

```python
from jax.tree_util import register_pytree_node
import collections

# Register an existing type
register_pytree_node(
    collections.namedtuple,
    lambda nt: (list(nt), nt._fields),
    lambda keys, values: collections.namedtuple(keys)(*values)
)
```

---

## 4. JVP and VJP Rules

### JVP rule (forward-mode)

```python
def my_op_jvp(primals, tangents):
    x, = primals
    dx, = tangents
    primal_out = my_op(x)
    tangent_out = some_jvp_rule(x, dx)
    return primal_out, tangent_out

ad.primitive_jvps[my_op_p] = my_op_jvp
```

### VJP rule (reverse-mode)

```python
def my_op_vjp(primals, result):
    x, = primals
    y, = result

    def vjp_fn(cotangent):
        # Compute dx from cotangent
        dx = compute_gradient(x, cotangent)
        return (dx,)

    return vjp_fn

ad.primitive_vjps[my_op_p] = my_op_vjp
```

### Using `custom_jvp`/`custom_vjp` (simpler API)

```python
@jax.custom_jvp
def my_op(x):
    return jnp.exp(x) * jnp.sin(x)

@my_op.defjvp
def my_op_jvp(primals, tangents):
    x, = primals
    dx, = tangents
    return my_op(x), (jnp.exp(x) * (jnp.sin(x) + jnp.cos(x))) * dx
```

---

## 5. MLIR Lowering Rules

```python
from jax.interpreters import mlir
import jaxlib.mlir.ir as ir

def my_op_lowering(ctx: mlir.LoweringRuleContext, x):
    """Lower primitive to MLIR/HLO."""
    # Get MLIR value from operand
    x_val = mlir.aval_to_ir_types(ctx.avals_in[0])

    # Build HLO operation
    result_type = mlir.aval_to_ir_types(ctx.avals_out[0])
    result = hlo.CustomCallOp(
        result_type,
        [x_val],
        call_target_name=ir.StringAttr.get("my_custom_op"),
        has_side_effect=ir.BoolAttr.get(False),
        operand_layouts=...,
        result_layouts=...,
    )

    return result.results

mlir.register_lowering(my_op_p, my_op_lowering)
```

---

## 6. Module System Best Practices

### Lazy imports

```python
# Use lazy imports for optional dependencies
def _get_scipy():
    import scipy
    return scipy
```

### Version checking

```python
import jax
jax_version = jax.__version_info__
```

### Configuration options

```python
# Add library-specific config
jax.config.update("my_lib_debug_mode", False)
```

---

## 7. jax.extend API

The `jax.extend` module provides access to JAX internals with explicit no-compatibility-guarantee warnings:

```python
from jax.extend import (
    core,           # Core types (Primitive, AbstractValue, etc.)
    ffi,            # Foreign Function Interface
    linear_util,    # Linear transformation utilities
    random,         # PRNG internals
    mesh,           # Mesh internals
)
```

### When to use jax.extend

- Building custom JAX transformations
- Implementing new hardware backends
- Creating domain-specific languages on top of JAX
- Research and experimentation

### When NOT to use jax.extend

- Application code — use the public `jax` API
- Production code requiring stability guarantees

---

## 8. Testing JAX Extensions

```python
import jax.test_util as jtu

class MyOpTest(jtu.JaxTestCase):
    def test_forward(self):
        x = jnp.array([1.0, 2.0, 3.0])
        result = my_op(x)
        expected = jnp.exp(x)
        self.assertArraysAllClose(result, expected)

    def test_grad(self):
        def f(x):
            return jnp.sum(my_op(x))
        jtu.check_grads(f, (jnp.array([1.0, 2.0]),), order=2)

    def test_jit(self):
        @jax.jit
        def f(x):
            return my_op(x)
        result = f(jnp.ones(3))
        self.assertEqual(result.shape, (3,))

    def test_vmap(self):
        def f(x):
            return my_op(x)
        result = jax.vmap(f)(jnp.ones((5, 3)))
        self.assertEqual(result.shape, (5, 3))
```

---

## 9. Documentation Standards

### Docstring format for JAX functions

```python
def my_function(x, y, z=1.0):
    """Short description.

    Longer description with details.

    Args:
        x: Description of x.
        y: Description of y.
        z: Optional parameter. Default: 1.0.

    Returns:
        Description of return value.

    Examples:
        >>> my_function(jnp.ones(3), jnp.zeros(3))
        Array([...], dtype=float32)
    """
```
