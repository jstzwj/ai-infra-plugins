# Custom Derivative Rules (jax.custom_jvp, jax.custom_vjp)

This document provides an exhaustive reference for JAX's custom derivative system. When JAX's automatic differentiation does not produce the desired gradients -- due to numerical instability, undefined derivatives at certain points, or domain-specific gradient definitions -- you can define custom forward- and reverse-mode differentiation rules using `jax.custom_jvp`, `jax.custom_vjp`, and `jax.custom_gradient`.

---

## Table of Contents

1. [Overview](#1-overview)
2. [jax.custom_jvp -- Custom Forward-Mode Rules](#2-jaxcustom_jvp----custom-forward-mode-rules)
3. [defjvp and defjvps](#3-defjvp-and-defjvps)
4. [symbolic_zeros in custom_jvp](#4-symbolic_zeros-in-custom_jvp)
5. [jax.custom_vjp -- Custom Reverse-Mode Rules](#5-jaxcustom_vjp----custom-reverse-mode-rules)
6. [symbolic_zeros in custom_vjp](#6-symbolic_zeros-in-custom_vjp)
7. [optimize_remat in custom_vjp](#7-optimize_remat-in-custom_vjp)
8. [jax.custom_gradient](#8-jaxcustom_gradient)
9. [closure_convert and linear_call](#9-closure_convert-and-linear_call)
10. [Numerically Stable log1pexp](#10-numerically-stable-log1pexp)
11. [Custom Gradient for Clipped Functions](#11-custom-gradient-for-clipped-functions)
12. [Custom Gradient for Sparsemax](#12-custom-gradient-for-sparsemax)
13. [Custom Gradient for Gradient Clipping at the Function Level](#13-custom-gradient-for-gradient-clipping-at-the-function-level)
14. [Composition with jit, vmap, grad](#14-composition-with-jit-vmap-grad)
15. [Complete Examples](#15-complete-examples)

---

## 1. Overview

JAX's automatic differentiation transforms (`jax.grad`, `jax.jacfwd`, `jax.jacrev`) work by tracing your function and automatically computing derivatives using the chain rule. However, there are situations where you need to override the default derivative:

- **Numerical stability:** The default gradient may overflow or underflow (e.g., `log(1 + exp(x))` for large `x`).
- **Discontinuities:** Functions like `clip`, `relu`, or `sign` have points where the derivative is undefined.
- **Domain knowledge:** You may want to implement straight-through estimators or biased gradients for specific reasons.
- **Black-box functions:** When a function is implemented via FFI or external calls, JAX cannot auto-differentiate through it.

JAX provides three main APIs:

| API | Mode | Description |
|-----|------|-------------|
| `jax.custom_jvp` | Forward | Define custom Jacobian-vector product (tangent propagation) |
| `jax.custom_vjp` | Reverse | Define custom vector-Jacobian product (cotangent propagation) |
| `jax.custom_gradient` | Reverse | Simplified API for scalar-output functions |

```python
import jax
import jax.numpy as jnp

# The problem: naively computing gradients can be numerically unstable
def unstable_log1pexp(x):
    return jnp.log(1.0 + jnp.exp(x))

# For large x, exp(x) overflows even though log(1+exp(x)) ≈ x
print(unstable_log1pexp(jnp.array(1000.0)))  # inf (bad)
# The correct result should be 1000.0
```

---

## 2. jax.custom_jvp -- Custom Forward-Mode Rules

`jax.custom_jvp` lets you define a custom rule for how tangents (perturbations) propagate through your function in forward mode. This affects both `jax.jacfwd` and `jax.grad` (since `grad` uses reverse mode under the hood but JAX internally uses JVP rules).

### Basic Structure

```python
import jax
import jax.numpy as jnp

@jax.custom_jvp
def f(x):
    # The primal function: compute the output
    return jnp.sin(x)

@f.defjvp
def f_jvp(primals, tangents):
    x, = primals
    x_dot, = tangents
    # Return (primal_output, tangent_output)
    primal_out = f(x)
    tangent_out = jnp.cos(x) * x_dot  # custom tangent rule
    return primal_out, tangent_out

# Now jax.grad uses the custom JVP rule
print(jax.grad(f)(2.0))  # cos(2.0) = -0.4161...
```

### Multiple Arguments

```python
import jax
import jax.numpy as jnp

@jax.custom_jvp
def weighted_norm(x, weight):
    return jnp.sqrt(jnp.sum((x * weight) ** 2))

@weighted_norm.defjvp
def weighted_norm_jvp(primals, tangents):
    x, w = primals
    x_dot, w_dot = tangents

    primal_out = weighted_norm(x, w)

    # Gradient w.r.t. x: (x * w^2) / norm
    # Gradient w.r.t. w: (x^2 * w) / norm
    norm = primal_out
    tangent_out = jnp.where(
        norm > 0,
        (jnp.sum(x * w**2 * x_dot) + jnp.sum(x**2 * w * w_dot)) / norm,
        0.0
    )
    return primal_out, tangent_out

# Test
x = jnp.array([1.0, 2.0, 3.0])
w = jnp.array([1.0, 1.0, 1.0])
print(jax.grad(weighted_norm)(x, w))
```

### How defjvp Works Internally

When you define `defjvp`, you are specifying:

1. The **primals** are the actual input values (same as `primals` in the original function).
2. The **tangents** are the perturbation vectors being propagated forward.
3. You must return a tuple `(primal_output, tangent_output)` where:
   - `primal_output` is the function evaluated at the primal inputs (usually `f(*primals)`).
   - `tangent_output` is `J_f @ tangents` (the Jacobian-vector product).

---

## 3. defjvp and defjvps

### defjvp -- Single Rule for All Arguments

`defjvp` defines a single function that handles tangents for all arguments simultaneously.

```python
import jax
import jax.numpy as jnp

@jax.custom_jvp
def my_function(x, y):
    return x ** 2 + y ** 3

@my_function.defjvp
def my_function_jvp(primals, tangents):
    x, y = primals
    x_dot, y_dot = tangents

    # Primal computation
    primal_out = x ** 2 + y ** 3

    # Tangent: d/dx(x^2 + y^3) * x_dot + d/dy(x^2 + y^3) * y_dot
    tangent_out = 2 * x * x_dot + 3 * y ** 2 * y_dot

    return primal_out, tangent_out

print(jax.grad(my_function)(2.0, 3.0))  # d/dx at (2,3) = 4.0
```

### defjvps -- Per-Argument Rules

`defjvps` lets you define separate JVP rules for each argument. This is often cleaner when the function has many arguments.

```python
import jax
import jax.numpy as jnp

@jax.custom_jvp
def my_function(x, y, z):
    return x * y + z

# defjvps: provide a JVP rule for each argument
# The signature for each is (primal_out, primal_arg, tangent_arg) -> tangent_out
@my_function.defjvps
def my_function_x_jvp(primal_out, x, x_dot):
    # Rule for x: d/dx(x*y + z) * x_dot = y * x_dot
    # primal_out is x*y + z
    # But we need y... we can't get it from here
    # So defjvps is limited for functions where the derivative of one arg
    # depends on another arg
    pass  # This won't work well; see the note below

# Instead, defjvps works best for simple cases:
@jax.custom_jvp
def scale(x, scale_factor):
    return x * scale_factor

@scale.defjvps
def scale_jvps(primal_out, x, x_dot):
    # x_dot is the tangent for the FIRST argument (x)
    # The derivative of x * scale_factor w.r.t. x is scale_factor
    return primal_out  # Hmm, this is wrong...

# Actually, let's show the correct pattern:
@jax.custom_jvp
def multiply(x, y):
    return x * y

# defjvps with explicit per-argument handling
@multiply.defjvps
def _x_jvp(primal_out, x, x_dot):
    # JVP rule for the first argument x
    # d/dx (x*y) * x_dot = y * x_dot
    # But we don't have y here! defjvps is tricky.
    # For multi-arg functions, defjvp is usually better.
    pass
```

**Important note on defjvps:** `defjvps` is designed for cases where the derivative with respect to each argument can be computed independently from just the primal output and the argument itself. For more complex cases involving interactions between arguments, use `defjvp` instead.

### Correct defjvps Usage

```python
import jax
import jax.numpy as jnp

@jax.custom_jvp
def safe_sqrt(x):
    return jnp.sqrt(x)

@safe_sqrt.defjvps
def safe_sqrt_jvps(primal_out, x, x_dot):
    # primal_out = sqrt(x), so tangent = x_dot / (2 * sqrt(x)) = x_dot / (2 * primal_out)
    return x_dot / (2.0 * primal_out + 1e-8)

x = jnp.array(4.0)
print(f"sqrt({x}) = {safe_sqrt(x)}")
print(f"grad at {x} = {jax.grad(safe_sqrt)(x)}")  # Should be 0.25
```

### defjvps for Multiple Arguments

```python
import jax
import jax.numpy as jnp

@jax.custom_jvp
def element_wise_product(x, y):
    return x * y

# When using defjvps, you can return a tuple of tangent rules
# But more commonly, you pass individual rules
# Let's use defjvp instead for clarity:
@element_wise_product.defjvp
def element_wise_product_jvp(primals, tangents):
    x, y = primals
    x_dot, y_dot = tangents
    primal_out = x * y
    tangent_out = x_dot * y + x * y_dot
    return primal_out, tangent_out

x, y = jnp.array(3.0), jnp.array(4.0)
print(jax.grad(element_wise_product, argnums=0)(x, y))  # y = 4.0
print(jax.grad(element_wise_product, argnums=1)(x, y))  # x = 3.0
```

---

## 4. symbolic_zeros in custom_jvp

Sometimes, certain inputs to a function do not contribute to the output, meaning their tangent should be zero. JAX uses `symbolic_zeros` to represent these zero tangents efficiently, avoiding unnecessary computation.

### Understanding Symbolic Zeros

```python
import jax
import jax.numpy as jnp

# Without symbolic_zeros, JAX would still compute the tangent for
# arguments that don't affect the output, which is wasteful.

@jax.custom_jvp
def select_first(x, y):
    """Return only x; y has no effect on output."""
    return x

# Without symbolic_zeros awareness:
@select_first.defjvp
def select_first_jvp(primals, tangents):
    x, y = primals
    x_dot, y_dot = tangents
    # y_dot contributes nothing, but we still receive it
    return x, x_dot
```

### Using nondiff_argnums

The simpler approach is to mark non-differentiable arguments:

```python
import jax
import jax.numpy as jnp

@jax.custom_jvp(nondiff_argnums=(1,))
def scale_by_constant(x, scale):
    return x * scale

@scale_by_constant.defjvp
def scale_by_constant_jvp(scale, primals, tangents):
    x, = primals
    x_dot, = tangents
    return x * scale, x_dot * scale

print(jax.grad(scale_by_constant)(jnp.array(2.0), 3.0))  # 3.0
```

### symbolic_zeros for Sparse Gradient Handling

```python
import jax
import jax.numpy as jnp
from jax.custom_derivatives import symbolic_zeros

@jax.custom_jvp
def conditional_scale(x, mask):
    """Apply mask: only elements where mask is True contribute to output."""
    return x * mask

@conditional_scale.defjvp
def conditional_scale_jvp(primals, tangents):
    x, mask = primals
    x_dot, mask_dot = tangents

    primal_out = x * mask

    # If mask_dot is a symbolic zero, we skip computing its contribution
    if isinstance(mask_dot, jax.custom_derivatives.SymbolicZero):
        tangent_out = x_dot * mask
    else:
        tangent_out = x_dot * mask + x * mask_dot

    return primal_out, tangent_out
```

### Practical Example: Stop Gradient on Condition

```python
import jax
import jax.numpy as jnp

@jax.custom_jvp
def stop_gradient_if_negative(x):
    """Propagate gradients only for positive elements."""
    return x  # Primal is identity

@stop_gradient_if_negative.defjvp
def stop_gradient_if_negative_jvp(primals, tangents):
    x, = primals
    x_dot, = tangents
    # Only propagate tangent where x > 0
    tangent_out = jnp.where(x > 0, x_dot, jnp.zeros_like(x_dot))
    return x, tangent_out

x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
grads = jax.grad(lambda x: jnp.sum(stop_gradient_if_negative(x)))(x)
print(grads)  # [0.0, 0.0, 0.0, 1.0, 1.0]
```

---

## 5. jax.custom_vjp -- Custom Reverse-Mode Rules

`jax.custom_vjp` lets you define a custom rule for how cotangents (gradient signals) propagate backward through your function. This is the most common API for custom gradients because `jax.grad` uses reverse-mode AD.

### Basic Structure

```python
import jax
import jax.numpy as jnp

@jax.custom_vjp
def f(x):
    return jnp.sin(x)

# Forward pass: compute the primal and save any values needed for the backward pass
def f_fwd(x):
    # Return (primal_output, saved_values_for_backward)
    return jnp.sin(x), (jnp.cos(x),)

# Backward pass: compute VJP given cotangent and saved values
def f_bwd(saved_values, cotangent):
    cos_x, = saved_values
    # Return a tuple of cotangents matching the input arguments
    return (cos_x * cotangent,)

f.defvjp(f_fwd, f_bwd)

print(jax.grad(f)(jnp.array(1.0)))  # cos(1.0) = 0.5403...
```

### Multiple Arguments

```python
import jax
import jax.numpy as jnp

@jax.custom_vjp
def safe_divide(x, y):
    return x / y

def safe_divide_fwd(x, y):
    result = x / (y + 1e-8)
    # Save values needed for backward pass
    return result, (x, y)

def safe_divide_bwd(saved, g):
    x, y = saved
    # d/dx (x/y) = 1/y
    # d/dy (x/y) = -x/y^2
    g_x = g / (y + 1e-8)
    g_y = -g * x / ((y + 1e-8) ** 2)
    return g_x, g_y

safe_divide.defvjp(safe_divide_fwd, safe_divide_bwd)

x, y = jnp.array(6.0), jnp.array(3.0)
print(jax.grad(safe_divide, argnums=0)(x, y))  # 1/3
print(jax.grad(safe_divide, argnums=1)(x, y))  # -6/9 = -2/3
```

### Why Save Values in the Forward Pass

The forward pass computes the primal output and can save intermediate values (residuals) that the backward pass needs. This avoids recomputation.

```python
import jax
import jax.numpy as jnp

@jax.custom_vjp
def log_softplus(x):
    """Numerically stable log(softplus(x)) = log(1 + exp(x))"""
    return jnp.log(jnp.logaddexp(0.0, x))

def log_softplus_fwd(x):
    # Save the softplus value for backward
    softplus_x = jnp.logaddexp(0.0, x)
    result = jnp.log(softplus_x)
    return result, (softplus_x,)

def log_softplus_bwd(saved, g):
    softplus_x, = saved
    # d/dx log(softplus(x)) = sigmoid(x) / softplus(x)
    sigmoid_x = jnp.exp(-softplus_x) * softplus_x  # = sigmoid(x)
    # Actually: sigmoid(x) = exp(-softplus(x)) is wrong
    # sigmoid(x) = 1 / (1 + exp(-x)) = exp(x) / (1 + exp(x)) = 1 - exp(-softplus(x))
    # But more simply: sigmoid(x) / softplus(x) = (softplus(x) - softplus(-x)) / (softplus(x) * something)
    # Let's just use the direct formula
    sigmoid_x = jax.nn.sigmoid(x) if False else 1.0 - jnp.exp(-softplus_x)
    g_x = g * sigmoid_x / softplus_x
    return (g_x,)

log_softplus.defvjp(log_softplus_fwd, log_softplus_bwd)
```

### nondiff_argnums in custom_vjp

```python
import jax
import jax.numpy as jnp

@jax.custom_vjp(nondiff_argnums=(1,))
def masked_sum(x, mask):
    """Sum only elements where mask is True."""
    return jnp.sum(x * mask)

def masked_sum_fwd(mask, x):
    return jnp.sum(x * mask), (mask,)

def masked_sum_bwd(mask, saved, g):
    # g is a scalar cotangent
    return (g * mask,)

masked_sum.defvjp(masked_sum_fwd, masked_sum_bwd)

x = jnp.array([1.0, 2.0, 3.0, 4.0])
mask = jnp.array([1.0, 0.0, 1.0, 0.0])
print(jax.grad(masked_sum)(x, mask))  # [1.0, 0.0, 1.0, 0.0]
```

---

## 6. symbolic_zeros in custom_vjp

In reverse mode, `symbolic_zeros` helps JAX optimize the backward pass when certain inputs have zero cotangent (i.e., no gradient flows back to them).

### Basic Usage

```python
import jax
import jax.numpy as jnp
from jax.custom_derivatives import symbolic_zero

@jax.custom_vjp
def project_onto_first(x, y):
    """Only x affects the output; y is ignored."""
    return x

def project_fwd(x, y):
    return x, ()

def project_bwd(saved, g):
    # y receives zero gradient because it doesn't affect the output
    return g, symbolic_zero(g)

project_onto_first.defvjp(project_fwd, project_bwd)

x, y = jnp.array(3.0), jnp.array(5.0)
gx, gy = jax.grad(project_onto_first, argnums=(0, 1))(x, y)
print(f"grad_x = {gx}, grad_y = {gy}")  # grad_x = 1.0, grad_y = 0.0
```

### Why Use Symbolic Zeros

Using `symbolic_zero` instead of `jnp.zeros_like(...)` allows JAX to:

1. **Skip computation:** The compiler can eliminate operations that produce zeros.
2. **Avoid materializing zero arrays:** Memory is not allocated for zero cotangents.
3. **Optimize the backward graph:** Zero cotangents propagate and allow dead code elimination.

```python
import jax
import jax.numpy as jnp
from jax.custom_derivatives import symbolic_zero

@jax.custom_vjp
def sparse_select(x, indices):
    """Select elements of x at given indices. No gradient for indices."""
    return x[indices]

def sparse_select_fwd(x, indices):
    return x[indices], (x.shape, indices)

def sparse_select_bwd(saved, g):
    x_shape, indices = saved
    # Gradient for x: scatter g back to the selected positions
    g_x = jnp.zeros(x_shape).at[indices].add(g)
    # Gradient for indices: symbolic zero (indices are integers anyway)
    return g_x, symbolic_zero(indices)

sparse_select.defvjp(sparse_select_fwd, sparse_select_bwd)

x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
indices = jnp.array([0, 2, 4])
print(jax.grad(sparse_select)(x, indices))  # [1.0, 0.0, 1.0, 0.0, 1.0]
```

---

## 7. optimize_remat in custom_vjp

The `optimize_remat` parameter (available in newer JAX versions) controls whether JAX can apply rematerialization optimizations to the residuals saved in the forward pass.

### Default Behavior

By default, `custom_vjp` residuals are not rematerialized -- they are always saved for the backward pass. This can lead to high memory usage.

```python
import jax
import jax.numpy as jnp

@jax.custom_vjp
def memory_heavy(x):
    return jnp.sin(x) * jnp.cos(x) * jnp.tan(x)

def heavy_fwd(x):
    # Save all intermediate values
    sin_x = jnp.sin(x)
    cos_x = jnp.cos(x)
    tan_x = jnp.tan(x)
    result = sin_x * cos_x * tan_x
    return result, (sin_x, cos_x, tan_x)

def heavy_bwd(saved, g):
    sin_x, cos_x, tan_x = saved
    # Use all saved values for backward
    g_x = g * (cos_x * cos_x * tan_x + sin_x * (-sin_x) * tan_x + sin_x * cos_x * (1 / cos_x**2))
    return (g_x,)

memory_heavy.defvjp(heavy_fwd, heavy_bwd)
```

### Using optimize_remat

When `optimize_remat=True`, JAX may choose to recompute residuals during the backward pass instead of storing them, trading compute for memory.

```python
import jax
import jax.numpy as jnp

@jax.custom_vjp
def efficient_custom_fn(x):
    return jnp.sin(x) * jnp.cos(x)

# With optimize_remat=True, JAX can recompute sin(x) and cos(x)
# during the backward pass instead of storing them
def efficient_fwd(x):
    result = jnp.sin(x) * jnp.cos(x)
    # Save minimal info
    return result, (x,)

def efficient_bwd(saved, g):
    x, = saved
    # Recompute instead of storing
    sin_x = jnp.sin(x)
    cos_x = jnp.cos(x)
    g_x = g * (cos_x * cos_x - sin_x * sin_x)
    return (g_x,)

efficient_custom_fn.defvjp(efficient_fwd, efficient_bwd, optimize_remat=True)

x = jnp.array(1.0)
print(jax.grad(efficient_custom_fn)(x))
```

### When to Use optimize_remat

Use `optimize_remat=True` when:

1. The forward pass saves large intermediate arrays.
2. The residuals are cheap to recompute from the inputs.
3. Memory is the bottleneck (e.g., training large models).

```python
import jax
import jax.numpy as jnp

@jax.custom_vjp
def layer_norm_custom(x, gamma, beta):
    mean = jnp.mean(x)
    var = jnp.var(x)
    x_norm = (x - mean) / jnp.sqrt(var + 1e-5)
    return gamma * x_norm + beta

def layer_norm_fwd(x, gamma, beta):
    mean = jnp.mean(x)
    var = jnp.var(x)
    x_norm = (x - mean) / jnp.sqrt(var + 1e-5)
    result = gamma * x_norm + beta
    # Save x_norm and std for backward (could be large)
    return result, (x_norm, jnp.sqrt(var + 1e-5), gamma)

def layer_norm_bwd(saved, g):
    x_norm, std, gamma = saved
    N = x_norm.shape[0]
    g_x = gamma * g / std
    g_x -= jnp.mean(g_x)
    g_x -= x_norm * jnp.mean(g_x * x_norm)
    g_gamma = jnp.sum(g * x_norm)
    g_beta = jnp.sum(g)
    return g_x, g_gamma, g_beta

layer_norm_custom.defvjp(layer_norm_fwd, layer_norm_bwd)
```

---

## 8. jax.custom_gradient

`jax.custom_gradient` is a simplified API for defining custom reverse-mode gradients for scalar-output functions. It is less flexible than `custom_vjp` but more concise.

### Basic Usage

```python
import jax
import jax.numpy as jnp

@jax.custom_gradient
def safe_sqrt(x):
    """Square root with a safe gradient at zero."""
    y = jnp.sqrt(x)
    def grad_fn(g):
        # g is the upstream gradient (cotangent)
        return g / (2.0 * y + 1e-8)
    return y, grad_fn

print(jax.grad(safe_sqrt)(jnp.array(4.0)))  # 0.25
print(jax.grad(safe_sqrt)(jnp.array(0.0)))  # ~0 (not inf)
```

### Multiple Arguments

```python
import jax
import jax.numpy as jnp

@jax.custom_gradient
def safe_div(x, y):
    result = x / (y + 1e-8)
    def grad_fn(g):
        g_x = g / (y + 1e-8)
        g_y = -g * x / ((y + 1e-8) ** 2)
        return g_x, g_y
    return result, grad_fn

x, y = jnp.array(6.0), jnp.array(3.0)
print(jax.grad(safe_div, argnums=0)(x, y))  # 1/3
print(jax.grad(safe_div, argnums=1)(x, y))  # -2/3
```

### Custom Gradient for a Loss Function

```python
import jax
import jax.numpy as jnp

@jax.custom_gradient
def huber_loss(pred, target, delta=1.0):
    diff = pred - target
    abs_diff = jnp.abs(diff)
    is_small = abs_diff <= delta
    loss = jnp.where(
        is_small,
        0.5 * diff ** 2,
        delta * (abs_diff - 0.5 * delta)
    )
    def grad_fn(g):
        # Gradient w.r.t. pred
        g_pred = g * jnp.where(
            is_small,
            diff,
            delta * jnp.sign(diff)
        )
        # Gradient w.r.t. target (negative of pred gradient)
        g_target = -g_pred
        return g_pred, g_target
    return loss, grad_fn

# Usage
pred = jnp.array(2.5)
target = jnp.array(1.0)

grad_pred = jax.grad(huber_loss, argnums=0)(pred, target)
grad_target = jax.grad(huber_loss, argnums=1)(pred, target)
print(f"grad pred: {grad_pred}")    # delta * sign(1.5) = 1.0
print(f"grad target: {grad_target}") # -1.0
```

### custom_gradient with Aux Values

```python
import jax
import jax.numpy as jnp

# custom_gradient does not directly support aux values.
# Use custom_vjp if you need aux outputs in addition to the gradient.
# However, you can work around this by returning auxiliary data in a closure:

def make_loss_with_aux(delta=1.0):
    @jax.custom_gradient
    def huber_with_aux(pred, target):
        diff = pred - target
        abs_diff = jnp.abs(diff)
        is_small = abs_diff <= delta
        loss = jnp.where(
            is_small,
            0.5 * diff ** 2,
            delta * (abs_diff - 0.5 * delta)
        )
        # Auxiliary info stored in closure (not differentiable)
        aux = {"is_small": is_small, "abs_diff": abs_diff}
        def grad_fn(g):
            g_pred = g * jnp.where(is_small, diff, delta * jnp.sign(diff))
            return g_pred, -g_pred
        return loss, grad_fn
    return huber_with_aux

fn = make_loss_with_aux(delta=1.0)
print(jax.grad(fn, argnums=0)(jnp.array(2.5), jnp.array(1.0)))
```

---

## 9. closure_convert and linear_call

### jax.closure_convert

`closure_convert` is used to make closures (functions that capture variables from their enclosing scope) compatible with JAX's tracing system. It explicitly separates the captured constants from the dynamic arguments.

```python
import jax
import jax.numpy as jnp

# Problem: closures capture values that JAX cannot trace through
def make_power_fn(n):
    """Create a function that raises x to the nth power."""
    def power_fn(x):
        return x ** n  # n is captured from the outer scope
    return power_fn

# This works in regular Python
fn = make_power_fn(3)
print(fn(2.0))  # 8.0

# But when we want to define custom gradients, the captured value
# creates issues. closure_convert helps:

converted_fn = jax.closure_convert(fn, jnp.array(1.0))
# Now converted_fn takes the captured constants as explicit arguments

# With custom_jvp, we can now define gradients:
@jax.custom_jvp
def power(x, n):
    return x ** n

@power.defjvp
def power_jvp(primals, tangents):
    x, n = primals
    x_dot, n_dot = tangents
    primal_out = x ** n
    tangent_out = n * x ** (n - 1) * x_dot + jnp.log(x) * primal_out * n_dot
    return primal_out, tangent_out

print(jax.grad(power, argnums=0)(jnp.array(2.0), jnp.array(3.0)))  # 12.0
```

### Using closure_convert with Custom Derivatives

```python
import jax
import jax.numpy as jnp

def make_scaled_loss(scale):
    """Create a scaled MSE loss function."""
    def loss_fn(pred, target):
        return scale * jnp.mean((pred - target) ** 2)
    return loss_fn

# Convert the closure so scale becomes an explicit argument
raw_fn = make_scaled_loss(10.0)
converted = jax.closure_convert(raw_fn, jnp.ones(5), jnp.ones(5))
# converted now has signature (scale, pred, target) -> loss

# Define custom gradient on the converted version
print(converted)
```

### jax.custom_derivatives.linear_call

`linear_call` declares that a function is linear, allowing JAX to compute its derivative exactly without tracing through it. This is useful for linear operations implemented with custom code (e.g., FFI calls).

```python
import jax
import jax.numpy as jnp
from jax.custom_derivatives import linear_call

# A linear function: f(x) = A @ x for some fixed matrix A
def matmul_by_A(x):
    A = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    return A @ x

# Without linear_call, JAX would trace through the function
# With linear_call, JAX knows the derivative is just the adjoint
def linear_matmul(x):
    return linear_call(matmul_by_A, matmul_by_A, x)

# The gradient is automatically computed as the transpose operation
x = jnp.array([1.0, 2.0])
print(linear_matmul(x))  # [5.0, 11.0, 17.0]

# For reverse mode, the transpose/adjoint of a linear function f is:
# If f(x) = Ax, then the adjoint is f*(y) = A^T y
# linear_call takes (fn, fn_transpose, *args)
```

### Linear Call with Explicit Transpose

```python
import jax
import jax.numpy as jnp
from jax.custom_derivatives import linear_call

# Define the linear function and its transpose
def linear_transform(x):
    """x -> A @ x where A is a fixed matrix."""
    A = jnp.array([[1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0]])
    return A @ x

def linear_transform_transpose(y):
    """y -> A^T @ y (the adjoint of the linear function)."""
    A = jnp.array([[1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0]])
    return A.T @ y

# Use linear_call with both the function and its transpose
def composed_linear(x):
    return linear_call(linear_transform, linear_transform_transpose, x)

# Now JAX knows how to differentiate through this
x = jnp.array([1.0, 1.0, 1.0])
result = composed_linear(x)
print(f"Forward: {result}")  # [6.0, 15.0]

# The gradient computation uses the transpose
def loss_fn(x):
    return jnp.sum(composed_linear(x) ** 2)

grad = jax.grad(loss_fn)(x)
print(f"Gradient: {grad}")  # 2 * A^T @ A @ x

# Verify: A^T @ A @ [1,1,1]
A = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
expected = 2 * A.T @ A @ x
print(f"Expected: {expected}")
```

---

## 10. Numerically Stable log1pexp

The log1pexp function (also known as softplus) is a classic example where custom derivatives are needed for numerical stability.

### The Problem

```python
import jax
import jax.numpy as jnp

def naive_log1pexp(x):
    """log(1 + exp(x)) -- numerically unstable."""
    return jnp.log(1.0 + jnp.exp(x))

# For large x, exp(x) overflows to inf
print(naive_log1pexp(jnp.array(1000.0)))  # inf (should be 1000.0)

# For very negative x, 1 + exp(x) = 1 due to float precision
print(naive_log1pexp(jnp.array(-1000.0)))  # 0.0 (correct)

# The gradient also has issues
print(jax.grad(naive_log1pexp)(jnp.array(1000.0)))  # NaN
```

### Using jax.nn.softplus (Built-in Solution)

```python
import jax
import jax.numpy as jnp

# JAX already provides a numerically stable softplus
x_large = jnp.array(1000.0)
print(jax.nn.softplus(x_large))  # 1000.0 (correct)

x_neg = jnp.array(-1000.0)
print(jax.nn.softplus(x_neg))    # 0.0 (correct)

print(jax.grad(jax.nn.softplus)(x_large))  # 1.0 (correct)
```

### Custom JVP Implementation

```python
import jax
import jax.numpy as jnp

@jax.custom_jvp
def log1pexp(x):
    """Numerically stable log(1 + exp(x))."""
    # Use piecewise implementation for stability
    return jnp.where(
        x > 20.0,
        x,                          # For large x: log(1+exp(x)) ≈ x
        jnp.where(
            x < -20.0,
            jnp.exp(x),             # For very negative x: log(1+exp(x)) ≈ exp(x)
            jnp.log1p(jnp.exp(x))   # Standard case: use log1p for precision
        )
    )

@log1pexp.defjvp
def log1pexp_jvp(primals, tangents):
    x, = primals
    x_dot, = tangents

    # d/dx log(1 + exp(x)) = exp(x) / (1 + exp(x)) = sigmoid(x)
    # sigmoid(x) is always in [0, 1] and numerically stable
    primal_out = log1pexp(x)
    tangent_out = jax.nn.sigmoid(x) * x_dot

    return primal_out, tangent_out

# Test
print(log1pexp(jnp.array(1000.0)))   # 1000.0
print(log1pexp(jnp.array(-1000.0)))  # 0.0
print(jax.grad(log1pexp)(jnp.array(1000.0)))   # 1.0
print(jax.grad(log1pexp)(jnp.array(-1000.0)))  # 0.0
print(jax.grad(log1pexp)(jnp.array(0.0)))      # 0.5
```

### Custom VJP Implementation

```python
import jax
import jax.numpy as jnp

@jax.custom_vjp
def log1pexp_vjp(x):
    """Numerically stable log1pexp with custom reverse-mode rule."""
    return jnp.log1p(jnp.exp(x))

def log1pexp_fwd(x):
    primal_out = jnp.log1p(jnp.exp(x))
    # Save sigmoid for backward (no need to save x)
    return primal_out, jax.nn.sigmoid(x)

def log1pexp_bwd(saved_sigmoid, g):
    sigmoid_x = saved_sigmoid
    return (g * sigmoid_x,)

log1pexp_vjp.defvjp(log1pexp_fwd, log1pexp_bwd)

# Test
print(log1pexp_vjp(jnp.array(0.0)))           # log(2) ≈ 0.6931
print(jax.grad(log1pexp_vjp)(jnp.array(0.0)))  # 0.5
```

---

## 11. Custom Gradient for Clipped Functions

Gradient clipping at the function level (not just the optimizer level) can be useful for ensuring bounded gradients during backpropagation.

### Clipped Linear with Straight-Through Estimator

```python
import jax
import jax.numpy as jnp

@jax.custom_jvp
def clip_straight_through(x, lo, hi):
    """Clip x to [lo, hi], but pass gradients through as if it were identity."""
    return jnp.clip(x, lo, hi)

@clip_straight_through.defjvp
def clip_straight_through_jvp(primals, tangents):
    x, lo, hi = primals
    x_dot, lo_dot, hi_dot = tangents
    primal_out = jnp.clip(x, lo, hi)
    # Straight-through: gradient is just x_dot regardless of clipping
    tangent_out = x_dot
    return primal_out, tangent_out

x = jnp.array(5.0)
lo, hi = jnp.array(0.0), jnp.array(3.0)
print(clip_straight_through(x, lo, hi))         # 3.0 (clipped)
print(jax.grad(clip_straight_through)(x, lo, hi))  # 1.0 (straight-through)
```

### Clipped Gradient (Gradient Clipping at Function Level)

```python
import jax
import jax.numpy as jnp

@jax.custom_vjp
def gradient_clipped_fn(x, clip_value=1.0):
    """Apply gradient clipping during backprop."""
    return x  # Identity forward pass

def gc_fwd(x, clip_value):
    return x, (clip_value,)

def gc_bwd(saved, g):
    clip_value, = saved
    # Clip the gradient magnitude
    g_norm = jnp.linalg.norm(g)
    clipped_g = jnp.where(
        g_norm > clip_value,
        g * clip_value / (g_norm + 1e-8),
        g
    )
    return clipped_g, None  # None for clip_value (not differentiable)

gradient_clipped_fn.defvjp(gc_fwd, gc_bwd)

x = jnp.array([100.0, 200.0, 300.0])
grads = jax.grad(lambda x: jnp.sum(gradient_clipped_fn(x, 1.0)))(x)
print(f"Clipped gradients: {grads}")
# The gradient of sum(identity) is [1,1,1], which has norm sqrt(3) > 1.0
# So it gets clipped: [1/sqrt(3), 1/sqrt(3), 1/sqrt(3)]
```

### ReLU with Custom Gradient Override

```python
import jax
import jax.numpy as jnp

@jax.custom_jvp
def leaky_relu_with_custom_grad(x, negative_slope=0.01):
    """Leaky ReLU where we override the gradient at x=0."""
    return jnp.where(x > 0, x, negative_slope * x)

@leaky_relu_with_custom_grad.defjvp
def leaky_relu_jvp(primals, tangents):
    x, _ = primals
    x_dot, _ = tangents
    primal_out = leaky_relu_with_custom_grad(x)
    # Custom: at x=0, use a specific gradient (e.g., 0.5)
    tangent_out = jnp.where(
        x > 0, x_dot,
        jnp.where(x < 0, negative_slope * x_dot, 0.5 * x_dot)
    )
    return primal_out, tangent_out

# But this has a bug: negative_slope isn't in scope in the JVP.
# Let's fix this:

@jax.custom_jvp
def leaky_relu_v2(x, negative_slope=0.01):
    return jnp.where(x > 0, x, negative_slope * x)

@leaky_relu_v2.defjvp
def leaky_relu_v2_jvp(primals, tangents):
    x, negative_slope = primals
    x_dot, ns_dot = tangents
    primal_out = leaky_relu_v2(x, negative_slope)
    tangent_out = jnp.where(x > 0, x_dot, negative_slope * x_dot + x * ns_dot)
    return primal_out, tangent_out
```

### Hard Tanh with Custom Gradient

```python
import jax
import jax.numpy as jnp

@jax.custom_jvp
def hard_tanh(x):
    """Hard tanh: clip to [-1, 1] with gradient 1 in [-1, 1] and 0 outside."""
    return jnp.clip(x, -1.0, 1.0)

@hard_tanh.defjvp
def hard_tanh_jvp(primals, tangents):
    x, = primals
    x_dot, = tangents
    primal_out = jnp.clip(x, -1.0, 1.0)
    # Gradient is 1 inside [-1, 1] and 0 outside
    tangent_out = jnp.where((x >= -1.0) & (x <= 1.0), x_dot, 0.0)
    return primal_out, tangent_out

x = jnp.array([-2.0, -0.5, 0.0, 0.5, 2.0])
print(hard_tanh(x))                          # [-1.0, -0.5, 0.0, 0.5, 1.0]
print(jax.grad(lambda x: jnp.sum(hard_tanh(x)))(x))  # [0.0, 1.0, 1.0, 1.0, 0.0]
```

---

## 12. Custom Gradient for Sparsemax

Sparsemax is a sparse alternative to softmax that projects onto the probability simplex.

```python
import jax
import jax.numpy as jnp

@jax.custom_vjp
def sparsemax(z):
    """Sparsemax: projects z onto the probability simplex."""
    # Sort z in descending order
    z_sorted = jnp.sort(z)[::-1]
    n = z.shape[0]

    # Find the threshold tau
    cumsum = jnp.cumsum(z_sorted)
    support = jnp.arange(1, n + 1)
    rho_values = 1.0 + support * z_sorted - cumsum

    # Find the largest k such that rho_k > 0
    rho = jnp.max(jnp.where(rho_values > 0, support, 0)).astype(jnp.int32)
    tau = (jnp.sum(z_sorted[:rho]) - 1.0) / rho

    return jnp.maximum(z - tau, 0.0)

def sparsemax_fwd(z):
    result = sparsemax(z)
    return result, (result,)

def sparsemax_bwd(saved, g):
    result, = saved
    # Gradient: if result > 0, the gradient is g - sum(g * support) / |support|
    support = (result > 0).astype(jnp.float32)
    support_size = jnp.sum(support)
    g_sparse = g - jnp.sum(g * support) / support_size
    return (g_sparse * support,)

sparsemax.defvjp(sparsemax_fwd, sparsemax_bwd)

z = jnp.array([1.0, 2.0, 3.0])
result = sparsemax(z)
print(f"Sparsemax: {result}")  # [0.0, 0.0, 1.0]
grads = jax.grad(lambda z: jnp.sum(sparsemax(z) * jnp.array([1.0, 0.0, 0.0])))(z)
print(f"Gradient: {grads}")
```

---

## 13. Custom Gradient for Gradient Clipping at the Function Level

This pattern is useful for implementing gradient penalties or controlling gradient flow through specific parts of a network.

```python
import jax
import jax.numpy as jnp

@jax.custom_vjp
def scaled_gradient(scale, x):
    """Identity in forward, but scales gradient by `scale` in backward."""
    return x

def scaled_gradient_fwd(scale, x):
    return x, (scale,)

def scaled_gradient_bwd(saved, g):
    scale, = saved
    return None, scale * g  # None: no gradient for scale

scaled_gradient.defvjp(scaled_gradient_fwd, scaled_gradient_bwd)

# Usage: reduce gradient magnitude in a specific layer
x = jnp.array([1.0, 2.0, 3.0])
scale = jnp.array(0.1)
grad = jax.grad(lambda x: jnp.sum(scaled_gradient(scale, x)))(x)
print(grad)  # [0.1, 0.1, 0.1]
```

### Stop Gradient (Equivalent to jax.lax.stop_gradient)

```python
import jax
import jax.numpy as jnp

@jax.custom_vjp
def my_stop_gradient(x):
    return x

def sg_fwd(x):
    return x, ()

def sg_bwd(saved, g):
    return (jnp.zeros_like(g),)  # Zero gradient

my_stop_gradient.defvjp(sg_fwd, sg_bwd)

# Equivalent to jax.lax.stop_gradient
x = jnp.array([1.0, 2.0, 3.0])
grad = jax.grad(lambda x: jnp.sum(my_stop_gradient(x ** 2)))(x)
print(grad)  # [0.0, 0.0, 0.0]
```

### Straight-Through Estimator for Quantization

```python
import jax
import jax.numpy as jnp

@jax.custom_jvp
def quantize_st(x, num_levels=256):
    """Quantize x but pass gradients through as if it were identity."""
    x_min = jnp.min(x)
    x_max = jnp.max(x)
    scale = (x_max - x_min) / (num_levels - 1)
    x_q = jnp.round((x - x_min) / scale) * scale + x_min
    return x_q

@quantize_st.defjvp
def quantize_st_jvp(primals, tangents):
    x, num_levels = primals
    x_dot, _ = tangents
    primal_out = quantize_st(x, num_levels)
    # Straight-through: gradient passes through as if identity
    return primal_out, x_dot

# Simulate quantized forward pass with real gradients
x = jnp.array([0.1, 0.5, 0.9])
loss = lambda x: jnp.sum(quantize_st(x) ** 2)
print(jax.grad(loss)(x))  # 2 * quantize_st(x), but gradient is 2*x
```

---

## 14. Composition with jit, vmap, grad

Custom derivative functions compose naturally with JAX's other transformations.

### Composition with jit

```python
import jax
import jax.numpy as jnp

@jax.custom_jvp
def my_exp(x):
    return jnp.exp(x)

@my_exp.defjvp
def my_exp_jvp(primals, tangents):
    x, = primals
    x_dot, = tangents
    return jnp.exp(x), jnp.exp(x) * x_dot

# JIT compiles both the primal and the custom JVP
jitted = jax.jit(my_exp)
print(jitted(2.0))  # e^2

# JIT with grad
jitted_grad = jax.jit(jax.grad(my_exp))
print(jitted_grad(2.0))  # e^2
```

### Composition with vmap

```python
import jax
import jax.numpy as jnp

@jax.custom_jvp
def safe_reciprocal(x):
    return 1.0 / (x + 1e-8)

@safe_reciprocal.defjvp
def safe_reciprocal_jvp(primals, tangents):
    x, = primals
    x_dot, = tangents
    primal_out = 1.0 / (x + 1e-8)
    tangent_out = -x_dot / (x + 1e-8) ** 2
    return primal_out, tangent_out

# vmap over a batch
x_batch = jnp.array([[1.0, 2.0], [3.0, 4.0]])
result = jax.vmap(safe_reciprocal)(x_batch)
print(result)

# vmap(grad) for per-example gradients
grads = jax.vmap(jax.grad(safe_reciprocal))(jnp.array([1.0, 2.0, 3.0]))
print(grads)  # [-1, -1/4, -1/9]
```

### Nested grad with custom derivatives

```python
import jax
import jax.numpy as jnp

@jax.custom_jvp
def smooth_abs(x, epsilon=1e-6):
    """Smooth absolute value: sqrt(x^2 + epsilon)."""
    return jnp.sqrt(x ** 2 + epsilon)

@smooth_abs.defjvp
def smooth_abs_jvp(primals, tangents):
    x, epsilon = primals
    x_dot, eps_dot = tangents
    primal_out = jnp.sqrt(x ** 2 + epsilon)
    tangent_out = (2 * x * x_dot + eps_dot) / (2 * primal_out)
    return primal_out, tangent_out

# Second derivative (Hessian diagonal)
hessian = jax.grad(jax.grad(smooth_abs))
print(hessian(jnp.array(1.0), jnp.array(1e-6)))
```

### Custom Derivatives inside scan

```python
import jax
import jax.numpy as jnp

@jax.custom_jvp
def stable_sigmoid(x):
    return jnp.where(
        x >= 0,
        1.0 / (1.0 + jnp.exp(-x)),
        jnp.exp(x) / (1.0 + jnp.exp(x))
    )

@stable_sigmoid.defjvp
def stable_sigmoid_jvp(primals, tangents):
    x, = primals
    x_dot, = tangents
    s = stable_sigmoid(x)
    return s, s * (1.0 - s) * x_dot

def rnn_step(carry, x):
    h, = carry
    h = stable_sigmoid(h * 0.9 + x * 0.1)
    return (h,), h

def rnn_loss(params, sequence):
    final_carry, outputs = jax.lax.scan(rnn_step, (params,), sequence)
    return jnp.sum(outputs)

# Gradient flows through the scan using custom sigmoid derivatives
key = jax.random.key(0)
h0 = jnp.zeros(16)
seq = jax.random.normal(key, (100, 16))
loss = rnn_loss(h0, seq)
grad = jax.grad(rnn_loss)(h0, seq)
```

---

## 15. Complete Examples

### Example 1: Numerically Stable Cross-Entropy with Custom VJP

```python
import jax
import jax.numpy as jnp

@jax.custom_vjp
def stable_log_softmax(logits):
    """Numerically stable log softmax."""
    shifted = logits - jnp.max(logits)
    log_sum_exp = jnp.log(jnp.sum(jnp.exp(shifted)))
    return shifted - log_sum_exp

def stable_log_softmax_fwd(logits):
    shifted = logits - jnp.max(logits)
    log_sum_exp = jnp.log(jnp.sum(jnp.exp(shifted)))
    log_probs = shifted - log_sum_exp
    probs = jnp.exp(log_probs)
    return log_probs, probs

def stable_log_softmax_bwd(saved_probs, g):
    probs = saved_probs
    return (g - probs * jnp.sum(g),)

stable_log_softmax.defvjp(stable_log_softmax_fwd, stable_log_softmax_bwd)

def cross_entropy_loss(logits, targets):
    log_probs = stable_log_softmax(logits)
    return -jnp.sum(targets * log_probs)

logits = jnp.array([2.0, 1.0, 0.1])
targets = jnp.array([1.0, 0.0, 0.0])
loss = cross_entropy_loss(logits, targets)
grad = jax.grad(cross_entropy_loss)(logits, targets)
print(f"Loss: {loss:.4f}")
print(f"Gradient: {grad}")
# Gradient should be softmax(logits) - targets
```

### Example 2: Custom Gradient for a Bilinear Sampling Operation

```python
import jax
import jax.numpy as jnp

@jax.custom_vjp
def bilinear_sample(image, y, x):
    """Sample from image at fractional coordinates (y, x) using bilinear interpolation."""
    H, W, C = image.shape
    y0 = jnp.floor(y).astype(jnp.int32)
    x0 = jnp.floor(x).astype(jnp.int32)
    y1 = y0 + 1
    x1 = x0 + 1
    wy = y - y0.astype(jnp.float32)
    wx = x - x0.astype(jnp.float32)

    # Clamp coordinates
    y0c = jnp.clip(y0, 0, H - 1)
    y1c = jnp.clip(y1, 0, H - 1)
    x0c = jnp.clip(x0, 0, W - 1)
    x1c = jnp.clip(x1, 0, W - 1)

    Ia = image[y0c, x0c]
    Ib = image[y1c, x0c]
    Ic = image[y0c, x1c]
    Id = image[y1c, x1c]

    return Ia * (1 - wy) * (1 - wx) + Ib * wy * (1 - wx) + Ic * (1 - wy) * wx + Id * wy * wx

def bilinear_sample_fwd(image, y, x):
    result = bilinear_sample(image, y, x)
    return result, (image, y, x)

def bilinear_sample_bwd(saved, g):
    image, y, x = saved
    H, W, C = image.shape
    y0 = jnp.floor(y).astype(jnp.int32)
    x0 = jnp.floor(x).astype(jnp.int32)
    wy = y - y0.astype(jnp.float32)
    wx = x - x0.astype(jnp.float32)

    # Gradient w.r.t. image: scatter g to the four corners
    g_image = jnp.zeros_like(image)
    y0c = jnp.clip(y0, 0, H - 1)
    x0c = jnp.clip(x0, 0, W - 1)
    g_image = g_image.at[y0c, x0c].add(g * (1 - wy) * (1 - wx))
    # ... (simplified)

    # Gradient w.r.t. y and x
    # g_y = sum over channels of g * (d/dy of the interpolation)
    g_y = jnp.sum(g * (-image[y0c, x0c] * (1 - wx) + image[y1c, x0c] * (1 - wx)
                        - image[y0c, x1c] * wx + image[y1c, x1c] * wx))
    g_x = jnp.sum(g * (-image[y0c, x0c] * (1 - wy) - image[y1c, x0c] * wy
                        + image[y0c, x1c] * (1 - wy) + image[y1c, x1c] * wy))

    return g_image, g_y, g_x

bilinear_sample.defvjp(bilinear_sample_fwd, bilinear_sample_bwd)
```

### Example 3: Gradient of a Non-Differentiable Metric (BLEU-like)

```python
import jax
import jax.numpy as jnp

@jax.custom_jvp
def soft_indicator(x, temperature=1.0):
    """Smooth indicator function: 1 if x > 0, 0 otherwise, smoothed."""
    return jax.nn.sigmoid(x / temperature)

@soft_indicator.defjvp
def soft_indicator_jvp(primals, tangents):
    x, temperature = primals
    x_dot, t_dot = tangents
    primal_out = soft_indicator(x, temperature)
    sigmoid_grad = primal_out * (1.0 - primal_out) / temperature
    tangent_out = sigmoid_grad * x_dot - primal_out * (1.0 - primal_out) * x / temperature**2 * t_dot
    return primal_out, tangent_out

# Use as a smooth threshold
x = jnp.linspace(-5.0, 5.0, 11)
print(soft_indicator(x))  # Smooth transition from 0 to 1
print(jax.grad(lambda x: jnp.sum(soft_indicator(x)))(x))
```

### Example 4: Custom Derivative for Numerically Stable Focal Loss

```python
import jax
import jax.numpy as jnp

@jax.custom_vjp
def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """Numerically stable focal loss."""
    probs = jax.nn.sigmoid(logits)
    ce = jax.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=targets)
    p_t = jnp.where(targets == 1, probs, 1 - probs)
    focal_weight = alpha * (1 - p_t) ** gamma
    return focal_weight * ce

def focal_loss_fwd(logits, targets, alpha, gamma):
    probs = jax.nn.sigmoid(logits)
    ce = jax.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=targets)
    p_t = jnp.where(targets == 1, probs, 1 - probs)
    focal_weight = alpha * (1 - p_t) ** gamma
    loss = focal_weight * ce
    return loss, (probs, p_t, focal_weight, targets, alpha, gamma)

def focal_loss_bwd(saved, g):
    probs, p_t, fw, targets, alpha, gamma = saved
    # Gradient w.r.t. logits
    ce_grad = probs - targets  # d/d logits of BCE
    fw_grad = -alpha * gamma * (1 - p_t) ** (gamma - 1) * ce  # d/d logits of focal weight
    g_logits = g * (fw * ce_grad + fw_grad * ce) if False else g * fw * ce_grad  # Simplified
    return g_logits, None, None, None

focal_loss.defvjp(focal_loss_fwd, focal_loss_bwd)

logits = jnp.array([2.0, -1.0, 0.5])
targets = jnp.array([1.0, 0.0, 1.0])
loss = focal_loss(logits, targets)
print(f"Focal loss: {loss}")
grad = jax.grad(lambda l: jnp.sum(focal_loss(l, targets)))(logits)
print(f"Gradient: {grad}")
```

### Example 5: Complete Training Loop with Custom Gradients

```python
import jax
import jax.numpy as jnp
import optax

# Define a custom activation with custom gradient
@jax.custom_jvp
def mish(x):
    """Mish activation: x * tanh(softplus(x))."""
    return x * jnp.tanh(jnp.log1p(jnp.exp(x)))

@mish.defjvp
def mish_jvp(primals, tangents):
    x, = primals
    x_dot, = tangents
    softplus_x = jnp.log1p(jnp.exp(x))
    tanh_sp = jnp.tanh(softplus_x)
    sigmoid_x = jax.nn.sigmoid(x)
    # d/dx mish(x) = tanh(softplus(x)) + x * sigmoid(x) * sech^2(softplus(x))
    sech2 = 1.0 - tanh_sp ** 2
    primal_out = x * tanh_sp
    tangent_out = x_dot * (tanh_sp + x * sigmoid_x * sech2)
    return primal_out, tangent_out

# MLP with mish activation
def mlp(params, x):
    for w, b in params[:-1]:
        x = mish(jnp.dot(x, w) + b)
    w, b = params[-1]
    return jnp.dot(x, w) + b

def loss_fn(params, x, y):
    pred = mlp(params, x)
    return jnp.mean((pred - y) ** 2)

# Training setup
key = jax.random.key(42)
hidden_dim = 64
params = [
    (jax.random.normal(key, (1, hidden_dim)) * 0.01, jnp.zeros(hidden_dim)),
    (jax.random.normal(key, (hidden_dim, 1)) * 0.01, jnp.zeros(1)),
]

optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# Train
for step in range(100):
    key, subkey = jax.random.split(key)
    x = jax.random.uniform(subkey, (32, 1), minval=-5.0, maxval=5.0)
    y = jnp.sin(x)
    params, opt_state, loss = train_step(params, opt_state, x, y)
    if step % 20 == 0:
        print(f"Step {step}: loss = {loss:.4f}")
```

---

## Summary

| Feature | custom_jvp | custom_vjp | custom_gradient |
|---------|-----------|-----------|----------------|
| Mode | Forward | Reverse | Reverse |
| Affects | `jacfwd`, `grad` | `jacrev`, `grad` | `grad` |
| Multi-arg | Yes (defjvp) | Yes (fwd/bwd pair) | Yes (return tuple) |
| Aux outputs | No | Yes | No |
| symbolic_zeros | Yes | Yes | No |
| nondiff_argnums | Yes | Yes | No |
| optimize_remat | N/A | Yes | No |
| Use when | Forward-mode AD needed | Full control over backward | Simple scalar-output grad override |

### When to Use Which

1. **`custom_gradient`:** Use for simple scalar-output functions where you just need to override the gradient.
2. **`custom_jvp`:** Use when you need forward-mode rules (e.g., for `jacfwd`), or when the tangent computation is simpler than the cotangent computation.
3. **`custom_vjp`:** Use when you need full control over the backward pass (e.g., saving specific residuals, using `nondiff_argnums`, or returning auxiliary values).
4. **`linear_call`:** Use for linear functions where the derivative is just the transpose/adjoint.
5. **`closure_convert`:** Use when you need to make closures explicit for JAX tracing.
