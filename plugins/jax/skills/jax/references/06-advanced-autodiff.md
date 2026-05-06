# 06 - Advanced Automatic Differentiation

## Overview

This chapter covers advanced AD topics: higher-order derivatives, efficient Jacobian computation, differentiation through complex control flow, and composing AD with other transformations.

---

## 1. Higher-Order Derivatives

### Arbitrary order differentiation

```python
import jax
import jax.numpy as jnp

def nth_derivative(f, n):
    """Compute the nth derivative of f."""
    if n == 0:
        return f
    return nth_derivative(jax.grad(f), n - 1)

# Example: Taylor series coefficient extraction
def f(x):
    return jnp.sin(x)

# d^n/dx^n sin(x) at x=0 gives Taylor coefficients
for n in range(6):
    d = nth_derivative(f, n)
    print(f"f^({n})(0) = {d(0.0)}")
# f^(0)(0) = 0.0     (sin(0))
# f^(1)(0) = 1.0     (cos(0))
# f^(2)(0) = -0.0    (-sin(0))
# f^(3)(0) = -1.0    (-cos(0))
# f^(4)(0) = 0.0     (sin(0))
# f^(5)(0) = 1.0     (cos(0))
```

### Forward-over-reverse

Most efficient for Hessian-vector products:

```python
def hvp(f, x, v):
    """Hessian-vector product using forward-over-reverse."""
    return jax.jvp(jax.grad(f), (x,), (v,))[1]

# Usage
def f(x):
    return jnp.sum(x ** 3)

x = jnp.array([1.0, 2.0, 3.0])
v = jnp.array([1.0, 0.0, 0.0])
hvp_result = hvp(f, x, v)  # d^2f/dx^2 @ v
```

### Reverse-over-forward

```python
def hvp_revfwd(f, x, v):
    """Hessian-vector product using reverse-over-forward."""
    def fwd(x):
        return jax.jvp(f, (x,), (v,))[1]
    return jax.grad(fwd)(x)
```

### Choosing the composition

| Goal | Composition | Cost |
|---|---|---|
| Hessian (small n) | `jacfwd(jacrev(f))` | O(n) forward + O(1) backward |
| Hessian (large n) | `jacrev(jacfwd(f))` | O(1) forward + O(n) backward |
| HVP | `jvp(grad(f), ...)` | O(n) forward + O(1) backward |
| VHP | `grad(jvp(f, ...))` | O(1) forward + O(n) backward |

---

## 2. Jacobian-Vector Products (JVP)

### Efficient directional derivatives

```python
def f(x):
    return jnp.stack([jnp.sin(x[0]) * jnp.cos(x[1]),
                      x[0] ** 2 + x[1] ** 2,
                      x[0] * x[1]])

x = jnp.array([1.0, 2.0])

# Directional derivative along each axis
_, df_dx0 = jax.jvp(f, (x,), (jnp.array([1.0, 0.0]),))
_, df_dx1 = jax.jvp(f, (x,), (jnp.array([0.0, 1.0]),))
```

### Batched JVPs with `linearize`

```python
# Compute JVP for many tangent vectors efficiently
primal, jvp_fn = jax.linearize(f, x)

# Can call jvp_fn multiple times without re-tracing
tangents = jnp.eye(2)  # Two tangent vectors
for i in range(2):
    tangent = jvp_fn(tangents[i])
    print(f"Column {i} of Jacobian: {tangent}")
```

---

## 3. Vector-Jacobian Products (VJP)

### Per-output gradients

```python
def f(x):
    return jnp.array([jnp.sin(x), jnp.cos(x), x ** 2])

x = 1.0
y, vjp_fn = jax.vjp(f, x)

# Gradient of output[0] w.r.t. x
g0, = vjp_fn(jnp.array([1.0, 0.0, 0.0]))  # cos(1.0)

# Gradient of output[1] w.r.t. x
g1, = vjp_fn(jnp.array([0.0, 1.0, 0.0]))  # -sin(1.0)

# Gradient of output[2] w.r.t. x
g2, = vjp_fn(jnp.array([0.0, 0.0, 1.0]))  # 2.0
```

### VJP with multiple inputs

```python
def f(x, y, z):
    return x * y + z ** 2

primals, vjp_fn = jax.vjp(f, 2.0, 3.0, 4.0)
result = primals  # 2*3 + 16 = 22
grads = vjp_fn(1.0)  # (dy/dx, dy/dy, dy/dz) = (3.0, 2.0, 8.0)
```

---

## 4. Differentiating Through Control Flow

### scan (recurrent loops)

```python
def rnn_loss(params, xs, y_target):
    def step(carry, x):
        h = jnp.tanh(params['w_h'] @ carry + params['w_x'] @ x + params['b'])
        return h, h

    h0 = jnp.zeros(hidden_size)
    _, hs = jax.lax.scan(step, h0, xs)
    return jnp.mean((hs[-1] - y_target) ** 2)

# grad works through scan
grads = jax.grad(rnn_loss)(params, xs, y_target)
```

### while_loop

```python
def newton_step(f, x0):
    df = jax.grad(f)

    def cond(state):
        x, i = state
        return jnp.abs(f(x)) > 1e-6

    def body(state):
        x, i = state
        x = x - f(x) / df(x)
        return x, i + 1

    x_final, _ = jax.lax.while_loop(cond, body, (x0, 0))
    return x_final

# Differentiate through Newton's method
grad_newton = jax.grad(lambda x0: newton_step(lambda x: x**3 - 2, x0))
```

### cond and switch

```python
def piecewise_loss(params, x, branch):
    def branch_a(p, x):
        return jnp.mean((p['w'] @ x) ** 2)

    def branch_b(p, x):
        return jnp.mean(jnp.abs(p['w'] @ x))

    loss = jax.lax.switch(branch, [branch_a, branch_b], params, x)
    return loss

# grad works through switch
grads = jax.grad(piecewise_loss)(params, x, branch_idx)
```

---

## 5. Implicit Function Differentiation

### Fixed-point differentiation

```python
def fixed_point_layer(params, x):
    """Differentiate through a fixed-point iteration."""
    def cond(state):
        z, i = state
        return jnp.linalg.norm(f(params, z, x) - z) > 1e-6

    def body(state):
        z, i = state
        return f(params, z, x), i + 1

    z_star, _ = jax.lax.while_loop(cond, body, (x, 0))
    return z_star

# JAX automatically differentiates through the unrolled loop
grads = jax.grad(lambda p: loss(fixed_point_layer(p, x)))(params)
```

---

## 6. Differentiating with respect to structures

### Dictionary parameters

```python
params = {
    'layer0': {'w': jnp.ones((4, 3)), 'b': jnp.zeros(3)},
    'layer1': {'w': jnp.ones((3, 2)), 'b': jnp.zeros(2)},
}

def loss(params, x, y):
    h = jnp.relu(params['layer0']['w'] @ x + params['layer0']['b'])
    pred = params['layer1']['w'] @ h + params['layer1']['b']
    return jnp.mean((pred - y) ** 2)

grads = jax.grad(loss)(params, x, y)
# grads has the same pytree structure as params
```

### Nested grad + vmap for ensemble gradients

```python
# Gradient for each member of an ensemble
def ensemble_loss(params, x, y):
    preds = jax.vmap(lambda p: predict(p, x))(params)
    return jnp.mean((preds - y) ** 2)

grads = jax.grad(ensemble_loss)(ensemble_params, x, y)
```

---

## 7. Efficient Gradient Computation

### Gradient accumulation

```python
@jax.jit
def accumulated_grad(params, xs, ys, batch_size):
    """Accumulate gradients over mini-batches."""
    def batch_grad(carry, batch_idx):
        params, total_grad = carry
        start = batch_idx * batch_size
        x_batch = jax.lax.dynamic_slice(xs, (start,), (batch_size,))
        y_batch = jax.lax.dynamic_slice(ys, (start,), (batch_size,))
        grads = jax.grad(loss)(params, x_batch, y_batch)
        new_total = jax.tree.map(lambda t, g: t + g, total_grad, grads)
        return (params, new_total), None

    n_batches = xs.shape[0] // batch_size
    zero_grads = jax.tree.map(jnp.zeros_like, params)
    (_, total_grads), _ = jax.lax.scan(
        batch_grad, (params, zero_grads),
        jnp.arange(n_batches)
    )
    return jax.tree.map(lambda g: g / n_batches, total_grads)
```

### Gradient checkpointing (remat)

See Chapter 20 for full details on `jax.checkpoint`/`jax.remat`.

```python
# Trade compute for memory
@jax.remat
def expensive_layer(x, w):
    for _ in range(100):
        x = jnp.tanh(x @ w)
    return x

# Now grad through this layer uses O(1) memory instead of O(100)
```

---

## 8. Differentiating Optimizers

### Meta-learning: gradient through optimization

```python
def inner_loop(params, x, y, lr=0.01, steps=5):
    """Inner optimization loop — differentiable."""
    def step(params, _):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
        return params, loss

    params, losses = jax.lax.scan(step, params, None, length=steps)
    return params, losses

# MAML: gradient of meta-loss through inner loop
def meta_loss(meta_params, x, y, x_test, y_test):
    adapted_params, _ = inner_loop(meta_params, x, y)
    test_loss = loss_fn(adapted_params, x_test, y_test)
    return test_loss

meta_grads = jax.grad(meta_loss)(meta_params, x, y, x_test, y_test)
```

---

## 9. Defining Custom Differentiation Rules

### `custom_jvp`

```python
@jax.custom_jvp
def f(x):
    return jnp.where(x > 0, x ** 2, 0.0)

@f.defjvp
def f_jvp(primals, tangents):
    x, = primals
    dx, = tangents
    # Custom forward-mode rule
    return f(x), jnp.where(x > 0, 2 * x * dx, 0.0)

# Now grad(f) uses the custom rule
```

### `custom_vjp`

```python
@jax.custom_vjp
def clip_gradient(x, threshold=1.0):
    # Forward: identity
    return x

def clip_fwd(x, threshold=1.0):
    return x, (threshold,)

def clip_bwd(res, g):
    threshold, = res
    # Reverse: clip gradient
    return (jnp.clip(g, -threshold, threshold),)

clip_gradient.defvjp(clip_fwd, clip_bwd)
```

### `custom_gradient`

```python
@jax.custom_gradient
def safe_sqrt(x):
    y = jnp.sqrt(jnp.maximum(x, 1e-8))
    def grad_fn(dy):
        return dy / (2 * y)
    return y, grad_fn
```

---

## 10. Numerical Stability in AD

### Log-sum-exp trick

```python
def logsumexp(x):
    c = jnp.max(x)
    return c + jnp.log(jnp.sum(jnp.exp(x - c)))

# JAX's built-in is stable
jnp.logsumexp(x)  # Same as scipy.special.logsumexp
```

### Softmax stability

```python
# Stable softmax: subtract max before exp
def stable_softmax(x):
    x = x - jnp.max(x, axis=-1, keepdims=True)
    return jnp.exp(x) / jnp.sum(jnp.exp(x), axis=-1, keepdims=True)
```

### Gradient clipping during computation

```python
def clipped_grad(f, max_norm):
    def grad_fn(*args):
        grads = jax.grad(f)(*args)
        norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree.leaves(grads)))
        scale = jnp.minimum(1.0, max_norm / (norm + 1e-6))
        return jax.tree.map(lambda g: g * scale, grads)
    return grad_fn
```

---

## 11. AD Performance Tips

### Prefer `value_and_grad` over separate calls

```python
# BAD: Two passes
loss = loss_fn(params, x, y)
grads = jax.grad(loss_fn)(params, x, y)

# GOOD: One pass
loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
```

### Use `has_aux` to avoid recomputation

```python
# BAD: Computing loss again after grad
grads = jax.grad(loss_fn)(params, x, y)
loss = loss_fn(params, x, y)  # Recomputes!

# GOOD: Return loss as aux
def loss_with_logits(params, x, y):
    logits = predict(params, x)
    loss = cross_entropy(logits, y)
    return loss, logits

grads, logits = jax.grad(loss_with_logits, has_aux=True)(params, x, y)
```

### JIT the entire training step

```python
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss
```
