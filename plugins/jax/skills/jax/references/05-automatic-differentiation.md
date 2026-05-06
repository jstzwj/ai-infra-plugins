# Automatic Differentiation in JAX

This document provides an exhaustive reference for JAX's automatic differentiation (autodiff) system. It covers reverse-mode differentiation (`jax.grad`), forward-mode differentiation (`jax.jvp`), Jacobian computations, higher-order derivatives, complex differentiation, and advanced patterns for differentiating with respect to nested data structures.

---

## Table of Contents

1. [Overview of JAX Autodiff](#1-overview-of-jax-autodiff)
2. [jax.grad: Reverse-Mode Differentiation](#2-jaxgrad-reverse-mode-differentiation)
3. [Higher-Order Derivatives](#3-higher-order-derivatives)
4. [jax.value_and_grad](#4-jaxvalue_and_grad)
5. [jax.jacfwd and jax.jacrev: Jacobians](#5-jaxjacfwd-and-jaxjacrev-jacobians)
6. [jax.hessian](#6-jaxhessian)
7. [jvp: Forward-Mode Differentiation](#7-jvp-forward-mode-differentiation)
8. [vjp: Reverse-Mode Differentiation (Explicit)](#8-vjp-reverse-mode-differentiation-explicit)
9. [jax.linearize and jax.linear_transpose](#9-jaxlinearize-and-jaxlinear_transpose)
10. [Differentiating w.r.t. Nested Structures (Pytrees)](#10-differentiating-wrt-nested-structures-pytrees)
11. [Checking Gradients Numerically](#11-checking-gradients-numerically)
12. [Complex Differentiation](#12-complex-differentiation)
13. [Custom Derivative Rules](#13-custom-derivative-rules)
14. [Practical Recipes](#14-practical-recipes)

---

## 1. Overview of JAX Autodiff

JAX provides two fundamental modes of automatic differentiation:

- **Reverse-mode** (aka backpropagation): Efficient when the function has many inputs and few outputs. The cost scales with the number of outputs. Implemented via `jax.grad`, `jax.vjp`, `jax.jacrev`.
- **Forward-mode**: Efficient when the function has few inputs and many outputs. The cost scales with the number of inputs. Implemented via `jax.jvp`, `jax.jacfwd`, `jax.linearize`.

Both modes are fully composable and support arbitrary-order derivatives through nesting.

```python
import jax
import jax.numpy as jnp

# Reverse-mode: one output -> gradient w.r.t. all inputs
grad_f = jax.grad(lambda x: jnp.sum(x ** 3))

# Forward-mode: push tangents through the function
tangent_out, primal_out = jax.jvp(lambda x: jnp.sum(x ** 3),
                                  (jnp.array([1.0, 2.0]),),
                                  (jnp.array([1.0, 1.0]),))
```

### Key Principles

1. **Functions, not tensors**: JAX differentiates *functions*, not computational graphs. You pass a Python function to `jax.grad`, and it returns a new function that computes the gradient.
2. **Scalar output requirement**: `jax.grad` requires the function to return a scalar. Use `jax.jacobian` or `jax.vjp` for vector-valued outputs.
3. **Pure functions**: Differentiated functions must be pure (no side effects, no mutation of inputs).
4. **Differentiable operations only**: All operations in the function must have defined differentiation rules. Most `jax.numpy` operations are differentiable; operations like sorting, indexing with integer arrays, or `jnp.where` with non-differentiable branches have specific gradient behaviors.

---

## 2. jax.grad: Reverse-Mode Differentiation

### Basic Usage

```python
import jax
import jax.numpy as jnp

# Simple scalar function
def f(x):
    return x ** 3 + 2 * x ** 2 - x + 1

# df/dx = 3x^2 + 4x - 1
df = jax.grad(f)

x = 3.0
print(f"f({x}) = {f(x)}")
print(f"f'({x}) = {df(x)}")
print(f"Expected: {3 * x**2 + 4 * x - 1}")
```

### Signature

```python
jax.grad(
    fun,            # Function to differentiate
    argnums=0,      # Which positional argument(s) to differentiate w.r.t.
    has_aux=False,  # Whether fun returns (value, aux) tuple
    holomorphic=False,  # Whether fun is holomorphic (complex-analytic)
    allow_int=False,    # Whether to allow integer arguments (returns zero grad)
    reduce_axes=(),     # Axes over which to reduce the output gradient
)
```

### argnums: Differentiating w.r.t. Multiple Arguments

`argnums` specifies which positional arguments to differentiate with respect to. It can be a single integer or a tuple of integers.

```python
import jax
import jax.numpy as jnp

# Function of two variables: f(x, y) = x^2 * y + y^3
def f(x, y):
    return x ** 2 * y + y ** 3

# Gradient w.r.t. x only (default argnums=0)
df_dx = jax.grad(f)
print(f"df/dx at (2, 3) = {df_dx(2.0, 3.0)}")  # 2*x*y = 12.0

# Gradient w.r.t. y only
df_dy = jax.grad(f, argnums=1)
print(f"df/dy at (2, 3) = {df_dy(2.0, 3.0)}")  # x^2 + 3*y^2 = 31.0

# Gradient w.r.t. both x and y (returns tuple of gradients)
df_dxy = jax.grad(f, argnums=(0, 1))
grad_x, grad_y = df_dxy(2.0, 3.0)
print(f"df/dx = {grad_x}, df/dy = {grad_y}")  # 12.0, 31.0
```

### has_aux: Returning Auxiliary Data

When the function returns auxiliary data (e.g., loss value along with metrics), set `has_aux=True`. The function should return `(scalar_value, aux_data)`.

```python
import jax
import jax.numpy as jnp

def loss_fn(params, x, y):
    """Returns (loss, auxiliary_data)."""
    predictions = params["w"] * x + params["b"]
    residuals = predictions - y
    loss = jnp.mean(residuals ** 2)
    # Auxiliary data: metrics, predictions, etc.
    aux = {
        "predictions": predictions,
        "mae": jnp.mean(jnp.abs(residuals)),
        "mse": loss,
    }
    return loss, aux

# With has_aux=True, grad returns (gradient, aux)
grad_fn = jax.grad(loss_fn, has_aux=True)

params = {"w": jnp.array(2.0), "b": jnp.array(0.5)}
x = jnp.array([1.0, 2.0, 3.0])
y = jnp.array([2.5, 4.5, 6.5])

grads, aux = grad_fn(params, x, y)
print(f"Gradients: {grads}")
print(f"Auxiliary MAE: {aux['mae']}")
print(f"Auxiliary predictions: {aux['predictions']}")
```

### holomorphic: Complex-Valued Functions

Set `holomorphic=True` when differentiating holomorphic (complex-analytic) functions. The function must accept complex inputs and return complex outputs.

```python
import jax
import jax.numpy as jnp

# Holomorphic function: f(z) = z^2 (analytic everywhere)
def f(z):
    return z ** 2

# df/dz = 2z (complex derivative)
df_dz = jax.grad(f, holomorphic=True)

z = 1.0 + 2.0j
print(f"f(z) = {f(z)}")         # (-3+4j)
print(f"df/dz = {df_dz(z)}")    # (2+4j)
print(f"Expected: {2 * z}")      # (2+4j)
```

### allow_int: Integer Arguments

By default, `jax.grad` raises an error if any differentiated argument has integer dtype. Set `allow_int=True` to permit integer arguments (the gradient w.r.t. integer arguments is zero).

```python
import jax
import jax.numpy as jnp

def f(x, n):
    """x raised to the power n. n is an integer."""
    return x ** n

# Without allow_int: error because n is integer
# df = jax.grad(f, argnums=(0, 1))  # TypeError!

# With allow_int: gradient w.r.t. n is zero
df = jax.grad(f, argnums=(0, 1), allow_int=True)
grad_x, grad_n = df(3.0, 4)
print(f"grad w.r.t. x = {grad_x}")  # 4 * 3^3 = 108.0
print(f"grad w.r.t. n = {grad_n}")  # 0 (integer, no gradient)
```

### reduce_axes: Summing Over Batch Dimensions

When the function returns a non-scalar output, you can use `reduce_axes` to specify axes over which to sum before differentiating. This is equivalent to computing `grad(lambda x: f(x).sum(axis=reduce_axes))`.

```python
import jax
import jax.numpy as jnp

def f(x):
    """Returns a vector, not a scalar."""
    return x ** 2  # shape (4,)

# Method 1: Manually sum
grad_manual = jax.grad(lambda x: jnp.sum(f(x)))

# Method 2: Use reduce_axes
grad_reduce = jax.grad(f, reduce_axes=0)

x = jnp.array([1.0, 2.0, 3.0, 4.0])
print(f"Manual: {grad_manual(x)}")   # [2., 4., 6., 8.]
print(f"Reduce: {grad_reduce(x)}")   # [2., 4., 6., 8.]
```

### Gradient of Vector-Valued Functions

`jax.grad` requires a scalar output. For vector-valued functions, use `jax.jacrev` or manually reduce to a scalar:

```python
import jax
import jax.numpy as jnp

def f(x):
    return jnp.array([x ** 2, x ** 3, jnp.sin(x)])

# Method 1: Sum the output
grad_sum = jax.grad(lambda x: jnp.sum(f(x)))
print(f"grad(sum(f(x))) at x=1.0: {grad_sum(1.0)}")

# Method 2: Use jacrev for the full Jacobian (see section 5)
J = jax.jacrev(f)
print(f"Jacobian at x=1.0:\n{J(1.0)}")
```

---

## 3. Higher-Order Derivatives

JAX supports arbitrary-order derivatives by composing `jax.grad`. This is a direct consequence of JAX's transformation-composition design.

### Second Derivative (f'')

```python
import jax
import jax.numpy as jnp

def f(x):
    return jnp.sin(x)

f_prime = jax.grad(f)       # cos(x)
f_double_prime = jax.grad(jax.grad(f))  # -sin(x)

x = 1.0
print(f"f(x) = {f(x)}")
print(f"f'(x) = {f_prime(x)}")
print(f"f''(x) = {f_double_prime(x)}")
print(f"Expected f'(x) = {jnp.cos(x)}")
print(f"Expected f''(x) = {-jnp.sin(x)}")
```

### nth-Order Derivatives

```python
import jax
import jax.numpy as jnp
import functools

def nth_derivative(f, n):
    """Compute the nth derivative of f."""
    if n == 0:
        return f
    return jax.grad(nth_derivative(f, n - 1))

# Example: derivatives of sin(x)
def sin_fn(x):
    return jnp.sin(x)

x = 0.5
for n in range(6):
    d = nth_derivative(sin_fn, n)
    print(f"d^{n}/dx^{n} sin({x}) = {d(x):.6f}")

# Output:
# d^0/dx^0 sin(0.5) = 0.479426  (sin)
# d^1/dx^1 sin(0.5) = 0.877583  (cos)
# d^2/dx^2 sin(0.5) = -0.479426 (-sin)
# d^3/dx^3 sin(0.5) = -0.877583 (-cos)
# d^4/dx^4 sin(0.5) = 0.479426  (sin)
# d^5/dx^5 sin(0.5) = 0.877583  (cos)
```

### Higher-Order Gradients of Multivariable Functions

```python
import jax
import jax.numpy as jnp

# f(x, y) = x^3 * y^2 + x * y
def f(x, y):
    return x ** 3 * y ** 2 + x * y

# First derivatives
df_dx = jax.grad(f, argnums=0)
df_dy = jax.grad(f, argnums=1)

# Mixed partial: d^2f / dx dy
d2f_dxdy = jax.grad(df_dy, argnums=0)

# Second partial: d^2f / dx^2
d2f_dx2 = jax.grad(df_dx, argnums=0)

x, y = 2.0, 3.0
print(f"df/dx = {df_dx(x, y)}")          # 3x^2*y^2 + y = 109.0
print(f"df/dy = {df_dy(x, y)}")          # 2x^3*y + x = 50.0
print(f"d2f/dxdy = {d2f_dxdy(x, y)}")   # 6x^2*y + 1 = 73.0
print(f"d2f/dx2 = {d2f_dx2(x, y)}")     # 6x*y^2 = 108.0
```

### Higher-Order Derivatives with jax.make_jaxpr

Inspect the computation graph of higher-order derivatives:

```python
import jax
import jax.numpy as jnp

def f(x):
    return jnp.sin(x) * jnp.exp(-x)

f_grad = jax.grad(f)
f_grad2 = jax.grad(f_grad)

# Inspect the JAX expression for the second derivative
jaxpr = jax.make_jaxpr(f_grad2)(1.0)
print(jaxpr)
```

---

## 4. jax.value_and_grad

`jax.value_and_grad` computes both the function value and its gradient in a single pass, which is more efficient than calling the function and then `jax.grad` separately.

### Basic Usage

```python
import jax
import jax.numpy as jnp

def loss_fn(params, x, y):
    return jnp.mean((params * x - y) ** 2)

# value_and_grad returns (value, gradient)
loss_and_grad = jax.value_and_grad(loss_fn)

params = jnp.array(2.0)
x = jnp.array([1.0, 2.0, 3.0])
y = jnp.array([1.5, 3.5, 5.5])

loss, grad = loss_and_grad(params, x, y)
print(f"Loss: {loss}")
print(f"Gradient: {grad}")
```

### With has_aux

```python
import jax
import jax.numpy as jnp

def loss_fn(params, x, y):
    predictions = params["w"] * x + params["b"]
    loss = jnp.mean((predictions - y) ** 2)
    aux = {"predictions": predictions, "loss": loss}
    return loss, aux

# When has_aux=True, value_and_grad returns ((value, aux), gradient)
loss_and_grad = jax.value_and_grad(loss_fn, has_aux=True)

params = {"w": jnp.array(1.5), "b": jnp.array(0.0)}
x = jnp.array([1.0, 2.0, 3.0])
y = jnp.array([2.0, 4.0, 6.0])

(loss, aux), grads = loss_and_grad(params, x, y)
print(f"Loss: {loss}")
print(f"Aux predictions: {aux['predictions']}")
print(f"Gradients: {grads}")
```

### Typical Training Loop Pattern

```python
import jax
import jax.numpy as jnp

def model(params, x):
    """Simple linear model."""
    return params["w"] @ x + params["b"]

def loss_fn(params, x, y):
    """MSE loss with L2 regularization."""
    preds = model(params, x)
    mse = jnp.mean((preds - y) ** 2)
    l2_reg = 0.01 * (jnp.sum(params["w"] ** 2) + jnp.sum(params["b"] ** 2))
    return mse + l2_reg

# Compute loss and gradients together
loss_and_grad = jax.value_and_grad(loss_fn)

# Initialize parameters
key = jax.random.key(42)
params = {
    "w": jax.random.normal(key, (3, 4)),
    "b": jnp.zeros(3),
}

# Simulated data
x = jax.random.normal(jax.random.key(0), (4, 10))
y = jax.random.normal(jax.random.key(1), (3, 10))

# One training step
def train_step(params, x, y, lr=0.01):
    (loss, grads) = loss_and_grad(params, x, y)
    # SGD update
    params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
    return params, loss

params, loss = train_step(params, x, y)
print(f"Loss after one step: {loss}")
```

---

## 5. jax.jacfwd and jax.jacrev: Jacobians

The Jacobian matrix of a function `f: R^n -> R^m` is an `m x n` matrix of partial derivatives. JAX provides two ways to compute it:

- `jax.jacrev`: Uses reverse-mode differentiation. Efficient when `m < n` (fewer outputs than inputs).
- `jax.jacfwd`: Uses forward-mode differentiation. Efficient when `n < m` (fewer inputs than outputs).

### jacrev (Reverse-Mode Jacobian)

```python
import jax
import jax.numpy as jnp

def f(x):
    """f: R^3 -> R^2"""
    return jnp.array([x[0] ** 2 + x[1] * x[2],
                      x[0] * x[1] - x[2] ** 2])

J = jax.jacrev(f)
x = jnp.array([1.0, 2.0, 3.0])
jacobian = J(x)

print(f"Jacobian shape: {jacobian.shape}")  # (2, 3)
print(f"Jacobian:\n{jacobian}")

# Manual verification:
# df1/dx0 = 2*x0 = 2,   df1/dx1 = x2 = 3,   df1/dx2 = x1 = 2
# df2/dx0 = x1 = 2,     df2/dx1 = x0 = 1,    df2/dx2 = -2*x2 = -6
```

### jacfwd (Forward-Mode Jacobian)

```python
import jax
import jax.numpy as jnp

def f(x):
    return jnp.array([jnp.sin(x[0]) * jnp.cos(x[1]),
                      jnp.exp(x[0] + x[1])])

# Forward-mode: efficient for few inputs, many outputs
J = jax.jacfwd(f)
x = jnp.array([0.0, 0.0])
print(f"Jacobian (forward-mode):\n{J(x)}")
# [[cos(0)*cos(0), -sin(0)*sin(0)],
#  [exp(0),         exp(0)         ]]
# = [[1.0, 0.0],
#    [1.0, 1.0]]
```

### argnums for Jacobians

```python
import jax
import jax.numpy as jnp

def f(x, y):
    """Function of two arguments."""
    return jnp.stack([x ** 2 + y, x * y, y ** 2 - x])

# Jacobian w.r.t. first argument (x)
J_x = jax.jacrev(f, argnums=0)
# Jacobian w.r.t. second argument (y)
J_y = jax.jacrev(f, argnums=1)
# Jacobian w.r.t. both arguments
J_xy = jax.jacrev(f, argnums=(0, 1))

x = jnp.array([1.0, 2.0])
y = jnp.array([3.0, 4.0])

Jx = J_x(x, y)   # shape (3, 2)
Jy = J_y(x, y)   # shape (3, 2)
Jx2, Jy2 = J_xy(x, y)  # tuple of (3, 2) arrays

print(f"J_x:\n{Jx}")
print(f"J_y:\n{Jy}")
```

### Jacobian-Vector and Vector-Jacobian Products

For large-scale problems, computing the full Jacobian matrix may be too expensive. Instead, compute Jacobian-vector products directly:

```python
import jax
import jax.numpy as jnp

def f(x):
    return jnp.array([x[0] ** 2 + x[1] ** 2, x[0] * x[1]])

x = jnp.array([1.0, 2.0])

# Jacobian-vector product (forward-mode): J @ v
v = jnp.array([1.0, 0.0])  # tangent vector
_, jvp_result = jax.jvp(f, (x,), (v,))
print(f"J @ v = {jvp_result}")  # df/dx0 evaluated at x

# Vector-Jacobian product (reverse-mode): v^T @ J
cotangent = jnp.array([1.0, 0.0])
_, vjp_fn = jax.vjp(f, x)
vjp_result, = vjp_fn(cotangent)
print(f"v^T @ J = {vjp_result}")
```

---

## 6. jax.hessian

`jax.hessian` computes the Hessian matrix (second-order partial derivatives) of a scalar-valued function.

### Basic Usage

```python
import jax
import jax.numpy as jnp

def f(x):
    """f(x) = x0^2 * x1 + x1^3"""
    return x[0] ** 2 * x[1] + x[1] ** 3

H = jax.hessian(f)
x = jnp.array([1.0, 2.0])
hessian = H(x)

print(f"Hessian shape: {hessian.shape}")  # (2, 2)
print(f"Hessian:\n{hessian}")

# Manual computation:
# df/dx0 = 2*x0*x1 -> d2f/dx0dx0 = 2*x1 = 4, d2f/dx0dx1 = 2*x0 = 2
# df/dx1 = x0^2 + 3*x1^2 -> d2f/dx1dx0 = 2*x0 = 2, d2f/dx1dx1 = 6*x1 = 12
```

### Implementation Detail

`jax.hessian(f)` is equivalent to `jax.jacfwd(jax.jacrev(f))`:

```python
import jax
import jax.numpy as jnp

def f(x):
    return jnp.sum(x ** 3)

x = jnp.array([1.0, 2.0, 3.0])

# Three equivalent ways to compute the Hessian:
H1 = jax.hessian(f)
H2 = jax.jacfwd(jax.jacrev(f))
H3 = jax.jacrev(jax.jacfwd(f))

print(f"jax.hessian:          \n{H1(x)}")
print(f"jacfwd(jacrev(f)):    \n{H2(x)}")
print(f"jacrev(jacfwd(f)):    \n{H3(x)}")
```

### Hessian-Vector Products

For large problems, computing the full Hessian is expensive. Use a Hessian-vector product instead:

```python
import jax
import jax.numpy as jnp

def f(x):
    return jnp.sum(x ** 2 * jnp.cos(x))

x = jnp.array([1.0, 2.0, 3.0])
v = jnp.array([1.0, 1.0, 1.0])

# Hessian-vector product: H @ v
# This uses only O(n) memory, not O(n^2) for the full Hessian
hvp = jax.grad(lambda x: jnp.dot(jax.grad(f)(x), v))
print(f"Hessian-vector product: {hvp(x)}")

# Alternative using jvp
def grad_f(x):
    return jax.grad(f)(x)

_, hvp_jvp = jax.jvp(grad_f, (x,), (v,))
print(f"HVP via jvp: {hvp_jvp}")
```

### Hessian for Optimization: Newton's Method

```python
import jax
import jax.numpy as jnp

def rosenbrock(x):
    """Rosenbrock function: a classic optimization test function."""
    return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

grad_fn = jax.grad(rosenbrock)
hess_fn = jax.hessian(rosenbrock)

x = jnp.array([0.0, 0.0])

for i in range(5):
    g = grad_fn(x)
    H = hess_fn(x)
    # Newton step: x_new = x - H^{-1} g
    delta = jnp.linalg.solve(H, g)
    x = x - delta
    print(f"Step {i}: x = {x}, f(x) = {rosenbrock(x):.6f}")
```

---

## 7. jvp: Forward-Mode Differentiation

`jax.jvp` (Jacobian-Vector Product) computes the forward-mode derivative. Given a function `f` and a tangent vector `v`, it computes `(f(x), J(x) @ v)`.

### Basic Usage

```python
import jax
import jax.numpy as jnp

def f(x):
    return x ** 3 + jnp.sin(x)

x = jnp.array(2.0)
v = jnp.array(1.0)  # tangent: differentiate w.r.t. x

# jvp returns (primal_out, tangent_out)
primal, tangent = jax.jvp(f, (x,), (v,))
print(f"f(x) = {primal}")      # f(2) = 8 + sin(2)
print(f"J @ v = {tangent}")    # f'(2) = 12 + cos(2)
```

### Multiple Arguments

```python
import jax
import jax.numpy as jnp

def f(x, y):
    return x ** 2 * y + y ** 3

x = jnp.array(2.0)
y = jnp.array(3.0)

# Tangent w.r.t. x only
primal, tangent = jax.jvp(f, (x, y), (jnp.array(1.0), jnp.array(0.0)))
print(f"f(x, y) = {primal}")
print(f"df/dx = {tangent}")  # 2*x*y = 12

# Tangent w.r.t. y only
primal, tangent = jax.jvp(f, (x, y), (jnp.array(0.0), jnp.array(1.0)))
print(f"df/dy = {tangent}")  # x^2 + 3*y^2 = 31

# Tangent w.r.t. both (directional derivative along (1, 1))
primal, tangent = jax.jvp(f, (x, y), (jnp.array(1.0), jnp.array(1.0)))
print(f"Directional derivative = {tangent}")  # 12 + 31 = 43
```

### Vector-Valued Functions

```python
import jax
import jax.numpy as jnp

def f(x):
    """f: R^3 -> R^2"""
    return jnp.array([
        x[0] * x[1] + x[2],
        x[0] ** 2 - x[1] * x[2],
    ])

x = jnp.array([1.0, 2.0, 3.0])
v = jnp.array([1.0, 0.0, 0.0])  # tangent along first component

primal, tangent = jax.jvp(f, (x,), (v,))
print(f"f(x) = {primal}")          # [5.0, -5.0]
print(f"J @ v = {tangent}")        # df/dx0 = [x1, 2*x0] = [2.0, 2.0]
```

### Computing the Full Jacobian via jvp

```python
import jax
import jax.numpy as jnp

def f(x):
    return jnp.array([jnp.sin(x[0]) * jnp.cos(x[1]),
                      x[0] ** 2 + x[1] ** 2])

x = jnp.array([0.5, 1.0])

# Compute full Jacobian by applying jvp with each basis vector
n = len(x)
eye = jnp.eye(n)

def jacobian_jvp(f, x):
    n = x.shape[0]
    jac_rows = []
    for i in range(n):
        _, col = jax.jvp(f, (x,), (eye[i],))
        jac_rows.append(col)
    return jnp.stack(jac_rows, axis=-1)

J = jacobian_jvp(f, x)
print(f"Jacobian (via jvp):\n{J}")

# Compare with jacfwd
J_jacfwd = jax.jacfwd(f)(x)
print(f"Jacobian (via jacfwd):\n{J_jacfwd}")
```

### jvp Inside JIT

```python
import jax
import jax.numpy as jnp

@jax.jit
def compute_jvp(x, v):
    """JIT-compatible forward-mode differentiation."""
    def f(x):
        return jnp.sum(x ** 3) + jnp.sin(x[0])

    return jax.jvp(f, (x,), (v,))

x = jnp.array([1.0, 2.0, 3.0])
v = jnp.array([1.0, 0.0, 0.0])

primal, tangent = compute_jvp(x, v)
print(f"Primal: {primal}")
print(f"Tangent: {tangent}")
```

---

## 8. vjp: Reverse-Mode Differentiation (Explicit)

`jax.vjp` (Vector-Jacobian Product) provides the explicit reverse-mode differentiation interface. It returns the function value and a callable that computes the VJP.

### Basic Usage

```python
import jax
import jax.numpy as jnp

def f(x):
    return x ** 3 + jnp.sin(x)

x = jnp.array(2.0)

# vjp returns (primal_out, vjp_fn)
primal, vjp_fn = jax.vjp(f, x)

# The vjp function takes a cotangent (gradient of the output)
# and returns the gradient w.r.t. inputs
cotangent = jnp.array(1.0)  # df/df = 1
grad, = vjp_fn(cotangent)

print(f"f(x) = {primal}")      # 8 + sin(2)
print(f"df/dx = {grad}")       # 12 + cos(2)
```

### Vector-Valued Functions

```python
import jax
import jax.numpy as jnp

def f(x):
    """f: R^3 -> R^2"""
    return jnp.array([x[0] * x[1], x[1] * x[2]])

x = jnp.array([1.0, 2.0, 3.0])

primal, vjp_fn = jax.vjp(f, x)

# Cotangent in output space (R^2)
cotangent = jnp.array([1.0, 0.0])  # gradient w.r.t. first output
grad1, = vjp_fn(cotangent)
print(f"v^T @ J for v=[1,0]: {grad1}")  # [x1, x0, 0] = [2, 1, 0]

cotangent = jnp.array([0.0, 1.0])  # gradient w.r.t. second output
grad2, = vjp_fn(cotangent)
print(f"v^T @ J for v=[0,1]: {grad2}")  # [0, x2, x1] = [0, 3, 2]
```

### Multiple Arguments

```python
import jax
import jax.numpy as jnp

def f(x, y):
    return x ** 2 * y + y ** 3

x = jnp.array(2.0)
y = jnp.array(3.0)

primal, vjp_fn = jax.vjp(f, x, y)
cotangent = jnp.array(1.0)

grad_x, grad_y = vjp_fn(cotangent)
print(f"f(x, y) = {primal}")
print(f"df/dx = {grad_x}")  # 2*x*y = 12
print(f"df/dy = {grad_y}")  # x^2 + 3*y^2 = 31
```

### Computing the Full Jacobian via vjp

```python
import jax
import jax.numpy as jnp

def f(x):
    """f: R^3 -> R^2"""
    return jnp.array([x[0] * x[1], x[1] * x[2], x[0] ** 2])

x = jnp.array([1.0, 2.0, 3.0])

primal, vjp_fn = jax.vjp(f, x)

# Compute full Jacobian by applying vjp with each basis vector in output space
m = primal.shape[0]
eye = jnp.eye(m)
jac_cols = []
for i in range(m):
    col, = vjp_fn(eye[i])
    jac_cols.append(col)
J = jnp.stack(jac_cols)

print(f"Jacobian (via vjp):\n{J}")
```

---

## 9. jax.linearize and jax.linear_transpose

### jax.linearize

`jax.linearize` computes both the primal value and a linear map (the forward-mode derivative function) that can be applied to tangent vectors multiple times. It is more efficient than calling `jvp` repeatedly when you need the derivative at the same point but with different tangent vectors.

```python
import jax
import jax.numpy as jnp

def f(x):
    return jnp.array([x[0] ** 2 * x[1], x[0] * x[1] ** 2])

x = jnp.array([2.0, 3.0])

# linearize returns (primal, linear_map)
primal, linear_map = jax.linearize(f, x)

print(f"Primal: {primal}")  # [12.0, 18.0]

# Evaluate the linearized function at different tangent vectors
v1 = jnp.array([1.0, 0.0])
v2 = jnp.array([0.0, 1.0])
v3 = jnp.array([1.0, 1.0])

# Each call to linear_map is O(1) -- no retracing needed
t1 = linear_map(v1)
t2 = linear_map(v2)
t3 = linear_map(v3)

print(f"J @ [1, 0] = {t1}")  # [2*x0*x1, x1^2] = [12, 9]
print(f"J @ [0, 1] = {t2}")  # [x0^2, 2*x0*x1] = [4, 12]
print(f"J @ [1, 1] = {t3}")  # [16, 21]

# Verify: J @ [1,1] should equal J @ [1,0] + J @ [0,1]
print(f"Sum: {t1 + t2}")
```

### jax.linear_transpose

`jax.linear_transpose` computes the transpose of a linear function. It is useful for implementing custom gradient rules and adjoint methods.

```python
import jax
import jax.numpy as jnp

# Define a linear function
def linear_fn(x):
    return 3.0 * x + 2.0 * jnp.roll(x, 1)

x = jnp.array([1.0, 2.0, 3.0, 4.0])

# Compute the transpose of the linear function
transpose_fn = jax.linear_transpose(linear_fn, x)

y = jnp.array([1.0, 1.0, 1.0, 1.0])
result, = transpose_fn(y)

print(f"Linear fn applied to x: {linear_fn(x)}")
print(f"Transpose applied to [1,1,1,1]: {result}")

# The transpose satisfies the adjoint property:
# <A*x, y> = <x, A^T*y>
lhs = jnp.dot(linear_fn(x), y)
rhs = jnp.dot(x, result)
print(f"<Ax, y> = {lhs}")
print(f"<x, A^T*y> = {rhs}")
print(f"Equal: {jnp.allclose(lhs, rhs)}")
```

### linear_transpose for Custom Gradients

```python
import jax
import jax.numpy as jnp

# Example: transpose of a matrix multiplication function
def matvec(W, x):
    """W @ x where W is a matrix."""
    return W @ x

W = jnp.array([[1.0, 2.0],
               [3.0, 4.0],
               [5.0, 6.0]])
x = jnp.array([1.0, 1.0])

# Transpose w.r.t. x (holding W fixed)
transpose_fn = jax.linear_transpose(lambda x: matvec(W, x), x)
y = jnp.array([1.0, 0.0, 0.0])
result, = transpose_fn(y)
print(f"W^T @ [1, 0, 0] = {result}")  # [1, 2] (first column of W)
```

---

## 10. Differentiating w.r.t. Nested Structures (Pytrees)

JAX can differentiate with respect to arbitrarily nested Python data structures (pytrees), including dicts, lists, tuples, and namedtuples.

### Dictionaries

```python
import jax
import jax.numpy as jnp

def loss_fn(params, x, y):
    """Neural network loss with nested parameter dict."""
    h = jnp.maximum(0, x @ params["layer1"]["w"] + params["layer1"]["b"])
    preds = h @ params["layer2"]["w"] + params["layer2"]["b"]
    return jnp.mean((preds - y) ** 2)

params = {
    "layer1": {
        "w": jax.random.normal(jax.random.key(0), (4, 8)),
        "b": jnp.zeros(8),
    },
    "layer2": {
        "w": jax.random.normal(jax.random.key(1), (8, 2)),
        "b": jnp.zeros(2),
    },
}

x = jax.random.normal(jax.random.key(2), (10, 4))
y = jax.random.normal(jax.random.key(3), (10, 2))

# Gradient has the same structure as params
grads = jax.grad(loss_fn)(params, x, y)
print(f"Gradient structure matches params: {jax.tree.structure(grads) == jax.tree.structure(params)}")
print(f"dL/d(layer1 w) shape: {grads['layer1']['w'].shape}")
print(f"dL/d(layer1 b) shape: {grads['layer1']['b'].shape}")
print(f"dL/d(layer2 w) shape: {grads['layer2']['w'].shape}")
print(f"dL/d(layer2 b) shape: {grads['layer2']['b'].shape}")
```

### Lists and Tuples

```python
import jax
import jax.numpy as jnp

def loss_fn(params, x):
    """params is a list of (weight, bias) tuples."""
    h = x
    for w, b in params:
        h = jnp.maximum(0, h @ w + b)
    return jnp.mean(h)

params = [
    (jax.random.normal(jax.random.key(0), (4, 8)), jnp.zeros(8)),
    (jax.random.normal(jax.random.key(1), (8, 3)), jnp.zeros(3)),
]

x = jax.random.normal(jax.random.key(2), (5, 4))

grads = jax.grad(loss_fn)(params, x)
print(f"Type of grads: {type(grads)}")       # list
print(f"Number of layers: {len(grads)}")     # 2
for i, (dw, db) in enumerate(grads):
    print(f"Layer {i}: dw shape = {dw.shape}, db shape = {db.shape}")
```

### NamedTuples

```python
import jax
import jax.numpy as jnp
from typing import NamedTuple

class Params(NamedTuple):
    w: jax.Array
    b: jax.Array

def loss_fn(params, x, y):
    preds = x @ params.w + params.b
    return jnp.mean((preds - y) ** 2)

params = Params(
    w=jax.random.normal(jax.random.key(0), (4, 2)),
    b=jnp.zeros(2),
)

x = jax.random.normal(jax.random.key(1), (10, 4))
y = jax.random.normal(jax.random.key(2), (10, 2))

grads = jax.grad(loss_fn)(params, x, y)
print(f"Type of grads: {type(grads)}")       # Params
print(f"grads.w shape: {grads.w.shape}")     # (4, 2)
print(f"grads.b shape: {grads.b.shape}")     # (2,)
```

### Mixed Structures

```python
import jax
import jax.numpy as jnp

def loss_fn(state, x):
    """State contains a dict of weights and a list of biases."""
    w = state["weights"]
    biases = state["biases"]
    h = x
    for i, (name, weight) in enumerate(w.items()):
        h = jnp.maximum(0, h @ weight + biases[i])
    return jnp.sum(h ** 2)

state = {
    "weights": {
        "layer0": jax.random.normal(jax.random.key(0), (3, 5)),
        "layer1": jax.random.normal(jax.random.key(1), (5, 2)),
    },
    "biases": [jnp.zeros(5), jnp.zeros(2)],
}

x = jax.random.normal(jax.random.key(2), (4, 3))
grads = jax.grad(loss_fn)(state, x)

# Gradient has identical structure
print(f"grad weights keys: {list(grads['weights'].keys())}")
print(f"grad biases length: {len(grads['biases'])}")
```

### Zero Gradients for Non-Differentiable Arguments

When using `argnums` to differentiate w.r.t. only some arguments, the structure of the gradient matches only the differentiated arguments:

```python
import jax
import jax.numpy as jnp

def f(params, data):
    return jnp.mean(params["w"] @ data)

params = {"w": jnp.array([1.0, 2.0, 3.0])}
data = jnp.array([0.5, 0.3, 0.2])

# Differentiate w.r.t. params only
grad = jax.grad(f, argnums=0)(params, data)
print(f"Gradient w.r.t. params: {grad}")
```

---

## 11. Checking Gradients Numerically

JAX provides `jax.check_grads` and you can also manually verify gradients using finite differences.

### jax.check_grads

```python
import jax
import jax.numpy as jnp

def f(x):
    return jnp.sum(jnp.sin(x) * jnp.exp(-x ** 2))

# check_grads verifies autodiff against numerical differentiation
x = jnp.array([0.5, 1.0, 1.5])

# Check first-order gradients
jax.check_grads(f, (x,), order=1)

# Check second-order gradients
jax.check_grads(f, (x,), order=2)

print("Gradient check passed!")
```

### Manual Finite Difference Check

```python
import jax
import jax.numpy as jnp

def finite_diff_grad(f, x, eps=1e-5):
    """Compute gradient using central finite differences."""
    grads = jnp.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.at[i].add(eps)
        x_minus = x.at[i].add(-eps)
        grads = grads.at[i].set((f(x_plus) - f(x_minus)) / (2 * eps))
    return grads

def f(x):
    return jnp.sum(x ** 3 + 2 * x ** 2 - x)

x = jnp.array([1.0, 2.0, 3.0])

# Autodiff gradient
autodiff_grad = jax.grad(f)(x)

# Numerical gradient
numerical_grad = finite_diff_grad(f, x)

print(f"Autodiff: {autodiff_grad}")
print(f"Numerical: {numerical_grad}")
print(f"Max difference: {jnp.max(jnp.abs(autodiff_grad - numerical_grad))}")
```

### Checking Gradients for Functions with Multiple Arguments

```python
import jax
import jax.numpy as jnp

def f(x, y, z):
    return jnp.sum(x ** 2 * y + z ** 3)

args = (jnp.array([1.0, 2.0]),
        jnp.array([3.0, 4.0]),
        jnp.array([0.5, 1.5]))

# Check gradients w.r.t. all arguments
jax.check_grads(f, args, order=1)

# Check only specific arguments
jax.check_grads(f, args, order=1, modes=["rev"])
jax.check_grads(f, args, order=1, modes=["fwd"])

print("All gradient checks passed!")
```

### check_grads with Pytree Arguments

```python
import jax
import jax.numpy as jnp

def loss_fn(params, x, y):
    preds = params["w"] @ x + params["b"]
    return jnp.mean((preds - y) ** 2)

params = {
    "w": jnp.array([[1.0, 2.0], [3.0, 4.0]]),
    "b": jnp.array([0.5, 0.5]),
}
x = jnp.array([1.0, 1.0])
y = jnp.array([3.5, 7.5])

jax.check_grads(loss_fn, (params, x, y), order=1)
print("Pytree gradient check passed!")
```

---

## 12. Complex Differentiation

### Holomorphic Functions

For holomorphic (complex-analytic) functions, JAX can compute complex gradients directly:

```python
import jax
import jax.numpy as jnp

# Holomorphic function: f(z) = z^2 + exp(z)
def f(z):
    return z ** 2 + jnp.exp(z)

grad_f = jax.grad(f, holomorphic=True)

z = jnp.array(1.0 + 0.5j)
print(f"f(z) = {f(z)}")
print(f"f'(z) = {grad_f(z)}")  # 2z + exp(z)
print(f"Expected: {2 * z + jnp.exp(z)}")
```

### Non-Holomorphic Functions (Wirtinger Calculus)

For non-holomorphic functions (e.g., functions involving `jnp.conj`, `jnp.abs`), JAX uses the convention of differentiating as if the function were a function of two real variables (real and imaginary parts).

```python
import jax
import jax.numpy as jnp

# |z|^2 is NOT holomorphic (involves conjugation)
def f(z):
    return jnp.real(z * jnp.conj(z))  # |z|^2

# For real-valued functions of complex variables,
# JAX computes the conjugate Wirtinger derivative
grad_f = jax.grad(f)

z = jnp.array(3.0 + 4.0j)
g = grad_f(z)
print(f"|z|^2 = {f(z)}")       # 25.0
print(f"grad = {g}")            # (6+0j), which is 2*Re(z)

# This is consistent with the Wirtinger calculus:
# d|z|^2/dz_conj = z = (3+4j)
# d|z|^2/dz = conj(z) = (3-4j)
# JAX returns 2*conj(z) for grad of |z|^2 because it treats
# real and imaginary parts as independent real variables
```

### Complex Neural Network Example

```python
import jax
import jax.numpy as jnp

def complex_linear(params, x):
    """Complex-valued linear layer."""
    return params["w"] @ x + params["b"]

def complex_mse_loss(params, x, y):
    """MSE loss for complex-valued predictions."""
    preds = complex_linear(params, x)
    return jnp.real(jnp.mean(jnp.abs(preds - y) ** 2))

params = {
    "w": jax.random.normal(jax.random.key(0), (3, 4)) + 1j * jax.random.normal(jax.random.key(1), (3, 4)),
    "b": jnp.zeros(3, dtype=jnp.complex64),
}

x = jax.random.normal(jax.random.key(2), (4,)) + 1j * jax.random.normal(jax.random.key(3), (4,))
y = jax.random.normal(jax.random.key(4), (3,)) + 1j * jax.random.normal(jax.random.key(5), (3,))

# Gradient computation works with complex parameters
grads = jax.grad(complex_mse_loss)(params, x, y)
print(f"Gradient for w: {grads['w']}")
print(f"Gradient for b: {grads['b']}")
```

### Real-to-Complex and Complex-to-Real

```python
import jax
import jax.numpy as jnp

# Real -> Complex: f(x) = exp(ix)
def real_to_complex(x):
    return jnp.exp(1j * x)

x = jnp.array(1.0)
g = jax.grad(lambda x: jnp.real(real_to_complex(x)))(x)
print(f"d/dx Re[exp(ix)] at x=1: {g}")  # -sin(1)

# Complex -> Real: f(z) = |z|
def complex_to_real(z):
    return jnp.abs(z)

z = jnp.array(3.0 + 4.0j)
g = jax.grad(complex_to_real)(z)
print(f"d/dz |z| at z=3+4i: {g}")  # z/|z| = (3+4j)/5
```

---

## 13. Custom Derivative Rules

### jax.custom_jvp: Custom Forward-Mode Rules

```python
import jax
import jax.numpy as jnp

@jax.custom_jvp
def safe_sqrt(x):
    """Square root with custom gradient that avoids NaN at zero."""
    return jnp.sqrt(x)

@safe_sqrt.defjvp
def safe_sqrt_jvp(primals, tangents):
    x, = primals
    dx, = tangents
    primal_out = safe_sqrt(x)
    # Custom tangent: clamp denominator to avoid division by zero
    tangent_out = dx / (2 * jnp.maximum(primal_out, 1e-8))
    return primal_out, tangent_out

# Usage
x = jnp.array(0.0)
print(f"sqrt(0) = {safe_sqrt(x)}")
print(f"grad at 0 = {jax.grad(safe_sqrt)(x)}")  # finite value, not NaN

x = jnp.array(4.0)
print(f"sqrt(4) = {safe_sqrt(x)}")
print(f"grad at 4 = {jax.grad(safe_sqrt)(x)}")  # 0.25 = 1/(2*2)
```

### jax.custom_vjp: Custom Reverse-Mode Rules

```python
import jax
import jax.numpy as jnp

@jax.custom_vjp
def clip_and_scale(x, threshold=1.0):
    """Clip values and scale, with custom backward pass."""
    return jnp.where(jnp.abs(x) > threshold, jnp.sign(x) * threshold, x) * 2.0

def clip_and_scale_fwd(x, threshold=1.0):
    """Forward pass: returns output and residuals for backward."""
    out = clip_and_scale(x, threshold)
    mask = jnp.abs(x) <= threshold  # which values were not clipped
    return out, (mask,)

def clip_and_scale_bwd(res, g):
    """Backward pass: zero gradient for clipped values."""
    mask, = res
    # Only propagate gradient through non-clipped values
    return (g * mask * 2.0, None)  # None for threshold (not differentiable)

clip_and_scale.defvjp(clip_and_scale_fwd, clip_and_scale_bwd)

# Usage
x = jnp.array([-2.0, -0.5, 0.0, 0.5, 2.0])
print(f"Output: {clip_and_scale(x)}")
print(f"Gradient: {jax.grad(lambda x: jnp.sum(clip_and_scale(x)))(x)}")
# Gradient is [0, 2, 2, 2, 0]: zero for clipped values, 2 for others
```

### custom_vjp for Gradient Checkpointing

```python
import jax
import jax.numpy as jnp

@jax.custom_vjp
def checkpointed_fn(x):
    """A function where we recompute forward pass during backward."""
    return jnp.sin(x) ** 2 + jnp.cos(x) ** 2  # always 1.0

def fwd(x):
    return checkpointed_fn(x), (x,)

def bwd(res, g):
    x, = res
    # Recompute forward pass to save memory (trade compute for memory)
    local_grad = 2 * jnp.sin(x) * jnp.cos(x) - 2 * jnp.cos(x) * jnp.sin(x)  # 0
    return (g * local_grad,)

checkpointed_fn.defvjp(fwd, bwd)

x = jnp.array(1.0)
print(f"f(x) = {checkpointed_fn(x)}")  # 1.0
print(f"grad = {jax.grad(checkpointed_fn)(x)}")  # 0.0
```

---

## 14. Practical Recipes

### Simple Gradient Descent

```python
import jax
import jax.numpy as jnp

def quadratic(params):
    """f(x) = 0.5 * x^T A x - b^T x"""
    A = jnp.array([[3.0, 1.0], [1.0, 2.0]])
    b = jnp.array([1.0, 2.0])
    return 0.5 * params @ A @ params - b @ params

grad_fn = jax.jit(jax.grad(quadratic))

x = jnp.array([5.0, 5.0])
for step in range(50):
    g = grad_fn(x)
    x = x - 0.1 * g
    if step % 10 == 0:
        print(f"Step {step}: x = {x}, f(x) = {quadratic(x):.6f}")
print(f"Final: x = {x}")
```

### Adam Optimizer with value_and_grad

```python
import jax
import jax.numpy as jnp

def adam_update(grads, state, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    """Single Adam optimizer step."""
    state["m"] = beta1 * state["m"] + (1 - beta1) * grads
    state["v"] = beta2 * state["v"] + (1 - beta2) * grads ** 2
    m_hat = state["m"] / (1 - beta1 ** state["t"])
    v_hat = state["v"] / (1 - beta2 ** state["t"])
    updates = lr * m_hat / (jnp.sqrt(v_hat) + eps)
    state["t"] += 1
    return updates, state

def loss_fn(params, x, y):
    return jnp.mean((x @ params["w"] + params["b"] - y) ** 2)

loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

# Setup
key = jax.random.key(0)
params = {
    "w": jax.random.normal(key, (4, 1)) * 0.01,
    "b": jnp.zeros(1),
}
adam_state = {
    "m": jax.tree.map(jnp.zeros_like, params),
    "v": jax.tree.map(jnp.zeros_like, params),
    "t": 1,
}

# Training data
x_data = jax.random.normal(jax.random.key(1), (100, 4))
y_data = x_data @ jnp.array([[1.0], [2.0], [-1.0], [0.5]]) + 0.1

# Training loop
for step in range(200):
    loss, grads = loss_and_grad(params, x_data, y_data)
    updates, adam_state = adam_update(grads, adam_state)
    params = jax.tree.map(lambda p, u: p - u, params, updates)
    if step % 50 == 0:
        print(f"Step {step}: loss = {loss:.6f}")
```

### Computing Per-Sample Gradients

```python
import jax
import jax.numpy as jnp

def loss_fn(params, x, y):
    """Per-sample MSE loss for a single sample."""
    pred = params["w"] @ x + params["b"]
    return jnp.mean((pred - y) ** 2)

# vmap over the batch dimension to get per-sample gradients
def per_sample_grads(params, x_batch, y_batch):
    grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))(
        params, x_batch, y_batch
    )
    return grads

params = {
    "w": jax.random.normal(jax.random.key(0), (3, 5)),
    "b": jnp.zeros(3),
}

x_batch = jax.random.normal(jax.random.key(1), (8, 5))   # 8 samples
y_batch = jax.random.normal(jax.random.key(2), (8, 3))

grads = per_sample_grads(params, x_batch, y_batch)
print(f"Per-sample grad for w shape: {grads['w'].shape}")  # (8, 3, 5)
print(f"Per-sample grad for b shape: {grads['b'].shape}")  # (8, 3)
```

### Gradient Clipping

```python
import jax
import jax.numpy as jnp

def clip_grad_norm(grads, max_norm=1.0):
    """Clip gradient by global norm."""
    leaves = jax.tree.leaves(grads)
    l2_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in leaves))
    scale = jnp.minimum(1.0, max_norm / (l2_norm + 1e-6))
    return jax.tree.map(lambda g: g * scale, grads), l2_norm

def loss_fn(params, x, y):
    return jnp.mean((x @ params - y) ** 2)

params = jax.random.normal(jax.random.key(0), (5, 3)) * 10.0
x = jax.random.normal(jax.random.key(1), (10, 5))
y = jax.random.normal(jax.random.key(2), (10, 3))

grads = jax.grad(loss_fn)(params, x, y)
clipped_grads, grad_norm = clip_grad_norm(grads, max_norm=1.0)
print(f"Original grad norm: {grad_norm:.4f}")
print(f"Clipped grad norm: {jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree.leaves(clipped_grads))):.4f}")
```

### Stop Gradient

```python
import jax
import jax.numpy as jnp

def loss_with_stop_grad(predictions, targets):
    """Use stop_gradient to prevent backprop through certain paths."""
    # Compute a baseline (moving average) but don't differentiate through it
    baseline = jax.lax.stop_gradient(jnp.mean(predictions))
    # Only differentiate through the advantage
    advantage = predictions - baseline
    return jnp.mean(advantage * targets)

predictions = jnp.array([1.0, 2.0, 3.0])
targets = jnp.array([1.5, 2.5, 3.5])

grads = jax.grad(loss_with_stop_grad)(predictions, targets)
print(f"Gradients: {grads}")  # Only gradient through advantage, not baseline
```

### Gradient Accumulation

```python
import jax
import jax.numpy as jnp

def loss_fn(params, x, y):
    return jnp.mean((params["w"] @ x + params["b"] - y) ** 2)

grad_fn = jax.grad(loss_fn)

params = {
    "w": jax.random.normal(jax.random.key(0), (3, 4)),
    "b": jnp.zeros(3),
}

# Simulate gradient accumulation over micro-batches
micro_batches_x = jnp.array_split(jax.random.normal(jax.random.key(1), (16, 4)), 4)
micro_batches_y = jnp.array_split(jax.random.normal(jax.random.key(2), (16, 3)), 4)

# Accumulate gradients
accumulated_grads = jax.tree.map(jnp.zeros_like, params)
for x_micro, y_micro in zip(micro_batches_x, micro_batches_y):
    grads = grad_fn(params, x_micro, y_micro)
    accumulated_grads = jax.tree.map(lambda a, g: a + g, accumulated_grads, grads)

# Average over micro-batches
num_micro = len(micro_batches_x)
accumulated_grads = jax.tree.map(lambda g: g / num_micro, accumulated_grads)
print(f"Accumulated gradient norm: {jax.tree.reduce(lambda a, g: a + jnp.sum(g**2), accumulated_grads, 0.0) ** 0.5:.4f}")
```
