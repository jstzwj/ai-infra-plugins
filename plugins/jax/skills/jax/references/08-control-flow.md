# Control Flow in JAX

This document provides an exhaustive reference for all control flow mechanisms in JAX. It covers JAX-specific control flow primitives (`jax.lax`), their differentiability properties, interaction with JIT and autodiff, and how they differ from standard Python control flow.

---

## Table of Contents

1. [Python Control Flow vs JAX Control Flow](#1-python-control-flow-vs-jax-control-flow)
2. [lax.cond](#2-laxcond)
3. [lax.select](#3-laxselect)
4. [lax.switch](#4-laxswitch)
5. [lax.while_loop](#5-laxwhile_loop)
6. [lax.fori_loop](#6-laxfori_loop)
7. [lax.scan](#7-laxscan)
8. [lax.associative_scan](#8-laxassociative_scan)
9. [jnp.where](#9-jnpwhere)
10. [jnp.piecewise](#10-jnppiecewise)
11. [jnp.select](#11-jnpselect)
12. [Logical Operators](#12-logical-operators)
13. [Control Flow with JIT](#13-control-flow-with-jit)
14. [Control Flow with Autodiff](#14-control-flow-with-autodiff)
15. [Summary Table of Differentiability](#15-summary-table-of-differentiability)

---

## 1. Python Control Flow vs JAX Control Flow

### The Problem with Python Control Flow in JAX

JAX traces Python functions to build a computation graph. Python-level `if`, `while`, and `for` statements are evaluated at trace time, not at runtime. This means:

- **Python `if`** on traced values: the condition is evaluated once during tracing. Both branches cannot be explored.
- **Python `while`** on traced values: the loop is either always entered or never entered during tracing.
- **Python `for`** with data-dependent range: the range must be known at trace time.

```python
import jax
import jax.numpy as jnp

# PROBLEM: Python if on traced value
@jax.jit
def bad_abs(x):
    if x > 0:       # x is a tracer, not a concrete value
        return x    # JAX can't evaluate this at trace time
    else:
        return -x   # Error: Abstract tracer value where concrete value is needed

# SOLUTION: Use JAX control flow or data-dependent operations
@jax.jit
def good_abs(x):
    return jnp.where(x > 0, x, -x)  # Works with traced values

# Python if with STATIC values works fine:
@jax.jit
def static_branch(x, flag):
    if flag:          # flag is a Python bool, not a traced value
        return x + 1
    else:
        return x - 1
```

### When to Use What

| Scenario | Use |
|----------|-----|
| Condition on static (trace-time known) values | Python `if` |
| Condition on traced (runtime) values, scalar | `lax.cond` |
| Condition on traced values, element-wise | `jnp.where` / `lax.select` |
| Multiple branches on a single index | `lax.switch` |
| Data-dependent loop count | `lax.while_loop` |
| Fixed iteration count | `lax.fori_loop` or Python `for` |
| Loop with carry state, fixed count | `lax.scan` |
| Element-wise conditional | `jnp.where` |

---

## 2. lax.cond

`jax.lax.cond` is a data-dependent conditional that works inside JIT-compiled functions. It evaluates one of two branches based on a scalar boolean predicate.

### Signature

```python
jax.lax.cond(pred, true_fun, false_fun, *operands, operand=None)
```

### Basic Usage

```python
import jax
import jax.numpy as jnp
from jax import lax

def true_fn(x):
    return x ** 2

def false_fn(x):
    return -x

@jax.jit
def conditional(x, flag):
    return lax.cond(flag, true_fn, false_fn, x)

print(conditional(5.0, True))    # 25.0 (true branch: x^2)
print(conditional(5.0, False))   # -5.0 (false branch: -x)
```

### With Multiple Operands

```python
import jax
import jax.numpy as jnp
from jax import lax

@jax.jit
def safe_divide(a, b):
    """Divide a by b, return 0 when b is zero."""
    def div(args):
        a, b = args
        return a / b

    def zero(args):
        a, b = args
        return 0.0

    return lax.cond(b != 0, div, zero, (a, b))

print(safe_divide(10.0, 2.0))   # 5.0
print(safe_divide(10.0, 0.0))   # 0.0
```

### With Pytree Operands

```python
import jax
import jax.numpy as jnp
from jax import lax

def update_adam(state):
    """Adam-style update."""
    return {
        "m": 0.9 * state["m"] + 0.1 * state["g"],
        "v": 0.999 * state["v"] + 0.001 * state["g"] ** 2,
    }

def update_sgd(state):
    """SGD-style update."""
    return {
        "m": state["g"],
        "v": jnp.zeros_like(state["v"]),
    }

@jax.jit
def conditional_update(state, use_adam):
    return lax.cond(
        use_adam,
        update_adam,
        update_sgd,
        state,
    )

state = {
    "m": jnp.array([0.1, 0.2]),
    "v": jnp.array([0.01, 0.02]),
    "g": jnp.array([1.0, 2.0]),
}

print("Adam:", conditional_update(state, True))
print("SGD:", conditional_update(state, False))
```

### Branches Must Have Same Output Structure

Both branches must return pytrees with the same structure and dtypes. This is required because JAX must know the output type at trace time.

```python
import jax
import jax.numpy as jnp
from jax import lax

# CORRECT: same output structure
@jax.jit
def good_cond(x, flag):
    def add_one(x):
        return x + 1.0
    def sub_one(x):
        return x - 1.0
    return lax.cond(flag, add_one, sub_one, x)

# WRONG: different output structures
# @jax.jit
# def bad_cond(x, flag):
#     def return_scalar(x):
#         return x[0]         # scalar
#     def return_array(x):
#         return x            # array
#     return lax.cond(flag, return_scalar, return_array, x)
# # TypeError: true_fun and false_fun outputs must have same types
```

### Nesting cond

```python
import jax
import jax.numpy as jnp
from jax import lax

@jax.jit
def piecewise_linear(x):
    """Piecewise linear function with 3 segments."""
    def positive_branch(x):
        return lax.cond(
            x > 2.0,
            lambda x: 2.0 * x + 1.0,
            lambda x: x + 3.0,
            x,
        )

    def negative_branch(x):
        return lax.cond(
            x > -2.0,
            lambda x: x + 1.0,
            lambda x: -2.0 * x - 3.0,
            x,
        )

    return lax.cond(x >= 0, positive_branch, negative_branch, x)

xs = jnp.array([-3.0, -1.0, 1.0, 3.0])
for x in xs:
    print(f"f({x}) = {piecewise_linear(float(x))}")
```

### Using the operand Keyword

```python
import jax
import jax.numpy as jnp
from jax import lax

@jax.jit
def with_operand_kw(pred, x):
    return lax.cond(
        pred,
        lambda x: x * 2,
        lambda x: x / 2,
        operand=x,
    )

print(with_operand_kw(True, 10.0))   # 20.0
print(with_operand_kw(False, 10.0))  # 5.0
```

---

## 3. lax.select

`jax.lax.select` performs element-wise selection between two arrays based on a boolean mask. Unlike `cond`, it evaluates both branches and then selects element-wise.

### Signature

```python
jax.lax.select(pred, on_true, on_false)
```

### Basic Usage

```python
import jax
import jax.numpy as jnp
from jax import lax

pred = jnp.array([True, False, True, False])
on_true = jnp.array([1.0, 2.0, 3.0, 4.0])
on_false = jnp.array([10.0, 20.0, 30.0, 40.0])

result = lax.select(pred, on_true, on_false)
print(result)  # [1.0, 20.0, 3.0, 40.0]
```

### Broadcasting

```python
import jax
import jax.numpy as jnp
from jax import lax

# Scalar condition -> broadcast
result = lax.select(True, jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]))
print(result)  # [1.0, 2.0]

# Broadcasting shapes
pred = jnp.array([[True, False], [False, True]])
on_true = jnp.array([1.0, 2.0])    # shape (2,) broadcasts to (2, 2)
on_false = jnp.array([10.0, 20.0])

result = lax.select(pred, on_true, on_false)
print(result)  # [[1.0, 20.0], [10.0, 2.0]]
```

### select_n: General N-way Select

```python
import jax
import jax.numpy as jnp
from jax import lax

# select_n uses an integer index to select from multiple arrays
idx = jnp.array([0, 1, 2, 0])
a = jnp.array([1.0, 1.0, 1.0, 1.0])
b = jnp.array([2.0, 2.0, 2.0, 2.0])
c = jnp.array([3.0, 3.0, 3.0, 3.0])

result = lax.select_n(idx, a, b, c)
print(result)  # [1.0, 2.0, 3.0, 1.0]
```

### lax.select vs jnp.where

`lax.select` and `jnp.where(condition, x, y)` are essentially the same operation. The difference is purely API style:

```python
import jax
import jax.numpy as jnp
from jax import lax

pred = jnp.array([True, False, True])
x = jnp.array([1.0, 2.0, 3.0])
y = jnp.array([4.0, 5.0, 6.0])

# These are equivalent:
r1 = lax.select(pred, x, y)
r2 = jnp.where(pred, x, y)
print(jnp.allclose(r1, r2))  # True
```

---

## 4. lax.switch

`jax.lax.switch` is a generalization of `lax.cond` that selects one of N branches based on an integer index. It is useful for dispatching to one of several implementations.

### Signature

```python
jax.lax.switch(index, branches, *operands)
```

### Basic Usage

```python
import jax
import jax.numpy as jnp
from jax import lax

def square(x):
    return x ** 2

def cube(x):
    return x ** 3

def negate(x):
    return -x

@jax.jit
def apply_op(op_idx, x):
    return lax.switch(op_idx, [square, cube, negate], x)

print(apply_op(0, 3.0))   # 9.0  (square)
print(apply_op(1, 3.0))   # 27.0 (cube)
print(apply_op(2, 3.0))   # -3.0 (negate)
```

### With Multiple Operands

```python
import jax
import jax.numpy as jnp
from jax import lax

def add(args):
    a, b = args
    return a + b

def multiply(args):
    a, b = args
    return a * b

def subtract(args):
    a, b = args
    return a - b

@jax.jit
def calculator(op_idx, a, b):
    return lax.switch(op_idx, [add, multiply, subtract], (a, b))

print(calculator(0, 3.0, 4.0))  # 7.0  (add)
print(calculator(1, 3.0, 4.0))  # 12.0 (multiply)
print(calculator(2, 3.0, 4.0))  # -1.0 (subtract)
```

### Activation Function Selector

```python
import jax
import jax.numpy as jnp
from jax import lax

activations = [
    lambda x: jnp.maximum(0, x),            # ReLU
    lambda x: x * (x > 0),                   # ReLU (alternative)
    lambda x: jnp.tanh(x),                   # Tanh
    lambda x: 1.0 / (1.0 + jnp.exp(-x)),     # Sigmoid
    lambda x: jnp.where(x > 0, x, x * 0.01), # Leaky ReLU
    lambda x: x * jnp.tanh(jnp.arcsinh(x)),  # Mish
]

@jax.jit
def apply_activation(idx, x):
    return lax.switch(idx, activations, x)

x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
for i, name in enumerate(["ReLU", "ReLU2", "Tanh", "Sigmoid", "LeakyReLU", "Mish"]):
    print(f"{name}: {apply_activation(i, x)}")
```

### switch with Data-Dependent Index

```python
import jax
import jax.numpy as jnp
from jax import lax

@jax.jit
def classify_and_transform(x):
    """Apply different transformations based on which range x falls in."""
    # Determine index based on value ranges
    idx = jnp.where(x < -1, 0,
           jnp.where(x < 0, 1,
           jnp.where(x < 1, 2, 3)))

    def transform_neg_large(x):
        return -jnp.exp(x)

    def transform_neg_small(x):
        return x ** 2 + 1

    def transform_pos_small(x):
        return jnp.sqrt(x + 2)

    def transform_pos_large(x):
        return jnp.log(x + 1)

    branches = [transform_neg_large, transform_neg_small,
                transform_pos_small, transform_pos_large]
    return lax.switch(idx, branches, x)

xs = jnp.array([-2.0, -0.5, 0.5, 2.0])
for x in xs:
    print(f"x={x:.1f}: {classify_and_transform(x):.4f}")
```

---

## 5. lax.while_loop

`jax.lax.while_loop` implements a data-dependent loop that continues while a condition function returns `True`. The loop body and condition must be JAX-traceable functions.

### Signature

```python
jax.lax.while_loop(cond_fun, body_fun, init_val)
```

### Basic Usage

```python
import jax
import jax.numpy as jnp
from jax import lax

@jax.jit
def countdown(n):
    """Count down from n to 0, accumulating the sum."""
    def cond(state):
        i, acc = state
        return i > 0

    def body(state):
        i, acc = state
        return (i - 1, acc + i)

    _, total = lax.while_loop(cond, body, (n, 0.0))
    return total

print(countdown(10))  # 55.0 = 1 + 2 + ... + 10
```

### Newton's Method for Square Root

```python
import jax
import jax.numpy as jnp
from jax import lax

@jax.jit
def newton_sqrt(x, tol=1e-8, max_iter=100):
    """Compute sqrt(x) using Newton's method."""
    def cond(state):
        guess, i = state
        diff = guess ** 2 - x
        return (jnp.abs(diff) > tol) & (i < max_iter)

    def body(state):
        guess, i = state
        new_guess = 0.5 * (guess + x / guess)
        return (new_guess, i + 1)

    init_guess = x / 2.0  # initial guess
    result, _ = lax.while_loop(cond, body, (init_guess, 0))
    return result

print(newton_sqrt(2.0))  # 1.4142135...
print(newton_sqrt(9.0))  # 3.0
print(newton_sqrt(100.0)) # 10.0
```

### Convergence with Tolerance

```python
import jax
import jax.numpy as jnp
from jax import lax

@jax.jit
def fixed_point_iteration(f, x0, tol=1e-6, max_iter=1000):
    """Find fixed point of f: x such that f(x) = x."""
    def cond(state):
        x, prev_x, i = state
        converged = jnp.max(jnp.abs(x - prev_x)) < tol
        return (~converged) & (i < max_iter)

    def body(state):
        x, prev_x, i = state
        return (f(x), x, i + 1)

    result, _, iterations = lax.while_loop(cond, body, (f(x0), x0, 0))
    return result, iterations

# Find fixed point of cos(x) = x
result, iters = fixed_point_iteration(jnp.cos, jnp.array(1.0))
print(f"Fixed point: {result:.8f}, iterations: {iters}")
# Should converge to ~0.73908513 (Dottie number)
```

### while_loop with Pytree State

```python
import jax
import jax.numpy as jnp
from jax import lax

@jax.jit
def gradient_descent(loss_fn, grad_fn, init_params, lr=0.01, max_iter=1000, tol=1e-6):
    """Simple gradient descent using while_loop."""
    def cond(state):
        params, grad_norm, i = state
        return (grad_norm > tol) & (i < max_iter)

    def body(state):
        params, grad_norm, i = state
        grads = grad_fn(params)
        new_params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
        new_grad_norm = jnp.sqrt(sum(
            jnp.sum(g ** 2) for g in jax.tree.leaves(grads)
        ))
        return (new_params, new_grad_norm, i + 1)

    init_grads = grad_fn(init_params)
    init_grad_norm = jnp.sqrt(sum(
        jnp.sum(g ** 2) for g in jax.tree.leaves(init_grads)
    ))

    final_params, _, iterations = lax.while_loop(
        cond, body, (init_params, init_grad_norm, 0)
    )
    return final_params, iterations

# Example usage
def quadratic(params):
    return jnp.sum(params ** 2)

grad_fn = jax.grad(quadratic)
init_params = jnp.array([5.0, 3.0, 1.0])
result, iters = gradient_descent(quadratic, grad_fn, init_params, lr=0.1)
print(f"Result: {result}, iterations: {iters}")
```

### Important Constraints

1. The condition function must return a scalar boolean.
2. The body function must return the same pytree structure as its input.
3. The loop bound must be deterministic for a given set of input shapes (JAX needs to know the maximum loop count for XLA compilation, but the actual number of iterations can be data-dependent).
4. `while_loop` is reverse-mode differentiable only if the loop bound is independent of the differentiated values (i.e., the number of iterations is the same for the primal and adjoint passes).

---

## 6. lax.fori_loop

`jax.lax.fori_loop` is a fixed-iteration-count loop. It is simpler than `while_loop` because the number of iterations is known at trace time.

### Signature

```python
jax.lax.fori_loop(lower, upper, body_fun, init_val)
```

### Basic Usage

```python
import jax
import jax.numpy as jnp
from jax import lax

@jax.jit
def sum_range(n):
    """Sum 0 + 1 + 2 + ... + (n-1)."""
    def body(i, acc):
        return acc + i.astype(acc.dtype)
    return lax.fori_loop(0, n, body, jnp.array(0.0))

print(sum_range(10))  # 45.0
```

### Iterative Refinement

```python
import jax
import jax.numpy as jnp
from jax import lax

@jax.jit
def iterative_average(x, n_iter=10):
    """Iteratively smooth a signal by averaging with neighbors."""
    def body(i, x):
        # Average each element with its neighbors
        left = jnp.roll(x, 1)
        right = jnp.roll(x, -1)
        return (x + left + right) / 3.0

    return lax.fori_loop(0, n_iter, body, x)

x = jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
print(f"Original:  {x}")
print(f"Smoothed:  {iterative_average(x, 5)}")
```

### Power Iteration for Eigenvalues

```python
import jax
import jax.numpy as jnp
from jax import lax

@jax.jit
def power_iteration(A, num_iters=50):
    """Compute the dominant eigenvector via power iteration."""
    key = jax.random.key(0)
    v = jax.random.normal(key, (A.shape[0],))

    def body(i, v):
        v_new = A @ v
        v_new = v_new / jnp.linalg.norm(v_new)
        return v_new

    v_final = lax.fori_loop(0, num_iters, body, v)
    eigenvalue = v_final @ A @ v_final
    return eigenvalue, v_final

A = jnp.array([[2.0, 1.0], [1.0, 3.0]])
eigenvalue, eigenvector = power_iteration(A)
print(f"Dominant eigenvalue: {eigenvalue:.4f}")  # Should be ~3.618
print(f"Eigenvector: {eigenvector}")
```

### fori_loop with Pytree Carry

```python
import jax
import jax.numpy as jnp
from jax import lax

@jax.jit
def train_step_loop(params, x, y, lr=0.01, n_steps=10):
    """Run multiple SGD steps using fori_loop."""

    def step(i, state):
        params, x, y, lr = state
        # Forward pass
        preds = x @ params["w"] + params["b"]
        loss = jnp.mean((preds - y) ** 2)
        # Backward pass
        grads = jax.grad(lambda p: jnp.mean((x @ p["w"] + p["b"] - y) ** 2))(params)
        # Update
        params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
        return (params, x, y, lr)

    final_state = lax.fori_loop(0, n_steps, step, (params, x, y, lr))
    return final_state[0]  # return only params

params = {
    "w": jax.random.normal(jax.random.key(0), (4, 2)) * 0.1,
    "b": jnp.zeros(2),
}
x = jax.random.normal(jax.random.key(1), (10, 4))
y = jax.random.normal(jax.random.key(2), (10, 2))

trained_params = train_step_loop(params, x, y)
print(f"Trained w norm: {jnp.linalg.norm(trained_params['w']):.4f}")
```

### Upper Bound Must Be Static or Traced

```python
import jax
import jax.numpy as jnp
from jax import lax

# Static upper bound: works fine
@jax.jit
def static_upper(x):
    return lax.fori_loop(0, 10, lambda i, x: x + 1, x)

# Dynamic upper bound: also works (upper bound is traced)
@jax.jit
def dynamic_upper(x, n):
    return lax.fori_loop(0, n, lambda i, x: x + 1, x)

print(static_upper(0.0))     # 10.0
print(dynamic_upper(0.0, 5)) # 5.0
```

---

## 7. lax.scan

`jax.lax.scan` is a loop that carries state forward and accumulates outputs at each step. It is the workhorse for implementing RNNs, cumulative operations, and sequential algorithms in JAX.

### Signature

```python
jax.lax.scan(f, init, xs, length=None, reverse=False, unroll=1)
```

### Basic Usage

```python
import jax
import jax.numpy as jnp
from jax import lax

def cumsum_scan(xs):
    """Compute cumulative sum using scan."""
    def body(carry, x):
        new_carry = carry + x
        return new_carry, new_carry

    final, cumulative = lax.scan(body, 0.0, xs)
    return final, cumulative

xs = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
final, cumulative = cumsum_scan(xs)
print(f"Final sum: {final}")           # 15.0
print(f"Cumulative: {cumulative}")     # [1, 3, 6, 10, 15]
```

### Simple RNN Cell

```python
import jax
import jax.numpy as jnp
from jax import lax

def rnn_scan(params, inputs, h0):
    """Simple RNN using scan: h_t = tanh(W_h h_{t-1} + W_x x_t + b)."""
    W_h, W_x, b = params

    def step(h, x):
        new_h = jnp.tanh(W_h @ h + W_x @ x + b)
        return new_h, new_h

    h_final, h_sequence = lax.scan(step, h0, inputs)
    return h_final, h_sequence

# Setup
key = jax.random.key(42)
hidden_size = 4
input_size = 3
seq_len = 10

params = (
    jax.random.normal(key, (hidden_size, hidden_size)) * 0.1,      # W_h
    jax.random.normal(jax.random.split(key)[0], (hidden_size, input_size)) * 0.1,  # W_x
    jnp.zeros(hidden_size),                                         # b
)
inputs = jax.random.normal(jax.random.split(key)[1], (seq_len, input_size))
h0 = jnp.zeros(hidden_size)

h_final, h_sequence = rnn_scan(params, inputs, h0)
print(f"Final hidden state: {h_final}")
print(f"Hidden state sequence shape: {h_sequence.shape}")  # (10, 4)
```

### scan with Multiple Carry Values

```python
import jax
import jax.numpy as jnp
from jax import lax

def fibonacci_scan(n):
    """Compute Fibonacci sequence using scan."""
    def step(carry, _):
        a, b = carry
        return (b, a + b), a

    init = (jnp.array(0), jnp.array(1))
    final, sequence = lax.scan(step, init, None, length=n)
    return sequence

fib = fibonacci_scan(10)
print(f"First 10 Fibonacci numbers: {fib}")
```

### Reverse Scan

```python
import jax
import jax.numpy as jnp
from jax import lax

def reverse_cumsum(xs):
    """Cumulative sum from right to left."""
    def body(carry, x):
        return carry + x, carry + x
    final, result = lax.scan(body, 0.0, xs, reverse=True)
    return result

xs = jnp.array([1.0, 2.0, 3.0, 4.0])
print(f"Forward cumsum: {lax.scan(lambda c, x: (c+x, c+x), 0.0, xs)[1]}")
print(f"Reverse cumsum: {reverse_cumsum(xs)}")
```

### scan for Running Statistics

```python
import jax
import jax.numpy as jnp
from jax import lax

def running_mean_std(xs):
    """Compute running mean and standard deviation using Welford's algorithm."""
    def step(state, x):
        count, mean, M2 = state
        count = count + 1
        delta = x - mean
        mean = mean + delta / count
        delta2 = x - mean
        M2 = M2 + delta * delta2
        variance = M2 / count
        return (count, mean, M2), (mean, variance)

    init = (jnp.array(0.0), jnp.array(0.0), jnp.array(0.0))
    final, (means, variances) = lax.scan(step, init, xs)
    return means, jnp.sqrt(variances)

xs = jax.random.normal(jax.random.key(0), (100,))
means, stds = running_mean_std(xs)
print(f"Final running mean: {means[-1]:.4f}")
print(f"Final running std: {stds[-1]:.4f}")
```

### scan with Pytree Carry

```python
import jax
import jax.numpy as jnp
from jax import lax

def optimizer_scan(params, grads_sequence, lr=0.01):
    """Apply a sequence of gradient updates using scan."""
    def step(params, grads):
        new_params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
        return new_params, new_params

    final_params, param_history = lax.scan(step, params, grads_sequence)
    return final_params, param_history

params = {"w": jnp.array([1.0, 2.0, 3.0]), "b": jnp.array([0.1])}
grads_seq = {
    "w": jnp.stack([jnp.array([0.1, 0.2, 0.3]) for _ in range(5)]),
    "b": jnp.stack([jnp.array([0.01]) for _ in range(5)]),
}

final, history = optimizer_scan(params, grads_seq)
print(f"Final params: {final}")
print(f"w history shape: {history['w'].shape}")  # (5, 3)
```

### Unrolling for Performance

```python
import jax
import jax.numpy as jnp
from jax import lax

# Unroll the scan loop to reduce overhead
@jax.jit
def fast_scan(x):
    def body(carry, x):
        return carry + x, carry + x
    # unroll=2 means the loop body is duplicated 2x in the compiled code
    return lax.scan(body, 0.0, x, unroll=2)

x = jnp.arange(100.0)
result = fast_scan(x)
print(result[1][-1])  # 4950.0
```

---

## 8. lax.associative_scan

`jax.lax.associative_scan` (also called `prefix_scan`) computes a cumulative (prefix) reduction using an associative binary operator. It is more parallel than `lax.scan` because it uses a tree-based algorithm with O(log n) parallel steps instead of O(n) sequential steps.

### Signature

```python
jax.lax.associative_scan(fn, elems, reverse=False, axis=0)
```

### Cumulative Sum

```python
import jax
import jax.numpy as jnp
from jax import lax

# Cumulative sum (prefix sum) using associative_scan
elems = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

# fn must be associative: fn(fn(a, b), c) == fn(a, fn(b, c))
result = lax.associative_scan(jnp.add, elems)
print(f"Prefix sum: {result}")  # [1, 3, 6, 10, 15]
```

### Cumulative Product

```python
import jax
import jax.numpy as jnp
from jax import lax

elems = jnp.array([1.0, 2.0, 3.0, 4.0])
result = lax.associative_scan(jnp.multiply, elems)
print(f"Prefix product: {result}")  # [1, 2, 6, 24]
```

### Cumulative Maximum

```python
import jax
import jax.numpy as jnp
from jax import lax

elems = jnp.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
result = lax.associative_scan(jnp.maximum, elems)
print(f"Running max: {result}")  # [3, 3, 4, 4, 5, 9, 9, 9]
```

### Custom Associative Operator (String Concatenation Analogy)

```python
import jax
import jax.numpy as jnp
from jax import lax

# Custom associative operator: element-wise min with index tracking
def associative_min(a, b):
    """Min that tracks which index produced the minimum."""
    val_a, idx_a = a
    val_b, idx_b = b
    min_val = jnp.minimum(val_a, val_b)
    # Choose the index from whichever had the smaller value
    min_idx = jnp.where(val_a <= val_b, idx_a, idx_b)
    return (min_val, min_idx)

values = jnp.array([5.0, 3.0, 7.0, 1.0, 4.0, 2.0, 6.0])
indices = jnp.arange(7)

# Track running minimum and the index where it first occurred
result_val, result_idx = lax.associative_scan(
    associative_min, (values, indices)
)
print(f"Running min values: {result_val}")  # [5, 3, 3, 1, 1, 1, 1]
print(f"Running min indices: {result_idx}") # [0, 1, 1, 3, 3, 3, 3]
```

### Reverse Associative Scan

```python
import jax
import jax.numpy as jnp
from jax import lax

elems = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

# Forward prefix sum
fwd = lax.associative_scan(jnp.add, elems, reverse=False)
print(f"Forward: {fwd}")  # [1, 3, 6, 10, 15]

# Reverse prefix sum (suffix sum)
rev = lax.associative_scan(jnp.add, elems, reverse=True)
print(f"Reverse: {rev}")  # [15, 14, 12, 9, 5]
```

### Multi-Dimensional along Specific Axis

```python
import jax
import jax.numpy as jnp
from jax import lax

x = jnp.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
])

# Cumulative sum along axis 0 (rows)
result_axis0 = lax.associative_scan(jnp.add, x, axis=0)
print(f"Along axis 0:\n{result_axis0}")
# [[1, 2, 3],
#  [5, 7, 9],
#  [12, 15, 18]]

# Cumulative sum along axis 1 (columns)
result_axis1 = lax.associative_scan(jnp.add, x, axis=1)
print(f"Along axis 1:\n{result_axis1}")
# [[1, 3, 6],
#  [4, 9, 15],
#  [7, 15, 24]]
```

### Parallel Prefix Sum vs Sequential scan

```python
import jax
import jax.numpy as jnp
from jax import lax
import time

n = 100000
x = jax.random.normal(jax.random.key(0), (n,))

# Sequential scan version of cumsum
@jax.jit
def sequential_cumsum(x):
    def body(carry, xi):
        return carry + xi, carry + xi
    return lax.scan(body, 0.0, x)[1]

# Parallel associative_scan version
@jax.jit
def parallel_cumsum(x):
    return lax.associative_scan(jnp.add, x)

# Both give the same result
r1 = sequential_cumsum(x)
r2 = parallel_cumsum(x)
print(f"Results match: {jnp.allclose(r1, r2)}")
print(f"Max diff: {jnp.max(jnp.abs(r1 - r2))}")
```

---

## 9. jnp.where

`jnp.where` is the element-wise conditional operation. It evaluates both branches but selects element-wise based on a condition. This is the most commonly used conditional in JAX.

### Signature

```python
jnp.where(condition, x, y)      # element-wise selection
jnp.where(condition)            # returns indices where condition is True
```

### Element-Wise Selection

```python
import jax
import jax.numpy as jnp

x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])

# ReLU via jnp.where
relu = jnp.where(x > 0, x, 0.0)
print(f"ReLU: {relu}")  # [0, 0, 0, 1, 2]

# Absolute value
abs_x = jnp.where(x >= 0, x, -x)
print(f"|x|: {abs_x}")  # [2, 1, 0, 1, 2]
```

### Safe Operations

```python
import jax
import jax.numpy as jnp

def safe_sqrt(x):
    """Square root that returns 0 for negative values."""
    return jnp.where(x >= 0, jnp.sqrt(jnp.abs(x)), 0.0)

def safe_divide(a, b):
    """Division that returns 0 where b is zero."""
    return jnp.where(b != 0, a / b, 0.0)

print(safe_sqrt(jnp.array([-1.0, 0.0, 4.0])))     # [0, 0, 2]
print(safe_divide(jnp.array([1.0, 2.0, 3.0]), jnp.array([2.0, 0.0, 1.0])))  # [0.5, 0, 3]
```

### Nested where

```python
import jax
import jax.numpy as jnp

x = jnp.array([-2.0, -0.5, 0.0, 0.5, 2.0])

# Piecewise: -1 if x < -1, x if -1 <= x <= 1, 1 if x > 1
result = jnp.where(x < -1.0, -1.0,
          jnp.where(x > 1.0,  1.0, x))
print(f"Clamped: {result}")  # [-1, -0.5, 0, 0.5, 1]
```

### where with Broadcasting

```python
import jax
import jax.numpy as jnp

# Condition shape (3, 1), x shape (3,), y shape (3,)
cond = jnp.array([[True], [False], [True]])  # (3, 1)
x = jnp.array([1.0, 2.0, 3.0])               # (3,)
y = jnp.array([10.0, 20.0, 30.0])            # (3,)

result = jnp.where(cond, x, y)
print(f"Broadcast where:\n{result}")
# [[ 1,  2,  3],
#  [10, 20, 30],
#  [ 1,  2,  3]]
```

### Finding Indices (Non-Zero Mode)

```python
import jax
import jax.numpy as jnp

x = jnp.array([0, 1, 0, 2, 0, 3, 0])

# Get indices where x is non-zero
indices = jnp.where(x > 0)
print(f"Non-zero indices: {indices}")  # (array([1, 3, 5]),)
```

### Masking Operations

```python
import jax
import jax.numpy as jnp

x = jnp.array([[1.0, 2.0, 3.0],
               [4.0, 5.0, 6.0],
               [7.0, 8.0, 9.0]])

# Create a lower triangular mask
mask = jnp.tril(jnp.ones((3, 3), dtype=bool))
print(f"Mask:\n{mask}")

# Apply mask: keep lower triangle, zero out upper
lower = jnp.where(mask, x, 0.0)
print(f"Lower triangle:\n{lower}")

# Replace NaN values
x_with_nan = jnp.array([1.0, jnp.nan, 3.0, jnp.nan, 5.0])
clean = jnp.where(jnp.isnan(x_with_nan), 0.0, x_with_nan)
print(f"Cleaned: {clean}")  # [1, 0, 3, 0, 5]
```

---

## 10. jnp.piecewise

`jnp.piecewise` evaluates a piecewise-defined function by applying different functions to different regions of the input defined by boolean conditions.

### Signature

```python
jnp.piecewise(x, condlist, funclist, *args, **kw)
```

### Basic Usage

```python
import jax
import jax.numpy as jnp

x = jnp.array([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0])

# Piecewise function:
#   -1      if x < -1
#    x      if -1 <= x < 0
#    x^2    if 0 <= x < 1
#    1      if x >= 1
condlist = [
    x < -1,
    (x >= -1) & (x < 0),
    (x >= 0) & (x < 1),
    x >= 1,
]
funclist = [
    lambda x: -1.0,
    lambda x: x,
    lambda x: x ** 2,
    lambda x: 1.0,
]

result = jnp.piecewise(x, condlist, funclist)
print(f"Piecewise result: {result}")
# [-1, -1, 0, 0.25, 1, 1]
```

### Absolute Value via piecewise

```python
import jax
import jax.numpy as jnp

x = jnp.array([-3.0, -1.0, 0.0, 1.0, 3.0])

result = jnp.piecewise(
    x,
    [x < 0, x >= 0],
    [lambda x: -x, lambda x: x]
)
print(f"|x|: {result}")  # [3, 1, 0, 1, 3]
```

### Piecewise with Additional Arguments

```python
import jax
import jax.numpy as jnp

def scaled_relu(x, scale):
    return scale * jnp.maximum(0, x)

def scaled_leaky(x, scale):
    return scale * jnp.minimum(0, x)

x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])

result = jnp.piecewise(
    x,
    [x < 0, x >= 0],
    [lambda x: scaled_leaky(x, 0.1),
     lambda x: scaled_relu(x, 1.0)]
)
print(f"Leaky ReLU: {result}")  # [-0.2, -0.1, 0, 1, 2]
```

### Default Function (Extra Element in funclist)

```python
import jax
import jax.numpy as jnp

x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])

# If funclist has one more element than condlist, the last function is the default
result = jnp.piecewise(
    x,
    [x < 0],
    [lambda x: -x,    # applied where x < 0
     lambda x: x],    # default: applied everywhere else
)
print(f"Result: {result}")  # [2, 1, 0, 1, 2]
```

---

## 11. jnp.select

`jnp.select` is a vectorized multi-way conditional. Given a list of conditions and corresponding choices, it selects the first matching condition for each element. It supports a default value for elements where no condition is true.

### Signature

```python
jnp.select(condlist, choicelist, default=0)
```

### Basic Usage

```python
import jax
import jax.numpy as jnp

x = jnp.array([-3.0, -1.0, 0.0, 1.0, 3.0])

conditions = [
    x < -2,
    x < 0,
    x == 0,
    x > 0,
]
choices = [
    jnp.full_like(x, -2.0),
    jnp.full_like(x, -1.0),
    jnp.full_like(x, 0.0),
    jnp.full_like(x, 1.0),
]

result = jnp.select(conditions, choices, default=0.0)
print(f"Selected: {result}")  # [-2, -1, 0, 1, 1]
```

### Discretization / Binning

```python
import jax
import jax.numpy as jnp

x = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])

# Discretize into bins
conditions = [
    x < 0.25,
    x < 0.5,
    x < 0.75,
    x >= 0.75,
]
choices = [
    jnp.zeros_like(x),
    jnp.ones_like(x),
    jnp.full_like(x, 2.0),
    jnp.full_like(x, 3.0),
]

result = jnp.select(conditions, choices, default=-1.0)
print(f"Discretized: {result}")  # [0, 1, 1, 2, 3]
```

### One-Hot Encoding via select

```python
import jax
import jax.numpy as jnp

indices = jnp.array([0, 2, 1, 3, 0])
num_classes = 4

# Create one-hot encoding using select
conditions = [indices == i for i in range(num_classes)]
choices = [jnp.ones_like(indices) * i for i in range(num_classes)]

result = jnp.select(conditions, choices, default=0)
print(f"Values: {result}")  # [0, 2, 1, 3, 0]

# Better approach: use one_hot
one_hot = jax.nn.one_hot(indices, num_classes)
print(f"One-hot:\n{one_hot}")
```

### Priority-Based Selection

```python
import jax
import jax.numpy as jnp

scores = jnp.array([0.9, 0.7, 0.5, 0.3, 0.1])

# Assign grades based on score thresholds (first match wins)
conditions = [
    scores >= 0.9,
    scores >= 0.8,
    scores >= 0.7,
    scores >= 0.6,
    scores >= 0.5,
]
choices = [
    jnp.full_like(scores, 4.0),  # A
    jnp.full_like(scores, 3.5),  # B+
    jnp.full_like(scores, 3.0),  # B
    jnp.full_like(scores, 2.5),  # C+
    jnp.full_like(scores, 2.0),  # C
]

grades = jnp.select(conditions, choices, default=0.0)
print(f"Grades: {grades}")  # [4, 3, 2, 0, 0]
```

---

## 12. Logical Operators

JAX provides element-wise logical operations that work on boolean arrays and are compatible with JIT and autodiff.

### Basic Logical Operations

```python
import jax
import jax.numpy as jnp

a = jnp.array([True, True, False, False])
b = jnp.array([True, False, True, False])

# Logical AND
print(f"AND: {jnp.logical_and(a, b)}")       # [T, F, F, F]

# Logical OR
print(f"OR:  {jnp.logical_or(a, b)}")        # [T, T, T, F]

# Logical NOT
print(f"NOT: {jnp.logical_not(a)}")           # [F, F, T, T]

# Logical XOR
print(f"XOR: {jnp.logical_xor(a, b)}")       # [F, T, T, F]
```

### Bitwise Operations on Boolean Arrays

```python
import jax
import jax.numpy as jnp

a = jnp.array([True, True, False, False])
b = jnp.array([True, False, True, False])

# For boolean arrays, bitwise and logical are equivalent
print(f"bitwise_and: {jnp.bitwise_and(a, b)}")  # [T, F, F, F]
print(f"bitwise_or:  {jnp.bitwise_or(a, b)}")   # [T, T, T, F]
print(f"bitwise_xor: {jnp.bitwise_xor(a, b)}")  # [F, T, T, F]

# Invert (same as logical_not for booleans)
print(f"invert: {jnp.invert(a)}")                 # [F, F, T, T]
```

### Comparison Operators (Produce Booleans)

```python
import jax
import jax.numpy as jnp

x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = jnp.array([2.0, 2.0, 2.0, 2.0, 2.0])

print(f"x == y: {x == y}")     # [F, T, F, F, F]
print(f"x != y: {x != y}")     # [T, F, T, T, T]
print(f"x > y:  {x > y}")      # [F, F, T, T, T]
print(f"x < y:  {x < y}")      # [T, F, F, F, F]
print(f"x >= y: {x >= y}")     # [F, T, T, T, T]
print(f"x <= y: {x <= y}")     # [T, T, F, F, F]
```

### Combining Logical Operations with Control Flow

```python
import jax
import jax.numpy as jnp

@jax.jit
def classify(x):
    """Classify values into categories using logical operations."""
    is_negative = x < 0
    is_zero = x == 0
    is_small = jnp.logical_and(x > 0, x < 1)
    is_medium = jnp.logical_and(x >= 1, x < 10)
    is_large = x >= 10

    # Use logical OR to combine conditions
    is_non_standard = jnp.logical_or(is_negative, jnp.logical_or(is_zero, is_large))

    result = jnp.where(is_negative, -1.0,
              jnp.where(is_zero,     0.0,
              jnp.where(is_small,    0.5,
              jnp.where(is_medium,   1.0, 10.0))))

    return result, is_non_standard

x = jnp.array([-1.0, 0.0, 0.5, 5.0, 15.0])
values, flags = classify(x)
print(f"Values: {values}")       # [-1, 0, 0.5, 1, 10]
print(f"Non-standard: {flags}")  # [T, T, F, F, T]
```

### all and any for Boolean Reduction

```python
import jax
import jax.numpy as jnp

x = jnp.array([True, True, False, True])
y = jnp.array([True, True, True, True])

print(f"all(x): {jnp.all(x)}")   # False
print(f"all(y): {jnp.all(y)}")   # True
print(f"any(x): {jnp.any(x)}")   # True
print(f"any(False): {jnp.any(jnp.array([False, False]))}")  # False

# Along an axis
matrix = jnp.array([[True, False], [True, True]])
print(f"all(axis=0): {jnp.all(matrix, axis=0)}")   # [True, False]
print(f"all(axis=1): {jnp.all(matrix, axis=1)}")   # [False, True]
print(f"any(axis=0): {jnp.any(matrix, axis=0)}")   # [True, True]
```

### lax.eq, lax.ne, lax.gt, lax.ge, lax.lt, lax.le

```python
import jax
import jax.numpy as jnp
from jax import lax

x = jnp.array([1, 2, 3, 4])
y = jnp.array([2, 2, 2, 2])

print(f"eq: {lax.eq(x, y)}")    # [F, T, F, F]
print(f"ne: {lax.ne(x, y)}")    # [T, F, T, T]
print(f"gt: {lax.gt(x, y)}")    # [F, F, T, T]
print(f"ge: {lax.ge(x, y)}")    # [F, T, T, T]
print(f"lt: {lax.lt(x, y)}")    # [T, F, F, F]
print(f"le: {lax.le(x, y)}")    # [T, T, F, F]
```

---

## 13. Control Flow with JIT

### What Works Inside JIT

```python
import jax
import jax.numpy as jnp
from jax import lax

@jax.jit
def works_fine(x):
    # Static condition (Python bool): works
    result = x + 1
    result = result * 2
    return result

# lax control flow: always works
@jax.jit
def lax_control(x, flag):
    return lax.cond(flag, lambda x: x ** 2, lambda x: -x, x)

# jnp.where: always works
@jax.jit
def where_control(x):
    return jnp.where(x > 0, x, 0.0)  # ReLU

# Python for with static range: works
@jax.jit
def python_for_static(x):
    for i in range(5):  # range is known at trace time
        x = x + 1
    return x
```

### What Does NOT Work Inside JIT

```python
import jax
import jax.numpy as jnp

# ERROR: Python if on traced value
# @jax.jit
# def bad_if(x):
#     if x > 0:        # ConcretizationTypeError
#         return x
#     return -x

# ERROR: Python while with traced condition
# @jax.jit
# def bad_while(x):
#     while x > 0:     # ConcretizationTypeError
#         x = x - 1
#     return x

# ERROR: Python for with traced range
# @jax.jit
# def bad_for(x, n):
#     for i in range(n):  # n is traced, not static
#         x = x + 1
#     return x

# ERROR: print of traced value
# @jax.jit
# def bad_print(x):
#     print(x)           # ConcretizationTypeError
#     return x
```

### Using static_argnums for Compile-Time Constants

```python
import jax
import jax.numpy as jnp

# Use static_argnums to make Python-level control flow work
@jax.jit(static_argnums=(1,))
def branch_with_static(x, use_square):
    if use_square:  # This is now a compile-time constant
        return x ** 2
    else:
        return x * 2

print(branch_with_static(3.0, True))   # 9.0
print(branch_with_static(3.0, False))  # 6.0
# Note: changing use_square triggers recompilation
```

### Using Block Ready Callbacks for Debugging

```python
import jax
import jax.numpy as jnp

@jax.jit
def debug_control_flow(x):
    # Use jax.debug.print to inspect traced values
    jax.debug.print("x = {x}", x=x)

    result = jnp.where(x > 0, x ** 2, -x)
    jax.debug.print("result = {r}", r=result)
    return result

debug_control_flow(3.0)
debug_control_flow(-2.0)
```

---

## 14. Control Flow with Autodiff

### Differentiability of Control Flow Primitives

Different JAX control flow primitives have different differentiability properties:

#### cond is Differentiable (Both Modes)

```python
import jax
import jax.numpy as jnp
from jax import lax

def f(x, flag):
    return lax.cond(flag, lambda x: x ** 3, lambda x: jnp.sin(x), x)

# Forward-mode works
primal, tangent = jax.jvp(f, (2.0, True), (1.0, 0.0))
print(f"Primal (x^3): {primal}, Tangent: {tangent}")  # 8.0, 12.0

primal, tangent = jax.jvp(f, (2.0, False), (1.0, 0.0))
print(f"Primal (sin): {primal}, Tangent: {tangent}")  # sin(2), cos(2)

# Reverse-mode works
grad_fn = jax.grad(f)
print(f"Grad (x^3 at 2): {grad_fn(2.0, True)}")    # 12.0
print(f"Grad (sin at 2): {grad_fn(2.0, False)}")   # cos(2)
```

#### select/where is Differentiable

```python
import jax
import jax.numpy as jnp

def relu(x):
    return jnp.where(x > 0, x, 0.0)

# Gradient of ReLU: 1 where x > 0, 0 where x < 0
grad_relu = jax.grad(relu)
print(f"grad ReLU at 2.0: {grad_relu(2.0)}")   # 1.0
print(f"grad ReLU at -1.0: {grad_relu(-1.0)}")  # 0.0
# At exactly 0: convention varies (subgradient)
```

#### scan is Differentiable

```python
import jax
import jax.numpy as jnp
from jax import lax

def scan_sum_square(xs):
    def body(carry, x):
        return carry + x ** 2, carry + x ** 2
    final, _ = lax.scan(body, 0.0, xs)
    return final

xs = jnp.array([1.0, 2.0, 3.0, 4.0])

# Forward-mode
primal, tangent = jax.jvp(scan_sum_square, (xs,), (jnp.ones(4),))
print(f"Primal: {primal}, Tangent: {tangent}")

# Reverse-mode
grad_fn = jax.grad(scan_sum_square)
grads = grad_fn(xs)
print(f"Gradients: {grads}")  # [2, 4, 6, 8] (derivative of sum of x^2)
```

#### fori_loop is Differentiable

```python
import jax
import jax.numpy as jnp
from jax import lax

def repeated_sin(x, n=5):
    """Apply sin n times."""
    def body(i, x):
        return jnp.sin(x)
    return lax.fori_loop(0, n, body, x)

x = 1.0
grad_fn = jax.grad(repeated_sin)
print(f"f(x) = {repeated_sin(x)}")
print(f"f'(x) = {grad_fn(x)}")
```

#### while_loop: Differentiable Under Constraints

`while_loop` is reverse-mode differentiable only when the number of iterations does NOT depend on the differentiated values. Forward-mode always works.

```python
import jax
import jax.numpy as jnp
from jax import lax

# Differentiable: iteration count does not depend on x
def fixed_iter_computation(x, n_iter=10):
    def cond(state):
        x, i = state
        return i < n_iter  # depends on i, not x
    def body(state):
        x, i = state
        return (x * 0.9 + 0.1, i + 1)
    result, _ = lax.while_loop(cond, body, (x, 0))
    return result

grad_fn = jax.grad(fixed_iter_computation)
print(f"Grad: {grad_fn(1.0)}")  # 0.9^10 ~ 0.3487
```

```python
import jax
import jax.numpy as jnp
from jax import lax

# NOT reverse-mode differentiable: iteration count depends on x
def convergence_loop(x):
    """NOT differentiable: loop count depends on x."""
    def cond(state):
        x, _ = state
        return jnp.abs(x) > 0.01  # condition depends on x!
    def body(state):
        x, i = state
        return (x * 0.5, i + 1)
    result, iters = lax.while_loop(cond, body, (x, 0))
    return result

# Forward-mode works
primal, tangent = jax.jvp(convergence_loop, (1.0,), (1.0,))
print(f"Forward-mode: primal={primal}, tangent={tangent}")

# Reverse-mode: will raise error or produce incorrect results
# grad_fn = jax.grad(convergence_loop)  # May error!
```

### Gradient Through scan for BPTT

```python
import jax
import jax.numpy as jnp
from jax import lax

def rnn_loss(params, inputs, targets, h0):
    """RNN loss using scan, differentiable for BPTT."""
    W_h, W_x, b, W_out = params

    def step(h, inputs_target):
        x, target = inputs_target
        h_new = jnp.tanh(W_h @ h + W_x @ x + b)
        output = W_out @ h_new
        loss = jnp.sum((output - target) ** 2)
        return h_new, loss

    h_final, losses = lax.scan(step, h0, (inputs, targets))
    return jnp.sum(losses)

# Setup
key = jax.random.key(0)
hidden_size = 4
input_size = 3
output_size = 2
seq_len = 8

params = (
    jax.random.normal(key, (hidden_size, hidden_size)) * 0.1,
    jax.random.normal(jax.random.split(key)[0], (hidden_size, input_size)) * 0.1,
    jnp.zeros(hidden_size),
    jax.random.normal(jax.random.split(key)[1], (output_size, hidden_size)) * 0.1,
)
inputs = jax.random.normal(jax.random.split(key)[2], (seq_len, input_size))
targets = jax.random.normal(jax.random.split(key)[3], (seq_len, output_size))
h0 = jnp.zeros(hidden_size)

# Compute gradients through time (BPTT)
loss, grads = jax.value_and_grad(rnn_loss)(params, inputs, targets, h0)
print(f"Loss: {loss:.4f}")
for i, g in enumerate(grads):
    print(f"Grad {i} shape: {g.shape}, norm: {jnp.linalg.norm(g):.4f}")
```

---

## 15. Summary Table of Differentiability

| Primitive | Forward-Mode (jvp) | Reverse-Mode (vjp/grad) | Notes |
|-----------|-------------------|------------------------|-------|
| `lax.cond` | Yes | Yes | Both branches must be differentiable |
| `lax.select` | Yes | Yes | Evaluates both sides (not lazy) |
| `lax.switch` | Yes | Yes | All branches must be differentiable |
| `jnp.where` | Yes | Yes | Same as `lax.select`; subgradient at boundary |
| `jnp.piecewise` | Yes | Yes | All pieces must be differentiable |
| `jnp.select` | Yes | Yes | All choices must be differentiable |
| `lax.scan` | Yes | Yes | Differentiable through carry and outputs |
| `lax.fori_loop` | Yes | Yes | Fixed iteration count |
| `lax.associative_scan` | Yes | Yes | Operator must be associative |
| `lax.while_loop` | Yes | **Conditional** | Reverse-mode only when iteration count does not depend on differentiated values |
| `logical_and/or/not` | N/A | N/A | Not differentiable (discrete) |
| Python `if` (static) | N/A | N/A | Resolved at trace time, no autodiff impact |
| Python `if` (traced) | Error | Error | Use `lax.cond` or `jnp.where` |

### Key Takeaways

1. **Use `jnp.where`** for element-wise conditionals -- simplest and most common.
2. **Use `lax.cond`** when you need lazy evaluation (avoid computing both branches).
3. **Use `lax.switch`** for multi-way dispatch on a single index.
4. **Use `lax.scan`** for sequential loops with carry state (RNNs, cumulative operations).
5. **Use `lax.while_loop`** for data-dependent loop termination, but be cautious with reverse-mode differentiation.
6. **Use `lax.fori_loop`** when the iteration count is known (cleaner than `while_loop`).
7. **Use `lax.associative_scan`** for parallel prefix operations (cumsum, cumprod, running max).
8. **Avoid Python control flow** (`if`, `while`) on traced values inside JIT.
9. **Logical operators** produce boolean outputs and are not differentiable.
10. **All `jnp.where`/`select` branches are evaluated** even if the condition selects one side -- use `lax.cond` for truly lazy evaluation.
