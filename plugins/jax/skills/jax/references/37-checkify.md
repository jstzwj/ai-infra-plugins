# Chapter 37: Checkify -- Functional Error Checking in JAX

## 37.1 Introduction

JAX programs are compiled (via XLA) before execution, which means standard
Python error handling (`try/except`, `assert`) does not work inside JIT-compiled
functions. Checkify provides a **functional error checking** mechanism that
works within JAX's compilation model.

Instead of raising exceptions, Checkify collects errors into an **Error object**
that is returned as a value from the compiled function. This allows error checking
inside `jax.jit`, `jax.vmap`, `jax.grad`, and other transformations.

---

## 37.2 Basic Usage

### 37.2.1 checkify.check

The `checkify.check` function is the primary assertion primitive:

```python
import jax
import jax.numpy as jnp
from jax.experimental import checkify

@checkify.checkify
def safe_divide(x, y):
    checkify.check(y != 0, "Division by zero!")
    return x / y

# Get the error and result
err, result = safe_divide(10.0, 2.0)
err.throw()  # No error -- does nothing
print(result)  # 5.0

# With an error
err, result = safe_divide(10.0, 0.0)
print(err)  # Error: Division by zero!
# err.throw()  # Would raise a ValueError
```

### 37.2.2 The Error Object

The `Error` object returned by `checkify.checkify` contains:

```python
err, result = safe_divide(10.0, 2.0)

# Check if there is an error
print(bool(err))  # False (no error)

# Get error message
print(err.get_message())  # "" (no message when no error)

# Throw the error (raises Python exception if there is one)
err.throw()  # No-op when no error

# String representation
print(str(err))  # "no error" or the error message
```

### 37.2.3 checkify.check Syntax

```python
from jax.experimental import checkify

# Basic check with message
checkify.check(condition, "Error message string")

# Check with formatted message
checkify.check(x > 0, f"Value must be positive, got shape {x.shape}")

# Multiple checks in one function
def validated_fn(x, y):
    checkify.check(x.shape == y.shape, "Shapes must match")
    checkify.check(jnp.all(jnp.isfinite(x)), "Input must be finite")
    checkify.check(jnp.all(y >= 0), "y must be non-negative")
    return x + y
```

---

## 37.3 Error Categories

Checkify organizes checks into categories. When you enable checkify, you
choose which categories of errors to check.

### 37.3.1 User Checks

The `user` category includes explicit `checkify.check` calls:

```python
from jax.experimental import checkify

# Enable only user-defined checks
@checkify.checkify(checkify.user_checks)
def my_function(x):
    checkify.check(x > 0, "x must be positive")
    return jnp.log(x)

err, result = my_function(5.0)
err.throw()  # OK

err, result = my_function(-1.0)
err.throw()  # Raises: x must be positive
```

### 37.3.2 NaN Checks

The `nan_checks` category automatically detects NaN values in computation results:

```python
from jax.experimental import checkify

@checkify.checkify(checkify.nan_checks)
def might_produce_nan(x):
    return jnp.sqrt(x)  # NaN for negative inputs

err, result = might_produce_nan(4.0)
err.throw()  # OK, result = 2.0

err, result = might_produce_nan(-1.0)
print(bool(err))  # True -- NaN detected
err.throw()  # Raises: NaN value detected
```

### 37.3.3 Division by Zero Checks

The `div_checks` category detects division by zero:

```python
from jax.experimental import checkify

@checkify.checkify(checkify.div_checks)
def safe_div(x, y):
    return x / y

err, result = safe_div(10.0, 2.0)
err.throw()  # OK

err, result = safe_div(10.0, 0.0)
print(bool(err))  # True -- division by zero
err.throw()  # Raises error about division by zero
```

### 37.3.4 Index Out of Bounds Checks

The `index_checks` category detects out-of-bounds array indexing:

```python
from jax.experimental import checkify

@checkify.checkify(checkify.index_checks)
def safe_index(arr, idx):
    return arr[idx]

arr = jnp.array([1.0, 2.0, 3.0])

err, result = safe_index(arr, 2)
err.throw()  # OK, result = 3.0

err, result = safe_index(arr, 5)
print(bool(err))  # True -- index out of bounds
err.throw()  # Raises error about out-of-bounds access
```

### 37.3.5 Float Checks

The `float_checks` category combines NaN checks and division checks:

```python
from jax.experimental import checkify

@checkify.checkify(checkify.float_checks)
def float_sensitive_fn(x, y):
    return x / y + jnp.sqrt(x)

# float_checks = nan_checks | div_checks
# Detects both NaN values and division by zero
```

### 37.3.6 All Checks

The `all_checks` category enables every type of check:

```python
from jax.experimental import checkify

# Enable all categories
@checkify.checkify(checkify.all_checks)
def fully_checked_fn(x, y, arr, idx):
    checkify.check(x > 0, "x must be positive")  # User check
    result = x / y                                  # Div check
    result = result + jnp.sqrt(result)              # NaN check
    return arr[idx]                                  # Index check

arr = jnp.array([1.0, 2.0, 3.0])
err, result = fully_checked_fn(1.0, 1.0, arr, 1)
err.throw()  # OK
```

### 37.3.7 Combining Categories

You can combine specific categories using the `|` operator:

```python
from jax.experimental import checkify

# Combine user checks with NaN checks
my_checks = checkify.user_checks | checkify.nan_checks

@checkify.checkify(my_checks)
def my_fn(x):
    checkify.check(jnp.all(x >= 0), "x must be non-negative")
    return jnp.log(x)

# Combine float and index checks
safe_checks = checkify.float_checks | checkify.index_checks

@checkify.checkify(safe_checks)
def safe_fn(x, y, arr, i):
    result = x / y
    return result * arr[i]
```

### 37.3.8 Category Summary

| Category | Constant | Detects |
|---|---|---|
| User | `checkify.user_checks` | Explicit `checkify.check` assertions |
| NaN | `checkify.nan_checks` | NaN values in computation results |
| Division | `checkify.div_checks` | Division by zero |
| Index | `checkify.index_checks` | Out-of-bounds array indexing |
| Float | `checkify.float_checks` | NaN + Division (combined) |
| All | `checkify.all_checks` | All of the above |

---

## 37.4 Usage with JIT Compilation

### 37.4.1 checkify Inside jit

Checkify must be applied *before* JIT compilation. The `checkify` decorator
wraps the function so that error checks become part of the compiled computation:

```python
from jax.experimental import checkify

# CORRECT: checkify first, then jit
@jax.jit
@checkify.checkify(checkify.all_checks)
def checked_fn(x, y):
    checkify.check(x > 0, "x must be positive")
    return x / y

# The function now returns (error, result)
err, result = checked_fn(10.0, 2.0)
err.throw()
print(result)  # 5.0
```

### 37.4.2 Functional Handling Pattern

Since checkify returns the error as a value, you can handle errors functionally
without exceptions:

```python
@jax.jit
@checkify.checkify(checkify.float_checks)
def safe_compute(x, y):
    return x / y

err, result = safe_compute(10.0, 0.0)

# Functional error handling
if err:
    print(f"Computation failed: {err}")
    # Use a fallback value
    result = jnp.inf
else:
    print(f"Result: {result}")
```

### 37.4.3 Disabling Checks in Production

For production, you can remove checkify overhead by simply not applying the
decorator, or by using a flag:

```python
ENABLE_CHECKS = False  # Set to True during development

def make_safe_divide():
    def safe_divide(x, y):
        if ENABLE_CHECKS:
            from jax.experimental import checkify

            @jax.jit
            @checkify.checkify(checkify.float_checks)
            def _checked(x, y):
                checkify.check(y != 0, "Division by zero")
                return x / y

            return _checked(x, y)
        else:
            return jax.jit(lambda x, y: x / y)(x, y)

    return safe_divide
```

---

## 37.5 Usage with vmap

### 37.5.1 checkify and vmap

Checkify works with `jax.vmap` -- each batch element is checked independently:

```python
from jax.experimental import checkify

@checkify.checkify(checkify.float_checks)
def checked_sqrt(x):
    return jnp.sqrt(x)

# vmap over the checked function
batched_sqrt = jax.vmap(checked_sqrt)

x = jnp.array([4.0, 9.0, -1.0, 16.0])
err, results = batched_sqrt(x)

# err will contain all errors from any batch element
print(bool(err))  # True (because of -1.0 producing NaN)
print(results)    # [2.0, 3.0, nan, 4.0]
```

### 37.5.2 Per-Element Error Handling

```python
@checkify.checkify(checkify.user_checks)
def safe_element_wise(x, threshold):
    checkify.check(x > threshold, "Value below threshold")
    return x * 2

batched = jax.vmap(safe_element_wise, in_axes=(0, None))
x = jnp.array([1.0, 2.0, 3.0, 0.5])
threshold = 0.0

err, results = batched(x, threshold)
# Since all x > 0, no error
err.throw()  # OK
print(results)  # [2.0, 4.0, 6.0, 1.0]
```

---

## 37.6 Usage with grad

### 37.6.1 checkify and grad

Checkify can be composed with `jax.grad`. Place checkify *outside* grad:

```python
from jax.experimental import checkify

def f(x):
    checkify.check(x > 0, "x must be positive for log")
    return jnp.log(x)

# Checkify wraps the gradient function
checked_grad = checkify.checkify(checkify.user_checks)(jax.grad(f))

x = 2.0
err, grad_val = checked_grad(x)
err.throw()  # OK
print(grad_val)  # 0.5 (= 1/x = 1/2.0)

x = -1.0
err, grad_val = checked_grad(x)
err.throw()  # Raises: x must be positive for log
```

### 37.6.2 Checking Gradient Values

```python
@checkify.checkify(checkify.nan_checks)
def checked_grad_fn(params, x):
    def loss(params, x):
        pred = jnp.dot(x, params)
        return jnp.mean(pred ** 2)

    return jax.grad(loss)(params, x)

params = jnp.array([1.0, 2.0, 3.0])
x = jnp.array([1.0, 1.0, 1.0])

err, grad_val = checked_grad_fn(params, x)
err.throw()  # OK
print(grad_val)  # [2.0, 4.0, 6.0]
```

### 37.6.3 Checking Training Step

```python
def checked_training_step(params, batch, lr=0.01):
    """Training step with checkify for catching NaN gradients."""
    def loss_fn(params, x, y):
        pred = jnp.dot(x, params)
        return jnp.mean((pred - y) ** 2)

    grads = jax.grad(loss_fn)(params, batch['x'], batch['y'])

    # Check for NaN gradients
    checkify.check(
        jnp.all(jnp.isfinite(jax.tree.leaves(grads)[0])),
        "NaN gradients detected!"
    )

    # Update
    new_params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
    return new_params

# Wrap with checkify and jit
safe_step = jax.jit(checkify.checkify(checkify.user_checks)(checked_training_step))

params = jnp.ones((5,))
batch = {'x': jnp.ones((4, 5)), 'y': jnp.ones((4,))}

err, new_params = safe_step(params, batch)
err.throw()  # OK if no NaN
print(new_params)
```

---

## 37.7 Advanced Patterns

### 37.7.1 Conditional Checks

```python
@checkify.checkify(checkify.user_checks)
def conditional_check(x, mode):
    # Check only applies in certain modes
    if mode == "strict":
        checkify.check(x > 0, "Strict mode: x must be positive")
    return jnp.abs(x)

# Note: The `mode` parameter must be a static argument for JIT
checked_fn = jax.jit(
    checkify.checkify(checkify.user_checks)(conditional_check),
    static_argnames=["mode"],
)

err, result = checked_fn(-1.0, mode="strict")
err.throw()  # Raises: Strict mode: x must be positive

err, result = checked_fn(-1.0, mode="lenient")
err.throw()  # OK
```

### 37.7.2 Shape Validation

```python
@checkify.checkify(checkify.user_checks)
def matrix_multiply(a, b):
    checkify.check(
        a.shape[1] == b.shape[0],
        f"Shape mismatch: a.shape={a.shape}, b.shape={b.shape}"
    )
    return jnp.dot(a, b)

# Works
err, result = matrix_multiply(jnp.ones((3, 4)), jnp.ones((4, 5)))
err.throw()  # OK

# Fails
err, result = matrix_multiply(jnp.ones((3, 4)), jnp.ones((5, 6)))
err.throw()  # Raises shape mismatch error
```

### 37.7.3 Checking Intermediate Values

```python
@checkify.checkify(checkify.user_checks | checkify.nan_checks)
def safe_computation(x):
    # Check intermediate values
    hidden = jnp.dot(x, weight_matrix)
    checkify.check(
        jnp.all(jnp.isfinite(hidden)),
        "Hidden layer produced non-finite values"
    )

    output = jax.nn.softmax(hidden)
    checkify.check(
        jnp.allclose(jnp.sum(output, axis=-1), 1.0, atol=1e-5),
        "Softmax output does not sum to 1"
    )

    return output
```

### 37.7.4 Debugging with checkify

```python
def debug_fn(x):
    """Use checkify for debugging numerical issues."""
    checkify.check(jnp.all(jnp.isfinite(x)), f"Input has non-finite values")

    y = x * 2.0
    checkify.check(jnp.all(jnp.isfinite(y)), f"After multiply: non-finite values")

    z = jnp.log(y)
    checkify.check(jnp.all(jnp.isfinite(z)), f"After log: non-finite values")

    w = 1.0 / z
    checkify.check(jnp.all(jnp.isfinite(w)), f"After reciprocal: non-finite values")

    return w

debug_checked = jax.jit(checkify.checkify(checkify.user_checks)(debug_fn))

# Test with problematic input
x = jnp.array([1.0, 0.0, -1.0])  # 0.0 and -1.0 will cause issues
err, result = debug_checked(x)
if err:
    print(f"Debug: {err}")  # Will show which check failed
```

---

## 37.8 Integration with scan and while_loop

### 37.8.1 checkify with jax.lax.scan

```python
from jax.experimental import checkify

@checkify.checkify(checkify.float_checks)
def checked_scan_fn(x):
    def body(carry, x_i):
        new_carry = carry + x_i
        checkify.check(jnp.isfinite(new_carry), "Non-finite carry in scan")
        return new_carry, new_carry

    final, outputs = jax.lax.scan(body, 0.0, x)
    return final

x = jnp.array([1.0, 2.0, 3.0, float('nan'), 5.0])
err, result = checked_scan_fn(x)
print(bool(err))  # True -- NaN detected
```

### 37.8.2 checkify with jax.lax.while_loop

```python
@checkify.checkify(checkify.user_checks)
def checked_iteration(x, max_iters=100):
    def cond(state):
        i, val = state
        checkify.check(i < max_iters, "Maximum iterations exceeded")
        return i < max_iters

    def body(state):
        i, val = state
        checkify.check(jnp.isfinite(val), "Non-finite value in iteration")
        return (i + 1, val * 0.5)

    final_i, final_val = jax.lax.while_loop(cond, body, (0, x))
    return final_val
```

---

## 37.9 Performance Considerations

### 37.9.1 Overhead

Checkify adds runtime overhead proportional to the number and complexity of
checks. The overhead comes from:

1. **Condition evaluation**: Each check evaluates a boolean condition
2. **Error state propagation**: Error state is threaded through the computation
3. **Memory**: Error metadata consumes additional memory

### 37.9.2 Guidelines

| Scenario | Recommendation |
|---|---|
| Development/debugging | Use `all_checks` freely |
| Testing | Use specific categories for targeted checks |
| Production | Remove checkify or use minimal `user_checks` |
| Performance-critical | Only use checkify in assertions/tests, not in hot loops |

```python
# Development: full checking
dev_fn = checkify.checkify(checkify.all_checks)(my_fn)

# Production: no checking (just call the function directly)
prod_fn = my_fn

# Testing: targeted checking
test_fn = checkify.checkify(checkify.float_checks)(my_fn)
```

---

## 37.10 Complete Example: Safe Neural Network Layer

```python
import jax
import jax.numpy as jnp
from jax.experimental import checkify

def safe_linear_layer(x, weight, bias, eps=1e-8):
    """Linear layer with comprehensive error checking."""
    # Validate input shapes
    checkify.check(
        x.shape[-1] == weight.shape[0],
        f"Input dim mismatch: x has {x.shape[-1]}, weight expects {weight.shape[0]}"
    )

    # Compute
    output = jnp.dot(x, weight) + bias

    # Check for numerical issues
    checkify.check(
        jnp.all(jnp.isfinite(output)),
        "Linear layer produced non-finite outputs"
    )

    # Apply activation (GELU)
    activated = output * jax.nn.sigmoid(1.702 * output)

    checkify.check(
        jnp.all(jnp.isfinite(activated)),
        "Activation produced non-finite outputs"
    )

    # Layer normalization
    mean = jnp.mean(activated, axis=-1, keepdims=True)
    var = jnp.var(activated, axis=-1, keepdims=True)
    normalized = (activated - mean) / jnp.sqrt(var + eps)

    checkify.check(
        jnp.all(jnp.isfinite(normalized)),
        "LayerNorm produced non-finite outputs"
    )

    return normalized

# Wrap with checkify and JIT
safe_layer = jax.jit(
    checkify.checkify(checkify.user_checks)(safe_linear_layer)
)

# Test with valid input
key = jax.random.key(0)
x = jax.random.normal(key, (4, 64))
w = jax.random.normal(jax.random.fold_in(key, 1), (64, 32)) * 0.1
b = jnp.zeros(32)

err, output = safe_layer(x, w, b)
err.throw()
print(f"Output shape: {output.shape}")  # (4, 32)
print(f"Output mean: {jnp.mean(output):.4f}")  # ~0
print(f"Output std: {jnp.std(output):.4f}")    # ~1
```

---

## 37.11 API Reference

```python
# Core API
checkify.check(condition, message)        # Assert condition, raise with message
checkify.checkify(errors)                 # Decorator to enable error checking
checkify.Error                            # Error object returned by checked functions

# Error categories
checkify.user_checks       # Explicit check() calls
checkify.nan_checks        # NaN value detection
checkify.div_checks        # Division by zero detection
checkify.index_checks      # Out-of-bounds indexing
checkify.float_checks      # nan_checks | div_checks
checkify.all_checks        # All categories combined

# Combining categories
combined = checkify.nan_checks | checkify.index_checks

# Error object methods
error = err            # Error returned from checked function
bool(error)            # True if error occurred
error.get_message()    # Get error message string
error.throw()          # Raise Python exception if error occurred
str(error)             # String representation
```
