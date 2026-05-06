# Chapter 26: Debugging and Error Handling

## Overview

Debugging JAX programs presents unique challenges because JAX transformations (jit, grad, vmap, etc.) trace through Python functions to build computation graphs. Standard Python debugging tools like `print()` and `breakpoint()` do not work inside traced code. JAX provides specialized debugging utilities that operate within the tracing and compilation pipeline.

This chapter covers every debugging tool available in JAX, common error types you will encounter, and systematic strategies for diagnosing issues.

**Key Principle:** JAX code runs in two phases -- Python tracing (builds the computation graph) and compiled execution (runs on accelerator). Debugging tools must respect which phase you are targeting.

## jax.debug.print -- Printing Inside JIT

### Basic Usage

Standard `print()` only executes during tracing (once per compiled function call), not during every execution of the compiled code. `jax.debug.print` is designed to print values at runtime inside JIT-compiled functions.

```python
import jax
import jax.numpy as jnp

@jax.jit
def f(x):
    # This prints EVERY time the compiled function runs
    jax.debug.print("x = {}", x)
    y = x * 2
    jax.debug.print("y = {}", y)
    return y

f(jnp.array([1.0, 2.0, 3.0]))
# x = [1. 2. 3.]
# y = [2. 4. 6.]

f(jnp.array([4.0, 5.0, 6.0]))
# x = [4. 5. 6.]
# y = [8. 10. 12.]
```

### Format String Syntax

`jax.debug.print` uses `{}` placeholders (similar to Python's `str.format()`), not f-string syntax. You cannot use f-strings because the values are tracers, not concrete Python values.

```python
@jax.jit
def print_examples(x, y):
    # Single value
    jax.debug.print("value: {}", x)

    # Multiple values (positional)
    jax.debug.print("x = {}, y = {}", x, y)

    # Named arguments
    jax.debug.print("x = {x}, y = {y}", x=x, y=y)

    # Mixed positional and named
    jax.debug.print("first: {}, x={x}, second: {}, y={y}", x, y, x=x, y=y)

    return x + y

print_examples(jnp.array(3.0), jnp.array(5.0))
```

### Printing Inside Control Flow

`jax.debug.print` works correctly inside JAX structured control flow primitives:

```python
import jax
import jax.numpy as jnp

@jax.jit
def debug_in_scan(x):
    # Works inside lax.scan
    def body(carry, _):
        jax.debug.print("carry = {}", carry)
        return carry + 1.0, None

    final, _ = jax.lax.scan(body, x, None, length=5)
    return final

debug_in_scan(jnp.array(0.0))
# carry = 0.0
# carry = 1.0
# carry = 2.0
# carry = 3.0
# carry = 4.0
```

```python
@jax.jit
def debug_in_cond(flag, x):
    # Works inside lax.cond
    def true_fn(val):
        jax.debug.print("true branch: val = {}", val)
        return val * 2

    def false_fn(val):
        jax.debug.print("false branch: val = {}", val)
        return val * 3

    return jax.lax.cond(flag, true_fn, false_fn, x)

debug_in_cond(jnp.bool_(True), jnp.array(5.0))
# true branch: val = 5.0
```

### Printing in vmap

When used inside `jax.vmap`, `jax.debug.print` prints for each element in the batch:

```python
@jax.vmap
def debug_vmap(x):
    jax.debug.print("element: {}", x)
    return x * 2

x = jnp.array([1.0, 2.0, 3.0, 4.0])
debug_vmap(x)
# element: 1.0
# element: 2.0
# element: 3.0
# element: 4.0
```

### Ordered vs Unordered Printing

By default, `jax.debug.print` output is ordered with respect to other ordered prints. To disable ordering guarantees (which can improve performance), use `ordered=False`:

```python
@jax.jit
def unordered_print(x, y):
    # These may print in any order relative to each other
    jax.debug.print("x = {}", x, ordered=False)
    jax.debug.print("y = {}", y, ordered=False)
    return x + y
```

### Limitations

- Only JAX array types can be printed (no Python objects, strings, etc.)
- Printing inside `jax.grad` is not supported (gradients do not propagate through print)
- Heavy use of `jax.debug.print` can slow down execution significantly

## jax.debug.breakpoint -- Interactive Debugging Inside JIT

`jax.debug.breakpoint` pauses execution inside a JIT-compiled function and opens an interactive debugger prompt. This is invaluable for inspecting intermediate values at specific points in your computation.

### Basic Usage

```python
import jax
import jax.numpy as jnp

@jax.jit
def f(x):
    y = x ** 2
    # Execution pauses here; opens interactive debugger
    jax.debug.breakpoint()
    z = y + 1
    return z

f(jnp.array([1.0, 2.0, 3.0]))
```

When the breakpoint is hit, you enter an interactive prompt:

```
Entering jax.debug.breakpoint:
> f(x) at line 5

Type "help" for additional information.

debug> x
Array([1., 2., 3.], dtype=float32)

debug> y
Array([1., 4., 9.], dtype=float32)

debug> x.shape
(3,)

debug> x.dtype
dtype('float32')
```

### Debugger Commands

| Command | Description |
|---------|-------------|
| `x` | Print the value of variable `x` |
| `x.shape` | Print the shape of `x` |
| `x.dtype` | Print the dtype of `x` |
| `help` | Show available commands |
| `continue` or `c` | Resume execution |
| `quit` or `q` | Quit the debugger (raises `DebuggerExit`) |
| `where` | Show stack trace context |
| `tree` | Print the pytree structure of a variable |

### Inspecting Multiple Variables

```python
@jax.jit
def complex_fn(params, x):
    hidden = jnp.dot(x, params["w1"]) + params["b1"]
    hidden = jax.nn.relu(hidden)

    jax.debug.breakpoint()  # Inspect hidden activations

    output = jnp.dot(hidden, params["w2"]) + params["b2"]
    return output

key = jax.random.PRNGKey(0)
k1, k2 = jax.random.split(key)
params = {
    "w1": jax.random.normal(k1, (4, 8)),
    "b1": jnp.zeros(8),
    "w2": jax.random.normal(k2, (8, 2)),
    "b2": jnp.zeros(2),
}
x = jnp.ones((1, 4))

complex_fn(params, x)
# In debugger:
# debug> hidden
# debug> params["w1"].shape
# debug> tree params
```

### Conditional Breakpoints

Breakpoints can be made conditional using `jax.lax.cond`:

```python
@jax.jit
def conditional_breakpoint(i, x):
    y = x * i

    # Only break when i == 5
    def do_break(val):
        jax.debug.breakpoint()
        return val

    def no_break(val):
        return val

    jax.lax.cond(i == 5, do_break, no_break, y)
    return y
```

### Limitations

- Only works in interactive terminal sessions (not in notebooks without configuration)
- Cannot modify values in the debugger
- Cannot step through code line by line
- Values are materialized to host, which can be slow for large arrays

## jax.debug.callback -- Arbitrary Callbacks

`jax.debug.callback` allows you to execute arbitrary Python functions during JIT-compiled execution. Unlike `jax.pure_callback`, debug callbacks are not guaranteed to be called a specific number of times and should not be used for computations that affect the result.

### Basic Usage

```python
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

def plot_histogram(values):
    """Arbitrary Python function that runs during JIT execution."""
    plt.figure()
    plt.hist(values)
    plt.title("Intermediate values")
    plt.savefig("debug_histogram.png")
    plt.close()

@jax.jit
def f(x):
    y = jax.nn.softmax(x)

    # Call arbitrary Python function with intermediate values
    jax.debug.callback(plot_histogram, y)

    return jnp.sum(y)

f(jax.random.normal(jax.random.PRNGKey(0), (100,)))
```

### Callback for Logging

```python
import logging

logger = logging.getLogger("jax_debug")

def log_stats(name, values):
    """Log statistics about an array."""
    logger.info(
        f"{name}: shape={values.shape}, "
        f"mean={values.mean():.4f}, "
        f"std={values.std():.4f}, "
        f"min={values.min():.4f}, "
        f"max={values.max():.4f}"
    )

@jax.jit
def training_step(params, x):
    hidden = jnp.dot(x, params["w"])
    jax.debug.callback(log_stats, "hidden", hidden)
    output = jax.nn.gelu(hidden)
    return jnp.mean(output)
```

### Callback with Multiple Arguments

```python
def multi_arg_callback(a, b, c):
    print(f"a: {a.shape}, b: {b.shape}, c: {c.shape}")
    print(f"a sum: {a.sum()}, b sum: {b.sum()}")

@jax.jit
def f(x, y):
    z = x + y
    jax.debug.callback(multi_arg_callback, x, y, z)
    return z
```

### Ordered Callbacks

Like `jax.debug.print`, callbacks can be ordered or unordered:

```python
@jax.jit
def ordered_callbacks(x):
    y = x + 1
    jax.debug.callback(lambda v: print(f"step 1: {v}"), y, ordered=True)
    z = y * 2
    jax.debug.callback(lambda v: print(f"step 2: {v}"), z, ordered=True)
    return z
```

### Differences from jax.pure_callback

| Feature | jax.debug.callback | jax.pure_callback |
|---------|--------------------|--------------------|
| Return value | None (ignored) | Can return JAX arrays |
| Execution guarantee | Best-effort | Guaranteed once per logical call |
| Use case | Debugging, logging | Integrating external computations |
| Affects result | No | Yes |
| Works with grad | No gradients through it | Gradients can be defined |

## Checkify -- Functional Error Checking

Checkify is JAX's functional error checking system. It allows you to embed assertions inside JIT-compiled functions that are checked at runtime without breaking functional semantics.

### Basic Checkify Usage

```python
import jax
import jax.numpy as jnp
from jax.experimental import checkify

def safe_divide(a, b):
    # Embed an assertion inside the function
    checkify.check(jnp.all(b != 0), "Division by zero!")
    return a / b

# Transform the function with checkify
checked_safe_divide = checkify.checkify(safe_divide)

# Call returns (error, result) tuple
err, result = checked_safe_divide(jnp.array(10.0), jnp.array(2.0))
print(result)  # 5.0
err.throw()    # No error -- does nothing

# With actual error
err, result = checked_safe_divide(jnp.array(10.0), jnp.array(0.0))
err.throw()    # Raises CheckifyError: Division by zero!
```

### Error Categories

Checkify supports several categories of automatic checks that do not require explicit `checkify.check` calls:

```python
from jax.experimental import checkify

# Available error categories:
# - checkify.user_checks:    Only explicit check() calls
# - checkify.nan_checks:     Automatically detect NaN values
# - checkify.float_checks:   Detect NaN and Inf values
# - checkify.div_checks:     Detect division by zero (integer)
# - checkify.index_checks:   Detect out-of-bounds indexing
# - checkify.all_checks:     All of the above combined
```

### NaN Checks

```python
import jax
import jax.numpy as jnp
from jax.experimental import checkify

def compute(x):
    return jnp.log(x)  # log of negative = NaN

checked_compute = checkify.checkify(compute, errors=checkify.nan_checks)

err, result = checked_compute(jnp.array(-1.0))
err.throw()
# Raises: CheckifyError: Numerical error: nan generated by elementwise
```

### Index Checks (Out-of-Bounds)

```python
from jax.experimental import checkify

def gather(x, idx):
    return x[idx]

checked_gather = checkify.checkify(gather, errors=checkify.index_checks)

x = jnp.array([1.0, 2.0, 3.0])
err, result = checked_gather(x, jnp.array(5))  # Index 5 is out of bounds
err.throw()
# Raises: CheckifyError: out-of-bounds indexing
```

### All Checks Combined

```python
from jax.experimental import checkify

def risky_function(x, idx):
    checkify.check(x.shape[0] > 0, "Input must not be empty")  # user check
    return x[idx] / (x[0] - x[0])  # potential: index OOB, division by zero, NaN

checked_fn = checkify.checkify(risky_function, errors=checkify.all_checks)

x = jnp.array([1.0, 2.0, 3.0])
err, result = checked_fn(x, jnp.array(1))
err.throw()  # Raises due to division by zero producing NaN
```

### Checkify with JIT

Checkified functions must be explicitly compiled with `checkify` before `jax.jit`:

```python
from jax.experimental import checkify
import jax
import jax.numpy as jnp

def my_fn(x):
    checkify.check(jnp.all(x >= 0), "x must be non-negative")
    return jnp.sqrt(x)

# Option 1: Checkify then JIT
checked_fn = checkify.checkify(my_fn, errors=checkify.user_checks)
jitted_checked = jax.jit(checked_fn)

err, result = jitted_checked(jnp.array([4.0, 9.0]))
err.throw()  # OK

err, result = jitted_checked(jnp.array([-1.0]))
err.throw()  # Raises CheckifyError
```

```python
# Option 2: Use checkify as a decorator-style approach
def make_checked_jitted_fn(fn, errors=checkify.user_checks):
    checked = checkify.checkify(fn, errors=errors)
    return jax.jit(checked)

checked_sqrt = make_checked_jitted_fn(my_fn)
err, result = checked_sqrt(jnp.array([4.0]))
err.throw()
```

### Error.get() vs Error.throw()

```python
from jax.experimental import checkify

def maybe_failing(x):
    checkify.check(x > 0, "x must be positive")
    return x * 2

checked = checkify.checkify(maybe_failing)

err, result = checked(jnp.array(-1.0))

# Option 1: throw() -- raises an exception if there is an error
try:
    err.throw()
except checkify.CheckifyError as e:
    print(f"Error thrown: {e}")

# Option 2: get() -- returns the error message without raising
error_data = err.get()
if error_data is not None:
    print(f"Error detected: {error_data}")
else:
    print("No error")
```

### Checkify in Training Loops

```python
import jax
import jax.numpy as jnp
from jax.experimental import checkify
import optax

def loss_fn(params, x, y):
    pred = jnp.dot(x, params["w"]) + params["b"]
    loss = jnp.mean((pred - y) ** 2)
    checkify.check(jnp.isfinite(loss), "Loss is not finite!")
    return loss

# Checkify the loss function
checked_loss = checkify.checkify(loss_fn, errors=checkify.all_checks)

def train_step(params, opt_state, x, y):
    # Use checkified loss inside grad
    def checked_and_graded(p):
        err, loss = checked_loss(p, x, y)
        return (err, loss), loss  # Return error alongside loss

    (err, loss), grads = jax.value_and_grad(checked_and_graded, has_aux=True)(params)

    updates, opt_state = optax.adam(1e-3).update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, err

jitted_step = jax.jit(train_step)
```

## NaN Debugging

### Enabling NaN Debugging Globally

JAX can automatically detect and trace NaN values during execution:

```python
import jax
import jax.numpy as jnp

# Enable NaN debugging
jax.config.update("jax_debug_nans", True)

@jax.jit
def nan_producer(x):
    return jnp.log(x)  # log(negative) = NaN

# This will raise an error pointing to the NaN-producing operation
nan_producer(jnp.array(-1.0))
# Raises: Floating point error: NaN generated
```

### When NaN Debugging Catches Issues

```python
import jax
import jax.numpy as jnp

jax.config.update("jax_debug_nans", True)

# Caught: NaN from mathematical operation
def f1(x):
    return jnp.sqrt(x)  # sqrt(negative) = NaN

# Caught: NaN from division
def f2(x):
    return x / 0.0

# Caught: NaN from subtraction of equal values (0/0 case)
def f3(x):
    return (x - x) / (x - x)  # 0/0 = NaN

# Each of these will raise when called with appropriate inputs
```

### Performance Impact

NaN debugging adds overhead to every floating-point operation. Only enable it during debugging sessions:

```python
import jax

# Enable only for debugging
jax.config.update("jax_debug_nans", True)

# ... debug your code ...

# Disable for production
jax.config.update("jax_debug_nans", False)
```

### NaN Debugging with grad

```python
import jax
import jax.numpy as jnp

jax.config.update("jax_debug_nans", True)

def unstable_loss(params, x):
    pred = jnp.dot(x, params)
    # If pred contains very large values, exp(pred) can overflow
    return jnp.sum(jnp.log(jnp.exp(pred) + 1))

params = jnp.array([1e10])
x = jnp.array([1.0])

# This may catch NaN in the gradient computation
grad_fn = jax.grad(unstable_loss)
grads = grad_fn(params, x)  # May raise NaN error
```

## Tracer Leak Detection

JAX tracers represent abstract values during tracing. A tracer "leak" occurs when a tracer value escapes the scope of the transformation that created it. Enabling tracer leak detection helps find these bugs.

### Enabling Tracer Leak Detection

```python
import jax

# Enable tracer leak detection
jax.check_tracer_leaks = True

# Or via config
jax.config.update("jax_check_tracer_leaks", True)
```

### Example of a Tracer Leak

```python
import jax
import jax.numpy as jnp

jax.check_tracer_leaks = True

saved_value = None

@jax.jit
def leaky_function(x):
    global saved_value
    saved_value = x  # LEAK: tracer escapes JIT scope
    return x + 1

leaky_function(jnp.array(1.0))
# Raises: Exception: Leaked tracer
```

### Common Causes of Tracer Leaks

```python
import jax
import jax.numpy as jnp

# 1. Storing tracers in global/module-level variables
_cache = {}

@jax.jit
def bad_cache(x):
    _cache["last_input"] = x  # Leak!
    return x * 2

# 2. Storing tracers in object attributes
class BadModel:
    def __init__(self):
        self.last_x = None

    @jax.jit
    def apply(self, x):
        self.last_x = x  # Leak!
        return x * 2

# 3. Appending tracers to lists outside the function scope
collected = []

@jax.jit
def bad_collect(x):
    collected.append(x)  # Leak!
    return x * 2
```

## Common Error Types and Their Meanings

### ConcretizationTypeError

This error occurs when JAX tries to use a traced value as a concrete Python value (e.g., in a Python `if` statement or as a loop bound).

```python
import jax
import jax.numpy as jnp

@jax.jit
def bad_if(x):
    # ERROR: cannot use traced value in Python if
    if x > 0:  # x is a tracer, not a concrete value
        return x
    else:
        return -x

# Raises: ConcretizationTypeError: Abstract tracer value encountered
# where concrete value is needed: Traced<ShapedArray(float32[], weak_type=True)>
```

**Fix:** Use `jax.lax.cond`:

```python
@jax.jit
def good_if(x):
    return jax.lax.cond(x > 0, lambda x: x, lambda x: -x, x)
```

**Common triggers:**

```python
# 1. Python if with traced condition
@jax.jit
def f(x):
    if x.sum() > 0:  # Error!
        return x
    return -x

# 2. Python for loop with traced range
@jax.jit
def f(x, n):
    for i in range(n):  # Error if n is traced!
        x = x + 1
    return x

# 3. Using traced value as array shape
@jax.jit
def f(x):
    return jnp.ones((x, 3))  # Error! x is traced

# 4. Using traced value in print formatting
@jax.jit
def f(x):
    print(f"x is {x}")  # Error during tracing if x is traced
    return x
```

### NonConcreteBooleanIndexError

Occurs when using a traced boolean array for indexing (boolean masking):

```python
import jax
import jax.numpy as jnp

@jax.jit
def bad_mask(x, mask):
    return x[mask]  # Error: mask is a traced boolean array

# Raises: NonConcreteBooleanIndexError: Array boolean indices must be concrete
```

**Fix:** Use `jnp.where` or `jnp.ndarray.at`:

```python
@jax.jit
def good_mask(x, mask):
    # Option 1: jnp.where
    return jnp.where(mask, x, 0.0)

    # Option 2: multiply by boolean mask
    # return x * mask
```

### TracerArrayConversionError

Occurs when trying to convert a JAX tracer to a NumPy array:

```python
import jax
import jax.numpy as jnp

@jax.jit
def bad_numpy(x):
    return np.array(x)  # Error: cannot convert tracer to numpy array

# Raises: TracerArrayConversionError: The numpy.ndarray conversion method
# __array__() was called on traced array
```

**Fix:** Avoid converting tracers to NumPy inside traced functions. Use JAX operations instead:

```python
@jax.jit
def good_numpy(x):
    # Stay in JAX land
    return x.astype(jnp.float32)
```

### UnexpectedTracerError

Occurs when a stale tracer from a previous trace is used in a new context:

```python
import jax
import jax.numpy as jnp

@jax.jit
def create_tracer(x):
    return x + 1

# First call creates a tracer
result = create_tracer(jnp.array(1.0))

# Using stale tracer in a new JIT compilation
@jax.jit
def use_stale(y):
    return result + y  # Error: result is from a different trace

use_stale(jnp.array(2.0))
# Raises: UnexpectedTracerError: Encountered an unexpected tracer
```

**Fix:** Pass all needed values explicitly as arguments:

```python
@jax.jit
def use_correctly(result, y):
    return result + y

result = create_tracer(jnp.array(1.0))
use_correctly(result, jnp.array(2.0))  # OK -- result is passed explicitly
```

## JIT Compilation Errors

### Shape Polymorphism Errors

```python
import jax
import jax.numpy as jnp

@jax.jit
def reshape_fn(x):
    # This works only for the specific shape seen during first call
    return x.reshape(2, 3)

reshape_fn(jnp.arange(6))   # OK: shape (6,) -> (2, 3)
reshape_fn(jnp.arange(12))  # Error: recompilation needed, shape mismatch
```

**Fix:** Use `static_argnums` for shape-dependent arguments or write shape-polymorphic code:

```python
@jax.jit
def reshape_fn_poly(x):
    return x.reshape(-1, x.shape[-1])  # Works for any shape with last dim

# Or use static_argnums
@jax.jit
def reshape_with_static(x, rows, cols):
    return x.reshape(rows, cols)

# Best: use shaped input hint
jax.jit(reshape_fn, static_argnums=())
```

### Recompilation Overhead

```python
import jax
import jax.numpy as jnp

@jax.jit
def my_fn(x):
    return x + 1

# Different shapes trigger recompilation
my_fn(jnp.ones(3))    # Compiles
my_fn(jnp.ones(4))    # Recompiles!
my_fn(jnp.ones(5))    # Recompiles again!

# Fix: use consistent shapes, or pad/batch to fixed size
```

## Shape and dtype Mismatch Errors

### Common Shape Errors

```python
import jax
import jax.numpy as jnp

@jax.jit
def matmul_error():
    a = jnp.ones((3, 4))
    b = jnp.ones((5, 6))
    return jnp.dot(a, b)  # Error: incompatible shapes for matmul

# Raises: TypeError: dot_general requires contracting dimensions to have
# the same shape, got (4,) and (5,).
```

### Debugging Shape Issues

```python
import jax
import jax.numpy as jnp

@jax.jit
def debug_shapes(x):
    # Print shapes during tracing (not runtime)
    # These are concrete because shapes are known at trace time
    print(f"Input shape: {x.shape}")
    print(f"Input dtype: {x.dtype}")

    y = jnp.dot(x, x.T)
    print(f"After dot shape: {y.shape}")

    z = y.reshape(-1)
    print(f"After reshape shape: {z.shape}")

    return z

debug_shapes(jnp.ones((3, 4)))
# Input shape: (3, 4)
# Input dtype: float32
# After dot shape: (3, 3)
# After reshape shape: (9,)
```

### dtype Mismatch

```python
import jax
import jax.numpy as jnp

@jax.jit
def dtype_mismatch():
    a = jnp.ones(3, dtype=jnp.float32)
    b = jnp.ones(3, dtype=jnp.int32)
    return a + b  # OK: JAX auto-promotes to float32

@jax.jit
def strict_dtype():
    a = jnp.ones(3, dtype=jnp.bfloat16)
    b = jnp.ones(3, dtype=jnp.float32)
    # This works but may not be desired -- implicit upcast to float32
    return a + b
```

## Debugging Strategies and Workflow

### Strategy 1: Progressive JIT Application

Start without JIT, add it incrementally:

```python
import jax
import jax.numpy as jnp

# Step 1: Define function WITHOUT jit -- use regular Python debugging
def my_complex_fn(x, params):
    hidden = jnp.dot(x, params["w1"]) + params["b1"]
    # Standard Python debugging works here
    print(f"hidden shape: {hidden.shape}, mean: {hidden.mean()}")
    hidden = jax.nn.relu(hidden)
    output = jnp.dot(hidden, params["w2"]) + params["b2"]
    return output

# Step 2: Test the function
params = {"w1": jnp.ones((4, 8)), "b1": jnp.zeros(8),
          "w2": jnp.ones((8, 2)), "b2": jnp.zeros(2)}
result = my_complex_fn(jnp.ones((1, 4)), params)
assert result.shape == (1, 2)

# Step 3: Add JIT only after confirming correctness
@jax.jit
def my_complex_fn(x, params):
    # Replace print with jax.debug.print
    hidden = jnp.dot(x, params["w1"]) + params["b1"]
    jax.debug.print("hidden shape: {}", hidden.shape)  # shape is static
    hidden = jax.nn.relu(hidden)
    output = jnp.dot(hidden, params["w2"]) + params["b2"]
    return output
```

### Strategy 2: Use jax.make_jaxpr for Inspection

```python
import jax
import jax.numpy as jnp

def my_fn(x, y):
    z = x + y
    w = jnp.dot(z, z.T)
    return jnp.sum(w)

# View the jaxpr (intermediate representation)
x = jnp.ones((3, 4))
y = jnp.ones((3, 4))
jaxpr = jax.make_jaxpr(my_fn)(x, y)
print(jaxpr)

# Output shows:
# { lambda ; a:f32[3,4] b:f32[3,4]. let
#     c:f32[3,4] = add a b
#     d:f32[3,3] = dot_general[c dimension_numbers=((), ()), ...] c c
#     e:f32[] = reduce_sum[axes=(0, 1)] d
#   in (e,) }
```

### Strategy 3: Disable JIT Temporarily

```python
import jax

# Disable all JIT compilation globally
jax.config.update("jax_disable_jit", True)

# Now all jax.jit-decorated functions run in eager mode
# Python debugging tools work normally

# ... debug ...

# Re-enable JIT
jax.config.update("jax_disable_jit", False)
```

### Strategy 4: Inspect Compilation with lower()

```python
import jax
import jax.numpy as jnp

@jax.jit
def my_fn(x):
    return jnp.sum(x ** 2)

# Lower to HLO without compiling
lowered = my_fn.lower(jnp.ones(5))
print(lowered.as_text())  # View the HLO IR

# Check compilation cost analysis
compiled = lowered.compile()
print(compiled.cost_analysis())
# Shows: {'flops': 10, 'bytes accessed': 40, ...}
```

### Strategy 5: Systematic Error Reduction

When encountering an error in a large function:

```python
import jax
import jax.numpy as jnp

@jax.jit
def large_fn(x):
    # Step 1: Comment out everything, return input
    return x

# Step 2: Add back one operation at a time
@jax.jit
def large_fn(x):
    a = x + 1
    return a

# Step 3: Continue adding until error appears
@jax.jit
def large_fn(x):
    a = x + 1
    b = jnp.dot(a, a.T)  # If error appears here, investigate shapes
    return b
```

### Strategy 6: Use jax.ShapeDtypeStruct for Mock Data

```python
import jax
import jax.numpy as jnp

# Create lightweight mock arrays for shape checking
mock_x = jax.ShapeDtypeStruct(shape=(32, 784), dtype=jnp.float32)
mock_params = {
    "w": jax.ShapeDtypeStruct(shape=(784, 256), dtype=jnp.float32),
    "b": jax.ShapeDtypeStruct(shape=(256,), dtype=jnp.float32),
}

# Use make_jaxpr to check shapes without computing
def predict(params, x):
    return jnp.dot(x, params["w"]) + params["b"]

jaxpr = jax.make_jaxpr(predict)(mock_params, mock_x)
print(f"Output shape: {jaxpr.out_avals[0].shape}")  # (32, 256)
```

### Strategy 7: Environment Variables for Debugging

```bash
# Show full traceback without filtering
export JAX_TRACEBACK_FILTERING=off

# Disable JIT globally
export JAX_DISABLE_JIT=1

# Enable NaN checking
export JAX_DEBUG_NANS=1

# Enable tracer leak detection
export JAX_CHECK_TRACER_LEAKS=1

# Print compilation logging
export JAX_LOGGING_LEVEL=DEBUG

# Show XLA compilation flags
export XLA_FLAGS="--xla_dump_to=/tmp/xla_dump"
```

## Summary: Debugging Quick Reference

| Problem | Solution |
|---------|----------|
| Print inside JIT | `jax.debug.print("x = {}", x)` |
| Inspect values at runtime | `jax.debug.breakpoint()` |
| Run arbitrary Python in JIT | `jax.debug.callback(fn, *args)` |
| Assert in JIT | `checkify.check(condition, "msg")` |
| Find NaN source | `jax.config.update("jax_debug_nans", True)` |
| Find tracer leaks | `jax.check_tracer_leaks = True` |
| Disable JIT for debugging | `jax.config.update("jax_disable_jit", True)` |
| View computation graph | `jax.make_jaxpr(fn)(*args)` |
| View HLO IR | `jax.jit(fn).lower(*args).as_text()` |
| Python if with traced value | Use `jax.lax.cond` |
| Python for with traced bound | Use `jax.lax.fori_loop` or `jax.lax.scan` |
| Boolean indexing with traced mask | Use `jnp.where(mask, x, default)` |
| Stale tracer error | Pass values explicitly as arguments |
| Shape/dtype mismatch | Use `jax.make_jaxpr` to inspect shapes |
| Recompilation overhead | Use consistent shapes or `static_argnums` |
