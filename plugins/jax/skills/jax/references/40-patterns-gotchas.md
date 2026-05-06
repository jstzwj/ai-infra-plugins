# Chapter 40: Common Patterns and Gotchas

## 40.1 Pure Functions

### 40.1.1 Why Purity Matters

JAX transformations (`jit`, `grad`, `vmap`, etc.) rely on functions being **pure**:
- Same inputs always produce same outputs
- No side effects (no mutation of global state, no I/O)
- No reliance on mutable external state

```python
import jax
import jax.numpy as jnp

# BAD: Impure function (relies on external state)
counter = 0
def impure_fn(x):
    global counter
    counter += 1  # Side effect!
    return x + counter

# jax.jit(impure_fn) will trace counter=0, then never update it
# The compiled function always uses the traced value of counter

# GOOD: Pure function (all state passed as arguments)
def pure_fn(x, counter):
    return x + counter + 1, counter + 1  # Return new state
```

### 40.1.2 Common Purity Violations

```python
# BAD: Using Python random
import random
def bad_random(x):
    return x + random.random()  # Non-deterministic at trace time

# GOOD: Use jax.random
def good_random(x, key):
    noise = jax.random.normal(key, x.shape)
    return x + noise

# BAD: Using print() inside JIT
@jax.jit
def bad_print(x):
    print(x)  # Only prints during tracing, not at runtime
    return x + 1

# GOOD: Use jax.debug.print for runtime printing
@jax.jit
def good_print(x):
    jax.debug.print("x = {}", x)
    return x + 1

# BAD: Mutating arrays
def bad_mutation(x):
    x[0] = 0  # JAX arrays are immutable!
    return x

# GOOD: Return a new array
def good_update(x):
    return x.at[0].set(0)  # Returns new array with element 0 set to 0
```

---

## 40.2 In-Place Updates Pattern

### 40.2.1 The .at[] API

JAX arrays are immutable. Use the `.at[]` indexer for "in-place" updates:

```python
import jax.numpy as jnp

x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

# Set a single element
y = x.at[2].set(0.0)
print(y)  # [1.0, 2.0, 0.0, 4.0, 5.0]
print(x)  # [1.0, 2.0, 3.0, 4.0, 5.0] -- original unchanged!

# Add to an element
y = x.at[0].add(10.0)
print(y)  # [11.0, 2.0, 3.0, 4.0, 5.0]

# Multiply an element
y = x.at[1].multiply(3.0)
print(y)  # [1.0, 6.0, 3.0, 4.0, 5.0]

# Subtract from an element
y = x.at[3].subtract(2.0)
print(y)  # [1.0, 2.0, 3.0, 2.0, 5.0]

# Power of an element
y = x.at[4].power(2.0)
print(y)  # [1.0, 2.0, 3.0, 4.0, 25.0]

# Minimum with a value
y = x.at[:].min(3.0)
print(y)  # [1.0, 2.0, 3.0, 3.0, 3.0]

# Maximum with a value
y = x.at[:].max(3.0)
print(y)  # [3.0, 3.0, 3.0, 4.0, 5.0]
```

### 40.2.2 Slicing with .at[]

```python
x = jnp.zeros((5, 5))

# Set a row
y = x.at[2, :].set(1.0)

# Set a column
y = x.at[:, 3].set(2.0)

# Set a submatrix
y = x.at[1:3, 2:4].set(jnp.ones((2, 2)))

# Apply function to a slice
y = x.at[0, :].apply(lambda row: row + jnp.arange(5))

# Update with boolean mask
mask = jnp.array([True, False, True, False, True])
y = x.at[mask].set(99.0)

# Update with integer array indexing
indices = jnp.array([0, 2, 4])
y = x.at[indices].set(1.0)
```

### 40.2.3 Updating PyTrees

```python
import jax

params = {
    'layer1': {'w': jnp.ones((3, 4)), 'b': jnp.zeros(4)},
    'layer2': {'w': jnp.ones((4, 2)), 'b': jnp.zeros(2)},
}

# Apply a function to all leaves
new_params = jax.tree.map(lambda x: x * 0.5, params)

# Apply with different functions per leaf
def update_param(param, grad, lr=0.01):
    return param - lr * grad

grads = jax.tree.map(lambda x: jnp.ones_like(x) * 0.1, params)
updated = jax.tree.map(update_param, params, grads)
```

---

## 40.3 Training Loop Pattern

### 40.3.1 Basic Training Loop

```python
import jax
import jax.numpy as jnp

def init_params(key, input_dim, hidden_dim, output_dim):
    """Initialize network parameters."""
    k1, k2, k3 = jax.random.split(key, 3)
    params = {
        'w1': jax.random.normal(k1, (input_dim, hidden_dim)) * 0.01,
        'b1': jnp.zeros(hidden_dim),
        'w2': jax.random.normal(k2, (hidden_dim, output_dim)) * 0.01,
        'b2': jnp.zeros(output_dim),
    }
    return params

def predict(params, x):
    """Forward pass."""
    hidden = jnp.dot(x, params['w1']) + params['b1']
    hidden = jax.nn.relu(hidden)
    output = jnp.dot(hidden, params['w2']) + params['b2']
    return output

def loss_fn(params, x, y):
    """Mean squared error loss."""
    pred = predict(params, x)
    return jnp.mean((pred - y) ** 2)

# Compile loss and gradient
@jax.jit
def train_step(params, x, y, lr=0.01):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    new_params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
    return new_params, loss

# Training loop
key = jax.random.key(0)
params = init_params(key, input_dim=10, hidden_dim=32, output_dim=1)

# Generate data
x = jax.random.normal(key, (100, 10))
y = jnp.sum(x[:, :3], axis=-1, keepdims=True)

for epoch in range(100):
    params, loss = train_step(params, x, y, lr=0.01)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: loss = {loss:.4f}")
```

### 40.3.2 Training with State (Optimizer, RNG)

```python
from typing import NamedTuple
import optax

class TrainState(NamedTuple):
    params: dict
    opt_state: optax.OptState
    key: jax.Array

def create_train_state(key, learning_rate=1e-3):
    params = init_params(key, input_dim=10, hidden_dim=32, output_dim=1)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    return TrainState(params=params, opt_state=opt_state, key=key)

@jax.jit
def train_step_with_state(state, x, y):
    key, new_key = jax.random.split(state.key)
    optimizer = optax.adam(1e-3)

    loss, grads = jax.value_and_grad(loss_fn)(state.params, x, y)
    updates, new_opt_state = optimizer.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)

    new_state = TrainState(
        params=new_params,
        opt_state=new_opt_state,
        key=new_key,
    )
    return new_state, loss

# Usage
state = create_train_state(jax.random.key(42))
for epoch in range(100):
    state, loss = train_step_with_state(state, x, y)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: loss = {loss:.4f}")
```

### 40.3.3 Mini-Batch Training with shmap

```python
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map

def create_minibatches(data, batch_size, key):
    """Create random minibatches."""
    n = data['x'].shape[0]
    perm = jax.random.permutation(key, n)
    x_shuffled = data['x'][perm]
    y_shuffled = data['y'][perm]
    num_batches = n // batch_size
    x_batches = x_shuffled[:num_batches * batch_size].reshape(num_batches, batch_size, -1)
    y_batches = y_shuffled[:num_batches * batch_size].reshape(num_batches, batch_size, -1)
    return x_batches, y_batches

@jax.jit
def epoch_step(params, x_batch, y_batch, lr=0.01):
    """Single epoch over all mini-batches."""
    def step(params, batch):
        x, y = batch
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        new_params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
        return new_params, loss

    batches = (x_batch, y_batch)
    params, losses = jax.lax.scan(step, params, batches)
    return params, jnp.mean(losses)
```

---

## 40.4 Common Gotchas

### 40.4.1 In-Place Mutation Does Not Work

```python
# BAD
x = jnp.array([1, 2, 3])
x[0] = 99  # TypeError: '<class 'jaxlib.xla_extension.ArrayImpl'>' object does not support item assignment

# GOOD
x = jnp.array([1, 2, 3])
x = x.at[0].set(99)
```

### 40.4.2 NumPy vs JAX Random

```python
# BAD: NumPy random is not JAX-traceable
import numpy as np
@jax.jit
def bad_random():
    return jnp.array(np.random.randn(3))  # Traced once, always same value!

# GOOD: Use jax.random
@jax.jit
def good_random(key):
    return jax.random.normal(key, (3,))

key = jax.random.key(0)
print(good_random(key))  # Different each call (with different key)
```

### 40.4.3 Python Control Flow vs JAX Control Flow

```python
# BAD: Python if/for with traced values
@jax.jit
def bad_conditional(x):
    if x > 0:  # Error: abstract tracer value used in Python boolean context
        return x
    return -x

# GOOD: Use jax.lax.cond or jnp.where
@jax.jit
def good_conditional(x):
    return jnp.where(x > 0, x, -x)

# Or use jax.lax.cond for more complex branches
@jax.jit
def good_conditional2(x):
    return jax.lax.cond(x > 0, lambda x: x, lambda x: -x, x)
```

```python
# BAD: Python for loop with traced bounds
@jax.jit
def bad_loop(x, n):
    for i in range(n):  # Error: n is traced, can't use as range argument
        x = x + 1
    return x

# GOOD: Use jax.lax.fori_loop
@jax.jit
def good_loop(x, n):
    body = lambda i, x: x + 1
    return jax.lax.fori_loop(0, n, body, x)

# GOOD: Use jax.lax.scan for collection
@jax.jit
def good_scan(x, n):
    def body(carry, _):
        return carry + 1, None
    result, _ = jax.lax.scan(body, x, None, length=n)
    return result
```

### 40.4.4 Array Shapes Must Be Static for JIT

```python
# BAD: Shape depends on data
@jax.jit
def bad_reshape(x):
    n = x.shape[0]
    nonzero = jnp.sum(x != 0)  # Data-dependent count
    return x[x != 0]  # Data-dependent shape! Can't JIT-compile.

# GOOD: Use fixed shapes or padded outputs
@jax.jit
def good_filter(x, threshold=0.0):
    mask = x > threshold
    # Return mask and let caller handle dynamic shapes
    return x * mask, mask
```

### 40.4.5 Float64 Disabled by Default

```python
# By default, float64 is disabled
x = jnp.array([1.0, 2.0, 3.0])
print(x.dtype)  # float32

x64 = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
print(x64.dtype)  # float32 (!) -- silently downcast

# Enable float64
jax.config.update("jax_enable_x64", True)
x64 = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
print(x64.dtype)  # float64

# Or via environment variable: JAX_ENABLE_X64=True
```

### 40.4.6 Tracing vs Runtime

```python
# Print during tracing (not runtime)
@jax.jit
def tracing_fn(x):
    print(f"Tracing with x.shape = {x.shape}")  # Printed once per shape
    return x + 1

tracing_fn(jnp.ones(3))   # Prints "Tracing with x.shape = (3,)"
tracing_fn(jnp.ones(3))   # Does NOT print (cached)
tracing_fn(jnp.ones(5))   # Prints "Tracing with x.shape = (5,)" (new trace)

# Print at runtime (during execution)
@jax.jit
def runtime_fn(x):
    jax.debug.print("x = {}", x)  # Printed every time function is called
    return x + 1
```

### 40.4.7 Gradient of Integer Operations

```python
# BAD: Gradients through integer operations
@jax.jit
def bad_grad(x):
    idx = jnp.argmax(x)  # Returns integer
    return x[idx].astype(jnp.float32)  # Argmax is not differentiable!

# GOOD: Use soft approximation
@jax.jit
def good_grad(x):
    weights = jax.nn.softmax(x * 10)  # Soft approximation of argmax
    return jnp.sum(x * weights)  # Differentiable
```

### 40.4.8 Device Placement

```python
# JAX arrays live on specific devices
x = jnp.ones(5)
print(x.device())  # TfmdCpuDevice(id=0) or GpuDevice(id=0)

# Explicit device placement
with jax.default_device(jax.devices("gpu")[0]):
    x_gpu = jnp.ones(5)  # Created on GPU

# Move between devices
x_cpu = jax.device_put(x_gpu, jax.devices("cpu")[0])
```

### 40.4.9 Donation for Memory Efficiency

```python
# Donate buffers that are no longer needed
@jax.jit(donate_argnums=(0,))  # First argument's buffer will be reused
def update_params(params, grads, lr=0.01):
    return jax.tree.map(lambda p, g: p - lr * g, params, grads)

# After calling update_params, the original params buffer is freed
# and its memory is reused for the result
new_params = update_params(old_params, grads)
# old_params is now invalid (donated)! Do not use it.
```

### 40.4.10 Side Effects in PyTree Leaves

```python
# BAD: Mixing numpy and jax arrays in computation
import numpy as np
@jax.jit
def bad_mix(x):
    np_val = np.array([1, 2, 3])  # NumPy array in computation
    return x + jnp.array(np_val)  # Works but forces device transfer

# GOOD: Convert to JAX array outside JIT
np_val = jnp.array(np.array([1, 2, 3]))

@jax.jit
def good_mix(x):
    return x + np_val  # Pure JAX computation
```

---

## 40.5 NumPy to JAX Migration Guide

### 40.5.1 Key Differences

| Feature | NumPy | JAX |
|---|---|---|
| Array mutation | `x[0] = 5` | `x = x.at[0].set(5)` |
| Random numbers | `np.random.seed(0)` | `key = jax.random.key(0)` |
| Control flow | Python `if`, `for` | `jax.lax.cond`, `jax.lax.scan` |
| Parallelism | Implicit (vectorized) | `jax.vmap`, `jax.pmap` |
| JIT compilation | N/A | `@jax.jit` |
| Gradients | N/A | `jax.grad` |
| Float64 default | Yes | No (opt-in) |
| Lazy evaluation | No (eager) | Yes (with JIT) |

### 40.5.2 API Compatibility

Most `numpy` functions have direct `jax.numpy` equivalents:

```python
import numpy as np
import jax.numpy as jnp

# Direct replacements (same API)
np.zeros((3, 4))    ->  jnp.zeros((3, 4))
np.ones((3, 4))     ->  jnp.ones((3, 4))
np.arange(10)       ->  jnp.arange(10)
np.linspace(0, 1, 5) -> jnp.linspace(0, 1, 5)
np.dot(a, b)        ->  jnp.dot(a, b)
np.sum(x)           ->  jnp.sum(x)
np.mean(x)          ->  jnp.mean(x)
np.max(x)           ->  jnp.max(x)
np.reshape(x, (3,4))->  jnp.reshape(x, (3, 4))
np.concatenate([a,b])->  jnp.concatenate([a, b])
np.exp(x)           ->  jnp.exp(x)
np.log(x)           ->  jnp.log(x)
np.sin(x)           ->  jnp.sin(x)
```

### 40.5.3 Random Number Migration

```python
# NumPy style
np.random.seed(42)
x = np.random.randn(3, 4)

# JAX style (functional, stateless)
key = jax.random.key(42)
x = jax.random.normal(key, (3, 4))

# Generating multiple random values
key = jax.random.key(0)
k1, k2, k3 = jax.random.split(key, 3)
x1 = jax.random.normal(k1, (3,))
x2 = jax.random.uniform(k2, (3,))
x3 = jax.random.bernoulli(k3, 0.5, (3,))

# Inside a training loop
def training_step(params, batch, key):
    k1, k2 = jax.random.split(key)
    dropout_mask = jax.random.bernoulli(k1, 0.5, batch.shape)
    # ... use k2 for next step ...
    return new_params, k2
```

### 40.5.4 Operations Without Direct Equivalents

```python
# NumPy: np.where with 3 args works in JAX too
result = jnp.where(condition, x, y)  # Same as NumPy

# NumPy: np.argmax/argmin work but return non-differentiable indices
idx = jnp.argmax(x)  # Works, but can't differentiate through idx

# NumPy: np.unique -- NOT supported in JAX (data-dependent output shape)
# Use jnp.unique but only outside JIT, or use a workaround
```

### 40.5.5 Full Migration Example

```python
# Before: NumPy implementation
import numpy as np

class NumpyMLP:
    def __init__(self, layer_sizes):
        self.weights = []
        self.biases = []
        for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.weights.append(np.random.randn(n_in, n_out) * 0.01)
            self.biases.append(np.zeros(n_out))

    def predict(self, x):
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            x = np.maximum(0, x @ w + b)  # ReLU
        x = x @ self.weights[-1] + self.biases[-1]
        return x

    def train_step(self, x, y, lr=0.01):
        # Manual gradient computation (complex!)
        pred = self.predict(x)
        loss = np.mean((pred - y) ** 2)
        # ... implement backprop manually ...
        return loss

# After: JAX implementation
import jax
import jax.numpy as jnp

class JaxMLP:
    def __init__(self, layer_sizes, key):
        self.params = {}
        for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            key, k = jax.random.split(key)
            self.params[f'w{i}'] = jax.random.normal(k, (n_in, n_out)) * 0.01
            self.params[f'b{i}'] = jnp.zeros(n_out)

    @staticmethod
    def predict(params, x):
        n_layers = len(params) // 2
        for i in range(n_layers - 1):
            x = jnp.dot(x, params[f'w{i}']) + params[f'b{i}']
            x = jax.nn.relu(x)
        x = jnp.dot(x, params[f'w{n_layers-1}']) + params[f'b{n_layers-1}']
        return x

    @staticmethod
    @jax.jit
    def train_step(params, x, y, lr=0.01):
        def loss_fn(params):
            pred = JaxMLP.predict(params, x)
            return jnp.mean((pred - y) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        new_params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
        return new_params, loss

# Usage
key = jax.random.key(42)
model = JaxMLP([10, 64, 32, 1], key)

x = jax.random.normal(key, (100, 10))
y = jnp.sum(x[:, :3], axis=-1, keepdims=True)

for epoch in range(200):
    model.params, loss = JaxMLP.train_step(model.params, x, y)
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: loss = {loss:.4f}")
```

---

## 40.6 Performance Patterns

### 40.6.1 Minimize JIT Recompilation

```python
# BAD: Different shapes trigger recompilation
@jax.jit
def process(x):
    return jnp.sum(x)

process(jnp.ones(10))   # Compiles for shape (10,)
process(jnp.ones(20))   # Recompiles for shape (20,)

# GOOD: Use padded/batched shapes, or static_argnums
@jax.jit
def process(x):
    return jnp.sum(x)

# Or use jax.jit with static arguments for non-array inputs
@jax.jit(static_argnums=(1,))
def process_with_size(x, size):
    return x[:size]
```

### 40.6.2 Use block_until_ready for Timing

```python
import time

# BAD: Doesn't wait for async computation
@jax.jit
def compute(x):
    return jnp.dot(x, x.T)

start = time.perf_counter()
result = compute(jnp.ones((1000, 1000)))
elapsed = time.perf_counter() - start  # Only measures dispatch time!

# GOOD: Wait for computation to complete
start = time.perf_counter()
result = compute(jnp.ones((1000, 1000)))
result.block_until_ready()  # Wait for actual computation
elapsed = time.perf_counter() - start  # Now measures actual time
```

### 40.6.3 Batch Operations

```python
# BAD: Process one at a time
def process_single(model, x):
    return jnp.dot(x, model)

results = [process_single(w, x_i) for x_i in x_batch]  # Slow

# GOOD: Batch process
def process_batch(model, x_batch):
    return jnp.dot(x_batch, model)  # Single matrix multiply

results = process_batch(w, x_batch)  # Fast
```

### 40.6.4 Avoid Python Loops Over Traced Values

```python
# BAD: Python loop over traced range
@jax.jit
def bad_loop(x, n):
    for i in range(n):  # n must be static (compile-time constant)
        x = x + 1
    return x

# GOOD: Use jax.lax for traced loop bounds
@jax.jit
def good_loop(x, n):
    return jax.lax.fori_loop(0, n, lambda i, x: x + 1, x)
```

---

## 40.7 Debugging Patterns

### 40.7.1 Inspecting Intermediate Values

```python
# Method 1: jax.debug.print
@jax.jit
def debug_fn(x):
    y = x * 2
    jax.debug.print("y = {}, shape = {}", y, y.shape)
    z = y + 1
    return z

# Method 2: jax.debug.breakpoint (interactive debugger)
@jax.jit
def interactive_debug(x):
    y = x * 2
    jax.debug.breakpoint()  # Drops into interactive debugger
    z = y + 1
    return z

# Method 3: Disable JIT for debugging
@jax.jit
def debuggable(x):
    # Temporarily disable JIT to use print/pdb
    return x + 1

# With JAX_DISABLE_JIT=True, runs eagerly with Python debugging
```

### 40.7.2 Checking Gradients

```python
# Numerical gradient checking
def check_grad(fn, x, eps=1e-4):
    """Compare JAX gradient with numerical gradient."""
    jax_grad = jax.grad(fn)(x)

    numerical_grad = jnp.zeros_like(x)
    for i in range(x.size):
        x_flat = x.flatten()
        x_plus = x_flat.at[i].add(eps).reshape(x.shape)
        x_minus = x_flat.at[i].add(-eps).reshape(x.shape)
        numerical_grad = numerical_grad.at[i].set(
            (fn(x_plus) - fn(x_minus)) / (2 * eps)
        )

    max_diff = jnp.max(jnp.abs(jax_grad - numerical_grad))
    print(f"Max gradient difference: {max_diff:.6e}")
    return max_diff < 1e-3

# Usage
def test_fn(x):
    return jnp.sum(x ** 2)

check_grad(test_fn, jnp.array([1.0, 2.0, 3.0]))
```

---

## 40.8 Summary of Gotchas

| Gotcha | Symptom | Solution |
|---|---|---|
| In-place mutation | TypeError | Use `.at[].set()` |
| Python random | Same value each call | Use `jax.random` |
| Python if/for with traced values | ConcretizationError | Use `jax.lax.cond`, `jax.lax.scan` |
| Data-dependent shapes | Shape error at trace time | Use fixed shapes, masks |
| Float64 disabled | Silently downcast to float32 | Set `JAX_ENABLE_X64=True` |
| Print inside JIT | Only prints during tracing | Use `jax.debug.print` |
| Not blocking for timing | Incorrect timing | Use `.block_until_ready()` |
| Donated buffers | Use-after-free | Don't use donated arguments |
| Non-differentiable ops | Zero/NaN gradients | Use differentiable approximations |
| Recompilation | Slow first call per shape | Batch inputs, use consistent shapes |
