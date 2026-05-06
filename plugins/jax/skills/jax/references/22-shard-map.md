# shard_map Reference

This reference provides comprehensive documentation of `jax.shard_map`, JAX's low-level API for writing per-device (SPMD) parallel programs. Unlike the automatic sharding approach where the compiler decides communication, `shard_map` gives you explicit control over what each device computes and when collective communication happens.

---

## 22.1 Overview

### 22.1.1 What is shard_map?

`shard_map` is a transformation that lets you write code from the perspective of a single device. Inside a `shard_map` function, each device sees only its local shard of the data. You must explicitly handle communication between devices using collective operations like `jax.lax.psum`, `jax.lax.all_gather`, and `jax.lax.ppermute`.

```
+------------------------------------------------------------------+
|               shard_map Programming Model                         |
|                                                                   |
|  Global View (outside shard_map)                                  |
|    x: jax.Array of shape (1024, 512)                              |
|    sharded across 4 devices as P('data', None)                    |
|                                                                   |
|  Local View (inside shard_map)                                    |
|    Each device sees: x_local of shape (256, 512)                  |
|    Device 0: rows 0-255                                           |
|    Device 1: rows 256-511                                         |
|    Device 2: rows 512-767                                         |
|    Device 3: rows 768-1023                                        |
|                                                                   |
|  The function body operates on local shards only.                 |
|  Communication via explicit collectives.                          |
+------------------------------------------------------------------+
```

### 22.1.2 shard_map vs pmap vs pjit

| Feature | `shard_map` | `pmap` (legacy) | GSPMD (pjit/auto) |
|---------|-------------|-----------------|-------------------|
| Communication | Explicit collectives | Explicit collectives | Automatic (compiler-inserted) |
| Programming model | Per-device view | Per-device view | Global view |
| Nesting | Composable | Limited | N/A |
| Control flow | Full | Limited | Full |
| Debugging | Straightforward | Moderate | Hard (compiler decisions) |
| Performance | Predictable | Moderate | Best (compiler-optimized) |
| Recommended | Yes (for manual control) | No (deprecated) | Yes (for ease of use) |

### 22.1.3 When to Use shard_map

- You need fine-grained control over when and how communication happens
- You want predictable, debuggable parallel programs
- You are implementing custom parallelism strategies (e.g., sequence parallelism, expert parallelism)
- You need to reason about per-device memory usage
- You want to overlap computation and communication

---

## 22.2 API Reference

### 22.2.1 jax.shard_map.shard_map

```python
from jax.shard_map import shard_map

def shard_map(
    fun,           # Function to apply per-device
    mesh,          # Mesh defining device arrangement
    in_specs,      # PartitionSpec(s) for input(s)
    out_specs,     # PartitionSpec(s) for output(s)
    check_rep=True # Whether to check output replication
):
    """Apply a function per-device on sharded data."""
```

Parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `fun` | Callable | The function to apply on each device's local shard |
| `mesh` | Mesh | Device mesh defining the parallelism layout |
| `in_specs` | PartitionSpec or tuple of PartitionSpec | How inputs are sharded. Determines what each device sees. |
| `out_specs` | PartitionSpec or tuple of PartitionSpec | How outputs should be reassembled into global arrays. |
| `check_rep` | bool | If True, verifies that replicated outputs are consistent across devices. |

### 22.2.2 Basic Example

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.shard_map import shard_map
import numpy as np

# Create a mesh
devices = jax.devices()
mesh = Mesh(np.array(devices), ('devices',))

# Simple element-wise operation
@shard_map(
    mesh,
    in_specs=P('devices',),      # 1D array sharded across devices
    out_specs=P('devices',),      # Output also sharded across devices
)
def double(x):
    # x is the local shard on each device
    # No communication needed for element-wise ops
    return x * 2

# Create input
x = jnp.arange(16.0)
y = double(x)
print(y)  # [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
```

### 22.2.3 Multiple Inputs and Outputs

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.shard_map import shard_map
import numpy as np

devices = jax.devices()
mesh = Mesh(np.array(devices).reshape(2, 2), ('data', 'model'))

@shard_map(
    mesh,
    in_specs=(P('data', None), P(None, 'model')),  # x sharded on dim0, y on dim1
    out_specs=P('data', 'model'),                     # output sharded on both
)
def matmul_per_device(x_local, y_local):
    # Each device computes a partial matmul with its local shards
    return x_local @ y_local

x = jnp.ones((1024, 512))
y = jnp.ones((512, 256))
result = matmul_per_device(x, y)
print(result.shape)  # (1024, 256)
```

Multiple outputs:

```python
@shard_map(
    mesh,
    in_specs=P('data', None),
    out_specs=(P('data', None), P('data', None)),  # Two outputs, both sharded
)
def split_and_process(x):
    return x ** 2, x ** 3

x = jnp.arange(16.0)
squares, cubes = split_and_process(x)
```

---

## 22.3 Rank-Preserving vs Rank-Reducing

### 22.3.1 Rank-Preserving (Default)

By default, `shard_map` preserves the rank of the input. Each device sees a shard of the same rank:

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.shard_map import shard_map
import numpy as np

devices = jax.devices()
mesh = Mesh(np.array(devices), ('data',))

@shard_map(
    mesh,
    in_specs=P('data', None),   # 2D input, sharded on dim 0
    out_specs=P('data', None),   # 2D output, sharded on dim 0
)
def rank_preserving(x):
    # x has the same rank as the input, just smaller in the sharded dimension
    # Input global shape: (1024, 512)
    # x local shape: (256, 512) per device (with 4 devices)
    print(f"Local shape: {x.shape}")  # (256, 512)
    return x + 1.0

x = jnp.ones((1024, 512))
result = rank_preserving(x)
print(f"Output shape: {result.shape}")  # (1024, 512)
```

### 22.3.2 Rank-Reducing

When an axis is completely consumed by sharding, the local view has a reduced rank. This happens when a dimension is fully sharded and the local shard has size 1 (or when using `None` in `out_specs`):

```python
@shard_map(
    mesh,
    in_specs=P('data',),       # 1D input, fully sharded
    out_specs=P(),              # 0D output per device -> gathered into 1D
)
def rank_reducing(x):
    # x is a scalar on each device (1D global -> 0D local)
    return jnp.sum(x)  # scalar

x = jnp.arange(16.0)
result = rank_reducing(x)
print(f"Output shape: {result.shape}")  # (4,) -- one value per device
```

### 22.3.3 Controlling Rank with PartitionSpec

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.shard_map import shard_map
import numpy as np

devices = jax.devices()
mesh = Mesh(np.array(devices).reshape(2, 2), ('data', 'model'))

# Rank-preserving: both dimensions sharded
@shard_map(
    mesh,
    in_specs=P('data', 'model'),
    out_specs=P('data', 'model'),
)
def both_dims_sharded(x):
    # Input: (1024, 512) -> Local: (512, 256) with 2x2 mesh
    return x * 2

# Partial sharding: only one dimension
@shard_map(
    mesh,
    in_specs=P('data', None),
    out_specs=P('data', None),
)
def one_dim_sharded(x):
    # Input: (1024, 512) -> Local: (512, 512) with 2 devices on 'data'
    return x * 2

# Fully replicated: no sharding
@shard_map(
    mesh,
    in_specs=P(None, None),
    out_specs=P(None, None),
)
def no_sharding(x):
    # Input: (1024, 512) -> Local: (1024, 512) on every device
    return x * 2
```

---

## 22.4 Collective Operations Inside shard_map

The power of `shard_map` comes from explicit collective communication. These operations coordinate across devices.

### 22.4.1 jax.lax.psum (All-Reduce Sum)

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.shard_map import shard_map
import numpy as np

devices = jax.devices()
mesh = Mesh(np.array(devices), ('data',))

@shard_map(
    mesh,
    in_specs=P('data',),       # 1D array sharded across devices
    out_specs=P(),               # Scalar per device -> 1D result
)
def global_sum(x_local):
    # x_local is a scalar on each device
    # psum sums across all devices along mesh axis 'data'
    return jax.lax.psum(x_local, 'data')

x = jnp.arange(16.0)  # sum = 120.0
result = global_sum(x)
print(result)  # [120. 120. 120. 120.]  -- replicated result

# If you want the result on each device as a scalar (not reassembled):
@shard_map(
    mesh,
    in_specs=P('data',),
    out_specs=P('data',),      # Keep the 'data' axis
)
def per_device_sum(x_local):
    # This sums over 'data' and broadcasts back
    total = jax.lax.psum(x_local, 'data')
    return jnp.broadcast_to(total, x_local.shape)

result2 = per_device_sum(x)
print(result2)  # Each device's shard has the full sum
```

#### psum with Multi-Dimensional Arrays

```python
devices = jax.devices()
mesh = Mesh(np.array(devices).reshape(2, 2), ('data', 'model'))

@shard_map(
    mesh,
    in_specs=P('data', None),
    out_specs=P('data', None),
)
def sum_over_model_axis(x_local):
    # x_local: (batch_shard, features)
    # Sum along 'model' axis (reduce across model-parallel devices)
    return jax.lax.psum(x_local, 'model')

x = jnp.ones((1024, 512))
result = sum_over_model_axis(x)
```

### 22.4.2 jax.lax.psum_scatter (Reduce-Scatter)

`psum_scatter` is a memory-efficient alternative to `psum` followed by sharding. It reduces across devices and each device gets a unique shard of the result:

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.shard_map import shard_map
import numpy as np

devices = jax.devices()
mesh = Mesh(np.array(devices), ('data',))

@shard_map(
    mesh,
    in_specs=P('data',),         # Each device has one element
    out_specs=P('data',),         # Each device gets one shard of the sum
)
def reduce_scatter_example(x_local):
    # x_local is a scalar
    # psum_scatter: sum all values, then each device gets a unique shard
    # For scalar input, this is like psum (each device gets the total)
    total = jax.lax.psum(x_local, 'data')
    return total

x = jnp.arange(16.0)
result = reduce_scatter_example(x)
print(result)
```

More practical example with vectors:

```python
mesh = Mesh(np.array(jax.devices()), ('data',))

@shard_map(
    mesh,
    in_specs=P('data', None),      # Each device has (chunk_size, features)
    out_specs=P('data',),           # Output: each device gets one value
)
def reduce_scatter_vector(x_local):
    # x_local: (chunk_size, features)
    # Sum across devices, then scatter the result
    # Each device ends up with a different slice of the summed vector
    per_device_sum = jnp.sum(x_local, axis=0)  # (features,)
    return jax.lax.psum_scatter(per_device_sum, 'data', scatter_dimension=0)

x = jnp.ones((16, 4))
result = reduce_scatter_vector(x)
```

### 22.4.3 jax.lax.all_gather

`all_gather` collects shards from all devices and concatenates them:

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.shard_map import shard_map
import numpy as np

devices = jax.devices()
mesh = Mesh(np.array(devices), ('data',))

@shard_map(
    mesh,
    in_specs=P('data',),         # Each device has one element
    out_specs=P(),                # Output: full array on each device
)
def gather_all(x_local):
    # Each device sees its local scalar
    # all_gather collects all scalars into a 1D array on each device
    full = jax.lax.all_gather(x_local, 'data')
    # full is shape (4,) on each device -- the reassembled array
    return full

x = jnp.arange(16.0)
result = gather_all(x)
print(result)  # Same value on all devices: the full array
```

#### all_gather with Multi-Dimensional Data

```python
@shard_map(
    mesh,
    in_specs=P('data', None),    # Each device has (chunk, features)
    out_specs=P(None, None),     # Full array replicated on each device
)
def gather_2d(x_local):
    # x_local: (chunk_size, features)
    # Gather along 'data' axis -> (full_batch, features) on each device
    return jax.lax.all_gather(x_local, 'data', axis=0)

x = jnp.arange(32.0).reshape(8, 4)
result = gather_2d(x)
print(f"Local shape: {result.shape}")  # (8, 4) on each device
```

### 22.4.4 jax.lax.ppermute

`ppermute` performs a collective permutation: each device sends its data to another device according to a permutation:

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.shard_map import shard_map
import numpy as np

devices = jax.devices()
mesh = Mesh(np.array(devices), ('data',))

# Define a rotation permutation: each device sends to the next device
num_devices = len(devices)
permutation = [(i, (i + 1) % num_devices) for i in range(num_devices)]
# With 4 devices: [(0,1), (1,2), (2,3), (3,0)]

@shard_map(
    mesh,
    in_specs=P('data',),
    out_specs=P('data',),
)
def rotate_right(x_local):
    # Each device sends its value to the next device
    return jax.lax.ppermute(x_local, 'data', perm=permutation)

x = jnp.arange(16.0)
result = rotate_right(x)
# Each device's value has been shifted to the next device
```

#### Ring Communication with ppermute

```python
# Ring-style all-reduce using ppermute
@shard_map(
    mesh,
    in_specs=P('data',),
    out_specs=P('data',),
)
def ring_sum(x_local):
    # Perform a ring reduction: accumulate partial sums
    result = x_local
    for step in range(num_devices - 1):
        # Send current result to next device, receive from previous
        received = jax.lax.ppermute(
            result, 'data',
            perm=[(i, (i + 1) % num_devices) for i in range(num_devices)]
        )
        result = result + received
    return result

x = jnp.arange(16.0)
result = ring_sum(x)
```

### 22.4.5 jax.lax.broadcast

```python
@shard_map(
    mesh,
    in_specs=P('data',),
    out_specs=P('data',),
)
def broadcast_example(x_local):
    # Broadcast a value from one device to all devices
    # axis_name: mesh axis to broadcast along
    # source: which position in the mesh axis is the source
    return jax.lax.broadcast(x_local, 'data', source=0)
```

### 22.4.6 Collective Summary Table

| Operation | Description | Communication |
|-----------|-------------|---------------|
| `psum(x, axis)` | Sum across devices | All-reduce (sum) |
| `pmean(x, axis)` | Mean across devices | All-reduce (sum) + divide |
| `pmax(x, axis)` | Max across devices | All-reduce (max) |
| `pmin(x, axis)` | Min across devices | All-reduce (min) |
| `psum_scatter(x, axis)` | Sum + scatter unique shards | Reduce-scatter |
| `all_gather(x, axis)` | Gather all shards | All-gather |
| `ppermute(x, axis, perm)` | Permute across devices | Point-to-point |
| `broadcast(x, axis, source)` | Broadcast from source | Broadcast |
| `all_to_all(x, axis)` | All-to-all exchange | All-to-all |

---

## 22.5 Composing shard_map with jit and grad

### 22.5.1 shard_map with jax.jit

`shard_map` composes naturally with `jax.jit`. In fact, `shard_map` is typically used inside a JIT-compiled function:

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.shard_map import shard_map
import numpy as np

devices = jax.devices()
mesh = Mesh(np.array(devices), ('data',))

@jax.jit
def compiled_sharded(x):
    @shard_map(mesh, in_specs=P('data',), out_specs=P('data',))
    def f(x_local):
        return x_local ** 2 + 1
    return f(x)

x = jnp.arange(16.0)
result = compiled_sharded(x)
```

You can also JIT the entire shard_map:

```python
@shard_map(mesh, in_specs=P('data',), out_specs=P('data',))
def f(x_local):
    return x_local ** 2 + 1

# shard_map functions are automatically JIT-compiled when called
result = f(jnp.arange(16.0))
```

### 22.5.2 shard_map with jax.grad

`shard_map` is differentiable. The gradient computation preserves the sharding structure:

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.shard_map import shard_map
import numpy as np

devices = jax.devices()
mesh = Mesh(np.array(devices), ('data',))

@shard_map(
    mesh,
    in_specs=(P('data', None), P(None, None)),  # x sharded, w replicated
    out_specs=P('data',),                        # loss per shard
)
def sharded_loss(x_local, w):
    # Each device computes loss on its local shard
    pred = x_local @ w
    return jnp.sum(pred ** 2)

# Compute gradients
def total_loss(x, w):
    # Sum the per-shard losses
    per_shard_loss = sharded_loss(x, w)
    return jnp.sum(per_shard_loss)

x = jnp.ones((1024, 128))
w = jnp.ones((128, 64))

# Gradient w.r.t. w
grad_fn = jax.grad(total_loss, argnums=1)
grads = grad_fn(x, w)
print(f"Gradient shape: {grads.shape}")  # (128, 64)
```

### 22.5.3 shard_map with jax.value_and_grad

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.shard_map import shard_map
import numpy as np
import optax

devices = jax.devices()
mesh = Mesh(np.array(devices), ('data',))

def init_params(key, dims):
    return jax.random.normal(key, dims) * 0.01

@shard_map(
    mesh,
    in_specs=(P('data', None), P(None, None)),
    out_specs=(P('data',), P(None, None)),
)
def sharded_forward(x_local, w):
    pred = x_local @ w
    loss = jnp.sum(pred ** 2)
    return loss, pred

@jax.jit
def train_step(x, w, opt_state):
    def loss_fn(w):
        per_shard_loss, _ = sharded_forward(x, w)
        return jnp.sum(per_shard_loss)

    loss, grads = jax.value_and_grad(loss_fn)(w)
    updates, opt_state = optax.adam(1e-3).update(grads, opt_state, w)
    w = optax.apply_updates(w, updates)
    return w, opt_state, loss

key = jax.random.PRNGKey(0)
w = init_params(key, (128, 64))
x = jnp.ones((1024, 128))
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(w)

w, opt_state, loss = train_step(x, w, opt_state)
print(f"Loss: {loss:.4f}")
```

---

## 22.6 Transposing shard_map (JEP 17111)

JAX supports transposing through `shard_map` (JEP 17111), which enables efficient reverse-mode differentiation. The transpose rule ensures that gradient computations maintain proper sharding:

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.shard_map import shard_map
import numpy as np

devices = jax.devices()
mesh = Mesh(np.array(devices), ('data',))

# Linear operation inside shard_map
@shard_map(
    mesh,
    in_specs=(P('data', None), P(None, None)),
    out_specs=P('data', None),
)
def sharded_linear(x_local, w):
    return x_local @ w

# The transpose (adjoint) of this operation is handled automatically
# by JAX's transpose machinery when computing gradients
x = jnp.ones((1024, 128))
w = jnp.ones((128, 64))

# Forward pass
y = sharded_linear(x, w)
print(f"Forward output shape: {y.shape}")  # (1024, 64)

# Backward pass (gradient)
# The transpose rule ensures gradients flow correctly through shard_map
def loss_fn(w):
    y = sharded_linear(x, w)
    return jnp.sum(y ** 2)

grad_w = jax.grad(loss_fn)(w)
print(f"Gradient shape: {grad_w.shape}")  # (128, 64)
```

### 22.6.1 How Transpose Works

When JAX transposes through `shard_map`:

1. The forward pass records how data flows through the per-device computation.
2. The backward pass applies the transpose of each operation on the same devices.
3. Collective operations are transposed appropriately:
   - `psum` transposes to a `broadcast` (replication)
   - `all_gather` transposes to `psum_scatter`
   - `psum_scatter` transposes to `all_gather`

```python
# Example showing transposition of collectives
@shard_map(
    mesh,
    in_specs=P('data', None),
    out_specs=P(None, None),  # Output is replicated
)
def forward_with_collective(x_local):
    # Sum across devices -> replicated result
    return jax.lax.psum(x_local, 'data')

# In the backward pass:
# - The cotangent is replicated (same as output)
# - Transpose of psum is broadcast -> each device gets the cotangent
# - Then the per-device transpose of the local function is applied

x = jnp.ones((1024, 128))
grad_x = jax.grad(lambda x: jnp.sum(forward_with_collective(x)))(x)
```

---

## 22.7 Parallelism Patterns with shard_map

### 22.7.1 Data Parallelism

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.shard_map import shard_map
import numpy as np

devices = jax.devices()
mesh = Mesh(np.array(devices), ('data',))

def init_params(key, input_dim, hidden_dim, output_dim):
    k1, k2 = jax.random.split(key)
    return {
        'w1': jax.random.normal(k1, (input_dim, hidden_dim)) * 0.01,
        'w2': jax.random.normal(k2, (hidden_dim, output_dim)) * 0.01,
    }

@shard_map(
    mesh,
    in_specs=(P('data', None), P(None, None)),
    out_specs=P('data',),
)
def data_parallel_loss(x_local, params):
    # Each device computes loss on its local data shard
    h = jax.nn.relu(x_local @ params['w1'])
    pred = h @ params['w2']
    return jnp.sum(pred ** 2)

@jax.jit
def train_step(params, x, opt_state):
    def total_loss(params):
        per_shard = data_parallel_loss(x, params)
        return jnp.sum(per_shard) / x.shape[0]  # Normalize by total batch

    loss, grads = jax.value_and_grad(total_loss)(params)
    # Gradients are automatically aggregated across devices
    updates, opt_state = optax.adam(1e-3).update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# Initialize
key = jax.random.PRNGKey(0)
params = init_params(key, 128, 64, 10)
import optax
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

# Training
x = jax.random.normal(key, (1024, 128))
params, opt_state, loss = train_step(params, x, opt_state)
print(f"Data parallel loss: {loss:.4f}")
```

### 22.7.2 Tensor (Model) Parallelism

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.shard_map import shard_map
import numpy as np

devices = jax.devices()
mesh = Mesh(np.array(devices), ('model',))

# Column-parallel linear layer
@shard_map(
    mesh,
    in_specs=(P(None, None), P(None, 'model')),   # x replicated, w sharded on dim1
    out_specs=P(None, 'model'),                      # output sharded on dim1
)
def column_parallel(x, w_shard):
    # Each device computes: x @ w_shard
    return x @ w_shard

# Row-parallel linear layer (follows column-parallel)
@shard_map(
    mesh,
    in_specs=(P(None, 'model'), P('model', None)),  # x sharded on dim1, w sharded on dim0
    out_specs=P(None, None),                          # output replicated
)
def row_parallel(x_shard, w_shard):
    # Each device computes partial matmul
    partial = x_shard @ w_shard  # (batch, features)
    # Sum across devices to get full result
    return jax.lax.psum(partial, 'model')

# Combined: Megatron-style transformer layer
def megatron_linear(x, w1, w2):
    """Column-parallel then row-parallel, with activation in between."""
    h = column_parallel(x, w1)
    h = jax.nn.relu(h)
    out = row_parallel(h, w2)
    return out

# Usage
key = jax.random.PRNGKey(0)
batch, feat, hidden = 32, 128, 256

# Shard weight matrices for tensor parallelism
w1 = jax.random.normal(key, (feat, hidden))
w2 = jax.random.normal(jax.random.fold_in(key, 1), (hidden, feat))
x = jax.random.normal(jax.random.fold_in(key, 2), (batch, feat))

# Place with appropriate shardings
from jax.sharding import NamedSharding
w1_sharding = NamedSharding(mesh, P(None, 'model'))
w2_sharding = NamedSharding(mesh, P('model', None))

result = megatron_linear(x, w1, w2)
print(f"Tensor parallel result shape: {result.shape}")
```

### 22.7.3 FSDP (Fully Sharded Data Parallelism)

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.shard_map import shard_map
from jax.sharding import NamedSharding
import numpy as np
import optax

devices = jax.devices()
mesh = Mesh(np.array(devices), ('fsdp',))

# FSDP: shard parameters along the FSDP axis
# All-gather before compute, reduce-scatter after

@shard_map(
    mesh,
    in_specs=(P('data', None), P('fsdp', None)),  # x: sharded on data, w: sharded on fsdp
    out_specs=P('data', None),
)
def fsdp_linear(x_shard, w_shard):
    # All-gather the weight along fsdp axis
    w_full = jax.lax.all_gather(w_shard, 'fsdp', axis=0)

    # Compute with full weight
    y = x_shard @ w_full

    # Could reduce-scatter the output for memory efficiency
    # y = jax.lax.psum_scatter(y, 'fsdp')
    return y

# More realistic FSDP example with a full training step
@shard_map(
    mesh,
    in_specs=(P('fsdp', None), P('fsdp',)),  # w1: (shard, features), w2: (shard,)
    out_specs=P('fsdp',),                      # loss per shard
)
def fsdp_layer_loss(w1_shard, b1_shard, x_shard):
    # All-gather the parameter shard to reconstruct full parameter
    w1_full = jax.lax.all_gather(w1_shard, 'fsdp', axis=0)
    b1_full = jax.lax.all_gather(b1_shard, 'fsdp', axis=0)

    # Compute with full parameters
    h = x_shard @ w1_full + b1_full
    h = jax.nn.relu(h)

    # Loss
    return jnp.mean(h ** 2)

@jax.jit
def fsdp_train_step(w1, b1, x, opt_state):
    def loss_fn(w1, b1):
        per_shard_loss = fsdp_layer_loss(w1, b1, x)
        return jnp.sum(per_shard_loss)

    loss, grads = jax.value_and_grad(loss_fn, argnums=(0, 1))(w1, b1)
    grads = jax.tree.map(lambda g: g / mesh.size, grads)  # Normalize
    updates, opt_state = optax.adam(1e-3).update(grads, opt_state, (w1, b1))
    w1, b1 = optax.apply_updates((w1, b1), updates)
    return w1, b1, opt_state, loss
```

### 22.7.4 Sequence Parallelism

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.shard_map import shard_map
import numpy as np

devices = jax.devices()
mesh = Mesh(np.array(devices), ('seq',))

@shard_map(
    mesh,
    in_specs=(P('seq', None), P(None, None)),
    out_specs=P('seq', None),
)
def sequence_parallel_layer_norm(x_shard, gamma):
    """Layer norm with sequence-parallel input.

    x_shard: (seq_shard, hidden) - local sequence chunk
    gamma: (hidden,) - replicated scale parameter
    """
    # For layer norm, we need global statistics (mean, var over all seq positions)
    # Local sum and sum-of-squares
    local_sum = jnp.sum(x_shard, axis=0)           # (hidden,)
    local_sq_sum = jnp.sum(x_shard ** 2, axis=0)   # (hidden,)

    # Global sum via all-reduce
    global_sum = jax.lax.psum(local_sum, 'seq')
    global_sq_sum = jax.lax.psum(local_sq_sum, 'seq')

    # Compute global mean and variance
    seq_len_global = x_shard.shape[0] * mesh.size
    mean = global_sum / seq_len_global
    var = global_sq_sum / seq_len_global - mean ** 2

    # Normalize with global statistics
    x_norm = (x_shard - mean) / jnp.sqrt(var + 1e-5)
    return x_norm * gamma

# Usage
seq_len, hidden = 1024, 768
x = jax.random.normal(jax.random.PRNGKey(0), (seq_len, hidden))
gamma = jnp.ones(hidden)

result = sequence_parallel_layer_norm(x, gamma)
print(f"Sequence parallel result shape: {result.shape}")
```

### 22.7.5 Pipeline Parallelism Simulation

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.shard_map import shard_map
import numpy as np

devices = jax.devices()
mesh = Mesh(np.array(devices), ('stage',))

# Simulate pipeline parallelism with sequential stages
@shard_map(
    mesh,
    in_specs=P(None, None),
    out_specs=P(None, None),
)
def pipeline_forward(x, w1, w2, w3):
    # Each "device" runs all stages in this simplified example
    # Real pipeline parallelism would use micro-batches and scheduling
    h1 = jax.nn.relu(x @ w1)
    h2 = jax.nn.relu(h1 @ w2)
    out = h2 @ w3
    return out

# In practice, pipeline parallelism in JAX is typically implemented
# using sequential computation with gradient checkpointing,
# rather than through shard_map directly
```

---

## 22.8 Debugging shard_map Programs

### 22.8.1 Printing from Inside shard_map

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.shard_map import shard_map
import numpy as np

devices = jax.devices()
mesh = Mesh(np.array(devices), ('data',))

@shard_map(
    mesh,
    in_specs=P('data',),
    out_specs=P('data',),
)
def debug_shard(x_local):
    # Use jax.debug.print to inspect per-device values
    jax.debug.print("Device {} sees local shape {} value {}",
                     jax.lax.axis_index('data'), x_local.shape, x_local)
    return x_local * 2

x = jnp.arange(16.0)
result = debug_shard(x)
# Output (one line per device):
# Device 0 sees local shape (4,) value [0. 1. 2. 3.]
# Device 1 sees local shape (4,) value [4. 5. 6. 7.]
# ...
```

### 22.8.2 Checking Sharding Consistency

```python
@shard_map(
    mesh,
    in_specs=P('data', None),
    out_specs=P('data', None),
    check_rep=True,  # Verify replicated outputs are consistent
)
def checked_fn(x_local):
    # If this function were to produce inconsistent replicated outputs
    # (i.e., different values on different devices for a replicated axis),
    # check_rep=True would raise an error
    return x_local + 1.0
```

### 22.8.3 Visualizing Data Flow

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.shard_map import shard_map
import numpy as np

devices = jax.devices()
mesh = Mesh(np.array(devices), ('data',))

# Step 1: Check input sharding
x = jnp.arange(16.0)
print(f"Input sharding: {x.sharding}")

# Step 2: Apply shard_map
@shard_map(
    mesh,
    in_specs=P('data',),
    out_specs=P('data',),
)
def step_with_debug(x_local):
    jax.debug.print("Input local: shape={}, values={}", x_local.shape, x_local)
    y = x_local ** 2
    jax.debug.print("After square: shape={}, values={}", y.shape, y)
    return y

y = step_with_debug(x)

# Step 3: Check output sharding
print(f"Output sharding: {y.sharding}")
print(f"Output shape: {y.shape}")
print(f"Output values: {y}")
```

### 22.8.4 Common Errors and Fixes

```python
# ERROR 1: Missing collective for reduction
# Problem: Trying to return a reduced value without proper communication
@shard_map(
    mesh,
    in_specs=P('data',),
    out_specs=P(),  # Expects scalar per device
)
def bad_sum(x_local):
    return jnp.sum(x_local)  # Each device sums its local shard
    # This gives LOCAL sums, not the GLOBAL sum!
    # The output will be [local_sum_0, local_sum_1, ...], not the total

# FIX: Use psum for global reduction
@shard_map(
    mesh,
    in_specs=P('data',),
    out_specs=P(),
)
def correct_sum(x_local):
    return jax.lax.psum(jnp.sum(x_local), 'data')  # Now all devices agree

# ERROR 2: Mismatched out_specs
# Problem: Output spec doesn't match actual output rank
@shard_map(
    mesh,
    in_specs=P('data', None),
    out_specs=P('data',),  # Expects 1D output
)
def bad_out_spec(x_local):
    return x_local @ jnp.ones((x_local.shape[1],))  # Returns 1D, but...

# FIX: Match out_specs to actual output
@shard_map(
    mesh,
    in_specs=P('data', None),
    out_specs=P('data',),  # OK if output is 1D
)
def correct_out_spec(x_local):
    return jnp.sum(x_local, axis=1)  # Returns 1D array
```

### 22.8.5 Performance Debugging

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.shard_map import shard_map
import numpy as np
import time

devices = jax.devices()
mesh = Mesh(np.array(devices), ('data',))

@shard_map(
    mesh,
    in_specs=P('data', None),
    out_specs=P('data', None),
)
def compute_heavy(x_local):
    # Simulate compute-heavy per-device work
    for _ in range(10):
        x_local = x_local @ jnp.eye(x_local.shape[1])
    return x_local

x = jax.device_put(jnp.ones((4096, 512)), jax.sharding.NamedSharding(mesh, P('data', None)))

# Warmup
_ = compute_heavy(x).block_until_ready()

# Time it
start = time.perf_counter()
for _ in range(100):
    result = compute_heavy(x)
result.block_until_ready()
elapsed = time.perf_counter() - start
print(f"Average time: {elapsed / 100 * 1000:.2f} ms")

# Check HLO to see communication patterns
# jax.jit(compute_heavy).lower(x).as_text()  # View the HLO IR
```

---

## 22.9 Advanced Patterns

### 22.9.1 Nested shard_map

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.shard_map import shard_map
import numpy as np

devices = jax.devices()
mesh_2d = Mesh(np.array(devices).reshape(2, 2), ('data', 'model'))

# Outer shard_map: data parallelism
@shard_map(
    mesh_2d,
    in_specs=P('data', None),
    out_specs=P('data', None),
)
def outer_data_parallel(x_shard):
    # x_shard: local data shard

    # Inner shard_map: model parallelism within the data shard
    mesh_model = Mesh(
        np.array(list(mesh_2d.devices)).reshape(1, -1),
        ('model_inner',)
    )

    @shard_map(
        mesh_model,
        in_specs=P(None, 'model_inner'),
        out_specs=P(None, None),
    )
    def model_parallel(w_shard):
        return jax.lax.psum(x_shard @ w_shard, 'model_inner')

    # This is conceptual; in practice, use a single 2D mesh with
    # appropriate in_specs and collectives
    return x_shard
```

### 22.9.2 Overlapping Communication and Computation

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.shard_map import shard_map
import numpy as np

devices = jax.devices()
mesh = Mesh(np.array(devices), ('data',))

@shard_map(
    mesh,
    in_specs=(P('data', None), P('data', None)),
    out_specs=P('data', None),
)
def overlap_comm_compute(x_local, w_local):
    # Strategy: while communicating, compute on local data
    # Step 1: Start communication (all-gather w)
    w_full = jax.lax.all_gather(w_local, 'data', axis=0)

    # Step 2: Compute local result
    local_result = x_local @ w_full

    return local_result

# Note: Actual overlapping depends on the XLA compiler's scheduling.
# The compiler may or may not overlap these operations depending on
# the target hardware and optimization level.
```

### 22.9.3 Conditional Collectives

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.shard_map import shard_map
import numpy as np

devices = jax.devices()
mesh = Mesh(np.array(devices), ('data',))

@shard_map(
    mesh,
    in_specs=P('data', None),
    out_specs=P('data', None),
)
def conditional_collective(x_local):
    # Use jax.lax.cond for conditional communication
    # All devices must agree on the condition!
    should_communicate = jnp.all(x_local > 0)

    def do_reduce(x):
        return jax.lax.psum(x, 'data') / mesh.size

    def skip_reduce(x):
        return x

    # IMPORTANT: In practice, all devices must take the same branch
    # for collective operations to work correctly.
    # This pattern is mainly useful for conditional computation
    # where the condition is the same on all devices.
    return jax.lax.cond(should_communicate, do_reduce, skip_reduce, x_local)
```

### 22.9.4 Custom Collective Patterns

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.shard_map import shard_map
import numpy as np

devices = jax.devices()
mesh = Mesh(np.array(devices), ('data',))

@shard_map(
    mesh,
    in_specs=P('data',),
    out_specs=P('data',),
)
def custom_all_reduce(x_local):
    """Implement all-reduce using reduce-scatter + all-gather."""
    # Step 1: Reduce-scatter
    reduced_shard = jax.lax.psum_scatter(
        jnp.array([x_local]), 'data', scatter_dimension=0
    )
    # Step 2: All-gather to replicate the result
    full_result = jax.lax.all_gather(reduced_shard, 'data', axis=0)
    return full_result[0]  # Each device gets the full sum

# Ring-based all-reduce for large tensors
num_devices = len(devices)

@shard_map(
    mesh,
    in_specs=P('data',),
    out_specs=P('data',),
)
def ring_all_reduce(x_local):
    """Ring all-reduce: reduces communication bandwidth."""
    result = x_local
    # Reduce-scatter phase
    for step in range(num_devices - 1):
        partner = (jax.lax.axis_index('data') + 1) % num_devices
        received = jax.lax.ppermute(
            result, 'data',
            perm=[(i, (i + 1) % num_devices) for i in range(num_devices)]
        )
        if step == 0:
            result = received + x_local
        else:
            result = received + result

    # All-gather phase
    for step in range(num_devices - 1):
        result = jax.lax.ppermute(
            result, 'data',
            perm=[(i, (i - 1) % num_devices) for i in range(num_devices)]
        )
    return result
```

---

## 22.10 Comparison with GSPMD (Automatic Sharding)

### 22.10.1 Same Computation, Two Approaches

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.shard_map import shard_map
import numpy as np

devices = jax.devices()
mesh = Mesh(np.array(devices), ('data',))

# Approach 1: GSPMD (automatic sharding)
@jax.jit
def gspmd_matmul(x, w):
    # The compiler handles everything
    return x @ w

x_dp = jax.device_put(jnp.ones((1024, 512)), NamedSharding(mesh, P('data', None)))
w_rep = jax.device_put(jnp.ones((512, 256)), NamedSharding(mesh, P(None, None)))
result_gspmd = gspmd_matmul(x_dp, w_rep)

# Approach 2: shard_map (manual sharding)
@shard_map(
    mesh,
    in_specs=(P('data', None), P(None, None)),
    out_specs=P('data', None),
)
def manual_matmul(x_local, w):
    return x_local @ w

result_manual = manual_matmul(x_dp, w_rep)

# Both produce the same result!
print(jnp.allclose(result_gspmd, result_manual))  # True
```

### 22.10.2 When Communication is Needed

```python
# GSPMD: compiler automatically inserts an all-reduce
@jax.jit
def gspmd_sum(x):
    # x is data-parallel: P('data', None)
    # Summing along dim 0 requires all-reduce
    return jnp.sum(x, axis=0)

result = gspmd_sum(x_dp)
# The compiler knows it needs to all-reduce across 'data'

# shard_map: you must explicitly write the collective
@shard_map(
    mesh,
    in_specs=P('data', None),
    out_specs=P(None,),  # Result is replicated
)
def manual_sum(x_local):
    local_sum = jnp.sum(x_local, axis=0)  # Sum local shard
    return jax.lax.psum(local_sum, 'data')  # All-reduce

result2 = manual_sum(x_dp)
```

---

## 22.11 API Reference Summary

### shard_map

| Parameter | Description |
|-----------|-------------|
| `fun` | Function to apply per-device |
| `mesh` | Mesh defining device arrangement |
| `in_specs` | PartitionSpec(s) for inputs; determines local shard shape |
| `out_specs` | PartitionSpec(s) for outputs; determines reassembly |
| `check_rep` | If True, verify replicated outputs are consistent |

### Collective Operations

| Operation | Signature | Description |
|-----------|-----------|-------------|
| `psum(x, axis_name)` | `x: Array, axis_name: str` | All-reduce sum along mesh axis |
| `pmean(x, axis_name)` | `x: Array, axis_name: str` | All-reduce mean along mesh axis |
| `pmax(x, axis_name)` | `x: Array, axis_name: str` | All-reduce max along mesh axis |
| `pmin(x, axis_name)` | `x: Array, axis_name: str` | All-reduce min along mesh axis |
| `psum_scatter(x, axis_name, scatter_dimension=0)` | All-reduce + scatter | Memory-efficient reduction |
| `all_gather(x, axis_name, axis=0)` | Gather shards | Collect all shards on each device |
| `ppermute(x, axis_name, perm)` | Permute values | Custom permutation across devices |
| `all_to_all(x, axis_name)` | Exchange shards | All-to-all data exchange |
| `axis_index(axis_name)` | Get device index | Returns this device's position along axis |

### Mesh and Sharding (used with shard_map)

| API | Description |
|-----|-------------|
| `Mesh(devices, axis_names)` | Create a device mesh |
| `jax.make_mesh(shape, axis_names)` | Helper to create meshes |
| `PartitionSpec(*dims)` | Specify data layout (alias `jax.P`) |
| `NamedSharding(mesh, spec)` | Combine mesh and spec into sharding |
| `jax.device_put(x, sharding)` | Create sharded array |

### Utility Functions

| API | Description |
|-----|-------------|
| `jax.lax.axis_index(axis_name)` | Index of current device along named axis |
| `jax.debug.print(fmt, *args)` | Print from inside shard_map |
| `jax.debug.visualize_array_sharding(x)` | ASCII visualization of sharding |
| `jax.jit(shard_map_fn)` | Compile shard_map function |
| `jax.grad(shard_map_fn)` | Differentiate through shard_map |
