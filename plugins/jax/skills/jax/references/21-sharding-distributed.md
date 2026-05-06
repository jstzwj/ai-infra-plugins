# Sharding and Distributed Computing Reference

This reference provides comprehensive documentation of JAX's distributed computing and sharding system. JAX uses a data-centric approach to parallelism: you describe how data is distributed across devices, and the compiler (XLA via GSPMD) automatically handles the parallel computation. This reference covers Mesh, Sharding, PartitionSpec, distributed arrays, sharding modes, multi-process computing, and all related APIs.

---

## 21.1 Overview

JAX's distributed programming model centers on three concepts:

1. **Mesh** -- a logical arrangement of devices into a multi-dimensional grid with named axes.
2. **Sharding** -- a description of how an array is distributed across the devices in a mesh.
3. **Distributed Arrays** -- `jax.Array` objects whose data physically resides on multiple devices.

The compiler uses the **GSPMD** (General and Scalable Parallelization for ML Computation Graphs) approach: you annotate inputs and outputs with sharding specifications, and XLA automatically inserts the necessary collective communication operations.

```
+------------------------------------------------------------------+
|                JAX Distributed Computing Stack                    |
|                                                                   |
|  User Code                                                        |
|    |  jax.device_put(data, sharding)                              |
|    |  @jax.jit  with sharded inputs                               |
|    v                                                              |
|  Sharding Layer                                                   |
|    |  Mesh + PartitionSpec -> NamedSharding                       |
|    |  jax.lax.with_sharding_constraint                            |
|    v                                                              |
|  GSPMD Compiler (XLA)                                            |
|    |  Inserts collective ops (all-reduce, all-gather, etc.)       |
|    |  Optimizes communication placement                           |
|    v                                                              |
|  Runtime                                                          |
|    |  jax.Array with per-device buffers                           |
|    |  NCCL / libtpu collectives                                   |
|    v                                                              |
|  Devices (GPU / TPU / CPU)                                       |
+------------------------------------------------------------------+
```

### 21.1.1 Sharding Modes

JAX supports three sharding modes that control how the compiler interprets sharding annotations:

| Mode | Description | When to Use |
|------|-------------|-------------|
| **Auto** | Compiler automatically decides shardings; user can provide hints | Easiest to use; recommended for most cases |
| **Explicit** | Shardings appear in array types; the compiler requires explicit sharding on every array | Maximum control; useful for debugging |
| **Manual** | Per-device view with `shard_map`; user writes single-device code | Fine-grained control over collectives |

```python
import os

# Set sharding mode via environment variable (before importing JAX)
# Options: "auto", "explicit", "manual"
os.environ["JAX_SHARDING_MODE"] = "auto"
```

---

## 21.2 Mesh

A `Mesh` is a logical multi-dimensional arrangement of physical devices. It provides named axes that are referenced in `PartitionSpec` objects to describe how data is distributed.

### 21.2.1 jax.sharding.Mesh

```python
import jax
from jax.sharding import Mesh

# Get all available devices
devices = jax.devices()
print(devices)
# [CudaDevice(id=0), CudaDevice(id=1), CudaDevice(id=2), CudaDevice(id=3)]

# Create a 1D mesh with all devices
mesh_1d = Mesh(devices, axis_names=('data',))

# Create a 2D mesh: e.g., 4 devices -> (2, 2) grid
import numpy as np
mesh_2d = Mesh(
    np.array(devices).reshape(2, 2),
    axis_names=('data', 'model')
)

# Create a 3D mesh: e.g., 8 devices -> (2, 2, 2)
mesh_3d = Mesh(
    np.array(devices).reshape(2, 2, 2),
    axis_names=('data', 'model', 'expert')
)

# Mesh properties
print(mesh_2d.devices)          # set of all devices in mesh
print(mesh_2d.shape)            # FrozenDict({'data': 2, 'model': 2})
print(mesh_2d.size)             # 4 (total number of devices)
print(mesh_2d.axis_names)       # ('data', 'model')
```

Meshes are used as context managers. Code inside the context can reference the mesh's axis names in `PartitionSpec`:

```python
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

devices = jax.devices()
mesh = Mesh(np.array(devices).reshape(2, 2), ('data', 'model'))

with mesh:
    # Inside this block, 'data' and 'model' are valid axis names
    sharding = NamedSharding(mesh, P('data', 'model'))
    x = jax.device_put(jnp.ones((1024, 512)), sharding)
    # x is sharded across the 'data' axis on dim 0 and 'model' axis on dim 1
```

### 21.2.2 jax.make_mesh

`jax.make_mesh` is a convenience function that creates a mesh without manually reshaping device arrays:

```python
import jax

# Create a 1D mesh with 4 devices
mesh = jax.make_mesh((4,), ('data',))

# Create a 2D mesh with shape (2, 2) and named axes
mesh = jax.make_mesh((2, 2), ('data', 'model'))

# Create a 2D mesh using specific devices
devices = jax.devices()[:4]
mesh = jax.make_mesh((2, 2), ('data', 'model'), devices=devices)

# For TPU pods or multi-host, you can specify devices across hosts
all_devices = jax.devices()  # includes devices from all hosts in multi-process
mesh = jax.make_mesh(
    (jax.process_count() * jax.local_device_count(), 1),
    ('data', 'model')
)
```

### 21.2.3 jax.set_mesh and jax.get_mesh

These functions allow you to set and query the current active mesh without using a `with` block:

```python
import jax
from jax.sharding import Mesh
import numpy as np

mesh = Mesh(np.array(jax.devices()).reshape(2, 2), ('data', 'model'))

# Set mesh globally (use with caution)
jax.set_mesh(mesh)

# Query current mesh
current_mesh = jax.get_mesh()
print(current_mesh)        # Mesh with axis names: ('data', 'model')
print(current_mesh.shape)  # {'data': 2, 'model': 2}

# Reset to no mesh
jax.set_mesh(None)
print(jax.get_mesh())      # None (or AbstractMesh with empty axes)
```

### 21.2.4 AbstractMesh vs Concrete Mesh

An `AbstractMesh` represents a mesh whose shape is not yet known at tracing time. This is useful when writing functions that should work with any mesh configuration:

```python
from jax.sharding import AbstractMesh

# AbstractMesh is used internally during tracing
# In user code, you typically only encounter concrete Mesh objects

# When jax.get_mesh() is called outside any mesh context, it may return
# an AbstractMesh (depending on configuration) rather than None
```

### 21.2.5 AxisType: Auto, Explicit, Manual

`AxisType` controls how each mesh axis is handled by the compiler:

```python
from jax.sharding import Mesh
from jax._internal_mesh import AxisType  # or via mesh constructor
import jax
import numpy as np

# Auto axes: the compiler decides how to shard
mesh_auto = Mesh(
    np.array(jax.devices()).reshape(2, 2),
    axis_names=('data', 'model'),
    axis_types=(AxisType.Auto, AxisType.Auto)
)

# Explicit axes: sharding must be specified in types
mesh_explicit = Mesh(
    np.array(jax.devices()).reshape(2, 2),
    axis_names=('data', 'model'),
    axis_types=(AxisType.Explicit, AxisType.Explicit)
)

# Manual axes: used with shard_map for per-device programming
mesh_manual = Mesh(
    np.array(jax.devices()).reshape(2, 2),
    axis_names=('data', 'model'),
    axis_types=(AxisType.Manual, AxisType.Manual)
)

# Mixed: some axes auto, some manual
mesh_mixed = Mesh(
    np.array(jax.devices()).reshape(2, 2),
    axis_names=('data', 'model'),
    axis_types=(AxisType.Auto, AxisType.Manual)
)
```

| AxisType | Behavior |
|----------|----------|
| `Auto` | Compiler automatically determines sharding for this axis. User can provide hints via `with_sharding_constraint`. |
| `Explicit` | The sharding of every array along this axis must be explicitly specified. Types include sharding info. |
| `Manual` | Used with `shard_map`. The user writes single-device code and manually manages collectives. |

---

## 21.3 Sharding

The `Sharding` base class and its subclasses describe how an array is partitioned across devices.

### 21.3.1 jax.sharding.Sharding (Base Class)

All sharding objects inherit from `jax.sharding.Sharding`. The base class defines the interface:

```python
from jax.sharding import Sharding

class Sharding:
    """Base class for all shardings."""

    @property
    def device_set(self) -> set:
        """Set of devices that this sharding spans."""
        raise NotImplementedError

    @property
    def is_fully_replicated(self) -> bool:
        """Whether the array is replicated on all devices."""
        raise NotImplementedError

    @property
    def is_fully_addressable(self) -> bool:
        """Whether all shards are on the current process's devices."""
        raise NotImplementedError

    def shard_shape(self, global_shape) -> tuple:
        """Shape of each individual shard."""
        raise NotImplementedError

    def devices_indices_map(self, global_shape) -> dict:
        """Mapping from device to the slice indices it holds."""
        raise NotImplementedError
```

### 21.3.2 jax.sharding.NamedSharding

`NamedSharding` is the most commonly used sharding. It combines a `Mesh` with a `PartitionSpec`:

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import numpy as np

devices = jax.devices()
mesh = Mesh(np.array(devices).reshape(2, 2), ('data', 'model'))

# Fully replicated (no axis names in spec)
sharding_replicated = NamedSharding(mesh, P(None, None))

# Shard dim 0 along 'data' axis, replicate dim 1
sharding_dp = NamedSharding(mesh, P('data', None))

# Replicate dim 0, shard dim 1 along 'model' axis
sharding_mp = NamedSharding(mesh, P(None, 'model'))

# Shard dim 0 along 'data', dim 1 along 'model'
sharding_2d = NamedSharding(mesh, P('data', 'model'))

# Fully sharded on 'data' for a 1D array
sharding_1d = NamedSharding(mesh, P('data'))

# Inspect properties
print(sharding_2d.device_set)            # {device0, device1, device2, device3}
print(sharding_2d.is_fully_replicated)   # False
print(sharding_2d.is_fully_addressable)  # True (single-process)

global_shape = (1024, 512)
print(sharding_2d.shard_shape(global_shape))  # (512, 256)

indices = sharding_2d.devices_indices_map(global_shape)
for dev, idx in indices.items():
    print(f"{dev}: {idx}")
# CudaDevice(0): (slice(0, 512), slice(0, 256))
# CudaDevice(1): (slice(0, 512), slice(256, 512))
# CudaDevice(2): (slice(512, 1024), slice(0, 256))
# CudaDevice(3): (slice(512, 1024), slice(256, 512))
```

### 21.3.3 jax.sharding.SingleDeviceSharding

Places an entire array on a single device:

```python
from jax.sharding import SingleDeviceSharding
import jax

device = jax.devices()[0]
sharding = SingleDeviceSharding(device)

print(sharding.device_set)            # {CudaDevice(0)}
print(sharding.is_fully_replicated)   # True (trivially: one copy)
print(sharding.is_fully_addressable)  # True

# Use with device_put
x = jax.device_put(jnp.ones((100, 100)), sharding)
print(x.sharding)  # SingleDeviceSharding(device=CudaDevice(0))
```

### 21.3.4 jax.sharding.GSPMDSharding

`GSPMDSharding` is a lower-level sharding that directly specifies how each device gets a slice of the array. It is used internally and can be useful for specifying shardings that cannot be expressed with `NamedSharding`:

```python
from jax.sharding import GSPMDSharding
import jax
import numpy as np

devices = jax.devices()

# GSPMDSharding takes a list of devices and a HLO sharding string
# This is an advanced API - most users should prefer NamedSharding
sharding = GSPMDSharding(
    devices,
    # HLO sharding specification string
    # Example: replicate across 2 devices
    sharding_spec="{devices=[2,1]}"
)

# More commonly, GSPMDSharding is encountered when inspecting
# the sharding of arrays created by the compiler
```

### 21.3.5 jax.sharding.PositionalSharding

`PositionalSharding` is another lower-level option that maps devices to array slices using positional arguments:

```python
from jax.sharding import PositionalSharding
import jax

devices = jax.devices()

# Create a positional sharding for 4 devices
sharding = PositionalSharding(devices)

# Replicate across all devices
sharding_replicated = sharding.replicate()

# Shard a 2D array: dim 0 across 2 devices, dim 1 across 2 devices
sharding_2d = sharding.reshape(2, 2)

# Use with device_put
x = jax.device_put(jnp.ones((1024, 512)), sharding_2d)
```

### 21.3.6 Sharding Properties

All `Sharding` subclasses share these key properties:

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import numpy as np

devices = jax.devices()
mesh = Mesh(np.array(devices).reshape(2, 2), ('data', 'model'))
sharding = NamedSharding(mesh, P('data', 'model'))

# device_set: set of devices that hold data
device_set = sharding.device_set
print(f"Devices: {device_set}")
print(f"Number of devices: {len(device_set)}")

# is_fully_replicated: True if every device has a complete copy
print(f"Is fully replicated: {sharding.is_fully_replicated}")  # False
print(f"Replicated sharding: {NamedSharding(mesh, P(None, None)).is_fully_replicated}")  # True

# is_fully_addressable: True if all shards are on this process's devices
# In single-process settings, this is always True
# In multi-process settings, it may be False
print(f"Is fully addressable: {sharding.is_fully_addressable}")

# shard_shape: shape of each shard for a given global shape
global_shape = (1024, 512)
print(f"Shard shape: {sharding.shard_shape(global_shape)}")  # (512, 256)

# devices_indices_map: mapping from device -> tuple of slices
indices_map = sharding.devices_indices_map(global_shape)
for device, slices in indices_map.items():
    print(f"  {device}: {slices}")
```

---

## 21.4 PartitionSpec (jax.P)

`PartitionSpec` (importable as `jax.P`) specifies how each dimension of an array maps to mesh axes. It is the primary tool for describing data layouts.

### 21.4.1 Creating PartitionSpecs

```python
from jax.sharding import PartitionSpec as P

# Basic syntax: P(axis_for_dim0, axis_for_dim1, ...)
# Each argument corresponds to one dimension of the array.

# 1D array: shard along 'data' mesh axis
spec_1d_sharded = P('data')

# 2D array: shard dim 0 along 'data', dim 1 along 'model'
spec_2d = P('data', 'model')

# 2D array: shard dim 0 along 'data', replicate dim 1
spec_dp = P('data', None)

# 2D array: fully replicated
spec_replicated = P(None, None)

# 3D array: various patterns
spec_3d = P('batch', 'seq', 'heads')  # all dims sharded
spec_3d_partial = P('batch', None, 'heads')  # dim 1 replicated

# None means "replicate this dimension across all devices"
# A string means "shard this dimension along the named mesh axis"
```

### 21.4.2 Unreduced Axes

Unreduced axes are a special feature for specifying that a dimension is sharded but no all-reduce should be inserted. This is relevant when using `shard_map`:

```python
from jax.sharding import PartitionSpec as P

# Standard sharding: dim 0 sharded on 'data', dim 1 replicated
spec_normal = P('data', None)

# Unreduced: dim 0 sharded on 'data', and 'model' axis is unreduced
# This means the 'model' axis won't trigger automatic reductions
spec_unreduced = P('data', None, unreduced_axes={'model'})

# Unreduced axes in a 2D context
spec = P('data', 'model', unreduced_axes={'model'})
```

### 21.4.3 Tuple Axis Names (Multi-axis Sharding)

You can shard a single array dimension across multiple mesh axes by passing a tuple:

```python
from jax.sharding import PartitionSpec as P

# Shard dim 0 across both 'data' and 'model' axes of the mesh
# The array dimension is split first by 'data', then each chunk by 'model'
spec_combined = P(('data', 'model'), None)

# This requires mesh to have both 'data' and 'model' axes
# For a mesh of shape (2, 2), an array of size 1024 would be split:
# 1024 -> 512 (by 'data') -> 256 (by 'model')
# Each of the 4 devices gets a shard of size 256
```

### 21.4.4 Common PartitionSpec Patterns

```python
from jax.sharding import PartitionSpec as P

# === Data Parallelism ===
# Batch dimension sharded, features replicated
dp_spec = P('data', None)

# === Tensor Parallelism (Megatron-style) ===
# Shard weight matrices along one dimension
tp_col_spec = P('model', None)    # Column-parallel: shard input dim
tp_row_spec = P(None, 'model')    # Row-parallel: shard output dim

# === Fully Sharded Data Parallelism (FSDP) ===
# Shard parameters along one axis
fsdp_spec = P('fsdp', None)

# === Pipeline Parallelism ===
# Typically uses different devices for different layers
# No special PartitionSpec needed; uses sequential device assignment

# === 2D/3D Parallelism ===
# Combined data + model parallelism
spec_2d = P('data', 'model')

# 3D: batch, sequence, heads each sharded
spec_3d = P('batch', 'seq', 'heads')

# === Attention: Q, K, V heads sharded ===
# Query: (batch, seq, num_heads, head_dim)
q_spec = P('data', None, 'model', None)
# Key/Value: same
kv_spec = P('data', None, 'model', None)
```

---

## 21.5 Distributed Arrays

### 21.5.1 Creating Sharded Arrays with jax.device_put

`jax.device_put` is the primary way to create sharded arrays:

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import numpy as np

devices = jax.devices()
mesh = Mesh(np.array(devices).reshape(2, 2), ('data', 'model'))

# Create data on host
x_host = np.random.randn(1024, 512)

# Shard and place on devices
sharding = NamedSharding(mesh, P('data', 'model'))
x = jax.device_put(x_host, sharding)

print(type(x))             # <class 'jax.Array'>
print(x.shape)             # (1024, 512)
print(x.dtype)             # float32
print(x.sharding)          # NamedSharding(mesh, P('data', 'model'))

# The array is now physically distributed across devices
# Operations on it will be parallelized automatically
y = x @ x.T  # matrix multiply - compiler handles collectives
```

You can also use `jax.device_put` with a specific device:

```python
# Place on a single device
x_single = jax.device_put(x_host, jax.devices()[0])

# Place with SingleDeviceSharding
from jax.sharding import SingleDeviceSharding
x_single2 = jax.device_put(x_host, SingleDeviceSharding(jax.devices()[0]))
```

### 21.5.2 jax.Array

`jax.Array` is the core array type that supports both single-device and multi-device (sharded) arrays:

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import numpy as np

# Regular single-device array
x_single = jnp.ones((4, 4))
print(type(x_single))           # <class 'jax.Array'>
print(x_single.sharding)        # SingleDeviceSharding

# Sharded array
devices = jax.devices()
mesh = Mesh(np.array(devices).reshape(2, 2), ('data', 'model'))
sharding = NamedSharding(mesh, P('data', None))
x_sharded = jax.device_put(jnp.ones((1024, 512)), sharding)
print(type(x_sharded))          # <class 'jax.Array'>
print(x_sharded.sharding)       # NamedSharding(mesh=..., spec=P('data', None))

# Both are jax.Array - the API is uniform
```

### 21.5.3 jax.typeof

`jax.typeof` inspects the full type of an array, including its sharding in explicit mode:

```python
import jax
import jax.numpy as jnp

# In auto mode (default), typeof shows shape and dtype
x = jnp.ones((4, 4))
print(jax.typeof(x))  # float32[4,4]

# In explicit mode, typeof also shows sharding info
# (requires explicit mode to be enabled)
```

### 21.5.4 Inspecting Shards

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import numpy as np

devices = jax.devices()
mesh = Mesh(np.array(devices).reshape(2, 2), ('data', 'model'))
sharding = NamedSharding(mesh, P('data', 'model'))

x = jax.device_put(jnp.arange(16.0).reshape(4, 4), sharding)

# addressable_shards: shards on this process's devices
for shard in x.addressable_shards:
    print(f"Device: {shard.device}")
    print(f"  Index: {shard.index}")
    print(f"  Data shape: {shard.data.shape}")
    print(f"  Data:\n{shard.data}")

# global_shards: all shards (including those on other processes)
# In single-process: same as addressable_shards
# In multi-process: includes shards from all processes
for shard in x.global_shards:
    print(f"Device: {shard.device}, Replica ID: {shard.replica_id}")

# Check if a specific shard is on this process
for shard in x.global_shards:
    print(f"Device {shard.device}: addressable={shard.device in jax.local_devices()}")
```

### 21.5.5 jax.debug.visualize_array_sharding

Visualize how an array is sharded using ASCII art:

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import numpy as np

devices = jax.devices()
mesh = Mesh(np.array(devices).reshape(2, 2), ('data', 'model'))

x = jax.device_put(jnp.ones((1024, 512)), NamedSharding(mesh, P('data', 'model')))
jax.debug.visualize_array_sharding(x)

# Output (example):
# ┌───────────┬───────────┐
# │           │           │
# │  GPU 0    │  GPU 1    │
# │           │           │
# ├───────────┼───────────┤
# │           │           │
# │  GPU 2    │  GPU 3    │
# │           │           │
# └───────────┴───────────┘

# For a replicated array
x_replicated = jax.device_put(jnp.ones((1024, 512)), NamedSharding(mesh, P(None, None)))
jax.debug.visualize_array_sharding(x_replicated)
# All devices show as having the full array

# Also works inside jitted functions (via jax.debug.print)
@jax.jit
def f(x):
    jax.debug.visualize_array_sharding(x, "Input sharding")
    return x * 2
```

---

## 21.6 Automatic vs Explicit vs Manual Modes

### 21.6.1 Auto Mode

In auto mode, the compiler decides shardings automatically. User code does not need to specify shardings -- the compiler propagates them through the computation:

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import numpy as np
import os

# Auto mode is the default
os.environ["JAX_SHARDING_MODE"] = "auto"

devices = jax.devices()
mesh = Mesh(np.array(devices).reshape(2, 2), ('data', 'model'))

with mesh:
    # Input is sharded
    x = jax.device_put(jnp.ones((1024, 512)), NamedSharding(mesh, P('data', None)))

    # Compiler propagates sharding through computation
    @jax.jit
    def f(x):
        # The compiler decides how intermediate values are sharded
        y = x @ jnp.ones((512, 256))
        z = jnp.sum(y, axis=0)
        return z

    result = f(x)
    print(result.shape)      # (256,)
    print(result.sharding)   # Compiler-chosen sharding
```

In auto mode, you can provide **hints** via `with_sharding_constraint` (see section 21.8).

### 21.6.2 Explicit Mode

In explicit mode, shardings must be explicitly specified on all array types. The compiler does not infer shardings:

```python
import os
os.environ["JAX_SHARDING_MODE"] = "explicit"

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import numpy as np

devices = jax.devices()
mesh = Mesh(np.array(devices).reshape(2, 2), ('data', 'model'))

with mesh:
    # Must specify sharding for all inputs
    x = jax.device_put(
        jnp.ones((1024, 512)),
        NamedSharding(mesh, P('data', None))
    )
    w = jax.device_put(
        jnp.ones((512, 256)),
        NamedSharding(mesh, P(None, 'model'))
    )

    @jax.jit
    def f(x, w):
        # In explicit mode, the output sharding is determined by the inputs
        # and the compiler will NOT insert resharding automatically
        y = x @ w
        return y

    result = f(x, w)
    # jax.typeof now includes sharding info
    print(jax.typeof(result))
```

### 21.6.3 Manual Mode

In manual mode, you use `shard_map` to write per-device code. See the `shard_map` reference for full details. Here is a brief example:

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.shard_map import shard_map
import numpy as np

devices = jax.devices()
mesh = Mesh(np.array(devices).reshape(4, 1), ('data',))

@shard_map(
    mesh,
    in_specs=P('data', None),
    out_specs=P('data', None),
)
def manual_matmul(x_shard):
    # x_shard is a per-device chunk of the input
    # Write single-device code here
    return x_shard * 2

x = jax.device_put(jnp.ones((1024, 512)), jax.sharding.NamedSharding(mesh, P('data', None)))
result = manual_matmul(x)
```

### 21.6.4 auto_axes and explicit_axes Decorators

These decorators provide fine-grained control over which mesh axes use which mode:

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import numpy as np

devices = jax.devices()
mesh = Mesh(np.array(devices).reshape(2, 2), ('data', 'model'))

# auto_axes: specify which axes should use auto sharding
# Other axes use the default mode
@jax.auto_axes(('data',))
def f_auto(x):
    # 'data' axis is in auto mode
    # 'model' axis follows the global setting
    return x @ x.T

# explicit_axes: specify which axes require explicit sharding
@jax.explicit_axes(('model',))
def f_explicit(x):
    # 'model' axis requires explicit sharding annotations
    # 'data' axis follows the global setting
    return x @ x.T
```

---

## 21.7 jax.lax.with_sharding_constraint

`with_sharding_constraint` provides hints to the compiler about how an intermediate value should be sharded. It is useful in auto mode to guide the compiler toward a better sharding strategy:

### 21.7.1 Basic Usage

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import jax.lax as lax
import numpy as np

devices = jax.devices()
mesh = Mesh(np.array(devices).reshape(2, 2), ('data', 'model'))

with mesh:
    @jax.jit
    def f(x, w):
        # x is sharded as P('data', None)
        # w is sharded as P(None, 'model')
        y = x @ w

        # Hint: we want y to be sharded as P('data', 'model')
        y = lax.with_sharding_constraint(y, NamedSharding(mesh, P('data', 'model')))

        # Continue computation with constrained sharding
        z = jnp.sum(y, axis=-1, keepdims=True)

        # Another constraint: make z replicated
        z = lax.with_sharding_constraint(z, NamedSharding(mesh, P('data', None)))

        return z

    x = jax.device_put(jnp.ones((1024, 512)), NamedSharding(mesh, P('data', None)))
    w = jax.device_put(jnp.ones((512, 256)), NamedSharding(mesh, P(None, 'model')))
    result = f(x, w)
```

### 21.7.2 When to Use Constraints

```python
# Use with_sharding_constraint when:
#
# 1. The compiler chooses a suboptimal sharding
# 2. You want to prevent unnecessary resharding
# 3. You need to control communication patterns
# 4. You want to force replication before a reduction

@jax.jit
def model(x, w1, w2):
    # Force intermediate to be sharded a particular way
    h = x @ w1
    h = jax.lax.with_sharding_constraint(h, NamedSharding(mesh, P('data', 'model')))
    h = jax.nn.relu(h)

    # Force replication before a collective
    out = h @ w2
    out = jax.lax.with_sharding_constraint(out, NamedSharding(mesh, P('data', None)))
    return out
```

### 21.7.3 Constraints in Explicit Mode

In explicit mode, `with_sharding_constraint` is more than a hint -- it may be required to specify shardings that the compiler cannot infer:

```python
# In explicit mode, shardings are part of the type system
# with_sharding_constraint is used to cast between sharding types

@jax.jit
def f(x):
    y = x * 2
    # In explicit mode, this constraint IS the sharding specification
    y = jax.lax.with_sharding_constraint(
        y, NamedSharding(mesh, P('data', None))
    )
    return y
```

---

## 21.8 Multi-Process Computing

### 21.8.1 jax.distributed.initialize()

For multi-host (multi-process) computation, each process must initialize JAX's distributed runtime:

```python
import jax
import jax.distributed

# Initialize distributed runtime
# This must be called before any JAX operations
jax.distributed.initialize()

# With explicit coordinator configuration:
# jax.distributed.initialize(
#     coordinator_address="host:port",  # Address of the coordinator process
#     num_processes=4,                  # Total number of processes
#     process_id=0,                     # This process's ID (0-indexed)
# )

# After initialization:
print(f"Process index: {jax.process_index()}")    # This process's rank
print(f"Process count: {jax.process_count()}")    # Total number of processes
print(f"Local devices: {jax.local_devices()}")    # Devices on this machine
print(f"All devices: {jax.devices()}")            # All devices across all processes
print(f"Local device count: {jax.local_device_count()}")
print(f"Total device count: {jax.device_count()}")
```

### 21.8.2 Typical Multi-Process Launch

```bash
# Launch on 2 hosts, 4 GPUs each (8 GPUs total)
# On host 0 (coordinator):
python train.py \
    --jax_distributed_coordinator_address=host0:12345 \
    --jax_distributed_num_processes=2 \
    --jax_distributed_process_id=0

# On host 1:
python train.py \
    --jax_distributed_coordinator_address=host0:12345 \
    --jax_distributed_num_processes=2 \
    --jax_distributed_process_id=1

# Or use the cluster launcher:
# Using jax.distributed.initialize() with environment variables
# set by your cluster manager (e.g., Slurm, Kubernetes)
```

### 21.8.3 Multi-Process Mesh

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import numpy as np

# Initialize distributed
jax.distributed.initialize()

# All devices across all processes
all_devices = jax.devices()  # Includes devices from all processes
total_devices = len(all_devices)
local_devices = jax.local_devices()
local_count = len(local_devices)

print(f"Process {jax.process_index()}/{jax.process_count()}")
print(f"Total devices: {total_devices}, Local devices: {local_count}")

# Create a mesh spanning all devices across all processes
mesh = Mesh(
    np.array(all_devices).reshape(jax.process_count(), local_count),
    ('host', 'device')
)

# Or a 1D mesh across all devices
mesh_1d = Mesh(np.array(all_devices), ('data',))
```

### 21.8.4 Creating Multi-Process Arrays

#### jax.make_array_from_process_local_data

Each process contributes its local portion of a globally-sharded array:

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import numpy as np

jax.distributed.initialize()

all_devices = jax.devices()
mesh = Mesh(np.array(all_devices), ('data',))
sharding = NamedSharding(mesh, P('data',))

# Global shape of the array
global_shape = (jax.device_count() * 128, 512)

# Each process provides its local data
# The local data should match the shard shape for this process's devices
local_data = np.random.randn(
    len(jax.local_devices()) * 128,  # Local batch size
    512
)

# Create a global array from process-local data
# Each process calls this with its own local_data
global_array = jax.make_array_from_process_local_data(
    sharding,
    local_data,
    global_shape=global_shape,
)

print(f"Process {jax.process_index()}: global_array.shape = {global_array.shape}")
```

#### jax.make_array_from_single_device_arrays

Combine single-device arrays into one sharded array:

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import numpy as np

jax.distributed.initialize()

all_devices = jax.devices()
mesh = Mesh(np.array(all_devices).reshape(2, -1), ('data', 'model'))
sharding = NamedSharding(mesh, P('data', 'model'))

global_shape = (256, 512)

# Create single-device arrays for each device
single_device_arrays = {}
for device in jax.local_devices():
    # Create the appropriate shard for this device
    indices = sharding.devices_indices_map(global_shape)[device]
    # Create data for this shard
    shard_shape = sharding.shard_shape(global_shape)
    data = np.random.randn(*shard_shape)
    single_device_arrays[device] = jax.device_put(data, device)

# Combine into a global sharded array
global_array = jax.make_array_from_single_device_arrays(
    global_shape,
    sharding,
    single_device_arrays,
)
```

### 21.8.5 Multi-Process Data Parallel Training Example

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import numpy as np
import optax

# Initialize distributed runtime
jax.distributed.initialize()

# Create mesh
all_devices = jax.devices()
mesh = Mesh(np.array(all_devices), ('data',))

# Define sharding
data_sharding = NamedSharding(mesh, P('data', None))  # Shard batch dim
param_sharding = NamedSharding(mesh, P(None, None))    # Replicate params

# Model
def init_params(key, input_dim, hidden_dim, output_dim):
    k1, k2, k3 = jax.random.split(key, 3)
    params = {
        'w1': jax.random.normal(k1, (input_dim, hidden_dim)) * 0.01,
        'b1': jnp.zeros(hidden_dim),
        'w2': jax.random.normal(k2, (hidden_dim, output_dim)) * 0.01,
        'b2': jnp.zeros(output_dim),
    }
    return params

def predict(params, x):
    h = jax.nn.relu(x @ params['w1'] + params['b1'])
    return h @ params['w2'] + params['b2']

def loss_fn(params, x, y):
    pred = predict(params, x)
    return jnp.mean((pred - y) ** 2)

@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    # Gradients are automatically averaged across devices in data-parallel mode
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# Initialize
key = jax.random.PRNGKey(0)
params = init_params(key, 784, 256, 10)
params = jax.device_put(params, param_sharding)
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

# Training loop
for step in range(1000):
    # Each process loads its own batch
    # The data loader should provide different data per process
    x_batch = np.random.randn(128, 784).astype(np.float32)
    y_batch = np.random.randn(128, 10).astype(np.float32)

    # Shard the batch across devices
    x_sharded = jax.device_put(x_batch, data_sharding)
    y_sharded = jax.device_put(y_batch, data_sharding)

    params, opt_state, loss = train_step(params, opt_state, x_sharded, y_sharded)

    if step % 100 == 0 and jax.process_index() == 0:
        print(f"Step {step}, Loss: {loss:.4f}")
```

---

## 21.9 Advanced Topics

### 21.9.1 Resharding

When two operations expect different shardings, JAX automatically inserts resharding operations (collective communication):

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import numpy as np

devices = jax.devices()
mesh = Mesh(np.array(devices).reshape(2, 2), ('data', 'model'))

@jax.jit
def resharding_example(x):
    # x comes in as P('data', None)
    # Force a different sharding
    y = jax.lax.with_sharding_constraint(
        x, NamedSharding(mesh, P(None, 'model'))
    )
    # The compiler inserts an all-to-all collective here
    return y

x = jax.device_put(jnp.ones((1024, 512)), NamedSharding(mesh, P('data', None)))
result = resharding_example(x)
```

### 21.9.2 Sharding and jax.jit

`jax.jit` respects input shardings and propagates them through the computation:

```python
@jax.jit
def f(x, y):
    # x and y may have different shardings
    # The compiler handles necessary resharding
    z = x + y  # element-wise: sharding should match
    w = x @ y  # matmul: compiler determines output sharding
    return z, w

x = jax.device_put(jnp.ones((1024, 512)), NamedSharding(mesh, P('data', None)))
y = jax.device_put(jnp.ones((512, 256)), NamedSharding(mesh, P(None, 'model')))
z, w = f(x, y)
print(f"z sharding: {z.sharding}")
print(f"w sharding: {w.sharding}")
```

### 21.9.3 Sharding and jax.grad

Gradients respect the sharding of the inputs. If parameters are replicated, gradients will be all-reduced across devices:

```python
@jax.jit
def compute_grads(params, x):
    loss = jnp.mean((params['w'] @ x) ** 2)
    return jax.grad(lambda p: loss)(params)

params = jax.device_put({'w': jnp.ones((256, 512))}, param_sharding)
x = jax.device_put(jnp.ones((512, 128)), data_sharding)
grads = compute_grads(params, x)
# grads['w'] has the same sharding as params['w']
```

### 21.9.4 Sharding and jax.vmap

`jax.vmap` can be combined with sharding for batch parallelism:

```python
@jax.jit
def f(x):
    # x: (batch, features)
    return jnp.sum(x, axis=-1)

# vmap over a sharded dimension
f_vmapped = jax.vmap(f)

x = jax.device_put(jnp.ones((1024, 128)), NamedSharding(mesh, P('data', None)))
result = f_vmapped(x)
```

### 21.9.5 Sharding Annotations for Common Model Patterns

#### Transformer Attention with 2D Sharding

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import numpy as np

devices = jax.devices()
mesh = Mesh(np.array(devices).reshape(2, 2), ('batch', 'heads'))

def attention(q, k, v, mesh):
    """Multi-head attention with batch and head parallelism."""
    # q, k, v: (batch, seq_len, num_heads, head_dim)
    batch_spec = P('batch', None, 'heads', None)

    # Scaled dot-product attention
    scale = jnp.sqrt(q.shape[-1]).astype(q.dtype)
    scores = jnp.einsum('bshd,bthd->bhst', q, k) / scale
    weights = jax.nn.softmax(scores, axis=-1)
    output = jnp.einsum('bhst,bthd->bshd', weights, v)

    # Constrain output sharding
    output = jax.lax.with_sharding_constraint(
        output, NamedSharding(mesh, batch_spec)
    )
    return output

with mesh:
    batch, seq, heads, dim = 8, 128, 16, 64
    q = jax.device_put(
        jnp.ones((batch, seq, heads, dim)),
        NamedSharding(mesh, P('batch', None, 'heads', None))
    )
    k = jax.device_put(
        jnp.ones((batch, seq, heads, dim)),
        NamedSharding(mesh, P('batch', None, 'heads', None))
    )
    v = jax.device_put(
        jnp.ones((batch, seq, heads, dim)),
        NamedSharding(mesh, P('batch', None, 'heads', None))
    )

    out = attention(q, k, v, mesh)
    print(f"Output shape: {out.shape}")
    print(f"Output sharding: {out.sharding}")
```

#### FSDP-Style Sharded Parameters

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import jax.lax as lax
import numpy as np

devices = jax.devices()
mesh = Mesh(np.array(devices), ('fsdp',))

def fsdp_layer(x, w, b, mesh):
    """A linear layer with FSDP-style parameter sharding."""
    # x: (batch, features) - sharded along batch
    # w: (features_in, features_out) - sharded along features_out (fsdp axis)
    # b: (features_out,) - sharded along fsdp axis

    # Local matmul: each device has a shard of w
    local_out = x @ w  # (batch, shard_of_features_out)

    # All-gather the output across fsdp axis to get full features_out
    # Or use reduce-scatter for memory efficiency
    out = local_out + b

    # Constrain back to batch-sharded for next layer
    out = lax.with_sharding_constraint(
        out, NamedSharding(mesh, P('fsdp', None))
    )
    return out

with mesh:
    batch, feat_in, feat_out = 64, 256, 128
    x = jax.device_put(
        jnp.ones((batch, feat_in)),
        NamedSharding(mesh, P('fsdp', None))
    )
    w = jax.device_put(
        jnp.ones((feat_in, feat_out)),
        NamedSharding(mesh, P(None, 'fsdp'))
    )
    b = jax.device_put(
        jnp.ones((feat_out,)),
        NamedSharding(mesh, P('fsdp'))
    )

    out = fsdp_layer(x, w, b, mesh)
```

### 21.9.6 Debugging Sharding Issues

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import numpy as np

devices = jax.devices()
mesh = Mesh(np.array(devices).reshape(2, 2), ('data', 'model'))

@jax.jit
def debug_sharding(x, w):
    y = x @ w

    # Print sharding at various points
    jax.debug.print("y sharding: {}", y.sharding)
    jax.debug.visualize_array_sharding(y, "After matmul")

    z = jnp.sum(y, axis=0)
    jax.debug.print("z sharding after sum: {}", z.sharding)
    jax.debug.visualize_array_sharding(z, "After reduction")

    return z

with mesh:
    x = jax.device_put(jnp.ones((1024, 512)), NamedSharding(mesh, P('data', None)))
    w = jax.device_put(jnp.ones((512, 256)), NamedSharding(mesh, P(None, 'model')))
    result = debug_sharding(x, w)
```

### 21.9.7 Checking Sharding Properties at Runtime

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import numpy as np

devices = jax.devices()
mesh = Mesh(np.array(devices).reshape(2, 2), ('data', 'model'))

x = jax.device_put(jnp.ones((1024, 512)), NamedSharding(mesh, P('data', None)))

# Check if an array is sharded
def is_sharded(arr):
    return not arr.sharding.is_fully_replicated

print(f"Is sharded: {is_sharded(x)}")  # True

# Check which devices hold data
print(f"Devices: {x.sharding.device_set}")

# Get shard shapes
print(f"Shard shape: {x.sharding.shard_shape(x.shape)}")

# Compare shardings
y = jax.device_put(jnp.ones((1024, 512)), NamedSharding(mesh, P('data', None)))
z = jax.device_put(jnp.ones((1024, 512)), NamedSharding(mesh, P('data', 'model')))

print(f"Same sharding: {x.sharding == y.sharding}")  # True
print(f"Different sharding: {x.sharding == z.sharding}")  # False
```

---

## 21.10 API Reference Summary

### Mesh API

| API | Description |
|-----|-------------|
| `Mesh(devices, axis_names)` | Create a mesh from device array and axis name tuple |
| `jax.make_mesh(shape, axis_names)` | Helper to create mesh from shape tuple |
| `jax.set_mesh(mesh)` | Set the current thread-local mesh |
| `jax.get_mesh()` | Get the current thread-local mesh |
| `mesh.devices` | Set of all devices in the mesh |
| `mesh.shape` | `FrozenDict` mapping axis names to sizes |
| `mesh.size` | Total number of devices |
| `mesh.axis_names` | Tuple of axis name strings |

### Sharding API

| API | Description |
|-----|-------------|
| `NamedSharding(mesh, spec)` | Sharding from mesh + PartitionSpec |
| `SingleDeviceSharding(device)` | Entire array on one device |
| `GSPMDSharding(devices, spec)` | Low-level GSPMD sharding |
| `PositionalSharding(devices)` | Positional device-to-slice mapping |
| `PartitionSpec(*dims)` | Specifies per-dimension sharding (alias `jax.P`) |
| `sharding.device_set` | Set of devices holding data |
| `sharding.is_fully_replicated` | True if all devices have full copy |
| `sharding.is_fully_addressable` | True if all shards on local process |
| `sharding.shard_shape(shape)` | Shape of each shard |
| `sharding.devices_indices_map(shape)` | Device-to-slice mapping |

### Array Shard Inspection API

| API | Description |
|-----|-------------|
| `array.sharding` | Sharding object for this array |
| `array.addressable_shards` | List of `Shard` objects on local devices |
| `array.global_shards` | List of all `Shard` objects |
| `jax.debug.visualize_array_sharding(arr)` | ASCII visualization of sharding |
| `jax.typeof(arr)` | Full type including sharding (explicit mode) |

### Multi-Process API

| API | Description |
|-----|-------------|
| `jax.distributed.initialize()` | Initialize multi-process runtime |
| `jax.process_index()` | This process's rank (0-indexed) |
| `jax.process_count()` | Total number of processes |
| `jax.make_array_from_process_local_data(sharding, data, global_shape)` | Create global array from per-process data |
| `jax.make_array_from_single_device_arrays(shape, sharding, arrays)` | Combine single-device arrays |
| `jax.devices()` | All devices (all processes) |
| `jax.local_devices()` | Devices on this process only |

### Constraint API

| API | Description |
|-----|-------------|
| `jax.lax.with_sharding_constraint(x, sharding)` | Hint/constrain intermediate sharding |
| `jax.auto_axes(axes)` | Decorator: auto-mode for specified axes |
| `jax.explicit_axes(axes)` | Decorator: explicit-mode for specified axes |
