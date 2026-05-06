# SPMD Partitioner

This document provides comprehensive documentation about XLA's SPMD (Single Program, Multiple Data) partitioner, which automatically partitions computations for execution across multiple devices.

## Table of Contents

- [Overview](#overview)
- [GSPMD Paper Overview](#gspmd-paper-overview)
- [ShardingPropagation Pass](#shardingpropagation-pass)
- [Communication Operations](#communication-operations)
- [Example Workflow](#example-workflow)

## Overview

XLA's SPMD partitioner is a compiler pass that transforms a single program written for the full (global) tensor shapes into a program that runs on each device with its local shard of the data. This enables efficient distributed computation without requiring the user to write device-specific code.

The key insight of SPMD is that all devices execute the same program (single program), but each device operates on its own portion of the data (multiple data). The partitioner automatically:

1. **Infers sharding**: Determines how tensors should be distributed across devices.
2. **Partitions operations**: Rewrites each operation to work on the local shard.
3. **Inserts communication**: Adds necessary collective operations (all-reduce, all-gather, collective-permute, etc.) to coordinate between devices.
4. **Overlaps computation and communication**: Schedules communication to overlap with independent computation where possible.

## GSPMD Paper Overview

The SPMD partitioner is based on the GSPMD (General and Scalable Parallelization for ML Computation Graphs) system described in the research paper. The key concepts are:

### Sharding Annotations

Users annotate tensors with sharding specifications that describe how they should be distributed across devices:

- **Replicated**: The entire tensor is copied to every device.
- **Sharded along dimension**: The tensor is split along a specific dimension, with each device holding a contiguous slice.

Sharding is specified using two concepts:

1. **Mesh**: Describes the physical device topology as a multi-dimensional grid.
2. **PartitionSpec**: Describes how each tensor dimension maps to mesh dimensions.

### Mesh

A mesh defines the logical arrangement of devices:

```python
import jax
from jax.sharding import Mesh
import jax.numpy as jnp

# Define a 2D mesh of devices
# 8 devices arranged as 4 x 2
devices = jax.devices()
mesh = Mesh(
    jax.numpy.array(devices[:8]).reshape(4, 2),
    axis_names=('data', 'model')
)
```

In this example:
- 8 physical devices are arranged in a 4x2 grid.
- The first mesh axis is named 'data' (size 4).
- The second mesh axis is named 'model' (size 2).

Meshes can be 1D, 2D, or higher-dimensional, depending on the desired parallelism strategy.

### PartitionSpec

A `PartitionSpec` describes how each dimension of a tensor maps to the mesh:

```python
from jax.sharding import PartitionSpec as P

# Tensor of shape (1024, 512)
# P('data', 'model') means:
#   - First dimension (1024) is sharded across 'data' mesh axis (4 devices)
#   - Second dimension (512) is sharded across 'model' mesh axis (2 devices)
spec = P('data', 'model')

# Each device holds a local shard of shape (256, 256)
```

Common sharding patterns:

| PartitionSpec | Shape (1024, 512) | Meaning |
|--------------|-------------------|---------|
| `P(None, None)` | (1024, 512) per device | Fully replicated |
| `P('data', None)` | (256, 512) per device | Sharded along dim 0 across 'data' |
| `P(None, 'model')` | (1024, 256) per device | Sharded along dim 1 across 'model' |
| `P('data', 'model')` | (256, 256) per device | Sharded along both dims |
| `P(('data', 'model'), None)` | (128, 512) per device | Dim 0 sharded across both mesh axes |

### jax.pjit Integration

In JAX, `pjit` (or `jit` with sharding) applies sharding annotations to compiled functions:

```python
from jax.sharding import Mesh, PartitionSpec as P
import jax

# Create mesh
devices = jax.devices()
mesh = Mesh(jax.numpy.array(devices[:8]).reshape(4, 2),
            axis_names=('data', 'model'))

# Define a function with sharding annotations
@jax.jit(
    in_shardings=(P('data', 'model'), P('model', None)),
    out_shardings=P('data', 'model')
)
def matmul(x, y):
    return jnp.dot(x, y)

# Run with the mesh context
with mesh:
    x = jnp.ones((1024, 512))
    y = jnp.ones((512, 256))
    result = matmul(x, y)
```

The `in_shardings` specify how inputs are sharded, and `out_shardings` specify how outputs should be sharded. The SPMD partitioner uses these annotations to generate the distributed code.

## ShardingPropagation Pass

### Automatic Sharding Propagation

The `ShardingPropagation` pass is a key component of the SPMD partitioner. It automatically infers sharding for operations that do not have explicit sharding annotations.

#### How It Works

1. **Start with annotations**: Operations with explicit sharding annotations provide the starting points.

2. **Propagate forward**: For each operation, infer the output sharding based on the input sharding and the operation's semantics:
   - Elementwise operations: Output sharding matches input sharding.
   - Dot (matmul): The output sharding is derived from the input sharding and the contraction dimensions.
   - Reduce: The output sharding drops the reduced dimension's sharding.
   - Reshape: The output sharding is mapped according to the reshape.

3. **Propagate backward**: When an output has a required sharding (e.g., from a user annotation), propagate the constraint backward to inputs.

4. **Resolve conflicts**: When forward and backward propagation produce conflicting sharding requirements, insert resharding operations (communication collectives) to convert between shardings.

#### Sharding Propagation Rules

For common HLO operations:

**Elementwise Operations** (add, multiply, etc.):
```
Input sharding: P('data', 'model')
Output sharding: P('data', 'model')
```
The output sharding matches the input sharding (elementwise operations apply to each element independently).

**Dot (Matrix Multiplication)**:
```
# For dot(lhs, rhs) with lhs [M, K] and rhs [K, N]
lhs sharding: P('data', 'model')
rhs sharding: P('model', None)
Output sharding: P('data', None)

# M is sharded across 'data' (from lhs)
# K is sharded across 'model' (from both lhs and rhs)
# N is replicated (from rhs)
# Result: output [M, N] is sharded along M across 'data'
```

**Reduce**:
```
# For reduce(x, axes=[1]) with x [M, N]
x sharding: P('data', 'model')  # N is sharded across 'model'
Output sharding: P('data', None)

# Reducing over dimension that is sharded requires an all-reduce
# The output has one fewer dimension, so 'model' is dropped
```

**Broadcast**:
```
# For broadcast(x, dimensions=[0]) from [M] to [M, N]
x sharding: P('data')
Output sharding: P('data', None)  # N is broadcast, so it's replicated
```

**Reshape**:
```
# For reshape from [M*N] to [M, N]
x sharding: P('data')
Output sharding: P('data', None)  # The single sharded dimension maps to dim 0

# For reshape from [M, N] to [M*N]
x sharding: P('data', 'model')
Output sharding: P(('data', 'model'))  # Both dims collapse into one
```

### Conflict Resolution

When propagation produces incompatible shardings, the partitioner inserts resharding operations:

1. **Sharded to replicated**: Insert `all-gather` to collect all shards.
2. **Replicated to sharded**: No operation needed (just take the local slice).
3. **Different sharding axis**: Insert `all-to-all` to redistribute.
4. **Partial sharding**: Insert `all-reduce` for partially reduced dimensions.

## Communication Operations

### Collective Operations Generated by Partitioner

The SPMD partitioner automatically inserts collective communication operations as needed:

#### All-Reduce

Used when reducing a dimension that is sharded across devices:

```
# Before partitioning (global view):
%result = reduce(%x, init_value), dimensions={1}, to_apply=add
# x: f32[1024, 512] sharded as P('data', 'model')

# After partitioning (per-device view):
%local_x = f32[1024, 128] parameter(0)  # local shard
%local_reduce = f32[1024] reduce(%local_x, init_value), dimensions={1}, to_apply=add
%result = f32[1024] all-reduce(%local_reduce), replica_groups={{0,1,2,3}}, to_apply=add
```

The all-reduce sums the partial reductions from each device to produce the global result.

#### All-Gather

Used when a replicated tensor is needed from a sharded one:

```
# Before partitioning:
%result = add(%x, %y)
# x: f32[1024, 512] sharded as P('data', None)
# y: f32[1024, 512] sharded as P(None, 'model')  -- different sharding!

# After partitioning:
%local_y = f32[1024, 256] parameter(1)  # local shard of y
%gathered_y = f32[1024, 512] all-gather(%local_y), replica_groups={{0,1}}, dimensions={1}
%local_x = f32[256, 512] parameter(0)  # local shard of x
%result = f32[256, 512] add(%local_x, %gathered_y)
```

The all-gather collects the shards of `y` so that each device has the full tensor for the addition.

#### All-To-All

Used when redistributing a tensor from one sharding to another:

```
# Redistribute from P('data', None) to P(None, 'data')
%local_x = f32[256, 512] parameter(0)
%redistributed = f32[1024, 128] all-to-all(%local_x), 
    replica_groups={{0,1,2,3}},
    source_dim=0, target_dim=1
```

#### Collective-Permute

Used for halo exchange in convolution partitioning or for pipeline parallelism:

```
# Shift data to the right in a ring
%local = f32[256, 512] parameter(0)
%shifted = f32[256, 512] collective-permute(%local),
    source_target_pairs={{0,1}, {1,2}, {2,3}, {3,0}}
```

#### Reduce-Scatter

Used as an optimization to combine all-reduce with sharding:

```
# Instead of all-reduce followed by sharding,
# use reduce-scatter for better memory efficiency
%partial_sum = f32[1024] parameter(0)
%result = f32[256] reduce-scatter(%partial_sum), 
    replica_groups={{0,1,2,3}}, 
    to_apply=add
```

### Overlap of Computation and Communication

The SPMD partitioner attempts to overlap communication with independent computation to hide latency:

1. **Schedule analysis**: The partitioner analyzes the data dependency graph to find operations that are independent of communication.

2. **Async communication**: Communication operations are split into start/done pairs:
   ```
   %async_start = all-gather-start(%local_data)
   // ... independent computation here ...
   %gathered = all-gather-done(%async_start)
   ```

3. **Overlap window**: The independent computation executes while the communication is in flight, hiding the communication latency behind useful work.

```
Timeline (without overlap):
  |--communication--|--computation--|

Timeline (with overlap):
  |--communication--|
  |--independent computation--|
  |--dependent computation--|

Total time: max(communication, independent) + dependent
```

## Example Workflow

### JAX Program with Sharding

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

# Set up 4 devices in a 1D mesh
devices = jax.devices()[:4]
mesh = Mesh(jax.numpy.array(devices), axis_names=('data',))

# Define a simple transformer layer with data parallelism
@jax.jit(
    in_shardings=NamedSharding(mesh, P('data', None)),
    out_shardings=NamedSharding(mesh, P('data', None))
)
def transformer_layer(x, wq, wk, wv, wo):
    """Simple transformer attention layer with data parallelism."""
    # Each device has a shard of x along the batch dimension
    # wq, wk, wv, wo are replicated (all devices have full weights)

    # QKV projection
    q = jnp.dot(x, wq)  # [batch*seq, d_model] x [d_model, d_k] -> [batch*seq, d_k]
    k = jnp.dot(x, wk)
    v = jnp.dot(x, wv)

    # Attention
    scores = jnp.dot(q, k.T) / jnp.sqrt(d_k)
    attn = jax.nn.softmax(scores, axis=-1)
    attn_out = jnp.dot(attn, v)

    # Output projection
    output = jnp.dot(attn_out, wo)
    return output
```

### HLO with Sharding Custom Calls

When JAX compiles this function, it generates HLO with sharding annotations encoded as custom calls:

```
HloModule transformer_layer, entry_computation_layout={(f32[1024,512]{1,0}, f32[512,64]{1,0}, f32[512,64]{1,0}, f32[512,64]{1,0}, f32[64,512]{1,0})->f32[1024,512]{1,0}}

ENTRY main {
  %x = f32[1024,512] parameter(0)
  %wq = f32[512,64] parameter(1)
  %wk = f32[512,64] parameter(2)
  %wv = f32[512,64] parameter(3)
  %wo = f32[64,512] parameter(4)

  // Sharding annotations (encoded as custom calls)
  %x_sharded = f32[1024,512] custom-call(%x), custom_call_target="Sharding",
      sharding={devices=[4,1]<=[4]}

  %wq_sharded = f32[512,64] custom-call(%wq), custom_call_target="Sharding",
      sharding={replicated}

  // QKV projections
  %q = f32[1024,64] dot(%x_sharded, %wq_sharded),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %k = f32[1024,64] dot(%x_sharded, %wk_sharded),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %v = f32[1024,64] dot(%x_sharded, %wv_sharded),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}

  // Attention computation
  %k_t = f32[64,1024] transpose(%k), dimensions={1,0}
  %scores = f32[1024,1024] dot(%q, %k_t),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %scaled = f32[1024,1024] multiply(%scores, constant(0.125))
  %attn = f32[1024,1024] softmax(%scaled)
  %attn_out = f32[1024,64] dot(%attn, %v),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}

  // Output projection
  %output = f32[1024,512] dot(%attn_out, %wo_sharded),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}

  ROOT %result = %output
}
```

### Partitioned HLO with Collectives

After the SPMD partitioner runs, the HLO is transformed to operate on local shards:

```
HloModule transformer_layer_partitioned

ENTRY main {
  // Local shard of x: [1024/4, 512] = [256, 512]
  %x_local = f32[256,512] parameter(0)

  // Replicated weights (same on all devices)
  %wq_local = f32[512,64] parameter(1)
  %wk_local = f32[512,64] parameter(2)
  %wv_local = f32[512,64] parameter(3)
  %wo_local = f32[64,512] parameter(4)

  // QKV projections: no communication needed
  // x is sharded along batch, weights are replicated
  // Each device computes its local QKV
  %q_local = f32[256,64] dot(%x_local, %wq_local),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %k_local = f32[256,64] dot(%x_local, %wk_local),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %v_local = f32[256,64] dot(%x_local, %wv_local),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}

  // Attention: need full k and v for attention scores
  // All-gather k and v across devices to get the full sequence
  %k_gathered = f32[1024,64] all-gather(%k_local),
      replica_groups={{0,1,2,3}}, all_gather_dim=0
  %v_gathered = f32[1024,64] all-gather(%v_local),
      replica_groups={{0,1,2,3}}, all_gather_dim=0

  // Compute attention with full k, v
  %k_t = f32[64,1024] transpose(%k_gathered), dimensions={1,0}
  %scores = f32[256,1024] dot(%q_local, %k_t),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %scaled = f32[256,1024] multiply(%scores, constant(0.125))
  %attn = f32[256,1024] softmax(%scaled)
  %attn_out = f32[256,64] dot(%attn, %v_gathered),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}

  // Output projection: no communication needed
  %output_local = f32[256,512] dot(%attn_out, %wo_local),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}

  ROOT %result = %output_local
}
```

Key observations about the partitioned HLO:

1. **Input shapes are reduced**: The local shard of `x` is `[256, 512]` instead of `[1024, 512]`.

2. **Weights are replicated**: All devices have the full weight matrices.

3. **All-gather is inserted**: To compute attention, each device needs the full `k` and `v` sequences. The partitioner inserts `all-gather` operations.

4. **Output is sharded**: The final output `[256, 512]` is the local shard of the global output `[1024, 512]`.

### Advanced: 2D Sharding (Megatron-Style)

For tensor parallelism combined with data parallelism:

```python
# 8 devices: 2-way data parallel, 4-way tensor parallel
mesh = Mesh(jax.numpy.array(devices[:8]).reshape(2, 4),
            axis_names=('data', 'model'))

# x: sharded along batch across 'data' axis
# w: sharded along input across 'model' axis
@jax.jit(
    in_shardings=(
        NamedSharding(mesh, P('data', None)),       # x: [B/2, D]
        NamedSharding(mesh, P(None, 'model')),       # w: [D, D/4]
    ),
    out_shardings=NamedSharding(mesh, P('data', 'model'))  # out: [B/2, D/4]
)
def linear(x, w):
    # Local computation: [B/2, D] x [D, D/4] = [B/2, D/4]
    local_out = jnp.dot(x, w)
    # The K dimension is sharded across 'model', so partial products
    # need to be all-reduced across the 'model' axis
    return local_out
```

The partitioned HLO for this case includes an `all-reduce` across the model dimension:

```
ENTRY main {
  %x_local = f32[B/2,D] parameter(0)      // local shard of x
  %w_local = f32[D,D/4] parameter(1)      // local shard of w
  %partial = f32[B/2,D/4] dot(%x_local, %w_local)  // partial matmul
  %result = f32[B/2,D/4] all-reduce(%partial),      // sum partial products
      replica_groups={{0,4}, {1,5}, {2,6}, {3,7}},
      to_apply=add
  ROOT %root = %result
}
```
