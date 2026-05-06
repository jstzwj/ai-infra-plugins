# Chapter 25: Pallas TPU Programming

## 25.1 TPU Architecture Overview

Google's Tensor Processing Unit (TPU) is an application-specific integrated circuit
(ASIC) designed for neural network workloads. The TPU architecture differs
fundamentally from GPU architectures, and understanding these differences is essential
for writing efficient Pallas TPU kernels.

### 25.1.1 TPU as a Sequential Wide-Vector Machine

Unlike GPUs which are SIMT (Single Instruction, Multiple Thread) processors with
thousands of concurrent threads, TPUs are **sequential machines with wide vector
registers**. The key architectural characteristics are:

- **Sequential execution**: Instructions execute one at a time (no thread-level
  parallelism within a core)
- **Wide vector registers**: Registers hold 8x128 tiles (8 rows x 128 columns)
  of data, enabling massive SIMD parallelism within each instruction
- **Systolic array**: The matrix multiply unit (MXU) uses a systolic array that
  processes 128x128 matrix blocks per cycle
- **VLIW-style**: Multiple operations can be issued per cycle (compute + memory + scalar)
- **Deterministic timing**: Execution time is predictable (no warp divergence,
  no cache misses)

### 25.1.2 TPU Core Layout

```
TPU v4 Core (TensorCore)
+-----------------------------------------------------------+
| Vector Unit                                                |
|  - 8 x 128 vector registers (VMEM-mapped)                |
|  - 8 x 128 bit in each register lane                      |
|  - 128-wide SIMD ALU operations                           |
|  - Supports: add, mul, compare, select, transpose, etc.   |
+-----------------------------------------------------------+
| Matrix Multiply Unit (MXU / Systolic Array)                |
|  - 128 x 128 systolic array                               |
|  - bfloat16 inputs, float32 accumulation                  |
|  - 128 x 128 x 128 multiply per cycle                     |
+-----------------------------------------------------------+
| Scalar Unit                                                |
|  - Scalar registers (SMEM)                                |
|  - Control flow (branching, loops)                        |
|  - Address computation                                    |
+-----------------------------------------------------------+
| Memory Subsystem                                           |
|  - VMEM (Vector Memory): 8 MB per core (TPU v4)          |
|  - HBM: 32-95 GB per chip                                 |
|  - SEM: Scalar memory for loop counters, indices          |
+-----------------------------------------------------------+
| Interconnect                                               |
|  - ICI (Inter-Chip Interconnect): 4.8 Tbps per link      |
|  - Torus topology between cores on same chip              |
|  - Links to other TPU chips in pod                        |
+-----------------------------------------------------------+
```

### 25.1.3 TPU Generations

| Feature | TPU v4 | TPU v5e | TPU v5p | TPU Trillium |
|---|---|---|---|---|
| MXU size | 128x128 | 128x128 | 128x128 | 256x256 |
| VMEM per core | 8 MB | 16 MB | 95 GB HBM | Larger |
| BF16 TOPS/chip | 275 | 197 | 459 | Higher |
| HBM capacity | 32 GB | 16 GB | 95 GB | Larger |
| HBM bandwidth | 1.2 TB/s | 0.8 TB/s | 2.6 TB/s | Higher |
| ICI bandwidth | 4.8 Tbps | 1.6 Tbps | 4.8 Tbps | Higher |
| Cores/chip | 2 (v4) | 1 | 1 | More |

### 25.1.4 Key Differences from GPU Programming

| Aspect | GPU | TPU |
|---|---|---|
| Programming model | SIMT (many threads) | Sequential + wide vector |
| Memory model | HBM, SMEM, registers | HBM, VMEM, SMEM |
| Matrix multiply | Tensor Cores (MMA) | Systolic array (MXU) |
| Synchronization | Barriers, atomics | Sequential (implicit ordering) |
| Programming difficulty | High (race conditions) | Lower (sequential model) |
| Parallelism | Thread-level parallelism | Data parallelism within vector ops |
| Register width | 32-bit per thread | 8x128 tiles per register |

---

## 25.2 Array Layouts on TPU

### 25.2.1 VMEM Tile Layout

The fundamental data unit in TPU vector processing is an 8x128 tile. VMEM stores
data as a collection of these tiles. The last two dimensions of any array map onto
the 8x128 tile structure.

```
VMEM Layout for array shape (M, N):
- M dimension: groups of 8 rows (M must be padded to multiple of 8)
- N dimension: groups of 128 columns (N must be padded to multiple of 128)

Example: Shape (16, 256)
- 2 tiles in M dimension (16 / 8 = 2)
- 2 tiles in N dimension (256 / 128 = 2)
- Total VMEM tiles: 2 * 2 = 4 tiles
```

### 25.2.2 Layout in Pallas

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# On TPU, the last two dimensions of arrays are mapped to 8x128 register tiles
# For a matrix of shape (M, N):
#   M rounds up to next multiple of 8
#   N rounds up to next multiple of 128

# Array with shape compatible with TPU tile layout
x = jnp.ones((16, 128), dtype=jnp.bfloat16)   # Fits in 2 VMEM tiles
y = jnp.ones((8, 256), dtype=jnp.bfloat16)    # Fits in 2 VMEM tiles
z = jnp.ones((24, 384), dtype=jnp.bfloat16)   # Fits in 3 * 3 = 9 VMEM tiles
```

### 25.2.3 Tiling Requirements

The systolic array operates on 128x128 blocks. For efficient matrix multiplication:
- The contracted dimension (K) should be a multiple of 128
- The M dimension should be a multiple of 8
- The N dimension should be a multiple of 128

```python
# Optimal shapes for TPU matmul
# A: (M, K) where M % 8 == 0, K % 128 == 0
# B: (K, N) where N % 128 == 0
# C: (M, N) result

def check_tpu_compat(shape, name="array"):
    M, N = shape[-2], shape[-1]
    if M % 8 != 0:
        print(f"Warning: {name} M={M} not multiple of 8")
    if N % 128 != 0:
        print(f"Warning: {name} N={N} not multiple of 128")
```

---

## 25.3 Grid Processing on TPU

### 25.3.1 Lexicographic Sequential Order

On TPU, grid programs execute in **lexicographic (sequential) order**. This is a
fundamental difference from GPUs, where programs execute concurrently.

For a 2D grid `(M, N)`:
- Programs execute as: (0,0), (0,1), ..., (0,N-1), (1,0), (1,1), ..., (M-1,N-1)
- Only one program runs at a time per TPU core
- No synchronization needed between programs (sequential ordering is guaranteed)

```python
def sequential_grid_kernel(x_ref, o_ref):
    """On TPU, this executes sequentially across grid points."""
    i = pl.program_id(0)
    j = pl.program_id(1)

    # No race conditions possible - only one program runs at a time
    x = x_ref[...]
    o_ref[...] = x * 2.0
```

### 25.3.2 Implications of Sequential Execution

Advantages:
- **No race conditions**: Each program completes before the next starts
- **Deterministic results**: Execution order is always the same
- **Simpler programming model**: No need for atomics or barriers
- **Predictable memory usage**: Only one program's data is live at a time

Disadvantages:
- **Lower throughput for small tiles**: Sequential execution limits parallelism
- **Must use large enough tiles** to amortize kernel launch overhead
- **Cannot overlap computation** between grid points (must use software pipelining)

---

## 25.4 Multicore TPU Programming

Modern TPU chips have multiple cores. Pallas can leverage multiple cores through
JAX's standard distributed computing APIs.

### 25.4.1 Multi-Core Execution via shard_map

```python
from jax.experimental import shard_map
import jax.sharding as jsharding

def multi_core_matmul(a: jax.Array, b: jax.Array) -> jax.Array:
    """Run matmul across multiple TPU cores using shard_map."""
    M, K = a.shape
    _, N = b.shape

    devices = jax.devices()
    mesh = jsharding.Mesh(devices, ("cores",))

    def matmul_shard(a_shard, b_shard):
        return jnp.dot(a_shard, b_shard)

    # Shard A along M dimension across cores
    # Replicate B across all cores (or shard along K/N as needed)
    a_spec = jsharding.PartitionSpec("cores", None)
    b_spec = jsharding.PartitionSpec(None, None)  # Replicated

    sharded_fn = shard_map.shard_map(
        matmul_shard,
        mesh=mesh,
        in_specs=(a_spec, b_spec),
        out_specs=jsharding.PartitionSpec("cores", None),
    )

    return sharded_fn(a, b)
```

### 25.4.2 Core-Level Kernel Launch

Each TPU core independently executes its portion of the grid. When using
`pallas_call` within `shard_map`, each core runs the kernel on its local data.

```python
def per_core_pallas_fn(a_shard, b_shard):
    """Each TPU core runs this Pallas kernel on its local shard."""
    return tpu_matmul(a_shard, b_shard)

sharded_matmul = shard_map.shard_map(
    per_core_pallas_fn,
    mesh=mesh,
    in_specs=(jsharding.PartitionSpec("cores", None), jsharding.PartitionSpec(None, None)),
    out_specs=jsharding.PartitionSpec("cores", None),
)
```

---

## 25.5 SMEM for Scalar Operations

TPU SMEM (Scalar Memory) stores scalar values used for control flow, index
computation, and scalar operations. It is separate from VMEM (Vector Memory).

### 25.5.1 Scalar vs Vector Operations

```python
def scalar_vector_kernel(x_ref, o_ref):
    """Example showing scalar and vector operations on TPU."""
    # Scalar operations (in SMEM)
    i = pl.program_id(0)           # Scalar integer
    scale = jnp.float32(2.0)       # Scalar constant

    # Vector operations (in VMEM)
    x = x_ref[...]                 # Vector: (8, 128) tile

    # Scalar-vector interaction: broadcast scalar across vector
    o_ref[...] = x * scale         # Vector result
```

### 25.5.2 SMEM in Block-Sparse Kernels

SMEM is particularly useful for storing block-sparse metadata (indices of non-zero
blocks) that control which blocks are processed.

```python
def sparse_meta_kernel(x_ref, mask_ref, o_ref):
    """Process only blocks indicated by the mask."""
    mask = mask_ref[...]  # Scalar: 0 or 1

    # Conditional: only compute if this block is non-zero
    if mask:
        x = x_ref[...]
        o_ref[...] = jnp.dot(x, x.T)
    else:
        o_ref[...] = jnp.zeros_like(o_ref[...])
```

---

## 25.6 PrefetchScalarGridSpec for Block-Sparse Kernels

`PrefetchScalarGridSpec` is a specialized grid specification for block-sparse
computations on TPU. It prefetches scalar metadata (e.g., block indices) into SMEM
before executing the vector kernel, enabling efficient sparse processing.

### 25.6.1 Motivation

In block-sparse operations, only a subset of blocks contain non-zero data. Processing
all blocks (including zero blocks) wastes computation and memory bandwidth.
`PrefetchScalarGridSpec` allows the kernel to skip zero blocks entirely.

### 25.6.2 API

```python
from jax.experimental.pallas.tpu import PrefetchScalarGridSpec

# Standard GridSpec: processes all grid points
grid_spec = pl.GridSpec(grid=(M // BM, N // BN))

# PrefetchScalarGridSpec: processes only grid points with valid metadata
sparse_grid_spec = PrefetchScalarGridSpec(
    grid=(num_nonzero_blocks,),     # Grid size = number of non-zero blocks
    prefetch_refs={
        # Prefetch scalar metadata into SMEM before kernel execution
        "row_idx": pl.BlockSpec((), lambda i: (i,)),   # Row block index
        "col_idx": pl.BlockSpec((), lambda i: (i,)),   # Column block index
    },
)
```

### 25.6.3 Block-Sparse Matmul with PrefetchScalarGridSpec

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas.tpu import PrefetchScalarGridSpec

BM, BN, BK = 128, 128, 128

def block_sparse_matmul_kernel(
    a_ref: pl.Ref,        # Full A matrix or blocked
    b_ref: pl.Ref,        # Full B matrix or blocked
    row_idx_ref: pl.Ref,  # Prefetched: row block index (scalar in SMEM)
    col_idx_ref: pl.Ref,  # Prefetched: column block index (scalar in SMEM)
    o_ref: pl.Ref,        # Output block
):
    """Kernel that processes one non-zero block of the output."""
    # Read prefetched scalar indices from SMEM
    row_block = row_idx_ref[()]    # Scalar integer
    col_block = col_idx_ref[()]    # Scalar integer

    # Accumulate across K dimension for this (row_block, col_block) output tile
    acc = jnp.zeros((BM, BN), dtype=jnp.float32)

    for k_block in range(K // BK):
        # Load the appropriate tiles of A and B using the block indices
        a_tile = a_ref[row_block * BM:(row_block + 1) * BM,
                       k_block * BK:(k_block + 1) * BK]
        b_tile = b_ref[k_block * BK:(k_block + 1) * BK,
                       col_block * BN:(col_block + 1) * BN]
        acc += jnp.dot(a_tile, b_tile)

    # Write to the appropriate output block
    o_ref[row_block * BM:(row_block + 1) * BM,
          col_block * BN:(col_block + 1) * BN] = acc

def block_sparse_matmul(
    a: jax.Array,
    b: jax.Array,
    row_indices: jax.Array,   # (num_blocks,) int32: row block indices
    col_indices: jax.Array,   # (num_blocks,) int32: column block indices
) -> jax.Array:
    M, K = a.shape
    _, N = b.shape
    num_blocks = row_indices.shape[0]

    # Use PrefetchScalarGridSpec to only process non-zero blocks
    sparse_grid = PrefetchScalarGridSpec(
        grid=(num_blocks,),
        prefetch_refs={
            "row_idx": pl.BlockSpec((), lambda i: (i,)),
            "col_idx": pl.BlockSpec((), lambda i: (i,)),
        },
    )

    # Note: This is a conceptual example showing the API pattern
    # Actual implementation would use the sparse grid in pallas_call
    return jnp.zeros((M, N), dtype=jnp.float32)
```

---

## 25.7 TPU Matrix Multiplication

### 25.7.1 Basic TPU Matmul

The TPU systolic array (MXU) performs 128x128 matrix multiply-accumulate operations
in bfloat16 with float32 accumulation.

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

# Tile sizes matching TPU MXU dimensions
TM = 8      # M tile (vector register row dimension)
TN = 128    # N tile (vector register column dimension)
TK = 128    # K tile (MXU systolic array dimension)

def tpu_matmul_kernel(a_ref, b_ref, c_ref):
    """Basic TPU matrix multiplication kernel.

    Uses the MXU systolic array via jnp.dot, which Pallas compiles to
    TPU dot primitives.
    """
    # Initialize accumulator
    acc = jnp.zeros((TM, TN), dtype=jnp.float32)

    # Accumulate over K tiles
    for k in range(K // TK):
        # Load tiles from VMEM
        a_tile = a_ref[:, k*TK:(k+1)*TK]    # (TM, TK) bfloat16
        b_tile = b_ref[k*TK:(k+1)*TK, :]    # (TK, TN) bfloat16

        # TPU systolic array multiply-accumulate
        # This compiles to TPU dot instruction using the MXU
        acc += jnp.dot(a_tile, b_tile)

    c_ref[...] = acc

def tpu_matmul(a: jax.Array, b: jax.Array) -> jax.Array:
    """Matrix multiplication using Pallas on TPU."""
    M, K = a.shape
    _, N = b.shape
    assert M % TM == 0, f"M={M} must be divisible by {TM}"
    assert N % TN == 0, f"N={N} must be divisible by {TN}"
    assert K % TK == 0, f"K={K} must be divisible by {TK}"

    return pl.pallas_call(
        tpu_matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
        grid=(M // TM, N // TN),
        in_specs=[
            pl.BlockSpec((TM, None), lambda i, j: (i, 0)),   # A rows
            pl.BlockSpec((None, TN), lambda i, j: (0, j)),   # B columns
        ],
        out_specs=pl.BlockSpec((TM, TN), lambda i, j: (i, j)),
    )(a, b)

# Usage
M, K, N = 1024, 1024, 1024
key = jax.random.PRNGKey(0)
a = jax.random.normal(key, (M, K), dtype=jnp.bfloat16)
b = jax.random.normal(jax.random.fold_in(key, 1), (K, N), dtype=jnp.bfloat16)
c = tpu_matmul(a, b)

# Verify
expected = jnp.dot(a, b)
print(f"Max error: {jnp.max(jnp.abs(c - expected)):.4f}")
```

### 25.7.2 Optimized TPU Matmul with VMEM Management

```python
def optimized_tpu_matmul_kernel(a_ref, b_ref, c_ref):
    """Optimized matmul with explicit VMEM tile management."""
    # VMEM tiles for double-buffered A
    a_tile_curr = pltpu.VMEM((TM, TK), jnp.bfloat16)
    a_tile_next = pltpu.VMEM((TM, TK), jnp.bfloat16)
    b_tile_curr = pltpu.VMEM((TK, TN), jnp.bfloat16)
    b_tile_next = pltpu.VMEM((TK, TN), jnp.bfloat16)

    # Prologue: load first tiles
    a_tile_curr[...] = a_ref[:, 0:TK]
    b_tile_curr[...] = b_ref[0:TK, :]

    acc = jnp.zeros((TM, TN), jnp.float32)

    num_k_tiles = K // TK
    for k in range(num_k_tiles):
        # Prefetch next tiles (overlapped with computation on TPU)
        if k + 1 < num_k_tiles:
            a_tile_next[...] = a_ref[:, (k+1)*TK:(k+2)*TK]
            b_tile_next[...] = b_ref[(k+1)*TK:(k+2)*TK, :]

        # Compute: systolic array multiply-accumulate
        acc += jnp.dot(a_tile_curr[...], b_tile_curr[...])

        # Swap buffers
        a_tile_curr, a_tile_next = a_tile_next, a_tile_curr
        b_tile_curr, b_tile_next = b_tile_next, b_tile_curr

    c_ref[...] = acc
```

---

## 25.8 Software Pipelining on TPU

### 25.8.1 TPU Pipeline Overview

TPU software pipelining overlaps memory loads with computation. Since TPU is a
sequential machine, pipelining is achieved through VLIW-style instruction scheduling
where load and compute instructions execute concurrently on different functional units.

### 25.8.2 Pipeline Stages

```python
from jax.experimental.pallas.tpu import pipeline

def pipelined_tpu_matmul_kernel(a_ref, b_ref, c_ref):
    """TPU matmul with software pipelining."""
    acc = jnp.zeros((TM, TN), jnp.float32)

    def pipeline_body(k, acc, a_buf, b_buf):
        # Stage 1: Compute on current buffer (systolic array)
        acc = acc + jnp.dot(a_buf[...], b_buf[...])

        # Stage 2: Load next buffer (memory unit, overlapped with compute)
        if k + 1 < K // TK:
            a_buf[...] = a_ref[:, (k+1)*TK:(k+2)*TK]
            b_buf[...] = b_ref[(k+1)*TK:(k+2)*TK, :]

        return acc, a_buf, b_buf

    # Allocate VMEM buffers
    a_buf = pltpu.VMEM((TM, TK), jnp.bfloat16)
    b_buf = pltpu.VMEM((TK, TN), jnp.bfloat16)

    # Load initial data
    a_buf[...] = a_ref[:, 0:TK]
    b_buf[...] = b_ref[0:TK, :]

    # Execute pipelined loop
    acc, _, _ = pipeline.emit_pipeline(
        K // TK,
        pipeline_body,
        init_state=(acc, a_buf, b_buf),
        num_stages=2,
    )

    c_ref[...] = acc
```

### 25.8.3 Three-Stage Pipeline

```python
def three_stage_pipeline_kernel(a_ref, b_ref, c_ref):
    """Three-stage pipeline: load_a, load_b, compute."""
    NUM_STAGES = 3

    a_bufs = [pltpu.VMEM((TM, TK), jnp.bfloat16) for _ in range(NUM_STAGES)]
    b_bufs = [pltpu.VMEM((TK, TN), jnp.bfloat16) for _ in range(NUM_STAGES)]

    # Prologue: fill pipeline
    for s in range(min(NUM_STAGES, K // TK)):
        a_bufs[s][...] = a_ref[:, s*TK:(s+1)*TK]
        b_bufs[s][...] = b_ref[s*TK:(s+1)*TK, :]

    acc = jnp.zeros((TM, TN), jnp.float32)

    # Steady state
    for k in range(K // TK):
        stage = k % NUM_STAGES
        acc += jnp.dot(a_bufs[stage][...], b_bufs[stage][...])

        # Prefetch into this buffer for future use
        future_k = k + NUM_STAGES
        if future_k < K // TK:
            a_bufs[stage][...] = a_ref[:, future_k*TK:(future_k+1)*TK]
            b_bufs[stage][...] = b_ref[future_k*TK:(future_k+1)*TK, :]

    c_ref[...] = acc
```

---

## 25.9 Distributed TPU Programming

### 25.9.1 TPU Topology

TPU chips are connected in a multi-dimensional torus topology via ICI (Inter-Chip
Interconnect) links. Each TPU chip has multiple high-bandwidth ICI links to its
neighbors in each topological dimension.

```
TPU Pod (2D Torus Example)
+---+    +---+    +---+    +---+
| 0 |<-->| 1 |<-->| 2 |<-->| 3 |
+---+    +---+    +---+    +---+
  ^        ^        ^        ^
  |        |        |        |
  v        v        v        v
+---+    +---+    +---+    +---+
| 4 |<-->| 5 |<-->| 6 |<-->| 7 |
+---+    +---+    +---+    +---+
  ^        ^        ^        ^
  |        |        |        |
  v        v        v        v
+---+    +---+    +---+    +---+
| 8 |<-->| 9 |<-->|10 |<-->|11 |
+---+    +---+    +---+    +---+

Each link: ICI at 4.8 Tbps (per direction)
Topology: wraps around (torus) - chip 0 also connects to chip 3, 8
```

Common topology shapes:
- **2D torus**: `topology = (4, 4)` for 16 chips
- **3D torus**: `topology = (4, 4, 2)` for 32 chips
- **4D torus**: `topology = (4, 4, 2, 2)` for 64 chips (TPU v5p)

### 25.9.2 ND Torus and ICI Interconnect

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P

# Create mesh matching TPU topology
devices = jax.devices()  # All TPU devices
num_devices = len(devices)

# Reshape devices to match physical topology
# For a 4x4 TPU pod slice:
mesh_shape = (4, 4)
mesh = Mesh(
    jax.devices().reshape(mesh_shape),
    ("x", "y"),
)

print(f"Mesh shape: {mesh.shape}")
print(f"Devices: {num_devices}")
```

### 25.9.3 RDMA Model: Push-Only Communication

TPU ICI uses an RDMA-like push-only model:
- The sender initiates data transfer to a remote core's VMEM
- The receiver does not pull data; it waits for the sender to push
- Communication is expressed through `make_async_remote_copy` or collective ops

```python
from jax.experimental.pallas.tpu import remote_copy

def push_example_kernel(local_data_ref, remote_dest_ref):
    """Push local data to a remote TPU core's VMEM."""
    # Create an async remote copy (push)
    copy = remote_copy.make_async_remote_copy(
        src_ref=local_data_ref,
        dst_ref=remote_dest_ref,
        remote_device=jax.devices()[1],  # Target device
    )

    # Start the push
    copy.start()

    # ... do other work while transfer is in flight ...

    # Wait for completion
    copy.wait()
```

### 25.9.4 DMA Semaphores and Routing

DMA (Direct Memory Access) transfers between TPU cores use semaphores for
synchronization. Data is routed through the ICI torus, with the hardware handling
routing automatically.

```python
def dma_transfer_kernel(send_buf_ref, recv_buf_ref, semaphore_ref):
    """DMA transfer with semaphore synchronization."""
    # Semaphore tracks DMA completion
    # 0 = not started, 1 = in progress, 2 = completed

    # Sender side:
    semaphore_ref[()] = 1  # Signal start
    # DMA engine handles the transfer through ICI
    # Hardware routes through optimal torus path

    # Receiver side (different core):
    # Wait until semaphore indicates completion
    # semaphore >= 2 means data is ready
```

### 25.9.5 Collective Operations

#### ppermute: Permutation Communication

```python
from jax.lax import ppermute

# Permute data across devices in a ring
# Each device sends its data to the next device
def ring_permute(x, mesh, axis_name):
    """Ring permutation: each device sends to its right neighbor."""
    num_devices = mesh.shape[axis_name]
    perm = [(i, (i + 1) % num_devices) for i in range(num_devices)]
    return ppermute(x, perm=perm, axis_name=axis_name)

# Usage
with mesh:
    result = ring_permute(x, mesh, "x")
```

#### all_gather: Gather All Shards

```python
from jax.lax import all_gather

def gather_all_shards(x, mesh, axis_name):
    """Gather all shards along an axis."""
    return all_gather(x, axis_name=axis_name, tiled=True)

# Usage: each device gets a copy of all shards
with mesh:
    full_data = gather_all_shards(local_shard, mesh, "x")
```

#### all-reduce: Reduce Across Devices

```python
from jax.lax import psum

def all_reduce_sum(x, mesh, axis_name):
    """Sum-reduce across all devices along axis."""
    return psum(x, axis_name=axis_name)

# Ring-based all-reduce for efficiency
def ring_all_reduce(x, mesh, axis_name):
    """Efficient ring all-reduce using reduce-scatter + all-gather."""
    num_devices = mesh.shape[axis_name]

    # Phase 1: Reduce-scatter (ring)
    chunk_size = x.shape[0] // num_devices
    acc = x
    for step in range(num_devices - 1):
        # Send chunk to right neighbor, receive from left neighbor
        shifted = ppermute(acc, perm=[(i, (i+1) % num_devices)
                                       for i in range(num_devices)],
                           axis_name=axis_name)
        # Add received data to local accumulation
        acc = acc + shifted

    # acc now contains the full reduction for one chunk per device
    # Phase 2: All-gather (ring)
    # ... (similar ring pattern to distribute all chunks)

    return acc
```

### 25.9.6 Double-Buffering Technique

Double-buffering overlaps communication with computation on TPU:

```python
def double_buffer_comm_kernel(
    compute_buf_ref,    # Buffer for computation
    comm_buf_ref,       # Buffer for incoming communication
    output_ref,
):
    """Overlap ICI communication with computation using double buffering."""
    num_steps = 8

    # Start first async receive
    comm_buf_ref[...] = remote_copy.async_receive(...)

    for step in range(num_steps):
        # Wait for current receive to complete
        remote_copy.wait(comm_buf_ref)

        # Swap buffers: start receiving into compute buffer while computing
        compute_buf_ref, comm_buf_ref = comm_buf_ref, compute_buf_ref

        # Start next async receive (into the now-free buffer)
        if step + 1 < num_steps:
            comm_buf_ref[...] = remote_copy.async_receive(...)

        # Compute on current data (overlapped with next receive)
        result = jnp.dot(compute_buf_ref[...], weight_matrix)
        output_ref[step] = result
```

### 25.9.7 Bi-Directional Reduce-Scatter

Bi-directional reduce-scatter sends data in both directions simultaneously on the
torus, halving the communication time compared to unidirectional.

```python
def bidirectional_reduce_scatter(x, mesh, axis_name):
    """Bi-directional reduce-scatter using both torus directions.

    Instead of sending data only clockwise, we send half the chunks clockwise
    and half counterclockwise, utilizing both directions of each ICI link.
    """
    num_devices = mesh.shape[axis_name]
    half = num_devices // 2
    chunk_size = x.shape[0] // num_devices

    acc = x
    for step in range(half):
        # Clockwise permutation
        cw_perm = [(i, (i + 1) % num_devices) for i in range(num_devices)]
        cw_data = ppermute(acc, perm=cw_perm, axis_name=axis_name)

        # Counter-clockwise permutation
        ccw_perm = [(i, (i - 1) % num_devices) for i in range(num_devices)]
        ccw_data = ppermute(acc, perm=ccw_perm, axis_name=axis_name)

        # Accumulate from both directions
        acc = acc + cw_data + ccw_data

    return acc
```

---

## 25.10 Block-Sparse Operations (Block-COO Format)

### 25.10.1 Block-COO Format

Block-COO (Coordinate Format) represents sparse matrices as a list of non-zero
blocks, each identified by its block row and column coordinates.

```python
import jax
import jax.numpy as jnp

# Dense matrix with 50% sparsity at the block level
# Matrix shape: (1024, 1024), block size: 128x128
BLOCK_SIZE = 128
M, N = 1024, 1024
num_blocks_m = M // BLOCK_SIZE
num_blocks_n = N // BLOCK_SIZE

# Block-COO representation
# row_indices: which row block each non-zero block is in
# col_indices: which column block each non-zero block is in
# values: the actual data for each non-zero block

# Example: diagonal blocks only (diagonal sparsity pattern)
num_nonzero = num_blocks_m  # One block per row
row_indices = jnp.arange(num_nonzero, dtype=jnp.int32)
col_indices = jnp.arange(num_nonzero, dtype=jnp.int32)
values = jax.random.normal(jax.random.PRNGKey(0),
                            (num_nonzero, BLOCK_SIZE, BLOCK_SIZE),
                            dtype=jnp.bfloat16)

# Convert block-COO to dense (for verification)
def block_coo_to_dense(row_indices, col_indices, values, shape, block_size):
    dense = jnp.zeros(shape, dtype=values.dtype)
    for idx in range(len(row_indices)):
        r = row_indices[idx]
        c = col_indices[idx]
        dense = dense.at[
            r*block_size:(r+1)*block_size,
            c*block_size:(c+1)*block_size
        ].set(values[idx])
    return dense
```

### 25.10.2 Block-Sparse Matmul in Pallas

```python
from jax.experimental import pallas as pl
from jax.experimental.pallas.tpu import PrefetchScalarGridSpec

BM, BN, BK = 128, 128, 128

def bcoo_matmul_kernel(
    a_ref: pl.Ref,         # Full A matrix (M, K)
    b_values_ref: pl.Ref,  # B block values (BM, BK) for this block
    row_idx_ref: pl.Ref,   # Prefetched row block index
    col_idx_ref: pl.Ref,   # Prefetched column block index
    o_ref: pl.Ref,         # Output block (BM, BN)
):
    """Multiply A @ B_sparse where B_sparse is in Block-COO format."""
    row_block = row_idx_ref[()]
    col_block = col_idx_ref[()]

    # Load corresponding A tile
    a_tile = a_ref[row_block * BM:(row_block + 1) * BM, :]  # (BM, K)

    acc = jnp.zeros((BM, BN), dtype=jnp.float32)

    # Accumulate: A_tile @ B_block for this non-zero B block
    for k in range(K // BK):
        a_k = a_tile[:, k*BK:(k+1)*BK]
        b_k = b_values_ref[k*BK:(k+1)*BK, :]
        acc += jnp.dot(a_k, b_k)

    # Scatter-add to output (multiple B blocks may contribute to same output tile)
    o_ref[...] = o_ref[...] + acc

def block_sparse_matmul(
    a: jax.Array,              # (M, K) dense
    b_row_indices: jax.Array,  # (num_blocks,) int32
    b_col_indices: jax.Array,  # (num_blocks,) int32
    b_values: jax.Array,       # (num_blocks, BM, BN) bfloat16
) -> jax.Array:
    M, K = a.shape
    num_blocks = b_row_indices.shape[0]

    # Initialize output
    o = jnp.zeros((M, N), dtype=jnp.float32)

    # Process each non-zero block
    # In practice, this would use PrefetchScalarGridSpec for efficiency
    for idx in range(num_blocks):
        r = b_row_indices[idx]
        c = b_col_indices[idx]
        b_block = b_values[idx]  # (BM, BN)

        a_tile = a[r*BM:(r+1)*BM, :]  # (BM, K)
        result_tile = jnp.dot(a_tile, b_block)  # (BM, BN)
        o = o.at[r*BM:(r+1)*BM, c*BN:(c+1)*BN].add(result_tile)

    return o
```

### 25.10.3 Block-Sparse Attention

```python
def block_sparse_attention_kernel(
    q_ref: pl.Ref,         # Query: (BM, D)
    k_ref: pl.Ref,         # Key: (num_blocks, BK, D)
    v_ref: pl.Ref,         # Value: (num_blocks, BK, D)
    row_idx_ref: pl.Ref,   # Block row index for Q
    col_idx_ref: pl.Ref,   # Block col index for K/V
    o_ref: pl.Ref,         # Output: (BM, D)
):
    """Block-sparse attention: Q @ K^T for selected blocks only."""
    q = q_ref[...]  # (BM, D)
    row_idx = row_idx_ref[()]
    col_idx = col_idx_ref[()]

    # Load the specific K, V block
    k_block = k_ref[col_idx]  # (BK, D)
    v_block = v_ref[col_idx]  # (BK, D)

    # Compute attention scores
    scores = jnp.dot(q, k_block.T)   # (BM, BK)
    scores = scores / jnp.sqrt(D).astype(jnp.float32)

    # Softmax (with online-softmax for accumulation across blocks)
    max_score = jnp.max(scores, axis=-1, keepdims=True)
    exp_scores = jnp.exp(scores - max_score)

    # Weighted sum of values
    attn_out = jnp.dot(exp_scores, v_block)  # (BM, D)

    # Accumulate with running sum for online softmax
    o_ref[...] = attn_out  # Would need online softmax accumulation in practice
```

---

## 25.11 core_map for Per-Core Programming

### 25.11.1 Overview

`core_map` is a Pallas TPU primitive that allows programming each TPU core
independently. Unlike `shard_map` which provides an SPMD model across cores,
`core_map` lets you specify different behavior for each core.

```python
from jax.experimental.pallas.tpu import core_map

def per_core_fn(core_id, *args):
    """Function that runs on each TPU core with its own core_id."""
    if core_id == 0:
        # Core 0: load and preprocess data
        return preprocess(args[0])
    elif core_id == 1:
        # Core 1: compute forward pass
        return forward_pass(args[0])
    else:
        # Other cores: compute attention
        return attention(args[0])

# Map function across all TPU cores
result = core_map(per_core_fn, inputs)
```

### 25.11.2 core_map with Different Work Per Core

```python
def heterogeneous_core_example():
    """Example: core 0 produces data, cores 1-3 consume it."""
    devices = jax.devices()
    num_cores = len(devices)

    def core_fn(core_id, input_data):
        if core_id == 0:
            # Producer core: compute and distribute
            partial = jnp.dot(input_data, weight_a)
            return partial
        else:
            # Consumer cores: receive and further process
            partial = receive_from_core(0)
            result = jnp.dot(partial, weight_b)
            return result

    mesh = jsharding.Mesh(devices, ("core",))
    with mesh:
        result = core_map(core_fn, x)
```

### 25.11.3 SparseCore Mapping

TPU v5 and later chips include dedicated SparseCore units that accelerate embedding
lookups and sparse operations. Pallas TPU provides primitives for mapping operations
to SparseCore units.

```python
from jax.experimental.pallas.tpu import sparse_core

def sparse_embedding_lookup(
    indices: jax.Array,      # (batch_size,) int32
    embedding_table: jax.Array,  # (vocab_size, embed_dim) float32
) -> jax.Array:
    """Embedding lookup using TPU SparseCore units."""
    # SparseCore handles sparse gather operations efficiently
    # The SparseCore unit has dedicated hardware for:
    # - Sparse gather (embedding lookup)
    # - Sparse scatter-add (gradient update)
    # - Feature crossing

    # Standard dense embedding lookup (falls back to dense path if no SparseCore)
    embeddings = embedding_table[indices]

    return embeddings

def sparse_gradient_update(
    grad_embeddings: jax.Array,   # (batch_size, embed_dim)
    indices: jax.Array,           # (batch_size,) int32
    embedding_table: jax.Array,   # (vocab_size, embed_dim)
) -> jax.Array:
    """Update embedding table using SparseCore scatter-add."""
    # Scatter-add gradients to the embedding table
    # SparseCore accelerates this by handling only the touched rows
    updated_table = embedding_table.at[indices].add(grad_embeddings)
    return updated_table
```

### 25.11.4 SparseCore for Recommendation Systems

```python
def sparse_mlp_layer(
    dense_features: jax.Array,     # (batch, dense_dim)
    sparse_indices: jax.Array,     # (batch, num_sparse_features)
    embedding_tables: list[jax.Array],  # List of (vocab_size, embed_dim)
    weight: jax.Array,             # (total_input_dim, hidden_dim)
) -> jax.Array:
    """MLP layer with sparse embedding features processed by SparseCore."""
    # SparseCore processes embedding lookups in parallel with dense features
    embeddings = []
    for i, table in enumerate(embedding_tables):
        emb = table[sparse_indices[:, i]]  # (batch, embed_dim)
        embeddings.append(emb)

    # Concatenate sparse embeddings with dense features
    sparse_concat = jnp.concatenate(embeddings, axis=-1)
    combined = jnp.concatenate([dense_features, sparse_concat], axis=-1)

    # Dense MLP computation on systolic array
    return jnp.dot(combined, weight)
```

---

## 25.12 Complete TPU Programming Examples

### 25.12.1 Fused Layer Normalization

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def layernorm_kernel(
    x_ref: pl.Ref,       # (BLOCK_M, N)
    gamma_ref: pl.Ref,   # (N,)
    beta_ref: pl.Ref,    # (N,)
    o_ref: pl.Ref,       # (BLOCK_M, N)
):
    """Fused layer normalization kernel for TPU."""
    x = x_ref[...]                       # (BLOCK_M, N)
    gamma = gamma_ref[...]               # (N,)
    beta = beta_ref[...]                 # (N,)

    eps = 1e-5

    # Compute mean and variance along last dimension
    mean = jnp.mean(x, axis=-1, keepdims=True)      # (BLOCK_M, 1)
    var = jnp.var(x, axis=-1, keepdims=True)         # (BLOCK_M, 1)

    # Normalize
    x_norm = (x - mean) / jnp.sqrt(var + eps)        # (BLOCK_M, N)

    # Scale and shift
    o_ref[...] = x_norm * gamma + beta

BLOCK_M = 8  # TPU vector register row size
N = 128      # TPU vector register column size

def fused_layernorm(x: jax.Array, gamma: jax.Array, beta: jax.Array) -> jax.Array:
    M = x.shape[0]
    assert x.shape == (M, N)
    assert M % BLOCK_M == 0

    return pl.pallas_call(
        layernorm_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
        grid=(M // BLOCK_M,),
        in_specs=[
            pl.BlockSpec((BLOCK_M, N), lambda i: (i, 0)),
            pl.BlockSpec((N,), lambda i: (0,)),          # Broadcast
            pl.BlockSpec((N,), lambda i: (0,)),          # Broadcast
        ],
        out_specs=pl.BlockSpec((BLOCK_M, N), lambda i: (i, 0)),
    )(x, gamma, beta)

# Usage
M = 1024
x = jax.random.normal(jax.random.PRNGKey(0), (M, N))
gamma = jnp.ones((N,))
beta = jnp.zeros((N,))
result = fused_layernorm(x, gamma, beta)

# Verify against JAX
expected = jax.nn.standardize(x, axis=-1) * gamma + beta
print(f"Max error: {jnp.max(jnp.abs(result - expected)):.6f}")
```

### 25.12.2 Fused Attention (Single-Head)

```python
def fused_attention_kernel(
    q_ref: pl.Ref,     # (BLOCK_M, D)
    k_ref: pl.Ref,     # (N_CTX, D)
    v_ref: pl.Ref,     # (N_CTX, D)
    o_ref: pl.Ref,     # (BLOCK_M, D)
):
    """Fused single-head attention kernel for TPU."""
    BLOCK_M = 8
    D = 128
    BLOCK_K = 128

    q = q_ref[...]  # (BLOCK_M, D)

    # Online softmax attention: accumulate max and sum across K blocks
    running_max = jnp.full((BLOCK_M, 1), -jnp.inf, dtype=jnp.float32)
    running_sum = jnp.zeros((BLOCK_M, 1), dtype=jnp.float32)
    running_out = jnp.zeros((BLOCK_M, D), dtype=jnp.float32)

    for k_block in range(N_CTX // BLOCK_K):
        # Load K and V tiles
        k_tile = k_ref[k_block*BLOCK_K:(k_block+1)*BLOCK_K, :]  # (BLOCK_K, D)
        v_tile = v_ref[k_block*BLOCK_K:(k_block+1)*BLOCK_K, :]  # (BLOCK_K, D)

        # Compute attention scores
        scores = jnp.dot(q, k_tile.T)  # (BLOCK_M, BLOCK_K)
        scores = scores / jnp.sqrt(D).astype(jnp.float32)

        # Online softmax update
        block_max = jnp.max(scores, axis=-1, keepdims=True)  # (BLOCK_M, 1)
        new_max = jnp.maximum(running_max, block_max)

        # Rescale running statistics
        exp_diff = jnp.exp(running_max - new_max)
        exp_scores = jnp.exp(scores - new_max)

        running_sum = running_sum * exp_diff + jnp.sum(exp_scores, axis=-1, keepdims=True)
        running_out = running_out * exp_diff + jnp.dot(exp_scores, v_tile)
        running_max = new_max

    # Final output
    o_ref[...] = running_out / running_sum

def fused_attention(
    q: jax.Array,   # (M, D)
    k: jax.Array,   # (N_CTX, D)
    v: jax.Array,   # (N_CTX, D)
) -> jax.Array:
    M = q.shape[0]
    N_CTX = k.shape[0]
    D = q.shape[1]

    assert M % 8 == 0
    assert D % 128 == 0
    assert N_CTX % 128 == 0

    return pl.pallas_call(
        fused_attention_kernel,
        out_shape=jax.ShapeDtypeStruct((M, D), jnp.float32),
        grid=(M // 8,),
        in_specs=[
            pl.BlockSpec((8, D), lambda i: (i, 0)),
            pl.BlockSpec((None, D), lambda i: (0, 0)),   # Full K
            pl.BlockSpec((None, D), lambda i: (0, 0)),   # Full V
        ],
        out_specs=pl.BlockSpec((8, D), lambda i: (i, 0)),
    )(q, k, v)
```

### 25.12.3 Distributed Matmul Across TPU Pod

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import shard_map

def distributed_matmul(
    a: jax.Array,   # (M, K) sharded along M
    b: jax.Array,   # (K, N) replicated
    mesh: Mesh,
) -> jax.Array:
    """Distributed matrix multiplication across a TPU mesh.

    A is sharded along M dimension across mesh.
    B is replicated on all devices.
    Result C is sharded along M dimension.
    """
    def matmul_shard(a_shard, b_shard):
        # Each device computes its local portion
        return jnp.dot(a_shard, b_shard)

    sharded_matmul = shard_map.shard_map(
        matmul_shard,
        mesh=mesh,
        in_specs=(P("data", None), P(None, None)),
        out_specs=P("data", None),
    )

    return sharded_matmul(a, b)

# Setup
devices = jax.devices()
mesh = Mesh(devices, ("data",))

M, K, N = 8192, 4096, 8192
key = jax.random.PRNGKey(0)
a = jax.random.normal(key, (M, K), dtype=jnp.bfloat16)
b = jax.random.normal(jax.random.fold_in(key, 1), (K, N), dtype=jnp.bfloat16)

# Shard A across devices
a_sharding = NamedSharding(mesh, P("data", None))
b_sharding = NamedSharding(mesh, P(None, None))

a = jax.device_put(a, a_sharding)
b = jax.device_put(b, b_sharding)

result = distributed_matmul(a, b, mesh)
```

### 25.12.4 Distributed Attention with Sequence Parallelism

```python
def distributed_sequence_attention(
    q: jax.Array,   # (M, D) sharded along M (sequence dim)
    k: jax.Array,   # (N_CTX, D) replicated
    v: jax.Array,   # (N_CTX, D) replicated
    mesh: Mesh,
) -> jax.Array:
    """Distributed attention with sequence parallelism on TPU.

    Q is sharded along the sequence dimension.
    K, V are replicated (or can be sharded with all-gather).
    Each device computes attention for its local Q shard.
    """
    def attention_shard(q_shard, k_shard, v_shard):
        # Scaled dot-product attention
        d = q_shard.shape[-1]
        scores = jnp.dot(q_shard, k_shard.T) / jnp.sqrt(d).astype(jnp.float32)
        attn_weights = jax.nn.softmax(scores, axis=-1)
        return jnp.dot(attn_weights, v_shard)

    sharded_attention = shard_map.shard_map(
        attention_shard,
        mesh=mesh,
        in_specs=(
            P("seq", None),    # Q sharded along sequence
            P(None, None),     # K replicated
            P(None, None),     # V replicated
        ),
        out_specs=P("seq", None),  # Output sharded like Q
    )

    return sharded_attention(q, k, v)
```

---

## 25.13 TPU-Specific Performance Considerations

### 25.13.1 Memory Hierarchy Optimization

```
Access Latency (approximate cycles):
  VMEM read:       ~10 cycles
  VMEM write:      ~10 cycles
  HBM read:        ~200 cycles
  HBM write:       ~200 cycles
  ICI transfer:    ~500 cycles (depends on distance)
  MXU dot:         ~1 cycle (128x128x128)

Key: Minimize HBM accesses, maximize VMEM reuse.
```

### 25.13.2 VMEM Capacity Management

```python
# VMEM capacity planning for tiled matmul
# TPU v4: 8 MB VMEM per core

# For bfloat16 (2 bytes per element):
# 8 MB = 4,194,304 elements

# Matmul with BM=128, BN=128, BK=128:
# A tile: 128 * 128 * 2 bytes = 32 KB
# B tile: 128 * 128 * 2 bytes = 32 KB
# Accumulator (float32): 128 * 128 * 4 bytes = 64 KB
# Total per tile: 128 KB

# With double buffering (2 stages):
# Total: 2 * (32 + 32) + 64 = 192 KB
# Well within 8 MB VMEM limit

# Maximum tile size for double-buffered matmul:
# VMEM / 2 / (2 bytes * 2 tiles + 4 bytes) ~= 8M / 16 = 500K elements
# i.e., can use very large tiles on TPU
```

### 25.13.3 MXU Utilization

```python
# TPU MXU operates on 128x128 blocks
# For maximum throughput, ensure dot operations use full 128x128 tiles

# Good: 128x128 dot -> full MXU utilization
result = jnp.dot(a_tile, b_tile)  # (128, 128) @ (128, 128) = (128, 128)

# OK: 8x128 dot -> one row of MXU
result = jnp.dot(a_small, b_tile)  # (8, 128) @ (128, 128) = (8, 128)

# Bad: small dimensions waste MXU cycles
result = jnp.dot(a_tiny, b_tiny)  # (3, 3) @ (3, 3) = poor utilization
```

### 25.13.4 ICI Communication Optimization

| Technique | Benefit | Description |
|---|---|---|
| Overlap compute and comm | 2x throughput | Double-buffering hides ICI latency |
| Bi-directional transfer | 2x bandwidth | Use both torus directions |
| Topology-aware sharding | Minimize hops | Place communicating shards adjacent |
| Collective pipelining | Overlap with compute | Pipeline all-reduce with computation |
| Compression | Reduce data volume | Use bfloat16 for transfers |

### 25.13.5 Performance Comparison: TPU vs GPU for Matmul

```
Matrix size: 4096 x 4096 x 4096 (FP16/BF16)

Peak TFLOPS:
  H100 GPU:    ~989 TFLOPS (dense FP16 with Tensor Core)
  TPU v4:      ~275 TFLOPS (BF16 with MXU)
  TPU v5p:     ~459 TFLOPS (BF16 with MXU)

Achievable with Pallas:
  H100 GPU:    ~900 TFLOPS (91% of peak)
  TPU v4:      ~240 TFLOPS (87% of peak)
  TPU v5p:     ~400 TFLOPS (87% of peak)

Key TPU advantages:
  - Predictable performance (no cache effects)
  - Simpler programming model (sequential)
  - Better for very large models (more HBM)

Key GPU advantages:
  - Higher peak FLOPS
  - More flexible (general-purpose Tensor Cores)
  - Larger software ecosystem
```

---

## 25.14 TPU Debugging and Profiling

### 25.14.1 Debugging Pallas TPU Kernels

```python
# Use interpret mode for debugging
def debug_tpu_kernel(x_ref, o_ref):
    x = x_ref[...]
    jax.debug.print("Input shape: {}", x.shape)
    jax.debug.print("Input sum: {}", jnp.sum(x))
    o_ref[...] = x * 2.0

# Run in emulation mode
result = pl.pallas_call(
    debug_tpu_kernel,
    out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
    grid=(x.shape[0] // 8,),
    in_specs=[pl.BlockSpec((8, x.shape[1]), lambda i: (i, 0))],
    out_specs=pl.BlockSpec((8, x.shape[1]), lambda i: (i, 0)),
    interpret=True,  # CPU emulation for debugging
)(x)
```

### 25.14.2 Performance Profiling

```python
import jax.profiler

# Profile TPU kernel execution
with jax.profiler.trace("/tmp/tpu_profile"):
    for _ in range(10):
        result = tpu_matmul(a, b)
    result.block_until_ready()

# View in TensorBoard
# tensorboard --logdir=/tmp/tpu_profile

# Timing
import time

# Warmup
_ = tpu_matmul(a, b).block_until_ready()

start = time.perf_counter()
for _ in range(100):
    result = tpu_matmul(a, b)
result.block_until_ready()
elapsed = time.perf_counter() - start

tflops = 2 * M * K * N * 100 / elapsed / 1e12
print(f"TPU Matmul Performance: {tflops:.2f} TFLOP/s")
```

### 25.14.3 Common TPU Pitfalls

| Pitfall | Symptom | Solution |
|---|---|---|
| Non-aligned dimensions | Compilation error | Pad M to multiple of 8, N/K to 128 |
| VMEM overflow | OOM or slow execution | Reduce tile sizes or number of buffers |
| BF16 precision loss | Numerical errors in output | Use float32 for accumulators |
| Poor MXU utilization | Low TFLOPS | Use full 128x128 tiles for dot ops |
| Excessive HBM traffic | Bandwidth bottleneck | Tile for VMEM reuse, pipeline loads |
| ICI hotspot | Communication bottleneck | Use topology-aware sharding |

---

## 25.15 Summary

### TPU Programming Model

| Concept | Description |
|---|---|
| **Sequential execution** | One instruction at a time, no thread parallelism |
| **Wide vector registers** | 8x128 tiles, 128-wide SIMD |
| **MXU systolic array** | 128x128 matrix multiply per cycle |
| **VMEM** | 8 MB+ fast memory per core |
| **ICI** | High-bandwidth interconnect, torus topology |
| **SparseCore** | Dedicated hardware for sparse operations (v5+) |

### Key APIs

| API | Purpose |
|---|---|
| `pallas_call` | Launch TPU kernel with grid and BlockSpec |
| `PrefetchScalarGridSpec` | Block-sparse kernel support with scalar prefetch |
| `core_map` | Per-core heterogeneous programming |
| `shard_map` | SPMD distributed computing across TPU cores |
| `remote_copy` | Direct RDMA-style ICI transfers |
| `ppermute` | Permutation-based communication |
| `all_gather` / `psum` | Collective communication primitives |

### Design Principles for TPU Kernels

1. **Match tile sizes to hardware**: M multiples of 8, N/K multiples of 128
2. **Maximize MXU utilization**: Use full 128x128 systolic array operations
3. **Reuse data in VMEM**: Keep data in VMEM across multiple operations
4. **Overlap communication and computation**: Use double-buffering and pipelining
5. **Use BF16 for data, FP32 for accumulators**: Standard mixed-precision pattern
6. **Be topology-aware**: Place communicating data on adjacent TPU cores
7. **Leverage SparseCore**: Use for embedding-heavy workloads on TPU v5+
