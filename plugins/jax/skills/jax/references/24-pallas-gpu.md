# Chapter 24: Pallas GPU Programming

## 24.1 GPU Architecture Model

Pallas GPU programming exposes a tile-based abstraction over NVIDIA GPU hardware.
Understanding the GPU architecture is essential for writing high-performance kernels.

### 24.1.1 Streaming Multiprocessors (SMs)

A modern NVIDIA GPU consists of multiple Streaming Multiprocessors (SMs), each
containing:

- **CUDA cores**: Scalar arithmetic units (FP32, INT32, FP64)
- **Tensor Cores**: Matrix multiply-accumulate units (MMA)
- **Shared memory (SMEM)**: Fast on-chip memory (up to 228 KB per SM on Hopper)
- **Register file**: Per-thread registers (256 KB per SM on Hopper)
- **L1 cache**: Closely coupled with shared memory
- **Warp schedulers**: Dispatch instructions to execution units

### 24.1.2 Execution Hierarchy

```
GPU
 |-- SM (Streaming Multiprocessor)
 |    |-- Warp Scheduler
 |    |    |-- Warp (32 threads)
 |    |    |    |-- Thread
 |    |    |-- Warpgroup (4 warps = 128 threads) [Hopper+]
 |    |-- Tensor Core (MMA unit)
 |    |-- Shared Memory (SMEM)
 |    |-- Register File
 |-- L2 Cache
 |-- HBM (High Bandwidth Memory / DRAM)
```

### 24.1.3 Warps and Warpgroups

- **Warp**: 32 threads executing in lockstep (SIMT execution)
- **Warpgroup**: 4 warps (128 threads) introduced in Hopper architecture
  - Used for WGMMA (Warp Group Matrix Multiply-Accumulate) operations
  - Each warpgroup can issue a single WGMMA instruction
  - Pallas maps a "program" to a warpgroup or CTAs (Cooperative Thread Arrays)

### 24.1.4 Tensor Cores

Tensor Cores perform matrix multiply-accumulate (MMA) operations in hardware:

| Architecture | Tensor Core | MMA Instruction | Input Types | Output Types |
|---|---|---|---|---|
| Ampere (A100) | 3rd gen | HMMA | FP16, BF16, INT8, INT4 | FP16, FP32, INT32 |
| Hopper (H100) | 4th gen | WGMMA | FP16, BF16, FP8, INT8 | FP32, FP16 |
| Blackwell (B200) | 5th gen | TCgen05 | FP16, BF16, FP8, FP4, INT8 | FP32, FP16 |

Tensor Cores operate on small matrix fragments:
- HMMA (Ampere): 16x16x16 or 16x8x16 (MxNxK)
- WGMMA (Hopper): Variable sizes up to 256x128x16 per warpgroup
- TCgen05 (Blackwell): Even larger tiles with TMEM support

---

## 24.2 Memory Hierarchy

### 24.2.1 Memory Spaces in Detail

```
+----------------------------------------------------------+
| HBM (High Bandwidth Memory) - DRAM                       |
| Capacity: 40-192 GB | Bandwidth: 1.5-8 TB/s             |
| Latency: ~200-800 cycles                                 |
+----------------------------------------------------------+
     |  TMA (Tensor Memory Accelerator) / Load/Store
     v
+----------------------------------------------------------+
| L2 Cache                                                 |
| Capacity: 40-60 MB | Shared across all SMs              |
+----------------------------------------------------------+
     |  Load/Store
     v
+----------------------------------------------------------+
| SMEM (Shared Memory) - Per SM                            |
| Capacity: 0-228 KB (configurable)                       |
| Bandwidth: ~19 TB/s | Latency: ~20-40 cycles            |
+----------------------------------------------------------+
     |  Register transfer
     v
+----------------------------------------------------------+
| Registers - Per thread / Per warp                        |
| Capacity: 256 KB total per SM (up to 255 per thread)    |
| Bandwidth: ~460 TB/s | Latency: ~1 cycle                |
+----------------------------------------------------------+

+----------------------------------------------------------+
| TMEM (Tensor Memory) - Per SM [Blackwell only]           |
| Capacity: tile-based storage for TCgen05 results         |
| Directly connected to Tensor Core pipeline               |
+----------------------------------------------------------+
```

### 24.2.2 Pallas GPU Memory Space Enums

```python
from jax.experimental.pallas import gpu as plgpu

# Memory spaces available on GPU
plgpu.MemorySpace.GMEM    # Global memory (HBM/DRAM) - largest, slowest
plgpu.MemorySpace.SMEM    # Shared memory - fast, per-SM
# Registers are implied when operating on local variables
```

### 24.2.3 Data Movement

The critical performance optimization on GPUs is minimizing HBM access by:
1. Loading data from HBM into SMEM once
2. Reusing SMEM data across multiple computations
3. Keeping intermediate results in registers

---

## 24.3 Array Layouts

Pallas GPU provides specific array layout annotations that control how data is
arranged in memory and registers for optimal Tensor Core usage.

### 24.3.1 WGMMA Layout

The `WGMMA` layout arranges data for Hopper's Warp Group MMA instructions. Data
is distributed across the 128 threads of a warpgroup in a specific swizzled pattern.

```python
from jax.experimental.pallas.gpu import layouts, mm

# WGMMA layout: optimized for Tensor Core consumption
# Used for operands fed to wgmma operations
wgmma_layout = layouts.WGMMA_LAYOUT

# Example: A matrix fragment in WGMMA layout
# Shape (M, K) where M is the warpgroup's M-dimension and K is the inner dim
```

### 24.3.2 WG_STRIDED Layout

The `WG_STRIDED` layout arranges data in a strided pattern across warpgroup threads.
This is useful for data that does not go through Tensor Cores but needs to be
efficiently distributed.

```python
# Strided layout across warpgroup
strided_layout = layouts.WG_STRIDED
```

### 24.3.3 WG_SPLAT Layout

The `WG_SPLAT` layout broadcasts (splats) a scalar or small value across all
threads in a warpgroup. Useful for bias values and scalar parameters.

```python
# Splat layout: same value replicated across all threads
splat_layout = layouts.WG_SPLAT
```

### 24.3.4 Layout Usage in Kernels

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import gpu as plgpu
from jax.experimental.pallas.gpu import layouts

def kernel_with_layouts(a_ref, b_ref, c_ref):
    # Allocate SMEM with specific layouts for Tensor Core consumption
    a_smem = plgpu.SMEM((BM, BK), jnp.float16, layouts.WGMMA_LAYOUT)
    b_smem = plgpu.SMEM((BK, BN), jnp.float16, layouts.WGMMA_LAYOUT)

    # Load from GMEM to SMEM
    a_smem[...] = a_ref[...]
    b_smem[...] = b_ref[...]

    # Compute using Tensor Cores (data is in WGMMA layout)
    acc = plgpu.wgmma(a_smem, b_smem, acc)

    c_ref[...] = acc
```

---

## 24.4 Memory Transforms

Memory transforms modify how data is arranged in shared memory to optimize access
patterns for Tensor Core consumption and avoid bank conflicts.

### 24.4.1 TileTransform

The `TileTransform` rearranges data in tiles to match the access granularity of
Tensor Core instructions.

```python
from jax.experimental.pallas.gpu import transforms

# Tile transform for SMEM layout
tile_transform = transforms.TileTransform(tile_shape=(16, 16))
```

### 24.4.2 SwizzleTransform

The `SwizzleTransform` applies XOR-based swizzling to shared memory addresses to
eliminate bank conflicts when multiple threads access shared memory simultaneously.

```python
# Swizzle transform to avoid bank conflicts
swizzle = transforms.SwizzleTransform(swizzle_bits=3)
```

Bank conflicts occur when multiple threads access different addresses within the
same 32-bit shared memory bank in the same cycle. Swizzling interleaves the data
so that consecutive threads access different banks.

### 24.4.3 TransposeTransform

The `TransposeTransform` transposes data layout in shared memory, useful when
matrix operands need to be transposed for the Tensor Core interface.

```python
# Transpose for SMEM layout
transpose = transforms.TransposeTransform()
```

### 24.4.4 Combining Transforms

```python
# Combine multiple transforms for optimal layout
combined_transform = transforms.Compose([
    transforms.SwizzleTransform(swizzle_bits=3),
    transforms.TileTransform(tile_shape=(16, 32)),
])
```

---

## 24.5 MMA Operations

Matrix Multiply-Accumulate (MMA) operations are the core of GPU compute performance.
Pallas GPU provides abstractions for Tensor Core operations across architectures.

### 24.5.1 Hopper WGMMA

Hopper (H100) introduced WGMMA (Warp Group MMA), which operates across an entire
warpgroup of 128 threads. This is the primary compute primitive for high-performance
matmul on Hopper GPUs.

```python
from jax.experimental.pallas.gpu import mm

def hopper_wgmma_example(a_ref, b_ref, c_ref):
    """Example using Hopper WGMMA for matrix multiply."""
    BM, BK = 128, 64
    BN = 128

    # Initialize accumulator in registers
    acc = jnp.zeros((BM, BN), dtype=jnp.float32)

    for k in range(K // BK):
        # Load A and B tiles from HBM to SMEM
        a_smem = a_ref[:, k*BK:(k+1)*BK]   # (BM, BK) float16
        b_smem = b_ref[k*BK:(k+1)*BK, :]   # (BK, BN) float16

        # WGMMA: accumulate C += A @ B
        # Input: float16, Accumulator: float32
        acc = plgpu.wgmma(a_smem, b_smem, acc)

    # Store result
    c_ref[...] = acc.astype(jnp.float16)
```

WGMMA characteristics:
- Operates on a full warpgroup (128 threads)
- Supports FP16, BF16, FP8, INT8 input types
- Accumulates in FP32 or FP16
- Single instruction processes up to 256x128x16 elements (MxNxK)

### 24.5.2 Blackwell TCgen05

Blackwell (B200) introduces TCgen05, the fifth-generation Tensor Core instruction
with support for TMEM (Tensor Memory) and larger tile sizes.

```python
def blackwell_tcgen05_example(a_ref, b_ref, c_ref):
    """Example using Blackwell TCgen05 for matrix multiply."""
    BM, BK = 128, 64
    BN = 256  # Larger N dimension on Blackwell

    # Initialize accumulator in TMEM
    acc = plgpu.tmem_alloc((BM, BN), jnp.float32)

    for k in range(K // BK):
        a_smem = a_ref[:, k*BK:(k+1)*BK]
        b_smem = b_ref[k*BK:(k+1)*BK, :]

        # TCgen05 MMA: results go directly to TMEM
        plgpu.tcgen05_mma(a_smem, b_smem, acc)

    # Store from TMEM to HBM
    c_ref[...] = plgpu.tmem_read(acc)
```

### 24.5.3 MMA Operation Types

```python
# Different MMA operation configurations
from jax.experimental.pallas.gpu import mm

# Standard FP16 matmul with FP32 accumulation
# a: (M, K) float16, b: (K, N) float16, c: (M, N) float32
c = plgpu.wgmma(a, b, c, input_dtype=jnp.float16, acc_dtype=jnp.float32)

# BF16 matmul
c = plgpu.wgmma(a, b, c, input_dtype=jnp.bfloat16, acc_dtype=jnp.float32)

# FP8 (E4M3 or E5M2) matmul on Hopper+
c = plgpu.wgmma(a, b, c, input_dtype=jnp.float8_e4m3fn, acc_dtype=jnp.float32)

# Mixed precision: FP8 input, FP16 accumulation
c = plgpu.wgmma(a, b, c, input_dtype=jnp.float8_e4m3fn, acc_dtype=jnp.float16)
```

---

## 24.6 Blackwell TMEM Operations

Blackwell introduces TMEM (Tensor Memory), a new memory space directly connected
to the Tensor Core pipeline. TMEM eliminates the register pressure associated with
holding accumulator fragments in thread registers.

### 24.6.1 TMEM Overview

- Located within each SM, dedicated to Tensor Core results
- Stores MMA accumulator tiles directly
- Reduces register pressure by offloading accumulator storage
- Enables larger tile sizes without register spills

### 24.6.2 TMEM API

```python
from jax.experimental.pallas.gpu import tmem

def tmem_matmul_kernel(a_ref, b_ref, c_ref):
    BM, BN, BK = 128, 256, 64

    # Allocate TMEM for accumulator
    acc_tmem = tmem.TMEM((BM, BN), jnp.float32)

    for k in range(K // BK):
        a_smem = a_ref[:, k*BK:(k+1)*BK]   # (BM, BK) in SMEM
        b_smem = b_ref[k*BK:(k+1)*BK, :]   # (BK, BN) in SMEM

        # MMA with TMEM accumulator
        tmem.tcgen05_mma(acc_tmem, a_smem, b_smem)

    # Read TMEM to registers/GMEM
    c_ref[...] = tmem.tmem_to_gmem(acc_tmem)
```

### 24.6.3 TMEM Benefits

| Feature | Registers (Hopper) | TMEM (Blackwell) |
|---|---|---|
| Accumulator storage | Thread registers | Dedicated tensor memory |
| Register pressure | High (large tiles) | Low |
| Max tile size | Limited by registers | Larger tiles possible |
| Read bandwidth | Register speed | Direct toGMEM store |
| Pipeline integration | Manual | Hardware-managed |

---

## 24.7 Synchronization Primitives

GPU parallel execution requires explicit synchronization to ensure correct ordering
of memory operations, especially when coordinating between warps or between SMs.

### 24.7.1 commit_smem

`commit_smem` ensures that all pending writes to shared memory are visible. This is
necessary after async copies from HBM to SMEM before the data can be safely read.

```python
from jax.experimental.pallas import gpu as plgpu

def kernel_with_commit_smem(a_ref, o_ref):
    # Initiate async copy from GMEM to SMEM
    a_smem = plgpu.async_copy(a_ref[...], smem_buffer)

    # Wait for copy to complete
    plgpu.commit_smem()

    # Safe to read from SMEM now
    data = smem_buffer[...]
    o_ref[...] = data * 2.0
```

### 24.7.2 Barrier

A `Barrier` synchronizes all warps within a single CTA (Cooperative Thread Array /
thread block). All warps must reach the barrier before any can proceed.

```python
from jax.experimental.pallas.gpu import barriers

def kernel_with_barrier(a_ref, b_ref, c_ref):
    BM, BN, BK = 64, 64, 32

    # Shared memory buffers for double buffering
    a_smem = [plgpu.SMEM((BM, BK), jnp.float16) for _ in range(2)]
    b_smem = [plgpu.SMEM((BK, BN), jnp.float16) for _ in range(2)]

    # Warp barrier
    barrier = barriers.Barrier(num_warps=8)

    acc = jnp.zeros((BM, BN), jnp.float32)

    for k in range(K // BK):
        # Load next tiles (one warp does the loading)
        a_smem[k % 2][...] = a_ref[:, k*BK:(k+1)*BK]
        b_smem[k % 2][...] = b_ref[k*BK:(k+1)*BK, :]

        # Ensure all warps see the loaded data
        barrier.wait()

        # Compute
        acc += jnp.dot(a_smem[k % 2][...], b_smem[k % 2][...])

        # Ensure computation is done before overwriting SMEM
        barrier.wait()

    c_ref[...] = acc
```

### 24.7.3 ClusterBarrier

On Hopper and later architectures, multiple CTAs can form a **cluster** that shares
distributed shared memory. `ClusterBarrier` synchronizes across CTAs within a cluster.

```python
from jax.experimental.pallas.gpu import barriers

# Cluster barrier for 2-CTA cluster
cluster_barrier = barriers.ClusterBarrier(num_ctas=2)

def clustered_kernel(a_ref, o_ref):
    # Each CTA processes its portion
    local_data = a_ref[...]
    # ... process ...
    cluster_barrier.wait()
    # Now can access distributed shared memory from other CTAs
```

### 24.7.4 Semaphore

Semaphores are general-purpose synchronization primitives for coordinating between
producers and consumers of data, commonly used in software pipelining.

```python
from jax.experimental.pallas.gpu import semaphore

def kernel_with_semaphore(a_ref, o_ref):
    # Semaphore for producer-consumer synchronization
    sem = semaphore.Semaphore(initial_value=0)

    # Producer: load data
    a_smem = plgpu.SMEM((64, 32), jnp.float16)
    a_smem[...] = a_ref[...]
    sem.signal(1)  # Signal that data is ready

    # Consumer: wait for data then compute
    sem.wait(1)    # Wait until signaled
    o_ref[...] = jnp.dot(a_smem[...], weight_matrix)
```

---

## 24.8 Software Pipelining

Software pipelining overlaps memory loads with computation to hide memory latency.
Instead of loading all data before computing, the kernel loads the next tile while
computing the current tile.

### 24.8.1 Manual Double Buffering

```python
def double_buffer_matmul(a_ref, b_ref, c_ref):
    """Matrix multiplication with manual double buffering."""
    BM, BN, BK = 128, 128, 64

    # Double buffers for A and B in SMEM
    a_buf = [plgpu.SMEM((BM, BK), jnp.float16) for _ in range(2)]
    b_buf = [plgpu.SMEM((BK, BN), jnp.float16) for _ in range(2)]

    barrier = barriers.Barrier(num_warps=8)
    acc = jnp.zeros((BM, BN), jnp.float32)

    # Prologue: load first tile
    a_buf[0][...] = a_ref[:, 0:BK]
    b_buf[0][...] = b_ref[0:BK, :]
    barrier.wait()

    num_stages = K // BK
    for k in range(num_stages):
        current = k % 2
        next_buf = (k + 1) % 2

        # Load next tile while computing current tile
        if k + 1 < num_stages:
            a_buf[next_buf][...] = a_ref[:, (k+1)*BK:(k+2)*BK]
            b_buf[next_buf][...] = b_ref[(k+1)*BK:(k+2)*BK, :]

        # Compute current tile
        acc += plgpu.wgmma(a_buf[current][...], b_buf[current][...], acc)

        # Wait for next load to complete
        if k + 1 < num_stages:
            barrier.wait()

    c_ref[...] = acc
```

### 24.8.2 emit_pipeline

Pallas GPU provides `emit_pipeline` for automatic software pipelining, which
generates the prologue, steady-state, and epilogue loops.

```python
from jax.experimental.pallas.gpu import pipeline

def pipelined_matmul(a_ref, b_ref, c_ref):
    """Matrix multiplication using emit_pipeline for automatic pipelining."""
    BM, BN, BK = 128, 128, 64
    num_k_tiles = K // BK

    # Define the pipeline body
    def body(k, acc, a_smem, b_smem):
        # Compute on current SMEM contents
        acc = plgpu.wgmma(a_smem[...], b_smem[...], acc)

        # Load next tile into SMEM (overlapped with computation)
        if k + 1 < num_k_tiles:
            a_smem[...] = a_ref[:, (k+1)*BK:(k+2)*BK]
            b_smem[...] = b_ref[(k+1)*BK:(k+2)*BK, :]

        return acc, a_smem, b_smem

    # Initialize
    acc = jnp.zeros((BM, BN), jnp.float32)
    a_smem = plgpu.SMEM((BM, BK), jnp.float16)
    b_smem = plgpu.SMEM((BK, BN), jnp.float16)

    # Load first tile
    a_smem[...] = a_ref[:, 0:BK]
    b_smem[...] = b_ref[0:BK, :]

    # Run pipelined loop
    acc, _, _ = pipeline.emit_pipeline(
        num_k_tiles,
        body,
        init_state=(acc, a_smem, b_smem),
    )

    c_ref[...] = acc
```

### 24.8.3 Pipeline Stages

The `emit_pipeline` API supports multiple pipeline stages:

```python
# 3-stage pipeline: 3 loads in flight simultaneously
pipeline.emit_pipeline(
    num_iterations,
    body_fn,
    init_state=state,
    num_stages=3,       # Number of pipeline stages
)
```

More stages hide more latency but require more SMEM buffers.

---

## 24.9 Warp Specialization

Warp specialization assigns different roles to different warps within a CTA.
Common patterns:
- **Load warps**: Dedicated to loading data from HBM to SMEM
- **Compute warps**: Dedicated to Tensor Core operations
- **Store warps**: Dedicated to writing results back to HBM

```python
from jax.experimental.pallas.gpu import warp_specialization

def warp_specialized_matmul(a_ref, b_ref, c_ref):
    BM, BN, BK = 128, 128, 64

    def load_fn(pipe_idx, a_smem, b_smem):
        """Load warp: copy data from HBM to SMEM."""
        a_smem[...] = a_ref[:, pipe_idx*BK:(pipe_idx+1)*BK]
        b_smem[...] = b_ref[pipe_idx*BK:(pipe_idx+1)*BK, :]

    def compute_fn(pipe_idx, acc, a_smem, b_smem):
        """Compute warps: run WGMMA."""
        acc = plgpu.wgmma(a_smem[...], b_smem[...], acc)
        return acc

    acc = jnp.zeros((BM, BN), jnp.float32)
    a_smem = plgpu.SMEM((BM, BK), jnp.float16)
    b_smem = plgpu.SMEM((BK, BN), jnp.float16)

    acc = warp_specialization.warp_specialized_loop(
        num_iterations=K // BK,
        load_fn=load_fn,
        compute_fn=compute_fn,
        init_acc=acc,
        buffers=(a_smem, b_smem),
        num_load_warps=1,
        num_compute_warps=7,
    )

    c_ref[...] = acc
```

---

## 24.10 Collective MMA Across 2 SMs

On Hopper and later, Pallas supports distributing a single MMA operation across
2 SMs using CTA clustering. This doubles the SMEM capacity and Tensor Core throughput
for a single matmul tile.

```python
def two_sm_mma_kernel(a_ref, b_ref, c_ref):
    """Matmul distributed across 2 SMs via CTA clustering."""
    BM, BN, BK = 256, 128, 64  # Larger M since we have 2 SMs

    # Each SM handles half the M dimension
    local_m = BM // 2
    cta_id = plgpu.cta_id()  # 0 or 1

    # Shared SMEM (visible to both CTAs via distributed shared memory)
    a_smem = plgpu.SMEM((BM, BK), jnp.float16)
    b_smem = plgpu.SMEM((BK, BN), jnp.float16)

    # CTA 0 loads A, CTA 1 also loads A (or use distributed SMEM)
    local_a = a_smem[cta_id * local_m:(cta_id + 1) * local_m, :]

    acc = jnp.zeros((local_m, BN), jnp.float32)
    for k in range(K // BK):
        # Load tiles
        a_smem[...] = a_ref[:, k*BK:(k+1)*BK]
        b_smem[...] = b_ref[k*BK:(k+1)*BK, :]

        # Synchronize across CTAs in the cluster
        plgpu.cluster_barrier_wait()

        # Each CTA computes its portion
        acc += plgpu.wgmma(local_a, b_smem, acc)

    # Store local portion
    c_ref[cta_id * local_m:(cta_id + 1) * local_m, :] = acc
```

---

## 24.11 Async Copies: GMEM <-> SMEM via TMA

### 24.11.1 Tensor Memory Accelerator (TMA)

TMA is a dedicated hardware unit on Hopper+ that performs async copies between
HBM and SMEM without involving CUDA cores. TMA handles:
- 1D, 2D, 3D, 4D, and 5D tensor descriptors
- Swizzled layouts
- Boundary checking and padding
- Out-of-order completion

```python
from jax.experimental.pallas.gpu import tma

def tma_copy_kernel(a_ref, b_ref, c_ref):
    BM, BN, BK = 128, 128, 64

    # TMA descriptor for A matrix
    a_desc = tma.make_descriptor(
        shape=(M, K),
        element_type=jnp.float16,
        block_shape=(BM, BK),
    )

    # Async load via TMA
    a_smem = plgpu.SMEM((BM, BK), jnp.float16)
    b_smem = plgpu.SMEM((BK, BN), jnp.float16)

    # Issue async TMA load (returns immediately)
    tma.async_load(a_smem, a_desc, block_coords=(i, k))

    # Wait for load completion
    tma.wait(a_smem)

    # Compute
    acc = plgpu.wgmma(a_smem, b_smem, acc)
```

### 24.11.2 TMA Copy Patterns

```python
# Bulk copy: large contiguous transfer
tma.bulk_copy(src=gmem_ref, dst=smem_ref, shape=(128, 64))

# Tiled copy: copy individual tiles with swizzling
tma.tiled_copy(
    src=gmem_ref,
    dst=smem_ref,
    tile_shape=(64, 32),
    transform=transforms.SwizzleTransform(3),
)

# Predicated copy: conditional copy with masks
tma.predicated_copy(
    src=gmem_ref,
    dst=smem_ref,
    mask=valid_mask,
)
```

### 24.11.3 Async Copy Pipeline

```python
def async_pipelined_matmul(a_ref, b_ref, c_ref):
    BM, BN, BK = 128, 128, 64
    NUM_STAGES = 3

    # Triple-buffered SMEM
    a_smem = [plgpu.SMEM((BM, BK), jnp.float16) for _ in range(NUM_STAGES)]
    b_smem = [plgpu.SMEM((BK, BN), jnp.float16) for _ in range(NUM_STAGES)]

    # TMA semaphore for async completion tracking
    sem = tma.Semaphore(NUM_STAGES)

    acc = jnp.zeros((BM, BN), jnp.float32)

    # Prologue: issue first N async loads
    for s in range(NUM_STAGES):
        tma.async_load(a_smem[s], a_desc, block_coords=(i, s))
        tma.async_load(b_smem[s], b_desc, block_coords=(s, j))

    # Steady state
    for k in range(K // BK):
        stage = k % NUM_STAGES

        # Wait for current stage's loads to complete
        tma.wait(sem, stage)

        # Compute
        acc = plgpu.wgmma(a_smem[stage], b_smem[stage], acc)

        # Issue next load (overwrite this stage's buffer)
        next_k = k + NUM_STAGES
        if next_k < K // BK:
            tma.async_load(a_smem[stage], a_desc, block_coords=(i, next_k))
            tma.async_load(b_smem[stage], b_desc, block_coords=(next_k, j))

    c_ref[...] = acc
```

---

## 24.12 NVLINK Inter-Device Transfers

NVLINK provides high-bandwidth inter-GPU communication. Pallas GPU can initiate
direct memory transfers between GPUs without host involvement.

```python
from jax.experimental.pallas.gpu import nvlink

def inter_device_kernel(local_ref, remote_ref):
    """Kernel that transfers data between two GPUs via NVLINK."""
    # NVLINK bandwidth: up to 900 GB/s (bidirectional, per link)
    # Initiate async copy from remote GPU's memory
    nvlink.async_copy_from_remote(
        src=remote_ref,
        dst=local_ref,
        shape=(128, 64),
    )

    # Wait for transfer
    nvlink.wait()

    # Now local_ref contains data from the other GPU
    result = local_ref[...] * 2.0
    local_ref[...] = result
```

### 24.12.1 NVLINK Topology Awareness

```python
# Query NVLINK topology
topology = nvlink.get_topology()
for i, j in topology.connected_pairs():
    bandwidth = topology.bandwidth(i, j)
    print(f"GPU {i} <-> GPU {j}: {bandwidth} GB/s")
```

---

## 24.13 Grid Tiling for L2 Cache Optimization

L2 cache performance can be significantly improved by controlling the order in which
tiles are processed. Pallas GPU supports grid tiling configurations that improve
data locality.

### 24.13.1 Cache-Aware Grid Traversal

By default, programs execute in lexicographic order of grid indices. Grid tiling
reorders execution to maximize L2 cache reuse between adjacent tiles.

```python
def cache_optimized_matmul(a: jax.Array, b: jax.Array) -> jax.Array:
    M, K = a.shape
    _, N = b.shape
    BM, BN, BK = 128, 128, 64

    grid = (M // BM, N // BN)

    # Configure grid tiling for L2 cache optimization
    # Tile size determines how many adjacent programs share L2 cache
    compiler_params = {
        "gpu": {
            "grid_tiling": (2, 2),  # Process in 2x2 super-tiles
        }
    }

    return pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
        grid=grid,
        in_specs=[
            pl.BlockSpec((BM, None), lambda i, j: (i, 0)),
            pl.BlockSpec((None, BN), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((BM, BN), lambda i, j: (i, j)),
        compiler_params=compiler_params,
    )(a, b)
```

### 24.13.2 L2 Cache Policy Hints

```python
compiler_params = {
    "gpu": {
        # Prefer cache level for read accesses
        "l2_cache_policy": "streaming",  # or "persistent"
        # Grid tiling for locality
        "grid_tiling": (4, 4),
    }
}
```

---

## 24.14 Progressive Optimization: From Basic to cuBLAS-Matching Matmul

This section walks through the progressive optimization of a matrix multiplication
kernel, from a basic implementation to one that matches cuBLAS performance.

### 24.14.1 Level 1: Basic Tiled Matmul

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

BM, BN, BK = 64, 64, 32

def matmul_v1_kernel(a_ref, b_ref, c_ref):
    """Basic tiled matmul: load from GMEM, compute in registers."""
    acc = jnp.zeros((BM, BN), dtype=jnp.float32)
    for k in range(K // BK):
        a_tile = a_ref[:, k*BK:(k+1)*BK]
        b_tile = b_ref[k*BK:(k+1)*BK, :]
        acc += jnp.dot(a_tile, b_tile)
    c_ref[...] = acc

def matmul_v1(a: jax.Array, b: jax.Array) -> jax.Array:
    M, K = a.shape
    _, N = b.shape
    return pl.pallas_call(
        matmul_v1_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
        grid=(M // BM, N // BN),
        in_specs=[
            pl.BlockSpec((BM, None), lambda i, j: (i, 0)),
            pl.BlockSpec((None, BN), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((BM, BN), lambda i, j: (i, j)),
    )(a, b)
```

Performance: ~30% of cuBLAS (limited by GMEM bandwidth, no SMEM reuse).

### 24.14.2 Level 2: SMEM Tiling

```python
def matmul_v2_kernel(a_ref, b_ref, c_ref):
    """Matmul with SMEM buffering for A and B tiles."""
    # Allocate SMEM buffers
    a_smem = plgpu.SMEM((BM, BK), jnp.float16)
    b_smem = plgpu.SMEM((BK, BN), jnp.float16)

    acc = jnp.zeros((BM, BN), dtype=jnp.float32)

    for k in range(K // BK):
        # Load tiles into SMEM
        a_smem[...] = a_ref[:, k*BK:(k+1)*BK]
        b_smem[...] = b_ref[k*BK:(k+1)*BK, :]

        # Compute from SMEM
        acc += jnp.dot(a_smem[...], b_smem[...])

    c_ref[...] = acc
```

Performance: ~50% of cuBLAS (GMEM traffic reduced, but still using jnp.dot not Tensor Cores).

### 24.14.3 Level 3: Tensor Core (WGMMA)

```python
def matmul_v3_kernel(a_ref, b_ref, c_ref):
    """Matmul using Hopper WGMMA for Tensor Core utilization."""
    a_smem = plgpu.SMEM((BM, BK), jnp.float16)
    b_smem = plgpu.SMEM((BK, BN), jnp.float16)

    acc = jnp.zeros((BM, BN), dtype=jnp.float32)

    for k in range(K // BK):
        a_smem[...] = a_ref[:, k*BK:(k+1)*BK]
        b_smem[...] = b_ref[k*BK:(k+1)*BK, :]

        # Use WGMMA instead of jnp.dot
        acc = plgpu.wgmma(a_smem[...], b_smem[...], acc)

    c_ref[...] = acc
```

Performance: ~70% of cuBLAS (Tensor Cores active, but no pipelining).

### 24.14.4 Level 4: Software Pipelining

```python
def matmul_v4_kernel(a_ref, b_ref, c_ref):
    """Matmul with WGMMA and software pipelining."""
    NUM_STAGES = 2
    a_smem = [plgpu.SMEM((BM, BK), jnp.float16) for _ in range(NUM_STAGES)]
    b_smem = [plgpu.SMEM((BK, BN), jnp.float16) for _ in range(NUM_STAGES)]

    barrier = barriers.Barrier(num_warps=8)
    acc = jnp.zeros((BM, BN), dtype=jnp.float32)

    # Prologue: load first tile
    a_smem[0][...] = a_ref[:, 0:BK]
    b_smem[0][...] = b_ref[0:BK, :]
    barrier.wait()

    for k in range(K // BK):
        curr = k % NUM_STAGES
        nxt = (k + 1) % NUM_STAGES

        # Prefetch next tile
        if k + 1 < K // BK:
            a_smem[nxt][...] = a_ref[:, (k+1)*BK:(k+2)*BK]
            b_smem[nxt][...] = b_ref[(k+1)*BK:(k+2)*BK, :]

        # Compute
        acc = plgpu.wgmma(a_smem[curr][...], b_smem[curr][...], acc)

        # Wait for next load
        if k + 1 < K // BK:
            barrier.wait()

    c_ref[...] = acc
```

Performance: ~85% of cuBLAS (latency hiding via double buffering).

### 24.14.5 Level 5: Warp Specialization + TMA + Cluster

```python
def matmul_v5_kernel(a_ref, b_ref, c_ref):
    """Near-cuBLAS matmul with warp specialization, TMA, and pipelining."""
    BM, BN, BK = 128, 128, 64
    NUM_STAGES = 3

    # Triple-buffered SMEM
    a_smem = [plgpu.SMEM((BM, BK), jnp.float16) for _ in range(NUM_STAGES)]
    b_smem = [plgpu.SMEM((BK, BN), jnp.float16) for _ in range(NUM_STAGES)]

    # TMA descriptors
    a_desc = tma.make_descriptor((M, K), jnp.float16, (BM, BK))
    b_desc = tma.make_descriptor((K, N), jnp.float16, (BK, BN))

    # Warp-specialized execution
    def load_fn(stage, a_buf, b_buf):
        k = stage
        tma.async_load(a_buf, a_desc, (i, k))
        tma.async_load(b_buf, b_desc, (k, j))
        tma.wait_group(0)  # Wait for these loads

    def compute_fn(stage, acc, a_buf, b_buf):
        return plgpu.wgmma(a_buf[...], b_buf[...], acc)

    acc = jnp.zeros((BM, BN), jnp.float32)

    acc = pipeline.emit_pipeline(
        K // BK,
        lambda k, acc: compute_fn(k, acc, a_smem[k % NUM_STAGES], b_smem[k % NUM_STAGES]),
        init_state=acc,
        num_stages=NUM_STAGES,
    )

    c_ref[...] = acc
```

Performance: ~95% of cuBLAS (near-optimal for large matrices).

---

## 24.15 Complete Blackwell Matmul Kernel

This section presents a complete, production-quality matmul kernel targeting
Blackwell (B200) GPUs with TCgen05 Tensor Cores and TMEM.

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import gpu as plgpu
from jax.experimental.pallas.gpu import tma, barriers, tmem, layouts, transforms

# Tile sizes optimized for Blackwell Tensor Cores
BM = 128    # M tile per CTA
BN = 256    # N tile per CTA (larger on Blackwell)
BK = 64     # K tile (inner dimension)
NUM_STAGES = 3  # Pipeline depth

def blackwell_matmul_kernel(
    a_desc_ref,    # TMA descriptor for A matrix
    b_desc_ref,    # TMA descriptor for B matrix
    c_ref,         # Output: (BM, BN)
):
    """Blackwell-optimized matmul using TCgen05 + TMEM + TMA pipelining.

    Features:
    - TCgen05 Tensor Core instructions for maximum throughput
    - TMEM for accumulator storage (reduced register pressure)
    - TMA for async GMEM->SMEM copies
    - 3-stage software pipeline
    - Swizzled SMEM layout for bank-conflict-free access
    """
    i = pl.program_id(0)
    j = pl.program_id(1)

    # Allocate triple-buffered SMEM with swizzled layout
    a_smem = [
        plgpu.SMEM(
            (BM, BK), jnp.float16,
            layout=layouts.WGMMA_LAYOUT,
            transform=transforms.SwizzleTransform(3),
        )
        for _ in range(NUM_STAGES)
    ]
    b_smem = [
        plgpu.SMEM(
            (BK, BN), jnp.float16,
            layout=layouts.WGMMA_LAYOUT,
            transform=transforms.SwizzleTransform(3),
        )
        for _ in range(NUM_STAGES)
    ]

    # Allocate TMEM accumulator (Blackwell feature)
    acc = tmem.TMEM((BM, BN), jnp.float32)

    # TMA semaphores for async completion tracking
    tma_sem = [tma.Semaphore() for _ in range(NUM_STAGES)]

    # Barrier for warp synchronization
    barrier = barriers.Barrier(num_warps=8)

    # --- Prologue: issue first NUM_STAGES async TMA loads ---
    for s in range(min(NUM_STAGES, K // BK)):
        tma.async_load(
            a_smem[s], a_desc_ref,
            block_coords=(i, s),
            semaphore=tma_sem[s],
        )
        tma.async_load(
            b_smem[s], b_desc_ref,
            block_coords=(s, j),
            semaphore=tma_sem[s],
        )

    # --- Steady state: overlap loads and compute ---
    for k in range(K // BK):
        stage = k % NUM_STAGES

        # Wait for this stage's TMA loads to complete
        tma.wait(tma_sem[stage])
        barrier.wait()

        # TCgen05 MMA: A @ B -> TMEM accumulator
        tmem.tcgen05_mma(
            acc,
            a_smem[stage],
            b_smem[stage],
        )

        # Issue TMA loads for a future stage (pipeline ahead)
        future_k = k + NUM_STAGES
        if future_k < K // BK:
            future_stage = future_k % NUM_STAGES
            # Ensure this buffer is no longer being used
            barrier.wait()

            tma.async_load(
                a_smem[future_stage], a_desc_ref,
                block_coords=(i, future_k),
                semaphore=tma_sem[future_stage],
            )
            tma.async_load(
                b_smem[future_stage], b_desc_ref,
                block_coords=(future_k, j),
                semaphore=tma_sem[future_stage],
            )

    # --- Store: TMEM -> GMEM ---
    c_ref[...] = tmem.tmem_to_array(acc)

def blackwell_matmul(
    a: jax.Array,
    b: jax.Array,
) -> jax.Array:
    """Production-quality Blackwell matmul.

    Args:
        a: Input matrix of shape (M, K), float16 or bfloat16
        b: Input matrix of shape (K, N), float16 or bfloat16

    Returns:
        Output matrix of shape (M, N), float32
    """
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"Dimension mismatch: a.shape[1]={K}, b.shape[0]={K2}"
    assert M % BM == 0, f"M={M} must be divisible by BM={BM}"
    assert N % BN == 0, f"N={N} must be divisible by BN={BN}"
    assert K % BK == 0, f"K={K} must be divisible by BK={BK}"

    # Create TMA descriptors for A and B
    a_desc = tma.make_descriptor(
        shape=(M, K),
        element_type=a.dtype,
        block_shape=(BM, BK),
    )
    b_desc = tma.make_descriptor(
        shape=(K, N),
        element_type=b.dtype,
        block_shape=(BK, BN),
    )

    grid = (M // BM, N // BN)

    result = pl.pallas_call(
        blackwell_matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
        grid=grid,
        in_specs=[
            pl.BlockSpec((), lambda i, j: ()),  # TMA descriptor (scalar)
            pl.BlockSpec((), lambda i, j: ()),  # TMA descriptor (scalar)
        ],
        out_specs=pl.BlockSpec((BM, BN), lambda i, j: (i, j)),
        compiler_params={
            "gpu": {
                "tma_descriptors": [a_desc, b_desc],
                "grid_tiling": (2, 2),
            }
        },
        input_output_aliases={},
    )(a_desc, b_desc)

    return result

# --- Usage ---
if __name__ == "__main__":
    M, K, N = 4096, 4096, 4096
    key = jax.random.PRNGKey(0)
    a = jax.random.normal(key, (M, K), dtype=jnp.float16)
    b = jax.random.normal(jax.random.fold_in(key, 1), (K, N), dtype=jnp.float16)

    c = blackwell_matmul(a, b)

    # Verify
    expected = jnp.dot(a.astype(jnp.float32), b.astype(jnp.float32))
    print(f"Max error: {jnp.max(jnp.abs(c - expected)):.6f}")
    print(f"Mean error: {jnp.mean(jnp.abs(c - expected)):.6f}")
```

---

## 24.16 Additional GPU Primitives

### 24.16.1 Atomic Operations

```python
def atomic_add_kernel(x_ref, o_ref):
    """Atomic addition in GMEM."""
    x = x_ref[...]
    # Atomic add to output
    o_ref.at[...].add(x)

def atomic_cas_kernel(addr_ref, expected, desired, o_ref):
    """Atomic compare-and-swap."""
    old_val = addr_ref.at[0].compare_and_swap(expected, desired)
    o_ref[0] = old_val
```

### 24.16.2 Warp-Level Primitives

```python
from jax.experimental.pallas.gpu import warp

def warp_shuffle_example(a_ref, o_ref):
    """Warp-level shuffle for inter-thread communication."""
    a = a_ref[...]  # Each thread has a value

    # Shuffle: exchange values within warp
    shuffled = warp.shfl_xor(a, mask=0x1)  # XOR shuffle
    # shfl_up: shift values up by delta lanes
    shifted = warp.shfl_up(a, delta=1)
    # shfl_down: shift values down
    shifted_down = warp.shfl_down(a, delta=1)

    o_ref[...] = shuffled + shifted + shifted_down
```

### 24.16.3 Shared Memory Allocation

```python
def smem_management_kernel(a_ref, b_ref, c_ref):
    """Dynamic SMEM allocation and management."""
    BM, BN, BK = 128, 128, 64

    # Static SMEM allocation
    a_buf = plgpu.SMEM((BM, BK), jnp.float16)
    b_buf = plgpu.SMEM((BK, BN), jnp.float16)

    # Total SMEM usage: BM*BK*2 + BK*BN*2 bytes
    # = 128*64*2 + 64*128*2 = 32768 bytes = 32 KB
    # Hopper SM limit: 228 KB

    # Load and compute
    a_buf[...] = a_ref[...]
    b_buf[...] = b_ref[...]
    c_ref[...] = jnp.dot(a_buf[...].astype(jnp.float32),
                          b_buf[...].astype(jnp.float32))
```

---

## 24.17 Performance Guidelines Summary

| Optimization | Expected Speedup | Complexity |
|---|---|---|
| Basic tiled matmul | Baseline | Low |
| SMEM buffering | 1.5-2x | Low |
| Tensor Core (WGMMA) | 2-4x over jnp.dot | Medium |
| Software pipelining | 1.2-1.5x | Medium |
| TMA async copies | 1.2-1.3x | Medium |
| Warp specialization | 1.1-1.2x | High |
| L2 cache tiling | 1.05-1.15x | Low |
| TMEM (Blackwell) | 1.1-1.2x | Medium |
| Cluster (2-SM) | 1.3-1.5x | High |
| Full combination | 95-100% cuBLAS | Very High |

### Key Performance Principles

1. **Maximize Tensor Core utilization**: Use WGMMA/TCgen05 for all matrix operations
2. **Hide memory latency**: Use software pipelining with at least 2 stages
3. **Minimize GMEM traffic**: Reuse data in SMEM across multiple K tiles
4. **Avoid bank conflicts**: Use swizzled SMEM layouts
5. **Choose optimal tile sizes**: Match hardware granularity (128x128x64 for Hopper)
6. **Use TMA for large copies**: Offload memory transfers from CUDA cores
7. **Consider warp specialization**: Separate load and compute warps when memory-bound

---

## 24.18 Debugging GPU Kernels

### 24.18.1 Common Issues

| Issue | Symptom | Solution |
|---|---|---|
| Bank conflicts | Lower than expected SMEM perf | Apply SwizzleTransform |
| Register spills | Slow kernels, local memory use | Reduce tile sizes |
| Insufficient occupancy | Low SM utilization | Reduce SMEM/register usage |
| Race condition | Non-deterministic results | Add barriers, check memory ordering |
| Uncoalesced access | Low GMEM bandwidth | Align block shapes to 128-byte boundaries |

### 24.18.2 Profiling

```python
# Use JAX profiler with Pallas kernels
import jax.profiler

with jax.profiler.trace("/tmp/pallas_profile"):
    result = blackwell_matmul(a, b)
    result.block_until_ready()

# Open in TensorBoard: tensorboard --logdir=/tmp/pallas_profile
```

```python
# Timing with block_until_ready
import time

# Warmup
_ = blackwell_matmul(a, b).block_until_ready()

# Benchmark
num_iters = 100
start = time.perf_counter()
for _ in range(num_iters):
    result = blackwell_matmul(a, b)
result.block_until_ready()
elapsed = time.perf_counter() - start

tflops = 2 * M * K * N * num_iters / elapsed / 1e12
print(f"Performance: {tflops:.2f} TFLOP/s")
```
