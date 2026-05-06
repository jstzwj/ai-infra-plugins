# Chapter 25: Gluon Experimental Language

Gluon is an experimental lower-level GPU programming model that provides explicit control over layouts, shared memory, and warp specialization.

## Overview

Gluon provides:
- Explicit memory layout control
- Shared memory management
- Warp specialization
- Architecture-specific intrinsics (Hopper, Blackwell, AMD)
- Lower-level than standard Triton

## Core API (`triton.experimental.gluon.language`)

### JIT Decorator

```python
from triton.experimental.gluon import jit, constexpr_function

@jit
def kernel(desc_a, desc_b, desc_c, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Gluon kernel with explicit layouts
    pass
```

### Tensor Types

```python
# Distributed tensor (in register file)
x: tl.tensor

# Shared memory descriptor
smem: shared_memory_descriptor
```

### Programming Model

```python
pid = tl.program_id(axis=0)
nprogs = tl.num_programs(axis=0)
nwarps = tl.num_warps()
nctas = tl.num_ctas()
```

### Memory Operations

```python
# Allocate shared memory
buf = allocate_shared_memory(shape, dtype, layout=SwizzledSharedLayout(...))

# Load from shared memory
data = tl.load(smem_desc, mask=None)

# Store to shared memory
tl.store(smem_desc, data, mask=None)
```

### Layout System

```python
from triton.experimental.gluon.language import (
    AutoLayout,
    BlockedLayout,
    CoalescedLayout,
    DistributedLinearLayout,
    DotOperandLayout,
    NVMMADistributedLayout,
    NVMMASharedLayout,
    PaddedSharedLayout,
    SharedLinearLayout,
    SliceLayout,
    SwizzledSharedLayout,
)

# Convert layouts
result = convert_layout(tensor, target_layout)

# Check bank conflicts
conflicts = bank_conflicts(layout)
```

### Warp Specialization

```python
# Specialize warps for different tasks
with tl.warp_specialize(num_warps_d=1, num_warps_lds=3):
    # Warp group 0: compute (1 warp)
    # Warp group 1: load/store (3 warps)
    pass
```

## NVIDIA Target Intrinsics

### Hopper (SM 90)

```python
from triton.experimental.gluon.language.nvidia.hopper import (
    async_copy,      # Async memory copy (TMA)
    cluster,         # Cluster-level operations
    mbarrier,        # Barrier synchronization
    tma,             # Tensor Memory Accelerator
)

# TMA descriptor
desc = tma.TensorDescriptor(ptr, shape, strides, block_shape, layout)

# Async copy with TMA
tma.async_copy(desc, coords, smem_buf)

# Cluster operations
cluster.barrier()
cluster.wait()
```

### Blackwell (SM 100+)

```python
from triton.experimental.gluon.language.nvidia.blackwell import (
    async_copy,           # Enhanced async copy
    clc,                  # Coordinated load-compute
    tma,                  # Enhanced TMA
    float2,               # FP2 operations
)

# Tensor memory operations
from triton.experimental.gluon.language.nvidia.blackwell import (
    allocate_tensor_memory,
    tensor_memory_descriptor,
    TensorMemoryLayout,
    TensorMemoryScalesLayout,
)
```

### Ampere (SM 80+)

```python
from triton.experimental.gluon.language.nvidia.ampere import (
    async_copy,      # Async copy (cp.async)
    mbarrier,        # Barrier (bar.sync)
)
```

## AMD Target Intrinsics

### CDNA 3 (gfx942)

```python
from triton.experimental.gluon.language.amd.cdna3 import (
    buffer_load,        # Buffer load operation
    buffer_store,       # Buffer store operation
    buffer_atomic_add,  # Buffer atomic add
    buffer_atomic_and,  # Buffer atomic and
    buffer_atomic_max,  # Buffer atomic max
    buffer_atomic_min,  # Buffer atomic min
    buffer_atomic_or,   # Buffer atomic or
    buffer_atomic_xchg, # Buffer atomic exchange
    buffer_atomic_xor,  # Buffer atomic xor
    mfma,               # Matrix Fused Multiply-Add
)
```

### CDNA 4 (gfx950)

```python
from triton.experimental.gluon.language.amd.cdna4 import (
    async_copy,          # Async memory copy
    buffer_load,         # Buffer load
    buffer_store,        # Buffer store
    mfma,                # MFMA operation
    mfma_scaled,         # Scaled MFMA (FP8)
    get_mfma_scale_layout, # Scale layout for MFMA
)
```

### GFX1250

```python
from triton.experimental.gluon.language.amd.gfx1250 import (
    async_copy,      # Async memory copy
    cluster,         # Cluster operations
    mbarrier,        # Barrier synchronization
    tdm,             # Tensor Data Movement
)
```

### Warp Pipeline

```python
from triton.experimental.gluon.language.amd import warp_pipeline_stage

# Multi-stage warp pipeline
stage0 = warp_pipeline_stage(0)
stage1 = warp_pipeline_stage(1)
```

## Runtime API (`triton.experimental.gluon`)

### GluonJITFunction

```python
from triton.experimental.gluon import jit, constexpr_function

@jit
def kernel(...):
    pass

# Launch
kernel[grid](args)

# Constexpr function
@constexpr_function
def compute_block_size(n):
    return min(1024, triton.next_power_of_2(n))
```

## Gluon Tensor Descriptors

```python
# NVIDIA Hopper TMA descriptor
from triton.experimental.gluon.language.nvidia.hopper.tma import TensorDescriptor

desc = TensorDescriptor(
    base=ptr,
    shape=(M, K),
    strides=(stride_m, stride_k),
    block_shape=(BLOCK_M, BLOCK_K),
    layout=NVMMASharedLayout(swizzle=3),
)

# Load entire block
data = desc.load([offset_m, offset_k])

# Store entire block
desc.store([offset_m, offset_k], data)
```

## Differences from Standard Triton

| Feature | Standard Triton | Gluon |
|---------|----------------|-------|
| Layout control | Automatic | Explicit |
| Shared memory | Implicit | Explicit allocation |
| Warp specialization | Automatic | Manual |
| Memory operations | High-level | Low-level descriptors |
| Target intrinsics | None | Architecture-specific |
| Dot operation | `tl.dot` | `dot_fma` or MMA intrinsics |
