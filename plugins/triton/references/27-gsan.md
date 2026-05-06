# Chapter 27: GPU Sanitizer (GSAN)

GSAN is an experimental GPU memory sanitization system for debugging.

## Overview

GSAN provides:
- Memory allocation tracking
- Buffer overflow detection
- Use-after-free detection
- Symmetric memory management for multi-GPU

## Allocator API

### Basic Usage

```python
from triton.experimental.gsan import create_mem_pool, get_allocator

# Create a memory pool with sanitization
pool = create_mem_pool()

# Get the allocator for use with PyTorch
allocator = get_allocator(pool)
```

### Integration with PyTorch

```python
import torch
from triton.experimental.gsan import get_allocator

# Set as CUDA pluggable allocator
allocator = get_allocator()
torch.cuda.memory.CUDAPluggableAllocator(
    torch.cuda.current_device(),
    allocator.malloc_fn,
    allocator.free_fn,
)
```

## Memory Pool

### `create_mem_pool()`

Creates a sanitized memory pool:

```python
from triton.experimental.gsan import create_mem_pool

pool = create_mem_pool()
# Pool tracks all allocations
# Detects double-free, use-after-free, buffer overflows
```

### `get_allocator(pool=None)`

Gets a CUDA pluggable allocator:

```python
allocator = get_allocator(pool)

# allocator has malloc and free functions
# that check for memory errors
```

## Symmetric Memory

Multi-GPU symmetric memory allocation:

```python
from triton.experimental.gsan import symmetric_memory

# Allocate symmetric memory across GPUs
sym_buf = symmetric_memory.allocate(size, num_gpus=4)
```

## Stream Synchronization

```python
from triton.experimental.gsan._stream_sync import stream_sync

# Synchronize streams for safe memory access
stream_sync(stream_a, stream_b)
```

## Testing Utilities

```python
from triton.experimental.gsan._testing import gsan_test_context

# Context manager for testing with sanitization
with gsan_test_context() as ctx:
    # Run code that should be checked
    result = my_kernel[grid](args)
    # Memory errors will be reported
```

## Error Detection

GSAN detects:
- **Buffer overflow:** Writing past allocation boundaries
- **Use-after-free:** Accessing freed memory
- **Double-free:** Freeing memory twice
- **Uninitialized read:** Reading before writing (limited support)

## C++ Extension

The allocator uses a C++ extension compiled at runtime:

```python
from triton.experimental.gsan._allocator import _compile_gsan_allocator

# Compiles the sanitizer allocator
so_path = _compile_gsan_allocator()
```

## Allocation Handles

```python
from triton.experimental.gsan._allocator import (
    export_allocation_handles,
    import_allocation_handles,
)

# Export handles for inter-process sharing
handles = export_allocation_handles()

# Import handles in another process
import_allocation_handles(handles)
```
