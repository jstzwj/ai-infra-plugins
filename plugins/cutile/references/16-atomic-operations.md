# Atomic Operations

Atomic operations provide thread-safe access to shared memory locations, allowing multiple threads to safely read-modify-write the same memory location without data races. In cuTile, atomic operations are designed to work efficiently on GPU hardware, supporting both intra-block and inter-block synchronization patterns.

## Overview of Atomic Operations

cuTile provides a comprehensive set of atomic operations that work on array elements at specified indices. Unlike traditional GPU programming models where atomic operations typically work on single addresses, cuTile's atomic operations support **bulk operations**—they can operate on multiple array elements specified by index arrays in a single call.

### Key Characteristics

1. **Bulk Operations**: Operate on multiple array elements in a single call
2. **Index-based**: Use index arrays to specify which elements to operate on
3. **Memory Ordering**: Fine-grained control over memory ordering semantics
4. **Memory Scoping**: Control visibility scope of atomic operations
5. **Return Values**: Most operations return the old value at each location

### Supported Data Types

Atomic operations in cuTile support the following data types:

- **Integers**: `int8`, `int16`, `int32`, `int64`
- **Unsigned Integers**: `uint8`, `uint16`, `uint32`, `uint64`
- **Floating Point**: `float32` (float16, float64 support varies by operation)

Not all operations support all data types. Refer to specific operation documentation for details.

## Atomic Operation API

### `ct.atomic_cas`

Compare-and-swap (CAS) is the fundamental atomic operation used to build lock-free data structures. It atomically compares the value at each target location with a comparison value and, if they match, stores a new value.

```python
ct.atomic_cas(array, indices, compare, val, *, memory_order, memory_scope)
```

**Parameters:**

- `array` (`tl.tensor`): Target array containing elements to modify
- `indices` (`tl.tensor`): Integer tensor specifying which elements to operate on
- `compare` (`tl.tensor` or scalar): Values to compare against existing array values
- `val` (`tl.tensor` or scalar): New values to store if comparison succeeds
- `memory_order` (`Optional[MemoryOrder]`): Memory ordering constraint (default: RELAXED)
- `memory_scope` (`Optional[MemoryScope]`): Visibility scope (default: BLOCK)

**Returns:** `tl.tensor` — Old values at each target location (before any potential swap)

**Supported dtypes:** All integer types, float32

**Memory Order Options:**
- `RELAXED`: No ordering guarantees (default)
- `ACQUIRE`: Acquire semantics for subsequent operations
- `RELEASE`: Release semantics for prior operations
- `ACQ_REL`: Combined acquire and release

**Memory Scope Options:**
- `BLOCK`: Ordering within same block (default)
- `DEVICE`: All threads on same GPU
- `SYS`: All threads across system (multi-GPU, host)

**Example: Lock-free Stack Push**

```python
import cuda.tile as ct
import torch

@ct.kernel
def lockfree_push(
    stack_ptr: ct.tensor,  # Pointer to top-of-stack index
    values: ct.tensor,     # Values to push
    output: ct.tensor      # Output array for stack
):
    # Each thread attempts to claim a stack slot
    pid = ct.program_id(0)
    old_top = ct.atomic_cas(
        stack_ptr, 
        indices=[0],  # Single shared counter
        compare=pid,  # Expect current value
        val=pid + 1   # Increment if successful
    )
    
    # Only write if we successfully claimed the slot
    if old_top == pid:
        output[old_top] = values[pid]
```

### `ct.atomic_xchg`

Atomic exchange swaps the value at each target location with a new value, returning the old value. This is useful for implementing locks and for simple value replacement scenarios.

```python
ct.atomic_xchg(array, indices, val, *, memory_order, memory_scope)
```

**Parameters:**

- `array` (`tl.tensor`): Target array containing elements to modify
- `indices` (`tl.tensor`): Integer tensor specifying which elements to operate on
- `val` (`tl.tensor` or scalar): New values to store
- `memory_order` (`Optional[MemoryOrder]`): Memory ordering constraint (default: RELAXED)
- `memory_scope` (`Optional[MemoryScope]`): Visibility scope (default: BLOCK)

**Returns:** `tl.tensor` — Old values at each target location

**Supported dtypes:** All integer types, float32, float64

**Example: Spin Lock Implementation**

```python
@ct.kernel
def spin_lock_acquire(
    lock_array: ct.tensor,  # Array of locks
    lock_id: ct.scalar,     # Which lock to acquire
    thread_id: ct.scalar    # Thread identifier
):
    # Try to exchange lock value (0 → thread_id)
    # Returns 0 if lock was free, non-zero if already held
    acquired = ct.atomic_xchg(
        lock_array,
        indices=[lock_id],
        val=thread_id
    )
    
    # Note: In real code, you'd loop here until acquired == 0
    # This is simplified for demonstration
```

### `ct.atomic_add`

Atomic addition adds a value to the target location, returning the old value. This is the most commonly used atomic operation, typically for global reductions and histogram computations.

```python
ct.atomic_add(array, indices, val, *, memory_order, memory_scope)
```

**Parameters:**

- `array` (`tl.tensor`): Target array containing elements to modify
- `indices` (`tl.tensor`): Integer tensor specifying which elements to operate on
- `val` (`tl.tensor` or scalar): Values to add
- `memory_order` (`Optional[MemoryOrder]`): Memory ordering constraint (default: RELAXED)
- `memory_scope` (`Optional[MemoryScope]`): Visibility scope (default: BLOCK)

**Returns:** `tl.tensor` — Old values at each target location (before addition)

**Supported dtypes:** All integer types, float32, float64

**Example: Global Histogram**

```python
@ct.kernel
def compute_histogram(
    data: ct.tensor,        # Input data values
    histogram: ct.tensor,   # Histogram bins (output)
    num_bins: ct.scalar     # Number of bins
):
    pid = ct.program_id(0)
    
    # Compute bin index for this element
    value = data[pid]
    bin_idx = ct.cast(value * num_bins, 'int32')
    
    # Clamp to valid range
    bin_idx = ct.max(0, ct.min(bin_idx, num_bins - 1))
    
    # Atomically increment the bin counter
    ct.atomic_add(histogram, indices=[bin_idx], val=1)
```

**Example: Global Sum Reduction**

```python
@ct.kernel
def atomic_sum_reduction(
    partial_sums: ct.tensor,  # Partial sums from each block
    global_sum: ct.tensor     # Global output (single element)
):
    pid = ct.program_id(0)
    partial = partial_sums[pid]
    
    # Accumulate partial sums into global counter
    ct.atomic_add(global_sum, indices=[0], val=partial)
```

### `ct.atomic_max`

Atomic maximum computes the maximum of the current value and a new value, storing the larger value and returning the old value. Useful for finding global maximums across parallel computations.

```python
ct.atomic_max(array, indices, val, *, memory_order, memory_scope)
```

**Parameters:**

- `array` (`tl.tensor`): Target array containing elements to modify
- `indices` (`tl.tensor`): Integer tensor specifying which elements to operate on
- `val` (`tl.tensor` or scalar): Values to compare
- `memory_order` (`Optional[MemoryOrder]`): Memory ordering constraint (default: RELAXED)
- `memory_scope` (`Optional[MemoryScope]`): Visibility scope (default: BLOCK)

**Returns:** `tl.tensor` — Old values at each target location

**Supported dtypes:** All integer types, float32, float64

**Example: Global Maximum Finder**

```python
@ct.kernel
def find_global_max(
    local_maxes: ct.tensor,  # Local maximums from each block
    global_max: ct.tensor    # Global output (single element)
):
    pid = ct.program_id(0)
    local_max = local_maxes[pid]
    
    # Update global maximum if local is larger
    ct.atomic_max(global_max, indices=[0], val=local_max)
```

### `ct.atomic_min`

Atomic minimum computes the minimum of the current value and a new value, storing the smaller value and returning the old value. Useful for finding global minimums and for distance computations.

```python
ct.atomic_min(array, indices, val, *, memory_order, memory_scope)
```

**Parameters:**

- `array` (`tl.tensor`): Target array containing elements to modify
- `indices` (`tl.tensor`): Integer tensor specifying which elements to operate on
- `val` (`tl.tensor` or scalar): Values to compare
- `memory_order` (`Optional[MemoryOrder]`): Memory ordering constraint (default: RELAXED)
- `memory_scope` (`Optional[MemoryScope]`): Visibility scope (default: BLOCK)

**Returns:** `tl.tensor` — Old values at each target location

**Supported dtypes:** All integer types, float32, float64

**Example: K-Nearest Neighbors**

```python
@ct.kernel
def update_knn_distances(
    query_idx: ct.scalar,        # Which query point
    point_distances: ct.tensor,  # Distances to candidate points
    knn_dists: ct.tensor,        # Current K nearest distances (output)
    k: ct.scalar                 # Number of neighbors
):
    pid = ct.program_id(0)
    dist = point_distances[pid]
    
    # For each slot in KNN array, update if closer
    for i in range(k):
        # Atomically update minimum distance
        ct.atomic_min(
            knn_dists, 
            indices=[query_idx, i], 
            val=dist
        )
```

### `ct.atomic_and`

Atomic bitwise AND performs a bitwise AND operation between the current value and a new value, storing the result and returning the old value. Commonly used for flag manipulation and bitfield operations.

```python
ct.atomic_and(array, indices, val, *, memory_order, memory_scope)
```

**Parameters:**

- `array` (`tl.tensor`): Target array containing elements to modify
- `indices` (`tl.tensor`): Integer tensor specifying which elements to operate on
- `val` (`tl.tensor` or scalar): Values to AND with
- `memory_order` (`Optional[MemoryOrder]`): Memory ordering constraint (default: RELAXED)
- `memory_scope` (`Optional[MemoryScope]`): Visibility scope (default: BLOCK)

**Returns:** `tl.tensor` — Old values at each target location

**Supported dtypes:** All integer types (signed and unsigned), but not floating-point

**Example: Clearing Flags**

```python
@ct.kernel
def clear_flag(
    flag_array: ct.tensor,  # Array of flag bitfields
    flag_idx: ct.scalar,    # Which flag to clear
    flag_mask: ct.scalar    # Bit mask for flag
):
    pid = ct.program_id(0)
    
    # Clear bit by AND-ing with complement of mask
    ct.atomic_and(
        flag_array,
        indices=[flag_idx],
        val=~flag_mask
    )
```

### `ct.atomic_or`

Atomic bitwise OR performs a bitwise OR operation between the current value and a new value, storing the result and returning the old value. Commonly used for setting flags and combining bitfields.

```python
ct.atomic_or(array, indices, val, *, memory_order, memory_scope)
```

**Parameters:**

- `array` (`tl.tensor`): Target array containing elements to modify
- `indices` (`tl.tensor`): Integer tensor specifying which elements to operate on
- `val` (`tl.tensor` or scalar): Values to OR with
- `memory_order` (`Optional[MemoryOrder]`): Memory ordering constraint (default: RELAXED)
- `memory_scope` (`Optional[MemoryScope]`): Visibility scope (default: BLOCK)

**Returns:** `tl.tensor` — Old values at each target location

**Supported dtypes:** All integer types (signed and unsigned), but not floating-point

**Example: Setting Flags**

```python
@ct.kernel
def set_flag(
    flag_array: ct.tensor,  # Array of flag bitfields
    flag_idx: ct.scalar,    # Which flag to set
    flag_mask: ct.scalar    # Bit mask for flag
):
    pid = ct.program_id(0)
    
    # Set bit by OR-ing with mask
    ct.atomic_or(
        flag_array,
        indices=[flag_idx],
        val=flag_mask
    )
```

### `ct.atomic_xor`

Atomic bitwise XOR performs a bitwise XOR operation between the current value and a new value, storing the result and returning the old value. Useful for toggling flags and for cryptographic operations.

```python
ct.atomic_xor(array, indices, val, *, memory_order, memory_scope)
```

**Parameters:**

- `array` (`tl.tensor`): Target array containing elements to modify
- `indices` (`tl.tensor`): Integer tensor specifying which elements to operate on
- `val` (`tl.tensor` or scalar): Values to XOR with
- `memory_order` (`Optional[MemoryOrder]`): Memory ordering constraint (default: RELAXED)
- `memory_scope` (`Optional[MemoryScope]`): Visibility scope (default: BLOCK)

**Returns:** `tl.tensor` — Old values at each target location

**Supported dtypes:** All integer types (signed and unsigned), but not floating-point

**Example: Toggling Flags**

```python
@ct.kernel
def toggle_flag(
    flag_array: ct.tensor,  # Array of flag bitfields
    flag_idx: ct.scalar,    # Which flag to toggle
    flag_mask: ct.scalar    # Bit mask for flag
):
    pid = ct.program_id(0)
    
    # Toggle bit by XOR-ing with mask
    ct.atomic_xor(
        flag_array,
        indices=[flag_idx],
        val=flag_mask
    )
```

## Memory Ordering and Scope

### Memory Order Constants

Memory ordering controls how atomic operations synchronize with other memory operations:

```python
import cuda.tile as ct

# Available memory ordering options
ct.MemoryOrder.RELAXED  # No ordering guarantees (default)
ct.MemoryOrder.ACQUIRE  # Acquire semantics
ct.MemoryOrder.RELEASE  # Release semantics
ct.MemoryOrder.ACQ_REL  # Combined acquire and release
```

**RELAXED**: No ordering guarantees beyond the atomicity of the operation itself. Multiple threads may see operations in different orders.

**ACQUIRE**: Subsequent read/write operations cannot be reordered before this atomic operation. Used when consuming data produced by another thread.

**RELEASE**: Previous read/write operations cannot be reordered after this atomic operation. Used when publishing data for other threads to consume.

**ACQ_REL**: Combines both acquire and release semantics. Used for operations that both consume and produce data.

### Memory Scope Constants

Memory scope controls the visibility of atomic operations:

```python
import cuda.tile as ct

# Available memory scope options
ct.MemoryScope.BLOCK   # Within same block (default)
ct.MemoryScope.DEVICE  # All threads on same GPU
ct.MemoryScope.SYS     # All threads across system
```

**BLOCK**: Ordering only guaranteed within the same thread block. Most efficient option for intra-block synchronization.

**DEVICE**: Ordering across all threads on the same GPU. Required for inter-block synchronization on single-GPU systems.

**SYS**: Ordering across all threads in the system, including multiple GPUs and the host. Required for multi-GPU synchronization.

## Common Atomic Operation Patterns

### Pattern 1: Global Reduction using atomic_add

Global reductions are one of the most common uses of atomic operations. When each thread computes a partial result that needs to be combined into a global result, `atomic_add` provides a simple (though not always optimal) solution.

```python
@ct.kernel
def global_histogram(
    data: ct.tensor,        # Input data [N]
    histogram: ct.tensor,   # Output histogram [B]
    num_bins: ct.scalar,    # Number of bins
    min_val: ct.scalar,     # Data minimum
    max_val: ct.scalar      # Data maximum
):
    pid = ct.program_id(0)
    value = data[pid]
    
    # Normalize value to [0, 1]
    normalized = (value - min_val) / (max_val - min_val)
    
    # Compute bin index
    bin_idx = ct.cast(normalized * num_bins, 'int32')
    
    # Clamp to valid range
    bin_idx = ct.max(0, ct.min(bin_idx, num_bins - 1))
    
    # Atomically increment histogram bin
    ct.atomic_add(histogram, indices=[bin_idx], val=1)
```

**When to use:** Simple reductions where correctness is more important than peak performance. For high-performance reductions, consider using tree-based reduction patterns followed by a single atomic operation.

### Pattern 2: Lock-Free Data Structures

Lock-free data structures avoid locks by using atomic CAS operations to coordinate between threads. The basic pattern is:

1. Read current value
2. Compute new value
3. Use atomic_cas to swap if value hasn't changed
4. Retry if CAS failed

```python
@ct.kernel
def lockfree_queue_enqueue(
    queue_data: ct.tensor,   # Queue data array
    queue_head: ct.tensor,   # Head index (atomic)
    queue_tail: ct.tensor,   # Tail index (atomic)
    capacity: ct.scalar,     # Queue capacity
    value: ct.scalar         # Value to enqueue
):
    pid = ct.program_id(0)
    
    # Try to claim a slot in the queue
    while True:
        # Read current tail
        current_tail = queue_tail[0]
        
        # Compute next tail position
        next_tail = (current_tail + 1) % capacity
        
        # Try to advance tail pointer
        old_tail = ct.atomic_cas(
            queue_tail,
            indices=[0],
            compare=current_tail,
            val=next_tail
        )
        
        # If successful, we claimed the slot
        if old_tail == current_tail:
            queue_data[current_tail] = value
            break
```

**When to use:** When you need shared data structures but want to avoid lock overhead. Lock-free structures are complex and error-prone—use with caution.

### Pattern 3: Finding Global Max/Min

Finding the global maximum or minimum across all threads is straightforward with `atomic_max` and `atomic_min`:

```python
@ct.kernel
def find_global_extremes(
    local_maxes: ct.tensor,   # Local maximums
    local_mins: ct.tensor,    # Local minimums
    global_max: ct.tensor,    # Global maximum (output)
    global_min: ct.tensor     # Global minimum (output)
):
    pid = ct.program_id(0)
    
    # Initialize global values on first thread
    if pid == 0:
        global_max[0] = local_maxes[0]
        global_min[0] = local_mins[0]
    
    # Synchronize to ensure initialization is visible
    ct.sync_threads()
    
    # Update global maximum
    ct.atomic_max(
        global_max,
        indices=[0],
        val=local_maxes[pid],
        memory_scope=ct.MemoryScope.DEVICE
    )
    
    # Update global minimum
    ct.atomic_min(
        global_min,
        indices=[0],
        val=local_mins[pid],
        memory_scope=ct.MemoryScope.DEVICE
    )
```

**When to use:** When you need global extrema and the number of updates is small compared to the computational cost.

### Pattern 4: Inter-Block Synchronization

While CUDA blocks cannot directly synchronize, atomic operations can be used for inter-block coordination:

```python
@ct.kernel
def inter_block_barrier(
    counter: ct.tensor,       # Arrival counter (atomic)
    num_blocks: ct.scalar,    # Total number of blocks
    flag_array: ct.tensor     # Flag array for signaling
):
    pid = ct.program_id(0)
    block_id = ct.program_id(1)
    
    # Each block announces arrival
    arrival_order = ct.atomic_add(
        counter,
        indices=[0],
        val=1,
        memory_scope=ct.MemoryScope.DEVICE
    )
    
    # Set flag for this block
    flag_array[block_id] = 1
    
    # Wait for all blocks to arrive
    while counter[0] < num_blocks:
        pass  # Spin wait
    
    # Additional work can go here
```

**Warning:** Inter-block synchronization via spin-waiting is inefficient and can lead to deadlock. Prefer kernel launches for major synchronization points.

## Performance Considerations

### Atomic Operation Cost

Atomic operations are significantly more expensive than regular memory operations:

1. **Contention**: When multiple threads simultaneously access the same atomic location, performance degrades sharply
2. **Cache Coherency**: Atomic operations require cache coherency traffic across streaming multiprocessors
3. **Serialization**: Operations on the same address are serialized, reducing parallelism

### Mitigation Strategies

**Strategy 1: Reduce Contention**

```python
# BAD: High contention
@ct.kernel
def bad_reduction(data: ct.tensor, output: ct.tensor):
    pid = ct.program_id(0)
    # All threads hammer the same location
    ct.atomic_add(output, indices=[0], val=data[pid])

# GOOD: Lower contention
@ct.kernel
def good_reduction(data: ct.tensor, output: ct.tensor):
    pid = ct.program_id(0)
    # Each thread updates a different location
    ct.atomic_add(output, indices=[pid % output.shape[0]], val=data[pid])
```

**Strategy 2: Use Shared Memory for Intermediate Results**

```python
@ct.kernel
def optimized_reduction(
    data: ct.tensor,
    output: ct.tensor,
    shared_buffer: ct.tensor  # In shared memory
):
    pid = ct.program_id(0)
    block_id = ct.program_id(1)
    
    # First pass: accumulate in shared memory
    shared_buffer[pid % 256] = data[pid]
    ct.sync_threads()
    
    # Reduce within block
    if pid % 256 == 0:
        block_sum = 0
        for i in range(256):
            block_sum += shared_buffer[i]
        
        # Single atomic operation per block
        ct.atomic_add(
            output,
            indices=[block_id],
            val=block_sum,
            memory_scope=ct.MemoryScope.DEVICE
        )
```

**Strategy 3: Use Appropriate Memory Scope**

```python
# Use BLOCK scope when possible for intra-block operations
ct.atomic_add(
    shared_array,
    indices=[idx],
    val=1,
    memory_scope=ct.MemoryScope.BLOCK  # More efficient than DEVICE
)

# Use DEVICE scope only for inter-block operations
ct.atomic_add(
    global_array,
    indices=[idx],
    val=1,
    memory_scope=ct.MemoryScope.DEVICE
)
```

## Common Pitfalls

### Pitfall 1: Assuming Return Value is Post-Operation

Atomic operations return the **old value**, not the new value:

```python
# WRONG: Assumes return value is after addition
new_value = ct.atomic_add(counter, indices=[0], val=1)
if new_value == threshold:  # This is wrong!
    pass

# CORRECT: Add 1 to get new value
old_value = ct.atomic_add(counter, indices=[0], val=1)
new_value = old_value + 1
if new_value == threshold:
    pass
```

### Pitfall 2: Missing Memory Ordering for Producer-Consumer

```python
# WRONG: No memory ordering guarantees
@ct.kernel
def producer(data: ct.tensor, flag: ct.tensor):
    flag[0] = 1  # Publish data

@ct.kernel
def consumer(data: ct.tensor, flag: ct.tensor):
    if flag[0] == 1:  # Check flag
        value = data[0]  # May see stale data!

# CORRECT: Use proper memory ordering
@ct.kernel
def producer(data: ct.tensor, flag: ct.tensor):
    data[0] = 42  # Write data
    ct.atomic_xchg(
        flag,
        indices=[0],
        val=1,
        memory_order=ct.MemoryOrder.RELEASE  # Ensure prior writes visible
    )

@ct.kernel
def consumer(data: ct.tensor, flag: ct.tensor):
    flag_val = ct.atomic_xchg(
        flag,
        indices=[0],
        val=0,
        memory_order=ct.MemoryOrder.ACQUIRE  # Ensure subsequent reads see latest writes
    )
    if flag_val == 1:
        value = data[0]  # Guaranteed to see latest data
```

### Pitfall 3: Using Wrong Memory Scope

```python
# WRONG: Uses BLOCK scope for inter-block coordination
@ct.kernel
def inter_block_counter(counter: ct.tensor):
    block_id = ct.program_id(1)
    ct.atomic_add(
        counter,
        indices=[0],
        val=1,
        memory_scope=ct.MemoryScope.BLOCK  # Only visible within block!
    )

# CORRECT: Use DEVICE scope
@ct.kernel
def inter_block_counter(counter: ct.tensor):
    block_id = ct.program_id(1)
    ct.atomic_add(
        counter,
        indices=[0],
        val=1,
        memory_scope=ct.MemoryScope.DEVICE  # Visible across blocks
    )
```

### Pitfall 4: Race Conditions in Read-Modify-Write

```python
# WRONG: Non-atomic read-modify-write
old_value = counter[0]  # Read
# ... potential other threads modify counter here ...
new_value = old_value + 1  # Modify
counter[0] = new_value  # Write (data race!)

# CORRECT: Use atomic operation
old_value = ct.atomic_add(counter, indices=[0], val=1)
```

## Advanced Examples

### Example 1: Sparse Matrix-Vector Multiplication

```python
@ct.kernel
def spmv_csr(
    row_ptr: ct.tensor,     # CSR row pointers
    col_idx: ct.tensor,     # CSR column indices
    values: ct.tensor,      # CSR non-zero values
    x: ct.tensor,           # Input vector
    y: ct.tensor            # Output vector
):
    row = ct.program_id(0)
    
    # Accumulate dot product for this row
    row_start = row_ptr[row]
    row_end = row_ptr[row + 1]
    
    row_sum = 0.0
    for i in range(row_start, row_end):
        col = col_idx[i]
        val = values[i]
        row_sum += val * x[col]
    
    # Atomic add to output (handles duplicate row assignments)
    ct.atomic_add(y, indices=[row], val=row_sum)
```

### Example 2: Parallel Prefix Sum with Atomics

```python
@ct.kernel
def parallel_prefix_sum(
    input: ct.tensor,
    output: ct.tensor,
    counter: ct.tensor
):
    pid = ct.program_id(0)
    
    # Claim position in output array
    pos = ct.atomic_add(counter, indices=[0], val=1)
    
    # Write to claimed position
    output[pos] = input[pid]
```

### Example 3: Collision Detection

```python
@ct.kernel
def detect_collisions(
    positions: ct.tensor,    # Particle positions [N, 2]
    collision_count: ct.tensor,  # Collision counter [M]
    grid_size: ct.scalar,    # Spatial grid size
    num_bins: ct.scalar      # Number of bins per dimension
):
    pid = ct.program_id(0)
    x, y = positions[pid, 0], positions[pid, 1]
    
    # Compute grid bin
    bin_x = ct.cast(x * num_bins / grid_size, 'int32')
    bin_y = ct.cast(y * num_bins / grid_size, 'int32')
    bin_idx = bin_y * num_bins + bin_x
    
    # Atomically increment collision counter for this bin
    ct.atomic_add(collision_count, indices=[bin_idx], val=1)
```

## Summary

Atomic operations in cuTile provide powerful tools for thread-safe parallel programming:

- **Bulk Operations**: Operate on multiple elements efficiently
- **Memory Ordering**: Fine-grained control over synchronization
- **Memory Scoping**: Control visibility across thread blocks and GPUs
- **Rich Operation Set**: CAS, exchange, arithmetic, and bitwise operations

Use atomic operations judiciously—they are powerful but expensive. Prioritize algorithms that minimize contention, and always use appropriate memory ordering and scope for correctness.
