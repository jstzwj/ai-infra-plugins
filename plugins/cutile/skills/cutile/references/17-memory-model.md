# Memory Model

cuTile's memory model provides fine-grained control over memory ordering and visibility across threads, enabling correct and efficient synchronization in parallel GPU programs. This chapter explains the memory model concepts, the available ordering and scope options, and how to use them correctly.

## Overview

### The Challenge of Memory Ordering

In GPU programming, the compiler and hardware can reorder memory operations for performance optimization. Without explicit synchronization:

- **Compiler Reordering**: The compiler may reorder independent load/store operations
- **Hardware Reordering**: The GPU's memory system may reorder operations
- **Cache Effects**: Different threads may see operations in different orders
- **Write Buffering**: Writes may be buffered and not immediately visible

These reorders are invisible within a single thread but create challenges when multiple threads need to coordinate through shared memory.

### Basic Guarantees

Without explicit memory ordering attributes:

1. **Intra-thread Order**: Operations by a single thread appear in program order from that thread's perspective
2. **No Cross-thread Guarantees**: Different threads may see operations in different orders
3. **Block Isolation**: Without synchronization, there are no ordering guarantees across thread blocks

### cuTile's Solution

cuTile provides two complementary attributes for memory coordination:

1. **Memory Order**: Controls ordering constraints relative to other operations
2. **Memory Scope**: Controls which threads can see the ordered operations

Together, these enable **per-element granularity synchronization**—you can synchronize on specific array elements rather than entire memory regions.

## Memory Order: `cuda.tile.MemoryOrder`

Memory order specifies the synchronization constraints for an atomic operation. It determines how the operation orders with respect to other memory operations.

### Memory Order Constants

```python
import cuda.tile as ct

# Memory order options
ct.MemoryOrder.WEAK     # Weak (non-atomic) ordering
ct.MemoryOrder.RELAXED  # No ordering guarantees
ct.MemoryOrder.ACQUIRE  # Acquire semantics
ct.MemoryOrder.RELEASE  # Release semantics
ct.MemoryOrder.ACQ_REL  # Combined acquire and release
```

### WEAK: Weak Ordering

**Constant**: `'weak'`  
**Value**: `0`  
**Description**: Weak (non-atomic) ordering. This is the default for regular load/store operations.

WEAK ordering provides no atomicity guarantees. It's used for normal memory operations that don't require synchronization:

```python
@ct.kernel
def weak_ordering_example(data: ct.tensor, output: ct.tensor):
    pid = ct.program_id(0)
    
    # These operations use WEAK ordering by default
    value = data[pid]       # Load with WEAK ordering
    output[pid] = value * 2  # Store with WEAK ordering
    
    # No atomicity or ordering guarantees
```

**Use cases:**
- Regular memory accesses
- Thread-private data
- Operations that don't require synchronization

**When NOT to use:**
- Shared data accessed by multiple threads
- Producer-consumer patterns
- Flag-based coordination

### RELAXED: Relaxed Ordering

**Constant**: `'relaxed'`  
**Value**: `1`  
**Description**: No ordering guarantees beyond atomicity. The operation is atomic, but cannot synchronize between threads.

RELAXED ordering provides atomicity—the operation completes indivisibly—but doesn't establish any happens-before relationships with other operations:

```python
@ct.kernel
def relaxed_counter_example(counter: ct.tensor):
    pid = ct.program_id(0)
    
    # Atomic increment, but no ordering guarantees
    old_value = ct.atomic_add(
        counter,
        indices=[0],
        val=1,
        memory_order=ct.MemoryOrder.RELAXED  # Default for atomics
    )
    
    # Other threads may see this increment before or after
    # subsequent operations in this thread
```

**Use cases:**
- Simple counters where ordering doesn't matter
- Statistics collection
- Operations where only the final value matters

**When NOT to use:**
- Producer-consumer patterns
- Publishing data for other threads
- Coordinating access to shared data structures

### ACQUIRE: Acquire Semantics

**Constant**: `'acquire'`  
**Value**: `2`  
**Description**: Acquire semantics. Subsequent read/write operations cannot be reordered before this operation.

ACQUIRE ordering is used on the consuming side of a producer-consumer relationship. It ensures that all operations after the atomic operation see the effects of all operations before the corresponding RELEASE operation:

```python
@ct.kernel
def consumer_example(
    data: ct.tensor,
    ready_flag: ct.tensor
):
    pid = ct.program_id(0)
    
    # Wait for data to be ready (acquire)
    flag = ct.atomic_xchg(
        ready_flag,
        indices=[pid],
        val=0,
        memory_order=ct.MemoryOrder.ACQUIRE
    )
    
    if flag == 1:
        # Guaranteed to see all writes from producer
        value = data[pid]
        
        # All subsequent operations see consistent memory state
        output = value * 2
```

**Key properties:**
- Prevents reordering of subsequent loads/stores before the atomic operation
- Pairs with a RELEASE operation on the producer side
- Establishes a happens-before relationship

**Use cases:**
- Reading data published by another thread
- Consuming from shared data structures
- Lock acquisition

**What it guarantees:**
- All reads/writes after the ACQUIRE see all writes before the paired RELEASE
- No subsequent operations can be reordered before the ACQUIRE

### RELEASE: Release Semantics

**Constant**: `'release'`  
**Value**: `3`  
**Description**: Release semantics. Prior read/write operations cannot be reordered after this operation.

RELEASE ordering is used on the producing side of a producer-consumer relationship. It ensures that all prior operations are visible before the atomic operation that publishes the result:

```python
@ct.kernel
def producer_example(
    data: ct.tensor,
    ready_flag: ct.tensor
):
    pid = ct.program_id(0)
    
    # Write data (prior operation)
    data[pid] = compute_value(pid)
    
    # Publish data with release semantics
    ct.atomic_xchg(
        ready_flag,
        indices=[pid],
        val=1,
        memory_order=ct.MemoryOrder.RELEASE
    )
    
    # All writes before RELEASE are visible to consumers
```

**Key properties:**
- Prevents reordering of prior loads/stores after the atomic operation
- Pairs with an ACQUIRE operation on the consumer side
- Establishes a happens-before relationship

**Use cases:**
- Publishing data for other threads to consume
- Releasing locks
- Signaling completion

**What it guarantees:**
- All reads/writes before the RELEASE are visible to threads performing the paired ACQUIRE
- No prior operations can be reordered after the RELEASE

### ACQ_REL: Acquire-Release Semantics

**Constant**: `'acq_rel'`  
**Value**: `4`  
**Description**: Combined acquire and release semantics. Provides both acquire and release guarantees.

ACQ_REL ordering is used for operations that both consume data produced by other threads and produce data for other threads. It's commonly used for read-modify-write operations:

```python
@ct.kernel
def update_shared_data(
    data: ct.tensor,
    counter: ct.tensor
):
    pid = ct.program_id(0)
    
    # Read-modify-write with acquire-release semantics
    old_value = ct.atomic_add(
        counter,
        indices=[0],
        val=1,
        memory_order=ct.MemoryOrder.ACQ_REL
    )
    
    # Guarantees:
    # - All prior operations visible to others (release)
    # - All subsequent operations see latest state (acquire)
```

**Key properties:**
- Combines both ACQUIRE and RELEASE semantics
- Prevents reordering on both sides of the atomic operation
- Used for bidirectional synchronization

**Use cases:**
- Read-modify-write operations on shared data
- Lock implementations (both acquire and release)
- Barrier implementations

**What it guarantees:**
- All prior operations visible to consumers
- All subsequent operations see latest producer writes
- No operations can be reordered across the atomic operation

## Memory Scope: `cuda.tile.MemoryScope`

Memory scope specifies the visibility domain of atomic operations. It determines which threads can see the ordered operations.

### Memory Scope Constants

```python
import cuda.tile as ct

# Memory scope options
ct.MemoryScope.NONE    # No scope
ct.MemoryScope.BLOCK   # Within same block
ct.MemoryScope.DEVICE  # All threads on same GPU
ct.MemoryScope.SYS     # All threads across system
```

### NONE: No Scope

**Constant**: `'none'`  
**Value**: `0`  
**Description**: No scope. Used only with WEAK ordering for regular load/store operations.

NONE scope is used with non-atomic operations:

```python
@ct.kernel
def none_scope_example(data: ct.tensor):
    pid = ct.program_id(0)
    
    # Regular load/store use NONE scope with WEAK ordering
    value = data[pid]  # Implicitly: memory_scope=NONE
```

### BLOCK: Block Scope

**Constant**: `'block'`  
**Value**: `1`  
**Description**: Ordering and visibility guaranteed only within the same thread block.

BLOCK scope is the default for atomic operations. It provides ordering guarantees only for threads within the same block:

```python
@ct.kernel
def block_scope_example(
    shared_counter: ct.tensor,
    shared_data: ct.tensor
):
    pid = ct.program_id(0)
    
    # This atomic only synchronizes threads within this block
    ct.atomic_add(
        shared_counter,
        indices=[0],
        val=1,
        memory_scope=ct.MemoryScope.BLOCK  # Default
    )
    
    # Synchronize within block
    ct.sync_threads()
    
    # Now shared_counter is consistent within block
    value = shared_counter[0]
```

**Use cases:**
- Shared memory operations within a block
- Intra-block synchronization
- Block-local reductions

**Performance:** BLOCK scope is the most efficient option for intra-block operations.

**When NOT to use:**
- Inter-block coordination
- Multi-GPU operations
- Host-device synchronization

### DEVICE: Device Scope

**Constant**: `'device'`  
**Value**: `2`  
**Description**: Ordering and visibility guaranteed across all threads on the same GPU.

DEVICE scope provides ordering guarantees for all threads on the GPU, regardless of which block they're in:

```python
@ct.kernel
def device_scope_example(
    global_counter: ct.tensor,
    global_data: ct.tensor
):
    pid = ct.program_id(0)
    block_id = ct.program_id(1)
    
    # This atomic synchronizes across all blocks
    ct.atomic_add(
        global_counter,
        indices=[0],
        val=1,
        memory_scope=ct.MemoryScope.DEVICE  # Required for inter-block
    )
    
    # All blocks see consistent global_counter
    # (but no explicit synchronization point)
```

**Use cases:**
- Inter-block coordination
- Global reductions
- Cross-block data structures

**Performance:** DEVICE scope is more expensive than BLOCK scope due to cache coherency across streaming multiprocessors.

**When NOT to use:**
- Single-block operations (use BLOCK instead)
- Multi-GPU operations (use SYS instead)

### SYS: System Scope

**Constant**: `'sys'`  
**Value**: `3`  
**Description**: Ordering and visibility guaranteed across all threads in the system, including multiple GPUs and the host.

SYS scope provides the strongest ordering guarantees, ensuring visibility across all devices and the host:

```python
@ct.kernel
def system_scope_example(
    global_counter: ct.tensor,
    flags: ct.tensor
):
    pid = ct.program_id(0)
    
    # This atomic synchronizes across GPUs and host
    ct.atomic_add(
        global_counter,
        indices=[0],
        val=1,
        memory_scope=ct.MemoryOrder.ACQ_REL,
        memory_scope=ct.MemoryScope.SYS  # Multi-GPU
    )
    
    # All devices and host see consistent state
```

**Use cases:**
- Multi-GPU coordination
- Host-device synchronization
- Cross-device data structures

**Performance:** SYS scope is the most expensive option. Use only when necessary.

## Synchronization Patterns

### Pattern 1: Producer-Consumer Between Blocks

The classic producer-consumer pattern uses RELEASE on the producer and ACQUIRE on the consumer:

```python
@ct.kernel
def producer(
    data: ct.tensor,
    ready_flag: ct.tensor,
    block_id: ct.scalar
):
    pid = ct.program_id(0)
    
    # Write data to shared location
    data[block_id, pid] = compute_value(pid)
    
    # Synchronize within block first
    ct.sync_threads()
    
    # Publish data with RELEASE (only first thread needs to publish)
    if pid == 0:
        ct.atomic_xchg(
            ready_flag,
            indices=[block_id],
            val=1,
            memory_order=ct.MemoryOrder.RELEASE,
            memory_scope=ct.MemoryScope.DEVICE
        )

@ct.kernel
def consumer(
    data: ct.tensor,
    ready_flag: ct.tensor,
    block_id: ct.scalar
):
    pid = ct.program_id(0)
    
    # Wait for data with ACQUIRE
    if pid == 0:
        while ct.load(ready_flag[block_id]) == 0:
            pass  # Spin wait
        
        # Acquire ensures we see the data
        ct.atomic_xchg(
            ready_flag,
            indices=[block_id],
            val=0,
            memory_order=ct.MemoryOrder.ACQUIRE,
            memory_scope=ct.MemoryScope.DEVICE
        )
    
    # Synchronize within block
    ct.sync_threads()
    
    # Now all threads in this block see the data
    value = data[block_id, pid]
```

**Key points:**
- Producer uses RELEASE to ensure data writes complete before flag
- Consumer uses ACQUIRE to ensure flag read happens before data reads
- DEVICE scope ensures visibility across blocks

### Pattern 2: Global Barrier Implementation

Implementing a global barrier across all blocks requires careful coordination:

```python
@ct.kernel
def global_barrier_phase1(
    arrival_counter: ct.tensor,
    departure_flag: ct.tensor
):
    # Each block announces arrival
    block_id = ct.program_id(1)
    
    # Atomic increment with RELEASE
    # Ensures all prior work is visible
    ct.atomic_add(
        arrival_counter,
        indices=[0],
        val=1,
        memory_order=ct.MemoryOrder.RELEASE,
        memory_scope=ct.MemoryScope.DEVICE
    )
    
    # Wait for all blocks
    pid = ct.program_id(0)
    if pid == 0:
        while ct.load(arrival_counter[0]) < num_blocks:
            pass

@ct.kernel
def global_barrier_phase2(
    arrival_counter: ct.tensor,
    departure_flag: ct.tensor
):
    # Reset counter and signal departure
    block_id = ct.program_id(1)
    pid = ct.program_id(0)
    
    if pid == 0:
        # Reset with ACQUIRE
        ct.atomic_xchg(
            arrival_counter,
            indices=[0],
            val=0,
            memory_order=ct.MemoryOrder.ACQUIRE,
            memory_scope=ct.MemoryScope.DEVICE
        )
```

**Warning:** Global barriers via spin-waiting are inefficient. Prefer kernel launches for major synchronization points.

### Pattern 3: Lock-Free Queue

A lock-free queue demonstrates proper use of memory ordering:

```python
@ct.kernel
def enqueue(
    queue: ct.tensor,
    head: ct.tensor,
    tail: ct.tensor,
    capacity: ct.scalar,
    value: ct.scalar
):
    pid = ct.program_id(0)
    
    while True:
        # Read current tail
        current_tail = ct.load(tail[0])
        
        # Check if queue is full
        next_tail = (current_tail + 1) % capacity
        if next_tail == ct.load(head[0]):
            return  # Queue full
        
        # Try to claim slot with ACQ_REL
        old_tail = ct.atomic_cas(
            tail,
            indices=[0],
            compare=current_tail,
            val=next_tail,
            memory_order=ct.MemoryOrder.ACQ_REL,
            memory_scope=ct.MemoryScope.DEVICE
        )
        
        if old_tail == current_tail:
            # Success: write value with RELEASE
            queue[current_tail] = value
            ct.atomic_store(
                queue,
                indices=[current_tail],
                val=value,
                memory_order=ct.MemoryOrder.RELEASE
            )
            break

@ct.kernel
def dequeue(
    queue: ct.tensor,
    head: ct.tensor,
    tail: ct.tensor,
    capacity: ct.scalar,
    output: ct.tensor
):
    pid = ct.program_id(0)
    
    while True:
        # Read current head
        current_head = ct.load(head[0])
        
        # Check if queue is empty
        if current_head == ct.load(tail[0]):
            return  # Queue empty
        
        # Try to claim slot with ACQ_REL
        old_head = ct.atomic_cas(
            head,
            indices=[0],
            compare=current_head,
            val=(current_head + 1) % capacity,
            memory_order=ct.MemoryOrder.ACQ_REL,
            memory_scope=ct.MemoryScope.DEVICE
        )
        
        if old_head == current_head:
            # Success: read value with ACQUIRE
            value = ct.atomic_load(
                queue,
                indices=[current_head],
                memory_order=ct.MemoryOrder.ACQUIRE
            )
            output[pid] = value
            break
```

### Pattern 4: Memory Fence

Explicit memory fences can establish ordering without atomic operations:

```python
@ct.kernel
def memory_fence_example(
    data: ct.tensor,
    flag: ct.tensor
):
    pid = ct.program_id(0)
    
    # Write data
    data[pid] = compute_value(pid)
    
    # Ensure all writes are visible
    ct.fence(
        memory_order=ct.MemoryOrder.RELEASE,
        memory_scope=ct.MemoryScope.DEVICE
    )
    
    # Now safe to set flag
    flag[pid] = 1
```

**Note:** cuTile provides explicit fence operations for fine-grained control.

## When to Use Each Memory Order

### Decision Tree

```
Is the operation atomic?
├─ No → Use WEAK (default for load/store)
└─ Yes → Is synchronization needed?
    ├─ No → Use RELAXED (default for atomics)
    └─ Yes → What is the pattern?
        ├─ Publishing data → Use RELEASE
        ├─ Consuming data → Use ACQUIRE
        └─ Both → Use ACQ_REL
```

### RELAXED

**Use when:**
- Only atomicity matters, not ordering
- Simple counters or statistics
- Operations where only the final value is important

**Example:**
```python
# Simple counter for statistics
ct.atomic_add(
    stats_counter,
    indices=[0],
    val=1,
    memory_order=ct.MemoryOrder.RELAXED
)
```

### ACQUIRE

**Use when:**
- Reading data published by another thread
- Consuming from a producer-consumer queue
- Acquiring a lock

**Example:**
```python
# Acquire lock
while ct.atomic_xchg(
    lock,
    indices=[0],
    val=1,
    memory_order=ct.MemoryOrder.ACQUIRE
) != 0:
    pass  # Spin
```

### RELEASE

**Use when:**
- Publishing data for other threads
- Releasing a lock
- Signaling completion

**Example:**
```python
# Release lock
ct.atomic_xchg(
    lock,
    indices=[0],
    val=0,
    memory_order=ct.MemoryOrder.RELEASE
)
```

### ACQ_REL

**Use when:**
- Both consuming and producing data
- Implementing locks (both acquire and release)
- Read-modify-write operations on shared data

**Example:**
```python
# Read-modify-write on shared data
ct.atomic_add(
    shared_counter,
    indices=[0],
    val=1,
    memory_order=ct.MemoryOrder.ACQ_REL
)
```

## Common Mistakes and Pitfalls

### Mistake 1: Missing Memory Ordering

```python
# WRONG: No memory ordering for producer-consumer
@ct.kernel
def producer(data: ct.tensor, flag: ct.tensor):
    data[0] = 42
    flag[0] = 1  # No RELEASE!

@ct.kernel
def consumer(data: ct.tensor, flag: ct.tensor):
    if flag[0] == 1:  # No ACQUIRE!
        value = data[0]  # May see stale data

# CORRECT: Use proper ordering
@ct.kernel
def producer(data: ct.tensor, flag: ct.tensor):
    data[0] = 42
    ct.atomic_xchg(
        flag,
        indices=[0],
        val=1,
        memory_order=ct.MemoryOrder.RELEASE  # Ensure data visible
    )

@ct.kernel
def consumer(data: ct.tensor, flag: ct.tensor):
    if ct.atomic_load(
        flag,
        indices=[0],
        memory_order=ct.MemoryOrder.ACQUIRE  # Ensure data visible
    ) == 1:
        value = data[0]  # Guaranteed to see latest data
```

### Mistake 2: Wrong Memory Scope

```python
# WRONG: Using BLOCK scope for inter-block coordination
@ct.kernel
def inter_block_counter(counter: ct.tensor):
    block_id = ct.program_id(1)
    ct.atomic_add(
        counter,
        indices=[0],
        val=1,
        memory_scope=ct.MemoryScope.BLOCK  # Wrong! Only visible within block
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

### Mistake 3: Over-Using Strong Ordering

```python
# INEFFICIENT: Using ACQ_REL when RELAXED suffices
@ct.kernel
def simple_counter(counter: ct.tensor):
    pid = ct.program_id(0)
    ct.atomic_add(
        counter,
        indices=[0],
        val=1,
        memory_order=ct.MemoryOrder.ACQ_REL  # Overkill!
    )

# EFFICIENT: Use RELAXED for simple operations
@ct.kernel
def simple_counter(counter: ct.tensor):
    pid = ct.program_id(0)
    ct.atomic_add(
        counter,
        indices=[0],
        val=1,
        memory_order=ct.MemoryOrder.RELAXED  # Correct
    )
```

### Mistake 4: Assuming Sequential Consistency

```python
# WRONG: Assuming all threads see operations in the same order
@ct.kernel
def writer(data1: ct.tensor, data2: ct.tensor):
    data1[0] = 1
    data2[0] = 2  # May be reordered!

@ct.kernel
def reader(data1: ct.tensor, data2: ct.tensor):
    # May see data2[0]=2 before data1[0]=1!
    val1 = data1[0]
    val2 = data2[0]

# CORRECT: Use explicit ordering
@ct.kernel
def writer(data1: ct.tensor, data2: ct.tensor):
    data1[0] = 1
    ct.fence(memory_order=ct.MemoryOrder.RELEASE)
    data2[0] = 2  # Guaranteed to be after data1 write
```

### Mistake 5: Forgetting sync_threads

```python
# WRONG: Atomic doesn't synchronize threads within block
@ct.kernel
def block_reduction(data: ct.tensor, output: ct.tensor):
    pid = ct.program_id(0)
    
    # First thread writes to shared memory
    if pid == 0:
        output[0] = data[0]
    
    # Other threads might not see this write!
    value = output[0]  # May be stale!

# CORRECT: Use sync_threads
@ct.kernel
def block_reduction(data: ct.tensor, output: ct.tensor):
    pid = ct.program_id(0)
    
    if pid == 0:
        output[0] = data[0]
    
    ct.sync_threads()  # Ensure all threads see the write
    
    value = output[0]  # Guaranteed to see latest value
```

## Performance Implications

### Ordering Cost Hierarchy

From least to most expensive:

1. **WEAK** (load/store) — Baseline cost
2. **RELAXED** — Atomic operations only
3. **ACQUIRE/RELEASE** — Ordering within scope
4. **ACQ_REL** — Both directions
5. **With larger scope** — BLOCK < DEVICE < SYS

### Scope Cost Hierarchy

1. **BLOCK** — Fastest (shared memory or L1 cache)
2. **DEVICE** — Moderate (cross-SMP cache coherency)
3. **SYS** — Slowest (cross-device coherence)

### Optimization Guidelines

1. **Use the weakest ordering that provides correctness**
2. **Use the smallest scope that covers all participating threads**
3. **Prefer block-local operations when possible**
4. **Batch operations to reduce synchronization overhead**
5. **Consider algorithmic alternatives to heavy synchronization**

## Advanced Topics

### Specifying Memory Order and Scope

Memory order and scope can be specified on any atomic operation:

```python
result = ct.atomic_add(
    array,
    indices=[idx],
    val=increment,
    memory_order=ct.MemoryOrder.ACQ_REL,
    memory_scope=ct.MemoryScope.DEVICE
)
```

### Default Values

- **Atomic operations**: Default to `RELAXED` ordering, `BLOCK` scope
- **Load/store operations**: Default to `WEAK` ordering, `NONE` scope

### Combining Ordering and Scope

The combination of ordering and scope provides precise control:

```python
# Inter-block producer-consumer
ct.atomic_xchg(
    flag,
    indices=[0],
    val=1,
    memory_order=ct.MemoryOrder.RELEASE,  # Publish prior writes
    memory_scope=ct.MemoryScope.DEVICE    # Visible across GPU
)
```

## Summary

cuTile's memory model provides:

- **Fine-grained control**: Per-element synchronization
- **Flexible ordering**: WEAK, RELAXED, ACQUIRE, RELEASE, ACQ_REL
- **Scoping**: NONE, BLOCK, DEVICE, SYS
- **Correctness**: Enables safe concurrent programming
- **Performance**: Allows optimization through careful ordering

**Key principles:**

1. Use weakest ordering that ensures correctness
2. Use smallest scope that covers all threads
3. Pair RELEASE with ACQUIRE for producer-consumer
4. Use ACQ_REL for read-modify-write operations
5. Remember that sync_threads() is still needed for intra-block synchronization

Understanding and properly using the memory model is essential for writing correct and efficient concurrent GPU programs in cuTile.
