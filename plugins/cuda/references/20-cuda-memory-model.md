# 20. CUDA C++ Memory Model and Execution Model

This section describes the CUDA C++ memory model, including thread scopes, atomicity guarantees, data race rules, message passing patterns, the execution model's forward progress guarantees, and synchronization primitives.

---

## 20.1 Thread Scopes

CUDA defines a hierarchy of thread scopes that determine the visibility of memory operations. Each atomic or barrier operation can specify the scope at which it is effective.

```cpp
namespace cuda {
    enum thread_scope {
        thread_scope_system,    // all threads in the system (host + all GPUs)
        thread_scope_device,    // all threads on the current GPU device
        thread_scope_block,     // all threads in the current thread block
        thread_scope_thread     // a single thread
    };
}
```

### 20.1.1 Scope Hierarchy

The scopes form a strict hierarchy:

```
thread_scope_system   (largest scope)
    |
thread_scope_device
    |
thread_scope_block
    |
thread_scope_thread  (smallest scope)
```

**Rules**:
- A scope S1 is at least as large as scope S2 if S1 encompasses S2 in the hierarchy above.
- `thread_scope_system >= thread_scope_device >= thread_scope_block >= thread_scope_thread`.
- An operation at a narrower scope may be implemented more efficiently than one at a wider scope, since it does not need to ensure visibility beyond that scope.

### 20.1.2 Scope Selection Guidelines

| Scenario | Recommended Scope |
|---|---|
| Synchronizing between threads in the same block | `thread_scope_block` |
| Synchronizing between threads in different blocks on the same GPU | `thread_scope_device` |
| Synchronizing between GPU threads and host threads, or between GPUs | `thread_scope_system` |
| Single-thread atomic RMW (read-modify-write) | `thread_scope_thread` |
| Performance-critical inner loops with block-level sharing | `thread_scope_block` |

```cpp
#include <cuda/atomic>

__global__ void scope_example(int* global_data, int* shared_flag) {
    // Block-scoped atomic: efficient, only visible within this block
    cuda::atomic_ref<int, cuda::thread_scope_block> block_atomic(*shared_flag);
    block_atomic.fetch_add(1, cuda::memory_order_relaxed);

    // Device-scoped atomic: visible to all threads on this GPU
    cuda::atomic_ref<int, cuda::thread_scope_device> device_atomic(global_data[0]);
    device_atomic.fetch_add(1, cuda::memory_order_relaxed);

    // System-scoped atomic: visible across host and all GPUs
    cuda::atomic_ref<int, cuda::thread_scope_system> system_atomic(global_data[1]);
    system_atomic.store(42, cuda::memory_order_release);
}
```

---

## 20.2 Atomicity

### 20.2.1 Definition

An atomic operation is one that no other thread can observe in a partially-completed state. In CUDA, an atomic operation A is **atomic at scope S** if:

1. **Scope is not `thread_scope_system`**: The operation is atomic at any scope smaller than `thread_scope_system`. All atomic operations in CUDA are atomic at `thread_scope_block` and `thread_scope_device` scopes by default.

2. **Scope is `thread_scope_system` AND specific conditions are met**: The atomic operation is atomic at `thread_scope_system` only if:
   - The memory location resides in **page-locked (pinned) host memory** or **device memory** that is accessible to all participating threads, AND
   - The operation uses appropriate memory ordering (at least `memory_order_acq_rel` for read-modify-write operations when cross-device visibility is required).

### 20.2.2 Atomic Operations

CUDA supports the following atomic operations (available via `cuda::atomic<T, Scope>` and `cuda::atomic_ref<T, Scope>`):

| Operation | Description |
|---|---|
| `store(val, order)` | Write value atomically |
| `load(order)` | Read value atomically |
| `exchange(val, order)` | Atomically replace with val, return old |
| `compare_exchange_weak(expected, desired, order)` | CAS: weak form (may spuriously fail) |
| `compare_exchange_strong(expected, desired, order)` | CAS: strong form (no spurious failure) |
| `fetch_add(val, order)` | Atomic add, return old value |
| `fetch_sub(val, order)` | Atomic subtract, return old value |
| `fetch_and(val, order)` | Atomic bitwise AND, return old value |
| `fetch_or(val, order)` | Atomic bitwise OR, return old value |
| `fetch_xor(val, order)` | Atomic bitwise XOR, return old value |
| `fetch_min(val, order)` | Atomic minimum, return old value |
| `fetch_max(val, order)` | Atomic maximum, return old value |
| `fetch_key_val(key, val, order)` | Atomic key-value update (CC 9.0+) |

```cpp
#include <cuda/atomic>

__global__ void atomicity_example(int* counter, int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Block-scoped atomic counter
    cuda::atomic_ref<int, cuda::thread_scope_block> block_ctr(counter[0]);
    int my_rank = block_ctr.fetch_add(1, cuda::memory_order_relaxed);

    // Device-scoped atomic update
    cuda::atomic_ref<int, cuda::thread_scope_device> dev_ctr(counter[1]);
    dev_ctr.fetch_add(1, cuda::memory_order_acq_rel);

    // System-scoped atomic for host visibility
    cuda::atomic_ref<int, cuda::thread_scope_system> sys_flag(counter[2]);
    sys_flag.store(1, cuda::memory_order_release);
}
```

### 20.2.3 System-Scope Atomicity Constraints

For `thread_scope_system` atomicity, the following conditions must be met:

- **Memory must be accessible from all agents**: The memory location must be in pinned host memory, or managed memory, or device memory mapped to other devices via unified memory or peer access.
- **Unified Virtual Addressing (UVA)**: The system must support UVA, which is available on all 64-bit platforms with compute capability 2.0+.
- **Atomic operations on system-scope memory**: Use `cuda::atomic<T, cuda::thread_scope_system>` or the `atomicAdd_system` family of functions.

```cpp
// System-scope atomic (host + GPU visible)
__global__ void system_atomic_example(cuda::atomic<int, cuda::thread_scope_system>* flag) {
    // This store is visible to the host and other GPUs
    flag->store(1, cuda::memory_order_release);
}

// Host side:
// cuda::atomic<int, cuda::thread_scope_system>* flag;
// cudaMallocManaged(&flag, sizeof(cuda::atomic<int, cuda::thread_scope_system>));
// system_atomic_example<<<1, 1>>>(flag);
// cudaDeviceSynchronize();
// int val = flag->load(cuda::memory_order_acquire);
// assert(val == 1);
```

---

## 20.3 Data Races

### 20.3.1 Definition

A **data race** occurs when:
1. Two or more threads access the same memory location, AND
2. At least one of the accesses is a write, AND
3. The accesses are potentially concurrent (not ordered by synchronization), AND
4. At least one of the accesses is not atomic at a scope that includes the other thread(s).

If a data race exists, the program has **undefined behavior**. This means the result of the conflicting accesses is unpredictable: the program may crash, produce incorrect results, or appear to work correctly in some runs but fail in others.

### 20.3.2 Examples

```cpp
// DATA RACE: Two threads write to the same non-atomic variable
__global__ void data_race_bad(int* data) {
    int idx = threadIdx.x;
    // Thread 0 and thread 1 both write to data[0] -- DATA RACE
    data[0] = idx;  // UB: conflicting non-atomic writes
}

// CORRECT: Use atomic operations
__global__ void data_race_fixed(cuda::atomic<int, cuda::thread_scope_block>* data) {
    data->store(threadIdx.x, cuda::memory_order_relaxed);
}

// DATA RACE: Write without proper synchronization across blocks
__device__ int flag;
__device__ int value;

__global__ void cross_block_race() {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        value = 42;
        // Missing memory fence / atomic release!
        flag = 1; // non-atomic write
    }
    if (blockIdx.x == 1 && threadIdx.x == 0) {
        if (flag == 1) { // non-atomic read -- may not see value=42
            // DATA RACE: value may or may not be 42
        }
    }
}
```

### 20.3.3 Well-Defined Concurrent Operations

The following concurrent invocations do NOT introduce data races:
- **`barrier`**: Concurrent `arrive` and `wait` operations on the same `cuda::barrier` are well-defined.
- **`latch`**: Concurrent `count_down` and `wait` operations on the same `cuda::latch` are well-defined.
- **`counting_semaphore`**: Concurrent `release` and `acquire` operations are well-defined.
- **Atomic operations**: Concurrent atomic operations on the same memory location, at compatible scopes, are well-defined.

```cpp
// Well-defined: concurrent barrier operations
__global__ void barrier_no_race() {
    __shared__ cuda::barrier<cuda::thread_scope_block> bar;
    auto tok = bar.arrive();  // concurrent arrives are safe
    // ... do work ...
    bar.wait(std::move(tok)); // concurrent waits are safe
}

// Well-defined: concurrent atomic operations
__global__ void atomic_no_race(cuda::atomic<int, cuda::thread_scope_device>* counter) {
    counter->fetch_add(1, cuda::memory_order_relaxed); // safe concurrent RMW
}
```

### 20.3.4 Rules for Avoiding Data Races

1. **Always use atomics for shared writable data** when threads from different blocks or devices access it concurrently.
2. **Use appropriate memory ordering**: `memory_order_relaxed` for simple atomicity, `memory_order_release`/`memory_order_acquire` for synchronization, `memory_order_seq_cst` for total ordering.
3. **Use scopes at least as wide as the participating threads**: If threads from different blocks communicate, use `thread_scope_device` or wider.
4. **Use barriers/fences to establish ordering**: `__syncthreads()` for block-level, `cuda::barrier` for explicit scope, `__threadfence()` for device-wide, `__threadfence_system()` for system-wide.
5. **Never mix atomic and non-atomic access to the same location** from potentially concurrent threads.

---

## 20.4 Message Passing Example

The classic message passing pattern demonstrates how to safely communicate data between threads using atomics and memory ordering.

### 20.4.1 Correct Pattern (Device Scope)

```cpp
#include <cuda/atomic>

__device__ int x;  // message payload
__device__ cuda::atomic<int, cuda::thread_scope_device> f{0};  // flag

__global__ void producer_consumer() {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // Producer: write data, then release flag
        x = 42;
        f.store(1, cuda::memory_order_release);
    }

    if (blockIdx.x == 1 && threadIdx.x == 0) {
        // Consumer: acquire flag, then read data
        while (f.load(cuda::memory_order_acquire) != 1) {
            // spin-wait
        }
        // GUARANTEED: x == 42
        // The release-acquire pair ensures that the write to x
        // happens-before the read of x.
        assert(x == 42);
    }
}
```

**Why this works**:
- `memory_order_release` on the store ensures all prior writes (including `x = 42`) are visible before the flag store becomes visible.
- `memory_order_acquire` on the load ensures all subsequent reads (including reading `x`) see values at least as recent as the flag load.
- Together, the release-acquire pair creates a **synchronizes-with** relationship, establishing **happens-before** ordering.

### 20.4.2 Undefined Behavior: Mismatched Scopes

```cpp
__device__ int x;
__device__ cuda::atomic<int, cuda::thread_scope_block> block_flag{0};

__global__ void mismatched_scope_bad() {
    // Thread 0, Block 0: uses BLOCK scope
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        x = 42;
        block_flag.store(1, cuda::memory_order_release);
        // block_flag is only visible within Block 0!
        // Block 1 has no guarantee of seeing this store.
    }

    // Thread 0, Block 1: tries to read BLOCK-scoped flag
    if (blockIdx.x == 1 && threadIdx.x == 0) {
        // UB: block_flag is block-scoped but we're in a different block
        // The atomic's scope does not encompass Block 1's threads
        while (block_flag.load(cuda::memory_order_acquire) != 1) {}
        // x == 42 is NOT guaranteed -- data race!
        assert(x == 42); // may fail
    }
}
```

**The fix**: Use `thread_scope_device` for cross-block communication:

```cpp
__device__ int x;
__device__ cuda::atomic<int, cuda::thread_scope_device> device_flag{0};

__global__ void matched_scope_fixed() {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        x = 42;
        device_flag.store(1, cuda::memory_order_release);
    }

    if (blockIdx.x == 1 && threadIdx.x == 0) {
        while (device_flag.load(cuda::memory_order_acquire) != 1) {}
        assert(x == 42); // guaranteed
    }
}
```

### 20.4.3 Message Passing Within a Block

```cpp
__global__ void block_message_passing() {
    __shared__ int shared_x;
    __shared__ cuda::atomic<int, cuda::thread_scope_block> shared_flag;

    if (threadIdx.x == 0) {
        // Producer thread
        shared_x = 100;
        shared_flag.store(1, cuda::memory_order_release);
    }

    // All other threads wait
    if (threadIdx.x != 0) {
        while (shared_flag.load(cuda::memory_order_acquire) != 1) {}
        // shared_x == 100 is guaranteed
        assert(shared_x == 100);
    }
}
```

### 20.4.4 System-Scope Message Passing (GPU <-> Host)

```cpp
// Device code
__global__ void gpu_producer(cuda::atomic<int, cuda::thread_scope_system>* flag,
                              int* data) {
    if (threadIdx.x == 0) {
        *data = 42;
        flag->store(1, cuda::memory_order_release);
    }
}

// Host code
void host_consumer(cuda::atomic<int, cuda::thread_scope_system>* flag, int* data) {
    while (flag->load(cuda::memory_order_acquire) != 1) {
        // spin-wait (host side)
    }
    assert(*data == 42); // guaranteed by release-acquire
}
```

---

## 20.5 Execution Model

### 20.5.1 Parallel Forward Progress

CUDA provides the following forward progress guarantees:

- **Thread block**: Once a thread in a thread block has been scheduled on an SM and has begun execution, the CUDA runtime guarantees that all threads in that block will eventually be scheduled. In other words, threads within a block make **parallel forward progress** relative to each other.

- **Cluster (CC 9.0+)**: Once any thread block in a cluster has begun execution, all thread blocks in that cluster will eventually be scheduled. Thread blocks within a cluster make parallel forward progress.

- **Grid**: Threads across different thread blocks (not in the same cluster) do NOT have parallel forward progress guarantees relative to each other. The runtime may schedule blocks in any order, and there is no guarantee that a block will be scheduled promptly once another block has made progress.

- **Device**: Once a device thread begins executing, it will eventually make progress as long as:
  1. It does not execute an infinite loop without observable side effects.
  2. It does not execute a blocking operation that depends on another thread that is not making progress.

### 20.5.2 Restrictions on Device Threads

Device threads do **NOT** support the following patterns:

1. **`std::this_thread::yield()`**: Not available in device code. There is no mechanism for a thread to voluntarily yield its execution slot.

2. **Volatile automatic storage loops**: A loop that reads a `volatile` automatic (stack) variable, where the loop has no other observable side effects, may be optimized away or may not terminate. For example:
   ```cpp
   __device__ void bad_volatile_loop() {
       volatile int flag = 0;
       while (flag == 0) {
           // This may not work: volatile on automatic storage
           // does not guarantee visibility from other threads
       }
   }
   ```

3. **Atomic automatic storage loops**: A loop that reads an atomic variable in automatic (stack) storage, waiting for another thread to modify it, will not work because automatic storage is private to each thread:
   ```cpp
   __device__ void bad_atomic_loop() {
       cuda::atomic<int, cuda::thread_scope_block> local_flag{0};
       // No other thread can see local_flag (it's on the stack!)
       while (local_flag.load(cuda::memory_order_acquire) == 0) {
           // Infinite loop: no other thread can modify local_flag
       }
   }
   ```

4. **Trivial infinite loops**: A loop that performs no observable side effects and has no exit condition is undefined behavior:
   ```cpp
   __device__ void bad_infinite_loop() {
       while (true) {
           // No side effects, no exit -- UB
       }
   }
   ```

### 20.5.3 Observable Side Effects

The following are considered observable side effects in CUDA device code:
- Writing to global or shared memory that may be read by other threads or the host.
- Atomic operations with side effects visible to other threads.
- Calling `printf()` (limited buffer).
- Terminating the kernel (e.g., `assert()` failure).

---

## 20.6 CUDA API Forward Progress

### 20.6.1 API Call Guarantees

Every CUDA API call (host-side or device-side) must satisfy one of the following:
1. The call **eventually returns** to the caller, OR
2. The call **ensures progress** of device threads (e.g., `cudaDeviceSynchronize()`).

This guarantee ensures that a well-formed CUDA program does not hang due to a host-side API call.

### 20.6.2 Polling Patterns

When using a polling pattern to check for GPU completion, the host thread must repeatedly call a query API:

```cpp
// Correct polling pattern
cudaStream_t stream;
cudaStreamCreate(&stream);

kernel<<<grid, block, 0, stream>>>();

// Poll for completion
while (cudaStreamQuery(stream) == cudaErrorNotReady) {
    // cudaStreamQuery returns eventually, allowing the host to make progress.
    // This is NOT a busy-wait on a volatile flag -- it's an API call.
    // The host can do other useful work here.
    do_other_work();
}

// Stream is complete
```

**Important**: Do NOT use a volatile flag or atomic variable in host memory polled by the host thread as a signal from the GPU, unless it is updated via a system-scope atomic from device code AND the host periodically calls a CUDA API to ensure the device makes forward progress.

### 20.6.3 Stream Dependencies

Cross-stream ordering in CUDA is achieved through stream dependencies (events):

```cpp
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

cudaEvent_t event;
cudaEventCreate(&event);

// Launch work on stream1
kernel1<<<grid1, block1, 0, stream1>>>();

// Record event after kernel1
cudaEventRecord(event, stream1);

// Make stream2 wait for the event
cudaStreamWaitEvent(stream2, event);

// kernel2 on stream2 will not start until kernel1 completes
kernel2<<<grid2, block2, 0, stream2>>>();
```

### 20.6.4 Callback Forward Progress

Callbacks registered with `cudaStreamAddCallback` execute on the host. They receive the stream's error status and a user-provided argument. The callback function must:
- Not call CUDA API functions within the callback (except `cudaStreamGetFlags` and a few others).
- Return promptly to allow the stream to continue.

---

## 20.7 Synchronization Primitives

CUDA provides a set of synchronization primitives that mirror the C++ standard library equivalents but are parameterized with CUDA thread scopes. Types in `cuda::std::` and `std::` namespaces have the same behavior as their `cuda::` counterparts when instantiated with `cuda::thread_scope_system`.

### 20.7.1 `cuda::atomic<T, Scope>`

A standalone atomic object. Behaves like `std::atomic<T>` but operates at the specified CUDA thread scope.

```cpp
#include <cuda/atomic>

// Declaration and initialization
__device__ cuda::atomic<int, cuda::thread_scope_device> device_counter{0};
__device__ cuda::atomic<int, cuda::thread_scope_system> system_flag{0};
__shared__ cuda::atomic<int, cuda::thread_scope_block> block_counter;

__global__ void atomic_example() {
    // Block-scoped atomic (shared memory)
    cuda::atomic<int, cuda::thread_scope_block> block_val{0};
    // (shared memory atomics must be initialized carefully)

    // Supported operations:
    device_counter.store(42, cuda::memory_order_release);
    int val = device_counter.load(cuda::memory_order_acquire);
    int old = device_counter.fetch_add(1, cuda::memory_order_acq_rel);
    int old2 = device_counter.exchange(0, cuda::memory_order_acq_rel);

    int expected = 42;
    bool success = device_counter.compare_exchange_strong(
        expected, 100, cuda::memory_order_acq_rel);

    device_counter.wait(0, cuda::memory_order_acquire);  // block until != 0 (CC 7.0+)
    device_counter.notify_one();  // wake one waiting thread (CC 7.0+)
    device_counter.notify_all();  // wake all waiting threads (CC 7.0+)
}
```

**Supported types**: `int`, `unsigned int`, `long`, `unsigned long`, `long long`, `unsigned long long`, `float`, `double`, and any trivially copyable type that fits within 8 bytes.

### 20.7.2 `cuda::atomic_ref<T, Scope>`

A non-owning reference to an atomic value. Useful when the underlying memory was not declared as `cuda::atomic`.

```cpp
#include <cuda/atomic>

__device__ int regular_int;  // non-atomic declaration

__global__ void atomic_ref_example() {
    // Create an atomic reference to a regular int
    cuda::atomic_ref<int, cuda::thread_scope_device> atomic_int(regular_int);

    // All standard atomic operations are available
    atomic_int.store(42, cuda::memory_order_release);
    int val = atomic_int.load(cuda::memory_order_acquire);
    int old = atomic_int.fetch_add(1, cuda::memory_order_relaxed);

    // CAS
    int expected = 43;
    bool success = atomic_int.compare_exchange_weak(
        expected, 100, cuda::memory_order_acq_rel, cuda::memory_order_relaxed);
}
```

**Key difference from `cuda::atomic`**: `cuda::atomic_ref` does not own the underlying storage. It provides atomic access to memory that may have been allocated or declared through non-atomic means.

### 20.7.3 `cuda::barrier<Scope>`

A barrier that synchronizes a fixed number of threads. All participating threads must arrive before any thread can proceed.

```cpp
#include <cuda/barrier>

// Block-scoped barrier for 256 threads
__global__ void barrier_example() {
    __shared__ cuda::barrier<cuda::thread_scope_block> bar;

    // One thread initializes the barrier with the expected arrival count
    if (threadIdx.x == 0) {
        init(&bar, blockDim.x);
    }
    __syncthreads(); // ensure initialization is visible

    // Each thread arrives and waits
    bar.arrive_and_wait();

    // All threads have arrived -- safe to proceed

    // Alternative: separate arrive and wait
    auto token = bar.arrive();
    // ... do work that does not depend on other threads arriving ...
    bar.wait(std::move(token));
}
```

**Construction and initialization**:

```cpp
// C++20-style initialization
__shared__ cuda::barrier<cuda::thread_scope_block> bar;
// Must be initialized before use (e.g., in thread 0):
if (threadIdx.x == 0) {
    init(&bar, blockDim.x);  // expected arrival count
}

// With completion function
auto on_completion = []() noexcept {
    // Called when all threads have arrived and the barrier completes
    // Executed by one unspecified thread
};
// cuda::barrier<cuda::thread_scope_block, decltype(on_completion)> bar_with_func;
// init(&bar_with_func, blockDim.x, on_completion);
```

**Barrier phases**: CUDA barriers support phase-based synchronization. After all threads arrive and the barrier completes, it automatically resets for the next phase. This means a barrier can be reused in a loop without reinitialization:

```cpp
__global__ void barrier_loop_example(float* data, int n, int iterations) {
    __shared__ cuda::barrier<cuda::thread_scope_block> bar;
    if (threadIdx.x == 0) init(&bar, blockDim.x);
    __syncthreads();

    for (int i = 0; i < iterations; i++) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            data[idx] += 1.0f;
        }
        bar.arrive_and_wait(); // re-usable across iterations
    }
}
```

### 20.7.4 `cuda::latch<Scope>`

A single-use countdown synchronization primitive. Threads can decrement the counter, and when it reaches zero, all waiting threads are unblocked. Unlike a barrier, `cuda::latch` is single-use and cannot be reset.

```cpp
#include <cuda/latch>

__global__ void latch_example(float* data, int n) {
    __shared__ cuda::latch<cuda::thread_scope_block> latch;

    // Initialize with expected count (e.g., number of producer threads)
    constexpr int NUM_PRODUCERS = 128;
    if (threadIdx.x == 0) {
        init(&latch, NUM_PRODUCERS);
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIdx.x < NUM_PRODUCERS) {
        // Producer: do work and count down
        if (idx < n) {
            data[idx] *= 2.0f;
        }
        latch.count_down(1);
    }

    // Consumer threads wait for all producers
    if (threadIdx.x >= NUM_PRODUCERS) {
        latch.wait();
        // All producers have finished -- safe to read data
    }
}
```

**Key methods**:
- `count_down(n)`: Decrement the internal counter by n. If counter reaches zero, waiting threads are unblocked.
- `wait()`: Block until the counter reaches zero.
- `try_wait()`: Non-blocking test; returns true if counter is zero.

**Differences from `cuda::barrier`**:
- `cuda::latch` is single-use; `cuda::barrier` is reusable across phases.
- `cuda::latch` allows a single thread to count down by more than 1; `cuda::barrier` typically expects one arrival per thread.
- `cuda::latch` allows some threads to only wait (not count down).

### 20.7.5 `cuda::std::` and `std::` Equivalence

Types from `cuda::std::` and `std::` behave identically to `cuda::` types when instantiated with `cuda::thread_scope_system`:

```cpp
// These have the same behavior (system scope):
cuda::atomic<int, cuda::thread_scope_system> a1;
cuda::std::atomic<int> a2;     // implicitly thread_scope_system
std::atomic<int> a3;           // same as cuda::std::atomic<int>

cuda::barrier<cuda::thread_scope_system> b1;
cuda::std::barrier<> b2;       // same behavior
std::barrier<> b3;             // same behavior

cuda::latch<cuda::thread_scope_system> l1;
cuda::std::latch<> l2;         // same behavior
std::latch<> l3;               // same behavior
```

This equivalence means you can use standard C++ synchronization primitives in device code, and they will operate at system scope by default. However, for finer-grained scope control (e.g., `thread_scope_block` for better performance), you must use the `cuda::` versions explicitly.

### 20.7.6 Memory Ordering

CUDA supports the same memory ordering constants as C++11:

```cpp
namespace cuda {
    enum memory_order {
        memory_order_relaxed,   // no synchronization, only atomicity
        memory_order_acquire,   // subsequent reads cannot be reordered before this
        memory_order_release,   // prior writes cannot be reordered after this
        memory_order_acq_rel,   // both acquire and release semantics
        memory_order_seq_cst    // total ordering across all seq_cst operations
    };
}
```

**Ordering guarantees**:

| Order | Guarantee | Use Case |
|---|---|---|
| `relaxed` | No ordering guarantee; only atomicity | Simple counters, statistics |
| `acquire` | Subsequent memory operations cannot be reordered before this load | Reading a flag to determine if data is ready |
| `release` | Prior memory operations cannot be reordered after this store | Writing data, then setting a flag |
| `acq_rel` | Both acquire and release (for read-modify-write) | Atomic update that both reads and writes |
| `seq_cst` | Total order across all `seq_cst` operations | When a global order of operations is needed |

### 20.7.7 Complete Synchronization Example

```cpp
#include <cuda/atomic>
#include <cuda/barrier>
#include <cuda/latch>

// Shared state
__device__ cuda::atomic<int, cuda::thread_scope_device> work_ready{0};
__device__ int work_data[1024];

__global__ void synchronization_pattern(int* output, int n) {
    // Block-level barrier for within-block sync
    __shared__ cuda::barrier<cuda::thread_scope_block> block_barrier;
    __shared__ int local_sum;

    if (threadIdx.x == 0) {
        init(&block_barrier, blockDim.x);
        local_sum = 0;
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // === Phase 1: Read data written by another block (device-scope sync) ===
    if (blockIdx.x > 0 && threadIdx.x == 0) {
        // Wait for block 0 to produce data using device-scope atomic
        while (work_ready.load(cuda::memory_order_acquire) == 0) {
            // spin-wait
        }
    }

    // === Phase 2: Process data with block-level synchronization ===
    int my_val = 0;
    if (idx < n) {
        my_val = work_data[idx % 1024] + idx;
    }

    // Reduce within block using barrier
    atomicAdd(&local_sum, my_val);
    block_barrier.arrive_and_wait();

    // Thread 0 of each block writes the block result
    if (threadIdx.x == 0) {
        output[blockIdx.x] = local_sum;

        // Block 0 signals that initial data is ready
        if (blockIdx.x == 0) {
            work_ready.store(1, cuda::memory_order_release);
        }
    }
}
```

### 20.7.8 Synchronization Primitive Summary

| Primitive | Header | Scope Support | Reusable | Key Use |
|---|---|---|---|---|
| `cuda::atomic<T, Scope>` | `<cuda/atomic>` | thread, block, device, system | N/A (always usable) | Thread-safe read-modify-write |
| `cuda::atomic_ref<T, Scope>` | `<cuda/atomic>` | thread, block, device, system | N/A | Atomic access to non-atomic memory |
| `cuda::barrier<Scope>` | `<cuda/barrier>` | block, device, system | Yes (phases) | N-thread rendezvous |
| `cuda::latch<Scope>` | `<cuda/latch>` | block, device, system | No (single-use) | Countdown synchronization |
| `__syncthreads()` | (built-in) | block | Yes | Simple block barrier |
| `__threadfence_block()` | (built-in) | block | N/A | Block-scope memory fence |
| `__threadfence()` | (built-in) | device | N/A | Device-scope memory fence |
| `__threadfence_system()` | (built-in) | system | N/A | System-scope memory fence |
| `cg::this_thread_block().sync()` | `<cooperative_groups.h>` | block | Yes | CG block barrier |
| `cg::this_grid().sync()` | `<cooperative_groups.h>` | grid | Yes | Grid-wide barrier (cooperative launch) |
| `cg::this_cluster().sync()` | `<cooperative_groups.h>` | cluster | Yes | Cluster barrier (CC 9.0+) |
| `__mbarrier_t` | `<cuda_awbarrier_primitives.h>` | block | Yes (phases) | Flexible arrival-counting barrier |
| `__pipeline_*` | `<cuda_pipeline.h>` | block | N/A | Async copy synchronization |
