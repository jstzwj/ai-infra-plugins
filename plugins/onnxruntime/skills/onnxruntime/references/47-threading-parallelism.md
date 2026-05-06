# ONNX Runtime Reference - Chapter 47: Threading and Parallelism

This chapter covers ONNX Runtime's threading model in detail, including thread pool types, configuration options, thread affinity, spin control, parallel execution strategies, and platform-specific behavior.

---

## 47.1 Threading Model Overview

### 47.1.1 Two-Level Parallelism

ONNX Runtime uses a two-level parallelism model:

```
┌─────────────────────────────────────────────────────────────┐
│                    Inter-Op Parallelism                      │
│    ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│    │ Node     │  │ Node     │  │ Node     │  ...            │
│    │ Group 0  │  │ Group 1  │  │ Group 2  │                │
│    │ (Thread  │  │ (Thread  │  │ (Thread  │                 │
│    │  Pool A) │  │  Pool B) │  │  Pool C) │                 │
│    │ ┌──────┐ │  │ ┌──────┐ │  │ ┌──────┐ │                │
│    │ │Conv0 │ │  │ │Conv1 │ │  │ │Relu  │ │                │
│    │ │Conv2 │ │  │ │Add3  │ │  │ │Conv5 │ │                │
│    │ │Relu4 │ │  │ │...   │ │  │ │...   │ │                │
│    │ └──────┘ │  │ └──────┘ │  │ └──────┘ │                │
│    └──────────┘  └──────────┘  └──────────┘               │
│                                                             │
│                  Intra-Op Parallelism                       │
│    Within each node, work is parallelized across threads     │
│    e.g., MatMul parallelized across rows/columns            │
└─────────────────────────────────────────────────────────────┘
```

- **Intra-op parallelism**: Parallelizes work *within* a single operator (e.g., parallelizing matrix multiplication across threads)
- **Inter-op parallelism**: Executes independent operators concurrently (when there are no data dependencies)

### 47.1.2 Thread Pool Hierarchy

```
OrtEnv
├── Global Thread Pools (optional)
│   ├── Intra-Op Thread Pool
│   └── Inter-Op Thread Pool
│
├── Session 1
│   ├── Per-Session Thread Pools (if not using global)
│   │   ├── Intra-Op Thread Pool
│   │   └── Inter-Op Thread Pool
│   └── Executors
│
└── Session 2
    ├── Per-Session Thread Pools
    └── Executors
```

---

## 47.2 Thread Pool Types

### 47.2.1 Intra-Op Thread Pool

The intra-op thread pool handles parallelism within individual operators. Each node that supports intra-op parallelism submits work to this pool.

```cpp
// onnxruntime/core/framework/parallel_executor.h
// Key characteristics:
// - Used for data-parallel operations within a single op
// - Fixed number of threads (configurable)
// - Eigen-based implementation
// - Supports work stealing
// - Thread-local task queues
```

### 47.2.2 Inter-Op Thread Pool

The inter-op thread pool handles concurrent execution of independent nodes in the graph.

```cpp
// Key characteristics:
// - Used for concurrent execution of independent ops
// - Only effective when graph has parallel branches
// - Sequential by default (inter_op_num_threads = 1)
// - Uses EigenNonBlockingThreadPool
```

### 47.2.3 ThreadPool Class

```cpp
// onnxruntime/core/platform/threadpool.h
class ThreadPool {
public:
    // Constructor
    ThreadPool(OrtEnv* env, const ThreadOptions& thread_options,
               const std::string& name, int thread_pool_size,
               bool allow_spinning);

    // Destructor
    ~ThreadPool();

    // Parallel loop: execute fn(i) for i in [0, total)
    void ParallelFor(int64_t total, const std::function<void(int64_t)>& fn);

    // Parallel loop with cost model
    void ParallelFor(int64_t total, double cost_per_unit,
                     const std::function<void(int64_t, int64_t)>& fn);

    // Parallel for with thread affinity
    void ParallelFor(int64_t total, double cost_per_unit, int64_t block_size,
                     const std::function<void(int64_t, int64_t)>& fn);

    // Simple parallel for (no cost model)
    void SimpleParallelFor(int64_t total,
                           const std::function<void(int64_t)>& fn);

    // Schedule a task
    void Schedule(std::function<void()> fn);

    // Get thread pool size
    int NumThreads() const;

    // Get current thread index (within the pool)
    int CurrentThreadId() const;

    // Steal work from other threads (work stealing)
    bool StealWork(int thief_index, std::function<void()>* stolen);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
```

---

## 47.3 EigenNonBlockingThreadPool Implementation

### 47.3.1 Overview

ONNX Runtime uses Eigen's `NonBlockingThreadPool` as the underlying thread pool implementation. This is a work-stealing thread pool that avoids blocking on condition variables.

```cpp
// Third-party/eigen/unsupported/Eigen/CXX11/src/ThreadPool/NonBlockingThreadPool.h
// Key features:
// 1. Non-blocking: Workers spin briefly before sleeping
// 2. Work stealing: Idle threads steal from busy threads
// 3. Per-thread queues: Each thread has its own task queue
// 4. Dynamic block sizing: Adjusts granularity based on workload
```

### 47.3.2 Thread Pool Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                EigenNonBlockingThreadPool                    │
│                                                             │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐       │
│  │ Thread 0│  │ Thread 1│  │ Thread 2│  │ Thread 3│  ...   │
│  │ ┌─────┐ │  │ ┌─────┐ │  │ ┌─────┐ │  │ ┌─────┐ │       │
│  │ │Queue│ │  │ │Queue│ │  │ │Queue│ │  │ │Queue│ │       │
│  │ │ ┌─┐ │ │  │ │ ┌─┐ │ │  │ │ ┌─┐ │ │  │ │ ┌─┐ │ │       │
│  │ │ │T│ │ │  │ │ │T│ │ │  │ │ │T│ │ │  │ │ │T│ │ │       │
│  │ │ │a│ │ │  │ │ │a│ │ │  │ │ │a│ │ │  │ │ │a│ │ │       │
│  │ │ │s│ │ │  │ │ │s│ │ │  │ │ │s│ │ │  │ │ │s│ │ │       │
│  │ │ │k│ │ │  │ │ │k│ │ │  │ │ │k│ │ │  │ │ │k│ │ │       │
│  │ │ └─┘ │ │  │ │ └─┘ │ │  │ │ └─┘ │ │  │ │ └─┘ │ │       │
│  │ └─────┘ │  │ └─────┘ │  │ └─────┘ │  │ └─────┘ │       │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘       │
│       │            │            │            │              │
│       └────────────┴────────────┴────────────┘              │
│                  Work Stealing Network                       │
│                                                             │
│  ┌──────────────────────────────────────────────────┐       │
│  │ Global Task Queue (overflow)                       │       │
│  └──────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

### 47.3.3 Thread Pool Lifecycle

```cpp
// Thread creation
class ThreadPool::Impl {
public:
    Impl(const ThreadOptions& options, const std::string& name,
         int num_threads, bool allow_spinning)
        : thread_options_(options),
          num_threads_(num_threads),
          allow_spinning_(allow_spinning) {

        // Create Eigen thread pool
        eigen_pool_ = std::make_unique<Eigen::NonBlockingThreadPool>(
            num_threads,
            allow_spinning,
            /*spin_duration=*/options.spin_duration_us);

        // Set thread affinity if specified
        if (options.affinity.size() > 0) {
            SetThreadAffinity(eigen_pool_.get(), options.affinity);
        }

        // Set thread names for debugging
        SetThreadNames(eigen_pool_.get(), name);
    }

    void ParallelFor(int64_t total, double cost_per_unit,
                     const std::function<void(int64_t, int64_t)>& fn) {
        if (total <= 0) return;
        if (total == 1) {
            fn(0, 1);
            return;
        }

        // Compute block size based on cost model
        int64_t block_size = ComputeBlockSize(total, cost_per_unit);

        // Compute number of blocks
        int64_t num_blocks = (total + block_size - 1) / block_size;

        if (num_blocks == 1) {
            // Single block: execute inline
            fn(0, total);
            return;
        }

        // Parallel execution
        std::atomic<int64_t> block_index{0};
        std::atomic<int> error_count{0};

        auto task = [&]() {
            while (true) {
                int64_t my_block = block_index.fetch_add(1);
                if (my_block >= num_blocks) break;

                int64_t start = my_block * block_size;
                int64_t end = std::min(start + block_size, total);

                try {
                    fn(start, end);
                } catch (...) {
                    error_count.fetch_add(1);
                }
            }
        };

        // Submit tasks to all threads
        for (int i = 0; i < num_threads_; ++i) {
            eigen_pool_->Schedule(task);
        }

        // Also execute on the main thread
        task();
    }

private:
    int64_t ComputeBlockSize(int64_t total, double cost_per_unit) {
        // Dynamic block sizing based on cost
        // High cost per unit → larger blocks (less overhead)
        // Low cost per unit → smaller blocks (better load balancing)

        double total_cost = total * cost_per_unit;

        // Minimum cost per block (in ns, empirically determined)
        constexpr double kMinCostPerBlock = 1000.0;  // 1us

        int64_t block_size = static_cast<int64_t>(
            kMinCostPerBlock / cost_per_unit);

        // Clamp block size
        block_size = std::max<int64_t>(1, block_size);
        block_size = std::min<int64_t>(total, block_size);

        return block_size;
    }

    ThreadOptions thread_options_;
    int num_threads_;
    bool allow_spinning_;
    std::unique_ptr<Eigen::NonBlockingThreadPool> eigen_pool_;
};
```

---

## 47.4 ThreadPool Configuration

### 47.4.1 Session-Level Configuration

```python
import onnxruntime as ort

options = ort.SessionOptions()

# Intra-op thread pool (default: number of CPU cores)
options.intra_op_num_threads = 8

# Inter-op thread pool (default: 0 = sequential execution)
options.inter_op_num_threads = 4

# Execution mode
options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL   # 0
# options.execution_mode = ort.ExecutionMode.ORT_PARALLEL    # 1

# Spin control
options.add_config_entry("session.allow_spinning", "1")     # Allow spinning
options.add_config_entry("session.spin_duration_us", "1000") # Spin for 1ms
options.add_config_entry("session.spin_backoff_max", "10000") # Max backoff 10ms

session = ort.InferenceSession("model.onnx", options)
```

### 47.4.2 Global Thread Pools

```python
import onnxruntime as ort

# Create global thread pools (shared across all sessions)
env = ort.Env(
    ort.LoggingLevel.ORT_LOGGING_LEVEL_WARNING,
    "app_name",
    global_thread_pool_options=ort.GlobalThreadPoolOptions(
        intra_op_num_threads=8,
        inter_op_num_threads=4,
        allow_spinning=True,
        spin_duration_us=1000,
    )
)

# All sessions created with this env share the thread pools
session1 = ort.InferenceSession("model1.onnx", ort.SessionOptions(), env)
session2 = ort.InferenceSession("model2.onnx", ort.SessionOptions(), env)
```

### 47.4.3 ThreadingOptions API (C++)

```cpp
// onnxruntime/core/session/onnxruntime_cxx_api.h
struct ThreadingOptions {
    // Thread counts
    int intra_op_thread_count = 0;   // 0 = use all cores
    int inter_op_thread_count = 0;   // 0 = sequential

    // Spin control
    bool allow_spinning = true;
    int spin_duration_us = 0;        // 0 = default (infinity)
    int spin_backoff_max = 0;        // 0 = default

    // Thread affinity
    std::vector<int> affinity;       // CPU core IDs for thread pinning

    // Thread naming
    std::string intra_op_thread_name = "ort-intra";
    std::string inter_op_thread_name = "ort-inter";

    // Stack size
    size_t thread_stack_size = 0;    // 0 = platform default
};

// Create environment with threading options
Ort::Env env(logging_level, "app", threading_options);

// Create session options with threading
Ort::SessionOptions options;
options.SetIntraOpNumThreads(8);
options.SetInterOpNumThreads(4);
options.SetExecutionMode(ORT_PARALLEL);
```

### 47.4.4 C API Threading Configuration

```c
// Intra-op threads
OrtStatus* OrtSessionOptionsSetIntraOpNumThreads(
    OrtSessionOptions* options, int intra_op_num_threads);

// Inter-op threads
OrtStatus* OrtSessionOptionsSetInterOpNumThreads(
    OrtSessionOptions* options, int inter_op_num_threads);

// Execution mode
OrtStatus* OrtSessionOptionsSetExecutionMode(
    OrtSessionOptions* options, ExecutionMode mode);

// Spin control
OrtStatus* OrtSessionOptionsAddConfigEntry(
    OrtSessionOptions* options, const char* key, const char* value);

// Global thread pools
OrtStatus* OrtEnvCreateWithGlobalThreadPools(
    OrtLoggingLevel logging_level, const char* logid,
    const OrtThreadingOptions* thread_options,
    OrtEnv** out);

// Per-session thread pools with custom options
OrtStatus* OrtSessionOptionsSetThreadAffinity(
    OrtSessionOptions* options, const char* affinity_string);
```

---

## 47.5 Thread Affinity Settings

### 47.5.1 Overview

Thread affinity binds threads to specific CPU cores, which can improve cache performance by reducing cache-line migration.

### 47.5.2 Setting Thread Affinity

```python
import onnxruntime as ort

options = ort.SessionOptions()

# Pin intra-op threads to cores 0-3
options.add_config_entry("session.intra_op_thread_affinity", "0;1;2;3")

# Pin inter-op threads to cores 4-5
options.add_config_entry("session.inter_op_thread_affinity", "4;5")

# Alternative: comma-separated
options.add_config_entry("session.intra_op_thread_affinity", "0,1,2,3")
```

### 47.5.3 Platform-Specific Affinity

```cpp
// Linux: Use pthread_setaffinity_np
void SetThreadAffinityLinux(std::thread& thread, const std::vector<int>& cores) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int core : cores) {
        CPU_SET(core, &cpuset);
    }
    int rc = pthread_setaffinity_np(thread.native_handle(),
                                     sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        LOGS_DEFAULT(WARNING) << "Failed to set thread affinity: " << strerror(rc);
    }
}

// Windows: Use SetThreadAffinityMask
void SetThreadAffinityWindows(std::thread& thread, const std::vector<int>& cores) {
    DWORD_PTR mask = 0;
    for (int core : cores) {
        mask |= (1ULL << core);
    }
    DWORD_PTR result = SetThreadAffinityMask(
        thread.native_handle(), mask);
    if (result == 0) {
        LOGS_DEFAULT(WARNING) << "Failed to set thread affinity: "
                              << GetLastError();
    }
}

// macOS: Thread affinity is advisory-only via thread_policy_set
void SetThreadAffinityMacOS(std::thread& thread, const std::vector<int>& cores) {
    // macOS uses affinity tags rather than core masks
    // Set thread affinity tag (same tag = same core group)
    thread_affinity_policy_data_t policy = {1};
    thread_policy_set(pthread_mach_thread_np(thread.native_handle()),
                      THREAD_AFFINITY_POLICY,
                      reinterpret_cast<thread_policy_t>(&policy),
                      1);
}
```

### 47.5.4 NUMA-Aware Affinity

```cpp
void SetNumaAwareAffinity(std::thread& thread, int numa_node) {
    // Get CPUs belonging to the NUMA node
    int num_cpus = numa_num_configured_cpus();
    struct bitmask* cpus = numa_allocate_cpumask();
    numa_node_to_cpus(numa_node, cpus);

    std::vector<int> cores;
    for (int i = 0; i < num_cpus; ++i) {
        if (numa_bitmask_isbitset(cpus, i)) {
            cores.push_back(i);
        }
    }
    numa_free_cpumask(cpus);

    SetThreadAffinityLinux(thread, cores);
}
```

---

## 47.6 Spin Control

### 47.6.1 Overview

Spin control determines how threads behave when there is no work available. The trade-off is between latency (spin = fast response) and CPU usage (spin = waste CPU cycles).

### 47.6.2 Spin Parameters

```python
import onnxruntime as ort

options = ort.SessionOptions()

# allow_spinning: Whether threads spin before sleeping
# Default: True (spinning enabled)
options.add_config_entry("session.allow_spinning", "1")

# spin_duration_us: How long to spin before sleeping (microseconds)
# Default: 0 (spin until work arrives or a long time passes)
# A value of 0 means "spin indefinitely" (actually limited by Eigen internally)
options.add_config_entry("session.spin_duration_us", "10000")  # 10ms

# spin_backoff_max: Maximum backoff duration during spinning (microseconds)
# Default: 0 (use Eigen default)
options.add_config_entry("session.spin_backoff_max", "100000")  # 100ms
```

### 47.6.3 Spin Behavior

```
Thread is idle:
    │
    ├── Phase 1: Spin (busy-wait)
    │   └── Check for work in a tight loop
    │       ├── spin_duration_us = 0: spin indefinitely
    │       └── spin_duration_us > 0: spin for specified duration
    │
    ├── Phase 2: Relaxed spin (yield-based)
    │   └── Periodically yield to OS scheduler
    │       └── spin_backoff_max controls maximum yield interval
    │
    └── Phase 3: Sleep
        └── Block on condition variable (zero CPU usage)
        └── Woken up when new work arrives
```

### 47.6.4 Spin Configuration Recommendations

| Workload | allow_spinning | spin_duration_us | Rationale |
|----------|---------------|-----------------|-----------|
| Low-latency inference | True | 0 (infinite) | Minimize latency |
| Batch inference | True | 10000 (10ms) | Balance latency and CPU |
| Long-running server | True | 1000 (1ms) | Moderate spinning |
| Multi-tenant server | False | N/A | Minimize CPU waste |
| Edge device | False | N/A | Save power |

---

## 47.7 Dynamic Block Sizing

### 47.7.1 Overview

Dynamic block sizing automatically adjusts the granularity of parallel work to balance overhead vs. load balancing.

### 47.7.2 Block Size Calculation

```cpp
// onnxruntime/core/util/math.h
// MLAS uses dynamic block sizing for GEMM operations

struct PartitionWorkParams {
    int64_t total_work;       // Total items to process
    double cost_per_item;     // Estimated cost per item (ns)
    int num_threads;          // Available threads
    int64_t min_block_size;   // Minimum block size
};

int64_t ComputeDynamicBlockSize(const PartitionWorkParams& params) {
    // Goal: each block should take at least ~10us to amortize scheduling overhead
    constexpr double kMinCostPerBlock = 10000.0;  // 10us in ns

    int64_t block_size = static_cast<int64_t>(
        kMinCostPerBlock / params.cost_per_item);

    // Ensure at least 2 blocks per thread for load balancing
    int64_t min_for_balancing = params.total_work /
                                (params.num_threads * 2);
    block_size = std::min(block_size, min_for_balancing);

    // Clamp to valid range
    block_size = std::max(block_size, params.min_block_size);
    block_size = std::min(block_size, params.total_work);

    return block_size;
}
```

### 47.7.3 Examples of Dynamic Block Sizing

```cpp
// MatMul: MxK * KxN
// Parallelize over M dimension
void ParallelMatMul(const float* A, const float* B, float* C,
                    int M, int K, int N, ThreadPool* tp) {
    double cost_per_row = 2.0 * K * N;  // FLOPs per row of C

    tp->ParallelFor(M, cost_per_row,
        [A, B, C, K, N](int64_t start, int64_t end) {
            for (int64_t i = start; i < end; ++i) {
                // Compute row i of C = A[i,:] * B
                const float* a_row = A + i * K;
                float* c_row = C + i * N;
                GemmRow(a_row, B, c_row, K, N);
            }
        });
}

// Element-wise operations (low cost per element)
void ParallelAdd(const float* a, const float* b, float* c,
                 int64_t count, ThreadPool* tp) {
    double cost_per_element = 1.0;  // 1 FLOP per element
    // Dynamic block sizing will create large blocks to amortize overhead

    tp->ParallelFor(count, cost_per_element,
        [a, b, c](int64_t start, int64_t end) {
            for (int64_t i = start; i < end; ++i) {
                c[i] = a[i] + b[i];
            }
        });
}

// Reduction (tree reduction pattern)
float ParallelReduce(const float* data, int64_t count, ThreadPool* tp) {
    // First phase: partial reductions in parallel
    int num_threads = tp->NumThreads();
    std::vector<float> partial_sums(num_threads, 0.0f);

    tp->ParallelFor(count, 2.0,  // Cost: load + add
        [&partial_sums, data](int64_t start, int64_t end) {
            int tid = /* get thread id */;
            float sum = 0.0f;
            for (int64_t i = start; i < end; ++i) {
                sum += data[i];
            }
            partial_sums[tid] += sum;
        });

    // Second phase: reduce partial sums (sequential for small num_threads)
    float total = 0.0f;
    for (auto s : partial_sums) total += s;
    return total;
}
```

---

## 47.8 Sequential vs Parallel Execution Modes

### 47.8.1 Sequential Mode (ORT_SEQUENTIAL)

```
Execution order (topological):

Node 0 ──→ Node 1 ──→ Node 2 ──→ Node 3 ──→ Node 4 ──→ Node 5
                                                      ↓
                                                       Output

- Single thread executes nodes one at a time
- Intra-op parallelism still applies (within each node)
- Default mode
- Lower overhead, deterministic execution
```

### 47.8.2 Parallel Mode (ORT_PARALLEL)

```
Execution order (when independent):

     Node 0 ──→ Node 1 ──┐
                         ├──→ Node 3 ──→ Output
     Node 2 ─────────────┘

- Independent nodes run concurrently on separate threads
- Requires inter_op_num_threads > 1
- Higher overhead but can improve throughput
- Only beneficial when graph has parallel branches
```

### 47.8.3 Execution Mode Configuration

```python
import onnxruntime as ort

# Sequential mode (default)
options = ort.SessionOptions()
options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
options.intra_op_num_threads = 8  # Still parallelizes within ops

# Parallel mode
options = ort.SessionOptions()
options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
options.intra_op_num_threads = 4
options.inter_op_num_threads = 4  # Required for parallel mode

session = ort.InferenceSession("model.onnx", options)
```

### 47.8.4 When to Use Parallel Mode

| Scenario | Recommended Mode | Rationale |
|----------|-----------------|-----------|
| Single branch model | Sequential | No parallelism to exploit |
| Multi-branch model (e.g., encoder-decoder) | Parallel | Independent branches |
| CPU inference | Sequential | Thread overhead may exceed benefit |
| GPU inference | Sequential | GPU handles parallelism internally |
| Batch processing | Sequential + intra-op | Intra-op parallelism is more effective |
| Multi-model server | Parallel + inter-op | Concurrent model execution |

---

## 47.9 OpenMP Integration

### 47.9.1 Overview

ONNX Runtime can optionally use OpenMP for parallelism in certain operations, particularly MLAS (Math Linear Algebra Subprograms) operations.

### 47.9.2 Enabling OpenMP

```python
import onnxruntime as ort

options = ort.SessionOptions()

# Enable OpenMP (if ORT was built with OpenMP support)
options.add_config_entry("session.use_openmp", "1")

# Set OpenMP thread count
options.add_config_entry("session.openmp_num_threads", "8")

# Set OpenMP thread affinity
options.add_config_entry("session.openmp_thread_affinity", "0,1,2,3,4,5,6,7")
```

### 47.9.3 OpenMP vs Thread Pool

```cpp
// MLAS can use either OpenMP or ORT's thread pool
// The choice depends on build configuration

#ifdef USE_OPENMP
    // Use OpenMP for parallel GEMM
    #pragma omp parallel for schedule(dynamic)
    for (int row = 0; row < M; ++row) {
        MlasGemmRow(A, B, C, row, K, N);
    }
#else
    // Use ORT thread pool
    thread_pool->ParallelFor(M, cost_per_row,
        [A, B, C, K, N](int64_t start, int64_t end) {
            for (int64_t row = start; row < end; ++row) {
                MlasGemmRow(A, B, C, row, K, N);
            }
        });
#endif
```

### 47.9.4 OpenMP Configuration Trade-offs

| Aspect | ORT Thread Pool | OpenMP |
|--------|----------------|--------|
| Overhead | Low (custom queues) | Medium (OpenMP runtime) |
| Integration | Native | External dependency |
| Work stealing | Yes | Varies by implementation |
| Thread reuse | Persistent | May create/destroy threads |
| Nested parallelism | Supported | Requires OMP_NESTED=TRUE |
| Configuration | OrtSessionOptions | Environment variables |

---

## 47.10 Global vs Per-Session Thread Pools

### 47.10.1 Per-Session Thread Pools (Default)

```
┌──────────┐     ┌──────────┐
│ Session1 │     │ Session2 │
│ ┌──────┐ │     │ ┌──────┐ │
│ │ 8    │ │     │ │ 8    │ │
│ │threads│ │     │ │threads│ │
│ └──────┘ │     │ └──────┘ │
└──────────┘     └──────────┘

Total threads: 16 (8 per session)
```

### 47.10.2 Global Thread Pools

```
┌──────────┐     ┌──────────┐
│ Session1 │     │ Session2 │
│    │     │     │    │     │
│    └─────┼─────┘    │     │
│          │          │     │
│    ┌─────┴──────────┘     │
│    │  8 shared threads    │
│    └─────────────────────┘
└───────────────────────────────┘

Total threads: 8 (shared)
```

### 47.10.3 When to Use Global Thread Pools

- **Multi-session deployment**: When running many models simultaneously
- **Memory-constrained**: Fewer thread stacks = less memory
- **CPU-constrained**: Avoids oversubscription from multiple pools
- **Server scenarios**: Sharing threads across sessions improves throughput

### 47.10.4 Configuration

```python
# Global thread pools
import onnxruntime as ort

# Create environment with global thread pools
thread_pool_options = ort.GlobalThreadPoolOptions(
    intra_op_num_threads=8,
    inter_op_num_threads=4,
)

env = ort.Env(
    ort.LoggingLevel.ORT_LOGGING_LEVEL_WARNING,
    "my_app",
    thread_pool_options
)

# All sessions share these thread pools
session1 = ort.InferenceSession("model1.onnx", ort.SessionOptions(), env)
session2 = ort.InferenceSession("model2.onnx", ort.SessionOptions(), env)
```

```c
// C API for global thread pools
OrtThreadingOptions* threading_options;
OrtCreateThreadingOptions(&threading_options);
OrtSetGlobalIntraOpNumThreads(threading_options, 8);
OrtSetGlobalInterOpNumThreads(threading_options, 4);
OrtSetGlobalSpinControl(threading_options, 1);  // Allow spinning
OrtSetGlobalSpinDuration(threading_options, 10000);  // 10ms

OrtEnv* env;
OrtEnvCreateWithGlobalThreadPools(
    ORT_LOGGING_LEVEL_WARNING, "my_app",
    threading_options, &env);

OrtReleaseThreadingOptions(threading_options);
```

---

## 47.11 Platform-Specific Threading

### 47.11.1 Windows

```cpp
// Windows thread creation and management
class WindowsThread : public IThread {
public:
    WindowsThread(const ThreadOptions& options, const std::string& name,
                  std::function<void()> fn)
        : fn_(std::move(fn)) {
        thread_ = std::thread([this]() {
            // Set thread name (for debugging)
            SetThreadDescription(GetCurrentThread(),
                               std::wstring(name.begin(), name.end()).c_str());

            // Set thread priority
            if (options.priority == ThreadPriority::HIGH) {
                SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
            }

            // Set thread affinity
            if (!options.affinity.empty()) {
                SetThreadAffinityWindows(thread_, options.affinity);
            }

            // Set stack size
            // (configured at thread creation, not changeable afterward)

            // Run the thread function
            fn_();
        });
    }

    ~WindowsThread() {
        if (thread_.joinable()) {
            thread_.join();
        }
    }

private:
    std::thread thread_;
    std::function<void()> fn_;
};
```

### 47.11.2 Linux

```cpp
// Linux thread creation and management
class LinuxThread : public IThread {
public:
    LinuxThread(const ThreadOptions& options, const std::string& name,
                std::function<void()> fn)
        : fn_(std::move(fn)) {
        pthread_attr_t attr;
        pthread_attr_init(&attr);

        // Set stack size
        if (options.stack_size > 0) {
            pthread_attr_setstacksize(&attr, options.stack_size);
        }

        thread_ = std::thread([this, options, name]() {
            // Set thread name (limited to 16 chars on Linux)
            std::string truncated_name = name.substr(0, 15);
            pthread_setname_np(pthread_self(), truncated_name.c_str());

            // Set thread affinity
            if (!options.affinity.empty()) {
                SetThreadAffinityLinux(/* get current thread */, options.affinity);
            }

            // Set scheduling policy
            if (options.realtime) {
                struct sched_param param;
                param.sched_priority = options.priority;
                pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
            }

            fn_();
        });

        pthread_attr_destroy(&attr);
    }

private:
    std::thread thread_;
    std::function<void()> fn_;
};
```

### 47.11.3 macOS

```cpp
// macOS thread creation and management
class MacOSThread : public IThread {
public:
    MacOSThread(const ThreadOptions& options, const std::string& name,
                std::function<void()> fn)
        : fn_(std::move(fn)) {
        thread_ = std::thread([this, options, name]() {
            // Set thread name
            pthread_setname_np(name.c_str());

            // macOS uses QoS classes instead of priorities
            // Map ORT priority to macOS QoS
            if (options.priority == ThreadPriority::HIGH) {
                pthread_set_qos_class_self_np(
                    QOS_CLASS_USER_INTERACTIVE, 0);
            } else {
                pthread_set_qos_class_self_np(
                    QOS_CLASS_DEFAULT, 0);
            }

            // Advisory affinity
            if (!options.affinity.empty()) {
                SetThreadAffinityMacOS(/* current thread */, options.affinity);
            }

            fn_();
        });
    }

private:
    std::thread thread_;
    std::function<void()> fn_;
};
```

---

## 47.12 Work Stealing

### 47.12.1 Overview

Work stealing allows idle threads to take work from busy threads' queues, improving load balancing.

```
Initial work distribution:
Thread 0: [Task A, Task B, Task C, Task D]  (4 tasks)
Thread 1: [Task E]                           (1 task)
Thread 2: [idle]                             (0 tasks)
Thread 3: [idle]                             (0 tasks)

After work stealing:
Thread 0: [Task A]          (1 task - working on it)
Thread 1: [Task E]          (1 task - working on it)
Thread 2: [Task B, Task C]  (2 tasks - stole from Thread 0)
Thread 3: [Task D]          (1 task - stole from Thread 0)
```

### 47.12.2 Work Stealing in Eigen

```cpp
// Eigen's NonBlockingThreadPool implements work stealing:
//
// 1. Each thread has a local run queue
// 2. New tasks go to the submitting thread's queue
// 3. When a thread's queue is empty, it:
//    a. Checks the global queue
//    b. Steals from other threads' queues
// 4. Stolen tasks are executed by the stealing thread
// 5. This provides automatic load balancing without central coordination
```

### 47.12.3 Enabling Work Stealing

```python
# Work stealing is always enabled in Eigen's NonBlockingThreadPool
# It cannot be disabled; it's fundamental to the implementation

# However, you can influence its effectiveness:
options = ort.SessionOptions()
# More threads = more stealing opportunities
options.intra_op_num_threads = 8
# Smaller blocks = more tasks = more stealing opportunities
# (dynamic block sizing handles this automatically)
```

---

## 47.13 Thread Safety Guarantees for Sessions

### 47.13.1 Session Thread Safety

```cpp
// Thread safety guarantees:
//
// 1. Ort::Session::Run() is thread-safe for concurrent calls
//    - Multiple threads can call Run() on the same session
//    - Each Run() call gets its own execution context
//
// 2. Ort::Session creation is NOT thread-safe
//    - Only one thread should create a session at a time
//
// 3. Ort::Env is thread-safe
//    - Multiple sessions can be created from the same env
//
// 4. Ort::SessionOptions is NOT thread-safe
//    - Configure options before creating sessions
//
// 5. Ort::Value (Tensor) is NOT thread-safe
//    - Each thread should have its own input/output tensors
```

### 47.13.2 Concurrent Inference Pattern

```cpp
// Thread-safe concurrent inference
void ConcurrentInference(Ort::Session& session,
                          const std::vector<InputBatch>& batches) {
    std::vector<std::thread> threads;
    std::atomic<int> error_count{0};

    for (const auto& batch : batches) {
        threads.emplace_back([&session, &batch, &error_count]() {
            try {
                // Each thread creates its own input/output tensors
                auto input_tensor = Ort::Value::CreateTensor<float>(
                    session.GetMemoryInfo(),
                    batch.data.data(), batch.data.size(),
                    batch.shape.data(), batch.shape.size());

                // Thread-safe Run() call
                auto output = session.Run(
                    Ort::RunOptions{},
                    session.GetInputNames().data(),
                    &input_tensor, 1,
                    session.GetOutputNames().data(),
                    session.GetOutputNames().size());

                // Process output...
            } catch (const Ort::Exception& e) {
                error_count.fetch_add(1);
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }
}
```

### 47.13.3 Thread Safety Summary

| Component | Thread-Safe? | Notes |
|-----------|-------------|-------|
| `Ort::Env` | Yes | Shared across sessions |
| `Ort::Session` (creation) | No | Create in single thread |
| `Ort::Session::Run()` | Yes | Multiple concurrent calls |
| `Ort::SessionOptions` | No | Configure before use |
| `Ort::Value` (Tensor) | No | Per-thread instances |
| `Ort::RunOptions` | Yes (const) | Create per Run() call |
| `Ort::Allocator` | Yes | Thread-safe by design |
| `Ort::ArenaCfg` | No | Configure before use |

---

## 47.14 Summary

| Topic | Key Points |
|-------|-----------|
| Two-Level Parallelism | Intra-op (within operator) and inter-op (between operators) |
| Thread Pool Types | EigenNonBlockingThreadPool with work stealing |
| Intra-Op Threads | Default: all CPU cores; parallelizes within individual operators |
| Inter-Op Threads | Default: 0 (sequential); parallelizes across independent operators |
| Thread Affinity | Bind threads to CPU cores for cache performance |
| Spin Control | Balance latency vs. CPU waste with spinning parameters |
| Dynamic Block Sizing | Automatic work granularity adjustment based on cost model |
| Sequential Mode | Default; nodes execute one at a time with intra-op parallelism |
| Parallel Mode | Independent nodes execute concurrently; needs inter-op threads |
| OpenMP | Optional alternative to ORT thread pool for MLAS operations |
| Global Thread Pools | Share threads across sessions; reduces total thread count |
| Work Stealing | Idle threads steal from busy threads for load balancing |
| Thread Safety | `Session::Run()` is thread-safe; creation is not |
