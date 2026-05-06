# 21. CUDA Environment Variables

CUDA exposes a wide range of environment variables that control device enumeration, JIT compilation, kernel execution, module loading, error logging, and other runtime behavior. These variables allow developers and system administrators to tune CUDA applications without recompilation, and are indispensable for debugging, deployment, and performance optimization.

---

## Table of Contents

1. [Device Enumeration](#211-device-enumeration)
2. [JIT Compilation](#212-jit-compilation)
3. [Execution](#213-execution)
4. [Module Loading](#214-module-loading)
5. [Error Log](#215-error-log)
6. [Other Variables](#216-other-variables)
7. [Quick Reference Table](#217-quick-reference-table)

---

## 21.1 Device Enumeration

These environment variables control which GPUs are visible to a CUDA application and the order in which they are enumerated.

### 21.1.1 CUDA_VISIBLE_DEVICES

Controls which GPUs are visible to CUDA applications and the order in which they appear. When set, only the listed devices are exposed; all others are hidden. Device indices in the rest of the application are renumbered starting from 0 in the order listed.

**Format:** `CUDA_VISIBLE_DEVICES=<device_list>`

The device list can be specified in several ways:

| Format | Example | Description |
|--------|---------|-------------|
| Comma-separated indices | `0,2,3` | Only devices 0, 2, and 3 are visible; renumbered as 0, 1, 2 |
| GPU UUIDs | `GPU-<UUID>` | Select devices by unique identifier |
| MIG device IDs | `MIG-<GPU UUID>/<GI ID>/<CI ID>` | Select specific MIG compute instances |

```bash
# Only use GPU 0 and GPU 1
export CUDA_VISIBLE_DEVICES=0,1

# Use GPUs in reverse order (GPU 3 becomes device 0, GPU 1 becomes device 1)
export CUDA_VISIBLE_DEVICES=3,1

# Select by UUID (useful in multi-node environments)
export CUDA_VISIBLE_DEVICES=GPU-3a7c1b9e-5d2f-4f8a-9c1d-6e3b2a1f0d9e

# Select a MIG compute instance
export CUDA_VISIBLE_DEVICES=MIG-3a7c1b9e-5d2f-4f8a-9c1d-6e3b2a1f0d9e/1/2
```

**Behavior details:**

- If the variable is set to an empty string or an invalid value, no devices are visible and `cudaGetDeviceCount()` returns 0.
- If the variable is not set, all available GPUs are visible.
- The renumbering affects all CUDA Runtime API and Driver API calls. For example, with `CUDA_VISIBLE_DEVICES=2,0`, the application sees GPU 2 as device 0 and GPU 0 as device 1.
- UUIDs can be obtained via `nvidia-smi -L` or `cudaGetDeviceProperties()` (the `uuid` field).
- When using MIG (Multi-Instance GPU), each MIG compute instance appears as a separate CUDA device.

```cpp
// Example: Verify which device the application sees
int deviceCount;
cudaGetDeviceCount(&deviceCount);
printf("Visible devices: %d\n", deviceCount);

for (int i = 0; i < deviceCount; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("  Device %d: %s (UUID: %s)\n", i, prop.name, prop.uuid);
}
```

### 21.1.2 CUDA_DEVICE_ORDER

Controls the order in which CUDA devices are enumerated by the runtime.

**Accepted values:**

| Value | Description |
|-------|-------------|
| `FASTEST_FIRST` | Devices are sorted by estimated compute power (default in older CUDA versions). This is not recommended for production use because the ordering can be inconsistent across runs. |
| `PCI_BUS_ID` | Devices are sorted by PCI bus ID in ascending order. This provides a deterministic, stable ordering that matches the output of `nvidia-smi`. |

```bash
# Deterministic ordering by PCI bus ID (recommended)
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Sort by estimated performance
export CUDA_DEVICE_ORDER=FASTEST_FIRST
```

**Recommendation:** Always use `PCI_BUS_ID` in production and multi-GPU setups to ensure consistent device ordering across application runs and machines. When `CUDA_DEVICE_ORDER=PCI_BUS_ID` is combined with `CUDA_VISIBLE_DEVICES`, the PCI ordering is applied first, then the filtering/reordering from `CUDA_VISIBLE_DEVICES` takes effect.

### 21.1.3 CUDA_MANAGED_FORCE_DEVICE_ALLOC

Forces all managed memory allocations (`cudaMallocManaged`) to be backed by device memory rather than allowing the driver to choose between host and device memory.

**Accepted values:** `1` (enable), `0` or unset (default behavior)

```bash
export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1
```

When enabled:
- All `cudaMallocManaged()` allocations are allocated in device memory.
- This can be useful for benchmarking or for ensuring predictable memory behavior on systems with non-uniform memory access.
- On systems without dedicated GPU memory (e.g., integrated GPUs), this variable may have no effect or cause allocations to fail.

---

## 21.2 JIT Compilation

CUDA kernels can be compiled to PTX (an intermediate representation) and then JIT-compiled to native GPU machine code (SASS) at runtime. These environment variables control the JIT compilation cache and behavior.

### 21.2.1 CUDA_CACHE_DISABLE

Disables the PTX JIT compilation cache. When disabled, every PTX-to-SASS compilation is performed from scratch, even for kernels that have been previously compiled in the same or a previous run.

**Accepted values:** `1` (disable caching), `0` or unset (caching enabled, default)

```bash
export CUDA_CACHE_DISABLE=1
```

Use cases:
- Debugging JIT compilation issues.
- Ensuring a clean compilation state.
- Benchmarking cold-start JIT compilation performance.

When caching is enabled (the default), the driver caches the compiled SASS code on disk so subsequent runs do not need to recompile the same PTX.

### 21.2.2 CUDA_CACHE_PATH

Specifies the filesystem path where the JIT compilation cache is stored.

**Default path:** `~/.nv/ComputeCache` on Linux and macOS, `%APPDATA%\NVIDIA\ComputeCache` on Windows.

```bash
# Custom cache location
export CUDA_CACHE_PATH=/mnt/ssd/cuda_cache

# Shared cache across users (ensure proper permissions)
export CUDA_CACHE_PATH=/shared/cuda_cache
```

Behavior notes:
- The directory is created automatically if it does not exist.
- If the specified path is not writable, caching is effectively disabled.
- A shared cache directory can speed up startup across multiple users or containers on the same machine.

### 21.2.3 CUDA_CACHE_MAXSIZE

Sets the maximum size of the JIT compilation cache in bytes. When the cache exceeds this limit, the driver evicts the least recently used entries.

**Default:** 1 GiB (1073741824 bytes)
**Maximum:** 4 GiB (4294967296 bytes)

```bash
# Set cache to 2 GiB
export CUDA_CACHE_MAXSIZE=2147483648

# Set cache to 512 MiB
export CUDA_CACHE_MAXSIZE=536870912
```

Behavior notes:
- The value must be specified in bytes (no suffixes like MB or GB).
- Values exceeding 4 GiB are clamped to 4 GiB.
- If the value is set to 0, caching is effectively disabled.

### 21.2.4 CUDA_FORCE_PTX_JIT

Forces the CUDA driver to JIT-compile embedded PTX code to native SASS even when a compatible cubin (pre-compiled binary) is available in the fat binary.

**Accepted values:** `1` (force PTX JIT), `0` or unset (default)

```bash
export CUDA_FORCE_PTX_JIT=1
```

When enabled:
- The driver ignores pre-compiled cubin sections in the fat binary and always JIT-compiles from PTX.
- This ensures the most optimized SASS for the specific GPU architecture, since PTX can be compiled to the exact target architecture while embedded cubins may target a lower architecture.
- This can increase startup time because all kernels must be JIT-compiled.
- This variable overrides `CUDA_FORCE_JIT=1` (if both are set, `CUDA_FORCE_PTX_JIT` takes precedence).

### 21.2.5 CUDA_FORCE_JIT

Forces the CUDA driver to JIT-compile from the highest available code version in the fat binary (which could be PTX or cubin). Unlike `CUDA_FORCE_PTX_JIT`, this does not specifically require PTX to be present.

**Accepted values:** `1` (force JIT), `0` or unset (default)

```bash
export CUDA_FORCE_JIT=1
```

- This variable is overridden by `CUDA_FORCE_PTX_JIT`. If both are set, only the PTX JIT behavior applies.
- Useful for ensuring that kernels are compiled for the current GPU rather than using an older embedded cubin.

### 21.2.6 CUDA_DISABLE_PTX_JIT

Disables PTX JIT compilation entirely. The driver will not attempt to JIT-compile PTX code and will only use pre-compiled cubins embedded in the fat binary.

**Accepted values:** `1` (disable PTX JIT), `0` or unset (default)

```bash
export CUDA_DISABLE_PTX_JIT=1
```

When enabled:
- If no compatible cubin is available for the target architecture, the kernel launch will fail.
- This can be useful for testing whether an application's fat binary contains the correct cubin for the target GPU.

### 21.2.7 CUDA_DISABLE_JIT

Disables all JIT compilation, including both PTX and cubin JIT. The driver will only use pre-compiled native code.

**Accepted values:** `1` (disable all JIT), `0` or unset (default)

```bash
export CUDA_DISABLE_JIT=1
```

When enabled:
- Only pre-compiled cubins that exactly match the target architecture can be used.
- If no exact cubin match is found, kernel launches will fail.

### 21.2.8 CUDA_FORCE_PRELOAD_LIBRARIES

Forces the CUDA runtime to preload NVVM and JIT-related libraries at application startup rather than loading them lazily on first use.

**Accepted values:** `1` (preload), `0` or unset (default)

```bash
export CUDA_FORCE_PRELOAD_LIBRARIES=1
```

When enabled:
- NVVM and JIT libraries are loaded during CUDA initialization rather than when the first JIT compilation is needed.
- This avoids latency spikes during the first kernel launch that requires JIT compilation.
- Particularly useful for latency-sensitive applications where predictable startup behavior is required.
- May slightly increase initial CUDA initialization time.

---

## 21.3 Execution

These environment variables control how CUDA dispatches and executes kernels and memory operations on the GPU.

### 21.3.1 CUDA_LAUNCH_BLOCKING

Forces all kernel launches to be synchronous. When enabled, every kernel launch blocks the host thread until the kernel completes on the device, effectively making all launches behave as if followed by an implicit `cudaDeviceSynchronize()`.

**Accepted values:** `1` (synchronous launches), `0` or unset (default, asynchronous launches)

```bash
export CUDA_LAUNCH_BLOCKING=1
```

This is the single most important debugging variable for CUDA:

- **Error localization:** Because CUDA errors are reported asynchronously by default, an error returned by one API call may actually have been caused by a much earlier kernel launch. With blocking launches, errors are reported immediately.
- **Race condition elimination:** Ensures kernels execute sequentially, making it impossible for race conditions between kernels to manifest.
- **Profiler correlation:** Makes it easier to correlate host-side API calls with device-side activity in profiling tools.

**Performance warning:** Do NOT use this in production. Synchronous launches severely degrade performance because the host CPU is idle while waiting for each kernel to finish, preventing overlap of computation and data transfers.

```cpp
// Example: Conditional blocking for debug builds
#ifdef DEBUG
    setenv("CUDA_LAUNCH_BLOCKING", "1", 1);
#endif
```

### 21.3.2 CUDA_DEVICE_MAX_CONNECTIONS

Sets the maximum number of concurrent connections (work queues) between the host and each device. Each connection represents an independent hardware queue that can carry a pipeline of work (kernel launches, memory copies, etc.).

**Accepted values:** `1` to `32` (default: `8`)

```bash
# Allow up to 16 concurrent work queues
export CUDA_DEVICE_MAX_CONNECTIONS=16

# Restrict to a single connection (serializes all work)
export CUDA_DEVICE_MAX_CONNECTIONS=1
```

Behavior notes:
- A higher number of connections allows more independent streams to execute concurrently.
- On GPUs with compute capability 3.5 and above, the default is 8 connections (sometimes reported as 16 on newer architectures).
- The actual number of hardware queues is limited by the GPU. Setting a value higher than the hardware limit has no additional effect.
- Increasing this value is useful for applications that use many streams and need to overlap many independent operations.
- Each connection consumes some driver and hardware resources. In rare cases, reducing the number can improve performance by reducing overhead.

```cpp
// Example: Check the effective number of connections
// Note: There is no direct API query, but you can observe the effect
// by launching many concurrent kernels in separate streams.
```

### 21.3.3 CUDA_DEVICE_MAX_COPY_CONNECTIONS

Sets the maximum number of connections dedicated to copy (memory transfer) operations. This is separate from the general connections controlled by `CUDA_DEVICE_MAX_CONNECTIONS`.

**Accepted values:** A non-negative integer (default varies by hardware)

```bash
# Allow up to 4 dedicated copy connections
export CUDA_DEVICE_MAX_COPY_CONNECTIONS=4
```

Behavior notes:
- Available on GPUs with compute capability 8.0 and above (Ampere and later).
- Dedicated copy engines can overlap memory transfers with computation without consuming general-purpose connections.
- Setting this to a higher value can improve the overlap of H2D and D2H transfers with kernel execution.

### 21.3.4 CUDA_SCALE_LAUNCH_QUEUES

Scales the number of launch queues (connections) relative to the default. This is an alternative to setting an absolute number via `CUDA_DEVICE_MAX_CONNECTIONS`.

**Accepted values:** `0.25`, `0.5`, `2`, `4` (multiplicative factors)

```bash
# Double the number of launch queues
export CUDA_SCALE_LAUNCH_QUEUES=2

# Halve the number of launch queues
export CUDA_SCALE_LAUNCH_QUEUES=0.5

# Quarter the number of launch queues
export CUDA_SCALE_LAUNCH_QUEUES=0.25

# Quadruple the number of launch queues
export CUDA_SCALE_LAUNCH_QUEUES=4
```

Behavior notes:
- The resulting number of queues is the default count multiplied by the specified factor, clamped to hardware limits.
- A factor of 0.5 or 0.25 can reduce resource overhead for applications that do not benefit from many concurrent queues.
- A factor of 2 or 4 can improve concurrency for applications with many independent streams.
- If `CUDA_DEVICE_MAX_CONNECTIONS` is also set, the explicit value takes precedence over scaling.

### 21.3.5 CUDA_GRAPHS_USE_NODE_PRIORITY

Enables per-node launch priorities in CUDA Graphs. When enabled, the relative priorities of graph nodes are respected during execution, allowing higher-priority nodes to preempt lower-priority ones.

**Accepted values:** `0` (disabled, default) or `1` (enabled)

```bash
export CUDA_GRAPHS_USE_NODE_PRIORITY=1
```

When enabled:
- Nodes in a CUDA graph can be assigned different launch priorities via `cudaGraphNodeSetParams()` or the graph node attributes.
- This allows fine-grained control over execution order within a graph beyond the strict topological ordering.
- Useful for real-time or latency-sensitive workloads where certain graph nodes should be prioritized over others.

### 21.3.6 CUDA_DEVICE_WAITS_ON_EXCEPTION

Causes the CUDA device to halt and the host to wait (spin) when a device-side exception occurs (e.g., an illegal memory access, out-of-bounds access, or other runtime error detected by the GPU).

**Accepted values:** `1` (halt on exception), `0` or unset (default)

```bash
export CUDA_DEVICE_WAITS_ON_EXCEPTION=1
```

When enabled:
- The device enters a blocking state upon encountering an exception.
- This is primarily intended for use with CUDA-GDB or other debugging tools.
- The debugger can then inspect the device state at the point of failure, including register values, memory contents, and the faulting instruction.
- Without this variable, the device may continue executing after an exception (or simply report an error asynchronously), making it difficult to pinpoint the exact cause.

**Note:** This is a debugging-only feature and should not be used in production because the device will hang indefinitely until a debugger is attached.

### 21.3.7 CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT

Sets the default percentage of the L2 cache that can be used for persisting (pinned) access. Persisting L2 cache lines are not evicted by normal (streaming) accesses, providing guaranteed cache residency for frequently accessed data.

**Accepted values:** `0` to `100` (percentage)

```bash
# Reserve 30% of L2 cache for persisting data
export CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT=30

# Disable persisting L2 cache (all L2 is streaming)
export CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT=0
```

Behavior notes:
- This sets the default limit; it can be overridden programmatically using `cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size)`.
- Available on GPUs with compute capability 8.0 and above (Ampere and later).
- Persisting L2 cache is beneficial for data that is accessed repeatedly across multiple kernel launches (e.g., lookup tables, frequently accessed weight matrices).
- Setting a high percentage reduces the L2 cache available for normal streaming accesses and may hurt overall performance if overused.

### 21.3.8 CUDA_DISABLE_PERF_BOOST

Prevents the CUDA driver from requesting a performance state (P-state) boost when a CUDA context is created or when kernels are launched. On Linux, the driver normally boosts the GPU to its highest performance P-state when CUDA work is detected.

**Accepted values:** `1` (disable boost), `0` or unset (default, boost enabled)

```bash
export CUDA_DISABLE_PERF_BOOST=1
```

When enabled:
- The GPU stays at its current P-state (often a lower power state) even during CUDA workloads.
- This is useful for power-constrained environments or for benchmarking at specific clock frequencies.
- It can also be used to reduce power consumption on shared systems where full GPU performance is not needed.

**Note:** This variable is only supported on Linux. On Windows, the WDDM driver manages P-states independently.

---

## 21.4 Module Loading

These environment variables control how CUDA modules (fat binaries containing kernel code) are loaded into the driver. Module loading can be a significant portion of application startup time, especially for applications with many kernels or large fat binaries.

### 21.4.1 CUDA_MODULE_LOADING

Controls the module loading strategy for CUDA fat binaries.

**Accepted values:**

| Value | Description |
|-------|-------------|
| `DEFAULT` | Use the driver's default strategy. Since CUDA 12.3, the default is `LAZY`. Before CUDA 12.3, the default was `EAGER`. |
| `LAZY` | Modules are loaded lazily. Kernel functions and other module symbols are resolved only when first used. This reduces startup time because only the kernels actually used are loaded. |
| `EAGER` | Modules are loaded eagerly at context creation or when the module is first referenced. All kernels in the module are resolved immediately. This increases startup time but eliminates latency spikes during execution. |

```bash
# Lazy loading (reduces startup time)
export CUDA_MODULE_LOADING=LAZY

# Eager loading (eliminates runtime JIT spikes)
export CUDA_MODULE_LOADING=EAGER

# Use driver default
export CUDA_MODULE_LOADING=DEFAULT
```

Behavior notes:
- Lazy loading is the recommended mode for most applications because it significantly reduces startup time, especially for applications that link against large CUDA libraries (cuBLAS, cuDNN, etc.) but only use a subset of their kernels.
- Eager loading is useful for real-time or latency-sensitive applications where unpredictable JIT compilation during execution is unacceptable.
- The `DEFAULT` value allows the driver to choose the best strategy based on the CUDA version and other factors.

### 21.4.2 CUDA_MODULE_DATA_LOADING

Controls how constant data and other module-level data sections are loaded, independent of the kernel code loading strategy.

**Accepted values:**

| Value | Description |
|-------|-------------|
| `LAZY` | Module data (e.g., `__constant__` variables, `__device__` variables) is loaded lazily when first accessed. |
| `EAGER` | Module data is loaded eagerly when the module is loaded. |

```bash
export CUDA_MODULE_DATA_LOADING=LAZY
```

Behavior notes:
- This variable is independent of `CUDA_MODULE_LOADING`. It is possible to have eager kernel loading with lazy data loading, or vice versa.
- Lazy data loading can further reduce startup time, especially for modules with large constant data arrays.

### 21.4.3 CUDA_BINARY_LOADER_THREAD_COUNT

Sets the number of CPU threads used by the CUDA driver to load binary modules in parallel.

**Accepted values:** A positive integer (default: `1`)

```bash
# Use 4 threads for module loading
export CUDA_BINARY_LOADER_THREAD_COUNT=4

# Use 8 threads for module loading on high-core-count systems
export CUDA_BINARY_LOADER_THREAD_COUNT=8
```

Behavior notes:
- Higher thread counts can reduce module loading time on systems with many CPU cores and many CUDA modules to load.
- This is most effective when combined with eager module loading, where multiple modules need to be loaded simultaneously.
- The optimal thread count depends on the number of CPU cores, the number of CUDA modules, and the I/O bandwidth of the storage subsystem.
- Setting this to a value much higher than the number of physical CPU cores may degrade performance due to contention.

---

## 21.5 Error Log

### 21.5.1 CUDA_LOG_FILE

Specifies the destination for CUDA error log messages. The CUDA driver and runtime can emit diagnostic messages when errors or warnings occur.

**Accepted values:**

| Value | Description |
|-------|-------------|
| `stdout` | Write log messages to standard output |
| `stderr` | Write log messages to standard error (default) |
| `<path>` | Write log messages to the specified file path |

```bash
# Log to stderr (default)
export CUDA_LOG_FILE=stderr

# Log to a file
export CUDA_LOG_FILE=/tmp/cuda_errors.log

# Log to stdout
export CUDA_LOG_FILE=stdout
```

Behavior notes:
- The log file is created (or appended to) when the first CUDA error or diagnostic message is emitted.
- On multi-process systems, using separate log files per process is recommended to avoid interleaved output.
- This variable is primarily useful for debugging; in production, applications should use the CUDA error checking APIs (`cudaGetLastError()`, `cudaPeekAtLastError()`, etc.) for error handling.

---

## 21.6 Other Variables

### 21.6.1 CUDA_API_PER_THREAD_DEFAULT_STREAM

Changes the default stream behavior from the legacy default stream (which synchronizes with all other streams) to a per-thread default stream (which does not synchronize with other streams and is equivalent to a separate stream per host thread).

**Usage:** Set the variable to any value (its presence enables the feature).

```bash
export CUDA_API_PER_THREAD_DEFAULT_STREAM=1
```

When enabled:
- Each host thread gets its own default stream instead of sharing the global legacy default stream.
- Operations in the per-thread default stream can overlap with operations in other streams (including other threads' default streams).
- This matches the behavior of the `--default-stream per-thread` nvcc flag.
- The per-thread default stream does NOT synchronize with explicit streams (non-default streams created with `cudaStreamCreate()`).

```cpp
// Compile-time equivalent
// nvcc --default-stream per-thread myapp.cu -o myapp

// Example: Two threads can overlap their default-stream work
#include <cuda_runtime.h>
#include <thread>

void worker(int deviceId) {
    cudaSetDevice(deviceId);
    // Kernel launches here use the per-thread default stream
    myKernel<<<grid, block>>>();  // No synchronization with other threads
    cudaDeviceSynchronize();
}

int main() {
    std::thread t1(worker, 0);
    std::thread t2(worker, 0);
    t1.join();
    t2.join();
}
```

**Related Driver API suffixes:**
When using per-thread default stream with the Driver API, use the `_ptsz` (per-thread stream) or `_ptds` (per-thread device synchronize) suffix variants of Driver API functions. These suffixes can be selected at symbol resolution time using `CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM`.

### 21.6.2 CUDA_FORCE_CDP1_IF_SUPPORTED

Forces the use of CUDA Dynamic Parallelism (CDP) version 1 even when the hardware and driver support CDP version 2 (which is available starting with compute capability 9.0 / Hopper architecture).

**Accepted values:** `1` (force CDP v1), `0` or unset (default, use latest supported CDP version)

```bash
export CUDA_FORCE_CDP1_IF_SUPPORTED=1
```

When enabled:
- Child kernels launched from device code use the CDP v1 programming model, even on GPUs that support CDP v2.
- CDP v1 has higher overhead than CDP v2 (it uses a separate device-side runtime and more complex synchronization).
- This variable is useful for backward compatibility testing or for debugging CDP-related issues.

**Note:** CDP v2 (available on Hopper and later) provides significantly lower overhead for device-side kernel launches compared to CDP v1. Only use this variable if you have a specific reason to force the older model.

---

## 21.7 Quick Reference Table

| Variable | Values | Default | Category |
|----------|--------|---------|----------|
| `CUDA_VISIBLE_DEVICES` | Index/UUID/MIG list | All GPUs | Device Enumeration |
| `CUDA_DEVICE_ORDER` | `FASTEST_FIRST`, `PCI_BUS_ID` | `FASTEST_FIRST` | Device Enumeration |
| `CUDA_MANAGED_FORCE_DEVICE_ALLOC` | `0`, `1` | `0` | Device Enumeration |
| `CUDA_CACHE_DISABLE` | `0`, `1` | `0` | JIT Compilation |
| `CUDA_CACHE_PATH` | Filesystem path | `~/.nv/ComputeCache` | JIT Compilation |
| `CUDA_CACHE_MAXSIZE` | Bytes (max 4 GiB) | 1 GiB | JIT Compilation |
| `CUDA_FORCE_PTX_JIT` | `0`, `1` | `0` | JIT Compilation |
| `CUDA_FORCE_JIT` | `0`, `1` | `0` | JIT Compilation |
| `CUDA_DISABLE_PTX_JIT` | `0`, `1` | `0` | JIT Compilation |
| `CUDA_DISABLE_JIT` | `0`, `1` | `0` | JIT Compilation |
| `CUDA_FORCE_PRELOAD_LIBRARIES` | `0`, `1` | `0` | JIT Compilation |
| `CUDA_LAUNCH_BLOCKING` | `0`, `1` | `0` | Execution |
| `CUDA_DEVICE_MAX_CONNECTIONS` | `1`-`32` | `8` | Execution |
| `CUDA_DEVICE_MAX_COPY_CONNECTIONS` | Integer | Varies | Execution |
| `CUDA_SCALE_LAUNCH_QUEUES` | `0.25`, `0.5`, `2`, `4` | None | Execution |
| `CUDA_GRAPHS_USE_NODE_PRIORITY` | `0`, `1` | `0` | Execution |
| `CUDA_DEVICE_WAITS_ON_EXCEPTION` | `0`, `1` | `0` | Execution |
| `CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT` | `0`-`100` | Varies | Execution |
| `CUDA_DISABLE_PERF_BOOST` | `0`, `1` | `0` | Execution |
| `CUDA_MODULE_LOADING` | `DEFAULT`, `LAZY`, `EAGER` | `LAZY` (12.3+) | Module Loading |
| `CUDA_MODULE_DATA_LOADING` | `LAZY`, `EAGER` | Varies | Module Loading |
| `CUDA_BINARY_LOADER_THREAD_COUNT` | Positive integer | `1` | Module Loading |
| `CUDA_LOG_FILE` | `stdout`, `stderr`, path | `stderr` | Error Log |
| `CUDA_API_PER_THREAD_DEFAULT_STREAM` | Any value | Not set | Other |
| `CUDA_FORCE_CDP1_IF_SUPPORTED` | `0`, `1` | `0` | Other |

---

## Usage Tips

### Debugging Configuration

```bash
# Recommended environment for debugging CUDA issues
export CUDA_LAUNCH_BLOCKING=1
export CUDA_DEVICE_WAITS_ON_EXCEPTION=1
export CUDA_LOG_FILE=stderr
```

### Performance Optimization Configuration

```bash
# Maximize concurrency for multi-stream workloads
export CUDA_DEVICE_MAX_CONNECTIONS=32
export CUDA_MODULE_LOADING=LAZY
export CUDA_CACHE_MAXSIZE=4294967296  # 4 GiB
```

### Deployment Configuration

```bash
# Deterministic device ordering and controlled visibility
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
export CUDA_MODULE_LOADING=LAZY
```
