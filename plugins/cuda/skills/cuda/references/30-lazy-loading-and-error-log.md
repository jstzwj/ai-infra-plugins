# Chapter 30: Lazy Loading and Error Log Management

CUDA provides two complementary features for improving developer experience and application behavior: **lazy loading** reduces CUDA driver initialization time by deferring module loading until functions are actually needed, and **error log management** provides structured, plain-English diagnostic output for debugging and troubleshooting.

## 30.1 Lazy Loading

### Concept

By default, when a CUDA application launches, the driver eagerly loads all code modules (PTX and cubin) embedded in the binary or in dependent libraries into each GPU's context. For applications with many kernels or large libraries (e.g., cuDNN, cuBLAS), this initialization overhead can be significant -- sometimes tens of seconds on multi-GPU systems.

Lazy loading defers this work: modules are loaded only when a specific kernel function is first referenced (e.g., via `cuModuleGetFunction` or `cudaFuncGetAttributes`). Functions that are never called are never loaded, saving both time and device memory.

### Change History

| CUDA Version | Behavior |
|---|---|
| 11.7 | Introduced as opt-in feature. Disabled by default. Controlled via `CUDA_MODULE_LOADING=LAZY`. |
| 12.2 | Enabled by default on Linux. Windows remained eager by default. |
| 12.3 | Enabled by default on all platforms (Linux and Windows). |

### Requirements

- **Runtime API version**: 11.7 or later.
- **Driver API version**: R515 (515.x) or later.
- **Environment variable**: `CUDA_MODULE_LOADING` controls the loading mode.

```
# Enable lazy loading explicitly
CUDA_MODULE_LOADING=LAZY ./my_application

# Force eager loading (restore pre-11.7 behavior)
CUDA_MODULE_LOADING=EAGER ./my_application

# Let CUDA decide (default since 12.3: LAZY)
unset CUDA_MODULE_LOADING
./my_application
```

### Checking the Loading Mode at Runtime

The Driver API provides a function to query which loading mode is active:

```cpp
#include <cuda.h>
#include <stdio.h>

void checkLoadingMode() {
    CUmoduleLoadingMode mode;
    CUresult res = cuModuleGetLoadingMode(&mode);
    if (res != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(res, &errStr);
        fprintf(stderr, "cuModuleGetLoadingMode failed: %s\n", errStr);
        return;
    }

    switch (mode) {
        case CU_MODULE_LOADING_MODE_EAGER:
            printf("Module loading mode: EAGER\n");
            break;
        case CU_MODULE_LOADING_MODE_LAZY:
            printf("Module loading mode: LAZY\n");
            break;
        case CU_MODULE_LOADING_MODE_COUNT:
        default:
            printf("Module loading mode: UNKNOWN (%d)\n", mode);
            break;
    }
}
```

### Forcing Eager Loading of Specific Modules

In some situations you may want lazy loading globally but need certain modules loaded eagerly. Calling any of the following functions triggers loading of the referenced module:

- **Driver API**: `cuModuleGetFunction()` -- forces the containing module to be fully loaded.
- **Runtime API**: `cudaFuncGetAttributes()` -- triggers loading of the module containing the specified kernel.

```cpp
// Force eager loading of a specific kernel's module
void preloadKernel(const void* funcPtr) {
    cudaFuncAttributes attr;
    cudaError_t err = cudaFuncGetAttributes(&attr, funcPtr);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to preload kernel: %s\n",
                cudaGetErrorString(err));
    } else {
        printf("Kernel preloaded: shared size %zu, const size %zu, "
               "local size %zu, max threads per block %d\n",
               attr.sharedSizeBytes, attr.constSizeBytes,
               attr.localSizeBytes, attr.maxThreadsPerBlock);
    }
}

// Usage with a kernel
__global__ void myKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] *= 2.0f;
}

void applicationInit() {
    // Preload myKernel's module so it's ready for first launch
    preloadKernel((const void*)myKernel);
}
```

### Potential Hazards and Mitigations

While lazy loading improves startup time, it introduces subtle behavioral differences that can affect certain workloads.

#### Hazard 1: Impact on Concurrent Kernel Execution

When lazy loading is active, the first launch of a kernel triggers module loading. If two streams attempt to launch different kernels concurrently, the second launch may block while the first module is still being loaded. This serializes what would otherwise be concurrent execution.

**Mitigation options:**

1. **Preload critical kernels** before the performance-sensitive section of the application.

```cpp
// Preload all kernels in a warmup phase
void preloadAllKernels() {
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, kernelA);
    cudaFuncGetAttributes(&attr, kernelB);
    cudaFuncGetAttributes(&attr, kernelC);
    // Now all modules are loaded; concurrent launches will truly overlap
}
```

2. **Use EAGER mode** if your application has many small concurrent kernels and startup time is not critical.

```bash
CUDA_MODULE_LOADING=EAGER ./my_application
```

#### Hazard 2: Large Memory Allocations

Some CUDA modules embed large constant data (e.g., lookup tables, weight matrices) that are loaded into device memory when the module is loaded. Under lazy loading, these allocations occur at first kernel use rather than at application startup. If this happens during a phase where most of the device memory is already in use, the allocation may fail or trigger unexpected eviction.

**Mitigation options:**

1. **Use `cudaMallocAsync`** instead of `cudaMalloc`. The stream-ordered allocator is more resilient to memory pressure and can better handle deferred allocations.

```cpp
// Prefer async allocation with lazy loading
cudaStream_t stream;
cudaStreamCreate(&stream);
void* buffer;
cudaMallocAsync(&buffer, size, stream);
// ... use buffer ...
cudaFreeAsync(buffer, stream);
```

2. **Add a memory buffer** to account for module data that will be loaded lazily.

```cpp
// Reserve extra memory before kernel launches
size_t freeMem, totalMem;
cudaMemGetInfo(&freeMem, &totalMem);
size_t reservedSize = 256ULL * 1024 * 1024; // 256 MB buffer
void* reserved;
cudaMalloc(&reserved, reservedSize);
// ... launch kernels (lazy module data allocations have room) ...
cudaFree(reserved); // release the buffer
```

3. **Preload modules** that contain large constant data early in the application.

#### Hazard 3: Performance Measurements

Lazy loading can distort benchmarking results. The first iteration of a kernel includes the module loading time, which is not representative of steady-state performance.

**Mitigation options:**

1. **Include a warmup iteration** that is excluded from timing measurements.

```cpp
// Warmup iteration (triggers lazy loading)
myKernel<<<grid, block>>>(d_data, n);
cudaDeviceSynchronize();

// Timed iterations
auto start = std::chrono::high_resolution_clock::now();
for (int i = 0; i < numIterations; i++) {
    myKernel<<<grid, block>>>(d_data, n);
}
cudaDeviceSynchronize();
auto end = std::chrono::high_resolution_clock::now();

float elapsed = std::chrono::duration<float, std::milli>(end - start).count();
printf("Average time: %.3f ms\n", elapsed / numIterations);
```

2. **Preload all kernels** before benchmarking begins.

```cpp
void benchmarkPreload() {
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, kernelToBenchmark);
    cudaDeviceSynchronize(); // Ensure loading completes
}
```

### Interaction with cuDNN and Other Libraries

Large CUDA libraries like cuDNN, cuBLAS, and cuFFT embed hundreds of kernels for different GPU architectures and algorithm variants. Lazy loading significantly reduces their initialization overhead. However, the first call to any library function still triggers loading of the relevant subset of kernels. For consistent performance in production workloads:

```cpp
// Initialize cuDNN and trigger loading of commonly-used kernels
cudnnHandle_t cudnn;
cudnnCreate(&cudnn);

// Optionally run a dummy operation to preload library internals
cudnnTensorDescriptor_t desc;
cudnnCreateTensorDescriptor(&desc);
cudnnSetTensor4dDescriptor(desc, CUDNN_DATA_FLOAT,
    CUDNN_TENSOR_NCHW, 1, 1, 1, 1);
cudnnDestroyTensorDescriptor(desc);

// Subsequent calls will not incur lazy loading overhead
```

## 30.2 Error Log Management

### Overview

CUDA error log management provides a structured, plain-English logging system for CUDA API errors, warnings, and informational messages. Rather than requiring developers to interpret numeric error codes, the log system outputs human-readable descriptions including context about what went wrong and potential causes.

The error log system is available through the Driver API and can output to files, memory buffers, or user-registered callback functions.

### Configuration

#### Setting the Log Output Destination

Use the `CUDA_LOG_FILE` environment variable to control where log output is written:

```bash
# Write logs to stdout
CUDA_LOG_FILE=stdout ./my_application

# Write logs to stderr
CUDA_LOG_FILE=stderr ./my_application

# Write logs to a specific file
CUDA_LOG_FILE=/tmp/cuda_errors.log ./my_application

# Disable file-based logging (callbacks still work)
unset CUDA_LOG_FILE
./my_application
```

#### Log Output Format

Each log entry follows a structured format:

```
[Time][TID][Source][Severity][API Entry Point] Message
```

Where:
- **Time**: Timestamp of the event (e.g., `2025-01-15T14:30:22.123456`).
- **TID**: Thread ID that generated the log entry.
- **Source**: The CUDA component that produced the message (e.g., `CUDA_DRIVER`, `CUDA_RUNTIME`, `MEMORY`).
- **Severity**: The severity level (e.g., `ERROR`, `WARNING`, `INFO`).
- **API Entry Point**: The CUDA function that was called when the log was generated.
- **Message**: Human-readable description of the event.

Example output:

```
[2025-01-15T14:30:22.123456][TID:28451][CUDA_DRIVER][ERROR][cuMemCreate] The requested allocation size exceeds available memory on NUMA node 0. Requested: 8589934592 bytes, Available: 4294967296 bytes. Consider using a smaller allocation or a different NUMA node.
[2025-01-15T14:30:22.456789][TID:28451][CUDA_DRIVER][WARNING][cuCtxCreate] Creating a context on device 0 with compute capability 9.0 but the module was compiled for compute capability 8.0. This may result in reduced performance.
```

### API Reference

#### Registering a Log Callback

Register a callback function to receive log entries in real time. The callback is invoked each time a log entry is generated.

```cpp
#include <cuda.h>
#include <stdio.h>

// Callback signature: void callback(const char* message, void* userData)
void myLogCallback(const char* message, void* userData) {
    // Forward CUDA logs to the application's logging system
    FILE* logFile = (FILE*)userData;
    fprintf(logFile, "[CUDA LOG] %s\n", message);
    fflush(logFile);
}

void setupLogging() {
    FILE* logFile = fopen("/tmp/cuda_app.log", "a");

    CUresult res = cuLogsRegisterCallback(myLogCallback, logFile);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "Failed to register log callback\n");
    }
}

void teardownLogging() {
    FILE* logFile = /* retrieve from application state */;
    cuLogsUnregisterCallback(myLogCallback);
    fclose(logFile);
}
```

#### Unregistering a Log Callback

Remove a previously registered callback when it is no longer needed:

```cpp
// Remove the callback (must match the exact function pointer used during registration)
cuLogsUnregisterCallback(myLogCallback);
```

#### Retrieving Current Log Entries

Retrieve log entries that have accumulated in the internal buffer:

```cpp
void dumpCurrentLogs() {
    // First call: determine the required buffer size
    unsigned int count = 0;
    CUresult res = cuLogsCurrent(NULL, &count);

    if (count == 0) {
        printf("No log entries in buffer\n");
        return;
    }

    // Allocate buffer (max 25600 bytes total, max 100 entries)
    const size_t bufferSize = 25600;
    char* buffer = (char*)malloc(bufferSize);

    // Second call: retrieve the entries
    res = cuLogsCurrent(buffer, &count);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "cuLogsCurrent failed: %d\n", res);
    } else {
        printf("Retrieved %u log entries:\n%s\n", count, buffer);
    }

    free(buffer);
}
```

#### Dumping Logs to a File

Write all buffered log entries to a file on disk:

```cpp
void dumpLogsToFile() {
    const char* path = "/tmp/cuda_error_dump.log";
    CUresult res = cuLogsDumpToFile(path);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "cuLogsDumpToFile failed: %d\n", res);
    } else {
        printf("Log entries written to %s\n", path);
    }
}
```

#### Dumping Logs to Memory

Retrieve log entries into a caller-supplied memory buffer with precise size control:

```cpp
void dumpLogsToMemory() {
    // Maximum buffer size: 25600 bytes
    // Maximum entries: 100
    const size_t bufferSize = 25600;
    char* buffer = (char*)malloc(bufferSize);
    unsigned int count = 0;

    CUresult res = cuLogsDumpToMemory(buffer, bufferSize, &count);
    if (res != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(res, &errStr);
        fprintf(stderr, "cuLogsDumpToMemory failed: %s\n", errStr);
    } else {
        printf("Dumped %u log entries to memory buffer\n", count);
        // Process the buffer contents as needed
        printf("--- Log Contents ---\n%s\n", buffer);
    }

    free(buffer);
}
```

### Complete Logging Example

The following example demonstrates a complete logging setup for a CUDA application:

```cpp
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static FILE* g_logFile = NULL;

void logCallback(const char* message, void* userData) {
    if (g_logFile) {
        fprintf(g_logFile, "%s\n", message);
        fflush(g_logFile);
    }
    // Also print to stderr for immediate visibility
    fprintf(stderr, "[CUDA] %s\n", message);
}

int main(int argc, char** argv) {
    // Initialize CUDA
    CUresult res = cuInit(0);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "cuInit failed\n");
        return 1;
    }

    // Setup logging
    g_logFile = fopen("cuda_application.log", "w");
    if (!g_logFile) {
        fprintf(stderr, "Could not open log file\n");
        return 1;
    }

    // Register callback for real-time logging
    res = cuLogsRegisterCallback(logCallback, NULL);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "Failed to register log callback\n");
    }

    // --- Application work ---
    CUdevice device;
    cuDeviceGet(&device, 0);
    CUcontext ctx;
    cuCtxCreate(&ctx, 0, device);

    // Intentional error: allocate an impossibly large amount
    CUmemGenericAllocationHandle handle;
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = 0;
    size_t absurdSize = (size_t)1024 * 1024 * 1024 * 1024 * 1024; // 1 PB
    res = cuMemCreate(&handle, absurdSize, &prop, 0);
    if (res != CUDA_SUCCESS) {
        printf("Expected failure: allocation of 1 PB\n");
    }

    // Dump buffered logs to file
    cuLogsDumpToFile("cuda_error_dump.log");

    // Also retrieve logs to memory for programmatic inspection
    const size_t bufSize = 25600;
    char logBuf[25600];
    unsigned int entryCount = 0;
    cuLogsDumpToMemory(logBuf, bufSize, &entryCount);
    printf("Captured %u log entries\n", entryCount);

    // Cleanup
    cuCtxDestroy(ctx);
    cuLogsUnregisterCallback(logCallback);
    fclose(g_logFile);

    return 0;
}
```

### Limitations

The error log system has several important limitations to be aware of:

1. **Buffer size**: The internal log buffer holds a maximum of **25,600 bytes** across a maximum of **100 entries**. When the buffer is full, older entries are overwritten. Use callbacks for reliable capture of all entries.

2. **API coverage**: Not all CUDA APIs generate log entries. The system primarily covers error-producing paths in memory management, context creation, module loading, and launch operations. Coverage is expected to expand in future CUDA releases.

3. **Language**: Log messages are currently available only in **US English**. There is no localization support.

4. **Driver API only**: The log management functions (`cuLogsRegisterCallback`, `cuLogsCurrent`, `cuLogsDumpToFile`, `cuLogsDumpToMemory`) are available only through the Driver API. Runtime API users can still benefit from log output by setting `CUDA_LOG_FILE`, but cannot programmatically access the buffer or register callbacks.

5. **Thread safety**: Callback functions are invoked synchronously from the thread that generated the log entry. Avoid making CUDA API calls from within a callback, as this may lead to deadlocks.

### Summary of Log API Functions

| Function | Description |
|---|---|
| `cuLogsRegisterCallback(callback, userData)` | Register a callback to receive log entries in real time. |
| `cuLogsUnregisterCallback(callback)` | Remove a previously registered callback. |
| `cuLogsCurrent(buffer, &count)` | Retrieve log entries currently in the internal buffer. Pass `NULL` for buffer to query count only. |
| `cuLogsDumpToFile(path)` | Write all buffered log entries to the specified file path. |
| `cuLogsDumpToMemory(buffer, size, &count)` | Write buffered log entries into a caller-supplied memory buffer. Returns the number of entries written. |
