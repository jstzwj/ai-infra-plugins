# Chapter 14: CUDA Driver API

The CUDA Driver API is a lower-level, handle-based interface to CUDA. It provides fine-grained control over GPU resources, including explicit context management, module loading, JIT compilation, and kernel execution. The Driver API is implemented as a dynamic library (`libcuda.so` on Linux, `cuda.dll` on Windows) and all entry points are prefixed with `cu`. This API complements the higher-level CUDA Runtime API (prefixed with `cuda`).

## 14.1 Overview

### 14.1.1 Driver API vs. Runtime API

| Aspect | Driver API | Runtime API |
|--------|-----------|-------------|
| Prefix | `cu` | `cuda` |
| Header | `cuda.h` | `cuda_runtime.h` |
| Library | `libcuda.so` / `cuda.dll` | `libcudart.so` / `cudart.dll` |
| Initialization | Explicit `cuInit(0)` | Implicit on first API call |
| Context management | Explicit push/pop/retain | Implicit (one context per device) |
| Module loading | Explicit `cuModuleLoad` | Automatic via nvcc (fatbin embedded) |
| Kernel launch | `cuLaunchKernel()` | `<<<grid, block>>>(...)` syntax |
| Error handling | `CUresult` return codes | `cudaError_t` return codes + `cudaGetLastError()` |
| ABI stability | Stable across CUDA versions | Stable across CUDA versions |
| Use case | Plugin systems, language runtimes, fine control | Application development, rapid prototyping |

### 14.1.2 Initialization

Before calling any Driver API function, you must initialize the driver:

```cpp
#include <cuda.h>

CUresult res = cuInit(0);
if (res != CUDA_SUCCESS) {
    const char* errStr;
    cuGetErrorString(res, &errStr);
    fprintf(stderr, "cuInit failed: %s\n", errStr);
    return -1;
}
```

The `cuInit()` call:
- Must be called before any other Driver API function.
- Takes a single `flags` argument that must be 0 (reserved for future use).
- Initializes the driver and discovers all CUDA devices.
- Can be called multiple times safely; subsequent calls are no-ops.

### 14.1.3 Error Handling

All Driver API functions return `CUresult`. A helper macro is recommended:

```cpp
#define CU_CHECK(expr) do { \
    CUresult res = (expr); \
    if (res != CUDA_SUCCESS) { \
        const char* errStr; \
        const char* errName; \
        cuGetErrorString(res, &errStr); \
        cuGetErrorName(res, &errName); \
        fprintf(stderr, "CUDA Driver Error: %s:%d: %s (%s)\n", \
                __FILE__, __LINE__, errName, errStr); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
```

## 14.2 Driver API Objects

The Driver API is handle-based. Each resource is represented by an opaque handle:

| Object | Handle Type | Description | Created By | Destroyed By |
|--------|------------|-------------|------------|--------------|
| Device | `CUdevice` (int) | A CUDA-enabled GPU | `cuDeviceGet` | N/A (not created/destroyed) |
| Context | `CUcontext` | Execution context (~CPU process analog) | `cuCtxCreate`, `cuDevicePrimaryCtxRetain` | `cuCtxDestroy`, `cuDevicePrimaryCtxRelease` |
| Module | `CUmodule` | Loaded code (~dynamic library) | `cuModuleLoad`, `cuModuleLoadDataEx` | `cuModuleUnload` |
| Function | `CUfunction` | A kernel within a module | `cuModuleGetFunction` | N/A (freed with module) |
| Device Memory | `CUdeviceptr` | Pointer to device memory | `cuMemAlloc`, `cuMemAllocManaged` | `cuMemFree` |
| Host Memory | `void*` | Pinned host memory | `cuMemAllocHost` | `cuMemFreeHost` |
| Stream | `CUstream` | Command queue | `cuStreamCreate` | `cuStreamDestroy` |
| Event | `CUevent` | Timestamp/sync marker | `cuEventCreate` | `cuEventDestroy` |
| Module JIT | `CUlinkState` | Linker state for multi-PTX linking | `cuLinkCreate` | `cuLinkDestroy` |

### 14.2.1 Object Lifecycle Example

```cpp
#include <cuda.h>

int main() {
    CU_CHECK(cuInit(0));

    // Device
    CUdevice device;
    CU_CHECK(cuDeviceGet(&device, 0));

    // Context
    CUcontext ctx;
    CU_CHECK(cuCtxCreate(&ctx, 0, device));

    // Module
    CUmodule module;
    CU_CHECK(cuModuleLoad(&module, "mykernel.fatbin"));

    // Function
    CUfunction kernel;
    CU_CHECK(cuModuleGetFunction(&kernel, module, "myKernel"));

    // Stream
    CUstream stream;
    CU_CHECK(cuStreamCreate(&stream, 0));

    // Memory
    CUdeviceptr dptr;
    CU_CHECK(cuMemAlloc(&dptr, 1024 * sizeof(float)));

    // ... use resources ...

    // Cleanup (reverse order)
    CU_CHECK(cuMemFree(dptr));
    CU_CHECK(cuStreamDestroy(stream));
    CU_CHECK(cuModuleUnload(module));
    CU_CHECK(cuCtxDestroy(ctx));

    return 0;
}
```

## 14.3 Context Management

A CUDA context is analogous to a CPU process. It encapsulates all GPU state: allocated memory, loaded modules, streams, events, and execution state. Each context is associated with a single device.

### 14.3.1 Primary Context

Each device has a primary context that is shared with the Runtime API. This is the recommended way to obtain a context when mixing Runtime and Driver API code:

```cpp
CUcontext ctx;
CU_CHECK(cuDevicePrimaryCtxRetain(&ctx, device));

// The primary context is now active. Runtime API calls on this device
// will use the same context.

// Push the context as current (needed for Driver API calls)
CU_CHECK(cuCtxPushCurrent(ctx));

// ... Driver API operations ...

// Pop the context
CUcontext poppedCtx;
CU_CHECK(cuCtxPopCurrent(&poppedCtx));
// poppedCtx == ctx

// Release the primary context when done
CU_CHECK(cuDevicePrimaryCtxRelease(device));
```

Important notes on primary contexts:
- The primary context is reference-counted. `cuDevicePrimaryCtxRetain` increments the refcount; `cuDevicePrimaryCtxRelease` decrements it.
- Runtime API functions implicitly retain and release the primary context.
- The primary context is lazily initialized on first use.
- You can configure the primary context before it is initialized:

```cpp
// Set primary context flags before first use
CU_CHECK(cuDevicePrimaryCtxSetFlags(device,
    CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST));
// Must be called before cuDevicePrimaryCtxRetain or any Runtime API call
```

### 14.3.2 Creating a New Context

You can also create additional contexts on a device (rarely needed):

```cpp
CUcontext ctx;
CU_CHECK(cuCtxCreate(&ctx, CU_CTX_SCHED_AUTO, device));

// The new context is now current

// Context flags:
// CU_CTX_SCHED_AUTO      - Default scheduling
// CU_CTX_SCHED_SPIN      - Spin-loop for synchronization
// CU_CTX_SCHED_YIELD     - Yield CPU during synchronization
// CU_CTX_SCHED_BLOCKING_SYNC - Block the thread during synchronization
// CU_CTX_MAP_HOST        - Allow mapped pinned memory
// CU_CTX_LMEM_RESIZE_TO_MAX - Resize local memory to max (avoid realloc)
```

### 14.3.3 Context Stack

CUDA maintains a per-thread stack of contexts. Only the top of the stack is "current":

```cpp
CUcontext ctx0, ctx1;
cuDevicePrimaryCtxRetain(&ctx0, 0);
cuDevicePrimaryCtxRetain(&ctx1, 1);

// Push contexts onto the stack
cuCtxPushCurrent(ctx0);
// ctx0 is now current

cuCtxPushCurrent(ctx1);
// ctx1 is now current (ctx0 is below on the stack)

CUcontext topCtx;
cuCtxGetCurrent(&topCtx);
// topCtx == ctx1

cuCtxPopCurrent(&topCtx);
// topCtx == ctx1, ctx0 is now current again

cuCtxPopCurrent(&topCtx);
// topCtx == ctx0, stack is empty
```

### 14.3.4 Context Query

```cpp
// Get the current context
CUcontext currentCtx;
CUresult res = cuCtxGetCurrent(&currentCtx);
if (res == CUDA_SUCCESS && currentCtx != NULL) {
    // A context is currently bound

    // Get the device for the current context
    CUdevice device;
    cuCtxGetDevice(&device);

    // Get context flags
    unsigned int flags;
    cuCtxGetFlags(&flags);

    // Get/set cache configuration
    CUfunc_cache cacheConfig;
    cuCtxGetCacheConfig(&cacheConfig);
    cuCtxSetCacheConfig(CU_FUNC_CACHE_PREFER_SHARED);
}
```

### 14.3.5 Context Synchronization

```cpp
// Synchronize on the current context (wait for all streams)
CU_CHECK(cuCtxSynchronize());

// Set the synchronization limit for the context
// (how long the driver spins before blocking)
```

## 14.4 Module Management

A CUDA module is the Driver API's analog of a dynamically loaded library. It contains compiled GPU code (PTX and/or cubin) for one or more kernels.

### 14.4.1 Loading a Module from File

```cpp
CUmodule module;
CUresult res = cuModuleLoad(&module, "mykernels.fatbin");
if (res != CUDA_SUCCESS) {
    const char* errStr;
    cuGetErrorString(res, &errStr);
    fprintf(stderr, "Failed to load module: %s\n", errStr);
}
```

Supported file formats:
- `.cubin` -- CUDA binary (device code for a specific architecture)
- `.fatbin` -- Fat binary (contains code for multiple architectures)
- `.ptx` -- PTX assembly text

### 14.4.2 Loading a Module from Data (JIT Compilation)

Load a module from a PTX string in memory, with JIT compilation options:

```cpp
const char* ptx = R"(
.version 8.0
.target sm_80
.address_size 64

.visible .entry vecAdd(
    .param .u64 vecAdd_param_0,
    .param .u64 vecAdd_param_1,
    .param .u64 vecAdd_param_2,
    .param .u32 vecAdd_param_3
) {
    .reg .f32  %f<4>;
    .reg .b32  %r<5>;
    .reg .b64  %rd<8>;

    ld.param.u64 %rd1, [vecAdd_param_0];
    ld.param.u64 %rd2, [vecAdd_param_1];
    ld.param.u64 %rd3, [vecAdd_param_2];
    ld.param.u32 %r1, [vecAdd_param_3];

    // ... kernel body ...
    ret;
}
)";

// JIT compilation options
int numOptions = 2;
CUjit_option options[2];
void* optionValues[2];

// Option 1: Generate debug info
options[0] = CU_JIT_GENERATE_DEBUG_INFO;
int debug = 1;
optionValues[0] = (void*)(intptr_t)debug;

// Option 2: Set log buffer for compilation messages
options[1] = CU_JIT_ERROR_LOG_BUFFER;
char logBuffer[4096];
optionValues[1] = logBuffer;

CUjit_option logBufferSizeOpt = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
void* logBufferSizeVal = (void*)(intptr_t)sizeof(logBuffer);
// Note: you would add this as a third option pair

CUmodule module;
CUresult res = cuModuleLoadDataEx(&module, ptx, numOptions,
                                   options, optionValues);
if (res != CUDA_SUCCESS) {
    fprintf(stderr, "JIT compilation failed:\n%s\n", logBuffer);
}
```

Available JIT options:

| Option | Description |
|--------|-------------|
| `CU_JIT_MAX_REGISTERS` | Max registers per thread |
| `CU_JIT_THREADS_PER_BLOCK` | Hint for threads per block |
| `CU_JIT_WALL_TIME` | Returns wall time of compilation |
| `CU_JIT_INFO_LOG_BUFFER` | Buffer for info log messages |
| `CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES` | Size of info log buffer |
| `CU_JIT_ERROR_LOG_BUFFER` | Buffer for error log messages |
| `CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES` | Size of error log buffer |
| `CU_JIT_OPTIMIZATION_LEVEL` | Optimization level (0-4) |
| `CU_JIT_TARGET_FROM_CUCONTEXT` | Derive target from current context |
| `CU_JIT_TARGET` | Target architecture (CUjit_target) |
| `CU_JIT_FALLBACK_STRATEGY` | Strategy when matching cubin not found |
| `CU_JIT_GENERATE_DEBUG_INFO` | Generate debug info (0 or 1) |
| `CU_JIT_LOG_VERBOSE` | Verbose logging (0 or 1) |
| `CU_JIT_GENERATE_LINE_INFO` | Generate line number info |
| `CU_JIT_CACHE_MODE` | Cache configuration hint |

### 14.4.3 Multi-PTX Linking

The Driver API provides a linker for combining multiple PTX or cubin inputs into a single module. This is useful for:
- Combining kernels from different source files at runtime
- Linking device code libraries
- Implementing runtime code generation pipelines

```cpp
CUlinkState linkState;

// Create linker with options
CUjit_option options[3];
void* optionValues[3];

options[0] = CU_JIT_MAX_REGISTERS;
int maxRegs = 64;
optionValues[0] = (void*)(intptr_t)maxRegs;

options[1] = CU_JIT_ERROR_LOG_BUFFER;
char errorLog[8192];
optionValues[1] = errorLog;

options[2] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
optionValues[2] = (void*)(intptr_t)sizeof(errorLog);

CUresult res = cuLinkCreate(3, options, optionValues, &linkState);
if (res != CUDA_SUCCESS) {
    fprintf(stderr, "cuLinkCreate failed\n");
}

// Add PTX data
const char* ptx1 = "..."; // PTX string for kernel 1
res = cuLinkAddData(linkState, CU_JIT_INPUT_PTX, (void*)ptx1,
                     strlen(ptx1) + 1, "kernel1.ptx", 0, NULL, NULL);

// Add cubin data (pre-compiled for a specific arch)
const void* cubin2 = ...; // Pointer to cubin data
size_t cubin2Size = ...;
res = cuLinkAddData(linkState, CU_JIT_INPUT_CUBIN, (void*)cubin2,
                     cubin2Size, "kernel2.cubin", 0, NULL, NULL);

// Add an object file
res = cuLinkAddFile(linkState, CU_JIT_INPUT_OBJECT, "kernels3.o",
                     0, NULL, NULL);

// Add a library file
res = cuLinkAddFile(linkState, CU_JIT_INPUT_LIBRARY, "libdevcode.a",
                     0, NULL, NULL);

// Complete linking
void* cubinOut;
size_t cubinOutSize;
res = cuLinkComplete(linkState, &cubinOut, &cubinOutSize);
if (res != CUDA_SUCCESS) {
    fprintf(stderr, "Link failed: %s\n", errorLog);
}

// Load the linked cubin as a module
CUmodule module;
res = cuModuleLoadData(&module, cubinOut);

// Get kernel functions
CUfunction kernel1, kernel2;
cuModuleGetFunction(&kernel1, module, "kernel1");
cuModuleGetFunction(&kernel2, module, "kernel2");

// ... use kernels ...

// Cleanup
cuModuleUnload(module);
cuLinkDestroy(linkState);
```

Input types for `cuLinkAddData`/`cuLinkAddFile`:

| Type | Description |
|------|-------------|
| `CU_JIT_INPUT_CUBIN` | CUDA binary (cubin) |
| `CU_JIT_INPUT_PTX` | PTX assembly text |
| `CU_JIT_INPUT_OBJECT` | Host object file with device code |
| `CU_JIT_INPUT_LIBRARY` | Device code library archive |

### 14.4.4 Getting Kernel Functions

```cpp
CUfunction kernel;
CUresult res = cuModuleGetFunction(&kernel, module, "myKernel");
if (res != CUDA_SUCCESS) {
    fprintf(stderr, "Kernel 'myKernel' not found in module\n");
}
```

### 14.4.5 Getting Global Variables

```cpp
CUdeviceptr globalVarPtr;
size_t globalVarSize;
CUresult res = cuModuleGetGlobal(&globalVarPtr, &globalVarSize,
                                  module, "myGlobalArray");
if (res == CUDA_SUCCESS) {
    // Copy data to/from the global variable
    cuMemcpyHtoD(globalVarPtr, hostData, globalVarSize);
}
```

### 14.4.6 Getting Texture and Surface References

```cpp
CUtexref texRef;
cuModuleGetTexRef(&texRef, module, "myTexture");

// Configure the texture reference
cuTexRefSetFormat(texRef, CU_AD_FORMAT_FLOAT, 1); // 1-component float
cuTexRefSetAddress(texRef, 0, dptr, size);        // Bind device memory

CUsurfref surfRef;
cuModuleGetSurfRef(&surfRef, module, "mySurface");
```

Note: Texture and surface object APIs (using `cudaTextureObject_t` / `cudaSurfaceObject_t`) are preferred over module-level texture/surface references in modern CUDA code.

## 14.5 Kernel Execution

### 14.5.1 Launching Kernels

The Driver API uses `cuLaunchKernel` for kernel launch, which is the explicit equivalent of the `<<<>>>` syntax:

```cpp
// Setup kernel parameters
float alpha = 2.0f;
CUdeviceptr d_A, d_B, d_C;
int N = 1024;

void* params[] = {
    &alpha,
    &d_A,
    &d_B,
    &d_C,
    &N
};

CUresult res = cuLaunchKernel(
    kernel,
    gridX, gridY, gridZ,    // grid dimensions
    blockX, blockY, blockZ, // block dimensions
    0,                       // shared memory bytes
    stream,                  // CUstream (or NULL for default)
    params,                  // kernel parameters (array of pointers)
    NULL                     // extra parameters (alternative param passing)
);
```

### 14.5.2 Parameter Passing: params vs. extra

There are two ways to pass parameters to `cuLaunchKernel`:

**Method 1: `kernelParams` (pointer-to-pointer array)**

```cpp
int value1 = 42;
float value2 = 3.14f;
CUdeviceptr dptr;

void* kernelParams[] = { &value1, &value2, &dptr };
cuLaunchKernel(kernel, 1, 1, 1, 256, 1, 1, 0, 0, kernelParams, NULL);
```

Each element of `kernelParams` points to the actual argument value. The array has one entry per kernel parameter, in declaration order.

**Method 2: `extra` (buffer-based)**

```cpp
#define CU_LAUNCH_PARAM_BUFFER_POINTER ((void*)0x01)
#define CU_LAUNCH_PARAM_BUFFER_SIZE    ((void*)0x02)
#define CU_LAUNCH_PARAM_END            ((void*)0x00)

struct {
    int value1;
    float value2;
    CUdeviceptr dptr;
} args;

args.value1 = 42;
args.value2 = 3.14f;
args.dptr = dptr;

void* extra[] = {
    CU_LAUNCH_PARAM_BUFFER_POINTER, &args,
    CU_LAUNCH_PARAM_BUFFER_SIZE,    (void*)(intptr_t)sizeof(args),
    CU_LAUNCH_PARAM_END,            NULL
};

cuLaunchKernel(kernel, 1, 1, 1, 256, 1, 1, 0, 0, NULL, extra);
```

The `extra` method packs all kernel arguments into a single buffer matching the kernel's parameter layout. This avoids creating an array of pointers and can be more efficient for kernels with many parameters.

### 14.5.3 Launch Attributes

CUDA 11.0+ supports setting launch attributes via `cuLaunchKernelEx`:

```cpp
CUlaunchConfig config;
config.gridDimX = gridX;
config.gridDimY = gridY;
config.gridDimZ = gridZ;
config.blockDimX = blockX;
config.blockDimY = blockY;
config.blockDimZ = blockZ;
config.sharedMemBytes = 0;
config.hStream = stream;

// Set launch attribute: cooperative kernel
CUlaunchAttribute coopAttr;
coopAttr.id = CU_LAUNCH_ATTRIBUTE_COOPERATIVE;
coopAttr.value cooperative = { 1 };
config.numAttributes = 1;
config.attrs = &coopAttr;

cuLaunchKernelEx(&config, kernel, params, NULL);
```

Available launch attributes:

| Attribute | Description |
|-----------|-------------|
| `CU_LAUNCH_ATTRIBUTE_COOPERATIVE` | Enable cooperative kernel launch |
| `CU_LAUNCH_ATTRIBUTE_SYNCHRONOUS` | Launch synchronously (block until kernel completes) |
| `CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION` | Cluster dimensions (CC 9.0+) |
| `CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_PREFERENCE` | Cluster scheduling hint |
| `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION` | Programmatic launch serialization |
| `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT` | Programmatic event launch |

### 14.5.4 Querying Kernel Attributes

```cpp
int maxThreadsPerBlock;
cuFuncGetAttribute(&maxThreadsPerBlock,
    CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, kernel);

int sharedSizeBytes;
cuFuncGetAttribute(&sharedSizeBytes,
    CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, kernel);

int constSizeBytes;
cuFuncGetAttribute(&constSizeBytes,
    CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, kernel);

int localSizeBytes;
cuFuncGetAttribute(&localSizeBytes,
    CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, kernel);

int numRegs;
cuFuncGetAttribute(&numRegs,
    CU_FUNC_ATTRIBUTE_NUM_REGS, kernel);

int ptxVersion;
cuFuncGetAttribute(&ptxVersion,
    CU_FUNC_ATTRIBUTE_PTX_VERSION, kernel);

int binaryVersion;
cuFuncGetAttribute(&binaryVersion,
    CU_FUNC_ATTRIBUTE_BINARY_VERSION, kernel);

// Set attributes
cuFuncSetAttribute(kernel,
    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 65536);
cuFuncSetAttribute(kernel,
    CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
    CU_SHAREDMEM_CARVEOUT_MAX_SHARED);
```

### 14.5.5 Occupancy Calculation

```cpp
int blockSize = 256;
size_t dynamicSharedMemPerBlock = 0;

// Calculate min grid size for maximum occupancy
int minGridSize, bestBlockSize;
cuOccupancyMaxPotentialBlockSize(&minGridSize, &bestBlockSize,
    kernel, 0, dynamicSharedMemPerBlock, 0);
printf("Best block size: %d, min grid size: %d\n",
       bestBlockSize, minGridSize);

// Calculate occupancy for a specific configuration
float occupancy;
cuOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, kernel,
    blockSize, dynamicSharedMemPerBlock);
cuOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize,
    &bestBlockSize, kernel, sharedMemCalculator, 0);

// Or use the attribute-based calculation
int maxActiveBlocks;
cuOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks,
    kernel, blockSize, dynamicSharedMemPerBlock);

// Get SM count for the device
int smCount;
cuDeviceGetAttribute(&smCount,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);

float occupancyPct = (float)(maxActiveBlocks * blockSize) /
    (float)(smCount * maxThreadsPerSM) * 100.0f;
```

## 14.6 Memory Management

### 14.6.1 Device Memory Allocation

```cpp
// Allocate device memory
CUdeviceptr dptr;
CUresult res = cuMemAlloc(&dptr, N * sizeof(float));
if (res != CUDA_SUCCESS) {
    fprintf(stderr, "cuMemAlloc failed\n");
}

// Allocate managed (unified) memory
CUdeviceptr managedPtr;
res = cuMemAllocManaged(&managedPtr, N * sizeof(float),
                         CU_MEM_ATTACH_GLOBAL);

// Allocate pinned host memory
void* hostPtr;
res = cuMemAllocHost(&hostPtr, N * sizeof(float));

// Allocate pinned host memory with specific flags
res = cuMemHostAlloc(&hostPtr, N * sizeof(float),
                      CU_MEMHOSTALLOC_PORTABLE |
                      CU_MEMHOSTALLOC_DEVICEMAP |
                      CU_MEMHOSTALLOC_WRITECOMBINED);

// Free
cuMemFree(dptr);
cuMemFreeManaged(managedPtr);
cuMemFreeHost(hostPtr);
```

### 14.6.2 Memory Copies

```cpp
// Host to device
cuMemcpyHtoD(dptr, hostData, size);

// Device to host
cuMemcpyDtoH(hostData, dptr, size);

// Device to device
cuMemcpyDtoD(dptrDst, dptrSrc, size);

// Async copies (require a stream)
cuMemcpyHtoDAsync(dptr, hostData, size, stream);
cuMemcpyDtoHAsync(hostData, dptr, size, stream);
cuMemcpyDtoDAsync(dptrDst, dptrSrc, size, stream);

// 2D copies
CUDA_MEMCPY2D copy2d = {};
copy2d.srcMemoryType = CU_MEMORYTYPE_HOST;
copy2d.srcHost = hostData;
copy2d.srcPitch = hostPitch;
copy2d.dstMemoryType = CU_MEMORYTYPE_DEVICE;
copy2d.dstDevice = dptr;
copy2d.dstPitch = devicePitch;
copy2d.WidthInBytes = width;
copy2d.Height = height;
cuMemcpy2D(&copy2d);

// 3D copies
CUDA_MEMCPY3D copy3d = {};
// ... set up src/dst fields ...
cuMemcpy3D(&copy3d);
```

### 14.6.3 Memory Initialization

```cpp
// Set device memory to a value (like memset)
cuMemsetD8(dptr, 0xFF, N);        // Set N bytes to 0xFF
cuMemsetD16(dptr, 0xFFFF, N);     // Set N uint16_t values
cuMemsetD32(dptr, 0xDEADBEEF, N); // Set N uint32_t values

// Async variants
cuMemsetD8Async(dptr, 0xFF, N, stream);
cuMemsetD16Async(dptr, 0xFFFF, N, stream);
cuMemsetD32Async(dptr, 0xDEADBEEF, N, stream);

// 2D memset
cuMemsetD2D8(dptr, pitch, 0xFF, width, height);
cuMemsetD2D32Async(dptr, pitch, 0, width, height, stream);
```

## 14.7 Stream and Event Management

### 14.7.1 Streams

```cpp
// Create a stream
CUstream stream;
cuStreamCreate(&stream, 0);

// Create a stream with priority
CUstream highPriorityStream;
int highestPriority, lowestPriority;
cuCtxGetStreamPriorityRange(&highestPriority, &lowestPriority);
cuStreamCreateWithPriority(&highPriorityStream, 0, highestPriority);

// Non-blocking stream (does not sync with default stream)
CUstream nonBlockingStream;
cuStreamCreateWithFlags(&nonBlockingStream, CU_STREAM_NON_BLOCK);

// Synchronize
cuStreamSynchronize(stream);

// Query status
CUresult res = cuStreamQuery(stream);
if (res == CUDA_SUCCESS) {
    // Stream is idle
} else if (res == CUDA_ERROR_NOT_READY) {
    // Stream has pending work
}

// Wait for an event
cuStreamWaitEvent(stream, event, 0 /* flags */);

// Callback
cuStreamAddCallback(stream, myCallback, userData, 0 /* flags */);

// Destroy
cuStreamDestroy(stream);
```

### 14.7.2 Events

```cpp
// Create an event
CUevent event;
cuEventCreate(&event, 0);

// Create event with flags
CUevent timedEvent;
cuEventCreateWithFlags(&timedEvent, CU_EVENT_DEFAULT);

CUevent blockingEvent;
cuEventCreateWithFlags(&blockingEvent, CU_EVENT_BLOCKING_SYNC);

CUevent disabledTimingEvent;
cuEventCreateWithFlags(&disabledTimingEvent, CU_EVENT_DISABLE_TIMING);

CUevent ipcEvent;
cuEventCreateWithFlags(&ipcEvent, CU_EVENT_INTERPROCESS | CU_EVENT_DISABLE_TIMING);

// Record
cuEventRecord(event, stream);

// Synchronize
cuEventSynchronize(event);

// Query
CUresult res = cuEventQuery(event);

// Elapsed time between two events
float ms;
cuEventElapsedTime(&ms, startEvent, stopEvent);

// Destroy
cuEventDestroy(event);
```

## 14.8 Runtime/Driver API Interoperability

The Runtime API and Driver API can be used together in the same application. This is common when an application uses the Runtime API but needs to call a library or framework that uses the Driver API.

### 14.8.1 Key Interop Rules

1. **Same context**: The Runtime API uses the primary context of the current device. Driver API code should use `cuDevicePrimaryCtxRetain` to access the same context, not `cuCtxCreate`.

2. **Pointer compatibility**: A `CUdeviceptr` from the Driver API can be used as a regular device pointer with the Runtime API, and vice versa:
```cpp
// Allocate with Driver API
CUdeviceptr dptr;
cuMemAlloc(&dptr, size);

// Use with Runtime API kernel launch
myKernel<<<grid, block>>>((float*)dptr, N);  // Cast CUdeviceptr to float*

// Allocate with Runtime API
float* dptr2;
cudaMalloc(&dptr2, size);

// Use with Driver API copy
cuMemcpyHtoD((CUdeviceptr)dptr2, hostData, size);  // Cast float* to CUdeviceptr
```

3. **Runtime libraries**: Libraries like cuBLAS, cuDNN, and cuFFT use the Runtime API internally. They can be called from Driver API code without issue, as long as the correct context is current.

4. **Context management**: If you create a context with `cuCtxCreate` (not the primary context), Runtime API calls will still use the primary context. This can lead to confusing behavior. Always prefer using the primary context when mixing APIs.

### 14.8.2 Mixed API Example

```cpp
#include <cuda.h>
#include <cuda_runtime.h>

int main() {
    // Initialize driver
    cuInit(0);

    // Set device via runtime API
    cudaSetDevice(0);

    // Get the primary context via driver API
    CUcontext ctx;
    cuDevicePrimaryCtxRetain(&ctx, 0);

    // Load a module via driver API
    CUmodule module;
    cuModuleLoad(&module, "mykernel.fatbin");
    CUfunction kernel;
    cuModuleGetFunction(&kernel, module, "vecAdd");

    // Allocate via runtime API
    float *d_A, *d_B, *d_C;
    int N = 1024;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    // Launch via driver API
    void* params[] = { &d_A, &d_B, &d_C, &N };
    cuLaunchKernel(kernel, (N+255)/256, 1, 1, 256, 1, 1,
                   0, 0, params, NULL);

    // Or launch via runtime API using the same pointers
    // vecAdd<<<(N+255)/256, 256>>>(d_A, d_B, d_C, N);

    // Synchronize via runtime API
    cudaDeviceSynchronize();

    // Read back via runtime API
    float h_C[1024];
    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cuModuleUnload(module);
    cuDevicePrimaryCtxRelease(0);

    return 0;
}
```

### 14.8.3 Getting the Runtime API Device from a Driver Context

```cpp
// From a Driver API context, get the device ordinal for Runtime API
CUcontext ctx;
cuCtxGetCurrent(&ctx);

CUdevice cuDevice;
cuCtxGetDevice(&cuDevice);

// CUdevice is an ordinal (int), same as used by cudaSetDevice
int deviceOrdinal = (int)cuDevice;
cudaSetDevice(deviceOrdinal);
```

### 14.8.4 Version Compatibility

```cpp
// Check driver version
int driverVersion;
cuDriverGetVersion(&driverVersion);
// driverVersion is encoded as 1000*major + 10*minor
// e.g., 13020 means CUDA 13.2

// Check runtime version
int runtimeVersion;
cudaRuntimeGetVersion(&runtimeVersion);

// The driver version must be >= runtime version for correct operation
if (driverVersion < runtimeVersion) {
    fprintf(stderr, "Driver too old: %d < %d\n",
            driverVersion, runtimeVersion);
}
```

### 14.8.5 Function Pointer Interop

You can obtain a `CUfunction` from a `__global__` function pointer:

```cpp
// Get CUfunction from a __global__ function
// The nvcc compiler generates a stub that can be cast to CUfunction
CUfunction kernelFunc = (CUfunction)(void*)myKernel;

// Or use cuModuleGetFunction if the kernel is in a loaded module
CUfunction kernelFromModule;
cuModuleGetFunction(&kernelFromModule, module, "myKernel");
```

## 14.9 Device Property Queries

### 14.9.1 Device Attributes

```cpp
CUdevice device;
cuDeviceGet(&device, 0);

// Common attributes
int major, minor;
cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);

int multiProcessorCount;
cuDeviceGetAttribute(&multiProcessorCount,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);

int maxThreadsPerBlock;
cuDeviceGetAttribute(&maxThreadsPerBlock,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device);

int maxThreadsPerMultiProcessor;
cuDeviceGetAttribute(&maxThreadsPerMultiProcessor,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, device);

int maxSharedMemoryPerBlock;
cuDeviceGetAttribute(&maxSharedMemoryPerBlock,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, device);

int maxSharedMemoryPerMultiprocessor;
cuDeviceGetAttribute(&maxSharedMemoryPerMultiprocessor,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, device);

int totalConstantMemory;
cuDeviceGetAttribute(&totalConstantMemory,
    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, device);

int warpSize;
cuDeviceGetAttribute(&warpSize,
    CU_DEVICE_ATTRIBUTE_WARP_SIZE, device);

int maxPitch;
cuDeviceGetAttribute(&maxPitch,
    CU_DEVICE_ATTRIBUTE_MAX_PITCH, device);

int maxTexture1DWidth;
cuDeviceGetAttribute(&maxTexture1DWidth,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH, device);

int concurrentKernels;
cuDeviceGetAttribute(&concurrentKernels,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS, device);

int eccEnabled;
cuDeviceGetAttribute(&eccEnabled,
    CU_DEVICE_ATTRIBUTE_ECC_ENABLED, device);

int pciBusId;
cuDeviceGetAttribute(&pciBusId,
    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, device);

int pciDeviceId;
cuDeviceGetAttribute(&pciDeviceId,
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, device);

int pciDomainId;
cuDeviceGetAttribute(&pciDomainId,
    CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, device);

int memoryClockRate;
cuDeviceGetAttribute(&memoryClockRate,
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device);

int globalMemoryBusWidth;
cuDeviceGetAttribute(&globalMemoryBusWidth,
    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, device);

int l2CacheSize;
cuDeviceGetAttribute(&l2CacheSize,
    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, device);
```

### 14.9.2 Device Memory Info

```cpp
size_t totalMemory, freeMemory;
cuMemGetInfo(&freeMemory, &totalMemory);
printf("Device memory: %.2f GB total, %.2f GB free\n",
       (double)totalMemory / (1ULL << 30),
       (double)freeMemory / (1ULL << 30));
```

### 14.9.3 Device Name and UUID

```cpp
char name[256];
cuDeviceGetName(name, sizeof(name), device);
printf("Device name: %s\n", name);

CUuuid uuid;
cuDeviceGetUuid(&uuid, device);
printf("UUID: %02x%02x%02x%02x-...\n",
       uuid.bytes[0], uuid.bytes[1], uuid.bytes[2], uuid.bytes[3]);
```

## 14.10 Complete Driver API Example

A self-contained example showing the full Driver API workflow:

```cpp
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CU_CHECK(expr) do { \
    CUresult _res = (expr); \
    if (_res != CUDA_SUCCESS) { \
        const char* _errStr; \
        cuGetErrorString(_res, &_errStr); \
        fprintf(stderr, "CUDA Driver Error at %s:%d: %s\n", \
                __FILE__, __LINE__, _errStr); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// PTX code for a simple vector addition kernel
const char* ptxSource = R"(
.version 8.5
.target sm_80
.address_size 64

.visible .entry vecAdd(
    .param .u64 vecAdd_param_0,
    .param .u64 vecAdd_param_1,
    .param .u64 vecAdd_param_2,
    .param .u32 vecAdd_param_3
) {
    .reg .f32  %f<5>;
    .reg .b32  %r<3>;
    .reg .b64  %rd<8>;

    ld.param.u64    %rd1, [vecAdd_param_0];
    ld.param.u64    %rd2, [vecAdd_param_1];
    ld.param.u64    %rd3, [vecAdd_param_2];
    ld.param.u32    %r1, [vecAdd_param_3];

    mov.u32         %r2, %ctaid.x;
    mad.lo.s32      %r2, %r2, %ntid.x, %tid.x;
    setp.ge.s32     %p1, %r2, %r1;
    @%p1 bra        $exit;

    cvta.to.global.u64  %rd4, %rd1;
    mul.wide.u32    %rd5, %r2, 4;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f1, [%rd6];

    cvta.to.global.u64  %rd4, %rd2;
    add.s64         %rd7, %rd4, %rd5;
    ld.global.f32   %f2, [%rd7];

    add.f32         %f3, %f1, %f2;

    cvta.to.global.u64  %rd4, %rd3;
    add.s64         %rd8, %rd4, %rd5;
    st.global.f32   [%rd8], %f3;

$exit:
    ret;
}
)";

int main() {
    const int N = 1024;
    const size_t size = N * sizeof(float);

    // 1. Initialize
    CU_CHECK(cuInit(0));

    // 2. Get device
    CUdevice device;
    CU_CHECK(cuDeviceGet(&device, 0));

    // 3. Create context
    CUcontext ctx;
    CU_CHECK(cuCtxCreate(&ctx, 0, device));

    // 4. JIT compile PTX
    CUmodule module;
    CUjit_option jitOptions[2];
    void* jitOptionValues[2];
    char errorLog[8192];

    jitOptions[0] = CU_JIT_ERROR_LOG_BUFFER;
    jitOptionValues[0] = errorLog;
    jitOptions[1] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
    jitOptionValues[1] = (void*)(intptr_t)sizeof(errorLog);

    CUresult res = cuModuleLoadDataEx(&module, ptxSource, 2,
                                       jitOptions, jitOptionValues);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "PTX JIT compilation failed:\n%s\n", errorLog);
        exit(EXIT_FAILURE);
    }

    // 5. Get kernel function
    CUfunction vecAdd;
    CU_CHECK(cuModuleGetFunction(&vecAdd, module, "vecAdd"));

    // 6. Allocate memory
    CUdeviceptr d_A, d_B, d_C;
    CU_CHECK(cuMemAlloc(&d_A, size));
    CU_CHECK(cuMemAlloc(&d_B, size));
    CU_CHECK(cuMemAlloc(&d_C, size));

    // 7. Initialize host data and copy to device
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        h_A[i] = (float)i;
        h_B[i] = (float)(i * 2);
    }

    CU_CHECK(cuMemcpyHtoD(d_A, h_A, size));
    CU_CHECK(cuMemcpyHtoD(d_B, h_B, size));

    // 8. Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    void* params[] = { &d_A, &d_B, &d_C, &N };
    CU_CHECK(cuLaunchKernel(vecAdd,
                             blocksPerGrid, 1, 1,
                             threadsPerBlock, 1, 1,
                             0, NULL, params, NULL));

    // 9. Copy result back
    CU_CHECK(cuMemcpyDtoH(h_C, d_C, size));

    // 10. Verify
    for (int i = 0; i < N; i++) {
        float expected = h_A[i] + h_B[i];
        if (fabsf(h_C[i] - expected) > 1e-5f) {
            fprintf(stderr, "Mismatch at index %d: got %f, expected %f\n",
                    i, h_C[i], expected);
        }
    }
    printf("Verification passed!\n");

    // 11. Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    CU_CHECK(cuMemFree(d_A));
    CU_CHECK(cuMemFree(d_B));
    CU_CHECK(cuMemFree(d_C));
    CU_CHECK(cuModuleUnload(module));
    CU_CHECK(cuCtxDestroy(ctx));

    return 0;
}
```

## 14.11 Best Practices for Driver API Usage

1. **Prefer the primary context** when mixing with Runtime API code. Use `cuDevicePrimaryCtxRetain` instead of `cuCtxCreate`.

2. **Always check `CUresult`** -- The Driver API does not have an equivalent of `cudaGetLastError()`. Every call must be checked.

3. **Use JIT compilation for portability** -- Shipping PTX and compiling at runtime allows your application to run on future GPU architectures.

4. **Set appropriate JIT options** -- Use `CU_JIT_MAX_REGISTERS` to control register usage, and `CU_JIT_ERROR_LOG_BUFFER` to capture compilation errors.

5. **Cache compiled modules** -- JIT compilation has overhead. Cache the resulting cubin with `cuLinkComplete` and save it to disk for subsequent runs.

6. **Use `cuMemAllocManaged`** for simplified memory management when porting Runtime API code to Driver API.

7. **Prefer `kernelParams` over `extra`** for parameter passing, as it is simpler and less error-prone.

8. **Destroy resources in reverse order** -- Contexts should outlive all resources created within them.

9. **Version-check the driver** -- Use `cuDriverGetVersion` to ensure the installed driver supports the features your application needs.

10. **Minimize context switches** -- Pushing and popping contexts has overhead. Group operations by context when possible.
