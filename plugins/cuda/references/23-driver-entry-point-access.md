# 23. Driver Entry Point Access

CUDA provides a mechanism for applications to retrieve the addresses of CUDA driver functions at runtime, similar to how `dlsym()` works on POSIX systems or `GetProcAddress()` on Windows. This capability, introduced in CUDA 11.3, enables dynamic resolution of driver API symbols with version-aware semantics, which is essential for building forward-compatible applications, plugin architectures, and frameworks that need to adapt to the available driver version at runtime.

---

## Table of Contents

1. [Introduction](#231-introduction)
2. [Function Typedefs](#232-function-typedefs)
3. [Retrieval APIs](#233-retrieval-apis)
4. [Per-Thread Default Stream](#234-per-thread-default-stream)
5. [Error Codes](#235-error-codes)
6. [Versioning Pitfalls](#236-versioning-pitfalls)
7. [Practical Examples](#237-practical-examples)

---

## 23.1 Introduction

The Driver Entry Point Access API addresses several important use cases:

**Why dynamic symbol resolution matters:**

- **Forward compatibility:** Applications can query whether a newer driver API is available before using it, enabling graceful degradation on older drivers.
- **Plugin systems:** Frameworks that load CUDA-using plugins at runtime can resolve driver symbols without linking against a specific CUDA version.
- **Version-aware dispatch:** Applications can retrieve a specific version of an API function, ensuring ABI compatibility across CUDA minor version updates.
- **Reduced dependencies:** Tools and profilers can access CUDA driver functions without requiring the full CUDA toolkit at link time.

The API is analogous to standard dynamic linking patterns:

| Platform | Standard Mechanism | CUDA Equivalent |
|----------|-------------------|-----------------|
| Linux/macOS | `dlsym()` | `cuGetProcAddress()` / `cudaGetDriverEntryPoint()` |
| Windows | `GetProcAddress()` | `cuGetProcAddress()` / `cudaGetDriverEntryPoint()` |

**Availability:** CUDA 11.3 and later for the Driver API variant; CUDA 11.3 and later for the Runtime API wrapper.

---

## 23.2 Function Typedefs

The CUDA SDK provides pre-defined function pointer typedefs for every driver API function. These typedefs encode the API version in their name, making it straightforward to declare correctly-typed function pointers.

### 23.2.1 Header Location

The typedefs are declared in the CUDA Driver API headers:

```
cudaTypedefs.h    -- function pointer typedefs for cuda.h symbols
```

This header is included automatically when you include `cuda.h` (Driver API) in CUDA 11.3+.

### 23.2.2 Naming Convention

Function pointer typedefs follow the convention:

```
PFN_<symbol>_<version>
```

Where:
- `PFN` = Pointer to Function
- `<symbol>` = The driver API function name (e.g., `cuStreamBeginCapture`)
- `<version>` = The API version the typedef corresponds to (e.g., `v10000` for CUDA 10.0, `v11030` for CUDA 11.3)

```cpp
// Example typedefs from cudaTypedefs.h
typedef CUresult (CUDAAPI *PFN_cuStreamBeginCapture_v10000)(
    CUstream hStream, CUstreamCaptureMode mode);

typedef CUresult (CUDAAPI *PFN_cuMemAlloc_v10000)(
    CUdeviceptr *dptr, size_t bytesize);

typedef CUresult (CUDAAPI *PFN_cuLaunchKernel_v10000)(
    CUfunction f,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    CUstream hStream,
    void **kernelParams,
    void **extra);
```

### 23.2.3 Using Typedefs

```cpp
#include <cuda.h>
// cudaTypedefs.h is included automatically

// Declare function pointer using the typedef
PFN_cuStreamBeginCapture_v10000 pfnStreamBeginCapture = nullptr;

// The typedef ensures correct parameter types and calling convention
// Any mismatch in the signature would be a compile-time error
```

### 23.2.4 Version Selection

Multiple typedefs may exist for the same function if its signature changed across CUDA versions:

```cpp
// If a function's signature changed in CUDA 12.0:
// Old version (CUDA 10.0 - 11.x)
typedef CUresult (CUDAAPI *PFN_cuSomeFunction_v10000)(
    CUstream stream, int param);

// New version (CUDA 12.0+)
typedef CUresult (CUDAAPI *PFN_cuSomeFunction_v12000)(
    CUstream stream, int param, int newParam);

// Application selects the version it was designed for
PFN_cuSomeFunction_v10000 pfnOld = nullptr;
PFN_cuSomeFunction_v12000 pfnNew = nullptr;
```

---

## 23.3 Retrieval APIs

CUDA provides two levels of API for retrieving driver function pointers: the Driver API version and the Runtime API wrapper.

### 23.3.1 Driver API: cuGetProcAddress

The primary function for retrieving driver entry points. It performs a version-aware symbol lookup.

```cpp
CUresult CUDAAPI cuGetProcAddress(
    const char* symbol,       // [in]  Null-terminated symbol name
    void** pfn,               // [out] Address of the function pointer
    cuuint64_t version,       // [in]  Requested API version (e.g., 11030)
    uint64_t flags,           // [in]  Option flags
    CUdriverProcAddress* driverProcAddress  // [out] Status and attributes
);
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `symbol` | The null-terminated name of the driver API function (e.g., `"cuStreamBeginCapture"`) |
| `pfn` | Output pointer where the function address will be stored |
| `version` | The CUDA API version the caller expects, encoded as `X * 1000 + Y * 10` (e.g., `11030` for CUDA 11.3, `12000` for CUDA 12.0) |
| `flags` | Option flags (see Section 23.4 for per-thread default stream flags) |
| `driverProcAddress` | Output structure containing the retrieval status and additional attributes |

**Version encoding:** The version number follows the pattern `XXYY0` where `XX` is the major version and `YY` is the minor version. For example:
- CUDA 10.0 = `10000`
- CUDA 11.0 = `11000`
- CUDA 11.3 = `11030`
- CUDA 12.0 = `12000`
- CUDA 12.3 = `12030`

```cpp
#include <cuda.h>
#include <stdio.h>

void retrieve_stream_capture() {
    // Declare the function pointer using the typedef
    PFN_cuStreamBeginCapture_v10000 pfnStreamBeginCapture = nullptr;

    // Prepare the output structure
    CUdriverProcAddress procAddr = {};

    // Retrieve the function address
    // Request version 10.0 (10000) compatibility
    CUresult result = cuGetProcAddress(
        "cuStreamBeginCapture",
        (void**)&pfnStreamBeginCapture,
        10000,                              // version: CUDA 10.0
        CU_GET_PROC_ADDRESS_DEFAULT,        // flags
        &procAddr                           // output status
    );

    if (result == CUDA_SUCCESS &&
        procAddr.status == CU_GET_PROC_ADDRESS_SUCCESS) {
        printf("Successfully retrieved cuStreamBeginCapture\n");

        // Use the function pointer
        CUstream stream;
        cuStreamCreate(&stream, 0);
        pfnStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_GLOBAL);
    } else {
        printf("Failed to retrieve cuStreamBeginCapture: %d (status: %d)\n",
               result, procAddr.status);
    }
}
```

### 23.3.2 Runtime API: cudaGetDriverEntryPoint

A convenience wrapper that calls `cuGetProcAddress()` internally. This is useful for applications that primarily use the Runtime API but need access to specific driver functions.

```cpp
cudaError_t CUDARTAPI cudaGetDriverEntryPoint(
    const char* symbol,       // [in]  Symbol name
    void** pfn,               // [out] Function pointer address
    uint64_t flags            // [in]  Option flags
);
```

```cpp
#include <cuda_runtime.h>
#include <stdio.h>

void use_runtime_entry_point() {
    void* funcPtr = nullptr;

    cudaError_t err = cudaGetDriverEntryPoint(
        "cuStreamBeginCapture",
        &funcPtr,
        cudaEnableDefaultStream
    );

    if (err == cudaSuccess) {
        // Cast to the appropriate function pointer type
        PFN_cuStreamBeginCapture_v10000 pfn =
            (PFN_cuStreamBeginCapture_v10000)funcPtr;
        printf("Got function pointer at %p\n", (void*)pfn);
    }
}
```

**Note:** `cudaGetDriverEntryPoint` uses the CUDA version of the runtime the application was compiled against as the version parameter. This means it always requests the version matching the static runtime.

### 23.3.3 Runtime API: cudaGetDriverEntryPointByVersion

An extended version that allows specifying the API version explicitly, similar to the `cuGetProcAddress` Driver API function.

```cpp
cudaError_t CUDARTAPI cudaGetDriverEntryPointByVersion(
    const char* symbol,       // [in]  Symbol name
    void** pfn,               // [out] Function pointer address
    cuuint64_t version,       // [in]  Requested API version
    uint64_t flags            // [in]  Option flags
);
```

```cpp
#include <cuda_runtime.h>
#include <stdio.h>

void use_versioned_entry_point() {
    void* funcPtr = nullptr;

    // Request the CUDA 11.0 version of the API
    cudaError_t err = cudaGetDriverEntryPointByVersion(
        "cuStreamBeginCapture",
        &funcPtr,
        11000,                              // version: CUDA 11.0
        cudaEnableDefaultStream
    );

    if (err == cudaSuccess) {
        PFN_cuStreamBeginCapture_v10000 pfn =
            (PFN_cuStreamBeginCapture_v10000)funcPtr;

        // Use the function pointer...
    } else {
        fprintf(stderr, "Symbol not available: %s\n",
                cudaGetErrorString(err));
    }
}
```

### 23.3.4 CUdriverProcAddress Structure

The `CUdriverProcAddress` structure returned by `cuGetProcAddress()` provides additional information about the symbol lookup:

```cpp
typedef struct CUdriverProcAddress_st {
    CUdriverProcAddressQueryResult status;
    // Reserved for future attributes
} CUdriverProcAddress;
```

The `status` field indicates the outcome of the lookup (see Section 23.5 for error codes).

### 23.3.5 Flags

| Flag | Value | Description |
|------|-------|-------------|
| `CU_GET_PROC_ADDRESS_DEFAULT` | `0` | Default behavior: resolve symbol with the legacy default stream variant |
| `CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM` | `1` | Resolve the per-thread default stream variant of the symbol |

---

## 23.4 Per-Thread Default Stream

CUDA's per-thread default stream feature changes how the implicit default stream behaves. By default, all host threads share a single legacy default stream that synchronizes with all other streams. With per-thread default stream, each host thread gets its own default stream that does NOT synchronize with other streams.

### 23.4.1 Enabling Per-Thread Default Stream

There are three ways to enable per-thread default stream:

```bash
# Method 1: Environment variable
export CUDA_API_PER_THREAD_DEFAULT_STREAM=1
```

```cpp
// Method 2: Compiler flag (nvcc)
// nvcc --default-stream per-thread myapp.cu -o myapp
```

```cpp
// Method 3: Define before including CUDA headers
#define CUDA_API_PER_THREAD_DEFAULT_STREAM 1
#include <cuda_runtime.h>
```

### 23.4.2 Driver API Suffixes

When using the Driver API with per-thread default stream, functions that operate on the default stream have suffixed variants:

| Suffix | Meaning | Example |
|--------|---------|---------|
| `_ptsz` | Per-Thread Stream | `cuLaunchKernel_ptsz` |
| `_ptds` | Per-Thread Device Synchronize | `cuDeviceSynchronize_ptds` |

The `_ptsz` suffix is used for functions that accept an implicit stream parameter, while `_ptds` is used for functions that perform device-wide synchronization.

### 23.4.3 Resolving Per-Thread Variants via cuGetProcAddress

When resolving driver symbols with `cuGetProcAddress()`, pass the `CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM` flag to retrieve the per-thread variant:

```cpp
#include <cuda.h>
#include <stdio.h>

void resolve_per_thread_variant() {
    // Resolve the legacy (default) variant
    PFN_cuLaunchKernel_v10000 pfnLegacy = nullptr;
    CUdriverProcAddress legacyStatus = {};
    cuGetProcAddress("cuLaunchKernel",
        (void**)&pfnLegacy,
        10000,
        CU_GET_PROC_ADDRESS_DEFAULT,
        &legacyStatus);

    // Resolve the per-thread default stream variant
    PFN_cuLaunchKernel_v10000 pfnPerThread = nullptr;
    CUdriverProcAddress ptStatus = {};
    cuGetProcAddress("cuLaunchKernel",
        (void**)&pfnPerThread,
        10000,
        CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM,
        &ptStatus);

    printf("Legacy:   %p (status: %d)\n",
           (void*)pfnLegacy, legacyStatus.status);
    printf("Per-thread: %p (status: %d)\n",
           (void*)pfnPerThread, ptStatus.status);

    // Both pointers may be different or the same, depending on the
    // driver implementation. The important thing is that the
    // per-thread variant respects the per-thread default stream
    // semantics when called.
}
```

### 23.4.4 Practical Implications

```cpp
// With per-thread default stream enabled:
// Thread A's default-stream work can overlap with Thread B's
// default-stream work.

#include <cuda_runtime.h>
#include <thread>

void thread_work(int deviceId) {
    cudaSetDevice(deviceId);

    // This kernel uses the per-thread default stream
    // It will NOT synchronize with other threads' default stream work
    kernel_a<<<grid, block>>>();

    // Explicit synchronization only within this thread's work
    cudaStreamSynchronize(0);  // 0 = current thread's default stream
}

int main() {
    std::thread t1(thread_work, 0);
    std::thread t2(thread_work, 0);

    t1.join();
    t2.join();
    return 0;
}
```

---

## 23.5 Error Codes

Symbol lookup can fail for several reasons. The `CUdriverProcAddress::status` field and the return value of `cuGetProcAddress()` provide detailed error information.

### 23.5.1 CUdriverProcAddressQueryResult Values

| Status Code | Value | Description |
|-------------|-------|-------------|
| `CU_GET_PROC_ADDRESS_SUCCESS` | `0` | The symbol was found and the requested version is supported. The function pointer in `pfn` is valid. |
| `CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT` | `1` | The symbol was found but the driver's version is older than the requested version. The function pointer may be `NULL` or point to a function with a different (older) signature. |
| `CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND` | `2` | The symbol was not found at all. The function pointer is `NULL`. |

### 23.5.2 Handling Lookup Results

```cpp
CUresult resolve_with_error_handling(const char* symbolName) {
    void* funcPtr = nullptr;
    CUdriverProcAddress procAddr = {};

    CUresult result = cuGetProcAddress(
        symbolName,
        &funcPtr,
        12000,  // Request CUDA 12.0 version
        CU_GET_PROC_ADDRESS_DEFAULT,
        &procAddr
    );

    // Check the cuGetProcAddress return value first
    if (result != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(result, &errStr);
        fprintf(stderr, "cuGetProcAddress failed: %s\n", errStr);
        return result;
    }

    // Then check the detailed status
    switch (procAddr.status) {
    case CU_GET_PROC_ADDRESS_SUCCESS:
        printf("Symbol '%s' resolved successfully at %p\n",
               symbolName, funcPtr);
        break;

    case CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT:
        fprintf(stderr,
            "Symbol '%s' found but driver version is older than "
            "requested (CUDA 12.0). Consider using an older "
            "API version or updating the driver.\n",
            symbolName);
        // funcPtr may still be usable for the older version's signature
        break;

    case CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND:
        fprintf(stderr,
            "Symbol '%s' not found in the driver. The function "
            "may not exist in this CUDA version.\n",
            symbolName);
        break;
    }

    return result;
}
```

### 23.5.3 Common Failure Scenarios

| Scenario | Result | Status | Resolution |
|----------|--------|--------|------------|
| Symbol exists, version matches | `CUDA_SUCCESS` | `SUCCESS` | Use the function pointer |
| Symbol exists, driver is older | `CUDA_SUCCESS` | `VERSION_NOT_SUFFICIENT` | Downgrade requested version or update driver |
| Symbol does not exist | `CUDA_SUCCESS` | `SYMBOL_NOT_FOUND` | Feature not available; use fallback path |
| Invalid symbol name | `CUDA_ERROR_INVALID_VALUE` | N/A | Fix the symbol name string |
| NULL pfn pointer | `CUDA_ERROR_INVALID_VALUE` | N/A | Provide a valid output pointer |

**Important:** Even when `cuGetProcAddress()` returns `CUDA_SUCCESS`, you must still check `procAddr.status` to determine whether the symbol was actually resolved successfully.

---

## 23.6 Versioning Pitfalls

Using the Driver Entry Point Access API requires careful attention to versioning to avoid subtle bugs.

### 23.6.1 ABI Mismatches Across Minor Versions

CUDA maintains binary compatibility within a major version (e.g., all CUDA 11.x versions). However, individual API functions may have their signatures extended in minor version updates.

```cpp
// CUDA 11.0 version: 2 parameters
typedef CUresult (CUDAAPI *PFN_cuSomeFunction_v11000)(
    CUstream stream, int param);

// CUDA 11.3 version: 3 parameters (added a new parameter)
typedef CUresult (CUDAAPI *PFN_cuSomeFunction_v11030)(
    CUstream stream, int param, int newParam);

// If you request version 11000 but the driver only has 11030,
// the returned function pointer is the 11000-compatible version.
// It is safe to call with 2 parameters.

// DANGER: If you request version 11030 but use the v11000 typedef,
// you may pass the wrong number of parameters, causing undefined behavior.
PFN_cuSomeFunction_v11000 pfn = nullptr;
CUdriverProcAddress procAddr = {};
cuGetProcAddress("cuSomeFunction", (void**)&pfn,
    11030,  // Request 11.3 version!
    CU_GET_PROC_ADDRESS_DEFAULT, &procAddr);

// BUG: pfn expects 2 params but the actual function expects 3
pfn(stream, param);  // Undefined behavior!
```

**Rule:** Always match the version number in `cuGetProcAddress()` with the version suffix of the typedef.

### 23.6.2 Static Runtime Version vs. Driver Version

The CUDA Runtime API is statically linked by default, which means the runtime version is fixed at compile time. However, the driver version is determined by the installed NVIDIA driver on the target system.

```cpp
// The static runtime version determines which version
// cudaGetDriverEntryPoint() uses implicitly.
// cudaGetDriverEntryPointByVersion() lets you override this.

// Check runtime version
int runtimeVersion;
cudaRuntimeGetVersion(&runtimeVersion);
printf("Runtime version: %d\n", runtimeVersion);  // e.g., 12030

// Check driver version
int driverVersion;
cudaDriverGetVersion(&driverVersion);
printf("Driver version: %d\n", driverVersion);  // e.g., 12000

// If driver < runtime, some features may not be available
if (driverVersion < runtimeVersion) {
    printf("Warning: Driver older than runtime. "
           "Some features may be unavailable.\n");
}
```

**Rule:** Always check the driver version before using newly introduced APIs. Use `cuGetProcAddress()` with the appropriate version and check `procAddr.status` for `VERSION_NOT_SUFFICIENT`.

### 23.6.3 API Version Bumps May Change Signatures

When an API function's signature changes between CUDA versions, both the old and new versions are typically available in the driver. The version parameter in `cuGetProcAddress()` determines which variant you get.

```cpp
// Safe pattern: Request the version matching your typedef,
// and handle the case where it is not available.

PFN_cuStreamBeginCapture_v10000 pfnCapture = nullptr;
CUdriverProcAddress procAddr = {};
CUresult res = cuGetProcAddress(
    "cuStreamBeginCapture",
    (void**)&pfnCapture,
    10000,  // Matches PFN_..._v10000 typedef
    CU_GET_PROC_ADDRESS_DEFAULT,
    &procAddr
);

if (res != CUDA_SUCCESS ||
    procAddr.status != CU_GET_PROC_ADDRESS_SUCCESS) {
    // cuStreamBeginCapture not available; use fallback
    fprintf(stderr, "Stream capture not supported\n");
    return;
}

// Safe to call with the v10000 signature
pfnCapture(stream, CU_STREAM_CAPTURE_MODE_GLOBAL);
```

### 23.6.4 Null Pointer Checks

Always verify that the returned function pointer is non-NULL before calling it, even when `procAddr.status` reports success:

```cpp
if (pfnCapture != nullptr) {
    pfnCapture(stream, CU_STREAM_CAPTURE_MODE_GLOBAL);
} else {
    // Unexpected: status was SUCCESS but pointer is NULL
    fprintf(stderr, "Unexpected NULL function pointer\n");
}
```

---

## 23.7 Practical Examples

### 23.7.1 Building a Version-Adaptive CUDA Plugin Loader

A plugin framework that dynamically loads CUDA functions based on the available driver version:

```cpp
#include <cuda.h>
#include <stdio.h>
#include <string.h>

// Plugin interface
typedef void (*KernelLaunchFunc)(CUfunction, dim3, dim3,
    unsigned int, CUstream, void**);

class CudaPluginLoader {
private:
    int driverVersion;

public:
    CudaPluginLoader() {
        cuDriverGetVersion(&driverVersion);
        printf("CUDA Driver version: %d.%d\n",
               driverVersion / 1000,
               (driverVersion % 1000) / 10);
    }

    // Try to retrieve the newest available version of a function
    template<typename FuncPtr>
    bool getFunction(const char* name, FuncPtr* pfn,
                     int minVersion, int maxVersion) {
        // Try versions from newest to oldest
        for (int ver = maxVersion; ver >= minVersion; ver -= 10) {
            CUdriverProcAddress procAddr = {};
            CUresult res = cuGetProcAddress(
                name, (void**)pfn, ver,
                CU_GET_PROC_ADDRESS_DEFAULT, &procAddr);

            if (res == CUDA_SUCCESS &&
                procAddr.status == CU_GET_PROC_ADDRESS_SUCCESS &&
                *pfn != nullptr) {
                printf("  Resolved '%s' at version %d\n", name, ver);
                return true;
            }
        }

        fprintf(stderr, "  Could not resolve '%s' "
                "(need version %d-%d)\n", name, minVersion, maxVersion);
        return false;
    }
};

// Usage
int main() {
    cuInit(0);

    CudaPluginLoader loader;

    using LaunchKernelFn = CUresult (*)(CUfunction,
        unsigned int, unsigned int, unsigned int,
        unsigned int, unsigned int, unsigned int,
        unsigned int, CUstream, void**, void**);

    PFN_cuLaunchKernel_v10000 pfnLaunch = nullptr;
    loader.getFunction("cuLaunchKernel", &pfnLaunch,
                       10000,  // min: CUDA 10.0
                       12030); // max: CUDA 12.3

    if (pfnLaunch) {
        // Use the resolved function
        CUstream stream;
        cuStreamCreate(&stream, 0);
        // pfnLaunch(f, gx, gy, gz, bx, by, bz, smem, stream, params, extra);
    }

    return 0;
}
```

### 23.7.2 Feature Detection Pattern

Use `cuGetProcAddress()` to detect whether optional CUDA features are available:

```cpp
#include <cuda.h>
#include <stdio.h>

bool has_cuda_graphs() {
    // CUDA Graphs were introduced in CUDA 10.0
    void* pfn = nullptr;
    CUdriverProcAddress procAddr = {};
    CUresult res = cuGetProcAddress(
        "cuGraphCreate",
        &pfn, 10000,
        CU_GET_PROC_ADDRESS_DEFAULT, &procAddr);
    return (res == CUDA_SUCCESS &&
            procAddr.status == CU_GET_PROC_ADDRESS_SUCCESS &&
            pfn != nullptr);
}

bool has_cuda_unified_addressing() {
    // Check for virtual memory management (CUDA 10.2+)
    void* pfn = nullptr;
    CUdriverProcAddress procAddr = {};
    CUresult res = cuGetProcAddress(
        "cuMemAddressReserve",
        &pfn, 10020,
        CU_GET_PROC_ADDRESS_DEFAULT, &procAddr);
    return (res == CUDA_SUCCESS &&
            procAddr.status == CU_GET_PROC_ADDRESS_SUCCESS &&
            pfn != nullptr);
}

bool has_cuda_cdp_v2() {
    // CDP v2 was introduced in CUDA 12.3 for Hopper+
    void* pfn = nullptr;
    CUdriverProcAddress procAddr = {};
    CUresult res = cuGetProcAddress(
        "cuLaunchKernelEx",
        &pfn, 12030,
        CU_GET_PROC_ADDRESS_DEFAULT, &procAddr);
    return (res == CUDA_SUCCESS &&
            procAddr.status == CU_GET_PROC_ADDRESS_SUCCESS &&
            pfn != nullptr);
}

int main() {
    cuInit(0);

    printf("CUDA Graphs: %s\n",
           has_cuda_graphs() ? "available" : "not available");
    printf("Virtual Memory Management: %s\n",
           has_cuda_unified_addressing() ? "available" : "not available");
    printf("CDP v2: %s\n",
           has_cuda_cdp_v2() ? "available" : "not available");

    return 0;
}
```

### 23.7.3 Safe Multi-Version Dispatch

A pattern for dispatching to the best available API version:

```cpp
#include <cuda.h>
#include <stdio.h>

// Three versions of a hypothetical API
typedef CUresult (*CuFuncV100)(CUstream, int);
typedef CUresult (*CuFuncV110)(CUstream, int, int);
typedef CUresult (*CuFuncV120)(CUstream, int, int, int);

void do_work(CUstream stream) {
    CUdriverProcAddress procAddr = {};
    CUresult res;

    // Try the newest version first
    void* pfnV120 = nullptr;
    res = cuGetProcAddress("cuDoSomething", &pfnV120,
        12000, CU_GET_PROC_ADDRESS_DEFAULT, &procAddr);
    if (res == CUDA_SUCCESS &&
        procAddr.status == CU_GET_PROC_ADDRESS_SUCCESS &&
        pfnV120 != nullptr) {
        CuFuncV120 fn = (CuFuncV120)pfnV120;
        fn(stream, 1, 2, 3);
        return;
    }

    // Fall back to v11
    void* pfnV110 = nullptr;
    res = cuGetProcAddress("cuDoSomething", &pfnV110,
        11000, CU_GET_PROC_ADDRESS_DEFAULT, &procAddr);
    if (res == CUDA_SUCCESS &&
        procAddr.status == CU_GET_PROC_ADDRESS_SUCCESS &&
        pfnV110 != nullptr) {
        CuFuncV110 fn = (CuFuncV110)pfnV110;
        fn(stream, 1, 2);
        return;
    }

    // Fall back to oldest supported version
    void* pfnV100 = nullptr;
    res = cuGetProcAddress("cuDoSomething", &pfnV100,
        10000, CU_GET_PROC_ADDRESS_DEFAULT, &procAddr);
    if (res == CUDA_SUCCESS &&
        procAddr.status == CU_GET_PROC_ADDRESS_SUCCESS &&
        pfnV100 != nullptr) {
        CuFuncV100 fn = (CuFuncV100)pfnV100;
        fn(stream, 1);
        return;
    }

    fprintf(stderr, "cuDoSomething not available on this driver\n");
}
```

### 23.7.4 Integrating with Dynamic Loading

For applications that dynamically load the CUDA driver library (libcuda.so / nvcuda.dll) rather than linking against it:

```cpp
#ifdef _WIN32
#include <windows.h>
#define dlopen(name) LoadLibrary(name)
#define dlsym(handle, name) GetProcAddress(handle, name)
#define dlclose(handle) FreeLibrary(handle)
typedef HMODULE DynLibHandle;
#else
#include <dlfcn.h>
#define dlopen(name) dlopen(name, RTLD_LAZY)
#define dlsym dlsym
#define dlclose dlclose
typedef void* DynLibHandle;
#endif

// Dynamically load the CUDA driver and resolve cuGetProcAddress
int dynamic_cuda_init() {
#ifdef _WIN32
    DynLibHandle handle = dlopen("nvcuda.dll");
#else
    DynLibHandle handle = dlopen("libcuda.so.1");
#endif

    if (!handle) {
        fprintf(stderr, "Failed to load CUDA driver library\n");
        return -1;
    }

    // Resolve cuGetProcAddress itself
    typedef CUresult (*CuGetProcAddressFn)(
        const char*, void**, cuuint64_t, uint64_t,
        CUdriverProcAddress*);

    CuGetProcAddressFn pfnGetProcAddr =
        (CuGetProcAddressFn)dlsym(handle, "cuGetProcAddress");

    if (!pfnGetProcAddr) {
        fprintf(stderr, "cuGetProcAddress not found; "
                "driver may be too old (need CUDA 11.3+)\n");
        dlclose(handle);
        return -1;
    }

    // Now resolve any driver function through pfnGetProcAddr
    CUdriverProcAddress procAddr = {};
    PFN_cuStreamCreate_v10000 pfnStreamCreate = nullptr;

    CUresult res = pfnGetProcAddr(
        "cuStreamCreate",
        (void**)&pfnStreamCreate,
        10000, CU_GET_PROC_ADDRESS_DEFAULT, &procAddr);

    if (res == CUDA_SUCCESS &&
        procAddr.status == CU_GET_PROC_ADDRESS_SUCCESS) {
        CUstream stream;
        pfnStreamCreate(&stream, 0);
        printf("Stream created via dynamically resolved API\n");
        cuStreamDestroy_v10000 pfnDestroy = nullptr;
        // ... resolve and call cuStreamDestroy
    }

    return 0;
}
```
