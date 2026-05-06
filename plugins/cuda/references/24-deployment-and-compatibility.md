# 24. Deployment and Compatibility

Deploying CUDA applications requires understanding versioning, binary compatibility, runtime redistribution, and the tools available for GPU management. This document covers CUDA toolkit versioning semantics, compatibility models that allow applications to run across driver and toolkit versions, build best practices for maximum compatibility, error handling patterns, and management tools (nvidia-smi and NVML).

---

## Table of Contents

1. [CUDA Toolkit Versioning](#241-cuda-toolkit-versioning)
2. [Compatibility Choices](#242-compatibility-choices)
3. [Building for Maximum Compatibility](#243-building-for-maximum-compatibility)
4. [Runtime Redistribution](#244-runtime-redistribution)
5. [Error Handling](#245-error-handling)
6. [nvidia-smi](#246-nvidia-smi)
7. [NVML](#247-nvml)
8. [Testing CUDA Availability](#248-testing-cuda-availability)

---

## 24.1 CUDA Toolkit Versioning (CUDA 11+)

Starting with CUDA 11.0, the CUDA Toolkit follows semantic versioning principles that define how versions interact and what compatibility guarantees they provide.

### 24.1.1 Version Format

CUDA uses a three-part version number: **X.Y.Z**

| Component | Name | Meaning |
|-----------|------|---------|
| **X** | Major version | Incremented for breaking API changes; binary compatibility may be broken. A new major version may require a newer NVIDIA driver. |
| **Y** | Minor version | Incremented for new features, new APIs, and deprecated APIs. Binary compatibility is **maintained** within the same major version. |
| **Z** | Patch version | Bug fixes, performance improvements, and documentation updates. No API changes. Fully compatible within the same X.Y. |

```cpp
// Query version at runtime
int driverVersion, runtimeVersion;
cudaDriverGetVersion(&driverVersion);
cudaRuntimeGetVersion(&runtimeVersion);

printf("Driver version:  %d (CUDA %d.%d)\n",
       driverVersion, driverVersion / 1000,
       (driverVersion % 1000) / 10);
printf("Runtime version:  %d (CUDA %d.%d)\n",
       runtimeVersion, runtimeVersion / 1000,
       (runtimeVersion % 1000) / 10);
```

### 24.1.2 Version Encoding

CUDA versions are encoded as integers for API purposes:

| CUDA Version | Encoded Value |
|-------------|---------------|
| CUDA 10.0 | 10000 |
| CUDA 10.1 | 10010 |
| CUDA 10.2 | 10020 |
| CUDA 11.0 | 11000 |
| CUDA 11.1 | 11010 |
| CUDA 11.2 | 11020 |
| CUDA 11.3 | 11030 |
| CUDA 12.0 | 12000 |
| CUDA 12.3 | 12030 |

The encoding formula is: `X * 1000 + Y * 10`

### 24.1.3 Version Compatibility Rules

| Relationship | Binary Compatible? | Source Compatible? |
|-------------|-------------------|-------------------|
| Same X.Y.Z | Yes | Yes |
| Different Z, same X.Y | Yes | Yes |
| Different Y, same X | Yes (within same major) | Mostly (new APIs not available in older) |
| Different X | Not guaranteed | No (breaking changes) |

**Key principle:** Within a major version (e.g., CUDA 11.x), all minor versions are binary-compatible. An application compiled with CUDA 11.0 can run on a system with the CUDA 11.8 runtime installed, and vice versa.

---

## 24.2 Compatibility Choices

CUDA provides multiple compatibility mechanisms that address different deployment scenarios. Understanding these helps choose the right strategy for your application.

### 24.2.1 Forward Compatible Upgrade (CUDA 10+)

Forward Compatible Upgrade allows CUDA applications built with a **newer** CUDA Toolkit to run on systems with an **older** NVIDIA driver. This is the most powerful compatibility mode because it decouples application updates from driver updates.

**How it works:**
- The CUDA compatibility library (`libcuda-compat.so` on Linux) acts as an interposition layer between the application and the installed driver.
- It translates newer CUDA API calls into operations the older driver can understand.
- The compatibility library is bundled with the CUDA Toolkit or the CUDA container.

**Requirements:**
- The hardware must be supported by the installed driver (even if the driver's CUDA version is older).
- The compatibility library must be from the same or newer CUDA version as the application.
- Available on Linux only.

```bash
# Example: Application built with CUDA 12.3 on a system with
# driver supporting CUDA 11.8

# The compatibility library intercepts CUDA calls
export LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH

# Verify the compatibility library is loaded
ldd myapp | grep cuda-compat
```

**Limitations:**
- Not all newer features can be emulated on older drivers. Features that require new hardware capabilities (e.g., new instructions, new hardware units) cannot be forward-compatible.
- There may be performance overhead from the translation layer.
- Some APIs may return `cudaErrorNotSupported` if the older driver cannot fulfill the request.

### 24.2.2 Enhanced Compatibility (CUDA 11.1+)

Enhanced Compatibility guarantees that an application built with a specific CUDA minor version will work with **all future** minor versions within the same major version, without recompilation.

**How it works:**
- When an application is compiled with CUDA 11.1+, the compiler embeds version metadata and compatibility markers into the binary.
- The runtime and driver negotiate the highest mutually supported version.
- API additions in future minor versions are available through dynamic symbol resolution (see Chapter 23: Driver Entry Point Access).

**Example:**

```cpp
// Application built with CUDA 11.1
// It can run on systems with CUDA 11.2, 11.3, ..., 11.8 drivers
// without recompilation

// To use features from newer minor versions, use dynamic resolution:
void* funcPtr = nullptr;
cudaGetDriverEntryPoint("cuSomeNewFunction", &funcPtr, 0);
if (funcPtr) {
    // New function is available on this driver
    // Cast and use it
}
```

**Key guarantee:** An application compiled with CUDA 11.1 will run on any driver that supports CUDA 11.1 or later within the CUDA 11.x family.

### 24.2.3 Binary Compatibility

Binary compatibility is the foundational guarantee that CUDA has provided since CUDA 3.2: applications compiled with an older CUDA Toolkit will run on systems with newer NVIDIA drivers.

**How it works:**
- The NVIDIA driver maintains backward-compatible entry points for all CUDA API functions.
- Fat binaries contain code for multiple architectures; the driver selects the best match at load time.
- PTX code in the fat binary can be JIT-compiled for newer architectures that did not exist when the application was compiled.

**Example:**

```
Application compiled with CUDA 10.0 -> Runs on driver supporting CUDA 11.x, 12.x
Application compiled with CUDA 11.0 -> Runs on driver supporting CUDA 12.x
Application compiled with CUDA 3.2  -> Still runs on modern drivers (with limitations)
```

**Limitations:**
- Binary compatibility only goes forward (old apps on new drivers), not backward.
- Deprecated APIs may be removed in a new major version.
- Features requiring newer hardware obviously require newer hardware, regardless of driver version.

### 24.2.4 Compatibility Decision Matrix

| Scenario | Mechanism | Setup Required |
|----------|-----------|---------------|
| Old app, new driver | Binary Compatibility | None (automatic) |
| New app, same major version, newer driver | Enhanced Compatibility | None (automatic with CUDA 11.1+) |
| New app, older driver (same major) | Forward Compatible Upgrade | Install compatibility library |
| New app, much older driver (different major) | May not work | Upgrade driver or use container |

---

## 24.3 Building for Maximum Compatibility

To ensure a CUDA application runs on the widest range of GPU architectures and driver versions, follow these build practices.

### 24.3.1 Specifying Multiple Architectures

Use the `-gencode` flag (or `--generate-code`) to compile kernels for multiple target architectures. Include both the virtual architecture (for PTX) and the real architecture (for cubin):

```bash
nvcc -gencode=arch=compute_70,code=sm_70 \
     -gencode=arch=compute_80,code=sm_80 \
     -gencode=arch=compute_86,code=sm_86 \
     -gencode=arch=compute_89,code=sm_89 \
     -gencode=arch=compute_90,code=sm_90 \
     -gencode=arch=compute_compute_90,code=compute_90 \
     -O2 mykernel.cu -o myapp
```

**Explanation of each gencode entry:**

| Entry | Meaning |
|-------|---------|
| `arch=compute_70,code=sm_70` | Generate SASS (cubin) for sm_70 (V100). Runs natively on V100. |
| `arch=compute_80,code=sm_80` | Generate SASS for sm_80 (A100). Runs natively on A100. |
| `arch=compute_86,code=sm_86` | Generate SASS for sm_86 (RTX 3090). Runs natively on GA10x. |
| `arch=compute_89,code=sm_89` | Generate SASS for sm_89 (RTX 4090). Runs natively on AD10x. |
| `arch=compute_90,code=sm_90` | Generate SASS for sm_90 (H100). Runs natively on H100. |
| `arch=compute_compute_90,code=compute_90` | Embed PTX for compute_90. Can be JIT-compiled for future architectures. |

**The last entry is critical for forward compatibility:** By embedding PTX for the highest virtual architecture you support, your application can be JIT-compiled for GPU architectures that did not exist when you built the application.

### 24.3.2 Minimal Multi-Architecture Build

For applications that need to support a broad range of GPUs with minimal binary size:

```bash
# Support Volta (sm_70) through Hopper (sm_90) with forward compat
nvcc -gencode=arch=compute_70,code=sm_70 \
     -gencode=arch=compute_80,code=sm_80 \
     -gencode=arch=compute_90,code=sm_90 \
     -gencode=arch=compute_compute_90,code=compute_90 \
     -O2 mykernel.cu -o myapp
```

This produces a fat binary with:
- Native cubins for sm_70, sm_80, and sm_90
- PTX for compute_90 (for JIT compilation on future architectures)
- The driver automatically selects the best matching code at load time

### 24.3.3 CMake Integration

```cmake
cmake_minimum_required(VERSION 3.18)
project(my_cuda_app LANGUAGES CUDA)

# Set CUDA architectures
set_target_properties(myapp PROPERTIES
    CUDA_ARCHITECTURES "70;80;86;89;90;90-virtual"
)

# The "90-virtual" entry generates PTX for compute_90
```

### 24.3.4 Build Flags for Compatibility

| Flag | Purpose |
|------|---------|
| `-gencode=arch=compute_X,code=sm_X` | Generate native code for sm_X |
| `-gencode=arch=compute_X,code=compute_X` | Embed PTX for future JIT |
| `-O2` | Optimization (recommended for release builds) |
| `--cudart=static` | Static link CUDA runtime (default, avoids DLL dependencies) |
| `--cudart=shared` | Dynamic link CUDA runtime (smaller binary, but requires cudart DLL) |
| `--ex-relaxed-constexpr` | Allow host constexpr in device code |
| `--extended-lambda` | Enable extended lambdas for device code |

### 24.3.5 Detecting Architecture at Runtime

```cpp
#include <cuda_runtime.h>
#include <stdio.h>

void select_kernel_variant() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("Device: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);

    // Dispatch to optimized kernel variant based on architecture
    int cc = prop.major * 10 + prop.minor;

    if (cc >= 90) {
        // Use Hopper-optimized kernel (e.g., with TMA, wgmma)
        kernel_hopper<<<grid, block>>>(...);
    } else if (cc >= 80) {
        // Use Ampere-optimized kernel (e.g., async copy, L2 persist)
        kernel_ampere<<<grid, block>>>(...);
    } else if (cc >= 70) {
        // Use Volta-optimized kernel (e.g., cooperative groups, tensor cores)
        kernel_volta<<<grid, block>>>(...);
    } else {
        // Fallback kernel
        kernel_fallback<<<grid, block>>>(...);
    }
}
```

---

## 24.4 Runtime Redistribution

When distributing a CUDA application, you must decide how to handle the CUDA runtime dependency.

### 24.4.1 Static Linking (Default)

By default, nvcc statically links the CUDA runtime (libcudart) into the application binary.

```bash
# Static linking (default)
nvcc myapp.cu -o myapp

# Explicit static linking
nvcc --cudart=static myapp.cu -o myapp
```

**Advantages:**
- No external runtime DLL/SO dependency.
- The application is self-contained and easier to deploy.
- Version conflicts between the application's runtime and the system's runtime are avoided.

**Disadvantages:**
- Larger binary size (the runtime library adds ~50-100 MB).
- Multiple CUDA applications each carry their own copy of the runtime.
- Runtime bug fixes require recompilation.

### 24.4.2 Dynamic Linking

Dynamic linking shares the CUDA runtime library across multiple applications.

```bash
# Dynamic linking
nvcc --cudart=shared myapp.cu -o myapp
```

**Advantages:**
- Smaller binary size.
- Runtime bug fixes can be applied by updating the shared library.
- Multiple applications share one copy of the runtime in memory.

**Disadvantages:**
- The application depends on the correct version of `libcudart.so` (Linux) or `cudart64_XY.dll` (Windows) being installed.
- Version conflicts can occur if multiple CUDA toolkit versions are installed.
- Deployment is more complex because the shared library must be distributed alongside the application.

### 24.4.3 Redistribution Files

When distributing an application, you may need to include certain CUDA libraries:

**Minimum redistribution set (dynamic linking):**

| Platform | Files |
|----------|-------|
| Linux | `libcudart.so.XY.Z` |
| Windows | `cudart64_XY.dll` |

**If using CUDA libraries (cuBLAS, cuDNN, etc.), also include:**

| Library | Linux | Windows |
|---------|-------|---------|
| cuBLAS | `libcublas.so.XY`, `libcublasLt.so.XY` | `cublas64_XY.dll`, `cublasLt64_XY.dll` |
| cuFFT | `libcufft.so.XY` | `cufft64_XY.dll` |
| cuRAND | `libcurand.so.XY` | `curand64_XY.dll` |
| cuSOLVER | `libcusolver.so.XY` | `cusolver64_XY.dll` |
| NPP | `libnppc.so.XY`, `libnppial.so.XY`, ... | `nppc64_XY.dll`, `nppial64_XY.dll`, ... |

**Container distribution:**

```dockerfile
FROM nvidia/cuda:12.3.0-runtime-ubuntu22.04

# The runtime image includes the CUDA runtime and essential libraries
COPY myapp /app/myapp
CMD ["/app/myapp"]
```

### 24.4.4 Compatibility Library Redistribution

For forward-compatible applications (see Section 24.2.1), include the compatibility library:

```bash
# On Linux, the compatibility library must be in the library search path
export LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH
```

In a container:

```dockerfile
FROM nvidia/cuda:12.3.0-base-ubuntu22.04
# The compat library is included in the CUDA container images
```

---

## 24.5 Error Handling

Robust error handling is critical for CUDA applications because many CUDA errors are reported asynchronously.

### 24.5.1 Error Return Convention

All CUDA Runtime API functions return `cudaError_t`. The only exception is `cudaGetLastError()`, which returns the last error and clears it.

```cpp
// Every CUDA call should be checked
cudaError_t err = cudaMalloc(&devPtr, size);
if (err != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed: %s\n",
            cudaGetErrorString(err));
    // Handle the error...
}
```

### 24.5.2 The CUDA_CHECK Macro Pattern

A common best practice is to wrap all CUDA calls in a macro that checks for errors:

```cpp
#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                               \
    do {                                                                \
        cudaError_t err = (call);                                       \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr,                                             \
                "CUDA error at %s:%d - %s: %s\n",                       \
                __FILE__, __LINE__,                                     \
                #call, cudaGetErrorString(err));                        \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

// Usage
CUDA_CHECK(cudaMalloc(&devPtr, size));
CUDA_CHECK(cudaMemcpy(devPtr, hostPtr, size, cudaMemcpyHostToDevice));
CUDA_CHECK(cudaFree(devPtr));
```

### 24.5.3 Kernel Launch Error Checking

Kernel launches are asynchronous, so errors from the launch itself are reported differently from errors during kernel execution.

```cpp
// Step 1: Check for launch errors (reported immediately)
myKernel<<<grid, block, sharedMem, stream>>>(args);
cudaError_t launchErr = cudaGetLastError();
if (launchErr != cudaSuccess) {
    fprintf(stderr, "Kernel launch failed: %s\n",
            cudaGetErrorString(launchErr));
}

// Step 2: Check for execution errors (reported asynchronously)
cudaError_t execErr = cudaDeviceSynchronize();
if (execErr != cudaSuccess) {
    fprintf(stderr, "Kernel execution error: %s\n",
            cudaGetErrorString(execErr));
}
```

**Important distinction:**
- `cudaGetLastError()` returns errors from the kernel launch configuration (e.g., too many threads, invalid parameters). It does NOT wait for the kernel to finish.
- `cudaDeviceSynchronize()` blocks until all pending work is complete and returns any errors that occurred during execution (e.g., out-of-bounds memory access, illegal instruction).

### 24.5.4 Per-Stream Error Checking

```cpp
// Check errors on a specific stream
cudaStreamSynchronize(stream);
cudaError_t streamErr = cudaStreamQuery(stream);
// Note: cudaStreamSynchronize already returns the error

// Check without synchronizing (polling)
cudaError_t queryErr = cudaStreamQuery(stream);
if (queryErr == cudaSuccess) {
    // Stream is idle, all operations completed successfully
} else if (queryErr == cudaErrorNotReady) {
    // Stream still has pending work
} else {
    // An error occurred in one of the stream's operations
    fprintf(stderr, "Stream error: %s\n",
            cudaGetErrorString(queryErr));
}
```

### 24.5.5 Driver API Error Handling

```cpp
#include <cuda.h>

#define CU_CHECK(call)                                                 \
    do {                                                                \
        CUresult err = (call);                                          \
        if (err != CUDA_SUCCESS) {                                      \
            const char* errStr;                                         \
            const char* errName;                                        \
            cuGetErrorString(err, &errStr);                             \
            cuGetErrorName(err, &errName);                              \
            fprintf(stderr,                                             \
                "Driver API error at %s:%d - %s: %s (%s)\n",            \
                __FILE__, __LINE__,                                     \
                #call, errName, errStr);                                \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

// Usage
CU_CHECK(cuMemAlloc(&devPtr, size));
CU_CHECK(cuLaunchKernel(function,
    gridX, gridY, gridZ,
    blockX, blockY, blockZ,
    sharedMem, stream,
    kernelParams, extra));
```

### 24.5.6 Common Error Codes

| Error Code | Description | Common Cause |
|------------|-------------|--------------|
| `cudaSuccess` | No error | N/A |
| `cudaErrorInvalidValue` | Invalid argument | NULL pointer, out-of-range value, wrong enum |
| `cudaErrorMemoryAllocation` | Allocation failed | Out of GPU memory, fragmentation |
| `cudaErrorInvalidDevice` | Invalid device ordinal | `CUDA_VISIBLE_DEVICES` filtering |
| `cudaErrorInvalidKernelImage` | Invalid kernel binary | No compatible cubin/PTX for target architecture |
| `cudaErrorLaunchOutOfResources` | Launch exceeded resources | Too many threads, too much shared memory |
| `cudaErrorIllegalAddress` | Out-of-bounds memory access | Buffer overrun, use-after-free on device |
| `cudaErrorMisalignedAddress` | Misaligned memory access | Reading 4-byte value at odd address |
| `cudaErrorInvalidPc` | Invalid program counter | Corrupted kernel code |
| `cudaErrorLaunchTimeout` | Kernel exceeded time limit | Infinite loop, TCC timeout |
| `cudaErrorNotReady` | Async operation not complete | Polling with `cudaStreamQuery()` |
| `cudaErrorNoDevice` | No CUDA-capable device | No GPU, driver not installed |
| `cudaErrorNotSupported` | Feature not supported | Hardware or driver limitation |

### 24.5.7 Error Handling Best Practices

```cpp
// 1. Always check return values
cudaError_t err = cudaMalloc(&ptr, size);
CHECK_CUDA(err);

// 2. Check kernel launch errors immediately
kernel<<<grid, block>>>();
CHECK_CUDA(cudaGetLastError());

// 3. Check execution errors after synchronization
CHECK_CUDA(cudaDeviceSynchronize());

// 4. In production, use a non-fatal error handler
void handle_cuda_error(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n",
                file, line, cudaGetErrorString(err));
        // Optionally: reset the device to recover
        cudaDeviceReset();
    }
}

// 5. For debugging, use CUDA_LAUNCH_BLOCKING=1
// This makes all kernel launches synchronous, localizing errors
```

---

## 24.6 nvidia-smi

`nvidia-smi` (NVIDIA System Management Interface) is a command-line utility for monitoring and managing NVIDIA GPU devices. It is installed with the NVIDIA driver.

### 24.6.1 Querying GPU Information

```bash
# Basic GPU status (all GPUs)
nvidia-smi

# Query specific information in a machine-readable format
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv

# Output example:
# index, name, utilization.gpu [%], memory.used [MiB], memory.total [MiB]
# 0, NVIDIA A100-SXM4-80GB, 45 %, 40960 MiB, 81920 MiB
# 1, NVIDIA A100-SXM4-80GB, 12 %, 10240 MiB, 81920 MiB
```

### 24.6.2 Common Query Targets

| Query Type | Flag | Example |
|-----------|------|---------|
| **GPU properties** | `--query-gpu` | Index, name, UUID, compute capability, memory total |
| **ECC errors** | `--query-gpu=ecc.errors.corrected.volatile.total` | Single-bit ECC error count |
| **Utilization** | `--query-gpu=utilization.gpu,utilization.memory` | GPU and memory utilization % |
| **Active processes** | `--query-compute-apps` | PID, name, used GPU memory |
| **Clocks** | `--query-gpu=clocks.sm,clocks.mem` | SM and memory clock frequencies |
| **Temperature** | `--query-gpu=temperature.gpu` | GPU temperature in C |
| **Power** | `--query-gpu=power.draw,power.limit` | Current draw and limit in watts |
| **PCI info** | `--query-gpu=pci.bus_id,pcie.link.gen.current` | Bus ID and PCIe link speed |
| **Driver version** | `--query-gpu=driver_version` | NVIDIA driver version |

### 24.6.3 Full Query Example

```bash
# Comprehensive GPU status
nvidia-smi --query-gpu=\
index,\
gpu_uuid,\
name,\
compute_cap,\
driver_version,\
temperature.gpu,\
utilization.gpu,\
utilization.memory,\
memory.total,\
memory.used,\
memory.free,\
power.draw,\
power.limit,\
clocks.sm,\
clocks.mem,\
ecc.mode.current,\
pstate,\
pcie.link.gen.current,\
pcie.link.width.current \
--format=csv -l 5
# -l 5 = refresh every 5 seconds
```

### 24.6.4 Modifying GPU State

```bash
# Enable/disable ECC memory (requires reboot)
nvidia-smi -i 0 --ecc-config=1    # Enable ECC
nvidia-smi -i 0 --ecc-config=0    # Disable ECC

# Set compute mode
nvidia-smi -i 0 --compute-mode=Exclusive    # Only one process at a time
nvidia-smi -i 0 --compute-mode=Default      # Multiple processes allowed
nvidia-smi -i 0 --compute-mode=Prohibited   # No compute allowed
nvidia-smi -i 0 --compute-mode=ExclusiveProcess  # One process, multiple threads

# Set persistence mode (keeps GPU initialized between processes)
nvidia-smi -i 0 --persistence-mode=1   # Enabled
nvidia-smi -i 0 --persistence-mode=0   # Disabled

# Reset GPU (useful after a hang)
nvidia-smi -i 0 --gpu-reset

# Set power limit
nvidia-smi -i 0 --power-limit=250    # Set to 250W

# Set application clock frequencies
nvidia-smi -i 0 --applications-clocks=2100,1401   # SM 2100 MHz, Mem 1401 MHz

# Reset application clocks to default
nvidia-smi -i 0 --reset-applications-clocks
```

### 24.6.5 Monitoring and Logging

```bash
# Continuous monitoring with 1-second interval
nvidia-smi dmon -s pucvmet -i 0
# -s flags: p=power, u=utilization, c=clocks, v=violations,
#           m=memory, e=ecc, t=throughput

# Log to file
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu \
    --format=csv -l 1 -f gpu_log.csv

# Show topology (GPU interconnects)
nvidia-smi topo -m
# Output shows NVLink, PCIe, and other interconnect topology
```

---

## 24.7 NVML

The NVIDIA Management Library (NVML) provides a C-based API for building GPU management and monitoring applications. It offers programmatic access to the same information displayed by `nvidia-smi`, and more.

### 24.7.1 Overview

NVML is designed for building:
- Cluster management tools
- GPU monitoring daemons
- Job schedulers with GPU awareness
- Automatic scaling systems
- Health monitoring and alerting

**Key characteristics:**
- C-based API (header: `nvml.h`)
- Backward-compatible: new versions add functions without breaking existing ones
- Available on Linux and Windows
- Does not require a CUDA context (can be used independently)
- Perl and Python bindings available (`nvidia-ml-py3` package)

### 24.7.2 Basic NVML Usage

```c
#include <nvml.h>
#include <stdio.h>

int main() {
    // Initialize NVML
    nvmlReturn_t result = nvmlInit();
    if (NVML_SUCCESS != result) {
        fprintf(stderr, "NVML init failed: %s\n", nvmlErrorString(result));
        return 1;
    }

    // Get device count
    unsigned int deviceCount;
    nvmlDeviceGetCount(&deviceCount);
    printf("Found %u GPU devices\n", deviceCount);

    // Iterate over devices
    for (unsigned int i = 0; i < deviceCount; i++) {
        nvmlDevice_t device;
        nvmlDeviceGetHandleByIndex(i, &device);

        // Get device name
        char name[NVML_DEVICE_NAME_BUFFER_SIZE];
        nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE);

        // Get memory info
        nvmlMemory_t memory;
        nvmlDeviceGetMemoryInfo(device, &memory);

        // Get utilization
        nvmlUtilization_t utilization;
        nvmlDeviceGetUtilizationRates(device, &utilization);

        // Get temperature
        unsigned int temp;
        nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp);

        // Get power usage
        unsigned int power;
        nvmlDeviceGetPowerUsage(device, &power);

        printf("GPU %u: %s\n", i, name);
        printf("  Memory: %llu / %llu MiB\n",
               memory.used / (1024 * 1024),
               memory.total / (1024 * 1024));
        printf("  GPU Util: %u%%, Memory Util: %u%%\n",
               utilization.gpu, utilization.memory);
        printf("  Temperature: %u C\n", temp);
        printf("  Power: %u mW (%.1f W)\n", power, power / 1000.0f);
    }

    // Shutdown NVML
    nvmlShutdown();
    return 0;
}
```

**Compile:** `gcc -lnvidia-ml monitor.c -o monitor`

### 24.7.3 Key NVML API Categories

| Category | Example Functions | Description |
|----------|------------------|-------------|
| **System** | `nvmlInit()`, `nvmlShutdown()`, `nvmlSystemGetDriverVersion()` | System-level initialization and queries |
| **Device** | `nvmlDeviceGetHandleByIndex()`, `nvmlDeviceGetName()`, `nvmlDeviceGetSerial()` | Device identification |
| **Memory** | `nvmlDeviceGetMemoryInfo()`, `nvmlDeviceGetBAR1MemoryInfo()` | Memory usage and capacity |
| **Utilization** | `nvmlDeviceGetUtilizationRates()`, `nvmlDeviceGetComputeRunningProcesses()` | GPU utilization and active processes |
| **Power** | `nvmlDeviceGetPowerUsage()`, `nvmlDeviceGetPowerManagementLimit()` | Power monitoring and limits |
| **Thermal** | `nvmlDeviceGetTemperature()`, `nvmlDeviceGetTemperatureThreshold()` | Temperature monitoring |
| **Clocks** | `nvmlDeviceGetClockInfo()`, `nvmlDeviceGetMaxClockInfo()` | Clock frequency monitoring |
| **ECC** | `nvmlDeviceGetTotalEccErrors()`, `nvmlDeviceGetMemoryErrorCounter()` | ECC error tracking |
| **Topology** | `nvmlDeviceGetTopologyCommonAncestor()`, `nvmlDeviceGetP2PStatus()` | GPU interconnect topology |
| **Events** | `nvmlDeviceRegisterEvents()`, `nvmlDeviceGetFieldValues()` | Event-driven monitoring |

### 24.7.4 Python Bindings

```python
import pynvml

pynvml.nvmlInit()
device_count = pynvml.nvmlDeviceGetCount()

for i in range(device_count):
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    name = pynvml.nvmlDeviceGetName(handle)
    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
    temperature = pynvml.nvmlDeviceGetTemperature(
        handle, pynvml.NVML_TEMPERATURE_GPU)

    print(f"GPU {i}: {name}")
    print(f"  Memory: {memory_info.used // (1024**2)} / "
          f"{memory_info.total // (1024**2)} MiB")
    print(f"  Utilization: GPU {utilization.gpu}%, "
          f"Memory {utilization.memory}%")
    print(f"  Temperature: {temperature} C")

pynvml.nvmlShutdown()
```

### 24.7.5 Monitoring Running Processes

```c
// Get processes currently using the GPU
unsigned int infoCount;
nvmlProcessInfo_t* infos = NULL;

// First call to get the count
nvmlDeviceGetComputeRunningProcesses(device, &infoCount, NULL);
infos = (nvmlProcessInfo_t*)malloc(infoCount * sizeof(nvmlProcessInfo_t));
nvmlDeviceGetComputeRunningProcesses(device, &infoCount, infos);

for (unsigned int i = 0; i < infoCount; i++) {
    printf("PID: %u, Used GPU Memory: %llu MiB\n",
           infos[i].pid,
           infos[i].usedGpuMemory / (1024 * 1024));
}
free(infos);
```

### 24.7.6 Event Monitoring

```c
// Register for event notifications
nvmlEventSet_t eventSet;
nvmlEventSetCreate(&eventSet);

// Register for specific events on a device
nvmlDeviceRegisterEvents(device,
    NVML_EVENT_CLOCK_CHANGE |
    NVML_EVENT_POWER_SOURCE_CHANGE |
    NVML_EVENT_SINGLE_BIT_ECC_ERROR,
    eventSet);

// Wait for events (blocking, with timeout)
nvmlEventData_t eventData;
nvmlReturn_t res = nvmlEventSetWait(eventSet, &eventData, 5000);  // 5s timeout

if (res == NVML_SUCCESS) {
    printf("Event on device %u, type: 0x%08x\n",
           eventData.deviceIndex, eventData.eventType);
}

nvmlEventSetFree(eventSet);
```

---

## 24.8 Testing CUDA Availability

Before using CUDA features, applications should verify that CUDA-capable hardware and software are available.

### 24.8.1 Basic Availability Check

```cpp
#include <cuda_runtime.h>
#include <stdio.h>

bool check_cuda_availability() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess || deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        if (err != cudaSuccess) {
            printf("Error: %s\n", cudaGetErrorString(err));
        }
        return false;
    }

    printf("Found %d CUDA device(s)\n", deviceCount);
    return true;
}
```

### 24.8.2 Detailed Device Query

```cpp
void print_device_info() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        printf("=== Device %d: %s ===\n", i, prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total global memory: %.1f GiB\n",
               prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Multiprocessor count: %d\n", prop.multiProcessorCount);
        printf("  Clock rate: %.2f GHz\n", prop.clockRate / 1e6);
        printf("  Memory clock rate: %.2f GHz\n", prop.memoryClockRate / 1e6);
        printf("  Memory bus width: %d-bit\n", prop.memoryBusWidth);
        printf("  L2 cache size: %d bytes\n", prop.l2CacheSize);
        printf("  Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max block dimensions: (%d, %d, %d)\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1],
               prop.maxThreadsDim[2]);
        printf("  Max grid dimensions: (%d, %d, %d)\n",
               prop.maxGridSize[0], prop.maxGridSize[1],
               prop.maxGridSize[2]);
        printf("  Warp size: %d\n", prop.warpSize);
        printf("  Shared memory per block: %zu bytes\n",
               prop.sharedMemPerBlock);
        printf("  Shared memory per SM: %zu bytes\n",
               prop.sharedMemPerMultiProcessor);
        printf("  Registers per block: %d\n", prop.regsPerBlock);
        printf("  Concurrent kernels: %s\n",
               prop.concurrentKernels ? "yes" : "no");
        printf("  ECC enabled: %s\n", prop.ECCEnabled ? "yes" : "no");
        printf("  PCI bus ID: %04x:%02x:%02x.%x\n",
               prop.pciDomainID, prop.pciBusID, prop.pciDeviceID, 0);
        printf("  UUID: %s\n", prop.uuid);
    }
}
```

### 24.8.3 Feature Detection

```cpp
bool check_feature_support(int device) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    int cc = prop.major * 10 + prop.minor;

    // Check compute capability requirements
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);

    // Unified memory
    printf("Unified addressing: %s\n",
           prop.unifiedAddressing ? "supported" : "not supported");

    // Managed memory
    int managedMemorySupport;
    cudaDeviceGetAttribute(&managedMemorySupport,
        cudaDevAttrManagedMemory, device);
    printf("Managed memory: %s\n",
           managedMemorySupport ? "supported" : "not supported");

    // Cooperative launch
    int cooperativeLaunch;
    cudaDeviceGetAttribute(&cooperativeLaunch,
        cudaDevAttrCooperativeLaunch, device);
    printf("Cooperative launch: %s\n",
           cooperativeLaunch ? "supported" : "not supported");

    // Compute preemption
    int computePreemption;
    cudaDeviceGetAttribute(&computePreemption,
        cudaDevAttrComputePreemptionSupported, device);
    printf("Compute preemption: %s\n",
           computePreemption ? "supported" : "not supported");

    return true;
}
```

### 24.8.4 Memory Allocation Test

```cpp
bool test_gpu_memory(size_t requiredMB) {
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    size_t totalMB = prop.totalGlobalMem / (1024 * 1024);
    size_t freeMB, totalFreeMB;
    cudaMemGetInfo(&freeMB, &totalFreeMB);
    freeMB /= (1024 * 1024);
    totalFreeMB /= (1024 * 1024);

    printf("GPU memory: %.0f MiB total, %zu MiB free\n",
           (double)totalMB, freeMB);

    if (requiredMB > freeMB) {
        printf("Insufficient GPU memory: need %zu MiB, have %zu MiB free\n",
               requiredMB, freeMB);
        return false;
    }

    // Test allocation
    void* testPtr;
    size_t testSize = requiredMB * 1024 * 1024;
    cudaError_t err = cudaMalloc(&testPtr, testSize);
    if (err != cudaSuccess) {
        printf("Allocation test failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    cudaFree(testPtr);

    printf("Allocation test passed: %zu MiB\n", requiredMB);
    return true;
}
```

### 24.8.5 Driver and Runtime Version Check

```cpp
bool check_version_compatibility(int minDriverVersion,
                                 int minRuntimeVersion) {
    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("Driver version:  %d.%d\n",
           driverVersion / 1000, (driverVersion % 1000) / 10);
    printf("Runtime version:  %d.%d\n",
           runtimeVersion / 1000, (runtimeVersion % 1000) / 10);

    if (driverVersion < minDriverVersion) {
        printf("Driver too old: need %d.%d, have %d.%d\n",
               minDriverVersion / 1000, (minDriverVersion % 1000) / 10,
               driverVersion / 1000, (driverVersion % 1000) / 10);
        return false;
    }

    if (runtimeVersion < minRuntimeVersion) {
        printf("Runtime too old: need %d.%d, have %d.%d\n",
               minRuntimeVersion / 1000, (minRuntimeVersion % 1000) / 10,
               runtimeVersion / 1000, (runtimeVersion % 1000) / 10);
        return false;
    }

    return true;
}
```

### 24.8.6 Complete Startup Check

```cpp
#include <cuda_runtime.h>
#include <stdio.h>

// Comprehensive CUDA availability and capability check
// Returns: 0 = ready, negative = fatal error
int cuda_startup_check(int requiredCC_Major, int requiredCC_Minor,
                       size_t requiredMemoryMB) {
    // Step 1: Check device count
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        return -2;
    }

    // Step 2: Find a suitable device
    int selectedDevice = -1;
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        if (prop.major >= requiredCC_Major &&
            prop.minor >= requiredCC_Minor) {
            size_t freeMem, totalMem;
            cudaMemGetInfo(&freeMem, &totalMem);
            if (freeMem >= requiredMemoryMB * 1024 * 1024) {
                selectedDevice = i;
                break;
            }
        }
    }

    if (selectedDevice < 0) {
        fprintf(stderr, "No device meets requirements "
                "(CC >= %d.%d, %zu MiB free memory)\n",
                requiredCC_Major, requiredCC_Minor, requiredMemoryMB);
        return -3;
    }

    // Step 3: Set the device
    cudaSetDevice(selectedDevice);

    // Step 4: Print summary
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, selectedDevice);
    printf("Using GPU %d: %s (CC %d.%d)\n",
           selectedDevice, prop.name, prop.major, prop.minor);

    int driverVer, runtimeVer;
    cudaDriverGetVersion(&driverVer);
    cudaRuntimeGetVersion(&runtimeVer);
    printf("Driver: %d.%d, Runtime: %d.%d\n",
           driverVer / 1000, (driverVer % 1000) / 10,
           runtimeVer / 1000, (runtimeVer % 1000) / 10);

    return 0;
}

// Usage
int main() {
    int result = cuda_startup_check(
        7,      // Require CC 7.0+ (Volta)
        0,
        4096    // Require 4 GiB free memory
    );

    if (result != 0) {
        fprintf(stderr, "CUDA startup check failed: %d\n", result);
        return 1;
    }

    // Proceed with CUDA application...
    return 0;
}
```
