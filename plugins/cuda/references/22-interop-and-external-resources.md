# 22. Interoperability and External Resources

CUDA provides a comprehensive set of interoperability APIs that allow GPU resources (memory buffers, synchronization primitives) to be shared with other graphics and compute APIs such as Vulkan, Direct3D, OpenGL, and NVIDIA's NvSci framework. These APIs are essential for building applications that combine CUDA computation with rendering, video processing, or cross-device data exchange.

---

## Table of Contents

1. [External Resource Interoperability](#221-external-resource-interoperability)
2. [Vulkan Interoperability](#222-vulkan-interoperability)
3. [Direct3D Interoperability](#223-direct3d-interoperability)
4. [NVSCI Interoperability](#224-nvsci-interoperability)
5. [OpenGL Interoperability (via Vulkan)](#225-opengl-interoperability-via-vulkan)
6. [General Interoperability Patterns](#226-general-interoperability-patterns)

---

## 22.1 External Resource Interoperability

The external resource interoperability API provides a unified mechanism for importing and exporting memory and synchronization objects between CUDA and other APIs. This API replaces older, API-specific interop functions with a more flexible and extensible framework.

### 22.1.1 External Memory

External memory allows a memory allocation from one API (e.g., Vulkan, D3D, NvSci) to be imported into CUDA and accessed as a device pointer or a CUDA array.

#### Importing External Memory

```cpp
cudaExternalMemory_t extMem;
cudaExternalMemoryHandleDesc handleDesc = {};

// Configure the handle description
handleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;  // or other handle type
handleDesc.handle.fd = fd;        // File descriptor (Linux)
// handleDesc.handle.win32.handle = handle;  // Windows NT handle
// handleDesc.handle.win32.name = name;      // Windows named handle
handleDesc.size = allocationSize;  // Size of the external allocation
handleDesc.flags = 0;              // Reserved, must be 0

cudaError_t err = cudaImportExternalMemory(&extMem, &handleDesc);
if (err != cudaSuccess) {
    fprintf(stderr, "Failed to import external memory: %s\n",
            cudaGetErrorString(err));
}
```

**Supported handle types:**

| Handle Type | Description | Platform |
|-------------|-------------|----------|
| `cudaExternalMemoryHandleTypeOpaqueFd` |Opaque POSIX file descriptor | Linux |
| `cudaExternalMemoryHandleTypeOpaqueWin32` | Windows NT handle | Windows |
| `cudaExternalMemoryHandleTypeOpaqueWin32Kmt` | Windows D3DKMT handle | Windows |
| `cudaExternalMemoryHandleTypeD3D12Heap` | D3D12 heap object | Windows |
| `cudaExternalMemoryHandleTypeD3D12Resource` | D3D12 committed resource | Windows |
| `cudaExternalMemoryHandleTypeD3D11Resource` | D3D11 resource | Windows |
| `cudaExternalMemoryHandleTypeD3D11ResourceKmt` | D3D11 resource (KMT) | Windows |
| `cudaExternalMemoryHandleTypeNvSciBuf` | NvSciBuf object | Linux (Tegra) |
| `cudaExternalMemoryHandleTypeDmaBuf` | DMA-BUF file descriptor | Linux |

**Important:** After calling `cudaImportExternalMemory()`, the file descriptor (or handle) is consumed by CUDA. The application must NOT close or use it again.

#### Mapping External Memory as a Linear Buffer

Once external memory is imported, it can be mapped as a linear device pointer for random access:

```cpp
void* devPtr = nullptr;
cudaExternalMemoryBufferDesc bufDesc = {};
bufDesc.offset = 0;            // Offset within the external allocation
bufDesc.size = bufferSize;     // Size of the mapping
bufDesc.flags = 0;             // Reserved, must be 0

cudaError_t err = cudaExternalMemoryGetMappedBuffer(
    &devPtr, extMem, &bufDesc);
if (err != cudaSuccess) {
    fprintf(stderr, "Failed to map external buffer: %s\n",
            cudaGetErrorString(err));
}

// Use devPtr like any other device pointer
myKernel<<<grid, block>>>(devPtr, bufferSize);

// Cleanup
cudaFree(devPtr);
cudaDestroyExternalMemory(extMem);
```

#### Mapping External Memory as a Mipmapped Array

External memory can also be mapped as a CUDA mipmapped array for texture or surface operations:

```cpp
cudaMipmappedArray_t mipmapArray = nullptr;
cudaExternalMemoryMipmappedArrayDesc mipmapDesc = {};
mipmapDesc.offset = 0;
mipmapDesc.formatDesc.x = 32;              // Bits per component
mipmapDesc.formatDesc.y = 32;
mipmapDesc.formatDesc.z = 32;
mipmapDesc.formatDesc.w = 32;
mipmapDesc.formatDesc.f = cudaChannelFormatKindFloat;
mipmapDesc.extent = make_cudaExtent(width, height, depth);
mipmapDesc.flags = 0;                      // cudaArrayDefault, etc.
mipmapDesc.numLevels = 1;                  // Number of mipmap levels

cudaError_t err = cudaExternalMemoryGetMappedMipmappedArray(
    &mipmapArray, extMem, &mipmapDesc);

// Get the level-0 array for surface write
cudaArray_t levelArray;
cudaGetMipmappedArrayLevel(&levelArray, mipmapArray, 0);

// Cleanup
cudaFreeMipmappedArray(mipmapArray);
cudaDestroyExternalMemory(extMem);
```

#### Destroying External Memory

```cpp
// All mapped resources must be freed BEFORE destroying external memory
cudaDestroyExternalMemory(extMem);
```

### 22.1.2 External Semaphores

External semaphores provide cross-API synchronization. They represent synchronization objects that can be signaled by one API and waited upon by another.

#### Importing External Semaphores

```cpp
cudaExternalSemaphore_t extSem;
cudaExternalSemaphoreHandleDesc semHandleDesc = {};
semHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
semHandleDesc.handle.fd = semaphoreFd;  // File descriptor
semHandleDesc.flags = 0;                // Reserved, must be 0

cudaImportExternalSemaphore(&extSem, &semHandleDesc);
```

**Supported semaphore handle types:**

| Handle Type | Description | Platform |
|-------------|-------------|----------|
| `cudaExternalSemaphoreHandleTypeOpaqueFd` | Opaque POSIX file descriptor | Linux |
| `cudaExternalSemaphoreHandleTypeOpaqueWin32` | Windows NT handle | Windows |
| `cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt` | Windows D3DKMT handle | Windows |
| `cudaExternalSemaphoreHandleTypeD3D12Fence` | D3D12 fence | Windows |
| `cudaExternalSemaphoreHandleTypeD3D11Fence` | D3D11 fence | Windows |
| `cudaExternalSemaphoreHandleTypeNvSciSync` | NvSciSync object | Linux (Tegra) |
| `cudaExternalSemaphoreHandleTypeKeyedMutex` | D3D11 keyed mutex | Windows |
| `cudaExternalSemaphoreHandleTypeSyncFile` | Sync file descriptor | Linux |

#### Signaling External Semaphores

```cpp
cudaExternalSemaphoreSignalParams signalParams = {};
signalParams.params.fence.value = signalValue;  // For fence-type semaphores
signalParams.flags = 0;

// Signal from a CUDA stream (GPU-side signal)
cudaSignalExternalSemaphoresAsync(
    &extSem, &signalParams, 1, stream);
```

#### Waiting on External Semaphores

```cpp
cudaExternalSemaphoreWaitParams waitParams = {};
waitParams.params.fence.value = waitValue;  // For fence-type semaphores
waitParams.flags = 0;

// Wait in a CUDA stream (GPU-side wait)
cudaWaitExternalSemaphoresAsync(
    &extSem, &waitParams, 1, stream);
```

#### Destroying External Semaphores

```cpp
cudaDestroyExternalSemaphore(extSem);
```

### 22.1.3 Complete Example: Round-Trip Memory Sharing

```cpp
#include <cuda_runtime.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>

// Process A: Export memory via a file descriptor
// (Assuming the exporting API provides an fd)

void import_and_use_memory(int fd, size_t size) {
    // Step 1: Import the external memory
    cudaExternalMemory_t extMem;
    cudaExternalMemoryHandleDesc memDesc = {};
    memDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    memDesc.handle.fd = fd;
    memDesc.size = size;
    cudaImportExternalMemory(&extMem, &memDesc);
    // fd is now consumed; do NOT close it

    // Step 2: Map as a linear buffer
    void* devPtr = nullptr;
    cudaExternalMemoryBufferDesc bufDesc = {};
    bufDesc.offset = 0;
    bufDesc.size = size;
    cudaExternalMemoryGetMappedBuffer(&devPtr, extMem, &bufDesc);

    // Step 3: Use in CUDA kernels
    process_kernel<<<256, 256>>>(devPtr, size);
    cudaStreamSynchronize(0);

    // Step 4: Cleanup
    cudaFree(devPtr);
    cudaDestroyExternalMemory(extMem);
}
```

---

## 22.2 Vulkan Interoperability

CUDA-Vulkan interoperability enables sharing of images, buffers, and synchronization primitives between CUDA and Vulkan. This is commonly used in applications that perform compute processing on Vulkan-rendered images, or that use CUDA for simulation/processing and Vulkan for visualization.

### 22.2.1 Prerequisites and Initialization

Vulkan must be initialized with the appropriate external memory and semaphore extensions before interoperability can be established.

**Required Vulkan instance extensions:**
- `VK_KHR_get_physical_device_properties2` (for UUID queries)

**Required Vulkan device extensions:**
- `VK_KHR_external_memory` (base)
- `VK_KHR_external_memory_fd` (Linux, for fd-based export)
- `VK_KHR_external_memory_win32` (Windows, for handle-based export)
- `VK_KHR_external_semaphore` (base)
- `VK_KHR_external_semaphore_fd` (Linux)
- `VK_KHR_external_semaphore_win32` (Windows)

```cpp
// Vulkan: Create device with required extensions
const char* deviceExtensions[] = {
    "VK_KHR_external_memory",
    "VK_KHR_external_memory_fd",
    "VK_KHR_external_semaphore",
    "VK_KHR_external_semaphore_fd",
    "VK_KHR_get_memory_requirements2",
    "VK_KHR_dedicated_allocation",
};

VkDeviceCreateInfo deviceCreateInfo = {};
deviceCreateInfo.enabledExtensionCount = 6;
deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions;
vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device);
```

### 22.2.2 Matching CUDA and Vulkan Devices

Before sharing resources, you must ensure that the CUDA device and Vulkan physical device refer to the same physical GPU. This is done by comparing device UUIDs.

```cpp
#include <cuda_runtime.h>
#include <vulkan/vulkan.h>

int find_cuda_device_for_vulkan(VkPhysicalDevice vkPhysicalDevice) {
    // Get Vulkan device UUID
    VkPhysicalDeviceIDPropertiesKHR idProps = {};
    idProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES_KHR;

    VkPhysicalDeviceProperties2KHR props2 = {};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2_KHR;
    props2.pNext = &idProps;
    vkGetPhysicalDeviceProperties2KHR(vkPhysicalDevice, &props2);

    // Iterate over CUDA devices to find a match
    int cudaDeviceCount;
    cudaGetDeviceCount(&cudaDeviceCount);

    for (int i = 0; i < cudaDeviceCount; i++) {
        cudaDeviceProp cudaProp;
        cudaGetDeviceProperties(&cudaProp, i);

        // Compare UUIDs (both are 16 bytes)
        if (memcmp(cudaProp.uuid, idProps.deviceUUID, 16) == 0) {
            printf("Matched Vulkan device to CUDA device %d (%s)\n",
                   i, cudaProp.name);
            return i;
        }
    }

    fprintf(stderr, "No matching CUDA device found for Vulkan device\n");
    return -1;
}
```

### 22.2.3 Exporting Vulkan Memory to CUDA

```cpp
// Vulkan: Create an image with external memory export capability
VkExternalMemoryImageCreateInfoKHR extMemImageInfo = {};
extMemImageInfo.sType =
    VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO_KHR;
extMemImageInfo.handleTypes =
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;

VkImageCreateInfo imageInfo = {};
imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
imageInfo.pNext = &extMemImageInfo;
imageInfo.imageType = VK_IMAGE_TYPE_2D;
imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
imageInfo.extent = {1920, 1080, 1};
imageInfo.mipLevels = 1;
imageInfo.arrayLayers = 1;
imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT |
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

VkImage vkImage;
vkCreateImage(device, &imageInfo, nullptr, &vkImage);

// Allocate memory with export info
VkExportMemoryAllocateInfoKHR exportAllocInfo = {};
exportAllocInfo.sType =
    VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
exportAllocInfo.handleTypes =
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;

VkMemoryRequirements memReqs;
vkGetImageMemoryRequirements(device, vkImage, &memReqs);

VkMemoryAllocateInfo allocInfo = {};
allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
allocInfo.pNext = &exportAllocInfo;
allocInfo.allocationSize = memReqs.size;
allocInfo.memoryTypeIndex = /* find suitable memory type */;

VkDeviceMemory vkMemory;
vkAllocateMemory(device, &allocInfo, nullptr, &vkMemory);
vkBindImageMemory(device, vkImage, vkMemory, 0);

// Export the memory as a file descriptor
VkMemoryGetFdInfoKHR getFdInfo = {};
getFdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
getFdInfo.memory = vkMemory;
getFdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;

int fd;
vkGetMemoryFdKHR(device, &getFdInfo, &fd);
```

### 22.2.4 Importing Vulkan Memory into CUDA

```cpp
// Import the Vulkan-exported file descriptor into CUDA
cudaExternalMemory_t extMem;
cudaExternalMemoryHandleDesc memHandleDesc = {};
memHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
memHandleDesc.handle.fd = fd;           // fd from vkGetMemoryFdKHR
memHandleDesc.size = allocationSize;    // Must match Vulkan allocation size
memHandleDesc.flags = 0;

cudaImportExternalMemory(&extMem, &memHandleDesc);
// fd is now consumed by CUDA

// Map as a linear buffer (for raw access)
void* devPtr = nullptr;
cudaExternalMemoryBufferDesc bufDesc = {};
bufDesc.offset = 0;
bufDesc.size = allocationSize;
cudaExternalMemoryGetMappedBuffer(&devPtr, extMem, &bufDesc);

// Or map as a CUDA array (for texture/surface operations)
cudaMipmappedArray_t mipmap = nullptr;
cudaExternalMemoryMipmappedArrayDesc mipmapDesc = {};
mipmapDesc.offset = 0;
mipmapDesc.formatDesc = {32, 32, 32, 32, cudaChannelFormatKindFloat};
mipmapDesc.extent = make_cudaExtent(1920, 1080, 0);
mipmapDesc.flags = cudaArraySurfaceLoadStore;
mipmapDesc.numLevels = 1;
cudaExternalMemoryGetMappedMipmappedArray(&mipmap, extMem, &mipmapDesc);
```

### 22.2.5 Exporting and Importing Vulkan Semaphores

```cpp
// Vulkan: Create exportable semaphore
VkExportSemaphoreCreateInfoKHR exportSemInfo = {};
exportSemInfo.sType =
    VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR;
exportSemInfo.handleTypes =
    VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;

VkSemaphoreCreateInfo semInfo = {};
semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
semInfo.pNext = &exportSemInfo;

VkSemaphore vkSemaphore;
vkCreateSemaphore(device, &semInfo, nullptr, &vkSemaphore);

// Export to file descriptor
VkSemaphoreGetFdInfoKHR getSemFdInfo = {};
getSemFdInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
getSemFdInfo.semaphore = vkSemaphore;
getSemFdInfo.handleType =
    VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;

int semFd;
vkGetSemaphoreFdKHR(device, &getSemFdInfo, &semFd);

// Import into CUDA
cudaExternalSemaphore_t cudaSem;
cudaExternalSemaphoreHandleDesc semHandleDesc = {};
semHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
semHandleDesc.handle.fd = semFd;
semHandleDesc.flags = 0;
cudaImportExternalSemaphore(&cudaSem, &semHandleDesc);
```

### 22.2.6 Synchronization Between CUDA and Vulkan

Use the imported semaphore to coordinate execution between CUDA and Vulkan. The typical pattern is:

1. Vulkan renders a frame and signals the semaphore.
2. CUDA waits on the semaphore, processes the frame, and signals another semaphore.
3. Vulkan waits on the second semaphore and presents the frame.

```cpp
// CUDA waits for Vulkan to finish rendering
cudaExternalSemaphoreWaitParams waitParams = {};
waitParams.flags = 0;
// For opaque fd semaphores, no fence value is needed
cudaWaitExternalSemaphoresAsync(&cudaSem, &waitParams, 1, cudaStream);

// Launch CUDA processing kernel
process_frame<<<grid, block, 0, cudaStream>>>(devPtr, width, height);

// Signal Vulkan that CUDA is done
cudaExternalSemaphoreSignalParams signalParams = {};
signalParams.flags = 0;
cudaSignalExternalSemaphoresAsync(&cudaSem, &signalParams, 1, cudaStream);
```

For timeline semaphores (Vulkan 1.2+), use fence values:

```cpp
// Wait for Vulkan to reach a specific timeline point
cudaExternalSemaphoreWaitParams waitParams = {};
waitParams.params.fence.value = timelineValue;
waitParams.flags = 0;
cudaWaitExternalSemaphoresAsync(&cudaSem, &waitParams, 1, cudaStream);

// Signal a new timeline point from CUDA
cudaExternalSemaphoreSignalParams signalParams = {};
signalParams.params.fence.value = timelineValue + 1;
signalParams.flags = 0;
cudaSignalExternalSemaphoresAsync(&cudaSem, &signalParams, 1, cudaStream);
```

### 22.2.7 Complete Vulkan-CUDA Interop Example

```cpp
#include <cuda_runtime.h>
#include <vulkan/vulkan.h>
#include <stdio.h>

// Full pipeline: Vulkan render -> CUDA process -> Vulkan present
void vulkan_cuda_pipeline(
    VkDevice vkDevice,
    VkPhysicalDevice vkPhysicalDevice,
    VkDeviceMemory vkMemory,
    VkSemaphore vkRenderComplete,
    VkSemaphore vkPresentReady,
    size_t allocationSize)
{
    // Step 1: Match devices
    int cudaDevice = find_cuda_device_for_vulkan(vkPhysicalDevice);
    cudaSetDevice(cudaDevice);

    // Step 2: Import memory
    VkMemoryGetFdInfoKHR memFdInfo = {};
    memFdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    memFdInfo.memory = vkMemory;
    memFdInfo.handleType =
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
    int memFd;
    vkGetMemoryFdKHR(vkDevice, &memFdInfo, &memFd);

    cudaExternalMemory_t extMem;
    cudaExternalMemoryHandleDesc memDesc = {};
    memDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    memDesc.handle.fd = memFd;
    memDesc.size = allocationSize;
    cudaImportExternalMemory(&extMem, &memDesc);

    void* devPtr;
    cudaExternalMemoryBufferDesc bufDesc = {};
    bufDesc.offset = 0;
    bufDesc.size = allocationSize;
    cudaExternalMemoryGetMappedBuffer(&devPtr, extMem, &bufDesc);

    // Step 3: Import semaphores
    // (Similar pattern: export from Vulkan, import into CUDA)

    // Step 4: Processing loop
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    for (int frame = 0; frame < 100; frame++) {
        // Wait for Vulkan render completion
        cudaExternalSemaphoreWaitParams waitParams = {};
        waitParams.flags = 0;
        cudaWaitExternalSemaphoresAsync(
            &renderCompleteSem, &waitParams, 1, stream);

        // Process with CUDA
        process_kernel<<<grid, block, 0, stream>>>(
            devPtr, 1920, 1080);

        // Signal that CUDA is done
        cudaExternalSemaphoreSignalParams signalParams = {};
        signalParams.flags = 0;
        cudaSignalExternalSemaphoresAsync(
            &presentReadySem, &signalParams, 1, stream);
    }

    cudaStreamSynchronize(stream);

    // Step 5: Cleanup
    cudaStreamDestroy(stream);
    cudaFree(devPtr);
    cudaDestroyExternalMemory(extMem);
    cudaDestroyExternalSemaphore(renderCompleteSem);
    cudaDestroyExternalSemaphore(presentReadySem);
}
```

---

## 22.3 Direct3D Interoperability

CUDA-Direct3D interoperability enables sharing of resources between CUDA and Direct3D 11/12. This is primarily used on Windows platforms for applications that combine D3D rendering with CUDA compute.

### 22.3.1 Device Matching by LUID

Unlike Vulkan (which uses UUIDs), Direct3D devices are matched to CUDA devices using the Locally Unique Identifier (LUID). The LUID is a system-assigned identifier that uniquely identifies a GPU adapter on the local machine.

```cpp
#include <cuda_runtime.h>
#include <d3d12.h>

int find_cuda_device_for_d3d12(IDXGIAdapter* adapter) {
    // Get the adapter LUID from DXGI
    DXGI_ADAPTER_DESC adapterDesc;
    adapter->GetDesc(&adapterDesc);

    // Search CUDA devices for matching LUID
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        // CUDA provides LUID in the device properties
        if (memcmp(prop.luid, &adapterDesc.AdapterLuid, sizeof(LUID)) == 0) {
            return i;
        }
    }
    return -1;
}
```

### 22.3.2 Memory Import Methods

There are three methods for importing D3D12 memory into CUDA, each using a different handle type:

#### Method 1: D3D12 Heap NT Handle

Import an entire D3D12 heap as external memory using its NT handle. This provides access to the full heap allocation.

```cpp
// D3D12: Create a heap with shared access
D3D12_HEAP_DESC heapDesc = {};
heapDesc.SizeInBytes = bufferSize;
heapDesc.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
heapDesc.Flags = D3D12_HEAP_FLAG_SHARED;

ID3D12Heap* d3dHeap;
device->CreateHeap(&heapDesc, IID_PPV_ARGS(&d3dHeap));

// Export the heap as an NT handle
HANDLE sharedHandle;
device->CreateSharedHandle(d3dHeap, nullptr,
    GENERIC_ALL, nullptr, &sharedHandle);

// Import into CUDA
cudaExternalMemory_t extMem;
cudaExternalMemoryHandleDesc handleDesc = {};
handleDesc.type = cudaExternalMemoryHandleTypeD3D12Heap;
handleDesc.handle.win32.handle = sharedHandle;
handleDesc.size = bufferSize;
handleDesc.flags = 0;
cudaImportExternalMemory(&extMem, &handleDesc);

// Map as buffer
void* devPtr;
cudaExternalMemoryBufferDesc bufDesc = {};
bufDesc.offset = 0;
bufDesc.size = bufferSize;
cudaExternalMemoryGetMappedBuffer(&devPtr, extMem, &bufDesc);
```

#### Method 2: Named Handle (D3D12 Resource)

Import a D3D12 committed resource using a named shared handle. This is useful when resources need to be shared across security boundaries.

```cpp
// D3D12: Create a committed resource with shared access
D3D12_RESOURCE_DESC resourceDesc = {};
resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
resourceDesc.Width = bufferSize;
resourceDesc.Height = 1;
resourceDesc.DepthOrArraySize = 1;
resourceDesc.MipLevels = 1;
resourceDesc.Format = DXGI_FORMAT_UNKNOWN;
resourceDesc.SampleDesc.Count = 1;
resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

ID3D12Resource* d3dResource;
device->CreateCommittedResource(
    &heapProperties, D3D12_HEAP_FLAG_SHARED,
    &resourceDesc, D3D12_RESOURCE_STATE_COMMON,
    nullptr, IID_PPV_ARGS(&d3dResource));

// Export with a name
LPCWSTR sharedName = L"MySharedResource";
HANDLE namedHandle;
device->CreateSharedHandle(d3dResource, nullptr,
    GENERIC_ALL, sharedName, &namedHandle);

// Import into CUDA using named handle
cudaExternalMemoryHandleDesc handleDesc = {};
handleDesc.type = cudaExternalMemoryHandleTypeD3D12Resource;
handleDesc.handle.win32.name = sharedName;
handleDesc.size = bufferSize;
cudaImportExternalMemory(&extMem, &handleDesc);
```

#### Method 3: Committed Resource NT Handle

Import a D3D12 committed resource using its NT handle (without a name).

```cpp
// Similar to Method 2, but use the NT handle directly
cudaExternalMemoryHandleDesc handleDesc = {};
handleDesc.type = cudaExternalMemoryHandleTypeD3D12Resource;
handleDesc.handle.win32.handle = sharedHandle;
handleDesc.size = bufferSize;
cudaImportExternalMemory(&extMem, &handleDesc);
```

### 22.3.3 Fence Import for Synchronization

D3D12 fences can be imported as CUDA external semaphores for cross-API synchronization. This uses the `cudaExternalSemaphoreHandleTypeD3D12Fence` handle type, which requires a fence value for signal and wait operations.

```cpp
// D3D12: Create a shared fence
ID3D12Fence* d3dFence;
device->CreateFence(0, D3D12_FENCE_FLAG_SHARED,
    IID_PPV_ARGS(&d3dFence));

HANDLE fenceHandle;
device->CreateSharedHandle(d3dFence, nullptr,
    GENERIC_ALL, nullptr, &fenceHandle);

// Import into CUDA
cudaExternalSemaphore_t cudaSem;
cudaExternalSemaphoreHandleDesc semDesc = {};
semDesc.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
semDesc.handle.win32.handle = fenceHandle;
semDesc.flags = 0;
cudaImportExternalSemaphore(&cudaSem, &semDesc);

// Wait for D3D12 to signal fence value 1
cudaExternalSemaphoreWaitParams waitParams = {};
waitParams.params.fence.value = 1;
waitParams.flags = 0;
cudaWaitExternalSemaphoresAsync(&cudaSem, &waitParams, 1, stream);

// Signal from CUDA with value 2 (D3D12 can wait on this)
cudaExternalSemaphoreSignalParams signalParams = {};
signalParams.params.fence.value = 2;
signalParams.flags = 0;
cudaSignalExternalSemaphoresAsync(&cudaSem, &signalParams, 1, stream);
```

### 22.3.4 D3D11 Interoperability

D3D11 resources and fences can also be imported using `cudaExternalMemoryHandleTypeD3D11Resource` and `cudaExternalSemaphoreHandleTypeD3D11Fence` respectively. The pattern is similar to D3D12 but uses D3D11-specific handle extraction.

```cpp
// D3D11: Get the shared handle from a texture
IDXGIResource* dxgiResource;
d3d11Texture->QueryInterface(
    IID_PPV_ARGS(&dxgiResource));
HANDLE sharedHandle;
dxgiResource->GetSharedHandle(&sharedHandle);
dxgiResource->Release();

// Import into CUDA
cudaExternalMemoryHandleDesc handleDesc = {};
handleDesc.type = cudaExternalMemoryHandleTypeD3D11Resource;
handleDesc.handle.win32.handle = sharedHandle;
handleDesc.size = allocationSize;
cudaImportExternalMemory(&extMem, &handleDesc);
```

---

## 22.4 NVSCI Interoperability

NVIDIA's System Controller Interface (NvSci) provides a framework for zero-copy buffer sharing and synchronization across different computing engines (GPU, ISP, VIC, display, etc.) on NVIDIA Tegra and other embedded platforms. CUDA supports NvSciBuf for memory sharing and NvSciSync for synchronization.

### 22.4.1 NvSciBuf: Buffer Sharing

NvSciBuf enables allocation and exchange of memory buffers across different engines without copying. Each engine describes its requirements, and NvSciBuf reconciles them to find a compatible allocation.

#### Key Buffer Attributes

| Attribute | Description |
|-----------|-------------|
| `NvSciBufGeneralAttrKey_GpuId` | Specifies which GPU the buffer is intended for |
| `NvSciBufGeneralAttrKey_NeedCpuAccess` | Whether CPU access is required |
| `NvSciBufGeneralAttrKey_Align` | Alignment requirement in bytes |
| `NvSciBufGeneralAttrKey_RequiredPerm` | Access permissions (read, write, readwrite) |
| `NvSciBufGeneralAttrKey_EnableGpuCache` | Enable GPU cache for the buffer |
| `NvSciBufGeneralAttrKey_EnableGpuCompression` | Enable GPU compression for the buffer |

#### CUDA-NvSciBuf Workflow

```cpp
#include <nvscibuf.h>
#include <cuda_runtime.h>

// Step 1: Create NvSciBuf attribute list
NvSciBufAttrList attrList;
NvSciBufAttrListCreate(sciBufModule, &attrList);

// Step 2: Set CUDA-specific attributes
NvSciBufAttrKeyValuePair keyvals[] = {
    { NvSciBufGeneralAttrKey_GpuId, &gpuId, sizeof(gpuId) },
    { NvSciBufGeneralAttrKey_NeedCpuAccess, &cpuAccess, sizeof(cpuAccess) },
    { NvSciBufGeneralAttrKey_Align, &alignment, sizeof(alignment) },
    { NvSciBufGeneralAttrKey_RequiredPerm,
      &requiredPerm, sizeof(requiredPerm) },
    { NvSciBufGeneralAttrKey_EnableGpuCache,
      &gpuCache, sizeof(gpuCache) },
    { NvSciBufGeneralAttrKey_EnableGpuCompression,
      &gpuCompression, sizeof(gpuCompression) },
};
NvSciBufAttrListSetAttrs(attrList, keyvals,
    sizeof(keyvals) / sizeof(keyvals[0]));

// Step 3: Reconcile attribute lists (if multiple engines)
NvSciBufAttrList reconciledList;
NvSciBufAttrListReconcile(&attrList, 1, &reconciledList, nullptr);

// Step 4: Allocate the buffer
NvSciBufObj bufObj;
NvSciBufObjAlloc(reconciledList, &bufObj);

// Step 5: Import into CUDA
cudaExternalMemory_t extMem;
cudaExternalMemoryHandleDesc handleDesc = {};
handleDesc.type = cudaExternalMemoryHandleTypeNvSciBuf;
handleDesc.handle.nvSciBufObject = bufObj;
handleDesc.size = bufferSize;
handleDesc.flags = 0;
cudaImportExternalMemory(&extMem, &handleDesc);

// Step 6: Map as a linear buffer
void* devPtr;
cudaExternalMemoryBufferDesc bufDesc = {};
bufDesc.offset = 0;
bufDesc.size = bufferSize;
cudaExternalMemoryGetMappedBuffer(&devPtr, extMem, &bufDesc);

// Step 7: Use in CUDA kernel
process_buffer<<<grid, block>>>(devPtr, bufferSize);
```

### 22.4.2 NvSciSync: Synchronization

NvSciSync provides cross-engine synchronization at operation boundaries. It allows one engine to signal completion and another engine to wait before starting its work.

#### CUDA-NvSciSync Workflow

```cpp
#include <nvscisync.h>
#include <cuda_runtime.h>

// Step 1: Create NvSciSync attributes
NvSciSyncAttrList syncAttrList;
NvSciSyncAttrListCreate(sciSyncModule, &syncAttrList);

// Step 2: Set CUDA-specific sync attributes
bool requireCpuWait = false;
NvSciSyncAttrKeyValuePair syncKeyvals[] = {
    { NvSciSyncAttrKey_NeedCpuWait, &requireCpuWait,
      sizeof(requireCWait) },
};
NvSciSyncAttrListSetAttrs(syncAttrList, syncKeyvals, 1);

// Step 3: Create the sync object
NvSciSyncObj syncObj;
NvSciSyncAttrList reconciledSyncList;
NvSciSyncAttrListReconcile(&syncAttrList, 1,
    &reconciledSyncList, nullptr);
NvSciSyncObjCreate(reconciledSyncList, &syncObj);

// Step 4: Import into CUDA
cudaExternalSemaphore_t cudaSem;
cudaExternalSemaphoreHandleDesc semDesc = {};
semDesc.type = cudaExternalSemaphoreHandleTypeNvSciSync;
semDesc.handle.nvSciSyncObj = syncObj;
semDesc.flags = 0;
cudaImportExternalSemaphore(&cudaSem, &semDesc);

// Step 5: Use for synchronization
cudaExternalSemaphoreWaitParams waitParams = {};
waitParams.params.nvSciSync.fence = &preFence;
waitParams.flags = cudaExternalSemaphoreWaitSkipNvSciBufBufSync;
cudaWaitExternalSemaphoresAsync(&cudaSem, &waitParams, 1, stream);

cudaExternalSemaphoreSignalParams signalParams = {};
signalParams.flags = 0;
cudaSignalExternalSemaphoresAsync(&cudaSem, &signalParams, 1, stream);
```

### 22.4.3 NvSciBuf Attribute Details

| Attribute | Type | Description |
|-----------|------|-------------|
| `GpuId` | `uint32_t` | GPU device ID for CUDA access |
| `NeedCpuAccess` | `bool` | Set to `true` if CPU needs to read/write the buffer |
| `Align` | `uint64_t` | Minimum alignment in bytes (e.g., 256 for optimal GPU access) |
| `RequiredPerm` | `NvSciBufAccessPerm` | `Read`, `Write`, `ReadWrite`, or `DontCare` |
| `EnableGpuCache` | `bool` | Allow GPU to cache buffer contents |
| `EnableGpuCompression` | `bool` | Allow GPU to compress buffer data (reduces bandwidth) |

---

## 22.5 OpenGL Interoperability (via Vulkan)

Direct OpenGL-CUDA interop through the legacy API (e.g., `cudaGraphicsGLRegisterBuffer`) has limitations and platform-specific quirks. The recommended modern approach is to use OpenGL's external memory and semaphore extensions, which provide Vulkan-compatible handles that can be shared with CUDA.

### 22.5.1 Required Extensions

| Extension | Purpose |
|-----------|---------|
| `GL_EXT_memory_object` | Create and manage external memory objects |
| `GL_EXT_memory_object_fd` | Import/export memory via file descriptors (Linux) |
| `GL_EXT_memory_object_win32` | Import/export memory via handles (Windows) |
| `GL_EXT_semaphore` | Create and manage external semaphore objects |
| `GL_EXT_semaphore_fd` | Import/export semaphores via file descriptors |
| `GL_EXT_semaphore_win32` | Import/export semaphores via handles |

### 22.5.2 OpenGL to CUDA via External Memory

```cpp
#include <GL/gl.h>
#include <GL/glext.h>

// Step 1: Create an OpenGL texture
GLuint glTexture;
glGenTextures(1, &glTexture);
glBindTexture(GL_TEXTURE_2D, glTexture);
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height,
             0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

// Step 2: Create a memory object from the texture
GLuint glMemObject;
glCreateMemoryObjectsEXT(1, &glMemObject);

// Step 3: Allocate storage with the memory object
// (Or import from an external fd)
GLint dedicated = GL_TRUE;
glMemoryObjectParameterivEXT(glMemObject,
    GL_DEDICATED_MEMORY_OBJECT_EXT, &dedicated);

// Step 4: Import external memory (from CUDA or another process)
// Export from CUDA first:
//   cudaExternalMemoryHandleDesc -> get fd
// Then import into OpenGL:
glImportMemoryFdEXT(glMemObject, allocationSize,
    GL_HANDLE_TYPE_OPAQUE_FD_EXT, fd);

// Step 5: Associate the memory object with the texture
glTexStorageMem2DEXT(glTexture, 1, GL_RGBA8,
    width, height, glMemObject, 0);
```

### 22.5.3 OpenGL Semaphore Synchronization

```cpp
// Create an OpenGL semaphore
GLuint glSemaphore;
glCreateSemaphoresEXT(1, &glSemaphore);

// Import from CUDA-exported semaphore fd
glImportSemaphoreFdEXT(glSemaphore,
    GL_HANDLE_TYPE_OPAQUE_FD_EXT, semaphoreFd);

// Wait for CUDA to finish before OpenGL uses the texture
GLenum srcLayout = GL_LAYOUT_GENERAL_EXT;
glWaitSemaphoreEXT(glSemaphore, 0, nullptr, 1,
    &glTexture, &srcLayout);

// Signal from OpenGL after rendering
GLenum dstLayout = GL_LAYOUT_GENERAL_EXT;
glSignalSemaphoreEXT(glSemaphore, 0, nullptr, 1,
    &glTexture, &dstLayout);
```

### 22.5.4 Full OpenGL-CUDA-CUDA Pipeline Example

```cpp
// CUDA side: Export memory and semaphore
void setup_cuda_exports(int* memFd, int* semFd, void** devPtr) {
    // Allocate CUDA device memory
    cudaExternalMemory_t extMem;
    // ... (create allocation, export as fd)
    *memFd = exportedMemFd;

    // Create and export semaphore
    cudaExternalSemaphore_t extSem;
    // ... (create semaphore, export as fd)
    *semFd = exportedSemFd;

    // Map the buffer for CUDA use
    cudaExternalMemoryBufferDesc bufDesc = {};
    bufDesc.offset = 0;
    bufDesc.size = width * height * 4;
    cudaExternalMemoryGetMappedBuffer(devPtr, extMem, &bufDesc);
}

// OpenGL side: Import and use
void setup_gl_imports(int memFd, int semFd, GLuint* texture) {
    // Import memory
    GLuint memObj;
    glCreateMemoryObjectsEXT(1, &memObj);
    glImportMemoryFdEXT(memObj, width * height * 4,
        GL_HANDLE_TYPE_OPAQUE_FD_EXT, memFd);

    // Create texture backed by imported memory
    glGenTextures(1, texture);
    glBindTexture(GL_TEXTURE_2D, *texture);
    glTexStorageMem2DEXT(*texture, 1, GL_RGBA8,
        width, height, memObj, 0);

    // Import semaphore
    GLuint semaphore;
    glCreateSemaphoresEXT(1, &semaphore);
    glImportSemaphoreFdEXT(semaphore,
        GL_HANDLE_TYPE_OPAQUE_FD_EXT, semFd);
}

// Render loop: CUDA processes -> OpenGL renders
void render_loop(cudaExternalSemaphore_t cudaSem,
                 GLuint glSem, GLuint texture, void* devPtr) {
    // CUDA writes to the buffer
    generate_frame<<<grid, block>>>((uchar4*)devPtr, width, height);

    // Signal from CUDA
    cudaExternalSemaphoreSignalParams sigParams = {};
    sigParams.flags = 0;
    cudaSignalExternalSemaphoresAsync(&cudaSem, &sigParams, 1, stream);

    // OpenGL waits and uses the texture
    GLenum layout = GL_LAYOUT_GENERAL_EXT;
    glWaitSemaphoreEXT(glSem, 0, nullptr, 1, &texture, &layout);

    // Render with the texture...
    draw_textured_quad(texture);
}
```

---

## 22.6 General Interoperability Patterns

### 22.6.1 Resource Lifetime Rules

When sharing resources across APIs, observe these lifetime rules:

1. **Imported handles are consumed.** After `cudaImportExternalMemory()` or `cudaImportExternalSemaphore()`, the original handle (fd, NT handle) belongs to CUDA. Do not close or reuse it.

2. **Mapped resources must be freed before external memory.** Call `cudaFree()` on mapped buffers and `cudaFreeMipmappedArray()` on mapped arrays before calling `cudaDestroyExternalMemory()`.

3. **Synchronize before destroying.** Ensure all CUDA work is complete (via `cudaStreamSynchronize()` or similar) before destroying shared resources.

4. **Handle the ordering carefully.** The producer must signal before the consumer waits. Use semaphores to enforce this ordering across API boundaries.

### 22.6.2 Performance Considerations

| Consideration | Recommendation |
|---------------|----------------|
| **Memory type** | Use optimal tiling for images; use linear for buffers accessed by address |
| **Dedicated allocations** | Use dedicated allocations (Vulkan: `VkDedicatedAllocationMemoryAllocateInfo`) for single-resource exports to avoid alignment issues |
| **Cache coherency** | Ensure cache flushes between API transitions; CUDA's `cudaDeviceSynchronize()` or semaphore wait/signal handles this |
| **Copy avoidance** | The primary benefit of interop is zero-copy sharing; avoid reading back to host and re-uploading |
| **Format matching** | Ensure the pixel format (channel format, bit depth) matches between APIs to avoid format conversion |

### 22.6.3 Common Pitfalls

1. **Size mismatches:** The `size` parameter in `cudaExternalMemoryHandleDesc` must exactly match the size of the external allocation. A mismatch causes import failure.

2. **Missing synchronization:** Without proper semaphore signaling and waiting, one API may access memory while the other is still writing, causing data corruption.

3. **Device mismatch:** Always verify that the CUDA device and the external API device refer to the same physical GPU (by UUID or LUID).

4. **Handle leaks:** Exported handles (fd, NT handles) must be closed by the importing party. After importing into CUDA, the original handle is consumed. On Windows, use `CloseHandle()` for any remaining copies.

5. **Format mismatch:** A `GL_RGBA8` OpenGL texture maps to `cudaChannelFormatKindUnsigned` with 8 bits per component in CUDA. Mismatches cause incorrect data interpretation.
