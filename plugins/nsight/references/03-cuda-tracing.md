# NVIDIA Nsight Systems -- CUDA Tracing Reference

This document provides a comprehensive reference for CUDA tracing features in Nsight Systems, including configuration options, memory tracing, CUDA graphs, Python backtraces, and complete default function trace lists.

## 1. Basic CUDA Trace Configuration

Nsight Systems is capable of capturing information about CUDA execution in the profiled process. The following information can be collected and presented on the timeline:

- **CUDA API trace** -- trace of CUDA Runtime and CUDA Driver calls made by the application
  - CUDA Runtime calls typically start with `cuda` prefix (e.g., `cudaLaunch`)
  - CUDA Driver calls typically start with `cu` prefix (e.g., `cuDeviceGetCount`)
- **CUDA workload trace** -- trace of activity happening on the GPU, including memory operations (e.g., Host-to-Device memory copies) and kernel executions
- On Workstation Edition: cuDNN, cuBLAS, and OpenACC API tracing

### 1.1 Enabling CUDA Trace

**CLI:**
```bash
nsys profile --trace=cuda <application>
```

**GUI:**
Select the "Collect CUDA trace" checkbox in the project settings.

### 1.2 Timeline Layout

Within the threads that use the CUDA API, additional child rows will appear in the timeline tree. Near the bottom of the timeline row tree, the GPU node will appear and contain a CUDA node. Within the CUDA node, each CUDA context used within the process will be shown along with its corresponding CUDA streams. Streams will contain memory operations and kernel launches on the GPU. Kernel launches are represented by blue, while memory transfers are displayed in red.

### 1.3 Additional Configuration Parameters

| Parameter | CLI Option | Description |
|-----------|-----------|-------------|
| CUDA API backtraces | `--cudabacktrace` | Turns on collection of CUDA API backtraces. Sets minimum time a CUDA API event must take before backtraces are collected. Setting too low causes high overhead. |
| Flush data periodically | `--cuda-flush-interval` | Specifies the period after which an attempt to flush CUDA trace data will be made. For collections over 30 seconds, 10 seconds recommended. |
| Skip some API calls | Default behavior | Avoids tracing insignificant CUDA Runtime API calls (cudaConfigureCall, cudaSetupArgument, cudaHostGetDevicePointers). Significantly reduces overhead. |
| Trace all APIs | `--cuda-trace-all-apis` | Forces tracing of all CUDA APIs, not just performance-relevant subset. |
| GPU Memory Usage | `--cuda-memory-usage` | Collects information for CUDA allocated memory graph. Increases overhead. |
| Unified Memory CPU page faults | `--cuda-um-cpu-page-faults` | Tracks page faults when CPU accesses device-resident memory. Not on Embedded Platforms Edition. |
| Unified Memory GPU page faults | `--cuda-um-gpu-page-faults` | Tracks page faults when GPU accesses host-resident memory. Not on Embedded Platforms Edition. |
| CUDA Graph trace | `--cuda-graph-trace` | Set to `graph` (whole-graph tracing) or `node` (per-node tracing). |
| cuDNN trace | `--trace=cudnn` | Trace cuDNN API calls (Workstation Edition, not on Windows). |
| cuBLAS trace | `--trace=cublas` or `--trace=cublas-verbose` | Trace cuBLAS API calls. |
| OpenACC trace | `--trace=openacc` | Trace OpenACC execution. Automatically enables CUDA. |

> **Note:** If your application crashes before all collected CUDA trace data has been copied out, some or all data might be lost.

> **Note:** Nsight Systems will not have information about CUDA events that were still in device buffers when analysis terminated. Call cudaDeviceReset before ending analysis when using cudaProfilerAPI.

---

## 2. CUDA Backtrace Options

When tracing CUDA APIs, Nsight Systems can collect backtraces when CUDA APIs are invoked. This is controlled by the `--cudabacktrace` option.

**CLI:**
```bash
nsys profile --cudabacktrace=<value> <application>
```

### 2.1 Backtrace Values

| Value | Description |
|-------|-------------|
| `all` | Collect backtraces for all CUDA API categories |
| `none` | No CUDA backtrace collection (default) |
| `kernel` | Collect backtraces for kernel launch APIs (cudaLaunchKernel, cuLaunchKernel, etc.) |
| `memory` | Collect backtraces for memory operation APIs (cudaMalloc, cudaMemcpy, etc.) |
| `sync` | Collect backtraces for synchronization APIs (cudaDeviceSynchronize, cudaStreamSynchronize, etc.) |
| `other` | Collect backtraces for all other CUDA APIs not in above categories |

### 2.2 Threshold Configuration

Values may be combined using commas. Each value (except `none`) may be appended with a threshold after `:`. The threshold is the duration in nanoseconds that CUDA APIs must execute before backtraces are collected.

```bash
# Collect kernel backtraces for APIs taking > 500ns
nsys profile --cudabacktrace=kernel:500 <application>

# Collect both kernel and memory backtraces with different thresholds
nsys profile --cudabacktrace=kernel:500,memory:2000 <application>

# Default threshold is 1000ns (1us) for each category
nsys profile --cudabacktrace=all <application>
```

> **Note:** Significant runtime overhead may occur with backtrace collection. CPU sampling must be enabled.

---

## 3. CUDA Memory Usage Tracking

The `--cuda-memory-usage` option tracks GPU memory usage by CUDA kernels.

**CLI:**
```bash
nsys profile --cuda-memory-usage=true --trace=cuda <application>
```

**GUI:**
Select "Collect GPU Memory Usage" from the CUDA trace configuration options.

This is not the same as the GPU memory graph generated during stutter analysis on Windows.

When enabled, Nsight Systems tracks CUDA GPU memory allocations and deallocations and presents a graph of this information in the timeline. Below, in a report where memory is allocated and freed during the collection, the graph shows allocation and deallocation patterns. If memory is allocated but not freed during the collection, the graph will show a monotonically increasing curve.

> **Note:** This feature may cause significant runtime overhead.

---

## 4. Launching Nsight Compute from a CUDA Kernel Context

After using CUDA trace in Nsight Systems to locate a potential problem kernel, you may want to run NVIDIA Nsight Compute on that specific kernel:

1. Right-click on the kernel in the Nsight Systems GUI to bring up a menu.
2. Select the option to run Nsight Compute.
3. On first use, configure the settings:
   - **Option 1:** Invoke the Nsight Compute UI with known parameters. Provide the location of the `ncu-ui` executable. Nsight Systems verifies path and executable validity.
   - **Option 2:** Generate a command line for running Nsight Compute on the remote target manually. Useful when Nsight Compute is not installed on the host.

Nsight Systems invokes NCU UI with relevant parameters pre-populated from the Nsight Systems run. Users may modify parameters before launching.

---

## 5. CUDA GPU Memory Allocation Graph

When "Collect GPU Memory Usage" is selected, Nsight Systems tracks CUDA GPU memory allocations and deallocations and presents a graph:

- Memory allocated and freed during collection: shows allocation/deallocation patterns
- Memory allocated but not freed: shows monotonically increasing usage
- Allocations on multiple GPUs: shows separate curves for each GPU

---

## 6. Unified Memory Transfer Trace

For Nsight Systems Workstation Edition, Unified Memory (also called Managed Memory) transfer trace is enabled automatically when CUDA trace is selected. It incurs no overhead in programs that do not perform any Unified Memory transfers.

Data is displayed in the Managed Memory area of the timeline.

### 6.1 Transfer Types

| Transfer Type | Description |
|---------------|-------------|
| **HtoD** | CUDA kernel accessed managed memory residing on the host. Kernel execution paused and transferred data to device. Heavy traffic indicates performance penalty. Consider using manual cudaMemcpy from pinned host memory. |
| **PtoP** | CUDA kernel accessed managed memory residing on a different device. Kernel execution paused and transferred data to this device. Heavy traffic indicates performance penalty. Consider using manual cudaMemcpyPeer. The row shows destination device; source device shown in tooltip. |
| **DtoH** | CPU accessed managed memory residing on a CUDA device. CPU execution paused and transferred data to system memory. Heavy traffic indicates performance penalty. Consider using manual cudaMemcpy from pinned host memory. |

### 6.2 Highlighted Transfer Causes

Some Unified Memory transfers are highlighted in red to indicate potential performance issues:

| Migration Cause | Description |
|----------------|-------------|
| **Coherence** | Migration occurred to guarantee data coherence. SMs stop until migration completes. |
| **Eviction** | Memory migrated to CPU because it was evicted to make room for another block on the GPU. Happens due to memory overcommitment (available on Linux with Compute Capability >= 6). |

---

## 7. Unified Memory CPU Page Faults

The Unified Memory CPU page faults feature tracks page faults when CPU code tries to access a memory page that resides on the device.

**CLI:**
```bash
nsys profile --cuda-um-cpu-page-faults=true --trace=cuda <application>
```

**GUI:**
Select "Collect Unified Memory CPU page faults" from the CUDA trace options.

> **Note:** Collecting Unified Memory CPU page faults can cause overhead of up to 70% in testing. Use this functionality only when needed. Not available on Nsight Systems Embedded Platforms Edition.

---

## 8. Unified Memory GPU Page Faults

The Unified Memory GPU page faults feature tracks page faults when GPU code tries to access a memory page that resides on the host.

**CLI:**
```bash
nsys profile --cuda-um-gpu-page-faults=true --trace=cuda <application>
```

**GUI:**
Select "Collect Unified Memory GPU page faults" from the CUDA trace options.

> **Note:** Collecting Unified Memory GPU page faults can cause overhead of up to 70% in testing. Use this functionality only when needed. Not available on Nsight Systems Embedded Platforms Edition.

---

## 9. CUDA Graph Trace

Nsight Systems captures information about CUDA graphs at either the graph or node granularity.

**CLI:**
```bash
# Trace at graph level (recommended, lower overhead)
nsys profile --cuda-graph-trace=graph --trace=cuda <application>

# Trace at node level (higher overhead, more detail)
nsys profile --cuda-graph-trace=node --trace=cuda <application>
```

**GUI:**
Set the CUDA graph trace dropdown in the CUDA trace configuration.

### 9.1 Graph vs Node Tracing

| Mode | Behavior | Requirements |
|------|----------|-------------|
| `graph` | Each graph appears as one item on the timeline. Significantly less overhead. | CUDA driver 515.43 or higher |
| `node` | Each graph appears as a set of individual nodes on the timeline. More detailed but may cause significant runtime overhead. | Any supported CUDA driver |

> **Note:** Default is `graph` if available (driver 515.43+), otherwise `node`.

---

## 10. CUDA Python Backtrace

Nsight Systems for Arm server (SBSA) platforms and x86 Linux targets can capture Python backtrace information when CUDA backtrace is being captured.

**CLI:**
```bash
nsys profile --python-backtrace=cuda --cudabacktrace=all --trace=cuda <application>
```

**GUI:**
Select "Collect Python backtrace for selected API calls" checkbox.

> **Note:** CUDA tracing, CUDA backtraces, and CPU sampling must all be enabled. The `--cudabacktrace` option must be set when using `--python-backtrace=cuda`.

---

## 11. OpenACC Trace

Nsight Systems for Linux x86_64 can capture OpenACC execution information.

- OpenACC versions 2.0, 2.5, and 2.6 are supported
- PGI runtime version 15.7 or later required
- PGI runtime 16.1 or later required to differentiate constructs
- GCC implementation of OpenACC is not currently supported

Under the CPU rows in the timeline tree, each thread using OpenACC will show OpenACC trace information. Clicking an OpenACC API call highlights correlation with underlying CUDA API calls. If the OpenACC API results in GPU work, that is also highlighted.

To capture OpenACC from GUI: select "Collect OpenACC trace" checkbox under CUDA trace configurations. Turning on OpenACC tracing also turns on CUDA tracing.

> **Note:** If your application crashes before all collected OpenACC trace data has been copied out, some or all data might be lost.

---

## 12. CUDA Default Function List for CLI

The following sections list all functions traced by default by Nsight Systems CLI for each CUDA-related API.

### 12.1 CUDA Runtime API Functions Traced by Default

```
cudaBindSurfaceToArray
cudaBindTexture
cudaBindTexture2D
cudaBindTextureToArray
cudaBindTextureToMipmappedArray
cudaConfigureCall
cudaCreateSurfaceObject
cudaCreateTextureObject
cudaD3D10MapResources
cudaD3D10RegisterResource
cudaD3D10UnmapResources
cudaD3D10UnregisterResource
cudaD3D9MapResources
cudaD3D9MapVertexBuffer
cudaD3D9RegisterResource
cudaD3D9RegisterVertexBuffer
cudaD3D9UnmapResources
cudaD3D9UnmapVertexBuffer
cudaD3D9UnregisterResource
cudaD3D9UnregisterVertexBuffer
cudaDestroySurfaceObject
cudaDestroyTextureObject
cudaDeviceReset
cudaDeviceSynchronize
cudaEGLStreamConsumerAcquireFrame
cudaEGLStreamConsumerConnect
cudaEGLStreamConsumerConnectWithFlags
cudaEGLStreamConsumerDisconnect
cudaEGLStreamConsumerReleaseFrame
cudaEGLStreamProducerConnect
cudaEGLStreamProducerDisconnect
cudaEGLStreamProducerReturnFrame
cudaEventCreate
cudaEventCreateFromEGLSync
cudaEventCreateWithFlags
cudaEventDestroy
cudaEventQuery
cudaEventRecord
cudaEventRecord_ptsz
cudaEventSynchronize
cudaFree
cudaFreeArray
cudaFreeHost
cudaFreeMipmappedArray
cudaGLMapBufferObject
cudaGLMapBufferObjectAsync
cudaGLRegisterBufferObject
cudaGLUnmapBufferObject
cudaGLUnmapBufferObjectAsync
cudaGLUnregisterBufferObject
cudaGraphicsD3D10RegisterResource
cudaGraphicsD3D11RegisterResource
cudaGraphicsD3D9RegisterResource
cudaGraphicsEGLRegisterImage
cudaGraphicsGLRegisterBuffer
cudaGraphicsGLRegisterImage
cudaGraphicsMapResources
cudaGraphicsUnmapResources
cudaGraphicsUnregisterResource
cudaGraphicsVDPAURegisterOutputSurface
cudaGraphicsVDPAURegisterVideoSurface
cudaHostAlloc
cudaHostRegister
cudaHostUnregister
cudaLaunch
cudaLaunchCooperativeKernel
cudaLaunchCooperativeKernelMultiDevice
cudaLaunchCooperativeKernel_ptsz
cudaLaunchKernel
cudaLaunchKernel_ptsz
cudaLaunch_ptsz
cudaMalloc
cudaMalloc3D
cudaMalloc3DArray
cudaMallocArray
cudaMallocHost
cudaMallocManaged
cudaMallocMipmappedArray
cudaMallocPitch
cudaMemGetInfo
cudaMemPrefetchAsync
cudaMemPrefetchAsync_ptsz
cudaMemcpy
cudaMemcpy2D
cudaMemcpy2DArrayToArray
cudaMemcpy2DArrayToArray_ptds
cudaMemcpy2DAsync
cudaMemcpy2DAsync_ptsz
cudaMemcpy2DFromArray
cudaMemcpy2DFromArrayAsync
cudaMemcpy2DFromArrayAsync_ptsz
cudaMemcpy2DFromArray_ptds
cudaMemcpy2DToArray
cudaMemcpy2DToArrayAsync
cudaMemcpy2DToArrayAsync_ptsz
cudaMemcpy2DToArray_ptds
cudaMemcpy2D_ptds
cudaMemcpy3D
cudaMemcpy3DAsync
cudaMemcpy3DAsync_ptsz
cudaMemcpy3DPeer
cudaMemcpy3DPeerAsync
cudaMemcpy3DPeerAsync_ptsz
cudaMemcpy3DPeer_ptds
cudaMemcpy3D_ptds
cudaMemcpyArrayToArray
cudaMemcpyArrayToArray_ptds
cudaMemcpyAsync
cudaMemcpyAsync_ptsz
cudaMemcpyFromArray
cudaMemcpyFromArrayAsync
cudaMemcpyFromArrayAsync_ptsz
cudaMemcpyFromArray_ptds
cudaMemcpyFromSymbol
cudaMemcpyFromSymbolAsync
cudaMemcpyFromSymbolAsync_ptsz
cudaMemcpyFromSymbol_ptds
cudaMemcpyPeer
cudaMemcpyPeerAsync
cudaMemcpyToArray
cudaMemcpyToArrayAsync
cudaMemcpyToArrayAsync_ptsz
cudaMemcpyToArray_ptds
cudaMemcpyToSymbol
cudaMemcpyToSymbolAsync
cudaMemcpyToSymbolAsync_ptsz
cudaMemcpyToSymbol_ptds
cudaMemcpy_ptds
cudaMemset
cudaMemset2D
cudaMemset2DAsync
cudaMemset2DAsync_ptsz
cudaMemset2D_ptds
cudaMemset3D
cudaMemset3DAsync
cudaMemset3DAsync_ptsz
cudaMemset3D_ptds
cudaMemsetAsync
cudaMemsetAsync_ptsz
cudaMemset_ptds
cudaPeerRegister
cudaPeerUnregister
cudaStreamAddCallback
cudaStreamAddCallback_ptsz
cudaStreamAttachMemAsync
cudaStreamAttachMemAsync_ptsz
cudaStreamCreate
cudaStreamCreateWithFlags
cudaStreamCreateWithPriority
cudaStreamDestroy
cudaStreamQuery
cudaStreamQuery_ptsz
cudaStreamSynchronize
cudaStreamSynchronize_ptsz
cudaStreamWaitEvent
cudaStreamWaitEvent_ptsz
cudaThreadSynchronize
cudaUnbindTexture
```

### 12.2 CUDA Primary (Driver) API Functions Traced by Default

```
cu64Array3DCreate
cu64ArrayCreate
cu64D3D9MapVertexBuffer
cu64GLMapBufferObject
cu64GLMapBufferObjectAsync
cu64MemAlloc
cu64MemAllocPitch
cu64MemFree
cu64MemGetInfo
cu64MemHostAlloc
cu64Memcpy2D
cu64Memcpy2DAsync
cu64Memcpy2DUnaligned
cu64Memcpy3D
cu64Memcpy3DAsync
cu64MemcpyAtoD
cu64MemcpyDtoA
cu64MemcpyDtoD
cu64MemcpyDtoDAsync
cu64MemcpyDtoH
cu64MemcpyDtoHAsync
cu64MemcpyHtoD
cu64MemcpyHtoDAsync
cu64MemsetD16
cu64MemsetD16Async
cu64MemsetD2D16
cu64MemsetD2D16Async
cu64MemsetD2D32
cu64MemsetD2D32Async
cu64MemsetD2D8
cu64MemsetD2D8Async
cu64MemsetD32
cu64MemsetD32Async
cu64MemsetD8
cu64MemsetD8Async
cuArray3DCreate
cuArray3DCreate_v2
cuArrayCreate
cuArrayCreate_v2
cuArrayDestroy
cuBinaryFree
cuCompilePtx
cuCtxCreate
cuCtxCreate_v2
cuCtxDestroy
cuCtxDestroy_v2
cuCtxSynchronize
cuD3D10CtxCreate
cuD3D10CtxCreateOnDevice
cuD3D10CtxCreate_v2
cuD3D10MapResources
cuD3D10RegisterResource
cuD3D10UnmapResources
cuD3D10UnregisterResource
cuD3D11CtxCreate
cuD3D11CtxCreateOnDevice
cuD3D11CtxCreate_v2
cuD3D9CtxCreate
cuD3D9CtxCreateOnDevice
cuD3D9CtxCreate_v2
cuD3D9MapResources
cuD3D9MapVertexBuffer
cuD3D9MapVertexBuffer_v2
cuD3D9RegisterResource
cuD3D9RegisterVertexBuffer
cuD3D9UnmapResources
cuD3D9UnmapVertexBuffer
cuD3D9UnregisterResource
cuD3D9UnregisterVertexBuffer
cuEGLStreamConsumerAcquireFrame
cuEGLStreamConsumerConnect
cuEGLStreamConsumerConnectWithFlags
cuEGLStreamConsumerDisconnect
cuEGLStreamConsumerReleaseFrame
cuEGLStreamProducerConnect
cuEGLStreamProducerDisconnect
cuEGLStreamProducerPresentFrame
cuEGLStreamProducerReturnFrame
cuEventCreate
cuEventCreateFromEGLSync
cuEventCreateFromNVNSync
cuEventDestroy
cuEventDestroy_v2
cuEventQuery
cuEventRecord
cuEventRecord_ptsz
cuEventSynchronize
cuGLCtxCreate
cuGLCtxCreate_v2
cuGLInit
cuGLMapBufferObject
cuGLMapBufferObjectAsync
cuGLMapBufferObjectAsync_v2
cuGLMapBufferObjectAsync_v2_ptsz
cuGLMapBufferObject_v2
cuGLMapBufferObject_v2_ptds
cuGLRegisterBufferObject
cuGLUnmapBufferObject
cuGLUnmapBufferObjectAsync
cuGLUnregisterBufferObject
cuGraphicsD3D10RegisterResource
cuGraphicsD3D11RegisterResource
cuGraphicsD3D9RegisterResource
cuGraphicsEGLRegisterImage
cuGraphicsGLRegisterBuffer
cuGraphicsGLRegisterImage
cuGraphicsMapResources
cuGraphicsMapResources_ptsz
cuGraphicsUnmapResources
cuGraphicsUnmapResources_ptsz
cuGraphicsUnregisterResource
cuGraphicsVDPAURegisterOutputSurface
cuGraphicsVDPAURegisterVideoSurface
cuInit
cuLaunch
cuLaunchCooperativeKernel
cuLaunchCooperativeKernelMultiDevice
cuLaunchCooperativeKernel_ptsz
cuLaunchGrid
cuLaunchGridAsync
cuLaunchKernel
cuLaunchKernel_ptsz
cuLinkComplete
cuLinkCreate
cuLinkCreate_v2
cuLinkDestroy
cuMemAlloc
cuMemAllocHost
cuMemAllocHost_v2
cuMemAllocManaged
cuMemAllocPitch
cuMemAllocPitch_v2
cuMemAlloc_v2
cuMemFree
cuMemFreeHost
cuMemFree_v2
cuMemGetInfo
cuMemGetInfo_v2
cuMemHostAlloc
cuMemHostAlloc_v2
cuMemHostRegister
cuMemHostRegister_v2
cuMemHostUnregister
cuMemPeerRegister
cuMemPeerUnregister
cuMemPrefetchAsync
cuMemPrefetchAsync_ptsz
cuMemcpy
cuMemcpy2D
cuMemcpy2DAsync
cuMemcpy2DAsync_v2
cuMemcpy2DAsync_v2_ptsz
cuMemcpy2DUnaligned
cuMemcpy2DUnaligned_v2
cuMemcpy2DUnaligned_v2_ptds
cuMemcpy2D_v2
cuMemcpy2D_v2_ptds
cuMemcpy3D
cuMemcpy3DAsync
cuMemcpy3DAsync_v2
cuMemcpy3DAsync_v2_ptsz
cuMemcpy3DPeer
cuMemcpy3DPeerAsync
cuMemcpy3DPeerAsync_ptsz
cuMemcpy3DPeer_ptds
cuMemcpy3D_v2
cuMemcpy3D_v2_ptds
cuMemcpyAsync
cuMemcpyAsync_ptsz
cuMemcpyAtoA
cuMemcpyAtoA_v2
cuMemcpyAtoA_v2_ptds
cuMemcpyAtoD
cuMemcpyAtoD_v2
cuMemcpyAtoD_v2_ptds
cuMemcpyAtoH
cuMemcpyAtoHAsync
cuMemcpyAtoHAsync_v2
cuMemcpyAtoHAsync_v2_ptsz
cuMemcpyAtoH_v2
cuMemcpyAtoH_v2_ptds
cuMemcpyDtoA
cuMemcpyDtoA_v2
cuMemcpyDtoA_v2_ptds
cuMemcpyDtoD
cuMemcpyDtoDAsync
cuMemcpyDtoDAsync_v2
cuMemcpyDtoDAsync_v2_ptsz
cuMemcpyDtoD_v2
cuMemcpyDtoD_v2_ptds
cuMemcpyDtoH
cuMemcpyDtoHAsync
cuMemcpyDtoHAsync_v2
cuMemcpyDtoHAsync_v2_ptsz
cuMemcpyDtoH_v2
cuMemcpyDtoH_v2_ptds
cuMemcpyHtoA
cuMemcpyHtoAAsync
cuMemcpyHtoAAsync_v2
cuMemcpyHtoAAsync_v2_ptsz
cuMemcpyHtoA_v2
cuMemcpyHtoA_v2_ptds
cuMemcpyHtoD
cuMemcpyHtoDAsync
cuMemcpyHtoDAsync_v2
cuMemcpyHtoDAsync_v2_ptsz
cuMemcpyHtoD_v2
cuMemcpyHtoD_v2_ptds
cuMemcpyPeer
cuMemcpyPeerAsync
cuMemcpyPeerAsync_ptsz
cuMemcpyPeer_ptds
cuMemcpy_ptds
cuMemcpy_v2
cuMemsetD16
cuMemsetD16Async
cuMemsetD16Async_ptsz
cuMemsetD16_v2
cuMemsetD16_v2_ptds
cuMemsetD2D16
cuMemsetD2D16Async
cuMemsetD2D16Async_ptsz
cuMemsetD2D16_v2
cuMemsetD2D16_v2_ptds
cuMemsetD2D32
cuMemsetD2D32Async
cuMemsetD2D32Async_ptsz
cuMemsetD2D32_v2
cuMemsetD2D32_v2_ptds
cuMemsetD2D8
cuMemsetD2D8Async
cuMemsetD2D8Async_ptsz
cuMemsetD2D8_v2
cuMemsetD2D8_v2_ptds
cuMemsetD32
cuMemsetD32Async
cuMemsetD32Async_ptsz
cuMemsetD32_v2
cuMemsetD32_v2_ptds
cuMemsetD8
cuMemsetD8Async
cuMemsetD8Async_ptsz
cuMemsetD8_v2
cuMemsetD8_v2_ptds
cuMipmappedArrayCreate
cuMipmappedArrayDestroy
cuModuleLoad
cuModuleLoadData
cuModuleLoadDataEx
cuModuleLoadFatBinary
cuModuleUnload
cuStreamAddCallback
cuStreamAddCallback_ptsz
cuStreamAttachMemAsync
cuStreamAttachMemAsync_ptsz
cuStreamBatchMemOp
cuStreamBatchMemOp_ptsz
cuStreamCreate
cuStreamCreateWithPriority
cuStreamDestroy
cuStreamDestroy_v2
cuStreamSynchronize
cuStreamSynchronize_ptsz
cuStreamWaitEvent
cuStreamWaitEvent_ptsz
cuStreamWaitValue32
cuStreamWaitValue32_ptsz
cuStreamWaitValue64
cuStreamWaitValue64_ptsz
cuStreamWriteValue32
cuStreamWriteValue32_ptsz
cuStreamWriteValue64
cuStreamWriteValue64_ptsz
cuSurfObjectCreate
cuSurfObjectDestroy
cuSurfRefCreate
cuSurfRefDestroy
cuTexObjectCreate
cuTexObjectDestroy
cuTexRefCreate
cuTexRefDestroy
cuVDPAUCtxCreate
cuVDPAUCtxCreate_v2
```

### 12.3 cuDNN API Functions Traced by Default

The following cuDNN API functions are traced on x86 Linux (not available on Windows targets):

```
cudnnActivationBackward
cudnnActivationBackward_v3
cudnnActivationBackward_v4
cudnnActivationForward
cudnnActivationForward_v3
cudnnActivationForward_v4
cudnnAddTensor
cudnnBatchNormalizationBackward
cudnnBatchNormalizationBackwardEx
cudnnBatchNormalizationForwardInference
cudnnBatchNormalizationForwardTraining
cudnnBatchNormalizationForwardTrainingEx
cudnnCTCLoss
cudnnConvolutionBackwardBias
cudnnConvolutionBackwardData
cudnnConvolutionBackwardFilter
cudnnConvolutionBiasActivationForward
cudnnConvolutionForward
cudnnCreate
cudnnCreateAlgorithmPerformance
cudnnDestroy
cudnnDestroyAlgorithmPerformance
cudnnDestroyPersistentRNNPlan
cudnnDivisiveNormalizationBackward
cudnnDivisiveNormalizationForward
cudnnDropoutBackward
cudnnDropoutForward
cudnnDropoutGetReserveSpaceSize
cudnnDropoutGetStatesSize
cudnnFindConvolutionBackwardDataAlgorithm
cudnnFindConvolutionBackwardDataAlgorithmEx
cudnnFindConvolutionBackwardFilterAlgorithm
cudnnFindConvolutionBackwardFilterAlgorithmEx
cudnnFindConvolutionForwardAlgorithm
cudnnFindConvolutionForwardAlgorithmEx
cudnnFindRNNBackwardDataAlgorithmEx
cudnnFindRNNBackwardWeightsAlgorithmEx
cudnnFindRNNForwardInferenceAlgorithmEx
cudnnFindRNNForwardTrainingAlgorithmEx
cudnnFusedOpsExecute
cudnnIm2Col
cudnnLRNCrossChannelBackward
cudnnLRNCrossChannelForward
cudnnMakeFusedOpsPlan
cudnnMultiHeadAttnBackwardData
cudnnMultiHeadAttnBackwardWeights
cudnnMultiHeadAttnForward
cudnnOpTensor
cudnnPoolingBackward
cudnnPoolingForward
cudnnRNNBackwardData
cudnnRNNBackwardDataEx
cudnnRNNBackwardWeights
cudnnRNNBackwardWeightsEx
cudnnRNNForwardInference
cudnnRNNForwardInferenceEx
cudnnRNNForwardTraining
cudnnRNNForwardTrainingEx
cudnnReduceTensor
cudnnReorderFilterAndBias
cudnnRestoreAlgorithm
cudnnRestoreDropoutDescriptor
cudnnSaveAlgorithm
cudnnScaleTensor
cudnnSoftmaxBackward
cudnnSoftmaxForward
cudnnSpatialTfGridGeneratorBackward
cudnnSpatialTfGridGeneratorForward
cudnnSpatialTfSamplerBackward
cudnnSpatialTfSamplerForward
cudnnTransformFilter
cudnnTransformTensor
cudnnTransformTensorEx
```

---

## 13. CUDA Trace Known Issues

- If a system is in CC-DevTools mode and tracing CUDA in an application using libcrypto, Nsight Systems may crash when the application exits. Workarounds: add cudaDeviceSynchronize before exit; add cudaProfilerStop with --flush-on-cudaprofilerstop=true; use collection duration; use capture ranges; or use CLI start/launch/stop commands.
- CUDA GPU trace collection requires a fraction of GPU memory. If the application utilizes all GPU memory, CUDA trace might not work.
- On Tegra platforms, CUDA trace requires root privileges.
- If the target application uses multiple streams from multiple threads, CUDA event buffers may not be released properly (error: "Couldn't allocate CUPTI buffer x times").
- When using CUDA Toolkit 10.X, tracing DtoD memory copy operations may crash. Update to 11.X or later.
- Nsight Systems will not trace kernels when a CDP (CUDA Dynamic Parallelism) kernel is found on Volta or later.
- The cudaMemPrefetchAsync() API allows specifying a stream, but Nsight Systems does not get stream information for UVM page migrations from the UVM backend.
- CUDA memory allocation graph generation is only guaranteed correct in the first profiling range when using interactive CLI start/stop.
