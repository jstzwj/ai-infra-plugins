# Graphics API Tracing and Stutter Analysis Reference

This document provides comprehensive reference material for graphics API tracing in NVIDIA Nsight Systems, including Direct3D 11, Direct3D 12, WDDM queues, Vulkan, OpenGL, OpenXR, GPUDirect Storage, and the stutter analysis framework. These features enable graphics developers to understand CPU-GPU interactions, diagnose rendering performance issues, and optimize frame delivery.

---

## Table of Contents

1. [Direct3D Trace Overview](#d3d-overview)
2. [D3D11 API Trace](#d3d11-trace)
   - [SLI Trace](#sli-trace)
3. [D3D12 API Trace](#d3d12-trace)
   - [Command List Creation](#d3d12-commandlist)
   - [GPU Rows and Queues](#d3d12-gpu-rows)
   - [DX12 API Memory Ops](#d3d12-memory-ops)
   - [API and Workload Correlation](#d3d12-correlation)
   - [Fence Synchronization](#d3d12-fence)
   - [D3D12 Work Graphs](#d3d12-workgraphs)
   - [DX12 GPU Workload Options](#d3d12-gpu-workload)
   - [DX12 Wait Calls](#d3d12-wait-calls)
4. [WDDM Queues and HW Scheduler](#wddm-queues)
   - [WDDM Queue Types](#wddm-queue-types)
   - [WDDM Additional Events](#wddm-additional-events)
   - [WDDM Backtraces](#wddm-backtraces)
   - [WDDM HW Scheduler](#wddm-hw-scheduler)
5. [Vulkan API Trace](#vulkan-trace)
   - [Vulkan Overview](#vulkan-overview)
   - [Command Buffer Creation](#vulkan-commandbuffer)
   - [Queue Rows](#vulkan-queues)
   - [Pipeline Creation Feedback](#vulkan-pipeline-feedback)
   - [Vulkan Memory Operations](#vulkan-memory-ops)
   - [Vulkan GPU Trace Notes](#vulkan-gpu-notes)
   - [Vulkan GPU Workload Options](#vulkan-gpu-workload)
6. [OpenGL Trace](#opengl-trace)
   - [CPU and GPU Trace](#opengl-cpu-gpu)
   - [eglSwapBuffers Visualization](#opengl-swapbuffers)
   - [KHR_debug Extension](#opengl-khr-debug)
   - [OpenGL Trace Using Command Line](#opengl-cli)
7. [OpenXR API Trace](#openxr-trace)
8. [GPUDirect Storage Trace](#gds-trace)
9. [Stutter Analysis](#stutter-analysis)
   - [Stutter Analysis Overview](#stutter-overview)
   - [FPS Overview](#fps-overview)
   - [Stutter Algorithm](#stutter-algorithm)
   - [OSC Detection](#osc-detection)
   - [Frame Duration Color Coding](#frame-color-coding)
   - [CPU Frame Duration](#cpu-frame-duration)
   - [GPU Frame Duration](#gpu-frame-duration)
   - [Reflex SDK](#reflex-sdk)
   - [Performance Warnings Row](#performance-warnings)
   - [Frame Health](#frame-health)
   - [Windows GPU Memory Utilization](#gpu-memory-utilization)
   - [VidMm Device Suspension](#vidmm-suspension)
   - [Demoted Memory](#demoted-memory)
   - [Resource Allocations](#resource-allocations)
   - [Resource Migrations](#resource-migrations)
   - [Memory Transfer](#memory-transfer)
   - [System Committed VRAM](#system-committed-vram)
   - [VRAM Resource Types Distribution](#vram-resource-types)
   - [Vertical Synchronization](#vsync)

---

## Direct3D Trace Overview

<a id="d3d-overview"></a>

Nsight Systems has the ability to trace both the Direct3D 11 API and the Direct3D 12 API on Windows targets.

---

## D3D11 API Trace

<a id="d3d11-trace"></a>

Nsight Systems can capture information about Direct3D 11 API calls made by the profiled process. This includes capturing the execution time of D3D11 API functions, performance markers, and frame durations.

### SLI Trace

<a id="sli-trace"></a>

You can trace SLI queries and peer-to-peer transfers of D3D11 applications. This requires SLI hardware and an active SLI profile definition in the NVIDIA console.

---

## D3D12 API Trace

<a id="d3d12-trace"></a>

Direct3D 12 is a low-overhead 3D graphics and compute API for Microsoft Windows. Information about Direct3D 12 can be found at the Direct3D 12 Programming Guide.

Nsight Systems can capture information about Direct3D 12 usage by the profiled process. This includes:

- Execution time of D3D12 API functions
- Corresponding workloads executed on the GPU
- Performance markers
- Frame durations

### Command List Creation

<a id="d3d12-commandlist"></a>

The **Command List Creation** row displays time periods when command lists were being created. This enables developers to improve their application's multi-threaded command list creation.

Command list creation time period is measured between the call to `ID3D12GraphicsCommandList::Reset` and the call to `ID3D12GraphicsCommandList::Close`.

### GPU Rows and Queues

<a id="d3d12-gpu-rows"></a>

The **GPU row** shows a compressed view of the D3D12 queue activity, color-coded by the queue type. Expanding it will show the individual queues and their corresponding API calls.

A **Command Queue row** is displayed for each D3D12 command queue created by the profiled application. The row's header displays the queue's running index and its type:

| Queue Type | Description |
|---|---|
| **Direct** | Graphics command queue |
| **Compute** | Compute command queue |
| **Copy** | Copy/transfer command queue |

### DX12 API Memory Ops

<a id="d3d12-memory-ops"></a>

The **DX12 API Memory Ops** row displays all API memory operations and non-persistent resource mappings. Event ranges in the row are color-coded by the heap type they belong to:

| Heap Type | Color Code |
|---|---|
| Default | - |
| Readback | - |
| Upload | - |
| Custom | - |
| CPU-Visible VRAM | - |

Usage warnings are highlighted in **yellow**.

A breakdown of the operations can be found by expanding the row to show rows for each individual heap type.

#### Operations and Warnings Tracked

The following operations and warnings are shown:

| Operation | Warning Condition |
|---|---|
| `ID3D12Device::CreateCommittedResource` | Warning if `D3D12_HEAP_FLAG_CREATE_NOT_ZEROED` is not set in the method's HeapFlags parameter |
| `ID3D12Device4::CreateCommittedResource1` | Warning if `D3D12_HEAP_FLAG_CREATE_NOT_ZEROED` is not set in the method's HeapFlags parameter |
| `ID3D12Device8::CreateCommittedResource2` | Warning if `D3D12_HEAP_FLAG_CREATE_NOT_ZEROED` is not set in the method's HeapFlags parameter |
| `ID3D12Device::CreateHeap` | Warning if `D3D12_HEAP_FLAG_CREATE_NOT_ZEROED` is not set in the Flags field of the method's pDesc parameter |
| `ID3D12Device4::CreateHeap1` | Warning if `D3D12_HEAP_FLAG_CREATE_NOT_ZEROED` is not set in the Flags field of the method's pDesc parameter |
| `ID3D12Resource::ReadFromSubResource` | Warning if the read is to a `D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE` CPU page or from a `D3D12_HEAP_TYPE_UPLOAD` resource |
| `ID3D12Resource::WriteToSubResource` | Warning if the write is from a `D3D12_CPU_PAGE_PROPERTY_WRITE_BACK` CPU page or to a `D3D12_HEAP_TYPE_READBACK` resource |
| `ID3D12Resource::Map` and `ID3D12Resource::Unmap` | Matched into `[Map, Unmap]` ranges for non-persistent mappings. If a mapping range is nested, only the most external range (reference count = 1) will be shown |

### API and Workload Correlation

<a id="d3d12-correlation"></a>

The **API row** displays time periods where `ID3D12CommandQueue::ExecuteCommandLists` was called. The **GPU Workload row** displays time periods where workloads were executed by the GPU. The workload's type (Graphics, Compute, Copy, etc.) is displayed on the bar representing the workload's GPU execution.

In addition, you can see the PIX command queue:
- CPU-side performance markers
- GPU-side performance markers
- GPU Command List performance markers

Each in their own row.

Clicking on a GPU workload highlights the corresponding `ID3D12CommandQueue::ExecuteCommandLists`, `ID3D12GraphicsCommandList::Reset`, and `ID3D12GraphicsCommandList::Close` API calls, and vice versa.

### Fence Synchronization

<a id="d3d12-fence"></a>

Detecting which CPU thread was blocked by a fence can be difficult in complex apps that run tens of CPU threads. The timeline view displays the 3 operations involved:

| Operation | Location | Display |
|---|---|---|
| CPU thread pushing a signal command and fence value into the command queue | DX12 Synchronization sub-row of the calling thread | Signal event |
| GPU executing that command, setting the fence value and signaling the fence | GPU Queue Synchronization sub-row | GPU signal event |
| CPU thread calling a Win32 wait API to block-wait until the fence is signaled | Thread's OS runtime libraries row | Wait event |

Clicking one of these will highlight it and the corresponding other two calls.

### D3D12 Work Graphs

<a id="d3d12-workgraphs"></a>

Nsight Systems D3D12 trace captures D3D12 Work Graphs dispatch calls to `DispatchGraph` and time boundaries of the GPU execution of the work graph.

- The **DX12 API row** displays `ID3D12GraphicsCommandList10::DispatchGraph` calls.
- The **GPU PIX Markers row** marks graph execution by the GPU with a custom marker captioned "D3D12 graph execution."

### DX12 GPU Workload Options

<a id="d3d12-gpu-workload"></a>

The `--dx12-gpu-workload` option controls how DX12 GPU workloads are displayed:

| Value | Description |
|---|---|
| `individual` | Each workload is displayed individually on the timeline |
| `batch` | Workloads are batched together for a more compact view |
| `none` | GPU workload display is disabled |

### DX12 Wait Calls

<a id="d3d12-wait-calls"></a>

The `--dx12-wait-calls` option enables tracing of DX12 wait calls, which can help identify synchronization bottlenecks between the CPU and GPU.

---

## WDDM Queues and HW Scheduler

<a id="wddm-queues"></a>

The Windows Display Driver Model (WDDM) architecture uses queues to send work packets from the CPU to the GPU. Each D3D device in each process is associated with one or more contexts. Graphics, compute, and copy commands that the profiled application uses are associated with a context, batched in a command buffer, and pushed into the relevant queue associated with that context.

Nsight Systems can capture the state of these queues during the trace session.

### WDDM Queue Types

<a id="wddm-queue-types"></a>

A command buffer in a WDDM queues may have one of the following types:

| Buffer Type | Description |
|---|---|
| **Render** | Graphics rendering commands |
| **Deferred** | Deferred command processing |
| **System** | Driver-internal system commands |
| **MMIOFlip** | Memory-mapped I/O flip operation |
| **Wait** | Wait/synchronization commands |
| **Signal** | Signal/synchronization commands |
| **Device** | Device management commands |
| **Software** | Software rendering commands |

It may also be marked as a **Present buffer**, indicating that the application has finished rendering and requests to display the source surface.

See the Microsoft documentation for the WDDM architecture and the `DXGKETW_QUEUE_PACKET_TYPE` enumeration.

### WDDM Additional Events

<a id="wddm-additional-events"></a>

Enabling the **"Collect additional range of ETW events"** option will also capture extended DxgKrnl events from the `Microsoft-Windows-DxgKrnl` provider, such as:

- Context status events
- Allocation events
- Sync wait events
- Signal events

### WDDM Backtraces

<a id="wddm-backtraces"></a>

To retain the .etl trace files captured, so that they can be viewed in other tools (e.g. GPUView), change the **"Save ETW log files in project folder"** option under **"Profile Behavior"** in Nsight Systems's global Options dialog.

The .etl files will appear in the same folder as the `.nsys-rep` file, accessible by right-clicking the report in the Project Explorer and choosing **"Show in Folder..."**.

Data collected from each ETW provider will appear in its own `.etl` file, and an additional `.etl` file named "Report XX-Merged-*.etl", containing the events from all captured sources, will be created as well.

### WDDM HW Scheduler

<a id="wddm-hw-scheduler"></a>

When GPU Hardware Scheduling is enabled in Windows 10 or newer, the Windows Display Driver Model (WDDM) uses the DxgKrnl ETW provider to expose report of NVIDIA GPUs' hardware scheduling context switches.

Nsight Systems can capture these context switch events, and display them under the GPUs in the timeline rows titled **WDDM HW Scheduler - [HW Queue type]**.

The ranges under each queue will show the **process name** and **PID** associated with the GPU work during the time period.

The events will be captured if:
- GPU Hardware Scheduling is enabled in the Windows System Display settings, AND
- **"Collect WDDM Trace"** is enabled in the Nsight Systems Project Settings

---

## Vulkan API Trace

<a id="vulkan-trace"></a>

### Vulkan Overview

<a id="vulkan-overview"></a>

Vulkan is a low-overhead, cross-platform 3D graphics and compute API, targeting a wide variety of devices from PCs to mobile phones and embedded platforms. The Vulkan API is defined by the Khronos Group.

Nsight Systems can capture information about Vulkan usage by the profiled process. This includes:
- Execution time of Vulkan API functions
- Corresponding GPU workloads
- Debug util labels
- Frame durations

Vulkan profiling is supported on both **Windows** and **x86 Linux** operating systems.

### Command Buffer Creation

<a id="vulkan-commandbuffer"></a>

The **Command Buffer Creation** row displays time periods when command buffers were being created. This enables developers to improve their application's multi-threaded command buffer creation.

Command buffer creation time period is measured between the call to `vkBeginCommandBuffer` and the call to `vkEndCommandBuffer`.

### Queue Rows

<a id="vulkan-queues"></a>

A **Queue row** is displayed for each Vulkan queue created by the profiled application:

- The **API sub-row** displays time periods where `vkQueueSubmit` was called.
- The **GPU Workload sub-row** displays time periods where workloads were executed by the GPU.

In addition, you can see Vulkan debug util labels on both the CPU and the GPU.

Clicking on a GPU workload highlights the corresponding `vkQueueSubmit` call, and vice versa.

### Pipeline Creation Feedback

<a id="vulkan-pipeline-feedback"></a>

When tracing target application calls to Vulkan pipeline creation APIs, Nsight Systems leverages the Pipeline Creation Feedback extension to collect more details about the duration of individual pipeline creation stages.

See the Pipeline Creation Feedback extension documentation for details about this extension.

Vulkan pipeline creation feedback is available on **NVIDIA driver release 435 or later**.

### Vulkan Memory Operations

<a id="vulkan-memory-ops"></a>

The **Vulkan Memory Operations** row contains an aggregation of all the Vulkan host-side memory operations, such as:

- Host-blocking writes
- Host-blocking reads
- Non-persistent map-unmap ranges

The row is separated into sub-rows by **heap index** and **memory type**. The tooltip for each row and the ranges inside show the heap flags and the memory property flags.

### Vulkan GPU Trace Notes

<a id="vulkan-gpu-notes"></a>

- Vulkan GPU trace is available only when tracing apps that use **NVIDIA GPUs**.
- The endings of Vulkan Command Buffers execution ranges on **Compute** and **Transfer** queues may appear earlier on the timeline than their actual occurrence.

### Vulkan GPU Workload Options

<a id="vulkan-gpu-workload"></a>

The `--vulkan-gpu-workload` option controls how Vulkan GPU workloads are displayed:

| Value | Description |
|---|---|
| `individual` | Each workload is displayed individually on the timeline |
| `batch` | Workloads are batched together for a more compact view |
| `none` | GPU workload display is disabled |

---

## OpenGL Trace

<a id="opengl-trace"></a>

OpenGL and OpenGL ES APIs can be traced to assist in the analysis of CPU and GPU interactions.

### CPU and GPU Trace

<a id="opengl-cpu-gpu"></a>

OpenGL trace feature in Nsight Systems consists of two different activities which will be shown in the CPU rows for those threads:

1. **CPU trace**: Interception of API calls that an application does to APIs (such as OpenGL, OpenGL ES, EGL, GLX, WGL, etc.).

2. **GPU trace (or workload trace)**: Trace of GPU workload (activity) triggered by use of OpenGL or OpenGL ES. Since draw calls are executed back-to-back, the GPU workload trace ranges include many OpenGL draw calls and operations in order to optimize performance overhead, rather than tracing each individual operation.

To collect GPU trace, the `glQueryCounter()` function is used to measure how much time batches of GPU workload take to complete.

### eglSwapBuffers Visualization

<a id="opengl-swapbuffers"></a>

A few usage examples:

- **Visualize how long `eglSwapBuffers`** (or similar) is taking. API trace can easily show correlations between thread state and graphics driver's behavior, uncovering where the CPU may be waiting on the GPU.
- **Spot bubbles of opportunity** on the GPU, where more GPU workload could be created.
- **Use KHR_debug extension** to trace GL events on both the CPU and GPU.

Ranges defined by the KHR_debug calls are represented similarly to OpenGL API and OpenGL GPU workload trace. GPU ranges in this case represent incremental draw cost. They cannot fully account for GPUs that can execute multiple draw calls in parallel. In this case, Nsight Systems will not show overlapping GPU ranges.

### OpenGL Trace Using Command Line

<a id="opengl-cli"></a>

For the CLI, the functions that are traced include:

```
glWaitSync  glReadPixels  glReadnPixelsKHR  glReadnPixelsEXT
glReadnPixelsARB  glReadnPixels  glFlush  glFinishFenceNV  glFinish
glClientWaitSync  glClearTexSubImage  glClearTexImage  glClearStencil
glClearNamedFramebufferuiv  glClearNamedFramebufferiv  glClearNamedFramebufferfv
glClearNamedFramebufferfi  glClearNamedBufferSubDataEXT  glClearNamedBufferSubData
glClearNamedBufferDataEXT  glClearNamedBufferData  glClearIndex  glClearDepthx
glClearDepthf  glClearDepthdNV  glClearDepth  glClearColorx  glClearColorIuiEXT
glClearColorIiEXT  glClearColor  glClearBufferuiv  glClearBufferSubData
glClearBufferiv  glClearBufferfv  glClearBufferfi  glClearBufferData  glClearAccum
glClear  glDispatchComputeIndirect  glDispatchComputeGroupSizeARB
glDispatchCompute  glComputeStreamNV
glNamedFramebufferDrawBuffers  glNamedFramebufferDrawBuffer
glMultiDrawElementsIndirectEXT  glMultiDrawElementsIndirectCountARB
glMultiDrawElementsIndirectBindlessNV  glMultiDrawElementsIndirectBindlessCountNV
glMultiDrawElementsIndirectAMD  glMultiDrawElementsIndirect
glMultiDrawElementsEXT  glMultiDrawElementsBaseVertex  glMultiDrawElements
glMultiDrawArraysIndirectEXT  glMultiDrawArraysIndirectCountARB
glMultiDrawArraysIndirectBindlessNV  glMultiDrawArraysIndirectBindlessCountNV
glMultiDrawArraysIndirectAMD  glMultiDrawArraysIndirect
glMultiDrawArraysEXT  glMultiDrawArrays
glListDrawCommandsStatesClientNV  glFramebufferDrawBuffersEXT
glFramebufferDrawBufferEXT  glDrawTransformFeedbackStreamInstanced
glDrawTransformFeedbackStream  glDrawTransformFeedbackNV
glDrawTransformFeedbackInstancedEXT  glDrawTransformFeedbackInstanced
glDrawTransformFeedbackEXT  glDrawTransformFeedback  glDrawTexxvOES
glDrawTexxOES  glDrawTextureNV  glDrawTexsvOES  glDrawTexsOES
glDrawTexivOES  glDrawTexiOES  glDrawTexfvOES  glDrawTexfOES
glDrawRangeElementsEXT  glDrawRangeElementsBaseVertexOES
glDrawRangeElementsBaseVertexEXT  glDrawRangeElementsBaseVertex
glDrawRangeElements  glDrawPixels
glDrawElementsInstancedNV  glDrawElementsInstancedEXT
glDrawElementsInstancedBaseVertexOES  glDrawElementsInstancedBaseVertexEXT
glDrawElementsInstancedBaseVertexBaseInstanceEXT
glDrawElementsInstancedBaseVertexBaseInstance
glDrawElementsInstancedBaseVertex  glDrawElementsInstancedBaseInstanceEXT
glDrawElementsInstancedBaseInstance  glDrawElementsInstancedARB
glDrawElementsInstanced  glDrawElementsIndirect
glDrawElementsBaseVertexOES  glDrawElementsBaseVertexEXT
glDrawElementsBaseVertex  glDrawElements  glDrawCommandsStatesNV
glDrawCommandsStatesAddressNV  glDrawCommandsNV  glDrawCommandsAddressNV
glDrawBuffersNV  glDrawBuffersATI  glDrawBuffersARB  glDrawBuffers
glDrawBuffer  glDrawArraysInstancedNV  glDrawArraysInstancedEXT
glDrawArraysInstancedBaseInstanceEXT  glDrawArraysInstancedBaseInstance
glDrawArraysInstancedARB  glDrawArraysInstanced  glDrawArraysIndirect
glDrawArraysEXT  glDrawArrays
eglSwapBuffersWithDamageKHR  eglSwapBuffers  glXSwapBuffers
glXQueryDrawable  glXGetCurrentReadDrawable  glXGetCurrentDrawable
glGetQueryObjectuivEXT  glGetQueryObjectuivARB  glGetQueryObjectuiv
glGetQueryObjectivARB  glGetQueryObjectiv
```

---

## OpenXR API Trace

<a id="openxr-trace"></a>

OpenXR is a royalty-free, open standard that provides high-performance access to Augmented Reality (AR) and Virtual Reality (VR) -- collectively known as XR -- platforms and devices.

Nsight Systems can capture information about OpenXR usage by the profiled process. This includes:
- Execution time of OpenXR API functions
- Debug labels
- Frame durations

OpenXR profiling is supported on **Windows** operating systems.

---

## GPUDirect Storage Trace

<a id="gds-trace"></a>

NVIDIA GPUDirect Storage (GDS) enables direct memory access (DMA) between storage and GPU memory. This avoids a bounce buffer through the CPU, increasing storage access bandwidth and decreasing latency and utilization load on the CPU.

Nsight Systems can capture information about GDS, specifically the various cuFile API calls made by the profiled process. GDS profiling is currently an **experimental feature**, and is supported on **Linux x64** and **SBSA** operating systems.

---

## Stutter Analysis

<a id="stutter-analysis"></a>

### Stutter Analysis Overview

<a id="stutter-overview"></a>

Nsight Systems on Windows targets displays stutter analysis visualization aids for profiled graphics applications that use either **OpenGL**, **D3D11**, **D3D12**, or **Vulkan**.

### FPS Overview

<a id="fps-overview"></a>

The **Frame Duration** section displays frame durations on both the CPU and the GPU.

The frame duration row displays live FPS statistics for the current timeline viewport. Values shown are:

| Value | Description |
|---|---|
| **Number of CPU frames shown** | Of the total number captured |
| **Average CPU frame time** | Of the currently displayed time range |
| **Minimal CPU frame time** | Of the currently displayed time range |
| **Maximal CPU frame time** | Of the currently displayed time range |
| **Average FPS value** | For the currently displayed frames |
| **99th percentile value** | Of the frame lengths (such that only 1% of the frames in the range are longer than this value) |

The values will update automatically when scrolling, zooming or filtering the timeline view.

### Stutter Algorithm

<a id="stutter-algorithm"></a>

The stutter row highlights frames that are significantly longer than the other frames in their immediate vicinity.

The stutter row uses an algorithm that compares the duration of each frame to the **median duration of the surrounding 19 frames**. Duration difference under **4 milliseconds** is never considered a stutter, to avoid cluttering the display with frames whose absolute stutter is small and not noticeable to the user.

#### Stutter Threshold Examples

For example, if the stutter threshold is set at 20%:

| Median Duration | Frame Duration | Stutter? | Reason |
|---|---|---|---|
| 10 ms | 13 ms | No | Relative difference > 20%, but absolute difference < 4 ms |
| 60 ms | 71 ms | No | Relative difference < 20%, absolute difference > 4 ms |
| 60 ms | 80 ms | **Yes** | Relative difference > 20% AND absolute difference > 4 ms |

### OSC Detection

<a id="osc-detection"></a>

The "19 frame window median" algorithm by itself may not work well with some cases of "oscillation" (consecutive fast and slow frames), resulting in some false positives. The median duration is not meaningful in cases of oscillation and can be misleading.

To address the issue and identify oscillating frames, the following method is applied:

1. For every frame, calculate the median duration, 1st and 3rd quartiles of 19-frames window.
2. Calculate the delta and ratio between 1st and 3rd quartiles.
3. If the 90th percentile of (3rd - 1st quartile delta array) > 4 ms AND the 90th percentile of (3rd/1st quartile array) > 1.2 (120%) then mark the results with **"OSC"** text.

### Frame Duration Color Coding

<a id="frame-color-coding"></a>

Right-clicking the Frame Duration row caption lets you choose the target frame rate (30, 60, 90 or custom frames per second).

By clicking the **Customize FPS Display** option, a customization dialog pops up. In the dialog, you can define the frame duration threshold to customize the view of the potentially problematic frames. In addition, you can define the threshold for the stutter analysis frames.

Frame duration bars are color-coded:

| Color | Meaning |
|---|---|
| **Green** | The frame duration is shorter than required by the target FPS ratio |
| **Yellow** | Duration is slightly longer than required by the target FPS rate |
| **Red** | Duration far exceeds that required to maintain the target FPS rate |

### CPU Frame Duration

<a id="cpu-frame-duration"></a>

The CPU Frame Duration row displays the CPU frame duration measured between the ends of consecutive frame boundary calls:

| API | Frame Boundary |
|---|---|
| **OpenGL** | `eglSwapBuffers` / `glXSwapBuffers` / `SwapBuffers` calls |
| **D3D11 and D3D12** | `IDXGISwapChainX::Present` calls |
| **Vulkan** | `vkQueuePresentKHR` calls |

The timing of the actual calls to the frame boundary calls can be seen in the blue bar at the bottom of the CPU frame duration row.

### GPU Frame Duration

<a id="gpu-frame-duration"></a>

The GPU Frame Duration row displays the time measured between:
- The start time of the first GPU workload execution of this frame
- The start time of the first GPU workload execution of the next frame

### Reflex SDK

<a id="reflex-sdk"></a>

NVIDIA Reflex SDK is a series of NVAPI calls that allow applications to integrate the Ultra Low Latency driver feature more directly into their game to further optimize synchronization between simulation and rendering stages and lower the latency between user input and final image rendering.

Nsight Systems will automatically capture NVAPI functions when either Direct3D 11, Direct3D 12, or Vulkan API trace are enabled.

The Reflex SDK row displays timeline ranges for the following types of latency markers:

| Marker Type | Description |
|---|---|
| **RenderSubmit** | Command buffer recording and submission phase |
| **Simulation** | Game logic, physics, AI update phase |
| **Present** | SwapChain Present call |
| **Driver** | GPU driver processing phase |
| **OS Render Queue** | OS compositor queue time |
| **GPU Render** | Actual GPU execution time |

### Performance Warnings Row

<a id="performance-warnings"></a>

This row shows performance warnings and common pitfalls that are automatically detected based on the enabled capture types. Warnings are reported for:

| Warning Type | Description |
|---|---|
| **ETW performance warnings** | Windows Event Tracing detected performance issues |
| **Slow vkQueueSubmit / ExecuteCommandList** | Vulkan calls to `vkQueueSubmit` and D3D12 calls to `ID3D12CommandQueue::ExecuteCommandList` that take a longer time to execute than the total time of the GPU workloads they generated |
| **D3D12 Memory Operation warnings** | Memory operations with suboptimal configurations |
| **Vulkan API performance issues** | Usage of Vulkan API functions that may adversely affect performance |
| **Vulkan device memory zeroing** | Creation of a Vulkan device with memory zeroing, whether by physical device default or manually |
| **Vulkan barrier inefficiencies** | Vulkan command buffer barriers which can be combined or removed, such as subsequent barriers or read-to-read barriers |

### Frame Health

<a id="frame-health"></a>

The **Frame Health** row displays actions that took significantly a longer time during the current frame, compared to the median time of the same actions executed during the surrounding 19 frames. This is a great tool for detecting the reason for frame time stuttering.

Such actions may be: shader compilation, present, memory mapping, and more. Nsight Systems measures the accumulated time of such actions in each frame. For example: calculating the accumulated time of shader compilations in each frame and comparing it to the accumulated time of shader compilations in the surrounding 19 frames.

### Windows GPU Memory Utilization

<a id="gpu-memory-utilization"></a>

Each GPU has two rows detailing its memory utilization:

| Row | Description |
|---|---|
| **GPU VRAM** | Memory consumed on the device |
| **GPU WDDM SYSMEM** | Memory consumed on the host computer RAM |

These rows show:
- A **green-colored line graph** for the memory budget for this memory segment
- An **orange-colored line graph** for the actual amount of memory used

Note that these graphs are scaled to fit the highest value encountered, as indicated by the "Y axis" value in the row header. You can use the vertical zoom slider in the top-right of the timeline view to make the row taller and view the graph in more detail.

Note that the value in the GPU VRAM row is not the same as the CUDA kernel memory allocation graph.

The GPU VRAM row also has several child rows, accessed by expanding the row in the tree view.

The events will be captured if **"Collect WDDM Trace"** and **"Collect additional range of ETW events, including context status, allocations, sync wait and signal events, etc."** are enabled in the Nsight Systems Project Settings.

### VidMm Device Suspension

<a id="vidmm-suspension"></a>

This row displays time ranges when the GPU memory manager suspended all memory transfer operations, pending the completion of a single memory transfer.

The events will be captured if **"Collect WDDM Trace"** and **"Collect additional range of ETW events, including context status, allocations, sync wait and signal events, etc."** are enabled in the Nsight Systems Project Settings.

### Demoted Memory

<a id="demoted-memory"></a>

This row displays the amount of VRAM that was demoted from GPU local memory to non-local memory (possibly due to exceeding the VRAM budget) as a **blue-colored line graph**.

High amounts of demoted memory could be indicative of:
- Video memory leaks
- Poor memory management

Note that the Demoted memory row is scaled to its highest value, similar to the GPU VRAM and GPU WDDM SYSMEM rows.

The events will be captured if **"Collect WDDM Trace"** and **"Collect additional range of ETW events, including context status, allocations, sync wait and signal events, etc."** are enabled in the Nsight Systems Project Settings.

### Resource Allocations

<a id="resource-allocations"></a>

This row shows markers indicating resource allocation events:
- **VRAM resources** are shown as **green markers**
- **SYSMEM resources** are shown in **gray**

Hovering over a marker or selecting it in the Events view will display all the allocation parameters as well as the call stack that led to the allocation event.

The events will be captured if **"Collect WDDM Trace"** and **"Collect additional range of ETW events, including context status, allocations, sync wait and signal events, etc."** are enabled in the Nsight Systems Project Settings.

### Resource Migrations

<a id="resource-migrations"></a>

This row displays a breakdown of resources' movement between VRAM and SYSMEM, focusing on resource evictions. The main row shows a timeline of total evicted resource memory and count as a **red-colored line graph**.

Each child row displays a timeline of the status of each resource, as reflected by WDDM events related to it. If the object has been named using PIX or `ID3D11Object::SetName` / `ID3D12Object::SetName`, the name will be shown in the row title. Whether named or not, the row title will also show the resource dimensions, format, priority, and the memory size migrated.

If the resource was migrated in parts using subresources, the row can be expanded to show the status for each subresource at any given time.

Expanding the row for a resource will show the individual WDDM events relevant to it and the call stacks that led to each event.

#### Sorting Options

By default, the resources are sorted by **Relevance** (most / largest migrations). Right-clicking the main Resource Migrations row header allows choosing between the following sorting options:

| Option | Description |
|---|---|
| **Relevance** | Most / largest migrations first |
| **Name** | Alphabetical by resource name |
| **Format** | By resource format |
| **Priority** | By resource priority |
| **Earliest allocation timestamp** | Order of appearance on the host |
| **Earliest migration timestamp** | Order of appearance on the device |

The top 5 resources are shown initially. If more than 5 resources exist, a row showing the number of hidden resources and buttons allowing to show more or fewer of them will appear below them. Right-click this row and select **"show all"** or **"show all collapsed"** to display all the resources at once.

The events will be captured if **"Collect WDDM Trace"** and **"Collect additional range of ETW events, including context status, allocations, sync wait and signal events, etc."** are enabled in the Nsight Systems Project Settings. Additionally, to correlate Graphics API debug name events with resource migration events, the **"Collect DX12"** or **"Collect Vulkan"** option should be enabled.

### Memory Transfer

<a id="memory-transfer"></a>

This row shows an overview of all memory transfer operations:

| Transfer Type | Color |
|---|---|
| **Device-to-host transfers** | Orange |
| **Host-to-device transfers** | Green |
| **Discarded device memory** | Light green |
| **Unknown events** | Dark gray |

The height of each event marker corresponds to the amount of memory that the event affected. Hovering over the marker will show the exact amount.

Expanding the row will show a breakdown of the events by each specific type.

The events will be captured if **"Collect WDDM Trace"** and **"Collect additional range of ETW events, including context status, allocations, sync wait and signal events, etc."** are enabled in the Nsight Systems Project Settings.

### System Committed VRAM

<a id="system-committed-vram"></a>

This row represents the total size of committed VRAM by all processes currently using the GPU. The stacked chart displays colored layers. Each layer corresponds to the VRAM commitment of a specific process.

To track VRAM commitment, enable the **"Collect WDDM Trace"** and **"Collect additional range of ETW events, including context status, allocations, sync wait and signal events, etc."** in Nsight Systems Project Settings.

### VRAM Resource Types Distribution

<a id="vram-resource-types"></a>

This row shows the distribution of VRAM usage across different resource types per process. It is color-coded to show the different resource types, and the height of each segment corresponds to the amount of VRAM used by that resource type.

Expand the chart's parent row to expose detailed separate rows for individual resource categories.

The events will be captured if **"Collect WDDM Trace"** and **"Collect additional range of ETW events, including context status, allocations, sync wait and signal events, etc."** are enabled in the Nsight Systems Project Settings. Additionally, to correlate Graphics API debug name events with resource migration events, the **"Collect DX12"** or **"Collect Vulkan"** option should be enabled.

### Vertical Synchronization

<a id="vsync"></a>

The **VSYNC rows** display when the monitor's vertical synchronizations occur.

---

## Quick Reference: CLI Options Summary

| Option | Values | Description |
|---|---|---|
| `--trace=d3d11` | - | Enable Direct3D 11 tracing |
| `--trace=d3d12` | - | Enable Direct3D 12 tracing |
| `--trace=vulkan` | - | Enable Vulkan tracing |
| `--trace=opengl` | - | Enable OpenGL tracing |
| `--trace=openxr` | - | Enable OpenXR tracing |
| `--dx12-gpu-workload` | `individual`, `batch`, `none` | D3D12 GPU workload display mode |
| `--dx12-wait-calls` | - | Enable D3D12 wait call tracing |
| `--vulkan-gpu-workload` | `individual`, `batch`, `none` | Vulkan GPU workload display mode |
| `--trace=wddm` | - | Enable WDDM queue tracing |

### Platform Availability

| Feature | Windows | Linux x86 | Linux Arm | QNX |
|---|---|---|---|---|
| D3D11 Trace | Yes | No | No | No |
| D3D12 Trace | Yes | No | No | No |
| WDDM Queues | Yes | No | No | No |
| WDDM HW Scheduler | Yes | No | No | No |
| Vulkan Trace | Yes | Yes | No | No |
| OpenGL Trace | Yes | Yes | Yes | No |
| OpenXR Trace | Yes | No | No | No |
| Stutter Analysis | Yes | No | No | No |
| GPUDirect Storage | No | Yes (x64, SBSA) | Yes (SBSA) | No |
