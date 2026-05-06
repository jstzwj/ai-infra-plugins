# Nsight Systems Graphics API Tracing Reference

## Table of Contents

- [Direct3D Trace](#direct3d-trace)
- [Vulkan API Trace](#vulkan-api-trace)
- [OpenGL Trace](#opengl-trace)
- [OpenXR API Trace](#openxr-api-trace)
- [GPUDirect Storage Trace](#gpudirect-storage-trace)
- [Custom ETW Trace](#custom-etw-trace)
- [Stutter Analysis](#stutter-analysis)
- [OpenMP Trace](#openmp-trace)

---

## Direct3D Trace

Nsight Systems supports tracing Direct3D 11 and Direct3D 12 applications on Windows.

### Enabling D3D Tracing

```bash
# Enable D3D11 tracing
nsys profile --trace=d3d11 my_application.exe

# Enable D3D12 tracing
nsys profile --trace=d3d12 my_application.exe

# Enable both D3D11 and D3D12
nsys profile --trace=d3d11,d3d12 my_application.exe

# Enable with additional features
nsys profile --trace=d3d12,nvtx,wddm --gpuctxsw=true my_application.exe
```

### D3D11 API Trace

D3D11 tracing captures the following:

| Category | Captured Events |
|---|---|
| **Device creation** | `D3D11CreateDevice`, `D3D11CreateDeviceAndSwapChain` |
| **Context operations** | `Draw`, `DrawIndexed`, `DrawInstanced`, `Dispatch` |
| **Resource operations** | `CopyResource`, `CopyBuffer`, `UpdateSubresource` |
| **Pipeline state** | `VSSetShader`, `PSSetShader`, `CSSetShader`, `OMSetRenderTargets` |
| **Synchronization** | `Flush`, `FinishFrameList`, `WaitForVBlank` |
| **Swap chain** | `Present`, `Present1` |

### D3D12 API Trace

D3D12 tracing provides detailed capture of the explicit low-level API:

#### Command Queue Operations

- `CreateCommandQueue` - Queue creation with type and priority
- `ExecuteCommandLists` - Command list submission
- `Signal` - Fence signal operations
- `Wait` - Fence wait operations
- `Present` - Swap chain present

#### Command List Operations

- `Open/close` - Command list lifetime
- `DrawInstanced`, `DrawIndexedInstanced`, `Dispatch`, `DispatchRays`, `ExecuteIndirect`
- `CopyBufferRegion`, `CopyTextureRegion`, `CopyResource`, `CopyTiles`
- `ResourceBarrier` - State transitions
- `SetPipelineState`, `SetComputeRootSignature`, `SetGraphicsRootSignature`
- `IASetVertexBuffers`, `IASetIndexBuffer`, `IASetPrimitiveTopology`
- `OMSetRenderTargets`, `RSSetViewports`, `RSSetScissorRects`

#### Resource Management

- `CreateCommittedResource` - Resource allocation
- `CreatePlacedResource` - Resource placement in heap
- `CreateReservedResource` - Tiled resource creation
- `CreateHeap` - Memory heap creation

#### Pipeline State

- `CreateGraphicsPipelineState` - Graphics pipeline creation
- `CreateComputePipelineState` - Compute pipeline creation
- `CreateRaytracingPipelineState` - RT pipeline creation
- `SetPipelineState1` - State object setting

### SLI Trace

For multi-GPU (SLI) configurations:

```bash
nsys profile --trace=d3d12 --sli=true my_application.exe
```

SLI tracing captures:

- Multi-GPU synchronization events
- Cross-GPU memory transfers
- Alternate frame rendering events
- SLI bridge utilization

| SLI Mode | Description |
|---|---|
| **AFR (Alternate Frame Rendering)** | Each GPU renders alternate frames |
| **SFR (Split Frame Rendering)** | Each GPU renders a portion of the frame |
| **Single GPU** | One GPU handles all rendering |

### WDDM Queues

The Windows Display Driver Model (WDDM) queue tracing captures GPU scheduling information:

#### Command Buffer Types

| Buffer Type | Description |
|---|---|
| **Render** | Graphics rendering commands |
| **Compute** | Compute shader dispatch commands |
| **Copy** | Resource copy and transfer commands |
| **Page flip** | Display presentation commands |
| **Paging** | GPU memory management (migration, eviction) |
| **System** | Driver-internal system commands |

#### Enabling WDDM Tracing

```bash
nsys profile --trace=wddm my_application.exe
```

WDDM tracing is Windows-only and provides visibility into:

- When the GPU driver submits work to the hardware
- Queue depths and scheduling behavior
- Preemption events
- Memory paging operations

### WDDM HW Scheduler

When the Hardware Scheduler is active (WDDM 3.0+), Nsight Systems captures:

- HW queue scheduling decisions
- HW context switch events
- Doorbell ring events
- HW fence signaling

```bash
nsys profile --trace=wddm --wddm-hwscheduler=true my_application.exe
```

---

## Vulkan API Trace

Nsight Systems captures Vulkan API calls for graphics and compute applications.

### Enabling Vulkan Tracing

```bash
nsys profile --trace=vulkan my_application
```

### Overview of Vulkan Tracing

Vulkan tracing captures the full command buffer lifecycle:

| Phase | Captured Operations |
|---|---|
| **Instance/Device** | `vkCreateInstance`, `vkCreateDevice`, `vkEnumeratePhysicalDevices` |
| **Queue operations** | `vkQueueSubmit`, `vkQueuePresentKHR`, `vkQueueBindSparse` |
| **Command buffers** | `vkBeginCommandBuffer`, `vkEndCommandBuffer`, `vkResetCommandBuffer` |
| **Render pass** | `vkCmdBeginRenderPass`, `vkCmdEndRenderPass`, `vkCmdNextSubpass` |
| **Draw/Dispatch** | `vkCmdDraw`, `vkCmdDrawIndexed`, `vkCmdDispatch`, `vkCmdDrawIndirect` |
| **Compute** | `vkCmdDispatch`, `vkCmdDispatchIndirect` |
| **Transfer** | `vkCmdCopyBuffer`, `vkCmdCopyImage`, `vkCmdCopyBufferToImage`, `vkCmdBlitImage` |
| **Synchronization** | `vkCmdPipelineBarrier`, `vkCmdSetEvent`, `vkCmdWaitEvents` |
| **Descriptors** | `vkAllocateDescriptorSets`, `vkUpdateDescriptorSets` |
| **Pipeline** | `vkCreateGraphicsPipelines`, `vkCreateComputePipelines` |

### Command Buffer Creation

Nsight Systems captures the recording of command buffers and displays them in the timeline:

- Each `vkQueueSubmit` appears as a group of work items.
- Individual commands within a command buffer are shown as sub-ranges.
- Command buffer boundaries are visible for correlation with application logic.

### Pipeline Creation Feedback

When Vulkan pipeline creation is traced, the timeline shows:

- `vkCreateGraphicsPipelines` duration
- `vkCreateComputePipelines` duration
- Cache hit/miss information (if supported)
- Pipeline compilation time breakdown

Pipeline creation feedback helps identify shader compilation stalls that affect frame timing.

---

## OpenGL Trace

Nsight Systems captures OpenGL and OpenGL ES API calls.

### Enabling OpenGL Tracing

```bash
# Full OpenGL trace
nsys profile --trace=opengl my_application

# OpenGL with specific extensions
nsys profile --trace=opengl --ogl-milliseconds-per-slice=1 my_application
```

### CPU Trace

OpenGL CPU-side tracing captures:

| Category | Events |
|---|---|
| **Draw calls** | `glDrawArrays`, `glDrawElements`, `glDrawArraysInstanced`, `glDrawElementsInstanced` |
| **Compute** | `glDispatchCompute`, `glDispatchComputeIndirect` |
| **State** | `glUseProgram`, `glBindTexture`, `glBindBuffer`, `glBindFramebuffer` |
| **Transfer** | `glBufferData`, `glBufferSubData`, `glTexSubImage2D` |
| **Synchronization** | `glFenceSync`, `glClientWaitSync`, `glWaitSync` |
| **Swap** | `eglSwapBuffers`, `glXSwapBuffers`, `wglSwapBuffers` |
| **Framebuffer** | `glFramebufferTexture2D`, `glReadPixels` |

### GPU Trace

OpenGL GPU tracing shows when the GPU actually executes submitted commands:

- GPU-side durations of draw calls and compute dispatches
- GPU-side memory operations
- GPU idle periods
- Correlation with CPU-side API calls

### KHR_debug

OpenGL `GL_KHR_debug` extension provides debug annotation:

```c
// Push a debug group
glPushDebugGroupKHR(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "Render Scene");

// Label an object
glObjectLabelKHR(GL_BUFFER, bufferId, -1, "Vertex Buffer");

// Pop the debug group
glPopDebugGroupKHR();
```

These annotations appear in the timeline as labeled ranges and object names.

---

## OpenXR API Trace

Nsight Systems supports tracing OpenXR applications for VR/AR profiling.

### Enabling OpenXR Tracing

```bash
nsys profile --trace=openxr my_application
```

### Captured OpenXR Events

| Category | Events |
|---|---|
| **Session** | `xrBeginSession`, `xrEndSession`, `xrRequestExitSession` |
| **Frame** | `xrWaitFrame`, `xrBeginFrame`, `xrEndFrame` |
| **Swapchain** | `xrCreateSwapchain`, `xrEnumerateSwapchainImages`, `xrAcquireSwapchainImage` |
| **Space** | `xrCreateSpace`, `xrLocateSpace`, `xrGetSpaceLocation` |
| **Actions** | `xrCreateAction`, `xrSuggestInteractionProfileBindings`, `xrSyncActions` |
| **Composition** | `xrCreateProjectionLayer`, `xrCreateQuadLayer` |

---

## GPUDirect Storage Trace

Nsight Systems can trace GPUDirect Storage (GDS) operations for direct GPU-to-storage data transfers.

### Enabling GDS Tracing

```bash
nsys profile --trace=gds my_application
```

### Captured GDS Events

| Event | Description |
|---|---|
| **cuFileRead** | Direct read from storage to GPU memory |
| **cuFileWrite** | Direct write from GPU memory to storage |
| **cuFileBatchIO** | Batch I/O submission |
| **Registration** | `cuFileHandleRegister`, `cuFileBufRegister` |

GDS tracing helps identify bottlenecks in data loading pipelines that use direct GPU-storage paths.

---

## Custom ETW Trace

On Windows, Nsight Systems supports custom Event Tracing for Windows (ETW) providers.

### Enabling Custom ETW

```bash
nsys profile --trace=etw --etw-provider="{GUID}" my_application.exe
```

### Configuration

| Parameter | Description |
|---|---|
| `--etw-provider` | ETW provider GUID to enable |
| `--etw-level` | Trace level (0-255, default: 255 for all) |
| `--etw-keywords` | Keyword bitmask to filter events |

### Example: Tracing DXGI

```bash
nsys profile --trace=etw --etw-provider="{CA13C140-4EB6-5FAB-5DC2-6A6B22A41925}" my_application.exe
```

---

## Stutter Analysis

Nsight Systems provides comprehensive stutter analysis for graphics applications, particularly games and interactive rendering.

### FPS Overview

The FPS Overview section provides frame-level analysis.

#### Frame Duration

| Metric | Description |
|---|---|
| **Frame time** | Duration from one Present call to the next |
| **Average FPS** | Mean frames per second over the trace |
| **1% Low FPS** | Average FPS of the slowest 1% of frames |
| **0.1% Low FPS** | Average FPS of the slowest 0.1% of frames |
| **Stutter count** | Number of frames that exceed the stutter threshold |

#### Stutter Algorithm

A frame is classified as a **stutter** if its duration exceeds 2x the average frame duration. For example:

- Target: 60 FPS (16.67ms per frame)
- Average frame time: 17ms
- Stutter threshold: 34ms
- Any frame > 34ms is counted as a stutter

#### OSC (Out-of-Sequence Completions) Detection

OSC detects frames that are displayed out of order, which can cause visible stuttering even when average FPS is high:

- **Triple buffering**: Can cause OSC when the display reads a frame out of order.
- **Pre-rendered frames**: Driver queue of pre-rendered frames can cause OSC.

### Reflex SDK

NVIDIA Reflex SDK integration provides low-latency pipeline analysis.

#### Pipeline Stages

| Stage | Description | Marker |
|---|---|---|
| **Simulation** | Game logic, physics, AI updates | `Reflex_Mark_Simulation` |
| **RenderSubmit** | Command buffer recording and submission | `Reflex_Mark_RenderSubmit` |
| **Present** | SwapChain Present call | `Reflex_Mark_Present` |
| **Driver** | GPU driver processing | Automatic |
| **OS Render Queue** | OS compositor queue time | Automatic |
| **GPU Render** | Actual GPU execution time | Automatic |

#### Reflex Metrics

```bash
nsys profile --trace=reflex,d3d12 my_application.exe
```

| Metric | Description |
|---|---|
| **Game to Render** | Total latency from input to display |
| **Render to Present** | Time from first render to Present call |
| **Present to Display** | Time from Present to actual scanout |
| **Input to Photon** | End-to-end latency (if input markers are used) |

### Performance Warnings Row

The Performance Warnings row in the timeline highlights:

| Warning | Description |
|---|---|
| **Excessive Present calls** | Present called too frequently without vsync |
| **GPU starved** | GPU idle while CPU is busy |
| **CPU-bound frame** | Frame time dominated by CPU processing |
| **Render target mismatch** | Mismatched render target sizes causing resolve operations |
| **Synchronization stall** | Unnecessary CPU-GPU synchronization |

### Frame Health Row

The Frame Health row provides per-frame quality indicators:

| Indicator | Color | Meaning |
|---|---|---|
| **Green** | Healthy | Frame within expected duration |
| **Yellow** | Warning | Frame slightly above expected duration |
| **Red** | Stutter | Frame significantly above expected duration |
| **Purple** | Dropped | Frame was dropped and never displayed |

### Windows GPU Memory Utilization

GPU memory utilization shows memory usage over time:

- **Committed memory**: Total GPU memory allocated
- **Resident memory**: Memory physically present on the GPU
- **Budget**: Available GPU memory (from DXGI query)

```bash
nsys profile --trace=d3d12 --gpumem=true my_application.exe
```

### Resource Allocations, Migrations, and Memory Transfer

| Event Type | Description |
|---|---|
| **Allocation** | New resource created on the GPU |
| **Deallocation** | Resource freed from GPU memory |
| **Migration** | Resource moved between memory segments (local, system, non-local) |
| **Page table update** | Virtual-to-physical mapping changes |
| **Eviction** | Resource evicted from GPU memory to system memory |
| **Make resident** | Resource made resident in GPU memory |

### Vertical Synchronization

VSync events show when frames are actually displayed:

- **VBlank**: Vertical blanking interval (display refresh)
- **Present**: When Present was called
- **Scanout**: When the frame was actually read by the display

Alignment (or misalignment) between Present calls and VBlank intervals reveals:
- VSync effectiveness
- Tearing potential
- Frame pacing quality

---

## OpenMP Trace

Nsight Systems supports tracing OpenMP applications using the OMPT (OpenMP Tools) interface.

### Enabling OpenMP Tracing

```bash
nsys profile --trace=openmp my_application
```

### OMPT Callbacks List

Nsight Systems captures the following OMPT callbacks:

#### Parallel Region Callbacks

| Callback | Description |
|---|---|
| `ompt_callback_parallel_begin` | Start of a parallel region |
| `ompt_callback_parallel_end` | End of a parallel region |
| `ompt_callback_implicit_task` | Implicit task creation within a parallel region |
| `ompt_callback_explicit_task` | Explicit task creation (`#pragma omp task`) |
| `ompt_callback_task_create` | Task creation event |
| `ompt_callback_task_schedule` | Task scheduling event |
| `ompt_callback_task_complete` | Task completion event |
| `ompt_callback_task_dependence` | Task dependency event |

#### Synchronization Callbacks

| Callback | Description |
|---|---|
| `ompt_callback_mutex_acquire` | Mutex acquisition begin |
| `ompt_callback_mutex_acquired` | Mutex successfully acquired |
| `ompt_callback_mutex_released` | Mutex released |
| `ompt_callback_nest_lock` | Nested lock operation |
| `ompt_callback_sync_region` | Synchronization region (barrier, critical, etc.) |
| `ompt_callback_sync_region_wait` | Waiting at a synchronization point |
| `ompt_callback_control_tool` | `omp control tool` directive |
| `ompt_callback_flush` | `#pragma omp flush` |

#### Worksharing Callbacks

| Callback | Description |
|---|---|
| `ompt_callback_work` | Worksharing construct (for, sections, single, etc.) |
| `ompt_callback_dispatch` | Loop dispatch event (iteration assignment) |
| `ompt_callback_reduction` | Reduction operation |

#### Device Callbacks

| Callback | Description |
|---|---|
| `ompt_callback_device_initialize` | Target device initialization |
| `ompt_callback_device_finalize` | Target device finalization |
| `ompt_callback_device_load` | Module load on target device |
| `ompt_callback_target` | Target region execution |
| `ompt_callback_target_data_op` | Data transfer to/from target |
| `ompt_callback_target_submit` | Kernel submission to target |
| `ompt_callback_target_map` | Memory mapping for target region |

### OpenMP Timeline Display

In the timeline, OpenMP events are shown per-thread with color coding:

| Color | Region Type |
|---|---|
| Blue | Parallel regions |
| Green | Worksharing constructs (for loops) |
| Orange | Synchronization (barriers) |
| Red | Critical sections |
| Purple | Task execution |
| Yellow | Task scheduling overhead |

### OpenMP Analysis Tips

1. **Imbalance detection**: Look for threads waiting at barriers while others are still working.
2. **Overhead measurement**: Compare time in OpenMP runtime vs. actual user code.
3. **Scaling analysis**: Profile with different thread counts to find optimal parallelism.
4. **False sharing**: Check for frequent cache line migrations between threads working on adjacent data.

---

## Cross-API Tracing

You can enable multiple API traces simultaneously:

```bash
# Trace CUDA + Vulkan + NVTX
nsys profile --trace=cuda,vulkan,nvtx my_application

# Trace D3D12 + OpenGL + OpenMP
nsys profile --trace=d3d12,opengl,openmp my_application.exe

# Full graphics and compute trace
nsys profile --trace=cuda,vulkan,opengl,nvtx,osrt my_application
```

### Trace Combination Matrix

| API | CUDA | Vulkan | OpenGL | D3D11 | D3D12 | OpenXR |
|---|---|---|---|---|---|---|
| **CUDA** | - | Yes | Yes | No | Yes | Yes |
| **Vulkan** | Yes | - | No | No | No | Yes |
| **OpenGL** | Yes | No | - | No | No | No |
| **D3D11** | No | No | No | - | No | No |
| **D3D12** | Yes | No | No | No | - | Yes |
| **OpenXR** | Yes | Yes | No | No | Yes | - |

> **Note**: D3D11 and D3D12 are Windows-only. Vulkan, OpenGL, and CUDA are cross-platform.

---

## See Also

- [CLI Reference](01-cli-usage.md)
- [GUI Report Analysis](07-gui-report-analysis.md)
- [Python and CPU Profiling](09-python-cpu-profiling.md)
- [Export Formats and SQLite Schema](11-export-sqlite-schema.md)
