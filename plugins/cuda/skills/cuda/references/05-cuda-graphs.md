# 5. CUDA Graphs

This document provides comprehensive coverage of CUDA Graphs -- a model for work submission that defines a series of operations connected by dependencies separately from execution. CUDA Graphs reduce CPU launch overhead and enable whole-workflow optimizations that are not possible with the piecewise work submission of streams.

**CUDA Toolkit Version:** 13.2 (March 2026)

---

## Table of Contents

1. [Overview](#51-overview)
2. [Node Types](#52-node-types)
3. [Building Graphs](#53-building-graphs)
4. [Graph Instantiation and Execution](#54-graph-instantiation-and-execution)
5. [Updating Graphs](#55-updating-graphs)
6. [Conditional Graph Nodes](#56-conditional-graph-nodes)
7. [Graph Memory Nodes](#57-graph-memory-nodes)
8. [Device Graph Launch](#58-device-graph-launch-cc-90)
9. [User Objects](#59-user-objects)

---

## 5.1 Overview

A CUDA graph is a series of operations (kernel launches, data movement, etc.) connected by dependencies, defined separately from its execution. This allows a graph to be defined once and then launched repeatedly.

### Benefits

1. **Reduced CPU launch costs** -- much of the setup is done in advance during graph definition and instantiation, rather than during each launch.
2. **Whole-workflow optimizations** -- presenting the entire workflow to CUDA enables optimizations not possible with piecewise stream submission.

### Three Stages of Graph Usage

1. **Definition** -- The graph is captured or constructed. This is done once.
2. **Instantiation** -- The graph is validated and optimized into an executable form. This is done once after definition.
3. **Execution** -- The instantiated graph is launched repeatedly. CPU overhead per launch is minimal.

### Edge Data

CUDA 12.3 introduced edge data on graphs. The only non-default edge type is `cudaGraphDependencyTypeProgrammatic`, which enables Programmatic Dependent Launch between two kernel nodes. Edge data is also available in stream capture APIs: `cudaStreamBeginCaptureToGraph()`, `cudaStreamGetCaptureInfo()`, and `cudaStreamUpdateCaptureDependencies()`.

---

## 5.2 Node Types

CUDA Graphs support the following node types:

| Node Type | Description |
|-----------|-------------|
| **Kernel** | Launches a CUDA kernel with specified grid/block configuration |
| **CPU function call (Host)** | Calls a function on the host |
| **Memory copy** | Performs `cudaMemcpy` operations (1D, 2D, 3D, peer) |
| **Memset** | Fills memory with a value (1D, 2D, 3D) |
| **Empty** | No-op node used as a synchronization point or placeholder |
| **Event wait** | Waits for a CUDA event |
| **Event record** | Records a CUDA event |
| **External semaphore signal** | Signals external semaphores |
| **External semaphore wait** | Waits on external semaphores |
| **Conditional** | IF / WHILE / SWITCH conditional execution (CUDA 12.4+) |
| **Memory alloc** | Allocates memory with GPU-ordered lifetime |
| **Memory free** | Frees memory allocated by a memory alloc node |
| **Child graph** | Embeds a sub-graph as a node |

---

## 5.3 Building Graphs

Graphs can be created via two mechanisms: the explicit **Graph API** and **Stream Capture**.

### 5.3.1 Graph API

The Graph API allows manual construction of graphs by creating nodes and specifying dependencies.

```cpp
cudaGraph_t graph;
cudaGraphCreate(&graph, 0);

cudaGraphNode_t nodes[4];
cudaGraphNodeParams kParams = { cudaGraphNodeTypeKernel };
kParams.kernel.func = (void*)myKernel;
kParams.kernel.gridDim = dim3(gridX, 1, 1);
kParams.kernel.blockDim = dim3(blockX, 1, 1);
kParams.kernel.sharedMemBytes = 0;
kParams.kernel.kernelParams = (void**)kernelArgs;

// Node 0: no dependencies (root)
cudaGraphAddNode(&nodes[0], graph, NULL, NULL, 0, &kParams);

// Nodes 1 and 2: depend on node 0
const cudaGraphNode_t deps0[] = { nodes[0] };
cudaGraphAddNode(&nodes[1], graph, deps0, NULL, 1, &kParams);
cudaGraphAddNode(&nodes[2], graph, deps0, NULL, 1, &kParams);

// Node 3: depends on nodes 1 and 2 (fan-in)
const cudaGraphNode_t deps12[] = { nodes[1], nodes[2] };
cudaGraphAddNode(&nodes[3], graph, deps12, NULL, 2, &kParams);
```

This creates a diamond-shaped dependency graph:
```
     [0]
    /   \
  [1]   [2]
    \   /
     [3]
```

For adding nodes of other types, use the appropriate `cudaGraphNodeType` value in `cudaGraphNodeParams`:
- `cudaGraphNodeTypeMemcpy` -- memory copy node
- `cudaGraphNodeTypeMemset` -- memory set node
- `cudaGraphNodeTypeHost` -- host function call node
- `cudaGraphNodeTypeEventRecord` -- event record node
- `cudaGraphNodeTypeEventWait` -- event wait node
- `cudaGraphNodeTypeMemAlloc` -- memory allocation node
- `cudaGraphNodeTypeMemFree` -- memory free node
- `cudaGraphNodeTypeConditional` -- conditional node
- `cudaGraphNodeTypeGraph` -- child graph node

### 5.3.2 Stream Capture

Stream capture provides a mechanism to create a graph from existing stream-based code. A section of code that launches work into streams is bracketed with `cudaStreamBeginCapture()` and `cudaStreamEndCapture()`.

```cpp
cudaGraph_t graph;
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

// All work launched into `stream` is captured, not executed
kernel_A<<<grid, block, 0, stream>>>(...);
cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream);
kernel_B<<<grid, block, 0, stream>>>(...);

cudaStreamEndCapture(stream, &graph);
```

During capture, work is **appended to an internal graph** rather than being enqueued for execution. The completed graph is returned by `cudaStreamEndCapture()`.

#### Capture Modes

```cpp
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);        // Default
cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);   // Thread-local capture
cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed);       // Relaxed ordering
```

#### Capture to Existing Graph

```cpp
cudaStreamBeginCaptureToGraph(stream, graph,
                               dependencies, edgeData,
                               numDeps, cudaStreamCaptureModeGlobal);
```

This captures work into a user-provided graph instead of an internal graph.

#### Cross-Stream Dependencies in Capture

Cross-stream dependencies via events are captured correctly:

```cpp
cudaStreamBeginCapture(stream1);

kernel_A<<<..., stream1>>>(..);
cudaEventRecord(event, stream1);

cudaStreamWaitEvent(stream2, event);
kernel_B<<<..., stream2>>>(..);

cudaEventRecord(event2, stream2);
cudaStreamWaitEvent(stream1, event2);

kernel_C<<<..., stream1>>>(..);

cudaStreamEndCapture(stream1, &graph);
```

**Important:** `cudaStreamEndCapture()` must be called on the same stream (the **origin stream**) where `cudaStreamBeginCapture()` was called. All other captured streams must be joined back to the origin stream before ending capture.

#### Prohibited Operations During Capture

- Synchronous APIs (`cudaMemcpy`, `cudaMemset`, `cudaDeviceSynchronize`, etc.)
- Synchronizing or querying a captured stream (`cudaStreamSynchronize`, `cudaStreamQuery`)
- `cudaEventQuery()` / `cudaEventSynchronize()` on events recorded in captured streams
- `cudaStreamWaitEvent()` on events from non-captured streams (except via event dependencies)

If an invalid operation is attempted, capture is **invalidated**. Subsequent use returns errors until `cudaStreamEndCapture()` is called, which returns an error and a NULL graph.

#### Introspection and Debugging

```cpp
// Get capture info for introspection
enum cudaStreamCaptureStatus captureStatus;
unsigned long long captureId;
cudaStreamGetCaptureInfo(stream, &captureStatus, &captureId,
                          NULL, NULL, NULL);

// Generate Graphviz DOT file for visualization
cudaGraphDebugDotPrint(graph, "graph.dot", 0);
// Then render: dot -Tpng graph.dot -o graph.png
```

### 5.3.3 Comparing Graph API vs Stream Capture

| Aspect | Graph API | Stream Capture |
|--------|-----------|----------------|
| Approach | Explicit node creation and dependency specification | Wraps existing stream-based code |
| Flexibility | Full control over graph topology | Limited to what streams can express |
| Error handling | Easier to pinpoint issues | Capture invalidation can be opaque |
| Use case | New code, complex topologies | Wrapping existing libraries, rapid prototyping |

---

## 5.4 Graph Instantiation and Execution

### 5.4.1 Instantiation

Once a graph has been created, it must be instantiated to create an executable graph:

```cpp
cudaGraphExec_t graphExec;
cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
```

Instantiation validates the graph, performs optimizations, and prepares all runtime structures needed for fast execution.

For instantiation with special flags:

```cpp
cudaGraphExec_t graphExec;
cudaGraphInstantiateWithFlags(&graphExec, graph,
    cudaGraphInstantiateFlagAutoFreeOnLaunch);  // Auto-free on relaunch
```

### 5.4.2 Execution

```cpp
cudaGraphLaunch(graphExec, stream);
```

The graph is launched into the specified stream for ordering with other asynchronous work. The stream is for **ordering only** -- it does not constrain the internal parallelism of the graph, nor does it affect where graph nodes execute.

### 5.4.3 Complete Workflow Example

```cpp
#define N 500000
__global__ void shortKernel(float* out_d, float* in_d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) out_d[idx] = 1.23f * in_d[idx];
}

bool graphCreated = false;
cudaGraph_t graph;
cudaGraphExec_t instance;

for (int istep = 0; istep < NSTEP; istep++) {
    if (!graphCreated) {
        // Stage 1: Capture the graph
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        for (int i = 0; i < NKERNEL; i++) {
            shortKernel<<<blocks, threads, 0, stream>>>(out_d, in_d);
        }
        cudaStreamEndCapture(stream, &graph);

        // Stage 2: Instantiate
        cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
        graphCreated = true;
    }

    // Stage 3: Execute (repeated, low overhead)
    cudaGraphLaunch(instance, stream);
    cudaStreamSynchronize(stream);
}

// Cleanup
cudaGraphExecDestroy(instance);
cudaGraphDestroy(graph);
```

### 5.4.4 Thread Safety and Concurrency

- `cudaGraph_t` objects are **not thread-safe**. Multiple threads must not concurrently access the same `cudaGraph_t`.
- A `cudaGraphExec_t` **cannot run concurrently with itself**. A launch is ordered after previous launches of the same executable graph.

---

## 5.5 Updating Graphs

When a workflow changes, the graph must be modified. For **topology changes**, re-instantiation is required. However, when only **node parameters** change (kernel args, memory addresses, etc.), CUDA provides lightweight update mechanisms.

### 5.5.1 Whole Graph Update

`cudaGraphExecUpdate()` updates an instantiated graph with parameters from a topologically identical graph:

```cpp
cudaGraphExec_t graphExec = NULL;

for (int i = 0; i < 10; i++) {
    cudaGraph_t graph;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    do_cuda_work(stream);  // user-defined stream-based workload
    cudaStreamEndCapture(stream, &graph);

    if (graphExec != NULL) {
        // Try lightweight update
        cudaGraphExecUpdateResult updateResult;
        cudaGraphNode_t errorNode;
        cudaGraphExecUpdate(graphExec, graph, &errorNode, &updateResult);

        if (updateResult != cudaGraphExecUpdateSuccess) {
            // Update failed; need full re-instantiation
            cudaGraphExecDestroy(graphExec);
            graphExec = NULL;
        }
    }

    if (graphExec == NULL) {
        cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    }

    cudaGraphDestroy(graph);
    cudaGraphLaunch(graphExec, stream);
    cudaStreamSynchronize(stream);
}
```

**Requirements for successful update:**

1. The topology of the updating graph must be **identical** to the original.
2. The order in which dependencies are specified must match.
3. Sink nodes (no dependents) must be consistently ordered.
4. For captured streams, API calls must be made in the same order.

### 5.5.2 Individual Node Update APIs

When only a few nodes need updating, individual node update is more efficient:

| API | Node Type |
|-----|-----------|
| `cudaGraphExecKernelNodeSetParams()` | Kernel |
| `cudaGraphExecMemcpyNodeSetParams()` | Memory copy |
| `cudaGraphExecMemsetNodeSetParams()` | Memset |
| `cudaGraphExecHostNodeSetParams()` | Host |
| `cudaGraphExecChildGraphNodeSetParams()` | Child graph |
| `cudaGraphExecEventRecordNodeSetEvent()` | Event record |
| `cudaGraphExecEventWaitNodeSetEvent()` | Event wait |
| `cudaGraphExecExternalSemSignNodeSetParams()` | External semaphore signal |
| `cudaGraphExecExternalSemWaitNodeSetParams()` | External semaphore wait |

Example:

```cpp
// Update a kernel node's parameters
cudaKernelNodeParams newParams = {0};
newParams.func = (void*)myUpdatedKernel;
newParams.gridDim = dim3(newGrid, 1, 1);
newParams.blockDim = dim3(newBlock, 1, 1);
newParams.kernelParams = (void**)newArgs;
cudaGraphExecKernelNodeSetParams(graphExec, kernelNode, &newParams);
```

### 5.5.3 Individual Node Enable/Disable

Kernel, memset, and memcpy nodes can be enabled or disabled:

```cpp
// Disable a node (becomes functionally equivalent to empty node)
cudaGraphNodeSetEnabled(graphExec, node, 0);

// Re-enable a node
cudaGraphNodeSetEnabled(graphExec, node, 1);

// Query enable state
unsigned int isEnabled;
cudaGraphNodeGetEnabled(graphExec, node, &isEnabled);
```

Disabled nodes are equivalent to empty nodes. Node parameters are preserved and take effect when re-enabled. Enable state is unaffected by individual node update or whole graph update.

### 5.5.4 Update Limitations

**Kernel nodes:**
- The owning context of the function cannot change.
- A node whose function did not use CDP cannot be updated to one that does.

**Memset/Memcpy nodes:**
- Device allocation locations cannot change.
- Source/destination must be from the same context as the original.
- Only 1D memset/memcpy nodes can change size.

**Memcpy additional restrictions:**
- Changing source/destination memory type or transfer kind is not supported.

**Conditional nodes:**
- Handle creation and assignment order must match.
- Changing node parameters (number of graphs, context) is not supported.

**Memory nodes:**
- Cannot update a `cudaGraphExec_t` if the source graph is currently instantiated as a different executable graph.

---

## 5.6 Conditional Graph Nodes

Conditional nodes (CUDA 12.4+) allow conditional execution and looping within a graph, enabling dynamic and iterative workflows without CPU involvement.

### 5.6.1 Types

| Type | Behavior |
|------|----------|
| **IF** | Execute body graph once if condition is non-zero. Optional else body if zero. |
| **WHILE** | Execute body graph repeatedly while condition is non-zero. |
| **SWITCH** | Execute the nth body graph once if condition equals n. |

### 5.6.2 Conditional Handles

A condition value is represented by `cudaGraphConditionalHandle`:

```cpp
cudaGraphConditionalHandle handle;

// Without default value (undefined at start of each execution)
cudaGraphConditionalHandleCreate(&handle, graph);

// With default value (set at beginning of each graph execution)
cudaGraphConditionalHandleCreate(&handle, graph, 1,
                                  cudaGraphCondAssignDefault);
```

- The handle must be associated with a single conditional node.
- Handles cannot be destroyed (no explicit cleanup needed).
- If `cudaGraphCondAssignDefault` is not provided, the condition value is **undefined** at the start of each execution and should not be assumed to persist.

### 5.6.3 Setting Condition Values from Device Code

```cpp
__global__ void setHandle(cudaGraphConditionalHandle handle, int value) {
    // Set the condition value
    cudaGraphSetConditional(handle, value);
}
```

### 5.6.4 IF Node Example

```cpp
void createIfNode() {
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    cudaGraphNode_t node;

    // Create graph and handle
    cudaGraphCreate(&graph, 0);
    cudaGraphConditionalHandle handle;
    cudaGraphConditionalHandleCreate(&handle, graph);

    // Upstream kernel sets the condition value
    void* kernelArgs[2];
    int value = 1;
    cudaGraphNodeParams params = { cudaGraphNodeTypeKernel };
    params.kernel.func = (void*)setHandle;
    params.kernel.gridDim = dim3(1, 1, 1);
    params.kernel.blockDim = dim3(1, 1, 1);
    params.kernel.kernelParams = kernelArgs;
    kernelArgs[0] = &handle;
    kernelArgs[1] = &value;
    cudaGraphAddNode(&node, graph, NULL, 0, &params);

    // Create IF conditional node
    cudaGraphNodeParams cParams = { cudaGraphNodeTypeConditional };
    cParams.conditional.handle = handle;
    cParams.conditional.type = cudaGraphCondTypeIf;
    cParams.conditional.size = 1;  // 1 body graph (no else)
    cudaGraphAddNode(&node, graph, &node, 1, &cParams);

    // Get and populate the body graph
    cudaGraph_t bodyGraph = cParams.conditional.phGraph_out[0];
    // ... add nodes to bodyGraph ...

    // Instantiate and launch
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    cudaGraphLaunch(graphExec, 0);
    cudaDeviceSynchronize();

    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
}
```

For IF-ELSE, set `cParams.conditional.size = 2` and use `phGraph_out[0]` (if body) and `phGraph_out[1]` (else body).

### 5.6.5 WHILE Node Example

```cpp
__global__ void loopKernel(cudaGraphConditionalHandle handle, char* dPtr) {
    if (--(*dPtr) == 0) {
        cudaGraphSetConditional(handle, 0);  // Stop loop
    }
}

void createWhileNode() {
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    char* dPtr;
    cudaMalloc((void**)&dPtr, 1);

    cudaGraphCreate(&graph, 0);

    // Handle with default value 1 (loop starts enabled)
    cudaGraphConditionalHandle handle;
    cudaGraphConditionalHandleCreate(&handle, graph, 1,
                                      cudaGraphCondAssignDefault);

    // Create WHILE node
    cudaGraphNodeParams cParams = { cudaGraphNodeTypeConditional };
    cParams.conditional.handle = handle;
    cParams.conditional.type = cudaGraphCondTypeWhile;
    cParams.conditional.size = 1;
    cudaGraphAddNode(&node, graph, NULL, 0, &cParams);

    // Populate body graph
    cudaGraph_t bodyGraph = cParams.conditional.phGraph_out[0];
    cudaGraphNodeParams params = { cudaGraphNodeTypeKernel };
    params.kernel.func = (void*)loopKernel;
    params.kernel.gridDim = dim3(1, 1, 1);
    params.kernel.blockDim = dim3(1, 1, 1);
    params.kernel.kernelParams = kernelArgs;
    kernelArgs[0] = &handle;
    kernelArgs[1] = &dPtr;
    cudaGraphAddNode(&node, bodyGraph, NULL, 0, &params);

    // Initialize dPtr to 10 -> loop runs 10 times
    cudaMemset(dPtr, 10, 1);

    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    cudaGraphLaunch(graphExec, 0);
    cudaDeviceSynchronize();

    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaFree(dPtr);
}
```

### 5.6.6 SWITCH Node Example

```cpp
void createSwitchNode() {
    cudaGraph_t graph;
    cudaGraphCreate(&graph, 0);

    cudaGraphConditionalHandle handle;
    cudaGraphConditionalHandleCreate(&handle, graph);

    // Upstream kernel sets handle value (0..4)
    // ... add setHandle kernel node ...

    // Create SWITCH with 5 body graphs
    cudaGraphNodeParams cParams = { cudaGraphNodeTypeConditional };
    cParams.conditional.handle = handle;
    cParams.conditional.type = cudaGraphCondTypeSwitch;
    cParams.conditional.size = 5;  // 5 cases
    cudaGraphAddNode(&node, graph, &prevNode, 1, &cParams);

    // Get body graphs
    cudaGraph_t* bodyGraphs = cParams.conditional.phGraph_out;
    // Populate bodyGraphs[0] through bodyGraphs[4]
    for (int i = 0; i < 5; i++) {
        // ... add nodes to bodyGraphs[i] ...
    }

    // If condition == n, bodyGraphs[n] executes
    // If condition does not match any, no body graph executes
}
```

### 5.6.7 Body Graph Requirements

- All nodes must reside on a **single device**.
- Body graphs can contain: kernel, empty, memcpy, memset, child graph, and conditional nodes.
- **No CUDA Dynamic Parallelism (CDP)** or Device Graph Launch by kernels.
- Cooperative launches are permitted (if MPS is not in use).
- Only copies/memsets involving device memory and/or pinned device-mapped host memory.
- Copies/memsets involving CUDA arrays are not permitted.
- Conditional nodes can be **nested**.

---

## 5.7 Graph Memory Nodes

Graph memory nodes allow graphs to create and own memory allocations with **GPU-ordered lifetime semantics**.

### 5.7.1 Overview

- Graph allocations have **fixed addresses** over the life of a graph (including repeated instantiations and launches).
- Virtual addresses are assigned at node creation time and remain stable even when CUDA changes the backing physical memory.
- Physical memory can be reused across allocations whose GPU-ordered lifetimes do not overlap (both within and across graphs).
- Memory nodes are ordered within a graph by dependency edges, just like other node types.

### 5.7.2 Node Types

- `cudaGraphNodeTypeMemAlloc` -- allocation node
- `cudaGraphNodeTypeMemFree` -- free node

### 5.7.3 Creating Memory Nodes (Graph API)

```cpp
cudaGraphCreate(&graph, 0);

// Create allocation node
cudaGraphNodeParams allocParams = { cudaGraphNodeTypeMemAlloc };
allocParams.alloc.poolProps.allocType = cudaMemAllocationTypePinned;
allocParams.alloc.poolProps.location.type = cudaMemLocationTypeDevice;
allocParams.alloc.poolProps.location.id = 0;  // device 0
allocParams.alloc.bytesize = size;
cudaGraphAddNode(&allocNode, graph, NULL, NULL, 0, &allocParams);
void* dptr = allocParams.alloc.dptr;  // virtual address is set here

// Create kernel nodes using the allocation
cudaGraphNodeParams kParams = { cudaGraphNodeTypeKernel };
kParams.kernel.kernelParams[0] = allocParams.alloc.dptr;
// ... set other kernel params ...
cudaGraphAddNode(&a, graph, &allocNode, 1, NULL, &kParams);
cudaGraphAddNode(&b, graph, &a, 1, NULL, &kParams);
cudaGraphAddNode(&c, graph, &a, 1, NULL, &kParams);

// Create free node (must depend on all users of the allocation)
cudaGraphNode_t deps[] = { b, c };
cudaGraphNodeParams freeParams = { cudaGraphNodeTypeMemFree };
freeParams.free.dptr = allocParams.alloc.dptr;
cudaGraphAddNode(&freeNode, graph, deps, NULL, 2, &freeParams);
```

### 5.7.4 Creating Memory Nodes (Stream Capture)

```cpp
cudaStreamBeginCapture(stream);

cudaMallocAsync(&dptr, size, stream);
kernel_A<<<..., stream>>>(dptr, ...);

cudaEventRecord(event1, stream);
cudaStreamWaitEvent(stream2, event1);
kernel_B<<<..., stream1>>>(dptr, ...);
kernel_C<<<..., stream2>>>(dptr, ...);

cudaEventRecord(event2, stream2);
cudaStreamWaitEvent(stream1, event2);

cudaFreeAsync(dptr, stream1);
cudaStreamEndCapture(stream1, &graph);
```

### 5.7.5 Accessing Graph Memory Outside the Allocating Graph

Graph allocations can persist beyond the allocating graph's execution and be accessed by subsequent CUDA operations, as long as proper ordering is maintained.

**Single-stream ordering:**

```cpp
// Launch allocating graph
cudaGraphLaunch(allocGraphExec, stream);
// Use allocation in stream work (ordered after graph)
kernel<<<..., stream>>>(dptr, ...);
// Free in the same stream
cudaFreeAsync(dptr, stream);
```

**Event-based ordering between streams:**

```cpp
cudaGraphLaunch(allocGraphExec, allocStream);
cudaEventRecord(allocEvent, allocStream);
cudaStreamWaitEvent(stream2, allocEvent);
kernel<<<..., stream2>>>(dptr, ...);
cudaEventRecord(useDoneEvent, stream2);
cudaStreamWaitEvent(stream3, useDoneEvent);
cudaGraphLaunch(freeGraphExec, stream3);
```

### 5.7.6 cudaGraphInstantiateFlagAutoFreeOnLaunch

Normally, a graph with unfreed memory allocations cannot be relaunched. This flag allows relaunching by automatically inserting an asynchronous free of unfreed allocations at relaunch:

```cpp
cudaGraphInstantiateWithFlags(&graphExec, graph,
    cudaGraphInstantiateFlagAutoFreeOnLaunch);
```

This is useful for **single-producer, multiple-consumer** patterns where a producer graph creates allocations consumed by varying sets of consumers.

```cpp
// Producer creates allocations
cudaGraphInstantiateWithFlags(&producer, graph,
    cudaGraphInstantiateFlagAutoFreeOnLaunch);

// Consumers access allocations
cudaGraphInstantiateWithFlags(&consumer1, graph, 0);
cudaGraphInstantiateWithFlags(&consumer2, graph, 0);

do {
    cudaGraphLaunch(producer, myStream);
    cudaGraphLaunch(consumer1, myStream);
    if (launchConsumer2) {
        cudaGraphLaunch(consumer2, myStream);
    }
} while (determineAction(&launchConsumer2));

// Explicit cleanup
cudaFreeAsync(data1, myStream);
cudaFreeAsync(data2, myStream);
```

### 5.7.7 Physical Memory Management

#### Memory Reuse Within a Graph

CUDA may reuse the same virtual address ranges for allocations with non-overlapping GPU-ordered lifetimes.

#### Physical Memory Sharing Between Graphs

Different graphs launched into the same stream may share the same physical memory because they cannot execute concurrently. This is done through **virtual aliasing**.

#### cudaGraphUpload()

Separates the cost of physical memory allocation from graph launch:

```cpp
cudaGraphUpload(graphExec, stream);
// Later, launching into the SAME stream avoids remapping
cudaGraphLaunch(graphExec, stream);
```

#### cudaDeviceGraphMemTrim()

Explicitly releases unused physical memory reserved by graph memory nodes:

```cpp
cudaDeviceGraphMemTrim(device);
```

This unmaps and releases physical memory not actively in use. Graphs that are scheduled or running are not affected.

#### Querying Graph Memory Footprint

```cpp
size_t reserved, used;
cudaDeviceGetGraphMemAttribute(device, cudaGraphMemAttrReservedMemCurrent, &reserved);
cudaDeviceGetGraphMemAttribute(device, cudaGraphMemAttrUsedMemCurrent, &used);
```

- `cudaGraphMemAttrReservedMemCurrent` -- total physical memory reserved for graph allocations.
- `cudaGraphMemAttrUsedMemCurrent` -- physical memory currently mapped by at least one graph.

### 5.7.8 Peer Access

Graph allocations can be configured for access from multiple GPUs:

```cpp
cudaGraphNodeParams allocParams = { cudaGraphNodeTypeMemAlloc };
allocParams.alloc.poolProps.allocType = cudaMemAllocationTypePinned;
allocParams.alloc.poolProps.location.type = cudaMemLocationTypeDevice;
allocParams.alloc.poolProps.location.id = 1;  // resident on device 1

cudaMemAccessDesc accessDescs[2];
accessDescs[0].flags = cudaMemAccessFlagsProtReadWrite;
accessDescs[0].location.type = cudaMemLocationTypeDevice;
accessDescs[0].location.id = 0;
accessDescs[1].flags = cudaMemAccessFlagsProtReadWrite;
accessDescs[1].location.type = cudaMemLocationTypeDevice;
accessDescs[1].location.id = 2;

allocParams.accessDescCount = 2;
allocParams.accessDescs = accessDescs;
// Allocation is accessible from devices 0, 1 (resident), and 2
cudaGraphAddNode(&allocNode, graph, NULL, 0, &allocParams);
```

For stream capture, the allocation node records the peer accessibility of the allocating pool at the time of capture.

### 5.7.9 Memory Nodes in Child Graphs (CUDA 12.9)

Child graphs can contain memory allocation and free nodes when **moved** to a parent graph:

```cpp
// Create child graph with memory nodes
cudaGraphCreate(&child, 0);
// ... add alloc and free nodes to child ...

// Move child to parent
cudaGraphNodeParams childParams = { cudaGraphNodeTypeGraph };
childParams.graph.graph = child;
childParams.graph.ownership = cudaGraphChildGraphOwnershipMove;
cudaGraphAddNode(&parentNode, parent, NULL, 0, &childParams);
```

Restrictions on moved child graphs:
- Cannot be independently instantiated or destroyed.
- Cannot be added as a child of a separate parent.
- Cannot be used with `cudaGraphExecUpdate`.
- Cannot have additional memory nodes added.

---

## 5.8 Device Graph Launch (CC 9.0+)

Device graph launch enables launching graphs from GPU device code, eliminating the need for a CPU round-trip for data-dependent decisions.

### 5.8.1 Requirements

The graph must be instantiated with `cudaGraphInstantiateFlagDeviceLaunch`:

```cpp
cudaGraphExec_t deviceGraphExec;
cudaGraphInstantiate(&deviceGraphExec, deviceGraph,
                      cudaGraphInstantiateFlagDeviceLaunch);
cudaGraphUpload(deviceGraphExec, stream);
```

**Device graph node restrictions:**
- All nodes must reside on a single device.
- Only kernel, memcpy, memset, and child graph nodes are allowed.
- No CUDA Dynamic Parallelism (CDP) by kernels in the graph.
- Only copies involving device memory and/or pinned device-mapped host memory.
- No CUDA array copies.

### 5.8.2 Device-Side Launch

```cpp
__global__ void launchGraphKernel(cudaGraphExec_t graph) {
    cudaGraphLaunch(graph, cudaStreamGraphFireAndForget);
}
```

Device graphs must be launched from within another graph (not directly from a stream). Launch is **per-thread** -- the user must select a single thread for each launch.

### 5.8.3 Launch Modes

| Stream | Mode | Description |
|--------|------|-------------|
| `cudaStreamGraphFireAndForget` | Fire-and-forget | Launches immediately, runs independently. Up to **120** per parent graph execution. |
| `cudaStreamGraphTailLaunch` | Tail launch | Executes when the parent's environment is complete. Up to **255** pending. |
| `cudaStreamGraphFireAndForgetAsSibling` | Sibling launch | Launches as a child of the parent's environment (not the launching graph). |

### 5.8.4 Fire-and-Forget Launch

```cpp
__global__ void launchFireAndForget(cudaGraphExec_t graph) {
    cudaGraphLaunch(graph, cudaStreamGraphFireAndForget);
}

void setup() {
    cudaGraphExec_t gExec1, gExec2;
    // Create, instantiate, and upload device graph
    create_graph(&g2);
    cudaGraphInstantiate(&gExec2, g2, cudaGraphInstantiateFlagDeviceLaunch);
    cudaGraphUpload(gExec2, stream);

    // Create launching graph via stream capture
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    launchFireAndForget<<<1, 1, 0, stream>>>(gExec2);
    cudaStreamEndCapture(stream, &g1);
    cudaGraphInstantiate(&gExec1, g1);

    cudaGraphLaunch(gExec1, stream);
}
```

### 5.8.5 Tail Launch

Tail launches execute when a graph's environment is complete (graph + all children done). Tail launches are ordered:

```cpp
__global__ void launchTail(cudaGraphExec_t graph) {
    cudaGraphLaunch(graph, cudaStreamGraphTailLaunch);
}
```

### 5.8.6 Tail Self-Launch

A device graph can enqueue itself for a tail launch (only one self-launch at a time):

```cpp
__device__ int relaunchCount = 0;

__global__ void relaunchSelf() {
    int relaunchMax = 100;
    if (threadIdx.x == 0) {
        if (relaunchCount < relaunchMax) {
            cudaGraphLaunch(cudaGetCurrentGraphExec(),
                           cudaStreamGraphTailLaunch);
        }
        relaunchCount++;
    }
}
```

`cudaGetCurrentGraphExec()` returns the handle of the currently running device graph, or NULL if not in a device graph.

### 5.8.7 Execution Environments

Each device graph launch creates an **execution environment** that encapsulates all work in the graph plus all generated fire-and-forget children. Environments are hierarchical. A graph is complete when it and all its children have finished. Host launches create a **stream environment** that parents the graph environment.

---

## 5.9 User Objects

CUDA User Objects manage the lifetime of resources used by asynchronous work, particularly useful with CUDA graphs and stream capture.

### 5.9.1 Concept

A user object associates a destructor callback with an internal reference count, similar to `std::shared_ptr`. References can be owned by:

- **User code** on the CPU (tracked manually)
- **CUDA graphs** (managed automatically)

### 5.9.2 Creating and Using User Objects

```cpp
cudaGraph_t graph;  // existing graph
Object* object = new Object;

cudaUserObject_t cuObject;
cudaUserObjectCreate(
    &cuObject,
    object,
    [] (void* ptr) { delete static_cast<Object*>(ptr); },  // destructor
    1,                         // initial refcount
    cudaUserObjectNoDestructorSync  // destructor cannot be waited on via CUDA
);

// Transfer ownership to the graph
cudaGraphRetainUserObject(
    graph, cuObject,
    1,                             // number of references
    cudaGraphUserObjectMove        // transfer ownership (don't modify total count)
);
// No user-owned references remain; no release call needed

// Instantiate and launch
cudaGraphExec_t graphExec;
cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
cudaGraphDestroy(graph);  // graphExec still owns a reference

cudaGraphLaunch(graphExec, 0);
cudaGraphExecDestroy(graphExec);  // deferred if launch not synced
cudaStreamSynchronize(0);         // destructor runs after sync
```

### 5.9.3 Automatic Management

- A **cloned** `cudaGraph_t` retains a copy of every reference from the source.
- An **instantiated** `cudaGraphExec_t` retains a copy of every reference from the source `cudaGraph_t`.
- If a `cudaGraphExec_t` is destroyed without synchronization, references are held until execution completes.
- References in child graph nodes are associated with the child graphs, not the parents.

### 5.9.4 Restrictions

- There is no CUDA API to wait on user object destructors. Users may signal a synchronization object manually from the destructor.
- **It is illegal to call CUDA APIs from the destructor**, similar to the `cudaLaunchHostFunc` restriction.
- It is legal to signal another thread to perform an API call from the destructor, if the dependency is one-way and the thread cannot block forward progress of CUDA work.
