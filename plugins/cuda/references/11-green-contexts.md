# 11. Green Contexts

Green Contexts are a CUDA feature that enables lightweight GPU context partitioning, allowing applications to statically divide GPU resources (such as streaming multiprocessors and work queues) among concurrent tasks within a single process. Unlike MIG (Multi-Instance GPU) or MPS (Multi-Process Service), Green Contexts operate entirely within a single process and provide fine-grained control over resource allocation with minimal overhead.

---

## 11.1 Motivation

Traditional CUDA streams offer concurrency within a single GPU context, but all streams compete for the same pool of GPU resources (SMs, memory bandwidth, etc.). When latency-sensitive and throughput-oriented workloads share a GPU, the throughput workloads can starve the latency-sensitive tasks of SM resources, causing unpredictable performance.

Green Contexts solve this by providing:

- **Static SM partitioning:** Dedicate a specific subset of SMs to a context, ensuring latency-sensitive work always has guaranteed compute resources
- **Lightweight overhead:** Much lighter than creating separate CUDA contexts or using MIG partitions
- **Single-process operation:** No need for multi-process setups, IPC, or MIG configuration
- **Work queue isolation:** Separate work submission queues prevent head-of-line blocking between contexts

**Comparison with alternatives:**

| Feature | Green Contexts | MIG | MPS |
|---------|---------------|-----|-----|
| Scope | Single process | System-wide (requires MIG mode) | Multi-process |
| SM partitioning | Static, per-context | Static, per-instance | Shared |
| Overhead | Minimal | High (requires MIG setup) | Low-medium |
| Memory isolation | Shared address space | Separate memory | Shared address space |
| Configuration | API calls | nvidia-smi / driver | Daemon process |

---

## 11.2 Device Resources

Green Contexts are built on the concept of device resources -- hardware resources that can be queried, partitioned, and allocated to contexts.

### 11.2.1 Resource Types

```cpp
// Top-level resource descriptor
typedef struct {
    cudaDevSmResource sm;       // Streaming multiprocessor resources
    cudaDevWorkqueueResource workqueue; // Work queue resources
} cudaDevResource;

// SM resource descriptor
typedef struct {
    unsigned int smCount;              // Number of SMs
    unsigned int minSmPartitionSize;   // Minimum granularity for SM partitioning
    unsigned int smCoscheduledAlignment; // Alignment constraint for coscheduling
    unsigned int flags;                // Additional flags
} cudaDevSmResource;

// Work queue resource descriptor
typedef struct {
    int device;                        // Device index
    unsigned int wqConcurrencyLimit;   // Maximum concurrent work submissions
    cudaDevWorkqueueSharingScope sharingScope; // Sharing scope
} cudaDevWorkqueueResource;
```

### 11.2.2 Key SM Resource Fields

| Field | Description |
|-------|-------------|
| `smCount` | Number of SMs available or allocated |
| `minSmPartitionSize` | The minimum number of SMs that can be allocated to a partition. Partitions must be multiples of this size. |
| `smCoscheduledAlignment` | The alignment constraint for SM coscheduling on the hardware. When the `IGNORE_COSCHEDULING` flag is not set, partitions are rounded up to this alignment. |
| `flags` | Control flags, such as `cudaDevSmResourceFlagIgnoreCoscheduling` |

### 11.2.3 Querying Available Resources

```cpp
void queryDeviceResources(int device)
{
    cudaDevResource resource;

    // Query all available SM resources on the device
    cudaError_t err = cudaDeviceGetDevResource(device, &resource,
                                                 cudaDevResourceTypeSm);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get SM resources: %s\n",
                cudaGetErrorString(err));
        return;
    }

    printf("Total SMs:             %u\n", resource.sm.smCount);
    printf("Min SM partition size: %u\n", resource.sm.minSmPartitionSize);
    printf("SM cosched alignment:  %u\n", resource.sm.smCoscheduledAlignment);
    printf("Flags:                 0x%x\n", resource.sm.flags);

    // Query work queue resources
    err = cudaDeviceGetDevResource(device, &resource,
                                    cudaDevResourceTypeWorkqueue);
    if (err == cudaSuccess) {
        printf("Workqueue concurrency: %u\n", resource.workqueue.wqConcurrencyLimit);
    }
}
```

---

## 11.3 Creation Steps

Creating a Green Context involves four steps: querying resources, partitioning them, generating a descriptor, and creating the context.

### 11.3.1 Step 1: Get Available Resources

```cpp
int device = 0;
cudaDevResource resource;

// Get all SM resources on the device
cudaDeviceGetDevResource(device, &resource, cudaDevResourceTypeSm);
// resource.sm.smCount now contains the total number of SMs
// resource.sm.minSmPartitionSize contains the partitioning granularity
```

### 11.3.2 Step 2: Partition SM Resources

There are two ways to partition SM resources: homogeneous (equal partitions) and heterogeneous (custom sizes).

#### Homogeneous Partitioning by Count

```cpp
// Partition SMs into equal groups
int numGroups = 2;       // Number of partitions
int minCount = 0;        // Minimum SMs per partition (0 = divide equally)
unsigned int flags = 0;  // No special flags

cudaDevResourceDesc groups[numGroups];
cudaDevResource remaining;

cudaError_t err = cudaDevSmResourceSplitByCount(
    groups,           // Output: array of resource descriptors
    &remaining,       // Output: remaining unallocated resources
    &resource.sm,     // Input: total SM resources to partition
    NULL,             // Optional: array of per-group flags
    numGroups,        // Number of groups to create
    minCount,         // Minimum SMs per group
    flags             // Global flags
);

if (err != cudaSuccess) {
    fprintf(stderr, "SM partition failed: %s\n", cudaGetErrorString(err));
}
```

#### Heterogeneous Partitioning

```cpp
// Partition SMs into groups with custom sizes
int numGroups = 3;

// Define custom parameters for each group
cudaDevResourceSplitGroupParams groupParams[numGroups];
groupParams[0].smCount = 40;  // Group 0: 40 SMs
groupParams[0].flags = 0;
groupParams[1].smCount = 40;  // Group 1: 40 SMs
groupParams[1].flags = 0;
groupParams[2].smCount = 52;  // Group 2: 52 SMs
groupParams[2].flags = 0;

cudaDevResourceDesc groups[numGroups];
cudaDevResource remaining;

cudaError_t err = cudaDevResourceSplit(
    groups,           // Output: array of resource descriptors
    numGroups,        // Number of groups
    resource,         // Input: total device resources
    groupParams       // Per-group parameters
);
```

### 11.3.3 Step 3: Create Descriptor

```cpp
cudaDevResourceGenerateDescOptions options = {};
options.smCount = numGroups; // Number of SM resource groups

cudaGreenCtxDesc desc;
cudaDevResourceGenerateDesc(&desc, groups, numGroups);
```

### 11.3.4 Step 4: Create Green Context

```cpp
cudaGreenCtx greenCtx;
unsigned int flags = 0;  // No special flags

cudaError_t err = cudaGreenCtxCreate(&greenCtx, desc, device, flags);
if (err != cudaSuccess) {
    fprintf(stderr, "Failed to create green context: %s\n",
            cudaGetErrorString(err));
    return;
}

printf("Green context created successfully\n");
```

### 11.3.5 Complete Creation Example

```cpp
#include <cuda_runtime.h>
#include <stdio.h>

int createGreenContexts(int device, cudaGreenCtx* contexts, int num_contexts)
{
    // Step 1: Query available SM resources
    cudaDevResource resource;
    cudaError_t err = cudaDeviceGetDevResource(device, &resource,
                                                 cudaDevResourceTypeSm);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device resources: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    printf("Device %d: %u SMs available (min partition: %u, cosched align: %u)\n",
           device, resource.sm.smCount,
           resource.sm.minSmPartitionSize,
           resource.sm.smCoscheduledAlignment);

    // Step 2: Partition into equal groups
    cudaDevResourceDesc groups[num_contexts];
    cudaDevResource remaining;

    err = cudaDevSmResourceSplitByCount(groups, &remaining, &resource.sm,
                                         NULL, num_contexts, 0, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to partition SMs: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    for (int i = 0; i < num_contexts; ++i) {
        printf("  Group %d: allocated\n", i);
    }

    // Step 3: Generate descriptor
    cudaGreenCtxDesc desc;
    cudaDevResourceGenerateDesc(&desc, groups, num_contexts);

    // Step 4: Create green contexts
    for (int i = 0; i < num_contexts; ++i) {
        err = cudaGreenCtxCreate(&contexts[i], desc, device, 0);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to create green context %d: %s\n",
                    i, cudaGetErrorString(err));
            return -1;
        }
        printf("  Green context %d created\n", i);
    }

    return 0;
}
```

### 11.3.6 Partition Behavior Example (CC 9.0, 132 SMs GH200)

The partitioning behavior depends on the `minCount` parameter and the `flags` (particularly `IGNORE_COSCHEDULING`). The following examples illustrate how partitioning works on a GH200 (CC 9.0) with 132 SMs:

**Example 1: Two equal groups with large minimum**

| Parameter | Value |
|-----------|-------|
| Requested groups | 2 |
| `minCount` | 72 |
| Flags | 0 (respect coscheduling) |

| Result | Value |
|--------|-------|
| Group 0 | 72 SMs |
| Group 1 | (not created -- only 60 SMs remain, below minCount of 72) |
| Remaining | 60 SMs |

With `minCount=72` and 2 groups requested, the first group gets 72 SMs. Only 60 SMs remain (132 - 72 = 60), which is less than the requested minimum of 72 for the second group, so only one group is created.

**Example 2: Six small groups**

| Parameter | Value |
|-----------|-------|
| Requested groups | 6 |
| `minCount` | 11 |
| Flags | 0 (respect coscheduling) |

| Result | Value |
|--------|-------|
| Each group | 16 SMs (rounded up from 132/6=22 to coscheduling alignment) |
| Actually | 6 groups of 16 SMs (due to coscheduling alignment of 16) = 96 SMs used |
| Remaining | 36 SMs |

Wait -- let me recalculate. With coscheduling alignment of 16, each group gets `ceil(132/6/16)*16 = ceil(1.375)*16 = 2*16 = 32`? No -- the actual behavior is:

The total SMs are divided among groups, and each group is rounded up to the coscheduling alignment. With 132 SMs and 6 groups: 132/6 = 22 SMs per group. With coscheduling alignment of 16, each group gets 32 SMs (next multiple of 16 above 22). But 6 * 32 = 192 > 132, so this would fail.

The actual behavior on GH200 (coscheduling alignment = 16, min partition = 8):

| Requested | `minCount` | Flags | Result | Remaining |
|-----------|------------|-------|--------|-----------|
| 2 groups | 72 | 0 | 1 group of 72 SMs | 60 SMs |
| 6 groups | 11 | 0 | 6 groups of 16 SMs each | 36 SMs |
| 6 groups | 11 | `IGNORE_COSCHEDULING` | 6 groups of 12 SMs each | 60 SMs |

The difference between the second and third rows demonstrates the effect of `IGNORE_COSCHEDULING`: without it, groups are rounded up to the coscheduling alignment (16), resulting in larger partitions. With it, groups are sized closer to the minimum requested count.

---

## 11.4 Launching Work on Green Contexts

Once a Green Context is created, work is launched on it by creating a stream associated with the context and then launching kernels on that stream.

### 11.4.1 Creating Streams on Green Contexts

```cpp
cudaGreenCtx greenCtx;  // Created via cudaGreenCtxCreate
cudaStream_t stream;

// Create a stream associated with the green context
cudaError_t err = cudaExecutionCtxStreamCreate(&stream, greenCtx, 0, 0);
if (err != cudaSuccess) {
    fprintf(stderr, "Failed to create stream on green context: %s\n",
            cudaGetErrorString(err));
    return;
}
```

**API signature:**

```cpp
cudaError_t cudaExecutionCtxStreamCreate(
    cudaStream_t* pStream,      // Output: newly created stream
    cudaGreenCtx greenCtx,       // Green context to associate with
    unsigned int flags,          // Stream creation flags (typically 0)
    int priority                 // Stream priority (0 = default)
);
```

### 11.4.2 Launching Kernels

Kernels launched on a green context stream execute only on the SMs allocated to that context:

```cpp
__global__ void myKernel(float* data, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = data[idx] * 2.0f;
    }
}

void launchOnGreenContext(cudaGreenCtx greenCtx, float* d_data, int N)
{
    cudaStream_t stream;
    cudaExecutionCtxStreamCreate(&stream, greenCtx, 0, 0);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // This kernel runs only on the SMs allocated to greenCtx
    myKernel<<<gridSize, blockSize, 0, stream>>>(d_data, N);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}
```

### 11.4.3 Multiple Green Contexts Concurrently

```cpp
void runConcurrentWorkloads(int device)
{
    const int NUM_CONTEXTS = 2;

    // Step 1: Query and partition resources
    cudaDevResource resource;
    cudaDeviceGetDevResource(device, &resource, cudaDevResourceTypeSm);

    cudaDevResourceDesc groups[NUM_CONTEXTS];
    cudaDevResource remaining;
    cudaDevSmResourceSplitByCount(groups, &remaining, &resource.sm,
                                   NULL, NUM_CONTEXTS, 0, 0);

    // Step 2: Create descriptors and green contexts
    cudaGreenCtxDesc desc;
    cudaDevResourceGenerateDesc(&desc, groups, NUM_CONTEXTS);

    cudaGreenCtx contexts[NUM_CONTEXTS];
    for (int i = 0; i < NUM_CONTEXTS; ++i) {
        cudaGreenCtxCreate(&contexts[i], desc, device, 0);
    }

    // Step 3: Create streams and launch work on each context
    cudaStream_t streams[NUM_CONTEXTS];

    for (int i = 0; i < NUM_CONTEXTS; ++i) {
        cudaExecutionCtxStreamCreate(&streams[i], contexts[i], 0, 0);
    }

    // Allocate data
    float *d_data[NUM_CONTEXTS];
    int N = 1024 * 1024;
    for (int i = 0; i < NUM_CONTEXTS; ++i) {
        cudaMalloc(&d_data[i], N * sizeof(float));
    }

    // Launch kernels concurrently on different green contexts
    for (int i = 0; i < NUM_CONTEXTS; ++i) {
        myKernel<<<(N + 255) / 256, 256, 0, streams[i]>>>(d_data[i], N);
    }

    // Wait for all contexts to complete
    for (int i = 0; i < NUM_CONTEXTS; ++i) {
        cudaStreamSynchronize(streams[i]);
    }

    // Cleanup
    for (int i = 0; i < NUM_CONTEXTS; ++i) {
        cudaStreamDestroy(streams[i]);
        cudaExecutionCtxDestroy(contexts[i]);
        cudaFree(d_data[i]);
    }
}
```

### 11.4.4 CUDA Graphs with Green Contexts

CUDA Graphs can be created and launched on Green Contexts. The green context is associated with a graph node at the time the node is created (during stream capture or explicit node creation).

```cpp
void createGraphOnGreenContext(cudaGreenCtx greenCtx)
{
    cudaStream_t stream;
    cudaExecutionCtxStreamCreate(&stream, greenCtx, 0, 0);

    // Begin stream capture on the green context's stream
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    // Launch kernels -- they will be captured into the graph
    // The green context is recorded during node creation
    myKernel<<<10, 256, 0, stream>>>(d_data, N);

    // End capture and instantiate the graph
    cudaGraph_t graph;
    cudaStreamEndCapture(stream, &graph);

    cudaGraphExec_t graphExec;
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

    // Launch the graph on the green context's stream
    cudaGraphLaunch(graphExec, stream);
    cudaStreamSynchronize(stream);

    // Cleanup
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
}
```

**Important:** The green context association is determined when graph nodes are created. Nodes created on a green context's stream will execute on that context's SMs when the graph is launched.

### 11.4.5 Thread Block Clusters with Green Contexts

When using Thread Block Clusters on a Green Context, use the green context's stream for occupancy queries to get accurate occupancy information for the allocated SMs:

```cpp
void clusterLaunchOnGreenContext(cudaGreenCtx greenCtx, float* d_data, int N)
{
    cudaStream_t stream;
    cudaExecutionCtxStreamCreate(&stream, greenCtx, 0, 0);

    // Query occupancy using the green context's stream
    int blockSize = 256;
    int numBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, myKernel,
                                                    blockSize, 0);

    // Note: occupancy is limited to the SMs in the green context
    // Use the green context's stream for cluster launch
    int cluster_size = 8; // 8 CTAs per cluster

    void* kernel_args[] = { &d_data, &N };
    dim3 gridDim(64, 1, 1);
    dim3 blockDim(256, 1, 1);

    // Launch with cluster dimensions using the green context stream
    cudaLaunchCooperativeKernel(
        myKernel, gridDim, blockDim, kernel_args,
        0, // shared memory size
        stream // green context stream
    );

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}
```

---

## 11.5 Additional APIs

### 11.5.1 Event Recording and Waiting

Green Contexts provide their own event recording and waiting APIs, analogous to `cudaEventRecord` and `cudaStreamWaitEvent`:

```cpp
// Record an event on a green context
// This records the current state of the green context into the event
cudaError_t cudaExecutionCtxRecordEvent(
    cudaGreenCtx greenCtx,
    cudaEvent_t event
);

// Make a green context wait for a previously recorded event
// The green context will not execute any further work until the event completes
cudaError_t cudaExecutionCtxWaitEvent(
    cudaGreenCtx greenCtx,
    cudaEvent_t event
);
```

**Example: Synchronizing between green contexts:**

```cpp
void synchronizeBetweenContexts(cudaGreenCtx ctx_a, cudaGreenCtx ctx_b,
                                float* d_data_a, float* d_data_b, int N)
{
    cudaStream_t stream_a, stream_b;
    cudaExecutionCtxStreamCreate(&stream_a, ctx_a, 0, 0);
    cudaExecutionCtxStreamCreate(&stream_b, ctx_b, 0, 0);

    // Create an event
    cudaEvent_t event;
    cudaEventCreate(&event);

    // Context A produces data
    producerKernel<<<(N + 255) / 256, 256, 0, stream_a>>>(d_data_a, N);

    // Record event after context A's work
    cudaExecutionCtxRecordEvent(ctx_a, event);

    // Context B waits for context A's event before consuming data
    cudaExecutionCtxWaitEvent(ctx_b, event);

    // Context B consumes data
    consumerKernel<<<(N + 255) / 256, 256, 0, stream_b>>>(d_data_a, d_data_b, N);

    cudaStreamSynchronize(stream_b);

    // Cleanup
    cudaEventDestroy(event);
    cudaStreamDestroy(stream_a);
    cudaStreamDestroy(stream_b);
}
```

### 11.5.2 Synchronization

```cpp
// Synchronize all work submitted to a green context
// Blocks the host until all previously submitted work completes
cudaError_t cudaExecutionCtxSynchronize(cudaGreenCtx greenCtx);
```

**Example:**

```cpp
// Launch work on the green context
for (int i = 0; i < num_iterations; ++i) {
    myKernel<<<grid, block, 0, stream>>>(d_data, N);
}

// Wait for all work on the green context to complete
cudaExecutionCtxSynchronize(greenCtx);
printf("All work on green context completed\n");
```

### 11.5.3 Querying Device

```cpp
// Get the device associated with a green context
cudaError_t cudaExecutionCtxGetDevice(
    cudaGreenCtx greenCtx,
    int* device   // Output: device index
);
```

**Example:**

```cpp
int device;
cudaExecutionCtxGetDevice(greenCtx, &device);
printf("Green context is on device %d\n", device);
```

### 11.5.4 Destroying a Green Context

```cpp
// Destroy a green context and release its resources
cudaError_t cudaExecutionCtxDestroy(cudaGreenCtx greenCtx);
```

**Important:** All streams associated with the green context must be synchronized or destroyed before calling `cudaExecutionCtxDestroy`. Failure to do so results in undefined behavior.

```cpp
void cleanupGreenContext(cudaGreenCtx greenCtx, cudaStream_t stream)
{
    // Synchronize and destroy the stream first
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    // Now destroy the green context
    cudaExecutionCtxDestroy(greenCtx);
}
```

### 11.5.5 Full API Reference

| API | Purpose |
|-----|---------|
| `cudaDeviceGetDevResource` | Query available device resources |
| `cudaDevSmResourceSplitByCount` | Partition SMs into equal groups |
| `cudaDevResourceSplit` | Partition SMs into heterogeneous groups |
| `cudaDevResourceGenerateDesc` | Generate a green context descriptor from resource groups |
| `cudaGreenCtxCreate` | Create a green context from a descriptor |
| `cudaExecutionCtxStreamCreate` | Create a stream on a green context |
| `cudaExecutionCtxRecordEvent` | Record an event on a green context |
| `cudaExecutionCtxWaitEvent` | Make a green context wait for an event |
| `cudaExecutionCtxSynchronize` | Synchronize all work on a green context |
| `cudaExecutionCtxGetDevice` | Query the device of a green context |
| `cudaExecutionCtxDestroy` | Destroy a green context |

### 11.5.6 Complete End-to-End Example

```cpp
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void latencyKernel(float* data, int N, int iterations)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = data[idx];
        for (int i = 0; i < iterations; ++i) {
            val = val * 1.0001f + 0.001f;
        }
        data[idx] = val;
    }
}

__global__ void throughputKernel(float* data, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        for (int i = 0; i < 1000; ++i) {
            data[idx] = data[idx] * 1.001f + 0.01f;
        }
    }
}

int main()
{
    int device = 0;
    cudaSetDevice(device);

    // Query device resources
    cudaDevResource resource;
    cudaDeviceGetDevResource(device, &resource, cudaDevResourceTypeSm);
    printf("Total SMs: %u\n", resource.sm.smCount);

    // Partition into two groups: one for latency, one for throughput
    const int NUM_GROUPS = 2;
    cudaDevResourceDesc groups[NUM_GROUPS];
    cudaDevResource remaining;

    // Create heterogeneous split: 40% for latency, rest for throughput
    cudaDevResourceSplitGroupParams groupParams[NUM_GROUPS];
    unsigned int latency_sms = (resource.sm.smCount * 40) / 100;
    // Round up to minSmPartitionSize
    unsigned int min_part = resource.sm.minSmPartitionSize;
    latency_sms = ((latency_sms + min_part - 1) / min_part) * min_part;
    unsigned int throughput_sms = resource.sm.smCount - latency_sms;

    groupParams[0].smCount = latency_sms;
    groupParams[0].flags = 0;
    groupParams[1].smCount = throughput_sms;
    groupParams[1].flags = 0;

    cudaDevResourceSplit(groups, NUM_GROUPS, resource, groupParams);

    // Generate descriptor and create green contexts
    cudaGreenCtxDesc desc;
    cudaDevResourceGenerateDesc(&desc, groups, NUM_GROUPS);

    cudaGreenCtx latencyCtx, throughputCtx;
    cudaGreenCtxCreate(&latencyCtx, desc, device, 0);
    cudaGreenCtxCreate(&throughputCtx, desc, device, 0);

    printf("Created green contexts:\n");
    printf("  Latency context:    %u SMs\n", latency_sms);
    printf("  Throughput context: %u SMs\n", throughput_sms);

    // Create streams
    cudaStream_t latencyStream, throughputStream;
    cudaExecutionCtxStreamCreate(&latencyStream, latencyCtx, 0, 0);
    cudaExecutionCtxStreamCreate(&throughputStream, throughputCtx, 0, 0);

    // Allocate data
    int N_latency = 256 * 1024;
    int N_throughput = 4 * 1024 * 1024;
    float *d_latency_data, *d_throughput_data;
    cudaMalloc(&d_latency_data, N_latency * sizeof(float));
    cudaMalloc(&d_throughput_data, N_throughput * sizeof(float));

    // Initialize
    cudaMemset(d_latency_data, 1, N_latency * sizeof(float));
    cudaMemset(d_throughput_data, 2, N_throughput * sizeof(float));

    // Launch latency-sensitive work on the latency context
    latencyKernel<<<(N_latency + 255) / 256, 256, 0, latencyStream>>>(
        d_latency_data, N_latency, 100);

    // Launch throughput work on the throughput context
    throughputKernel<<<(N_throughput + 255) / 256, 256, 0, throughputStream>>>(
        d_throughput_data, N_throughput);

    // Both kernels run concurrently on their respective SMs
    // The latency kernel is guaranteed SM resources and is not
    // affected by the throughput kernel's resource usage

    // Synchronize
    cudaExecutionCtxSynchronize(latencyCtx);
    cudaExecutionCtxSynchronize(throughputCtx);

    printf("Both workloads completed\n");

    // Cleanup
    cudaStreamDestroy(latencyStream);
    cudaStreamDestroy(throughputStream);
    cudaExecutionCtxDestroy(latencyCtx);
    cudaExecutionCtxDestroy(throughputCtx);
    cudaFree(d_latency_data);
    cudaFree(d_throughput_data);

    return 0;
}
```
