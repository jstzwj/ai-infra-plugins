# 10. L2 Cache Control

The L2 cache is a shared resource on NVIDIA GPUs that sits between the streaming multiprocessors (SMs) and global memory. Proper management of the L2 cache can significantly improve performance for data-intensive workloads, particularly those with predictable data access patterns. CUDA provides APIs to control how data is cached in L2, including the ability to persistently retain specific data and to tune cache hit policies.

This section covers the L2 cache management APIs available in CUDA, including cache set-aside, access policy windows, access properties, hit ratio tuning, and reset/query operations.

---

## 10.1 L2 Cache Set-Aside

CUDA allows applications to reserve a portion of the L2 cache for persistent data. Persistently cached data is preferentially retained in L2 across kernel launches, reducing the need to re-fetch frequently accessed data from global memory.

### 10.1.1 Setting the Persisting L2 Cache Size

```cpp
cudaDeviceProp prop;
int device;

cudaGetDevice(&device);
cudaGetDeviceProperties(&prop, device);

// Set aside the maximum available persisting L2 cache size
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, prop.persistingL2CacheMaxSize);
```

The `cudaLimitPersistingL2CacheSize` limit controls how much of the L2 cache is reserved for persistent caching. The maximum available size is reported by `cudaDeviceProp::persistingL2CacheMaxSize`.

**Key points:**

- The set-aside size must be set before launching kernels that rely on persistent caching
- Setting the size to 0 disables persistent caching
- The actual amount of L2 cache available for persistence depends on the GPU model and current utilization

### 10.1.2 Constraints

**MIG (Multi-Instance GPU) mode:** L2 cache persistence is disabled when the GPU is partitioned using MIG. Each MIG instance has its own L2 cache partition, and the persistence APIs have no effect.

**MPS (Multi-Process Service):** Under MPS, the L2 cache persistence behavior can be controlled via the environment variable:

```bash
CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT=<percentage>
```

This environment variable sets the default percentage of L2 cache that can be set aside for persistence across all MPS clients. When using MPS, coordinate L2 cache usage across clients to avoid contention.

### 10.1.3 Example: Enabling L2 Cache Persistence

```cpp
void enableL2Persistence(int device)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    if (prop.persistingL2CacheMaxSize > 0) {
        // Reserve 50% of the maximum persisting L2 cache size
        size_t persist_size = prop.persistingL2CacheMaxSize / 2;
        cudaError_t err = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,
                                              persist_size);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to set L2 persisting cache size: %s\n",
                    cudaGetErrorString(err));
        }
    }
}
```

---

## 10.2 Access Policy Window

The access policy window defines a region of global memory and specifies how accesses to that region should be treated by the L2 cache. It is associated with a CUDA stream and applies to all kernels launched on that stream.

### 10.2.1 Setting an Access Policy Window

```cpp
cudaStreamAttrValue stream_attr;

// Define the memory region
stream_attr.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(device_ptr);
stream_attr.accessPolicyWindow.num_bytes = num_bytes;

// Set the cache hit and miss policies
stream_attr.accessPolicyWindow.hitRatio = 1.0;  // 100% of accesses use hitProp
stream_attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;  // Retain in L2
stream_attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;  // Evict from L2

// Apply the access policy to the stream
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attr);
```

### 10.2.2 Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `base_ptr` | `void*` | Base address of the memory region. This can be any address within the desired region; CUDA uses it as a hint for the memory range. |
| `num_bytes` | `size_t` | Number of bytes in the memory region starting from `base_ptr`. |
| `hitRatio` | `float` | Fraction (0.0 to 1.0) of accesses within the window that receive the `hitProp` policy. The remaining fraction receives `missProp`. |
| `hitProp` | `cudaAccessProperty` | Cache policy applied to hits (accesses that land in the policy window). |
| `missProp` | `cudaAccessProperty` | Cache policy applied to misses (accesses outside the policy window or the fraction not receiving `hitProp`). |

### 10.2.3 Complete Example

```cpp
#include <cuda_runtime.h>

void setupL2CachePolicy(float* d_data, size_t data_size, cudaStream_t stream)
{
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    // Step 1: Set aside a portion of L2 cache for persistence
    size_t persist_size = min(data_size, prop.persistingL2CacheMaxSize);
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, persist_size);

    // Step 2: Define the access policy window
    cudaStreamAttrValue stream_attr;
    stream_attr.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(d_data);
    stream_attr.accessPolicyWindow.num_bytes = data_size;
    stream_attr.accessPolicyWindow.hitRatio = 1.0;
    stream_attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    stream_attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;

    // Step 3: Apply to the stream
    cudaError_t err = cudaStreamSetAttribute(stream,
                                              cudaStreamAttributeAccessPolicyWindow,
                                              &stream_attr);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to set stream attribute: %s\n",
                cudaGetErrorString(err));
    }
}
```

### 10.2.4 Multiple Access Policy Windows

Each stream can have at most one access policy window active at a time. However, different streams can have different policies. To protect multiple memory regions, use the `hitRatio` parameter to distribute persistence across regions, or use separate streams for different regions.

```cpp
// Stream 1 protects region A
cudaStreamAttrValue attr_a;
attr_a.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(d_region_a);
attr_a.accessPolicyWindow.num_bytes = size_a;
attr_a.accessPolicyWindow.hitRatio = 1.0;
attr_a.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
attr_a.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
cudaStreamSetAttribute(stream_a, cudaStreamAttributeAccessPolicyWindow, &attr_a);

// Stream 2 protects region B
cudaStreamAttrValue attr_b;
attr_b.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(d_region_b);
attr_b.accessPolicyWindow.num_bytes = size_b;
attr_b.accessPolicyWindow.hitRatio = 1.0;
attr_b.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
attr_b.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
cudaStreamSetAttribute(stream_b, cudaStreamAttributeAccessPolicyWindow, &attr_b);
```

---

## 10.3 Access Properties

CUDA defines three access properties that control how the L2 cache handles memory accesses. These properties determine whether cached data is preferentially retained, evicted, or treated normally.

### 10.3.1 cudaAccessPropertyStreaming

```cpp
cudaAccessPropertyStreaming
```

- **Behavior:** Data accessed with this property is preferentially evicted from the L2 cache
- **Use case:** Data that is accessed once or infrequently (e.g., input data that is only read once, intermediate results that will not be reused)
- **Effect:** Marks cache lines as "streaming," making them the first candidates for eviction when L2 cache space is needed

**When to use:**

```cpp
// Example: Marking output buffer as streaming (written once, not reused)
cudaStreamAttrValue stream_attr;
stream_attr.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(d_output);
stream_attr.accessPolicyWindow.num_bytes = output_size;
stream_attr.accessPolicyWindow.hitRatio = 1.0;
stream_attr.accessPolicyWindow.hitProp = cudaAccessPropertyStreaming;
stream_attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attr);
```

### 10.3.2 cudaAccessPropertyPersisting

```cpp
cudaAccessPropertyPersisting
```

- **Behavior:** Data accessed with this property is preferentially retained in the L2 cache across kernel launches
- **Use case:** Data that is accessed repeatedly across multiple kernel launches (e.g., weight matrices in deep learning, lookup tables, frequently accessed data structures)
- **Effect:** Marks cache lines as "persistent," protecting them from eviction. Only evicted when the L2 cache is under severe pressure and no streaming lines are available.

**When to use:**

```cpp
// Example: Keeping weight data persistent across kernel launches
cudaStreamAttrValue stream_attr;
stream_attr.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(d_weights);
stream_attr.accessPolicyWindow.num_bytes = weights_size;
stream_attr.accessPolicyWindow.hitRatio = 1.0;
stream_attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
stream_attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attr);
```

### 10.3.3 cudaAccessPropertyNormal

```cpp
cudaAccessPropertyNormal
```

- **Behavior:** Resets the persisting status of cache lines, returning them to normal caching behavior
- **Use case:** When persistent data is no longer needed and its cache lines should be returned to the normal eviction pool
- **Effect:** Removes the "persistent" or "streaming" designation from cache lines, allowing them to be managed by the default L2 replacement policy

**When to use:**

```cpp
// Example: After finishing with persistent data, reset its cache status
cudaStreamAttrValue stream_attr;
stream_attr.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(d_weights);
stream_attr.accessPolicyWindow.num_bytes = weights_size;
stream_attr.accessPolicyWindow.hitRatio = 1.0;
stream_attr.accessPolicyWindow.hitProp = cudaAccessPropertyNormal;
stream_attr.accessPolicyWindow.missProp = cudaAccessPropertyNormal;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attr);
```

### 10.3.4 Property Summary

| Property | Eviction Priority | Persistence | Typical Use |
|----------|-------------------|-------------|-------------|
| `cudaAccessPropertyStreaming` | High (evicted first) | None | One-time data access |
| `cudaAccessPropertyPersisting` | Low (evicted last) | Across kernel launches | Frequently reused data |
| `cudaAccessPropertyNormal` | Default | None | Reset to default behavior |

---

## 10.4 Hit Ratio Tuning

The `hitRatio` parameter in the access policy window controls what fraction of memory accesses within the policy window receive the `hitProp` policy versus the `missProp` policy. This is a critical tuning knob for optimizing L2 cache utilization.

### 10.4.1 How Hit Ratio Works

```
hitRatio = 0.0  -->  0% of accesses get hitProp, 100% get missProp
hitRatio = 0.5  -->  50% of accesses get hitProp, 50% get missProp
hitRatio = 1.0  -->  100% of accesses get hitProp, 0% get missProp
```

The selection of which accesses receive the `hitProp` policy is determined by the hardware using a hash of the access address. This means the distribution is approximately uniform across the memory region.

### 10.4.2 When Persistent Data Fits in L2

When the persistent data is smaller than the available persisting L2 cache, setting `hitRatio = 1.0` maximizes the caching benefit. In this regime, all accesses to the policy window are cached persistently, and performance can increase by up to 50% compared to no L2 persistence.

```cpp
// Optimal: persistent data fits entirely in L2
// Hit ratio = 1.0 (all accesses are cached persistently)
size_t persistent_data_size = 4 * 1024 * 1024; // 4 MB
size_t l2_available = prop.persistingL2CacheMaxSize;

if (persistent_data_size <= l2_available) {
    stream_attr.accessPolicyWindow.hitRatio = 1.0;
    stream_attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    // ... up to 50% performance increase
}
```

### 10.4.3 When Persistent Data Exceeds L2

When the persistent data is larger than the available persisting L2 cache, setting `hitRatio = 1.0` causes thrashing -- cache lines are persistently loaded but immediately evicted to make room for new persistent lines. This can lead to a performance drop of approximately 10% compared to no L2 persistence.

The fix is to reduce `hitRatio` so that only a subset of the data receives the persisting policy, ensuring that subset fits in the available L2 cache.

```cpp
// Persistent data is larger than available L2
// Reduce hitRatio to prevent thrashing
size_t persistent_data_size = 32 * 1024 * 1024; // 32 MB
size_t l2_available = 4 * 1024 * 1024;          // 4 MB available

// Calculate the fraction that fits in L2
float ratio = static_cast<float>(l2_available) / static_cast<float>(persistent_data_size);

stream_attr.accessPolicyWindow.num_bytes = persistent_data_size;
stream_attr.accessPolicyWindow.hitRatio = ratio;  // ~0.125 in this example
stream_attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
stream_attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
```

### 10.4.4 Tuning Strategy

1. **Fix `num_bytes`** to the size of the data region you want to protect
2. **Start with `hitRatio = 1.0`** if the data fits in L2
3. **If data exceeds L2**, calculate `hitRatio = l2_available / num_bytes`
4. **Fine-tune experimentally** by adjusting `hitRatio` up or down and measuring performance
5. **Monitor** using CUDA profiling tools (NVIDIA Nsight Compute, Nsight Systems)

```cpp
float computeOptimalHitRatio(size_t data_size, size_t l2_persist_size)
{
    if (data_size <= l2_persist_size) {
        return 1.0f;
    }
    return static_cast<float>(l2_persist_size) / static_cast<float>(data_size);
}

void applyL2Policy(float* d_data, size_t data_size, cudaStream_t stream)
{
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    // Set aside L2 cache for persistence
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, prop.persistingL2CacheMaxSize);

    float hit_ratio = computeOptimalHitRatio(data_size, prop.persistingL2CacheMaxSize);

    cudaStreamAttrValue stream_attr;
    stream_attr.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(d_data);
    stream_attr.accessPolicyWindow.num_bytes = data_size;
    stream_attr.accessPolicyWindow.hitRatio = hit_ratio;
    stream_attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    stream_attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;

    cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attr);
}
```

### 10.4.5 Performance Impact Summary

| Scenario | Hit Ratio | Performance Impact |
|----------|-----------|-------------------|
| Data fits in L2, `hitRatio = 1.0` | 1.0 | Up to +50% improvement |
| Data exceeds L2, `hitRatio = 1.0` | 1.0 | ~10% degradation (thrashing) |
| Data exceeds L2, tuned `hitRatio` | `l2_size / data_size` | Improvement restored |

---

## 10.5 Reset and Query

### 10.5.1 Resetting the Persisting L2 Cache

When an application no longer needs persistent data in the L2 cache, it should reset the cache to allow other data to use the persisting portion. This is especially important in multi-kernel or multi-phase applications.

```cpp
// Reset all persisting L2 cache lines to normal status
cudaCtxResetPersistingL2Cache();
```

This function flushes the persisting portion of the L2 cache and returns all persisting cache lines to normal (non-persistent) status. Subsequent kernels will not benefit from previously persisted data.

**When to call:**

- At the end of a computation phase where persistent data was used
- Before switching to a different data access pattern
- When the persistent data is no longer needed by subsequent kernels

### 10.5.2 Querying L2 Cache Properties

The `cudaDeviceProp` structure provides several fields related to L2 cache management:

```cpp
cudaDeviceProp prop;
int device;

cudaGetDevice(&device);
cudaGetDeviceProperties(&prop, device);

// Total L2 cache size in bytes
size_t l2_total = prop.l2CacheSize;

// Maximum size that can be set aside for persistent caching
size_t l2_persist_max = prop.persistingL2CacheMaxSize;

// Maximum window size for the access policy
size_t policy_max_window = prop.accessPolicyMaxWindowSize;

printf("L2 Cache Size:                 %zu bytes (%.1f MB)\n",
       l2_total, l2_total / (1024.0 * 1024.0));
printf("Max Persisting L2 Cache Size:  %zu bytes (%.1f MB)\n",
       l2_persist_max, l2_persist_max / (1024.0 * 1024.0));
printf("Max Access Policy Window Size: %zu bytes (%.1f MB)\n",
       policy_max_window, policy_max_window / (1024.0 * 1024.0));
```

### 10.5.3 Complete Lifecycle Example

The following example demonstrates the complete lifecycle of L2 cache persistence: setup, use, and cleanup.

```cpp
#include <cuda_runtime.h>
#include <stdio.h>

// Phase 1: Setup L2 cache persistence for weight data
void setupPersistence(float* d_weights, size_t weights_size,
                      cudaStream_t stream)
{
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    // Reserve L2 cache for persistence
    size_t persist_size = prop.persistingL2CacheMaxSize;
    if (persist_size > 0) {
        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, persist_size);
    }

    // Set the access policy for the weights region
    float hit_ratio = 1.0f;
    if (weights_size > persist_size) {
        hit_ratio = static_cast<float>(persist_size) /
                    static_cast<float>(weights_size);
    }

    cudaStreamAttrValue stream_attr;
    stream_attr.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(d_weights);
    stream_attr.accessPolicyWindow.num_bytes = weights_size;
    stream_attr.accessPolicyWindow.hitRatio = hit_ratio;
    stream_attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    stream_attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow,
                           &stream_attr);

    printf("L2 persistence configured: %zu bytes, hitRatio=%.3f\n",
           weights_size, hit_ratio);
}

// Phase 2: Run kernels that benefit from persistent L2 data
// (weights stay cached in L2 across multiple kernel launches)
__global__ void computeKernel(const float* __restrict__ weights,
                              const float* __restrict__ input,
                              float* __restrict__ output,
                              int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Access weights (persistent in L2) and input (streaming)
        float sum = 0.0f;
        for (int w = 0; w < 256; ++w) {
            sum += weights[w] * input[idx * 256 + w];
        }
        output[idx] = sum;
    }
}

// Phase 3: Cleanup -- reset L2 cache when done
void cleanupPersistence(cudaStream_t stream)
{
    // Reset the access policy to normal
    cudaStreamAttrValue stream_attr;
    stream_attr.accessPolicyWindow.base_ptr = nullptr;
    stream_attr.accessPolicyWindow.num_bytes = 0;
    stream_attr.accessPolicyWindow.hitRatio = 0.0;
    stream_attr.accessPolicyWindow.hitProp = cudaAccessPropertyNormal;
    stream_attr.accessPolicyWindow.missProp = cudaAccessPropertyNormal;
    cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow,
                           &stream_attr);

    // Reset persisting L2 cache lines
    cudaCtxResetPersistingL2Cache();

    // Optionally release the set-aside L2 cache
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0);

    printf("L2 persistence cleaned up\n");
}

// Main driver
int main()
{
    int device = 0;
    cudaSetDevice(device);

    // Allocate data
    size_t weights_size = 4 * 1024 * 1024; // 4 MB of weights
    size_t input_size = 64 * 1024 * 1024;  // 64 MB of input
    size_t output_size = 256 * 1024;        // 256 KB of output

    float *d_weights, *d_input, *d_output;
    cudaMalloc(&d_weights, weights_size);
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, output_size);

    // Initialize weights (would normally load from somewhere)
    cudaMemset(d_weights, 1, weights_size);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Setup L2 persistence
    setupPersistence(d_weights, weights_size, stream);

    // Run multiple kernel launches -- weights stay in L2 across launches
    int num_iterations = 100;
    for (int i = 0; i < num_iterations; ++i) {
        int N = output_size / sizeof(float);
        int block_size = 256;
        int grid_size = (N + block_size - 1) / block_size;

        computeKernel<<<grid_size, block_size, 0, stream>>>(
            d_weights, d_input, d_output, N);
    }

    // Cleanup
    cleanupPersistence(stream);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaFree(d_weights);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
```

### 10.5.4 Querying Current L2 Cache Limits

```cpp
// Query the current persisting L2 cache size limit
size_t current_limit;
cudaDeviceGetLimit(&current_limit, cudaLimitPersistingL2CacheSize);
printf("Current L2 persisting limit: %zu bytes\n", current_limit);
```

### 10.5.5 Removing Stream Attributes

To remove an access policy window from a stream, use `cudaStreamSetAttribute` with `hitProp` and `missProp` set to `cudaAccessPropertyNormal`:

```cpp
cudaStreamAttrValue stream_attr;
stream_attr.accessPolicyWindow.base_ptr = nullptr;
stream_attr.accessPolicyWindow.num_bytes = 0;
stream_attr.accessPolicyWindow.hitRatio = 0.0;
stream_attr.accessPolicyWindow.hitProp = cudaAccessPropertyNormal;
stream_attr.accessPolicyWindow.missProp = cudaAccessPropertyNormal;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attr);
```

This effectively disables the access policy window for the stream, returning all cache behavior to the default policy.
