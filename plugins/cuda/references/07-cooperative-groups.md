# 7. Cooperative Groups

## 7.1 Introduction

Cooperative Groups (CG) is a CUDA extension introduced in CUDA 9 that provides a model for organizing and synchronizing groups of cooperating threads. It supersedes the older `__syncthreads()` and `__ballot()` / `__shfl*()` warp-level primitives by offering a type-safe, composable, and future-proof API.

### Motivation

Traditional CUDA synchronization primitives have several limitations:

- `__syncthreads()` synchronizes all threads in a block, but cannot synchronize subsets of threads safely.
- Warp-level primitives (`__shfl_sync()`, `__ballot_sync()`, etc.) require an explicit mask and are fragile under warp divergence.
- There is no standard way to synchronize across thread blocks or across an entire grid without resorting to multiple kernel launches.

Cooperative Groups addresses these issues by:

- Representing groups of threads as first-class objects.
- Providing a uniform `sync()` method that works on any group.
- Supporting hierarchical partitioning (grid, block, warp, sub-warp tiles).
- Enabling cross-block and cross-grid synchronization when combined with cooperative kernel launches.

### Header

```cpp
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
```

### Key Concepts

- **Group**: A collection of threads that can cooperate. Every group exposes a common interface (`thread_rank()`, `num_threads()`, `sync()`, etc.).
- **Implicit group**: Obtained from the hardware/hierarchy without explicit construction (`this_thread_block()`, `this_grid()`, etc.).
- **Partitioned group**: Created by subdividing an existing group (`tiled_partition`, `labeled_partition`, `binary_partition`).
- **Scope**: The visibility of synchronization -- block, device, system.

---

## 7.2 Implicit Groups

Implicit groups are obtained from the execution hierarchy. They are not constructed explicitly; instead, they are queried.

| Group Handle | Scope | Description |
|---|---|---|
| `cg::this_thread_block()` | Block | All threads in the current thread block |
| `cg::this_grid()` | Grid (Device) | All threads in the grid (requires cooperative launch) |
| `cg::coalesced_threads()` | Warp | Currently active (coalesced) threads in the current warp |
| `cg::this_cluster()` | Cluster | All threads in the current cluster (requires CC 9.0+) |

### Obtaining Implicit Groups

```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void kernel() {
    // Block-level group
    auto block = cg::this_thread_block();

    // Grid-level group (requires cooperative launch)
    auto grid = cg::this_grid();

    // Coalesced (active) threads in warp
    auto active = cg::coalesced_threads();

#if __CUDA_ARCH__ >= 900
    // Cluster-level group (requires CC 9.0+)
    auto cluster = cg::this_cluster();
#endif
}
```

### `this_thread_block()`

Returns a `thread_block` handle representing all threads in the current block. This is the most commonly used implicit group.

```cpp
auto block = cg::this_thread_block();

// Equivalent to __syncthreads() but extensible
block.sync();
```

### `this_grid()`

Returns a `grid_group` handle representing all threads across all blocks in the grid. This requires the kernel to be launched using `cudaLaunchCooperativeKernel()` or `cudaLaunchCooperativeKernelMultiDevice()`.

```cpp
auto grid = cg::this_grid();
if (grid.is_valid()) {
    grid.sync(); // Synchronize all threads in the entire grid
}
```

### `coalesced_threads()`

Returns a `coalesced_group` containing only the currently active (non-diverged) threads within the current warp. This is useful for performing collective operations on a dynamically determined subset of warp lanes.

```cpp
__global__ void kernel(int* result) {
    // Only threads where (threadIdx.x % 2 == 0) proceed
    if (threadIdx.x % 2 == 0) {
        auto active = cg::coalesced_threads();
        // active.num_threads() may be less than 32
        int sum = cg::reduce(active, threadIdx.x, cg::plus<int>());
        if (active.thread_rank() == 0) {
            atomicAdd(result, sum);
        }
    }
}
```

### `this_cluster()` (CC 9.0+)

Returns a `cluster_group` representing all threads in the current Thread Block Cluster. Clusters allow multiple thread blocks to cooperate directly through shared memory.

```cpp
__global__ void __cluster_dims__(2, 1, 1) kernel() {
    auto cluster = cg::this_cluster();
    printf("Cluster has %d blocks, I am block %d\n",
           cluster.num_blocks(), cluster.block_rank());
}
```

---

## 7.3 Member Functions (Common to All Groups)

All group types expose a common interface. The following member functions are available on any cooperative group object.

### Query Functions

| Function | Return Type | Description |
|---|---|---|
| `thread_rank()` | `unsigned int` | Rank of the calling thread within the group, in `[0, num_threads())` |
| `num_threads()` | `unsigned int` | Total number of threads in the group |
| `dim_threads()` | `dim3` | Dimensions of the group (for `thread_block`, returns block dimensions) |
| `group_index()` | `dim3` | 3D index of this group within the parent hierarchy (e.g., block index within grid) |
| `thread_index()` | `dim3` | 3D index of the calling thread within this group |

### Synchronization Functions

| Function | Description |
|---|---|
| `sync()` | Synchronize all threads in the group (barrier) |
| `barrier_arrive()` | Arrive at the barrier and receive a token (non-blocking) |
| `barrier_wait(token)` | Wait on a previously arrived barrier token (blocking) |

### Usage Examples

```cpp
__global__ void kernel(int* data) {
    auto block = cg::this_thread_block();

    // Each thread writes its rank
    data[block.thread_rank()] = block.thread_rank();

    // Synchronize so all writes are visible
    block.sync();

    // Now all threads can safely read any element
    int val = data[(block.thread_rank() + 1) % block.num_threads()];

    // Query 3D structure (for 2D/3D blocks)
    dim3 blockDim = block.dim_threads();
    dim3 blockIdx3d = block.group_index();  // block's position in grid
    dim3 threadIdx3d = block.thread_index(); // thread's position in block
}
```

### Split Barrier Example

The split barrier allows overlapping independent work with the synchronization:

```cpp
__global__ void kernel(float* output, const float* input) {
    auto block = cg::this_thread_block();
    __shared__ float shared[256];

    // Phase 1: Each thread loads data
    shared[threadIdx.x] = input[threadIdx.x];

    // Arrive at barrier -- does NOT block
    auto token = block.barrier_arrive();

    // Independent work that doesn't depend on shared[] being fully loaded
    float local_val = some_computation(input[threadIdx.x]);

    // Now wait for all threads to finish loading
    block.barrier_wait(std::move(token));

    // Safe to read all of shared[]
    float result = shared[(threadIdx.x + 1) % 256] + local_val;
    output[threadIdx.x] = result;
}
```

### Group-Specific Details

For `thread_block`:
- `group_index()` returns `blockIdx` (the block's index in the grid).
- `thread_index()` returns `threadIdx` (the thread's index within the block).
- `dim_threads()` returns `blockDim` (the block dimensions).

For `grid_group`:
- `group_index()` returns `{0, 0, 0}` (there is only one grid).
- `thread_index()` returns the linear thread rank within the grid.
- `dim_threads()` returns `{gridDim.x * blockDim.x, gridDim.y * blockDim.y, gridDim.z * blockDim.z}`.

For `coalesced_group`:
- `group_index()` and `thread_index()` are not meaningful; use `thread_rank()`.

For `cluster_group`:
- `group_index()` returns the block rank within the cluster.
- `thread_index()` returns the thread's local index within its block.

---

## 7.4 `thread_block_tile`

`thread_block_tile` represents a fixed-size tile of threads partitioned from a parent group. It provides efficient warp-level and sub-warp-level collective operations.

### Declaration

```cpp
template <unsigned int Size, typename ParentT = void>
class thread_block_tile;
```

- `Size` must be a power of 2 and less than or equal to 1024.
- For `Size <= 32`, the tile maps to a subset of warp lanes and uses hardware shuffle/vote instructions.
- For `Size > 32` on CC <= 7.5, use `block_tile_memory` to coordinate via shared memory.

### Creating Tiles

```cpp
auto block = cg::this_thread_block();

// Partition into tiles of 32 threads (warp-sized)
auto tile32 = cg::tiled_partition<32>(block);

// Partition into tiles of 4 threads (sub-warp)
auto tile4 = cg::tiled_partition<4>(block);

// Partition into tiles of 8 threads
auto tile8 = cg::tiled_partition<8>(block);

// Further partition an existing tile
auto sub_tile = cg::tiled_partition<2>(tile4);
```

### Shuffle Operations

Shuffle operations allow threads within a tile to exchange register values without using shared memory.

```cpp
auto tile = cg::tiled_partition<32>(block);
int val = threadIdx.x;

// Direct shuffle: get val from thread with specified rank
int from_rank5 = tile.shfl(val, 5);

// Shuffle up: get val from thread (my_rank - delta)
int from_above = tile.shfl_up(val, 1);  // value from rank-1

// Shuffle down: get val from thread (my_rank + delta)
int from_below = tile.shfl_down(val, 1); // value from rank+1

// Shuffle XOR: get val from thread (my_rank ^ mask)
int from_xor = tile.shfl_xor(val, 0x1);  // swap with neighbor
```

#### Shuffle Details

| Operation | Signature | Description |
|---|---|---|
| `shfl(v, rank)` | `T shfl(T v, unsigned int rank)` | Returns value `v` from the thread at `rank` |
| `shfl_up(v, delta)` | `T shfl_up(T v, unsigned int delta)` | Returns value `v` from thread at `my_rank - delta`. For rank < delta, returns `v` from rank 0. |
| `shfl_down(v, delta)` | `T shfl_down(T v, unsigned int delta)` | Returns value `v` from thread at `my_rank + delta`. For rank >= size - delta, returns `v` from last rank. |
| `shfl_xor(v, mask)` | `T shfl_xor(T v, unsigned int mask)` | Returns value `v` from thread at `my_rank ^ mask` |

```cpp
// Example: Warp-level reduction using shfl_down
__device__ int warp_reduce_sum(int val) {
    auto tile = cg::tiled_partition<32>(cg::this_thread_block());
    for (int offset = tile.num_threads() / 2; offset > 0; offset /= 2) {
        val += tile.shfl_down(val, offset);
    }
    return val; // Only rank 0 has the full sum
}
```

### Vote Operations

Vote operations allow threads to collectively evaluate a predicate.

```cpp
auto tile = cg::tiled_partition<32>(block);
bool my_predicate = (threadIdx.x > 10);

// any(): returns true if ANY thread in tile has predicate == true
bool any_result = tile.any(my_predicate);

// all(): returns true if ALL threads in tile have predicate == true
bool all_result = tile.all(my_predicate);

// ballot(): returns a bitmask where bit i is set if thread i's predicate is true
unsigned int mask = tile.ballot(my_predicate);
int count = __popc(mask); // Count number of true predicates
```

| Operation | Return Type | Description |
|---|---|---|
| `any(pred)` | `bool` | True if any thread in the tile has `pred == true` |
| `all(pred)` | `bool` | True if all threads in the tile have `pred == true` |
| `ballot(pred)` | `unsigned int` | Bitmask of predicate results across the tile |

### Match Operations

Match operations find threads with matching values.

```cpp
auto tile = cg::tiled_partition<32>(block);
int val = threadIdx.x % 4;

// match_any(): bitmask of threads with same value as calling thread
unsigned int same_mask = tile.match_any(val);

// match_all(): bitmask of ALL threads if all have same value, else 0
unsigned int all_mask = tile.match_all(val);
bool all_same = (all_mask != 0);
```

| Operation | Return Type | Description |
|---|---|---|
| `match_any(val)` | `unsigned int` | Bitmask of threads in tile with value equal to `val` of calling thread |
| `match_all(val)` | `unsigned int` | Bitmask of all threads if ALL threads have same `val`, otherwise 0 |

### Tiles Larger Than 32 Threads

For tiles with `Size > 32` on architectures before CC 8.0, Cooperative Groups uses shared memory to implement collective operations. On CC 8.0+ (Ampere and later), larger tiles can leverage hardware features.

```cpp
// On CC 8.0+, tiles > 32 work directly
auto tile64 = cg::tiled_partition<64>(block);
int val = cg::reduce(tile64, threadIdx.x, cg::plus<int>());
```

### Complete Tile Example: Parallel Prefix Sum

```cpp
__device__ float tile_prefix_sum(float val) {
    auto tile = cg::tiled_partition<32>(cg::this_thread_block());
    unsigned int rank = tile.thread_rank();
    unsigned int size = tile.num_threads();

    // Build prefix sum using shuffle_up
    for (int offset = 1; offset < size; offset *= 2) {
        float other = tile.shfl_up(val, offset);
        if (rank >= offset) {
            val += other;
        }
    }
    return val;
}
```

---

## 7.5 `cluster_group` (CC 9.0+)

Thread Block Clusters, introduced with the Hopper architecture (CC 9.0), allow multiple thread blocks to form a cluster. Blocks within a cluster can directly access each other's shared memory, enabling efficient inter-block cooperation.

### Obtaining a Cluster Group

```cpp
__global__ void __cluster_dims__(2, 2, 1) cluster_kernel() {
    auto cluster = cg::this_cluster();
    // ...
}
```

The `__cluster_dims__` annotation specifies the cluster dimensions (number of blocks in x, y, z).

### Member Functions

| Function | Return Type | Description |
|---|---|---|
| `block_rank()` | `unsigned int` | Linear rank of this block within the cluster |
| `num_blocks()` | `unsigned int` | Total number of blocks in the cluster |
| `dim_blocks()` | `dim3` | 3D dimensions of the cluster (number of blocks in each dimension) |
| `block_index()` | `dim3` | 3D index of this block within the cluster |
| `thread_rank()` | `unsigned int` | Linear rank of this thread across all threads in the cluster |
| `num_threads()` | `unsigned int` | Total threads in the cluster |
| `sync()` | `void` | Synchronize all threads in the cluster |
| `query_shared_rank(void* addr)` | `unsigned int` | Returns the block rank that owns the shared memory at `addr` |
| `map_shared_rank(void* addr, unsigned int rank)` | `void*` | Returns a pointer to the shared memory of block at `rank`, mapped from the same offset as `addr` |

### Cross-Block Shared Memory Access

The most powerful feature of clusters is the ability for one block to directly read and write another block's shared memory.

```cpp
__global__ void __cluster_dims__(4, 1, 1) cluster_kernel(float* output) {
    auto cluster = cg::this_cluster();
    extern __shared__ float shared_data[];

    // Each block writes its rank into shared memory
    shared_data[0] = cluster.block_rank();

    cluster.sync();

    // Each block reads shared memory from a neighbor block
    unsigned int neighbor = (cluster.block_rank() + 1) % cluster.num_blocks();

    // Map neighbor's shared memory into our address space
    float* neighbor_shared = (float*)cluster.map_shared_rank(shared_data, neighbor);

    // Read neighbor's data
    float neighbor_rank = neighbor_shared[0];
    output[threadIdx.x + blockIdx.x * blockDim.x] = neighbor_rank;
}
```

### Cluster Launch Configuration

Clusters can be launched using the CUDA runtime or driver API:

```cpp
// Runtime API with cluster dimensions
cudaLaunchConfig_t config = {0};
config.gridDim = {8, 1, 1};
config.blockDim = {128, 1, 1};

cudaLaunchAttribute clusterAttr;
clusterAttr.id = cudaLaunchAttributeClusterDimension;
clusterAttr.val.clusterDim.x = 2;
clusterAttr.val.clusterDim.y = 1;
clusterAttr.val.clusterDim.z = 1;
config.attrs = &clusterAttr;
config.numAttrs = 1;

cudaLaunchKernelEx(&config, my_kernel, arg1, arg2);
```

### Use Cases

- **Halo exchange**: Neighboring blocks sharing border data without going through global memory.
- **Collaborative work**: Multiple blocks cooperatively processing a shared tile of data.
- **Reduced global memory traffic**: Sharing intermediate results across blocks via shared memory.

---

## 7.6 `grid_group`

A `grid_group` represents all threads in the entire kernel grid. It enables synchronization across all thread blocks without terminating the kernel.

### Requirements

- The kernel must be launched with `cudaLaunchCooperativeKernel()` or `cudaLaunchCooperativeKernelMultiDevice()`.
- All blocks must be resident on the GPU simultaneously (the grid size cannot exceed the maximum number of concurrent blocks).
- The device must support cooperative launch. Check with:
  ```cpp
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  if (prop.cooperativeLaunch) {
      // Device supports cooperative kernel launch
  }
  ```

### API

```cpp
auto grid = cg::this_grid();

// Check if grid sync is valid
if (grid.is_valid()) {
    // All threads in the entire grid synchronize here
    grid.sync();
}
```

| Function | Description |
|---|---|
| `is_valid()` | Returns `true` if the grid group was created via cooperative launch (i.e., `sync()` is legal) |
| `sync()` | Synchronizes all threads across all blocks in the grid |
| `thread_rank()` | Linear rank of the calling thread across the grid |
| `num_threads()` | Total number of threads in the grid |
| `dim_threads()` | 3D dimensions of the grid (total threads in each dimension) |
| `group_index()` | Returns `{0, 0, 0}` (only one grid) |
| `thread_index()` | Linear thread index within the grid |

### Cooperative Kernel Launch

```cpp
// Standard launch (does NOT support grid sync)
// my_kernel<<<gridSize, blockSize>>>(args...);

// Cooperative launch (supports grid sync)
void* args[] = { &arg1, &arg2 };
cudaLaunchCooperativeKernel(
    (void*)my_kernel,
    gridSize,    // grid dimensions
    blockSize,   // block dimensions
    args,        // kernel arguments
    shmemSize,   // shared memory size
    stream       // CUDA stream
);
```

### Grid Size Constraints

The maximum grid size for cooperative launch is limited by the number of blocks that can be resident simultaneously:

```cpp
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);

// Maximum cooperative grid size
int maxBlocksPerSM = prop.maxBlocksPerMultiProcessor;
int numSMs = prop.multiProcessorCount;
int maxCooperativeBlocks = maxBlocksPerSM * numSMs;

// Ensure grid size does not exceed this
dim3 gridSize = dim3(min(desiredBlocks, maxCooperativeBlocks), 1, 1);
```

### Example: Global Reduction with Grid Sync

```cpp
__global__ void grid_reduce(float* data, float* partial, int n) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    // Phase 1: Block-level reduction
    float sum = 0.0f;
    for (int i = grid.thread_rank(); i < n; i += grid.num_threads()) {
        sum += data[i];
    }

    // Block-level reduction
    __shared__ float block_sum[256];
    block_sum[threadIdx.x] = sum;
    block.sync();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_sum[threadIdx.x] += block_sum[threadIdx.x + s];
        }
        block.sync();
    }

    // Each block writes its partial sum
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = block_sum[0];
    }

    // Synchronize the ENTIRE GRID
    grid.sync();

    // Phase 2: Single block reduces the partial sums
    if (blockIdx.x == 0) {
        float final_sum = 0.0f;
        for (int i = threadIdx.x; i < gridDim.x; i += blockDim.x) {
            final_sum += partial[i];
        }
        block_sum[threadIdx.x] = final_sum;
        block.sync();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                block_sum[threadIdx.x] += block_sum[threadIdx.x + s];
            }
            block.sync();
        }

        if (threadIdx.x == 0) {
            data[0] = block_sum[0];
        }
    }
}
```

---

## 7.7 Partitioning Operations

Cooperative Groups supports dynamic partitioning of groups into subgroups based on various criteria.

### `tiled_partition<Size>(parent)`

Partitions the parent group into fixed-size tiles. The parent group size must be evenly divisible by `Size`. Threads are assigned to tiles in row-major order.

```cpp
auto block = cg::this_thread_block();

// Create warp-sized tiles
auto warp = cg::tiled_partition<32>(block);

// Create sub-warp tiles of size 8
auto tile8 = cg::tiled_partition<8>(block);

// Create tile of size 4 from a tile of size 8
auto tile4 = cg::tiled_partition<4>(tile8);
```

**Constraints**:
- `Size` must be a power of 2.
- `Size` must be <= 1024.
- `parent.num_threads()` must be evenly divisible by `Size`.

### `labeled_partition(group, label)`

Partitions a group into subgroups based on a label value. Threads with the same label end up in the same subgroup. This is useful for grouping threads that need to cooperate on the same task.

```cpp
auto warp = cg::tiled_partition<32>(cg::this_thread_block());

// Group threads by their warp-lane modulo 4
int label = warp.thread_rank() % 4;
auto labeled = cg::labeled_partition(warp, label);

// Threads in 'labeled' all have the same label value
printf("Thread %d -> label %d, subgroup size %d\n",
       warp.thread_rank(), label, labeled.num_threads());
```

### `binary_partition(group, predicate)`

Partitions a group into two subgroups based on a boolean predicate: one for threads where `predicate` is true, and one for threads where it is false.

```cpp
auto warp = cg::tiled_partition<32>(cg::this_thread_block());

bool predicate = (warp.thread_rank() < 16);
auto subgroup = cg::binary_partition(warp, predicate);

// 'subgroup' contains only threads with the same predicate value
if (predicate) {
    // All threads here have predicate == true
    printf("True group: rank %d, size %d\n",
           subgroup.thread_rank(), subgroup.num_threads());
}
```

### Partitioning Summary

| Operation | Description | Example |
|---|---|---|
| `tiled_partition<Size>(parent)` | Fixed-size subgroups in row-major order | `cg::tiled_partition<16>(block)` |
| `labeled_partition(g, label)` | Subgroups by label value | `cg::labeled_partition(warp, key)` |
| `binary_partition(g, pred)` | Two subgroups by boolean | `cg::binary_partition(warp, flag)` |

### Dynamic Partitioning Example

```cpp
__global__ void dynamic_partition_kernel(int* keys, float* values, float* results) {
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    int key = keys[threadIdx.x];
    float val = values[threadIdx.x];

    // Group threads by key
    auto group = cg::labeled_partition(warp, key);

    // Reduce within the group (only threads with same key)
    float sum = cg::reduce(group, val, cg::plus<float>());

    if (group.thread_rank() == 0) {
        results[key] = sum;
    }
}
```

---

## 7.8 Collective Operations

Cooperative Groups provides collective operations that perform computations across all threads in a group.

### Reduce

Performs a reduction across all threads in the group using the specified operator.

```cpp
// General form
auto result = cg::reduce(group, value, operator);
```

**Available Operators**:

| Operator | Description | Header |
|---|---|---|
| `cg::plus<T>` | Sum | `<cooperative_groups/reduce.h>` |
| `cg::less<T>` | Minimum | `<cooperative_groups/reduce.h>` |
| `cg::greater<T>` | Maximum | `<cooperative_groups/reduce.h>` |
| `cg::bit_and<T>` | Bitwise AND | `<cooperative_groups/reduce.h>` |
| `cg::bit_or<T>` | Bitwise OR | `<cooperative_groups/reduce.h>` |
| `cg::bit_xor<T>` | Bitwise XOR | `<cooperative_groups/reduce.h>` |

**Hardware Acceleration**: On CC 8.0+ (Ampere and later), reduce is hardware-accelerated for 4-byte types (int, unsigned int, float) within warp-sized tiles.

```cpp
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__global__ void reduce_kernel(int* global_sum, int N) {
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);

    int local_sum = 0;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x;
         i < N;
         i += blockDim.x * gridDim.x) {
        local_sum += some_value(i);
    }

    // Warp-level reduce (hardware accelerated on CC 8.0+)
    int warp_sum = cg::reduce(tile, local_sum, cg::plus<int>());

    // Write per-warp results to shared memory
    __shared__ int warp_sums[32]; // blockDim.x / 32
    if (tile.thread_rank() == 0) {
        warp_sums[threadIdx.x / 32] = warp_sum;
    }
    block.sync();

    // Final block-level reduce by first warp
    if (threadIdx.x < blockDim.x / 32) {
        int block_sum = cg::reduce(tile, warp_sums[tile.thread_rank()],
                                   cg::plus<int>());
        if (tile.thread_rank() == 0) {
            atomicAdd(global_sum, block_sum);
        }
    }
}
```

### Scan (Prefix Sum)

Performs inclusive or exclusive scan (prefix sum) across all threads in the group.

```cpp
#include <cooperative_groups/scan.h>

// Inclusive scan: result[i] = op(val[0], val[1], ..., val[i])
auto inclusive = cg::inclusive_scan(group, value, operator);

// Exclusive scan: result[i] = op(val[0], val[1], ..., val[i-1])
auto exclusive = cg::exclusive_scan(group, value, operator);

// With initial value for exclusive scan
auto exclusive_with_init = cg::exclusive_scan(group, value, init, operator);
```

**Variants with update**:

These variants also return whether the calling thread holds the final accumulated value and what that value is.

```cpp
// inclusive_scan_update returns pair<scan_result, is_final>
auto [result, is_final] = cg::inclusive_scan_update(group, value, op);

// exclusive_scan_update returns pair<scan_result, pair<is_final, final_value>>
auto [exc_result, final_info] = cg::exclusive_scan_update(group, value, op);
bool is_final = final_info.first;
auto final_val = final_info.second;
```

```cpp
#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>
namespace cg = cooperative_groups;

__global__ void scan_kernel(int* output, const int* input, int n) {
    auto tile = cg::tiled_partition<32>(cg::this_thread_block());

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int val = (idx < n) ? input[idx] : 0;

    // Compute inclusive scan within tile
    int prefix = cg::inclusive_scan(tile, val, cg::plus<int>());

    if (idx < n) {
        output[idx] = prefix;
    }
}
```

### `memcpy_async`

Asynchronous memory copy collective that operates across all threads in a group. This is particularly useful for loading data from global memory into shared memory asynchronously.

```cpp
#include <cooperative_groups/memcpy_async.h>

// Basic async copy
cg::memcpy_async(group, dst, src, size);

// Typed async copy
cg::memcpy_async(group, dst, src, count); // count elements

// Block-level async copy with pipeline (staged)
cg::memcpy_async(block, dst_shared, src_global, block.size() * sizeof(float));
cg::wait(block); // Wait for completion

// Wait with pipeline stage
cg::wait_prior<N>(block); // Wait for all stages older than N
```

**With Pipeline Stages**:

```cpp
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
namespace cg = cooperative_groups;

__global__ void pipeline_copy_kernel(float* output, const float* input, int n) {
    __shared__ float buffer[2][256]; // Double buffer
    auto block = cg::this_thread_block();

    for (int base = 0; base < n; base += 256) {
        int stage = (base / 256) % 2;

        // Asynchronously copy next chunk into shared memory
        cg::memcpy_async(block, buffer[stage], &input[base],
                         min(256, n - base) * sizeof(float));
        cg::wait(block);

        // Process data in shared memory
        int count = min(256, n - base);
        if (threadIdx.x < count) {
            buffer[stage][threadIdx.x] *= 2.0f; // Example processing
        }
        block.sync();

        // Copy results back
        if (threadIdx.x < count) {
            output[base + threadIdx.x] = buffer[stage][threadIdx.x];
        }
    }
}
```

### Collective Operations Summary

| Operation | Signature | Description |
|---|---|---|
| `reduce(group, val, op)` | `T` | Reduce `val` across group with `op` |
| `inclusive_scan(group, val, op)` | `T` | Inclusive prefix scan |
| `exclusive_scan(group, val, op)` | `T` | Exclusive prefix scan |
| `exclusive_scan(group, val, init, op)` | `T` | Exclusive scan with initial value |
| `inclusive_scan_update(group, val, op)` | `pair<T, bool>` | Inclusive scan with final flag |
| `exclusive_scan_update(group, val, op)` | `pair<T, pair<bool, T>>` | Exclusive scan with final info |
| `memcpy_async(group, dst, src, size)` | `void` | Async memory copy |
| `wait(group)` | `void` | Wait for pending memcpy_async |
| `wait_prior<N>(group)` | `void` | Wait for stages older than N |

---

## 7.9 Synchronization

### Full Barrier Synchronization

The simplest form of synchronization. All threads in the group must reach the `sync()` call before any can proceed.

```cpp
auto block = cg::this_thread_block();
auto grid = cg::this_grid();
auto cluster = cg::this_cluster();

// Block-level sync (replaces __syncthreads())
block.sync();

// Grid-level sync (requires cooperative launch)
if (grid.is_valid()) {
    grid.sync();
}

// Cluster-level sync (CC 9.0+)
cluster.sync();
```

### Split Barrier (Arrive-Wait)

Split barriers separate the synchronization into two phases: arrival and wait. This allows threads to perform independent work between arriving and waiting, which can improve performance by overlapping computation with synchronization.

```cpp
auto block = cg::this_thread_block();

// Phase 1: Arrive at the barrier (non-blocking)
auto token = block.barrier_arrive();

// Independent work that doesn't depend on other threads' barrier-side effects
do_independent_work();

// Phase 2: Wait for all threads to arrive (blocking)
block.barrier_wait(std::move(token));
```

**Rules for split barriers**:
- Each thread must call `barrier_arrive()` before `barrier_wait()`.
- The token from `barrier_arrive()` must be moved to `barrier_wait()`.
- Independent work between arrive and wait must not depend on data that other threads produce before their own arrive.
- All threads must eventually call both arrive and wait to avoid deadlock.

### Split Barrier with Producer-Consumer Pattern

```cpp
__shared__ float data[256];
auto block = cg::this_thread_block();

// Producer: odd threads write data
if (threadIdx.x % 2 == 0) {
    data[threadIdx.x] = compute_value(threadIdx.x);
}

auto token = block.barrier_arrive();

// Consumer: even threads can do independent work
float local = compute_something_else();

block.barrier_wait(std::move(token));

// Now all producer writes are visible
float consumed = data[(threadIdx.x + 128) % 256];
```

### Synchronization and Memory Visibility

When `sync()` completes, it guarantees:
1. All memory writes by threads in the group before the `sync()` are visible to all threads in the group after the `sync()`.
2. No thread reads or writes to the same memory location around a `sync()` without an intervening `sync()` (data races are undefined behavior).

```cpp
// Correct usage:
__shared__ int data[256];
auto block = cg::this_thread_block();

data[threadIdx.x] = threadIdx.x;  // Write
block.sync();                       // Synchronize
int val = data[255 - threadIdx.x]; // Safe read of other threads' writes
block.sync();                       // Synchronize again before next write
data[threadIdx.x] = val * 2;       // Safe write
```

### Synchronization Scope Comparison

| Synchronization Method | Scope | Prerequisites |
|---|---|---|
| `block.sync()` | Thread block | None |
| `tile.sync()` | Tile within block | None |
| `cluster.sync()` | Cluster of blocks | CC 9.0+, cluster launch |
| `grid.sync()` | Entire grid | Cooperative kernel launch |
| `coalesced.sync()` | Active warp threads | None (but only in convergent code) |

---

## 7.10 Best Practices and Common Patterns

### Warp-Level Reduction Pattern

```cpp
__device__ int warp_sum(int val) {
    auto tile = cg::tiled_partition<32>(cg::this_thread_block());
    return cg::reduce(tile, val, cg::plus<int>());
}
```

### Block-Level Reduction with Shared Memory

```cpp
__device__ int block_sum(int val) {
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);
    __shared__ int warp_results[32];

    // Reduce within each warp
    int warp_val = cg::reduce(tile, val, cg::plus<int>());

    // First thread in each warp writes result
    if (tile.thread_rank() == 0) {
        warp_results[threadIdx.x / 32] = warp_val;
    }
    block.sync();

    // First warp reduces the warp results
    if (threadIdx.x < blockDim.x / 32) {
        int warp_sum = cg::reduce(tile, warp_results[tile.thread_rank()],
                                  cg::plus<int>());
        if (tile.thread_rank() == 0) {
            warp_results[0] = warp_sum;
        }
    }
    block.sync();

    return warp_results[0];
}
```

### Choosing the Right Group

| Scenario | Recommended Group |
|---|---|
| Synchronize all threads in a block | `this_thread_block()` |
| Warp-wide shuffle/vote operations | `tiled_partition<32>(block)` |
| Sub-warp cooperation (e.g., 8 threads) | `tiled_partition<8>(block)` |
| Dynamic subgroup by condition | `coalesced_threads()` or `binary_partition()` |
| Cross-block shared memory access | `this_cluster()` (CC 9.0+) |
| Global synchronization without kernel termination | `this_grid()` with cooperative launch |

### Common Pitfalls

1. **Deadlock from divergent sync**: Never call `sync()` inside a divergent branch unless all threads in the group take the same branch.
   ```cpp
   // WRONG: deadlock if some threads take different paths
   if (threadIdx.x < 16) {
       block.sync(); // Only half the threads arrive -> deadlock!
   }

   // CORRECT: sync outside the branch
   if (threadIdx.x < 16) {
       data[threadIdx.x] = val;
   }
   block.sync(); // All threads sync together
   ```

2. **Using grid sync without cooperative launch**: `this_grid().sync()` will produce undefined behavior or deadlock if the kernel was not launched with `cudaLaunchCooperativeKernel()`.

3. **Exceeding maximum cooperative grid size**: The total number of blocks must be <= `maxBlocksPerMultiProcessor * multiProcessorCount` for cooperative launch.

4. **Invalid tile sizes**: `tiled_partition<Size>()` requires `Size` to be a power of 2 and the parent group size must be divisible by `Size`.
