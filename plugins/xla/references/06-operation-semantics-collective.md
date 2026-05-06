# XLA Operation Semantics: Collective Operations

This reference provides comprehensive documentation of all collective XLA operations. Collective operations coordinate computation across multiple devices (replicas) in a distributed XLA computation. They are fundamental to implementing distributed training and multi-device inference in frameworks such as JAX, TensorFlow, and PyTorch (via torch-xla).

---

## Table of Contents

1. [AllGather](#allgather)
2. [AllReduce](#allreduce)
3. [AllToAll](#alltoall)
4. [RaggedAllToAll](#raggedalltoall)
5. [CollectiveBroadcast](#collectivebroadcast)
6. [CollectivePermute](#collectivepermute)
7. [ReduceScatter](#reducescatter)
8. [Common Patterns](#common-patterns)
9. [StableHLO Cross-References](#stablehlo-cross-references)

---

## AllGather

`AllGather` concatenates the values of an operand from all replicas (or from a subset of replicas as defined by replica groups) along a specified dimension. Each replica contributes its local operand value, and every participating replica receives the full concatenated result.

### Signature

```
AllGather(operand, all_gather_dimension, shard_count, replica_groups,
          channel_id, layout, use_global_device_ids)
```

### Arguments

| Argument | Type | Description |
|---|---|---|
| `operand` | `XlaOp` | The input tensor to be gathered from each replica. The shape of this operand represents the local shard held by each replica. |
| `all_gather_dimension` | `int64` | The dimension along which the operand values from different replicas are concatenated. Must be in the range `[0, rank(operand))`. |
| `shard_count` | `int64` | The number of replicas (or devices) participating in the gather within each replica group. This must equal the size of each replica group. |
| `replica_groups` | `std::vector<std::vector<int64>>` | A vector of vectors specifying groups of replicas that participate together. Each inner vector is a group of replica IDs. If empty (i.e., `{}`), all replicas form a single group. |
| `channel_id` | `std::optional<int64>` | An optional channel ID used to match corresponding collective operations across different HLO modules (cross-module communication). If `std::nullopt`, the operation is matched within the same module. |
| `layout` | `std::optional<Layout>` | The desired layout of the output. If not specified, a default layout is used. |
| `use_global_device_ids` | `bool` | When `true`, the replica IDs in `replica_groups` are interpreted as global device IDs rather than per-replica IDs. This is relevant in multi-module computations where devices are assigned globally. Default is `false`. |

### Semantics

Each replica holds a local shard of a logically unified tensor. `AllGather` reassembles these shards by concatenating them along `all_gather_dimension`. Within each replica group, the shards are concatenated in the order of the replica IDs listed in the group definition.

**Output Shape**: The output shape is identical to the input shape except that `all_gather_dimension` is multiplied by `shard_count`:

```
output_shape = operand_shape
output_shape[all_gather_dimension] *= shard_count
```

The output on every replica in the same group is identical.

### Example: 2 Replicas

Consider two replicas, each holding a `f32[4]` tensor:

- **Replica 0**: `[a0, a1, a2, a3]`
- **Replica 1**: `[b0, b1, b2, b3]`

After `AllGather(operand, all_gather_dimension=0, shard_count=2, replica_groups={{0, 1}})`:

- **Replica 0** output: `[a0, a1, a2, a3, b0, b1, b2, b3]` (shape `f32[8]`)
- **Replica 1** output: `[a0, a1, a2, a3, b0, b1, b2, b3]` (shape `f32[8]`)

Both replicas receive the same concatenated result.

#### HLO Text Representation

```
%all_gather = f32[8]{0} all-gather(f32[4]{0} %operand),
  all_gather_dimension=0, shard_count=2,
  replica_groups={{0,1}}
```

#### Multi-Dimensional Example

Two replicas, each holding `f32[2, 3]`:

- **Replica 0**: `[[1, 2, 3], [4, 5, 6]]`
- **Replica 1**: `[[7, 8, 9], [10, 11, 12]]`

After `AllGather(operand, all_gather_dimension=1, shard_count=2, replica_groups={{0, 1}})`:

- Output shape: `f32[2, 6]`
- **Both replicas** receive: `[[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]`

### Internal Decomposition: AllGatherStart / AllGatherDone

Internally, `AllGather` decomposes into two separate operations to enable overlap of communication and computation:

1. **AllGatherStart**: Initiates the asynchronous all-gather operation. Returns a target tensor that serves as a future. Does not block the calling computation.

   ```
   %target = all-gather-start(f32[4] %operand),
     all_gather_dimension=0, shard_count=2,
     replica_groups={{0,1}}
   ```

2. **AllGatherDone**: Blocks until the all-gather operation initiated by `AllGatherStart` completes. Takes the target tensor as input and returns the result.

   ```
   %result = all-gather-done(f32[8] %target)
   ```

This decomposition is critical for performance optimization, allowing the compiler to schedule independent computations while the collective operation is in flight.

### Replica Groups in Detail

The `replica_groups` parameter controls which subsets of replicas communicate:

- **Single group (all replicas)**: `replica_groups={}` or `replica_groups={{0,1,2,3}}`
  All replicas participate in one collective operation.

- **Multiple groups**: `replica_groups={{0,1}, {2,3}}`
  Replicas 0 and 1 form one group; replicas 2 and 3 form another. Each group performs an independent all-gather.

- **Ordering matters**: Within a group, the order of replica IDs determines the concatenation order in the output. `{{1,0}}` would place replica 1's data before replica 0's.

### Constraints

- `shard_count` must be positive and must match the size of each replica group.
- All replica groups must be non-empty and the same size.
- Each replica ID must appear in exactly one group.
- `all_gather_dimension` must be a valid dimension index for the operand.

---

## AllReduce

`AllReduce` performs a reduction operation (specified by a computation) across all replicas (or replica groups) and makes the result available on all participating replicas. It is the most commonly used collective operation in distributed training for gradient aggregation.

### Signature

```
AllReduce(operand, computation, replica_groups, channel_id,
          shape_with_layout, use_global_device_ids)
```

### Arguments

| Argument | Type | Description |
|---|---|---|
| `operand` | `XlaOp` | The input tensor to be reduced across replicas. Array of any supported element type. |
| `computation` | `XlaComputation` | A reduction function that takes two scalar inputs and produces one scalar output. This function must be associative and commutative for correctness (e.g., addition, multiplication, min, max). |
| `replica_groups` | `std::vector<std::vector<int64>>` | Groups of replicas. See `AllGather` for detailed semantics. If empty `{}`, all replicas form one group. |
| `channel_id` | `std::optional<int64>` | Optional channel ID for cross-module communication matching. |
| `shape_with_layout` | `std::optional<Shape>` | The desired shape and layout of the output. |
| `use_global_device_ids` | `bool` | Whether to interpret replica IDs in `replica_groups` as global device IDs. Default `false`. |

### Semantics

For each element in the operand tensor, `AllReduce` applies the `computation` across all replicas in the same group. The result is the same tensor shape as the input, where every element has been reduced across replicas.

The computation is applied element-wise: for element at position `(i, j, ...)` in the operand, the reduction combines `operand[i,j,...]` from replica 0, `operand[i,j,...]` from replica 1, and so on, producing a single scalar that is written to position `(i, j, ...)` of every replica's output.

**Output shape**: Identical to the input shape.

### CrossReplicaSum

`CrossReplicaSum` is a specialized convenience function that performs `AllReduce` with addition as the computation:

```cpp
XlaOp CrossReplicaSum(XlaOp operand,
                      std::vector<std::vector<int64_t>> replica_groups);
```

It is equivalent to:

```cpp
// Build an add computation
XlaBuilder add_builder("add");
auto x = Parameter(&add_builder, 0, ShapeUtil::MakeShape(F32, {}), "x");
auto y = Parameter(&add_builder, 1, ShapeUtil::MakeShape(F32, {}), "y");
auto add_computation = Add(x, y);
// ...
AllReduce(operand, add_computation, replica_groups, ...);
```

### Example: Sum Across 4 Replicas

Four replicas, each holding `f32[2]`:

- **Replica 0**: `[1.0, 2.0]`
- **Replica 1**: `[3.0, 4.0]`
- **Replica 2**: `[5.0, 6.0]`
- **Replica 3**: `[7.0, 8.0]`

After `AllReduce(operand, add_computation, replica_groups={{0,1,2,3}})`:

- **All replicas** output: `[16.0, 20.0]`

#### HLO Text Representation

```
%add = ((f32[], f32[]) -> f32[]) add(F32[] %x, F32[] %y) {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %x, f32[] %y)
}

%all_reduce = f32[2]{0} all-reduce(f32[2]{0} %operand),
  to_apply=%add, replica_groups={{0,1,2,3}}
```

### Deadlock Warning: While Loops with Infeed

A critical correctness concern arises when `AllReduce` is used inside a `While` loop that also contains `Infeed` operations. This creates a potential deadlock:

**Scenario**: If different replicas take different loop paths (due to data-dependent control flow involving infeed data), one replica may execute `AllReduce` while another does not. The `AllReduce` is a blocking collective operation -- it waits for all participating replicas to reach it. If a replica never reaches the `AllReduce`, the other replicas will deadlock.

**Mitigation strategies**:
1. Ensure all replicas follow identical control flow paths (SPMD programming model).
2. Avoid placing collective operations inside conditionally executed code.
3. Use `channel_id` carefully to ensure proper matching of cross-module collectives.
4. Consider restructuring the computation to move collectives outside of conditional control flow.

### Internal Decomposition: AllReduceStart / AllReduceDone

Like `AllGather`, `AllReduce` decomposes into start/done pairs for asynchronous execution:

1. **AllReduceStart**: Begins the asynchronous reduction. Returns a target future.

   ```
   %target = all-reduce-start(f32[2] %operand),
     to_apply=%add, replica_groups={{0,1,2,3}}
   ```

2. **AllReduceDone**: Awaits completion and returns the reduced result.

   ```
   %result = all-reduce-done(f32[2] %target)
   ```

This pattern enables computation-communication overlap, which is essential for scaling distributed training to large cluster sizes.

### Common Reduction Computations

| Reduction | Computation Body | Use Case |
|---|---|---|
| Sum | `Add(x, y)` | Gradient aggregation |
| Product | `Mul(x, y)` | Probability computations |
| Maximum | `Max(x, y)` | Synchronization of max values |
| Minimum | `Min(x, y)` | Finding global minimum |
| Logical AND | `And(x, y)` | Consensus / barrier |
| Logical OR | `Or(x, y)` | Any-replica signaling |

### Replica Group Examples for AllReduce

**Data-parallel training** (all replicas in one group):
```
replica_groups = {}  // or {{0,1,2,...,N-1}}
```
All replicas' gradients are summed together.

**Model-parallel with data-parallel groups**:
```
// 8 GPUs: 2 model-parallel groups of 4 data-parallel replicas each
replica_groups = {{0,1,2,3}, {4,5,6,7}}
```
GPUs 0-3 sum their gradients independently from GPUs 4-7.

---

## AllToAll

`AllToAll` is a collective operation that performs a scatter from each replica to all other replicas, followed by a gather from all other replicas to each replica. It effectively re-partitions a tensor across the device dimension.

### Two Phases

1. **Scatter phase**: Each replica splits its local operand along `split_dimension` into `shard_count` parts and sends each part to a different replica.
2. **Gather phase**: Each replica concatenates the received parts along `concat_dimension`.

### Signature

```
AllToAll(operand, split_dimension, concat_dimension, split_count,
         replica_groups, layout, channel_id, use_global_device_ids)
```

### Arguments

| Argument | Type | Description |
|---|---|---|
| `operand` | `XlaOp` | The input tensor. |
| `split_dimension` | `int64` | The dimension along which the operand is split during the scatter phase. Must be in `[0, rank)`. |
| `concat_dimension` | `int64` | The dimension along which received shards are concatenated during the gather phase. Must be in `[0, rank)`. |
| `split_count` | `int64` | The number of splits (and thus the size of each replica group). The `split_dimension` of the operand must be evenly divisible by `split_count`. |
| `replica_groups` | `std::vector<std::vector<int64>>` | Groups of replicas. See `AllGather` for semantics. |
| `layout` | `std::optional<Layout>` | Optional desired output layout. |
| `channel_id` | `std::optional<int64>` | Optional channel ID for cross-module matching. |
| `use_global_device_ids` | `bool` | Whether replica IDs are global device IDs. Default `false`. |

### Semantics

For a replica group of size `split_count`, each replica splits its operand into `split_count` equal-sized slices along `split_dimension`. The i-th slice from each replica is sent to replica i. Then each replica concatenates all received slices along `concat_dimension`.

**Output shape**: The output shape differs from the input shape in two dimensions:
- `output_shape[split_dimension] = input_shape[split_dimension] / split_count`
- `output_shape[concat_dimension] = input_shape[concat_dimension] * split_count`

If `split_dimension == concat_dimension`, the output shape along that dimension is `input_shape[split_dimension]` (i.e., it stays the same, but the data is redistributed).

### Example 1: 4 Cores with f32[4,16] Operand

Four replicas (cores), each with a `f32[4,16]` operand:

- `split_dimension = 0` (split along the first dimension, size 4)
- `concat_dimension = 1` (concatenate along the second dimension)
- `split_count = 4`

**Scatter phase** (each replica splits along dim 0 into 4 parts of size `f32[1,16]`):
- Replica 0: splits `A[0:1,:]`, `A[1:2,:]`, `A[2:3,:]`, `A[3:4,:]`
- Replica 1: splits `B[0:1,:]`, `B[1:2,:]`, `B[2:3,:]`, `B[3:4,:]`
- Replica 2: splits `C[0:1,:]`, `C[1:2,:]`, `C[2:3,:]`, `C[3:4,:]`
- Replica 3: splits `D[0:1,:]`, `D[1:2,:]`, `D[2:3,:]`, `D[3:4,:]`

Send: Replica i's j-th split goes to replica j.

**Gather phase** (each replica concatenates along dim 1):
- Replica 0 receives: `A[0:1,:]`, `B[0:1,:]`, `C[0:1,:]`, `D[0:1,:]` -> `f32[1,64]`
- Replica 1 receives: `A[1:2,:]`, `B[1:2,:]`, `C[1:2,:]`, `D[1:2,:]` -> `f32[1,64]`
- Replica 2 receives: `A[2:3,:]`, `B[2:3,:]`, `C[2:3,:]`, `D[2:3,:]` -> `f32[1,64]`
- Replica 3 receives: `A[3:4,:]`, `B[3:4,:]`, `C[3:4,:]`, `D[3:4,:]` -> `f32[1,64]`

#### HLO Text Representation

```
%all_to_all = f32[1,64]{1,0} all-to-all(f32[4,16]{1,0} %operand),
  split_dimension=0, concat_dimension=1, split_count=4,
  replica_groups={{0,1,2,3}}
```

### Example 2: StableHLO with 2 Replicas

Two replicas, each with `f32[2, 4]`:

- `split_dimension = 1` (split along dim 1, size 4, into 2 parts of size 2)
- `concat_dimension = 0` (concatenate along dim 0)
- `split_count = 2`

- **Replica 0** operand:
  ```
  [[1, 2, 3, 4],
   [5, 6, 7, 8]]
  ```
- **Replica 1** operand:
  ```
  [[9, 10, 11, 12],
   [13, 14, 15, 16]]
  ```

After scatter (split along dim 1):
- Replica 0 sends `[:, 0:2]` = `[[1,2],[5,6]]` to replica 0, `[:, 2:4]` = `[[3,4],[7,8]]` to replica 1
- Replica 1 sends `[:, 0:2]` = `[[9,10],[13,14]]` to replica 0, `[:, 2:4]` = `[[11,12],[15,16]]` to replica 1

After gather (concat along dim 0):
- **Replica 0** output `f32[4, 2]`:
  ```
  [[1, 2],
   [5, 6],
   [9, 10],
   [13, 14]]
  ```
- **Replica 1** output `f32[4, 2]`:
  ```
  [[3, 4],
   [7, 8],
   [11, 12],
   [15, 16]]
  ```

### Constraints

- `split_dimension` of the operand must be divisible by `split_count`.
- `split_count` must equal the size of each replica group.
- All replica groups must have the same size.
- Each replica ID must appear in exactly one group.

---

## RaggedAllToAll

`RaggedAllToAll` extends `AllToAll` to support **ragged (variable-sized) tensors**, where the data contributed by each replica may have different sizes along the split dimension.

### Signature

```
RaggedAllToAll(data, offsets, sizes, split_dimension, concat_dimension,
               replica_groups, layout, channel_id, use_global_device_ids)
```

### Arguments

| Argument | Type | Description |
|---|---|---|
| `data` | `XlaOp` | The input data tensor. The concatenation of all shards from all replicas. |
| `offsets` | `XlaOp` | A 1D integer tensor of offsets specifying where each replica's data begins in the `data` tensor along `split_dimension`. Shape: `[num_replicas_in_group]`. |
| `sizes` | `XlaOp` | A 1D integer tensor specifying the size (along `split_dimension`) of each replica's contribution. Shape: `[num_replicas_in_group]`. |
| `split_dimension` | `int64` | Dimension along which data is split and redistributed. |
| `concat_dimension` | `int64` | Dimension along which received data is concatenated. |
| `replica_groups` | `std::vector<std::vector<int64>>` | Groups of replicas participating. |
| `layout` | `std::optional<Layout>` | Optional output layout. |
| `channel_id` | `std::optional<int64>` | Optional channel ID. |
| `use_global_device_ids` | `bool` | Whether to use global device IDs. |

### Semantics

Unlike the regular `AllToAll` where each replica contributes equally-sized data, `RaggedAllToAll` allows variable-sized contributions. The `offsets` and `sizes` tensors describe the layout of data from each replica within the concatenated `data` tensor.

For replica `j` in the group:
- Its data begins at `offsets[j]` along `split_dimension`
- Its data has size `sizes[j]` along `split_dimension`

This is particularly useful for workloads with variable-length sequences (e.g., natural language processing with variable token counts per device).

### Data / Offsets / Sizes Relationship

Given `N` replicas in a group:

```
offsets = [0, s0, s0+s1, s0+s1+s2, ...]  // cumulative sizes
sizes   = [s0, s1, s2, ..., s_{N-1}]      // per-replica sizes
```

The `data` tensor's `split_dimension` size equals `sum(sizes)`. Each replica's contribution occupies a contiguous slice of `data`.

### Example

2 replicas with ragged data:

- Replica 0 contributes 3 elements along dim 0: `offsets[0]=0, sizes[0]=3`
- Replica 1 contributes 5 elements along dim 0: `offsets[1]=3, sizes[1]=5`

Data tensor (shape `f32[8, 4]`): all data from both replicas concatenated.

After `RaggedAllToAll` with `split_dimension=0, concat_dimension=0`:
- Data is re-partitioned between replicas according to the ragged structure, potentially with a different size distribution after redistribution.

---

## CollectiveBroadcast

`CollectiveBroadcast` sends the operand from a single source replica to all other replicas in a replica group. It is essentially a one-to-all broadcast operation.

### Signature

```
CollectiveBroadcast(operand, replica_groups, channel_id,
                    use_global_device_ids)
```

### Arguments

| Argument | Type | Description |
|---|---|---|
| `operand` | `XlaOp` | The input tensor. Only the value from the source replica is used. |
| `replica_groups` | `std::vector<std::vector<int64>>` | Groups of replicas. The **first** replica ID in each inner vector is the source; all others are targets. |
| `channel_id` | `std::optional<int64>` | Optional channel ID for cross-module matching. |
| `use_global_device_ids` | `bool` | Whether replica IDs are global device IDs. Default `false`. |

### Semantics

Within each replica group, the first replica listed acts as the source. Its operand value is sent to all other replicas in the group. After the operation completes, every replica in the group holds the same tensor value (the value from the source replica).

**Output shape**: Identical to the input shape.

### Example

With `replica_groups = {{0, 1, 2, 3}}`:
- Replica 0's operand `[5.0, 10.0]` is broadcast to replicas 1, 2, and 3.
- All four replicas now hold `[5.0, 10.0]`.

#### HLO Text Representation

```
%broadcast = f32[2]{0} collective-broadcast(f32[2]{0} %operand),
  replica_groups={{0,1,2,3}}
```

### Multiple Groups

```
replica_groups = {{0, 1}, {2, 3}}
```

Replica 0 broadcasts to replica 1; replica 2 broadcasts to replica 3. Each group operates independently.

### Constraints

- Each replica group must have at least one member (the source).
- The first replica ID in each group is always the source.
- All source and target replicas must participate (no partial groups).

---

## CollectivePermute

`CollectivePermute` is a peer-to-peer collective operation that sends data between pairs of replicas according to a specified mapping. It enables arbitrary data permutation patterns across devices.

### Signature

```
CollectivePermute(operand, source_target_pairs, channel_id,
                  layout, use_global_device_ids)
```

### Arguments

| Argument | Type | Description |
|---|---|---|
| `operand` | `XlaOp` | The input tensor to be permuted. |
| `source_target_pairs` | `std::vector<std::pair<int64, int64>>` | A list of `(source, target)` pairs. Each pair specifies that the operand from replica `source` is sent to replica `target`. |
| `channel_id` | `std::optional<int64>` | Optional channel ID for cross-module communication matching. |
| `layout` | `std::optional<Layout>` | Optional desired output layout. |
| `use_global_device_ids` | `bool` | Whether replica IDs are global device IDs. Default `false`. |

### Semantics

For each `(source, target)` pair in `source_target_pairs`:
- The operand from replica `source` is sent to replica `target`.

Each target replica concatenates all values it receives from different sources along the first dimension (dimension 0). If a replica receives no data, the output is an empty tensor of the appropriate shape.

A replica can appear as a source in multiple pairs (sending its data to multiple targets). A replica can appear as a target in multiple pairs (receiving data from multiple sources, which are concatenated). A replica can be both a source and a target simultaneously.

### Restrictions on Pairs

1. **No duplicate targets from the same source**: A given source replica cannot appear in two pairs with the same target. That is, all `(source, target)` pairs must be unique.

2. **Pairs must form valid communication**: The pairs must describe a pattern that can be realized by the interconnect. In practice, this means the system must support the required peer-to-peer transfers.

3. **Self-loops**: A pair like `(i, i)` is technically valid and means replica `i` sends data to itself. The behavior is that replica `i`'s operand is included in its own output (concatenated with any other incoming data).

### Example: Ring Shift

Shift data to the right in a ring of 4 replicas:

```
source_target_pairs = {(0, 1), (1, 2), (2, 3), (3, 0)}
```

- Replica 0 receives from replica 3: gets replica 3's operand
- Replica 1 receives from replica 0: gets replica 0's operand
- Replica 2 receives from replica 1: gets replica 1's operand
- Replica 3 receives from replica 2: gets replica 2's operand

Each replica holds a single value (one incoming source), so no concatenation occurs.

#### HLO Text Representation

```
%permute = f32[4]{0} collective-permute(f32[4]{0} %operand),
  source_target_pairs={{0,1},{1,2},{2,3},{3,0}}
```

### Example: All-to-All via Permute

```
source_target_pairs = {(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)}
```

Each replica sends to every other replica. Each replica receives from two sources, and the results are concatenated along dimension 0.

### Internal Decomposition

`CollectivePermute` decomposes into `CollectivePermuteStart` and `CollectivePermuteDone` for asynchronous execution:

1. **CollectivePermuteStart**: Initiates the peer-to-peer transfers.

   ```
   %target = collective-permute-start(f32[4] %operand),
     source_target_pairs={{0,1},{1,2},{2,3},{3,0}}
   ```

2. **CollectivePermuteDone**: Waits for completion and returns the result.

   ```
   %result = collective-permute-done(f32[4] %target)
   ```

### Common Use Cases

| Pattern | Pairs | Description |
|---|---|---|
| Ring shift right | `{(i, (i+1)%N)}` for all i | Pipeline parallelism |
| Ring shift left | `{((i+1)%N, i)}` for all i | Reverse pipeline |
| Neighbor exchange | `{(i, i-1), (i, i+1)}` | Halo exchange in stencil computations |
| Broadcast from 0 | `{(0, i)}` for i > 0 | Parameter server pattern |

---

## ReduceScatter

`ReduceScatter` combines `AllReduce` and a scatter operation. It reduces data across replicas (like `AllReduce`) but then scatters the result so each replica holds only a distinct shard of the reduced output. This is more communication-efficient than `AllReduce` when each replica only needs a portion of the result.

### Signature

```
ReduceScatter(operand, computation, scatter_dimension, shard_count,
              replica_groups, channel_id, layout, use_global_device_ids)
```

### Arguments

| Argument | Type | Description |
|---|---|---|
| `operand` | `XlaOp` | The input tensor to be reduced and scattered. |
| `computation` | `XlaComputation` | The reduction computation (must be associative and commutative). Takes two scalars, produces one scalar. |
| `scatter_dimension` | `int64` | The dimension along which the reduced result is split and scattered to replicas. Must be in `[0, rank)`. |
| `shard_count` | `int64` | Number of shards. Must equal the size of each replica group. `scatter_dimension` must be divisible by `shard_count`. |
| `replica_groups` | `std::vector<std::vector<int64>>` | Groups of replicas. See `AllGather` for detailed semantics. |
| `channel_id` | `std::optional<int64>` | Optional channel ID for cross-module matching. |
| `layout` | `std::optional<Layout>` | Optional desired output layout. |
| `use_global_device_ids` | `bool` | Whether replica IDs are global device IDs. Default `false`. |

### Semantics

`ReduceScatter` performs two logical steps:

1. **Reduce**: Perform an `AllReduce` across all replicas in the group using `computation`. All replicas compute the same fully-reduced tensor.

2. **Scatter**: Split the reduced result along `scatter_dimension` into `shard_count` equal parts. Replica `i` (the i-th replica in the group) receives the i-th shard.

**Output shape**:
```
output_shape = input_shape
output_shape[scatter_dimension] = input_shape[scatter_dimension] / shard_count
```

The output is a shard of the reduced result. Each replica in the group receives a different shard.

### Example: 4 Replicas

Four replicas, each holding `f32[8]`:

- **Replica 0**: `[1, 2, 3, 4, 5, 6, 7, 8]`
- **Replica 1**: `[2, 3, 4, 5, 6, 7, 8, 9]`
- **Replica 2**: `[3, 4, 5, 6, 7, 8, 9, 10]`
- **Replica 3**: `[4, 5, 6, 7, 8, 9, 10, 11]`

With `scatter_dimension=0, shard_count=4, computation=add`:

**Step 1 (Reduce)**: Sum across replicas:
```
[10, 14, 18, 22, 26, 30, 34, 38]
```

**Step 2 (Scatter)**: Split into 4 parts of size 2:
- Replica 0 gets `[10, 14]`
- Replica 1 gets `[18, 22]`
- Replica 2 gets `[26, 30]`
- Replica 3 gets `[34, 38]`

#### HLO Text Representation

```
%reduce_scatter = f32[2]{0} reduce-scatter(f32[8]{0} %operand),
  to_apply=%add, scatter_dimension=0, shard_count=4,
  replica_groups={{0,1,2,3}}
```

### Relationship to AllReduce and AllGather

`ReduceScatter` is complementary to `AllGather`:

- `ReduceScatter` followed by `AllGather` on the same `scatter_dimension`/`all_gather_dimension` and matching `shard_count` is equivalent to `AllReduce`.
- This decomposition is often used in practice (e.g., in NCCL's ring-based algorithms) because `ReduceScatter + AllGather` can be more bandwidth-efficient than a direct `AllReduce`.

### Internal Decomposition

`ReduceScatter` decomposes into `ReduceScatterStart` and `ReduceScatterDone`:

```
%target = reduce-scatter-start(f32[8] %operand),
  to_apply=%add, scatter_dimension=0, shard_count=4,
  replica_groups={{0,1,2,3}}

%result = reduce-scatter-done(f32[2] %target)
```

---

## Common Patterns

### Pattern 1: Data-Parallel Gradient Aggregation

```python
# Each replica computes gradients locally
local_gradients = compute_gradients(model, data_shard)

# Sum gradients across all replicas
global_gradients = AllReduce(
    local_gradients,
    computation=add,
    replica_groups={{}}  # all replicas
)

# Update model with aggregated gradients
model = apply_updates(model, global_gradients)
```

### Pattern 2: Tensor-Parallel Column Parallelism

```python
# Split weight matrix columns across replicas
local_weight = weight_shard  # [hidden, hidden/N]

# Each replica computes its portion of the output
local_output = dot(input, local_weight)  # [batch, seq, hidden/N]

# AllGather to reconstruct full output
full_output = AllGather(
    local_output,
    all_gather_dimension=-1,  # gather along feature dim
    shard_count=N
)
```

### Pattern 3: Sequence-Parallel ReduceScatter + AllGather

```python
# After layer norm (computed in sequence-parallel mode)
# ReduceScatter to get local shard of the sum
local_sum = ReduceScatter(
    tensor,
    computation=add,
    scatter_dimension=1,  # sequence dim
    shard_count=N
)

# ... compute with local shard ...

# AllGather to reconstruct full tensor
full_tensor = AllGather(
    local_result,
    all_gather_dimension=1,
    shard_count=N
)
```

### Pattern 4: Pipeline Parallelism with CollectivePermute

```python
# Stage 0 on device 0, Stage 1 on device 1, etc.
# Forward pass: send activations to next stage
activations = CollectivePermute(
    stage_output,
    source_target_pairs={(0, 1), (1, 2), (2, 3)}
)

# Backward pass: send gradients to previous stage
gradients = CollectivePermute(
    grad_output,
    source_target_pairs={(3, 2), (2, 1), (1, 0)}
)
```

### Pattern 5: Expert Parallelism with AllToAll

```python
# In MoE (Mixture of Experts): tokens are dispatched to experts
# AllToAll redistributes tokens from "dispatched by token" to "dispatched by expert"
expert_input = AllToAll(
    dispatched_tokens,  # [num_experts, tokens_per_expert, hidden]
    split_dimension=1,   # split tokens
    concat_dimension=0,  # concatenate into expert groups
    split_count=num_experts
)
```

---

## StableHLO Cross-References

Most XLA collective operations have direct counterparts in the StableHLO opset:

| XLA Operation | StableHLO Operation | Notes |
|---|---|---|
| AllGather | `stablehlo.all_gather` | Direct correspondence; same semantics |
| AllReduce | `stablehlo.all_reduce` | Direct correspondence; same semantics |
| AllToAll | `stablehlo.all_to_all` | Direct correspondence; same semantics |
| CollectiveBroadcast | `stablehlo.collective_broadcast` | Direct correspondence |
| CollectivePermute | `stablehlo.collective_permute` | Direct correspondence |
| ReduceScatter | `stablehlo.reduce_scatter` | Direct correspondence |

### StableHLO-Specific Notes

1. **Channel Handle**: StableHLO uses `channel_id` as an attribute (`si64`), with `0` representing no channel (local within module). Positive values represent cross-module channels.

2. **Replica Groups**: In StableHLO, `replica_groups` is a 2D tensor of `si64` values. An empty `replica_groups` means all replicas participate as one group.

3. **Use Global Device IDs**: StableHLO models this as a boolean attribute `use_global_device_ids`. When `true`, the IDs in `replica_groups` refer to global device IDs assigned by the runtime.

4. **AllReduce Computation**: In StableHLO, the `computation` is a region that takes two tensors and returns one tensor, applying the reduction element-wise. The region must implement an associative and commutative binary operation.

### StableHLO Example: AllReduce

```mlir
%result = stablehlo.all_reduce(%operand) ({
  ^bb0(%x: tensor<f32>, %y: tensor<f32>):
    %sum = stablehlo.add %x, %y : tensor<f32>
    stablehlo.return %sum : tensor<f32>
}) {
  replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
  channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>,
  use_global_device_ids = false
} : (tensor<2xf32>) -> tensor<2xf32>
```

### StableHLO Example: AllToAll

```mlir
%result = stablehlo.all_to_all(%operand) {
  split_dimension = 1 : i64,
  concat_dimension = 0 : i64,
  split_count = 2 : i64,
  replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
  channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
} : (tensor<2x4xf32>) -> tensor<4x2xf32>
```

### StableHLO Example: CollectivePermute

```mlir
%result = stablehlo.collective_permute(%operand) {
  source_target_pairs = dense<[[0, 1], [1, 2], [2, 3], [3, 0]]> : tensor<4x2xi64>,
  channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
} : (tensor<4xf32>) -> tensor<4xf32>
```

---

## Appendix: Collective Operation Comparison Table

| Operation | Communication Pattern | Output Size per Replica | Key Use Case |
|---|---|---|---|
| AllGather | All-to-all (concatenate) | Larger than input | Reassembling sharded tensors |
| AllReduce | All-to-all (reduce) | Same as input | Gradient aggregation |
| AllToAll | All-to-all (repartition) | Potentially different | Expert parallelism, resharding |
| CollectiveBroadcast | One-to-all | Same as input | Parameter distribution |
| CollectivePermute | Peer-to-peer | Same or larger | Pipeline parallelism, halo exchange |
| ReduceScatter | All-to-all (reduce + scatter) | Smaller than input | Bandwidth-efficient reduction |
| RaggedAllToAll | All-to-all (ragged) | Potentially different | Variable-length sequences |

### Bandwidth and Latency Characteristics

For `N` replicas and a tensor of size `M` elements:

| Operation | Data Transferred per Replica | Time Complexity (ring) |
|---|---|---|
| AllGather | `O(M * (N-1)/N)` | `O(M * (N-1)/N * latency + M * bandwidth)` |
| AllReduce | `O(M * 2*(N-1)/N)` | `O(M * 2*(N-1)/N * latency + M * bandwidth)` |
| ReduceScatter | `O(M * (N-1)/N)` | `O(M * (N-1)/N * latency + M * bandwidth)` |
| AllToAll | `O(M * (N-1)/N)` | `O(M/N * (N-1) * latency + M * bandwidth)` |
| CollectiveBroadcast | `O(M)` | `O(M * log(N) * latency + M * bandwidth)` |
| CollectivePermute | `O(M)` per pair | Depends on pattern |
