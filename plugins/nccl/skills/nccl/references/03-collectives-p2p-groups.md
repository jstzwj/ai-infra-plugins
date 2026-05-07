# 03 - Collectives, Point-to-Point Operations, and Group Semantics

## Primary source files

- `sources/nccl/src/nccl.h.in`: public operation semantics and prototypes.
- `sources/nccl/src/collectives.cc`: API wrappers that create `ncclInfo` and call `ncclEnqueueCheck`.
- `sources/nccl/src/include/info.h`: normalized API-call descriptor.
- `sources/nccl/src/group.cc`: group start/end, async jobs, grouped launch.
- `sources/nccl/docs/examples/02_point_to_point/*`: P2P ring example.
- `sources/nccl/docs/examples/03_collectives/*`: AllReduce example.

## Enqueue semantics

NCCL collective and P2P API calls generally return after work is **enqueued on a CUDA stream**, not
after communication is complete. Host code must synchronize the relevant CUDA stream or use CUDA events
when it needs the data.

This distinction matters for:

- Correctness checks after collectives.
- Buffer lifetime and reuse.
- Destroy/finalize timing.
- CUDA graph capture.
- Async error handling.

## Collective operation table

| API | Data movement | Reduction op? | Root? | In-place rule |
|---|---|---:|---:|---|
| `ncclReduce` | all ranks reduce into root recv buffer | yes | yes | `sendbuff == recvbuff` |
| `ncclBroadcast` | root data copied to every rank | no | yes | `sendbuff == recvbuff` |
| `ncclBcast` | deprecated in-place broadcast | no | yes | implicitly in-place |
| `ncclAllReduce` | reduce and distribute result to all ranks | yes | no | `sendbuff == recvbuff` |
| `ncclReduceScatter` | reduce then scatter equal blocks | yes | no | `recvbuff == sendbuff + rank * recvcount` |
| `ncclAllGather` | gather equal blocks from all ranks | no | no | `sendbuff == recvbuff + rank * sendcount` |
| `ncclAlltoAll` | each rank sends equal count to every rank | no | no | layout-dependent |
| `ncclGather` | every rank sends equal count to root | no | yes | root: `sendbuff == recvbuff + root * count` |
| `ncclScatter` | root sends equal count to every rank | no | yes | root: `recvbuff == sendbuff + root * count` |

The `root` parameter is always a **rank**, not a CUDA device ID.

## Data types

`ncclDataType_t` values from the public header:

| Type | Aliases/notes |
|---|---|
| `ncclInt8` | `ncclChar` |
| `ncclUint8` | unsigned 8-bit |
| `ncclInt32` | `ncclInt` |
| `ncclUint32` | unsigned 32-bit |
| `ncclInt64` | signed 64-bit |
| `ncclUint64` | unsigned 64-bit |
| `ncclFloat16` | `ncclHalf` |
| `ncclFloat32` | `ncclFloat` |
| `ncclFloat64` | `ncclDouble` |
| `ncclBfloat16` | bfloat16 |
| `ncclFloat8e4m3` | FP8 e4m3 |
| `ncclFloat8e5m2` | FP8 e5m2 |

## Reduction operations

Built-in `ncclRedOp_t` values:

| Op | Meaning |
|---|---|
| `ncclSum` | sum |
| `ncclProd` | product |
| `ncclMax` | maximum |
| `ncclMin` | minimum |
| `ncclAvg` | average |

Dynamic reduction operation support:

```c
ncclRedOpCreatePreMulSum(&op, scalar, datatype, residence, comm);
ncclRedOpDestroy(op, comm);
```

`ncclScalarResidence_t` controls when the scalar is read:

| Residence | Meaning |
|---|---|
| `ncclScalarDevice` | scalar is in device-visible memory and read while the collective runs |
| `ncclScalarHostImmediate` | scalar is read from host-visible memory before create returns |

## `ncclInfo`: how public calls enter internals

`collectives.cc` turns each API call into an `ncclInfo` record and calls `ncclEnqueueCheck`.
Examples:

- `ncclAllReduce` sets `func=ncclFuncAllReduce`, operation name `"AllReduce"`, datatype, op, count,
  communicator, stream, and allreduce chunk/slice steps.
- `ncclSend` and `ncclRecv` set `func=ncclFuncSend` / `ncclFuncRecv`, peer in `root`, and `chunkSteps=1`.
- RMA/signal APIs set peer window, signal index, context, flags, or wait descriptors.

This is important for source debugging: public API bugs usually route through:

```text
nccl.h.in declaration
  -> collectives.cc wrapper
  -> ncclEnqueueCheck(info)
  -> taskAppend / scheduling in enqueue.cc
```

## Group semantics

### Why groups exist

The public header explains two major reasons:

1. When one host thread manages multiple GPUs, NCCL calls for different ranks may need inter-CPU
   synchronization. Grouping lets them be submitted as one operation.
2. Grouping fuses multiple operations to improve performance or allow concurrent progress, especially
   for multiple send/recv operations that would otherwise deadlock.

### API

```c
ncclGroupStart();
// NCCL calls only; no dependent CUDA work between start/end.
ncclGroupEnd();
```

`ncclGroupStart` queues NCCL calls until `ncclGroupEnd`. Nothing starts on the CUDA stream until group
end. `ncclGroupEnd` starts the fused operation and returns when operations have been enqueued, not when
they have completed on device.

### Simulate end

```c
ncclSimInfo_t sim = NCCL_SIM_INFO_INITIALIZER;
ncclGroupSimulateEnd(&sim);
printf("estimated time = %f\n", sim.estimatedTime);
```

Use simulation for planning/introspection, not as a replacement for benchmarking.

## Collective call ordering

All ranks in a communicator clique must call matching collective operations in compatible order. Mismatched
operation order, datatype, count, root, or participation usually causes hangs or asynchronous errors.

For example, if rank 0 calls AllReduce then Broadcast while rank 1 calls Broadcast then AllReduce, both
ranks can wait forever because the device kernels and proxy work are not matching the same operation
sequence.

## Single-process multi-GPU collective pattern

```cpp
ncclGroupStart();
for (int r = 0; r < nranks; ++r) {
  cudaSetDevice(devices[r]);
  ncclAllReduce(send[r], recv[r], count, ncclFloat32, ncclSum, comms[r], streams[r]);
}
ncclGroupEnd();

for (int r = 0; r < nranks; ++r) {
  cudaSetDevice(devices[r]);
  cudaStreamSynchronize(streams[r]);
}
```

Use this shape when one host thread loops over devices.

## P2P operations

### `ncclSend`

```c
ncclSend(sendbuff, count, datatype, peer, comm, stream);
```

Sends data from this rank to `peer`. The peer must call `ncclRecv` with the same datatype and count
from this rank.

### `ncclRecv`

```c
ncclRecv(recvbuff, count, datatype, peer, comm, stream);
```

Receives data from `peer`. The peer must call matching `ncclSend`.

### P2P blocking rule

The public header states that P2P operations are blocking for the GPU. If multiple sends and receives
must progress concurrently to complete, they must be fused within `ncclGroupStart` / `ncclGroupEnd`.

### Ring P2P example shape

```cpp
int prev = (rank - 1 + nranks) % nranks;
int next = (rank + 1) % nranks;

ncclGroupStart();
ncclSend(sendbuf, count, ncclFloat32, next, comm, stream);
ncclRecv(recvbuf, count, ncclFloat32, prev, comm, stream);
ncclGroupEnd();
```

In a single process managing all GPUs, the group must include all local ranks' sends and recvs.

## AllReduce example logic

The in-repo collective example uses a simple verification pattern:

1. Create one communicator per GPU with `ncclCommInitAll`.
2. Allocate per-GPU send/recv buffers and streams.
3. Fill send buffer with rank-derived values.
4. Group `ncclAllReduce(..., ncclSum, ...)` across ranks.
5. Synchronize streams.
6. Verify each rank received the expected global sum.
7. Finalize/destroy comms and free buffers.

This is a good template for correctness examples but not for measuring peak performance.

## Rooted collectives: reduce, broadcast, gather, scatter

When using rooted collectives, keep two namespaces separate:

- `root` is a communicator rank.
- CUDA device IDs are chosen by the application.

This is a common bug in MPI applications where `localRank` and global rank differ. Use
`ncclCommUserRank` or application rank mapping when in doubt.

## Count and layout rules

### AllGather

Each rank sends `sendcount` elements. `recvbuff` must contain at least `nranks * sendcount` elements.
Data from rank `i` is at `recvbuff + i * sendcount`.

### ReduceScatter

Each rank receives `recvcount` elements. `sendbuff` must contain at least `nranks * recvcount` elements.
The reduced result block for rank `i` goes to rank `i`.

### AlltoAll

Each rank sends `count` elements to every other rank. Data for destination rank `j` is read from
`sendbuff + j * count`; data received from source rank `i` is written to `recvbuff + i * count`.

### Gather

Each rank sends `count` elements to `root`. On root, data from rank `i` is placed at
`recvbuff + i * count`. Non-root `recvbuff` is unused.

### Scatter

Root sends `count` elements to every rank. Root reads rank `i`'s data at `sendbuff + i * count`.
Non-root `sendbuff` is unused.

## CUDA graph capture considerations

NCCL tracks graph capture in the enqueue planner (`planner->capturingGraph`). Grouped calls, captured
streams, persistent work buffers, and destructors have special handling. When answering graph-capture
questions:

- All streams in a grouped NCCL operation need compatible capture state.
- NCCL may record kernel launches rather than launch immediately during graph capture.
- Persistent plans and registration cleanup differ from normal launches.
- Debug with `NCCL_DEBUG=INFO` and check whether the user is mixing captured and non-captured streams.

## Practical deadlock checklist

For a hang involving collectives/P2P:

1. Does every rank call the same operation sequence?
2. Are `count`, `datatype`, `root`, and `peer` values compatible?
3. Are all ranks in the communicator participating?
4. Are multi-device single-thread calls wrapped in `ncclGroupStart/End`?
5. Are concurrent P2P sends/recvs grouped?
6. Is CUDA stream work before the NCCL operation blocking progress?
7. Did any rank return an immediate error that the application ignored?
8. What is the last `NCCL_DEBUG=INFO` line per rank?
