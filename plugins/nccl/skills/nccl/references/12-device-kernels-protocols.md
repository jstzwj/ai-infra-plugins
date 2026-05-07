# 12 - Device Kernels and Protocol Primitives

## Primary source files

- `sources/nccl/src/device/common.cu`: generic device kernel entry.
- `sources/nccl/src/device/common_kernel.h`: kernel execution framework and `RunWork*` dispatch.
- `sources/nccl/src/device/primitives.h`: high-level protocol primitive abstraction.
- `sources/nccl/src/device/prims_simple.h`: SIMPLE protocol primitives.
- `sources/nccl/src/device/prims_ll.h`: LL protocol primitives.
- `sources/nccl/src/device/prims_ll128.h`: LL128 protocol primitives.
- `sources/nccl/src/device/all_reduce.h`
- `sources/nccl/src/device/broadcast.h`
- `sources/nccl/src/device/reduce.h`
- `sources/nccl/src/device/reduce_scatter.h`
- `sources/nccl/src/device/all_gather.h`
- `sources/nccl/src/device/all_gather_v.h`
- `sources/nccl/src/device/sendrecv.h`
- `sources/nccl/src/device/reduce_kernel.h`
- `sources/nccl/src/device/generate.py`: generated kernel/header support.
- `sources/nccl/src/include/device.h`: device-side structures and constants.

## Device kernel role

Host enqueue builds work descriptors and launches NCCL device kernels. The device kernel reads uploaded
work batches and executes collective/P2P/RMA algorithms using protocol primitives over channel connectors.

Conceptual flow:

```text
ncclKernelPlan on host
  -> ncclDevKernelArgs / work FIFO / work batches
  -> ncclDevKernel_Generic or specialized kernel
  -> RunWorkBatch / RunWorkColl / RunWorkP2p
  -> algorithm-specific device header
  -> protocol primitive send/recv/reduce/copy steps
  -> transport connector buffers / peer memory / network proxy-visible FIFOs
```

## Kernel specialization dimensions

Device code specializes or dispatches by:

- collective function,
- datatype,
- reduction op,
- algorithm,
- protocol,
- number of channels,
- work type,
- registered/window/symmetric variants,
- specialized kernels vs generic kernel.

This is why adding a new operation or datatype may require updates in generator scripts, enum mapping,
reduce op support, and multiple device headers.

## Protocols

### SIMPLE

SIMPLE is the bandwidth-oriented protocol. It tends to be used for larger messages where throughput
matters more than per-message latency. Source file: `device/prims_simple.h`.

### LL

LL is a low-latency protocol for small messages. Source file: `device/prims_ll.h`.

### LL128

LL128 is another low-latency protocol using 128-bit-oriented behavior. Source file:
`device/prims_ll128.h`.

The public tuning layer exposes protocol names `LL`, `LL128`, and `SIMPLE`. Internally, protocol choice
affects primitive layout, synchronization, step size, and proxy/network behavior.

## Primitive abstraction

`primitives.h` provides the abstraction used by collective implementations. Depending on algorithm and
protocol, primitives perform combinations of:

- send,
- receive,
- copy,
- reduce,
- reduce and send,
- receive and reduce,
- direct peer reads/writes,
- synchronization through step counters/FIFOs.

The primitive layer hides many details of connector state and protocol-specific buffering from collective
algorithm code.

## Collective device implementations

### AllReduce

`device/all_reduce.h` implements allreduce variants over ring/tree/NVLS/CollNet-style algorithms. It
combines reduce and broadcast/distribution phases depending on algorithm.

Typical algorithmic shapes:

- Ring allreduce: reduce-scatter phase + allgather phase over ring channels.
- Tree allreduce: reduction up tree + broadcast down tree.
- NVLS/CollNet paths: use topology-specific local/network acceleration.

### Broadcast

`device/broadcast.h` copies root data to all ranks, using ring/tree or other selected structures.
Root is a rank, not CUDA device ID.

### Reduce

`device/reduce.h` reduces all rank inputs into root output. Non-root `recvbuff` may be unused/null for
host API semantics, but device work still follows topology paths.

### ReduceScatter

`device/reduce_scatter.h` reduces then scatters blocks to ranks. In-place semantics depend on rank offset.

### AllGather / AllGatherV

`device/all_gather.h` gathers fixed-size blocks from all ranks. `all_gather_v.h` handles variable-size or
specialized allgatherv patterns.

### SendRecv

`device/sendrecv.h` handles P2P send/recv work. Because P2P operations can be blocking for the GPU, host
grouping must provide all operations needed for concurrent progress.

## Work descriptors and batches

The device kernel does not receive a C++ object graph. It receives compact work descriptors uploaded by
host enqueue. Structures include work batch headers and operation-specific payloads such as collective or
P2P descriptors.

When editing work descriptors:

1. Update host writer in `enqueue.cc`.
2. Update device reader in `device/common_kernel.h` or operation headers.
3. Check alignment and size assumptions.
4. Check CUDA graph persistent storage behavior.
5. Update profiler/device event metadata if needed.

## Channel connectors

Each channel has peer send/recv connectors. Device code uses connector state to coordinate with peer GPUs
or proxy/network paths. Connector memory includes cache-line-aligned head/tail and FIFOs (`ncclSendMem`,
`ncclRecvMem` in `comm.h`).

Important concepts:

- Steps are tracked with head/tail counters.
- FIFOs carry offsets/sizes/connection info.
- Some protocols embed completion flags in data.
- Network paths may rely on proxy threads reading/writing connector state.

## Reduction operations on device

`collectives.cc` maps `ncclRedOp_t` to device reduction operation strings. Device reduction support must
handle:

- built-in sum/prod/min/max/avg,
- dynamic pre-multiply sum,
- datatype-specific behavior,
- scalar residence for dynamic ops,
- possible post-division for average.

If adding a datatype or operation, update both host string/type mapping and device reduce/copy logic.

## Generated code

`device/generate.py` and build rules generate or assemble device kernels/headers. NCCL uses generation to
cover cross-products of function, datatype, reduction op, algorithm, and protocol without hand-writing
every specialization.

When a symbol seems missing, inspect generated build outputs as well as static source headers.

## One-rank path

`device/onerank.cu` handles degenerate single-rank cases. These can bypass normal communication and are
important for correctness tests with one GPU/rank.

## Device kernel debugging

1. Verify host selected the expected algorithm/protocol in `TUNING` logs.
2. Verify work descriptor was uploaded and channel mask is nonzero.
3. Check whether the generic or specialized kernel was launched.
4. Check proxy ops if the protocol/transport requires proxy progress.
5. Use CUDA tools to find whether the kernel is running, stalled, or not launched.
6. For custom Device API kernels, distinguish NCCL-owned kernels from user kernels using `nccl_device`.

## Common source-change traps

| Change | Required follow-up |
|---|---|
| add collective enum | update string conversion, public wrapper, enqueue task formation, device dispatch |
| add datatype | update `ncclDatatypeToString`, type size logic, reduce/copy primitives, generated kernels |
| change work descriptor | update host upload and device reader together |
| change protocol constants | update tuning thresholds, primitive assumptions, proxy behavior |
| alter connector layout | update host and device copies, alignment, transport setup |
| add profiler event | update descriptors, activation masks, event start/stop paths |

## Relationship to Device API

Do not confuse NCCL's internal device kernels with the public Device API. Internal kernels execute NCCL
host-enqueued work. Device API lets user kernels perform communication using `ncclDevComm_t`, teams,
barriers, windows, LSA, and GIN. They share some lower-level concepts but have different entry points and
lifetime rules.
