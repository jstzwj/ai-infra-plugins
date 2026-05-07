# 11 - Group, Enqueue, Scheduler, and Launch Pipeline

## Primary source files

- `sources/nccl/src/group.cc`: group nesting, async jobs, launch grouping.
- `sources/nccl/src/include/group.h`: group helper declarations/inlines.
- `sources/nccl/src/enqueue.cc`: central task and launch pipeline.
- `sources/nccl/src/include/enqueue.h`: launch phases and planner helpers.
- `sources/nccl/src/include/info.h`: `ncclInfo` API-call descriptor.
- `sources/nccl/src/include/comm.h`: task/planner/plan structures.
- `sources/nccl/src/scheduler/symmetric_sched.cc`: symmetric collective scheduler.
- `sources/nccl/src/scheduler/allgatherv_sched.cc`: allgatherv scheduler.
- `sources/nccl/src/include/scheduler.h`: scheduler extension declarations.

## High-level pipeline

```text
Public API wrapper
  -> ncclInfo
  -> ncclEnqueueCheck
  -> taskAppend
  -> planner accumulation
  -> ncclPrepareTasks
  -> ncclLaunchPrepare
  -> scheduleCollTasksToPlan / scheduleP2pTasksToPlan / RMA/CE/symmetric schedulers
  -> uploadWork / uploadProxyOps
  -> ncclLaunchKernel
  -> ncclLaunchFinish
```

This pipeline is the heart of NCCL operation execution.

## Group layer

### Nested groups

`ncclGroupStart` increments group depth. `ncclGroupEnd` decrements it and only triggers launch when the
outermost group ends. This allows nested library code to participate in a user's group without launching
prematurely.

### Group state responsibilities

The group layer coordinates:

- accumulated communicators in the group,
- async jobs such as communicator init/mutation,
- operation launch ordering,
- preconnect/setup work,
- blocking/nonblocking behavior,
- cleanup when group launch fails.

### `groupLaunch`

The internals survey identifies `groupLaunch` as ordering:

1. communicator preparation,
2. preconnect,
3. launch planning,
4. cleanup across all communicators in the group.

This means grouping is not just API batching; it is also where NCCL aligns transport setup and kernel
launch across ranks/devices.

## `ncclInfo`

`ncclInfo` is the normalized representation of a public API call. It carries:

- function enum,
- name string,
- send/recv buffer pointers,
- count,
- datatype,
- reduction op,
- root/peer,
- communicator,
- CUDA stream,
- chunk/slice steps,
- optional RMA/window/signal fields.

`ncclEnqueueCheck` validates and appends it to the communicator planner.

## Task append

`taskAppend` in `enqueue.cc` converts `ncclInfo` into one of:

- `ncclTaskColl` for collectives,
- `ncclTaskP2p` for send/recv,
- `ncclTaskRma` for RMA/signal/wait,
- specialized task variants when needed.

Task append is where API-level metadata becomes scheduler-level work.

Important transformations:

- compute bytes from count/datatype,
- convert host reduction op to device reduction op,
- attach profiler handles,
- detect stream/graph capture state,
- track buffer registration/window info,
- queue by peer for P2P.

## Task preparation

`ncclPrepareTasks` aggregates and sorts tasks before launch planning. It separates work into categories:

| Queue/category | Meaning |
|---|---|
| normal collective tasks | standard collective kernel path |
| CollNet/NVLS tasks | algorithm-specific paths |
| CE collectives | copy-engine or specialized collectives where supported |
| symmetric tasks | symmetric memory/window optimized collectives |
| RMA tasks | one-sided/signal operations |
| P2P queues | per-peer send/recv work |

Collective tasks are roughly sorted by descending size using `ncclTaskCollSorter`, which reduces poor
packing when operations of different sizes are grouped.

## Launch preparation

`ncclLaunchPrepare` builds one or more `ncclKernelPlan` objects subject to budgets such as:

- work FIFO bytes,
- argument bytes,
- channel masks,
- proxy op queues,
- CUDA graph capture persistence,
- specialized kernel availability,
- symmetric/RMA/CE constraints.

A large group can become multiple kernel plans.

## Scheduling collective tasks

`scheduleCollTasksToPlan` places collective work onto channels and builds device work descriptors.
Inputs include:

- chosen algorithm,
- chosen protocol,
- number of channels,
- chunk/slice steps,
- registration/window mode,
- channel topology (ring/tree/NVLS/CollNet),
- work budget.

The result is a plan with work entries and, for network-assisted paths, proxy operations.

## Scheduling P2P tasks

`scheduleP2pTasksToPlan` pairs send/recv tasks, chooses channel(s), and creates work batches. P2P scheduling
has to respect:

- peer ordering,
- operation counters/epochs,
- concurrent progress requirements,
- LL threshold behavior,
- channel budget,
- proxy requirements for network paths.

Group semantics are essential because scheduling can only see and match the P2P operations that have
been accumulated.

## Upload work

`uploadWork` moves host-side work descriptors into device-visible storage. Depending on the plan, storage
can be FIFO/persistent/captured. `uploadProxyOps` prepares proxy progress descriptors for CPU/network work.

Important concerns:

- CUDA graph capture changes lifetime and persistence.
- Device work descriptors must match what `device/common_kernel.h` expects.
- Proxy operations must be visible to proxy threads before/around kernel execution.
- Cleanup callbacks reclaim host and device allocations after safe completion.

## Kernel launch

`ncclLaunchKernel` launches the selected NCCL kernel or records it into a CUDA graph capture. Launch
parameters come from the plan:

- kernel function pointer,
- kernel args pointer/size,
- dynamic shared memory,
- thread-per-block count,
- channel mask,
- target CUDA stream,
- mem sync domain and launch attributes.

`ncclLaunchFinish` handles post-launch bookkeeping and cleanup registration.

## CUDA graph capture

The planner tracks a `capturingGraph`. NCCL restricts grouped operations so all involved streams are
compatible with the same capture state. In capture mode:

- work buffers may become persistent,
- destructors/callbacks differ,
- kernel launch event means "recorded" rather than actually executed,
- registration cleanup must not invalidate graph replay.

If graph capture fails, inspect both stream capture state and whether the NCCL path requires unsupported
runtime allocation/registration.

## Symmetric scheduler

`scheduler/symmetric_sched.cc` groups compatible symmetric collective tasks and schedules symmetric work
into kernel plans. Source parameters include:

| Variable | Meaning |
|---|---|
| `NCCL_SYM_CTAS` | symmetric CTA count override |
| `NCCL_SYM_GIN_KERNELS_ENABLE` | enable symmetric GIN kernels |
| `NCCL_SYM_TMA_ENABLE` | enable TMA usage |
| `NCCL_SYM_CE_THRESHOLD` | CE threshold for symmetric path |
| `NCCL_SYM_NOWIN_ENABLE` | symmetric no-window enable |

Symmetric scheduling is relevant for window-registered collectives and advanced device/network paths.

## AllGatherV scheduler

`scheduler/allgatherv_sched.cc` implements specialized scheduling for all-gather-v style behavior. It is
separate because variable-size gather patterns require different metadata and scheduling than fixed-count
AllGather.

## Copy-engine / CE collectives

`ce_coll.cc` and related enqueue queues handle copy-engine or specialized collective paths where NCCL can
use alternate execution resources. When debugging unexpected plan types, check `isCeColl` in
`ncclKernelPlan` and `collCeTaskQueue` in the planner.

## Source-reading recipes

### A new API call does not launch

Trace:

```text
collectives.cc wrapper
  -> ncclEnqueueCheck
  -> taskAppend
  -> ncclPrepareTasks
  -> ncclLaunchPrepare
  -> schedule*TasksToPlan
  -> ncclLaunchKernel
```

Check whether the task count increases and whether it is sorted into an unexpected queue.

### P2P deadlock or no progress

Trace:

```text
ncclSend/ncclRecv wrappers
  -> taskAppend P2P queues
  -> ncclPrepareTasks peer queues
  -> scheduleP2pTasksToPlan
  -> proxy op creation
  -> proxy.cc progress
```

Verify group boundaries include all matching sends/recvs.

### CUDA graph bug

Trace:

```text
stream capture detection in enqueue
  -> planner->capturingGraph
  -> persistent plan/work storage
  -> uploadWork
  -> ncclLaunchKernel capture path
  -> cleanup callbacks/destructors
```

### Performance regression after scheduler change

Inspect:

- task sorting order,
- number of kernel plans,
- channel masks,
- algorithm/protocol selected,
- proxy op count,
- work FIFO bytes,
- graph capture persistence.

## Invariants to preserve when editing

1. A task must be owned by exactly one queue at a time.
2. P2P send/recv matching must be visible to the scheduler before launch.
3. Device work descriptor layout must match device kernel readers.
4. Proxy ops must be created for transports/protocols that require CPU/network progress.
5. Cleanup must happen after device/proxy work no longer references memory.
6. CUDA graph capture must not use transient allocations that disappear before replay.
7. Profiler event hierarchy must remain parented correctly across group/api/kernel/proxy events.
