# 04 - Memory Registration, Symmetric Windows, Suspend/Resume, and RMA Signals

## Primary source files

- `sources/nccl/src/nccl.h.in`: public memory, registration, window, suspend, RMA/signal APIs.
- `sources/nccl/src/allocator.cc`: `ncclMemAlloc` / `ncclMemFree` implementation.
- `sources/nccl/src/mem_manager.cc`: communicator memory tracking and suspend/resume.
- `sources/nccl/src/include/mem_manager.h`: tracked memory metadata.
- `sources/nccl/src/register/register.cc`, `coll_reg.cc`, `sendrecv_reg.cc`: registration internals.
- `sources/nccl/src/rma/*`: one-sided put/signal/wait implementation.
- `sources/nccl/docs/examples/04_user_buffer_registration/*`
- `sources/nccl/docs/examples/05_symmetric_memory/*`

## NCCL allocation helpers

```c
ncclMemAlloc(void** ptr, size_t size);
ncclMemFree(void* ptr);
```

`ncclMemAlloc` allocates memory in a form suitable for NCCL optimizations such as user buffer
registration and symmetric windows. The actual allocated size may be larger than requested due to
alignment/granularity requirements.

Use `ncclMemAlloc` instead of raw `cudaMalloc` when:

- The buffer will be registered with `ncclCommRegister`.
- The buffer will be used as a symmetric memory window.
- You are following Device API examples.
- You want NCCL to choose a compatible allocation strategy for advanced transports.

## User buffer registration

### Public API

```c
ncclCommRegister(comm, buff, size, &handle);
ncclCommDeregister(comm, handle);
```

Registration lets NCCL operate directly on user-allocated buffers, avoiding repeated per-call
registration/staging overhead. The in-repo examples describe it as useful for repeated collectives on
the same buffers and as a prerequisite for advanced features such as symmetric memory or Device API calls.

### Typical shape

```c
void* d_send = NULL;
void* d_recv = NULL;
void* send_handle = NULL;
void* recv_handle = NULL;

ncclMemAlloc(&d_send, size_bytes);
ncclMemAlloc(&d_recv, size_bytes);

ncclCommRegister(comm, d_send, size_bytes, &send_handle);
ncclCommRegister(comm, d_recv, size_bytes, &recv_handle);

ncclAllReduce(d_send, d_recv, count, ncclFloat32, ncclSum, comm, stream);

ncclCommDeregister(comm, send_handle);
ncclCommDeregister(comm, recv_handle);
ncclMemFree(d_send);
ncclMemFree(d_recv);
```

### When registration helps

| Scenario | Why registration helps |
|---|---|
| repeated collectives on same buffers | amortizes registration and transport setup |
| RDMA-capable networking | avoids repeated memory registration with NIC |
| symmetric memory / Device API | required or strongly expected by the feature |
| channel/resource pressure | can reduce internal NCCL buffering/resource usage |

### Pitfalls

- Do not deregister before all CUDA stream work using the buffer has completed.
- Match registration handles with the communicator that created them.
- Use NCCL allocation helpers for examples unless you know the transport's registration requirements.
- Registration may not improve one-off small operations; benchmark representative workloads.

## Symmetric memory windows

### Public API

```c
ncclCommWindowRegister(comm, buff, size, &win, winFlags);
ncclCommWindowDeregister(comm, win);
ncclWinGetUserPtr(comm, win, &ptr);
```

Window flags from `nccl.h.in`:

| Flag | Meaning |
|---|---|
| `NCCL_WIN_DEFAULT` | default behavior |
| `NCCL_WIN_COLL_SYMMETRIC` | register window for symmetric collective optimizations |
| `NCCL_WIN_STRICT_ORDERING` | request stricter ordering behavior |

`NCCL_WIN_REQUIRED_ALIGNMENT` is 4096.

### Symmetric collective window shape

```c
void* buffer = NULL;
ncclWindow_t win;

ncclMemAlloc(&buffer, size_bytes);
ncclCommWindowRegister(comm, buffer, size_bytes, &win, NCCL_WIN_COLL_SYMMETRIC);

ncclAllReduce(buffer, buffer, count, ncclFloat32, ncclSum, comm, stream);

ncclCommWindowDeregister(comm, win);
ncclMemFree(buffer);
```

### Why symmetric windows exist

The examples describe symmetric windows as enabling optimized collective protocols when all ranks use
consistent memory layouts. Memory must be allocated through CUDA Virtual Memory Management-compatible
paths and registered with NCCL. Symmetric memory is especially relevant for:

- Very low latency local collectives.
- Device API LSA peer access.
- Multimem-style local collectives where supported.
- Future-proof buffer layouts for NCCL internals.

## Communicator memory suspend/resume

### Public API

```c
#define NCCL_SUSPEND_MEM 0x01
ncclCommSuspend(comm, NCCL_SUSPEND_MEM);
ncclCommResume(comm);
```

Suspend releases suspendable dynamic GPU allocations tracked by NCCL. The communicator cannot be used
while suspended. Resume restores previously suspended resources.

### Memory statistics

```c
ncclCommMemStats(comm, ncclStatGpuMemSuspend, &value);
ncclCommMemStats(comm, ncclStatGpuMemSuspended, &value);
ncclCommMemStats(comm, ncclStatGpuMemPersist, &value);
ncclCommMemStats(comm, ncclStatGpuMemTotal, &value);
```

Stats:

| Stat | Meaning |
|---|---|
| `ncclStatGpuMemSuspend` | bytes of GPU memory that can be suspended |
| `ncclStatGpuMemSuspended` | whether suspendable memory is suspended (`0` or `1`) |
| `ncclStatGpuMemPersist` | bytes of GPU memory that cannot be suspended |
| `ncclStatGpuMemTotal` | total NCCL-tracked GPU memory |

Use this when an application needs to temporarily reduce NCCL memory footprint, e.g. between phases or
while another subsystem owns GPU memory.

## One-sided RMA/signal APIs

### `ncclPutSignal`

```c
ncclPutSignal(localbuff, count, datatype,
              peer, peerWin, peerWinOffset,
              sigIdx, ctx, flags, comm, stream);
```

Writes data from a local buffer to a remote peer's registered memory window and associates the operation
with a signal index/context. The target process does not explicitly post a matching receive in the same
way as `ncclRecv`.

Parameters to pay attention to:

| Parameter | Meaning |
|---|---|
| `peer` | target rank |
| `peerWin` | memory window registered by target peer |
| `peerWinOffset` | byte offset inside peer window |
| `sigIdx` | signal index identifier |
| `ctx` | context identifier |
| `flags` | reserved for future use in public header |

### `ncclSignal`

```c
ncclSignal(peer, sigIdx, ctx, flags, comm, stream);
```

Sends a signal to a peer without transferring data.

### `ncclWaitSignal`

```c
typedef struct {
  int opCnt;
  int peer;
  int sigIdx;
  int ctx;
} ncclWaitSignalDesc_t;

ncclWaitSignal(nDesc, signalDescs, comm, stream);
```

Waits for one or more signal descriptors. Each descriptor specifies how many signal operations to wait
for from a given peer/signal/context combination.

## RMA vs P2P vs collectives

| Need | Prefer |
|---|---|
| standard distributed training gradients | `ncclAllReduce` or reduce-scatter/all-gather decomposition |
| explicit pairwise exchange | `ncclSend` / `ncclRecv` with groups |
| one-sided writes into registered peer windows | `ncclPutSignal` + `ncclWaitSignal` |
| custom in-kernel peer access | Device API with windows/LSA/GIN |
| repeated collectives on fixed buffers | `ncclCommRegister` or symmetric windows where applicable |

## Registration internals map

- `src/register/register.cc`: generic local/user registration logic.
- `src/register/coll_reg.cc`: collective registration paths.
- `src/register/sendrecv_reg.cc`: send/recv registration paths.
- `src/mem_manager.cc`: tracks communicator allocations, peer import/export, suspend/resume, stats.
- `src/transport/net.cc` and `src/transport/net_ib/reg.cc`: network transport registration behavior.

When modifying registration code, trace both the public handle lifetime and the transport-specific memory
handle lifetime.

## Practical safety checklist

1. Allocate/initialize buffers before registration.
2. Register with every communicator that will use the buffer.
3. Enqueue NCCL work only after registration succeeds.
4. Synchronize streams or otherwise prove work completion before deregistration/free.
5. Keep window offsets aligned where required.
6. Do not assume peer window address equality unless using APIs that explicitly expose peer pointers.
7. For hangs, verify all ranks register compatible windows and reach the same signal/wait sequence.
