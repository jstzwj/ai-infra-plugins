# 08 - Debugging, Troubleshooting, Logging, and Profiling

## Primary source files

- `sources/nccl/src/debug.cc`: logging subsystem and debug behavior.
- `sources/nccl/src/collectives.cc`: NVTX payload wrapping of public operations.
- `sources/nccl/src/init.cc`: init-time warnings and setup.
- `sources/nccl/src/enqueue.cc`: launch/capture/scheduler failure points.
- `sources/nccl/src/proxy.cc`: proxy progress engine.
- `sources/nccl/src/ras/*`: fault handling/RAS.
- `sources/nccl/plugins/profiler/README.md`: profiler plugin API.
- `sources/nccl/docs/examples/common/include/nccl_utils.h`: example `NCCLCHECK` macro style.

## First debug command

For most NCCL application issues, start with:

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,NET,TUNING ./your_app
```

If the hang involves many ranks, capture logs per rank:

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,NET,TUNING \
  mpirun -np 8 ./your_app 2>&1 | tee nccl.log
```

For production clusters, configure the launcher to write one rank log per file so the last line from
each rank is visible.

## Error checking pattern

Educational examples use macros similar to:

```c
#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r != ncclSuccess) {                           \
    fprintf(stderr, "%s:%d NCCL failure %s: %s\n",  \
            __FILE__, __LINE__, #cmd,               \
            ncclGetErrorString(r));                 \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while (0)
```

For libraries, return errors rather than exiting, but keep:

- file/line or callsite context,
- failed operation string,
- `ncclGetErrorString`,
- `ncclGetLastError(comm)` when a communicator exists,
- async error polling when operations may fail later.

## Async errors

NCCL work is asynchronous with CUDA stream execution. A call can enqueue successfully and fail later.
Use:

```c
ncclResult_t asyncErr = ncclSuccess;
ncclCommGetAsyncError(comm, &asyncErr);
if (asyncErr != ncclSuccess) {
  fprintf(stderr, "async NCCL error: %s last=%s\n",
          ncclGetErrorString(asyncErr), ncclGetLastError(comm));
  ncclCommAbort(comm);
}
```

Poll async errors in long-running distributed jobs, especially while waiting for CUDA streams/events or
for remote ranks.

## Hangs: triage checklist

### Communicator init hang

Check:

1. Every rank received the same `ncclUniqueId`.
2. Every rank calls `ncclCommInitRank` with same `nranks` and unique `rank`.
3. Each rank calls `cudaSetDevice` before init.
4. Rank-to-GPU mapping is valid and does not oversubscribe unintentionally.
5. Firewalls/container networking allow bootstrap communication.
6. If MPI is used, `MPI_Bcast` covers exactly the communicator ranks.
7. Logs show all ranks reach the same bootstrap/init phase.

Useful logs:

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,BOOTSTRAP,NET
```

### Collective hang

Check:

1. All ranks call the same collective sequence.
2. Counts, datatypes, roots, and communicator objects match.
3. Single-thread multi-device loops are grouped.
4. No rank skipped the collective due to conditional logic.
5. CUDA work before the NCCL call on the same stream is not stuck.
6. Application did not free/deregister buffers too early.

### P2P hang

Check:

1. Every `ncclSend` has a matching `ncclRecv` with same count/datatype and opposite peer.
2. Concurrent send/recv cycles are inside `ncclGroupStart/End`.
3. Peer IDs are ranks, not device IDs.
4. The group includes all P2P operations needed to make progress.

### Shutdown hang

Check:

1. Streams using NCCL work have progressed.
2. Ranks call finalize/destroy in compatible order.
3. No rank aborted while peers wait for normal finalize.
4. Use `ncclCommGetAsyncError` and `ncclCommAbort` on error paths.

## Performance triage order

1. **Establish baseline** with `nccl-tests` on the same nodes/GPU mapping.
2. **Check topology logs** with `NCCL_DEBUG_SUBSYS=INIT,GRAPH,NET,TUNING`.
3. **Check transport**: P2P/SHM/NET/IB/socket/external plugin.
4. **Check algorithm/protocol**: ring/tree/CollNet/NVLS/PAT and LL/LL128/SIMPLE.
5. **Check channel count** and CTA/thread settings.
6. **Check buffer registration** for repeated large operations.
7. **Check CUDA stream synchronization** and whether application serializes communication unintentionally.
8. **Profile** with Nsight Systems/NVTX and NCCL profiler plugin if deeper instrumentation is needed.

## Common symptoms table

| Symptom | Likely causes | Next action |
|---|---|---|
| `ncclCommInitRank` never returns | rank count/ID mismatch, bootstrap networking, wrong launch | print rank/env/device before init; enable `INIT,BOOTSTRAP,NET` logs |
| first collective hangs | mismatched operation order or missing rank | log operation sequence per rank |
| P2P ring hangs | not grouped, wrong peer formula | wrap send/recv in group; verify prev/next ranks |
| slow multi-node bandwidth | socket fallback, wrong NIC, GDR disabled, topology mismatch | inspect `NET` logs; compare `nccl-tests` |
| slow small messages | protocol/algorithm not suitable, launch overhead, no grouping | inspect `TUNING`; test LL/LL128 behavior |
| CUDA graph capture failure | mixed capture streams, unsupported path, registration issue | isolate capture; check `enqueue.cc` graph handling logs |
| errors only after stream sync | async NCCL/CUDA failure | poll `ncclCommGetAsyncError`, print `ncclGetLastError` |
| registered-buffer crash | freed/deregistered before stream completed | synchronize before deregister/free |

## NVTX instrumentation

`collectives.cc` wraps public APIs with NVTX payload macros, including operation name, comm hash, count
in bytes, root/peer, and op where relevant. This helps Nsight Systems correlate application calls with
NCCL kernels and proxy/network work.

Disable NVTX if needed:

```bash
export NCCL_NVTX_DISABLE=1
```

## Profiler plugin overview

Profiler plugins were introduced to extract performance data from NCCL and integrate with frameworks.
They load as:

```text
libnccl-profiler.so
libnccl-profiler-${NCCL_PROFILER_PLUGIN}.so
```

or by setting `NCCL_PROFILER_PLUGIN` to a pathname.

The profiler plugin exports versioned symbols such as `ncclProfiler_v5`.

### Main v5 interface

```c
typedef struct {
  const char* name;
  ncclResult_t (*init)(void** context, uint64_t commId, int* eActivationMask,
                       const char* commName, int nNodes, int nranks, int rank,
                       ncclDebugLogger_t logfn);
  ncclResult_t (*startEvent)(void* context, void** eHandle,
                             ncclProfilerEventDescr_v5_t* eDescr);
  ncclResult_t (*stopEvent)(void* eHandle);
  ncclResult_t (*recordEventState)(void* eHandle,
                                   ncclProfilerEventState_v5_t eState,
                                   ncclProfilerEventStateArgs_v5_t* eStateArgs);
  ncclResult_t (*finalize)(void* context);
} ncclProfiler_v5_t;
```

Profiler generated errors generally should not alter normal NCCL behavior. The docs advise returning
`ncclSuccess` except `init`, where failure can disable the plugin.

### Event types

The profiler docs list event categories including:

- `ncclProfileGroupApi`
- `ncclProfileCollApi`
- `ncclProfileP2pApi`
- `ncclProfileKernelLaunch`
- `ncclProfileGroup`
- `ncclProfileColl`
- `ncclProfileP2p`
- `ncclProfileProxyOp`
- `ncclProfileProxyStep`
- `ncclProfileProxyCtrl`
- `ncclProfileKernelCh`
- `ncclProfileNetPlugin`

This is the right extension surface when a framework wants structured NCCL timing without parsing logs.

## Proxy progress debugging

Network and some transport paths rely on CPU proxy progress. Relevant source:

- `src/proxy.cc`
- `src/include/proxy.h`
- transport implementations that enqueue `ncclProxyOp`

Proxy-related profiler events distinguish:

- proxy op state,
- individual proxy steps,
- proxy control idle/active/sleep/wakeup,
- append of new network work.

If GPU kernels are launched but network communication does not progress, inspect proxy thread state,
network plugin completions, and whether helper threads are running.

## RAS and fault handling

RAS source files (`src/ras/*`) implement a background reliability/availability/serviceability subsystem
with peer tracking, keepalive/retry behavior, local notifications, and a diagnostic client. For user-facing
failure handling, combine:

- async error polling,
- `ncclCommRevoke`,
- `ncclCommShrink(... NCCL_SHRINK_ABORT ...)`,
- `ncclCommAbort` on unrecoverable paths.

See `16-ras-fault-handling.md` for subsystem details.

## Minimal data to request from a user reporting NCCL issues

Ask for:

1. NCCL version and how it was installed/built.
2. CUDA driver/runtime version and GPU model.
3. Number of nodes, GPUs per node, process count, launcher command.
4. Rank-to-GPU mapping logic.
5. Relevant `NCCL_*` variables.
6. `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,NET,TUNING` logs from all ranks.
7. Whether `nccl-tests` passes on the same allocation.
8. Exact operation where hang/error occurs.

This usually separates application ordering bugs from NCCL topology/network problems.
