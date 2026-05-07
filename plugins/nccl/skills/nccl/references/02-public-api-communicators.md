# 02 - Public API and Communicators

## Primary source files

- `sources/nccl/src/nccl.h.in`: public C API template.
- `sources/nccl/src/init.cc`: implementation of communicator lifecycle APIs.
- `sources/nccl/src/dev_runtime.cc`: Device API host entry points such as `ncclCommQueryProperties`.
- `sources/nccl/docs/examples/01_communicators/*`: initialization examples.

## Version macros

The generated public header exposes:

```c
#define NCCL_MAJOR 2
#define NCCL_MINOR 30
#define NCCL_PATCH 4
#define NCCL_VERSION_CODE ...
#define NCCL_VERSION(X,Y,Z) ...
```

Use `ncclGetVersion(int* version)` at runtime when an application must adapt to the loaded shared
library rather than the compile-time header.

## Public opaque handles

| Type | Meaning |
|---|---|
| `ncclComm_t` | communicator handle; internally points to `struct ncclComm` |
| `ncclWindow_t` | registered symmetric/window memory handle |
| `ncclUniqueId` | 128-byte unique ID used by ranks to create the same communicator clique |
| `ncclParamHandle_t` | opaque handle to a runtime parameter |

`NCCL_COMM_NULL` is `NULL`.

## Error enum

`ncclResult_t` values:

| Value | Typical interpretation |
|---|---|
| `ncclSuccess` | operation succeeded or was successfully enqueued |
| `ncclUnhandledCudaError` | CUDA runtime/driver call failed |
| `ncclSystemError` | OS/system/network/library call failed |
| `ncclInternalError` | NCCL internal invariant failure or unexpected state |
| `ncclInvalidArgument` | invalid API argument |
| `ncclInvalidUsage` | API used incorrectly for current state or constraints |
| `ncclRemoteError` | remote peer failed or reported an error |
| `ncclInProgress` | nonblocking/finalize/revoke operation is still progressing |
| `ncclTimeout` | timeout condition |

Diagnostics:

```c
const char* ncclGetErrorString(ncclResult_t result);
const char* ncclGetLastError(ncclComm_t comm);
ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t* asyncError);
```

`ncclGetErrorString` describes a return code. `ncclGetLastError` is communicator-specific and often
more useful after an operation fails. `ncclCommGetAsyncError` is important because NCCL work is
asynchronous with respect to host code and errors may surface after enqueue.

## Communicator configuration

`ncclConfig_t` is versioned and must be initialized with `NCCL_CONFIG_INITIALIZER`:

```c
ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
config.blocking = 1;
config.netName = "IB";
config.commName = "trainer-data-parallel";
ncclCommInitRankConfig(&comm, nranks, id, rank, &config);
```

Important fields from `ncclConfig_t`:

| Field | Purpose |
|---|---|
| `blocking` | blocking vs nonblocking communicator behavior; `NCCL_CONFIG_UNDEF_INT` means default/env |
| `cgaClusterSize` | cluster size for cooperative group array launch style where supported |
| `minCTAs`, `maxCTAs` | bound CTA usage per operation |
| `netName` | request a specific network implementation name |
| `splitShare` | resource sharing behavior for `ncclCommSplit` |
| `trafficClass` | traffic/QoS class passed to network plugin configuration |
| `commName` | user-readable communicator name, useful for logs/profiler plugins |
| `collnetEnable` | CollNet enable/disable override |
| `CTAPolicy` | CTA policy: default, efficiency, or zero |
| `shrinkShare` | resource sharing behavior for shrink |
| `nvlsCTAs` | NVLS CTA count override |
| `nChannelsPerNetPeer` | channel count per network peer override |
| `nvlinkCentricSched` | NVLink-utilization-centric scheduling mode |
| `graphUsageMode` | CUDA graph usage behavior |
| `numRmaCtx` | RMA context count |
| `maxP2pPeers` | maximum P2P peer count |

Do not manually fill `size`, `magic`, or `version`. NCCL validates these fields.

## Initialization APIs

### `ncclCommInitAll`

```c
ncclResult_t ncclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist);
```

Use for single-process, single-node code where one process manages multiple GPUs. It creates `ndev`
communicators, one per device, and rank order follows `devlist` if provided. If `devlist == NULL`,
NCCL uses the first `ndev` CUDA devices.

Best for:

- Examples and quick tests.
- Single-node applications with one process controlling all GPUs.
- Avoiding external bootstrap code.

Limitations:

- Not a multi-node launcher.
- Less representative of production distributed training than one rank per GPU.

### `ncclGetUniqueId` + `ncclCommInitRank`

```c
ncclUniqueId id;
if (rank == 0) ncclGetUniqueId(&id);
// broadcast id to all ranks with MPI/socket/shared memory/etc.
cudaSetDevice(localDevice);
ncclCommInitRank(&comm, nranks, id, rank);
```

Rules:

1. Call `ncclGetUniqueId` once for the communicator clique.
2. Distribute the ID to every rank before `ncclCommInitRank`.
3. Each rank must have a unique `rank` in `[0, nranks)`.
4. The CUDA device must be set before communicator initialization.
5. `ncclCommInitRank` implicitly synchronizes with other ranks, so call it from separate processes,
   separate threads, or use group semantics when a single host thread initializes multiple ranks.

### `ncclCommInitRankConfig`

Same as `ncclCommInitRank`, but accepts `ncclConfig_t*`.

Use when you need communicator-specific settings such as `commName`, `trafficClass`, `netName`, or
blocking behavior. If the user is debugging network plugin QoS or forcing a net implementation, this
is the public entry point to mention.

### `ncclCommInitRankScalable`

```c
ncclCommInitRankScalable(ncclComm_t* newcomm, int nranks, int myrank,
                         int nId, ncclUniqueId* commIds, ncclConfig_t* config);
```

This allows more than one unique ID, up to one per rank, to accelerate initialization. The number and
order of IDs must be identical on every rank.

Use for very large communicator initialization bottlenecks; keep standard `ncclCommInitRank` for normal
cases.

## Lifecycle APIs

### `ncclCommFinalize`

```c
ncclCommFinalize(comm);
```

Flushes issued communication and marks communicator state as `ncclInProgress`. The state changes to
`ncclSuccess` when globally quiescent and related resources have been freed. After that,
`ncclCommDestroy` can free local resources without blocking on global communication.

Use this for orderly shutdown.

### `ncclCommDestroy`

```c
ncclCommDestroy(comm);
```

Frees local resources associated with the communicator. In examples, the clean shutdown shape is:

```c
ncclCommFinalize(comm);
ncclCommDestroy(comm);
```

### `ncclCommAbort`

```c
ncclCommAbort(comm);
```

Frees communicator resources and aborts operations that might still be running on the device. Use for
error paths, timeouts, or failed ranks when normal finalize cannot complete. Do not present abort as a
normal teardown substitute unless the user is explicitly handling failures.

### `ncclCommRevoke`

```c
ncclCommRevoke(comm, NCCL_REVOKE_DEFAULT);
```

Stops in-flight operations and marks communicator state as `ncclInProgress` until quiescent. After
revoke, management operations such as destroy, split, and shrink can proceed safely. Calling finalize
after revoke is invalid. Resource sharing through split/shrink is disabled while revoked.

## Resizing and membership APIs

### `ncclCommSplit`

```c
ncclCommSplit(comm, color, key, &newcomm, config);
```

Ranks with the same `color` form a new communicator. `key` orders ranks within the new communicator.
`NCCL_SPLIT_NOCOLOR` excludes a rank and returns a null communicator. If `config == NULL`, the new
communicator inherits configuration from the parent.

### `ncclCommShrink`

```c
ncclCommShrink(comm, excludeRanksList, excludeRanksCount, &newcomm, config, shrinkFlags);
```

Removes ranks listed in `excludeRanksList`; new ranks are compacted to fill gaps. Flags:

| Flag | Meaning |
|---|---|
| `NCCL_SHRINK_DEFAULT` | shrink parent communicator |
| `NCCL_SHRINK_ABORT` | first terminate ongoing parent operations, then shrink |

Use shrink in failure recovery or elastic membership scenarios where the remaining ranks continue.

### `ncclCommGetUniqueId` + `ncclCommGrow`

`ncclCommGetUniqueId(comm, &id)` generates a per-communicator ID for grow. Constraints in the header:

- Cannot generate a new UID while a previous UID is unconsumed.
- Each UID can be used only once.
- Must wait for grow to complete before calling again.

`ncclCommGrow` usage by role:

| Role | `comm` | `uniqueId` | `rank` |
|---|---|---|---|
| existing non-root | existing communicator | `NULL` | `-1` |
| existing root | existing communicator | `&id` | `-1` |
| new rank | `NULL` | `&id` | assigned rank |

## Query APIs

```c
ncclCommCount(comm, &count);
ncclCommCuDevice(comm, &device);
ncclCommUserRank(comm, &rank);
ncclCommGetAsyncError(comm, &asyncError);
```

Use these in library integrations to validate the communicator matches the caller's rank/device
expectations.

Device API property query is declared in `src/include/nccl_device/core.h`:

```c
ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
ncclCommQueryProperties(comm, &props);
```

Important properties:

| Field | Meaning |
|---|---|
| `rank`, `nRanks` | communicator rank and size |
| `cudaDev`, `nvmlDev` | CUDA/NVML device identifiers |
| `deviceApiSupport` | whether Device API can be used |
| `multimemSupport` | support for multimem windows |
| `ginType` | GIN support type (`NONE`, `PROXY`, `GDAKI`) |
| `nLsaTeams` | number of local load/store-accessible teams |
| `hostRmaSupport` | host RMA support |
| `railedGinType` | railed GIN support type |

## Initialization examples and when to choose them

| Pattern | API | Multi-node | Best for |
|---|---|---:|---|
| single process manages all GPUs | `ncclCommInitAll` | no | examples, quick tests, simple single-node apps |
| one pthread per GPU | `ncclCommInitRank` | no | single-node threaded runtimes |
| one process per GPU with MPI | `ncclCommInitRank` | yes | production distributed training and clusters |

The MPI pattern broadcasts `ncclUniqueId`, maps each local rank to a CUDA device, calls
`cudaSetDevice(localRankDevice)`, and initializes one communicator per process.

## Common pitfalls

1. **Calling `ncclCommInitRank` sequentially from one thread for multiple local ranks.** It synchronizes
   across ranks; use separate threads/processes or group semantics.
2. **Forgetting `cudaSetDevice` before init.** The communicator binds to the current CUDA device.
3. **Destroying immediately after enqueue without stream synchronization/finalize.** NCCL work is
   asynchronous with CUDA streams.
4. **Mixing communicator groups incorrectly.** All ranks in a clique must call compatible operations.
5. **Assuming rank equals CUDA device ID.** Rank is user order; device mapping is application-defined.
