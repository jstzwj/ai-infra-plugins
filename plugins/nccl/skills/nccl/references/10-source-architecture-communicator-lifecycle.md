# 10 - Source Architecture and Communicator Lifecycle Internals

## Primary source files

- `sources/nccl/src/include/comm.h`: central host-side runtime structures.
- `sources/nccl/src/init.cc`: communicator init, split/shrink/grow/finalize/destroy/abort.
- `sources/nccl/src/bootstrap.cc`: rank bootstrap and unique ID exchange.
- `sources/nccl/src/channel.cc`: channel helpers.
- `sources/nccl/src/enhcompat.cc`: enhanced compatibility/version shims.
- `sources/nccl/src/devcomm/*`: Device API compatibility layers for older ABI versions.
- `sources/nccl/src/include/core.h`, `checks.h`, `argcheck.h`: common API checking and return-code patterns.

## Mental model of NCCL host internals

At runtime, a public API call mostly moves through this pipeline:

```text
public API call
  -> argument checks / graph capture checks / profiler API event
  -> ncclInfo descriptor
  -> group/enqueue task accumulation
  -> task preparation and sorting
  -> topology/tuning selection if needed
  -> ncclKernelPlan creation
  -> proxy op creation for CPU/network-assisted paths
  -> upload device work descriptors
  -> CUDA kernel launch or CUDA graph capture
  -> proxy progress + device protocol execution
  -> cleanup callbacks/destructors
```

`struct ncclComm` is the central object tying all of these layers together.

## `comm.h`: core structures

### `ncclComm`

The full `ncclComm` definition is large and appears later in `comm.h`, but conceptually it owns:

- rank/world metadata,
- CUDA device metadata,
- peer information for every rank,
- local rank mapping,
- channels and channel peers,
- topology system and graphs,
- shared resources such as streams and proxy state,
- scheduler/planner state,
- memory manager,
- RAS state,
- plugin/tuner/profiler state,
- async error and lifecycle state.

When debugging internals, find the field in `comm.h` before chasing callsites.

### `ncclSharedResources`

Shared resources support communicator split/shrink sharing and common per-process resources:

| Field group | Purpose |
|---|---|
| `refCount`, `owner` | lifetime and ownership |
| `peers`, `devPeers` | host/device peer connector arrays per channel |
| `p2pOpCount`, `collOpCount` | shared operation counters |
| `tp*` fields | top-parent rank/channel information for shared resources |
| `deviceStream`, `hostStream` | internal strong streams |
| `launchEvent`, `scratchEvent` | internal CUDA events |
| `proxyState` | proxy progress engine state |
| `ginState` | GIN state |

### `ncclChannel`

A channel is a logical communication lane. It contains:

- host peer connector array (`peers`),
- device peer connector array (`devPeers`),
- host pointer to device peer array (`devPeersHostPtr`),
- ring structure,
- tree structure,
- CollNet chain/direct structures,
- NVLS structures,
- channel ID,
- work FIFO production state,
- sharable CollNet/NVLS peer resources.

Algorithms map work onto channels. Channel count and connectivity are major performance dimensions.

### Task structures

| Structure | Meaning |
|---|---|
| `ncclTaskColl` | collective task accumulated from `ncclInfo` |
| `ncclTaskP2p` | send/recv task |
| `ncclTaskRma` | one-sided/signal task |
| `ncclTaskBcast` | specialized broadcast task |
| `ncclTaskCollSorter` | approximate descending-size sorter for collective tasks |

`ncclTaskColl` includes operation metadata, computed algorithm/protocol/channel data, registration/window
metadata, profiler handles, and cleanup state.

### `ncclKernelPlanner`

The planner accumulates tasks between group boundaries and builds launchable plans.

Important state groups:

1. **Accumulation state**: coll sorter, per-peer P2P queues, task counts, captured streams.
2. **Prepared task queues**: normal collectives, CE collectives, symmetric collectives, RMA queues.
3. **WIP plan**: per-channel work batches and proxy op queues.
4. **Plan queue**: built `ncclKernelPlan` objects waiting for launch.
5. **Graph capture state**: capture graph and persistence behavior.

### `ncclKernelPlan`

A plan is a launch unit. It contains:

- callback/reclaimer as first field,
- owning communicator,
- persistent/captured status,
- flags for symmetric, CE, RMA, host callback enqueue,
- work storage type,
- kernel function/arguments,
- channel mask,
- proxy op presence,
- thread/block settings,
- counts and work queues,
- cleanup queue,
- profiler handles.

If a user asks "why did NCCL launch N kernels?", inspect how tasks were partitioned into kernel plans.

## Communicator initialization pipeline

The high-level initialization path from `init.cc`:

```text
ncclCommInitRank / ncclCommInitRankConfig
  -> ncclCommInitRankDev
  -> create async init job
  -> ncclCommInitRankFunc
  -> rank bootstrap and peer info exchange
  -> initTransportsRank
  -> topology discovery and graph search
  -> channels and transport connections
  -> tuning model initialization
  -> proxy/RAS/device-side state
  -> communicator returned/ready
```

The implementation uses `ncclAsyncJob` and group semantics so communicator creation can be blocking or
nonblocking depending on config/environment and grouping.

## `initTransportsRank`

The internals survey identifies `initTransportsRank` as the major handoff from communicator metadata into:

- topology discovery,
- channel construction,
- transport initialization,
- proxy setup,
- tuning,
- RAS,
- device-side state.

When initialization succeeds but performance is wrong, this is the phase where the chosen topology and
transport are usually determined.

## Bootstrap

`bootstrap.cc` is responsible for out-of-band rank communication needed before NCCL transports are ready.
It distributes peer information, connection handles, and unique ID metadata. Parameters include UID
staggering and optional out-of-band net behavior.

Common bootstrap failure causes:

- ranks disagree on world size,
- unique ID not distributed to all ranks,
- network/firewall/container issue,
- launcher environment differs across ranks,
- process exits before peers complete bootstrap.

## Lifecycle operations

### Finalize/destroy/abort

`ncclCommFinalize` transitions a communicator toward global quiescence; `ncclCommDestroy` frees local
resources; `ncclCommAbort` terminates in-flight work and frees resources. Internally these operations
must coordinate with proxy state, asynchronous jobs, outstanding plans, graph captures, and destructors.

### Split/shrink/grow

`ncclCommSplit`, `ncclCommShrink`, and `ncclCommGrow` reuse the async job model. They have to derive new
rank membership, possibly share resources, update bootstrap/topology state, and ensure parent communicator
state permits mutation.

Important resource-sharing knobs:

- `config.splitShare`
- `config.shrinkShare`
- `NCCL_COMM_SPLIT_SHARE_RESOURCES`
- `NCCL_COMM_SHRINK_SHARE_RESOURCES`

### Revoke

Revoke stops in-flight operations and waits for quiescence so management operations can proceed. After
revoke, finalize is invalid. Split/shrink resource sharing is disabled while revoked.

## Async job model

`group.cc` and `init.cc` use `ncclAsyncJob` to represent work that may block or progress asynchronously.
This unifies communicator init/mutation with grouped operation launch.

Conceptually:

- create job with function and destructor,
- enqueue job in group state,
- progress/launch at group end or blocking wait,
- propagate result to communicator state,
- clean up resources.

For debugging, distinguish a job that is still `ncclInProgress` from one that failed and set async error.

## Profiler and NVTX integration

Public API wrappers create NVTX events/payloads. `ncclKernelPlan` and task structures carry profiler event
handles for group API, collective API, P2P API, kernel launch, and device/proxy events. Plugin activation
masks decide which events are emitted.

When adding a new operation type, update:

- enum/string conversion,
- public wrapper,
- task formation,
- profiler descriptors,
- device work descriptor,
- scheduling and kernel dispatch.

## Source-reading paths

| Question | Start here |
|---|---|
| how does a public API become device work? | `collectives.cc` -> `enqueue.cc` -> `device/common_kernel.h` |
| why communicator init hangs? | `init.cc` -> `bootstrap.cc` -> `graph/*` -> `transport.cc` |
| where are ranks/channels stored? | `include/comm.h` |
| how are split/shrink/grow implemented? | `init.cc`, lifecycle API functions |
| how are tasks grouped? | `group.cc`, `include/group.h`, `enqueue.cc` |
| where are proxy ops created/progressed? | `enqueue.cc`, `proxy.cc`, transport files |
| how does Device API query properties? | `dev_runtime.cc`, `include/nccl_device/core.h` |

## Modification caution

NCCL internals are tightly coupled across host scheduling, device work descriptors, proxy operations, and
plugin ABI. A change that compiles in `collectives.cc` may still require updates in device kernels,
profiler payloads, tuner logic, and compatibility shims. Always grep the operation enum/function name
across the tree before editing.
