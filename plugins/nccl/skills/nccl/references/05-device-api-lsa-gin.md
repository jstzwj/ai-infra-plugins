# 05 - NCCL Device API, LSA, GIN, Teams, and Device-Side Communication

## Primary source files

- `sources/nccl/src/include/nccl_device.h`: umbrella header for Device API.
- `sources/nccl/src/include/nccl_device/core.h`: host/device core types and declarations.
- `sources/nccl/src/include/nccl_device/coop.h`: cooperative launch/device coordination helpers.
- `sources/nccl/src/include/nccl_device/barrier.h`, `lsa_barrier`, `gin_barrier`: barrier surfaces.
- `sources/nccl/src/include/nccl_device/ll_a2a.h`: low-latency all-to-all helpers.
- `sources/nccl/src/include/nccl_device/ptr.h`: pointer/window helpers.
- `sources/nccl/src/include/nccl_device/reduce_copy.h`: reduce/copy primitives for device code.
- `sources/nccl/src/dev_runtime.cc`: host runtime implementation.
- `sources/nccl/src/gin/*`: GIN host/proxy support.
- `sources/nccl/docs/examples/06_device_api/*`: LSA, GIN, and hybrid examples.

## What the Device API enables

NCCL's host collectives enqueue NCCL-owned kernels. The Device API enables application CUDA kernels to
perform communication directly. This lets applications:

- Fuse communication with custom compute.
- Implement custom collective algorithms in CUDA kernels.
- Use load/store-accessible peer memory for local ranks.
- Use GPU-Initiated Networking (GIN) for remote peers where supported.
- Build MoE/expert-parallel primitives such as NCCL EP dispatch/combine.

The examples frame the Device API as a way to schedule communication from inside CUDA kernels.

## Required include and build context

Host and device code include:

```cpp
#include <nccl.h>
#include <nccl_device.h>
```

Device API examples are CUDA `.cu` programs and require a NCCL build with Device API support. For
multi-node GIN paths, the hardware/network stack must support the chosen GIN backend.

## Capability discovery

Before using Device API features, query communicator properties:

```cpp
ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
NCCLCHECK(ncclCommQueryProperties(comm, &props));

if (!props.deviceApiSupport) {
  // Device API not available for this communicator/platform.
}
if (props.ginType == NCCL_GIN_TYPE_NONE) {
  // GIN not available.
}
```

Important fields from `ncclCommProperties_t`:

| Field | Meaning |
|---|---|
| `rank` | this communicator rank |
| `nRanks` | communicator size |
| `cudaDev` | associated CUDA device |
| `nvmlDev` | associated NVML device |
| `deviceApiSupport` | whether `ncclDevCommCreate` can be used |
| `multimemSupport` | whether multimem pointers/handles are supported |
| `ginType` | `NCCL_GIN_TYPE_NONE`, `NCCL_GIN_TYPE_PROXY`, or `NCCL_GIN_TYPE_GDAKI` |
| `nLsaTeams` | number of load-store-accessible teams |
| `hostRmaSupport` | host RMA availability |
| `railedGinType` | GIN type for railed GIN |

GIN type values:

| Value | Meaning |
|---|---|
| `NCCL_GIN_TYPE_NONE` | no GIN support |
| `NCCL_GIN_TYPE_PROXY` | proxy-backed GIN path |
| `NCCL_GIN_TYPE_GDAKI` | GPU Direct Async Kernel-Initiated path |

## Device communicator creation

### Requirements structure

```cpp
ncclDevCommRequirements_t reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
```

Important fields:

| Field | Meaning |
|---|---|
| `resourceRequirementsList` | linked list of device resource buffer requirements |
| `teamRequirementsList` | linked list of team/multimem requirements |
| `lsaMultimem` | enable multimem on LSA team |
| `barrierCount` | generic barrier count |
| `lsaBarrierCount` | LSA barrier count |
| `railGinBarrierCount` | rail GIN barrier count |
| `lsaLLA2ABlockCount`, `lsaLLA2ASlotCount` | LSA low-latency all-to-all resources |
| `ginForceEnable` | force GIN enablement request |
| `ginContextCount` | GIN context hint |
| `ginSignalCount`, `ginCounterCount` | allocated signal/counter ranges |
| `ginConnectionType` | none/full/rail GIN connection type |
| `ginExclusiveContexts` | exclusive GIN context request |
| `ginQueueDepth` | queue depth hint |
| `ginTrafficClass` | traffic class/QoS |
| `worldGinBarrierCount` | world-level GIN barrier count |

### Host creation/destruction

```cpp
ncclDevComm_t devComm;
ncclDevCommRequirements_t reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
reqs.lsaBarrierCount = NCCL_DEVICE_CTA_COUNT;

NCCLCHECK(ncclDevCommCreate(comm, &reqs, &devComm));
// launch kernels using devComm
NCCLCHECK(ncclDevCommDestroy(comm, &devComm));
```

The actual example code registers windows and creates the device communicator before launching custom
kernels.

## Teams

`ncclTeam_t` describes a rank subset with fields:

```c
struct ncclTeam {
  int nRanks;
  int rank;
  int stride;
};
```

Team helpers include:

| Helper | Meaning |
|---|---|
| `ncclTeamWorld` | all ranks in communicator |
| `ncclTeamLsa` | load-store-accessible local team |
| `ncclTeamRail` | rail team, equivalent to outer factor of LSA team |
| `ncclTeamRankIsMember` | membership test |
| `ncclTeamRankToTeam` | translate rank from one team to another |
| `ncclTeamRankToWorld` | translate team rank to world rank |
| `ncclTeamRankToLsa` | translate team rank to LSA rank |
| `ncclTeamInnerFactor`, `ncclTeamOuterFactor` | derive subteams from layout factors |
| `ncclTeamRankInDifference` | rank in set difference of parent and subset |

Use teams when writing kernels that choose local LSA behavior for some peers and GIN/network behavior
for others.

## Windows and device pointers

Host-side pointer helpers:

```c
ncclGetLsaMultimemDevicePointer(window, offset, &ptr);
ncclGetMultimemDevicePointer(window, offset, multimemHandle, &ptr);
ncclGetLsaDevicePointer(window, offset, lsaRank, &ptr);
ncclGetPeerDevicePointer(window, offset, peer, &ptr);
```

Device-side pointer helpers include:

```cpp
ncclGetLocalPointer(window, offset);
ncclGetLsaPointer(window, offset, peer);
ncclGetPeerPointer(window, offset, peer);
ncclGetPeerPointer(window, offset, team, peer);
ncclGetMultimemPointer(window, offset, multimemHandle);
ncclGetLsaMultimemPointer(window, offset, devComm);
```

Resource-buffer helper variants map `ncclDevResourceHandle` to local, LSA, peer, and multimem pointers.

## LSA: Load Store Access

LSA is the local peer-memory path exposed through windows and teams. The LSA allreduce example uses:

1. Device communicator with LSA barrier support.
2. Symmetric memory windows for send/recv buffers.
3. `ncclGetLsaPointer` or related pointer helpers to access peer memory.
4. Device-side barriers for correctness.
5. Manual reduction inside a CUDA kernel.

Typical use case: local GPUs in the same LSA team where peer memory is load/store accessible.

## GIN: GPU-Initiated Networking

GIN lets GPU kernels initiate network operations for remote peers. The examples cover:

- Pure GIN AlltoAll: use GIN for all peers.
- Hybrid AlltoAll: use LSA for local peers and GIN for remote peers.
- GIN barriers/signals to order puts and detect completion.

For multi-node RDMA GIN in NCCL EP docs, a recommended environment example is:

```bash
export NCCL_GIN_TYPE=3  # GDAKI
```

GIN availability depends on hardware, CUDA/NCCL build, net device support, and plugin/backend support.
Always query `ncclCommProperties_t` before assuming support.

## Example 1: LSA AllReduce structure

Host:

```cpp
ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
ncclCommQueryProperties(comm, &props);
if (!props.deviceApiSupport || props.nLsaTeams != 1) { /* fallback or exit */ }

ncclWindow_t sendWin, recvWin;
ncclCommWindowRegister(comm, d_send, bytes, &sendWin, NCCL_WIN_COLL_SYMMETRIC);
ncclCommWindowRegister(comm, d_recv, bytes, &recvWin, NCCL_WIN_COLL_SYMMETRIC);

ncclDevCommRequirements_t reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
reqs.lsaBarrierCount = NCCL_DEVICE_CTA_COUNT;
ncclDevComm_t devComm;
ncclDevCommCreate(comm, &reqs, &devComm);

simpleAllReduceKernel<<<NCCL_DEVICE_CTA_COUNT, NCCL_DEVICE_THREADS_PER_CTA, 0, stream>>>(
    sendWin, 0, recvWin, 0, count, devComm);
```

Device:

- Use LSA barriers for cross-GPU synchronization.
- Load peer values through LSA pointers.
- Reduce in-kernel.
- Store result to local output.

## Example 2: Pure GIN AlltoAll

The pure GIN example creates a device communicator with GIN support and uses network barriers/signals.
Its communication is network-only, so it is useful as a baseline for multi-node all-to-all behavior.

Checklist:

1. Query `props.ginType != NCCL_GIN_TYPE_NONE`.
2. Configure `reqs` for GIN contexts/signals/barriers.
3. Create `devComm`.
4. Launch kernel that uses GIN `put`, barriers, and completion signaling.
5. Destroy resources and windows after stream completion.

## Example 3: Hybrid LSA + GIN AlltoAll

Hybrid kernels classify peers:

- **Local**: ranks in the LSA team (`ncclTeamLsa`), typically same node or same NVLink domain.
- **Remote**: world ranks outside the local LSA team; use GIN.

This is the production-shaped pattern for multi-node custom kernels: choose the lowest-overhead path per
peer instead of forcing all communication through one mechanism.

## Device API troubleshooting

1. Query properties and print `deviceApiSupport`, `ginType`, `nLsaTeams`, `multimemSupport`.
2. Confirm buffers are allocated/registered as windows with compatible flags.
3. Confirm all ranks create compatible device communicator requirements.
4. Confirm kernel launch dimensions match the barriers/resources requested.
5. For GIN, confirm `NCCL_GIN_TYPE`, net plugin/device support, and multi-node network setup.
6. Use `NCCL_DEBUG=INFO` and `NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH` before assuming a kernel bug.
7. If a custom kernel hangs, distinguish barrier mismatch from network put/signal mismatch.

## Source modification map

| Task | Start files |
|---|---|
| add/query a property | `src/include/nccl_device/core.h`, `src/dev_runtime.cc` |
| change device communicator creation | `src/dev_runtime.cc`, `src/include/dev_runtime.h` |
| add LSA pointer/team helper | `src/include/nccl_device/core.h`, implementation headers under `nccl_device/impl` |
| change GIN host behavior | `src/gin/gin_host.cc`, `src/gin/gin_host_proxy.cc` |
| change net-backed GIN | `src/transport/net_ib/gdaki/*` |
| update examples | `docs/examples/06_device_api/*` |
