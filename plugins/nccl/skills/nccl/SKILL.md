---
name: nccl
description: >
  Comprehensive reference documentation and skill for NVIDIA NCCL (Collective Communications Library),
  the GPU communication library for multi-GPU and multi-node collectives. Use this skill whenever the
  user mentions NCCL, all-reduce, all-gather, reduce-scatter, broadcast, gather/scatter, all-to-all,
  ncclSend/ncclRecv, communicator initialization, CUDA stream group semantics, distributed training
  communication, NVLink/NVSwitch/InfiniBand/TCP transport behavior, NCCL environment variables,
  debugging NCCL hangs or performance, NCCL plugins (net/tuner/profiler/env), Device API, GIN, LSA,
  symmetric memory, user buffer registration, RAS, or NCCL source-code internals.
version: 2.30.4
---

# NCCL - NVIDIA Collective Communications Library

## How to use this skill

Use this skill as a code-aware NCCL reference manual. When a user asks about NCCL usage, debugging,
performance, plugin development, or internals, first identify which layer they are working at:

1. **Application API usage**: communicator lifecycle, collectives, P2P, group semantics, errors.
2. **Advanced user features**: buffer registration, symmetric memory windows, one-sided RMA, Device API.
3. **Runtime tuning/debugging**: environment variables, topology, algorithms/protocols, transport choice.
4. **Plugin development**: net, tuner, profiler, env, GIN/plugin loading.
5. **Source-code internals**: init, group/enqueue/scheduler, device kernels, topology graph, proxy/transport.
6. **Bindings/contrib**: nccl4py and NCCL EP for MoE expert parallel communication.

When the answer needs precise behavior, cite the source file path from `sources/nccl` and, if you read
current files, cite `path:line`. If a reference here names a symbol and the user is about to modify code,
verify the current source first because NCCL internals change quickly.

## NCCL in one page

NCCL is a standalone GPU communication library implementing collective and point-to-point primitives
optimized for PCIe, NVLink, NVSwitch, InfiniBand Verbs, and TCP/IP sockets. It supports arbitrary GPU
counts in a single node or across multiple nodes, and can be used from single-process, multi-threaded,
or multi-process/MPI applications.

**Core public API families:**
- Communicator lifecycle: `ncclGetUniqueId`, `ncclCommInitRank`, `ncclCommInitAll`,
  `ncclCommInitRankConfig`, `ncclCommInitRankScalable`, `ncclCommFinalize`, `ncclCommDestroy`,
  `ncclCommAbort`, `ncclCommSplit`, `ncclCommShrink`, `ncclCommGrow`, `ncclCommRevoke`.
- Collectives: `ncclReduce`, `ncclBroadcast`, `ncclAllReduce`, `ncclReduceScatter`,
  `ncclAllGather`, `ncclAlltoAll`, `ncclGather`, `ncclScatter`.
- P2P and one-sided: `ncclSend`, `ncclRecv`, `ncclPutSignal`, `ncclSignal`, `ncclWaitSignal`.
- Group semantics: `ncclGroupStart`, `ncclGroupEnd`, `ncclGroupSimulateEnd`.
- Memory/registration: `ncclMemAlloc`, `ncclMemFree`, `ncclCommRegister`, `ncclCommDeregister`,
  `ncclCommWindowRegister`, `ncclCommWindowDeregister`, `ncclWinGetUserPtr`.
- Device API: `ncclCommQueryProperties`, `ncclDevCommCreate`, `ncclDevCommDestroy`, LSA/GIN pointer,
  team, barrier, and device communication helpers from `nccl_device` headers.
- Parameters: `ncclParamBind`, typed `ncclParamGet*`, `ncclParamGetParameter`,
  `ncclParamGetAllParameterKeys`, `ncclParamDumpAll`.

## Quick usage patterns

### Single-process multi-GPU allreduce
```cpp
int ndev = 0;
cudaGetDeviceCount(&ndev);
std::vector<ncclComm_t> comms(ndev);
ncclCommInitAll(comms.data(), ndev, nullptr);

ncclGroupStart();
for (int r = 0; r < ndev; ++r) {
  cudaSetDevice(r);
  ncclAllReduce(send[r], recv[r], count, ncclFloat32, ncclSum, comms[r], streams[r]);
}
ncclGroupEnd();

for (int r = 0; r < ndev; ++r) cudaStreamSynchronize(streams[r]);
for (int r = 0; r < ndev; ++r) {
  ncclCommFinalize(comms[r]);
  ncclCommDestroy(comms[r]);
}
```

### Multi-process initialization shape
```cpp
ncclUniqueId id;
if (rank == 0) ncclGetUniqueId(&id);
// Broadcast id with MPI, sockets, shared memory, or another launcher mechanism.
cudaSetDevice(local_device);
ncclComm_t comm;
ncclCommInitRank(&comm, world_size, id, rank);
```

### P2P ring pattern must use grouping
```cpp
ncclGroupStart();
ncclSend(sendbuf, count, ncclFloat32, next_rank, comm, stream);
ncclRecv(recvbuf, count, ncclFloat32, prev_rank, comm, stream);
ncclGroupEnd();
```

### Debugging first line
```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,NET,TUNING ./your_app
```

## Documentation map

### User-facing API and usage
- [01 - Overview, Build, Install](references/01-overview-build-install.md): what NCCL is, build/package/test flow, examples layout.
- [02 - Public API and Communicators](references/02-public-api-communicators.md): public types, errors, config, initialization, lifecycle, split/shrink/grow.
- [03 - Collectives, P2P, and Groups](references/03-collectives-p2p-groups.md): operation semantics, in-place rules, count/layout rules, P2P deadlocks, grouping.
- [04 - Memory Registration, Symmetric Windows, RMA](references/04-memory-registration-symmetric-rma.md): `ncclMemAlloc`, buffer registration, symmetric windows, suspend/resume, signals.
- [05 - Device API, LSA, GIN](references/05-device-api-lsa-gin.md): device communicators, teams, windows, LSA/GPU-Initiated Networking, examples.
- [06 - Example Patterns](references/06-example-patterns.md): single process, pthread, MPI, AllReduce, ring P2P, advanced examples.

### Runtime configuration and performance
- [07 - Environment Variables and Parameters](references/07-environment-parameters.md): source-derived parameter map and public parameter API.
- [08 - Debugging, Troubleshooting, Profiling](references/08-debugging-troubleshooting-profiling.md): hangs, async errors, transport visibility, profiler plugins, logging.
- [09 - Algorithms, Protocols, Tuning](references/09-algorithms-protocols-tuning.md): Ring/Tree/CollNet/NVLS/PAT, LL/LL128/SIMPLE, cost model and tuner plugin hooks.

### Source internals
- [10 - Source Architecture and Communicator Lifecycle](references/10-source-architecture-communicator-lifecycle.md): key source files, `ncclComm`, init pipeline.
- [11 - Group, Enqueue, Scheduler, Launch Pipeline](references/11-group-enqueue-scheduler-launch.md): `ncclInfo` → tasks → kernel plans → CUDA launch/proxy ops.
- [12 - Device Kernels and Protocol Primitives](references/12-device-kernels-protocols.md): generated kernels, `RunWork*`, SIMPLE/LL/LL128 primitives.
- [13 - Topology, Graph Search, Channels](references/13-topology-graph-channels.md): topology discovery, ring/tree/NVLS/CollNet graph search, channel connection.
- [14 - Transports, Proxy, Networking](references/14-transports-proxy-networking.md): P2P/SHM/NET/CollNet/NVLS transports, proxy progress, sockets/IB.

### Extension surfaces
- [15 - Plugin Development](references/15-plugin-development.md): net, tuner, profiler, env, mixed plugins, ABI/versioning/load names.
- [16 - RAS and Fault Handling](references/16-ras-fault-handling.md): RAS thread, peers, keepalive, communicator revoke/shrink patterns.
- [17 - Python Bindings and NCCL EP](references/17-python-bindings-nccl-ep.md): nccl4py, NCCL EP dispatch/combine for MoE.
- [18 - Source File Index](references/18-source-file-index.md): source-code map by module for fast navigation.

## Answering rules for NCCL tasks

- For API questions, explain both **host-side enqueue semantics** and **CUDA stream completion semantics**. NCCL calls usually return after enqueueing work, not after communication finishes.
- For P2P questions, check whether multiple sends/recvs must progress concurrently. If yes, recommend `ncclGroupStart/End`.
- For multi-GPU single-thread code, default to grouping per-device NCCL calls.
- For multi-process code, make `cudaSetDevice(localRank)` before communicator init explicit.
- For hangs, ask for `NCCL_DEBUG=INFO`, rank count, launcher, device mapping, network interface/NIC, and the last log lines per rank.
- For performance, distinguish topology/algorithm/protocol/channel issues from application stream synchronization and buffer registration issues.
- For source changes, start from `sources/nccl/src/nccl.h.in`, `collectives.cc`, `group.cc`, `enqueue.cc`, `init.cc`, and `src/include/comm.h` depending on the layer.
