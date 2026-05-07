# 18 - NCCL Source File Index

Use this index to jump to the right source area before making claims or edits.

## Public API and examples

| Path | Purpose |
|---|---|
| `sources/nccl/src/nccl.h.in` | public C API template: types, errors, communicator APIs, collectives, memory/window/RMA/parameter APIs |
| `sources/nccl/README.md` | top-level overview, build/install/test commands |
| `sources/nccl/docs/examples/README.md` | examples overview and build/run variables |
| `sources/nccl/docs/examples/01_communicators` | communicator init patterns |
| `sources/nccl/docs/examples/02_point_to_point` | P2P ring pattern |
| `sources/nccl/docs/examples/03_collectives` | basic collective examples |
| `sources/nccl/docs/examples/04_user_buffer_registration` | buffer registration examples |
| `sources/nccl/docs/examples/05_symmetric_memory` | symmetric window examples |
| `sources/nccl/docs/examples/06_device_api` | Device API LSA/GIN/hybrid examples |
| `sources/nccl/docs/examples/common` | shared example utilities |

## Build and packaging

| Path | Purpose |
|---|---|
| `sources/nccl/Makefile` | top-level Make targets |
| `sources/nccl/CMakeLists.txt` | CMake build entry |
| `sources/nccl/makefiles/common.mk` | common Make settings, CUDA arch defaults |
| `sources/nccl/makefiles/version.mk` | NCCL version (`2.30.4` in this checkout) |
| `sources/nccl/pkg/debian` | Debian package files |
| `sources/nccl/pkg/redhat` | RPM package files |
| `sources/nccl/pkg/txz`, `pkg/srctxz` | tarball packaging |

## Core host runtime

| Path | Purpose |
|---|---|
| `sources/nccl/src/init.cc` | communicator initialization/lifecycle/split/shrink/grow/revoke |
| `sources/nccl/src/collectives.cc` | public collective/P2P/RMA API wrappers and string conversions |
| `sources/nccl/src/group.cc` | group semantics, async job handling, grouped launch |
| `sources/nccl/src/enqueue.cc` | task append, prepare, schedule, upload, launch |
| `sources/nccl/src/channel.cc` | channel helpers |
| `sources/nccl/src/bootstrap.cc` | bootstrap communication before transports are ready |
| `sources/nccl/src/debug.cc` | logging/debug subsystem |
| `sources/nccl/src/allocator.cc` | `ncclMemAlloc` / `ncclMemFree` |
| `sources/nccl/src/mem_manager.cc` | tracked communicator memory, suspend/resume/stats |
| `sources/nccl/src/dev_runtime.cc` | Device API host runtime |
| `sources/nccl/src/enhcompat.cc` | enhanced compatibility behavior |

## Core internal headers

| Path | Purpose |
|---|---|
| `sources/nccl/src/include/comm.h` | central structs: communicator, channel, tasks, planner, plans |
| `sources/nccl/src/include/info.h` | `ncclInfo` API-call descriptor |
| `sources/nccl/src/include/group.h` | group helpers |
| `sources/nccl/src/include/enqueue.h` | enqueue/launch declarations |
| `sources/nccl/src/include/collectives.h` | collective enums/constants |
| `sources/nccl/src/include/device.h` | device work structures/constants |
| `sources/nccl/src/include/transport.h` | transport interfaces |
| `sources/nccl/src/include/proxy.h` | proxy structures/messages |
| `sources/nccl/src/include/graph.h` | topology graph declarations |
| `sources/nccl/src/include/param.h` | legacy/internal `NCCL_PARAM` macro |
| `sources/nccl/src/include/param/param.h` | typed parameter registry macros |
| `sources/nccl/src/include/checks.h` | error-checking helpers |
| `sources/nccl/src/include/argcheck.h` | argument checks |

## Device kernels

| Path | Purpose |
|---|---|
| `sources/nccl/src/device/common.cu` | generic device kernel entry |
| `sources/nccl/src/device/common_kernel.h` | work-batch execution and dispatch framework |
| `sources/nccl/src/device/primitives.h` | protocol primitive abstraction |
| `sources/nccl/src/device/prims_simple.h` | SIMPLE protocol |
| `sources/nccl/src/device/prims_ll.h` | LL protocol |
| `sources/nccl/src/device/prims_ll128.h` | LL128 protocol |
| `sources/nccl/src/device/all_reduce.h` | AllReduce device implementation |
| `sources/nccl/src/device/broadcast.h` | Broadcast device implementation |
| `sources/nccl/src/device/reduce.h` | Reduce device implementation |
| `sources/nccl/src/device/reduce_scatter.h` | ReduceScatter device implementation |
| `sources/nccl/src/device/all_gather.h` | AllGather device implementation |
| `sources/nccl/src/device/all_gather_v.h` | AllGatherV device implementation |
| `sources/nccl/src/device/sendrecv.h` | P2P send/recv device implementation |
| `sources/nccl/src/device/reduce_kernel.h` | reduce kernel helpers |
| `sources/nccl/src/device/generate.py` | device code generation support |
| `sources/nccl/src/device/onerank.cu` | single-rank degenerate path |

## Device API

| Path | Purpose |
|---|---|
| `sources/nccl/src/include/nccl_device.h` | Device API umbrella header |
| `sources/nccl/src/include/nccl_device/core.h` | core Device API types and host/device helpers |
| `sources/nccl/src/include/nccl_device/coop.h` | cooperative helpers |
| `sources/nccl/src/include/nccl_device/barrier.h` | barrier API |
| `sources/nccl/src/include/nccl_device/ptr.h` | pointer/window helpers |
| `sources/nccl/src/include/nccl_device/reduce_copy.h` | reduce/copy helpers |
| `sources/nccl/src/include/nccl_device/ll_a2a.h` | low-latency all-to-all helpers |
| `sources/nccl/src/include/nccl_device/impl/*` | inline implementation/types |
| `sources/nccl/src/devcomm/*` | Device API compatibility versions |

## Topology and graph

| Path | Purpose |
|---|---|
| `sources/nccl/src/graph/topo.cc` | topology discovery/build |
| `sources/nccl/src/graph/paths.cc` | path quality/bandwidth computation |
| `sources/nccl/src/graph/search.cc` | topology graph search |
| `sources/nccl/src/graph/connect.cc` | channel connectivity from selected graphs |
| `sources/nccl/src/graph/rings.cc` | ring helpers |
| `sources/nccl/src/graph/trees.cc` | tree helpers |
| `sources/nccl/src/graph/tuning.cc` | algorithm/protocol cost model |
| `sources/nccl/src/graph/xml.cc` | topology XML |

## Transports and proxy

| Path | Purpose |
|---|---|
| `sources/nccl/src/transport.cc` | transport multiplexer and connection setup |
| `sources/nccl/src/transport/p2p.cc` | CUDA P2P transport |
| `sources/nccl/src/transport/shm.cc` | shared-memory transport |
| `sources/nccl/src/transport/net.cc` | network transport layer |
| `sources/nccl/src/transport/net_socket.cc` | socket net implementation |
| `sources/nccl/src/transport/net_ib/init.cc` | IB init/device discovery |
| `sources/nccl/src/transport/net_ib/connect.cc` | IB connection/QP setup |
| `sources/nccl/src/transport/net_ib/p2p.cc` | IB P2P send/recv behavior |
| `sources/nccl/src/transport/net_ib/reg.cc` | IB memory registration |
| `sources/nccl/src/transport/net_ib/common.cc` | IB common helpers |
| `sources/nccl/src/transport/net_ib/p2p_resiliency*.cc` | IB resiliency/failover/recovery |
| `sources/nccl/src/transport/net_ib/gdaki/*` | GDAKI/GIN IB support |
| `sources/nccl/src/transport/coll_net.cc` | CollNet transport |
| `sources/nccl/src/transport/nvls.cc` | NVLS transport |
| `sources/nccl/src/proxy.cc` | proxy progress engine |

## Plugins

| Path | Purpose |
|---|---|
| `sources/nccl/src/plugin/plugin_open.cc` | dynamic library open/probe helpers |
| `sources/nccl/src/plugin/net.cc` | net plugin integration |
| `sources/nccl/src/plugin/tuner.cc` | tuner plugin integration |
| `sources/nccl/src/plugin/profiler.cc` | profiler plugin integration |
| `sources/nccl/src/plugin/env.cc` | env plugin integration |
| `sources/nccl/src/include/plugin/nccl_net.h` | net plugin ABI |
| `sources/nccl/src/include/plugin/nccl_tuner.h` | tuner plugin ABI |
| `sources/nccl/src/include/plugin/nccl_profiler.h` | profiler plugin ABI |
| `sources/nccl/src/include/plugin/nccl_env.h` | env plugin ABI |
| `sources/nccl/plugins/net` | net plugin docs/examples |
| `sources/nccl/plugins/tuner` | tuner plugin docs/examples |
| `sources/nccl/plugins/profiler` | profiler plugin docs/examples |
| `sources/nccl/plugins/env` | env plugin docs/examples |
| `sources/nccl/plugins/mixed` | combined plugin example |

## Registration, memory, RMA, RAS

| Path | Purpose |
|---|---|
| `sources/nccl/src/register/register.cc` | generic registration |
| `sources/nccl/src/register/coll_reg.cc` | collective registration |
| `sources/nccl/src/register/sendrecv_reg.cc` | P2P registration |
| `sources/nccl/src/rma/*` | one-sided RMA/signal implementation |
| `sources/nccl/src/ras/ras.cc` | RAS main thread/message handling |
| `sources/nccl/src/ras/rasnet.cc` | RAS networking/keepalive/retry |
| `sources/nccl/src/ras/peers.cc` | peer state tracking |
| `sources/nccl/src/ras/collectives.cc` | RAS collectives |
| `sources/nccl/src/ras/client.cc` | RAS diagnostic client |

## Scheduler extensions and special subsystems

| Path | Purpose |
|---|---|
| `sources/nccl/src/scheduler/symmetric_sched.cc` | symmetric collective scheduling |
| `sources/nccl/src/scheduler/allgatherv_sched.cc` | AllGatherV scheduling |
| `sources/nccl/src/ce_coll.cc` | CE/special collective support |
| `sources/nccl/src/gin/gin_host.cc` | GIN host support |
| `sources/nccl/src/gin/gin_host_proxy.cc` | GIN proxy support |
| `sources/nccl/src/init_nvtx.cc` | NVTX init/disable behavior |
| `sources/nccl/src/misc/*` | sockets, CUDA wrappers, NVML/IB wrappers, utilities |
| `sources/nccl/src/os/*` | OS abstractions |

## Bindings and contrib

| Path | Purpose |
|---|---|
| `sources/nccl/bindings/nccl4py` | Python/Cython bindings |
| `sources/nccl/bindings/ir` | Device API/IR binding wrapper |
| `sources/nccl/contrib/nccl_ep` | Expert Parallelism dispatch/combine extension |

## Fast grep recipes

```bash
# Find public API declarations
grep -n "ncclResult_t .*nccl" sources/nccl/src/nccl.h.in

# Find environment parameters
grep -R "NCCL_PARAM(" -n sources/nccl/src

# Trace a public operation
grep -R "ncclAllReduce\|ncclFuncAllReduce" -n sources/nccl/src

# Trace a plugin ABI version
grep -R "ncclNet_v\|ncclProfiler_v\|ncclEnvPlugin_v\|ncclTuner" -n sources/nccl

# Find Device API symbols
grep -R "ncclDevCommCreate\|ncclCommQueryProperties" -n sources/nccl/src sources/nccl/docs/examples
```
