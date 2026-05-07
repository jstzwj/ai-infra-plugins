# 07 - Environment Variables and Runtime Parameters

## Primary source files

- `sources/nccl/src/include/param.h`: `NCCL_PARAM(name, env, default)` macro.
- `sources/nccl/src/include/param/param.h`: newer typed parameter definitions.
- `sources/nccl/src/param/param.cc`: public parameter registry flags.
- `sources/nccl/src/param/c_api.cc`: public parameter API glue.
- `sources/nccl/src/nccl.h.in`: `ncclParam*` public APIs.
- Parameter definitions across `sources/nccl/src/**/*.cc`.

## How NCCL parameters are named

Most internal parameters are declared with:

```c
NCCL_PARAM(Name, "ENV_SUFFIX", defaultValue)
```

This resolves to an environment variable named:

```text
NCCL_<ENV_SUFFIX>
```

Example:

```c
NCCL_PARAM(CommBlocking, "COMM_BLOCKING", NCCL_CONFIG_UNDEF_INT);
```

Environment variable:

```bash
export NCCL_COMM_BLOCKING=1
```

## Parameter caching and dumping

Public header APIs:

```c
ncclParamBind(&handle, "NCCL_DEBUG");
ncclParamGetStr(handle, &value);
ncclParamGetParameter("NCCL_DEBUG", &value, &valueLen);
ncclParamGetAllParameterKeys(&table, &tableLen);
ncclParamDumpAll();
```

Special registry variables from `param.cc`:

| Variable | Meaning |
|---|---|
| `NCCL_PARAM_DUMP_ALL=true` | include private/internal parameters in parameter dumps |
| `NCCL_NO_CACHE` | disables parameter caching for named keys or `ALL` |

Most users should set environment variables before process start. Hot-changing variables during a run is
not a reliable tuning mechanism unless a specific parameter API path says so.

## High-value runtime variables

These are common variables users ask about. Some are source-derived internal parameters; public support
and exact behavior can vary by NCCL version, so verify in current source for code changes.

### Debug/logging

| Variable | Source/default | Use |
|---|---|---|
| `NCCL_DEBUG` | documented runtime variable | `WARN`, `INFO`, etc.; start with `INFO` |
| `NCCL_DEBUG_SUBSYS` | documented runtime variable | restrict logs, e.g. `INIT,GRAPH,NET,TUNING` |
| `NCCL_SET_THREAD_NAME` | `debug.cc`, default `0` | set helper/proxy thread names |
| `NCCL_NVTX_DISABLE` | `init_nvtx.cc`, default `0` | disable NVTX instrumentation |

### Communicator/init behavior

| Variable | Source/default | Use |
|---|---|---|
| `NCCL_COMM_BLOCKING` | `init.cc`, config undef | default blocking behavior for communicators |
| `NCCL_GROUP_CUDA_STREAM` | `init.cc` | group CUDA stream behavior |
| `NCCL_CHECK_POINTERS` | `init.cc`, `0` | pointer validation/debugging |
| `NCCL_RUNTIME_CONNECT` | `init.cc`, `1` | runtime transport connection behavior |
| `NCCL_SET_CPU_STACK_SIZE` | `init.cc`, `1` | CPU stack-size management |
| `NCCL_SET_STACK_SIZE` | `init.cc`, `0` | stack-size override behavior |
| `NCCL_MULTI_RANK_GPU_ENABLE` | `init.cc`, `0` | multiple ranks per GPU behavior |
| `NCCL_UID_STAGGER_RATE` | `bootstrap.cc`, `7000` | stagger UID/bootstrap behavior |
| `NCCL_UID_STAGGER_THRESHOLD` | `bootstrap.cc`, `256` | threshold for UID staggering |
| `NCCL_OOB_NET_ENABLE` | `bootstrap.cc`, `0` | out-of-band network bootstrap |

### Channels, CTAs, launch

| Variable | Source/default | Use |
|---|---|---|
| `NCCL_MIN_NCHANNELS` | `graph/connect.cc`, `-2` | minimum channels |
| `NCCL_MAX_NCHANNELS` | `graph/connect.cc`, `-2` | maximum channels |
| `NCCL_MIN_NRINGS` | `graph/connect.cc`, `-2` | minimum rings legacy/tuning alias |
| `NCCL_MAX_NRINGS` | `graph/connect.cc`, `-2` | maximum rings legacy/tuning alias |
| `NCCL_NTHREADS` | `graph/tuning.cc`, `-2` | threads for algorithms/protocols |
| `NCCL_LL128_NTHREADS` | `graph/tuning.cc`, `-2` | LL128 thread count |
| `NCCL_MIN_CTAS` | `init.cc`, config undef | minimum CTAs |
| `NCCL_MAX_CTAS` | `init.cc`, config undef | maximum CTAs |
| `NCCL_CGA_CLUSTER_SIZE` | `init.cc`, config undef | CGA cluster size |
| `NCCL_L1_SHARED_MEMORY_CARVEOUT` | `enqueue.cc`, `0` | launch shared-memory carveout |
| `NCCL_LAUNCH_ORDER_IMPLICIT` | `enqueue.cc`, `0` | launch ordering mode |
| `NCCL_MEM_SYNC_DOMAIN` | `enqueue.cc`, CUDA remote domain default | CUDA launch mem sync domain |
| `NCCL_LAUNCH_RACE_FATAL` | `misc/strongstream.cc`, `1` | launch race fatal behavior |

### Algorithm/protocol selection

`graph/tuning.cc` parses prefix-list syntax for algorithm/protocol filters. Examples from source comments:

```bash
export NCCL_ALGO="ring,collnetdirect;allreduce:tree,collnetdirect;broadcast:ring"
export NCCL_PROTO="LL,Simple;allreduce:^LL"
export NCCL_PROTO="^LL128;allreduce:LL128"
```

Syntax:

- Entries separated by `;`.
- Optional prefix (`allreduce`, `broadcast`, etc.) followed by `:`.
- Elements separated by `,`.
- Leading `^` means exclude listed elements instead of include.
- A prefix-less first entry applies to all operation types.

Other algorithm variables:

| Variable | Source/default | Use |
|---|---|---|
| `NCCL_COLLNET_ENABLE` | `init.cc`, config undef | CollNet enable override |
| `NCCL_COLLNET_NODE_THRESHOLD` | `init.cc`, `2` | CollNet node threshold |
| `NCCL_NVLS_ENABLE` | `transport/nvls.cc`, `2` | NVLS auto/enable/disable behavior |
| `NCCL_NVLS_NCHANNELS` | `init.cc`, config undef | NVLS channel count |
| `NCCL_NVLS_CHUNKSIZE` | `transport/nvls.cc`, `128*1024` | NVLS chunk size |
| `NCCL_NVLSTREE_MAX_CHUNKSIZE` | `transport/nvls.cc`, `-2` | NVLS tree max chunk |
| `NCCL_PAT_ENABLE` | `graph/tuning.cc`, `2` | PAT enable/auto |
| `NCCL_MNNVL_ENABLE` | `init.cc`, `2` | MNNVL enable/auto |
| `NCCL_MNNVL_UUID` | `init.cc`, `-1` | MNNVL UUID override |
| `NCCL_MNNVL_CLIQUE_ID` | `init.cc`, `-1` | MNNVL clique ID override |
| `NCCL_MNNVL_CROSS_CLIQUE` | `init.cc`, `0` | cross-clique MNNVL behavior |

### P2P and local transports

| Variable | Source/default | Use |
|---|---|---|
| `NCCL_P2P_READ_ENABLE` | `transport/p2p.cc`, `-2` | P2P read behavior |
| `NCCL_P2P_DIRECT_DISABLE` | `transport/p2p.cc`, `0` | disable direct P2P path |
| `NCCL_P2P_USE_CUDA_MEMCPY` | `transport/p2p.cc`, `0` | use CUDA memcpy path |
| `NCCL_P2P_LL_THRESHOLD` | `enqueue.cc`, `16384` | LL threshold for P2P |
| `NCCL_P2P_MAX_PEERS` | `init.cc`, config undef | max P2P peers |
| `NCCL_P2P_PXN_LEVEL` | `graph/search.cc`, `2` | PXN level for P2P |
| `NCCL_PXN_DISABLE` | `graph/paths.cc`, `0` | disable PXN |
| `NCCL_PXN_C2C` | `graph/paths.cc`, `1` | C2C PXN behavior |
| `NCCL_SHM_DISABLE` | `transport/shm.cc`, `0` | disable SHM transport |
| `NCCL_SHM_USE_CUDA_MEMCPY` | `transport/shm.cc`, `0` | SHM CUDA memcpy path |
| `NCCL_SHM_MEMCPY_MODE` | `transport/shm.cc`, send-side default | SHM memcpy side/mode |
| `NCCL_SHM_LOCALITY` | `transport/shm.cc`, recv-side default | SHM locality policy |

### Network and socket transport

| Variable | Source/default | Use |
|---|---|---|
| `NCCL_NET_PLUGIN` | plugin loader | select/load external net plugin library suffix/path |
| `NCCL_NET` | net plugin docs | select network implementation name |
| `NCCL_NET_GDR_READ` | `graph/paths.cc`, `-2` | GPU Direct RDMA read behavior |
| `NCCL_NET_GDR_C2C` | `graph/paths.cc`, `1` | C2C GDR behavior |
| `NCCL_NET_FORCE_FLUSH` | `graph/paths.cc`, `0` | force network flush behavior |
| `NCCL_NET_DISABLE_INTRA` | `graph/paths.cc`, `0` | disable intra-node network use |
| `NCCL_NET_SHARED_BUFFERS` | `transport/net.cc`, `-2` | network shared buffer mode |
| `NCCL_NET_SHARED_COMMS` | `transport/net.cc`, `1` | network shared comms |
| `NCCL_NET_OPTIONAL_RECV_COMPLETION` | `transport/net.cc`, `1` | optional recv completion for LL/LL128 |
| `NCCL_NET_OVERHEAD` | `graph/tuning.cc`, `-2` | network post overhead model |
| `NCCL_NCHANNELS_PER_NET_PEER` | `init.cc`, config undef | channels per net peer |
| `NCCL_P2P_NET_CHUNKSIZE` | `init.cc`, 128 KiB | net P2P chunk size |
| `NCCL_SOCKET_NTHREADS` | `transport/net_socket.cc`, `-2` | socket helper threads |
| `NCCL_NSOCKS_PERTHREAD` | `transport/net_socket.cc`, `-2` | sockets per helper thread |
| `NCCL_SOCKET_INLINE` | `transport/net_socket.cc`, 128 B | socket inline size |
| `NCCL_SOCKET_MIN_TASKSIZE` | `transport/net_socket.cc`, 64 KiB | min socket task size |
| `NCCL_SOCKET_RETRY_CNT` | `misc/socket.cc`, `34` | socket retry count |
| `NCCL_SOCKET_RETRY_SLEEP_MSEC` | `misc/socket.cc`, `100` | socket retry sleep |
| `NCCL_SOCKET_POLL_TIMEOUT_MSEC` | `misc/socket.cc`, `0` | socket poll timeout |
| `NCCL_SOCKET_RCVBUF` | `misc/socket.cc`, `-1` | socket receive buffer |
| `NCCL_SOCKET_SNDBUF` | `misc/socket.cc`, `-1` | socket send buffer |

### InfiniBand / Verbs transport

| Variable | Source/default | Use |
|---|---|---|
| `NCCL_IB_DISABLE` | `transport/net_ib/init.cc`, `0` | disable IB transport |
| `NCCL_IB_GID_INDEX` | `transport/net_ib/connect.cc`, `-1` | GID index |
| `NCCL_IB_ROUTABLE_FLID_GID_INDEX` | `connect.cc`, `1` | routable FLID GID index |
| `NCCL_IB_ROCE_VERSION_NUM` | `connect.cc`, `2` | RoCE version |
| `NCCL_IB_TIMEOUT` | `connect.cc`, `20` | IB timeout |
| `NCCL_IB_RETRY_CNT` | `connect.cc`, `7` | retry count |
| `NCCL_IB_PKEY` | `connect.cc`, `0` | partition key |
| `NCCL_IB_USE_INLINE` | `connect.cc`, `0` | inline send use |
| `NCCL_IB_SL` | `connect.cc`, `-1` | service level |
| `NCCL_IB_TC` | `connect.cc`, `-1` | traffic class |
| `NCCL_IB_FIFO_TC` | `connect.cc`, `-1` | FIFO traffic class |
| `NCCL_IB_ECE_ENABLE` | `connect.cc`, `1` | ECE behavior |
| `NCCL_IB_QPS_PER_CONNECTION` | `connect.cc`, `1` | QPs per connection |
| `NCCL_IB_PCI_RELAXED_ORDERING` | `init.cc`, `2` | PCI relaxed ordering |
| `NCCL_IB_ADAPTIVE_ROUTING` | `init.cc`, `-2` | adaptive routing |
| `NCCL_IB_DATA_DIRECT` | `init.cc`, `1` | data direct |
| `NCCL_IB_OOO_RQ` | `init.cc`, `0` | out-of-order receive queue |
| `NCCL_IB_MERGE_VFS` | `init.cc`, `1` | merge virtual functions |
| `NCCL_IB_MERGE_NICS` | `init.cc`, `1` | merge NICs |
| `NCCL_IB_DEVICE_PCI_ORDER` | `init.cc`, `1` | PCI ordering for devices |
| `NCCL_IB_AR_THRESHOLD` | `p2p.cc`, `-2` | adaptive routing threshold |
| `NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME` | `p2p.cc`, `-2` | receiver matching scheme |
| `NCCL_IB_WARN_RAIL_LOCAL` | `connect.cc`, `0` | rail-local warning |

### CUDA memory / DMA-BUF / GDRCopy

| Variable | Source/default | Use |
|---|---|---|
| `NCCL_CUMEM_ENABLE` | `misc/cudawrap.cc`, `-2` | cuMem allocation path |
| `NCCL_CUMEM_HOST_ENABLE` | `misc/cudawrap.cc`, `-1` | cuMem host path |
| `NCCL_DMABUF_ENABLE` | `init.cc`, `1` | DMA-BUF support |
| `NCCL_GDRCOPY_ENABLE` | `init.cc`, `0` | GDRCopy enable |
| `NCCL_GDRCOPY_FIFO_ENABLE` | `init.cc`, `1` | GDRCopy FIFO behavior |
| `NCCL_GDRCOPY_SYNC_ENABLE` | `transport/net.cc`, `1` | GDRCopy sync |
| `NCCL_GDRCOPY_FLUSH_ENABLE` | `transport/net.cc`, `0` | GDRCopy flush |
| `NCCL_GDR_FLUSH_DISABLE` | `transport/net_ib/connect.cc`, `0` | disable GDR flush |
| `NCCL_LEGACY_CUDA_REGISTER` | `transport/p2p.cc`, `0` | legacy CUDA registration path |
| `NCCL_LOCAL_REGISTER` | `register/register.cc`, `1` | local registration enable |
| `NCCL_MULTI_SEGMENT_REGISTER` | `transport/generic.cc`, `1` | multi-segment registration |
| `NCCL_SINGLE_PROC_MEM_REG_ENABLE` | `group.cc`, `0` | single-process mem registration |

### Device API / GIN / symmetric features

| Variable | Source/default | Use |
|---|---|---|
| `NCCL_WIN_ENABLE` | `init.cc`, `1` | window feature enable |
| `NCCL_WIN_STRIDE` | `dev_runtime.cc`, `-1` | window stride |
| `NCCL_ENABLE_VERSION_CHECK` | `dev_runtime.cc`, `1` | Device API version check |
| `NCCL_ELASTIC_BUFFER_REGISTER` | `dev_runtime.cc`, `1` | elastic buffer registration |
| `NCCL_SYM_REUSE_SYSMEM_HANDLES` | `dev_runtime.cc`, `0` | reuse sysmem handles |
| `NCCL_LSA_TEAM_SIZE` | `dev_runtime.cc`, `0` | LSA team size override |
| `NCCL_GIN_ENABLE` | `gin/gin_host.cc`, `1` | GIN enable |
| `NCCL_GIN_TYPE` | `transport/net_ib/gin.cc`, `-1` | GIN type override |
| `NCCL_GIN_NCONNECTIONS` | `gin/gin_host.cc`, `-2` | GIN connection count |
| `NCCL_GIN_PROXY_QUEUE_SIZE` | `gin/gin_host_proxy.cc`, `-1` | GIN proxy queue size |
| `NCCL_GIN_PLUGIN_REF_COUNT` | `plugin/gin.cc`, `0` | plugin ref count debug |
| `NCCL_NUM_RMA_CTX` | `init.cc`, config undef | RMA contexts |
| `NCCL_RMA_PROXY_QUEUE_SIZE` | `rma/rma_proxy.cc`, `-1` | RMA proxy queue size |

### Plugin variables

| Variable | Use |
|---|---|
| `NCCL_NET_PLUGIN` | external net plugin selection/path/suffix |
| `NCCL_TUNER_PLUGIN` | tuner plugin selection/path/suffix |
| `NCCL_PROFILER_PLUGIN` | profiler plugin selection/path/suffix |
| `NCCL_ENV_PLUGIN` | environment plugin selection/path/suffix |
| `NCCL_NET_PLUGIN_REF_COUNT` | net plugin ref count behavior/debug |

## Practical tuning order

1. Start with no overrides and `NCCL_DEBUG=INFO`.
2. Confirm topology and transport selection in logs.
3. Use `NCCL_DEBUG_SUBSYS=INIT,GRAPH,NET,TUNING` for focused logs.
4. Only then test algorithm/protocol/channel overrides.
5. Change one variable at a time and validate with `nccl-tests`.
6. Remove overrides that do not improve representative workloads.

## Warning about internal parameters

Many variables above are internal/source-derived. They are valuable for debugging and code reading, but
not all are stable public contract. For production guidance, prefer documented NCCL user-guide variables
and communicator config fields. For source work, verify current definitions with:

```bash
grep -R "NCCL_PARAM(" -n sources/nccl/src
```
