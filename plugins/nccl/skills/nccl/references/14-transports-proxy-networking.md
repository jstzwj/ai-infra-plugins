# 14 - Transports, Proxy Progress, Networking, SHM, P2P, CollNet, and NVLS

## Primary source files

- `sources/nccl/src/transport.cc`: transport multiplexer and P2P setup.
- `sources/nccl/src/include/transport.h`: transport interface structures.
- `sources/nccl/src/transport/p2p.cc`: CUDA P2P/direct GPU transport.
- `sources/nccl/src/transport/shm.cc`: shared-memory transport.
- `sources/nccl/src/transport/net.cc`: generic network transport layer.
- `sources/nccl/src/transport/net_socket.cc`: socket network implementation.
- `sources/nccl/src/transport/net_ib/*`: InfiniBand Verbs/RDMA implementation.
- `sources/nccl/src/transport/coll_net.cc`: CollNet transport.
- `sources/nccl/src/transport/nvls.cc`: NVLS transport.
- `sources/nccl/src/proxy.cc`: CPU proxy progress engine.
- `sources/nccl/src/include/proxy.h`: proxy structures/messages.
- `sources/nccl/src/plugin/net.cc`: external net plugin loading.
- `sources/nccl/plugins/net/README.md`: net plugin API.

## Transport selection

`transport.cc` defines the transport order:

```c
struct ncclTransport* ncclTransports[NTRANSPORTS+1] = {
  &p2pTransport,
  &shmTransport,
  &netTransport,
  &collNetTransport,
  &profilerTransport
};
```

For a connection, NCCL loops over transports and calls:

```text
transport->canConnect(...)
  -> transportComm->setup(...)
  -> later transportComm->connect(...)
```

If no transport can connect, NCCL logs a warning and returns `ncclSystemError`.

## Connector setup

`ncclTransportP2pSetup` wires send/recv connectors for channels and peers. It:

1. marks which peers/channels need send/recv connections,
2. selects transports for each connection,
3. exchanges `ncclConnect` data over bootstrap,
4. repeatedly calls connect until channels are connected,
5. copies connection info to device peer connector memory,
6. uses internal host/device strong streams for setup.

Connection setup is batched by `NCCL_CONNECT_ROUND_MAX_PEERS` and can report progress with
`NCCL_REPORT_CONNECT_PROGRESS`.

## P2P transport

Source: `transport/p2p.cc`.

P2P handles local GPU-to-GPU communication using CUDA peer capabilities and direct paths when topology
allows.

Important variables:

| Variable | Use |
|---|---|
| `NCCL_P2P_READ_ENABLE` | controls P2P read behavior |
| `NCCL_P2P_DIRECT_DISABLE` | disables direct P2P |
| `NCCL_P2P_USE_CUDA_MEMCPY` | uses CUDA memcpy path |
| `NCCL_LEGACY_CUDA_REGISTER` | legacy registration path |

`ncclTransportCheckP2pType` checks local ranks for CUDA P2P connectivity and whether direct mode applies.

## SHM transport

Source: `transport/shm.cc`.

Shared-memory transport is an intra-node path when direct P2P is unavailable or not selected. It uses host
shared memory and CUDA memory-copy behavior depending on configuration.

Variables:

| Variable | Use |
|---|---|
| `NCCL_SHM_DISABLE` | disable SHM transport |
| `NCCL_SHM_USE_CUDA_MEMCPY` | choose CUDA memcpy behavior |
| `NCCL_SHM_MEMCPY_MODE` | sender/receiver/both-side memcpy mode |
| `NCCL_SHM_LOCALITY` | locality policy |

Disabling SHM can force network/socket paths for local ranks; use only for diagnosis or specific topology
constraints.

## NET transport

Source: `transport/net.cc`.

NET is the abstraction over external net plugins and internal socket/IB implementations. It handles:

- device selection,
- memory registration,
- send/recv operation posting,
- optional recv completion,
- flush behavior,
- shared buffers/comms,
- proxy operation creation.

Variables:

| Variable | Use |
|---|---|
| `NCCL_NET` | choose net implementation by plugin-reported name |
| `NCCL_NET_PLUGIN` | select/load net plugin library suffix/path |
| `NCCL_NET_SHARED_BUFFERS` | network shared buffers |
| `NCCL_NET_SHARED_COMMS` | network shared comms |
| `NCCL_NET_OPTIONAL_RECV_COMPLETION` | optimize LL/LL128 recv completion behavior |
| `NCCL_NET_GDR_READ` | GDR read behavior |
| `NCCL_NET_FORCE_FLUSH` | force flushes |
| `NCCL_NET_DISABLE_INTRA` | avoid net for intra-node paths |

## Socket transport

Source: `transport/net_socket.cc`, `misc/socket.cc`.

Socket path is portable and useful as fallback, but usually slower than RDMA/IB on GPU clusters.

Variables:

| Variable | Use |
|---|---|
| `NCCL_SOCKET_NTHREADS` | socket progress threads |
| `NCCL_NSOCKS_PERTHREAD` | sockets per thread |
| `NCCL_SOCKET_INLINE` | inline size |
| `NCCL_SOCKET_MIN_TASKSIZE` | min task size |
| `NCCL_SOCKET_RETRY_CNT` | retry count |
| `NCCL_SOCKET_RETRY_SLEEP_MSEC` | retry sleep |
| `NCCL_SOCKET_POLL_TIMEOUT_MSEC` | poll timeout |
| `NCCL_SOCKET_RCVBUF`, `NCCL_SOCKET_SNDBUF` | socket buffer sizes |

If logs show socket fallback on an IB cluster, investigate plugin loading, `NCCL_IB_DISABLE`, container
permissions, NIC visibility, and library dependencies.

## InfiniBand transport

Source directories:

- `transport/net_ib/init.cc`
- `transport/net_ib/connect.cc`
- `transport/net_ib/p2p.cc`
- `transport/net_ib/reg.cc`
- `transport/net_ib/common.cc`
- resiliency files and GDAKI/GIN files.

IB transport handles RDMA-capable multi-node communication, QP setup, memory registration, adaptive
routing, relaxed ordering, GDR flush, and resiliency options.

Important variable groups:

### Device/init

- `NCCL_IB_DISABLE`
- `NCCL_IB_MERGE_VFS`
- `NCCL_IB_MERGE_NICS`
- `NCCL_IB_DEVICE_PCI_ORDER`
- `NCCL_IB_PCI_RELAXED_ORDERING`
- `NCCL_IB_ADAPTIVE_ROUTING`
- `NCCL_IB_DATA_DIRECT`
- `NCCL_IB_OOO_RQ`

### Connection/QP

- `NCCL_IB_GID_INDEX`
- `NCCL_IB_ROCE_VERSION_NUM`
- `NCCL_IB_TIMEOUT`
- `NCCL_IB_RETRY_CNT`
- `NCCL_IB_PKEY`
- `NCCL_IB_USE_INLINE`
- `NCCL_IB_SL`
- `NCCL_IB_TC`
- `NCCL_IB_FIFO_TC`
- `NCCL_IB_ECE_ENABLE`
- `NCCL_IB_QPS_PER_CONNECTION`

### Resiliency

- `NCCL_IB_RESILIENCY_PORT_FAILOVER`
- `NCCL_IB_RESILIENCY_PORT_FAILOVER_MAX_ATTEMPTS`
- `NCCL_IB_RESILIENCY_PORT_FAILOVER_PROBE_DELAY`
- `NCCL_IB_RESILIENCY_PORT_RECOVERY`
- `NCCL_IB_RESILIENCY_PORT_RECOVERY_*`

## CollNet transport

Source: `transport/coll_net.cc` plus net plugin CollNet support.

CollNet allows in-network collective acceleration when the network/plugin supports it. It is optional and
versioned alongside the net plugin API. If unavailable or mismatched, NCCL can fall back to other algorithms.

Relevant variables:

- `NCCL_COLLNET_ENABLE`
- `NCCL_COLLNET_NODE_THRESHOLD`
- `NCCL_IGNORE_COLLNET_MISMATCH`

## NVLS transport

Source: `transport/nvls.cc`.

NVLS is a local NVLink/NVSwitch-oriented transport/algorithm path. It is topology-specific and may use
special chunk sizes/tree variants.

Variables:

- `NCCL_NVLS_ENABLE`
- `NCCL_NVLS_CHUNKSIZE`
- `NCCL_NVLSTREE_MAX_CHUNKSIZE`
- `NCCL_NVLS_NCHANNELS`

## Proxy progress engine

Source: `proxy.cc`.

Some transports/protocols need CPU-side progress for network operations, connection setup, memory
registration, or profiler polling. Proxy messages include operations such as:

- init,
- setup,
- connect,
- start,
- close,
- abort,
- file descriptor exchange,
- register/deregister.

Proxy concepts:

| Concept | Meaning |
|---|---|
| proxy state | communicator/process-level proxy thread and resources |
| proxy op | per-channel/per-peer network work item |
| proxy step | finer-grained transfer step inside a proxy op |
| progress thread | CPU thread polling and advancing operations |
| async RPC | decouples launch/setup from progress |

Variables:

| Variable | Use |
|---|---|
| `NCCL_PROXY_APPEND_BATCH_SIZE` | batch appending proxy ops |
| `NCCL_CREATE_THREAD_CONTEXT` | thread context behavior |
| `NCCL_PROXY_DUMP_SIGNAL` | proxy signal debugging |
| `NCCL_PROGRESS_APPENDOP_FREQ` | append-op frequency |

## Net plugin API flow

A net plugin implements versioned `ncclNet_vX` structs. Core flow:

```text
init -> devices -> getProperties
listen (receiver) -> handle exchange -> connect (sender) -> accept (receiver)
regMr / regMrDmaBuf
isend / irecv / iflush / test
closeSend / closeRecv / closeListen
deregMr
```

Important nonblocking rule: `connect`, `accept`, `isend`, and `irecv` may return success with output
comm/request set to `NULL` to mean "try again later".

See `15-plugin-development.md` for the full plugin development reference.

## Debugging transport selection

1. Enable `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH`.
2. Confirm P2P/SHM/NET selection in logs.
3. Confirm external plugin library is found if expected.
4. Confirm net plugin `name` matches `NCCL_NET` if set.
5. Confirm NIC properties include valid PCI path, speed, latency, pointer support.
6. Confirm memory registration path is compatible with CUDA pointer/GDR/DMABUF expectations.
7. Confirm proxy threads are alive for network paths.
8. Compare with `nccl-tests`.

## Editing transport code safely

- Preserve the `canConnect/setup/connect` contract.
- Do not block in plugin/transport calls documented as retry/nonblocking.
- Keep host/device connector structures synchronized.
- Update proxy op creation and progress logic together.
- Test P2P, SHM fallback, socket fallback, IB, and plugin paths separately.
- Validate CUDA graph capture paths if setup happens during capture.
