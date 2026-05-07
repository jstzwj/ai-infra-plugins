# 16 - RAS, Fault Handling, Revoke, Shrink, and Resiliency

## Primary source files

- `sources/nccl/src/include/ras.h`: RAS integration declarations.
- `sources/nccl/src/ras/ras.cc`: RAS thread, local notifications, message framing, polling loop.
- `sources/nccl/src/ras/rasnet.cc`: RAS socket lifecycle, keepalive, retries, fallback links, timeouts.
- `sources/nccl/src/ras/peers.cc`: peer discovery/update propagation, dead-peer tracking, link reinit.
- `sources/nccl/src/ras/collectives.cc`: RAS-level distributed collectives.
- `sources/nccl/src/ras/client.cc`: `ncclras` diagnostic client.
- `sources/nccl/src/init.cc`: communicator revoke/shrink/grow/finalize/abort APIs.
- `sources/nccl/src/transport/net_ib/p2p_resiliency*.cc`: IB resiliency/failover/recovery.

## RAS purpose

RAS stands for reliability, availability, and serviceability. NCCL's RAS subsystem provides background
monitoring and peer-state propagation so failures and diagnostic state can be detected and surfaced more
systematically than a raw transport hang.

RAS is not a replacement for application-level failure handling. Applications still need to poll async
errors, abort/revoke failed communicators, and decide whether to shrink/restart.

## RAS integration API

`include/ras.h` exposes a small internal integration surface:

```c
ncclRasCommInit(...);
ncclRasCommFini(...);
ncclRasAddRanks(...);
```

These are called as communicators are initialized/finalized or ranks are added.

## RAS source components

| File | Role |
|---|---|
| `ras.cc` | central RAS thread, local notifications, message handling, polling, termination |
| `rasnet.cc` | socket connections, keepalive, retry/fallback, timeouts |
| `peers.cc` | peer database, peer updates, dead-peer tracking, link reinitialization |
| `collectives.cc` | RAS-level collectives to aggregate/distribute RAS state |
| `client.cc` | diagnostic client (`ncclras`) |

## RAS-related parameters

Source-derived variables:

| Variable | Default | Meaning |
|---|---:|---|
| `NCCL_RAS_ENABLE` | `1` | enable RAS subsystem |
| `NCCL_RAS_TIMEOUT_FACTOR` | `1` | scale RAS timeouts |

IB resiliency variables:

| Variable | Default | Meaning |
|---|---:|---|
| `NCCL_IB_RESILIENCY_PORT_FAILOVER` | `0` | enable port failover |
| `NCCL_IB_RESILIENCY_PORT_FAILOVER_MAX_ATTEMPTS` | `1` | max failover attempts |
| `NCCL_IB_RESILIENCY_PORT_FAILOVER_PROBE_DELAY` | `10 ms` | probe delay |
| `NCCL_IB_RESILIENCY_PORT_RECOVERY` | `0` | enable port recovery |
| `NCCL_IB_RESILIENCY_PORT_RECOVERY_START_DELAY` | `200 ms` | recovery start delay |
| `NCCL_IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_BATCH_INTERVAL` | `500 ms` | alive batch interval |
| `NCCL_IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_BATCH_SIZE` | `5` | alive batch size |
| `NCCL_IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_SEQUENCE_SIZE` | `5` | alive sequence size |
| `NCCL_IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_TIMEOUT` | `4000 ms` | alive timeout |
| `NCCL_IB_RESILIENCY_PORT_RECOVERY_ACK_TIMEOUT` | `5000 ms` | ack timeout |
| `NCCL_IB_RESILIENCY_PORT_RECOVERY_ATTEMPTS_MAX` | `5` | max recovery attempts |

Treat resiliency variables as advanced; validate against current NCCL docs/source and cluster vendor
guidance before recommending deployment changes.

## Public failure-handling APIs

### Async error polling

```c
ncclResult_t asyncErr;
ncclCommGetAsyncError(comm, &asyncErr);
```

Poll during long waits. If `asyncErr != ncclSuccess`, use `ncclGetLastError(comm)` for details and decide
whether to abort/revoke/shrink.

### Abort

```c
ncclCommAbort(comm);
```

Use when a communicator cannot complete outstanding operations normally. Abort frees resources and stops
operations that may still run on device.

### Revoke

```c
ncclCommRevoke(comm, NCCL_REVOKE_DEFAULT);
```

Stops in-flight operations and waits for quiescence. After revoke, destroy/split/shrink can proceed.
Calling `ncclCommFinalize` after revoke is invalid.

### Shrink

```c
ncclCommShrink(comm, excludeRanksList, excludeRanksCount, &newcomm,
               config, NCCL_SHRINK_ABORT);
```

Use to remove failed ranks and continue with a smaller communicator. `NCCL_SHRINK_ABORT` first terminates
ongoing parent operations, then shrinks.

## Failure-handling flow patterns

### Normal shutdown

```text
all ranks finish enqueued work
  -> ncclCommFinalize
  -> wait/poll until quiescent if needed
  -> ncclCommDestroy
```

### Fatal error shutdown

```text
rank detects immediate or async NCCL error
  -> notify application control plane if any
  -> ncclCommAbort on affected comms
  -> clean up CUDA/application resources
  -> restart or fail job
```

### Recover by shrink

```text
detect failed/excluded ranks
  -> revoke or shrink with NCCL_SHRINK_ABORT
  -> new communicator returned for surviving ranks
  -> rebuild application parallel groups/state
  -> resume with smaller world if algorithm supports it
```

This requires application-level support. NCCL can produce the smaller communicator, but it cannot fix model
parallel layouts, optimizer sharding, checkpoint consistency, or data-loader state by itself.

## Diagnosing suspected rank failure

Ask for:

1. Which rank first observed error/hang.
2. Last log line for every rank.
3. Whether one process died or was OOM-killed.
4. Network error counters/logs if multi-node.
5. `ncclCommGetAsyncError` values on surviving ranks.
6. RAS logs if enabled.
7. Whether the application used abort/revoke/shrink or waited indefinitely.

## Interaction with proxy and transports

Network failures often surface through proxy progress and transport request completion. A GPU kernel may
be waiting on connector state while proxy/network cannot make progress. Profiler plugin events can expose
proxy op/step states, and RAS can expose peer-level state.

For IB-specific failures, inspect both NCCL logs and verbs/network-driver logs. NCCL variables can adjust
retry/failover behavior, but physical/network misconfiguration must be fixed outside NCCL.

## Source modification cautions

- Failure paths must be async-safe with respect to in-flight kernels and proxy ops.
- Do not free connector/proxy memory while device work can still reference it.
- Revoke/shrink/grow interact with resource sharing; test with split/shrink shared resources enabled and disabled.
- RAS message handling must avoid blocking the progress path.
- Keep diagnostic client compatibility in mind if changing message formats.
