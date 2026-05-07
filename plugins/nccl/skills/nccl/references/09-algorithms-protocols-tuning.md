# 09 - Algorithms, Protocols, Tuning, and Cost Model

## Primary source files

- `sources/nccl/src/graph/tuning.cc`: algorithm/protocol model, parsing, cost estimates.
- `sources/nccl/src/graph/search.cc`: topology graph search.
- `sources/nccl/src/graph/connect.cc`: channel/ring/tree connection construction.
- `sources/nccl/src/include/collectives.h`: algorithm/protocol/function enums and constants.
- `sources/nccl/src/include/nccl_tuner.h`: tuner interface used internally.
- `sources/nccl/plugins/tuner/README.md`: external tuner plugin docs.
- `sources/nccl/src/plugin/tuner.cc`: tuner plugin loading/integration.

## NCCL operation selection dimensions

For each collective, NCCL chooses across several dimensions:

1. **Collective function**: AllReduce, Broadcast, Reduce, ReduceScatter, AllGather, AlltoAll, Gather,
   Scatter, P2P, RMA.
2. **Algorithm**: Ring, Tree, CollNet direct/chain, NVLS, NVLS tree, PAT, or specialized schedules.
3. **Protocol**: LL, LL128, SIMPLE.
4. **Channels**: number of channels/rings and per-peer channel allocation.
5. **Threads/CTAs**: kernel launch resource choices.
6. **Transport**: P2P, SHM, NET, CollNet, NVLS, plugin/IB/socket.
7. **Registration/window mode**: normal, user-registered, symmetric, device API.

Performance tuning is about identifying which dimension is wrong for the workload/topology.

## Algorithms

`collectives.cc` exposes string names for algorithms:

| Internal algorithm | String | Typical role |
|---|---|---|
| `NCCL_ALGO_TREE` | `TREE` | latency-friendly reductions/broadcast patterns |
| `NCCL_ALGO_RING` | `RING` | bandwidth-oriented large collectives |
| `NCCL_ALGO_COLLNET_DIRECT` | `COLLNET_DIRECT` | in-network/multi-node collective acceleration |
| `NCCL_ALGO_COLLNET_CHAIN` | `COLLNET_CHAIN` | CollNet chain variant |
| `NCCL_ALGO_NVLS` | `NVLS` | NVLink/NVSwitch local collectives |
| `NCCL_ALGO_NVLS_TREE` | `NVLS_TREE` | NVLS tree variant |
| `NCCL_ALGO_PAT` | `PAT` | PAT algorithm; source enables only under constraints |

Do not assume every algorithm is valid for every collective/topology/message size. NCCL computes
availability and estimated time.

## Protocols

`collectives.cc` exposes string names:

| Protocol | String | Typical role |
|---|---|---|
| `NCCL_PROTO_LL` | `LL` | low-latency protocol for small messages |
| `NCCL_PROTO_LL128` | `LL128` | low-latency protocol using 128-bit-oriented paths |
| `NCCL_PROTO_SIMPLE` | `SIMPLE` | bandwidth-oriented protocol for larger messages |

Protocol thresholds and thread counts are tuned per topology and architecture. Environment overrides can
force bad choices; use them experimentally.

## Algorithm/protocol filter syntax

`graph/tuning.cc` parses a mapping string of operation prefixes to element lists.

Examples from source comments:

```bash
NCCL_ALGO="ring,collnetdirect;allreduce:tree,collnetdirect;broadcast:ring"
NCCL_PROTO="LL,Simple;allreduce:^LL"
NCCL_PROTO="^LL128;allreduce:LL128"
```

Semantics:

- `;` separates mapping entries.
- `prefix:list` applies only to a collective prefix.
- A first entry without prefix applies to all prefixes.
- `,` separates elements.
- A leading `^` excludes the listed elements.
- All entries after the first must have a prefix if the first was prefix-less.
- Unknown prefixes/elements return `ncclInvalidUsage`.

Use filters for diagnosis or controlled deployments, not as first-line tuning.

## Cost model inputs

`tuning.cc` initializes default tuner constants with tables for:

- base latencies by algorithm/protocol,
- hardware latencies for NVLINK, PCI, NET,
- max bandwidths for LL,
- per-channel LL128 ring/tree bandwidths,
- per-channel tree bandwidths,
- per-channel NVLS tree bandwidths,
- architecture-specific factors for Volta, Ampere, Hopper, Blackwell.

The model also considers:

- number of ranks,
- number of nodes,
- min/max compute capability,
- CPU architecture/vendor for network overhead defaults,
- graph bandwidths and channel counts,
- net device type,
- plugin-provided tuner constants or cost modifications.

## PAT enablement

Source behavior in `ncclPatEnable`:

- Requires SM60 or higher for CUDA atomics.
- If `NCCL_PAT_ENABLE` is explicitly not auto (`2`), the explicit value is used.
- Auto mode disables PAT unless `nNodes == nRanks` (one GPU per node).
- Auto mode disables PAT when net device type is not host.

So PAT is not a generic all-topologies algorithm; it is constrained.

## Thread-count tuning

Variables:

```bash
NCCL_NTHREADS
NCCL_LL128_NTHREADS
```

`getNthreads` validates:

- multiple of warp size,
- maximum bound,
- minimum bound.

Invalid values are logged and clamped/defaulted. If a user sets thread counts, advise validating logs
because NCCL may not use the exact requested value.

## Channel/ring tuning

Important variables:

| Variable | Meaning |
|---|---|
| `NCCL_MIN_NCHANNELS` | lower bound for channels |
| `NCCL_MAX_NCHANNELS` | upper bound for channels |
| `NCCL_MIN_NRINGS` | legacy/ring lower bound |
| `NCCL_MAX_NRINGS` | legacy/ring upper bound |
| `NCCL_NCHANNELS_PER_NET_PEER` | per-network-peer channel count |
| `NCCL_NVLS_NCHANNELS` | NVLS channel count |
| `NCCL_P2P_SCHEDULE_GROUP_SIZE` | P2P schedule group size |

Increasing channels can improve bandwidth but also increases resource use and can worsen small-message
latency or contend with compute kernels.

## Tuner plugin interface

External tuner plugins modify NCCL's algorithm/protocol selection by changing cost tables and channel
counts without recompiling NCCL.

From `plugins/tuner/README.md`, the interface includes:

```c
ncclResult_t (*init)(size_t nRanks, size_t nNodes,
                     ncclDebugLogger_t logFunction, void **context);

ncclResult_t (*getCollInfo)(void* context, ncclFunc_t collType, size_t nBytes,
                            int numPipeOps, float** collCostTable,
                            int numAlgo, int numProto,
                            int regBuff, int* nChannels);

ncclResult_t (*destroy)(void* context);
```

Tuner plugins can:

- set cost to `0.0` to prefer a combination,
- set cost to `NCCL_ALGO_PROTO_IGNORE` to disable a combination,
- adjust `nChannels`,
- implement topology/workload-aware strategies.

Loading:

```bash
export LD_LIBRARY_PATH=/path/to/plugin:$LD_LIBRARY_PATH
export NCCL_TUNER_PLUGIN=example
# or
export NCCL_TUNER_PLUGIN=libnccl-tuner-example.so
# or
export NCCL_TUNER_PLUGIN=/absolute/path/libnccl-tuner-example.so
```

Debugging tuner behavior:

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=TUNING
```

## Tuner plugin best practices

1. Keep `getCollInfo` lightweight; it can run on critical paths.
2. Cache expensive topology/workload decisions in plugin context.
3. Return `ncclSuccess` for ignored/no-op cases.
4. Avoid returning errors from `getCollInfo`; initialization is a safer failure point.
5. Test across message sizes and rank/node counts.
6. Compare against no plugin with `nccl-tests` and representative application workloads.
7. Document every forced ignore/preference because it can become wrong on new hardware.

## Manual tuning workflow

1. Baseline with no overrides.
2. Capture `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=GRAPH,TUNING,NET`.
3. Identify chosen algorithm/protocol/channels in logs.
4. Run `nccl-tests` for target collective/message sizes.
5. Try one override at a time:
   - algorithm filter,
   - protocol filter,
   - channel bounds,
   - transport-specific variable.
6. Measure median and tail, not just best run.
7. Remove overrides that help microbenchmarks but hurt application overlap.
8. Prefer tuner plugin for systematic deployment-specific policy.

## When not to tune manually

Avoid manual overrides when:

- the root cause is mismatched rank/device mapping,
- NCCL is falling back to sockets due to network setup,
- CUDA stream synchronization serializes communication,
- message sizes vary widely and one forced algorithm hurts other phases,
- the cluster has mixed GPU/NIC topology,
- a framework already has NCCL/tensor-parallel scheduling assumptions.

Fix the topology/usage issue first.
