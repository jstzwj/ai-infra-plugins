# 13 - Topology Discovery, Graph Search, Channels, and Tuning Inputs

## Primary source files

- `sources/nccl/src/graph/topo.cc`: topology system construction.
- `sources/nccl/src/graph/topo.h`: topology declarations.
- `sources/nccl/src/graph/xml.cc`, `xml.h`: XML topology representation.
- `sources/nccl/src/graph/paths.cc`: path computation and trimming.
- `sources/nccl/src/graph/search.cc`: graph search for rings, trees, NVLS, CollNet, etc.
- `sources/nccl/src/graph/connect.cc`: convert selected graphs into channel connectivity.
- `sources/nccl/src/graph/rings.cc`, `rings.h`: ring helpers.
- `sources/nccl/src/graph/trees.cc`: tree helpers.
- `sources/nccl/src/graph/tuning.cc`: performance model and algorithm/protocol costs.
- `sources/nccl/src/include/graph.h`: internal graph declarations.

## Topology pipeline

```text
hardware/system discovery or XML input
  -> ncclTopoSystem
  -> path computation between GPUs/CPUs/NICs/NVSwitch/GIN devices
  -> graph search for algorithm-specific patterns
  -> channel/ring/tree/NVLS/CollNet connection construction
  -> tuning model estimates algorithm/protocol/channel costs
  -> communicator stores selected graphs/channels
```

NCCL's performance depends heavily on this pipeline. Many "NCCL is slow" reports are actually topology
or transport-selection problems.

## Topology system

`topo.cc` builds a topology graph from sources such as:

- PCI/NVML discovery,
- GPUs and NVLink/NVSwitch relationships,
- CPUs and NUMA relationships,
- NICs/network devices,
- GIN/device-networking nodes,
- optional XML topology dumps/overrides.

The topology system captures both connectivity and bandwidth/path quality.

## XML topology

`xml.cc` and `xml.h` implement XML representation. Topology XML is useful for:

- dumping discovered topology for debugging,
- replaying/inspecting topology offline,
- comparing expected vs actual GPU/NIC relationships,
- filing reproducible topology bugs.

Related variables include:

| Variable | Meaning |
|---|---|
| `NCCL_TOPO_DUMP_FILE_RANK` | rank controlling topology dump file behavior |
| `NCCL_GRAPH_DUMP_FILE_RANK` | rank controlling graph dump file behavior |

Exact dump file variables/names should be verified in current source/user docs before instructing a user
to rely on them in production.

## Path computation

`paths.cc` computes path quality and bandwidth between topology nodes. It handles:

- GPU-GPU P2P path availability,
- PCI distance and intermediate devices,
- NVLink/NVSwitch paths,
- NIC locality,
- GDR read/flush capability,
- PXN paths,
- disabled or unavailable P2P paths,
- local vs network routing decisions.

Important source parameters:

| Variable | Default/source | Purpose |
|---|---|---|
| `NCCL_NVB_DISABLE` | `0` | disable NVB behavior |
| `NCCL_IGNORE_DISABLED_P2P` | `0` | ignore disabled P2P paths |
| `NCCL_NET_GDR_READ` | `-2` | GDR read selection |
| `NCCL_NET_GDR_C2C` | `1` | C2C GDR behavior |
| `NCCL_NET_FORCE_FLUSH` | `0` | network flush forcing |
| `NCCL_NET_DISABLE_INTRA` | `0` | disable intra-node network use |
| `NCCL_PXN_DISABLE` | `0` | disable PXN |
| `NCCL_PXN_C2C` | `1` | C2C PXN behavior |
| `NCCL_P2P_PER_CHANNEL_NET_BW` | `14 GB/s` | P2P per-channel network bandwidth model |
| `NCCL_MIN_P2P_NCHANNELS` | `1` | min P2P channels |
| `NCCL_MAX_P2P_NCHANNELS` | `MAXCHANNELS` | max P2P channels |

## Graph search

`search.cc` recursively searches for graph patterns that implement algorithms efficiently. Key functions
called out by internals survey:

- `ncclTopoSearchInit`
- `ncclTopoCompute`
- `ncclTopoSearchRec`

Search targets include:

- rings,
- trees,
- NVLS graphs,
- CollNet graphs,
- channel patterns,
- MNNVL/network rail layouts.

Important parameters:

| Variable | Meaning |
|---|---|
| `NCCL_CROSS_NIC` | cross-NIC policy |
| `NCCL_MNNVL_SCATTER_NETS_ENABLE` | MNNVL scatter net behavior |
| `NCCL_MNNVL_RAIL_PER_HOST` | rail-per-host override |
| `NCCL_P2P_PXN_LEVEL` | P2P PXN search level |

## Rings and trees

Rings and trees are the classical NCCL graph structures.

### Ring

A ring gives each rank a predecessor and successor per channel. Ring algorithms often maximize bandwidth
for large messages by pipelining chunks around the ring.

### Tree

A tree gives parent/children relationships. Tree algorithms often reduce latency for reductions and
broadcast-like patterns, especially for smaller messages or multi-node hierarchies.

`rings.cc`, `rings.h`, and `trees.cc` contain helpers for constructing and validating these structures.

## Channel connection

`connect.cc` converts chosen topology graphs into communicator channel connectivity. It sets up ring/tree
peer relationships and controls channel/ring count bounds.

Important parameters:

| Variable | Meaning |
|---|---|
| `NCCL_MIN_NRINGS`, `NCCL_MAX_NRINGS` | ring count bounds |
| `NCCL_MIN_NCHANNELS`, `NCCL_MAX_NCHANNELS` | channel count bounds |
| `NCCL_UNPACK_DOUBLE_NCHANNELS` | unpack-related channel doubling |
| `NCCL_NVB_PRECONNECT` | NVB preconnect behavior |

## Tuning model

`tuning.cc` estimates time for algorithm/protocol combinations. It starts from default constants and then
uses topology-specific graph bandwidth and latency. External tuner plugins can modify the cost table.

Inputs include:

- number of ranks/nodes,
- algorithm graph bandwidth,
- channels,
- protocol/thread count,
- compute capability class,
- CPU architecture/vendor,
- network overhead,
- CollNet/NVLS availability,
- registered buffer mode.

## CollNet

CollNet supports in-network collective operations when the network/plugin can accelerate reductions. Net
plugins may expose CollNet structures in addition to normal net API structures. CollNet algorithms include
direct and chain variants.

Common tuning/debug variables:

- `NCCL_COLLNET_ENABLE`
- `NCCL_COLLNET_NODE_THRESHOLD`
- `NCCL_IGNORE_COLLNET_MISMATCH`

## NVLS

NVLS uses NVLink/NVSwitch local collectives where hardware/topology supports it. Files include
`transport/nvls.cc` and graph/tuning/search code.

Variables:

- `NCCL_NVLS_ENABLE`
- `NCCL_NVLS_NCHANNELS`
- `NCCL_NVLS_CHUNKSIZE`
- `NCCL_NVLSTREE_MAX_CHUNKSIZE`

NVLS is topology-dependent; do not assume availability on non-NVSwitch systems.

## MNNVL

MNNVL parameters appear in init/search:

- `NCCL_MNNVL_ENABLE`
- `NCCL_MNNVL_UUID`
- `NCCL_MNNVL_CLIQUE_ID`
- `NCCL_MNNVL_CROSS_CLIQUE`
- `NCCL_MNNVL_SCATTER_NETS_ENABLE`
- `NCCL_MNNVL_RAIL_PER_HOST`

Treat MNNVL as an advanced topology-specific feature; inspect logs/source for exact platform behavior.

## NIC locality and cross-NIC behavior

NCCL considers NIC PCI path, GUID, port, speed, latency, and topology locality. Net plugin properties feed
into topology detection. Cross-NIC policy determines whether traffic can use NICs not closest to a GPU.

For multi-rail systems, inspect:

- NIC names and PCI paths in logs,
- whether multiple devices share GUID/port,
- `NCCL_CROSS_NIC`,
- plugin `getProperties` output,
- channel-to-NIC mapping.

## Debugging topology problems

1. Enable `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,NET,TUNING`.
2. Confirm every GPU and NIC is discovered.
3. Confirm GPU/NIC PCI locality matches expectation.
4. Confirm selected transport is not unintended socket fallback.
5. Compare channel count and algorithm/protocol with expected topology.
6. Run `nccl-tests` with same rank mapping.
7. Dump topology/graph if necessary and compare across nodes.

## Source modification advice

Topology code is highly heuristic. When editing:

- Add logging for new path decisions.
- Preserve XML dump/replay usefulness.
- Validate single-node, multi-node, no-NIC, multi-NIC, NVSwitch, and P2P-disabled cases.
- Run all relevant collectives, not just AllReduce.
- Check both performance and connection correctness.
