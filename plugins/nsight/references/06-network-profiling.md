# Network and Communication Profiling Reference

## Overview

Nsight Systems provides comprehensive network and communication profiling capabilities for distributed computing workloads. This includes tracing of MPI, OpenSHMEM, UCX, NCCL, and NVSHMEM communication libraries, as well as NIC metric sampling, InfiniBand statistics, and storage I/O profiling. These features are essential for understanding communication patterns, identifying bottlenecks, and optimizing multi-node GPU application performance.

---

## MPI API Trace

### Supported Implementations

Nsight Systems supports MPI tracing for the following implementations:

| Implementation | Versions | Notes |
|---------------|----------|-------|
| **Open MPI** | 4.x, 5.x | Most widely tested; requires `--trace=mpi` |
| **MPICH** | 3.x, 4.x | Supported with `--trace=mpi` |

Other MPI implementations that are ABI-compatible with Open MPI or MPICH may also work but are not officially tested.

### Enabling MPI Tracing

```bash
# Basic MPI tracing
nsys profile --trace=mpi mpirun -np 4 ./my_mpi_app

# MPI tracing with CUDA and NVTX
nsys profile --trace=cuda,mpi,nvtx mpirun -np 4 ./my_mpi_app

# MPI tracing with communication parameters
nsys profile --trace=mpi --mpi-impl=openmpi mpirun -np 4 ./my_mpi_app
```

| CLI Option | Description | Default |
|------------|-------------|---------|
| `--trace=mpi` | Enable MPI API tracing | Disabled |
| `--mpi-impl` | Specify MPI implementation (`openmpi`, `mpich`) | Auto-detected |
| `--mpi-msg-size` | Minimum message size to trace (bytes) | 0 (all messages) |

### MPI Trace Output

MPI trace data appears in the Nsight Systems timeline as:

- **Point-to-point operations**: Shown as arrows between ranks with message size annotations.
- **Collective operations**: Shown as synchronized bars across all participating ranks.
- **RMA operations**: Shown as specialized annotations on the originating rank.

Each MPI operation on the timeline displays:
- Function name (e.g., `MPI_Send`, `MPI_Recv`, `MPI_Allreduce`)
- Duration
- Message size (where applicable)
- Source/destination rank
- Communicator information

---

## MPI Communication Parameters

Nsight Systems can capture and display communication parameters for MPI operations, providing additional context about each transfer.

### Enabling Communication Parameters

```bash
nsys profile --trace=mpi --mpi-comm-params=yes mpirun -np 4 ./my_mpi_app
```

Captured parameters include:

| Parameter | Description |
|-----------|-------------|
| Source rank | Originating process for receives |
| Destination rank | Target process for sends |
| Tag | Message tag used for matching |
| Communicator | MPI communicator handle |
| Count | Number of data elements |
| Datatype | MPI datatype of the data |
| Message size | Computed size in bytes |

---

## Complete MPI Functions Traced List

The following MPI functions are traced when MPI tracing is enabled:

### Point-to-Point Communication

| Function | Description |
|----------|-------------|
| `MPI_Send` | Standard mode send |
| `MPI_Ssend` | Synchronous mode send |
| `MPI_Rsend` | Ready mode send |
| `MPI_Bsend` | Buffered mode send |
| `MPI_Recv` | Blocking receive |
| `MPI_Sendrecv` | Send and receive in one call |
| `MPI_Sendrecv_replace` | Send and receive with a single buffer |
| `MPI_Isend` | Nonblocking standard mode send |
| `MPI_Issend` | Nonblocking synchronous mode send |
| `MPI_Irsend` | Nonblocking ready mode send |
| `MPI_Ibsend` | Nonblocking buffered mode send |
| `MPI_Irecv` | Nonblocking receive |
| `MPI_Send_init` | Build a persistent send handle |
| `MPI_Ssend_init` | Build a persistent synchronous send handle |
| `MPI_Rsend_init` | Build a persistent ready send handle |
| `MPI_Bsend_init` | Build a persistent buffered send handle |
| `MPI_Recv_init` | Build a persistent receive handle |
| `MPI_Start` | Start a persistent communication request |
| `MPI_Startall` | Start multiple persistent requests |
| `MPI_Wait` | Wait for a nonblocking operation to complete |
| `MPI_Waitall` | Wait for all given operations to complete |
| `MPI_Waitany` | Wait for any given operation to complete |
| `MPI_Waitsome` | Wait for some given operations to complete |
| `MPI_Test` | Test for completion of a nonblocking operation |
| `MPI_Testall` | Test for completion of all given operations |
| `MPI_Testany` | Test for completion of any given operation |
| `MPI_Testsome` | Test for completion of some given operations |
| `MPI_Probe` | Blocking test for a message |
| `MPI_Iprobe` | Nonblocking test for a message |
| `MPI_Mprobe` | Blocking matched probe |
| `MPI_Improbe` | Nonblocking matched probe |
| `MPI_Mrecv` | Blocking receive after matched probe |
| `MPI_Imrecv` | Nonblocking receive after matched probe |

### Collective Communication

| Function | Description |
|----------|-------------|
| `MPI_Barrier` | Synchronization barrier |
| `MPI_Bcast` | Broadcast data to all processes |
| `MPI_Gather` | Gather data from all processes |
| `MPI_Gatherv` | Gather data with variable counts |
| `MPI_Scatter` | Scatter data to all processes |
| `MPI_Scatterv` | Scatter data with variable counts |
| `MPI_Allgather` | Gather data from all and distribute to all |
| `MPI_Allgatherv` | Gather with variable counts and distribute to all |
| `MPI_Alltoall` | Personalized all-to-all communication |
| `MPI_Alltoallv` | Personalized all-to-all with variable counts |
| `MPI_Alltoallw` | Personalized all-to-all with variable counts and displacements |
| `MPI_Reduce` | Reduce values from all processes |
| `MPI_Allreduce` | Reduce and distribute to all processes |
| `MPI_Reduce_scatter` | Reduce with scatter |
| `MPI_Reduce_scatter_block` | Reduce with scatter (block) |
| `MPI_Scan` | Prefix reduction |
| `MPI_Exscan` | Exclusive prefix reduction |
| `MPI_Ibarrier` | Nonblocking barrier |
| `MPI_Ibcast` | Nonblocking broadcast |
| `MPI_Igather` | Nonblocking gather |
| `MPI_Igatherv` | Nonblocking gather with variable counts |
| `MPI_Iscatter` | Nonblocking scatter |
| `MPI_Iscatterv` | Nonblocking scatter with variable counts |
| `MPI_Iallgather` | Nonblocking allgather |
| `MPI_Iallgatherv` | Nonblocking allgatherv |
| `MPI_Ialltoall` | Nonblocking alltoall |
| `MPI_Ialltoallv` | Nonblocking alltoallv |
| `MPI_Ialltoallw` | Nonblocking alltoallw |
| `MPI_Ireduce` | Nonblocking reduce |
| `MPI_Iallreduce` | Nonblocking allreduce |
| `MPI_Ireduce_scatter` | Nonblocking reduce scatter |
| `MPI_Ireduce_scatter_block` | Nonblocking reduce scatter block |
| `MPI_Iscan` | Nonblocking scan |
| `MPI_Iexscan` | Nonblocking exscan |

### RMA (One-Sided Communication)

| Function | Description |
|----------|-------------|
| `MPI_Put` | Put data into remote process memory |
| `MPI_Get` | Get data from remote process memory |
| `MPI_Accumulate` | Accumulate data into remote process memory |
| `MPI_Get_accumulate` | Get and accumulate atomically |
| `MPI_Fetch_and_op` | Atomic fetch and operate |
| `MPI_Compare_and_swap` | Atomic compare and swap |
| `MPI_Rput` | Nonblocking put |
| `MPI_Rget` | Nonblocking get |
| `MPI_Raccumulate` | Nonblocking accumulate |
| `MPI_Rget_accumulate` | Nonblocking get and accumulate |
| `MPI_Win_fence` | Synchronize RMA window |
| `MPI_Win_start` | Start RMA exposure epoch |
| `MPI_Win_complete` | Complete RMA exposure epoch |
| `MPI_Win_post` | Post RMA exposure epoch |
| `MPI_Win_wait` | Wait for RMA exposure epoch |
| `MPI_Win_lock` | Lock a remote window |
| `MPI_Win_unlock` | Unlock a remote window |
| `MPI_Win_lock_all` | Lock all remote windows |
| `MPI_Win_unlock_all` | Unlock all remote windows |
| `MPI_Win_flush` | Flush RMA operations |
| `MPI_Win_flush_all` | Flush all RMA operations |
| `MPI_Win_flush_local` | Local flush RMA operations |
| `MPI_Win_flush_local_all` | Local flush all RMA operations |
| `MPI_Win_sync` | Synchronize private and public windows |

---

## OpenSHMEM Library Trace

### Overview

OpenSHMEM is a PGAS (Partitioned Global Address Space) communication library. Nsight Systems can trace OpenSHMEM API calls to visualize one-sided communication patterns.

### Enabling OpenSHMEM Tracing

```bash
nsys profile --trace=oshmem oshrun -np 4 ./my_oshmem_app
```

### Traced OpenSHMEM Functions

| Function | Description |
|----------|-------------|
| `shmem_put` | Put data to a remote PE |
| `shmem_get` | Get data from a remote PE |
| `shmem_put_nbi` | Nonblocking put |
| `shmem_get_nbi` | Nonblocking get |
| `shmem_atomic_add` | Atomic add |
| `shmem_atomic_inc` | Atomic increment |
| `shmem_atomic_fetch_add` | Atomic fetch and add |
| `shmem_atomic_fetch_inc` | Atomic fetch and increment |
| `shmem_atomic_set` | Atomic set |
| `shmem_atomic_fetch` | Atomic fetch |
| `shmem_atomic_swap` | Atomic swap |
| `shmem_atomic_compare_swap` | Atomic compare and swap |
| `shmem_barrier` | Barrier synchronization |
| `shmem_barrier_all` | Barrier across all PEs |
| `shmem_fence` | Memory fence |
| `shmem_quiet` | Quiet (completion) |
| `shmem_broadcast` | Broadcast |
| `shmem_collect` | Collect |
| `shmem_fcollect` | Fenced collect |
| `shmem_alltoall` | All-to-all |
| `shmem_alltoalls` | All-to-all with stride |
| `shmem_reduce` | Reduction |
| `shmem_team_sync` | Team synchronization |

---

## UCX API Trace

### Overview

Unified Communication X (UCX) is a low-level communication framework used by MPI, OpenSHMEM, and other HPC libraries. Tracing UCX provides visibility into the actual network operations underlying higher-level communication APIs.

### Enabling UCX Tracing

```bash
nsys profile --trace=ucx mpirun -np 4 ./my_mpi_app
```

### Environment Variables

| Variable | Values | Description |
|----------|--------|-------------|
| `NSYS_UCP_COMM_SUBMIT` | `0` or `1` | Trace UCP communication submission operations (send/recv init). Default: `1` |
| `NSYS_UCP_COMM_PROGRESS` | `0` or `1` | Trace UCP communication progress operations (actual data transfer). Default: `1` |
| `NSYS_UCP_COMM_PARAMS` | `0` or `1` | Capture communication parameters (buffer address, size, endpoint). Default: `1` |

Example:

```bash
# Only trace UCX submission (not progress) to reduce overhead
export NSYS_UCP_COMM_SUBMIT=1
export NSYS_UCP_COMM_PROGRESS=0
nsys profile --trace=ucx mpirun -np 4 ./my_mpi_app
```

### UCX Functions Traced

| Function | Description |
|----------|-------------|
| `ucp_tag_send_nb` | Nonblocking tagged send |
| `ucp_tag_send_sync_nb` | Nonblocking synchronous tagged send |
| `ucp_tag_recv_nb` | Nonblocking tagged receive |
| `ucp_tag_msg_recv_nb` | Nonblocking tagged message receive |
| `ucp_stream_send_nb` | Nonblocking stream send |
| `ucp_stream_recv_nb` | Nonblocking stream receive |
| `ucp_put_nb` | Nonblocking remote memory put |
| `ucp_get_nb` | Nonblocking remote memory get |
| `ucp_atomic_add_nb` | Nonblocking atomic add |
| `ucp_atomic_fetch_nb` | Nonblocking atomic fetch |
| `ucp_am_send_nb` | Nonblocking active message send |
| `ucp_am_recv_nb` | Nonblocking active message receive |
| `ucp_worker_progress` | Progress the UCX worker |
| `ucp_worker_fence` | Worker fence |
| `ucp_worker_flush_nb` | Nonblocking worker flush |
| `ucp_ep_flush_nb` | Nonblocking endpoint flush |
| `ucp_request_cancel` | Cancel a pending request |
| `ucp_request_free` | Free a completed request |

### UCX Functions NOT Traced

The following UCX functions are not traced because they are management or setup operations that do not directly involve data movement:

| Function | Reason Not Traced |
|----------|-------------------|
| `ucp_init` | Initialization (one-time) |
| `ucp_cleanup` | Cleanup (one-time) |
| `ucp_worker_create` | Worker creation |
| `ucp_worker_destroy` | Worker destruction |
| `ucp_worker_query` | Worker query |
| `ucp_ep_create` | Endpoint creation |
| `ucp_ep_destroy` | Endpoint destruction |
| `ucp_ep_modify_nb` | Endpoint modification |
| `ucp_ep_query` | Endpoint query |
| `ucp_listener_create` | Listener creation |
| `ucp_listener_destroy` | Listener destruction |
| `ucp_listener_query` | Listener query |
| `ucp_mem_map` | Memory registration |
| `ucp_mem_unmap` | Memory deregistration |
| `ucp_mem_query` | Memory query |
| `ucp_rkey_pack` | Remote key packing |
| `ucp_rkey_unpack` | Remote key unpacking |
| `ucp_rkey_release` | Remote key release |
| `ucp_config_read` | Configuration read |
| `ucp_config_release` | Configuration release |
| `ucp_config_modify` | Configuration modify |
| `ucp_context_query` | Context query |
| `ucp_dump_all` | Debug dump |

---

## NVIDIA NVSHMEM and NCCL Trace

### NCCL Trace

NVIDIA Collective Communications Library (NCCL) provides optimized collective operations for multi-GPU and multi-node communication. Nsight Systems traces NCCL API calls to visualize collective patterns.

#### Enabling NCCL Tracing

```bash
nsys profile --trace=nccl mpirun -np 4 ./my_nccl_app
```

NCCL tracing captures:

- Collective operation type (AllReduce, Broadcast, AllGather, ReduceScatter, Send, Recv, etc.)
- Message sizes
-参与组信息 (participating GPU group)
- Duration and timing of each collective phase

#### Traced NCCL Functions

| Function | Description |
|----------|-------------|
| `ncclAllReduce` | All-reduce across all GPUs |
| `ncclReduce` | Reduce to a single GPU |
| `ncclBroadcast` | Broadcast from one GPU to all |
| `ncclAllGather` | Gather data from all GPUs and distribute |
| `ncclReduceScatter` | Reduce and scatter across GPUs |
| `ncclSend` | Point-to-point send |
| `ncclRecv` | Point-to-point receive |
| `ncclGroupStart` | Begin a group of operations |
| `ncclGroupEnd` | End a group of operations |
| `ncclCommInitRank` | Initialize communicator for a rank |
| `ncclCommDestroy` | Destroy communicator |
| `ncclCommAbort` | Abort communicator |

### NVSHMEM Trace

NVIDIA SHMEM (NVSHMEM) provides a PGAS communication library with GPU-initiated communication capabilities. Nsight Systems traces NVSHMEM API calls for both host-side and device-initiated operations.

#### Enabling NVSHMEM Tracing

```bash
nsys profile --trace=nvshmem mpirun -np 4 ./my_nvshmem_app
```

#### Traced NVSHMEM Functions

| Function Category | Functions |
|-------------------|-----------|
| **Put operations** | `nvshmem_put`, `nvshmem_iput`, `nvshmem_put_nbi` |
| **Get operations** | `nvshmem_get`, `nvshmem_iget`, `nvshmem_get_nbi` |
| **Atomics** | `nvshmem_atomic_add`, `nvshmem_atomic_inc`, `nvshmem_atomic_fetch`, `nvshmem_atomic_swap`, `nvshmem_atomic_compare_swap` |
| **Collectives** | `nvshmem_barrier`, `nvshmem_barrier_all`, `nvshmem_broadcast`, `nvshmem_collect`, `nvshmem_reduce` |
| **Memory ordering** | `nvshmem_fence`, `nvshmem_quiet` |
| **Teams** | `nvshmem_team_sync`, `nvshmem_team_broadcast`, `nvshmem_team_reduce` |

---

## NIC Metric Sampling

### Overview

Network Interface Controller (NIC) metric sampling captures throughput and utilization metrics from network devices (Ethernet, InfiniBand, and other high-speed interconnects). This data helps identify network bottlenecks in distributed workloads.

### Limitations and Requirements

| Requirement | Details |
|-------------|---------|
| **Driver support** | Mellanox OFED or inbox drivers with ethtool support |
| **NIC types** | Mellanox ConnectX (all generations), NVIDIA BlueField DPU |
| **Metrics source** | ethtool, /sys/class/net statistics, and hardware counters |
| **Sampling overhead** | Low (reads hardware counters) |
| **Privilege requirements** | No special privileges needed for basic counters |

### Collecting NIC Metrics via CLI

```bash
# Enable NIC metric sampling on all detected NICs
nsys profile --nic-metrics=all ./my_app

# Specify specific NIC devices
nsys profile --nic-metrics=mlx5_0,mlx5_1 ./my_app

# Set NIC metric sampling frequency
nsys profile --nic-metrics=all --nic-metrics-frequency=10 ./my_app
```

| Option | Description | Default |
|--------|-------------|---------|
| `--nic-metrics` | Comma-separated NIC device names, or `all` | Disabled |
| `--nic-metrics-frequency` | Sampling frequency in Hz | 10 Hz |

### Available NIC Metrics

| Metric | Unit | Description |
|--------|------|-------------|
| RX Bytes | Bytes/s | Total received bytes per second |
| TX Bytes | Bytes/s | Total transmitted bytes per second |
| RX Packets | Packets/s | Total received packets per second |
| TX Packets | Packets/s | Total transmitted packets per second |
| RX Errors | Count/s | Receive errors per second |
| TX Errors | Count/s | Transmit errors per second |
| RX Drops | Count/s | Received packets dropped per second |
| TX Drops | Count/s | Transmitted packets dropped per second |
| RX Frame Errors | Count/s | Frame errors on receive |
| TX Carrier Errors | Count/s | Carrier errors on transmit |
| RX Multicast | Packets/s | Multicast packets received |
| RX Broadcast | Packets/s | Broadcast packets received |
| TX Multicast | Packets/s | Multicast packets transmitted |
| TX Broadcast | Packets/s | Broadcast packets transmitted |

---

## InfiniBand Switch Metric Sampling

### Overview

Nsight Systems can sample metrics from InfiniBand switches to monitor fabric-level congestion and throughput. This requires access to the switch management interface (via InfiniBand subnet manager).

### Enabling InfiniBand Switch Metrics

```bash
nsys profile --ib-switch-metrics=all ./my_app
```

### Available Metrics

| Metric | Unit | Description |
|--------|------|-------------|
| Port Xmit Data | Bytes/s | Data transmitted on a switch port |
| Port Rcv Data | Bytes/s | Data received on a switch port |
| Port Xmit Packets | Packets/s | Packets transmitted on a switch port |
| Port Rcv Packets | Packets/s | Packets received on a switch port |
| Port Xmit Wait | Count | Times transmit was delayed |
| Symbol Error | Count | Physical layer symbol errors |
| VL15 Dropped | Count | VL15 packets dropped |
| Port Buffer Overrun | Count | Port buffer overrun events |

---

## InfiniBand Switch Congestion Events

### Overview

InfiniBand Congestion Control (IB CC) provides real-time congestion notification. Nsight Systems can capture congestion events to identify hotspots in the fabric.

### Enabling Congestion Event Collection

```bash
nsys profile --ib-switch-congestion=all ./my_app
```

### Captured Congestion Data

| Data | Description |
|------|-------------|
| Congested switch GUID | Unique identifier of the congested switch |
| Congested port | Port number experiencing congestion |
| Congestion severity | Degree of congestion (low/medium/high based on threshold) |
| Timestamp | When the congestion event occurred |
| Duration | How long the congestion persisted |

---

## InfiniBand Network Information

### Overview

Nsight Systems can collect InfiniBand fabric topology and network information for context during analysis.

### Enabling IB Network Information

```bash
nsys profile --ib-network-info=yes ./my_app
```

### Collected Information

| Information | Source | Description |
|-------------|--------|-------------|
| HCA GUID | ibstat | Host Channel Adapter globally unique identifier |
| Port state | ibstat | Port physical and logical state (Active, Down, etc.) |
| Link width | ibstat | Active link width (1x, 2x, 4x, 8x, 12x) |
| Link speed | ibstat | Active link speed (SDR, DDR, QDR, FDR, EDR, HDR, NDR) |
| LID | ibstat | Local Identifier assigned by the subnet manager |
| SM LID | ibstat | Subnet Manager LID |
| Port GUID | ibstat | Port-level GUID |
| Topology | ibnetdiscover | Switch-to-host connectivity map |
| Rate | ibstat | Effective link rate in Gbps |

This information is embedded in the report file and provides context for interpreting NIC metrics and communication traces.

---

## Amazon AWS EFA Metrics

### Overview

Elastic Fabric Adapter (EFA) is AWS's custom network interface for HPC and ML workloads. Nsight Systems can collect EFA-specific metrics when running on AWS instances with EFA attached.

### Enabling EFA Metrics

```bash
nsys profile --efa-metrics=yes ./my_app
```

### Available EFA Metrics

| Metric | Description |
|--------|-------------|
| EFA RX Bytes | Bytes received via EFA |
| EFA TX Bytes | Bytes transmitted via EFA |
| EFA RX Packets | Packets received via EFA |
| EFA TX Packets | Packets transmitted via EFA |
| EFA RDMA Read Bytes | Bytes transferred via RDMA read operations |
| EFA RDMA Write Bytes | Bytes transferred via RDMA write operations |
| EFA Unreliable Datagram Bytes | Bytes transferred via unreliable datagram |
| EFA Connection Errors | Connection-level error count |

### Requirements

- AWS EC2 instance with EFA attached (e.g., p4d.24xlarge, p5.48xlarge, Hpc6a, Trn1)
- AWS EFA driver installed
- Libfabric (OFI) configured for EFA provider

---

## Network Interface Metrics

### Overview

Beyond NIC hardware counters, Nsight Systems can collect general network interface metrics from the operating system for all detected network interfaces.

### Enabling Network Interface Metrics

```bash
nsys profile --network-metrics=all ./my_app
```

### Available Metrics

| Metric | Source | Description |
|--------|--------|-------------|
| Interface name | /sys/class/net | Network interface name (eth0, ib0, etc.) |
| RX bytes | /sys/class/net/statistics/rx_bytes | Total bytes received |
| TX bytes | /sys/class/net/statistics/tx_bytes | Total bytes transmitted |
| RX packets | /sys/class/net/statistics/rx_packets | Total packets received |
| TX packets | /sys/class/net/statistics/tx_packets | Total packets transmitted |
| RX errors | /sys/class/net/statistics/rx_errors | Receive errors |
| TX errors | /sys/class/net/statistics/tx_errors | Transmit errors |
| RX dropped | /sys/class/net/statistics/rx_dropped | Received packets dropped |
| TX dropped | /sys/class/net/statistics/tx_dropped | Transmitted packets dropped |
| Collisions | /sys/class/net/statistics/collisions | Packet collisions |
| Link speed | ethtool | Negotiated link speed |
| Duplex mode | ethtool | Half/full duplex |

---

## Storage Metrics Profiling

### Overview

Nsight Systems can profile storage I/O for distributed filesystems and high-performance storage protocols. This helps identify I/O bottlenecks that affect application performance.

### NFS (Network File System) Profiling

#### Enabling NFS Trace

```bash
nsys profile --trace=nfs ./my_app
```

NFS profiling captures:

| Data | Description |
|------|-------------|
| File operations | open, close, read, write, stat, getattr, setattr |
| Transfer sizes | Bytes per read/write operation |
| Latency | Time spent in each NFS operation |
| Server information | NFS server address and export path |

### Lustre Filesystem Profiling

#### Enabling Lustre Trace

```bash
nsys profile --trace=lustre ./my_app
```

Lustre profiling captures:

| Data | Description |
|------|-------------|
| OST statistics | Object Storage Target throughput and latency |
| MDT operations | Metadata Target operation counts |
| Job ID mapping | Lustre job statistics mapped to application |
| Read/Write sizes | Per-operation transfer sizes |
| Stripe information | File striping configuration |
| RPC latency | Lustre RPC round-trip time |

### NVMeOF (NVMe over Fabrics) Profiling

#### Enabling NVMeOF Trace

```bash
nsys profile --trace=nvmeof ./my_app
```

NVMeOF profiling captures:

| Data | Description |
|------|-------------|
| I/O commands | NVMe read/write command submissions and completions |
| Transfer sizes | Bytes per I/O command |
| Latency | I/O command round-trip latency |
| Queue depth | NVMe submission queue utilization |
| Controller info | NVMeOF controller and namespace details |
| Error counts | I/O errors and timeouts |

### GDS (GPU Direct Storage) Profiling

#### Overview

GPU Direct Storage (GDS) enables direct data transfer between GPU memory and storage devices, bypassing the CPU. Nsight Systems traces GDS operations to identify storage I/O bottlenecks in GPU-accelerated pipelines.

#### Enabling GDS Trace

```bash
nsys profile --trace=gds ./my_app
```

GDS profiling captures:

| Data | Description |
|------|-------------|
| Read operations | cuFile read from storage to GPU memory |
| Write operations | cuFile write from GPU memory to storage |
| Batch operations | Batched cuFile submissions |
| Transfer sizes | Bytes per GDS operation |
| Latency | Time from submission to completion |
| File handle info | File path and handle mapping |
| Error status | cuFile error codes for failed operations |

#### GDS Performance Analysis

When analyzing GDS trace data, compare:

1. **GDS throughput vs. PCIe throughput**: GDS should approach PCIe bandwidth limits. If significantly lower, check batch sizes and alignment.
2. **GDS vs. staged copies**: Compare GDS direct paths against traditional staged copies (storage -> CPU -> GPU) to quantify benefit.
3. **Latency outliers**: Identify GDS operations with unexpectedly high latency, which may indicate storage contention or misalignment.

#### Example: Correlate GDS with GPU Metrics

```bash
nsys profile --trace=cuda,gds --gpu-metrics-devices=all ./my_gds_app
```

This combination shows whether GPU computation overlaps with GDS data loading, and whether GDS transfers compete with other PCIe traffic.

---

## Combined Network Profiling Example

### Multi-Node Training Profiling Command

```bash
# Complete profiling command for distributed training
nsys profile \
    --trace=cuda,nvtx,mpi,nccl,ucx,osrt \
    --gpu-metrics-devices=all \
    --nic-metrics=all \
    --network-metrics=all \
    --ib-network-info=yes \
    --sample=cpu \
    --output=my_profile_%p \
    mpirun -np 8 ./my_training_app
```

This command captures:

- CUDA kernel execution and memory operations
- NVTX annotations from the application
- MPI communication patterns
- NCCL collective operations
- UCX low-level network operations
- OS runtime calls (I/O, synchronization)
- GPU hardware metrics (utilization, bandwidth)
- NIC throughput counters
- Network interface statistics
- InfiniBand fabric information
- CPU sampling

### SQLite Query: Identify Communication Hotspots

```sql
-- Find the longest MPI operations
SELECT
    n.timestamp,
    n.end - n.timestamp AS duration_ns,
    n.name AS mpi_function,
    (n.end - n.timestamp) / 1000000.0 AS duration_ms
FROM StringIds s
JOIN FUNCTION_NAMES n ON n.nameId = s.id
WHERE s.value LIKE 'MPI_%'
  AND (n.end - n.timestamp) > 10000000  -- Longer than 10ms
ORDER BY duration_ns DESC
LIMIT 20;
```

### SQLite Query: Correlate NCCL with NIC Throughput

```sql
-- Find time periods where NIC bandwidth is high during NCCL operations
SELECT
    nccl.timestamp AS nccl_start,
    nccl.end AS nccl_end,
    nic.timestamp AS nic_sample_time,
    nic.value AS rx_bytes_per_sec
FROM
    (SELECT timestamp, end, nameId FROM FUNCTION_NAMES WHERE nameId IN
     (SELECT id FROM StringIds WHERE value LIKE 'nccl%')) nccl
JOIN
    (SELECT timestamp, value FROM NIC_METRICS
     WHERE metricId = (SELECT id FROM ENUM_NIC_METRICS WHERE name = 'RX Bytes')) nic
ON nic.timestamp BETWEEN nccl.timestamp AND nccl.end
WHERE nic.value > 1000000000  -- More than 1 GB/s
ORDER BY nic.value DESC;
```

---

## Quick Reference: Network Trace Options

| CLI Option | Description |
|------------|-------------|
| `--trace=mpi` | Trace MPI API calls |
| `--trace=oshmem` | Trace OpenSHMEM API calls |
| `--trace=ucx` | Trace UCX API calls |
| `--trace=nccl` | Trace NCCL collective operations |
| `--trace=nvshmem` | Trace NVSHMEM operations |
| `--nic-metrics=all` | Sample NIC hardware counters |
| `--nic-metrics-frequency=N` | NIC sampling frequency (Hz) |
| `--ib-switch-metrics=all` | Sample InfiniBand switch counters |
| `--ib-switch-congestion=all` | Capture InfiniBand congestion events |
| `--ib-network-info=yes` | Collect IB fabric topology information |
| `--efa-metrics=yes` | Collect AWS EFA metrics |
| `--network-metrics=all` | Collect OS-level network interface metrics |
| `--trace=nfs` | Trace NFS file operations |
| `--trace=lustre` | Trace Lustre filesystem operations |
| `--trace=nvmeof` | Trace NVMe over Fabrics operations |
| `--trace=gds` | Trace GPU Direct Storage operations |
| `--mpi-impl=IMPL` | Specify MPI implementation |
| `--mpi-msg-size=N` | Minimum MPI message size to trace (bytes) |
| `--mpi-comm-params=yes` | Capture MPI communication parameters |
