# Network Communication and Storage Profiling Reference

This document provides comprehensive reference material for network communication profiling, NIC metric sampling, InfiniBand analysis, and storage metrics profiling in NVIDIA Nsight Systems. These features enable developers to understand network utilization, identify communication bottlenecks, and optimize multi-node application performance.

---

## Table of Contents

1. [Network Communication Profiling Overview](#network-overview)
2. [MPI API Trace](#mpi-api-trace)
   - [Configuration](#mpi-configuration)
   - [MPI Communication Parameters](#mpi-communication-parameters)
   - [MPI Functions Traced](#mpi-functions-traced)
3. [OpenSHMEM Library Trace](#openshmem-library-trace)
   - [Functions Traced](#openshmem-traced)
   - [Functions NOT Traced](#openshmem-not-traced)
4. [UCX API Trace](#ucx-api-trace)
   - [Environment Variables](#ucx-environment-variables)
   - [UCX Functions Traced](#ucx-functions-traced)
   - [UCX Functions NOT Traced](#ucx-functions-not-traced)
5. [NVIDIA NVSHMEM and NCCL Trace](#nvshmem-nccl-trace)
6. [NIC Metric Sampling](#nic-metric-sampling)
   - [Overview](#nic-overview)
   - [Requirements](#nic-requirements)
   - [Collecting NIC Metrics via CLI](#nic-cli)
   - [Available Metrics](#nic-available-metrics)
   - [Usage Examples](#nic-usage-examples)
7. [InfiniBand Switch Metric Sampling](#ib-switch-metrics)
8. [InfiniBand Switch Congestion Events](#ib-switch-congestion)
   - [Configuration Switches](#ib-congestion-switches)
9. [InfiniBand Network Information](#ib-network-info)
10. [Amazon AWS EFA Metrics](#aws-efa-metrics)
11. [Network Interface Metrics](#network-interface-metrics)
12. [Storage Metrics Profiling](#storage-metrics)
    - [Available Arguments](#storage-arguments)
    - [NFS Storage Example](#nfs-example)
    - [Lustre Storage Example](#lustre-example)
    - [Local / NVMeOF Storage Example](#nvmeof-example)
    - [GPUDirect Storage Metrics](#gds-metrics)
13. [GPUDirect Storage Trace](#gds-trace)

---

## Network Communication Profiling Overview

<a id="network-overview"></a>

Nsight Systems can be used to profile several popular network communication protocols. To enable this, select the **Communication profiling options** dropdown in the GUI, then select the libraries to trace. The corresponding Nsight Systems CLI `--trace`/`-t` options are `mpi`, `oshmem`, and `ucx`.

For multi-node runs, refer to the section on Handling Application Launchers in the Profiling From the CLI documentation.

---

## MPI API Trace

<a id="mpi-configuration"></a>

### Configuration

Nsight Systems has built-in API trace support for Open MPI and MPICH-based MPI implementations.

- **CLI**: Use `--trace=mpi` or select the MPI checkbox under Network profiling options.
- **Specify implementation**: If the auto-detection of the MPI implementation fails, specify it via `--mpi-impl=[openmpi|mpich]` or the respective checkbox in the GUI.

```bash
# Auto-detect MPI implementation
nsys profile --trace=mpi my_mpi_application

# Explicitly specify OpenMPI
nsys profile --trace=mpi --mpi-impl=openmpi my_mpi_application

# Explicitly specify MPICH
nsys profile --trace=mpi --mpi-impl=mpich my_mpi_application
```

#### Custom NVTX Wrappers for MPI

If you require more control over the list of traced APIs or if you are using a different MPI implementation, you can use the NVTX wrappers for MPI on GitHub. Choose an NVTX domain name other than "MPI," since it is filtered out by Nsight Systems when MPI tracing is not enabled.

```bash
nsys profile -e LD_PRELOAD=${PATH_TO_YOUR_NVTX_MPI_LIB} --trace=nvtx my_mpi_application
```

> **Note:** If not all ranks are traced, `NSYS_MPI_STORE_TEAMS_PER_RANK` has to be set to 1. If communicator tracking is still causing issues, it can be disabled by setting `NSYS_MPI_DISABLE_COMMUNICATOR_TRACKING=1`.

<a id="mpi-communication-parameters"></a>

### MPI Communication Parameters

Nsight Systems can get additional information about MPI communication parameters. Currently, the parameters are only visible in the mouseover tooltips or in the event log. This means that the data is only available via the GUI. Future versions of the tool will export this information into the SQLite data files for post-run analysis.

In order to fully interpret MPI communications, data for all ranks associated with a communication operation must be loaded into Nsight Systems.

#### Communication Parameters Captured

The following parameters are captured for MPI communication calls:

| Parameter | Description |
|---|---|
| `size` | Communicator size (number of ranks) |
| `rank` | Rank of the calling process within the communicator |
| `tag` | Message tag |
| `dest` | Destination rank (for send operations) |
| `source` | Source rank (for receive operations) |
| `count` | Number of data elements |
| `datatype` | MPI datatype of the data elements |

#### Communicator Display

Here is an example of `MPI_COMM_WORLD` data. This does not require any additional team data, since local rank is the same as global rank.

When not all processes that are involved in an MPI communication are loaded into Nsight Systems:

- Encoding: `MPI_COMM[*team size*]*global-group-root-rank*.*group-ID*`

When all reports are loaded into Nsight Systems:

- World rank is shown in addition to group-local rank "(world rank X)."
- Encoding: `MPI_COMM[*team size*]{rank0, rank1, ...}`.
- At most 8 ranks are shown (the numbers represent world ranks, the position in the list is the group-local rank).

<a id="mpi-functions-traced"></a>

### MPI Functions Traced

Nsight Systems will trace a subset of the MPI API, including blocking and non-blocking point-to-point and collective communications as well as MPI one-sided communications, file I/O, and pack operations.

#### Initialization and Finalization

```
MPI_Init           MPI_Init_thread    MPI_Finalize
```

#### Point-to-Point Communication (Blocking)

```
MPI_Send           MPI_Bsend          MPI_Ssend          MPI_Rsend
MPI_Recv           MPI_Mrecv          MPI_Sendrecv       MPI_Sendrecv_replace
```

#### Collective Communication (Blocking)

```
MPI_Barrier        MPI_Bcast
MPI_Scatter        MPI_Scatterv
MPI_Gather         MPI_Gatherv
MPI_Allgather      MPI_Allgatherv
MPI_Alltoall       MPI_Alltoallv      MPI_Alltoallw
MPI_Allreduce      MPI_Reduce         MPI_Reduce_scatter
MPI_Reduce_scatter_block            MPI_Reduce_local
MPI_Scan           MPI_Exscan
```

#### Point-to-Point Communication (Non-Blocking)

```
MPI_Isend          MPI_Ibsend         MPI_Issend         MPI_Irsend
MPI_Irecv          MPI_Imrecv
MPI_Send_init      MPI_Bsend_init     MPI_Ssend_init     MPI_Rsend_init
MPI_Recv_init
MPI_Start          MPI_Startall
MPI_Wait           MPI_Waitall        MPI_Waitany        MPI_Waitsome
```

#### Collective Communication (Non-Blocking)

```
MPI_Ibarrier       MPI_Ibcast
MPI_Iscatter       MPI_Iscatterv
MPI_Igather        MPI_Igatherv
MPI_Iallgather     MPI_Iallgatherv
MPI_Ialltoall      MPI_Ialltoallv     MPI_Ialltoallw
MPI_Iallreduce     MPI_Ireduce        MPI_Ireduce_scatter
MPI_Ireduce_scatter_block
MPI_Iscan          MPI_Iexscan
```

#### One-Sided Communication (RMA)

```
MPI_Put            MPI_Rput           MPI_Get            MPI_Rget
MPI_Accumulate     MPI_Raccumulate
MPI_Get_accumulate MPI_Rget_accumulate
MPI_Fetch_and_op   MPI_Compare_and_swap
```

#### RMA Window Management

```
MPI_Win_allocate              MPI_Win_allocate_shared
MPI_Win_create                MPI_Win_create_dynamic
MPI_Win_attach                MPI_Win_detach
MPI_Win_free
MPI_Win_fence
MPI_Win_start     MPI_Win_complete   MPI_Win_post       MPI_Win_wait
MPI_Win_lock      MPI_Win_unlock     MPI_Win_lock_all   MPI_Win_unlock_all
MPI_Win_flush     MPI_Win_flush_local MPI_Win_flush_all  MPI_Win_flush_local_all
MPI_Win_sync
```

#### File I/O

```
MPI_File_open                MPI_File_close
MPI_File_delete              MPI_File_sync
MPI_File_read                MPI_File_write
MPI_File_read_all            MPI_File_write_all
MPI_File_read_all_begin      MPI_File_write_all_begin
MPI_File_read_all_end        MPI_File_write_all_end
MPI_File_read_at             MPI_File_write_at
MPI_File_read_at_all         MPI_File_write_at_all
MPI_File_read_at_all_begin   MPI_File_write_at_all_begin
MPI_File_read_at_all_end     MPI_File_write_at_all_end
MPI_File_read_shared         MPI_File_write_shared
MPI_File_read_ordered        MPI_File_write_ordered
MPI_File_read_ordered_begin  MPI_File_write_ordered_begin
MPI_File_read_ordered_end    MPI_File_write_ordered_end
MPI_File_iread               MPI_File_iwrite
MPI_File_iread_all           MPI_File_iwrite_all
MPI_File_iread_at            MPI_File_iwrite_at
MPI_File_iread_at_all        MPI_File_iwrite_at_all
MPI_File_iread_shared        MPI_File_iwrite_shared
MPI_File_set_size            MPI_File_set_view            MPI_File_set_info
MPI_File_get_size            MPI_File_get_view            MPI_File_get_info
MPI_File_get_group           MPI_File_get_amode
MPI_File_preallocate
```

#### Pack/Unpack Operations

```
MPI_Pack             MPI_Pack_external
MPI_Unpack           MPI_Unpack_external
```

---

## OpenSHMEM Library Trace

<a id="openshmem-traced"></a>

If OpenSHMEM library trace is selected, Nsight Systems will trace the subset of OpenSHMEM API functions that are most likely to be involved in performance bottlenecks. To keep overhead low, Nsight Systems does not trace all functions.

<a id="openshmem-not-traced"></a>

### OpenSHMEM 1.5 Functions NOT Traced

The following OpenSHMEM functions are excluded from tracing to keep overhead low:

```
shmem_my_pe                        shmem_n_pes
shmem_global_exit                  shmem_pe_accessible
shmem_addr_accessible              shmem_ctx_create
shmem_ctx_destroy                  shmem_ctx_get_team
shmem_global_exit                  shmem_info_get_version
shmem_info_get_name                shmem_my_pe
shmem_n_pes                        shmem_pe_accessible
shmem_ptr                          shmem_query_thread
shmem_team_create_ctx              shmem_team_destroy
shmem_team_get_config              shmem_team_my_pe
shmem_team_n_pes                   shmem_team_translate_pe
shmem_team_split_2d                shmem_team_split_strided
shmem_test*
```

---

## UCX API Trace

If UCX API trace is selected, Nsight Systems will trace the subset of functions of the UCX protocol layer UCP that are most likely to be involved in performance bottlenecks. To keep overhead low, Nsight Systems does not trace all functions.

<a id="ucx-environment-variables"></a>

### Environment Variables

The following environment variables control what is recorded:

| Variable | Default | Description |
|---|---|---|
| `NSYS_UCP_COMM_SUBMIT` | Enabled | If set to 0, UCP communication submission calls are not recorded. These calls are usually short, because the communication itself is handled in a worker thread. |
| `NSYS_UCP_COMM_PROGRESS` | Enabled | If set to 0, tracking of (process-local) UCP communication progress is disabled. The progress tracking uses UCP completion callbacks. |
| `NSYS_UCP_COMM_PARAMS` | Enabled | If set to 0, UCP communication parameters (tag, remote worker UID, packed message size, buffer address) will not be recorded. Recording the remote worker UID requires UCX >= 1.12.0. Recording the packed message size requires UCX >= 1.14.0. |

<a id="ucx-functions-traced"></a>

### UCX Functions Traced

```
ucp_am_send_nb[x]              ucp_am_recv_data_nbx
ucp_am_data_release

ucp_atomic_add{32,64}          ucp_atomic_cswap{32,64}
ucp_atomic_fadd{32,64}         ucp_atomic_swap{32,64}
ucp_atomic_post                 ucp_atomic_fetch_nb
ucp_atomic_op_nbx

ucp_cleanup                     ucp_config_modify
ucp_config_read                 ucp_config_release

ucp_disconnect_nb

ucp_dt_create_generic          ucp_dt_destroy

ucp_ep_create                  ucp_ep_destroy
ucp_ep_modify_nb               ucp_ep_close_nbx
ucp_ep_flush                   ucp_ep_flush_nb
ucp_ep_flush_nbx

ucp_listener_create            ucp_listener_destroy
ucp_listener_query             ucp_listener_reject

ucp_mem_advise                 ucp_mem_map
ucp_mem_unmap                  ucp_mem_query

ucp_put[_nbi]                  ucp_get[_nbi]
ucp_put_nb[x]                  ucp_get_nb[x]

ucp_request_alloc              ucp_request_cancel
ucp_request_is_completed

ucp_rkey_buffer_release        ucp_rkey_destroy
ucp_rkey_pack                   ucp_rkey_ptr

ucp_stream_data_release        ucp_stream_recv_data_nb
ucp_stream_send_nb[x]          ucp_stream_recv_nb[x]
ucp_stream_worker_poll

ucp_tag_msg_recv_nb[x]         ucp_tag_send_nbr
ucp_tag_recv_nb[x]             ucp_tag_send_nb[x]
ucp_tag_send_sync_nb[x]

ucp_worker_create              ucp_worker_destroy
ucp_worker_get_address         ucp_worker_get_efd
ucp_worker_arm                 ucp_worker_fence
ucp_worker_wait                ucp_worker_signal
ucp_worker_wait_mem            ucp_worker_flush
ucp_worker_flush_nb            ucp_worker_flush_nbx

ucp_worker_set_am_handler      ucp_worker_set_am_recv_handler
```

<a id="ucx-functions-not-traced"></a>

### UCX Functions NOT Traced

```
ucp_config_print               ucp_conn_request_query
ucp_context_query               ucp_context_print_info
ucp_get_version                 ucp_get_version_string
ucp_ep_close_nb                ucp_ep_print_info
ucp_ep_query                   ucp_ep_rkey_unpack
ucp_mem_print_info
ucp_request_check_status       ucp_request_free
ucp_request_query              ucp_request_release
ucp_request_test
ucp_stream_recv_request_test
ucp_tag_probe_nb               ucp_tag_recv_request_test
ucp_worker_address_query       ucp_worker_print_info
ucp_worker_progress            ucp_worker_query
ucp_worker_release_address
```

Additional API functions from other UCX layers may be added in a future version of the product.

---

## NVIDIA NVSHMEM and NCCL Trace

<a id="nvshmem-nccl-trace"></a>

The NVIDIA network communication libraries NVSHMEM and NCCL have been instrumented using NVTX annotations. To enable tracing these libraries in Nsight Systems, turn on NVTX tracing in the GUI or CLI.

To enable the NVTX instrumentation of the NVSHMEM library, make sure that the environment variable `NVSHMEM_NVTX` is set properly; e.g., `NVSHMEM_NVTX=common`.

```bash
# Trace NCCL via NVTX
nsys profile --trace=nvtx my_nccl_application

# Trace NVSHMEM with NVTX instrumentation enabled
NVSHMEM_NVTX=common nsys profile --trace=nvtx my_nvshmem_application
```

---

## NIC Metric Sampling

<a id="nic-overview"></a>

### Overview

NVIDIA ConnectX smart network interface cards (smart NICs) offer advanced hardware offloads and accelerations for network operations. Viewing smart NICs metrics on the Nsight Systems timeline enables developers to better understand their application's network usage. Developers can use this information to optimize the application's performance.

<a id="nic-requirements"></a>

### Requirements

- NIC metric sampling supports NVIDIA ConnectX boards starting with **ConnectX 5**
- NIC metric sampling is supported on **Linux x86_64** and **Arm Server (SBSA)** machines only
- Minimum Linux kernel **4.12**
- Minimum **MLNX_OFED 4.1**
- If collecting NIC metrics within a container, make sure that the container has access to the driver on the host machine

To check if OFED is installed and get its version:

```bash
/usr/bin/ofed_info

# Alternative method
cat /sys/module/"$(cat /proc/modules | grep -o -E "^mlx._core")"/version
```

To check if the target system meets the requirements for NIC metrics collection:

```bash
nsys status --network
```

You can download the latest and archived versions of the MLX_OFED driver from the MLNX_OFED Download Center.

<a id="nic-cli"></a>

### Collecting NIC Metrics Using the Command Line

To collect NIC performance metrics using Nsight Systems CLI, add the `--nic-metrics` command line switch:

```bash
nsys profile --nic-metrics=true my_app
```

<a id="nic-available-metrics"></a>

### Available Metrics

| Metric | Description |
|---|---|
| **Bytes sent** | Number of bytes sent through all NIC ports |
| **Bytes received** | Number of bytes received by all NIC ports |
| **Average sent packet size** | Average byte size of packets sent through all NIC ports |
| **Average received packet size** | Average byte size of packets received by all NIC ports |
| **CNPs sent** | Number of congestion notification packets sent by the NIC |
| **CNPs received** | Number of congestion notification packets received and handled by the NIC |
| **Send waits** | The number of ticks during which ports had data to transmit but no data was sent during the entire tick (either because of insufficient credits or because of lack of arbitration) |

> **Note:** Each one of the mentioned metrics is shown only if it has a non-zero value during profiling.

<a id="nic-usage-examples"></a>

### Usage Examples

- **Bytes sent/sec** and **Bytes received/sec** metrics enable identifying idle and busy NIC times. Developers may shift network operations from busy to idle times to reduce network congestion and latency.
- Developers can use idle NIC times to send additional data without reducing application performance.
- **CNPs** (congestion notification packets) received/sent and **Send waits** metrics may explain network latencies. A developer seeing the time periods when the network was congested may rewrite their algorithm to avoid the observed congestions.

---

## InfiniBand Switch Metric Sampling

<a id="ib-switch-metrics"></a>

NVIDIA Quantum InfiniBand switches offer high-bandwidth, low-latency communication. Viewing switch metrics on the Nsight Systems timeline enables developers to better understand their application's network usage.

### Requirements

- IB switch metric sampling supports all **NVIDIA Quantum switches**
- The user needs to have permission to query the InfiniBand switch metrics
- To check if the current user has permissions, check that the user has permission to access `/dev/infiniband/umad*`
- To give user permissions to query InfiniBand switch metrics on RedHat systems, follow the directions at RedHat Solutions

### Collecting InfiniBand Switch Metrics via CLI

To collect InfiniBand switch performance metrics, add the `--ib-switch-metrics-device` command line switch, followed by a comma-separated list of InfiniBand switch GUIDs:

```bash
nsys profile --ib-switch-metrics-device=<IB switch GUID> my_app
```

To get a list of InfiniBand switches reachable by a given NIC:

```bash
sudo ibswitches -C <nic name>
```

### Available Switch Metrics

| Metric | Description |
|---|---|
| **Bytes sent** | Number of bytes sent through all switch ports |
| **Bytes received** | Number of bytes received by all switch ports |
| **Send waits** | The number of ticks during which switch ports, selected by PortSelect, had data to transmit but no data was sent during the entire tick (either because of insufficient credits or because of lack of arbitration) |
| **Average sent packet size** | Average sent InfiniBand packet size |
| **Average received packet size** | Average received InfiniBand packet size |

---

## InfiniBand Switch Congestion Events

<a id="ib-switch-congestion"></a>

### Overview

NVIDIA Quantum InfiniBand switches offer high-bandwidth, low-latency communication. When a switch egress port is congested, packets wait in the egress port queue before being sent out of the switch. This increases the latency of these packets.

Nsight Systems Workstation Edition gives you the ability to view when switch egress ports are congested on the Nsight Systems timeline. This enables developers to better understand latencies that are caused by the application's network usage.

### Requirements

- **Quantum 2 switch** or newer
- Firmware version **31.2012.1068** or higher
- User needs to have permission to send management datagrams
- To get a list of InfiniBand switches reachable by a given NIC: `sudo ibswitches -C <nic name>`
- To check if the current user has permissions, check access to `/dev/umad`

<a id="ib-congestion-switches"></a>

### Configuration Switches

To collect InfiniBand switch congestion events using the Nsight Systems CLI, add the following command line switches:

| Switch | Description |
|---|---|
| `--ib-switch-congestion-device` | Followed by a comma-separated list of InfiniBand switch GUIDs from which congestion events will be collected. |
| `--ib-switch-congestion-nic-device` | Followed by the name of the NIC (HCA) through which InfiniBand switches will be accessed. The profiled InfiniBand switches should be reachable by this NIC. |
| `--ib-switch-congestion-percent` | Defines the percent of InfiniBand switch congestion events to be collected. This option enables reducing the network bandwidth consumed by reporting congestion events. Values are in the **[1,100]** range. |
| `--ib-switch-congestion-threshold-high` | Defines the high threshold for InfiniBand switch egress port queue size. When a packet enters an InfiniBand switch, its data is stored at an ingress port buffer. A pointer to the packet's data is inserted into the egress port's queue, from which the packet will be exiting the switch. At that point, the threshold given by this command switch is compared to the egress queue data size. If the queue data size exceeds the threshold, a congestion event is reported. The threshold is given in percent of the ingress port size. An egress port queue can point to data coming from multiple ingress port buffers, therefore the threshold can be bigger than 100%. Values are in the **(1,1023]** range. |

---

## InfiniBand Network Information

<a id="ib-network-info"></a>

### Overview

By default, Nsight Systems displays low-level identifiers like LIDs (Local Identifiers) and GUIDs (Globally Unique Identifiers). Instead, Nsight Systems can leverage InfiniBand network information to display the actual names of nodes and switches. This makes the Nsight Systems reports much more intuitive and easier to understand at a glance.

InfiniBand network information discovery is done using the `ibdiagnet` utility. There are two approaches:

1. **Pre-generated files**: Run `ibdiagnet` and store the generated network information files to be later used by Nsight Systems. This method is useful for large networks, where `ibdiagnet`'s network discovery time may be long, and for networks where only administrators have permissions to query the network information.

2. **Runtime discovery**: Ask Nsight Systems to run `ibdiagnet` to collect the network information during the profiling session. This method is useful for small networks.

### Requirements

The user needs to have permission to send MADs (management datagrams). To check if you have permission to send MADs, check if you can access the `/dev/infiniband/umad*` files. To give user permissions to send MADs on RedHat systems, follow the directions given at RedHat Solutions.

### Relevant CLI Switches

| Switch | Description |
|---|---|
| `--ib-net-info-devices` | Followed by a comma-separated list of NIC names from which `ibdiagnet` will run network discovery. The results of the network discovery will be automatically loaded into Nsight Systems. |
| `--ib-net-info-files` | Followed by a comma-separated list of pre-generated `ibdiagnet` db_csv file paths which Nsight Systems will read. |
| `--ib-net-info-output` | Followed by a path of a directory into which Nsight Systems will store the `ibdiagnet` network discovery data. These files will be used by the `--ib-net-info-devices` command line switch. This command line switch can only be used together with `--ib-net-info-devices`. |

---

## Amazon AWS EFA Metrics

<a id="aws-efa-metrics"></a>

Nsight Systems can now periodically sample performance counters for AWS Elastic Fabric Adapters (EFAs) and plot them on the timeline in the GUI. This enables developers to analyze how network communications may be involved with the critical path of their multi-node application.

Created in collaboration with AWS, this plugin works on AWS EC2 NVIDIA GPU accelerated compute instances.

### Enabling EFA Metrics

To enable the AWS EFA metrics, add the following option to the `nsys profile` or `nsys start` commands:

```
--enable efa_metrics[,arg1[=value1],arg2[=value2], ...]
```

There are no spaces following the `efa_metrics` plugin name. It is followed by a comma-separated list of arguments or `argument=value` pairs. Arguments with spaces should be enclosed in double quotes.

### Supported Arguments

| Name | Possible Parameters | Default | Switch Description |
|---|---|---|---|
| `-efa-non-rdma` | `true`, `false` | `false` | Sample InfiniBand non-RDMA counters |
| `-efa-sysfs` | `<path>` | `/sys/class/infiniband` | Root directory for EFA counters sysfs |
| `-efa-work-requests` | `true`, `false` | `false` | Sample InfiniBand WorkRequest counters |
| `-errors` | `true`, `false` | `false` | Sample error counters |
| `-freq` | integer (negative means 1/F frequency) | `10` | Target sample frequency in hertz |
| `-mode` | `throughput`, `delta`, `total` | `throughput` | Report sampled counters as a value per second, delta since previous sample, or an accumulated sum |
| `-packets` | `true`, `false` | `false` | Sample packet counters |

### Usage Examples

```bash
# Sample all EFA adapters, display as bytes per second
nsys profile --enable efa_metrics ...

# Sample all available EFA adapter counters
nsys profile --enable efa_metrics,-packets,-errors,-efa-non-rdma ...

# Sample all EFA adapters, display as total value sum since profiling start
nsys profile --enable efa_metrics,-mode=total ...

# Look for EFA counters in a different sysfs directory (useful in some k8s environments)
nsys profile --enable efa_metrics,-efa-counters-sysfs="/mnt/nv/sys" ...
```

This collector is the first use case for the Nsight Systems Plugins (Preview) system.

---

## Network Interface Metrics

<a id="network-interface-metrics"></a>

Nsight Systems can periodically sample performance counters for network interface devices and plot them on the timeline in the GUI.

### Enabling Network Interface Metrics

To enable the network device metrics, add the following option to the `nsys profile` or `nsys start` commands:

```
--enable network_interface[,arg1[=value1],arg2[=value2], ...]
```

There are no spaces following the `network_interface` plugin name. It is followed by a comma-separated list of arguments or `argument=value` pairs. Arguments with spaces should be enclosed in double quotes.

### Supported Arguments

| Short Name | Long Name | Possible Parameters | Default | Switch Description |
|---|---|---|---|---|
| `-i` | `--interval` | integer | `100000` | Sampling interval in microseconds |
| `-d` | `--device` | regular expression | `".+"` (and filtering for physical devices) | Device(s) to sample |
| `-m` | `--metric` | regular expression | `".*_bytes"` | Metric(s) to sample |
| `-h` | `--help` | - | - | Print help message |

### Usage Examples

```bash
# Sample bytes metrics for all physical network devices every 100ms
nsys profile --enable network_interface ...

# Sample bytes metrics for all network devices every 100ms
nsys profile --enable network_interface,-dall ...

# Sample all metrics, for all network devices, every 10ms
nsys profile --enable network_interface,-i10000,-dall,-m".+"
```

For general information on Nsight Systems plugins, refer to the Nsight Systems Plugins (Preview) system documentation.

---

## Storage Metrics Profiling

<a id="storage-metrics"></a>

Nsight Systems can profile several major storage / remote storage protocols. To activate this feature, use the Nsight Systems CLI `--storage-metrics` option, followed by a comma-separated list of the desired arguments.

<a id="storage-arguments"></a>

### Available Arguments

| Argument | Description |
|---|---|
| `--nfs-volumes={all \| volume1[,volume2][,volume3..]}` | Enable NFS storage profiling for the specified volume(s). Specify `all` to profile all volumes. |
| `--lustre-volumes={all \| volume1[,volume2][,volume3..]}` | Enable Lustre storage profiling for the specified volume(s). Specify `all` to profile all volumes. |
| `--lustre-llite-dir=<path>` | Specifies the path of the llite directory mount. This is the `/sys/kernel/debug/lustre/llite` directory mount point (mandatory if Lustre profiling is enabled). |
| `--storage-devices={all \| device1[,device2][,device3..]}` | Enable storage profiling of the specified local storage or NVMeOF device(s). Specify `all` to profile all devices. |
| `--gds-metrics={driver}` | Enable GPUDirect Storage Kernel-Space metrics profiling. |

### Viewing Storage Metrics

In the report file, under **Timeline view**, the storage metrics can be viewed in the **Mounts** section. Each row contains metrics for one volume or device, with the storage type next to the volume / device name. Expanding each row will show the collected metrics for that volume / device.

The stdout and stderr log files for the storage metrics collection process can be viewed under the **Files** section, which may assist in debugging.

### Read/Write Metric Types

There are two types of Read/Write metrics:

| Type | Description |
|---|---|
| **Application-level Read/Write** | Displays quantities of data read/written to the storage device by applications (in Bytes). |
| **Driver-level Read/Write** | Displays throughput of data read/written to the storage device by the driver (in Bytes/sec). |

For example, when an application uses the `write` POSIX function to write 10 MB of data into a file, the entire 10 MB will appear, in a single sampling point, at the Application-level Write counter. The same 10 MB of data may be spread across multiple Driver-level Write counter sampling points, since it may take a bit of time for the NFS driver to write 10 MB of data into the NFS storage server.

### Exposing Lustre Driver Counters to Non-Privileged Users

The Lustre driver exposes performance counters via virtual files residing under `/sys/kernel/debug/lustre`. However, this path is not accessible to non-privileged users.

To expose the Lustre counters to non-privileged users, a superuser should create a mount point to `/sys/kernel/debug/lustre`:

```bash
su - root
mkdir /mnt/lustre-stats
mount --bind /sys/kernel/debug/lustre /mnt/lustre-stats
```

The `--lustre-llite-dir=` command line argument should point to the llite directory under this mount point; this will enable Nsight Systems to read the Lustre counters. For example: `--lustre-llite-dir=/mnt/lustre-stats/llite`

<a id="nfs-example"></a>

### NFS Storage Example

Example Nsight Systems command line for NFS storage profiling:

```bash
./nsys profile --storage-metrics --nfs-volumes=all <target-application>
```

<a id="lustre-example"></a>

### Lustre Storage Example

Example Nsight Systems command line for Lustre storage profiling:

```bash
./nsys profile --storage-metrics --lustre-volumes=dtdata --lustre-llite-dir=/mnt/lustre-stats/llite <target-application>
```

<a id="nvmeof-example"></a>

### Local / NVMeOF Storage Example

Example Nsight Systems command line for local storage and NVMeOF device profiling:

```bash
./nsys profile --storage-metrics --storage-devices=all <target-application>
```

It is also possible to use combinations of these arguments to profile multiple storage protocols at once:

```bash
./nsys profile --storage-metrics \
  --nfs-volumes=all \
  --lustre-volumes=all \
  --storage-devices=<device_name1>,<device_name2> \
  --lustre-llite-dir=<path_to_llite_directory> \
  <target-application>
```

<a id="gds-metrics"></a>

### GPUDirect Storage Installation and Metrics Collection

Before collecting GDS (GPUDirect Storage) metrics, ensure that NVIDIA GPUDirect Storage is installed on your system. For installation instructions, refer to the NVIDIA GPUDirect Storage Installation and Troubleshooting Guide.

Once GDS is installed, you can enable GDS Kernel-Space metrics profiling by using the `--gds-metrics=driver` command line argument. The GDS metrics can be viewed in the **GPUDirect Storage** section of the report file, under **Timeline view**.

Example Nsight Systems command line for GDS Kernel-Space profiling:

```bash
./nsys profile --storage-metrics --gds-metrics=driver <target-application>
```

---

## GPUDirect Storage Trace

<a id="gds-trace"></a>

NVIDIA GPUDirect Storage (GDS) enables direct memory access (DMA) between storage and GPU memory. This avoids a bounce buffer through the CPU, increasing storage access bandwidth and decreasing latency and utilization load on the CPU. Information about GDS can be found at NVIDIA Magnum IO GPUDirect Storage.

Nsight Systems can capture information about GDS, specifically the various cuFile API calls made by the profiled process. GDS profiling is currently an **experimental feature**, and is supported on **Linux x64** and **SBSA** operating systems.

### Key Information

- GDS trace captures cuFile API calls (cuFileRead, cuFileWrite, etc.)
- Enables analysis of direct GPU-to-storage data transfers
- Helps identify bottlenecks in data loading pipelines that use direct GPU-storage paths
- Requires NVIDIA GPUDirect Storage to be installed

---

## Quick Reference: CLI Options Summary

| Option | Description |
|---|---|
| `--trace=mpi` | Enable MPI API tracing |
| `--mpi-impl=openmpi\|mpich` | Specify MPI implementation |
| `--trace=oshmem` | Enable OpenSHMEM tracing |
| `--trace=ucx` | Enable UCX API tracing |
| `--trace=nvtx` | Enable NVSHMEM/NCCL via NVTX |
| `--nic-metrics=true` | Enable NIC metric sampling |
| `--ib-switch-metrics-device=<GUID>` | Enable IB switch metric sampling |
| `--ib-switch-congestion-device=<GUID>` | Enable IB switch congestion events |
| `--ib-switch-congestion-nic-device=<name>` | Specify NIC for congestion events |
| `--ib-switch-congestion-percent=<N>` | Congestion event sampling percentage [1,100] |
| `--ib-switch-congestion-threshold-high=<N>` | Congestion queue threshold (1,1023] |
| `--ib-net-info-devices=<NICs>` | IB network info discovery |
| `--ib-net-info-files=<paths>` | Pre-generated IB network info files |
| `--ib-net-info-output=<dir>` | IB network info output directory |
| `--enable efa_metrics` | Enable AWS EFA metrics |
| `--enable network_interface` | Enable network interface metrics |
| `--storage-metrics` | Enable storage metrics profiling |
| `--nfs-volumes=<volumes>` | NFS volumes to profile |
| `--lustre-volumes=<volumes>` | Lustre volumes to profile |
| `--lustre-llite-dir=<path>` | Lustre llite directory path |
| `--storage-devices=<devices>` | Storage devices to profile |
| `--gds-metrics=driver` | Enable GPUDirect Storage metrics |
