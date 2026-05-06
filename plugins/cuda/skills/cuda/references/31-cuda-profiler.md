---
title: CUDA Profiler Tools Reference
---

# CUDA Profiler Tools (Visual Profiler, nvprof, NVTX)

NVIDIA profiling tools for understanding and optimizing CUDA application performance. Includes the Visual Profiler (nvvp) GUI, nvprof command-line profiler, and the NVIDIA Tools Extension (NVTX) API for application annotation.

> **Note:** Visual Profiler and nvprof are **deprecated** since CUDA 12.8. NVIDIA Volta is the last architecture fully supported. Use **NVIDIA Nsight Systems** (system-wide tracing) and **NVIDIA Nsight Compute** (kernel profiling) instead. See the Migration section at the end.

**Document Version:** Release 12.9 (May 2025)

---

## Table of Contents

1. [Profiling Overview & Terminology](#profiling-overview--terminology)
2. [Focused Profiling](#focused-profiling)
3. [Marking CPU Regions & NVTX](#marking-cpu-regions--nvtx)
4. [Naming Resources](#naming-resources)
5. [Flush Profile Data](#flush-profile-data)
6. [Visual Profiler (nvvp)](#visual-profiler-nvvp)
7. [nvprof Command-Line Tool](#nvprof-command-line-tool)
8. [NVIDIA Tools Extension (NVTX) API](#nvidia-tools-extension-nvtx-api)
9. [Remote Profiling](#remote-profiling)
10. [MPI Profiling](#mpi-profiling)
11. [MPS Profiling](#mps-profiling)
12. [Dependency Analysis](#dependency-analysis)
13. [Metrics Reference](#metrics-reference)
14. [Warp State Analysis](#warp-state-analysis)
15. [Migrating to Nsight Tools](#migrating-to-nsight-tools)
16. [Known Issues](#known-issues)

---

## Profiling Overview & Terminology

### Terminology

| Term | Definition |
|------|-----------|
| **Event** | A countable activity on a device corresponding to a single hardware counter value, collected during kernel execution. List with `nvprof --query-events`. |
| **Metric** | A characteristic calculated from one or more event values. List with `nvprof --query-metrics`. |
| **Timeline** | Chronological view of CPU and GPU activities (kernel executions, memory transfers, API calls). |
| **Critical Path** | Longest path through the event dependency graph without wait states; optimizing it directly reduces runtime. |

### Profiling Workflow

1. **Profile** - Collect data with nvprof or Visual Profiler (no code changes required)
2. **Analyze** - Identify bottlenecks via guided analysis or manual inspection
3. **Optimize** - Apply targeted improvements
4. **Verify** - Re-profile to confirm improvements

---

## Focused Profiling

Limit profiling to performance-critical regions to reduce data volume and focus analysis.

### API

```cpp
#include <cuda_profiler_api.h>

// Start profiling
cudaProfilerStart();

// ... performance-critical code ...

// Stop profiling
cudaProfilerStop();

// Driver API equivalent
#include <cudaProfiler.h>
cuProfilerStart();
cuProfilerStop();
```

### Configuration

| Tool | Flag to disable auto-profiling |
|------|-------------------------------|
| nvprof | `--profile-from-start off` |
| Visual Profiler | Uncheck "Start execution with profiling enabled" in Settings View |

### Common Use Cases

- Test harness applications: profile only the CUDA algorithm, not initialization/verification
- Multi-phase applications: profile each phase independently
- Iterative algorithms: profile a subset of iterations

---

## Marking CPU Regions & NVTX

Use the NVIDIA Tools Extension (NVTX) to annotate CPU activity regions visible in the profiler timeline.

```cpp
#include <nvToolsExt.h>

void criticalSection() {
    nvtxRangePushA("critical_section");
    // ... work ...
    nvtxMarkA("checkpoint");
    // ... more work ...
    nvtxRangePop();
}
```

nvprof shows NVTX markers and ranges in API trace output (timeline) and summary mode (associated CUDA activities).

---

## Naming Resources

Custom names for CPU threads, CUDA devices, contexts, and streams improve readability in profiling output.

```cpp
#include <nvToolsExt.h>

// Name OS thread
nvtxNameOsThreadA(pthread_self(), "MAIN_THREAD");

// Name CUDA device and stream
nvtxNameCudaDeviceA(0, "GPU_0");
cudaStream_t stream;
cudaStreamCreate(&stream);
nvtxNameCudaStreamA(stream, "data_load_stream");
```

---

## Flush Profile Data

Profile data is buffered and flushed asynchronously. To avoid data loss:

```cpp
// Ensure all GPU work completes
cudaDeviceSynchronize();

// Force flush of buffered profile data
cudaProfilerStop();  // or cuProfilerStop()
```

For applications with display loops that may exit unexpectedly, use nvprof `--timeout <seconds>` to force a flush before timeout.

---

## Visual Profiler (nvvp)

GUI profiler displaying CPU/GPU activity timelines with automated analysis engine.

### Launch

```bash
# Start new session
nvvp [executableName [args...]]

# Import nvprof data
nvvp data.nvprof

# Import multi-process data
nvvp data1.nvprof data2.nvprof ...
```

### Session Types

| Type | Description |
|------|-------------|
| **Executable Session** | Run and profile an application from within nvvp |
| **Import Session** | Import data collected by nvprof (read-only analysis) |

### Session Options

**CUDA Options:**
- Start execution with profiling enabled (default: on)
- Enable concurrent kernel profiling
- Enable CUDA API tracing
- Enable power, clock, thermal profiling
- Enable unified memory profiling
- Replay application to collect events/metrics
- Run guided analysis

**CPU Options:**
- Profile execution on CPU
- Enable OpenACC profiling (Linux, PGI 19.1+)
- Enable CPU thread tracing (Pthreads, Linux only)

### Timeline View Components

| Row Type | Shows |
|----------|-------|
| Process | One per profiled application |
| Thread | CPU threads making CUDA calls |
| Runtime API / Driver API | Duration of each CUDA API call |
| OpenACC / OpenMP | Parallel runtime activities |
| Markers and Ranges | NVTX annotations |
| Profiling Overhead | Profiler-induced overhead |
| Device | GPU devices with compute utilization |
| Unified Memory | CPU/GPU page faults, data migrations |
| Context / Stream | CUDA context and stream activities |
| Memcpy | Memory transfers (HtoD, DtoH, DtoD, P2P) |
| Compute / Kernel | GPU kernel execution durations |

### Analysis Modes

**Guided Analysis:** Step-by-step analysis from application-level down to kernel-specific issues.

**Unguided Analysis:** Manual exploration of all analysis stages:
- Compute utilization
- Memory analysis
- Kernel profiling (PC sampling, instruction execution)
- Dependency analysis

### PC Sampling

Available on CC 5.2+ (non-mobile). Samples PC and warp state at regular intervals:
- **Warp State View** (CC 5.2+): Shows why warps stalled
- **Latency Reasons View** (CC 6.0+): Shows reasons for holes in issue pipeline

### Memory Statistics View

Shows memory hierarchy usage during kernel execution (CC 5.0+):
- Cache hit rates for L1, L2, texture caches
- Data path throughput between SMs and memory spaces
- Read/write operation direction indicators

### Source-Disassembly View

Source-level analysis results (requires `-lineinfo` compiler flag):
- Global Memory Access Pattern Analysis
- Shared Memory Access Pattern Analysis
- Divergent Execution Analysis
- Instruction Execution Analysis
- PC Sampling Analysis

---

## nvprof Command-Line Tool

Command-line profiler for collecting and viewing profiling data.

### Basic Usage

```bash
nvprof [options] [application] [application-arguments]
nvprof --help  # Full help
```

### CUDA Profiling Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `--aggregate-mode` | on, off | on | Aggregate events/metrics across GPU or show per-unit |
| `--analysis-metrics` | N/A | N/A | Collect data for Visual Profiler analysis mode |
| `--annotate-mpi` | off, openmpi, mpich | off | Auto-annotate MPI calls with NVTX |
| `--concurrent-kernels` | on, off | on | Enable/disable concurrent kernel execution |
| `--cpu-thread-tracing` | on, off | off | Collect CPU thread API activity |
| `--dependency-analysis` | N/A | N/A | Run dependency analysis |
| `--device-buffer-size` | {MB} | 8 MB | Device memory for profiling data |
| `--devices` | {IDs}, all | all | Scope events/metrics to specific devices |
| `--events` (-e) | {names}, all | N/A | Events to profile |
| `--kernels` | {filter} | all | Scope to specific kernels (regex supported) |
| `--metrics` (-m) | {names}, all | N/A | Metrics to profile |
| `--profile-all-processes` | N/A | N/A | Profile all CUDA processes by same user |
| `--profile-child-processes` | N/A | N/A | Profile child processes |
| `--profile-from-start` | on, off | on | Enable/disable profiling from app start |
| `--replay-mode` | disabled, kernel, application | kernel | How to replay for multi-pass profiling |
| `--system-profiling` | on, off | off | Enable power/clock/thermal sampling |
| `--timeout` (-t) | {seconds} | N/A | Execution timeout |
| `--unified-memory-profiling` | per-process-device, off | per-process-device | UM profiling configuration |

### CPU Profiling Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `--cpu-profiling` | on, off | off | Enable CPU profiling |
| `--cpu-profiling-frequency` | {Hz} | 100Hz | Sampling frequency (max 500Hz) |
| `--cpu-profiling-mode` | flat, top-down, bottom-up | bottom-up | Output organization |
| `--cpu-profiling-max-depth` | {depth} | 0 (unlimited) | Max call stack depth |

### Print Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `--csv` | N/A | N/A | CSV output format |
| `--demangling` | on, off | on | C++ name demangling |
| `--normalized-time-unit` (-u) | s, ms, us, ns, auto | auto | Time unit in output |

### IO Options

| Option | Description |
|--------|-------------|
| `--export-profile` (-o) {file} | Export profile data (SQLite format) |
| `--import-profile` (-i) {file} | Import previous profile data |
| `--log-file` {file} | Redirect output (%1=stdout, %2=stderr, %p=PID, %h=hostname) |
| `--force-overwrite` (-f) | Overwrite existing output files |

### Profiling Modes

#### Summary Mode (Default)

One result line per kernel function and memcpy type:
```bash
nvprof ./matrixMul
# Output: Time(%), Time, Calls, Avg, Min, Max, Name
# Plus API call summary
```

#### GPU-Trace Mode

Timeline of all GPU activities with detailed per-invocation data:
```bash
nvprof --print-gpu-trace ./matrixMul
# Output: Start, Duration, Grid Size, Block Size, Regs, SSMem, DSMem,
#         Size, Throughput, Device, Context, Stream, Name
```

#### API-Trace Mode

Timeline of all CUDA runtime/driver API calls:
```bash
nvprof --print-api-trace ./matrixMul
# Output: Start, Duration, Name
```

#### Event/Metric Summary Mode

```bash
nvprof --events warps_launched,local_load --metrics ipc ./matrixMul
# Output per kernel: Min, Max, Avg for each event/metric
```

#### Event/Metric Trace Mode

Per-kernel-execution values with optional per-SM breakdown:
```bash
nvprof --aggregate-mode off --events local_load --print-gpu-trace ./matrixMul
```

### Kernel Filtering Syntax

```
--kernels "<context>:<stream>:<kernel_name>:<invocation>"
```

Each field supports Perl regex. Examples:
```bash
# All kernels containing "gemm"
--kernels "gemm"

# 2nd invocation of kernel "bar" on context 1, stream "foo"
--kernels "1:foo:bar:2"

# All 3rd invocations of every kernel
--kernels ":::3"
```

### Replay Modes

| Mode | Behavior |
|------|----------|
| `kernel` (default) | Replay each kernel individually for multi-pass collection |
| `application` | Re-run entire application per pass (better for large device memory) |
| `disabled` | No replay; drop events/metrics that can't be collected in one pass |

### Profiling Scope Examples

```bash
# Collect ipc metric on device 0
nvprof --devices 0 --metrics ipc ./app

# Collect events for specific kernel
nvprof --devices 0 --kernels "1:foo:bar:2" --events local_load ./app

# Analysis metrics for a specific kernel
nvprof --kernels "myKernel" --analysis-metrics -o analysis.prof ./app
```

### Output Redirection

```bash
# To file
nvprof --log-file output.txt ./app

# To stdout
nvprof --log-file %1 ./app

# Per-process files (MPI)
nvprof -o output.%h.%p.%q{OMPI_COMM_WORLD_RANK} ./mpi_app
```

### CPU Sampling

```bash
nvprof --cpu-profiling on --cpu-profiling-mode bottom-up ./app
```

Limitations: Not supported on mobile, multi-process mode, or with CSV output.

---

## NVIDIA Tools Extension (NVTX) API

C-based API for annotating events, code ranges, and resources in applications.

### Files

| File | Description |
|------|-------------|
| `nvToolsExt.h` | Core NVTX API |
| `nvToolsExtCuda.h` | CUDA-specific extensions |
| `nvToolsExtCudaRt.h` | CUDA Runtime extensions |
| `libnvToolsExt.so` (Linux) | Shared library |

### NVTX Markers

Instantaneous event annotation:
```cpp
nvtxMarkA("My mark");

// With attributes
nvtxEventAttributes_t eventAttrib = {0};
eventAttrib.version = NVTX_VERSION;
eventAttrib.size = NVTX_EVENT_ATTRIBSTRUCT_SIZE;
eventAttrib.colorType = NVTX_COLOR_ARGB;
eventAttrib.color = 0xFFFF0000; // Red
eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
eventAttrib.message.ascii = "Mark with attributes";
nvtxMarkEx(&eventAttrib);
```

### NVTX Range Start/Stop

Arbitrary (potentially non-nested, cross-thread) time spans:
```cpp
nvtxRangeId_t id1 = nvtxRangeStartA("My range");
// ... work ...
nvtxRangeEnd(id1);

// Overlapping ranges
nvtxRangeId_t r1 = nvtxRangeStartA("Range 0");
nvtxRangeId_t r2 = nvtxRangeStartA("Range 1");
nvtxRangeEnd(r1);
nvtxRangeEnd(r2);
```

### NVTX Range Push/Pop

Nested time spans (must be same thread):
```cpp
nvtxRangePushA("outer");
nvtxRangePushA("inner");
nvtxRangePop(); // end "inner"
nvtxRangePop(); // end "outer"
```

### Event Attributes Structure

```cpp
nvtxEventAttributes_t eventAttrib = {0};
eventAttrib.version = NVTX_VERSION;              // Required
eventAttrib.size = NVTX_EVENT_ATTRIBSTRUCT_SIZE; // Required
// Optional attributes:
eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
eventAttrib.message.ascii = "My event";
eventAttrib.category = 1;             // Grouping ID
eventAttrib.colorType = NVTX_COLOR_ARGB;
eventAttrib.color = 0xFF00FF00;       // Green
```

| Attribute | Description | Default |
|-----------|-------------|---------|
| message | String label | NVTX_MESSAGE_UNKNOWN |
| category | User-controlled grouping ID | 0 |
| color | ARGB color for visualization | Default color |
| payload | Additional numeric data | None |

### NVTX Synchronization Markers

Track custom synchronization (e.g., spinlocks, atomic-based mutexes):
```cpp
// Create user sync object
nvtxSyncUserAttributes_t attributes = {0};
attributes.version = NVTX_VERSION;
attributes.size = NVTX_SYNC_USER_ATTRIBSTRUCT_SIZE;
attributes.messageType = NVTX_MESSAGE_TYPE_ASCII;
attributes.message.ascii = "my_mutex";
nvtxSyncUser_t hSync = nvtxDomainSyncUserCreate(domain, &attributes);

// Acquire
nvtxDomainSyncUserAcquireStart(hSync);
bool acquired = __sync_bool_compare_and_swap(&bLocked, 0, 1);
if (acquired) nvtxDomainSyncUserAcquireSuccess(hSync);
else nvtxDomainSyncUserAcquireFailed(hSync);

// Release
nvtxDomainSyncUserReleasing(hSync);
nvtxDomainSyncUserDestroy(hSync);
```

### NVTX Domains

Scope annotations to avoid conflicts:
```cpp
nvtxDomainHandle_t domain = nvtxDomainCreateA("Domain_A");
// Use domain-scoped APIs: nvtxDomainMarkEx, nvtxDomainRangePushEx, etc.
nvtxDomainDestroy(domain);
```

Each domain maintains its own categories, thread range stacks, and registered strings.

### NVTX Resource Naming

```cpp
// OS Thread
nvtxNameOsThreadA(pthread_self(), "WORKER_THREAD");

// CUDA Runtime
nvtxNameCudaDeviceA(0, "my_device");
nvtxNameCudaStreamA(stream, "my_stream");

// CUDA Driver
nvtxNameCuDeviceA(device, "my_device");
nvtxNameCuContextA(context, "my_context");
nvtxNameCuStreamA(stream, "my_stream");
```

### NVTX String Registration

Improve performance by registering strings once:
```cpp
nvtxDomainHandle_t domain = nvtxDomainCreateA("Domain_A");
nvtxStringHandle_t message = nvtxDomainRegisterStringA(domain, "registered string");
nvtxEventAttributes_t eventAttrib = {0};
eventAttrib.version = NVTX_VERSION;
eventAttrib.size = NVTX_EVENT_ATTRIBSTRUCT_SIZE;
eventAttrib.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
eventAttrib.message.registered = message;
```

---

## Remote Profiling

### With nvprof (Recommended)

Collect data on remote system, view on host:

```bash
# On remote system: collect timeline
nvprof -o timeline.prof ./app

# On remote system: collect metrics
nvprof --metrics achieved_occupancy,IPC -o metrics.prof ./app

# On remote system: collect analysis for specific kernel
nvprof --kernels "kernelName" --analysis-metrics -o analysis.prof ./app

# On host: import into Visual Profiler
nvvp timeline.prof
```

### With Visual Profiler

Direct remote profiling via SSH. Requirements:
- Same CUDA Toolkit version on host and remote
- Remote must be Linux, accessible via SSH
- Host does not need NVIDIA GPU

### One-Hop Profiling

For setups with an intermediate login node between the Visual Profiler host and the compute node, use the one-hop profiling Perl script.

---

## MPI Profiling

### Automatic MPI Annotation

```bash
# OpenMPI
mpirun -np 2 nvprof --annotate-mpi openmpi ./my_mpi_app

# MPICH
mpirun -np 2 nvprof --annotate-mpi mpich ./my_mpi_app
```

### Manual MPI Profiling

```bash
# Per-process output files
mpirun -np 2 nvprof -o output.%h.%p.%q{OMPI_COMM_WORLD_RANK} ./mpi_app

# Profile all processes on a node
nvprof --profile-all-processes -o output.%h.%p
```

### Named Resources for MPI

```bash
nvprof --process-name "MPI Rank %q{OMPI_COMM_WORLD_RANK}" \
        --context-name "MPI Rank %q{OMPI_COMM_WORLD_RANK}" \
        -o output.%h.%p.%q{OMPI_COMM_WORLD_RANK} ./mpi_app
```

---

## MPS Profiling

### With nvprof

```bash
# Start MPS daemon
nvidia-cuda-mps-control -d

# Profile all MPS clients
nvprof --profile-all-processes -o output_%p

# Run application in separate terminal
./my_app

# Exit nvprof with Ctrl-c

# Import multi-process data into Visual Profiler
nvvp output_*
```

### With Visual Profiler

1. Launch MPS daemon: `nvidia-cuda-mps-control -d`
2. Create session with "Profile all processes" option
3. Run application in separate terminal
4. Press "Cancel" in Visual Profiler to stop and load data

Note: Event/metric profiling serializes MPS clients (one at a time).

---

## Dependency Analysis

Analyzes execution dependencies between CPU threads and CUDA streams to find optimization opportunities.

### Metrics

| Metric | Description |
|--------|-------------|
| **Waiting Time** | Duration an activity is blocked waiting on another thread/stream. Indicates load imbalance. |
| **Time on Critical Path** | Duration on the longest dependency path without wait states. Optimizing these activities directly reduces runtime. |

### Usage

```bash
# With nvprof
nvprof --dependency-analysis ./app

# With trace output
nvprof --dependency-analysis --print-dependency-analysis-trace ./app

# For multi-threaded apps, enable CPU thread tracing
nvprof --cpu-thread-tracing on --dependency-analysis ./app
```

### Supported APIs

- CUDA runtime and driver API
- POSIX threads (Pthreads), mutexes, condition variables

### Limitations

- Does not model resource contention (e.g., single copy engine)
- Does not track custom busy-wait synchronization
- Limited CDP support
- POSIX semaphores not supported
- Does not support `cudaLaunchCooperativeKernelMultiDevice`

---

## Metrics Reference

### Key Metrics by Category

#### Occupancy & Execution

| Metric | Description |
|--------|-------------|
| `achieved_occupancy` | Ratio of average active warps to maximum number of warps |
| `sm_efficiency` | Percentage of time at least one warp is active on an SM |
| `ipc` | Instructions executed per cycle |
| `issued_ipc` | Instructions issued per cycle |
| `eligible_warps_per_cycle` | Average warps eligible to issue per active cycle |
| `warp_execution_efficiency` | Ratio of average active threads per warp to maximum |

#### Memory Throughput

| Metric | Description |
|--------|-------------|
| `gld_efficiency` | Ratio of requested to required global load throughput |
| `gst_efficiency` | Ratio of requested to required global store throughput |
| `gld_throughput` | Global memory load throughput |
| `gst_throughput` | Global memory store throughput |
| `dram_read_throughput` | Device memory read throughput |
| `dram_write_throughput` | Device memory write throughput |
| `l2_read_throughput` | L2 cache read throughput |
| `l2_write_throughput` | L2 cache write throughput |
| `shared_load_throughput` | Shared memory load throughput |
| `shared_store_throughput` | Shared memory store throughput |

#### Memory Transactions

| Metric | Description |
|--------|-------------|
| `gld_transactions` | Number of global memory load transactions |
| `gst_transactions` | Number of global memory store transactions |
| `gld_transactions_per_request` | Average global load transactions per request |
| `gst_transactions_per_request` | Average global store transactions per request |
| `l2_read_transactions` | L2 cache read transactions |
| `l2_write_transactions` | L2 cache write transactions |
| `atomic_transactions` | Global memory atomic/reduction transactions |
| `l2_atomic_throughput` | L2 cache throughput for atomic/reduction requests |

#### FLOPS

| Metric | Description |
|--------|-------------|
| `flop_count_sp` | Single-precision FLOPs executed |
| `flop_count_dp` | Double-precision FLOPs executed |
| `flop_count_hp` | Half-precision FLOPs executed |
| `flop_sp_efficiency` | Ratio of achieved to peak SP FLOPs |
| `flop_dp_efficiency` | Ratio of achieved to peak DP FLOPs |

#### Cache

| Metric | Description |
|--------|-------------|
| `tex_cache_hit_rate` | Unified cache hit rate |
| `l2_tex_hit_rate` | L2 hit rate for texture cache requests |
| `global_hit_rate` | Hit rate for global loads in L1/tex cache |
| `local_hit_rate` | Hit rate for local loads and stores |
| `shared_efficiency` | Ratio of requested to required shared memory throughput |

#### Stall Reasons

| Metric | Description |
|--------|-------------|
| `stall_not_selected` | Warp ready but not selected for issue |
| `stall_memory_dependency` | Waiting for previous memory access |
| `stall_exec_dependency` | Waiting for input from earlier instructions |
| `stall_sync` | Blocked at `__syncthreads()` barrier |
| `stall_inst_fetch` | Next instruction not yet available |
| `stall_memory_throttle` | Too many outstanding memory requests |
| `stall_texture` | Texture sub-system fully utilized |
| `stall_pipe_busy` | Required functional unit is busy |
| `stall_constant_memory_dependency` | Constant cache miss |
| `stall_other` | Uncommon/unknown reasons |

#### Interconnect

| Metric | Description |
|--------|-------------|
| `pcie_total_data_received` | Total bytes received via PCIe |
| `pcie_total_data_transmitted` | Total bytes transmitted via PCIe |
| `nvlink_total_data_received` | Total bytes received via NVLink (CC 6.0+) |
| `nvlink_total_data_transmitted` | Total bytes transmitted via NVLink (CC 6.0+) |
| `nvlink_receive_throughput` | NVLink receive throughput (CC 6.0+) |
| `nvlink_transmit_throughput` | NVLink transmit throughput (CC 6.0+) |

### Compute Capability Support

| CC | Metrics Support |
|----|----------------|
| 5.x | Full metric set (single/multi-context) |
| 6.x | Adds NVLink metrics, `unique_warps_launched` |
| 7.x | Adds `stall_sleeping`, `tensor_precision_fu_utilization`, `tensor_int_fu_utilization` |
| 8.0+ | Not supported by Visual Profiler/nvprof; use Nsight Compute |

---

## Warp State Analysis

Detailed stall reason analysis for optimizing kernel performance.

### Stall Reasons and Mitigation

#### Instruction Fetch Stall
Next instruction not available.
**Fix:** Reduce loop unrolling, inline small functions, fuse short kernels, use larger thread blocks with occasional `__syncthreads()`.

#### Execution Dependency Stall
Waiting for earlier instruction inputs.
**Fix:** Increase instruction-level parallelism (ILP) via loop unrolling, process multiple elements per thread.

#### Memory Dependency Stall
Waiting for previous memory access to complete.
**Fix:** Improve memory coalescing, increase memory-level parallelism (MLP), use shared memory, reduce register spilling.

#### Memory Throttle Stall
Too many outstanding memory requests.
**Fix:** Combine memory transactions, use 64-bit requests, minimize uncoalesced accesses, use LDG for read-only data.

#### Texture Stall
Texture sub-system overloaded.
**Fix:** Combine texture fetches, use shared memory, re-compute instead of fetch, reduce LDG usage on CC 3.x.

#### Sync Stall
Waiting at `__syncthreads()`.
**Fix:** Improve load balance between sync points, reduce thread block size, minimize `__threadfence()` calls, consider warp shuffle operations.

#### Constant Memory Stall
Constant cache miss.
**Fix:** Reduce `__constant__` usage, increase kernel runtime, process more items per thread, merge kernels using same constant data.

#### Pipe Busy Stall
Required functional unit busy.
**Fix:** Use float instead of double where precision allows, look for arithmetic order-of-operation improvements.

#### Not Selected Stall
Warp ready but another warp was chosen.
**Fix:** May indicate good optimization; could decrease occupancy to improve cache hit rates.

---

## Migrating to Nsight Tools

### Tool Mapping

| Feature | Nsight Systems | Nsight Compute |
|---------|---------------|----------------|
| Timeline/Activity/API Tracing | Yes | |
| CPU Sampling | Yes | |
| OpenACC / OpenMP / MPI | Yes | |
| MPS Tracing | Yes | |
| Dependency Analysis | Yes | |
| Unified Memory Transfers | Yes | |
| Events & Metrics (per kernel) | | Yes |
| Guided Kernel Analysis | | Yes |
| Source-Disassembly View | | Yes |
| PC Sampling | | Yes |
| NVTX | Yes | Yes |
| Remote Profiling | Yes | Yes |

### Architecture Support

| Architecture | nvprof/nvvp | Nsight Systems | Nsight Compute |
|-------------|-------------|----------------|----------------|
| Maxwell (5.x) | Yes | No | No |
| Pascal (6.x) | Yes | Yes | No |
| Volta (7.0) | Yes | Yes | Yes |
| Turing (7.5) | Tracing only | Yes | Yes |
| Ampere+ (8.0+) | No | Yes | Yes |

### Quick Migration

```bash
# nvprof timeline → Nsight Systems
nsys profile -o report ./app
nsys-ui report.qdrep

# nvprof kernel profiling → Nsight Compute
ncu --set full -o report ./app
ncu-ui report.ncu-rep
```

---

## Known Issues

### Critical Limitations

- **CC 8.0+ not supported** by Visual Profiler and nvprof
- **macOS not supported** as target platform (CUDA 11.0+)
- **CC 7.5+ events/metrics** only in Nsight Compute (not nvprof)
- **CUDA Graph kernel nodes** cannot be profiled
- **Multi-device cooperative kernels** not supported
- **OptiX applications** cannot be profiled (CUDA 12.4+)

### Profiling Accuracy

- Events/metrics collection serializes all kernel executions
- Kernel replay may produce incorrect results for IPC/peer communication
- Multi-context applications may only collect single-context metrics
- Compute preemption (CC 6.0+) can affect kernel timing and event counts
- Auto-boost can cause inconsistent results; nvprof tries to disable it

### Platform Issues

- ARM64 SBSA, vGPU, WSL, CMP not supported
- Windows timer resolution may cause "invalid records" warning
- LD_PRELOAD with MPI may crash; use SUID nvprof as workaround
- Non-root/non-admin: GPU performance counters restricted (driver 418.43+/419.17+)

### Data Collection

- Always call `cudaDeviceSynchronize()` then `cudaProfilerStop()` before exit
- High kernel launch rate may overflow device profiling buffers
- Concurrent kernel mode adds overhead for many short kernels
- Large profiles may exceed JVM heap; increase `-Xmx` in `nvvp.ini`
