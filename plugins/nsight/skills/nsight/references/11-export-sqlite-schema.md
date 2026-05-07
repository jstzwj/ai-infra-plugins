# Nsight Systems Export Formats and SQLite Schema Reference

## Table of Contents

- [Available Export Formats](#available-export-formats)
- [SQLite Export](#sqlite-export)
- [Schema Versioning](#schema-versioning)
- [Complete SQLite Schema Reference](#complete-sqlite-schema-reference)
- [Expert Systems Analysis](#expert-systems-analysis)
- [Multi-Report Analysis](#multi-report-analysis)

---

## Available Export Formats

Nsight Systems supports exporting profiling data to several formats for programmatic analysis and integration with other tools.

### Format Overview

| Format | Extension | CLI Option | Description |
|---|---|---|---|
| **SQLite** | `.sqlite` | `-t sqlite` | Relational database with full schema |
| **JSON** | `.json` | `-t json` | JSON format for web and script integration |
| **HDF5** | `.h5` | `-t hdf5` | Hierarchical Data Format for scientific computing |
| **Text** | `.txt` | `-t text` | Plain text human-readable output |
| **CSV** | `.csv` | `-t csv` | Comma-separated values for spreadsheet import |
| **ParaVer** | `.prv` | `-t paraver` | For use with ParaVer trace analyzer |

### Export Command Syntax

```bash
# Basic export
nsys export -t sqlite -o output.sqlite report.nsys-rep

# Export with filtering
nsys export -t sqlite -o output.sqlite \
    --start=1.0 --duration=5.0 \
    --gpu-id=0 \
    report.nsys-rep

# Export specific tables
nsys export -t sqlite -o output.sqlite \
    --tables=CUPTI_ACTIVITY_KIND_KERNEL,CUPTI_ACTIVITY_KIND_MEMCPY \
    report.nsys-rep

# Export to JSON
nsys export -t json -o output.json report.nsys-rep

# Export to CSV
nsys export -t csv -o output.csv report.nsys-rep
```

### Export Options

| Option | Description | Default |
|---|---|---|
| `-t, --type` | Output format type | Required |
| `-o, --output` | Output file path | Required |
| `--start` | Start time offset (seconds) | 0 (beginning) |
| `--duration` | Duration to export (seconds) | Full trace |
| `--gpu-id` | Filter by GPU ID | All GPUs |
| `--tables` | Comma-separated list of tables to export | All tables |
| `--force` | Overwrite existing output file | false |
| `--separate` | Create separate files per table | false |
| `--lz4` | Compress output with LZ4 | false |

---

## SQLite Export

SQLite export provides a relational database containing all profiling data, suitable for custom analysis using SQL queries.

### Exporting to SQLite

```bash
# Basic SQLite export
nsys export -t sqlite -o report.sqlite report.nsys-rep

# Export with time range
nsys export -t sqlite -o report.sqlite --start=2.0 --duration=10.0 report.nsys-rep

# Export with specific tables only
nsys export -t sqlite -o report.sqlite \
    --tables=CUPTI_ACTIVITY_KIND_KERNEL,CUPTI_ACTIVITY_KIND_MEMCPY,TARGET_INFO_SESSION_START_TIME \
    report.nsys-rep
```

### Querying the SQLite Database

```bash
# Open the database
sqlite3 report.sqlite

# List all tables
.tables

# Get schema for a specific table
.schema CUPTI_ACTIVITY_KIND_KERNEL

# Query top 10 longest GPU kernels
SELECT name, sum(end - start) as total_time_ns, count(*) as count
FROM CUPTI_ACTIVITY_KIND_KERNEL
GROUP BY name
ORDER BY total_time_ns DESC
LIMIT 10;
```

### Common SQL Queries

#### Top GPU Kernels by Total Time

```sql
SELECT
    k.name,
    COUNT(*) AS launch_count,
    SUM(k.end - k.start) AS total_ns,
    AVG(k.end - k.start) AS avg_ns,
    MIN(k.end - k.start) AS min_ns,
    MAX(k.end - k.start) AS max_ns
FROM CUPTI_ACTIVITY_KIND_KERNEL k
GROUP BY k.name
ORDER BY total_ns DESC
LIMIT 20;
```

#### Memory Transfer Summary

```sql
SELECT
    CASE copyKind
        WHEN 0 THEN 'HostToHost'
        WHEN 1 THEN 'HostToDevice'
        WHEN 2 THEN 'DeviceToHost'
        WHEN 3 THEN 'DeviceToDevice'
        WHEN 4 THEN 'Default'
        ELSE 'Unknown'
    END AS direction,
    COUNT(*) AS count,
    SUM(end - start) AS total_ns,
    SUM(bytes) AS total_bytes,
    SUM(bytes) * 1.0 / SUM((end - start) / 1000000000.0) / 1073741824.0 AS throughput_GBps
FROM CUPTI_ACTIVITY_KIND_MEMCPY
GROUP BY copyKind;
```

#### API Overhead Analysis

```sql
SELECT
    name,
    COUNT(*) AS call_count,
    SUM(end - start) AS total_ns,
    AVG(end - start) AS avg_ns
FROM CUPTI_ACTIVITY_KIND_RUNTIME
GROUP BY name
ORDER BY total_ns DESC
LIMIT 15;
```

---

## Schema Versioning

The SQLite schema uses semantic versioning (major.minor.micro).

### Version Format

| Component | Description |
|---|---|
| **Major** | Breaking schema changes (table renames, column removals). Reports with different major versions are incompatible. |
| **Minor** | Additive changes (new tables, new columns). Older queries still work; new queries may not work on older databases. |
| **Micro** | Bug fixes and minor corrections. Fully compatible. |

### Checking Schema Version

```sql
SELECT * FROM META_DATA WHERE key = 'schema_version';
-- Returns: major.minor.micro, e.g., "5.1.0"
```

### Version History

| Version | Nsight Systems Release | Key Changes |
|---|---|---|
| 5.x | 2025.x | Latest schema with all current tables |
| 4.x | 2024.x | Added Vulkan pipeline tables |
| 3.x | 2023.x | Added GPU metrics tables |
| 2.x | 2022.x | Added WDDM tables |
| 1.x | 2021.x | Initial comprehensive schema |

---

## Complete SQLite Schema Reference

### StringIds Table

Maps integer IDs to string values for efficient storage.

```sql
CREATE TABLE StringIds (
    id INTEGER PRIMARY KEY,
    value TEXT NOT NULL
);
```

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER | Unique string identifier |
| `value` | TEXT | The actual string value |

### ANALYSIS_FILE Table

Metadata about the analysis file itself.

```sql
CREATE TABLE ANALYSIS_FILE (
    fileName TEXT NOT NULL,
    filePath TEXT NOT NULL,
    fileSize INTEGER NOT NULL,
    fileDate INTEGER NOT NULL
);
```

### ThreadNames Table

Thread names and process associations.

```sql
CREATE TABLE ThreadNames (
    pid INTEGER NOT NULL,
    tid INTEGER NOT NULL,
    name TEXT,
    startTimestamp INTEGER,
    globalTid INTEGER
);
```

### ProcessStreams Table

Process stream associations.

```sql
CREATE TABLE ProcessStreams (
    globalPid INTEGER NOT NULL,
    globalTid INTEGER,
    streamId INTEGER
);
```

---

### TARGET_INFO_* Tables

#### TARGET_INFO_SYSTEM_ENV

System environment information at the time of profiling.

```sql
CREATE TABLE TARGET_INFO_SYSTEM_ENV (
    name TEXT NOT NULL,
    value TEXT NOT NULL
);
```

Common entries:
- `OS`: Operating system name and version
- `CPU`: CPU model and frequency
- `GPU`: GPU device names
- `Kernel`: Kernel version
- `Driver`: NVIDIA driver version
- `CUDA`: CUDA toolkit version
- `Memory`: System memory size
- `Hostname`: Machine hostname

#### TARGET_INFO_NIC_INFO

Network Interface Card information.

```sql
CREATE TABLE TARGET_INFO_NIC_INFO (
    id INTEGER PRIMARY KEY,
    name TEXT,
    speed INTEGER,
    ip TEXT,
    mac TEXT
);
```

#### TARGET_INFO_SESSION_START_TIME

Session timing information.

```sql
CREATE TABLE TARGET_INFO_SESSION_START_TIME (
    globalPid INTEGER NOT NULL,
    utcEpochNs INTEGER NOT NULL,
    monotonicNs INTEGER NOT NULL,
    boottimeNs INTEGER NOT NULL
);
```

#### TARGET_INFO_GPU

GPU device information.

```sql
CREATE TABLE TARGET_INFO_GPU (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    pciBusId TEXT,
    computeCapability TEXT,
    totalMemory INTEGER,
    smCount INTEGER,
    maxClockFrequency INTEGER,
    gpuFamily TEXT,
    uuid TEXT
);
```

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER | GPU device index |
| `name` | TEXT | GPU model name (e.g., "NVIDIA A100-SXM4-80GB") |
| `pciBusId` | TEXT | PCI bus address |
| `computeCapability` | TEXT | Compute capability (e.g., "8.0") |
| `totalMemory` | INTEGER | Total GPU memory in bytes |
| `smCount` | INTEGER | Number of streaming multiprocessors |
| `maxClockFrequency` | INTEGER | Maximum clock frequency in MHz |
| `gpuFamily` | TEXT | GPU architecture family (e.g., "Ampere") |
| `uuid` | TEXT | GPU unique identifier |

#### TARGET_INFO_CUDA_DEVICE

CUDA device properties.

```sql
CREATE TABLE TARGET_INFO_CUDA_DEVICE (
    id INTEGER PRIMARY KEY,
    gpuId INTEGER,
    computeCapability TEXT,
    pciBusId TEXT,
    pciDeviceId INTEGER,
    maxThreadsPerBlock INTEGER,
    maxBlockDimX INTEGER,
    maxBlockDimY INTEGER,
    maxBlockDimZ INTEGER,
    maxGridDimX INTEGER,
    maxGridDimY INTEGER,
    maxGridDimZ INTEGER,
    maxSharedMemoryPerBlock INTEGER,
    totalConstantMemory INTEGER,
    warpSize INTEGER,
    maxRegistersPerBlock INTEGER,
    clockRate INTEGER,
    multiProcessorCount INTEGER,
    integrated INTEGER,
    canMapHostMemory INTEGER,
    concurrentKernels INTEGER,
    ECCEnabled INTEGER,
    asyncEngineCount INTEGER,
    unifiedAddressing INTEGER,
    memoryClockRate INTEGER,
    memoryBusWidth INTEGER,
    l2CacheSize INTEGER,
    maxThreadsPerMultiProcessor INTEGER
);
```

#### TARGET_INFO_PROCESS

Process information from the profiling session.

```sql
CREATE TABLE TARGET_INFO_PROCESS (
    pid INTEGER NOT NULL,
    name TEXT,
    exePath TEXT,
    startTime INTEGER,
    endTime INTEGER,
    globalPid INTEGER,
    ppId INTEGER
);
```

---

### META_DATA_* Tables

#### META_DATA

General metadata key-value store.

```sql
CREATE TABLE META_DATA (
    key TEXT NOT NULL,
    value TEXT NOT NULL
);
```

Common keys:
- `schema_version`: Schema version string
- `report_version`: Report format version
- `tool_version`: Nsight Systems version
- `start_utc`: UTC start time of profiling session
- `duration_ns`: Duration of profiling session in nanoseconds
- `platform`: Target platform name

#### META_DATA_PROPERTIES

Extended metadata properties.

```sql
CREATE TABLE META_DATA_PROPERTIES (
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    section TEXT,
    type TEXT
);
```

---

### ENUM_* Tables

Enumeration tables provide mappings for numeric codes used throughout the schema.

#### ENUM_CUDA_MEMCPY_KIND

```sql
CREATE TABLE ENUM_CUDA_MEMCPY_KIND (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);
-- Values: 0=HostToHost, 1=HostToDevice, 2=DeviceToHost,
--         3=DeviceToDevice, 4=Default
```

#### ENUM_CUDA_MEMSET_KIND

```sql
CREATE TABLE ENUM_CUDA_MEMSET_KIND (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);
```

#### ENUM_CUDA_SYNC_TYPE

```sql
CREATE TABLE ENUM_CUDA_SYNC_TYPE (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);
```

#### ENUM_CUDA_EVENT_TYPE

```sql
CREATE TABLE ENUM_CUDA_EVENT_TYPE (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);
```

#### ENUM_CUDA_GRAPH_NODE_TYPE

```sql
CREATE TABLE ENUM_CUDA_GRAPH_NODE_TYPE (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);
```

#### ENUM_CUPTI_ACTIVITY_KIND

```sql
CREATE TABLE ENUM_CUPTI_ACTIVITY_KIND (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);
```

#### ENUM_GPU_CONTEXT_SWITCH_TYPE

```sql
CREATE TABLE ENUM_GPU_CONTEXT_SWITCH_TYPE (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);
```

#### ENUM_NVTX_PAYLOAD_SCHEMA_TYPE

```sql
CREATE TABLE ENUM_NVTX_PAYLOAD_SCHEMA_TYPE (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);
```

---

### CUPTI_ACTIVITY_KIND_* Tables

These tables contain the core CUDA profiling activity data collected via the CUPTI (CUDA Profiling Tools Interface) library.

#### CUPTI_ACTIVITY_KIND_MEMCPY

Records CUDA memory copy operations.

```sql
CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (
    start INTEGER NOT NULL,
    end INTEGER NOT NULL,
    deviceId INTEGER,
    contextId INTEGER,
    streamId INTEGER,
    correlationId INTEGER,
    bytes INTEGER,
    copyKind INTEGER,
    srcMemoryKind INTEGER,
    dstMemoryKind INTEGER,
    srcId INTEGER,
    dstId INTEGER,
    async INTEGER,
    copiedBytes INTEGER,
    deviceIdSrc INTEGER,
    deviceIdDst INTEGER
);
```

| Column | Type | Description |
|---|---|---|
| `start` | INTEGER | Start timestamp (ns) |
| `end` | INTEGER | End timestamp (ns) |
| `bytes` | INTEGER | Number of bytes to copy |
| `copyKind` | INTEGER | Memcpy type (see ENUM_CUDA_MEMCPY_KIND) |
| `srcMemoryKind` | INTEGER | Source memory type (host, device, etc.) |
| `dstMemoryKind` | INTEGER | Destination memory type |
| `srcId` | INTEGER | Source device ID |
| `dstId` | INTEGER | Destination device ID |
| `correlationId` | INTEGER | Links to the corresponding API call |
| `streamId` | INTEGER | CUDA stream ID |

#### CUPTI_ACTIVITY_KIND_MEMSET

Records CUDA memory set operations.

```sql
CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET (
    start INTEGER NOT NULL,
    end INTEGER NOT NULL,
    deviceId INTEGER,
    contextId INTEGER,
    streamId INTEGER,
    correlationId INTEGER,
    bytes INTEGER,
    memoryKind INTEGER,
    elementSize INTEGER,
    dstId INTEGER,
    async INTEGER
);
```

#### CUPTI_ACTIVITY_KIND_KERNEL

Records GPU kernel executions.

```sql
CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
    start INTEGER NOT NULL,
    end INTEGER NOT NULL,
    deviceId INTEGER,
    contextId INTEGER,
    streamId INTEGER,
    correlationId INTEGER,
    nameId INTEGER,
    name TEXT,
    demangledName TEXT,
    launched INTEGER,
    launchedGridX INTEGER,
    launchedGridY INTEGER,
    launchedGridZ INTEGER,
    launchedBlockX INTEGER,
    launchedBlockY INTEGER,
    launchedBlockZ INTEGER,
    cacheConfig INTEGER,
    sharedMemoryConfig INTEGER,
    registersPerThread INTEGER,
    sharedMemory INTEGER,
    staticSharedMemory INTEGER,
    dynamicSharedMemory INTEGER,
    localMemoryPerThread INTEGER,
    localMemoryTotal INTEGER,
    correlationGlobalPid INTEGER,
    envStart INTEGER,
    envEnd INTEGER,
    parentCorrelationId INTEGER,
    hasCategory INTEGER
);
```

| Column | Type | Description |
|---|---|---|
| `name` | TEXT | Kernel function name (mangled) |
| `demangledName` | TEXT | Demangled kernel name |
| `launchedGridX/Y/Z` | INTEGER | Grid dimensions |
| `launchedBlockX/Y/Z` | INTEGER | Block dimensions |
| `registersPerThread` | INTEGER | Register usage per thread |
| `sharedMemory` | INTEGER | Total shared memory used (bytes) |
| `staticSharedMemory` | INTEGER | Statically allocated shared memory |
| `dynamicSharedMemory` | INTEGER | Dynamically allocated shared memory |
| `parentCorrelationId` | INTEGER | For graph-launched kernels |

#### CUPTI_ACTIVITY_KIND_SYNCHRONIZATION

Records CUDA synchronization events.

```sql
CREATE TABLE CUPTI_ACTIVITY_KIND_SYNCHRONIZATION (
    start INTEGER NOT NULL,
    end INTEGER NOT NULL,
    deviceId INTEGER,
    contextId INTEGER,
    streamId INTEGER,
    correlationId INTEGER,
    syncType INTEGER,
    eventId INTEGER,
    streamIdSynched INTEGER,
    status INTEGER
);
```

| syncType | Meaning |
|---|---|
| 0 | Unknown |
| 1 | Stream synchronize |
| 2 | Event synchronize |
| 3 | Event query |
| 4 | Context synchronize |

#### CUPTI_ACTIVITY_KIND_CUDA_EVENT

Records CUDA event operations.

```sql
CREATE TABLE CUPTI_ACTIVITY_KIND_CUDA_EVENT (
    start INTEGER NOT NULL,
    end INTEGER NOT NULL,
    deviceId INTEGER,
    contextId INTEGER,
    streamId INTEGER,
    correlationId INTEGER,
    eventId INTEGER,
    eventKind INTEGER
);
```

#### CUPTI_ACTIVITY_KIND_GRAPH_TRACE

Records CUDA Graph trace events.

```sql
CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE (
    start INTEGER NOT NULL,
    end INTEGER NOT NULL,
    deviceId INTEGER,
    contextId INTEGER,
    streamId INTEGER,
    correlationId INTEGER,
    graphId INTEGER,
    graphExecId INTEGER
);
```

#### CUPTI_ACTIVITY_KIND_RUNTIME

Records CUDA Runtime API calls.

```sql
CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
    start INTEGER NOT NULL,
    end INTEGER NOT NULL,
    correlationId INTEGER,
    globalPid INTEGER,
    globalTid INTEGER,
    nameId INTEGER,
    name TEXT,
    returnValue INTEGER,
    callChainId INTEGER
);
```

| Column | Type | Description |
|---|---|---|
| `name` | TEXT | API function name (e.g., `cudaLaunchKernel`) |
| `returnValue` | INTEGER | Return code (0 = cudaSuccess) |
| `callChainId` | INTEGER | Links to call chain backtrace |

#### CUPTI_ACTIVITY_KIND_BLOCK_TRACE

Records block-level trace data for detailed kernel execution analysis.

```sql
CREATE TABLE CUPTI_ACTIVITY_KIND_BLOCK_TRACE (
    start INTEGER NOT NULL,
    end INTEGER NOT NULL,
    deviceId INTEGER,
    contextId INTEGER,
    streamId INTEGER,
    correlationId INTEGER,
    blockX INTEGER,
    blockY INTEGER,
    blockZ INTEGER
);
```

#### CUPTI_ACTIVITY_KIND_WARP_TRACE

Records warp-level trace data.

```sql
CREATE TABLE CUPTI_ACTIVITY_KIND_WARP_TRACE (
    start INTEGER NOT NULL,
    end INTEGER NOT NULL,
    deviceId INTEGER,
    contextId INTEGER,
    streamId INTEGER,
    correlationId INTEGER,
    blockX INTEGER,
    blockY INTEGER,
    blockZ INTEGER,
    warpId INTEGER
);
```

---

### CUDA_* Tables

#### CUDA_UM_CPU_PAGE_FAULT_EVENTS

Unified Memory CPU page fault events.

```sql
CREATE TABLE CUDA_UM_CPU_PAGE_FAULT_EVENTS (
    start INTEGER NOT NULL,
    address INTEGER NOT NULL,
    pc INTEGER,
    globalPid INTEGER,
    globalTid INTEGER,
    accessCount INTEGER,
    accessKind INTEGER,
    flags INTEGER
);
```

#### CUDA_UM_GPU_PAGE_FAULT_EVENTS

Unified Memory GPU page fault events.

```sql
CREATE TABLE CUDA_UM_GPU_PAGE_FAULT_EVENTS (
    start INTEGER NOT NULL,
    address INTEGER NOT NULL,
    pc INTEGER,
    globalPid INTEGER,
    globalTid INTEGER,
    deviceId INTEGER,
    accessCount INTEGER,
    accessKind INTEGER,
    flags INTEGER
);
```

#### CUDA_GPU_MEMORY_USAGE_EVENTS

GPU memory usage tracking events.

```sql
CREATE TABLE CUDA_GPU_MEMORY_USAGE_EVENTS (
    start INTEGER NOT NULL,
    end INTEGER NOT NULL,
    deviceId INTEGER,
    contextId INTEGER,
    globalPid INTEGER,
    kind INTEGER,
    memoryUsed INTEGER,
    totalMemory INTEGER
);
```

#### CUDA_GPU_MEMORY_POOL_EVENTS

GPU memory pool operations.

```sql
CREATE TABLE CUDA_GPU_MEMORY_POOL_EVENTS (
    start INTEGER NOT NULL,
    end INTEGER NOT NULL,
    deviceId INTEGER,
    contextId INTEGER,
    globalPid INTEGER,
    kind INTEGER,
    bytes INTEGER,
    poolId INTEGER
);
```

#### CUDA_CALLCHAINS

Call chain backtrace data.

```sql
CREATE TABLE CUDA_CALLCHAINS (
    callChainId INTEGER NOT NULL,
    functionId INTEGER,
    offset INTEGER,
    moduleId INTEGER,
    address INTEGER
);
```

#### CUDA_GRAPH_NODE_EVENTS

CUDA Graph node execution events.

```sql
CREATE TABLE CUDA_GRAPH_NODE_EVENTS (
    start INTEGER NOT NULL,
    end INTEGER NOT NULL,
    deviceId INTEGER,
    contextId INTEGER,
    streamId INTEGER,
    correlationId INTEGER,
    graphId INTEGER,
    graphNodeId INTEGER,
    nodeType INTEGER,
    parentGraphNodeId INTEGER
);
```

#### CUDA_GRAPH_EVENTS

CUDA Graph creation and execution events.

```sql
CREATE TABLE CUDA_GRAPH_EVENTS (
    start INTEGER NOT NULL,
    end INTEGER NOT NULL,
    deviceId INTEGER,
    contextId INTEGER,
    graphId INTEGER,
    globalPid INTEGER,
    globalTid INTEGER,
    kind INTEGER,
    correlationId INTEGER
);
```

---

### NVTX_* Tables

#### NVTX_EVENTS

NVIDIA Tools Extension (NVTX) annotation events.

```sql
CREATE TABLE NVTX_EVENTS (
    start INTEGER,
    end INTEGER,
    eventType INTEGER NOT NULL,
    globalPid INTEGER,
    globalTid INTEGER,
    domainId INTEGER,
    eventId INTEGER,
    category INTEGER,
    colorType INTEGER,
    color INTEGER,
    textId INTEGER,
    text TEXT,
    payloadId INTEGER,
    payloadText TEXT,
    payloadValueType INTEGER,
    payloadValue REAL,
    scopeId INTEGER,
    scopeType INTEGER,
    nestingLevel INTEGER,
    parentEventId INTEGER
);
```

| Column | Type | Description |
|---|---|---|
| `eventType` | INTEGER | 0=RangeStart, 1=RangeEnd, 2=Mark, 3=RangePush, 4=RangePop |
| `domainId` | INTEGER | NVTX domain ID (0 = default domain) |
| `category` | INTEGER | User-defined category ID |
| `color` | INTEGER | ARGB color value |
| `text` | TEXT | Event annotation text |
| `payloadValue` | REAL | Numeric payload value |
| `payloadText` | TEXT | Text payload value |
| `scopeType` | INTEGER | Scope: process, thread, or custom |

#### NVTX_PAYLOAD_SCHEMAS

Schema definitions for structured NVTX payloads.

```sql
CREATE TABLE NVTX_PAYLOAD_SCHEMAS (
    schemaId INTEGER PRIMARY KEY,
    domainId INTEGER,
    name TEXT,
    numEntries INTEGER
);
```

#### NVTX_PAYLOAD_SCHEMA_ENTRIES

Individual entries within a payload schema.

```sql
CREATE TABLE NVTX_PAYLOAD_SCHEMA_ENTRIES (
    schemaId INTEGER NOT NULL,
    entryIndex INTEGER NOT NULL,
    key TEXT,
    type INTEGER,
    unit TEXT,
    description TEXT
);
```

#### NVTX_PAYLOAD_ENUMS

Enumeration values for payload schema entries.

```sql
CREATE TABLE NVTX_PAYLOAD_ENUMS (
    schemaId INTEGER NOT NULL,
    entryIndex INTEGER NOT NULL,
    enumValue INTEGER NOT NULL,
    enumName TEXT
);
```

#### NVTX_SCOPES

NVTX scope definitions.

```sql
CREATE TABLE NVTX_SCOPES (
    scopeId INTEGER NOT NULL,
    scopeType INTEGER NOT NULL,
    domainId INTEGER,
    globalPid INTEGER,
    globalTid INTEGER,
    name TEXT
);
```

---

### OSRT_API and OSRT_CALLCHAINS

#### OSRT_API

Operating System Runtime API call trace.

```sql
CREATE TABLE OSRT_API (
    start INTEGER NOT NULL,
    end INTEGER NOT NULL,
    globalPid INTEGER,
    globalTid INTEGER,
    nameId INTEGER,
    name TEXT,
    returnValue INTEGER,
    callChainId INTEGER,
    argSetId INTEGER
);
```

Common OS runtime functions traced:
- `pthread_create`, `pthread_join`, `pthread_mutex_lock`
- `open`, `read`, `write`, `close`
- `mmap`, `munmap`
- `poll`, `select`, `epoll_wait`
- `fork`, `exec`, `waitpid`

#### OSRT_CALLCHAINS

Backtrace call chains for OS runtime events.

```sql
CREATE TABLE OSRT_CALLCHAINS (
    callChainId INTEGER NOT NULL,
    functionId INTEGER,
    offset INTEGER,
    moduleId INTEGER,
    address INTEGER
);
```

---

### SCHED_EVENTS

Thread scheduling events from the Linux kernel.

```sql
CREATE TABLE SCHED_EVENTS (
    timestamp INTEGER NOT NULL,
    globalPid INTEGER,
    globalTid INTEGER,
    cpu INTEGER,
    targetTid INTEGER,
    targetGlobalTid INTEGER,
    eventType INTEGER,
    priority INTEGER,
    policy INTEGER
);
```

| eventType | Meaning |
|---|---|
| 0 | Switch in (thread starts running on CPU) |
| 1 | Switch out (thread leaves CPU) |
| 2 | Wake up (thread becomes runnable) |

---

### COMPOSITE_EVENTS

Composite events combining multiple event sources.

```sql
CREATE TABLE COMPOSITE_EVENTS (
    start INTEGER NOT NULL,
    end INTEGER NOT NULL,
    globalPid INTEGER,
    globalTid INTEGER,
    eventType INTEGER,
    name TEXT,
    deviceId INTEGER,
    streamId INTEGER
);
```

---

### SAMPLING_CALLCHAINS

CPU sampling backtrace data.

```sql
CREATE TABLE SAMPLING_CALLCHAINS (
    sampleId INTEGER NOT NULL,
    globalPid INTEGER,
    globalTid INTEGER,
    cpu INTEGER,
    timestamp INTEGER NOT NULL,
    functionId INTEGER,
    offset INTEGER,
    moduleId INTEGER,
    address INTEGER,
    depth INTEGER
);
```

| Column | Type | Description |
|---|---|---|
| `sampleId` | INTEGER | Groups frames from the same sample |
| `depth` | INTEGER | Stack depth (0 = leaf, increasing = callers) |
| `functionId` | INTEGER | References StringIds for function name |
| `moduleId` | INTEGER | References module information |
| `address` | INTEGER | Instruction pointer value |

---

### PROFILER_OVERHEAD

Profiling overhead measurements.

```sql
CREATE TABLE PROFILER_OVERHEAD (
    start INTEGER NOT NULL,
    end INTEGER NOT NULL,
    globalPid INTEGER,
    globalTid INTEGER,
    overheadKind INTEGER,
    overheadValue INTEGER
);
```

---

### MPI_* Tables

#### MPI_EVENTS

MPI (Message Passing Interface) communication events.

```sql
CREATE TABLE MPI_EVENTS (
    start INTEGER NOT NULL,
    end INTEGER NOT NULL,
    globalPid INTEGER,
    globalTid INTEGER,
    nameId INTEGER,
    name TEXT,
    returnValue INTEGER,
    rank INTEGER,
    size INTEGER,
    tag INTEGER,
    communicator INTEGER,
    dataType INTEGER,
    count INTEGER
);
```

#### MPI_CALLCHAINS

Call chains for MPI events.

```sql
CREATE TABLE MPI_CALLCHAINS (
    callChainId INTEGER NOT NULL,
    functionId INTEGER,
    offset INTEGER,
    moduleId INTEGER,
    address INTEGER
);
```

---

### UCP_* Tables

UCP (Unified Communication Protocol) events for UCX-based communication.

```sql
CREATE TABLE UCP_EVENTS (
    start INTEGER NOT NULL,
    end INTEGER NOT NULL,
    globalPid INTEGER,
    globalTid INTEGER,
    name TEXT,
    operation INTEGER,
    bytes INTEGER,
    remoteProc INTEGER,
    tag INTEGER
);
```

---

### OPENGL_API and OPENGL_WORKLOAD

#### OPENGL_API

OpenGL API call trace.

```sql
CREATE TABLE OPENGL_API (
    start INTEGER NOT NULL,
    end INTEGER NOT NULL,
    globalPid INTEGER,
    globalTid INTEGER,
    nameId INTEGER,
    name TEXT,
    returnValue INTEGER,
    contextId INTEGER,
    callChainId INTEGER
);
```

#### OPENGL_WORKLOAD

OpenGL GPU workload events.

```sql
CREATE TABLE OPENGL_WORKLOAD (
    start INTEGER NOT NULL,
    end INTEGER NOT NULL,
    deviceId INTEGER,
    contextId INTEGER,
    name TEXT,
    drawType INTEGER,
    vertexCount INTEGER,
    instanceCount INTEGER
);
```

#### KHR_DEBUG_EVENTS

OpenGL KHR_debug annotation events.

```sql
CREATE TABLE KHR_DEBUG_EVENTS (
    start INTEGER,
    end INTEGER,
    globalPid INTEGER,
    globalTid INTEGER,
    source INTEGER,
    type INTEGER,
    id INTEGER,
    severity INTEGER,
    message TEXT
);
```

---

### DX12_* Tables

#### DX12_API

Direct3D 12 API call trace.

```sql
CREATE TABLE DX12_API (
    start INTEGER NOT NULL,
    end INTEGER NOT NULL,
    globalPid INTEGER,
    globalTid INTEGER,
    nameId INTEGER,
    name TEXT,
    returnValue INTEGER,
    queueId INTEGER,
    commandListId INTEGER,
    callChainId INTEGER
);
```

#### DX12_WORKLOAD

Direct3D 12 GPU workload events.

```sql
CREATE TABLE DX12_WORKLOAD (
    start INTEGER NOT NULL,
    end INTEGER NOT NULL,
    deviceId INTEGER,
    contextId INTEGER,
    queueId INTEGER,
    commandListId INTEGER,
    name TEXT,
    drawType INTEGER,
    vertexCount INTEGER,
    instanceCount INTEGER
);
```

#### DX12_MEMORY_OPERATION

Direct3D 12 memory operation events.

```sql
CREATE TABLE DX12_MEMORY_OPERATION (
    start INTEGER NOT NULL,
    end INTEGER NOT NULL,
    deviceId INTEGER,
    operationType INTEGER,
    size INTEGER,
    alignment INTEGER,
    heapType INTEGER,
    resourceFormat INTEGER
);
```

---

### VULKAN_* Tables

#### VULKAN_API

Vulkan API call trace.

```sql
CREATE TABLE VULKAN_API (
    start INTEGER NOT NULL,
    end INTEGER NOT NULL,
    globalPid INTEGER,
    globalTid INTEGER,
    nameId INTEGER,
    name TEXT,
    returnValue INTEGER,
    queueId INTEGER,
    commandBufferId INTEGER,
    callChainId INTEGER
);
```

#### VULKAN_WORKLOAD

Vulkan GPU workload events.

```sql
CREATE TABLE VULKAN_WORKLOAD (
    start INTEGER NOT NULL,
    end INTEGER NOT NULL,
    deviceId INTEGER,
    queueId INTEGER,
    commandBufferId INTEGER,
    name TEXT,
    drawType INTEGER,
    vertexCount INTEGER,
    instanceCount INTEGER,
    shaderCount INTEGER
);
```

#### VULKAN_DEBUG_API

Vulkan debug marker API events.

```sql
CREATE TABLE VULKAN_DEBUG_API (
    start INTEGER,
    end INTEGER,
    globalPid INTEGER,
    globalTid INTEGER,
    name TEXT,
    objectType INTEGER,
    objectHandle INTEGER,
    message TEXT,
    colorR REAL,
    colorG REAL,
    colorB REAL,
    colorA REAL
);
```

#### VULKAN_PIPELINE_CREATION

Vulkan pipeline creation events.

```sql
CREATE TABLE VULKAN_PIPELINE_CREATION (
    start INTEGER NOT NULL,
    end INTEGER NOT NULL,
    deviceId INTEGER,
    pipelineType INTEGER,
    pipelineId INTEGER,
    shaderCount INTEGER,
    cacheHit INTEGER,
    feedbackStatus INTEGER
);
```

---

### GPU_CONTEXT_SWITCH_EVENTS

GPU context switch events showing when GPU work is scheduled.

```sql
CREATE TABLE GPU_CONTEXT_SWITCH_EVENTS (
    timestamp INTEGER NOT NULL,
    deviceId INTEGER NOT NULL,
    contextId INTEGER NOT NULL,
    streamId INTEGER,
    switchType INTEGER NOT NULL,
    globalPid INTEGER,
    globalTid INTEGER
);
```

---

### OPENMP_EVENT_KIND_* Tables

#### OPENMP_EVENT_KIND_PARALLEL

OpenMP parallel region events.

```sql
CREATE TABLE OPENMP_EVENT_KIND_PARALLEL (
    start INTEGER NOT NULL,
    end INTEGER NOT NULL,
    globalPid INTEGER,
    globalTid INTEGER,
    parallelId INTEGER,
    teamSize INTEGER,
    requestedParallelism INTEGER
);
```

#### OPENMP_EVENT_KIND_WORK

OpenMP worksharing events.

```sql
CREATE TABLE OPENMP_EVENT_KIND_WORK (
    start INTEGER NOT NULL,
    end INTEGER NOT NULL,
    globalPid INTEGER,
    globalTid INTEGER,
    parallelId INTEGER,
    workType INTEGER,
    chunkSize INTEGER,
    iterationCount INTEGER
);
```

#### OPENMP_EVENT_KIND_SYNC

OpenMP synchronization events.

```sql
CREATE TABLE OPENMP_EVENT_KIND_SYNC (
    start INTEGER NOT NULL,
    end INTEGER NOT NULL,
    globalPid INTEGER,
    globalTid INTEGER,
    parallelId INTEGER,
    syncType INTEGER,
    barrierId INTEGER
);
```

#### OPENMP_EVENT_KIND_TASK

OpenMP task events.

```sql
CREATE TABLE OPENMP_EVENT_KIND_TASK (
    start INTEGER NOT NULL,
    end INTEGER NOT NULL,
    globalPid INTEGER,
    globalTid INTEGER,
    taskId INTEGER,
    parentTaskId INTEGER,
    taskType INTEGER,
    threadId INTEGER
);
```

#### OPENMP_EVENT_KIND_FLUSH

OpenMP flush events.

```sql
CREATE TABLE OPENMP_EVENT_KIND_FLUSH (
    timestamp INTEGER NOT NULL,
    globalPid INTEGER,
    globalTid INTEGER,
    flushId INTEGER
);
```

---

### WDDM_* Tables

#### WDDM_SUBMISSION

WDDM command buffer submission events.

```sql
CREATE TABLE WDDM_SUBMISSION (
    start INTEGER NOT NULL,
    end INTEGER NOT NULL,
    deviceId INTEGER,
    contextId INTEGER,
    queueType INTEGER,
    submitSequence INTEGER,
    processId INTEGER,
    globalPid INTEGER
);
```

#### WDDM_CONTEXT_CREATE

WDDM context creation events.

```sql
CREATE TABLE WDDM_CONTEXT_CREATE (
    timestamp INTEGER NOT NULL,
    deviceId INTEGER,
    contextId INTEGER,
    contextType INTEGER,
    processId INTEGER
);
```

#### WDDM_QUEUE_PACKET

WDDM queue packet events.

```sql
CREATE TABLE WDDM_QUEUE_PACKET (
    start INTEGER NOT NULL,
    end INTEGER NOT NULL,
    deviceId INTEGER,
    contextId INTEGER,
    packetType INTEGER,
    submitSequence INTEGER,
    dmaBufferAddress INTEGER,
    dmaBufferSize INTEGER
);
```

#### WDDM_HW_QUEUE

WDDM hardware queue events.

```sql
CREATE TABLE WDDM_HW_QUEUE (
    timestamp INTEGER NOT NULL,
    deviceId INTEGER,
    queueId INTEGER,
    queueType INTEGER,
    contextId INTEGER
);
```

---

### NVVIDEO_* Tables

#### NVVIDEO_ENCODER

NVIDIA video encoder events.

```sql
CREATE TABLE NVVIDEO_ENCODER (
    start INTEGER NOT NULL,
    end INTEGER NOT NULL,
    deviceId INTEGER,
    globalPid INTEGER,
    globalTid INTEGER,
    codecType INTEGER,
    resolutionX INTEGER,
    resolutionY INTEGER,
    frameType INTEGER,
    frameSize INTEGER
);
```

#### NVVIDEO_DECODER

NVIDIA video decoder events.

```sql
CREATE TABLE NVVIDEO_DECODER (
    start INTEGER NOT NULL,
    end INTEGER NOT NULL,
    deviceId INTEGER,
    globalPid INTEGER,
    globalTid INTEGER,
    codecType INTEGER,
    resolutionX INTEGER,
    resolutionY INTEGER,
    frameType INTEGER,
    frameSize INTEGER
);
```

---

### GPU_METRICS and SOC_METRICS

#### GPU_METRICS

GPU metric samples over time.

```sql
CREATE TABLE GPU_METRICS (
    timestamp INTEGER NOT NULL,
    deviceId INTEGER NOT NULL,
    metricId INTEGER NOT NULL,
    value REAL NOT NULL
);
```

Common metrics:
- GPU utilization (%)
- SM occupancy (%)
- Memory utilization (%)
- Power draw (W)
- SM clock frequency (MHz)
- Memory clock frequency (MHz)
- PCIe RX/TX bandwidth (MB/s)

#### SOC_METRICS

System-on-Chip metric samples.

```sql
CREATE TABLE SOC_METRICS (
    timestamp INTEGER NOT NULL,
    metricId INTEGER NOT NULL,
    value REAL NOT NULL
);
```

#### TARGET_INFO_GPU_METRICS

Metadata about available GPU metrics.

```sql
CREATE TABLE TARGET_INFO_GPU_METRICS (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    unit TEXT,
    deviceId INTEGER,
    domainId INTEGER
);
```

#### TARGET_INFO_SOC_METRICS

Metadata about available SoC metrics.

```sql
CREATE TABLE TARGET_INFO_SOC_METRICS (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    unit TEXT,
    domainId INTEGER
);
```

---

### NET_IB_* and NIC_ID_MAP

#### NET_IB_EVENTS

InfiniBand network events.

```sql
CREATE TABLE NET_IB_EVENTS (
    timestamp INTEGER NOT NULL,
    deviceId INTEGER NOT NULL,
    portId INTEGER,
    metricType INTEGER,
    value REAL NOT NULL
);
```

#### NIC_ID_MAP

Network Interface Card ID mapping.

```sql
CREATE TABLE NIC_ID_MAP (
    nicId INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    speed INTEGER,
    busId TEXT
);
```

---

## Expert Systems Analysis

Nsight Systems includes expert system rules that automatically analyze profiling data for common performance issues.

### Rules

#### cuda_memcpy_async

**Rule ID**: `cuda_memcpy_async`

**Description**: Detects synchronous memory copies that could be made asynchronous.

**Detection Logic**: Identifies `cudaMemcpy` calls that are not using the `_Async` variant and where the source or destination is on the host.

**Impact**: Synchronous copies block the CPU, preventing overlap of data transfer with computation.

**Recommendation**: Replace `cudaMemcpy` with `cudaMemcpyAsync` and use CUDA streams to enable copy-compute overlap.

```sql
-- Manual detection query
SELECT name, COUNT(*), SUM(end - start)
FROM CUPTI_ACTIVITY_KIND_RUNTIME
WHERE name = 'cudaMemcpy'
GROUP BY name;
```

#### cuda_memcpy_sync

**Rule ID**: `cuda_memcpy_sync`

**Description**: Detects implicit synchronization caused by memory operations.

**Detection Logic**: Identifies pinned memory copies or copies that force GPU synchronization.

**Impact**: Can cause significant stalls in the CUDA pipeline.

**Recommendation**: Use `cudaMemcpyAsync` with proper stream ordering.

#### cuda_memset_sync

**Rule ID**: `cuda_memset_sync`

**Description**: Detects synchronous memory set operations.

**Detection Logic**: Identifies `cudaMemset` calls that block the CPU.

**Impact**: Prevents overlap of memory initialization with computation.

**Recommendation**: Use `cudaMemsetAsync` with CUDA streams.

#### cuda_api_sync

**Rule ID**: `cuda_api_sync`

**Description**: Detects CUDA API calls that cause implicit synchronization.

**Detection Logic**: Identifies calls like `cudaDeviceSynchronize`, `cudaStreamSynchronize`, `cudaMemcpy` (synchronous), etc.

**Impact**: Forces the CPU to wait for GPU operations to complete, breaking asynchronous execution.

**Recommendation**: Minimize explicit synchronization; use event-based synchronization instead.

#### gpu_gaps

**Rule ID**: `gpu_gaps`

**Description**: Detects significant gaps in GPU utilization.

**Detection Logic**: Finds time ranges where the GPU has no kernels or memory operations queued.

**Impact**: GPU is idle while there may be work to do, reducing overall throughput.

**Recommendation**: Ensure the CPU is submitting work fast enough; use CUDA streams for concurrent execution.

#### gpu_time_util

**Rule ID**: `gpu_time_util`

**Description**: Reports overall GPU time utilization.

**Detection Logic**: Computes the ratio of active GPU time to total profiling time.

**Impact**: Low utilization indicates the GPU is underused.

**Recommendation**: Increase workload parallelism, optimize data pipeline, or batch more work.

**Thresholds**:

| Utilization | Assessment |
|---|---|
| > 90% | Excellent |
| 70-90% | Good |
| 50-70% | Fair, may have room for improvement |
| 30-50% | Poor, investigate GPU starvation |
| < 30% | Critical, significant optimization needed |

#### dx12_mem_ops

**Rule ID**: `dx12_mem_ops`

**Description**: Detects suboptimal memory operations in Direct3D 12 applications.

**Detection Logic**: Identifies frequent resource barriers, unnecessary heap transitions, and large copies.

**Impact**: Excessive memory operations reduce rendering performance.

**Recommendation**: Batch resource barriers, use explicit barriers only when needed, minimize resource state transitions.

### Running Expert Analysis

```bash
# Run all expert rules
nsys analyze report.nsys-rep

# Run specific rules
nsys analyze --rules=cuda_memcpy_async,gpu_gaps report.nsys-rep

# Export analysis results
nsys analyze --output=analysis.json report.nsys-rep
```

---

## Multi-Report Analysis

Nsight Systems can analyze multiple reports together for comparative and aggregated analysis.

### Running Multi-Report Analysis

```bash
# Analyze multiple reports
nsys analyze report1.nsys-rep report2.nsys-rep report3.nsys-rep

# Use recipes for specific analyses
nsys analyze --recipe=kernel-comparison report1.nsys-rep report2.nsys-rep
```

### Recipes

Recipes are pre-built analysis workflows for common comparison scenarios:

| Recipe | Description |
|---|---|
| `kernel-comparison` | Compare GPU kernel execution times across reports |
| `api-overhead-comparison` | Compare CUDA API overhead between runs |
| `memory-transfer-comparison` | Compare memory transfer patterns |
| `gpu-utilization-comparison` | Compare GPU utilization over time |
| `scaling-analysis` | Analyze performance scaling across different configurations |
| `regression-detection` | Detect performance regressions against a baseline |
| `kernel-timeline-diff` | Show differences in kernel execution timelines |
| `cpu-gpu-overlap-comparison` | Compare CPU-GPU overlap efficiency |
| `stutter-comparison` | Compare frame timing and stutter metrics |
| `power-comparison` | Compare power consumption across runs |
| `memory-bandwidth-comparison` | Compare memory bandwidth utilization |
| `occupancy-comparison` | Compare SM occupancy patterns |

### Recipe Usage

```bash
# Run kernel comparison recipe
nsys analyze --recipe=kernel-comparison \
    --baseline=baseline.nsys-rep \
    --current=current.nsys-rep

# Run scaling analysis
nsys analyze --recipe=scaling-analysis \
    1gpu.nsys-rep 2gpu.nsys-rep 4gpu.nsys-rep 8gpu.nsys-rep

# Export recipe results
nsys analyze --recipe=regression-detection \
    --output=regression_report.json \
    --threshold=0.10 \
    baseline.nsys-rep current.nsys-rep
```

### Dask Configuration

Multi-report analysis can leverage Dask for parallel processing of large report sets.

```bash
# Enable Dask for multi-report analysis
nsys analyze --dask --dask-scheduler=local \
    --recipe=kernel-comparison \
    report*.nsys-rep

# Configure Dask workers
nsys analyze --dask \
    --dask-scheduler=tcp://scheduler:8786 \
    --dask-workers=4 \
    --dask-memory=8GB \
    report*.nsys-rep
```

#### Dask Configuration Parameters

| Parameter | Description | Default |
|---|---|---|
| `--dask` | Enable Dask parallel processing | false |
| `--dask-scheduler` | Scheduler address (`local` or `tcp://host:port`) | `local` |
| `--dask-workers` | Number of Dask worker processes | Auto (CPU count) |
| `--dask-memory` | Memory limit per worker | Auto |
| `--dask-chunk-size` | Chunk size for data processing | 100 MB |

---

## See Also

- [CLI Reference](02-cli-reference.md)
- [GUI Report Analysis](07-gui-report-analysis.md)
- [Python and CPU Profiling](09-python-cpu-profiling.md)
- [Release Notes and Troubleshooting](12-release-notes-troubleshooting.md)
