# GPU Metrics Reference

## Overview

NVIDIA Nsight Systems provides GPU hardware metric sampling capabilities that enable developers to understand GPU utilization, throughput, and resource usage patterns during application execution. GPU metrics are sampled periodically and stored alongside timeline data, allowing correlation between application behavior and hardware performance.

### Purpose

GPU metric sampling answers critical performance questions such as:

- **Is the GPU fully utilized?** Check SM occupancy metrics (SMs Active, SM Issue) to determine if streaming multiprocessors are being saturated.
- **Where is the memory bottleneck?** Analyze DRAM and VRAM read/write bandwidth metrics to identify saturation points.
- **Is the kernel compute-bound or memory-bound?** Compare GR Active against DRAM bandwidth utilization.
- **Are tensor cores being used effectively?** Examine Tensor Active and Tensor Active / FP16 Active metrics.
- **Is PCIe or NVLink the data transfer bottleneck?** Review PCIe and NVLink throughput counters.
- **What is the compute vs. graphics balance?** Inspect compute warps, pixel warps, and vertex/geometry warps in flight.
- **Are encoder or OFA engines utilized?** Track NVENC and OFA active metrics for multimedia and optical flow workloads.

---

## Launching from the CLI

### Basic GPU Metrics Collection

Enable GPU metric sampling on a specific device:

```bash
nsys profile --gpu-metrics-devices=all ./my_application
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--gpu-metrics-devices` | Comma-separated list of GPU device IDs to sample, or `all` for all GPUs | Disabled (no metrics) |
| `--gpu-metrics-set` | Select a predefined metric set to collect | Default set |
| `--gpu-metrics-frequency` | Sampling frequency in Hz | 10000 (10 kHz) |

### Examples

Collect GPU metrics on devices 0 and 1 at 50 kHz:

```bash
nsys profile --gpu-metrics-devices=0,1 --gpu-metrics-frequency=50000 ./my_application
```

Collect GPU metrics on all devices using the full metric set:

```bash
nsys profile --gpu-metrics-devices=all --gpu-metrics-set=full ./my_application
```

Collect GPU metrics at a low frequency for long-running traces (minimize overhead):

```bash
nsys profile --gpu-metrics-devices=all --gpu-metrics-frequency=100 ./my_application
```

### Metric Sets

The `--gpu-metrics-set` option controls which metrics are collected:

| Metric Set | Description |
|------------|-------------|
| `default` | Collects the standard set of GPU metrics suitable for most profiling tasks |
| `full` | Collects all available metrics for the target GPU architecture |

Using `full` may increase per-sample overhead and buffer consumption. Choose `default` unless you specifically need additional metrics.

---

## Launching from the GUI

To enable GPU metrics from the Nsight Systems GUI:

1. Start Nsight Systems GUI.
2. Create a new project or open an existing one.
3. In the **Project Properties** dialog, navigate to the **GPU Metrics** section.
4. Check **Collect GPU metrics**.
5. Select the target GPU devices from the device list.
6. Choose the desired **Metric Set** (Default or Full).
7. Set the **Sampling Frequency** (in Hz).
8. Launch the profiling session.

The GUI will display sampled metrics in a dedicated row per GPU device on the timeline. You can zoom in to see individual sample points and hover over them for exact values.

---

## Sampling Frequency Details

### Supported Frequency Range

| Parameter | Value |
|-----------|-------|
| Minimum frequency | 10 Hz (1 sample every 100 ms) |
| Default frequency | 10,000 Hz (10 kHz, 1 sample every 100 us) |
| Maximum frequency | 200,000 Hz (200 kHz, 1 sample every 5 us) |

### Choosing a Sampling Frequency

- **10 Hz - 100 Hz**: Suitable for long-running applications (>10 minutes). Minimal overhead, but may miss short kernel-level dynamics.
- **100 Hz - 10 kHz**: Good balance for most profiling tasks. Captures kernel-level utilization patterns.
- **10 kHz - 100 kHz**: High-resolution analysis of individual kernel execution phases. Use for short captures.
- **100 kHz - 200 kHz**: Maximum resolution. Only useful for very short captures (<1 second). High risk of buffer overflow.

### Buffer Overflow

At high sampling frequencies or for long profiling sessions, the internal metric buffer may overflow. When this occurs:

- Older samples are discarded to make room for newer ones.
- A warning message is displayed in the profiling output.
- The resulting report file will have a gap in metric data for the overflow period.

**Mitigation strategies:**

1. Reduce sampling frequency if full-duration coverage is more important than resolution.
2. Limit the profiling duration using `--duration` to capture only the region of interest.
3. Use `--gpu-metrics-set=default` to reduce the number of metrics per sample, lowering buffer consumption.
4. Increase the report file size limit with `--report` options if applicable.

### Sampling Overhead

The overhead of metric sampling depends on:

- **Number of GPUs**: Each monitored GPU adds independent sampling overhead.
- **Sampling frequency**: Higher frequencies increase CPU overhead for reading and storing counters.
- **Metric set**: The `full` metric set requires reading more hardware counters per sample.
- **GPU architecture**: Newer architectures may expose more counters.

Typical overhead at the default 10 kHz frequency on a single GPU is less than 1% of total CPU time and negligible GPU time.

---

## Available Metrics

The following sections describe all GPU metrics available in Nsight Systems. Metric availability depends on the GPU architecture (Volta, Turing, Ampere, Ada Lovelace, Hopper, Blackwell). Metrics not supported by the target hardware are reported as zero or omitted.

### Clock Frequency Metrics

#### GPC Clock Frequency

| Attribute | Value |
|-----------|-------|
| **Name** | GPC Clock Frequency |
| **Unit** | MHz |
| **Description** | Average GPC (Graphics Processing Cluster) clock frequency during the sample interval. |

This metric reports the actual clock speed at which the GPCs are running. It reflects the effect of GPU Boost (automatic overclocking) and power/thermal throttling. Compare against the base and boost clock specifications to determine if the GPU is running at expected frequencies.

**Interpretation:**

- If the GPC clock frequency is significantly below the boost clock, the GPU may be thermally or power throttled.
- Large variations in clock frequency across samples indicate unstable operating conditions.
- Correlate with NVML power metrics to determine if throttling is power-limited.

#### SYS Clock Frequency

| Attribute | Value |
|-----------|-------|
| **Name** | SYS Clock Frequency |
| **Unit** | MHz |
| **Description** | Average system clock frequency during the sample interval. |

The SYS clock domain governs the GPU's system-level interconnect and memory controller operations. Changes in this clock can affect NVLink and PCIe throughput independently of the GPC clock.

---

### Compute and Graphics Activity Metrics

#### GR Active

| Attribute | Value |
|-----------|-------|
| **Name** | GR Active |
| **Unit** | Percentage (0-100%) |
| **Description** | Percentage of time the graphics/compute pipeline was active during the sample interval. |

GR Active indicates whether any graphics or compute work was being processed on the GPU. This is a high-level utilization metric.

**Interpretation:**

- **100%**: The GPU was fully occupied with graphics or compute work for the entire sample.
- **0%**: The GPU was completely idle; no kernels were running.
- **Intermediate values**: The GPU had periods of activity and idle time within the sample window. This can indicate insufficient work submission or gaps between kernel launches.

#### Sync Compute In Flight

| Attribute | Value |
|-----------|-------|
| **Name** | Sync Compute In Flight |
| **Unit** | Percentage (0-100%) |
| **Description** | Percentage of time at least one synchronous compute task was in flight on the GPU. |

Synchronous compute tasks are those submitted to a CUDA stream without explicit asynchronous flags that the GPU hardware tracks for synchronization purposes.

#### Async Compute In Flight

| Attribute | Value |
|-----------|-------|
| **Name** | Async Compute In Flight |
| **Unit** | Percentage (0-100%) |
| **Description** | Percentage of time at least one asynchronous compute task was in flight on the GPU. |

Asynchronous compute tasks run independently of the main graphics pipeline, allowing overlap between compute and graphics work. High async compute in flight alongside high GR active indicates effective concurrent execution.

#### Draw Started

| Attribute | Value |
|-----------|-------|
| **Name** | Draw Started |
| **Unit** | Count per sample |
| **Description** | Number of draw calls that started during the sample interval. |

This metric is relevant for graphics workloads (Vulkan, OpenGL, DirectX). It counts draw call submissions, not completions.

#### Dispatch Started

| Attribute | Value |
|-----------|-------|
| **Name** | Dispatch Started |
| **Unit** | Count per sample |
| **Description** | Number of compute dispatches (kernel launches) that started during the sample interval. |

For CUDA and compute API workloads, this tracks how many kernels began execution within each sample window.

---

### Warp-Level Metrics

#### Vertex/Tess/Geometry Warps in Flight

| Attribute | Value |
|-----------|-------|
| **Name** | Vertex/Tess/Geometry Warps in Flight |
| **Unit** | Percentage (0-100%) |
| **Description** | Percentage of time at least one warp from the vertex, tessellation, or geometry shader stages was in flight. |

This metric covers the geometry processing pipeline stages. High values indicate heavy vertex processing workloads.

#### Pixel Warps in Flight

| Attribute | Value |
|-----------|-------|
| **Name** | Pixel Warps in Flight |
| **Unit** | Percentage (0-100%) |
| **Description** | Percentage of time at least one pixel shader warp was in flight. |

High pixel warp activity indicates heavy fragment shading workloads. This is common in fill-rate-bound rendering scenarios.

#### Compute Warps in Flight

| Attribute | Value |
|-----------|-------|
| **Name** | Compute Warps in Flight |
| **Unit** | Percentage (0-100%) |
| **Description** | Percentage of time at least one compute warp was in flight. |

This is the primary compute utilization indicator for CUDA and OpenCL kernels. Compare with SMs Active and SM Issue to understand how effectively compute warps fill the GPU.

---

### SM Utilization Metrics

#### Active SM Unused Warp Slots

| Attribute | Value |
|-----------|-------|
| **Name** | Active SM Unused Warp Slots |
| **Unit** | Percentage (0-100%) |
| **Description** | Percentage of warp slots on active SMs that were not occupied. |

An SM is "active" if it has at least one warp assigned. This metric measures the degree to which active SMs could have hosted additional warps. High values indicate underutilization due to low occupancy.

**Interpretation:**

- **High value**: SMs are active but not fully occupied. Consider increasing block/grid dimensions or reducing register/shared memory usage to improve occupancy.
- **Low value**: Active SMs are well-utilized with warps. The GPU is effectively using its compute resources.
- Combine with SMs Active to determine if the issue is too few active SMs or underutilized active SMs.

#### Idle SM Unused Warp Slots

| Attribute | Value |
|-----------|-------|
| **Name** | Idle SM Unused Warp Slots |
| **Unit** | Percentage (0-100%) |
| **Description** | Percentage of warp slots on idle SMs that were not occupied. |

Idle SMs have no warps assigned at all. This metric indicates the potential for more parallelism if the workload could distribute across more SMs.

**Interpretation:**

- **Non-zero value**: Some SMs are completely idle. The kernel launch configuration may be too small to occupy all SMs.
- **Zero**: All SMs have at least one warp assigned. The workload is large enough to fill the GPU.

#### SMs Active

| Attribute | Value |
|-----------|-------|
| **Name** | SMs Active |
| **Unit** | Percentage (0-100%) |
| **Description** | Percentage of SMs that were active (had at least one warp assigned) averaged over the sample interval. |

This is a key metric for understanding GPU-wide compute utilization.

**Interpretation:**

- **100%**: All SMs have at least one warp. The workload spans the entire GPU.
- **Low values**: Only a fraction of SMs are in use. Consider launching more thread blocks or using larger grid sizes.
- Compare with SM Issue to distinguish between occupancy (having warps) and compute intensity (warps actually issuing instructions).

#### SM Issue

| Attribute | Value |
|-----------|-------|
| **Name** | SM Issue |
| **Unit** | Percentage (0-100%) |
| **Description** | Percentage of time at least one warp on an SM was issuing an instruction. |

SM Issue is more granular than SMs Active. An SM can have warps assigned (counted as "active") but those warps may be stalled waiting for memory or dependencies. SM Issue reflects actual instruction throughput.

**Interpretation:**

- **High SM Issue + Low SMs Active**: A few SMs are heavily loaded but the workload does not span all SMs. Consider increasing parallelism.
- **High SMs Active + Low SM Issue**: Many SMs have warps but they are stalled. Likely memory-bound or latency-bound. Improve memory access patterns or increase occupancy to hide latency.
- **High SMs Active + High SM Issue**: Near-optimal utilization.
- **Low SMs Active + Low SM Issue**: The GPU is underutilized. More work can be submitted.

---

### Tensor Core Metrics

#### Tensor Active

| Attribute | Value |
|-----------|-------|
| **Name** | Tensor Active |
| **Unit** | Percentage (0-100%) |
| **Description** | Percentage of time the tensor core pipeline was active. |

Tensor cores are specialized hardware units for matrix multiply-accumulate operations (HMMA, IMMA, DFMA instructions). This metric indicates how heavily tensor cores are being used.

**Interpretation:**

- **High Tensor Active**: The workload is effectively using tensor cores for matrix operations (common in deep learning training and inference).
- **Zero Tensor Active**: Either the workload does not use tensor cores, or the operations are mapped to the regular FP32/FP64 datapaths instead.
- Compare with GR Active to determine the tensor core fraction of total GPU work.

#### Tensor Active / FP16 Active

| Attribute | Value |
|-----------|-------|
| **Name** | Tensor Active / FP16 Active |
| **Unit** | Ratio (0.0-1.0) |
| **Description** | Ratio of tensor core activity to FP16 activity. A value near 1.0 indicates most FP16 operations are using tensor cores. |

This metric helps determine if FP16 workloads are being accelerated by tensor cores or running on the regular ALU datapath.

**Interpretation:**

- **Close to 1.0**: FP16 workloads are efficiently dispatched to tensor cores.
- **Close to 0.0**: FP16 operations are not using tensor cores. This may indicate unsupported data types or memory layouts.
- **Intermediate values**: Some FP16 work uses tensor cores and some does not.

---

### Memory Bandwidth Metrics

#### DRAM Read Bandwidth

| Attribute | Value |
|-----------|-------|
| **Name** | DRAM Read Bandwidth |
| **Unit** | GB/s |
| **Description** | Average data read bandwidth from DRAM (device memory) during the sample interval. |

This metric measures the throughput of read transactions from the GPU's onboard HBM or GDDR memory. Compare against the theoretical peak bandwidth of the GPU to determine memory saturation.

**Peak bandwidth reference (theoretical):**

| GPU | Memory Type | Peak Bandwidth |
|-----|-------------|----------------|
| A100 80GB | HBM2e | 2,039 GB/s |
| A100 40GB | HBM2 | 1,555 GB/s |
| H100 SXM | HBM3 | 3,350 GB/s |
| H200 | HBM3e | 4,800 GB/s |
| B200 | HBM3e | 8,000 GB/s |
| RTX 4090 | GDDR6X | 1,008 GB/s |

#### DRAM Write Bandwidth

| Attribute | Value |
|-----------|-------|
| **Name** | DRAM Write Bandwidth |
| **Unit** | GB/s |
| **Description** | Average data write bandwidth to DRAM (device memory) during the sample interval. |

Write bandwidth is often lower than read bandwidth in compute workloads. Write-heavy patterns may indicate large output buffers or intermediate result storage.

#### VRAM Read Bandwidth

| Attribute | Value |
|-----------|-------|
| **Name** | VRAM Read Bandwidth |
| **Unit** | GB/s |
| **Description** | Total read bandwidth from VRAM including all traffic types (compute, display, video, etc.). |

VRAM Read Bandwidth may differ from DRAM Read Bandwidth because it includes traffic from all GPU clients, not just the compute pipeline.

#### VRAM Write Bandwidth

| Attribute | Value |
|-----------|-------|
| **Name** | VRAM Write Bandwidth |
| **Unit** | GB/s |
| **Description** | Total write bandwidth to VRAM including all traffic types. |

---

### Encoder and OFA Metrics

#### NVENC Active

| Attribute | Value |
|-----------|-------|
| **Name** | NVENC Active |
| **Unit** | Percentage (0-100%) |
| **Description** | Percentage of time the NVIDIA hardware encoder (NVENC) was active. |

Relevant for video encoding workloads. Each GPU has a fixed number of NVENC sessions. This metric shows hardware encoder utilization.

#### NVENC Read Throughput

| Attribute | Value |
|-----------|-------|
| **Name** | NVENC Read Throughput |
| **Unit** | MB/s |
| **Description** | Data read throughput of the NVENC engine during the sample interval. |

#### NVENC Write Throughput

| Attribute | Value |
|-----------|-------|
| **Name** | NVENC Write Throughput |
| **Unit** | MB/s |
| **Description** | Data write throughput of the NVENC engine during the sample interval. |

#### OFA Active

| Attribute | Value |
|-----------|-------|
| **Name** | OFA Active |
| **Unit** | Percentage (0-100%) |
| **Description** | Percentage of time the Optical Flow Accelerator (OFA) was active. |

The OFA engine accelerates optical flow and disparity estimation calculations. Relevant for computer vision and video processing workloads.

#### OFA Read Throughput

| Attribute | Value |
|-----------|-------|
| **Name** | OFA Read Throughput |
| **Unit** | MB/s |
| **Description** | Data read throughput of the OFA engine during the sample interval. |

#### OFA Write Throughput

| Attribute | Value |
|-----------|-------|
| **Name** | OFA Write Throughput |
| **Unit** | MB/s |
| **Description** | Data write throughput of the OFA engine during the sample interval. |

---

### Interconnect Metrics

#### NVLink Bytes Received

| Attribute | Value |
|-----------|-------|
| **Name** | NVLink Bytes Received |
| **Unit** | Bytes per sample |
| **Description** | Total number of bytes received across all NVLink connections during the sample interval. |

NVLink is the high-speed GPU-to-GPU and GPU-to-CPU interconnect. This metric aggregates inbound traffic across all active NVLink lanes.

**NVLink peak bandwidth per direction (per link):**

| NVLink Version | Bandwidth per Link |
|----------------|-------------------|
| NVLink 2.0 (Volta) | 300 GB/s total (bidirectional) |
| NVLink 3.0 (Ampere) | 600 GB/s total (bidirectional) |
| NVLink 4.0 (Hopper) | 900 GB/s total (bidirectional) |
| NVLink 5.0 (Blackwell) | 1,800 GB/s total (bidirectional) |

#### NVLink Bytes Transmitted

| Attribute | Value |
|-----------|-------|
| **Name** | NVLink Bytes Transmitted |
| **Unit** | Bytes per sample |
| **Description** | Total number of bytes transmitted across all NVLink connections during the sample interval. |

High NVLink traffic indicates heavy multi-GPU communication (e.g., tensor parallelism, pipeline parallelism, or collective operations).

#### PCIe Read Throughput

| Attribute | Value |
|-----------|-------|
| **Name** | PCIe Read Throughput |
| **Unit** | MB/s |
| **Description** | Average data read throughput across the PCIe bus during the sample interval. |

PCIe bandwidth measures data transfer between the host CPU and GPU. Common sources include cudaMemcpy operations, unified memory page migrations, and peer-to-peer transfers over PCIe.

**PCIe peak bandwidth (per direction):**

| PCIe Version | x16 Bandwidth |
|--------------|---------------|
| PCIe 3.0 x16 | ~16 GB/s |
| PCIe 4.0 x16 | ~32 GB/s |
| PCIe 5.0 x16 | ~64 GB/s |

#### PCIe Write Throughput

| Attribute | Value |
|-----------|-------|
| **Name** | PCIe Write Throughput |
| **Unit** | MB/s |
| **Description** | Average data write throughput across the PCIe bus during the sample interval. |

#### PCIe Read Requests to BAR1

| Attribute | Value |
|-----------|-------|
| **Name** | PCIe Read Requests to BAR1 |
| **Unit** | Count per sample |
| **Description** | Number of PCIe read requests targeting BAR1 (Base Address Register 1) during the sample interval. |

BAR1 is used for Memory-Mapped I/O (MMIO) access to GPU memory. High BAR1 traffic indicates frequent small accesses from the CPU to GPU memory, which can be inefficient. This pattern is common with Unified Memory page fault handling or direct CPU access to GPU memory.

#### PCIe Write Requests to BAR1

| Attribute | Value |
|-----------|-------|
| **Name** | PCIe Write Requests to BAR1 |
| **Unit** | Count per sample |
| **Description** | Number of PCIe write requests targeting BAR1 during the sample interval. |

---

## Exporting and Querying Data

GPU metric samples are stored in the `.nsys-rep` report file and can be exported to SQLite for programmatic analysis.

### Exporting to SQLite

```bash
nsys export -t sqlite -o report.sqlite my_profile.nsys-rep
```

### Querying GPU Metrics from SQLite

The GPU metric data is stored in the `GPU_METRICS` table. Key columns:

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Row ID |
| `gpuId` | INTEGER | GPU device index |
| `metricId` | INTEGER | Foreign key to `ENUM_GPU_METRICS` |
| `timestamp` | INTEGER | Sample timestamp in nanoseconds |
| `value` | REAL | Metric value |

#### List Available Metrics

```sql
SELECT * FROM ENUM_GPU_METRICS ORDER BY id;
```

#### Query SM Utilization Over Time

```sql
SELECT
    m.timestamp,
    m.value AS sm_active_pct,
    e.name AS metric_name
FROM GPU_METRICS m
JOIN ENUM_GPU_METRICS e ON m.metricId = e.id
WHERE e.name = 'SMs Active [%]'
  AND m.gpuId = 0
ORDER BY m.timestamp;
```

#### Calculate Average DRAM Bandwidth

```sql
SELECT
    AVG(m.value) AS avg_read_bw_gb_s
FROM GPU_METRICS m
JOIN ENUM_GPU_METRICS e ON m.metricId = e.id
WHERE e.name = 'DRAM Read Bandwidth [GB/s]'
  AND m.gpuId = 0;
```

#### Find Time Intervals with High SM Issue

```sql
SELECT
    m.timestamp,
    m.value AS sm_issue_pct
FROM GPU_METRICS m
JOIN ENUM_GPU_METRICS e ON m.metricId = e.id
WHERE e.name = 'SM Issue [%]'
  AND m.gpuId = 0
  AND m.value > 80.0
ORDER BY m.value DESC;
```

#### Correlate SM Utilization with DRAM Bandwidth

```sql
SELECT
    a.timestamp,
    a.value AS sm_active_pct,
    d.value AS dram_read_gb_s
FROM
    (SELECT timestamp, value FROM GPU_METRICS
     WHERE metricId = (SELECT id FROM ENUM_GPU_METRICS WHERE name = 'SMs Active [%]')
       AND gpuId = 0) a
JOIN
    (SELECT timestamp, value FROM GPU_METRICS
     WHERE metricId = (SELECT id FROM ENUM_GPU_METRICS WHERE name = 'DRAM Read Bandwidth [GB/s]')
       AND gpuId = 0) d
ON a.timestamp = d.timestamp
ORDER BY a.timestamp;
```

#### Time Range Selection

```sql
-- Filter metrics to a specific time window (timestamps in nanoseconds)
SELECT
    m.timestamp,
    e.name,
    m.value
FROM GPU_METRICS m
JOIN ENUM_GPU_METRICS e ON m.metricId = e.id
WHERE m.gpuId = 0
  AND m.timestamp >= 1000000000   -- Start at 1 second
  AND m.timestamp <= 5000000000   -- End at 5 seconds
ORDER BY m.timestamp, e.name;
```

---

## Limitations

### General Limitations

1. **Architecture dependency**: Not all metrics are available on all GPU architectures. Older GPUs (Pascal, Maxwell) may report zero for some counters.
2. **Sample resolution**: GPU metrics are sampled periodically, not traced per instruction. Very short kernels (< sample interval) may not appear in metric data.
3. **Buffer overflow**: High sampling frequencies combined with long trace durations can cause internal buffer overflow, resulting in data gaps.
4. **Overhead**: While typically low (<1%), overhead increases with the number of monitored GPUs and sampling frequency.
5. **Metric granularity**: Metrics are GPU-wide aggregates. Per-SM or per-warp breakdowns are not available in Nsight Systems (use Nsight Compute for per-kernel analysis).
6. **No per-kernel attribution**: GPU metrics cannot be directly attributed to individual kernels. Correlation with the kernel timeline requires manual time-range alignment.
7. **Multi-instance GPU (MIG)**: When MIG is enabled, metrics reflect the entire GPU, not individual MIG instances (unless using MIG-aware profiling).

### Metric-Specific Caveats

- **Tensor Active / FP16 Active**: May report undefined values on architectures without tensor cores or when no FP16 work is running.
- **NVLink metrics**: Only available when NVLink connections are active and configured.
- **PCIe throughput**: Includes all PCIe traffic (DMA, BAR1, peer-to-peer), which may overstate application-specific transfer rates.
- **Clock frequencies**: Reported as averages over the sample interval;瞬态频率变化 may not be captured.

---

## NVML Power and Temperature Metrics (Preview)

NVIDIA Management Library (NVML) provides system-level telemetry including power draw and temperature. Nsight Systems can sample these alongside GPU hardware metrics.

### Available NVML Metrics

| Metric | Unit | Description |
|--------|------|-------------|
| Power Draw | W (Watts) | Instantaneous GPU power consumption |
| GPU Temperature | C (Celsius) | Current GPU die temperature |
| Fan Speed | % | Current fan speed as a percentage of maximum |

### Enabling NVML Metrics

```bash
nsys profile --gpu-metrics-devices=all --gpu-metrics-set=full ./my_application
```

NVML power and temperature metrics are included in the `full` metric set and are displayed in the GUI alongside other GPU metrics.

### Use Cases

- **Power throttling diagnosis**: Correlate GPC Clock Frequency drops with Power Draw spikes to identify power-limited throttling.
- **Thermal analysis**: Track GPU Temperature over time to identify cooling issues.
- **Energy efficiency**: Calculate energy consumed per operation by integrating power over time.

### Example: Detect Power Throttling

```sql
SELECT
    clock.timestamp,
    clock.value AS gpc_clock_mhz,
    power.value AS power_watts
FROM
    (SELECT timestamp, value FROM GPU_METRICS
     WHERE metricId = (SELECT id FROM ENUM_GPU_METRICS WHERE name = 'GPC Clock Frequency [MHz]')
       AND gpuId = 0) clock
JOIN
    (SELECT timestamp, value FROM GPU_METRICS
     WHERE metricId = (SELECT id FROM ENUM_GPU_METRICS WHERE name = 'Power Draw [W]')
       AND gpuId = 0) power
ON clock.timestamp = power.timestamp
WHERE clock.value < 1000  -- Flag samples where clock dropped below expected
ORDER BY clock.timestamp;
```

---

## SoC Metrics

### Overview

System-on-Chip (SoC) metrics apply to NVIDIA SoC platforms such as Jetson and Grace Hopper. These platforms integrate GPU and CPU on the same die, and SoC metrics provide visibility into fabric traffic, memory controller utilization, and interconnect behavior unique to these architectures.

### Available SoC Metrics

| Metric | Description |
|--------|-------------|
| Fabric Clock | Clock frequency of the internal SoC fabric |
| GPU Fabric Bandwidth | Bandwidth utilization of the GPU-to-fabric link |
| CPU Fabric Bandwidth | Bandwidth utilization of the CPU-to-fabric link |
| EMC Bandwidth | External Memory Controller bandwidth (total system memory traffic) |
| EMC Read Bandwidth | Read bandwidth through the EMC |
| EMC Write Bandwidth | Write bandwidth through the EMC |

### Launching SoC Metrics from CLI

```bash
# Collect SoC metrics alongside GPU metrics
nsys profile --gpu-metrics-devices=all --gpu-metrics-set=full --soc-metrics=yes ./my_application
```

| Option | Description | Default |
|--------|-------------|---------|
| `--soc-metrics` | Enable SoC metric sampling (`yes`/`no`) | `no` |

### Launching SoC Metrics from GUI

1. Open **Project Properties**.
2. Navigate to the **GPU Metrics** section.
3. Enable **Collect SoC metrics** checkbox.
4. This option is only available on supported SoC platforms.

### SoC Metrics Analysis

SoC metrics are particularly useful for:

- **Grace Hopper Superchip**: Understanding the NVLink-C2C interconnect between Grace CPU and Hopper GPU, and analyzing unified memory traffic patterns.
- **Jetson platforms**: Identifying memory bandwidth bottlenecks in the shared memory subsystem (CPU and GPU share the same physical memory).
- **Power budgeting**: Correlating fabric and memory controller activity with power draw to optimize power-performance trade-offs.

### Example: Query SoC Fabric Metrics

```sql
SELECT
    m.timestamp,
    e.name,
    m.value
FROM GPU_METRICS m
JOIN ENUM_GPU_METRICS e ON m.metricId = e.id
WHERE e.name LIKE '%Fabric%'
  AND m.gpuId = 0
ORDER BY m.timestamp;
```

---

## Metric Quick Reference Table

| Metric | Unit | Category |
|--------|------|----------|
| GPC Clock Frequency | MHz | Clock |
| SYS Clock Frequency | MHz | Clock |
| GR Active | % | Compute |
| Sync Compute In Flight | % | Compute |
| Async Compute In Flight | % | Compute |
| Draw Started | Count | Graphics |
| Dispatch Started | Count | Compute |
| Vertex/Tess/Geometry Warps in Flight | % | Graphics |
| Pixel Warps in Flight | % | Graphics |
| Compute Warps in Flight | % | Compute |
| Active SM Unused Warp Slots | % | SM Utilization |
| Idle SM Unused Warp Slots | % | SM Utilization |
| SMs Active | % | SM Utilization |
| SM Issue | % | SM Utilization |
| Tensor Active | % | Tensor Core |
| Tensor Active / FP16 Active | Ratio | Tensor Core |
| DRAM Read Bandwidth | GB/s | Memory |
| DRAM Write Bandwidth | GB/s | Memory |
| VRAM Read Bandwidth | GB/s | Memory |
| VRAM Write Bandwidth | GB/s | Memory |
| NVENC Active | % | Encoder |
| NVENC Read Throughput | MB/s | Encoder |
| NVENC Write Throughput | MB/s | Encoder |
| OFA Active | % | Accelerator |
| OFA Read Throughput | MB/s | Accelerator |
| OFA Write Throughput | MB/s | Accelerator |
| NVLink Bytes Received | Bytes | Interconnect |
| NVLink Bytes Transmitted | Bytes | Interconnect |
| PCIe Read Throughput | MB/s | Interconnect |
| PCIe Write Throughput | MB/s | Interconnect |
| PCIe Read Requests to BAR1 | Count | Interconnect |
| PCIe Write Requests to BAR1 | Count | Interconnect |
