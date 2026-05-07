# NVIDIA Nsight Systems -- GPU Metrics Reference

This document provides a comprehensive reference for GPU hardware metrics, context switch tracing, SoC metrics, and NVIDIA video profiling in Nsight Systems.

## 1. GPU Context Switch Tracing

Nsight Systems provides the ability to trace GPU context switches.

**CLI:**
```bash
nsys profile --gpuctxsw=true <application>
```

**Requirements:** Driver r435.17 or later and root permission.

### 1.1 Behavior by Privilege Level

| Privilege | Record Contents |
|-----------|----------------|
| **Root** | Records about contexts from all processes. Valid context IDs, process IDs, and full-precision timestamps. |
| **Normal user** | Records about all processes. For your user's processes: valid context ID, PID, full-precision timestamps. For other users: context ID = 0, PID = 0, reduced-precision timestamps (still in correct order). |
| **vGPU** | Same rules apply within your VM. No records for contexts on other VMs. Timeline may show gaps when vGPU is switched to another VM's context(s). |

> **Note:** GPU context switch data collection on a host system where vGPUs are in use by VMs is not currently supported.

---

## 2. GPU Metrics Overview

The GPU Metrics feature is intended to identify performance limiters in applications using GPU for computations and graphics. It uses periodic sampling to gather performance metrics and detailed timing statistics associated with different GPU hardware units, taking advantage of specialized hardware to capture this data in a single pass with minimal overhead.

> **Note:** GPU Metrics gives precise device-level information but does not know which process or context is involved. GPU context switch trace provides less precise information but gives process and context information.

### 2.1 What GPU Metrics Provides

These metrics provide an overview of GPU efficiency over time within compute, graphics, and I/O activities:

- **IO throughputs:** PCIe, NVLink, and GPU memory bandwidth
- **SM utilization:** SMs activity, tensor core activity, instructions issued, warp occupancy, and unassigned warp slots

### 2.2 Key Questions GPU Metrics Helps Answer

| Question | Relevant Metrics |
|----------|-----------------|
| Is my GPU idle? | SMs Active, GR Active |
| Is my GPU full? Enough kernel grids and streams? | SMs Active, Active SM Unused Warp Slots |
| Am I using Tensor Cores? | Tensor Active |
| Is my instruction rate high? | SM Issue |
| Am I blocked on IO or number of warps? | PCIe Read/Write, NVLink, DRAM throughput |

### 2.3 Availability and Requirements

**Platforms:** Linux x86-64, aarch64, and Windows targets.
**Architecture:** NVIDIA Turing or newer.
**Permissions:** Elevated privileges required (sudo on Linux, admin on Windows).

**Minimum driver versions:**

| Architecture | Minimum Driver |
|-------------|---------------|
| NVIDIA Turing TU10x, TU11x | r440 |
| NVIDIA Ampere GA100 | r450 |
| NVIDIA Ampere GA100 MIG | r470 TRD1 |
| NVIDIA Ampere GA10x | r455 |

> **Note:** Tensor Core utilization can be found under the SM instructions/Tensor Active row when running `nsys profile --gpu-metrics-devices all`. It is not practical to expect 100% Tensor Core utilization due to overheads.

---

## 3. Launching GPU Metrics from the CLI

GPU Metrics is controlled with 3 CLI switches:

| Switch | Values | Default | Description |
|--------|--------|---------|-------------|
| `--gpu-metrics-devices` | all, cuda-visible, none, \<index\> | none | Select GPUs to sample |
| `--gpu-metrics-set` | \<alias\>, file:\<filename\> | First suitable set | Select metric set |
| `--gpu-metrics-frequency` | 10..200000 | 10000 | Sampling frequency in Hz |

### 3.1 Basic Usage

```bash
# Profile with GPU metrics on GPU 1
# Must have elevated permissions or be root/admin
nsys profile --gpu-metrics-devices=1 ./my-app
```

### 3.2 Listing Available GPUs

```bash
nsys profile --gpu-metrics-devices=help
```

Example output:
```
Possible --gpu-metrics-devices values are:
1: Turing TU104 | GeForce RTX 2070 SUPER PCI[0000:65:00.0]
all: Select all supported GPUs
cuda-visible: Select GPUs that match CUDA_VISIBLE_DEVICES
none: Disable GPU Metrics [Default]

Some GPUs are not supported:
0: Volta GV100 | Quadro GV100 PCI[0000:17:00.0]
```

### 3.3 Listing Available Metric Sets

```bash
nsys profile --gpu-metrics-devices=all --gpu-metrics-set=help
```

Example output:
```
Possible --gpu-metrics-set values are:
tu10x        : General Metrics for NVIDIA TU10x (any frequency)
tu10x-gfxt   : Graphics Throughput Metrics for NVIDIA TU10x (frequency >= 10kHz)
file:<file name> : use metric set from a given file
```

### 3.4 Setting Sampling Frequency

```bash
# Default is 10 kHz
nsys profile --gpu-metrics-frequency=20000 --gpu-metrics-devices=all ./my-app
```

---

## 4. Launching GPU Metrics from the GUI

1. When launching analysis in Nsight Systems, select "Collect GPU Metrics".
2. Select the GPUs dropdown to pick which GPUs to sample.
3. Select the Metric set dropdown to choose which available metric set to use.

> **Note:** Metric sets for GPUs that are not being sampled will be greyed out.

---

## 5. Sampling Frequency

Sampling frequency can be selected from 10 Hz to 200 kHz. The default is 10 kHz.

### 5.1 Frequency Considerations

The maximum sampling frequency without buffer overflow depends on:
- GPU (SM count)
- GPU load intensity
- Overall system load

The bigger the chip and higher the load, the lower the maximum frequency. Increase frequency until you see "Buffer overflow" in the Diagnostics Summary.

Each metric set has a recommended frequency range in its description. If you observe Inconsistent Data or Missing Data ranges on timeline, adjust frequency closer to the recommended range.

---

## 6. Available GPU Metrics

### 6.1 Clock Frequency Metrics

**GPC Clock Frequency** -- `gpc__cycles_elapsed.avg.per_second`
The average GPC clock frequency in hertz. In public documentation the GPC clock may be called the "Application" clock, "Graphic" clock, "Base" clock, or "Boost" clock.

> **Note:** The collection mechanism for GPC can result in a small fluctuation between samples.

**SYS Clock Frequency** -- `sys__cycles_elapsed.avg.per_second`
The average SYS clock frequency in hertz. The GPU front end (command processor), copy engines, and the performance monitor run at the SYS clock. On Turing and GA100 GPUs, the sampling frequency is based upon a period of SYS clocks (not time) so samples per second will vary with SYS clock. On GA10x GPUs, the sampling frequency is based upon a fixed frequency clock.

### 6.2 Graphics/Compute Engine Metrics

**GR Active** -- `gr__cycles_active.sum.pct_of_peak_sustained_elapsed`
The percentage of cycles the graphics/compute engine is active. The engine is active if there is any work in the graphics pipe or if the compute pipe is processing work.

> **Note for GA100 MIG:** MIG is not yet supported. This counter reports the activity of the primary GR engine.

**Sync Compute In Flight** -- `gr__dispatch_cycles_active_queue_sync.avg.pct_of_peak_sustained_elapsed`
The percentage of cycles with synchronous compute in flight. CUDA reports synchronous queue only with MPS configured with 64 sub-context (VEID=0). Graphics: true if any compute work from the direct queue is in flight.

**Async Compute in Flight** -- `gr__dispatch_cycles_active_queue_async.avg.pct_of_peak_sustained_elapsed`
The percentage of cycles with asynchronous compute in flight. CUDA: all compute work is asynchronous (exception: MPS with 64 sub-contexts). Graphics: true if compute work from a compute queue is in flight.

**Draw Started** -- `fe__draw_count.avg.pct_of_peak_sustained_elapsed`
The ratio of draw calls issued to the graphics pipe to the maximum sustained rate.

> **Note:** The percentage will always be very low as the front end can issue draw calls significantly faster than the pipe can execute them.

**Dispatch Started** -- `gr__dispatch_count.avg.pct_of_peak_sustained_elapsed`
The ratio of compute grid launches (dispatches) to the maximum sustained rate.

> **Note:** The percentage will always be very low as the front end can issue grid launches significantly faster than execution.

### 6.3 SM Warp Metrics

**Vertex/Tess/Geometry Warps in Flight** -- `tpc__warps_active_shader_vtg_realtime.avg.pct_of_peak_sustained_elapsed`
The ratio of active vertex, geometry, tessellation, and meshlet shader warps resident on the SMs to the maximum number of warps per SM as a percentage.

**Pixel Warps in Flight** -- `tpc__warps_active_shader_ps_realtime.avg.pct_of_peak_sustained_elapsed`
The ratio of active pixel/fragment shader warps resident on the SMs to the maximum number of warps per SM as a percentage.

**Compute Warps in Flight** -- `tpc__warps_active_shader_cs_realtime.avg.pct_of_peak_sustained_elapsed`
The ratio of active compute shader warps resident on the SMs to the maximum number of warps per SM as a percentage.

**Active SM Unused Warp Slots** -- `tpc__warps_inactive_sm_active_realtime.avg.pct_of_peak_sustained_elapsed`
The ratio of inactive warp slots on the SMs to the maximum number of warps per SM. Indicates how many more warps may fit if not limited by resources (max warps, shared memory, registers, thread blocks per SM).

**Idle SM Unused Warp Slots** -- `tpc__warps_inactive_sm_idle_realtime.avg.pct_of_peak_sustained_elapsed`
The ratio of inactive warp slots due to idle SMs to the maximum warps per SM. Indicates the workload is not sufficient to put work on all SMs. Possible causes:
- CPU starving the GPU
- Current work is too small to saturate the GPU
- Current work is trailing off but blocking next work

### 6.4 SM Utilization Metrics

**SMs Active** -- `sm__cycles_active.avg.pct_of_peak_sustained_elapsed`
The ratio of cycles SMs had at least 1 warp in flight (allocated on SM) to the number of cycles as a percentage. A value of 0 indicates all SMs were idle. A value of 50% can indicate either all SMs active 50% of the time or 50% of SMs active 100% of the time.

**SM Issue** -- `sm__inst_executed_realtime.avg.pct_of_peak_sustained_elapsed`
The ratio of cycles that SM sub-partitions (warp schedulers) issued an instruction to the number of cycles in the sample period as a percentage.

**Tensor Active** -- `sm__pipe_tensor_cycles_active_realtime.avg.pct_of_peak_sustained_elapsed`
The ratio of cycles the SM tensor pipes were active issuing tensor instructions to the number of cycles as a percentage.

> **Note for TU102/4/6:** This metric is not available on TU10x for periodic sampling. See Tensor Active/FP16 Active instead.

**Tensor Active / FP16 Active** -- `sm__pipe_shared_cycles_active_realtime.avg.pct_of_peak_sustained_elapsed`
TU102/4/6 only. The ratio of cycles the SM tensor pipes or FP16x2 pipes were active issuing tensor instructions.

### 6.5 Memory Metrics

**DRAM Read Bandwidth** -- `dramc__read_throughput.avg.pct_of_peak_sustained_elapsed`, `dram__read_throughput.avg.pct_of_peak_sustained_elapsed`

**VRAM Read Bandwidth** -- `FBPA.TriageA.dramc__read_throughput.avg.pct_of_peak_sustained_elapsed`, `FBSP.TriageSCG.dramc__read_throughput.avg.pct_of_peak_sustained_elapsed`, `FBSP.TriageAC.dramc__read_throughput.avg.pct_of_peak_sustained_elapsed`

The ratio of cycles the DRAM interface was active reading data to the elapsed cycles as a percentage.

**DRAM Write Bandwidth** -- `dramc__write_throughput.avg.pct_of_peak_sustained_elapsed`, `dram__write_throughput.avg.pct_of_peak_sustained_elapsed`

**VRAM Write Bandwidth** -- `FBPA.TriageA.dramc__write_throughput.avg.pct_of_peak_sustained_elapsed`, `FBSP.TriageSCG.dramc__write_throughput.avg.pct_of_peak_sustained_elapsed`, `FBSP.TriageAC.dramc__write_throughput.avg.pct_of_peak_sustained_elapsed`

The ratio of cycles the DRAM interface was active writing data to the elapsed cycles as a percentage.

### 6.6 Encoder/Decoder Metrics

**NVENC Active** -- `NVENC.TriageTop.nvenc__cycles_active.avg.pct_of_peak_sustained_elapsed`
The ratio of cycles the NVENC unit was actively processing a command to the number of cycles as a percentage.

**NVENC Read Throughput** -- `NVENC.TriageTop.nvenc__memif2nvenc_read_throughput.avg.pct_of_peak_sustained_elapsed`

**NVENC Write Throughput** -- `NVENC.TriageTop.nvenc__nvenc2memif_write_throughput.avg.pct_of_peak_sustained_elapsed`

The ratio of cycles the NVENC unit was actively processing read/write operations.

**OFA Active** -- `OFA.TriageTop.ofa_cycles_active.avg.pct_of_peak_sustained_elapsed`
The ratio of cycles the OFA (Optical Flow Accelerator) was actively processing a command.

**OFA Read Throughput** -- `OFA.TriageTop.ofa__memif2ofa_read_throughput.avg.pct_of_peak_sustained_elapsed`

**OFA Write Throughput** -- `OFA.TriageTop.ofa__ofa2memif_write_throughput.avg.pct_of_peak_sustained_elapsed`

The ratio of cycles the OFA was actively processing read/write operations.

### 6.7 Interconnect Metrics

**NVLink bytes received** -- `nvlrx__bytes.avg.pct_of_peak_sustained_elapsed`
The ratio of bytes received on the NVLink interface to the maximum receivable bytes as a percentage. Includes protocol overhead.

**NVLink bytes transmitted** -- `nvltx__bytes.avg.pct_of_peak_sustained_elapsed`
The ratio of bytes transmitted on the NVLink interface to the maximum transmittable bytes as a percentage. Includes protocol overhead.

**PCIe Read Throughput** -- `pcie__read_bytes.avg.pct_of_peak_sustained_elapsed`
The ratio of bytes received on the PCIe interface to the maximum receivable bytes as a percentage. Calculated based on PCIe generation and number of lanes. Includes protocol overhead.

**PCIe Write Throughput** -- `pcie__write_bytes.avg.pct_of_peak_sustained_elapsed`
The ratio of bytes transmitted on the PCIe interface to the maximum receivable bytes as a percentage. Calculated based on PCIe generation and number of lanes. Includes protocol overhead.

**PCIe Read Requests to BAR1** -- `pcie__rx_requests_aperture_bar1_op_read.sum`

**PCIe Write Requests to BAR1** -- `pcie__rx_requests_aperture_bar1_op_write.sum`

BAR1 is a PCIe interface used to allow the CPU or other devices to directly access GPU memory. The GPU normally transfers memory with copy engines (would not show as BAR1 activity). Heavier traffic is typically from:
- **Linux:** GPU Direct, GPU Direct RDMA, GPU Direct Storage
- **Windows:** Direct3D12 resources made accessible to CPU via NVAPI functions

---

## 7. Exporting and Querying GPU Metrics Data

It is possible to access metric values for automated processing using the Nsight Systems CLI export capabilities.

### 7.1 Example: Extract SMs Active Values

```bash
nsys export -t sqlite report.nsys-rep
sqlite3 report.sqlite "SELECT timestamp, value FROM GPU_METRICS JOIN TARGET_INFO_GPU_METRICS USING (metricId) WHERE value != 0 and metricName == 'SMs Active' LIMIT 10;"
```

Example output:
```
309277039|80
309301295|99
309325583|99
309349776|99
309373872|60
309397872|19
309421840|100
309446000|100
309470096|100
309494161|99
```

Values are integer percentages (0..100).

---

## 8. GPU Metrics Limitations

### 8.1 NVLink Inactive Links

If metric sets with NVLink are used but the links are not active, they may appear as fully utilized.

### 8.2 Counter Subscription Conflicts

Only one tool that subscribes to these counters can be used at a time. Nsight Systems GPU Metrics cannot be used simultaneously with:

- Nsight Graphics
- Nsight Compute
- DCGM (Data Center GPU Manager)

To pause DCGM:
```bash
dcgmi profile --pause
# Resume later
dcgmi profile --resume
```

Or use the API:
```bash
dcgmProfPause
dcgmProfResume
```

Non-NVIDIA products using CUPTI sampling or DCGM library will also conflict.

### 8.3 Memory Limits

Nsight Systems limits the amount of memory for storing GPU Metrics samples. Analysis with higher sampling rates or on GPUs with more SMs has a risk of exceeding this limit, leading to gaps filled with "Missing Data" ranges.

---

## 9. NVML Power and Temperature Metrics (Preview)

Nsight Systems can periodically sample power and temperature metrics from GPUs and plot them on the timeline.

- **Power metrics:** Provided in milliwatts (mW) via `nvmlDeviceGetPowerUsage`
- **Temperature metrics:** Provided in degrees Celsius (C) via `nvmlDeviceGetTemperature`

### 9.1 Enabling NVML Metrics

Add to `nsys profile` or `nsys start` commands:

```bash
nsys profile --enable nvml_metrics[,arg1[=value1],arg2[=value2],...] ...
```

No spaces after `nvml_metrics`. Arguments are comma-separated.

### 9.2 Supported Arguments

| Short | Long | Parameters | Default | Description |
|-------|------|------------|---------|-------------|
| `-i` | `--interval` | integer | 100 | Sampling interval in milliseconds |
| `-h` | `--help` | | | Print help message |

### 9.3 Usage Examples

```bash
# Sample power and temperature every 100ms (default)
nsys profile --enable nvml_metrics ./my-app

# Sample every 10ms
nsys profile --enable nvml_metrics,-i10 ./my-app
```

---

## 10. SoC Metrics (Embedded Platforms Only)

### 10.1 Overview

SoC Metrics identifies performance limiters in applications running on NVIDIA SoCs and is similar to GPU Metrics. Available for Linux and QNX targets on aarch64. Requires NVIDIA Orin architecture or newer.

### 10.2 CLI Switches

| Switch | Values | Default | Description |
|--------|--------|---------|-------------|
| `--soc-metrics` | true, false | false | Enable SoC Metrics sampling |
| `--soc-metrics-set` | \<alias\>, file:\<filename\> | First suitable | Metric set to use |
| `--soc-metrics-frequency` | 100..1000000 | 10000 | Sampling frequency in Hz |

Basic usage:
```bash
# Must be root or in 'debug' group
nsys profile --soc-metrics=true ./my-app
```

### 10.3 Available SoC Metrics

#### CPU Throughput Metrics

| Metric | Internal Name | Description |
|--------|--------------|-------------|
| CPU Read Throughput | `mcc__dram_throughput_srcnode_cpu_op_read.avg.pct_of_peak_sustained_elapsed` | Cycles SoC memory controllers actively processing CPU read operations |
| CPU Write Throughput | `mcc__dram_throughput_srcnode_cpu_op_write.avg.pct_of_peak_sustained_elapsed` | Cycles SoC memory controllers actively processing CPU write operations |

#### GPU Throughput Metrics

| Metric | Internal Name | Description |
|--------|--------------|-------------|
| GPU Read Throughput | `mcc__dram_throughput_srcnode_gpu_op_read.avg.pct_of_peak_sustained_elapsed` | Cycles SoC memory controllers actively processing GPU read operations |
| GPU Write Throughput | `mcc__dram_throughput_srcnode_gpu_op_write.avg.pct_of_peak_sustained_elapsed` | Cycles SoC memory controllers actively processing GPU write operations |

#### DBB Throughput Metrics

| Metric | Internal Name | Description |
|--------|--------------|-------------|
| DBB Read Throughput | `mcc__dram_throughput_srcnode_dbb_op_read.avg.pct_of_peak_sustained_elapsed` | Cycles SoC memory controllers actively processing non-CPU/non-GPU read operations |
| DBB Write Throughput | `mcc__dram_throughput_srcnode_dbb_op_write.avg.pct_of_peak_sustained_elapsed` | Cycles SoC memory controllers actively processing non-CPU/non-GPU write operations |

#### DRAM Aggregate Metrics

| Metric | Internal Name | Description |
|--------|--------------|-------------|
| DRAM Read Throughput | `mcc__dram_throughput_op_read.avg.pct_of_peak_sustained_elapsed` | Cycles SoC memory controllers actively processing all read operations |
| DRAM Write Throughput | `mcc__dram_throughput_op_write.avg.pct_of_peak_sustained_elapsed` | Cycles SoC memory controllers actively processing all write operations |

#### DLA Metrics (DLA0/DLA1)

| Metric | Internal Name | Description |
|--------|--------------|-------------|
| DLA Active | `nvdla__cycles_active.avg.pct_of_peak_sustained_elapsed` | Cycles DLA (Deep Learning Accelerator) actively processing commands |
| DLA Read Throughput | `nvdla__dbb2nvdla_read_throughput.avg.pct_of_peak_sustained_elapsed` | Cycles DLA actively processing read operations |
| DLA Write Throughput | `nvdla__nvdla2dbb_write_throughput.avg.pct_of_peak_sustained_elapsed` | Cycles DLA actively processing write operations |

#### NVENC Metrics

| Metric | Internal Name | Description |
|--------|--------------|-------------|
| NVENC Active | `nvenc__cycles_active.avg.pct_of_peak_sustained_elapsed` | Cycles NVENC actively processing commands |
| NVENC Read Throughput | `nvenc__memif2nvenc_read_throughput.avg.pct_of_peak_sustained_elapsed` | Cycles NVENC actively processing read operations |
| NVENC Write Throughput | `nvenc__nvenc2memif_write_throughput.avg.pct_of_peak_sustained_elapsed` | Cycles NVENC actively processing write operations |

#### PVA VPU Metrics

| Metric | Internal Name | Description |
|--------|--------------|-------------|
| PVA VPU Active | `pvavpu__vpu_cycles_active.avg.pct_of_peak_sustained_elapsed` | Cycles PVA (Programmable Vision Accelerator) VPU actively processing commands |
| PVA DMA Read Throughput | `pva__dbb2pvadma_read_throughput.avg.pct_of_peak_sustained_elapsed` | Cycles PVA DMA actively processing read operations |
| PVA DMA Write Throughput | `pva__pvadma2dbb_write_throughput.avg.pct_of_peak_sustained_elapsed` | Cycles PVA DMA actively processing write operations |

> **Note:** To enable PVA trace on DRIVE 6.0.8.0, run before mounting additional partitions:
```bash
echo 1 >/dev/nvpvadebugfs/pva0/tracing
echo 2 >/dev/nvpvadebugfs/pva0/trace_level
```

#### OFA Metrics

| Metric | Internal Name | Description |
|--------|--------------|-------------|
| OFA Active | `ofa_cycles_active.avg.pct_of_peak_sustained_elapsed` | Cycles OFA (Optical Flow Accelerator) actively processing commands |
| OFA Read Throughput | `ofa__memif2ofa_read_throughput.avg.pct_of_peak_sustained_elapsed` | Cycles OFA actively processing read operations |
| OFA Write Throughput | `ofa__ofa2memif_write_throughput.avg.pct_of_peak_sustained_elapsed` | Cycles OFA actively processing write operations |

#### VIC Metrics

| Metric | Internal Name | Description |
|--------|--------------|-------------|
| VIC Active | `vic_cycles_active.avg.pct_of_peak_sustained_elapsed` | Cycles VIC (Video Image Compositor) actively processing commands |
| VIC Read Throughput | `vic__dbb2vic_read_throughput.avg.pct_of_peak_sustained_elapsed` | Cycles VIC actively processing read operations |
| VIC Write Throughput | `vic__vic2dbb_write_throughput.avg.pct_of_peak_sustained_elapsed` | Cycles VIC actively processing write operations |

### 10.4 Launching SoC Metrics from the GUI

When launching analysis, select "Collect SoC Metrics". Settings are similar to GPU Metrics.

---

## 11. NVIDIA Video Profiling

### 11.1 NVIDIA Video Hardware Profiling

#### Requirements

- Linux (x86_64 or Arm) and Windows (x86_64)
- Desktop platforms running ResMan kernel driver
- Driver version >= 535
- GPU architecture Turing+

**Not supported for:**
- Mobile platforms
- Driver version < 535
- GPU architecture < Turing
- GSP enabled and Driver < 545.31
- MIG enabled
- Confidential computing enabled
- vGPU

#### Disabling GSP (if needed)

```bash
# Permanent disable
sudo su -c 'echo options nvidia NVreg_EnableGpuFirmware=0 > /etc/modprobe.d/nvidia-gsp.conf'
sudo update-initramfs -u  # Ubuntu-based
# Then reboot
```

Temporary disable (until next reboot):
```bash
sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia && \
sudo insmod /lib/modules/$(uname -r)/updates/dkms/nvidia.ko NVreg_EnableGpuFirmware=0
for i in $(seq 0 7); do sudo nvidia-smi -i $i -pm ENABLED; done
```

#### Running from the CLI

The feature is enabled through the `--gpu-video-device` option:

```bash
# List supported devices
nsys profile --gpu-video-device help
```

Example output:
```
Possible --gpu-video-device values are:
0: NVIDIA GeForce RTX 3070 PCI[0000:65:00.0]
all: Select all supported GPUs
none: Disable GPU video accelerator tracing [Default]

Some GPUs don't support video accelerator tracing:
Quadro P620 PCI[0000:04:00.0] (reason = Arch Pascal < Turing)
```

Arguments:
- `help` -- list supported devices and IDs
- `none` -- disable (default)
- `all` -- enable on all supported devices
- `<id1,id2,...>` -- enable on specific devices

> **Note:** This is a system-wide feature; it does not require a program to be launched.

### 11.2 NVIDIA Video Codec SDK Trace

Nsight Systems for x86 Linux and Windows can trace calls from the NV Video Codec SDK.

**CLI:**
```bash
nsys profile --trace=nvvideo <application>
```

**GUI:**
Enable NV Video Codec SDK trace selection.

On the timeline, calls on the CPU to the NV Encoder API and NV Decoder API will be shown.

### 11.3 NV Encoder API Functions Traced by Default

```
NvEncodeAPICreateInstance
nvEncOpenEncodeSession
nvEncGetEncodeGUIDCount
nvEncGetEncodeGUIDs
nvEncGetEncodeProfileGUIDCount
nvEncGetEncodeProfileGUIDs
nvEncGetInputFormatCount
nvEncGetInputFormats
nvEncGetEncodeCaps
nvEncGetEncodePresetCount
nvEncGetEncodePresetGUIDs
nvEncGetEncodePresetConfig
nvEncGetEncodePresetConfigEx
nvEncInitializeEncoder
nvEncCreateInputBuffer
nvEncDestroyInputBuffer
nvEncCreateBitstreamBuffer
nvEncDestroyBitstreamBuffer
nvEncEncodePicture
nvEncLockBitstream
nvEncUnlockBitstream
nvEncLockInputBuffer
nvEncUnlockInputBuffer
nvEncGetEncodeStats
nvEndGetSequenceParams
nvEncRegisterAsyncEvent
nvEncUnregisterAsyncEvent
nvEncMapInputResource
nvEncUnmapInputResource
nvEncDestroyEncoder
nvEncInvalidateRefFrames
nvEncOpenEncodeSessionEx
nvEncRegisterResource
nvEncUnregisterResource
nvEncReconfigureEncoder
nvEncCreateMVBuffer
nvEncDestroyMVBuffer
nvEncRunMotionEstimationOnly
nvEncGetLastErrorString
nvEncSetIOCudaStreams
nvEncGetSequenceParamEx
```

### 11.4 NV Decoder API Functions Traced by Default

```
cuvidCreateVideoSource
cuvidCreateVideoSourceW
cuvidDestroyVideoSource
cuvidSetVideoSourceState
cudaVideoState
cuvidGetSourceVideoFormat
cuvidGetSourceAudioFormat
cuvidCreateVideoParser
cuvidParseVideoData
cuvidDestroyVideoParser
cuvidCreateDecoder
cuvidDestroyDecoder
cuvidDecodePicture
cuvidGetDecodeStatus
cuvidReconfigureDecoder
cuvidMapVideoFrame
cuvidUnmapVideoFrame
cuvidMapVideoFrame64
cuvidUnmapVideoFrame64
cuvidCtxLockCreate
cuvidCtxLockDestroy
cuvidCtxLock
cuvidCtxUnlock
```

### 11.5 NV JPEG API Functions Traced by Default

```
nvjpegBufferDeviceCreate
nvjpegBufferDeviceDestroy
nvjpegBufferDeviceRetrieve
nvjpegBufferPinnedCreate
nvjpegBufferPinnedDestroy
nvjpegBufferPinnedRetrieve
nvjpegCreate
nvjpegCreateEx
nvjpegCreateSimple
nvjpegDecode
nvjpegDecodeBatched
nvjpegDecodeBatchedEx
nvjpegDecodeBatchedInitialize
nvjpegDecodeBatchedPreAllocate
nvjpegDecodeBatchedSupported
nvjpegDecodeBatchedSupportedEx
nvjpegDecodeJpeg
nvjpegDecodeJpegDevice
nvjpegDecodeJpegHost
nvjpegDecodeJpegTransferToDevice
nvjpegDecodeParamsCreate
nvjpegDecodeParamsDestroy
nvjpegDecodeParamsSetAllowCMYK
nvjpegDecodeParamsSetOutputFormat
nvjpegDecodeParamsSetROI
nvjpegDecodeParamsSetScaleFactor
nvjpegDecoderCreate
nvjpegDecoderDestroy
nvjpegDecoderJpegSupported
nvjpegDecoderStateCreate
nvjpegDestroy
nvjpegEncodeGetBufferSize
nvjpegEncodeImage
nvjpegEncodeRetrieveBitstream
nvjpegEncodeRetrieveBitstreamDevice
nvjpegEncoderParamsCopyHuffmanTables
nvjpegEncoderParamsCopyMetadata
nvjpegEncoderParamsCopyQuantizationTables
nvjpegEncoderParamsCreate
nvjpegEncoderParamsDestroy
nvjpegEncoderParamsSetEncoding
nvjpegEncoderParamsSetOptimizedHuffman
nvjpegEncoderParamsSetQuality
nvjpegEncoderParamsSetSamplingFactors
nvjpegEncoderStateCreate
nvjpegEncoderStateDestroy
nvjpegEncodeYUV
nvjpegGetCudartProperty
nvjpegGetDeviceMemoryPadding
nvjpegGetImageInfo
nvjpegGetPinnedMemoryPadding
nvjpegGetProperty
nvjpegJpegStateCreate
nvjpegJpegStateDestroy
nvjpegJpegStreamCreate
nvjpegJpegStreamDestroy
nvjpegJpegStreamGetChromaSubsampling
nvjpegJpegStreamGetComponentDimensions
nvjpegJpegStreamGetComponentsNum
nvjpegJpegStreamGetFrameDimensions
nvjpegJpegStreamGetJpegEncoding
nvjpegJpegStreamParse
nvjpegJpegStreamParseHeader
nvjpegSetDeviceMemoryPadding
nvjpegSetPinnedMemoryPadding
nvjpegStateAttachDeviceBuffer
nvjpegStateAttachPinnedBuffer
```
