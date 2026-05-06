# Containers, Migration, Video Profiling, Embedded VMs, and Plugins Reference

This document provides comprehensive reference material for container-based profiling, migrating from nvprof, NVIDIA video profiling, embedded virtual machine profiling, and Nsight Systems plugins.

---

## Table of Contents

- [Container and Scheduler Support](#container-and-scheduler-support)
  - [Collecting Data Within a Container](#collecting-data-within-a-container)
  - [Enable Docker Collection](#enable-docker-collection)
  - [Launch Docker Collection](#launch-docker-collection)
  - [Profiling Services Launched via Kubernetes](#profiling-services-launched-via-kubernetes)
  - [Nsight Streamer for Nsight Systems](#nsight-streamer-for-nsight-systems)
  - [GUI VNC Container](#gui-vnc-container)
    - [Available Parameters](#available-parameters)
    - [Ports](#ports)
    - [Volumes](#volumes)
    - [Environment Variables](#environment-variables)
    - [Usage Examples](#usage-examples)
- [Migrating from NVIDIA nvprof](#migrating-from-nvidia-nvprof)
  - [Using the Nsight Systems CLI nvprof Command](#using-the-nsight-systems-cli-nvprof-command)
  - [CLI nvprof Command Switch Options](#cli-nvprof-command-switch-options)
  - [Next Steps](#next-steps)
- [NVIDIA Video Profiling](#nvidia-video-profiling)
  - [NVIDIA Video Hardware Profiling](#nvidia-video-hardware-profiling)
    - [Limitations and Requirements](#limitations-and-requirements)
    - [Running from the CLI](#running-from-the-cli)
  - [NVIDIA Video Codec SDK Trace](#nvidia-video-codec-sdk-trace)
    - [NV Encoder API Functions Traced by Default](#nv-encoder-api-functions-traced-by-default)
    - [NV Decoder API Functions Traced by Default](#nv-decoder-api-functions-traced-by-default)
    - [NV JPEG API Functions Traced by Default](#nv-jpeg-api-functions-traced-by-default)
- [Profiling Embedded Virtual Machines](#profiling-embedded-virtual-machines)
  - [XHV Trace Configuration](#xhv-trace-configuration)
  - [Specific Command Line Options](#specific-command-line-options)
  - [Config File for Kernel Symbols](#config-file-for-kernel-symbols)
  - [Symbol Files](#symbol-files)
  - [XHV Profiling from the GUI](#xhv-profiling-from-the-gui)
- [Adding Your Own Collection to a Report](#adding-your-own-collection-to-a-report)
- [Nsight Systems Plugins (Preview)](#nsight-systems-plugins-preview)
  - [What is a Plugin](#what-is-a-plugin)
  - [Manifest File Contents](#manifest-file-contents)
  - [How to Launch a Plugin](#how-to-launch-a-plugin)
  - [How to Pass Arguments to a Plugin](#how-to-pass-arguments-to-a-plugin)
  - [Supported Platforms](#supported-platforms)
  - [Sample Plugin](#sample-plugin)

---

## Container and Scheduler Support

### Collecting Data Within a Container

While examples in this section use Docker container semantics, other containers work much the same. The following information assumes the reader is knowledgeable regarding Docker containers.

**Best Practices:**
- It is strongly recommended to use the **CLI** to profile in a container
- Best container practice is to split services across containers when they do not require colocation
- The Nsight Systems GUI is not needed to profile and brings in many dependencies, so the CLI is recommended
- If you wish, the GUI can be in a separate side-car container you use to view your report
- All you need is a shared folder between the containers

### Enable Docker Collection

When starting the Docker to perform a Nsight Systems collection, additional steps are required to enable the `perf_event_open` system call. This is required in order to utilize the Linux kernel's perf subsystem which provides sampling information to Nsight Systems.

**Three ways to enable perf_event_open syscall:**

1. **Privileged mode:**

```bash
docker run --privileged=true ...
```

2. **Add SYS_ADMIN capability:**

```bash
docker run --cap-add=SYS_ADMIN ...
```

3. **Set seccomp security profile:**

Secure computing mode (seccomp) is a feature of the Linux kernel that can be used to restrict an application's access. This feature is available only if the kernel is enabled with seccomp support.

**To check for seccomp support:**

```bash
$ grep CONFIG_SECCOMP= /boot/config-$(uname -r)
```

**To configure the seccomp profile:**

Download the default seccomp profile file, `default.json`, relevant to your Docker version. If `perf_event_open` is already listed in the file as guarded by `CAP_SYS_ADMIN`, then remove the `perf_event_open` line. Add the following lines under "syscalls" and save the resulting file as `default_with_perf.json`:

```json
{
    "name": "perf_event_open",
    "action": "SCMP_ACT_ALLOW",
    "args": []
}
```

Then use the following switch when starting the Docker to apply the new seccomp profile:

```
--security-opt seccomp=default_with_perf.json
```

### Launch Docker Collection

Here is an example command that has been used to launch a Docker for testing with Nsight Systems:

```bash
sudo nvidia-docker run --network=host \
    --security-opt seccomp=default_with_perf.json \
    --rm -ti caffe-demo2 bash
```

**Known Issue:** There is a known issue where Docker collections terminate prematurely with older versions of the driver and the CUDA Toolkit. If collection is ending unexpectedly, please update to the latest versions.

After the Docker has been started, use the Nsight Systems CLI to launch a collection within the Docker. The resulting file can be imported into the Nsight Systems host like any other CLI result.

### Profiling Services Launched via Kubernetes

Nsight Systems can provide profiling via sidecar injection without need to modify your containers or k8/helm specs.

**Features:**
- Data collected can be filtered by namespace or pod using Kubernetes labels
- Data can be filtered within a container process using command-line regex
- Compatible with various cloud service provider's in-house managed Kubernetes variants including AKS, EKS, GKE, and OKE

**Documentation and Download:**
Available at NGC Nsight Operator.

### Nsight Streamer for Nsight Systems

A self-hosted NVIDIA Nsight Systems GUI running inside a Docker container enables remote access through a web browser. This configuration is particularly useful for analyzing data on remote servers or clusters.

**More Information:**
Visit Nsight Streamer for Nsight Systems on NGC.

### GUI VNC Container

Nsight Systems provides a build script to build a self-isolated Docker container with the Nsight Systems GUI and VNC server.

**Build Script Location:**
`host-linux-x64/Scripts/VncContainer` directory (or similar on other architectures) under your Nsight Systems installation directory.

**Requirements:**
- Docker
- Python 3.5 or later

#### Available Parameters

| Short Name | Full Name | Description |
|------------|-----------|-------------|
| | `--vnc-password` | (optional) Default password for VNC access (at least 6 characters). If it is specified and empty user will be asked during the build. Can be changed when running a container. |
| `-aba` | `--additional-build-arguments` | (optional) Additional arguments, which will be passed to the docker build command. |
| `-hd` | `--nsys-host-directory` | (optional) The directory with Nsight Systems host binaries (with GUI). |
| `-td` | `--nsys-target-directory` | (optional, repeatable) The directory with Nsight Systems target binaries (can be specified multiple times). |
| | `--tigervnc` | (optional) Use TigerVNC instead of x11vnc. |
| | `--http` | (optional) Install noVNC in the Docker container for HTTP access. |
| | `--rdp` | (optional) Install xRDP in the Docker for RDP access. |
| | `--geometry` | (optional) Default VNC server resolution in the format WidthxHeight (default 1920x1080). |
| | `--build-directory` | (optional) The directory to save temporary files (with the write access for the current user). By default, script or tmp directory will be used. |

#### Ports

These ports can be published from the container to provide access to the Docker container:

| Port | Purpose | Condition |
|------|---------|-----------|
| TCP 5900 | Port for VNC access | Always available |
| TCP 80 | Port for HTTP access to noVNC server | Container is built with `--http` parameter |
| TCP 3389 | Port for RDP access | Container is built with `--rdp` parameter |

#### Volumes

| Docker Folder | Purpose | Description |
|---------------|---------|-------------|
| `/mnt/host` | Root path for shared folders | Folder owned by the Docker user (inner content can be accessed from Nsight Systems GUI) |
| `/mnt/host/Projects` | Folder with projects and reports | Created by Nsight Systems UI in container |
| `/mnt/host/logs` | Folder with inner services logs | May be useful to send reports to developers |

#### Environment Variables

| Variable Name | Purpose |
|---------------|---------|
| `VNC_PASSWORD` | Password for VNC access (at least 6 characters) |
| `NSYS_WINDOW_WIDTH` | Width of VNC server display (in pixels) |
| `NSYS_WINDOW_HEIGHT` | Height of VNC server display (in pixels) |

#### Usage Examples

**VNC access on port 5916:**

```bash
sudo docker run -p 5916:5900/tcp -ti nsys-ui-vnc:1.0
```

**VNC access on port 5916 and HTTP access on port 8080:**

```bash
sudo docker run -p 5916:5900/tcp -p 8080:80/tcp -ti nsys-ui-vnc:1.0
```

**VNC access on port 5916, HTTP access on port 8080, and RDP access on port 33890:**

```bash
sudo docker run -p 5916:5900/tcp -p 8080:80/tcp -p 33890:3389/tcp -ti nsys-ui-vnc:1.0
```

**VNC access on port 5916, shared HOME folder, custom resolution and VNC password:**

```bash
sudo docker run -p 5916:5900/tcp \
    -v $HOME:/mnt/host/home \
    -e NSYS_WINDOW_WIDTH=3840 \
    -e NSYS_WINDOW_HEIGHT=2160 \
    -e VNC_PASSWORD=7654321 \
    -ti nsys-ui-vnc:1.0
```

**VNC access on port 5916, shared HOME and projects folder:**

```bash
sudo docker run -p 5916:5900/tcp \
    -v $HOME:/mnt/host/home \
    -v /opt/NsysProjects:/mnt/host/Projects \
    -ti nsys-ui-vnc:1.0
```

---

## Migrating from NVIDIA nvprof

### Using the Nsight Systems CLI nvprof Command

The `nvprof` command of the Nsight Systems CLI is intended to help former nvprof users transition to `nsys`. Many nvprof switches are not supported by `nsys`, often because they are now part of NVIDIA Nsight Compute.

**References:**
- Full nvprof documentation: https://docs.nvidia.com/cuda/profiler-users-guide
- nvprof transition guide for Nsight Compute: https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#nvprof-guide

**Important Notes:**
- Any nvprof switch not listed below is not supported by the `nsys` nvprof command
- No additional `nsys` functionality is available through this command
- New features will not be added to this command in the future

**Usage:**

```bash
nsys nvprof [options]
```

### CLI nvprof Command Switch Options

After choosing the nvprof command switch, the following options are available. When you are ready to move to using Nsight Systems CLI directly, see the Command Line Options documentation for the `nsys` switch(es) given below. Note that the `nsys` implementation and output may vary from nvprof.

| Switch | Parameters (Default in Bold) | nsys Switch | Switch Description |
|--------|-------------------------------|-------------|-------------------|
| `--annotate-mpi` | **off**, openmpi, mpich | `--trace=mpi` AND `--mpi-impl` | Automatically annotate MPI calls with NVTX markers. Specify the MPI implementation installed on your machine. Only OpenMPI and MPICH implementations are supported. |
| `--cpu-thread-tracing` | **on**, off | `--trace=osrt` | Collect information about CPU thread API activity. |
| `--profile-api-trace` | none, runtime, driver, **all** | `--trace=cuda` | Turn on/off CUDA runtime and driver API tracing. For Nsight Systems there is no separate CUDA runtime and CUDA driver trace, so selecting runtime or driver is equivalent to selecting all. |
| `--profile-from-start` | **on**, off | if off use `--capture-range=cudaProfilerApi` | Enable/disable profiling from the start of the application. If disabled, the application can use {cu,cuda}Profiler{Start,Stop} to turn on/off profiling. |
| `-t` / `--timeout` | <nanoseconds> default=**0** | `--duration=seconds` | If greater than 0, stop the collection and kill the launched application after timeout seconds. nvprof started counting when the CUDA driver is initialized. nsys starts counting immediately. |
| `--cpu-profiling` | **on**, off | `--sampling=cpu` | Turn on/off CPU profiling |
| `--openacc-profiling` | **on**, off | `--trace=openacc` to turn on | Enable/disable recording information from the OpenACC profiling interface. Note: OpenACC profiling interface depends on the presence of the OpenACC runtime. |
| `-o` / `--export-profile` | <filename> | `--output={filename}` and/or `--export=sqlite` | Export named file to be imported or opened in the Nsight Systems GUI. `%q{ENV_VAR}` in string will be replaced with the value of the environment variable. If not set this is an error. `%h` is replaced with the system hostname. `%%` in the string is replaced with `%`. `%p` in the string is not supported currently. Any other character following `%` is illegal. The default is report1, with the number incrementing to avoid overwriting files, in the user's working directory. |
| `-f` / `--force-overwrite` | | `--force-overwrite=true` | Force overwriting all output files with same name. |
| `-h` / `--help` | | `--help` | Print Nsight Systems CLI help |
| `-V` / `--version` | | `--version` | Print Nsight Systems CLI version information |

### Next Steps

NVIDIA Visual Profiler (NVVP) and NVIDIA nvprof are deprecated. New GPUs and features will not be supported by those tools. We encourage you to make the move to Nsight Systems now.

For additional information, suggestions, and rationale, see the blog series in Other Resources.

---

## NVIDIA Video Profiling

### NVIDIA Video Hardware Profiling

#### Limitations and Requirements

**Requirements:**
- Linux (x86_64 or Arm) and Windows (x86_64)
- Only covers desktop platforms running ResMan kernel driver
- Driver version >= 535
- GPU architecture Turing+

**Not Supported On:**
- Mobile platforms
- Driver version < 535
- GPU architecture < Turing
- GSP is enabled and Driver < 545.31
- MIG is enabled
- Confidential computing is enabled
- vGPU

**Disabling GSP:**

To turn off GSP permanently:

```bash
sudo su -c 'echo options nvidia NVreg_EnableGpuFirmware=0 > /etc/modprobe.d/nvidia-gsp.conf'
sudo update-initramfs -u  # for Ubuntu-based systems
# Then reboot.
```

To disable GSP until the next reboot:

```bash
sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia && \
sudo insmod /lib/modules/$(uname -r)/updates/dkms/nvidia.ko NVreg_EnableGpuFirmware=0
for i in $(seq 0 7); do sudo nvidia-smi -i $i -pm ENABLED; done
```

#### Running from the CLI

The feature is enabled through the `--gpu-video-device` option. It is available from the `nsys profile`, `nsys launch` and `nsys start` commands.

The option behaves exactly like `--gpu-metrics-device` and accepts the following arguments:

| Argument | Description |
|----------|-------------|
| `help` | List supported devices and their IDs; list unsupported devices (if any) and the reason |
| `none` | Turn the feature off |
| `all` | Turn the feature on on all supported devices (error if no devices support it) |
| `<id1,id2,...>` | Turn the feature on the specified devices (ID corresponds to what `help` returns) |

**Example:**

```bash
$ nsys profile --gpu-video-device help
Possible --gpu-video-device values are:
0: NVIDIA GeForce RTX 3070 PCI[0000:65:00.0]
all: Select all supported GPUs
none: Disable GPU video accelerator tracing [Default]

Some GPUs don't support video accelerator tracing:
Quadro P620 PCI[0000:04:00.0] (reason = Arch Pascal < Turing)
```

**Note:** This is a system-wide feature; it does not require a program to be launched.

### NVIDIA Video Codec SDK Trace

Nsight Systems for x86 Linux and Windows targets can trace calls from the NV Video Codec SDK. This software trace can be launched from the GUI or using `--trace nvvideo` from the CLI.

On the timeline, calls on the CPU to the NV Encoder API and NV Decoder API will be shown.

#### NV Encoder API Functions Traced by Default

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

#### NV Decoder API Functions Traced by Default

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

#### NV JPEG API Functions Traced by Default

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

---

## Profiling Embedded Virtual Machines

Nsight Systems and DRIVE Hypervisor support periodic CPU sampling with call stacks. It works both on DRIVE Linux and QNX.

The call stacks are collected using frame pointers. The Linux kernel, QNX kernel, and user space libraries provided by NVIDIA are compiled with frame pointers. To ensure correct call stacks, compile all application code with frame pointer support using `-fno-omit-frame-pointer` with GCC, Clang, and QCC.

**Important Notes:**
- This is an experimental feature and is expected to change in the future
- The symbols can be resolved both for user space code and for kernel space code

**Symbol Resolution:**

In the user space, the Cross-Hypervisor (XHV) sampling events are matched with the CPU thread state trace coming from Linux Perf and QNX Tracelogger. After that, Nsight Systems can know the module filename and can resolve symbols directly from these files if they are unstripped, or by looking up additional files with symbols.

In the kernel space (Linux kernel, QNX kernel, and additional service VMs), the symbols are resolved using the ELF file with symbols specified. `kernel_symbols.json` input file specifies the location of this ELF file.

### XHV Trace Configuration

**Steps to set up XHV profiling:**

1. Flash the devkit
2. Copy necessary files: `pct.json`, eventlib schema files, and `kernel_symbols.json`
3. Compose `kernel_symbols.json` to allow resolving symbols
4. See example CLI commands to collect data

**Known Issues:**
- This feature is not compatible with standard CPU sampling on Linux and QNX
- When enabled together, hypervisor trace plus XHV sampling can write too much data into the same eventlib buffers, and the Nsight Systems agent might not be able to keep up with the rate, losing events. If that happens, disable hypervisor trace events with `--xhv-trace-events=none`.

**Flashing DRIVE OS QNX/Linux:**

Log into the NVIDIA GPU Cloud (NGC):

```bash
sudo docker login nvcr.io
```

Username: `$oauthtoken` Password: \<NGC API key\>

Docker command:

```bash
sudo docker run --rm --privileged --net host \
    -v /dev/bus/usb:/dev/bus/usb \
    -v /tmp:/drive_flashing \
    -it <docker image>
```

**Examples:**

6.0.8.0 QNX:

```bash
sudo docker run --rm --privileged --net host \
    -v /dev/bus/usb:/dev/bus/usb \
    -v /tmp:/drive_flashing \
    -it nvcr.io/{MY_NGC_ORG}/driveos-pdk/drive-agx-orin-qnx-aarch64-pdk-build-x86:6.0.8.0-0003
```

6.0.9.1 QNX:

```bash
sudo docker run --rm --privileged --net host \
    -v /dev/bus/usb:/dev/bus/usb \
    -v /tmp:/drive_flashing \
    -it nvcr.io/{MY_NGC_ORG}/driveos-pdk/drive-agx-orin-qnx-aarch64-pdk-build-x86:6.0.9.1-latest
```

6.0.8.0 Linux:

```bash
sudo docker run --rm --privileged --net host \
    -v /dev/bus/usb:/dev/bus/usb \
    -v /tmp:/drive_flashing \
    -it nvcr.io/{MY_NGC_ORG}/driveos-pdk/drive-agx-orin-linux-aarch64-pdk-build-x86:6.0.8.0-0003
```

Inside of container, flash with flash.py:

```bash
cd /drive
./flash.py <aurix> <board>
```

- `<board>` -- Target board base name: `p3710` or `p3663`
- `<aurix>` -- Aurix serial port, for example: `/dev/ttyACM1`, `/dev/ttyUSB1`

Examples:

```bash
# Firespray p3710
./flash.py /dev/ttyACM1 p3710

# Drive Orin p3663
./flash.py /dev/ttyUSB1 p3663
```

**Creating XHV Directory:**

QNX:

```bash
cd /drive_flashing
mkdir -p xhv/hypervisor/configs/t234ref-release/pct/qnx xhv/schemas
cp -rv /drive/drive-foundation/virtualization/hypervisor/t23x/configs/t234ref-release/pct/p3710-10-a03/qnx/pct.json ./xhv/hypervisor/configs/t234ref-release/pct/qnx/
cp -rv /drive/drive-foundation/schemas/event ./xhv/schemas/
```

Linux:

```bash
cd /drive_flashing
mkdir -p xhv/hypervisor/configs/t234ref-release/pct/linux xhv/schemas
cp -rv /drive/drive-foundation/virtualization/hypervisor/t23x/configs/t234ref-release/pct/p3710-10-a03/linux/pct.json ./xhv/hypervisor/configs/t234ref-release/pct/linux/
cp -rv /drive/drive-foundation/schemas/event ./xhv/schemas/
```

**XHV Directory Structure:**

```
xhv/
  ├── hypervisor/
  │   └── configs/
  │       └── t234ref-release/
  │           └── pct/
  │               └── linux/
  │                   └── pct.json
  └── schemas/
      └── event/
          ├── audioserver_events.json
          ├── bpmp_events.json
          ├── cem_events.json
          ├── hv_events.json
          ├── i2c_events.json
          ├── Makefile.gen-event-headers.tmk
          ├── monitor_events.json
          ├── se_events.json
          ├── sysmgr_events.json
          └── vsc_events.json
```

Copy XHV directory to target:

```bash
scp -r xhv <user>@<target-IP>
```

### Specific Command Line Options

| Option | Possible Parameters | Default | Switch Description |
|--------|-------------------|---------|-------------------|
| `--sample` | process-tree, system-wide, xhv, xhv-system-wide, none | process-tree | Select `xhv` or `xhv-system-wide` to enable Cross-Hypervisor (XHV) sampling; requires root privileges |
| `--xhv-vm-symbols` | <filepath kernel_symbols.json> | none | XHV sampling config (optional, for kernel symbols) |
| `--xhv-trace` | <filepath pct.json> | none | Collect hypervisor trace |
| `--xhv-trace-events` | all, none, core, sched, irq, trap | all | HV trace events |

**Example Commands:**

```bash
# XHV process-tree sampling with NVTX, OSRT, CUDA trace
nsys profile --sample=xhv --trace=nvtx,osrt,cuda \
    --xhv-vm-symbols=/root/kernel_symbols.json \
    --xhv-trace=/root/xhv/hypervisor/configs/p3710-10-a01/pct/qnx/pct.json \
    --xhv-trace-events=none sleep 5

# XHV system-wide sampling
nsys profile --sample=xhv-system-wide \
    --xhv-vm-symbols=/root/kernel_symbols.json \
    --xhv-trace=/root/xhv/hypervisor/configs/p3710-10-a01/pct/qnx/pct.json \
    --xhv-trace-events=none sleep 5
```

### Config File for Kernel Symbols

**QNX kernel_symbols.json example:**

```json
{
    "guest_cfg": [
        {
            "guest_id": 0,
            "guest_name": "Guest VM 0",
            "symbols": "/root/symbols/procnto-smp-instr-safety.guest_vm.bin.sym"
        },
        {
            "guest_id": 1,
            "guest_name": "Update service",
            "symbols": "/root/symbols/procnto-smp-instr-safety.update_vm.bin.sym"
        },
        {
            "guest_id": 2,
            "guest_name": "Resource Manager Server"
        },
        {
            "guest_id": 3,
            "guest_name": "Storage Server"
        },
        {
            "guest_id": 4,
            "guest_name": "Ethernet Server"
        },
        {
            "guest_id": 5,
            "guest_name": "Debug Server"
        }
    ],
    "symbol_files": {
        "Sidekick": "/root/symbols/sidekick.unstripped"
    }
}
```

**Linux kernel_symbols.json example:**

```json
{
    "guest_cfg": [
        {
            "guest_id": 0,
            "guest_name": "Guest VM 0",
            "symbols": "/home/nvidia/vmlinux"
        },
        {
            "guest_id": 1,
            "guest_name": "Update service"
        }
    ],
    "symbol_files": {}
}
```

### Symbol Files

**Search Path Configuration:**

CLI: `DbgFileSearchPath` config option

```bash
NSYS_CONFIG_DIRECTIVES='DbgFileSearchPath="/lib:/root/symbols"' nsys profile \
    --sample=xhv \
    --xhv-vm-symbols=/root/kernel_symbols.json \
    --xhv-trace=/root/xhv/hypervisor/configs/p3710-10-a01/pct/qnx/pct.json \
    --xhv-trace-events=none sleep 5
```

GUI: Symbol location button.

**Note:** The search is non-recursive.

**Symbol File Search Methods** (Nsight Systems tries them sequentially for each target file):

1. **Build-id debug files** (CLI only)
   - `<symbol directory>/.build-id/...` directories with debug files (or links to debug files)

2. **Debuglink files** (CLI only)
   - `<symbol directory>/<symbol file>` both filename and CRC from debuglink section must be matched

3. **File name and build-id** (CLI/GUI)
   - `<symbol directory>/<symbol file>` by filename and build-id

**Default Search Paths:**
- Linux: `/usr/lib/debug`
- QNX: No default path

### XHV Profiling from the GUI

**XHV GUI Configuration Dialog Options:**

| Option | Description |
|--------|-------------|
| Collect HV Trace | Enable XHV tracing |
| pct.json location | The location of pct.json file on the host. There is predefined hierarchy of XHV JSON files |
| Collect VM Profile | Enable XHV sampling (depends on Collect HV Trace) |
| Event mask | Select XHV trace events (can be set to None) |
| kernel_symbols.json location | The location of kernel_symbols.json file on the host. Note that this file contains target paths to the kernel symbol files |
| Skip idle checkbox | Deprecated |
| Combine EL0 checkbox | Deprecated |

---

## Adding Your Own Collection to a Report

Nsight Systems allows the user to add additional information to a report file for display with other Nsight Systems options.

---

## Nsight Systems Plugins (Preview)

### What is a Plugin

Nsight Systems plugins are standalone applications that can be profiled along with the main application or without one in a system-wide profiling. The NVTX events emitted by a plugin are displayed in the same timeline as the main application events. Additionally, any stdout and stderr streams are captured the same way as for a target application.

**How to make a plugin available:**
Create a directory with a manifest file, `nsys-plugin.yaml`, then place it in a "plugins" directory next to the Nsight Systems target CLI binary. The manifest file describes the plugin and its configuration.

### Manifest File Contents

The manifest file is a YAML file with the following required fields:

```yaml
PluginName: SamplePlugin
ExecutablePath: PluginExecutableRelativeToManifest
Description: This is a sample plugin.
```

### How to Launch a Plugin

Plugins are supported in `nsys profile` and `nsys start` commands. Plugin processes are launched by Nsight Systems as if it was a target application and terminated at the end of profiling. It is possible to launch multiple instances of the same plugin by using multiple `--enable` options.

```bash
# Launch with a plugin
nsys profile --enable=SamplePlugin -- myApp

# Launch multiple instances of the same plugin
nsys profile --enable=SamplePlugin --enable=SamplePlugin -- myApp
```

### How to Pass Arguments to a Plugin

To pass arguments to a plugin, specify them as a part of `--enable` option after plugin name when launching the target application.

**Argument Rules:**
- Arguments should be separated by commas only (no spaces)
- Commas can be escaped with a backslash `\,`
- The backslash itself can be escaped by another backslash `\\`
- To include spaces in an argument, enclose the argument in double quotes `"`

```bash
# Pass arguments to a plugin
nsys profile --enable=SamplePlugin,arg1,arg2 -- myApp

# Arguments with spaces
nsys profile --enable=SamplePlugin,"arg with spaces",arg2 -- myApp
```

### Supported Platforms

Currently plugins are supported on **x86_64 and arm64 Linux**.

### Sample Plugin

You can look at the Nsight Systems installation path, for example `/opt/nvidia/nsight-systems/2024.5.1/target-linux-x64`, under the directory `samples` for the source code of a sample plugin.

The `NetworkPlugin.cpp` source file is the exact source code for the `network_interface` plugin that ships in binary form with Nsight Systems. Users can modify this plugin or use it as a guide to create their own plugin for profiling their intended source of metrics.
