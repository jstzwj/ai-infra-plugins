# Nsight Systems Release Notes and Troubleshooting Reference

## Table of Contents

- [Release Notes](#release-notes)
  - [Nsight Systems 2025.2 Highlights](#nsight-systems-20252-highlights)
  - [New Features by Category](#new-features-by-category)
- [Known Issues](#known-issues)
  - [General Issues](#general-issues)
  - [vGPU Issues](#vgpu-issues)
  - [Docker Issues](#docker-issues)
  - [CUDA Trace Issues](#cuda-trace-issues)
  - [Multi-Report Analysis Issues](#multi-report-analysis-issues)
- [Troubleshooting](#troubleshooting)
  - [General Troubleshooting](#general-troubleshooting)
  - [CLI Troubleshooting](#cli-troubleshooting)
  - [GUI Troubleshooting](#gui-troubleshooting)
  - [Symbol Resolution](#symbol-resolution)
  - [Logging and Diagnostics](#logging-and-diagnostics)
  - [Environment Variables](#environment-variables)
  - [Profiling Games](#profiling-games)
  - [WebGL Testing](#webgl-testing)
  - [QNX Target Issues](#qnx-target-issues)
  - [Launch Processes in Stopped State](#launch-processes-in-stopped-state)
  - [Broken Backtraces on Tegra](#broken-backtraces-on-tegra)
  - [Debug Versions of ELF Files](#debug-versions-of-elf-files)
  - [Common Error Messages and Solutions](#common-error-messages-and-solutions)
  - [Performance Overhead Tips](#performance-overhead-tips)
  - [Container Profiling Issues](#container-profiling-issues)
  - [WSL Timestamp Issues](#wsl-timestamp-issues)
- [Additional Resources](#additional-resources)

---

## Release Notes

### Nsight Systems 2025.2 Highlights

Nsight Systems 2025.2 introduces significant new features and enhancements across GPU tracing, system tracing, Python profiling, and GUI capabilities.

| Category | Feature | Status |
|---|---|---|
| **Python** | Dask API trace (`--dask`) | New |
| **Python** | PyTorch enhancements | Enhanced |
| **Python** | Python 3.13 support | New |
| **CUDA Trace** | Hardware-based low-overhead CUDA trace for NVIDIA Blackwell GPUs (`--trace=cuda-hw`) | Beta |
| **CUDA Trace** | GPU Direct Storage trace (`--trace=gds`) | New |
| **CUDA Trace** | CUDA device-side event trace (`--cuda-event-trace`) | New |
| **CUDA Trace** | Graph trace improvements | Enhanced |
| **CUDA Trace** | Kernel CGA dimensions and policy added to kernel tooltips | New |
| **CUDA Trace** | Stream priority in Timeline legend tooltips | New |
| **Security** | NVIDIA Confidential Compute support improvements | Enhanced |
| **Windows** | GPU Frame Duration for DLSS Frame Generation | New |
| **Windows** | GPU resource trace tracks pre-start allocation names | Enhanced |
| **Windows** | Graphics Hotspot Analysis recipe | New |
| **Linux** | Syscall trace enhancements | Enhanced |
| **Linux** | Syscall trace reduces requirements to `CAP_BPF` and `CAP_PERFMON` | Enhanced |
| **Linux** | System-wide mode (`--syscall=pid-namespace`) | New |
| **Linux** | Backtrace collection for syscalls | New |
| **Linux** | OS Runtime Trace (OSRT) VFS POSIX functions trace (`--osrt-file-access=true`) | New |
| **Grace CPU** | Topdown analysis recipe for PMU events based on NVTX range annotations | New |
| **Grace CPU** | Updates to available counters and metrics | Enhanced |
| **NVTX** | Payloads Extensions | New |
| **NVTX** | Counters Extensions | New |
| **NVTX** | Deferred Events Extensions | New |
| **Plugins** | Callback for last-chance to submit NVTX deferred events on stop | New |
| **Plugins** | Windows support | New |
| **GUI** | macOS GUI now available for arm64 | New |
| **GUI** | Go to range toolbar for jumping to longest, shortest, and median ranges | New |
| **Streamer** | Nsight Streamer available on NGC for Kubernetes and Docker | New |
| **Operator** | Nsight Operator releasing soon on NGC for Kubernetes | Upcoming |

### New Features by Category

#### CUDA Tracing Enhancements

**Hardware-Based CUDA Trace (Beta)**

The new `--trace=cuda-hw` option enables low-overhead hardware-based CUDA tracing for NVIDIA Blackwell GPUs. This feature uses hardware capabilities to reduce profiling overhead compared to software-based tracing.

```bash
# Enable hardware-based CUDA trace on Blackwell GPUs
nsys profile --trace=cuda-hw -o hw_trace_report ./my_application
```

**GPU Direct Storage Trace**

Trace GPU Direct Storage (GDS) operations for storage I/O profiling:

```bash
nsys profile --trace=gds -o gds_report ./gds_application
```

**CUDA Device-Side Event Trace**

The `--cuda-event-trace` option enables tracing of CUDA events created and recorded on the device side:

```bash
nsys profile --cuda-event-trace --trace=cuda -o cuda_event_report ./my_application
```

**Graph Trace Improvements**

Enhanced CUDA Graph tracing with better visibility into graph instantiation, execution, and node-level details. Kernel tooltips now include CGA (Cooperative Group Array) dimensions and policy information.

#### Python Profiling Enhancements

**Dask API Trace**

New `--dask` flag enables tracing of Dask API calls for distributed Python workloads:

```bash
nsys profile --trace=cuda,nvtx,osrt --dask=true -o dask_report python dask_workflow.py
```

**Python 3.13 Support**

Nsight Systems 2025.2 adds full support for Python 3.13, including:
- Backtrace sampling in Python 3.13 applications
- Python functions trace (`--python-functions-trace`)
- GIL tracing

#### Linux System Trace Enhancements

**Syscall Trace Improvements**

- Reduced capability requirements from full root to `CAP_BPF` and `CAP_PERFMON`
- System-wide mode via `--syscall=pid-namespace`
- Backtrace collection for syscall events

```bash
# System-wide syscall tracing with reduced privileges
nsys profile --syscall=pid-namespace --sample=cpu -o syscall_report ./my_app

# Collect backtraces with syscall trace
nsys profile --trace=syscall --syscall-backtrace=true -o bt_report ./my_app
```

**OS Runtime Trace (OSRT) VFS POSIX Functions**

New option to trace file access operations:

```bash
nsys profile --osrt-file-access=true -o file_access_report ./my_application
```

#### NVIDIA Grace CPU Support

**Topdown Analysis Recipe**

New recipe for PMU events-based topdown analysis on NVIDIA Grace CPUs with NVTX range annotations:

```bash
# Step 1: Collect data
<path>/cpu/collect_grace_topdown.sh ./myApp

# Step 2: Run recipe
nsys recipe nvtx_cpu_topdown --input .
```

The recipe produces a Jupyter notebook with:
- NVTX Summary with range instances and median durations
- Topdown Level 1 metrics (Frontend Bound, Backend Bound, Bad Speculation, Retiring)
- Detailed sub-metrics for each topdown category
- Report summary with PMU core events and CPU core metrics

#### NVTX API Enhancements

**Payloads Extensions**

NVTX Payloads Extensions allow attaching structured data to NVTX ranges for richer annotation:

```c
// Example: Attach structured payload to an NVTX range
nvtxEventAttributes_t attr = {0};
attr.payload.type = NVTX_PAYLOAD_TYPE_INT64;
attr.payload.llValue = tensor_size_bytes;
nvtxDomainRangePushEx(domain, &attr);
```

**Counters Extensions**

NVTX Counters Extensions enable applications to report custom performance counters:

```c
// Example: Report a custom counter
nvtxCounterDataStruct counter = {
    .name = "batch_throughput",
    .value = throughput_items_per_sec,
    .unit = "items/s"
};
nvtxCounterSample(&counter);
```

**Deferred Events Extensions**

NVTX Deferred Events allow postponing event submission until a specific condition is met, reducing overhead when profiling is not active:

```c
// Example: Deferred event submission
nvtxEventAttributes_t deferred_attr = {0};
nvtxDomainRangePushDeferred(domain, &deferred_attr);
```

#### Plugin System Enhancements

**Callback for Deferred NVTX Events**

Plugins can now register a callback that provides a last-chance opportunity to submit NVTX deferred events when collection stops:

```python
# Plugin callback example
def on_stop_callback(context):
    """Called when collection stops - last chance to submit deferred events"""
    for pending_event in context.get_pending_events():
        context.submit_nvtx_event(pending_event)
```

**Windows Support for Plugins**

The Nsight Systems Plugin system is now supported on Windows platforms, in addition to the existing Linux support.

#### GUI Improvements

**Go to Range Toolbar**

A new timeline toolbar allows quick navigation to specific range instances:
- Jump to the longest range instance
- Jump to the shortest range instance
- Jump to the median range instance
- Navigate forward/backward through range instances

**macOS arm64 GUI**

The Nsight Systems GUI is now available for macOS on Apple Silicon (arm64) for viewing and analyzing reports collected on remote systems.

#### Nsight Streamer

Nsight Streamer is now available on NGC for viewing reports on remote headless servers. It supports:
- Kubernetes deployment
- Docker deployment
- On-demand profiling via API
- Continuous monitoring for trend analysis
- Centralized collection from multiple agents

---

## Known Issues

### General Issues

#### Architecture Support Changes

| Issue | Details |
|---|---|
| **Pascal and Volta dropped** | Nsight Systems versions starting with 2025.2 do not support Pascal or Volta architectures. Use an older version downloadable from https://developer.nvidia.com/gameworksdownload. |
| **Power PC dropped** | Nsight Systems versions starting with 2024.2 do not support Power PC. Use an older version. |
| **cuBLAS versions prior to 11.4 dropped** | Nsight Systems versions starting with 2024.4 do not support cuBLAS versions prior to 11.4. |

#### WSL Timestamp Issue

The default time conversion used by Nsight Systems is not reliable on WSL (Windows Subsystem for Linux). A fallback to a safer, but less precise, time system is required.

**Fix:**

```bash
mkdir -p "$(dirname "$(nsys -z)")"
echo 'CuptiUseRawGpuTimestamps=false' >> "$(nsys -z)"
```

This sets the config file option `CuptiUseRawGpuTimestamps` to false. This will be corrected in a future version.

#### Session and Executable Name Length Limits

| Limit | Maximum Length |
|---|---|
| Session name | 127 characters |
| Executable name for `nsys profile` | 111 characters |

These limitations will be removed in a future version.

#### Thread Scheduling Collection Overhead

Nsight Systems 2020.4 introduced collection of thread scheduling information without full sampling. While this allows system information at a lower cost, it does add overhead.

**Fix:** Turn off thread schedule information collection:

```bash
nsys profile --cpuctxsw=none ./my_application
```

Or disable it in the GUI settings.

#### Profiling Duration Limitations

Profiling greater than 5 minutes is not officially supported. Long profiling sessions with high activity applications can:
- Create very large result files
- Take a very long time to load
- Run out of memory
- Lock up the system

**Recommendation:** Start with short sessions of no more than 5 minutes. If your application has a repeating pattern (frame, iteration), you typically only need a few of these.

#### Attach/Re-attach from GUI

Attaching or re-attaching to a process from the GUI is not supported with the x86_64 Linux target.

**Workaround:** Use the interactive CLI to launch the process and then start/stop analysis at multiple points:

```bash
nsys launch ./my_application
nsys start
nsys stop
nsys quit
```

#### API Trace Subset

To reduce overhead, Nsight Systems traces a subset of API calls rather than all possible calls. There is currently no way to change the subset being traced when using the CLI.

#### Trace Event Buffer Size Limit

There is an upper bound on the default size used by the tool to record trace events during collection. If you see the following diagnostic error, Nsight Systems hit the upper limit:

```
Reached the size limit on recording trace events for this process.
       Try reducing the profiling duration or reduce the number of features
       traced.
```

**Workaround:** Reduce profiling duration, reduce traced features, or increase buffer size:

```bash
nsys profile --buffer-size=2G -o report ./my_application
```

#### CUPTI Conflicts

When profiling a framework or application that uses CUPTI (like some versions of TensorFlow), Nsight Systems will not be able to trace CUDA usage due to CUPTI limitations.

**Workaround:** Turn off the application's use of CUPTI if CUDA tracing is required.

#### Thread Safety Requirement

Tracing an application that uses a memory allocator that is not thread-safe is not supported.

#### glibc Symbol Preloading

Tracing OS Runtime libraries in an application that preloads glibc symbols is unsupported and can lead to undefined behavior.

#### Virtual Window Managers

Nsight Systems cannot profile applications launched through a virtual window manager like GNU Screen.

#### MPI and Darshan Module Conflict

Using Nsight Systems MPI trace functionality with the Darshan runtime module can lead to segfaults.

**Fix:**

```bash
module unload darshan-runtime
```

#### MPI Fortran API Memory Corruption

Profiling MPI Fortran APIs with `MPI_Status` as an argument (e.g., `MPI_Recv`, `MPI_Test[all]`, `MPI_Wait[all]`) can cause memory corruption for MPICH versions 3.0.x due to different `MPI_Status` structure layouts.

**Affected:** MPICH 3.0.x only. Versions 2.1.x and >=3.1.x are not affected.

#### SQLite Export File Locking

Using `nsys export` to export to an SQLite database will fail if the destination filesystem does not support file locking:

```
std::exception::what: database is locked
```

**Workaround:** Export to a filesystem that supports file locking (not NFS with certain configurations, or use `--force`).

#### VNC Rendering Issues

On some Linux systems when VNC is used, some widgets can be rendered incorrectly, or Nsight Systems can crash when opening Analysis Summary or Diagnostics Summary pages.

**Fix:** Force a specific software renderer:

```bash
GALLIUM_DRIVER=llvmpipe nsys-ui
```

#### Open MPI 4.0.1 Bug

Due to a known bug in Open MPI 4.0.1, the target application may crash at the end of execution when being profiled by Nsight Systems.

**Fix:** Use a different Open MPI version, or add the following option:

```bash
mpirun --mca btl ^vader -np 4 ./my_application
```

#### Python Multiprocessing Fork Mode

The Python multiprocessing module defaults to using "fork" mode on Linux. According to POSIX, fork without exec leads to undefined behavior, making it difficult for tools like Nsight Systems to collect profiling information.

**Fix:** Use the `spawn` start method:

```python
import multiprocessing as mp

if __name__ == '__main__':
    mp.set_start_method('spawn')
    # ... rest of the application
```

Reference: https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods

#### Process Exit Gracefulness

Users must ensure processes exit gracefully (using `close` and `join` methods for multiprocessing objects). Otherwise, Nsight Systems cannot flush buffers properly and traces may be missing.

#### LinuxPerf DFS Race Condition

When using the CLI sequence launch, start, stop to profile a process-tree, LinuxPerf does a depth-first search (DFS) to find all threads. If threads are created during the DFS, they may not be found.

**Note:** Once a thread is programmed via `perf_event_open`, any subsequent children processes or threads will be tracked (inherit bit is set). System-wide mode does not suffer from this issue.

### vGPU Issues

#### Profiler Grant Required

When running Nsight Systems on vGPU, you must use the profiler grant. Without it:
- Unexpected migrations may crash a running session
- May report an error and abort
- May silently produce a corrupted report with inaccurate data

**Action:** See Virtual GPU Software Documentation for details on enabling CUDA Toolkit profilers for vGPUs.

#### Device-Level Metrics on vGPU

Starting with vGPU 13.0, device-level metrics collection is exposed to end users. Note that device-level metrics provide info about all work being executed on the GPU, including work from other VMs running on the same physical GPU.

#### License Timing

As of CUDA 11.4 and R470 TRD1 driver release, Nsight Systems is supported in vGPU environments which require a vGPU license. If the license is not obtained after 20 minutes, the tool will still work but reported GPU performance metrics data will be inaccurate due to a performance reduction feature in vGPU.

### Docker Issues

#### Kernel Version Requirement

In a Docker container, when the system host utilizes a kernel older than v4.3, it is not possible for Nsight Systems to collect sampling data unless both the host and Docker are running RHEL or CentOS with kernel version 3.10.1-693 or newer.

#### docker exec Shell Hang

When `docker exec` is called on a running container and stdout is kept open from a command inside that shell, the exec shell hangs until the command exits.

**Fix:** Run with `--tty`:

```bash
docker exec --tty container_id nsys profile -o /tmp/report ./my_app
```

Related bug reports:
- [moby/moby#33039](https://github.com/moby/moby/issues/33039)
- [drud/ddev#732](https://github.com/drud/ddev/issues/732)

### CUDA Trace Issues

#### CC-DevTools Mode Crash with libcrypto

If a system is in CC-DevTools mode (Confidential Compute) and Nsight Systems traces CUDA in an application using libcrypto, Nsight Systems may crash when the application exits. The crash causes profiler data loss.

**Workarounds (in order of preference):**

1. Add `cudaDeviceSynchronize()` immediately before application exit
2. Add `cudaProfilerStop()` before exit and use `--flush-on-cudaprofilerstop=true`:
   ```bash
   nsys profile --flush-on-cudaprofilerstop=true -o report ./my_app
   ```
3. End the profile before application exit using one of:
   - Set a duration: `--duration=10`
   - Use a capture range: `--capture-range`
   - Set CUDA flush interval for frequent flushes
   - Use CLI start/launch/stop commands

#### UVM Page Migration Stream Info

The `cudaMemPrefetchAsync()` API allows specifying a stream, but Nsight Systems does not receive stream information for UVM page migrations from the UVM backend. Stream information cannot be correctly correlated with `cudaMemPrefetchAsync()` calls.

#### CUDA Toolkit 10.X DtoD Copy Crash

When using CUDA Toolkit 10.X, tracing of device-to-device (DtoD) memory copy operations may result in a crash.

**Fix:** Update CUDA Toolkit to 11.X or later.

#### CDP Kernel Tracing

Nsight Systems will not trace kernels when a CUDA Dynamic Parallelism (CDP) kernel is found in a target application on Volta devices or later.

#### Tegra Root Privileges

On Tegra platforms, CUDA trace requires root privileges. Use the "Launch as root" checkbox in project settings.

#### Multi-Stream Multi-Thread Buffer Issue

If the target application uses multiple streams from multiple threads, CUDA event buffers may not be released properly. The following diagnostic error appears:

```
Couldn't allocate CUPTI bufer x times. Some CUPTI events may
       be missing.
Please contact the Nsight Systems team.
```

#### CUDA Memory Allocation Graph Limitation

When starting and stopping profiling inside an application using the interactive CLI, CUDA memory allocation graph generation is only guaranteed to be correct in the first profiling range.

#### GPU Memory Requirement

CUDA GPU trace collection requires a fraction of GPU memory. If the application utilizes all available GPU memory, CUDA trace might not work or can break the application.

**Example:** cuDNN can crash with `CUDNN_STATUS_INTERNAL_ERROR` if GPU memory allocation fails.

**Workaround:** Reserve GPU memory for profiling by pre-allocating slightly less memory in your application.

#### Short-Lived Applications on Older Kernels

For Linux kernels prior to 4.4, when profiling very short-lived applications (~1 second) that exit in the middle of the profiling session, Nsight Systems may not show CUDA events on the timeline.

#### Serialized Kernel Event Order

When more than 64k serialized CUDA kernels and memory copies are executed:

```
InvalidArgumentException: "Wrong event order detected"
```

**Fix:** Upgrade to the CUDA 9.2 driver at minimum. If unable to upgrade, use the CLI for partial analysis (may miss a large fraction of CUDA events).

#### Vibrante NAT Configuration

When running a profiling session with multiple targets that are guest VMs in a CCC configuration behind a NAT:

```
Failed to sync time on device.
```

**Fix:** Edit group connection settings, select "Targets on the same SoC" checkbox, and retry.

#### Driver 455 Crash

When using the 455 driver (shipped with CUDA Toolkit 11.1), tracing CUDA with Nsight Systems may cause a crash when the application exits.

**Fix:** End the profiling session before the application exits, or update the driver.

### Multi-Report Analysis Issues

Setting up Dask analysis on a workstation requires additional system configuration. For small data inputs, running the recipes without Dask may be faster.

```bash
# Without Dask (for small datasets)
nsys analyze --recipe=kernel-comparison report1.nsys-rep report2.nsys-rep

# With Dask (for large datasets)
nsys analyze --dask --recipe=kernel-comparison report*.nsys-rep
```

---

## Troubleshooting

### General Troubleshooting

If the profiler behaves unexpectedly during the profiling session, or the profiling session fails to start, follow these steps:

1. **Close the host application.**
2. **Restart the target device.**
3. **Start the host application and connect to the target device.**

Nsight Systems uses a settings file (`NVIDIA Nsight Systems.ini`) on the host to store information about:
- Loaded projects
- Report files
- Window layout configuration

The location of the settings file is described in the Help > About dialog. Deleting the settings file restores Nsight Systems to a fresh state, but all projects and reports will disappear from the Project Explorer.

### CLI Troubleshooting

#### .nsys-rep File Will Not Load

If you collected a report file using the CLI and the report will not open in the GUI:

**Cause:** Your GUI version is older than the CLI version used to collect the report.

**Fix:** Download a new version of the Nsight Systems GUI. The GUI version must be the same or greater than the CLI version.

This situation occurs most frequently when updating Nsight Systems using a CLI-only package, such as the package available from the NVIDIA HPC SDK.

#### .nsys-rep File Not Generated

The CLI initially generates a `.qdstrm` file (intermediate result file, not intended for multiple imports). It needs to be processed into a `.nsys-rep` file. Usually this happens automatically. If it does not, use the standalone QdstrmImporter utility.

**Requirements:**
- The CLI and QdstrmImporter versions must match
- The resulting `.nsys-rep` file can be opened in the same version or more recent versions of the GUI

**On the host system:** Find the QdstrmImporter binary in the `Host-x86_64` directory in your installation. QdstrmImporter is available for all host platforms.

**On the target system:** Copy the Linux `Host-x86_64` directory to the target Linux system or install Nsight Systems for Linux host directly on the target. The Windows or macOS host QdstrmImporter will not work on a Linux target.

**QdstrmImporter Options:**

| Short | Long | Parameter | Description |
|---|---|---|---|
| `-h` | `--help` | | Help message providing information about available options |
| `-v` | `--version` | | Output QdstrmImporter version information |
| `-i` | `--input-file` | filename or path | Import .qdstrm file from this location |
| `-o` | `--output-file` | filename or path | Provide a different file name or path for the resulting .nsys-rep file. Default is the same name and path as the .qdstrm file |

**Example:**

```bash
# Convert a .qdstrm file to .nsys-rep
QdstrmImporter -i report.qdstrm -o report.nsys-rep
```

### GUI Troubleshooting

#### Empty or Black Pages in Analysis or Diagnostics Summary

If the Analysis Summary or Diagnostics Summary pages appear empty or black, this may be caused by rendering issues related to OpenGL or Vulkan drivers.

**Fix:** Run Nsight Systems with the following command:

```bash
QTWEBENGINE_CHROMIUM_FLAGS="--no-sandbox" \
QMLSCENE_DEVICE=softwarecontext \
[installation_path]/host-linux-[arch]/nsys-ui
```

#### Missing xcb-cursor Package

If you encounter the following error, you are missing the required xcb-cursor package:

```
qt.qpa.plugin: From 6.5.0, xcb-cursor0 or libxcb-cursor0 is needed to
load the Qt xcb platform plugin.
```

This issue typically occurs on RHEL but may also affect other distributions.

**Fix by OS:**

| Distribution | Command |
|---|---|
| RHEL / CentOS / Fedora | `sudo dnf install -y xcb-util-cursor` |
| OpenSUSE | `sudo zypper install -y xcb-util-cursor` |
| Debian / Ubuntu | `sudo apt-get install -y libxcb-cursor0` |

#### Missing Qt Libraries

If opening the Nsight Systems Linux GUI fails with either:

```
This application failed to start because it could not find or load the
Qt platform plugin "xcb" in "". Available platform plugins are: xcb.
Reinstalling the application may fix this problem.
```

or:

```
error while loading shared libraries: [library_name]: cannot open shared
object file: No such file or directory
```

**With root privileges (Ubuntu 18.04/20.04/22.04, CentOS 7/8/9):**

```bash
[installation_path]/host-linux-[arch]/Scripts/DependenciesInstaller/install-dependencies.sh
```

Then launch the Linux GUI as usual.

**Without root privileges (Ubuntu 18.04/20.04/22.04, CentOS 7/8/9):**

```bash
# Install dependencies to a user-writable directory
[installation_path]/host-linux-[arch]/Scripts/DependenciesInstaller/install-dependencies-without-root.sh [dependencies_path]

# Launch with environment set up
source [installation_path]/host-linux-[arch]/Scripts/DependenciesInstaller/setup-dependencies-environment.sh [dependencies_path] && \
[installation_path]/host-linux-[arch]/nsys-ui
```

**For other platforms or if the above does not help:**

```bash
# Determine which libraries are missing
QT_DEBUG_PLUGINS=1 [installation_path]/host-linux-[arch]/nsys-ui
```

If the workload does not run when launched via Nsight Systems or the timeline is empty, check the `stderr.log` and `stdout.log` files (click on the drop-down menu showing "Timeline View" and click on "Files") to see errors encountered by the app.

### Symbol Resolution

If stack trace information is missing symbols and you have a symbol file, you can manually re-resolve using the ResolveSymbols utility. This can be done by:
- Right-clicking the report file in the Project Explorer window and selecting "Resolve Symbols..."
- Using the ResolveSymbols executable found in the `[installation_path]\Host` directory

The utility works with:
- ELF format files
- Windows PDB directories and symbol servers
- Text files with format: `<start> <length> <name>`

**ResolveSymbols Options:**

| Short | Long | Argument | Description |
|---|---|---|---|
| `-h` | `--help` | | Help message |
| `-l` | `--process-list` | | Print global process IDs list |
| `-s` | `--sym-file` | filename | Path to symbol file |
| `-b` | `--base-addr` | address | If set, `<start>` in symbol file is treated as relative address starting from this base address |
| `-p` | `--global-pid` | pid | Which process in the report should be resolved. May be omitted if there is only one process. |
| `-f` | `--force` | | Force use of a given symbol file |
| `-i` | `--report` | filename | Path to the report with unresolved symbols |
| `-o` | `--output` | filename | Path and name of the output file. If omitted, "resolved" suffix is added to the original filename. |
| `-d` | `--directories` | directory paths | List of symbol folder paths, separated by semi-colon characters. Windows only. |
| `-v` | `--servers` | server URLs | List of symbol servers using the same format as `_NT_SYMBOL_PATH` environment variable, i.e., `srv*<LocalStore>*<SymbolServerURL>`. Windows only. |
| `-n` | `--ignore-nt-sym-path` | | Ignore symbol locations stored in the `_NT_SYMBOL_PATH` environment variable. Windows only. |

**Example:**

```bash
# Resolve symbols for a specific process
ResolveSymbols -i report.nsys-rep -s /path/to/symbols.sym -p 12345 -o resolved_report.nsys-rep

# Use a base address with a relative symbol file
ResolveSymbols -i report.nsys-rep -s relative_symbols.sym -b 0x400000 -o resolved_report.nsys-rep
```

### Logging and Diagnostics

#### Host Logging Configuration

To enable logging on the host, refer to the config file template:

```
host-linux-x64/nvlog.config.template
```

When reporting bugs, include:
- Build version number (from Help > About dialog)
- Log files
- Report (.nsys-rep) files

#### Verbose Remote Logging on Linux Targets

Verbose logging is available when connecting to a Linux-based device from the GUI on the host. This extra debug information is not available when launching via the CLI.

Nsight Systems installs executable and library files into:
```
/opt/nvidia/nsight_systems/
```

**To enable verbose logging on the target device (launched from host):**

1. Close the host application.
2. Restart the target device.
3. Place `nvlog.config` from the host directory to `/opt/nvidia/nsight_systems/` on the target.
4. From SSH console, launch:
   ```bash
   sudo /opt/nvidia/nsight_systems/nsys --daemon --debug
   ```
5. Start the host application and connect to the target device.

Logs are collected into: `nsys.log` in the directory where `nsys` was launched.

**Warning:** Debug logging can significantly slow down the profiler.

#### Verbose CLI Logging on Linux Targets

To enable verbose logging of the Nsight Systems CLI and the target application's injection behavior:

1. In the `target-linux-x64` directory, rename `nvlog.config.template` to `nvlog.config`.
2. Inside that file, change the line:
   ```
   $ nsys-ui.log
   ```
   to:
   ```
   $ nsys-agent.log
   ```
3. Run a collection. The `target-linux-x64` directory should include `nsys-agent.log`.

**Warning:** Debug logging can significantly slow down the profiler.

#### Verbose Logging on Windows Targets

Verbose logging is available when connecting to a Windows-based device from the GUI on the host. Nsight Systems installs files to:

```
C:\Program Files\NVIDIA Corporation\Nsight Systems 2023.3
```

**To enable verbose logging on a Windows target:**

1. Close the host application.
2. Terminate the `nsys` process.
3. Place `nvlog.config` from the host directory next to the Nsight Systems Windows agent:
   - **Local Windows target:**
     ```
     C:\Program Files\NVIDIA Corporation\Nsight Systems 2023.3\target-windows-x64
     ```
   - **Remote Windows target:**
     ```
     C:\Users\<user name>\AppData\Local\Temp\nvidia\nsight_systems
     ```
4. Start the host application and connect to the target device.

Logs are collected into: `nsight-sys.log` in the same directory as the Nsight Systems Windows agent.

**Warning:** Debug logging can significantly slow down the profiler.

### Environment Variables

#### TMPDIR - Temporary File Location

By default, Nsight Systems writes temporary files to `/tmp`. If your system does not allow writing to `/tmp` or has limited storage there, use the `TMPDIR` environment variable:

```bash
TMPDIR=/testdata ./bin/nsys profile -t cuda matrixMul
```

#### Windows Environment Variable Workaround

Environment variable control for Windows target trace is not directly available. Use this workaround:

1. Create a batch file that sets the environment variables and launches your application:
   ```batch
   @echo off
   set MY_VAR=value
   set ANOTHER_VAR=another_value
   my_application.exe
   ```

2. Set Nsight Systems to launch the batch file as its target (set the project settings target path to the batch file path).

3. Start the trace. Nsight Systems will launch the batch file in a new `cmd` instance and trace the whole process tree whose root is the `cmd` running your batch file.

### Profiling Games

In launcher-based platforms (like Steam), if you attempt to run the game executable directly from Nsight Systems, the game will detect that the launcher is missing. It will launch the launcher and then self-terminate.

Nsight Systems on Windows automatically attaches to child processes spawned by the target process.

**Workflow:**

1. Verify the Steam client is not running. Select Quit to terminate Steam if running.
2. Configure Nsight Systems to launch the Steam client with a manual collection option. Check the hotkey checkbox to begin data collection from within the game.
3. Click Start. Nsight Systems will launch the Steam client.
4. Use the Steam GUI to launch the game.
5. When you have reached the scene you want to profile, press **F12** to start data collection.
6. Let the game continue running while Nsight Systems collects profiling data (typically 10-60 seconds).
7. Press **F12** again to end the collection.

### WebGL Testing

Nsight Systems cannot profile using the default Chrome launch command. To profile WebGL, use the following command structure:

```bash
"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" \
    --inprocess-gpu \
    --no-sandbox \
    --disable-gpu-watchdog \
    --use-angle=gl \
    https://webglsamples.org/aquarium/aquarium.html
```

Required Chrome flags:

| Flag | Purpose |
|---|---|
| `--inprocess-gpu` | Run GPU process in the main process for profiling |
| `--no-sandbox` | Disable sandbox for tool injection |
| `--disable-gpu-watchdog` | Prevent watchdog from killing GPU during profiling |
| `--use-angle=gl` | Force OpenGL backend |

### QNX Target Issues

When working with QNX targets, ensure:

| Requirement | Details |
|---|---|
| tracelogger utility | Must be available and runnable on the target |
| /tmp directory | Must be accessible and support sub-directories |
| Version switching | Kill all processes related to the previous version before using a new version. Reboot the target if issues persist after switching versions. |

### Launch Processes in Stopped State

In many cases, it is important to profile an application from the very beginning of its execution. When launching processes, Nsight Systems ensures the profiling session is fully initialized before making the `exec()` system call on Linux.

If the process launch capabilities are not sufficient, the application should be launched manually, and the profiler configured to attach to the already launched process.

Two mechanisms can be used on Linux without recompiling the application. Both ensure that between the time the process is created (PID is known) and the time application code is called, the process is stopped and waits for a signal.

#### Method 1: LD_PRELOAD

This mechanism uses the `LD_PRELOAD` environment variable. It only works with dynamically linked binaries (static binaries do not invoke the runtime linker).

**Library paths by platform:**

| Platform | Library Path |
|---|---|
| ARMv7 binaries | `/opt/nvidia/nsight_systems/libLauncher32.so` |
| Running from host (other) | `/opt/nvidia/nsight_systems/libLauncher64.so` |
| Running from CLI | `[installation_directory]/libLauncher64.so` |

**Usage:**

```bash
LD_PRELOAD=/opt/nvidia/nsight_systems/libLauncher64.so ./my-aarch64-binary --arguments
```

When loaded, this library sends itself a `SIGSTOP` signal (equivalent to Ctrl+Z). The process becomes a background job controllable with `jobs`, `fg`, and `bg`. Use `jobs -l` to see the PID.

When attaching to a stopped process, Nsight Systems sends `SIGCONT` (equivalent to `bg`).

#### Method 2: Launcher Utility

The launcher mechanism can be used with any binary (including statically linked):

```bash
# Launch the application in stopped state
/opt/nvidia/nsight_systems/launcher ./my-binary --arguments
```

The process will be launched, daemonized, and wait for `SIGUSR1` signal. After attaching with Nsight Systems, manually resume:

```bash
# Resume the process
pkill -USR1 launcher
```

**Note:** `pkill` sends the signal to any process with the matching name. Use `kill` to send to a specific PID if multiple launcher processes exist.

Standard output and error are redirected to:
- `/tmp/stdout_<PID>.txt`
- `/tmp/stderr_<PID>.txt`

The launcher mechanism is more complex and less automated than `LD_PRELOAD`, but gives more control to the user.

### Broken Backtraces on Tegra

In Nsight Systems Embedded Platforms Edition, the symbols table has a special entry called "Broken backtraces." This denotes the point in the call chain where the unwinding algorithms could not determine the next (caller) function.

Broken backtraces happen because there is no unwind information available for the current function. In the Top-Down view, functions with broken backtraces are immediate children of the "Broken backtraces" row.

**To eliminate broken backtraces**, modify the build system to provide at least one kind of unwind information:

#### For ARMv7 Binaries

| Unwind Information Type | ELF Sections | Compiler Flag |
|---|---|---|
| DWARF debug info | `.debug_frame`, `.zdebug_frame`, `.eh_frame`, `.eh_frame_hdr` | `-g` |
| Exception handling (EHABI) | `.ARM.exidx`, `.ARM.extab` | `-funwind-tables` |
| Frame pointers | Built into `.text` section | `-fno-omit-frame-pointer` |

#### For AArch64 Binaries

| Unwind Information Type | ELF Sections | Compiler Flag |
|---|---|---|
| DWARF debug info | `.debug_frame`, `.zdebug_frame`, `.eh_frame`, `.eh_frame_hdr` | `-g` |
| Frame pointers | Built into `.text` section | `-fno-omit-frame-pointer` |

**Notes:**
- The following ELF sections should be considered empty if they have a size of 4 bytes: `.debug_frame`, `.eh_frame`, `.ARM.exidx`. These only contain termination records and no useful information.
- EHABI and DWARF information is compiled per-unit (every .cpp/.c file and static library can be built with or without it). Presence of ELF sections does not guarantee every function has unwind information.
- Frame pointers are required by the AArch64 Procedure Call Standard. The performance impact is usually negligible.
- To check default compiler flags:
  ```bash
  gcc -Q --help=common
  ```
- To see actual compiler flags being used:
  ```bash
  gcc -### [compilation options]
  ```

### Debug Versions of ELF Files

After building a binary with debug information (`-g`), it often gets stripped before deployment. ELF sections containing useful information (non-export function names, unwind information) can get stripped as well.

**Solutions:**

1. **Deploy unstripped library:** Install the original unstripped library instead of the stripped one.

2. **Debug symbol packages (Ubuntu):** For target devices with Ubuntu, use debug symbol packages. These install debug ELF files with the `/usr/lib/debug` prefix. Nsight Systems can find and use matching debug libraries.

   Many packages have debug companions installable with APT:
   ```bash
   # Packages with -dbg suffix
   apt-get install <package>-dbg

   # Packages with -dbgsym suffix (requires debug repo setup)
   apt-get install <package>-dbgsym
   ```

3. **Verify debug library usage:** To verify that a debug version of a library has been picked up, check the Module Summary section of Analysis Summary for "Debug library has been used."

### Common Error Messages and Solutions

#### Error: "Failed to initialize CUPTI"

```
Error: Failed to initialize CUPTI. CUPTI activity tracing is not available.
```

| Cause | Solution |
|---|---|
| Incompatible CUDA driver version | Update NVIDIA driver to match CUDA toolkit version (R515+) |
| CUPTI library not found | Ensure CUDA toolkit is installed and `LD_LIBRARY_PATH` includes `extras/CUPTI/lib64` |
| CUPTI permissions issue | Ensure the user has permissions to access CUPTI device files |
| GPU in exclusive mode | Check `nvidia-smi` compute mode and set to DEFAULT if needed |

```bash
# Verify CUPTI is available
ls /usr/local/cuda/extras/CUPTI/lib64/libcupti.so

# Set LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# Check GPU compute mode
nvidia-smi --query-gpu=compute_mode --format=csv
nvidia-smi -c EXCLUSIVE_PROCESS  # Set if needed
```

#### Error: "Permission denied for perf_event_open"

```
Error: perf_event_open() failed: Operation not permitted
```

| Cause | Solution |
|---|---|
| `perf_event_paranoid` too high | Set to 0 or -1: `sudo sh -c 'echo 0 > /proc/sys/kernel/perf_event_paranoid'` |
| Seccomp blocking in container | Use `--privileged` or custom seccomp profile allowing `perf_event_open` |
| Missing capabilities | Add `CAP_SYS_ADMIN` capability |

#### Error: "Trace buffer overflow"

```
Warning: Trace buffer overflow detected. Some events may be missing.
```

| Cause | Solution |
|---|---|
| Too many events for buffer size | Increase buffer: `--buffer-size=512` (in MB) |
| Very high event rate | Reduce trace scope: `--trace=cuda` only (exclude `osrt` if not needed) |
| Long profiling duration | Shorten the profiling window or use NVTX to focus on specific code |

```bash
# Increase buffer size to 512 MB
nsys profile --buffer-size=512 my_application

# Use NVTX to limit the traced region
nsys profile -c cudaProfilerApi my_application
# (Call cudaProfilerStart/Stop in application code)
```

#### Error: "Cannot connect to target"

```
Error: Failed to connect to target host: Connection refused
```

| Cause | Solution |
|---|---|
| SSH not running on target | Start SSH daemon: `sudo systemctl start sshd` |
| Firewall blocking connection | Open required ports (default SSH port 22) |
| Target hostname incorrect | Verify hostname/IP address and network connectivity |
| SSH key authentication failure | Verify key file path and permissions (chmod 600) |

#### Error: "Symbol resolution failed"

```
Warning: Could not resolve symbols for: /path/to/library.so
```

| Cause | Solution |
|---|---|
| Debug symbols not installed | Install debug packages: `sudo apt install library-dbgsym` |
| Stripped binary | Use unstripped version during profiling |
| Symbol path not configured | Add paths in GUI preferences or CLI `--symbol-path` |

```bash
# Add symbol search paths
nsys profile --symbol-path=/path/to/symbols my_application

# Check if binary has symbols
file my_binary
nm my_binary | head
readelf -S my_binary | grep debug
```

#### Error: "CUDA context not found"

```
Warning: No CUDA context was captured during profiling.
```

| Cause | Solution |
|---|---|
| Application creates context after profiling window | Start profiling before context creation |
| CUDA tracing not enabled | Add `--trace=cuda` option |
| Application does not use CUDA | Verify the application uses CUDA and GPU is visible |

### Performance Overhead Tips

#### Minimizing Profiling Overhead

1. **Reduce trace scope**: Only trace the APIs you need.
   ```bash
   # Instead of tracing everything
   nsys profile --trace=cuda,nvtx --sample=cpu my_app

   # Trace only CUDA if CPU profiling is not needed
   nsys profile --trace=cuda my_app
   ```

2. **Use NVTX ranges**: Focus profiling on specific code sections.
   ```bash
   nsys profile -c nvtx -e my_app  # Wait for NVTX start/stop
   ```

3. **Reduce sampling frequency**: Lower frequency means less overhead.
   ```bash
   nsys profile --sample=cpu --sampling-frequency=100 my_app  # 100 Hz instead of 1000
   ```

4. **Increase OS runtime threshold**: Skip very short OS events.
   ```bash
   nsys profile --osrt-threshold=10000 my_app  # Only events > 10 us
   ```

5. **Limit profiling duration**: Shorter traces have less overhead.
   ```bash
   nsys profile --duration=10 my_app  # Profile for 10 seconds
   ```

#### Overhead by Profiling Mode

| Configuration | Approximate Overhead | Best For |
|---|---|---|
| `--trace=cuda` only | < 3% | GPU-bound analysis |
| `--trace=cuda --sample=cpu` | 5-10% | General profiling |
| `--trace=cuda,osrt --sample=cpu` | 10-20% | Comprehensive CPU+GPU |
| `--trace=cuda,nvtx,osrt --sample=cpu --python-sampling` | 15-30% | Full stack Python+CUDA |
| All options enabled | 30-100%+ | Debugging only |

### Container Profiling Issues

#### Issue: "Cannot profile inside container"

**Diagnosis**:

```bash
# Check if GPU is visible
nvidia-smi

# Check perf_event_paranoid (from host)
cat /proc/sys/kernel/perf_event_paranoid

# Check capabilities
capsh --print | grep perf
```

**Solutions**:

```bash
# Run with full privileges
docker run --privileged --gpus all my_image nsys profile my_app

# Or minimal capabilities
docker run --cap-add=SYS_PTRACE --cap-add=SYS_ADMIN \
    --security-opt seccomp=unconfined \
    --gpus all my_image nsys profile my_app
```

#### Issue: "Report file lost after container exits"

**Solution**: Always mount a volume for output.

```bash
docker run --privileged --gpus all \
    -v $(pwd)/nsys_reports:/reports \
    my_image nsys profile -o /reports/report my_app
```

#### Issue: "nsys binary not found in container"

**Solution**: Install Nsight Systems in the Dockerfile.

```dockerfile
FROM nvidia/cuda:12.0.0-runtime-ubuntu22.04

# Install Nsight Systems CLI
RUN apt-get update && apt-get install -y wget && \
    wget -q https://developer.download.nvidia.com/devtools/nsight-systems/2025_2/NsightSystemsLinux-public-2025.2.deb && \
    dpkg -i NsightSystemsLinux-public-2025.2.deb && \
    rm NsightSystemsLinux-public-2025.2.deb

ENTRYPOINT ["nsys", "profile"]
```

### WSL Timestamp Issues

Windows Subsystem for Linux (WSL) has known timestamp-related issues.

#### Issue: Timestamps appear incorrect or inconsistent

**Cause**: WSL1 uses software-emulated timers that may not be synchronized with hardware clocks. WSL2 uses a real Linux kernel but may have timer drift.

**WSL2 Solutions**:

```bash
# Verify WSL version
wsl --list --verbose

# Update WSL
wsl --update

# Use WSL2 (recommended for profiling)
wsl --set-version Ubuntu 2
```

#### Issue: GPU timestamps not synchronized with CPU timestamps

**Cause**: TSC (Time Stamp Counter) offset between WSL guest and Windows host.

**Workaround**: Use the `--clock-profile` option to specify the clock domain:

```bash
nsys profile --clock-profile=monotonic my_application
```

#### Issue: "CUPTI not available in WSL"

**Cause**: CUPTI requires NVIDIA driver support for WSL GPU profiling.

**Solution**:

1. Ensure you are using WSL2 (not WSL1).
2. Install the latest NVIDIA driver on Windows (not inside WSL).
3. Verify GPU access: `nvidia-smi` should work inside WSL.

```bash
# Inside WSL2
nvidia-smi
# Should show GPU without installing driver inside WSL

# Install CUDA toolkit (without driver)
sudo apt install cuda-toolkit-12-0
```

#### WSL Profiling Limitations

| Feature | WSL2 Support |
|---|---|
| CUDA API tracing | Supported |
| GPU kernel tracing | Supported |
| CPU sampling | Partial (may have gaps) |
| Context switch tracing | Limited |
| Hardware counters | Not supported |
| GPU metrics | Partial |
| PCIe metrics | Not supported |

### Additional Troubleshooting Tips

#### Checking Profiling Support

```bash
# Verify CUDA and driver compatibility
nvidia-smi
nvcc --version

# Check kernel version for CPU profiling support
uname -r

# Check perf_event_paranoid
cat /proc/sys/kernel/perf_event_paranoid

# Verify CUPTI availability
ls /usr/local/cuda/extras/CUPTI/lib64/

# Check Nsight Systems version
nsys --version
```

#### Report File Validation

```bash
# Validate a report file
nsys stats --report summary report.nsys-rep

# Check report file info
nsys info report.nsys-rep

# Export and inspect schema
nsys export -t sqlite -o check.sqlite report.nsys-rep
sqlite3 check.sqlite "SELECT * FROM META_DATA;"
```

#### Cleaning Up Stale Files

```bash
# Remove temporary nsys files
rm -f /tmp/.nsys-*

# Remove old report files
find /tmp -name "*.nsys-rep" -mtime +7 -delete

# Clean nsys session files
rm -rf ~/.nv/nsight-systems/
```

---

## Additional Resources

### Training Seminars

| Resource | Description |
|---|---|
| NVIDIA Deep Learning Institute | Self-Paced Online Course: Optimizing CUDA Machine Learning Codes With Nsight Profiling Tools |
| CUDA Developer Tools YouTube | Intro to NVIDIA Nsight Systems |
| NCSA Blue Waters Webinar (2018) | Introduction to NVIDIA Nsight Systems |

### Blog Posts

| Year | Title |
|---|---|
| 2021 | Optimizing DX12 Resource Uploads to the GPU Using CPU-Visible VRAM |
| 2020 | Understanding the Visualization of Overhead and Latency in Nsight Systems |
| 2019 | Migrating to NVIDIA Nsight Tools from NVVP and nvprof |
| 2019 | Transitioning to Nsight Systems from NVIDIA Visual Profiler / nvprof |
| 2019 | NVIDIA Nsight Systems Add Vulkan Support |
| 2019 | TensorFlow Performance Logging Plugin nvtx-plugins-tf Goes Public |

### Feature Videos

- OpenMP Trace Feature Spotlight
- Command Line Sessions Video Spotlight
- Direct3D11 Feature Spotlight
- Vulkan Trace
- Statistics Driven Profiling
- Analyzing NCCL Usage with NVIDIA Nsight Systems

### Conference Presentations

| Year / Event | Title |
|---|---|
| GTC 2024 | Achieving Higher Performance From Your Data Center and Cloud Application |
| Jetson Edge AI Developer Days 2023 | Getting the Most Out of Your Jetson Orin Using NVIDIA Nsight Developer Tools |
| GTC 2023 | Optimizing at Scale: Investigating Hidden Bottlenecks for Multi-Node Workloads |
| GTC 2023 | Optimize Multi-Node System Workloads With NVIDIA Nsight Systems |
| GTC 2023 | Ray-Tracing Development using NVIDIA Nsight Graphics and Nsight Systems |
| GTC 2022 | Killing Cloud Monsters Has Never Been Smoother |
| GTC 2022 | Optimizing Communication with Nsight Systems Network Profiling |
| GTC 2022 | Optimizing Vulkan 1.3 Applications with Nsight Graphics & Nsight Systems |
| GTC 2021 | Tuning GPU Network and Memory Usage in Apache Spark |
| GTC 2020 | Rebalancing the Load: Profile-Guided Optimization of NAMD for Modern GPUs |
| GTC 2020 | Scaling the Transformer Model Implementation in PyTorch Across Multiple Nodes |
| GTC 2019 | Using Nsight Tools to Optimize the NAMD Molecular Dynamics Simulation Program |
| GTC 2019 | Optimizing Facebook AI Workloads for NVIDIA GPUs |
| GTC 2018 | Optimizing HPC Simulation and Visualization Codes Using NVIDIA Nsight Systems |
| GTC 2018 Israel | Boost DNN Training Performance using NVIDIA Tools |
| Siggraph 2018 | Taming the Beast; Using NVIDIA Tools to Unlock Hidden GPU Performance |

### Support

To file a bug report or ask a question on the Nsight Systems forums, register with the NVIDIA Developer Program. Registration is not required to read the forums.

- **Documentation:** https://docs.nvidia.com/nsight-systems/
- **Nsight Systems Forums:** https://forums.developer.nvidia.com/c/developer-tools/nsight-systems/
- **GitHub Issues:** https://github.com/NVIDIA/NsightSystems/issues
- **NVIDIA Developer:** https://developer.nvidia.com/nsight-systems
- **GUI Feedback:** Help > Send Feedback (enter email for a response)

---

## See Also

- [CLI Reference](02-cli-reference.md)
- [GPU Metrics](04-gpu-metrics.md)
- [Export Formats and SQLite Schema](11-export-sqlite-schema.md)
- [Containers, Migration, and Plugins](10-containers-migration.md)
