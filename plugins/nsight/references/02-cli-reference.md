# Chapter 2: CLI Command Reference

> NVIDIA Nsight Systems -- Comprehensive Reference Manual

---

## Table of Contents

- [Global Options](#global-options)
- [Command Switches Overview](#command-switches-overview)
- [profile](#profile)
  - [Profile Options (A-E)](#profile-options-a-e)
  - [Profile Options (F-M)](#profile-options-f-m)
  - [Profile Options (N-S)](#profile-options-n-s)
  - [Profile Options (T-Z)](#profile-options-t-z)
- [analyze](#analyze)
- [cancel](#cancel)
- [export](#export)
- [launch](#launch)
- [nvprof](#nvprof)
- [recipe](#recipe)
- [sessions](#sessions)
- [shutdown](#shutdown)
- [start](#start)
- [stats](#stats)
- [status](#status)
- [stop](#stop)
- [Example Single Command Lines](#example-single-command-lines)
- [Example Interactive CLI Sequences](#example-interactive-cli-sequences)
- [Example Stats Command Sequences](#example-stats-command-sequences)

---

## Global Options

These options apply to all commands.

| Option | Description |
|--------|-------------|
| `-h`, `--help` | Show help message and exit. Can be combined with a command name for command-specific help (e.g., `nsys profile -h`). |
| `-v`, `--version` | Print the Nsight Systems CLI version and exit. |

```bash
# Show global help
nsys --help

# Show version
nsys --version

# Show command-specific help
nsys profile --help
nsys stats --help
```

---

## Command Switches Overview

| Command | Description | Interactive Mode |
|---------|-------------|:----------------:|
| `profile` | Profile an application by launching it under Nsight Systems | No |
| `analyze` | Analyze an existing report file | No |
| `cancel` | Cancel the active trace session (interactive mode) | Yes |
| `export` | Export report data to various formats | No |
| `launch` | Launch the GUI with an optional report file | No |
| `nvprof` | Legacy nvprof compatibility mode | No |
| `recipe` | Run a predefined analysis recipe on a report | No |
| `sessions` | List active trace sessions (interactive mode) | Yes |
| `shutdown` | Shut down the Nsight Systems daemon | Yes |
| `start` | Start a new trace session (interactive mode) | Yes |
| `stats` | Generate statistical reports from a trace file | No |
| `status` | Display the status of the Nsight Systems daemon | Yes |
| `stop` | Stop the active trace session (interactive mode) | Yes |

---

## profile

The `profile` command traces the target application and generates a report file. This is the most commonly used command.

### Syntax

```bash
nsys profile [options] [--] <application> [application-args]
```

### Profile Options (A-E)

| Option | Parameters | Default | Description |
|--------|-----------|---------|-------------|
| `--accelerator-trace` | `=on\|off` | `off` | Enable or disable accelerator tracing. |
| `--accelerator-wattson` | `=on\|off` | `off` | Enable Wattson GPU power analysis and estimation. |
| `--android` | `=on\|off` | `auto` | Enable Android target profiling. |
| `--android-activity` | `=<string>` | (none) | Launch the specified Android activity for profiling. |
| `--android-attach` | `=<pid>` | (none) | Attach to a running Android process. |
| `--android-launch-args` | `=<string>` | (none) | Additional arguments for the Android activity launch. |
| `--android-package` | `=<string>` | (none) | Android package name to profile. |
| `--capture-range` | `=none\|cudaProfilerApi\|nvtx\|nvtx-and-cuda-profiler-api` | `none` | Only capture data when the specified profiler API is active. `cudaProfilerApi` requires `cudaProfilerStart/Stop`, `nvtx` uses `NVTX_RANGE_PUSH/POP` with `nsys` domain. |
| `--capture-range-end` | `=none\|cudaProfilerApi\|nvtx\|stop-on-last-exit` | `none` | Stop capture at the specified boundary. |
| `--continue-on-exception` | `=on\|off` | `off` | Continue profiling even if an exception occurs in the target. |
| `--cpuctxsw` | `=none\|process\|thread` | `none` | Trace CPU context switches. `process` traces per-process switches; `thread` traces per-thread switches. Requires root on some systems. |
| `--cuda-flush-interval` | `=<milliseconds>` | `10000` | Interval (ms) at which CUDA trace buffers are flushed. Lower values reduce buffer overflow risk but may increase overhead. |
| `--cuda-memory-usage` | `=true\|false` | `false` | Track CUDA memory allocation and deallocation events. Records `cudaMalloc`, `cudaFree`, etc. |
| `--cuda-um-cpu-page-faults` | `=true\|false` | `false` | Trace Unified Memory CPU page faults. Requires root on Linux. |
| `--cuda-um-gpu-page-faults` | `=true\|false` | `false` | Trace Unified Memory GPU page faults. |
| `--cudabacktrace` | `=true\|false` | `false` | Capture CUDA API call backtraces. Records the call stack leading to each CUDA API call. Increases overhead significantly. |
| `--cudagraph` | `=true\|false` | `true` | Trace CUDA Graph capture, instantiation, and execution events. |
| `--cudagrpcpu` | `=true\|false` | `false` | Trace CUDA Graph-related CPU-side events. |
| `--cudaProfilingApi` | `=auto\|cdp1\|cdp2` | `auto` | CUDA Dynamic Parallelism profiling mode. `cdp1` for CUDA DP v1, `cdp2` for CUDA DP v2. |
| `--delay` | `=<seconds>` | `0` | Delay (in seconds) before starting the trace. Useful for skipping initialization phases. |
| `--duration` | `=<seconds>` | `0` | Duration (in seconds) for the trace. `0` means trace until the application exits. |
| `--duration-override` | `=<seconds>` | (none) | Override the trace duration without modifying other settings. |
| `--enable*` | Various | (varies) | A family of options to enable specific tracing features. See individual options. |
| `--env-var` | `=<KEY=VALUE>` | (none) | Set an environment variable for the target process. Can be specified multiple times. |

### Profile Options (F-M)

| Option | Parameters | Default | Description |
|--------|-----------|---------|-------------|
| `--ftrace` | `=on\|off` | `off` | Enable Linux kernel ftrace (function tracing). Requires root access. |
| `--ftrace-events` | `=<function1,function2,...>` | (none) | Specify which kernel functions to trace with ftrace. Separate with commas. |
| `--force-overwrite` | `=true\|false` | `false` | Overwrite existing report file without prompting. |
| `--gpu-metrics-device` | `=all\|none\|<device_id>` | `none` | Collect GPU hardware metrics. `all` collects from all GPUs, a number targets a specific GPU. |
| `--gpu-metrics-frequency` | `=<frequency_hz>` | `10000` | Frequency (Hz) at which GPU metrics are sampled. Range: 100-100000. Higher frequencies increase overhead. |
| `--gpu-temp-memory` | `=true\|false` | `false` | Track GPU memory temperature metrics. |
| `--gpu-temp-power` | `=true\|false` | `false` | Track GPU power metrics. |
| `--gpuctxsw` | `=true\|false` | `false` | Trace GPU context switch events. |
| `--help` | (none) | N/A | Show profile command help and exit. |
| `--hostname` | `=<hostname>` | `localhost` | Hostname or IP address of the target machine for remote profiling. |
| `--hot-clock` | `=true\|false` | `false` | Enable hot clock frequency tracking. |
| `--kill` | `=none\|sigkill\|sigterm\|<signal>` | `none` | Send a signal to the target process when the trace is stopped. `none` lets the application exit naturally. |
| `--launch-attach` | `=<pid>` | (none) | Attach to an already running process instead of launching a new one. |
| `--launch-forward` | `=on\|off` | `on` | Forward stdout/stderr from the target process. |
| `--launch-watchdog` | `=on\|off` | `off` | Enable a watchdog to detect hung target processes. |
| `--magic-repr` | `=true\|false` | `true` | Enable magic representation for binary data in the trace. |
| `--malloc-tracking` | `=true\|false` | `false` | Track `malloc`/`free` calls for memory allocation analysis. |
| `--metrics` | `=<metric_group1,metric_group2,...>` | (none) | Specify which GPU metric groups to collect. Use `nsys stats --report gpu-metrics` to see available groups. |
| `--module` | `=<module>` | (none) | Specify the CUDA module to trace. |

### Profile Options (N-S)

| Option | Parameters | Default | Description |
|--------|-----------|---------|-------------|
| `--name` | `=<session_name>` | (auto) | Name for the trace session. Used to identify sessions in interactive mode. |
| `--nic` | `=all\|none\|<device_id>` | `none` | Trace NIC (Network Interface Card) metrics. `all` traces all NICs, a number targets a specific NIC. |
| `--nic-frequency` | `=<frequency_hz>` | `100` | Frequency (Hz) for NIC metrics sampling. Range: 1-1000. |
| `--nv-nsight-cli` | (none) | N/A | Internal use. |
| `--nvprof` | (none) | N/A | Legacy nvprof compatibility mode. |
| `--nvtx-include` | `=<domain1:range1,domain2:range2,...>` | (none) | Only capture NVTX ranges matching the specified filter. Format: `domain:range_name`. Supports wildcards. |
| `--nvtx-domain` | `=<domain>` | (none) | Filter NVTX ranges by domain name. |
| `--output` | `=<filename>` | `report<#>` | Output filename. `%q{ENV_VAR}` inserts environment variable value. `%%` inserts `%`. `%p` inserts PID. `%h` inserts hostname. `##` inserts sequential number. |
| `--output-fmt` | `=<format>` | (auto from extension) | Force output format. Options: `sqlite` (`.nsys-rep`), `hdf5` (`.h5`), `text` (`.txt`). |
| `--override` | `=true\|false` | `false` | Override all conflicting settings and force the specified options. |
| `--pennant` | `=true\|false` | `false` | Enable PENNANT tracing support. |
| `--power-transition` | `=true\|false` | `false` | Track GPU power state transitions. |
| `--python-api-tracing` | `=true\|false` | `false` | Trace Python function calls. |
| `--python-backtrace` | `=true\|false` | `false` | Capture Python backtraces in CPU samples. |
| `--python-sampling` | `=true\|false` | `false` | Enable Python-specific sampling. |
| `--qnx-fork-stats` | `=true\|false` | `false` | Track fork statistics on QNX. |
| `--qnx-name` | `=<name>` | (none) | QNX process name filter. |
| `--qnx-pid` | `=<pid>` | (none) | QNX process ID filter. |
| `--qnx-tid` | `=<tid>` | (none) | QNX thread ID filter. |
| `--report` | `=<report_type>` | (none) | Generate specific report type after profiling. |
| `--roi-activity` | `=global\|thread\|off` | `global` | How NVTX ROI (Region of Interest) activities are attributed. |
| `--sample` | `=cpu\|none` | `cpu` | Enable CPU sampling. `cpu` enables instruction-level sampling; `none` disables it. |
| `--sample-frequency` | `=<frequency>` | `1000` | CPU sampling frequency in Hz. Higher values give more precise data but increase overhead. Range: 1-100000. |
| `--session` | `=<session_id>` | `new` | Specify a session ID for interactive mode. `new` creates a new session. |
| `--show-progress` | `=true\|false` | `true` | Display progress information during profiling. |
| `--sm-clock` | `=true\|false` | `false` | Track SM clock frequency. |
| `--stats` | `=true\|false` | `false` | Generate a default stats report after profiling. Equivalent to running `nsys stats` after profiling. |
| `--stop-on-exit` | `=true\|false` | `true` | Stop the trace when the target process exits. |
| `--stop-on-disconnect` | `=true\|false` | `true` | Stop the trace when the CLI disconnects (interactive mode). |
| `--switch` | `=on\|off` | `off` | Enable switch tracing (network). |

### Profile Options (T-Z)

| Option | Parameters | Default | Description |
|--------|-----------|---------|-------------|
| `-t`, `--trace` | `=<trace_features>` | `cuda,nvtx,osrt` | Comma-separated list of features to trace. See trace features table below. |
| `--trace-fork` | `=true\|false` | `true` | Trace child processes created via `fork()`. |
| `--trey` | `=true\|false` | `false` | Enable Trey tracing support. |
| `--use-app-ctx` | `=true\|false` | `false` | Use the application's context for tracing. |
| `-w`, `--wait` | `=primary\|all` | `primary` | Wait for either the primary or all traced processes to finish before stopping. |
| `--warmup` | `=<seconds>` | `0` | Warm-up period before tracing begins. Trace data collected during warmup is discarded. |
| `--xhv-trace` | `=true\|false` | `false` | Enable XHV (Xen Hypervisor) tracing. |
| `--xhv-trace-events` | `=<events>` | (none) | Specify XHV events to trace. |

### Trace Features (`-t` / `--trace`)

The `-t` option accepts a comma-separated list of the following features:

| Feature | Description | Platform |
|---------|-------------|----------|
| `cuda` | CUDA Driver and Runtime API tracing, kernel launches, memory transfers | All |
| `nvtx` | NVIDIA Tools Extension markers and ranges | All |
| `osrt` | OS Runtime API (pthreads, semaphores, I/O) | Linux/QNX |
| `cuda_rt` | CUDA Runtime API only (subset of `cuda`) | All |
| `cuda_driver` | CUDA Driver API only (subset of `cuda`) | All |
| `cublas` | cuBLAS library calls | All |
| `cudnn` | cuDNN library calls | All |
| `cublaslt` | cuBLASLt library calls | All |
| `cufft` | cuFFT library calls | All |
| `curand` | cuRAND library calls | All |
| `cusolver` | cuSOLVER library calls | All |
| `cusparse` | cuSPARSE library calls | All |
| `nvjpeg` | nvJPEG library calls | All |
| `nvmpi` | NVIDIA MPI library calls | Linux |
| `nvvideo` | NVIDIA Video codec calls | Linux |
| `nvmedia` | NvMedia calls | QNX |
| `opengl` | OpenGL API calls | Linux/Windows |
| `openglx` | OpenGL extension calls | Linux |
| `vulkan` | Vulkan API calls | Linux/Windows |
| `vulkan-loader` | Vulkan loader calls | Linux/Windows |
| `dx11` | DirectX 11 | Windows |
| `dx12` | DirectX 12 | Windows |
| `dx12-d3d` | DirectX 12 D3D calls | Windows |
| `dx12-residency` | DirectX 12 residency tracking | Windows |
| `openacc` | OpenACC runtime calls | All |
| `openmp` | OpenMP runtime events | All |
| `mpi` | MPI communication events | Linux |
| `python` | Python function tracing | All |
| `numpy` | NumPy API tracing | All |
| `os` | OS-level events (generic) | All |
| `ftrace` | Linux kernel ftrace (requires root) | Linux |
| `syscalls` | Linux system call tracing | Linux |
| `etw` | Event Tracing for Windows | Windows |
| `wddm` | WDDM (Windows Display Driver Model) tracing | Windows |
| `xnvctrl` | XNVCTRL (X11 NV Control) tracing | Linux |
| `nvapi` | NVAPI tracing | Windows |
| `dxcore` | DXCore tracing | Windows |
| `dxgi` | DXGI (DirectX Graphics Infrastructure) tracing | Windows |
| `uvm` | Unified Virtual Memory events | Linux |
| `cufile` | cuFile (GDS) operations | Linux |
| `pennant` | PENNANT proxy tracing | Linux |
| `video` | Video codec tracing | All |
| `wgl` | WGL (Windows Graphics Library) | Windows |
| `gdal` | GDAL tracing | Linux |

### Sample Output Filenames

```bash
# Default naming (auto-increment)
nsys profile ./my_app
# Creates: report1.nsys-rep, report2.nsys-rep, ...

# Custom filename
nsys profile -o my_profile ./my_app
# Creates: my_profile.nsys-rep

# With PID in filename
nsys profile -o "profile_%p" ./my_app
# Creates: profile_12345.nsys-rep

# With hostname
nsys profile -o "profile_%h" ./my_app
# Creates: profile_myserver.nsys-rep

# With environment variable
nsys profile -o "profile_%q{EXPERIMENT_NAME}" ./my_app
# Creates: profile_baseline.nsys-rep (if EXPERIMENT_NAME=baseline)

# Force overwrite
nsys profile -o my_profile --force-overwrite=true ./my_app
```

---

## analyze

The `analyze` command performs analysis on an existing report file.

### Syntax

```bash
nsys analyze [options] <report-file>
```

### Options

| Option | Parameters | Default | Description |
|--------|-----------|---------|-------------|
| `--help` | (none) | N/A | Show analyze command help. |
| `--report` | `=<report_type>` | (default set) | Specify which reports to generate. Same report types as `stats`. |
| `--format` | `=<format>` | `text` | Output format: `text`, `csv`, `json`. |
| `--output` | `=<filename>` | stdout | Output filename for the analysis results. |
| `--force-overwrite` | `=true\|false` | `false` | Overwrite existing output file. |
| `--time-range` | `=<start:end>` | (full trace) | Analyze only the specified time range (in nanoseconds). |
| `--gpu` | `=<device_id>` | all | Filter analysis to a specific GPU. |

---

## cancel

The `cancel` command cancels the active trace session in interactive mode. Data collected so far is discarded.

### Syntax

```bash
nsys cancel [options]
```

### Options

| Option | Parameters | Default | Description |
|--------|-----------|---------|-------------|
| `--help` | (none) | N/A | Show cancel command help. |
| `--session` | `=<session_id>` | (active) | Specify which session to cancel. |

---

## export

The `export` command exports report data to various formats for further analysis.

### Syntax

```bash
nsys export [options] <report-file>
```

### Options

| Option | Parameters | Default | Description |
|--------|-----------|---------|-------------|
| `--help` | (none) | N/A | Show export command help. |
| `-t`, `--type` | `=sqlite\|hdf5\|text\|csv\|json` | `sqlite` | Export format type. |
| `-o`, `--output` | `=<filename>` | (input name with new extension) | Output filename. |
| `--force-overwrite` | `=true\|false` | `false` | Overwrite existing output file. |
| `--lz4` | `=on\|off` | `on` | Use LZ4 compression for SQLite output. |
| `--exclude` | `=<table1,table2,...>` | (none) | Exclude specific tables from export. |
| `--include` | `=<table1,table2,...>` | (all) | Include only specific tables in export. |
| `--time-range` | `=<start:end>` | (full trace) | Export only the specified time range (nanoseconds). |
| `--separator` | `=<char>` | `,` | CSV separator character. |
| `--gpu` | `=<device_id>` | all | Filter to a specific GPU device. |

### Export Examples

```bash
# Export to SQLite
nsys export -t sqlite -o my_report.sqlite my_report.nsys-rep

# Export to HDF5
nsys export -t hdf5 -o my_report.h5 my_report.nsys-rep

# Export only CUDA kernel data to CSV
nsys export -t csv -o kernels.csv --include=CUPTI_ACTIVITY_KIND_KERNEL my_report.nsys-rep

# Export a specific time range
nsys export -t sqlite --time-range=1000000000:5000000000 -o range.sqlite my_report.nsys-rep
```

---

## launch

The `launch` command opens the Nsight Systems GUI, optionally loading a report file.

### Syntax

```bash
nsys launch [options] [report-file]
```

### Options

| Option | Parameters | Default | Description |
|--------|-----------|---------|-------------|
| `--help` | (none) | N/A | Show launch command help. |
| `--hostname` | `=<hostname>` | `localhost` | Hostname for remote GUI display. |
| `--port` | `=<port>` | `auto` | Port for GUI communication. |
| `--style` | `=<style>` | (system default) | Qt style for the GUI (`fusion`, `windows`, etc.). |
| `--dpi-scaling` | `=<factor>` | (auto) | DPI scaling factor for the GUI. |
| `--session` | `=<session_id>` | (none) | Connect to a specific interactive session. |

---

## nvprof

The `nvprof` command provides backward compatibility with the legacy NVIDIA Visual Profiler (nvprof). It translates nvprof-style command-line arguments to Nsight Systems equivalents.

### Syntax

```bash
nsys nvprof [nvprof-options] <application> [application-args]
```

### Supported nvprof Options

| nvprof Option | Nsight Systems Equivalent |
|---------------|--------------------------|
| `--analysis-metrics` | `--gpu-metrics-device=all` |
| `--export-profile` | `-o <filename>` |
| `--kernels <regex>` | Kernel filtering in stats |
| `--metrics <list>` | `--metrics <list>` |
| `--print-gpu-trace` | `nsys stats --report gpukernsum` |
| `--print-summary` | `nsys stats` |
| `--profile-from-start` | `--delay` |
| `--system-profiling` | `-s cpu` |
| `--trace <list>` | `-t <list>` |
| `--unified-memory-profiling` | `--cuda-um-cpu-page-faults --cuda-um-gpu-page-faults` |

---

## recipe

The `recipe` command runs predefined analysis recipes on a report file.

### Syntax

```bash
nsys recipe [options] <recipe-name> <report-file>
```

### Options

| Option | Parameters | Default | Description |
|--------|-----------|---------|-------------|
| `--help` | (none) | N/A | Show recipe command help. |
| `--list` | (none) | N/A | List all available recipes. |
| `--output` | `=<filename>` | stdout | Output file for recipe results. |
| `--force-overwrite` | `=true\|false` | `false` | Overwrite existing output file. |

### Available Recipes

List available recipes with:

```bash
nsys recipe --list
```

Common recipes include:
- **gpu_speed_of_light**: Overall GPU utilization summary
- **gpu_memcpy**: Memory transfer analysis
- **cuda_api**: CUDA API usage summary
- **kernel_latency**: Kernel launch latency analysis
- **memory**: Memory usage analysis
- **nccl**: NCCL collective communication analysis

---

## sessions

The `sessions` command lists all active trace sessions in interactive mode.

### Syntax

```bash
nsys sessions [options]
```

### Options

| Option | Parameters | Default | Description |
|--------|-----------|---------|-------------|
| `--help` | (none) | N/A | Show sessions command help. |
| `--format` | `=text\|csv\|json` | `text` | Output format for session listing. |

---

## shutdown

The `shutdown` command shuts down the Nsight Systems daemon on the target machine.

### Syntax

```bash
nsys shutdown [options]
```

### Options

| Option | Parameters | Default | Description |
|--------|-----------|---------|-------------|
| `--help` | (none) | N/A | Show shutdown command help. |
| `--hostname` | `=<hostname>` | `localhost` | Hostname of the target machine. |
| `--force` | `=true\|false` | `false` | Force shutdown even if sessions are active. |

---

## start

The `start` command begins a new trace session in interactive mode. Use with `stop` to control the trace window.

### Syntax

```bash
nsys start [options]
```

### Options

The `start` command accepts the same tracing options as `profile` (e.g., `-t`, `--sample`, `--delay`, `--duration`, `--gpu-metrics-device`, etc.), plus:

| Option | Parameters | Default | Description |
|--------|-----------|---------|-------------|
| `--help` | (none) | N/A | Show start command help. |
| `--name` | `=<session_name>` | `auto` | Name for the trace session. |
| `--hostname` | `=<hostname>` | `localhost` | Target hostname for remote tracing. |

---

## stats

The `stats` command generates statistical reports from an existing trace file. It is one of the most useful commands for quick analysis without opening the GUI.

### Syntax

```bash
nsys stats [options] <report-file>
```

### Options

| Option | Parameters | Default | Description |
|--------|-----------|---------|-------------|
| `--help` | (none) | N/A | Show stats command help. |
| `-r`, `--report` | `=<report_type>` | (default set) | Specify report type(s) to generate. See report types table below. Multiple reports can be specified. |
| `--format` | `=text\|csv\|json\|markdown` | `text` | Output format for the report. |
| `-o`, `--output` | `=<filename>` | stdout | Output file for the report. |
| `--force-overwrite` | `=true\|false` | `false` | Overwrite existing output file. |
| `--time-range` | `=<start:end>` | (full trace) | Restrict analysis to the specified time range (nanoseconds). |
| `--report-all` | (none) | N/A | Generate all available reports. |
| `--gpu` | `=<device_id>` | all | Filter to a specific GPU device. |
| `--user-data` | `=<path>` | (none) | Path to user-defined report definition file. |

### Report Types

| Report Type | Description |
|------------|-------------|
| `cuda_api_sum` | CUDA API summary (time per API function) |
| `cuda_api_kern_sum` | CUDA API kernel launch summary |
| `cuda_api_mem_sum` | CUDA API memory operation summary |
| `cuda_api_sync_sum` | CUDA API synchronization summary |
| `cuda_gpu_kern_sum` | GPU kernel execution summary |
| `cuda_gpu_kern_trace` | Detailed GPU kernel trace (every launch) |
| `cuda_gpu_mem_trace` | Detailed GPU memory operation trace |
| `cuda_gpu_mem_sum` | GPU memory operation summary |
| `cuda_gpu_stride_sum` | GPU stride access summary |
| `cuda_gpu_stall` | GPU stall reason analysis |
| `cuda_gpu_warp` | GPU warp execution summary |
| `cuda_gpu_occ` | GPU occupancy summary |
| `cuda_omp` | OpenMP + CUDA overlap analysis |
| `cuda_uvm` | Unified Memory activity summary |
| `cuda_hw_metrics` | GPU hardware metrics summary |
| `cuda_gpu_speed_of_light` | GPU speed-of-light analysis |
| `nvtx_sum` | NVTX range summary |
| `nvtx_push_pop` | NVTX push/pop ranges |
| `nvtx_start_end` | NVTX start/end ranges |
| `osrt_sum` | OS Runtime API summary |
| `osrt_api_sum` | OS Runtime API function summary |
| `omp_sum` | OpenMP summary |
| `mpi_sum` | MPI communication summary |
| `cpu_samples` | CPU sampling summary |
| `cpu_samples_raw` | Raw CPU sample data |
| `cuda_graph` | CUDA Graph summary |
| `python_sum` | Python function summary |
| `vulkan_sum` | Vulkan API summary |
| `dx12_sum` | DirectX 12 summary |
| `gpu_metrics` | GPU hardware metrics time series |
| `nic` | NIC (network) metrics summary |
| `pwr` | Power metrics summary |
| `thread` | Thread activity summary |

### Stats Examples

```bash
# Default stats report
nsys stats my_profile.nsys-rep

# Generate only CUDA kernel summary
nsys stats --report cuda_gpu_kern_sum my_profile.nsys-rep

# Generate multiple reports
nsys stats --report cuda_api_sum,cuda_gpu_kern_sum,osrt_sum my_profile.nsys-rep

# Generate all reports
nsys stats --report-all my_profile.nsys-rep

# Output as CSV
nsys stats --report cuda_gpu_kern_sum --format csv -o kernels.csv my_profile.nsys-rep

# Output as JSON
nsys stats --report cuda_api_sum --format json -o api.json my_profile.nsys-rep

# Analyze a specific time range
nsys stats --time-range=1000000000:5000000000 my_profile.nsys-rep
```

---

## status

The `status` command displays the current status of the Nsight Systems daemon and any active sessions.

### Syntax

```bash
nsys status [options]
```

### Options

| Option | Parameters | Default | Description |
|--------|-----------|---------|-------------|
| `--help` | (none) | N/A | Show status command help. |
| `--environment` | (none) | N/A | Show environment information (CUDA version, driver version, GPU info). |
| `--session` | `=<session_id>` | (all) | Show status for a specific session. |
| `--hostname` | `=<hostname>` | `localhost` | Target hostname for remote status check. |

### Status Examples

```bash
# Check daemon status
nsys status

# Show environment information
nsys status --environment

# Check status on a remote machine
nsys status --hostname=192.168.1.100
```

---

## stop

The `stop` command stops the active trace session in interactive mode and saves the collected data.

### Syntax

```bash
nsys stop [options]
```

### Options

| Option | Parameters | Default | Description |
|--------|-----------|---------|-------------|
| `--help` | (none) | N/A | Show stop command help. |
| `--session` | `=<session_id>` | (active) | Specify which session to stop. |
| `--output` | `=<filename>` | (auto) | Output filename for the trace data. |
| `--force-overwrite` | `=true\|false` | `false` | Overwrite existing output file. |
| `--stats` | `=true\|false` | `false` | Generate a stats report after stopping. |

---

## Example Single Command Lines

### Version and Help

```bash
# Display version
nsys --version

# Display global help
nsys --help

# Display profile-specific help
nsys profile --help

# Display stats-specific help
nsys stats --help
```

### Default Profile Run

```bash
# Profile with default settings (traces cuda, nvtx, osrt)
nsys profile ./my_application

# Profile with custom output name
nsys profile -o my_profile ./my_application

# Profile with forced overwrite
nsys profile -o my_profile --force-overwrite=true ./my_application
```

### Limited Trace

```bash
# Trace only CUDA (no OS runtime, no NVTX)
nsys profile -t cuda -o cuda_only ./my_application

# Trace only CUDA and NVTX
nsys profile -t cuda,nvtx -o cuda_nvtx ./my_application

# Trace CUDA with CPU sampling disabled
nsys profile -t cuda,nvtx -s none -o no_sampling ./my_application
```

### Delayed Start

```bash
# Skip first 10 seconds (e.g., initialization)
nsys profile --delay=10 -t cuda,nvtx -o skip_init ./my_application

# Profile only 30 seconds after 5-second warmup
nsys profile --delay=5 --duration=30 -t cuda,nvtx -o middle_section ./my_application

# Use warmup to allow JIT compilation to complete before tracing
nsys profile --warmup=15 -t cuda,nvtx -o after_jit python train.py
```

### ftrace (Linux Kernel Function Tracing)

```bash
# Trace specific kernel functions (requires root)
sudo nsys profile -t cuda,ftrace --ftrace-events=nvidia,pthread_create -o ftrace_profile ./my_application

# Trace all ftrace events matching pattern
sudo nsys profile -t cuda,ftrace --ftrace-events='nvidia*' -o nvidia_ftrace ./my_application
```

### GPU Metrics

```bash
# Collect GPU metrics from all GPUs
nsys profile -t cuda --gpu-metrics-device=all -o gpu_metrics ./my_application

# Collect GPU metrics from GPU 0 at 50kHz
nsys profile -t cuda --gpu-metrics-device=0 --gpu-metrics-frequency=50000 -o high_freq_metrics ./my_application

# Collect specific metric groups
nsys profile -t cuda --gpu-metrics-device=all --metrics=gpc__cycles_active,sm__cycles_active -o specific_metrics ./my_application

# Collect GPU power and temperature metrics
nsys profile -t cuda --gpu-metrics-device=all --gpu-temp-power=true -o power_metrics ./my_application
```

### CPU Events and Sampling

```bash
# Enable CPU sampling with high frequency
nsys profile -t cuda,nvtx -s cpu --sample-frequency=10000 -o high_sample ./my_application

# Enable CPU sampling with backtraces
nsys profile -t cuda,nvtx -s cpu --cpu-backtrace=true -o with_backtraces ./my_application

# Trace OS runtime (pthreads, I/O)
nsys profile -t cuda,nvtx,osrt -o osrt_trace ./my_application

# Trace CPU context switches (requires root)
sudo nsys profile -t cuda --cpuctxsw=thread -o ctxsw ./my_application

# Trace system calls
nsys profile -t cuda,syscalls -o syscalls_trace ./my_application
```

### ETW (Windows Event Tracing)

```bash
# Trace with ETW on Windows
nsys profile -t cuda,etw -o etw_profile my_application.exe

# Trace DirectX 12 with ETW
nsys profile -t cuda,dx12,etw -o dx12_profile my_application.exe

# Trace WDDM events
nsys profile -t cuda,wddm -o wddm_profile my_application.exe
```

### Python Profiling

```bash
# Profile a Python script with CUDA tracing
nsys profile -t cuda,nvtx,osrt -s cpu -o python_profile python my_script.py

# Profile with Python function tracing enabled
nsys profile -t cuda,nvtx,osrt,python -s cpu --python-backtrace=true -o python_funcs python my_script.py

# Profile with NumPy tracing
nsys profile -t cuda,nvtx,osrt,python,numpy -s cpu -o python_numpy python my_script.py

# Profile PyTorch with NVTX integration
nsys profile -t cuda,nvtx,osrt -s cpu -o pytorch_profile python train.py

# Profile TensorFlow with CUDA tracing
nsys profile -t cuda,nvtx,osrt -s cpu -o tf_profile python train.py

# Profile a Jupyter notebook cell
nsys profile -t cuda,nvtx,osrt -s cpu -o notebook_profile jupyter execute notebook.ipynb
```

### Vulkan Profiling

```bash
# Profile a Vulkan application
nsys profile -t vulkan,nvtx -s cpu -o vulkan_profile ./vulkan_app

# Profile Vulkan with GPU metrics
nsys profile -t vulkan --gpu-metrics-device=all -o vulkan_gpu ./vulkan_app

# Profile Vulkan loader calls
nsys profile -t vulkan,vulkan-loader -s cpu -o vulkan_loader ./vulkan_app
```

### CUDA Memory and Unified Memory

```bash
# Profile with CUDA memory usage tracking
nsys profile -t cuda --cuda-memory-usage=true -o mem_usage ./my_application

# Profile with Unified Memory transfer tracing
nsys profile -t cuda,uvm -o uvm_transfers ./my_application

# Profile with Unified Memory page faults
sudo nsys profile -t cuda,uvm --cuda-um-cpu-page-faults=true --cuda-um-gpu-page-faults=true -o um_page_faults ./my_application

# Profile with CUDA backtraces
nsys profile -t cuda --cudabacktrace=true -o cuda_bt ./my_application
```

### CUDA Graph Profiling

```bash
# Profile with CUDA Graph tracing (enabled by default)
nsys profile -t cuda --cudagraph=true -o cuda_graph ./my_application

# Profile with CUDA Graph node-level detail
nsys profile -t cuda --cudagraph=true --cudagrpcpu=true -o graph_detail ./my_application
```

### Multi-Process Profiling

```bash
# Profile an MPI application (one rank)
mpirun -np 1 nsys profile -t cuda,nvtx,mpi -o mpi_rank_%p ./my_mpi_app

# Profile all MPI ranks (each gets its own file)
mpirun -np 4 nsys profile -t cuda,nvtx,mpi -o rank_%p ./my_mpi_app
```

---

## Example Interactive CLI Sequences

Interactive mode allows you to control tracing with `start` and `stop` commands. This is useful when you want to trace a specific portion of a long-running application.

### Basic Interactive Sequence

```bash
# Terminal 1: Start the daemon and trace session
nsys start --name=my_session -t cuda,nvtx -s cpu

# Terminal 2: Launch the application normally
./my_application

# Terminal 1: Start tracing when ready
nsys start --name=my_session

# ... wait for the interesting part ...

# Terminal 1: Stop tracing and save
nsys stop --name=my_session -o interactive_profile

# Or cancel without saving
nsys cancel --name=my_session
```

### Delayed Start with Interactive Control

```bash
# Start daemon
nsys start --name=gpu_session -t cuda --gpu-metrics-device=all

# Launch application (tracing starts automatically)
./my_application &

# Wait for the interesting phase, then stop
nsys stop --name=gpu_session -o gpu_session_profile

# Check status at any time
nsys status
```

### Remote Interactive Profiling

```bash
# On host: start daemon on remote target
nsys start --hostname=192.168.1.100 --name=remote_session -t cuda,nvtx -s cpu

# On target: launch the application
./my_application

# On host: stop the trace
nsys stop --hostname=192.168.1.100 --name=remote_session -o remote_profile

# On host: open the report locally
nsys-ui remote_profile.nsys-rep
```

### Attach to Running Process

```bash
# Attach to a running process by PID
nsys profile --launch-attach=12345 -t cuda,nvtx -o attached_profile

# Trace for 10 seconds then detach
nsys profile --launch-attach=12345 --duration=10 -t cuda,nvtx -o ten_second_trace
```

---

## Example Stats Command Sequences

### Quick Overview

```bash
# Generate default stats (summary of all major categories)
nsys stats my_profile.nsys-rep
```

This produces tables covering:
1. **CUDA API Statistics**: Total time and count for each CUDA API function
2. **CUDA GPU Kernel Statistics**: Min/max/avg time and count per kernel
3. **CUDA GPU Memory Statistics**: Transfer sizes and durations
4. **OS Runtime API Statistics**: Time in OS functions

### Focused Analysis

```bash
# Focus on GPU kernel performance
nsys stats --report cuda_gpu_kern_sum my_profile.nsys-rep

# Focus on CUDA API overhead
nsys stats --report cuda_api_sum my_profile.nsys-rep

# Focus on memory transfers
nsys stats --report cuda_gpu_mem_sum my_profile.nsys-rep

# Focus on CPU hot spots
nsys stats --report cpu_samples my_profile.nsys-rep

# Focus on OS runtime overhead
nsys stats --report osrt_sum my_profile.nsys-rep
```

### Exporting Reports

```bash
# Export kernel summary as CSV
nsys stats --report cuda_gpu_kern_sum --format csv -o kernels.csv my_profile.nsys-rep

# Export multiple reports as JSON
nsys stats --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_sum --format json -o full_report.json my_profile.nsys-rep

# Export all reports to a directory
nsys stats --report-all --format csv -o ./reports/ my_profile.nsys-rep
```

### Time-Range Analysis

```bash
# Analyze only the first second of the trace
nsys stats --time-range=0:1000000000 my_profile.nsys-rep

# Analyze a specific time window (2s to 5s)
nsys stats --time-range=2000000000:5000000000 --report cuda_gpu_kern_sum my_profile.nsys-rep

# Combine with report type
nsys stats --time-range=1000000000:3000000000 --report cuda_api_sum,cuda_gpu_kern_sum my_profile.nsys-rep
```

### Detailed Trace Analysis

```bash
# Show every single kernel launch with full details
nsys stats --report cuda_gpu_kern_trace my_profile.nsys-rep

# Show every memory transfer
nsys stats --report cuda_gpu_mem_trace my_profile.nsys-rep

# Show NVTX range summary
nsys stats --report nvtx_sum my_profile.nsys-rep
```

### GPU Metrics Analysis

```bash
# Show GPU hardware metrics
nsys stats --report cuda_hw_metrics my_profile.nsys-rep

# Show GPU speed-of-light analysis
nsys stats --report cuda_gpu_speed_of_light my_profile.nsys-rep

# Show occupancy information
nsys stats --report cuda_gpu_occ my_profile.nsys-rep
```

### Combined Workflow

```bash
# Step 1: Profile the application
nsys profile -t cuda,nvtx,osrt -s cpu --gpu-metrics-device=all --stats=true -o my_profile ./my_application

# The --stats=true flag automatically generates a default stats report.
# For more detailed analysis:

# Step 2: Generate detailed kernel analysis
nsys stats --report cuda_gpu_kern_trace --format csv -o kernels.csv my_profile.nsys-rep

# Step 3: Generate API overhead analysis
nsys stats --report cuda_api_sum --format csv -o api.csv my_profile.nsys-rep

# Step 4: Generate CPU hot-spot analysis
nsys stats --report cpu_samples --format csv -o cpu.csv my_profile.nsys-rep

# Step 5: Export full database for custom SQL queries
nsys export -t sqlite -o my_profile.sqlite my_profile.nsys-rep

# Step 6: Query with custom SQL
sqlite3 my_profile.sqlite "SELECT displayName, AVG(duration)/1000 as avg_us, COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL GROUP BY displayName ORDER BY SUM(duration) DESC LIMIT 10;"
```

---

## Next Steps

- [Chapter 1: Overview & Getting Started](01-overview.md) -- Installation and first steps.
- [Chapter 3: CUDA Tracing Reference](03-cuda-tracing.md) -- Detailed CUDA tracing configuration and function lists.
