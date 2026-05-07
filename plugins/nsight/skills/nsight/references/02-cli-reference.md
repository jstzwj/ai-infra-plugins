# NVIDIA Nsight Systems -- CLI Command Reference

This document provides a comprehensive reference for all Nsight Systems CLI commands, options, and usage examples.

## 1. CLI Syntax Forms

The Nsight Systems command lines can have one of two forms:

```bash
nsys [global_option]
```

or

```bash
nsys [command_switch][optional command_switch_options][application] [optional application_options]
```

All command line options are case-sensitive. For command switch options:
- **Short options**: parameters follow the switch after a space (e.g., `-s process-tree`)
- **Long options**: the switch is followed by an equal sign and then the parameter(s) (e.g., `--sample=process-tree`)

For this version of Nsight Systems, if you launch a process from the command line to begin analysis, the launched process will be terminated when collection is complete, including runs with `--duration` set, unless the user specifies the `--kill none` option. The exception is that if the user uses NVTX, cudaProfilerStart/Stop, or hotkeys to control the duration, the application will continue unless `--kill` is set.

The Nsight Systems CLI supports concurrent analysis by using sessions. Each Nsight Systems session is defined by a sequence of CLI commands that define one or more collections (e.g., when and what data is collected). A session begins with either a start, launch, or profile command. A session ends with a shutdown command, when a profile command terminates, or, if requested, when all the process tree(s) launched in the session exit. Multiple sessions can run concurrently on the same system.

---

## 2. CLI Global Options

| Short | Long | Description |
|-------|------|-------------|
| `-h` | `--help` | Help message providing information about available command switches and their options. |
| `-v` | `--version` | Output Nsight Systems CLI version information. |

---

## 3. CLI Command Switches

The Nsight Systems command line interface can be used in two modes. You may launch your application and begin analysis with options specified to the `nsys` profile command. Alternatively, you can control the launch of an application and data collection using interactive CLI commands.

| Command | Description |
|---------|-------------|
| `analyze` | Post process existing Nsight Systems result, either in .nsys-rep or SQLite format, to generate expert systems report. |
| `cancel` | Cancels an existing collection started in interactive mode. All data already collected in the current collection is discarded. |
| `export` | Generates an export file from an existing .nsys-rep file. For more information about the exported formats see the /documentation/nsys-exporter directory in your Nsight Systems installation directory. |
| `launch` | In interactive mode, launches an application in an environment that supports the requested options. The launch command can be executed before or after a start command. |
| `nvprof` | Special option to help with transition from legacy NVIDIA nvprof tool. Calling `nsys nvprof [options]` will provide the best available translation of `nvprof [options]`. See Migrating from NVIDIA nvprof topic for details. No additional functionality of nsys will be available when using this option. |
| `profile` | A fully formed profiling description requiring and accepting no further input. The command switch options used determine when the collection starts, stops, what collectors are used, what processes are monitored, etc. |
| `recipe` | Post process multiple existing Nsight Systems results to generate statistical information and create various plots. See the Multi-Report Analysis topic for details. |
| `sessions` | Gives information about all sessions running on the system. |
| `shutdown` | Disconnects the CLI process from the launched application and forces the CLI process to exit. If a collection is pending or active, it is canceled. |
| `start` | Starts a collection in interactive mode. The start command can be executed before or after a launch command. |
| `stats` | Post process existing Nsight Systems result, either in .nsys-rep or SQLite format, to generate statistical information. |
| `status` | Reports on the status of a CLI-based collection or the suitability of the profiling environment. |
| `stop` | Stops a collection that was started in interactive mode. When executed, all active collections stop, the CLI process terminates but the application continues running. |

---

## 4. CLI Profile Command Switch Options

After choosing the profile command switch, the following options are available.

**Usage:**

```bash
nsys [global-options] profile [options] [application] [application-arguments]
```

| Short | Long | Parameters | Default | Description |
|-------|------|------------|---------|-------------|
| | `--accelerator-trace` | none, tegra-accelerators | none | Collect other accelerators workload trace from the hardware engine units. Available in Nsight Systems Embedded Platforms Edition only. |
| | `--auto-report-name` | true, false | false | Derive report file name from collected data using details of the profiled graphics application. Format: [Process Name][GPU Name][Window Resolution][Graphics API] Timestamp .nsys-rep. If true, automatically generate report file names. |
| `-b` | `--backtrace` | auto, fp, lbr, dwarf, none | | Select the backtrace method to use while sampling. lbr uses Intel Last Branch Record registers (Haswell+). fp is frame pointer. dwarf uses DWARF CFI. Setting to none reduces collection overhead. |
| `-c` | `--capture-range` | none, cudaProfilerApi, hotkey, nvtx | none | When --capture-range is used, profiling will start only when an appropriate start API or hotkey is invoked. If set to none, start/stop API calls and hotkeys will be ignored. Hotkey works for graphic applications only. |
| | `--capture-range-end` | none, stop, stop-shutdown, repeat[:N], repeat-shutdown:N | stop-shutdown | Behavior when a capture range ends. none = ignore end. stop = stop collection, ignore subsequent. stop-shutdown = stop and shutdown session. repeat[:N] = collect N capture ranges. repeat-shutdown:N = repeat N then shutdown. Use --kill to control target app termination. |
| | `--clock-frequency-changes` | true, false | false | Collect clock frequency changes. Available only in Nsight Systems Embedded Platforms Edition and Arm server (SBSA) platforms. |
| | `--command-file` | \<filename\> | none | Open a file that contains profile switches and parse the switches. Additional switches on the command line override switches in the file. Can be specified more than once. |
| | `--cpu-cluster-events` | 0x16, 0x17, ..., none | none | Collect per-cluster Uncore PMU counters. Multiple values separated by commas (no spaces). Use --cpu-cluster-events=help for full list. Embedded Platforms Edition only. |
| | `--cpu-core-events` (Embedded) | 0x11,0x13,...,none | none | Collect per-core PMU counters. Multiple values separated by commas. Use --cpu-core-events=help for full list. |
| | `--cpu-core-events` (Workstation) | 'help' or end user events 'x,y' | '2' (Instructions Retired) | Select CPU Core events to sample. Use --cpu-core-events=help for full list. Use --event-sample to enable. |
| | `--cpu-core-metrics` | 0,1,2,...,none | none | Collect metrics on the CPU core. Use --cpu-core-metrics=help for full list. Use --event-sample to enable. Only available on Grace. |
| | `--cpu-socket-events` (Embedded) | 0x2a,0x2c,...,none | none | Collect per-socket Uncore PMU counters. Embedded Platforms Edition only. |
| | `--cpu-socket-events` (Workstation) | 'help' or events 'x,y' | none | Select Uncore CPU Socket events to sample. Use --event-sample to enable. |
| | `--cpu-socket-metrics` | 0,1,2,...,none | none | Collect Uncore metrics on the CPU socket. Use --event-sample to enable. Only available on Grace. |
| | `--cpuctxsw` | process-tree, system-wide, none | process-tree | Trace OS thread scheduling activity. Select none to disable. Some values require root. If --sample is not none, --cpuctxsw is set to same value as --sample. Requires --sampling-trigger=perf in Embedded Platforms Edition. |
| | `--cuda-flush-interval` | milliseconds | See desc | Interval when buffered CUDA data is saved. For collections over 30 seconds, 10 seconds recommended. Default: 10000 for Embedded Platforms Edition, 0 otherwise. |
| | `--cuda-graph-trace` | graph, node | graph | If graph, CUDA graphs traced as a whole (requires driver 515.43+). If node, individual node activities collected (may cause significant overhead). |
| | `--cuda-memory-usage` | true, false | false | Track GPU memory usage by CUDA kernels. Only when CUDA tracing is enabled. May cause significant overhead. |
| | `--cuda-trace-all-apis` | true, false | false | Trace all CUDA APIs including less relevant ones. Default skips some non-critical APIs. May cause significant overhead. |
| | `--cuda-um-cpu-page-faults` | true, false | false | Track page faults when CPU code accesses device-resident memory. May cause significant overhead. Not available on Embedded Platforms Edition. |
| | `--cuda-um-gpu-page-faults` | true, false | false | Track page faults when GPU code accesses host-resident memory. May cause significant overhead. Not available on Embedded Platforms Edition. |
| | `--cudabacktrace` | all, none, kernel, memory, sync, other | none | Enable backtrace collection when CUDA API is invoked. Significant overhead. Values combinable with ','. Each may have threshold after ':' (default 1000ns). CPU sampling must be enabled. |
| `-y` | `--delay` | \<seconds\> | 0 | Collection start delay in seconds. |
| `-d` | `--duration` | \<seconds\> | NA | Collection duration in seconds (must be > 0). Launched process terminated unless --kill none. |
| | `--duration-frames` | 60 \<= integer | disabled | Stop recording after this many frames captured. Cannot include other stop options. |
| | `--dx-force-declare-adapter-removal-support` | true, false | false | Call DXGIDeclareAdapterRemovalSupport() before device creation. Requires DX11 or DX12 trace. |
| | `--dx12-gpu-workload` | true, false, individual, batch, none | individual | DX12 GPU workload tracing mode. individual = per-workload. batch = per-ExecuteCommandLists batch. none = no GPU trace. Requires --trace=dx12. Windows only. |
| | `--dx12-wait-calls` | true, false | true | Trace wait calls blocking on fences for DX12. Requires --trace=dx12. Windows only. |
| | `--xhv-vm-symbols` | \<filepath\> | none | XHV sampling config file. Embedded Platforms Edition only. |
| `-e` | `--env-var` | A=B | NA | Set environment variables for the launched application. Multiple: A=B,C=D. |
| | `--enable` | \<plugin\>[,arg1,arg2,...] | NA | Use specified plugin. Can be specified multiple times. Use --enable=help to list all plugins. |
| | `--etw-provider` | "\<name\>,\<guid\>" or JSON file | none | Add custom ETW trace provider(s). Can be used multiple times. Windows only. |
| | `--event-sample` | system-wide, none | none | Enable event sampling. Use --cpu-core-events=help and --os-events=help for available events. Not on Embedded Platforms Edition. |
| | `--event-sampling-frequency` | 1 to 20 Hz | 3 | Sampling frequency for event counts. Not on Embedded Platforms Edition. |
| | `--export` | arrow, arrowdir, hdf, json, parquetdir, sqlite, text, none | none | Create additional export files. Can be given more than once. Warning: large data may take minutes. |
| | `--flush-on-cudaprofilerstop` | true, false | true | If true, cudaProfilerStop() flushes CUDA trace buffers. |
| `-f` | `--force-overwrite` | true, false | false | Overwrite existing result files with same name. |
| | `--ftrace` | subsystem1/event1,subsystem2/event2 | | Collect ftrace events. Requires root. No ftrace events by default. |
| | `--ftrace-keep-user-config` | | | Skip initial ftrace setup, collect already configured events. |
| | `--gpu-metrics-devices` | GPU ID, help, all, none | none | Collect GPU Metrics from specified devices. Use help to determine GPU IDs. |
| | `--gpu-metrics-frequency` | integer | 10000 | GPU Metrics sampling frequency in Hz. Min 10, Max 200000. |
| | `--gpu-metrics-set` | alias, file:\<filename\> | see desc | Metric set for GPU Metrics. Use help for aliases. Default: first suitable set. |
| | `--gpu-video-device` | help, \<id1,id2,...\>, all, none | none | Analyze video devices. help lists supported devices and IDs. |
| | `--gpuctxsw` | true, false | false | Trace GPU context switches. Requires driver r435.17+ and root. |
| | `--help` | \<tag\> | none | Print help message. Optional tag filters relevant options. |
| | `--hotkey-capture` | 'F1' to 'F12' | 'F12' | Hotkey to trigger profiling. Requires --capture-range=hotkey. |
| | `--ib-net-info-devices` | \<NIC names\> | none | Comma-separated NIC names for ibdiagnet network discovery. |
| | `--ib-net-info-files` | \<file paths\> | none | Paths of existing ibdiagnet db_csv files. |
| | `--ib-net-info-output` | \<directory\> | none | Directory for ibdiagnet output. |
| | `--ib-switch-congestion-device` | \<IB switch GUIDs\> | none | IB switch GUIDs for congestion events. System scope. Repeatable. |
| | `--ib-switch-congestion-nic-device` | \<NIC name\> | none | NIC for accessing IB switches. Default: first active NIC. |
| | `--ib-switch-congestion-percent` | 1-100 | 50 | Percent of IB switch congestion events to collect. |
| | `--ib-switch-congestion-threshold-high` | 1-1023 | 75 | High threshold percentage for IB switch egress port buffer. |
| | `--ib-switch-metrics-device` | \<IB switch GUIDs\> | none | IB switch GUIDs for metrics. System scope. Repeatable. |
| | `--ib-switch-metrics-nic-device` | \<NIC name\> | none | NIC for accessing IB switches for metrics. |
| `-n` | `--inherit-environment` | true, false | true | true = current + tool env vars. false = only tool env vars. |
| | `--injection-use-detours` | true, false | true | Use detours for injection. false = use windows hooks (bypasses anti-cheat). |
| | `--isr` | true, false | false | Trace ISRs and DPCs. Requires admin. Windows only. |
| | `--kill` | none, sigkill, sigterm, signal number | sigterm | Signal sent to target application's process group. |
| | `--mpi-impl` | openmpi, mpich | openmpi | MPI implementation. Auto-detected if not specified. Requires --trace=mpi. |
| | `--nic-metrics` | true, false | false | Collect NIC/HCA device metrics. System scope. Not on Embedded Platforms Edition. |
| `-p` | `--nvtx-capture` | range@domain, range, range@* | none | NVTX range/domain to trigger profiling. Requires --capture-range=nvtx. |
| | `--nvtx-domain-exclude` | default, \<domains\> | | Exclude NVTX events from specified domains. Mutually exclusive with --nvtx-domain-include. Requires --trace=nvtx. |
| | `--nvtx-domain-include` | default, \<domains\> | | Only include NVTX events from specified domains. Mutually exclusive with --nvtx-domain-exclude. Requires --trace=nvtx. |
| | `--python-functions-trace` | \<json_file\> | | Path to JSON file containing requested NVTX annotations. |
| | `--opengl-gpu-workload` | true, false | true | Trace OpenGL GPU workload. Requires --trace=opengl. |
| | `--os-events` | 'help' or 'x,y' | | OS events to sample. Use help for list. Requires --event-sample. Not on Embedded Platforms Edition. |
| | `--osrt-backtrace-depth` | integer | 24 | Backtrace depth for OS runtime libraries calls. |
| | `--osrt-backtrace-stack-size` | integer | 6144 | Stack dump size in bytes for OSRT backtraces. |
| | `--osrt-backtrace-threshold` | nanoseconds | 80000 | Duration threshold for OSRT backtrace collection. |
| | `--osrt-threshold` | \<nanoseconds\> | 1000 ns | Duration threshold for OSRT API tracing. Values much less than 1000 may cause overhead. Ignored for file APIs when --osrt-file-access=true. |
| | `--osrt-file-access` | true, false | false | Collect file access data for OSRT APIs. Overrides --osrt-threshold for file APIs. |
| `-o` | `--output` | \<filename\> | report# | Report file name. Patterns: %q{ENV_VAR}, %h (hostname), %p (PID), %%. Default: report# in working directory. |
| | `--process-scope` | main, process-tree, system-wide | main | Process scope. Embedded Platforms Edition only. Workstation Edition always system-wide. |
| | `--python-backtrace` | cuda, none | none | Python backtrace on selected API trigger. Arm SBSA and x86 Linux only. Requires --cudabacktrace. |
| | `--python-sampling` | true, false | false | Python backtrace sampling. Arm SBSA, x86 Linux and Windows. Consider disabling CPU sampling for Python-only workflows. |
| | `--python-sampling-frequency` | 1-2000 | 1000 | Python sampling frequency in Hz. Ignored if --python-sampling=false. |
| | `--pytorch` | autograd-nvtx, autograd-shapes-nvtx, functions-trace, none | none | Enable PyTorch function annotations. |
| | `--dask` | functions-trace, none | none | Enable Dask function annotations. |
| | `--qnx-kernel-events` | class/event, ... | none | QNX kernel events. Use help for list. Embedded Platforms Edition only. |
| | `--qnx-kernel-events-mode` | system, process, fast, wide | system:fast | Default mode for QNX kernel events. Embedded Platforms Edition only. |
| | `--resolve-symbols` | true, false | true | Resolve symbols of captured samples and backtraces. |
| | `--retain-etw-files` | true, false | false | Retain and merge ETW files to output directory. |
| | `--run-as` | \<username\> | none | Run target as specified user. Requires root. Linux only. |
| `-s` | `--sample` | process-tree, system-wide, xhv, xhv-system-wide, none | process-tree | CPU IP/backtrace sample collection mode. none disables CPU sampling. Some modes require root. |
| | `--samples-per-backtrace` | integer \<= 32 | 1 (4 for DWARF) | CPU IP samples per backtrace sample. Lower = more data. Not on Embedded Platforms or non-Linux. |
| | `--sampling-frequency` | 100-8000 | 1000 | Sampling/backtracing frequency in Hz. QNX, L4T, and Windows only. |
| | `--sampling-period` (Embedded) | integer | dynamic | CPU Cycles before IP sample. Requires --sampling-trigger=perf. |
| | `--sampling-period` (Workstation) | integer | dynamic | Events before IP sample. Dynamically determined event type. Linux only. |
| | `--sampling-trigger` | timer, sched, perf, cuda | timer,sched | Backtrace collection trigger. Embedded Platforms Edition only. |
| | `--session-new` | [a-Z][0-9,a-Z,spaces] | profile-\<id\>-\<app\> | Session name. Starts with alpha. Supports %q{ENV_VAR}, %h, %%. |
| `-w` | `--show-output` | true, false | true | true = stdout/stderr to console AND report files. false = only to report files. |
| | `--soc-metrics` | true, false | false | Collect SoC Metrics. Embedded Platforms Edition only. |
| | `--soc-metrics-frequency` | integer | 10000 | SoC Metrics frequency in Hz. Min 100, Max 1000000. Embedded Platforms Edition only. |
| | `--soc-metrics-set` | alias, file:\<filename\> | see desc | SoC Metrics set. Embedded Platforms Edition only. |
| | `--start-frame-index` | 1 \<= integer | disabled | Start recording at this frame index. Cannot include other start options. |
| `-Y` | `--start-later` | true, false | false | Delay collection until nsys start is executed. Overrides --delay. |
| | `--stats` | true, false | false | Generate summary statistics. Creates SQLite database. Warning: large data may take minutes. |
| `-x` | `--stop-on-exit` | true, false | true | Stop on process exit or duration expiry. If false, duration must be set. Runs > 5 min unsupported. |
| | `--syscall` (experimental) | process-tree, pid-namespace, none | none | Collect system calls. process-tree = app only. pid-namespace = current namespace + children. |
| `-t` | `--trace` | cuda, nvtx, cublas, cublas-verbose, cusparse, cusparse-verbose, cudnn, cudla, cudla-verbose, cusolver, cusolver-verbose, opengl, opengl-annotations, openacc, openmp, osrt, mpi, nvvideo, vulkan, vulkan-annotations, dx11, dx11-annotations, dx12, dx12-annotations, openxr, openxr-annotations, oshmem, ucx, wddm, tegra-accelerators, python-gil, gds(experimental), none | cuda, opengl, nvtx, osrt | APIs to trace. Multiple values comma-separated. OpenACC/cuXXX auto-enable CUDA. Reflex SDK auto-collected with DX/Vulkan. cuDNN not on Windows. If \<api\>-annotations selected, base API also traced. |
| | `--trace-fork-before-exec` | true, false | false | Trace child processes after fork, before exec. May cause crash. Linux only. |
| | `--vsync` | true, false | false | Collect vsync events. Also captures display ftrace events. Embedded Platforms Edition only. |
| | `--vulkan-gpu-workload` | true, false, individual, batch, none | individual | Vulkan GPU workload tracing. Requires --trace=vulkan. Not on QNX. |
| | `--wait` | primary, all | all | primary = wait on app process. all = also wait on re-parented processes. |
| | `--wddm-additional-events` | true, false | true | Collect extended ETW events. Requires --trace=wddm. Windows only. |
| | `--wddm-backtraces` | true, false | false | Collect WDDM event backtraces. Requires --trace=wddm. Windows only. |
| | `--xhv-trace` | \<filepath\> | none | Hypervisor trace config. Embedded Platforms Edition only. |
| | `--xhv-trace-events` | all, none, core, sched, irq, trap | all | Hypervisor trace events. Embedded Platforms Edition only. |

---

## 5. CLI Analyze Command Switch Options

The `nsys` analyze command generates reports using expert system rules on existing results.

**Usage:**

```bash
nsys [global-options] analyze [options] [input-file]
```

| Short | Long | Parameters | Default | Description |
|-------|------|------------|---------|-------------|
| | `--help` | \<tag\> | none | Print help message with optional tag filter. |
| `-f` | `--format` | column, table, csv, tsv, json, hdoc, htable, . | | Output format. "." = default for given output. Can use multiple times or comma-separated list. |
| | `--force-export` | true, false | false | Force re-export of SQLite from .nsys-rep. |
| | `--force-overwrite` | true, false | false | Overwrite existing output files. |
| | `--help-formats` | \<name\>, ALL, [none] | none | List available output formats. |
| | `--help-rules` | \<name\>, ALL, [none] | none | List available analysis rules. |
| `-o` | `--output` | -, @\<cmd\>, \<basename\>, . | - | Output destination. "-" = console. "@" prefix = pipe to command. Otherwise file basename. |
| `-q` | `--quiet` | | | Suppress verbose messages, show only errors. |
| `-r` | `--rule` | cuda_memcpy_async, cuda_memcpy_sync, cuda_memset_sync, cuda_api_sync, gpu_gaps, gpu_time_util, dx12_mem_ops | all | Analysis rules to execute. Can use multiple times or comma-separated. |
| | `--sqlite` | \<file.sqlite\> | | Specify SQLite filename. Created from .nsys-rep if needed. |
| | `--timeunit` | nsec, usec, msec, seconds | nanoseconds | Basic unit of time. Longest prefix matching. |

---

## 6. CLI Cancel Command Switch Options

**Usage:**

```bash
nsys [global-options] cancel [options]
```

| Long | Parameters | Default | Description |
|------|------------|---------|-------------|
| `--help` | \<tag\> | none | Print help message. |
| `--session` | \<session identifier\> | none | Cancel collection in given session. Supports %q{ENV_VAR}, %h, %% patterns. |

---

## 7. CLI Export Command Switch Options

**Usage:**

```bash
nsys [global-options] export [options] [nsys-rep-file]
```

| Short | Long | Parameters | Default | Description |
|-------|------|------------|---------|-------------|
| | `--append` | | | Don't error on existing directory-format export files. |
| `-f` | `--force-overwrite` | true, false | false | Overwrite existing result files. |
| | `--help` | \<tag\> | none | Print help message. |
| | `--include-blobs` | true, false | false | Export NVTX payloads as binary data. Affects SQLite, Arrow, Parquet. |
| | `--include-json` | true, false | false | Include repetitive JSON blocks in export. |
| `-l` | `--lazy` | true, false | true | Lazy table creation (only when data present). Affects SQLite, HDF5, Arrow, Parquet. |
| `-o` | `--output` | \<filename\> | \<input\>.ext | Set output filename. |
| `-q` | `--quiet` | true, false | false | Suppress progress bar. |
| | `--separate-strings` | true, false | false | Output strings separately, one per line. JSON and text only. |
| `-t` | `--type` | sqlite, hdf, text, json, info, arrow, arrowdir, parquetdir | sqlite | Export format. HDF only on x86_64 Linux and Windows. |
| | `--tables` | \<pattern\>[,...] | | POSIX regex patterns for table filtering. Advanced feature. Affects SQLite, HDF5, Arrow, Parquet. |
| | `--times` | \<range\>[,...] | | Time range filter for events. Advanced feature. Affects SQLite, HDF5, Arrow, Parquet. |
| | `--ts-normalize` | true, false | false | Shift timestamps to UTC wall-clock time. Limited by clock sync precision. |
| | `--ts-shift` | signed integer (ns) | 0 | Shift all timestamps by given nanoseconds. |

---

## 8. CLI Launch Command Switch Options

**Usage:**

```bash
nsys [global-options] launch [options] <application> [application-arguments]
```

The launch command shares most options with the profile command. Launch-specific options:

| Short | Long | Parameters | Default | Description |
|-------|------|------------|---------|-------------|
| | `--session` | session identifier | none | Launch in indicated session. Supports %q{ENV_VAR}, %h, %% patterns. |
| | `--session-new` | [a-Z][0-9,a-Z,spaces] | profile-\<id\>-\<app\> | Name for the new session. Supports %q{ENV_VAR}, %h, %%. |

All other collection options (backtrace, cpuctxsw, cuda-*, cudabacktrace, cuda-graph-trace, dx12-*, env-var, gpu-video-device, hotkey-capture, inherit-environment, injection-use-detours, isr, mpi-impl, nvtx-*, opengl-gpu-workload, os-events, osrt-*, python-*, pytorch, dask, qnx-*, resolve-symbols, retain-etw-files, run-as, sample, sampling-*, show-output, trace, trace-fork-before-exec, vulkan-gpu-workload, wait, wddm-*) are the same as the Profile command.

---

## 9. CLI Sessions Command Switch Subcommands

**Usage:**

```bash
nsys [global-options] sessions [subcommand]
```

| Command | Description |
|---------|-------------|
| `list` | List all active sessions including ID, name, and state information |

### Sessions List Options

| Short | Long | Parameters | Default | Description |
|-------|------|------------|---------|-------------|
| | `--help` | \<tag\> | none | Print help message. |
| `-p` | `--show-header` | true, false | true | Whether to show header in output. |

---

## 10. CLI Shutdown Command Switch Options

**Usage:**

```bash
nsys [global-options] shutdown [options]
```

| Long | Parameters | Default | Description |
|------|------------|---------|-------------|
| `--help` | \<tag\> | none | Print help message. |
| `--kill` | Linux: one, sigkill, sigterm, signal number. Windows: true, false | sigterm | Signal to send to target. |
| `--session` | session identifier | none | Session to shutdown. Supports %q{ENV_VAR}, %h, %% patterns. |

---

## 11. CLI Start Command Switch Options

**Usage:**

```bash
nsys [global-options] start [options]
```

The start command supports all collection options from the profile command: accelerator-trace, backtrace, capture-range, capture-range-end, clock-frequency-changes, cpu-core-events, cpu-core-metrics, cpu-socket-events, cpu-socket-metrics, cpuctxsw, enable, xhv-vm-symbols, etw-provider, event-sample, event-sampling-frequency, export, flush-on-cudaprofilerstop, force-overwrite, ftrace, ftrace-keep-user-config, gpu-metrics-devices, gpu-metrics-frequency, gpu-metrics-set, gpu-video-device, gpuctxsw, ib-net-info-*, ib-switch-*, isr, nic-metrics, os-events, output, process-scope, retain-etw-files, sample, samples-per-backtrace, sampling-frequency, sampling-period, sampling-trigger, session-new, show-output, soc-metrics, soc-metrics-frequency, soc-metrics-set, stats, stop-on-exit, syscall, vsync, xhv-trace, xhv-trace-events.

---

## 12. CLI Stats Command Switch Options

Reports are processed using a three-tuple: (report, format, output). The first report uses the first format and first output, the second uses the second, etc. Lists are expanded by repeating the last element.

**Usage:**

```bash
nsys [global-options] stats [options] [input-file]
```

| Short | Long | Parameters | Default | Description |
|-------|------|------------|---------|-------------|
| | `--help` | \<tag\> | none | Print help message. |
| `-f` | `--format` | column, table, csv, tsv, json, hdoc, htable, . | | Output format. Console default: column. File/process default: csv. |
| | `--force-export` | true, false | false | Force re-export of SQLite from .nsys-rep. |
| | `--force-overwrite` | true, false | false | Overwrite existing report files. |
| | `--help-formats` | \<name\>, ALL, [none] | none | List output formats. |
| | `--help-reports` | \<name\>, ALL, [none] | none | List available reports. |
| `-o` | `--output` | -, @\<cmd\>, \<basename\>, . | - | Output destination. "-" = console. "@" = pipe to command. Otherwise file basename. |
| `-q` | `--quiet` | | | Only show errors. |
| `-r` | `--report` | See Report Scripts | default set | Reports to generate. Default: nvtx_sum, osrt_sum, cuda_api_sum, cuda_gpu_kern_sum, cuda_gpu_mem_time_sum, cuda_gpu_mem_size_sum, openmp_sum, opengl_khr_range_sum, vulkan_marker_sum, dx12_gpu_marker_sum, etc. |
| | `--report-dir` | \<path\> | | Add directory to report script search path. Can be used multiple times. |
| | `--sqlite` | \<file.sqlite\> | | Specify SQLite filename. |
| | `--timeunit` | nsec, usec, msec, seconds | nanoseconds | Basic unit of time. |

---

## 13. CLI Status Command Switch Options

**Usage:**

```bash
nsys [global-options] status [options]
```

| Short | Long | Parameters | Default | Description |
|-------|------|------------|---------|-------------|
| | `--all` | | | Print all profiling environment information. |
| `-e` | `--environment` | | | System suitability for profiling. |
| | `--help` | \<tag\> | none | Print help message. |
| `-n` | `--network` | | | System suitability for network profiling. |
| | `--session` | session identifier | none | Status of indicated session. Supports %q{ENV_VAR}, %h, %% patterns. |

---

## 14. CLI Stop Command Switch Options

**Usage:**

```bash
nsys [global-options] stop [options]
```

| Long | Parameters | Default | Description |
|------|------------|---------|-------------|
| `--help` | \<tag\> | none | Print help message. |
| `--keep` | time in seconds | 0 | Seconds of data to retain before stop. 0 = retain all. |
| `--session` | session identifier | none | Session to stop. Supports %q{ENV_VAR}, %h, %% patterns. |

---

## 15. Example Single Command Lines

### Version Information

```bash
nsys -v
```
Prints tool version information to the screen.

### Run with Elevated Privilege

```bash
sudo nsys profile <app>
```
Nsight Systems CLI (and target application) will run with elevated privilege. Necessary for features like FTrace or system-wide CPU sampling. Use --run-as to avoid elevating the target application.

### Default Analysis Run

```bash
nsys profile <application> [application-arguments]
```
Launch application. Start collecting immediately. End collection when application stops. Trace CUDA, OpenGL, NVTX, and OSRT APIs. Collect CPU sampling and thread scheduling. Generate report#.nsys-rep.

### Limited Trace Only Run

```bash
nsys profile --trace=cuda,nvtx -d 20 --sample=none --cpuctxsw=none -o my_test <application>
```
Trace CUDA and NVTX only for 20 seconds. No CPU sampling or thread scheduling. Output: my_test.nsys-rep.

### Delayed Start Run

```bash
nsys profile -e TEST_ONLY=0 -y 20 <application>
```
Set TEST_ONLY=0. Start collecting after 20 seconds. End at application exit. Default trace and sampling.

### Collect ftrace Events

```bash
nsys profile --ftrace=drm/drm_vblank_event -d 20
```
Collect ftrace drm_vblank_event events for 20 seconds. Requires root. List events: `sudo cat /sys/kernel/debug/tracing/available_events`

### Run GPU Metric Sampling on One TU10x

```bash
nsys profile --gpu-metrics-devices=0 --gpu-metrics-set=tu10x-gfxt <application>
```
Collect GPU metrics for GPU 0 (TU10x) using tu10x-gfxt metric set at 10 kHz.

### Run GPU Metric Sampling on All GPUs

```bash
nsys profile --gpu-metrics-devices=all --gpu-metrics-frequency=20000 <application>
```
Collect GPU metrics for all GPUs at 20 kHz sampling frequency.

### Collect CPU IP/backtrace and CPU Context Switch

```bash
nsys profile --sample=system-wide --duration=5
```
System-wide CPU IP/backtrace samples and context switch trace for 5 seconds. Requires root.

### Get Available CPU Core Events

```bash
nsys profile --cpu-core-events=help
```
Lists CPU events and maximum number that can be collected concurrently.

### Collect System-wide CPU Events

```bash
nsys profile --event-sample=system-wide --cpu-core-events='1,2' --event-sampling-frequency=5 <app>
```
System-wide CPU sampling + "CPU Cycles" and "Instructions Retired" every 200 ms. Requires root.

### Collect Custom ETW Trace (Windows)

```bash
nsys profile --etw-provider=file.JSON
```
Configure custom ETW collectors from JSON file. Collect for 20 seconds.

ETW level values: TRACE_LEVEL_CRITICAL, TRACE_LEVEL_ERROR, TRACE_LEVEL_WARNING, TRACE_LEVEL_INFORMATION, TRACE_LEVEL_VERBOSE.

ETW flag values: EVENT_TRACE_FLAG_ALPC, EVENT_TRACE_FLAG_CSWITCH, EVENT_TRACE_FLAG_DBGPRINT, EVENT_TRACE_FLAG_DISK_FILE_IO, EVENT_TRACE_FLAG_DISK_IO, EVENT_TRACE_FLAG_DISK_IO_INIT, EVENT_TRACE_FLAG_DISPATCHER, EVENT_TRACE_FLAG_DPC, EVENT_TRACE_FLAG_DRIVER, EVENT_TRACE_FLAG_FILE_IO, EVENT_TRACE_FLAG_FILE_IO_INIT, EVENT_TRACE_FLAG_IMAGE_LOAD, EVENT_TRACE_FLAG_INTERRUPT, EVENT_TRACE_FLAG_JOB, EVENT_TRACE_FLAG_MEMORY_HARD_FAULTS, EVENT_TRACE_FLAG_MEMORY_PAGE_FAULTS, EVENT_TRACE_FLAG_NETWORK_TCPIP, EVENT_TRACE_FLAG_NO_SYSCONFIG, EVENT_TRACE_FLAG_PROCESS, EVENT_TRACE_FLAG_PROCESS_COUNTERS, EVENT_TRACE_FLAG_PROFILE, EVENT_TRACE_FLAG_REGISTRY, EVENT_TRACE_FLAG_SPLIT_IO, EVENT_TRACE_FLAG_SYSTEMCALL, EVENT_TRACE_FLAG_THREAD, EVENT_TRACE_FLAG_VAMAP, EVENT_TRACE_FLAG_VIRTUAL_ALLOC.

### Profile a Python Script with CUDA

```bash
nsys profile --trace=cuda,cudnn,cublas,osrt,nvtx --delay=60 python my_dnn_script.py
```
Trace CUDA, cuDNN, cuBLAS, OSRT, and NVTX. Start profiling 60 seconds after launch.

### Profile a Vulkan Application

```bash
nsys profile --trace=vulkan,osrt,nvtx --delay=60 ./myapp
```
Trace Vulkan, OSRT, and NVTX. Start profiling 60 seconds after launch.

---

## 16. Example Interactive CLI Command Sequences

### Collect from Beginning, End Manually

```bash
nsys start --stop-on-exit=false
nsys launch --trace=cuda,nvtx --sample=none <application>
nsys stop
```
Create interactive CLI. Begin collecting on application launch. Trace CUDA and NVTX. Stop only when explicitly requested.

> **Warning:** If you start a collection and fail to stop it, storage may fill with collected data. Nsight Systems does not support runs over 5 minutes.

### Run Application, Begin Collection Manually

```bash
nsys launch -w true <application>
nsys start
```
Create interactive CLI. Launch application. No data collected until `nsys start`. Profile until application ends.

> **Note:** If application exits before start is called, Nsight Systems creates an empty .nsys-rep file.

### Run Application, Name Session, Keep Last Seconds

```bash
nsys start --session-new=mysession
nsys launch --session=mysession myapp
nsys stop --session=mysession --keep=3
```
Create named session. Launch app with default options. Stop and keep only last 3 seconds of data.

### Use cudaProfilerStart/Stop

```bash
nsys start -c cudaProfilerApi
nsys launch -w true <application>
```
Begin collecting on cudaProfilerStart(). Stop at cudaProfilerStop(), nsys stop, or process termination.

> **Note:** Use --capture-range-end=repeat to capture separate reports for each cudaProfilerStart/Stop pair.

### Use NVTX Capture Range

```bash
nsys start -c nvtx
nsys launch -w true -p MESSAGE@DOMAIN <application>
```
Begin collecting when NVTX range with given message in given domain opens.

NVTX capture range formats:
- `Message@Domain` -- ranges with given message in given domain
- `Message@*` -- ranges with given message in all domains
- `Message` -- ranges with given message in default domain

Enable non-registered NVTX strings:
```bash
nsys launch -w true -p profiler@service -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0 ./app
```

### Multiple Start/Stop Cycles

```bash
nsys launch <application>
nsys start
nsys stop
nsys start
nsys stop
nsys shutdown --kill sigkill
```
First start/stop generates report#.qstrm. Second generates report#.nsys-rep. Shutdown sends sigkill.

> **Note:** `nsys cancel` after `nsys start` cancels collection without generating a report.

---

## 17. Example Stats Command Sequences

### Display Default Statistics

```bash
nsys stats report1.nsys-rep
```
Export SQLite from .nsys-rep (if not existing). Print default reports in column format.

Equivalent:
```bash
nsys profile --stats=true <application>
```

### Display Specific Report

```bash
nsys stats --report cuda_gpu_trace report1.nsys-rep
```
Print the cuda_gpu_trace report.

### Multiple Reports, Formats, Outputs

```bash
nsys stats --report cuda_gpu_trace --report cuda_gpu_kern_sum --report cuda_api_sum \
  --format csv,column --output .,- report1.nsys-rep
```
Three reports. First to CSV file. Other two to console as columns.

### Pipe Report to Command

```bash
nsys stats --report cuda_api_sum --format table \
  --output @"grep -E (-|Name|cudaFree" test.sqlite
```
Pipe table-formatted cuda_api_sum output through grep. Limitations: no shell expansions, no pipes, no redirections. Use shell scripts for complex commands.

---

## 18. System Wide API Trace on Windows

Trace DX11, DX12, or Vulkan in already-running applications:

```bash
nsys profile --trace=dx12-annotations,wddm --dx12-gpu-workload=individual --duration=20
```
Then click each target application window. For DX11/DX12, the application must gain system focus. For Vulkan, the application must be launched after the nsys profile command.

---

## 19. Handling Application Launchers

### Single-Node Profiling

Prefix nsys before program or launcher:
```bash
nsys profile [nsys args] mpirun [mpirun args] ...
```

### Multi-Node Profiling

Prefix nsys before application, not launcher:
```bash
mpirun [mpirun args] nsys profile [nsys args] ...
```

Use rank/ID in output filename:
- `%q{OMPI_COMM_WORLD_RANK}` -- Open MPI
- `%q{PMI_RANK}` -- MPICH
- `%q{SLURM_PROCID}` -- Slurm
- `%p` -- PID

> **Warning:** Multiple processes writing to the same report file will cause an error.

### Profile Single Process (Rank 0 Only)

```bash
#!/bin/bash
# Use $PMI_RANK for MPICH and $SLURM_PROCID with srun
if [ $OMPI_COMM_WORLD_RANK -eq 0 ]; then
    nsys profile -e NSYS_MPI_STORE_TEAMS_PER_RANK=1 -t mpi "$@"
else
    "$@"
fi
```

Execute: `mpirun [mpirun options] ./nsys_profile.sh ./myapp [app options]`

> **Note:** When profiling subset of MPI ranks, set NSYS_MPI_STORE_TEAMS_PER_RANK=1.

### DeepSpeed

```bash
#!/bin/bash
nsys profile -t cuda,mpi,nvtx,cudnn -o rname.%p python ...
```

Use with: `deepspeed --no_python [deepspeed args] ./nsys_profile.sh`

### GPU and NIC Metrics (Single Rank)

```bash
#!/bin/bash
if [ $OMPI_COMM_WORLD_LOCAL_RANK -eq 0 ]; then
    nsys profile --nic-metrics=true --gpu-metrics-devices=all "$@"
else
    nsys profile "$@"
fi
```

Per-rank GPU metrics:
```bash
#!/bin/bash
nsys profile -e CUDA_VISIBLE_DEVICES=${OMPI_COMM_WORLD_LOCAL_RANK} \
  --gpu-metrics-devices=${OMPI_COMM_WORLD_LOCAL_RANK} "$@"
```

---

## 20. CLI nvprof Command Switch Options

The nvprof command helps former nvprof users transition to nsys. No additional nsys functionality available.

**Usage:**

```bash
nsys nvprof [options]
```

| Switch | Parameters | nsys Equivalent | Description |
|--------|-----------|-----------------|-------------|
| `--annotate-mpi` | off, openmpi, mpich | --trace=mpi AND --mpi-impl | Annotate MPI calls with NVTX markers. |
| `--cpu-thread-tracing` | on, off | --trace=osrt | Collect CPU thread API activity. |
| `--profile-api-trace` | none, runtime, driver, all | --trace=cuda | CUDA API tracing. runtime or driver = all for nsys. |
| `--profile-from-start` | on, off | if off: --capture-range=cudaProfilerApi | Enable/disable profiling from start. |
| `-t` / `--timeout` | \<nanoseconds\> (default 0) | --duration=seconds | Stop after timeout. nsys starts counting immediately. |
| `--cpu-profiling` | on, off | --sampling=cpu | CPU profiling toggle. |
| `--openacc-profiling` | on, off | --trace=openacc | OpenACC profiling. |
| `-o` / `--export-profile` | \<filename\> | --output={filename} and/or --export=sqlite | Export file. Supports %q{ENV_VAR}, %h, %%. |
| `-f` / `--force-overwrite` | | --force-overwrite=true | Force overwrite. |
| `-h` / `--help` | | --help | Print help. |
| `-V` / `--version` | | --version | Print version. |

> **Note:** NVIDIA Visual Profiler (NVVP) and NVIDIA nvprof are deprecated. Migrate to Nsight Systems.

---

## 21. Opening Command Line Results

### Open in GUI

The .nsys-rep file can be opened in any GUI of the same version or later. Very large files may consume all host memory.

### Import Windows ETL Files

ETL files from Xperf or GPUView can be imported via the Import dialog to create .nsys-rep files.
