# Python and CPU Profiling Reference

This document provides comprehensive reference material for CPU profiling on Linux and Python profiling features in NVIDIA Nsight Systems.

---

## Table of Contents

- [CPU Profiling on Linux](#cpu-profiling-on-linux)
  - [Features Overview](#features-overview)
  - [CPU Instruction Pointer / Backtrace Sampling](#cpu-instruction-pointer--backtrace-sampling)
  - [CPU Context Switch Tracing](#cpu-context-switch-tracing)
  - [CPU Event Sampling](#cpu-event-sampling)
  - [CPU Core Metrics](#cpu-core-metrics)
  - [System Requirements](#system-requirements)
    - [Paranoid Level](#paranoid-level)
    - [Kernel Version](#kernel-version)
    - [perf_event_open Syscall](#perf_event_open-syscall)
    - [Sampling Trigger](#sampling-trigger)
  - [Checking Your Target System](#checking-your-target-system)
  - [Configuring a CPU Profiling Collection](#configuring-a-cpu-profiling-collection)
  - [Visualizing CPU Profiling Results](#visualizing-cpu-profiling-results)
  - [Arm Topdown Analysis - Preview Feature](#arm-topdown-analysis---preview-feature)
  - [Common Issues](#common-issues)
    - [Reducing Overhead Caused By Sampling](#reducing-overhead-caused-by-sampling)
    - [Throttling](#throttling)
    - [Sample Intervals Are Irregular](#sample-intervals-are-irregular)
    - [No CPU Profiling Data Is Collected](#no-cpu-profiling-data-is-collected)
- [Python Profiling](#python-profiling)
  - [Python Backtrace Sampling](#python-backtrace-sampling)
  - [Python Functions Trace](#python-functions-trace)
  - [Python GIL Tracing](#python-gil-tracing)
  - [PyTorch Profiling](#pytorch-profiling)
  - [Dask Profiling](#dask-profiling)

---

## CPU Profiling on Linux

Nsight Systems on Linux targets utilizes the Linux OS' perf subsystem to sample CPU Instruction Pointers (IPs) and backtraces, trace CPU context switches, and sample CPU and OS event counts. The Linux perf tool utilizes the same perf subsystem.

Nsight Systems Embedded Platforms Edition on Linux kernel prior to v5.15 uses a custom kernel module to collect the same data. The Nsight Systems CLI command `nsys status --environment` indicates when the kernel module is used instead of the Linux OS' perf subsystem.

### Features Overview

CPU profiling on Linux includes the following capabilities:

| Feature | CLI Switch | Description |
|---------|-----------|-------------|
| IP/Backtrace Sampling | `--sample` | Periodically sample CPU instruction pointers and backtraces |
| Context Switch Tracing | `--cpuctxsw` | Trace every OS thread schedule/un-schedule event |
| Event Sampling | `--event-sample` | Periodically sample CPU hardware and OS event counts |
| CPU Core Metrics | `--cpu-core-metrics` | Access CPU core metric information (Grace only) |

### CPU Instruction Pointer / Backtrace Sampling

Nsight Systems can sample CPU Instruction Pointers (IPs) and backtraces periodically. The collection of a sample is triggered by a hardware event overflow -- for example, a sample is collected after every 1 million CPU reference cycles on a per-thread basis.

**Display Locations:**
- Individual thread timelines in the GUI
- Event Viewer
- Top Down, Bottom Up, or Flat views (histogram-like summaries)

**Modes:**
- **Process-tree mode** -- Nsight Systems will sample the process, and any of its descendants, launched by the tool.
- **System-wide mode** -- Nsight Systems will sample all processes running on the system, including any processes launched by the tool.

**Configuration:**

```bash
# Process-tree sampling (default)
nsys profile --sample=process-tree -- myApp

# System-wide sampling
nsys profile --sample=system-wide -- myApp

# Disable sampling
nsys profile --sample=none -- myApp
```

### CPU Context Switch Tracing

Nsight Systems can trace every time the OS schedules a thread on a logical CPU and every time the OS thread gets unscheduled from a logical CPU. The data is used to show CPU utilization and OS thread utilization within the Nsight Systems GUI.

**Modes:**
- **Process-tree mode** -- Nsight Systems will trace the process, and any of its descendants, launched by Nsight Systems.
- **System-wide mode** -- Nsight Systems will trace all processes running on the system, including any processes launched by Nsight Systems.

**CLI Option:** `--cpuctxsw`

| Parameter | Description |
|-----------|-------------|
| `process-tree` | Trace only launched process tree (default when app is launched) |
| `system-wide` | Trace all system processes |
| `none` | Disable context switch tracing |

**Important Notes:**
- If the `--sample` switch is set to a value other than `none`, the `--cpuctxsw` setting is hardcoded to the same value as the `--sample` switch.
- If `--sample=none` and a target application is launched, the default is `process-tree`.
- If `--sample=none` and no target application is launched, the default is `none`.
- On Nsight Systems Embedded Platforms Edition, this requires `--sampling-trigger=perf`.

### CPU Event Sampling

Nsight Systems can periodically sample CPU hardware event counts and OS event counts and show the event's rate over time in the Nsight Systems GUI.

**Mode:**
- **System-wide mode only** -- Nsight Systems will sample event counts of all CPUs and the OS event counts running on the system. Event counts are not directly associated with processes or threads.

**CLI Options:**

```bash
# Enable event sampling
nsys profile --event-sample=system-wide -- myApp

# Specify CPU core events to sample
nsys profile --event-sample=system-wide --cpu-core-events=2 -- myApp

# View available events
nsys profile --cpu-core-events=help
```

| Switch | Parameters | Default | Description |
|--------|-----------|---------|-------------|
| `--event-sample` | none, system-wide | none | Enable event sampling |
| `--cpu-core-events` | help, or event IDs | '2' (Instructions Retired) | Select CPU Core events to sample |
| `--cpu-socket-events` | help, or event IDs | none | Select Uncore CPU Socket events to sample |
| `--event-sampling-frequency` | frequency value | N/A | Set the event sampling frequency |

**Note:** Use the `--cpu-core-events=help` switch to see the full list of events and the number of events that can be collected simultaneously.

### CPU Core Metrics

Nsight Systems can access and make available information about CPU core metrics. This functionality is available **only on Linux** and **only for the NVIDIA Grace (TM) CPU**.

The `--cpu-core-metrics=help` command will list 39 different metrics. Those metrics are described in the Grace Performance Tuning Guide. Selected option IDs can then be fed into the `--cpu-core-metrics` switch.

**Usage:**

```bash
# List available metrics
nsys profile --cpu-core-metrics=help

# Select specific metrics
nsys profile --cpu-core-metrics=0,1,2 --event-sample=system-wide -- myApp

# Collect CPU socket metrics (Grace only)
nsys profile --cpu-socket-metrics=0,1,2 --event-sample=system-wide -- myApp
```

| Switch | Parameters | Description |
|--------|-----------|-------------|
| `--cpu-core-metrics` | 0,1,2,...,none | Collect metrics on the CPU core (Grace only) |
| `--cpu-socket-metrics` | 0,1,2,...,none | Collect Uncore metrics on the CPU socket (Grace only) |

### System Requirements

#### Paranoid Level

The system's paranoid level must be 2 or lower. The paranoid level controls access to performance monitoring data.

**Detailed Paranoid Level Table:**

| Paranoid Level | CPU IP/Backtrace Sampling (process-tree) | CPU IP/Backtrace Sampling (system-wide) | CPU Context Switch Tracing (process-tree) | CPU Context Switch Tracing (system-wide) | Event Sampling (system-wide) |
|---|---|---|---|---|---|
| **3 or greater** | not available | not available | not available | not available | not available |
| **2** | User mode IP/backtrace samples only | not available | available | not available | not available |
| **1** | Kernel and user mode IP/backtrace samples | not available | available | not available | not available |
| **0, -1** | Kernel and user mode IP/backtrace samples | Kernel and user mode IP/backtrace samples | available | available | hardware and OS events |

**To check paranoid level:**

```bash
cat /proc/sys/kernel/perf_event_paranoid
```

**To set paranoid level:**

```bash
# Set to unrestricted (requires root)
sudo sh -c 'echo 0 > /proc/sys/kernel/perf_event_paranoid'

# Set to per-process kernel+user mode (level 1)
sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid'
```

| Paranoid Value | Access Level |
|----------------|-------------|
| `-1` | Unrestricted access (all features available) |
| `0` | Unrestricted access (all features available) |
| `1` | Per-process kernel and user mode samples |
| `2` | Per-process user mode samples only |
| `3+` | No sampling available |

#### Kernel Version

To support the CPU profiling features utilized by Nsight Systems, the kernel version must be greater than or equal to **v4.3**.

RedHat has backported the required features to the v3.10.0-693 kernel. RedHat distros and their derivatives (e.g., CentOS) require a **3.10.0-693** or later kernel.

**To check kernel version:**

```bash
uname -r
```

#### perf_event_open Syscall

The `perf_event_open` syscall needs to be available. When running within a Docker container, the default seccomp settings will normally block the `perf_event_open` syscall.

**Workarounds:**
- Use the Docker run `--privileged` switch when launching the docker
- Modify the docker's seccomp settings
- Some VMs (virtual machines), e.g., AWS, may also block the `perf_event_open` syscall

#### Sampling Trigger

In some rare cases, a sampling trigger is not available. The sampling trigger is either a hardware or software event that causes a sample to be collected. Some VMs block hardware events from being accessed and therefore prevent hardware events from being used as sampling triggers. In those cases, Nsight Systems will fall back to using a software trigger if possible.

### Checking Your Target System

Use the `nsys status --environment` command to check if a system meets the Nsight Systems CPU profiling requirements.

**Important Notes:**
- This command does not check for Linux capability overrides (i.e., if the user or executable files have `CAP_SYS_ADMIN` or `CAP_PERFMON` capability).
- This command does not indicate if system-wide mode can be used.

**Example Output:**

```
$ nsys status --environment
Environment status output
```

### Configuring a CPU Profiling Collection

When configuring Nsight Systems for CPU Profiling from the CLI, use some or all of the following options:

| Option | Description |
|--------|-------------|
| `--sample` | Enable/disable IP/backtrace sampling (process-tree, system-wide, none) |
| `--cpuctxsw` | Enable/disable context switch tracing (process-tree, system-wide, none) |
| `--event-sample` | Enable/disable event sampling (system-wide, none) |
| `--backtrace` | Select backtrace method (auto, fp, lbr, dwarf, none) |
| `--cpu-core-events` | Select CPU core PMU events to sample |
| `--event-sampling-frequency` | Set event sampling frequency |
| `--os-events` | Select OS events |
| `--samples-per-backtrace` | Control number of backtraces per sample |
| `--sampling-period` | Set the sampling period |

**Example Commands:**

```bash
# Basic CPU profiling with default settings
nsys profile --sample=process-tree --cpuctxsw=process-tree -- myApp

# System-wide CPU profiling with backtraces
nsys profile --sample=system-wide --backtrace=fp -- myApp

# CPU profiling with event sampling
nsys profile --sample=process-tree --event-sample=system-wide \
    --cpu-core-events=2 -- myApp

# CPU profiling on Grace with core metrics
nsys profile --sample=process-tree --event-sample=system-wide \
    --cpu-core-metrics=0,1 -- myApp

# Reduced overhead CPU profiling
nsys profile --sample=process-tree --backtrace=none \
    --sampling-period=10000 -- myApp

# Intel LBR backtrace method
nsys profile --sample=process-tree --backtrace=lbr -- myApp

# DWARF backtrace method
nsys profile --sample=process-tree --backtrace=dwarf -- myApp

# Frame pointer backtrace method
nsys profile --sample=process-tree --backtrace=fp -- myApp
```

**Backtrace Methods:**

| Method | Description |
|--------|-------------|
| `auto` | Automatically select the best available method |
| `fp` | Frame pointer (assumes frame pointers enabled during compilation) |
| `lbr` | Intel Last Branch Record registers (Haswell and later CPUs only) |
| `dwarf` | DWARF's CFI (Call Frame Information) |
| `none` | Disable backtrace collection (reduces overhead) |

**GUI Configuration:**
When configuring from the GUI, the following options are available in the CPU profiling section of the project settings. The configuration used during CPU profiling is documented in:
- Analysis Summary
- Diagnostics Summary

### Visualizing CPU Profiling Results

#### CPU IP/Backtrace Data

In the timeline, yellow-orange marks can be found under each thread's timeline that indicate the moment an IP / backtrace sample was collected on that thread. Hovering the cursor over a mark will cause a tooltip to display the backtrace for that sample.

Below the Timeline is a drop-down list with multiple options for viewing CPU IP / backtrace sampling data:
- **Events View** -- Chronological listing of events
- **Top-Down View** -- Hierarchical view from caller to callee
- **Bottom-Up View** -- Hierarchical view from callee to caller
- **Flat View** -- Aggregated function-level summary

#### CPU Event Sampling

Event sampling samples hardware or software event counts during a collection and then graphs those events as rates on the Timeline.

- **Core and cache events** are graphed under the associated CPU row
- **Uncore and OS events** are graphed in their own row
- Hovering the cursor over an event sampling row in the timeline shows the event's rate at that moment

### Arm Topdown Analysis - Preview Feature

Arm Topdown methodology supports performance analysis, workload characterization, and microarchitecture exploration. Nsight Systems provides scripting to support running this analysis for the Grace CPU.

**Location:** In your `target-linux-sbsa-armv8/cpu` directory, look for a script named `collect_grace_topdown.sh`.

**What the Script Does:**
The script simplifies collecting all PMU core event and metric data needed to perform a traditional CPU Topdown analysis of the workload's CPU performance. The script runs multiple system-wide `nsys profile` commands sequentially to collect the data.

**Switch Restrictions:**
You can add additional Nsight Systems options to the command line as per usual, with the following exceptions:
- `--event-sample`, `--event-sampling-interval`, `--cpu-core-events`, and `--cpu-core-metrics` switches are set by the script for Topdown analysis
- `-f` / `--force-overwrite` switch is set to true by the script
- `-o` / `--output` switch is set by the script to generate a list of predefined output .nsys-rep files
- `--kill` switch is set to the default value of `sigterm`

**Example Command:**

```bash
collect_grace_topdown.sh --trace=osrt,nvtx,cuda -- myApp arg1 arg2
```

**Output:**
Output files will be written to the current working directory. The output consists of a collection of `.nsys-rep` files that contain the metric data required to do a Topdown analysis of the workload. These files can be opened in the Nsight Systems GUI to view the metric results on the timeline.

**Further Analysis:**
You can use the NVTX CPU Topdown recipe (`nsys recipe nvtx_cpu_topdown --input .`) to process the data from the `.nsys-rep` files and generate an output with CPU Topdown Methodology metrics computed for NVTX ranges.

**Important Notes:**
- Arm Topdown analysis requires multiple system-wide collections and may take a significantly long time to run and post-process.
- For details and use cases of the nvtx_cpu_topdown recipe, see the nvtx_cpu_topdown Recipe documentation.

### Common Issues

#### Reducing Overhead Caused By Sampling

There are several ways to reduce overhead caused by sampling:

1. **Disable sampling entirely** -- Use the `--sampling=none` switch.

```bash
nsys profile --sample=none -- myApp
```

2. **Increase the sampling period** (reduce the sampling rate) -- Use the `--sampling-period` switch.

```bash
nsys profile --sampling-period=100000 -- myApp
```

3. **Stop collecting backtraces** -- Use the `--backtrace=none` switch or collect more efficient backtraces.

```bash
# Disable backtraces entirely
nsys profile --backtrace=none -- myApp

# Use more efficient LBR backtraces (if available)
nsys profile --backtrace=lbr -- myApp
```

4. **Reduce the number of backtraces collected per sample** -- See documentation for the `--samples-per-backtrace` switch.

#### Throttling

The Linux operating system enforces a maximum time to handle sampling interrupts. This means that if collecting samples takes more than a specified amount of time, the OS will throttle (i.e., slow down) the sampling rate to prevent the perf subsystem from causing too much overhead.

**Symptoms:**
- Sampling data may become irregular even though the thread is very busy
- Irregular intervals of sampling tickmarks on the thread timeline
- The number of times a collection throttled is provided in the Nsight Systems GUI's Diagnostics messages

**Remediation:**
If a collection throttles frequently (e.g., 1000s of times), increasing the sampling period should help reduce throttling.

**Reset the OS Maximum Sampling Rate:**

When throttling occurs, the OS sets a new (lower) maximum sampling rate in the procfs. This value must be reset before the sampling rate can be increased again:

```bash
echo '100000' | sudo tee /proc/sys/kernel/perf_event_max_sample_rate
```

#### Sample Intervals Are Irregular

**Common Questions:**
- My samples are not periodic -- why?
- My samples are clumped up -- why?
- There are gaps in between the samples -- why?

**Likely Reasons:**

1. **Throttling** -- As described above, the OS may throttle sampling when overhead is too high.

2. **Paranoid level is set to 2** -- If the paranoid level is set to 2, anytime the workload makes a system call and spends time executing kernel mode code, samples will not be collected and there will be gaps in the sampling data.

3. **Non-periodic sampling trigger** -- If the trigger event is not periodic (for example, the "Instructions Retired" event), sample collection will primarily occur when cache misses are occurring.

#### No CPU Profiling Data Is Collected

There are a few common issues that cause CPU profiling data to not be collected:

1. **System requirements are not met** -- Check your system settings with the `nsys status --environment` command and see the System Requirements section above.

2. **Docker container without perf_event_open** -- By default, Docker containers prevent the `perf_event_open` syscall from being utilized. To override this behavior, launch the Docker with the `--privileged` switch or modify the Docker's seccomp settings.

3. **Docker container running Ubuntu 20+ on a CentOS host with old kernel** -- If profiling a workload in a Docker container running Ubuntu 20+ on top of a host system running CentOS with a kernel version < 3.10.0-693, the `nsys status --environment` command may incorrectly indicate that CPU profiling is supported. The host OS kernel version determines if CPU profiling is allowed and a CentOS host with a version < 3.10.0-693 is too old.

---

## Python Profiling

Nsight Systems has several features to enhance users optimizing their Python code.

**Important Note Regarding Output Buffering:**
You may find that all of your Python application output comes at the end of the run instead of as events happen. Python will change the buffering of stdout depending on whether it points to a tty or something else. Nsight Systems redirects the application stdout to a pipe to demultiplex stdout to both a file and the terminal. As a side effect, it makes Python change stdout buffering from line-buffered to page-buffered.

**Workaround:**

```bash
# Use python -u option
nsys profile -- python -u my_script.py

# Or set the PYTHONUNBUFFERED environment variable
PYTHONUNBUFFERED=1 nsys profile -- python my_script.py
```

### Python Backtrace Sampling

Nsight Systems for Arm server (SBSA) platforms, x86 Linux and Windows targets, is capable of periodically capturing Python backtrace information.

**Requirements:**
- Python interpreters of version **3.9 or later**
- CPython interpreter
- Capturing Python backtraces is done in periodic samples
- Selected frequency ranging from **1Hz - 2KHz** with a default value of **1KHz**

**When profiling Python-only workflows**, consider disabling the CPU sampling option to reduce overhead.

**CLI Configuration:**

```bash
# Enable Python backtrace sampling with default frequency (1KHz)
nsys profile --python-sampling=true -- python my_script.py

# Set custom sampling frequency
nsys profile --python-sampling=true --python-sampling-frequency=500 -- python my_script.py
```

| Switch | Parameters | Description |
|--------|-----------|-------------|
| `--python-sampling` | true, false | Enable/disable Python backtrace sampling |
| `--python-sampling-frequency` | 1 - 2000 | Sampling frequency in Hz (default: 1000) |

**GUI Configuration:**
Select the "Collect Python backtrace samples" checkbox in the project settings.

### Python Functions Trace

Nsight Systems for Arm server (SBSA) platforms, x86 Linux and Windows targets, is capable of using NVTX to annotate Python functions. The Python source code does not require any changes.

**Requirements:**
- CPython interpreter, release **3.8 or later**

**Annotations Configuration:**
The annotations are configured in a JSON file. An example file is located in the Nsight Systems installation folder:

```
<target-platform-folder>/PythonFunctionsTrace/annotations.json
```

**Predefined Annotation Files:**

| Framework | File Location |
|-----------|--------------|
| PyTorch | `<target-platform-folder>/PythonFunctionsTrace/pytorch.json` |
| Dask | `<target-platform-folder>/PythonFunctionsTrace/dask.json` |

**Limitation:** Annotating a function from the module `__main__` is not supported.

**CLI Configuration:**

```bash
# Enable Python functions trace with custom annotations
nsys profile --python-functions-trace=/path/to/annotations.json -- python my_script.py

# Using predefined PyTorch annotations
nsys profile --python-functions-trace=<nsys_install_dir>/<target-arch>/PythonFunctionsTrace/pytorch.json -- python my_script.py
```

| Switch | Parameters | Description |
|--------|-----------|-------------|
| `--python-functions-trace` | <json_file> | Path to JSON configuration file for function tracing |

**GUI Configuration:**
Select the "Python Functions trace" checkbox and specify the JSON file in the project settings.

**Example JSON Configuration:**

```json
{
    "functions": [
        {
            "module": "numpy",
            "function_name": "array"
        },
        {
            "module": "torch",
            "function_name": "tensor"
        }
    ]
}
```

### Python GIL Tracing

Nsight Systems for Arm server (SBSA) platforms, x86 Linux and Windows targets, is capable of tracing when Python threads are waiting to hold and holding the GIL (Global Interpreter Lock).

**Requirements:**
- CPython interpreter, release **3.9 or later**
- The Python source code does not require any changes
- **Not supported** on Python that was compiled with `Py_GIL_DISABLED=1` (See Python documentation for details)

**CLI Configuration:**

```bash
# Enable Python GIL tracing
nsys profile --trace=python-gil -- python my_script.py

# Combine with other traces
nsys profile --trace=cuda,osrt,python-gil -- python my_script.py
```

| Switch | Parameters | Description |
|--------|-----------|-------------|
| `--trace` | python-gil | Enable Python GIL tracing (combined with other trace values) |

**GUI Configuration:**
Select the "Trace GIL" checkbox under Python profiling options in the project settings.

### PyTorch Profiling

Nsight Systems for Arm server (SBSA) platforms, x86 Linux and Windows targets, is capable of automatically annotating common PyTorch operations with execution time ranges.

**Requirements:**
- CPython interpreter, release **3.8 or later**
- The Python source code does not require any changes

**CLI Options:**

| Switch Value | Description |
|-------------|-------------|
| `--pytorch=autograd-nvtx` | Enables `torch.autograd.profiler.emit_nvtx(record_shapes=False)` (implies `--trace=nvtx`) |
| `--pytorch=autograd-shapes-nvtx` | Enables `torch.autograd.profiler.emit_nvtx(record_shapes=True)` (implies `--trace=nvtx`) |
| `--pytorch=functions-trace` | Alias to `--python-functions-trace=<nsys_install_dir>/<target-arch>/PythonFunctionsTrace/pytorch.json`; provides additional annotations for PyTorch functions |

**Combining Options:**
`autograd-nvtx` and `autograd-shapes-nvtx` options can be combined with the `functions-trace` option by adding them separated by a comma.

**Example Commands:**

```bash
# Enable PyTorch autograd NVTX annotations
nsys profile --pytorch=autograd-nvtx -- python train.py

# Enable PyTorch autograd with shapes
nsys profile --pytorch=autograd-shapes-nvtx -- python train.py

# Combine autograd with functions trace
nsys profile --pytorch=autograd-nvtx,function-trace -- python train.py

# Use PyTorch functions trace only
nsys profile --pytorch=functions-trace -- python train.py

# Full PyTorch profiling
nsys profile --trace=cuda,osrt,nvtx --pytorch=autograd-shapes-nvtx \
    --python-sampling=true -- python train.py
```

### Dask Profiling

Nsight Systems for Arm server (SBSA) platforms, x86 Linux and Windows targets, is capable of automatically annotating common Dask functions with execution time ranges.

**Requirements:**
- CPython interpreter, release **3.8 or later**
- The Python source code does not require any changes

**CLI Configuration:**

```bash
# Enable Dask functions trace
nsys profile --dask=functions-trace -- python my_dask_app.py
```

**What `--dask=functions-trace` does:**
- Sets `--python-functions-trace=<nsys_install_dir>/<target-arch>/PythonFunctionsTrace/dask.json`
- Renames relevant threads to 'Dask Worker' and 'Dask Scheduler'

**Customization:**
The `dask.json` file can be modified to include additional functions to be traced from any Python module.

| Switch | Parameters | Description |
|--------|-----------|-------------|
| `--dask` | functions-trace | Enable Dask functions trace with thread renaming |

**Example JSON for Custom Dask Tracing:**

```json
{
    "functions": [
        {
            "module": "dask",
            "function_name": "compute"
        },
        {
            "module": "dask",
            "function_name": "persist"
        }
    ]
}
```

---

## Quick Reference: CPU Profiling CLI Options

| Short | Long | Parameters | Default | Description |
|-------|------|------------|---------|-------------|
| | `--backtrace` | auto, fp, lbr, dwarf, none | auto | Select the backtrace method to use while sampling |
| | `--cpuctxsw` | process-tree, system-wide, none | process-tree | Trace OS thread scheduling activity |
| | `--cpu-core-events` | help, or event IDs | '2' (Instructions Retired) | Select CPU Core events to sample |
| | `--cpu-core-metrics` | 0,1,2,...,none | none | Collect metrics on the CPU core (Grace only) |
| | `--cpu-socket-events` | help, or event IDs | none | Select Uncore CPU Socket events to sample |
| | `--cpu-socket-metrics` | 0,1,2,...,none | none | Collect Uncore metrics on the CPU socket (Grace only) |
| | `--event-sample` | none, system-wide | none | Enable event sampling |
| | `--event-sampling-frequency` | frequency | N/A | Set the event sampling frequency |
| | `--sample` | process-tree, system-wide, none | process-tree | CPU IP/backtrace sampling mode |
| | `--sampling-period` | period value | N/A | Set the sampling period |
| | `--samples-per-backtrace` | count | N/A | Number of backtraces per sample |
| | `--python-sampling` | true, false | false | Enable Python backtrace sampling |
| | `--python-sampling-frequency` | 1 - 2000 | 1000 | Python sampling frequency in Hz |
| | `--python-functions-trace` | <json_file> | none | Path to JSON configuration for Python function tracing |
| | `--pytorch` | autograd-nvtx, autograd-shapes-nvtx, functions-trace | none | PyTorch profiling mode |
| | `--dask` | functions-trace | none | Dask profiling mode |
