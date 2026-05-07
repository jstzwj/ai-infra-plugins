# Nsight Systems GUI Report Analysis Reference

This document provides comprehensive reference material for using the Nsight Systems GUI to profile targets, analyze reports, navigate the timeline, and use the various analysis views. It covers profiling from the GUI for Linux, Windows, QNX, and JupyterLab targets, as well as all report analysis features including timeline navigation, function tables, events view, multi-report views, and flame graph generation.

---

## Table of Contents

1. [Profiling from the GUI](#profiling-from-gui)
2. [Profiling Linux Targets from the GUI](#profiling-linux)
   - [Connecting to the Target Device](#connecting-linux)
   - [Open Ports](#open-ports)
   - [Kernel Version Number](#kernel-version)
   - [Netcat Requirement](#netcat-requirement)
3. [System-Wide Profiling Options](#system-wide-options)
4. [Target Sampling Options](#target-sampling)
5. [Hotkey Trace Start/Stop](#hotkey-trace)
6. [Launching Processes](#launching-processes)
7. [Profiling Windows Targets from the GUI](#profiling-windows)
   - [Remoting to a Windows Based Machine](#remoting-windows)
   - [Hotkey Trace Start/Stop (Windows)](#hotkey-windows)
   - [Changing the Default Hotkey Binding](#changing-hotkey)
   - [Target Sampling Options on Windows](#sampling-windows)
   - [Thread Activity Option](#thread-activity)
   - [Symbol Locations](#symbol-locations)
8. [Profiling QNX Targets from the GUI](#profiling-qnx)
9. [Profiling within JupyterLab](#profiling-jupyterlab)
10. [Report Management](#report-management)
    - [ImportNvtxt Commands](#importnvtxt)
    - [Generating a New Report](#generating-report)
    - [Opening an Existing Report](#opening-report)
    - [Sharing a Report File](#sharing-report)
    - [Report Tab](#report-tab)
11. [Analysis Summary View](#analysis-summary)
12. [Diagnostics Summary View](#diagnostics-summary)
13. [Symbol Resolution Logs View](#symbol-resolution)
14. [Timeline View](#timeline-view)
    - [Timeline Navigation](#timeline-navigation)
    - [Zoom and Scroll](#zoom-scroll)
    - [Timeline/Events Correlation](#timeline-events-correlation)
    - [Row Height](#row-height)
    - [Row Percentage](#row-percentage)
    - [Timeline Options](#timeline-options)
15. [Events View](#events-view)
16. [Function Table Modes](#function-table-modes)
    - [Top-Down View](#top-down-view)
    - [Bottom-Up View](#bottom-up-view)
    - [Flat View](#flat-view)
    - [Function Table Notes](#function-table-notes)
17. [Filter Dialog](#filter-dialog)
18. [Example: Using Timeline with Function Table](#timeline-function-table-example)
19. [Backtraces](#backtraces)
20. [Multi-Report Timeline Views](#multi-report-views)
    - [Viewing Multiple Reports in Separate Panes](#separate-panes)
    - [Viewing Multiple Reports in the Same Timeline](#same-timeline)
    - [Time Synchronization](#time-sync)
    - [Timeline Hierarchy](#timeline-hierarchy)
    - [Example: MPI](#mpi-example)
    - [Limitations](#multi-report-limitations)
21. [Add-on Graphs - Flame Graph](#flame-graph)

---

## Profiling from the GUI

<a id="profiling-from-gui"></a>

Nsight Systems provides a graphical user interface for launching profiling sessions on local and remote targets. The GUI simplifies the collection process by providing dialogs for all configurable options.

---

## Profiling Linux Targets from the GUI

<a id="profiling-linux"></a>

### Connecting to the Target Device

<a id="connecting-linux"></a>

Nsight Systems provides a simple interface to profile on localhost or manage multiple connections to Linux or Windows based devices via SSH. The network connections manager can be launched through the device selection dropdown.

On **x86_64** and **Tegra** platforms, the dialog has simple controls that allow adding, removing, and modifying connections.

#### Security Notice

SSH is only used to establish the initial connection to a target device, perform checks, and upload necessary files. The actual profiling commands and data are transferred through a raw, unencrypted socket. **Nsight Systems should not be used in a network setup where attacker-in-the-middle attack is possible, or where untrusted parties may have network access to the target device.**

While connecting to the target device, you will be prompted to input the user's password. Note that if you choose to remember the password, it will be stored in **plain text** in the configuration file on the host. Stored passwords are bound to the public key fingerprint of the remote device.

#### No Authentication Option

The No authentication option is useful for devices configured for passwordless login using root username. To enable such a configuration, edit the file `/etc/ssh/sshd_config` on the target and specify the following option:

```ini
PermitRootLogin yes
```

Then set empty password using `passwd` and restart the SSH service with `service ssh restart`.

### Open Ports

<a id="open-ports"></a>

The Nsight Systems daemon requires **port 22** and **port 45555** to be open for listening. You can confirm that these ports are open with the following command:

```bash
sudo firewall-cmd --list-ports --permanent
sudo firewall-cmd --reload
```

To open a port use the following command (skip `--permanent` option to open only for this session):

```bash
sudo firewall-cmd --permanent --add-port 45555/tcp
sudo firewall-cmd --reload
```

Likewise, if you are running on a cloud system, you must open port 22 and port 45555 for ingress.

### Kernel Version Number

<a id="kernel-version"></a>

To check for the version number of the kernel support of Nsight Systems on a target device, run the following command on the remote device:

```bash
cat /proc/quadd/version
```

Minimal supported version is **1.82**.

### Netcat Requirement

<a id="netcat-requirement"></a>

Presence of the Netcat command (`nc`) is required on the target device. For example, on Ubuntu this package can be installed using the following command:

```bash
sudo apt-get install netcat-openbsd
```

---

## System-Wide Profiling Options

<a id="system-wide-options"></a>

System-wide profiling captures activity from all processes running on the target, not just the launched application. The options are configurable from the GUI in the project settings.

---

## Target Sampling Options

<a id="target-sampling"></a>

Target sampling behavior is somewhat different for Nsight Systems Workstation Edition and Nsight Systems Embedded Platforms Edition.

The sampling configuration controls how CPU profiling data is collected:

| Parameter | Description |
|---|---|
| **Sampling frequency** | Number of samples per second per CPU core. Available values: 100 Hz, 1 KHz (default), 2 KHz, 4 KHz, 8 KHz |
| **Sample backtraces** | Collect call stack backtraces at each sample |
| **Wait for CPU** | Include time threads spent waiting to be scheduled on a CPU |

---

## Hotkey Trace Start/Stop

<a id="hotkey-trace"></a>

Nsight Systems Workstation Edition can use hotkeys to control profiling. Press the hotkey to start and/or stop a trace session from within the target application's graphic window. This is useful when tracing games and graphic applications that use fullscreen display. In these scenarios, switching to Nsight Systems' UI would unnecessarily introduce the window manager's footprint into the trace.

To enable the use of Hotkey, check the **Hotkey checkbox** in the project settings page.

The **default hotkey is F12**.

---

## Launching Processes

<a id="launching-processes"></a>

Nsight Systems can launch new processes for profiling on target devices. The profiler ensures that all environment variables are set correctly to successfully collect trace information.

The **Edit arguments...** link will open an editor window, where every command line argument is edited on a separate line. This is convenient when arguments contain spaces or quotes.

---

## Profiling Windows Targets from the GUI

<a id="profiling-windows"></a>

Profiling on Windows devices is similar to the profiling on Linux devices. The major differences on the platforms are listed below.

### Remoting to a Windows Based Machine

<a id="remoting-windows"></a>

To perform remote profiling to a target Windows based machine, install and configure an **OpenSSH Server** on the target machine.

### Hotkey Trace Start/Stop (Windows)

<a id="hotkey-windows"></a>

Nsight Systems Workstation Edition can use hotkeys to control profiling. Press the hotkey to start and/or stop a trace session from within the target application's graphic window. This is useful when tracing games and graphic applications that use fullscreen display.

To enable the use of Hotkey, check the **Hotkey checkbox** in the project settings page. The default hotkey is **F12**.

#### Changing the Default Hotkey Binding

<a id="changing-hotkey"></a>

A different hotkey binding can be configured by setting the `HotKeyIntValue` configuration field in the `config.ini` file.

Set the decimal numeric identifier of the hotkey you would like to use for triggering start/stop from the target app graphics window. The default value is **123** which corresponds to **0x7B**, or the **F12** key.

Virtual key identifiers are detailed in MSDN's Virtual-Key Codes documentation.

Note that you must convert the hexadecimal values detailed in the page to their decimal counterpart before using them in the file. For example, to use the **F1** key as a start/stop trace hotkey, use the following settings in the config.ini file:

```ini
HotKeyIntValue=112
```

### Target Sampling Options on Windows

<a id="sampling-windows"></a>

Nsight Systems can sample one process tree. Sampling here means interrupting each processor periodically. The sampling rate is defined in the project settings and is either **100 Hz**, **1 KHz** (default value), **2 KHz**, **4 KHz**, or **8 KHz**.

### Thread Activity Option

<a id="thread-activity"></a>

On Windows, Nsight Systems can collect thread activity of one process tree. Collecting thread activity means that each thread context switch event is logged and (optionally) a backtrace is collected at the point that the thread is scheduled back for execution. Thread states are displayed on the timeline.

If it was collected, the thread backtrace is displayed when hovering over a region where the thread execution is blocked.

### Symbol Locations

<a id="symbol-locations"></a>

Symbol resolution happens on host, and therefore does not affect performance of profiling on the target.

Press the **Symbol locations...** button to open the Configure debug symbols location dialog. Use this dialog to specify:

- Paths of PDB files
- Symbol servers
- The location of the local symbol cache

To use a symbol server:

1. Install **Debugging Tools for Windows**, a part of the Windows 10 SDK.
2. Add the symbol server URL using the **Add Server** button.
3. Information about Microsoft's public symbol server, which enables getting Windows operating system related debug symbols, can be found in the Microsoft documentation.

---

## Profiling QNX Targets from the GUI

<a id="profiling-qnx"></a>

Profiling on QNX devices is similar to the profiling on Linux devices. The major differences are listed below:

- **Backtrace sampling is not supported**. Instead backtraces are collected for long OS runtime libraries calls. Refer to the OS Runtime Libraries Trace section for detailed documentation.
- **CUDA support is limited to CUDA 9.0+**.
- **Filesystem on QNX device might be mounted read-only**. In that case Nsight Systems is not able to install target-side binaries required to run the profiling session. Please make sure that target filesystem is writable before connecting to QNX target. For example, make sure the following command works:

```bash
echo XX > /xx && ls -l /xx
```

---

## Profiling within JupyterLab

<a id="profiling-jupyterlab"></a>

The JupyterLab Nsight extension integrates Nsight Systems profiling into JupyterLab for profiling of Jupyter notebook cells. CUDA kernels launched by the cells as well as CUDA and Python code execution can be profiled and analyzed.

For more information and to install the extension, go to **JupyterLab Nsight extension on PyPI**.

---

## Report Management

<a id="report-management"></a>

### ImportNvtxt Commands

<a id="importnvtxt"></a>

Nsight Systems supports importing NVTXT (external counter data) files. Time stamps can be based on different clock sources:

| Timestamp Type | Description |
|---|---|
| Global | Timestamps are considered to be nanoseconds |
| TSC | Timestamps use the Timestamp Counter |
| CNTVCT | Timestamps use the generic timer (CNTVCT) |

#### Help Message

```
-h [ --help ]
```

#### Info Command

Find out report's start and end time:

```bash
ImportNvtxt --cmd info -i [--input] arg
```

Example:

```
ImportNvtxt info Report.nsys-rep
Analysis start (ns) 83501026500000
Analysis end (ns)   83506375000000
```

#### Create Command

Create a report file using an existing NVTXT file:

```bash
ImportNvtxt --cmd create -n [--nvtxt] arg -o [--output] arg [-m [--mode] mode_name mode_args] [--target <Hw:Vm>] [--update_report_time]
```

Example:

```bash
ImportNvtxt --cmd create -n Sample.nvtxt -o Report.nsys-rep
```

The output will be a new generated report file which can be opened and viewed by Nsight Systems.

#### Merge Command

Merge an NVTXT file with an existing report file:

```bash
ImportNvtxt --cmd merge -i [--input] arg -n [--nvtxt] arg -o [--output] arg [-m [--mode] mode_name mode_args] [--target <Hw:Vm>] [--update_report_time]
```

Example:

```bash
ImportNvtxt --cmd merge -i Report.nsys-rep -n Sample.nvtxt -o NewReport.nsys-rep
```

#### Modes

Available modes for time conversion:

| Mode | Description |
|---|---|
| `lerp` | Insert with linear interpolation |
| `lin` | Insert with linear equation |

**lerp mode:**

```
--mode lerp --ns_a arg --ns_b arg [--nvtxt_a arg --nvtxt_b arg]
```

| Parameter | Description |
|---|---|
| `ns_a` | A nanoseconds value |
| `ns_b` | A nanoseconds value (greater than ns_a) |
| `nvtxt_a` | An nvtxt file's time unit value corresponding to ns_a nanoseconds |
| `nvtxt_b` | An nvtxt file's time unit value corresponding to ns_b nanoseconds |

If `nvtxt_a` and `nvtxt_b` are not specified, they are respectively set to the nvtxt file's minimum and maximum time value.

**lin mode:**

```
--mode lin --ns_a arg --freq arg [--nvtxt_a arg]
```

| Parameter | Description |
|---|---|
| `ns_a` | A nanoseconds value |
| `freq` | The nvtxt file's timer frequency |
| `nvtxt_a` | An nvtxt file's time unit value corresponding to ns_a nanoseconds |

If `nvtxt_a` is not specified, it is set to the nvtxt file's minimum time value.

**Common parameters:**

| Parameter | Description |
|---|---|
| `--target <Hw:Vm>` | Specify target id, e.g., `--target 0:1` |
| `--update_report_time` | Prolong report's profiling session time while merging if needed. Without this option all events outside the profiling session time window will be skipped during merging. |

Time values in `<filename.nvtxt>` are assumed to be nanoseconds if no mode is specified.

### Generating a New Report

<a id="generating-report"></a>

Users can generate a new report by stopping a profiling session. If a profiling session has been canceled, a report will **not** be generated, and all collected data will be discarded.

A new `.nsys-rep` file will be created and put into the same directory as the project file (`.qdproj`).

### Opening an Existing Report

<a id="opening-report"></a>

An existing `.nsys-rep` file can be opened using **File > Open...**.

### Sharing a Report File

<a id="sharing-report"></a>

Report files (`.nsys-rep`) are **self-contained** and can be shared with other users of Nsight Systems. The only requirement is that the **same or newer version** of Nsight Systems is always used to open report files.

Project files (`.qdproj`) are currently not shareable, since they contain full paths to the report files.

To quickly navigate to the directory containing the report file, right click on it in the Project Explorer, and choose **Show in folder...** in the context menu.

### Report Tab

<a id="report-tab"></a>

While generating a new report or loading an existing one, a new tab will be created. The most important parts of the report tab are:

| Component | Description |
|---|---|
| **View selector** | Allows switching between Multi-report view (absent for single reports), Analysis Summary, Timeline View, Diagnostics Summary, and Symbol Resolution Logs views |
| **Timeline** | This is where all charts are displayed |
| **Function table** | Located below the timeline, it displays statistical information about functions in the target application in multiple ways |
| **Zoom slider** | Allows you to vertically zoom the charts on the timeline |

---

## Analysis Summary View

<a id="analysis-summary"></a>

This view shows a summary of the profiling session. In particular, it is useful to review the project configuration used to generate this report. Information from this view can be selected and copied using the mouse cursor.

Key information displayed includes:

- Project configuration settings
- Application name and arguments
- Duration of the profiling session
- Target device information
- UTC time at t=0
- TSC value at t=0
- Report alignment source (for multi-report views)

---

## Diagnostics Summary View

<a id="diagnostics-summary"></a>

This view shows important messages. Some of them were generated during the profiling session, while some were added while processing and analyzing data in the report. Messages can be one of the following types:

| Type | Description |
|---|---|
| **Informational messages** | General information about the profiling session |
| **Warnings** | Non-critical issues that may affect data quality |
| **Errors** | Critical issues that may have affected data collection |

To draw attention to important diagnostics messages, a **summary line** is displayed on the timeline view in the top right corner.

Information from this view can be selected and copied using the mouse cursor.

---

## Symbol Resolution Logs View

<a id="symbol-resolution"></a>

This view shows all messages related to the process of resolving symbols. It might be useful to debug issues when some of the symbol names in the symbols table of the timeline view are unresolved.

---

## Timeline View

<a id="timeline-view"></a>

The timeline view consists of two main controls: the timeline at the top, and a bottom pane that contains the events view and the function table. In some cases, when sampling of a process has not been enabled, the function table might be empty and hidden.

The bottom view selector sets the view that is displayed in the bottom pane.

### Timeline

Timeline is a versatile control that contains a tree-like hierarchy on the left, a line labels column in the center, and the corresponding charts on the right. The line labels column can be hidden by using the timeline options.

Contents of the hierarchy depend on the project settings used to collect the report. For example, if a certain feature has not been enabled, corresponding rows will not be shown on the timeline.

To generate a timeline screenshot without opening the full GUI, use the command:

```bash
nsys-ui.exe --screenshot filename.nsys-rep
```

Hovering over elements in the GUI will cause a tooltip to pop open as appropriate to give additional information, such as the parameters of that function call or the call stack. Tooltips can be copied by hovering and right clicking to bring up the **Copy Tooltip** option in the context menu.

### Timeline Navigation

<a id="timeline-navigation"></a>

#### Zoom and Scroll

<a id="zoom-scroll"></a>

At the upper right portion of the Nsight Systems GUI, there is a vertical slider that sets the vertical size of screen rows, and a magnifying glass that resets it to the original settings.

There are many ways to zoom and scroll horizontally through the timeline:

| Action | Description |
|---|---|
| Mouse wheel | Scroll or zoom depending on modifier key |
| Click and drag | Select a time range |
| `+` / `-` keys | Zoom in/out |
| Arrow keys | Pan left/right |
| `Ctrl+Home` | Reset to full trace view |

#### Timeline/Events Correlation

<a id="timeline-events-correlation"></a>

To display trace events in the Events View, right-click a timeline row and select the **Show in Events View** command. The events of the selected row and all of its sub-rows will be displayed in the Events View. Note that the events displayed will correspond to the current zoom in the timeline; zooming in or out will reset the event pane filter.

If a timeline row has been selected for display in the Events View, then double-clicking a timeline item on that row will automatically scroll the content of the Events View to make the corresponding events view item visible and select it. If that event has tool tip information, it will be displayed in the right hand pane.

Likewise, double-clicking on a particular instance in the Events View will highlight the corresponding event in the timeline.

#### Row Height

<a id="row-height"></a>

Several of the rows in the timeline use height as a way to model the percent utilization of resources. This gives the user insight into what is going on even when the timeline is zoomed all the way out.

Nsight Systems calculates the average occupancy for the period of time represented by a particular pixel width of screen. It then uses that average to set the top of the colored section. So, for instance, if 25% of that timeslice the kernel is active, the bar goes 25% of the distance to the top of the row.

In order to make the difference clear, if the percentage of the row height is non-zero, but would be represented by less than one vertical pixel, Nsight Systems displays it as one pixel high. The gray height represents the maximum usage in that time range.

This row height coding is used in the **CPU utilization**, **thread and process occupancy**, **kernel occupancy**, and **memory transfer activity** rows.

#### Row Percentage

<a id="row-percentage"></a>

The percentage shown in front of the stream indicates the proportion of context running time this particular stream takes:

```
% stream = 100.0 * streamUsage / contextUsage
```

Where:
- `streamUsage` = total amount of time this stream is active on GPU
- `contextUsage` = total amount of time all streams for this context are active on GPU

So "26% Stream 1" means that Stream 1 takes 26% of its context's total running time.

Total running time = sum of durations of all kernels and memory ops that run in this context.

### Timeline Options

<a id="timeline-options"></a>

We strongly recommend using the OS/Desktop defaults for size and color, but if you would like to set them for yourself, they are available using the **Tools > Options** dialog.

The above will change the options globally for this GUI. It's also possible to change some options for a particular open report. There is an **Options...** button near the View Selector.

This button will show a dialog that allows showing/hiding the following:

| Option | Description |
|---|---|
| **Correlation arrows** | Show/hide arrows linking correlated CPU and GPU events |
| **Line labels** | Show/hide the line labels column |
| **CPU occupancy chart** | Show/hide the CPU occupancy visualization |

By default, the timeline will be based on **session time**. If you would like to switch to **global time**, click on the small arrow at the top of the leftmost column to reveal the dropdown.

---

## Events View

<a id="events-view"></a>

The Events View provides a tabular display of the trace events. The view contents can be searched and sorted.

Double-clicking an item in the Events View automatically focuses the Timeline View on the corresponding timeline item.

API calls, GPU executions, and debug markers that occurred within the boundaries of a debug marker are displayed nested to that debug marker. Multiple levels of nesting are supported.

Events View recognizes these types of debug markers:

| Marker Type | Description |
|---|---|
| **NVTX** | NVIDIA Tools Extension markers |
| **Vulkan** | `VK_EXT_debug_marker` markers, `VK_EXT_debug_utils` labels |
| **PIX** | PIX events and markers |
| **OpenGL** | `KHR_debug` markers |

You can copy and paste from the Events View by highlighting rows, using Shift or Ctrl to enable multi-select. Right clicking on the selection will give you a copy option.

Pasting into text gives you a tab-separated view. Pasting into a spreadsheet properly copies into rows and columns.

---

## Function Table Modes

<a id="function-table-modes"></a>

The function table can work in three modes:

### Top-Down View

<a id="top-down-view"></a>

In this mode, expanding top-level functions provides information about the callee functions. One of the top-level functions is typically the `main` function of your application, or another entry point defined by the runtime libraries.

**Columns:**

| Column | Description |
|---|---|
| **Self** | The relative amount of time spent executing instructions of this particular function |
| **Total** | How much time has been spent executing this function, including all other functions called from this one. Total values of sibling rows sum up to the Total value of the parent row, or 100% for the top-level rows |

### Bottom-Up View

<a id="bottom-up-view"></a>

This is a reverse of the Top-Down view. On the top level, there are functions directly hit by the sampling profiler. To explore all possible call chains leading to these functions, you need to expand the subtrees of the top-level functions.

**Columns:**

| Column | Description |
|---|---|
| **Self (top-level)** | Shows how much time has been spent directly in this function. Self times of all top-level rows add up to 100% |
| **Self (children)** | Breaks down the value of the parent row based on the various call chains leading to that function. Self times of sibling rows add up to the value of the parent row |

### Flat View

<a id="flat-view"></a>

This view enumerates all functions ever observed by the profiler, even if they have never been directly hit, but just appeared somewhere on the call stack. This view typically provides a high-level overview of which parts of the code are CPU-intensive.

**Column:**

| Column | Description |
|---|---|
| **Flat** | Shows how much time this function has been anywhere on the call stack. Values in this column do not add up or have other significant relationships |

### When to Use Each View

Each of the views helps understand particular performance issues:

| Scenario | Recommended View |
|---|---|
| Finding specific bottleneck functions that can be optimized | **Bottom-Up** view -- examine the top few functions and expand them to understand contexts |
| Navigating the call tree, searching for algorithms that consume unexpectedly large CPU time | **Top-Down** view |
| Quickly assessing which high-level parts consume significant CPU time | **Flat** view |

> **Note:** If low-impact functions have been filtered out, values may not add up correctly to 100%, or to the value of the parent row. This filtering can be disabled.

### Filtering and the Symbols Table

Contents of the symbols table is tightly related to the timeline. Users can apply and modify filters on the timeline, and they will affect which information is displayed in the symbols table:

- **Per-thread filtering** -- Each thread that has sampling information associated with it has a checkbox next to it on the timeline. Only threads with selected checkboxes are represented in the symbols table.
- **Time filtering** -- A time filter can be set up on the timeline by pressing the left mouse button, dragging over a region of interest on the timeline, and then choosing **Filter by selection** in the dropdown menu. In this case, only sampling information collected during the selected time range will be used to build the symbols table.

> **Note:** If too little sampling data is being used to build the symbols table (for example, when the sampling rate is configured to be low, and a short period of time is used for time-based filtering), the numbers in the symbols table might not be representative or accurate in some cases.

### Function Table Notes

<a id="function-table-notes"></a>

#### Last Branch Records vs. Frame Pointers

Two of the mechanisms available for collecting backtraces are Intel Last Branch Records (LBRs) and frame pointers.

**LBRs** are used to trace every branch instruction via a limited set of hardware registers. They can be configured to generate backtraces but have finite depth based on the CPU's microarchitecture. LBRs are effectively free to collect but may not be as deep as you need.

**Frame pointers** only work when a binary is compiled with the `-fno-omit-frame-pointer` compiler switch. To determine if frame pointers are enabled on an x86_64 binary running on Linux, dump a binary's assembly code:

```bash
objdump -d [binary_file]
```

Look for this pattern at the beginning of all functions:

```asm
push   %rbp
mov    %rsp,%rbp
```

When frame pointers are available in a binary, full stack traces will be captured. Note that libraries that frequently used by applications and ship with the operating system, such as libc, are generated in release mode and therefore do not include frame pointers.

#### Kernel Samples

When an IP sample is captured while a kernel mode (i.e. operating system) function is executing, the sample will be shown with an address that starts with `0xffffffff` and map to the `[kernel.kallsyms]` module.

#### [vdso] Entries

Samples may be collected while a CPU is executing functions in the Virtual Dynamic Shared Object. In this case, the sample will be resolved (i.e., mapped) to the `[vdso]` module.

The vDSO ("virtual dynamic shared object") is a small shared library that the kernel automatically maps into the address space of all user-space applications. Applications usually do not need to concern themselves with these details as the vDSO is most commonly called by the C library.

#### [Unknown] Entries

When an address can not be resolved (i.e., mapped to a module), its address within the process's address space will be shown and its module will be marked as `[Unknown]`.

---

## Filter Dialog

<a id="filter-dialog"></a>

The Filter dialog provides options for managing the display of sampling data:

| Option | Description |
|---|---|
| **Collapse unresolved lines** | Useful if some of the binary code does not have symbols. In this case, subtrees that consist of only unresolved symbols get collapsed in the Top-Down view, since they provide very little useful information. |
| **Hide functions with CPU usage below X%** | Useful for large applications, where the sampling profiler hits lots of functions just a few times. To filter out the "long tail," which is typically not important for CPU performance bottleneck analysis, this checkbox should be selected. |

---

## Example: Using Timeline with Function Table

<a id="timeline-function-table-example"></a>

Here is an example walkthrough of using the timeline and function table with Instruction Pointer (IP)/backtrace Sampling Data.

### Timeline

When a collection result is opened in the Nsight Systems GUI, there are multiple ways to view the CPU profiling data:

In the timeline, **yellow-orange marks** can be found under each thread's timeline that indicate the moment an IP / backtrace sample was collected on that thread. Hovering the cursor over a mark will cause a tooltip to display the backtrace for that sample.

### Sampling Summary

Below the Timeline is a drop-down list with multiple options including Events View, Top-Down View, Bottom-Up View, and Flat View. All four of these views can be used to view CPU IP / backtrace sampling data.

If the Bottom-Up View is selected, the sampling summary is shown in the bottom half of the Timeline View screen. The summary includes the phrase "65,022 samples are used," indicating how many samples are summarized. By default, functions that were found in less than 0.5% of the samples are not shown. Use the filter button to modify that setting.

### Filtering

When sampling data is filtered, the Sampling Summary will summarize the selected samples. Samples can be filtered on an OS thread basis, on a time basis, or both.

Deselecting a checkbox next to a thread removes its samples from the sampling summary. Dragging the cursor over the timeline and selecting "Filter and Zoom In" chooses the samples during the time selected. The sample summary includes the phrase "0.35% (225 samples) of data is shown due to applied filters."

Click on the down arrow next to a thread and choose **Show Only This Thread** to deselect all threads except that thread.

If the Events View is selected, right-click on a specific thread and choose **Show in Events View**. The samples collected while that thread executed will be shown in the Events View. Double-clicking on a specific sample in the Events view causes the timeline to show when that sample was collected.

---

## Backtraces

<a id="backtraces"></a>

To understand the code path used to get to a specific function shown in the sampling summary, right-click on a function and select **Expand**.

The backtrace shows the full call chain leading to the function. The `[Max depth]` string marks the end of the collected backtrace.

By default, backtraces with less than 0.5% of the total backtraces are hidden. This behavior can make the percentage results hard to understand. If all backtraces are shown (i.e., the filter is disabled), the results look very different and the numbers add up as expected. To disable the filter, click on the **Filter...** button and uncheck the **Hide functions with CPU usage below X%** checkbox.

When backtraces are collected, the whole sample (IP and backtrace) is handled as a single sample. If two samples have the exact same IP and backtrace, they are summed in the final results. If two samples have the same IP but a different backtrace, they will be shown as having the same leaf (i.e., IP) but a different backtrace.

When backtraces end, they are marked with the `[Max depth]` string (unless the backtrace can be traced back to its origin; e.g., `__libc_start_main`) or the backtrace breaks because an IP cannot be resolved.

---

## Multi-Report Timeline Views

<a id="multi-report-views"></a>

### Viewing Multiple Reports in Separate Panes

<a id="separate-panes"></a>

You have the option of looking at two or more Nsight Systems results files in separate panes. To do so:

1. Open each report in a tab.
2. Grab one of the tabs and undock it.
3. When you hover with the cursor in the middle of the GUI, you will see options for where to dock the pane.
4. Multiple reports can be docked in the window.

### Viewing Multiple Reports in the Same Timeline

<a id="same-timeline"></a>

You can open several reports in a single timeline. This could be done using one of these methods:

- **File > Open...** in the main menu, and select several report files.
- **File > New multi-report view** in the main menu, add report files that you want to open in the Multi-report view, and click the **Apply** button.

Multi-report view contains a simple editor that allows you to add/remove report files and will load them all on a single timeline after applying that set of reports.

When reports are loaded, you can use the View Selector to open the Multi-report view again, change the set of reports, and click on **Apply** button to reload the timeline with the new set of reports.

The selected set of reports can be saved as a Multi-report view document and could be opened later to load the same set again.

### Time Synchronization

<a id="time-sync"></a>

When multiple reports are loaded into a single timeline, timestamps between them need to be adjusted, such that events that happened at the same time appear to be aligned.

#### UTC-Based Synchronization (Default)

Nsight Systems can automatically adjust timestamps based on UTC time recorded around the collection start time. This method is used by default when other more precise methods are not available.

This time can be seen as **UTC time at t=0** in the Analysis Summary page of the report file. Refer to your OS documentation to learn how to sync the software clock using the Network Time Protocol (NTP).

**NTP-based time synchronization is not very precise**, with the typical errors on the scale of one to tens of milliseconds.

#### TSC-Based Synchronization

Reports collected on the same physical machine can use synchronization based on **Timestamp Counter (TSC)** values. These are platform-specific counters, typically accessed in user space applications using the RDTSC instruction on x86_64 architecture, or by reading the CNTVCT register on Arm64.

Their values converted to nanoseconds can be seen as **TSC value at t=0** in the Analysis Summary page of the report file. Reports synchronized using TSC values can be aligned with **nanoseconds-level precision**.

TSC-based time synchronization is activated automatically, when Nsight Systems detects that reports come from same target and that the same TSC value corresponds to very close UTC times. Targets are considered to be the same when either:

- Explicitly set environment variables `NSYS_HW_ID` are the same for both reports, OR
- Target hostnames are the same and `NSYS_HW_ID` is not set for either target

The difference between UTC and TSC time offsets must be below **1 second** to choose TSC-based time synchronization.

To find out which synchronization method was used, navigate to the Analysis Summary tab of an added report and check the **Report alignment source** property of a target. Note that the first report will not have this parameter.

> **Important:** When loading multiple reports into a single timeline, it is always advisable to first check that time synchronization looks correct, by zooming into synchronization or communication events that are expected to be aligned.

### Timeline Hierarchy

<a id="timeline-hierarchy"></a>

When reports are added to the same timeline Nsight Systems will automatically line them up by timestamps. If you want Nsight Systems to also recognize matching process or hardware information, you will need to set environment variables `NSYS_SYSTEM_ID` and `NSYS_HW_ID` at the time of report collection.

When loading a pair of report files into the same timeline, they will be merged in one of the following configurations:

| Configuration | Description |
|---|---|
| **Different hardware** | Used when reports are coming from different physical machines, and no hardware resources are shared. This mode is used when neither `NSYS_HW_ID` or `NSYS_SYSTEM_ID` is set and target hostnames are different or absent. Can be additionally signalled by specifying different `NSYS_HW_ID` values. |
| **Different systems, same hardware** | Used when reports are collected on different virtual machines (VMs) or containers on the same physical machine. To activate this mode, specify the same value of `NSYS_HW_ID` when collecting the reports. |
| **Same system** | Used when reports are collected within the same operating system (or container) environment. In this mode a process identifier (PID) 100 will refer to the same process in both reports. To manually activate this mode, specify the same value of `NSYS_SYSTEM_ID` when collecting the reports. This mode is automatically selected when target hostnames are the same and neither `NSYS_HW_ID` or `NSYS_SYSTEM_ID` is provided. |

### Example: MPI

<a id="mpi-example"></a>

A typical scenario is when a computing job is run using one of the MPI implementations. Each instance of the app can be profiled separately, resulting in multiple report files.

```bash
# Run MPI job without the profiler:
mpirun <mpirun-options> ./myApp

# Run MPI job and profile each instance of the application:
mpirun <mpirun-options> nsys profile -o report-%p <nsys-options> ./myApp
```

When each MPI rank runs on a different node, the command above works fine, since the default pairing mode (different hardware) will be used.

When all MPI ranks run on localhost only, use this command (value "A" was chosen arbitrarily, it can be any non-empty string):

```bash
NSYS_SYSTEM_ID=A mpirun <mpirun-options> nsys profile -o report-%p <nsys-options> ./myApp
```

For convenience, the MPI rank can be encoded into the report filename:

- **Open MPI**: Use `OMPI_COMM_WORLD_RANK` environment variable

```bash
mpirun <mpirun-options> nsys profile -o report-%q{OMPI_COMM_WORLD_RANK} <nsys-options> ./myApp
```

- **MPICH-based implementations**: Set the environment variable `PMI_RANK`
- **Slurm (srun)**: Provides the global MPI rank in `SLURM_PROCID`

### Limitations on Syncing Multiple Reports in Timeline

<a id="multi-report-limitations"></a>

- Only report files collected with Nsight Systems version **2021.3** and newer are fully supported.
- Sequential reports collected in a single CLI profiling session cannot be loaded into a single timeline yet.

---

## Add-on Graphs - Flame Graph

<a id="flame-graph"></a>

The generation of Flame Graphs from Nsight Systems reports is not a built-in feature, but it is possible to create such graphs from Nsight Systems reports with the script `stackcollapse_nsys.py` located at `<nsys-install-dir>/<host-folder>/Scripts/Flamegraph/`. There is also a `README.md` file at that location with additional usage details.

### Requirements

- `flamegraph.pl` from Brendan Gregg's FlameGraph GitHub repository
- Perl

### Usage

Generating flamegraph from Nsight Systems report file on **Linux**:

```bash
python3 stackcollapse_nsys.py report.nsys-rep | ./flamegraph.pl > result_flamegraph.svg
```

Generating flamegraph from Nsight Systems report file on **Windows**:

```powershell
PowerShell -Command "python stackcollapse_nsys.py report.nsys-rep | perl flamegraph.pl > result_flamegraph.svg"
```

The script exports the report to SQLite, queries the CPU samples and passes them as input to `flamegraph.pl`.

### Parameters

| Short | Long | Default | Description |
|---|---|---|---|
| `--nsys` | - | Current Nsight Systems CLI installation location | Path to the Nsight Systems CLI directory |
| `-o` | `--out` | Output is written to stdout | Path to a result file containing data suitable for `flamegraph.pl` |
| - | `--full_function_names` | `False` | Use full function names with return type, arguments and expanded templates, if available |

By default, the script tries to shorten function definitions (eliminating return type, arguments and templates). In some complex cases shortening may fail and return a full function definition. To disable shortening, define the `--full_function_names=False` argument.

### Example

Here is an example of a Flame Graph generated from an Nsight Systems report. The program was a debug build of GROMACS, running on two ranks, each running two OpenMP threads.

---

## Keyboard Shortcuts Summary

| Shortcut | Action |
|---|---|
| `Ctrl+O` | Open report |
| `Ctrl+W` | Close current tab |
| `Ctrl+Home` | Reset zoom to full trace |
| `Ctrl+G` | Go to time |
| `Ctrl+F` | Find event |
| `Ctrl+Plus` | Zoom in |
| `Ctrl+Minus` | Zoom out |
| `Ctrl+Shift+Z` | Zoom to selection |
| `Tab` | Switch between views |
| `Escape` | Cancel current selection |
| Double-click event | Focus timeline on event / highlight in Events View |
