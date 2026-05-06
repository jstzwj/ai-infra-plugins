# Nsight Systems GUI Report Analysis Reference

## Table of Contents

- [Profiling from the GUI](#profiling-from-the-gui)
- [System-Wide Profiling Options](#system-wide-profiling-options)
- [Target Sampling Options](#target-sampling-options)
- [Hotkey Trace Start/Stop](#hotkey-trace-startstop)
- [Launching Processes](#launching-processes)
- [Windows Target Profiling](#windows-target-profiling)
- [QNX Target Profiling](#qnx-target-profiling)
- [Generating, Opening, and Sharing Reports](#generating-opening-and-sharing-reports)
- [Report Tab Structure](#report-tab-structure)
- [Analysis Summary View](#analysis-summary-view)
- [Diagnostics Summary View](#diagnostics-summary-view)
- [Symbol Resolution Logs](#symbol-resolution-logs)
- [Timeline View](#timeline-view)
- [Timeline Options](#timeline-options)
- [Events View](#events-view)
- [Function Table Modes](#function-table-modes)
- [Backtraces Explanation](#backtraces-explanation)
- [Multi-Report Timeline Views](#multi-report-timeline-views)
- [Flame Graph Generation](#flame-graph-generation)

---

## Profiling from the GUI

Nsight Systems provides a graphical user interface for launching profiling sessions on local and remote targets. The GUI simplifies the collection process by providing dialogs for all configurable options.

### Connecting to a Target

1. Launch Nsight Systems GUI (`nsys-ui` on Linux, Nsight Systems from Start Menu on Windows).
2. Create a new project or open an existing one.
3. In the **Target** field, specify the connection:
   - **Local target**: Leave as `localhost` or select from the dropdown.
   - **Remote target**: Enter the hostname or IP address.
4. Configure the connection method:

| Connection Method | Description |
|---|---|
| **SSH** | Default method for Linux-to-Linux and Windows-to-Linux profiling. Uses the system SSH client. |
| **SSH with key** | Provide a private key file for authentication instead of password. |
| **Custom** | Use a custom connection script for non-standard setups. |

### SSH Connection Details

- Nsight Systems uses SSH to deploy the CLI binary to the remote target if it is not already installed.
- The default SSH port is `22`. You can specify a custom port in the hostname field using the format `hostname:port`.
- SSH tunneling is used for communication between the GUI and the target CLI.

### Security Considerations

- The GUI transfers the `nsys` CLI binary to the target via SCP if the target does not have Nsight Systems installed.
- Ensure the SSH user has sufficient permissions to run profiling workloads.
- For containerized environments, ensure the target has the necessary capabilities (see [Container Support](10-containers-migration.md)).

### Port Configuration

- Nsight Systems uses ephemeral ports for GUI-to-target communication.
- If a firewall is present, ensure the specified port range is open.
- You can configure the port range in **File > Preferences > Connection**.

### Kernel Version Requirements

- Linux targets require kernel version 3.10 or later for basic profiling.
- CPU sampling (backtrace) requires kernel version 4.3 or later.
- The target must support `perf_event_open` system call for sampling-based profiling.

---

## System-Wide Profiling Options

System-wide profiling captures activity from all processes running on the target, not just the launched application.

| Option | Description | Default |
|---|---|---|
| **Duration** | How long to profile (seconds). `0` means profile until manually stopped. | 0 |
| **CPU sampling** | Enable/disable CPU IP and backtrace sampling. | Enabled |
| **CPU context switches** | Record thread context switch events. | Enabled |
| **GPU metrics** | Collect GPU utilization and memory metrics. | Disabled |
| **PCIe metrics** | Collect PCIe bandwidth metrics. | Disabled |
| **Network metrics** | Collect network interface throughput metrics. | Disabled |
| **Power/energy metrics** | Collect power consumption data. | Disabled |
| **Clock frequency metrics** | Collect CPU and GPU clock frequency data. | Disabled |

### System-Wide Profiling Workflow

1. Select **Profile > System Wide** from the menu.
2. Configure target connection and options.
3. Click **Start** to begin collection.
4. Click **Stop** (or use hotkey) to end collection.
5. The report is automatically transferred to the GUI host.

---

## Target Sampling Options

The sampling configuration controls how CPU profiling data is collected.

| Parameter | Description | Default |
|---|---|---|
| **Sampling frequency** | Number of samples per second per CPU core. | 1000 Hz |
| **Sample backtraces** | Collect call stack backtraces at each sample. | Enabled |
| **Wait for CPU** | Include time threads spent waiting to be scheduled on a CPU. | Enabled |

### Sampling Frequency Recommendations

| Frequency | Use Case | Overhead |
|---|---|---|
| 100 Hz | Long-running applications, minimal overhead | Very Low |
| 1000 Hz | General purpose profiling | Low |
| 5000 Hz | Detailed hotspot analysis | Moderate |
| 10000 Hz | Fine-grained per-function analysis | High |
| 20000 Hz+ | Maximum detail, short captures only | Very High |

---

## Hotkey Trace Start/Stop

Nsight Systems supports hotkeys to start and stop tracing on the target without GUI interaction.

### Linux Hotkeys

| Action | Default Hotkey | Notes |
|---|---|---|
| Start trace | `Ctrl+T` | Only works in the terminal where `nsys` is running |
| Stop trace | `Ctrl+T` | Toggles between start and stop |

To use hotkey-based tracing from the CLI:

```bash
nsys start --hotkey
# Press Ctrl+T to start tracing
# Press Ctrl+T again to stop tracing
nsys stop
```

### Windows Hotkeys

| Action | Default Hotkey | Notes |
|---|---|---|
| Start trace | `Ctrl+T` | Only in the console window where nsys is running |
| Stop trace | `Ctrl+T` | Toggles between start and stop |

### Notes

- Hotkeys only work in interactive terminal sessions. They are not supported in batch scripts or SSH sessions without a TTY.
- For non-interactive environments, use `nsys start` and `nsys stop` commands, or use NVTX range annotations to control trace windows programmatically.

---

## Launching Processes

When using the GUI to launch a process for profiling:

1. **Application Path**: Full path to the executable on the target system.
2. **Working Directory**: The directory from which the application will be launched.
3. **Command Line Arguments**: Arguments passed to the application.
4. **Environment Variables**: Key-value pairs set before launch.
5. **Run as**: User context (for remote profiling).

### Launch Mode Options

| Mode | Description |
|---|---|
| **Launch** | Start a new process and profile it from the beginning. |
| **Attach** | Attach to an already running process by PID. |
| **Launch with warmup** | Start the process, then begin tracing after a delay or on trigger. |
| **Profile entire system** | Profile all processes system-wide. |

### Environment Variable Examples

```bash
# Set CUDA device
CUDA_VISIBLE_DEVICES=0

# Enable NVTX in CUDA applications
NVTX_INJECTION64_PATH=libnvtx.so

# Set library path
LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

---

## Windows Target Profiling

### OpenSSH Requirements

Windows target profiling requires OpenSSH to be installed and running on the Windows target.

1. Install OpenSSH Server:
   ```powershell
   # In PowerShell as Administrator
   Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
   Start-Service sshd
   Set-Service -Name sshd -StartupType Automatic
   ```

2. Verify the service is running:
   ```powershell
   Get-Service sshd
   ```

3. Ensure the SSH user is a member of the `Performance Log Users` group or has Administrator privileges for full profiling capability.

### Symbol Locations

For proper function name resolution in CPU sampling:

- PDB files must be accessible on the target or through a symbol server.
- Configure symbol search paths in **File > Preferences > Symbols**.
- Supported symbol sources:
  - Local file system paths
  - Symbol servers (using `symsrv.dll`)
  - `_NT_SYMBOL_PATH` environment variable

### PDB File Handling

| Scenario | Solution |
|---|---|
| PDB next to EXE/DLL | Automatically found |
| PDB in a different directory | Add the directory to symbol search paths |
| PDB on a symbol server | Configure `_NT_ALT_SYMBOL_PATH` or use the GUI preference |
| Stripped PDB (public symbols only) | Limited function name resolution |

---

## QNX Target Profiling

Nsight Systems supports profiling on QNX Neutrino RTOS targets.

### Prerequisites

- QNX SDP 7.0 or later
- SSH or serial connection to the QNX target
- `nsys` binary for QNX (cross-compiled)

### Connection Setup

1. Select **Profile > New Project**.
2. Set the target platform to **QNX**.
3. Enter the target hostname/IP and credentials.
4. Nsight Systems will deploy the QNX binary via SSH.

### Limitations on QNX

- CPU sampling uses `devc-perf` interface instead of Linux `perf_event_open`.
- GPU tracing may have limited support depending on the GPU.
- Some metrics (PCIe, power) may not be available.

---

## Generating, Opening, and Sharing Reports

### Generating Reports

Reports are generated automatically after a profiling session completes. The output format is `.nsys-rep`.

From the CLI:
```bash
nsys profile -o my_report my_application
# Produces my_report.nsys-rep
```

From the GUI: Reports are created in the temporary directory and can be saved to a permanent location.

### Opening Reports

| Method | Description |
|---|---|
| **File > Open** | Open a `.nsys-rep` file from disk. |
| **Drag and drop** | Drag a `.nsys-rep` file onto the GUI window. |
| **Command line** | `nsys-ui report.nsys-rep` |
| **Auto-open** | After a profiling session, the report opens automatically. |

### Sharing Reports

- `.nsys-rep` files are self-contained and can be shared with other users.
- Recipients need Nsight Systems GUI (same or later version) to open the report.
- For version compatibility: newer GUI versions can open older reports, but older GUI versions may not open newer reports.
- Export to SQLite for programmatic analysis:
  ```bash
  nsys export -t sqlite -o report.sqlite report.nsys-rep
  ```

### Report File Sizes

| Profiling Type | Typical Size |
|---|---|
| Short CPU-only trace | 1-10 MB |
| GPU + CPU trace (1 minute) | 10-100 MB |
| System-wide with metrics | 100 MB - 1 GB |
| Long traces with sampling | 1 GB+ |

---

## Report Tab Structure

When a report is opened, it is displayed in a tab with the following structure:

### View Selector (Left Pane)

The left pane contains the view selector with the following sections:

| Section | Description |
|---|---|
| **Analysis Summary** | High-level overview of profiling results with key metrics. |
| **Diagnostics** | Warnings and errors encountered during profiling. |
| **Timeline** | Visual timeline of CPU, GPU, and API activity. |
| **Events** | Detailed list of all recorded events with filtering. |
| **Function Table** | Aggregated function statistics (Top-Down, Bottom-Up, Flat). |

### Timeline Area (Center Pane)

The center pane displays the timeline visualization with rows for:
- Process threads
- GPU streams
- HW engines
- API calls
- NVTX ranges
- Memory operations

### Function/Details Table (Bottom Pane)

The bottom pane shows the function table or details for the selected timeline row, depending on the active view.

---

## Analysis Summary View

The Analysis Summary View provides a high-level overview of the profiling session.

### Sections

| Section | Content |
|---|---|
| **Project Summary** | Application name, arguments, duration, date, target info. |
| **GPU Summary** | Per-GPU utilization, kernel counts, memcpy counts and sizes. |
| **CPU Summary** | Thread count, process tree, context switch counts. |
| **Warning Summary** | Performance warnings and suggestions. |
| **CUDA API Summary** | Time spent in CUDA runtime and driver APIs. |
| **Memory Transfer Summary** | Sizes and durations of memory copies. |
| **Kernel Summary** | Top GPU kernels by total duration. |
| **Sampling Summary** | Top CPU functions by sample count. |

### Key Metrics

| Metric | Description |
|---|---|
| **GPU Time** | Total time GPUs spent executing work. |
| **GPU Utilization** | Percentage of time the GPU was busy. |
| **CPU Utilization** | Per-thread CPU usage percentage. |
| **Memory Throughput** | Aggregate memcpy bandwidth. |
| **Kernel Count** | Total number of GPU kernel launches. |

---

## Diagnostics Summary View

The Diagnostics Summary View shows warnings, errors, and informational messages from the profiling session.

### Message Categories

| Category | Description |
|---|---|
| **Error** | Critical issues that may have affected data collection. |
| **Warning** | Non-critical issues that may affect data quality. |
| **Info** | Informational messages about the profiling configuration. |

### Common Diagnostic Messages

| Message | Cause | Resolution |
|---|---|---|
| "CPU sampling rate was throttled" | System-level throttling of perf events | Reduce sampling frequency or shorten trace duration |
| "GPU context was not captured" | GPU trace started after context creation | Use `--cuda-memory-usage` or restart with full trace |
| "Symbol resolution failed for module X" | Missing symbol files | Add symbol paths in preferences |
| "perf_event_open failed" | `perf_event_paranoid` too restrictive | Set `/proc/sys/kernel/perf_event_paranoid` to -1 or 0 |
| "Trace buffer overflow" | Too many events for the buffer size | Increase buffer size with `--buffer-size` option |
| "NVTX domain not found" | Application uses NVTX but domain was not enabled | Enable NVTX tracing with `--trace=nvtx` |

---

## Symbol Resolution Logs

The Symbol Resolution Logs provide details about symbol loading and resolution during report analysis.

### Log Contents

- **Loaded modules**: List of all binaries loaded during profiling.
- **Symbol search paths**: Paths searched for debug symbols.
- **Resolved symbols**: Successfully resolved symbol files.
- **Unresolved symbols**: Modules for which symbols could not be found.
- **Symbol load errors**: Errors encountered during symbol loading.

### Accessing Symbol Logs

1. Open a report in the GUI.
2. Select **View > Symbol Resolution Log** or check the Diagnostics view.
3. Review the log for any unresolved modules.

### Configuring Symbol Paths

```bash
# Linux: Set symbol search paths
export Nsight_Systems_SymbolPath=/path/to/symbols:/another/path

# Windows: Use _NT_SYMBOL_PATH
set _NT_SYMBOL_PATH=SRV*C:\Symbols*https://msdl.microsoft.com/download/symbols
```

In the GUI: **File > Preferences > Symbols > Add Path**

---

## Timeline View

The Timeline View is the primary visualization for understanding application behavior over time.

### Navigation

| Action | Shortcut | Description |
|---|---|---|
| Zoom in | `Ctrl+Mouse Wheel` or `+` | Zoom in on the timeline center. |
| Zoom out | `Ctrl+Mouse Wheel` or `-` | Zoom out from the timeline center. |
| Zoom to selection | `Ctrl+Shift+Z` | Zoom to fit the selected range. |
| Zoom to full trace | `Ctrl+Home` | Show the entire trace. |
| Pan left | `Left Arrow` or `Shift+Mouse Wheel` | Scroll timeline to the left. |
| Pan right | `Right Arrow` or `Shift+Mouse Wheel` | Scroll timeline to the right. |
| Select range | `Click and drag` | Select a time range. |
| Select event | `Click` | Select a single event. |
| Go to time | `Ctrl+G` | Jump to a specific timestamp. |
| Reset view | `Ctrl+Home` | Reset to full trace view. |

### Zoom and Scroll

- **Horizontal zoom**: Changes the time resolution (nanoseconds per pixel).
- **Vertical zoom**: Changes the row height (pixels per row).
- **Minimap**: The bar at the top shows the full trace with a viewport indicator.
- **Time ruler**: Shows absolute and relative timestamps.

### Timeline and Events Correlation

- Clicking an event in the timeline highlights it and shows details in the bottom pane.
- Hovering over an event shows a tooltip with duration, API name, and arguments.
- Right-clicking an event provides context options:
  - **Zoom to Event**
  - **Find Similar Events**
  - **View Source** (if symbols are available)
  - **Copy Event Details**

### Row Height and Row Percentage

| Option | Description |
|---|---|
| **Auto row height** | Automatically adjusts row height based on content. |
| **Fixed row height** | Set a specific pixel height for all rows. |
| **Row percentage** | Show percentage of visible time occupied by events in each row. |

To configure: Right-click on the timeline header area and select **Row Height** or **Show Percentages**.

---

## Timeline Options

The Timeline Options panel provides configuration for the timeline display.

### Display Options

| Option | Description | Default |
|---|---|---|
| **Show idle threads** | Display threads with no activity. | Enabled |
| **Show GPU migrations** | Show GPU context migration events. | Enabled |
| **Show memory transfers** | Display cudaMemcpy and similar operations. | Enabled |
| **Merge rows** | Combine similar rows into groups. | Disabled |
| **NVTX nesting** | Show nested NVTX ranges hierarchically. | Enabled |
| **Row labels** | Choose between thread name, thread ID, or process name. | Thread name |

### Time Format

| Format | Example | Description |
|---|---|---|
| **Absolute** | `12:34:56.789123456` | Wall clock time. |
| **Relative to trace start** | `1.234 s` | Time from the beginning of the trace. |
| **Relative to selection** | `0.000 s` | Time from the start of the selected range. |
| **Ticks** | `1234567890` | Raw timestamp counter value. |

### Color Coding

- API calls are color-coded by category (CUDA Runtime, CUDA Driver, OS Runtime, etc.).
- GPU kernels use a distinct color palette from CPU operations.
- NVTX ranges use user-defined colors or default palette.
- Customize colors in **File > Preferences > Colors**.

---

## Events View

The Events View shows a tabular list of all recorded events with powerful filtering capabilities.

### Event Columns

| Column | Description |
|---|---|
| **Start** | Start timestamp of the event. |
| **Duration** | Duration of the event. |
| **PID** | Process ID. |
| **TID** | Thread ID. |
| **API/Event Name** | Name of the API call or event. |
| **Result** | Return value of the API call. |
| **Arguments** | Arguments passed to the API. |
| **Correlation ID** | Links related CPU and GPU events. |

### Debug Markers

Nsight Systems supports several debug marker APIs for annotating traces:

#### NVTX (NVIDIA Tools Extension)

- NVTX ranges appear as labeled regions in the timeline.
- Support for nested ranges, categories, and custom colors.
- Domain support for isolating ranges from different libraries.
- Payload support for attaching key-value data.

#### Vulkan Debug Markers

- `VK_EXT_debug_marker` and `VK_EXT_debug_utils` extensions.
- `vkCmdBeginDebugUtilsLabelEXT` / `vkCmdEndDebugUtilsLabelEXT`
- `vkSetDebugUtilsObjectNameEXT`
- Appear in the Vulkan command buffer rows of the timeline.

#### PIX Debug Markers

- Windows-only, used with Direct3D applications.
- `PIXBeginEvent` / `PIXEndEvent` / `PIXSetMarker`
- Displayed in D3D12 command list rows.

#### OpenGL KHR_debug

- `GL_KHR_debug` extension markers.
- `glPushDebugGroupKHR` / `glPopDebugGroupKHR`
- `glObjectLabelKHR`
- Shown in OpenGL API trace rows.

### Filtering Events

- Filter by event type (Kernel, Memcpy, API, etc.).
- Filter by duration (minimum/maximum).
- Filter by name pattern (regex support).
- Filter by PID/TID.
- Filter by correlation ID.

---

## Function Table Modes

The Function Table provides aggregated statistics about CPU function execution from sampling data.

### Top-Down View

The Top-Down view shows the call hierarchy starting from the outermost functions (e.g., `main`) and drilling down into called functions.

```
main
├── train_loop
│   ├── forward_pass
│   │   ├── conv2d (Self: 15%, Total: 45%)
│   │   └── relu (Self: 5%, Total: 10%)
│   └── backward_pass
│       ├── conv2d_backward (Self: 20%, Total: 25%)
│       └── ...
└── data_loader
    └── read_batch (Self: 10%, Total: 10%)
```

**Columns:**

| Column | Description |
|---|---|
| **Total Samples** | Number of samples where this function was anywhere in the call stack. |
| **Total %** | Percentage of all samples where this function appeared in the call stack. |
| **Self Samples** | Number of samples where this function was at the top of the call stack (the currently executing function). |
| **Self %** | Percentage of all samples where this function was the leaf function. |

### Bottom-Up View

The Bottom-Up view starts from the leaf functions (most frequently sampled) and shows their callers.

```
conv2d_kernel (Self: 25%)
├── conv2d (called from: forward_pass)
│   └── train_loop
│       └── main
├── im2col (called from: conv2d)
│   └── ...
└── [Unknown] (called from inlined code)

gemm_kernel (Self: 18%)
├── matmul
│   ├── linear
│   └── ...
```

**Columns:**

| Column | Description |
|---|---|
| **Self Samples** | Number of samples at this exact function. |
| **Self %** | Percentage of samples at this exact function. |
| **Total Samples** | Total samples in this function and all its callees. |
| **Total %** | Percentage of total samples. |

### Flat View

The Flat view shows all functions sorted by their self-sample count without call hierarchy.

| Function | Module | Self Samples | Self % | Total Samples | Total % |
|---|---|---|---|---|---|
| `conv2d_kernel` | libcudnn.so | 12500 | 25.0% | 12500 | 25.0% |
| `gemm_kernel` | libcublas.so | 9000 | 18.0% | 9000 | 18.0% |
| `memcpy` | libc.so | 5000 | 10.0% | 5000 | 10.0% |
| `[Unknown]` | N/A | 3500 | 7.0% | 3500 | 7.0% |
| `elementwise_kernel` | libtorch_cuda.so | 2800 | 5.6% | 2800 | 5.6% |

**Column Explanations:**

- **Self**: Time spent directly in this function (not including called functions). This identifies the actual hotspots.
- **Total**: Time spent in this function and all functions it called. This identifies which code paths are expensive.
- **Flat**: In flat view, Self and Total are the same since there is no hierarchy.

---

## Backtraces Explanation

Backtraces (call stacks) are collected at each sampling point to show the full execution path.

### LBR (Last Branch Record) vs Frame Pointers

| Method | Description | Pros | Cons |
|---|---|---|---|
| **LBR** | Intel hardware feature that records recent branch targets. Provides limited-depth backtraces without frame pointers. | No recompilation needed; works with optimized code. | Limited depth (typically 4-32 frames); Intel only. |
| **Frame Pointers** | Traditional backtrace method using the frame pointer register (`rbp` on x86-64). | Unlimited depth; works on all architectures. | Requires `-fno-omit-frame-pointer` compilation flag; not available with heavy optimization. |
| **DWARF** | Uses DWARF debug info to unwind stacks without frame pointers. | Works with optimized code; unlimited depth. | Slower to collect; requires debug info (`-g`). |
| **ORC** | Linux kernel-specific unwinder (ORC unwinder). | Used for kernel stack traces. | Kernel only. |

### Kernel Samples

Kernel-space samples appear with the `[kernel]` prefix or module name:

- `[kernel].schedule` - Scheduler function
- `[kernel].__schedule` - Internal scheduler
- `[kernel].do_page_fault` - Page fault handler
- `[kernel].copy_user_enhanced_fast_string` - Memory copy in kernel

### [vdso] Entries

The `[vdso]` (Virtual Dynamic Shared Object) is a kernel-provided shared library mapped into every process:

- Contains fast-path implementations of system calls like `clock_gettime`, `gettimeofday`.
- Samples in `[vdso]` indicate time spent in these frequently-called functions.
- Not a performance concern unless excessive.

### [Unknown] Entries

`[Unknown]` appears when the backtrace cannot be resolved to a function name:

| Cause | Solution |
|---|---|
| Missing symbol files | Install debug packages or add symbol paths. |
| Stripped binaries | Use unstripped versions during profiling. |
| JIT-compiled code | Register JIT debug info with the runtime. |
| Inlined functions | Compile with `-g` to get inline information. |
| Dynamic code generation | Use runtime-specific symbol registration (e.g., `__jit_debug_register_code`). |

---

## Multi-Report Timeline Views

Nsight Systems supports viewing multiple reports simultaneously for comparison.

### Separate Panes Mode

- Each report is displayed in its own timeline pane, stacked vertically.
- Scrolling and zooming operate independently per pane.
- Useful for comparing different runs of the same application.

### Same Timeline Mode

- Multiple reports are overlaid on the same timeline axis.
- Events from different reports are shown in different colors or row groups.
- Useful for comparing A/B test results.

### Time Synchronization

| Mode | Description |
|---|---|
| **Synchronized** | Zooming and scrolling in one pane mirrors in all others. |
| **Independent** | Each pane has independent zoom and scroll. |
| **Aligned by start** | Aligns all reports so their start times coincide. |
| **Aligned by event** | Aligns reports using a specific event type (e.g., first kernel launch). |

### Opening Multiple Reports

1. Open the first report normally.
2. Drag and drop additional `.nsys-rep` files onto the GUI.
3. Select **View > Multi-Report > Tile Horizontally/Vertically**.
4. Use the time sync controls in the toolbar.

### Use Cases

- **Before/after comparison**: Compare performance before and after a code change.
- **Scaling analysis**: Compare runs with different numbers of GPUs or threads.
- **Configuration comparison**: Compare different optimization flags or parameters.
- **Regression testing**: Compare current results against a known baseline.

---

## Flame Graph Generation

Nsight Systems can generate flame graphs from CPU sampling data for visual hotspot analysis.

### Creating a Flame Graph

1. Open a report with CPU sampling data.
2. Navigate to the **Function Table** view.
3. Click the **Flame Graph** button in the toolbar.
4. Alternatively, right-click on the Function Table and select **Show Flame Graph**.

### Flame Graph Features

| Feature | Description |
|---|---|
| **Interactive** | Click on a frame to zoom in; right-click to zoom out. |
| **Search** | Type a function name to highlight matching frames. |
| **Color coding** | Frames are colored by module or category. |
| **Tooltips** | Hover over a frame to see function name, sample count, and percentage. |
| **Call path** | Each stack level shows the cumulative path from the root. |

### Flame Graph Interpretation

- **Width**: Represents the total number of samples (time) for that function and its callees.
- **Height**: Represents the stack depth.
- **Flat top**: A frame with no children indicates a leaf function (actual execution point).
- **Wide frames**: Functions occupying a large portion of the graph are performance hotspots.

### Exporting Flame Graphs

```bash
# Export flame graph data from CLI
nsys export -t sqlite -o report.sqlite report.nsys-rep

# Use external tools for advanced flame graph visualization
# e.g., FlameGraph tools by Brendan Gregg
nsys stats --report function-summary report.nsys-rep
```

### Flame Graph vs Function Table

| Aspect | Flame Graph | Function Table |
|---|---|---|
| **Visualization** | Graphical, intuitive | Tabular, precise |
| **Navigation** | Click to zoom | Expand/collapse rows |
| **Search** | Visual highlighting | Text filtering |
| **Precision** | Approximate widths | Exact sample counts |
| **Use case** | Quick overview, presentations | Detailed analysis, scripting |

---

## Keyboard Shortcuts Summary

| Shortcut | Action |
|---|---|
| `Ctrl+O` | Open report |
| `Ctrl+S` | Save report |
| `Ctrl+W` | Close current tab |
| `Ctrl+Z` | Undo |
| `Ctrl+Home` | Reset zoom |
| `Ctrl+G` | Go to time |
| `Ctrl+F` | Find event |
| `Ctrl+Plus` | Zoom in |
| `Ctrl+Minus` | Zoom out |
| `Ctrl+Shift+Z` | Zoom to selection |
| `Space` | Play/Pause (for animated replay) |
| `Tab` | Switch between views |
| `F5` | Refresh current view |
| `F11` | Toggle fullscreen |
| `Escape` | Cancel current selection |

---

## See Also

- [CLI Reference](01-cli-usage.md)
- [Export Formats and SQLite Schema](11-export-sqlite-schema.md)
- [Release Notes and Troubleshooting](12-release-notes-troubleshooting.md)
