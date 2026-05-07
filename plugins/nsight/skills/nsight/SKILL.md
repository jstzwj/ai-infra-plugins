# NVIDIA Nsight Systems - Comprehensive Reference Manual

> **Version Coverage**: Nsight Systems 2025.x (CLI version 1.0.x)
> **Last Updated**: 2026-05-07
> **Official Documentation**: [https://docs.nvidia.com/nsight-systems/](https://docs.nvidia.com/nsight-systems/)

---

## Table of Contents

### Reference Chapters

| # | Chapter | File | Description |
|---|---------|------|-------------|
| 1 | Overview & Getting Started | [01-overview.md](references/01-overview.md) | What is Nsight Systems, system requirements, installation, first steps |
| 2 | CLI Command Reference | [02-cli-reference.md](references/02-cli-reference.md) | Complete command-line interface documentation with all switches and options |
| 3 | CUDA Tracing Reference | [03-cuda-tracing.md](references/03-cuda-tracing.md) | CUDA API tracing, memory tracking, Unified Memory, CUDA Graphs, function lists |
| 4 | GPU Metrics & Hardware Profiling | [04-gpu-metrics.md](references/04-gpu-metrics.md) | GPU metrics sampling, hardware counters, SoC metrics, NVML power metrics |
| 5 | NVTX & OS Runtime Tracing | [05-nvtx-osrt-tracing.md](references/05-nvtx-osrt-tracing.md) | NVTX annotations, OS Runtime Libraries, OpenMP, syscall trace |
| 6 | Network Communication Profiling | [06-network-profiling.md](references/06-network-profiling.md) | MPI, OpenSHMEM, UCX, NCCL, NVSHMEM, NIC metrics, InfiniBand |
| 7 | GUI, Reports & Timeline Analysis | [07-gui-report-analysis.md](references/07-gui-report-analysis.md) | GUI usage, report management, timeline navigation, analysis views |
| 8 | Graphics APIs Trace | [08-graphics-apis.md](references/08-graphics-apis.md) | Direct3D 11/12, Vulkan, OpenGL, OpenXR, WDDM, stutter analysis |
| 9 | Python & CPU Profiling | [09-python-cpu-profiling.md](references/09-python-cpu-profiling.md) | Python backtrace, GIL tracing, PyTorch, Dask, CPU sampling, Arm Topdown |
| 10 | Containers, Migration & Video | [10-containers-migration.md](references/10-containers-migration.md) | Docker, Kubernetes, Nsight Streamer, nvprof migration, video profiling |
| 11 | Export, SQLite Schema & Analysis | [11-export-sqlite-schema.md](references/11-export-sqlite-schema.md) | Export formats, SQLite schema, expert systems, multi-report analysis |
| 12 | Release Notes & Troubleshooting | [12-release-notes-troubleshooting.md](references/12-release-notes-troubleshooting.md) | Release notes, known issues, troubleshooting guide |

---

## Overview

NVIDIA Nsight Systems is a system-wide performance analysis tool that delivers an at-a-glance view of how an application uses the compute resources of the target machine. It is designed to help developers understand and optimize the performance of their GPU-accelerated applications by providing detailed timelines and metrics.

### What Nsight Systems Provides

- **Timeline Visualization**: See exactly when and where CPU threads, GPU kernels, memory transfers, and API calls occur relative to each other.
- **System-Wide Tracing**: Capture activity from multiple CPUs, GPUs, and accelerators simultaneously.
- **Low Overhead**: Designed to minimize profiling impact so that the collected data reflects real-world behavior.
- **Multi-API Support**: Trace CUDA, CUDA Runtime, OpenACC, OpenMP, MPI, Vulkan, DirectX, OpenGL, and more.
- **Hardware Metrics**: Collect GPU hardware metrics including SM utilization, memory throughput, and PCIe bandwidth.
- **Sampling & Backtraces**: CPU instruction-level sampling with call-stack backtraces for hot-spot analysis.
- **NVTX Integration**: Full support for NVIDIA Tools Extension (NVTX) ranges and markers for custom instrumentation.

### Key Concepts

| Concept | Definition |
|---------|-----------|
| **Profiling** | The process of measuring where an application spends its time and resources. Nsight Systems collects timeline traces and sampling data to create a profile. |
| **Sampling** | Periodically interrupting execution to record the instruction pointer and call stack. Used to identify CPU hot spots without instrumenting every function. |
| **Tracing** | Recording the start and end times of specific events (API calls, kernel launches, memory transfers) to build a detailed timeline. |
| **Trace Session** | A collection of traced data from one or more profiling runs. Stored in `.nsys-rep` format. |
| **NVTX** | NVIDIA Tools Extension SDK. An API that allows developers to annotate their code with custom ranges, markers, and strings that appear in the profiling output. |
| **Report** | A processed view of trace data that summarizes performance metrics. Can be generated via CLI or viewed in the GUI. |
| **Range Profiler** | A feature that allows collecting hardware metrics for a specific range of execution (e.g., a single kernel or an NVTX range). |
| **Backtrace** | A snapshot of the call stack at a particular point during execution, useful for understanding where CPU time is being spent. |

### Typical Workflow

1. **Profile** the application using the CLI (`nsys profile`) or GUI.
2. **Open** the resulting `.nsys-rep` file in the Nsight Systems GUI for timeline visualization.
3. **Analyze** the timeline to identify bottlenecks (e.g., low GPU utilization, excessive synchronization, memory transfer overhead).
4. **Iterate** by adding NVTX annotations and re-profiling to drill down into specific code sections.

---

## Quick Reference Card

### Common CLI Commands

```bash
# Basic profiling with default settings
nsys profile -o my_app ./my_application

# Profile with CUDA and GPU metrics
nsys profile -t cuda,nvtx,osrt -s cpu -o my_app ./my_application

# Profile with sampling and backtraces
nsys profile -t cuda,nvtx,osrt -s cpu --cpu-backtrace=true -o my_app ./my_application

# Profile Python script
nsys profile -t cuda,nvtx,osrt -s cpu -o my_script python my_script.py

# Profile with delayed start (skip initialization)
nsys profile --delay=5 -t cuda,nvtx -s cpu -o my_app ./my_application

# Generate a text report from a trace file
nsys stats my_app.nsys-rep

# Export trace to SQLite database
nsys export -t sqlite -o my_app.sqlite my_app.nsys-rep
```

### GUI Launch

```bash
# Launch GUI (local)
nsys-ui

# Launch GUI and open a specific report
nsys-ui my_app.nsys-rep
```

---

## Supported APIs and Technologies

| API / Technology | Trace Support | Sampling Support | Notes |
|-----------------|:------------:|:----------------:|-------|
| CUDA Driver API | Yes | N/A | `cu*` functions |
| CUDA Runtime API | Yes | N/A | `cuda*` functions |
| cuDNN | Yes | N/A | Deep learning primitives |
| cuBLAS | Yes | N/A | Linear algebra |
| cuFFT | Yes | N/A | Fast Fourier transforms |
| cuRAND | Yes | N/A | Random number generation |
| cuSPARSE | Yes | N/A | Sparse matrix operations |
| cuSOLVER | Yes | N/A | Numerical solvers |
| NVTX | Yes | N/A | Custom annotations |
| OpenACC | Yes | N/A | Directive-based parallelism |
| OpenMP | Yes | Yes | Thread-level tracing |
| MPI | Yes | N/A | Message passing |
| Vulkan | Yes | N/A | Graphics/compute API |
| OpenGL | Yes | N/A | Graphics API |
| DirectX 11/12 | Yes | N/A | Windows only |
| OS Runtime (pthreads) | Yes | Yes | Thread activity |
| Python | Yes | N/A | Function tracing via `nsys` |
| PyTorch | Yes | N/A | Profiler integration |
| TensorFlow | Yes | N/A | Profiler integration |
| ETW | Yes | N/A | Windows Event Tracing |
| ftrace | Yes | N/A | Linux kernel function tracing |
| Syscalls | Yes | N/A | Linux system call tracing |

---

## File Formats

| Extension | Description |
|-----------|-------------|
| `.nsys-rep` | Default Nsight Systems report format (SQLite-based) |
| `.qdrep` | Legacy report format (for backward compatibility) |
| `.qdrep.zip` | Compressed legacy report format |
| `.sqlite` | Exported SQLite database for programmatic access |
| `.hdf5` | Exported HDF5 format |
| `.csv` | Exported comma-separated values |

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `NSYS_ARGS` | Default arguments passed to all `nsys` commands |
| `NSYS_JIT_SERVICE_PATH` | Path to the JIT service binary |
| `NSYS_LOG_LEVEL` | Logging verbosity (`0`-`6`, default `3`) |
| `NSYS_TARGET_PROCESS` | Target process for attaching (Linux) |
| `CUDA_INJECTION64_PATH` | Path to the Nsight Systems injection library |
| `LD_LIBRARY_PATH` | Must include Nsight Systems lib directory |

---

## Integration with Other Tools

### Nsight Compute Integration

Nsight Systems can launch Nsight Compute (`ncu`) for detailed kernel-level profiling from within a trace session. This enables a two-level analysis workflow:

1. Use Nsight Systems for system-wide timeline analysis
2. Launch Nsight Compute for specific kernels identified as bottlenecks

```bash
# Profile with Nsight Systems, then use ncu on specific kernels
nsys profile -o my_app ./my_application
ncu --set full -k "my_kernel_name" ./my_application
```

### PyTorch Integration

```python
import torch
import torch.autograd.profiler as profiler

with profiler.profile(use_cuda=True, use_nsys=True) as prof:
    # Your PyTorch code here
    output = model(input)
```

### TensorFlow Integration

```python
import tensorflow as tf

tf.profiler.experimental.start('nsys_profile')
# Your TensorFlow code here
tf.profiler.experimental.stop()
```

---

## Version History Highlights

| Version | Key Features |
|---------|-------------|
| 2025.2 | Dask API trace, CUDA HW trace (Blackwell), GDS trace, syscall trace, Python 3.13, CGA dimensions |
| 2025.1 | Enhanced GPU memory metrics, improved Python profiling, ARM SBSA support updates |
| 2024.x | Wattson GPU power tracing, improved CUDA Graph support, expanded HW metrics |
| 2023.x | Range Profiler for HW metrics per NVTX range, improved CLI stats |
| 2022.x | CUDA Graph tracing, improved Vulkan support, enhanced sampling |
| 2021.x | Initial Python function tracing, expanded ETW support |

---

## Chapter Summaries

### Chapter 1: Overview & Getting Started
Covers what Nsight Systems is, system requirements across all supported platforms (Linux x86_64, Linux ARM SBSA, Windows, QNX), installation via package managers and manual methods, CLI setup, GUI launching, and Jupyter notebook integration.

### Chapter 2: CLI Command Reference
Comprehensive documentation of every CLI command and option. Covers global options, all command switches (`profile`, `analyze`, `cancel`, `export`, `launch`, `nvprof`, `recipe`, `sessions`, `shutdown`, `start`, `stats`, `status`, `stop`), with every parameter documented including defaults and descriptions. Includes extensive examples for single commands, interactive sequences, and stats workflows.

### Chapter 3: CUDA Tracing Reference
Deep dive into CUDA tracing capabilities. Covers basic CUDA trace configuration, backtrace setup, memory usage tracking, Unified Memory transfer tracing (HtoD, PtoP, DtoH), Unified Memory CPU/GPU page fault analysis, CUDA Graph tracing at graph and node levels, and CUDA Python backtraces. Includes complete reference lists for all traceable CUDA Runtime API functions, CUDA Primary (Driver) API functions, and cuDNN functions. Also covers launching Nsight Compute from a kernel context.

### Chapter 4: GPU Metrics & Hardware Profiling
Covers GPU metrics sampling with full metric tables, GPU context switch tracing, SoC metrics for embedded platforms, NVML power and temperature metrics, and video hardware profiling. Documents all available GPU metrics (GPC clock, SM utilization, tensor core activity, memory throughput, PCIe/NVLink bandwidth), sampling frequency configuration, and limitations.

### Chapter 5: NVTX & OS Runtime Tracing
Covers NVTX trace configuration including payloads, counters, domains, and categories. Documents OS Runtime Libraries trace with default function lists (libc syscalls, POSIX threads, I/O functions, miscellaneous). Also covers OpenMP trace and the experimental syscall trace feature.

### Chapter 6: Network Communication Profiling
Covers MPI API trace (parameters and functions traced), OpenSHMEM, UCX, NCCL, NVSHMEM. Includes NIC metric sampling, InfiniBand switch metrics and congestion events, network information collection, Amazon AWS EFA metrics, and network interface/storage metrics.

### Chapter 7: GUI, Reports & Timeline Analysis
Covers GUI profiling for Linux, Windows, and QNX targets. Report management (create, merge, share). Timeline navigation, zoom, scroll, events correlation. Analysis Summary, Diagnostics Summary, Function Table modes, filter dialog, backtraces view. Multi-report timeline views and synchronization. Flame graph add-on.

### Chapter 8: Graphics APIs Trace
Covers Direct3D 11/12 API trace, WDDM queues and HW scheduler, Vulkan API trace with pipeline creation feedback, OpenGL trace, OpenXR trace, and stutter analysis including FPS, Reflex SDK, frame health, GPU memory utilization, and vertical synchronization.

### Chapter 9: Python & CPU Profiling
Covers Python backtrace sampling, Python functions trace, GIL tracing, PyTorch profiling integration, Dask profiling. CPU profiling on Linux including IP/backtrace sampling, context switch tracing, event sampling, Arm Topdown analysis on Grace, paranoid levels, and common issues.

### Chapter 10: Containers, Migration & Video
Covers Docker collection enablement, Kubernetes profiling, Nsight Streamer, GUI VNC container, nvprof migration guide, and NVIDIA video profiling (encoder/decoder/JPEG API trace).

### Chapter 11: Export, SQLite Schema & Analysis
Complete SQLite schema reference with all table definitions. Export formats (SQLite, HDF5, Arrow, Parquet, JSON, text). Expert Systems analysis rules. Multi-report analysis with Dask-based recipes.

### Chapter 12: Release Notes & Troubleshooting
Nsight Systems 2025.2 release notes, known issues (general, vGPU, Docker, CUDA trace, multi-report), and troubleshooting guide.

---

## Additional Resources

- **Official Documentation**: [https://docs.nvidia.com/nsight-systems/](https://docs.nvidia.com/nsight-systems/)
- **CUDA Toolkit Download**: [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
- **Nsight Systems Forum**: [https://forums.developer.nvidia.com/c/developer-tools/nsight-systems/](https://forums.developer.nvidia.com/c/developer-tools/nsight-systems/)
- **NVTX Documentation**: [https://github.com/NVIDIA/NVTX](https://github.com/NVIDIA/NVTX)
- **Nsight Systems CLI Guide**: Included with the installation at `<install_dir>/docs/`
- **CUDA Programming Guide**: [https://docs.nvidia.com/cuda/cuda-c-programming-guide/](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- **CUDA Best Practices Guide**: [https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
