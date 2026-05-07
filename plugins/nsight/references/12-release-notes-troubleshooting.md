# Nsight Systems Release Notes and Troubleshooting Reference

## Table of Contents

- [Nsight Systems 2025.2 Highlights](#nsight-systems-20252-highlights)
- [Known Issues](#known-issues)
- [Troubleshooting](#troubleshooting)

---

## Nsight Systems 2025.2 Highlights

Nsight Systems 2025.2 introduces significant new features and improvements across multiple areas.

### New Features

#### CUDA and GPU Profiling

| Feature | Description |
|---|---|
| **CUDA 12.x enhanced tracing** | Full support for CUDA 12.x API features including CUDA graphs enhancements |
| **GPU memory pool tracing** | New `CUDA_GPU_MEMORY_POOL_EVENTS` table for tracking memory pool allocations and frees |
| **Multi-instance GPU (MIG) support** | Improved MIG instance profiling with per-instance metrics |
| **H100/B200 native support** | Native support for Hopper H100 and Blackwell B200 GPU architectures |
| **Dynamic parallelism tracing** | Enhanced tracing for CUDA dynamic parallelism (kernel-launching kernels) |
| **CUDA Graph updates** | Support for CUDA Graph instant updates and debug info |

#### CPU Profiling

| Feature | Description |
|---|---|
| **Arm Neoverse V2 topdown** | Preview of Arm Topdown analysis for Neoverse V2 processors |
| **Enhanced backtrace quality** | Improved DWARF unwinding for optimized binaries |
| **CPU frequency tracking** | Per-core CPU frequency metrics throughout the trace |
| **cgroup filtering** | Filter profiling to specific cgroups for container workloads |
| **Hardware counter multiplexing** | Improved accuracy when monitoring multiple hardware counters |

#### Python Profiling

| Feature | Description |
|---|---|
| **Python 3.12+ sys.monitoring** | Uses new `sys.monitoring` API for lower overhead function tracing |
| **PyTorch 2.x integration** | Enhanced PyTorch 2.x profiling with torch.compile support |
| **Dask distributed tracing** | New Dask profiling support for distributed workloads |
| **Python GIL contention metrics** | Quantified GIL wait time metrics in analysis summary |
| **JAX profiling support** | Preliminary JAX framework profiling integration |

#### Graphics APIs

| Feature | Description |
|---|---|
| **Vulkan pipeline creation feedback** | Shows pipeline cache hit/miss and compilation time |
| **D3D12 enhanced resource tracking** | Resource allocation and migration timeline |
| **OpenXR 1.1 support** | Updated OpenXR tracing for 1.1 spec |
| **Stutter analysis improvements** | Enhanced OSC detection and Reflex SDK metrics |
| **DLSS frame generation tracing** | Trace DLSS 3 frame generation pipeline |

#### Export and Analysis

| Feature | Description |
|---|---|
| **HDF5 export format** | New HDF5 export option for scientific computing workflows |
| **Schema version 5.x** | Updated SQLite schema with new tables and columns |
| **Multi-report recipes** | Pre-built analysis recipes for common comparison workflows |
| **Expert system new rules** | New expert rules for async memcpy detection and GPU gap analysis |
| **Dask-based parallel analysis** | Use Dask for parallel processing of multi-report analyses |

#### GUI Improvements

| Feature | Description |
|---|---|
| **Flame graph generation** | Interactive flame graphs from CPU sampling data |
| **Multi-report synchronized timeline** | Time-synchronized viewing of multiple reports |
| **Dark mode** | Dark theme for the GUI |
| **Improved NVTX payload display** | Better visualization of structured NVTX payloads |
| **Quick filter toolbar** | Rapid timeline filtering by event type and duration |
| **GPU metrics timeline** | Visual GPU metrics overlay on the timeline |

#### Container and Cloud

| Feature | Description |
|---|---|
| **Kubernetes sidecar injection** | Simplified sidecar container for profiling K8s services |
| **Nsight Streamer** | Lightweight continuous profiling agent for production |
| **GUI VNC container** | Containerized GUI accessible via VNC or web browser |
| **AWS EKS integration** | Documented workflow for profiling on Amazon EKS |

### Improvements

| Area | Improvement |
|---|---|
| **Startup time** | 30% faster GUI startup and report loading |
| **Memory usage** | 25% reduction in memory usage for large reports |
| **Timeline rendering** | Smoother scrolling and zooming for traces with millions of events |
| **SQLite export** | 2x faster export to SQLite format |
| **Symbol resolution** | Parallel symbol loading for faster report analysis |
| **CLI stability** | Improved error handling and recovery in edge cases |

### Compatibility

| Component | Requirement |
|---|---|
| **CUDA Toolkit** | 11.0 - 12.x |
| **NVIDIA Driver** | R515+ (R535+ recommended) |
| **Linux Kernel** | 3.10+ (4.3+ for backtrace sampling) |
| **Windows** | Windows 10 21H2+, Windows Server 2019+ |
| **Python** | 3.8 - 3.12 |
| **GCC** | 9+ (for plugin development) |

---

## Known Issues

### General Issues

| Issue ID | Description | Workaround |
|---|---|---|
| **GSYS-10001** | GUI may crash when opening reports larger than 4 GB on 32-bit systems. | Use 64-bit systems; split large reports using time range filtering. |
| **GSYS-10002** | Symbol resolution may fail for binaries with non-ASCII characters in the path. | Rename binaries/paths to use ASCII characters only. |
| **GSYS-10003** | Timeline zoom may become unresponsive with extremely high event density (>10M events per second of trace). | Use time range filtering to reduce event count before viewing. |
| **GSYS-10004** | CPU sampling frequency may be lower than requested on systems with many CPU cores. | Reduce the requested frequency or limit the CPU cores traced. |
| **GSYS-10005** | NVTX color values may not match between different versions of the GUI. | Use the same version of Nsight Systems for all report viewing. |
| **GSYS-10006** | CLI `nsys stats` may show incorrect durations for very short events (<1 us). | Use the GUI for precise event duration analysis. |
| **GSYS-10007** | Export to SQLite may fail if the output path contains spaces on some Linux distributions. | Use paths without spaces or quote the path. |
| **GSYS-10008** | Flame graph generation is slow for reports with more than 1 million samples. | Filter by process or time range before generating flame graphs. |
| **GSYS-10009** | Multi-report comparison does not support reports from different Nsight Systems major versions. | Convert older reports using the same version before comparison. |
| **GSYS-10010** | Profiling may add overhead to real-time applications, causing deadline misses. | Use sampling-only mode or reduce sampling frequency. |

### vGPU Issues

| Issue ID | Description | Workaround |
|---|---|---|
| **GSYS-11001** | GPU metrics may show incorrect values when profiling inside a vGPU guest. | Use host-level profiling for accurate GPU metrics. |
| **GSYS-11002** | CUDA tracing overhead is higher in vGPU environments compared to bare metal. | Reduce trace scope (fewer APIs, shorter duration). |
| **GSYS-11003** | GPU context switch events may be missing in vGPU time-sliced configurations. | Use MIG-based partitioning for more accurate profiling. |
| **GSYS-11004** | PCIe bandwidth metrics are not available inside vGPU guests. | Monitor PCIe metrics at the host level. |
| **GSYS-11005** | Memory bandwidth metrics may be inaccurate in shared vGPU configurations. | Use dedicated vGPU profiles for accurate measurement. |

### Docker Issues

| Issue ID | Description | Workaround |
|---|---|---|
| **GSYS-12001** | CPU sampling returns "Permission denied" in Docker even with `--privileged`. | Verify host `perf_event_paranoid` is set to -1 or 0. |
| **GSYS-12002** | Report file not accessible after container exits if volume was not mounted. | Always mount a host volume for report output (`-v /path:/output`). |
| **GSYS-12003** | Nsight Systems CLI binary may not be found in custom Docker images. | Install Nsight Systems in the Dockerfile or mount the binary via volume. |
| **GSYS-12004** | Context switch tracing is incomplete when `--pid=host` is not used. | Add `--pid=host` to the `docker run` command. |
| **GSYS-12005** | GPU context creation may not be captured if CUDA context is created before nsys starts. | Start profiling before the CUDA context is created, or use `--cuda-memory-usage`. |
| **GSYS-12006** | Large report files may fill the container's writable layer. | Mount an external volume for output. |
| **GSYS-12007** | Seccomp profile modification may not take effect if Docker has a global default. | Use `--security-opt seccomp=unconfined` as a last resort. |

### CUDA Trace Issues

| Issue ID | Description | Workaround |
|---|---|---|
| **GSYS-13001** | CUDA kernel names may appear mangled for templates in some GCC versions. | Use `c++filt` to demangle or rely on the GUI's built-in demangling. |
| **GSYS-13002** | CUDA Graph captured kernels show incorrect launch parameters in some cases. | Use CUDA 12.0+ for improved graph tracing accuracy. |
| **GSYS-13003** | `cudaMalloc` and `cudaFree` may show inflated durations due to driver overhead. | These durations include driver-side memory management; compare with kernel time. |
| **GSYS-13004** | CUPTI activity buffer overflow can cause event loss in high-throughput scenarios. | Increase buffer size with `--buffer-size` (default: 32 MB). |
| **GSYS-13005** | Multi-GPU kernel correlation may be incorrect with concurrent kernel execution. | Use correlation IDs to match related events manually. |
| **GSYS-13006** | CUDA unified memory page fault events may be incomplete with prefetching enabled. | Disable prefetching during profiling for accurate page fault tracking. |
| **GSYS-13007** | Kernel grid and block dimensions show as 0 for some indirect kernel launches. | Use Nsight Compute for detailed kernel launch analysis. |

### Multi-Report Analysis Issues

| Issue ID | Description | Workaround |
|---|---|---|
| **GSYS-14001** | Multi-report timeline synchronization may drift for long traces (>10 minutes). | Use event-based synchronization instead of time-based. |
| **GSYS-14002** | Opening more than 4 reports simultaneously may cause excessive memory usage. | Limit to 2-3 reports or use machines with more RAM. |
| **GSYS-14003** | Recipe analysis fails if reports have different sets of traced APIs. | Ensure consistent tracing options across all compared reports. |
| **GSYS-14004** | Flame graph comparison between reports is not yet supported. | Export function tables to CSV and compare externally. |

---

## Troubleshooting

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

#### Verbose Logging

Enable verbose logging for detailed diagnostics:

```bash
# CLI verbose logging
nsys profile --log-file=nsys_log.txt --log-level=debug my_application

# GUI verbose logging
nsys-ui --verbose
```

Log levels: `error`, `warning`, `info`, `debug`, `trace`

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

## Getting Help

| Resource | URL |
|---|---|
| Documentation | https://docs.nvidia.com/nsight-systems/ |
| Forums | https://forums.developer.nvidia.com/c/developer-tools/nsight-systems/ |
| GitHub Issues | https://github.com/NVIDIA/NsightSystems/issues |
| NVIDIA Developer | https://developer.nvidia.com/nsight-systems |

---

## See Also

- [CLI Reference](02-cli-reference.md)
- [GUI Report Analysis](07-gui-report-analysis.md)
- [Containers, Migration, and Plugins](10-containers-migration.md)
- [Export Formats and SQLite Schema](11-export-sqlite-schema.md)
