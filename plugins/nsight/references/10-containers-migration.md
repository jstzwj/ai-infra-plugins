# Nsight Systems Containers, Migration, and Plugins Reference

## Table of Contents

- [Container Support](#container-support)
- [Migrating from NVIDIA nvprof](#migrating-from-nvidia-nvprof)
- [Nsight Systems Plugins](#nsight-systems-plugins)
- [Handling Application Launchers](#handling-application-launchers)

---

## Container Support

Nsight Systems supports profiling applications running inside Docker containers and Kubernetes pods. Proper configuration is required to grant the container access to performance monitoring hardware.

### Enable Docker Collection

#### Seccomp Configuration

Docker's default seccomp profile blocks the `perf_event_open` system call required for CPU sampling. You must adjust the configuration to allow it.

**Option 1: Use the `--privileged` flag (simplest, least secure):**

```bash
docker run --privileged my_container
```

**Option 2: Use a custom seccomp profile:**

Create a custom seccomp profile (`nsys-seccomp.json`) based on the default profile but with `perf_event_open` allowed:

```json
{
    "defaultAction": "SCMP_ACT_ERRNO",
    "architectures": ["SCMP_ARCH_X86_64", "SCMP_ARCH_AARCH64"],
    "syscalls": [
        {
            "names": ["perf_event_open"],
            "action": "SCMP_ACT_ALLOW"
        }
    ]
}
```

Then run with:

```bash
docker run --security-opt seccomp=nsys-seccomp.json my_container
```

**Option 3: Use `--cap-add=SYS_PTRACE` and `--cap-add=SYS_ADMIN`:**

```bash
docker run --cap-add=SYS_PTRACE --cap-add=SYS_ADMIN my_container
```

#### perf_event_open Access

For CPU sampling inside containers, ensure:

| Requirement | How to Verify |
|---|---|
| `perf_event_paranoid <= 0` | `cat /proc/sys/kernel/perf_event_paranoid` on the host |
| `perf_event_open` allowed | Check seccomp profile |
| Device access | `--device /dev/perf` or `--privileged` |
| Kernel headers | May be needed for some sampling modes |

```bash
# Set paranoid level on the host (before running container)
sudo sh -c 'echo 0 > /proc/sys/kernel/perf_event_paranoid'

# Verify from inside container
docker run --rm --privileged ubuntu cat /proc/sys/kernel/perf_event_paranoid
```

### Launch Docker Collection

#### Basic Docker Profiling

```bash
# Profile an application inside a Docker container
docker run --privileged --gpus all \
    -v /path/to/nsys:/usr/local/bin/nsys \
    my_container \
    nsys profile -o /tmp/report my_application

# Or with nsys installed in the container image
docker run --privileged --gpus all \
    my_container \
    nsys profile -o /tmp/report python train.py
```

#### NVIDIA Container Toolkit

When using the NVIDIA Container Toolkit (formerly nvidia-docker):

```bash
docker run --gpus all --privileged \
    -e NVIDIA_VISIBLE_DEVICES=0 \
    -v /path/to/output:/output \
    my_container \
    nsys profile -o /output/report \
    --trace=cuda,nvtx \
    --sample=cpu \
    python train.py
```

#### Extracting Reports from Containers

```bash
# Mount a host volume for report output
docker run --gpus all --privileged \
    -v $(pwd)/reports:/reports \
    my_container \
    nsys profile -o /reports/report python train.py

# Or copy from a stopped container
docker cp container_id:/tmp/report.nsys-rep ./report.nsys-rep
```

#### Docker Run Options Summary

| Option | Purpose |
|---|---|
| `--privileged` | Full device access (simplest for profiling) |
| `--gpus all` | Enable GPU access via NVIDIA Container Toolkit |
| `--cap-add=SYS_PTRACE` | Allow ptrace (needed for some sampling) |
| `--cap-add=SYS_ADMIN` | Allow perf_event_open |
| `--security-opt seccomp=...` | Custom seccomp profile |
| `--pid=host` | Share PID namespace with host |
| `-v /path:/path` | Mount volumes for output and tools |

### Profiling Services via Kubernetes

#### Sidecar Injection

Nsight Systems can profile services running in Kubernetes using a sidecar container approach.

**Pod Configuration:**

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: profiled-service
spec:
  shareProcessNamespace: true
  containers:
  - name: application
    image: my-application:latest
    command: ["python", "service.py"]
    securityContext:
      capabilities:
        add: ["SYS_PTRACE", "SYS_ADMIN"]
  - name: nsys-sidecar
    image: nvidia/nsight-systems:latest
    command: ["nsys", "profile", "-o", "/output/profile"]
    securityContext:
      privileged: true
    volumeMounts:
    - name: output
      mountPath: /output
  volumes:
  - name: output
    emptyDir: {}
```

#### Key Requirements

| Requirement | Configuration |
|---|---|
| Shared PID namespace | `shareProcessNamespace: true` |
| Privileged mode | `securityContext.privileged: true` |
| GPU access | `nvidia.com/gpu` resource request |
| Output volume | Shared emptyDir or PVC |
| Host paranoid level | Set on the node, not configurable per-pod |

#### Attaching to a Running Pod

```bash
# Use kubectl exec to run nsys in an existing pod
kubectl exec -it my-pod -- nsys profile --duration=10 -o /tmp/profile

# Or use a debug container (Kubernetes 1.18+)
kubectl debug my-pod -it --image=nvidia/nsight-systems -- nsys profile --duration=10
```

### Nsight Streamer

Nsight Streamer provides a lightweight agent for continuous profiling in production environments.

#### Architecture

```
[Application Container] --> [Nsight Streamer Agent] --> [Nsight Systems Collector]
                                      |
                                      v
                              [Report Storage]
```

#### Configuration

```bash
# Start the streamer agent
nsys-streamer start --port=8900 --output=/data/profiles

# Profile from the streamer
nsys-streamer profile --target=pod://my-service --duration=30
```

#### Streamer Features

| Feature | Description |
|---|---|
| **On-demand profiling** | Trigger profiles via API without restarting the service |
| **Continuous monitoring** | Periodic profiling for trend analysis |
| **Centralized collection** | Multiple agents report to a central collector |
| **Low overhead** | Minimal impact when not actively profiling |

### GUI VNC Container

For remote GUI access to Nsight Systems in a containerized environment, use the VNC container image.

#### Container Parameters

| Parameter | Description | Default |
|---|---|---|
| `VNC_PORT` | Port for VNC connections | `5900` |
| `NOVNC_PORT` | Port for web-based VNC | `8080` |
| `VNC_PASSWORD` | Password for VNC access | (none) |
| `RESOLUTION` | Display resolution | `1920x1080` |

#### Ports

| Port | Protocol | Purpose |
|---|---|---|
| `5900` | TCP | Direct VNC connection |
| `8080` | TCP | noVNC web client |
| `6000` | TCP | X11 (optional) |

#### Volumes

| Volume | Purpose |
|---|---|
| `/data` | Report storage |
| `/home/user` | User preferences and settings |
| `/tmp` | Temporary files |

#### Environment Variables

| Variable | Description | Example |
|---|---|---|
| `DISPLAY` | X11 display number | `:0` |
| `VNC_PASSWORD` | VNC access password | `my_password` |
| `RESOLUTION` | Virtual display resolution | `1920x1080` |
| `NSYS_HOME` | Nsight Systems installation path | `/opt/nvidia/nsight-systems` |

#### Example: Running the GUI VNC Container

```bash
# Basic VNC container
docker run -d \
    -p 5900:5900 -p 8080:8080 \
    -e VNC_PASSWORD=mypassword \
    -e RESOLUTION=1920x1080 \
    -v $(pwd)/reports:/data \
    --gpus all \
    --privileged \
    nvcr.io/nvidia/nsight-systems:latest

# Access via web browser at http://localhost:8080
# Or via VNC client at localhost:5900
```

---

## Migrating from NVIDIA nvprof

nvprof has been deprecated in favor of Nsight Systems and Nsight Compute. This section provides migration guidance.

### CLI nvprof Command Switch Options

| nvprof Option | Nsight Systems Equivalent | Notes |
|---|---|---|
| `nvprof ./app` | `nsys profile ./app` | Basic profiling |
| `--export-profile` / `-o` | `nsys profile -o report` | Output file |
| `--log-file` | `nsys profile --log-file log.txt` | Log file |
| `--analysis-metrics` | `nsys profile --stats=true` | Summary statistics |
| `--print-gpu-trace` | `nsys stats --report gpu-trace` | Detailed GPU trace |
| `--print-summary` | `nsys stats --report summary` | Summary report |
| `--print-api-summary` | `nsys stats --report cuda-api-summary` | CUDA API summary |
| `--print-gpu-summary` | `nsys stats --report cuda-api-summary` | GPU kernel summary |
| `--print-kernel-summary` | `nsys stats --report kernel-summary` | Per-kernel summary |
| `--print-memory-summary` | `nsys stats --report mem-summary` | Memory transfer summary |
| `--print-openmp-summary` | `nsys stats --report openmp-summary` | OpenMP summary |
| `--metrics` | Use Nsight Compute (`ncu`) | Detailed GPU metrics |
| `--events` | Use Nsight Compute (`ncu`) | Hardware events |
| `--kernels` `--kernel-name` | `nsys profile` + filter in GUI | Kernel filtering |
| `--unified-memory-profiling` | `nsys profile --trace=cuda` | UM profiling (automatic) |
| `--cpu-profiling` | `nsys profile --sample=cpu` | CPU sampling |
| `--cpu-thread-tracing` | `nsys profile --trace=osrt` | Thread tracing |
| `--profile-from-start off` | `nsys profile -c cudaProfilerApi` | Deferred start |
| `--profile-api-trace` | `nsys profile --trace=cuda` | API tracing |
| `--device-buffer-size` | `nsys profile --buffer-size` | Buffer size |
| `--concurrent-kernels` | `nsys profile --trace=cuda` | Concurrent kernel view |
| `--openacc-profiling off` | `nsys profile --trace=cuda,nvtx` | OpenACC (via NVTX) |
| `--dependency-events` | `nsys profile --trace=cuda` | Dependency tracking |
| `--replay-mode` | N/A in nsys (use ncu) | Kernel replay |
| `--clock-period` | N/A in nsys | Sampling period |

### Next Steps After Migration

1. **Replace nvprof with nsys for timeline analysis**: Use `nsys profile` for collecting traces and the Nsight Systems GUI for visualization.

2. **Use Nsight Compute for kernel profiling**: For detailed per-kernel metrics (occupancy, memory throughput, instruction mix), use `ncu` (Nsight Compute CLI):
   ```bash
   ncu --set full -o report ./my_application
   ```

3. **Update scripts**: Replace nvprof CLI invocations with nsys equivalents:
   ```bash
   # Old
   nvprof --export-profile profile.nvprof ./my_application

   # New
   nsys profile -o profile ./my_application
   nsys stats profile.nsys-rep
   ```

4. **Learn the new workflow**:
   - `nsys profile` collects the trace.
   - `nsys stats` generates text-based reports.
   - `nsys export` converts to SQLite or other formats.
   - Nsight Systems GUI provides interactive visualization.

5. **Key differences to remember**:
   - nsys produces `.nsys-rep` files (not `.nvprof`).
   - nsys GUI is separate from the CLI (launch with `nsys-ui`).
   - nsys has much richer CPU profiling capabilities.
   - nsys supports multi-GPU and system-wide profiling.
   - nsys does not replace detailed kernel analysis (use ncu).

---

## Nsight Systems Plugins

Nsight Systems Plugins (Preview) extend the functionality of Nsight Systems through custom analysis modules.

### What is a Plugin

A plugin is a collection of files that adds new analysis capabilities to Nsight Systems:

- **Manifest file**: Describes the plugin metadata and capabilities.
- **Python scripts**: Implement the analysis logic.
- **Resource files**: Icons, strings, and other assets.

Plugins can:
- Add custom analysis passes to the report pipeline.
- Provide new views in the GUI.
- Export data in custom formats.
- Implement domain-specific analysis rules.

### Manifest File

The manifest file (`manifest.yml`) describes the plugin:

```yaml
# manifest.yml
name: My Custom Analysis
version: 1.0.0
description: Custom analysis plugin for my domain
author: My Name
nsys_version: "2025.2"

# Entry points
entry_points:
  - type: analysis
    name: custom-analysis
    description: Runs custom analysis on profiling data
    script: custom_analysis.py
    function: run_analysis

  - type: export
    name: custom-export
    description: Exports data in custom format
    script: custom_export.py
    function: export_data

# Configuration schema
config:
  properties:
    threshold:
      type: number
      default: 0.5
      description: Analysis threshold (0.0 to 1.0)
    verbose:
      type: boolean
      default: false
      description: Enable verbose output

# Supported platforms
platforms:
  - linux-x86_64
  - linux-aarch64
  - windows-x86_64
```

### How to Launch and Pass Arguments

#### Launching with a Plugin

```bash
# Run plugin analysis on a report
nsys plugin run --name=my-plugin report.nsys-rep

# With arguments
nsys plugin run --name=my-plugin --args="--threshold=0.8 --verbose" report.nsys-rep

# List available plugins
nsys plugin list

# Install a plugin
nsys plugin install /path/to/plugin/directory
```

#### Plugin Arguments

Arguments can be passed via:

1. **Command line**: `--args="key=value"`
2. **Config file**: `--config=plugin_config.json`
3. **Environment variables**: `NSYS_PLUGIN_<NAME>_<KEY>=value`

### Supported Platforms

| Platform | Architecture | Status |
|---|---|---|
| Linux | x86_64 | Supported |
| Linux | aarch64 (ARM64) | Supported |
| Windows | x86_64 | Supported |
| QNX | aarch64 | Limited support |

### ImportNvtxt Utility

The ImportNvtxt utility creates and manages NVTX text (`.nvtxt`) annotation files that can be imported into Nsight Systems reports.

#### Info Command

Display information about an `.nvtxt` file:

```bash
nsys-import-nvtxt info annotations.nvtxt
```

Output:
```
NVTX Text File Information:
  Version: 1.0
  Domains: 3
  Total ranges: 1500
  Total events: 200
  Time range: 0.000s - 10.234s
```

#### Create Command

Create a new `.nvtxt` file:

```bash
nsys-import-nvtxt create \
    --output=annotations.nvtxt \
    --format=csv \
    annotations.csv
```

Input CSV format:
```csv
timestamp_ns,name,domain,category,color
1000000,Initialize, MyApp,1,#FF0000
2000000,Train, MyApp,2,#00FF00
8000000,Validate, MyApp,3,#0000FF
```

#### Merge Command

Merge an `.nvtxt` file with an existing report:

```bash
nsys-import-nvtxt merge \
    --input=annotations.nvtxt \
    --report=profile.nsys-rep \
    --output=merged_report.nsys-rep
```

Merge options:

| Option | Description |
|---|---|
| `--input` | Path to the `.nvtxt` file |
| `--report` | Path to the `.nsys-rep` file |
| `--output` | Path for the merged output |
| `--time-offset` | Offset to add to all timestamps (ns) |
| `--domain-prefix` | Prefix for all domain names |
| `--overwrite` | Overwrite existing output file |

---

## Handling Application Launchers

When profiling distributed or multi-process applications, the application is often launched through a launcher like `mpirun`, `torchrun`, or `deepspeed`. Special handling is required.

### Single Process / Subset Profiling with Wrapper Scripts

#### Profiling a Single Rank with mpirun

Create a wrapper script that only profiles a specific rank:

```bash
#!/bin/bash
# profile_rank0.sh
if [ "$OMPI_COMM_WORLD_RANK" = "0" ]; then
    nsys profile -o report_rank0 "$@"
else
    exec "$@"
fi
```

Usage:
```bash
mpirun -np 4 ./profile_rank0.sh python train.py
```

#### Profiling a Subset of Ranks

```bash
#!/bin/bash
# profile_ranks.sh
RANK=${OMPI_COMM_WORLD_RANK:-$PMI_RANK}
case $RANK in
    0|1)
        nsys profile -o report_rank${RANK} "$@"
        ;;
    *)
        exec "$@"
        ;;
esac
```

#### Environment Variables for Rank Detection

| Launcher | Variable | Description |
|---|---|---|
| Open MPI | `OMPI_COMM_WORLD_RANK` | Global rank |
| Open MPI | `OMPI_COMM_WORLD_LOCAL_RANK` | Per-node rank |
| MPICH | `PMI_RANK` | Global rank |
| SLURM | `SLURM_PROCID` | Global task ID |
| SLURM | `SLURM_LOCALID` | Local task ID |
| torchrun | `LOCAL_RANK` | Local rank |
| torchrun | `RANK` | Global rank |
| DeepSpeed | `LOCAL_RANK` | Local rank |
| DeepSpeed | `RANK` | Global rank |

### DeepSpeed Profiling

#### Basic DeepSpeed Profiling

```bash
# Profile all ranks (generates one report per rank)
deepspeed --num_gpus=4 train.py --profile

# Or use nsys directly with DeepSpeed
nsys profile -o ds_report \
    deepspeed --num_gpus=4 train.py
```

#### Profile a Single DeepSpeed Rank

```bash
#!/bin/bash
# ds_profile.sh
if [ "$LOCAL_RANK" = "0" ]; then
    nsys profile -o ds_rank0 \
        --trace=cuda,nvtx,osrt \
        --sample=cpu \
        --python-sampling=true \
        --output=ds_rank0 \
        "$@"
else
    exec "$@"
fi
```

```bash
deepspeed --num_gpus=4 ./ds_profile.sh train.py
```

#### DeepSpeed with Nsight Systems Integration

DeepSpeed has built-in profiling support:

```json
// ds_config.json
{
    "comms_logger": {
        "enabled": true
    },
    "flops_profiler": {
        "enabled": true,
        "profile_step": 5,
        "module_depth": -1,
        "top_modules": 3
    }
}
```

#### Profiling Multi-Node DeepSpeed

```bash
# On each node, run with appropriate rank filtering
deepspeed --num_nodes=2 --num_gpus=4 --hostfile=hostfile \
    ./ds_profile.sh train.py
```

Ensure the output directory is shared (NFS) or use unique filenames per rank:

```bash
# In the wrapper script
nsys profile -o /shared/output/ds_node${SLURM_NODEID}_rank${LOCAL_RANK} "$@"
```

### Profiling torchrun

```bash
# Profile a specific rank with torchrun
torchrun --nproc_per_node=4 train.py

# Using a wrapper script
#!/bin/bash
if [ "$LOCAL_RANK" = "0" ]; then
    nsys profile -o torch_rank0 --trace=cuda,nvtx python "$@"
else
    python "$@"
fi

torchrun --nproc_per_node=4 ./wrapper.sh train.py
```

### Common Issues with Application Launchers

| Issue | Cause | Solution |
|---|---|---|
| Multiple reports overwrite each other | All ranks use same output filename | Include rank ID in filename: `-o report_${RANK}` |
| Disk space exhaustion | Each rank generates a large report | Profile only a subset of ranks |
| Startup synchronization issues | Profiling adds delay to profiled rank | Use warmup period or deferred start |
| Signal propagation | nsys intercepts signals | Use `--kill=none` or adjust signal handling |
| Environment not propagated | Launcher does not forward env vars | Use wrapper script to set environment |

---

## See Also

- [CLI Reference](02-cli-reference.md)
- [Python and CPU Profiling](09-python-cpu-profiling.md)
- [Export Formats and SQLite Schema](11-export-sqlite-schema.md)
- [Release Notes and Troubleshooting](12-release-notes-troubleshooting.md)
