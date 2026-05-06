# Launcher

## Overview

The DeepSpeed launcher (`deepspeed/launcher/`) provides a comprehensive multi-process and multi-node orchestration system for launching distributed training jobs. It handles resource discovery, process spawning, environment configuration, SSH-based remote execution, elastic training coordination, and log management. The launcher is invoked primarily through the `deepspeed` CLI command and serves as the primary entry point for starting distributed DeepSpeed training.

## Source Code Structure

```
deepspeed/launcher/
    __init__.py               # Module exports
    runner.py                 # Main launch logic, argument parsing, process spawning
    multinode_runner.py       # Multi-node SSH-based remote execution
    launcher_helper.py        # Helper functions for environment setup, PID tracking
    constants.py              # Launcher constants and defaults
```

Additionally, the CLI entry points are defined in:

```
bin/
    deepspeed                 # Main CLI entry point
    ds_report                 # System environment report
    ds_ssh                    # SSH helper for multi-node
    ds_elastic                # Elastic training launcher
    ds_bench                  # Benchmarking utility
    ds_io                     # IO benchmark utility
    ds_nvme_tune              # NVMe tuning utility
```

## CLI Entry Point: deepspeed

### Basic Usage

```bash
# Single-node, all GPUs
deepspeed train.py --deepspeed ds_config.json

# Single-node, specific GPU count
deepspeed --num_gpus=4 train.py --deepspeed ds_config.json

# Single-node, specific GPUs
deepspeed --include="localhost:0,1,2,3" train.py --deepspeed ds_config.json

# Multi-node via hostfile
deepspeed --hostfile=myhostfile --num_nodes=2 train.py --deepspeed ds_config.json

# Multi-node with explicit hosts
deepspeed --hostfile=none --num_nodes=2 --num_gpus=4 \
    --master_addr=192.168.1.1 --master_port=29500 \
    train.py --deepspeed ds_config.json

# With autotuning
deepspeed --autotuning=run train.py
```

### Full CLI Syntax

```
deepspeed [DEEPSPEED_ARGS] ENTRY_PROGRAM [USER_ARGS]

Positional Arguments:
  ENTRY_PROGRAM             Python training script to launch

DeepSpeed Arguments:
  -H, --hostfile            Hostfile path (default: None, uses /job/hostfile)
  --include STR             Specify resources to include (e.g., "node1:0-3,node2:0-3")
  --exclude STR             Specify resources to exclude
  --num_nodes INT           Number of nodes to use (default: -1, uses all available)
  --num_gpus INT            Number of GPUs per node (default: -1, uses all available)
  --master_addr STR         Master node address (default: "127.0.0.1")
  --master_port INT         Master node port (default: 29500)
  --force_multi             Force multi-node execution even with single node
  --launcher LOADER         Launcher backend: "pdsh", "openmpi", "mvapich", "slurm" (default: "pdsh")
  --launcher_args STR       Additional arguments for the launcher backend
  --module                  Run ENTRY_PROGRAM as a Python module
  --no_python               ENTRY_PROGRAM is not a Python script
  --enable_each_rank_log    Enable per-rank log files
  --output LOGDIR           Directory for per-rank log output
  --elastic_training        Enable elastic training mode
  --bind_cores_to_rank      Bind CPU cores to each rank
  --bind_core_list STR      List of CPU cores to bind to (e.g., "0-31")
  --autotuning STR          Autotuning mode: "run" to enable

User Arguments:
  All arguments after ENTRY_PROGRAM are passed to the training script
  --deepspeed STR           Path to DeepSpeed configuration JSON
```

## parse_args(): Launch Argument Parsing

### Function Signature

```python
def parse_args():
    """Parse command-line arguments for the DeepSpeed launcher."""
    parser = argparse.ArgumentParser(
        description="DeepSpeed Distributed Training Launcher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ... argument definitions ...
    return parser.parse_args()
```

### Parsed Arguments Object

The parsed arguments are returned as a namespace with the following attributes:

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `hostfile` | `str` or `None` | `None` | Path to hostfile |
| `include` | `str` or `None` | `None` | Resource inclusion filter |
| `exclude` | `str` or `None` | `None` | Resource exclusion filter |
| `num_nodes` | `int` | `-1` | Number of nodes (-1 = all) |
| `num_gpus` | `int` | `-1` | GPUs per node (-1 = all) |
| `master_addr` | `str` | `"127.0.0.1"` | Master address |
| `master_port` | `int` | `29500` | Master port |
| `force_multi` | `bool` | `False` | Force multi-node mode |
| `launcher` | `str` | `"pdsh"` | Launcher backend |
| `launcher_args` | `str` or `None` | `None` | Extra launcher arguments |
| `module` | `bool` | `False` | Run as module |
| `no_python` | `bool` | `False` | Not a Python script |
| `enable_each_rank_log` | `bool` | `False` | Per-rank logging |
| `output` | `str` | `""` | Log output directory |
| `elastic_training` | `bool` | `False` | Elastic training |
| `bind_cores_to_rank` | `bool` | `False` | CPU core binding |
| `bind_core_list` | `str` or `None` | `None` | CPU core list |
| `autotuning` | `str` | `None` | Autotuning mode |
| `user_args` | `list[str]` | `[]` | Arguments for training script |
| `deepspeed_config` | `str` or `None` | `None` | Path to ds_config.json |

## launch(): Main Launch Function

### Function Signature

```python
def launch(args):
    """Main launch function that orchestrates distributed training startup."""
```

### Launch Flow

```
launch(args)
    |
    +-- 1. Resource Discovery
    |   +-- parse_hostfile() or infer_resources()
    |   +-- Apply include/exclude filters
    |   +-- Determine num_nodes and num_gpus
    |
    +-- 2. Environment Setup
    |   +-- Set MASTER_ADDR, MASTER_PORT
    |   +-- Generate world_size = num_nodes * num_gpus
    |   +-- Construct launch command
    |
    +-- 3. Process Launch
    |   +-- Single-node: fetch_host(localhost) + spawn_processes()
    |   +-- Multi-node: MultiNodeRunner.start()
    |       +-- SSH to each node
    |       +-- Launch worker processes
    |       +-- Coordinate via rendezvous
    |
    +-- 4. Monitoring & Cleanup
        +-- Monitor running processes
        +-- Handle failures and restarts
        +-- Collect logs
        +-- Clean up on completion
```

## Multi-Node Coordination

### Hostfile Format

The hostfile defines the available nodes and their GPU resources. It uses a simple text format:

```
# Comment lines start with #
# Format: hostname slots=N
worker1 slots=4
worker2 slots=8
worker3 slots=8
worker4 slots=4
```

**Rules:**
- Each line specifies one host and its number of GPU slots
- Lines starting with `#` are comments
- Empty lines are ignored
- The first host listed becomes the master node (unless `master_addr` is set explicitly)
- `slots` indicates the number of available GPUs on that host

### Hostfile Parsing

```python
def parse_hostfile(hostfile_path):
    """Parse a hostfile and return a dictionary of hosts and their slot counts."""
    hosts = {}
    with open(hostfile_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            hostname = parts[0]
            slots = int(parts[1].split("=")[1]) if len(parts) > 1 else 1
            hosts[hostname] = slots
    return hosts
```

### Default Hostfile Locations

DeepSpeed searches for the hostfile in the following order:

1. `--hostfile` command-line argument
2. `DS_HOSTFILE` environment variable
3. `/job/hostfile` (common in cluster environments)
4. If none found: single-node mode using local GPU count

### Resource Filtering: include/exclude

The `--include` and `--exclude` flags allow fine-grained control over which GPUs are used.

**Include syntax:**
```bash
# Use specific GPUs on specific nodes
--include="worker1:0-3,worker2:0-1"

# Use all GPUs on specific nodes
--include="worker1,worker2"

# Mix formats
--include="worker1:0,2,worker2:0-7"
```

**Exclude syntax:**
```bash
# Exclude specific GPUs
--exclude="worker1:0,1"

# Exclude entire nodes
--exclude="worker3"

# Exclude GPU ranges
--exclude="worker1:4-7"
```

**Priority:** `--include` and `--exclude` are mutually exclusive. If both are specified, `--include` takes precedence. After applying inclusion, exclusion is applied to the result.

### SSH Setup for Multi-Node

For multi-node training, DeepSpeed requires passwordless SSH access from the master node to all worker nodes.

**Setup steps:**

```bash
# 1. Generate SSH key (if not already present)
ssh-keygen -t rsa -b 4096

# 2. Copy key to all worker nodes
ssh-copy-id user@worker1
ssh-copy-id user@worker2

# 3. Verify connectivity
ds_ssh "hostname"  # Should print all worker hostnames

# 4. Alternative: use ds_ssh to test
deepspeed --hostfile=myhostfile --num_nodes=2 \
    --launcher_args="--timeout=30" \
    train.py --deepspeed ds_config.json
```

**SSH Configuration Options:**
- DeepSpeed uses `pdsh` by default for multi-node launch
- Set `PDSH_RCMD_TYPE=ssh` to ensure SSH is used
- The `--launcher_args` flag passes additional arguments to the launcher backend
- For environments without `pdsh`, use `--launcher=openmpi` or `--launcher=slurm`

### Multi-Node Runner

The `MultiNodeRunner` class in `multinode_runner.py` handles remote process execution:

```python
class MultiNodeRunner:
    def __init__(self, launcher, world_info, hostfile):
        self.launcher = launcher          # "pdsh", "openmpi", "mvapich", "slurm"
        self.world_info = world_info      # Dict mapping hosts to rank assignments
        self.hostfile = hostfile          # Parsed hostfile dictionary

    def start(self, cmd, env):
        """Launch processes on all nodes."""
        if self.launcher == "pdsh":
            return self._start_pdsh(cmd, env)
        elif self.launcher == "openmpi":
            return self._start_openmpi(cmd, env)
        elif self.launcher == "mvapich":
            return self._start_mvapich(cmd, env)
        elif self.launcher == "slurm":
            return self._start_slurm(cmd, env)
```

**Launcher Backends:**

| Backend | Description | Requirements |
|---------|-------------|--------------|
| `pdsh` | Uses pdsh for parallel remote execution (default) | `pdsh` installed on all nodes |
| `openmpi` | Uses OpenMPI's `mpirun` | OpenMPI installation |
| `mvapich` | Uses MVAPICH's `mpirun` | MVAPICH installation |
| `slurm` | Uses Slurm's `srun` | Slurm workload manager |

## Resource Management

### GPU Allocation

```python
# Resource allocation logic (simplified)
def allocate_resources(hostfile, num_nodes, num_gpus, include, exclude):
    """
    Allocate GPU resources based on hostfile and filters.

    Returns:
        dict: {hostname: [list of GPU indices to use]}
    """
    # Start with all resources from hostfile
    resources = {host: list(range(slots)) for host, slots in hostfile.items()}

    # Apply include filter
    if include:
        resources = apply_include(resources, include)

    # Apply exclude filter
    if exclude:
        resources = apply_exclude(resources, exclude)

    # Limit number of nodes
    if num_nodes > 0:
        resources = dict(list(resources.items())[:num_nodes])

    # Limit GPUs per node
    if num_gpus > 0:
        resources = {host: gpus[:num_gpus] for host, gpus in resources.items()}

    return resources
```

### GPU Exclusion

DeepSpeed can exclude specific GPUs from training. This is useful when:

- Some GPUs are faulty
- Some GPUs are needed for other tasks
- Running on heterogeneous hardware

```bash
# Exclude GPUs 4-7 on node worker1
deepspeed --exclude="worker1:4-7" train.py --deepspeed ds_config.json

# Exclude an entire node
deepspeed --exclude="worker3" --hostfile=myhostfile train.py --deepspeed ds_config.json

# Use only GPUs 0-3 on all nodes
deepspeed --num_gpus=4 train.py --deepspeed ds_config.json

# Use only GPUs 2 and 3 on worker1
deepspeed --include="worker1:2,3" train.py --deepspeed ds_config.json
```

### CUDA_VISIBLE_DEVICES Management

DeepSpeed automatically sets `CUDA_VISIBLE_DEVICES` for each process based on the resource allocation:

```python
# For a rank assigned to GPU 2 on a node with 8 GPUs:
# The launcher sets:
env["CUDA_VISIBLE_DEVICES"] = "2"

# For a rank assigned to GPUs 0,2 on a node:
# Only one GPU per process (standard DDP), so:
env["CUDA_VISIBLE_DEVICES"] = "2"  # The specific GPU for this rank
```

## Process Binding to CPU Cores

### Overview

DeepSpeed can bind each training process to specific CPU cores to improve cache locality and reduce CPU contention. This is especially beneficial for high-performance data loading and CPU-bound operations.

### Configuration

```bash
# Enable automatic core binding
deepspeed --bind_cores_to_rank train.py --deepspeed ds_config.json

# Specify which cores to use
deepspeed --bind_cores_to_rank --bind_core_list="0-31" train.py --deepspeed ds_config.json
```

### Core Allocation Logic

```python
def bind_process_to_core(local_rank, num_processes, core_list):
    """
    Bind a process to a specific set of CPU cores.

    Args:
        local_rank: Local rank of the process (0-indexed)
        num_processes: Total number of processes on this node
        core_list: List of available CPU core indices
    """
    cores_per_process = len(core_list) // num_processes
    start_core = local_rank * cores_per_process
    end_core = start_core + cores_per_process
    my_cores = core_list[start_core:end_core]

    # Set CPU affinity
    os.sched_setaffinity(0, my_cores)
```

### Example: 8 GPUs, 64 CPU Cores

```
Without binding:
  All 8 processes compete for all 64 cores

With binding (--bind_cores_to_rank):
  Rank 0: cores 0-7
  Rank 1: cores 8-15
  Rank 2: cores 16-23
  Rank 3: cores 24-31
  Rank 4: cores 32-39
  Rank 5: cores 40-47
  Rank 6: cores 48-55
  Rank 7: cores 56-63
```

## Log File Management

### Per-Rank Logging

When `--enable_each_rank_log` is enabled, DeepSpeed creates separate log files for each rank:

```bash
deepspeed --enable_each_rank_log --output=/logs/train \
    train.py --deepspeed ds_config.json
```

**Log file structure:**
```
/logs/train/
    rank0.log
    rank1.log
    rank2.log
    rank3.log
    ...
```

### Log File Naming Convention

```
{output_dir}/rank{RANK}.log
```

Where:
- `output_dir` is specified by `--output`
- `RANK` is the global rank of the process (0-indexed)

### stdout/stderr Redirection

When per-rank logging is enabled, the launcher redirects `stdout` and `stderr` for each process:

```python
# Internal redirection logic
if enable_rank_log:
    log_dir = Path(output_dir) / f"rank{rank}"
    log_dir.mkdir(parents=True, exist_ok=True)

    stdout_file = open(log_dir / "rank{rank}.log", "w")
    stderr_file = open(log_dir / "rank{rank}_error.log", "w")

    process = subprocess.Popen(
        cmd,
        stdout=stdout_file,
        stderr=stderr_file,
        env=env,
    )
```

### Log Viewing

```bash
# View all rank logs simultaneously
tail -f /logs/train/rank*.log

# View a specific rank's log
cat /logs/train/rank0.log

# Search for errors across all ranks
grep -l "Error" /logs/train/rank*_error.log
```

## PID Tracking

### Overview

DeepSpeed tracks process IDs (PIDs) for all launched worker processes to enable monitoring, health checks, and cleanup.

### PID File Location

```
{output_dir}/.deepspeed_pids
```

Or in the system temporary directory if `output_dir` is not specified:

```
/tmp/deepspeed_{timestamp}/.deepspeed_pids
```

### PID File Format

```
# node_hostname, pid, rank, local_rank, command
worker1,12345,0,0,python train.py --deepspeed ds_config.json
worker1,12346,1,1,python train.py --deepspeed ds_config.json
worker1,12347,2,2,python train.py --deepspeed ds_config.json
worker1,12348,3,3,python train.py --deepspeed ds_config.json
worker2,23456,4,0,python train.py --deepspeed ds_config.json
worker2,23457,5,1,python train.py --deepspeed ds_config.json
worker2,23458,6,2,python train.py --deepspeed ds_config.json
worker2,23459,7,3,python train.py --deepspeed ds_config.json
```

### PID Management Functions

```python
# launcher_helper.py

def write_pid_file(pid_info, output_dir):
    """Write PID tracking file."""
    pid_path = os.path.join(output_dir, ".deepspeed_pids")
    with open(pid_path, "w") as f:
        for info in pid_info:
            f.write(f"{info['host']},{info['pid']},{info['rank']},{info['local_rank']},{info['cmd']}\n")

def read_pid_file(output_dir):
    """Read PID tracking file."""
    pid_path = os.path.join(output_dir, ".deepspeed_pids")
    if not os.path.exists(pid_path):
        return []
    with open(pid_path, "r") as f:
        return [line.strip().split(",") for line in f if line.strip()]

def kill_all_processes(output_dir):
    """Kill all tracked processes."""
    for info in read_pid_file(output_dir):
        host, pid = info[0], int(info[1])
        if host == socket.gethostname():
            os.kill(pid, signal.SIGTERM)
        else:
            subprocess.run(["ssh", host, f"kill {pid}"])
```

## World Info Propagation

### Environment Variables

The launcher propagates world information to each worker process through environment variables:

| Variable | Description | Example Value |
|----------|-------------|---------------|
| `MASTER_ADDR` | IP address or hostname of the master node | `"192.168.1.1"` |
| `MASTER_PORT` | Port for the rendezvous server | `"29500"` |
| `RANK` | Global rank of this process (0-indexed) | `"0"` |
| `WORLD_SIZE` | Total number of distributed processes | `"16"` |
| `LOCAL_RANK` | Local rank within the current node | `"0"` |
| `LOCAL_SIZE` | Number of processes on the current node | `"8"` |
| `CUDA_VISIBLE_DEVICES` | GPU(s) visible to this process | `"0"` |

### Environment Construction

```python
def construct_env(rank, local_rank, world_size, local_size,
                  master_addr, master_port, device_per_node):
    """Construct environment variables for a worker process."""
    env = os.environ.copy()
    env.update({
        "MASTER_ADDR": str(master_addr),
        "MASTER_PORT": str(master_port),
        "RANK": str(rank),
        "WORLD_SIZE": str(world_size),
        "LOCAL_RANK": str(local_rank),
        "LOCAL_SIZE": str(local_size),
        "CUDA_VISIBLE_DEVICES": str(device_per_node[local_rank]),
    })
    return env
```

### Multi-Node Environment Propagation

For multi-node training, the launcher propagates environment variables to remote nodes via SSH:

```python
def launch_remote_worker(hostname, cmd, env):
    """Launch a worker process on a remote node via SSH."""
    # Build environment variable exports
    env_exports = " ".join(f"{k}={v}" for k, v in env.items())

    # Construct remote command
    remote_cmd = f"cd {workdir} && {env_exports} {cmd}"

    # Execute via SSH
    ssh_cmd = ["ssh", hostname, remote_cmd]
    process = subprocess.Popen(ssh_cmd)
    return process
```

### Rank Assignment

Ranks are assigned sequentially across nodes:

```
Node 0: GPUs 0-3 -> Ranks 0-3
Node 1: GPUs 0-3 -> Ranks 4-7
Node 2: GPUs 0-3 -> Ranks 8-11
Node 3: GPUs 0-3 -> Ranks 12-15

Rank = node_index * gpus_per_node + local_rank
```

## Elastic Training Support

### Overview

DeepSpeed supports elastic training, which allows training to continue when workers join or leave the training job. This is particularly useful for:

- Shared computing environments where nodes may be preempted
- Spot/preemptible instance training
- Dynamic scaling of resources

### Enabling Elastic Training

```bash
# Via CLI
deepspeed --elastic_training --hostfile=myhostfile train.py --deepspeed ds_config.json

# Via ds_elastic command
ds_elastic train.py --deepspeed ds_config.json
```

### Elastic Configuration

```json
{
    "elastic_training": {
        "enabled": true
    }
}
```

### Elastic Training Flow

```
1. Launcher starts initial workers
2. Workers begin training
3. If a worker fails:
   a. Launcher detects failure (via PID monitoring or heartbeat)
   b. Remaining workers are notified
   c. World size is recomputed
   d. Model state is restored from latest checkpoint
   e. Training resumes with fewer workers
4. If new workers become available:
   a. Launcher starts new worker processes
   b. New workers are integrated into the process group
   c. State is synchronized from existing workers
   d. Training continues with more workers
```

### Integration with Torchrun

DeepSpeed elastic training integrates with PyTorch's `torchrun` for rendezvous and restart:

```bash
# Using torchrun with DeepSpeed
torchrun --nnodes=2 --nproc_per_node=4 --rdzv_id=100 \
    --rdzv_backend=c10d --rdzv_endpoint=master:29500 \
    train.py --deepspeed ds_config.json
```

## DeepSpeed CLI Utilities

### ds_report: System Environment Report

Generates a comprehensive report of the system environment for debugging:

```bash
$ ds_report
```

**Output includes:**
```
--------------------------------------------------
DeepSpeed C++/CUDA extension op report
--------------------------------------------------
NOTE: Ops not installed on the system will only be logged for levels
      higher than 0 or when DS_BUILD_OPS is specified

 aten Adams ....................................... [OKAY]
 aten FusedAdam .................................. [OKAY]
 async_io ......................................... [OKAY]
 cpu_adam ......................................... [OKAY]
 cpu_lion ......................................... [OKAY]
 contiguous_data_movement ......................... [OKAY]
 cpu_adagrad ...................................... [OKAY]
 fused_lamb ....................................... [OKAY]
 fused_lion ....................................... [OKAY]
 quantizer ........................................ [OKAY]
 random_ltd ....................................... [OKAY]
 sparse_attn ...................................... [OKAY]
 spatial_inference ................................ [OKAY]
 transformer_inference ............................ [OKAY]
 utils ............................................ [OKAY]

--------------------------------------------------
DeepSpeed general environment info:
--------------------------------------------------
CUDA device count ......................... 8
CUDA device 0 name ....................... NVIDIA A100-SXM4-80GB
CUDA device 0 CUDA capability ............ 8.0
CUDA device 0 total memory ............... 80.00 GB
CUDA version ............................. 11.8
PyTorch version .......................... 2.1.0
DeepSpeed version ........................ 0.16.0
Python version ........................... 3.10.12
OS version ............................... Ubuntu 22.04
NCCL version ............................. 2.18.1
...
```

### ds_ssh: SSH Helper

Tests SSH connectivity to all nodes in the hostfile:

```bash
# Test connectivity to all nodes
ds_ssh --hostfile myhostfile "hostname"

# Run a command on all nodes
ds_ssh --hostfile myhostfile "nvidia-smi"

# Check GPU status across cluster
ds_ssh --hostfile myhostfile "nvidia-smi --query-gpu=name,memory.total --format=csv"
```

### ds_elastic: Elastic Training Launcher

Launches elastic training with automatic restart and scaling:

```bash
ds_elastic --hostfile myhostfile --max_restarts=3 \
    train.py --deepspeed ds_config.json
```

**Options:**
| Flag | Description |
|------|-------------|
| `--hostfile` | Path to hostfile |
| `--max_restarts` | Maximum number of restart attempts |
| `--workflow` | Workflow configuration file |
| `--rdzv_endpoint` | Rendezvous endpoint |
| `--rdzv_backend` | Rendezvous backend (c10d, etcd) |

### ds_bench: Benchmarking Utility

Runs DeepSpeed benchmarks for various configurations:

```bash
# Run all benchmarks
ds_bench

# Run specific benchmark
ds_bench --benchmarks=flops,comm

# Run with specific configuration
ds_bench --model=bert-base --batch_size=32 --dp=4 --pp=1 --tp=1
```

### ds_io: IO Benchmark Utility

Benchmarks IO performance for checkpoint and data loading:

```bash
# Run IO benchmarks
ds_io --output_dir=/tmp/ds_io_bench

# Benchmark NVMe performance
ds_io --nvme --nvme_path=/mnt/nvme0
```

### ds_nvme_tune: NVMe Tuning Utility

Automatically tunes NVMe configuration for optimal DeepSpeed performance:

```bash
# Auto-tune NVMe settings
ds_nvme_tune --nvme_path=/mnt/nvme0 --output_file=ds_nvme_config.json

# Tune with specific parameters
ds_nvme_tune --nvme_path=/mnt/nvme0 --io_size=4K --num_threads=16
```

**Generated configuration:**
```json
{
    "aio": {
        "block_size": 1048576,
        "queue_depth": 8,
        "thread_count": 2,
        "single_submit": false,
        "overlap_events": true
    }
}
```

## Configuration Examples

### Single-Node, All GPUs

```bash
# Launch with all available GPUs
deepspeed train.py --deepspeed ds_config.json
```

**Equivalent manual setup:**
```bash
# Without DeepSpeed launcher
torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) \
    train.py --deepspeed ds_config.json
```

### Single-Node, Specific GPU Count

```bash
# Use only 4 GPUs
deepspeed --num_gpus=4 train.py --deepspeed ds_config.json
```

**ds_config.json:**
```json
{
    "train_batch_size": 32,
    "gradient_accumulation_steps": 1,
    "zero_optimization": {
        "stage": 2
    },
    "bf16": {
        "enabled": true
    }
}
```

### Multi-Node with Hostfile

**hostfile:**
```
master slots=8
worker1 slots=8
worker2 slots=8
```

**Launch command:**
```bash
deepspeed --hostfile=hostfile --num_nodes=3 \
    train.py --deepspeed ds_config.json
```

**ds_config.json:**
```json
{
    "train_batch_size": 192,
    "gradient_accumulation_steps": 4,
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "contiguous_gradients": true
    },
    "bf16": {
        "enabled": true
    },
    "gradient_accumulation_steps": 4
}
```

### Multi-Node with Resource Filtering

```bash
# Use only GPUs 0-3 on each node
deepspeed --hostfile=hostfile --num_gpus=4 \
    train.py --deepspeed ds_config.json

# Exclude specific GPUs
deepspeed --hostfile=hostfile --exclude="master:4-7" \
    train.py --deepspeed ds_config.json

# Include only specific nodes and GPUs
deepspeed --include="master:0-3,worker1:0-3" \
    train.py --deepspeed ds_config.json
```

### Multi-Node with OpenMPI

```bash
# Use OpenMPI launcher instead of pdsh
deepspeed --hostfile=hostfile --launcher=openmpi \
    train.py --deepspeed ds_config.json

# With extra MPI arguments
deepspeed --hostfile=hostfile --launcher=openmpi \
    --launcher_args="-mca btl_tcp_if_include eth0" \
    train.py --deepspeed ds_config.json
```

### Slurm Integration

```bash
# Use Slurm launcher
deepspeed --launcher=slurm train.py --deepspeed ds_config.json
```

**Alternatively, use Slurm's native srun with DeepSpeed:**
```bash
#!/bin/bash
#SBATCH --job-name=deepspeed-training
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8

srun deepspeed train.py --deepspeed ds_config.json
```

### Elastic Training Launch

```bash
# Enable elastic training with automatic restart
deepspeed --elastic_training --hostfile=hostfile \
    train.py --deepspeed ds_config.json
```

**ds_config.json with elastic config:**
```json
{
    "train_batch_size": 64,
    "elastic_training": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2
    }
}
```

### Per-Rank Logging

```bash
# Enable per-rank log files
deepspeed --enable_each_rank_log --output=/logs/training \
    --num_gpus=8 train.py --deepspeed ds_config.json

# View logs during training
tail -f /logs/training/rank0.log

# Search for errors
grep -r "Error" /logs/training/
```

### CPU Core Binding

```bash
# Auto-bind cores to ranks
deepspeed --bind_cores_to_rank --num_gpus=4 \
    train.py --deepspeed ds_config.json

# Specify core range
deepspeed --bind_cores_to_rank --bind_core_list="0-31" --num_gpus=4 \
    train.py --deepspeed ds_config.json
```

### Mixed Precision Training Launch

```bash
# BF16 training
deepspeed --num_gpus=8 train.py --deepspeed ds_config_bf16.json

# FP16 training
deepspeed --num_gpus=8 train.py --deepspeed ds_config_fp16.json
```

**ds_config_bf16.json:**
```json
{
    "train_batch_size": 64,
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu"
        }
    }
}
```

### Pipeline + Tensor Parallelism Launch

```bash
# 2-way pipeline parallelism, 4-way tensor parallelism, 2-way data parallelism
# Total: 16 GPUs across 2 nodes
deepspeed --hostfile=hostfile --num_nodes=2 --num_gpus=8 \
    train.py --deepspeed ds_config_3d.json \
    --pipeline_model_parallel_size 2 \
    --tensor_model_parallel_size 4
```

**ds_config_3d.json:**
```json
{
    "train_batch_size": 128,
    "gradient_accumulation_steps": 8,
    "zero_optimization": {
        "stage": 0
    },
    "bf16": {
        "enabled": true
    },
    "pipeline": {
        "enabled": true,
        "parallel_size": 2
    },
    "tensor_pipeline": {
        "enabled": true,
        "tp_size": 4
    }
}
```

## Environment Variables Reference

### DeepSpeed Launcher Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DS_HOSTFILE` | Default hostfile path | `/job/hostfile` |
| `DS_REPORTER` | Reporter backend for launcher | None |
| `DS_ACCELERATOR` | Accelerator type override | Auto-detected |

### PyTorch Distributed Variables

| Variable | Description | Set By |
|----------|-------------|--------|
| `MASTER_ADDR` | Master node address | DeepSpeed launcher |
| `MASTER_PORT` | Master node port | DeepSpeed launcher |
| `RANK` | Global rank | DeepSpeed launcher |
| `WORLD_SIZE` | Total process count | DeepSpeed launcher |
| `LOCAL_RANK` | Local rank on node | DeepSpeed launcher |
| `TORCH_DISTRIBUTED_DEBUG` | PyTorch distributed debug level | User |
| `TORCH_CPP_LOG_LEVEL` | C++ log level | User |

### NCCL Variables (NVIDIA)

| Variable | Description | Typical Value |
|----------|-------------|---------------|
| `NCCL_SOCKET_IFNAME` | Network interface for TCP | `"eth0"`, `"ib0"` |
| `NCCL_IB_DISABLE` | Disable InfiniBand | `"0"` or `"1"` |
| `NCCL_IB_GID_INDEX` | IB GID index | `"3"` |
| `NCCL_DEBUG` | NCCL debug level | `"INFO"`, `"WARN"` |
| `NCCL_NET_GDR_LEVEL` | GPUDirect RDMA level | `"5"` |
| `NCCL_ALGO` | NCCL algorithm | `"Ring"`, `"Tree"` |
| `NCCL_PROTO` | NCCL protocol | `"Simple"`, `"LL"` |
| `NCCL_MIN_NCHANNELS` | Min number of channels | `"4"` |
| `NCCL_MAX_NCHANNELS` | Max number of channels | `"32"` |

### CCL Variables (Intel)

| Variable | Description | Typical Value |
|----------|-------------|---------------|
| `CCL_ATL_TRANSPORT` | CCL transport | `"ofi"`, `"shm"` |
| `CCL_KVS` | Key-value store | `"mpi"` |
| `CCL_DEBUG` | CCL debug level | `"info"`, `"warn"` |

## Troubleshooting

### Common Issues

**1. "NCCL error: unhandled system error"**
```bash
# Solution: Set network interface explicitly
export NCCL_SOCKET_IFNAME=eth0
deepspeed --hostfile=hostfile train.py --deepspeed ds_config.json
```

**2. "SSH connection refused"**
```bash
# Solution: Verify SSH connectivity
ds_ssh --hostfile hostfile "echo connected"

# If using non-default SSH port
export PDSH_SSH_ARGS="-p 2222"
```

**3. "Port already in use"**
```bash
# Solution: Use a different port
deepspeed --master_port=29501 train.py --deepspeed ds_config.json
```

**4. "CUDA out of memory"**
```bash
# Solution: Reduce batch size or use ZeRO offloading
# In ds_config.json:
{
    "train_batch_size": 16,
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "cpu"}
    }
}
```

**5. "Rank-0 heartbeat timeout"**
```bash
# Solution: Increase timeout or check network
export NCCL_TIMEOUT=600  # 10 minutes
deepspeed --hostfile=hostfile train.py --deepspeed ds_config.json
```
