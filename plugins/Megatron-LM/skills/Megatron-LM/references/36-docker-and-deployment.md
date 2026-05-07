# Chapter 36: Docker and Deployment

## Source Files
- `sources/Megatron-LM/docker/` - Docker configurations
- `sources/Megatron-LM/Dockerfile` - Main Dockerfile

## Overview

Megatron-LM supports deployment through Docker containers and various cluster management systems. The recommended approach uses NVIDIA NGC containers with pre-installed dependencies.

## Docker Options

### Option 1: NGC PyTorch Container (Recommended)

```bash
# Pull the NGC container
docker pull nvcr.io/nvidia/pytorch:26.01-py3

# Run with GPU access
docker run --gpus all -it --rm \
    --shm-size=256g \
    -v /path/to/data:/workspace/data \
    -v /path/to/checkpoints:/workspace/checkpoints \
    -e PIP_CONSTRAINT= \
    nvcr.io/nvidia/pytorch:26.01-py3

# Install Megatron Core inside container
pip install uv
uv pip install --no-build-isolation "megatron-core[training,dev]"
```

### Option 2: Build Custom Docker Image

```dockerfile
FROM nvcr.io/nvidia/pytorch:26.01-py3

# Unset pip constraints
ENV PIP_CONSTRAINT=

# Install Megatron-LM
RUN pip install uv && \
    uv pip install --no-build-isolation "megatron-core[training,dev]"

# Clone for examples
RUN git clone https://github.com/NVIDIA/Megatron-LM.git /workspace/Megatron-LM

WORKDIR /workspace/Megatron-LM
```

```bash
# Build
docker build -t megatron-lm:latest .

# Run
docker run --gpus all -it --rm \
    --shm-size=256g \
    -v /path/to/data:/workspace/data \
    megatron-lm:latest
```

### Docker Run Options

| Flag | Description | Recommended |
|---|---|---|
| `--gpus all` | Access all GPUs | Required |
| `--shm-size=256g` | Shared memory size | 256g for large models |
| `--ulimit memlock=-1` | Unlimited memory lock | Recommended |
| `--ulimit stack=67108864` | Stack size | Recommended |
| `--network=host` | Host networking | For multi-node |
| `--ipc=host` | Host IPC | For multi-node |

## Multi-Node Deployment

### Docker Swarm
```bash
# Initialize swarm
docker swarm init --advertise-addr MASTER_IP

# On worker nodes
docker swarm join --token TOKEN MASTER_IP:2377

# Deploy with docker-compose
docker stack deploy -c docker-compose.yml megatron
```

### Kubernetes

```yaml
# megatron-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: megatron-training
spec:
  parallelism: 4  # Number of nodes
  template:
    spec:
      containers:
      - name: megatron
        image: megatron-lm:latest
        resources:
          limits:
            nvidia.com/gpu: 8
        command: ["torchrun"]
        args:
          - "--nproc_per_node=8"
          - "--nnodes=4"
          - "pretrain_gpt.py"
          - [additional args...]
      restartPolicy: OnFailure
```

## SLURM Integration

### Basic SLURM Script
```bash
#!/bin/bash
#SBATCH --job-name=megatron-gpt
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --output=megatron-%j.log

# Load modules
module load cuda/12.1
module load nccl/2.18

# Set up environment
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=6000
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Run training
srun torchrun \
    --nproc_per_node=8 \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    pretrain_gpt.py \
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 2 \
    --num-layers 32 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --seq-length 4096 \
    --micro-batch-size 2 \
    --global-batch-size 512 \
    --train-iters 100000 \
    --lr 1e-4 \
    --bf16 \
    --data-path /data/training_data_text_document \
    --save /checkpoints/gpt-7b \
    --save-interval 1000
```

### Advanced SLURM with Pyxis (Enroot)
```bash
#!/bin/bash
#SBATCH --job-name=megatron-ngc
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --container-image=nvcr.io/nvidia/pytorch:26.01-py3
#SBATCH --container-mounts=/data:/workspace/data,/checkpoints:/workspace/checkpoints

export PIP_CONSTRAINT=
pip install uv && uv pip install --no-build-isolation "megatron-core[training,dev]"

srun torchrun --nproc_per_node=8 --nnodes=$SLURM_JOB_NUM_NODES \
    --master_addr=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n1) \
    --master_port=6000 \
    pretrain_gpt.py [args...]
```

## Network Configuration

### InfiniBand
```bash
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=5
export NCCL_IB_TC=106
```

### Ethernet Fallback
```bash
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
```

### NVLink Only (Single Node)
```bash
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
```

## Storage Configuration

### Shared Filesystem
```bash
# Recommended: Use NFS or Lustre for shared storage
# Checkpoint directory should be on shared filesystem
--save /shared/checkpoints/megatron-gpt

# Data directory
--data-path /shared/data/training_data_text_document
```

### Local Storage
```bash
# Use local SSD for temporary data
--data-cache-path /local/cache

# Copy data to local storage before training
srun --distribution=block:block --ntasks-per-node=1 \
    cp /shared/data/* /local/data/
```
