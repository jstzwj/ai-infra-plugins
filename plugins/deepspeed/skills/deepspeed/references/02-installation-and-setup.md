# DeepSpeed Reference - Chapter 2: Installation and Setup

This chapter covers all aspects of installing DeepSpeed, from basic pip installation to advanced compilation options, Docker containers, cloud platform support, and environment configuration.

---

## 2.1 Prerequisites

### 2.1.1 Software Requirements

| Requirement | Minimum Version | Recommended Version | Notes |
|-------------|----------------|---------------------|-------|
| Python | 3.8 | 3.10+ | Python 3.12+ supported from DeepSpeed 0.14 |
| PyTorch | 1.9 | 2.1+ | Required for all features; 2.0+ for torch.compile integration |
| CUDA Toolkit | 11.0 | 12.1+ | Required for NVIDIA GPUs; version must match PyTorch CUDA |
| cuDNN | 8.0 | 8.9+ | Required for CUDA operations |
| NVIDIA Driver | 450.x | 525.x+ | Must support the CUDA toolkit version |
| MPI (optional) | Any | OpenMPI 4.x | For MPI-based launcher |
| C++ Compiler | C++14 | C++17 | GCC 9+ or Clang 10+ |

### 2.1.2 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU Memory | 8 GB | 40 GB+ (A100, H100) |
| System RAM | 32 GB | 128 GB+ (for CPU offloading) |
| NVMe SSD | Not required | Required for ZeRO-Infinity NVMe offload |
| Network | 1 GbE | InfiniBand HDR (200 Gb/s) or RoCE |

### 2.1.3 Checking Your Environment

Before installing DeepSpeed, verify your environment:

```bash
# Check Python version
python --version

# Check PyTorch installation and CUDA version
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Check GPU info
nvidia-smi

# Check NVIDIA driver version
cat /proc/driver/nvidia/version
```

---

## 2.2 Basic Installation

### 2.2.1 pip Install (Recommended)

The simplest way to install DeepSpeed is via pip:

```bash
# Install latest stable release
pip install deepspeed

# Install a specific version
pip install deepspeed==0.15.0

# Install with all extras
pip install deepspeed[all]

# Install specific extras
pip install deepspeed[autotuning]
pip install deepspeed[1bit]
pip install deepspeed[sparse_attention]
pip install deepspeed[audit]
```

**Available Extras:**

| Extra | Description |
|-------|-------------|
| `[all]` | Install all optional dependencies |
| `[autotuning]` | Dependencies for autotuning feature |
| `[1bit]` | 1-bit Adam/Lamb optimizer dependencies |
| `[sparse_attention]` | Sparse attention dependencies (triton) |
| `[audit]` | Security audit tools |
| `[readthedocs]` | Documentation build dependencies |

### 2.2.2 Install from Source

Installing from source gives you the latest development version and allows customization of the build:

```bash
# Clone the repository
git clone https://github.com/microsoft/DeepSpeed.git
cd DeepSpeed

# Install in development mode (editable)
pip install -e .

# Install with all extras from source
pip install -e .[all]

# Install a specific branch
git checkout v0.15-release
pip install -e .
```

### 2.2.3 Install from Source with Custom Build

For custom build configurations:

```bash
git clone https://github.com/microsoft/DeepSpeed.git
cd DeepSpeed

# Build with specific CUDA architectures
DS_BUILD_OPS=1 DS_BUILD_FUSED_ADAM=1 DS_BUILD_UTILS=1 pip install -e .

# Build all ops from source
DS_BUILD_OPS=1 pip install -e .

# Skip building ops entirely (use PyTorch native ops)
DS_BUILD_OPS=0 pip install -e .
```

---

## 2.3 Build System and Custom Ops

### 2.3.1 Overview of the Op Builder System

DeepSpeed includes custom CUDA/C++ operations that provide significant performance improvements over PyTorch-native implementations. These ops are managed by the `op_builder` system, which handles:

- Detecting available compilers (nvcc, gcc/g++, hipcc for ROCm)
- Determining CUDA/ROCm architecture targets
- Compiling and linking CUDA kernels
- Caching compiled ops to avoid recompilation

The op builder system supports three modes:

1. **JIT (Just-In-Time) Compilation**: Ops are compiled on first use and cached for future runs.
2. **Pre-compiled Installation**: Ops are compiled during `pip install` and included in the package.
3. **Skip Compilation**: Ops are not built; DeepSpeed falls back to PyTorch-native implementations.

### 2.3.2 Build Environment Variables

The following environment variables control the build process:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DS_BUILD_OPS` | `0/1` | `1` | Global toggle: build all ops (1) or none (0) |
| `DS_BUILD_FUSED_ADAM` | `0/1` | `1` | Build fused Adam optimizer kernel |
| `DS_BUILD_FUSED_LAMB` | `0/1` | `1` | Build fused Lamb optimizer kernel |
| `DS_BUILD_FUSED_LION` | `0/1` | `1` | Build fused Lion optimizer kernel |
| `DS_BUILD_CPU_ADAM` | `0/1` | `1` | Build CPU Adam optimizer |
| `DS_BUILD_CPU_ADAGRAD` | `0/1` | `1` | Build CPU Adagrad optimizer |
| `DS_BUILD_UTILS` | `0/1` | `1` | Build utility CUDA ops |
| `DS_BUILD_TRANSFORMER` | `0/1` | `1` | Build transformer inference kernels |
| `DS_BUILD_TRANSFORMER_INFERENCE` | `0/1` | `1` | Build transformer inference-specific kernels |
| `DS_BUILD_QUANTIZER` | `0/1` | `1` | Build quantization kernels |
| `DS_BUILD_SPARSE_ATTN` | `0/1` | `1` | Build sparse attention kernels |
| `DS_BUILD_STOCHASTIC_TRANSFORMER` | `0/1` | `1` | Build stochastic transformer kernels |
| `DS_BUILD_AIO` | `0/1` | `1` | Build async I/O ops (for NVMe offload) |
| `DS_BUILD_CROSS_ENTROPY` | `0/1` | `1` | Build cross-entropy loss kernel |
| `DS_BUILD_NORMS` | `0/1` | `1` | Build custom normalization kernels |
| `DS_BUILD_RAGGED_MODULE_OPS` | `0/1` | `1` | Build ragged tensor ops |
| `DS_BUILD_GDS` | `0/1` | `1` | Build GPUDirect Storage ops |
| `DS_BUILD_CCL` | `0/1` | `0` | Build Intel oneCCL ops |
| `DS_BUILD_MCCL` | `0/1` | `0` | Build Moore Threads MCCL ops |
| `DS_BUILD_MSCCL` | `0/1` | `0` | Build MSDI MSCCL ops |
| `DS_BUILD_HPU` | `0/1` | `0` | Build Habana HPU ops |
| `DS_BUILD_NPU` | `0/1` | `0` | Build Ascend NPU ops |
| `DS_BUILD_XPU` | `0/1` | `0` | Build Intel XPU ops |
| `DS_BUILD_MLU` | `0/1` | `0` | Build Cambricon MLU ops |
| `DS_BUILD_SDAA` | `0/1` | `0` | Build Moore Threads SDAA ops |
| `TORCH_CUDA_ARCH_LIST` | `str` | Auto | CUDA architectures to target (e.g., "8.0;8.6;9.0") |
| `DS_CUDA_VERSION` | `str` | Auto | Override CUDA version detection |
| `DS_HIP_VERSION` | `str` | Auto | Override HIP version detection |
| `MAX_BUILD_JOBS` | `int` | `nproc` | Maximum parallel compilation jobs |

### 2.3.3 CUDA Architecture Selection

DeepSpeed builds CUDA kernels targeting specific GPU architectures. The architecture list controls which GPU generations the kernels are optimized for:

```bash
# Target only A100 (SM 8.0)
TORCH_CUDA_ARCH_LIST="8.0" pip install deepspeed

# Target A100 and H100
TORCH_CUDA_ARCH_LIST="8.0;9.0" pip install deepspeed

# Target all common data center GPUs
TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0" pip install deepspeed

# Let DeepSpeed auto-detect (default)
pip install deepspeed
```

**CUDA Architecture Reference:**

| Architecture | SM Version | GPUs |
|-------------|-----------|------|
| Volta | 7.0 | V100 |
| Turing | 7.5 | T4, RTX 2080 |
| Ampere | 8.0 | A100 |
| Ampere | 8.6 | A10, A30, RTX 3090 |
| Ada Lovelace | 8.9 | L4, L40, RTX 4090 |
| Hopper | 9.0 | H100, H200 |
| Blackwell | 10.0 | B200, GB200 |

### 2.3.4 JIT Compilation

When pre-compiled ops are not available (e.g., when running from source without building all ops), DeepSpeed compiles ops on first use. JIT-compiled ops are cached in `~/.cache/torch_extensions/`.

```python
import deepspeed

# Ops will be JIT-compiled on first use
model_engine, _, _, _ = deepspeed.initialize(model=model, config=ds_config)

# Check which ops are available
from deepspeed.ops.op_builder import get_ops
for op_name, op_builder in get_ops().items():
    print(f"{op_name}: {op_builder.is_compatible()}")
```

**Force JIT recompilation:**

```bash
# Clear JIT cache
rm -rf ~/.cache/torch_extensions/

# Or set environment variable to force rebuild
DS_REBUILD_OPS=1 python train.py
```

### 2.3.5 Available Custom Operations

| Operation | Module | Description | Performance Impact |
|-----------|--------|-------------|-------------------|
| Fused Adam | `ops.adam.fused_adam` | Fused Adam optimizer kernel | 5-10% training speedup |
| Fused Lamb | `ops.adam_lamb.fused_lamb` | Fused Lamb optimizer kernel | 5-10% training speedup |
| Fused Lion | `ops.lion.fused_lion` | Fused Lion optimizer kernel | 5-10% training speedup |
| CPU Adam | `ops.adam.cpu_adam` | CPU-based Adam optimizer | Required for CPU offloading |
| CPU Adagrad | `ops.adagrad.cpu_adagrad` | CPU-based Adagrad optimizer | Required for CPU offloading |
| Transformer Inference | `ops.transformer.inference` | Optimized transformer kernels | 2-4x inference speedup |
| Sparse Attention | `ops.sparse_attention` | Block-sparse attention kernels | Enables long sequence training |
| Quantizer | `ops.quantizer` | INT8/INT4 quantization kernels | Enables inference quantization |
| Async I/O | `ops.aio` | NVMe async I/O operations | Required for NVMe offload |
| Cross Entropy | `ops.cross_entropy` | Fused cross-entropy loss | Minor training speedup |
| Norms | `ops.norms` | Custom normalization kernels | Minor training speedup |
| Ragged Ops | `ops.ragged` | Ragged tensor operations | For variable-length inputs |

---

## 2.4 ROCm (AMD GPU) Installation

### 2.4.1 ROCm Requirements

| Requirement | Version |
|-------------|---------|
| ROCm | 5.0+ (5.7+ recommended) |
| PyTorch for ROCm | Matching ROCm version |
| HIP compiler (hipcc) | Included with ROCm |

### 2.4.2 ROCm Installation Steps

```bash
# Install PyTorch with ROCm support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

# Install DeepSpeed
pip install deepspeed

# DeepSpeed auto-detects ROCm and uses hipcc for compilation
# Verify ROCm support
ds_report
```

### 2.4.3 ROCm-Specific Build Options

```bash
# Build with ROCm support from source
git clone https://github.com/microsoft/DeepSpeed.git
cd DeepSpeed
DS_BUILD_OPS=1 pip install -e .

# Override HIP version if auto-detection fails
DS_HIP_VERSION=6.0 DS_BUILD_OPS=1 pip install -e .

# Target specific AMD GPU architectures
HCC_AMDGPU_TARGET="gfx90a;gfx942" DS_BUILD_OPS=1 pip install -e .
```

**AMD GPU Architecture Reference:**

| Architecture | Target | GPUs |
|-------------|--------|------|
| CDNA2 | gfx90a | MI250X, MI210 |
| CDNA3 | gfx942 | MI300X, MI300A |
| RDNA3 | gfx1100 | RX 7900 XTX |

---

## 2.5 Other Accelerator Installation

### 2.5.1 Intel Habana HPU

```bash
# Install Habana framework
pip install habana-torch-plugin
pip install habana-torch-dataloader

# Install DeepSpeed with HPU support
DS_BUILD_HPU=1 pip install -e .

# Verify
ds_report
```

### 2.5.2 Huawei Ascend NPU

```bash
# Install torch_npu
pip install torch_npu

# Install DeepSpeed with NPU support
DS_BUILD_NPU=1 pip install -e .
```

### 2.5.3 Intel XPU

```bash
# Install Intel extension for PyTorch
pip install intel_extension_for_pytorch

# Install DeepSpeed with XPU support
DS_BUILD_XPU=1 pip install -e .
```

### 2.5.4 Cambricon MLU

```bash
# Install Cambricon PyTorch plugin
pip install torch_mlu

# Install DeepSpeed with MLU support
DS_BUILD_MLU=1 pip install -e .
```

### 2.5.5 Moore Threads SDAA

```bash
# Install Moore Threads PyTorch plugin
pip install torch_musa

# Install DeepSpeed with SDAA support
DS_BUILD_SDAA=1 pip install -e .
```

---

## 2.6 Windows Support

### 2.6.1 Windows Limitations

DeepSpeed on Windows has significant limitations:

- **No CUDA ops compilation**: Most custom CUDA ops cannot be built on Windows. DeepSpeed falls back to PyTorch-native implementations.
- **No NCCL**: Windows does not support NCCL. Use Gloo backend for single-node or MPI for multi-node.
- **Limited testing**: Windows support is community-maintained and less tested than Linux.

### 2.6.2 Windows Installation

```powershell
# Install without building custom ops
$env:DS_BUILD_OPS = "0"
pip install deepspeed

# Install with Windows-compatible ops only
$env:DS_BUILD_CPU_ADAM = "1"
pip install deepspeed

# For WSL2 (Windows Subsystem for Linux) - full support
# Install DeepSpeed normally inside WSL2
wsl
pip install deepspeed
```

### 2.6.3 WSL2 Setup (Recommended for Windows Users)

```powershell
# Install WSL2
wsl --install -d Ubuntu-22.04

# Inside WSL2
sudo apt update
sudo apt install python3-pip nvidia-cuda-toolkit

# Verify CUDA access from WSL2
nvidia-smi

# Install PyTorch and DeepSpeed
pip install torch torchvision torchaudio
pip install deepspeed
```

---

## 2.7 Docker Support

### 2.7.1 Official DeepSpeed Docker Images

```bash
# Pull the official image
docker pull deepspeed/deepspeed:latest

# Specific version
docker pull deepspeed/deepspeed:v0.15.0

# With specific CUDA version
docker pull deepspeed/deepspeed:v0.15.0-cuda12.1

# With PyTorch pre-installed
docker pull deepspeed/deepspeed:v0.15.0-torch2.1-cuda12.1
```

### 2.7.2 Building Custom Docker Images

**Dockerfile example:**

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Install DeepSpeed with all ops
ENV DS_BUILD_OPS=1
RUN pip install deepspeed

# Verify installation
RUN ds_report

WORKDIR /workspace
```

**Build and run:**

```bash
# Build the image
docker build -t my-deepspeed:latest .

# Run single-GPU training
docker run --gpus all -v $(pwd):/workspace my-deepspeed \
    python train.py --deepspeed ds_config.json

# Run multi-GPU training
docker run --gpus all --net=host -v $(pwd):/workspace my-deepspeed \
    deepspeed --num_gpus=4 train.py --deepspeed ds_config.json
```

### 2.7.3 Docker with NVMe Offload

For ZeRO-Infinity with NVMe offload, the container needs access to the host's NVMe devices:

```bash
docker run --gpus all \
    --device /dev/nvme0n1 \
    -v /dev/nvme0n1:/dev/nvme0n1 \
    -v $(pwd):/workspace \
    my-deepspeed \
    python train.py --deepspeed ds_config_nvme.json
```

### 2.7.4 Docker Compose for Multi-Node

```yaml
version: '3.8'
services:
  deepspeed-worker:
    image: my-deepspeed:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - MASTER_ADDR=master
      - MASTER_PORT=29500
      - WORLD_SIZE=8
    volumes:
      - ./workspace:/workspace
    network_mode: host
    command: deepspeed --num_gpus=4 train.py --deepspeed ds_config.json
```

---

## 2.8 Azure Support

### 2.8.1 Azure ML Integration

DeepSpeed integrates with Azure Machine Learning for cloud-based training:

```python
# Azure ML job definition (YAML)
$schema: https://azuremlsdk2.blob.core.windows.net/latest/azureJob.schema.json
type: command
command: >-
    deepspeed --num_gpus=$AZUREML_GPU_COUNT train.py
    --deepspeed ds_config.json
    --output_dir ${{outputs.model}}
environment: azureml:deepspeed-env:latest
resources:
  instance_type: Standard_ND96asr_v4  # 8x A100 80GB
  instance_count: 2
outputs:
  model:
    type: mlflow_model
```

### 2.8.2 Azure VM Types for DeepSpeed

| VM Type | GPUs | GPU Model | GPU Memory | Interconnect | Use Case |
|---------|------|-----------|-----------|-------------|----------|
| Standard_NC24ads_A100_v4 | 1 | A100 | 80 GB | PCIe | Development, testing |
| Standard_ND96asr_v4 | 8 | A100 | 80 GB each | NVLink + InfiniBand | Training |
| Standard_ND96amsr_A100_v4 | 8 | A100 | 80 GB each | NVLink + InfiniBand | Large model training |
| Standard_ND96isr_H100_v5 | 8 | H100 | 80 GB each | NVLink + InfiniBand | Maximum performance |
| Standard_NC24ads_A100_v4 | 1 | A100 | 80 GB | PCIe | Inference |
| Standard_ND40rs_v2 | 8 | V100 | 32 GB each | NVLink + InfiniBand | Legacy workloads |

### 2.8.3 Azure CycleCloud Setup

Azure CycleCloud provides cluster management for DeepSpeed training:

```bash
# Install CycleCloud CLI
pip install cyclecloud

# Create a DeepSpeed cluster
cyclecloud create_cluster deepspeed \
    --nodes 4 \
    --gpu-type A100 \
    --gpus-per-node 8
```

---

## 2.9 Post-Installation Verification

### 2.9.1 ds_report

The `ds_report` command provides a comprehensive diagnostic of the DeepSpeed installation:

```bash
ds_report
```

**Example output:**

```
--------------------------------------------------
DeepSpeed C++/CUDA extension op report
--------------------------------------------------
NOTE: Ops not installed will be just-in-time (JIT) compiled at
      runtime if needed. Op compatibility means that your installation
      the op is ready for use and will be JIT compiled if needed.

JIT compiled ops requires ninja that could be installed with
"pip install ninja". Installing ninja is highly recommended.

op name .................... installed ....... compatible
  async_io .................. [YES] ........... [OKAY]
  fused_adam ................ [YES] ........... [OKAY]
  fused_lamb ................ [YES] ........... [OKAY]
  fused_lion ................ [YES] ........... [OKAY]
  cpu_adam .................. [YES] ........... [OKAY]
  cpu_adagrad ............... [YES] ........... [OKAY]
  transformer ............... [YES] ........... [OKAY]
  transformer_inference ..... [YES] ........... [OKAY]
  stochastic_transformer .... [YES] ........... [OKAY]
  sparse_attention .......... [YES] ........... [OKAY]
  quantizer ................. [YES] ........... [OKAY]
  cross_entropy ............. [YES] ........... [OKAY]
  random_ltd ................ [YES] ........... [OKAY]
  ragged_module_ops ......... [YES] ........... [OKAY]
  norms ..................... [YES] ........... [OKAY]

--------------------------------------------------
DeepSpeed general environment info:
CUDA device count ........... 8
CUDA capability ............ 8.0
CUDA total memory ........... 80 GB
PyTorch version ............. 2.1.0+cu121
DeepSpeed version ........... 0.15.0
--------------------------------------------------
```

### 2.9.2 Quick Smoke Test

```python
# test_deepspeed.py
import deepspeed
import torch
import json

# Create a simple model
model = torch.nn.Linear(1024, 1024)

# DeepSpeed configuration
ds_config = {
    "train_batch_size": 16,
    "gradient_accumulation_steps": 1,
    "fp16": {"enabled": True},
    "zero_optimization": {"stage": 2},
}

# Initialize DeepSpeed
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config_params=ds_config,
)

# Test forward pass
x = torch.randn(4, 1024).cuda()
y = model_engine(x)
loss = y.sum()
model_engine.backward(loss)
model_engine.step()

print("DeepSpeed smoke test passed!")
print(f"ZeRO stage: {model_engine.zero_optimization_stage}")
print(f"FP16 enabled: {model_engine.fp16_enabled}")
```

```bash
# Run the smoke test
deepspeed --num_gpus=1 test_deepspeed.py
```

---

## 2.10 Environment Variables

### 2.10.1 Runtime Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DEEPSPEED_TIMEOUT` | `int` (seconds) | `1800` | Timeout for collective operations (30 minutes) |
| `DS_ACCELERATOR` | `str` | Auto-detect | Force specific accelerator (cuda, rocm, hpu, npu, xpu, mlu, sdaa, cpu) |
| `DS_BUILD_OPS` | `0/1` | `1` | Build custom ops during installation |
| `DS_REBUILD_OPS` | `0/1` | `0` | Force rebuild JIT ops at runtime |
| `DS_DUMP_CPU_OP_GRAPH` | `0/1` | `0` | Dump CPU op computation graph for debugging |
| `DS_KERNEL_INJECT_PATH` | `str` | None | Custom path to kernel injection libraries |
| `DS_DEBUG` | `0/1` | `0` | Enable debug logging |
| `DS_LOG_LEVEL` | `str` | `INFO` | Log level (DEBUG, INFO, WARNING, ERROR) |
| `DS_COMM_BLOCKING` | `0/1` | `0` | Use blocking communication (for debugging) |
| `DS_DISABLE_JIT` | `0/1` | `0` | Disable JIT compilation entirely |
| `DS_USE_SYS_LEVEL_LOG` | `0/1` | `0` | Use Python system-level logging |
| `DEEPSPEED_PROFILING_ENABLE` | `0/1` | `0` | Enable profiling by default |
| `DEEPSPEED_PROFILING_OUTPUT_DIR` | `str` | `./profiling` | Profiling output directory |
| `DEEPSPEED_CONFIG_FILE` | `str` | None | Default config file path |

### 2.10.2 Distributed Training Environment Variables

These variables are typically set by the DeepSpeed launcher, but can be set manually:

| Variable | Description |
|----------|-------------|
| `MASTER_ADDR` | Hostname of the master node |
| `MASTER_PORT` | Port for distributed rendezvous |
| `RANK` | Global rank of the process |
| `LOCAL_RANK` | Local rank on the current node |
| `WORLD_SIZE` | Total number of processes |
| `LOCAL_SIZE` | Number of processes on the current node |
| `NODE_RANK` | Rank of the current node |

### 2.10.3 CUDA-Related Environment Variables

| Variable | Description |
|----------|-------------|
| `CUDA_VISIBLE_DEVICES` | Restrict visible GPUs |
| `NCCL_DEBUG` | NCCL debug level (INFO, WARN) |
| `NCCL_DEBUG_SUBSYS` | NCCL subsystems to debug |
| `NCCL_SOCKET_IFNAME` | Network interface for NCCL |
| `NCCL_IB_DISABLE` | Disable InfiniBand (0/1) |
| `NCCL_P2P_DISABLE` | Disable P2P transfers (0/1) |
| `NCCL_SHM_DISABLE` | Disable shared memory (0/1) |
| `CUDA_LAUNCH_BLOCKING` | Synchronize CUDA operations (for debugging) |

### 2.10.4 Memory-Related Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DS_OFFLOAD_CPU_PIN_MEMORY` | `0/1` | `1` | Pin CPU memory for offloaded tensors |
| `DS_OFFLOAD_BUFFER_SIZE` | `int` | Auto | Buffer size for CPU offloading |
| `DS_NVME_WRITE_THRU` | `0/1` | `1` | Enable write-through caching for NVMe |
| `DS_AIO_BLOCK_SIZE` | `int` | `1048576` | AIO block size in bytes |
| `DS_AIO_QUEUE_DEPTH` | `int` | `8` | AIO queue depth |
| `DS_AIO_THREAD_COUNT` | `int` | `4` | Number of AIO threads |
| `DS_AIO_SINGLE_SUBMIT` | `0/1` | `0` | Submit AIO requests individually |
| `DS_AIO_OVERLAP_EVENTS` | `0/1` | `1` | Overlap AIO events |

---

## 2.11 Version Compatibility Matrix

### 2.11.1 DeepSpeed / PyTorch / CUDA Compatibility

| DeepSpeed | PyTorch | CUDA | Python | Status |
|-----------|---------|------|--------|--------|
| 0.17.x | 2.2 - 2.5 | 11.8 - 12.4 | 3.9 - 3.12 | Current |
| 0.16.x | 2.1 - 2.4 | 11.8 - 12.3 | 3.9 - 3.12 | Supported |
| 0.15.x | 2.0 - 2.3 | 11.7 - 12.2 | 3.9 - 3.11 | Maintained |
| 0.14.x | 1.13 - 2.2 | 11.6 - 12.1 | 3.8 - 3.11 | Maintenance |
| 0.13.x | 1.12 - 2.1 | 11.6 - 12.0 | 3.8 - 3.11 | Maintenance |
| 0.12.x | 1.12 - 2.0 | 11.3 - 11.8 | 3.8 - 3.10 | End of life |
| 0.11.x | 1.12 - 2.0 | 11.3 - 11.8 | 3.8 - 3.10 | End of life |
| 0.10.x | 1.9 - 1.13 | 11.1 - 11.7 | 3.8 - 3.10 | End of life |
| 0.9.x | 1.9 - 1.12 | 11.0 - 11.6 | 3.7 - 3.9 | End of life |
| 0.8.x | 1.8 - 1.12 | 11.0 - 11.5 | 3.7 - 3.9 | End of life |

### 2.11.2 HuggingFace Transformers Compatibility

| DeepSpeed | Transformers | Notes |
|-----------|-------------|-------|
| 0.17.x | 4.38+ | Full support for all features |
| 0.16.x | 4.35+ | Full support |
| 0.15.x | 4.30+ | Full support |
| 0.14.x | 4.25+ | Full support |
| 0.13.x | 4.20+ | Some newer models may not work |
| 0.12.x | 4.20+ | ZeRO-3 with newer models may have issues |

---

## 2.12 Troubleshooting Installation

### 2.12.1 Common Installation Issues

**Issue 1: `ninja` not found during build**

```bash
# Solution: Install ninja
pip install ninja
```

**Issue 2: CUDA version mismatch**

```bash
# Check CUDA versions
nvcc --version                    # CUDA toolkit version
python -c "import torch; print(torch.version.cuda)"  # PyTorch CUDA version

# They must match. Reinstall PyTorch with correct CUDA:
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**Issue 3: GCC version incompatibility**

```bash
# Check GCC version
gcc --version

# DeepSpeed requires GCC 7-12 for CUDA compilation
# If GCC 13+ is installed, install a compatible version:
sudo apt install gcc-11 g++-11
export CC=gcc-11
export CXX=g++-11
pip install deepspeed
```

**Issue 4: Out of disk space during JIT compilation**

```bash
# Move JIT cache to a larger partition
export TORCH_EXTENSIONS_DIR=/mnt/large_disk/torch_extensions
```

**Issue 5: Permission denied on NVMe devices**

```bash
# Give user access to NVMe devices
sudo chmod a+rw /dev/nvme0n1
```

**Issue 6: NCCL timeout during multi-node training**

```bash
# Increase timeout
export NCCL_TIMEOUT=3600000  # 1 hour in milliseconds
export DEEPSPEED_TIMEOUT=3600  # 1 hour in seconds
```

### 2.12.2 Diagnostic Commands

```bash
# Full environment report
ds_report

# Check CUDA ops compatibility
python -c "
from deepspeed.ops.op_builder import get_ops
for name, builder in get_ops().items():
    try:
        installed = builder.is_installed()
        compatible = builder.is_compatible()
        print(f'{name}: installed={installed}, compatible={compatible}')
    except Exception as e:
        print(f'{name}: ERROR - {e}')
"

# Check distributed setup
python -c "
import torch.distributed as dist
import deepspeed
print(f'DeepSpeed version: {deepspeed.__version__}')
print(f'NCCL available: {torch.cuda.nccl.is_available()}')
"

# Check GPU topology
nvidia-smi topo -m
```

---

## 2.13 Advanced Installation Topics

### 2.13.1 Installing for Development

```bash
# Clone and set up development environment
git clone https://github.com/microsoft/DeepSpeed.git
cd DeepSpeed

# Create virtual environment
python -m venv ds-dev
source ds-dev/bin/activate

# Install in editable mode with dev dependencies
pip install -e .[all]
pip install pre-commit pytest pytest-timeout

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/
```

### 2.13.2 Building with Debug Symbols

```bash
# Build with debug symbols for profiling
DS_BUILD_OPS=1 CMAKE_BUILD_TYPE=Debug pip install -e .
```

### 2.13.3 Cross-Compilation

For building DeepSpeed on a machine different from the target:

```bash
# Build for a different CUDA architecture
TORCH_CUDA_ARCH_LIST="9.0" DS_BUILD_OPS=1 pip install -e .

# Transfer the installed package to the target machine
pip install deepspeed --target ./ds_package
tar czf deepspeed_package.tar.gz ./ds_package
# On target machine:
tar xzf deepspeed_package.tar.gz
pip install ./ds_package/deepspeed-*.whl
```

### 2.13.4 Installing in Air-Gapped Environments

```bash
# Download all dependencies on an internet-connected machine
pip download deepspeed -d ./ds_packages
pip download torch -d ./ds_packages

# Transfer to air-gapped environment and install
pip install --no-index --find-links ./ds_packages deepspeed
```

---

## 2.14 Quick Start Checklist

Use this checklist to verify your installation is ready for training:

```bash
# 1. Verify Python version (3.8+)
python --version

# 2. Verify PyTorch installation
python -c "import torch; print(f'PyTorch {torch.__version__}')"

# 3. Verify CUDA is available
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"

# 4. Verify DeepSpeed installation
python -c "import deepspeed; print(f'DeepSpeed {deepspeed.__version__}')"

# 5. Run full diagnostic
ds_report

# 6. Run smoke test
deepspeed --num_gpus=1 -m deepspeed.runtime.benchmark --model_name gpt2 --batch_size 8
```
