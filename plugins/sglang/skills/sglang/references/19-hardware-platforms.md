# SGLang Hardware Platforms Reference

This document provides a comprehensive reference for running SGLang on all supported hardware platforms. Each platform section covers installation, configuration, supported models, optimization tips, and platform-specific features.

---

## Table of Contents

1. [Platform Overview](#platform-overview)
2. [NVIDIA GPUs](#nvidia-gpus)
3. [AMD GPUs](#amd-gpus)
4. [Intel Xeon CPUs](#intel-xeon-cpus)
5. [Google TPUs](#google-tpus)
6. [Huawei Ascend NPUs](#huawei-ascend-npus)
7. [Intel XPU (GPU)](#intel-xpu-gpu)
8. [NVIDIA Jetson Orin](#nvidia-jetson-orin)
9. [Moore Threads GPUs (MUSA)](#moore-threads-gpus-musa)
10. [Platform Plugin System](#platform-plugin-system)

---

## Platform Overview

SGLang supports a wide range of hardware accelerators for LLM inference:

| Platform | Backend | Status | Best For |
|----------|---------|--------|----------|
| NVIDIA GPUs | CUDA | Production | General inference, highest performance |
| AMD GPUs | ROCm | Production | MI300X, MI355 inference |
| Intel Xeon CPUs | CPU | Production | CPU-only servers with AMX |
| Google TPUs | JAX (sglang-jax) | Production | Google Cloud TPU inference |
| Ascend NPUs | CANN | Production | Atlas 800I A2/A3 inference |
| Intel XPU | XPU | Beta | Intel Arc GPU acceleration |
| NVIDIA Jetson Orin | CUDA | Beta | Edge inference |
| Moore Threads | MUSA | Beta | MTT S4000+ GPUs |

---

## NVIDIA GPUs

### Overview

NVIDIA GPUs are the primary and most mature platform for SGLang. All major features are supported and optimized for NVIDIA hardware.

### Supported GPUs

| GPU Series | Key Models | Memory | Notes |
|------------|-----------|--------|-------|
| Hopper | H100, H200 | 80-141 GB | Best performance, FP8 support |
| Blackwell | B200, GB200 | 192 GB | Next-gen, highest memory |
| Ada Lovelace | L40S, RTX 4090 | 24-96 GB | Good for development |
| Ampere | A100, A6000 | 40-80 GB | Widely available |
| H20 | H20 | 96 GB | China-specific, good for large models |

### CUDA Versions

SGLang Docker images support multiple CUDA versions:

| CUDA Version | Compute Capability | Architectures |
|-------------|-------------------|---------------|
| 12.6.1 | sm_90 | Hopper |
| 12.8.1 | sm_90, sm_100 | Hopper, Blackwell |
| 12.9.1 | sm_90, sm_100, sm_103 | Hopper, Blackwell |
| 13.0.1 | sm_90, sm_100, sm_103 | Hopper, Blackwell |

### Installation

```bash
# Install from source
pip install --upgrade pip
pip install "sglang[all]"

# Or install from source
git clone https://github.com/sgl-project/sglang.git
cd sglang
pip install -e "python[all]"
```

### Docker

```bash
docker pull lmsysorg/sglang:latest

docker run --gpus all -it --rm --network=host \
    --ipc=host \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct
```

### Performance Tips

- Use `--mem-fraction-static 0.88-0.93` to maximize GPU memory utilization
- Enable CUDA graphs (default) for optimal decode performance
- Use FP8 quantization on Hopper+ GPUs for 2x memory savings
- Use `--enable-torch-compile` with `--torch-compile-max-bs` for CPU-bound workloads
- Set `SGLANG_ENABLE_JIT_DEEPGEMM=1` for DeepSeek models

---

## AMD GPUs

### Overview

SGLang runs on AMD GPUs using the ROCm backend. The MI300X is the primary target platform.

### Supported GPUs

| GPU | Architecture | Memory | HBM | Notes |
|-----|-------------|--------|-----|-------|
| MI300X | CDNA3 | 192 GB HBM3 | 5.3 TB/s | Primary target |
| MI355 | CDNA4 | 288 GB HBM3e | 8 TB/s | Next-gen |
| MI250X | CDNA2 | 128 GB HBM2e | 3.2 TB/s | Supported |

### System Configuration

**Update GRUB** in `/etc/default/grub`:

```
GRUB_CMDLINE_LINUX="... pci=realloc=off iommu=pt"
```

Then run `sudo update-grub` and reboot.

**Disable NUMA auto-balancing:**

```bash
sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'
```

**Reference guides:**
- [AMD MI300X Tuning Guides](https://rocm.docs.amd.com/en/latest/how-to/tuning-guides/mi300x/index.html)
- [AMD Instinct MI300X System Optimization](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/mi300x.html)

### Installation

**From source:**

```bash
git clone -b v0.5.9 https://github.com/sgl-project/sglang.git
cd sglang

# Compile sgl-kernel
cd sgl-kernel
python setup_rocm.py install
cd ..

# Install sglang
rm -rf python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml
pip install -e "python[all_hip]"
```

**Using Docker (recommended):**

```bash
docker build -t sglang_image -f rocm.Dockerfile .

alias drun='docker run -it --rm --network=host --privileged \
    --device=/dev/kfd --device=/dev/dri \
    --ipc=host --shm-size 16G --group-add video \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined'

drun -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=<secret>" \
    sglang_image \
    python3 -m sglang.launch_server \
    --model-path NousResearch/Meta-Llama-3.1-8B \
    --host 0.0.0.0 --port 30000
```

### Aiter Acceleration

[Aiter](https://github.com/ROCm/aiter) provides optimized kernels for AMD:

```bash
export SGLANG_USE_AITER=1
```

### Quantization on AMD

| Method | Notes |
|--------|-------|
| FP8 | Works via Aiter or Triton. Pre-quantized models work out of the box |
| AWQ | Uses Triton dequantization (no Marlin path) |
| MXFP4 | Requires CDNA3/CDNA4 and `SGLANG_USE_AITER=1` |
| petit_nvfp4 | Enables NVFP4 models on MI250/MI300X via Petit |
| GPTQ | Supported |
| W8A8 | Supported |
| quark_int4fp8_moe | AMD-only online quantization for MoE models |

Methods that depend on Marlin/NVIDIA-specific kernels (`awq_marlin`, `gptq_marlin`, `gguf`, `modelopt_fp8`, `modelopt_fp4`) are NOT supported on AMD.

### Examples

**DeepSeek-V3:**

```bash
drun -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host --env "HF_TOKEN=<secret>" \
    sglang_image \
    python3 -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --tp 8 --trust-remote-code \
    --host 0.0.0.0 --port 30000
```

---

## Intel Xeon CPUs

### Overview

SGLang is optimized for Intel Xeon CPUs equipped with Intel AMX instructions (4th generation or newer Intel Xeon Scalable Processors).

### Supported Models

| Model | BF16 | W8A8_INT8 | FP8 |
|-------|------|-----------|-----|
| DeepSeek-R1 | - | meituan/DeepSeek-R1-Channel-INT8 | deepseek-ai/DeepSeek-R1 |
| DeepSeek-V3.1-Terminus | - | IntervitensInc/DeepSeek-V3.1-Terminus-Channel-int8 | deepseek-ai/DeepSeek-V3.1-Terminus |
| Llama-3.2-3B | meta-llama/Llama-3.2-3B-Instruct | RedHatAI/Llama-3.2-3B-quantized.w8a8 | - |
| Llama-3.1-8B | meta-llama/Llama-3.1-8B-Instruct | RedHatAI/Meta-Llama-3.1-8B-quantized.w8a8 | - |
| QwQ-32B | - | RedHatAI/QwQ-32B-quantized.w8a8 | - |
| Qwen3-235B | - | - | Qwen/Qwen3-235B-A22B-FP8 |

### Installation

**Docker (recommended):**

```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang/docker
docker build -t sglang-cpu:latest -f xeon.Dockerfile .

docker run -it --privileged --ipc=host --network=host \
    -v /dev/shm:/dev/shm \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 30000:30000 \
    -e "HF_TOKEN=<secret>" \
    sglang-cpu:latest /bin/bash
```

**From source:**

```bash
# Create venv
cd /opt
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12
source .venv/bin/activate

# Configure torch CPU channels
export UV_CONFIG_FILE=/opt/.venv/uv.toml

# Install
git clone https://github.com/sgl-project/sglang.git
cd sglang/python
cp pyproject_cpu.toml pyproject.toml
uv pip install .

cd ../sgl-kernel
cp pyproject_cpu.toml pyproject.toml
uv pip install .
```

**Required environment variables:**

```bash
export SGLANG_USE_CPU_ENGINE=1
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
export LD_PRELOAD=${LD_PRELOAD}:/opt/.venv/lib/libiomp5.so:${LD_LIBRARY_PATH}/libtcmalloc.so.4:${LD_LIBRARY_PATH}/libtbbmalloc.so.2
```

### Running

```bash
sglang serve \
    --model-path <MODEL_ID_OR_PATH> \
    --trust-remote-code \
    --disable-overlap-schedule \
    --device cpu \
    --host 0.0.0.0 \
    --tp 6
```

**For W8A8 quantized models:**

```bash
sglang serve \
    --model-path RedHatAI/Meta-Llama-3.1-8B-quantized.w8a8 \
    --trust-remote-code \
    --disable-overlap-schedule \
    --device cpu \
    --quantization w8a8_int8 \
    --enable-torch-compile \
    --torch-compile-max-bs 4 \
    --host 0.0.0.0 \
    --tp 6
```

### Tensor Parallelism on CPU

On CPU, each TP rank corresponds to a sub-NUMA cluster (SNC). Use `lscpu` to check available SNCs.

Custom core binding with `SGLANG_CPU_OMP_THREADS_BIND`:

```bash
# TP=6, using first 40 cores of each SNC on Xeon 6980P
export SGLANG_CPU_OMP_THREADS_BIND="0-39|43-82|86-125|128-167|171-210|214-253"
```

---

## Google TPUs

### Overview

SGLang TPU support is implemented via the SGLang-JAX backend, a dedicated JAX-based inference engine at [https://github.com/sgl-project/sglang-jax](https://github.com/sgl-project/sglang-jax).

### Supported TPUs

| TPU Type | HBM Memory | Availability |
|----------|-----------|--------------|
| TPU v6e | 32 GB | Google Cloud |
| TPU v7 | 96 GB per core | Google Cloud |

### Feature Support

| Feature | Status |
|---------|--------|
| Continuous Batching | Supported |
| Radix Tree KV Cache | Supported |
| FlashAttention Backend | Supported |
| Tensor Parallelism | Supported |
| Speculative Decoding (EAGLE/EAGLE3) | Supported (20-40% improvement) |
| Chunked Prefill | Supported |
| OpenAI-Compatible API | Supported |
| Data Parallel Attention | In Development |
| Quantization | In Development |
| Multi-LoRA | In Development |

### Optimized Models

- Qwen 3 and Qwen 3 MoE: Recommended for production
- Gemma 2: Verified on TPU
- Qwen 2, Llama, Grok-2: Available but needs improvement

### Installation

```bash
pip install sglang-jax

# Or from source
git clone https://github.com/sgl-project/sglang-jax
cd sglang-jax
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e "python[all]"
```

### Running

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python3 -u -m sgl_jax.launch_server \
    --model-path Qwen/Qwen-7B-Chat \
    --trust-remote-code \
    --dist-init-addr=0.0.0.0:10011 \
    --nnodes=1 --tp-size=4 --device=tpu \
    --mem-fraction-static=0.8 \
    --max-prefill-tokens=8192 \
    --dtype=bfloat16 \
    --host 0.0.0.0 --port 30000
```

**With speculative decoding (EAGLE3):**

```bash
python3 -u -m sgl_jax.launch_server \
    --model-path Qwen/Qwen3-32B \
    --tp-size=4 --device=tpu \
    --attention-backend=fa \
    --speculative-algorithm=EAGLE3 \
    --speculative-draft-model-path=AngelSlim/Qwen3-32B_eagle3 \
    --speculative-eagle-topk=1 \
    --speculative-num-steps=3 \
    --speculative-num-draft-tokens=4
```

### Optimization Tips

- Always set `JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache` for faster startup
- Use `--dtype=bfloat16` (TPU native)
- Use `--attention-backend=fa` for production
- Match `--tp-size` to TPU core count (1, 4, or 8)
- Use `--skip-server-warmup` to defer compilation to first request

---

## Huawei Ascend NPUs

### Overview

SGLang runs on Huawei Ascend NPUs using the CANN backend. Supports Atlas 800I A2 and A3 inference servers.

### Supported Devices

- **Atlas 800I A2**: Ascend 910B series
- **Atlas 800I A3**: Next-gen Ascend accelerators

### Component Version Mapping

| Component | Version | Source |
|-----------|---------|--------|
| HDK | 25.5.2 | [Link](https://www.hiascend.com/hardware/firmware-drivers/commercial) |
| CANN | 8.5.0 | Docker images |
| PyTorch Adapter | 7.3.0 | [Link](https://gitcode.com/Ascend/pytorch/releases) |
| MemFabric | 1.0.5 | pip install |
| Triton | 3.2.0 | pip install triton-ascend |
| SGLang NPU Kernel | Latest | [Link](https://github.com/sgl-project/sgl-kernel-npu/releases) |

### Installation

**From source:**

```bash
# Python 3.11 required
conda create --name sglang_npu python=3.11
conda activate sglang_npu

# Install CANN 8.5.0 Toolkit, Kernels, and NNAL

# Install dependencies
pip install memfabric-hybrid==1.0.5
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
pip install torch_npu==2.8.0.post2
pip install triton-ascend
pip install "setuptools<80"

# Install SGLang
git clone https://github.com/sgl-project/sglang.git
cd sglang
mv python/pyproject_npu.toml python/pyproject.toml
pip install -e python[all_npu]
```

**Using Docker:**

```bash
# Atlas 800I A3
docker pull quay.io/ascend/cann:8.5.0-a3-ubuntu22.04-py3.11

# Atlas 800I A2
docker pull quay.io/ascend/cann:8.5.0-910b-ubuntu22.04-py3.11
```

### System Settings

```bash
# CPU performance mode
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Disable NUMA balancing
sudo sysctl -w kernel.numa_balancing=0

# Reduce swap
sudo sysctl -w vm.swappiness=10
```

### Running

**PD Mixed (single server):**

```bash
export SGLANG_SET_CPU_AFFINITY=1
python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --attention-backend ascend
```

**PD Disaggregation:**

```bash
# Prefill server
export ASCEND_MF_STORE_URL="tcp://PREFILL_IP:PORT"
python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend ascend \
    --attention-backend ascend \
    --device npu --tp-size 1 \
    --host 127.0.0.1 --port 8000

# Decode server
export ASCEND_MF_STORE_URL="tcp://PREFILL_IP:PORT"
python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend ascend \
    --attention-backend ascend \
    --device npu --tp-size 1 \
    --host 127.0.0.1 --port 8001

# Router
python3 -m sglang_router.launch_router \
    --pd-disaggregation --policy cache_aware \
    --prefill http://127.0.0.1:8000 \
    --decode http://127.0.0.1:8001 \
    --host 127.0.0.1 --port 6688
```

### Supported Models (LLMs)

DeepSeek V3/V3.1, DeepSeek-R1, Qwen3 series (0.6B to 235B), Qwen2.5, Llama 3/4, Mistral, Gemma 3, GLM-4, InternLM 2, Phi-4, Grok-2, Kimi-K2, and many more.

### Supported Models (Multimodal)

Qwen2.5-VL, Qwen3-VL, DeepSeek-VL2, Janus-Pro, MiniCPM-V, Llama 3.2 Vision, GLM-4.5V, Kimi-VL, and more.

### Environment Variables

**SGLang NPU variables:**

| Variable | Description | Default |
|----------|-------------|---------|
| `SGLANG_NPU_USE_MLAPO` | Use MLAPO fusion operator for MLA attention | false |
| `SGLANG_USE_FIA_NZ` | Reshape KV Cache for FIA NZ format | false |
| `SGLANG_NPU_USE_MULTI_STREAM` | Enable dual-stream for shared/routing experts | false |
| `SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT` | Disable ACL format weight casting | false |
| `SGLANG_NPU_FORWARD_NATIVE_GELUTANH` | Native GELU tanh for specific models | false |
| `SGLANG_NPU_FORWARD_NATIVE_GEMMA_RMS_NORM` | Native RMS norm for specific models | false |
| `SGLANG_NPU_FUSED_MOE_MODE` | Fused MoE mode | 1 |

**DeepEP Ascend variables:**

| Variable | Description | Default |
|----------|-------------|---------|
| `DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS` | Tokens per round in dispatch stage | 8192 |
| `DEEPEP_NORMAL_LONG_SEQ_ROUND` | Rounds per dispatch | 1 |
| `DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ` | Enable long seq in combine stage | 0 |
| `MOE_ENABLE_TOPK_NEG_ONE` | Handle expert ID -1 in DEEPEP | 0 |
| `DEEP_NORMAL_MODE_USE_INT8_QUANT` | Quantize x to int8 in dispatch | 0 |

**System-level variables:**

| Variable | Description | Default |
|----------|-------------|---------|
| `TASK_QUEUE_ENABLE` | Task queue optimization level | 1 |
| `INF_NAN_MODE_ENABLE` | INF_NAN mode vs saturation mode | 1 |
| `STREAMS_PER_DEVICE` | Max streams in pool | 32 |
| `ASCEND_MF_STORE_URL` | MemFabric config store address | - |
| `ASCEND_LAUNCH_BLOCKING` | Synchronous operator execution | 0 |
| `HCCL_BUFFSIZE` | HCCL buffer size (MB) | 200 |
| `HCCL_SOCKET_IFNAME` | Network card for HCCL | - |
| `GLOO_SOCKET_IFNAME` | Network interface for GLOO | - |

### Quantization on Ascend

Supported quantization schemes include:

- **ModelSlim**: W4A4, W8A8 (static/dynamic) for Linear and MoE layers
- **AWQ**: W4A16, W8A16 for Linear and MoE layers
- **GPTQ**: W4A16, W8A16 for Linear and MoE layers
- **Auto-round**: W4A16, W8A16 for Linear and MoE layers
- **Compressed-tensors**: W8A8, W4A8, W4A16 for MoE
- **GGUF**: All types (standard, K-quant) for Linear and MoE
- **MXFP8**: Work in progress for A5

---

## Intel XPU (GPU)

### Overview

SGLang supports Intel Arc Pro B-Series and Arc B-Series GPUs via the XPU backend.

### Supported Models

| Model | BF16 |
|-------|------|
| Llama-3.2-3B | meta-llama/Llama-3.2-3B-Instruct |
| Llama-3.1-8B | meta-llama/Llama-3.1-8B-Instruct |
| Qwen2.5-1.5B | Qwen/Qwen2.5-1.5B |

Verified on Intel Arc B580 Graphics.

### Installation

```bash
conda create -n sgl-xpu python=3.12 -y
conda activate sgl-xpu

pip3 install torch==2.11.0+xpu torchao torchvision torchaudio==2.11.0+xpu \
    --index-url https://download.pytorch.org/whl/xpu
pip3 install xgrammar --no-deps

git clone https://github.com/sgl-project/sglang.git
cd sglang/python
cp pyproject_xpu.toml pyproject.toml
pip install -v . --extra-index-url https://download.pytorch.org/whl/xpu
```

### Running

```bash
sglang serve \
    --model-path <MODEL_ID_OR_PATH> \
    --trust-remote-code \
    --disable-overlap-schedule \
    --device xpu \
    --host 0.0.0.0 \
    --tp 2 \
    --attention-backend intel_xpu \
    --page-size 64
```

The `intel_xpu` attention backend supports page sizes of 32, 64, or 128.

---

## NVIDIA Jetson Orin

### Overview

SGLang runs on NVIDIA Jetson AGX Orin devices for edge inference.

### Prerequisites

- NVIDIA Jetson AGX Orin Devkit with JetPack 6.1+
- CUDA Toolkit and cuDNN installed
- High-performance mode: `sudo nvpmodel -m 0`

### Installation

```bash
git clone https://github.com/dusty-nv/jetson-containers.git
bash jetson-containers/install.sh
jetson-containers build sglang
jetson-containers run $(autotag sglang)
```

### Running

```bash
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --device cuda \
  --dtype half \
  --attention-backend flashinfer \
  --mem-fraction-static 0.8 \
  --context-length 8192
```

### Quantization with TorchAO

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --device cuda \
    --dtype bfloat16 \
    --attention-backend flashinfer \
    --mem-fraction-static 0.8 \
    --context-length 8192 \
    --torchao-config int4wo-128
```

---

## Moore Threads GPUs (MUSA)

### Overview

SGLang supports Moore Threads GPUs using the MUSA backend.

### Installation

```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang

# Compile sgl-kernel
cd sgl-kernel
python setup_musa.py install
cd ..

# Install sglang
rm -f python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml
pip install -e "python[all_musa]"
```

---

## Platform Plugin System

### Overview

The SGLang plugin system allows hardware vendors and developers to extend SGLang without modifying the main repository code. Plugins are automatically discovered via Python's `setuptools` entry_points.

### Plugin Types

| Type | Entry Point Group | Purpose |
|------|-------------------|---------|
| Hardware Platform Plugin | `sglang.srt.platforms` | Register custom hardware platform (device operations, KV cache pools, attention backends, etc.) |
| General Plugin | `sglang.srt.plugins` | Inject hooks into any function/method or replace entire classes |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `SGLANG_PLATFORM` | Select platform plugin by entry_point name. Required when multiple plugins would activate. |
| `SGLANG_PLUGINS` | Comma-separated whitelist of general plugin names to load. If unset, all discovered plugins are loaded. |

### Platform Hierarchy

```
DeviceMixin (shared device identity + operations)
  +-- SRTPlatform(DeviceMixin)           # + graph runner, KV pool, ...
  |   +-- MySRTPlatform(SRTPlatform, MyDeviceMixin)   # OOT plugin
  +-- MMPlatform(DeviceMixin)            # + attention backend, VAE, ... (future)
      +-- MyMMPlatform(MMPlatform, MyDeviceMixin)      # OOT plugin
```

### Platform Interface Methods

**Identity Queries:**
- `is_cuda()`, `is_rocm()`, `is_npu()`, `is_cpu()`, `is_xpu()`, `is_musa()`, `is_cuda_alike()`, `is_out_of_tree()`

**Device Operations:**
- `set_device()`, `get_device_name()`, `get_device_total_memory()`, `get_current_memory_usage()`, `get_device_capability()`, `empty_cache()`, `synchronize()`, `get_torch_distributed_backend_str()`

**Capability Flags:**
- `support_cuda_graph()`, `support_piecewise_cuda_graph()`, `supports_fp8()`, `is_pin_memory_available()`

**Subsystem Factories:**
- `get_default_attention_backend()`, `get_graph_runner_cls()`, `get_mha_kv_pool_cls()`, `get_mla_kv_pool_cls()`, `get_paged_allocator_cls()`, `get_compile_backend()`

### Creating a Hardware Platform Plugin

**1. Package structure:**

```
my_platform_plugin/
  +-- pyproject.toml
  +-- my_platform_plugin/
      +-- __init__.py    # activate() function
      +-- device.py      # MyDeviceMixin
      +-- platform.py    # MySRTPlatform
```

**2. pyproject.toml:**

```toml
[project]
name = "my-platform-plugin"
version = "0.1.0"

[project.entry-points."sglang.srt.platforms"]
my_device = "my_platform_plugin:activate"
```

**3. __init__.py:**

```python
def activate():
    """Return fully-qualified class name to activate, or None to skip."""
    if _my_device_is_available():
        return "my_platform_plugin.platform.MySRTPlatform"
    return None
```

**4. device.py:**

```python
from sglang.srt.platforms.device_mixin import DeviceMixin, PlatformEnum

class MyDeviceMixin(DeviceMixin):
    _enum = PlatformEnum.OOT
    device_name = "my_device"
    device_type = "my_device"  # torch device type

    def set_device(self, device) -> None: ...
    def get_device_name(self, device_id=0) -> str: ...
    def get_device_total_memory(self, device_id=0) -> int: ...
    def get_current_memory_usage(self, device=None) -> float: ...
    def get_device_capability(self, device_id=0): ...
    def get_torch_distributed_backend_str(self) -> str: ...
```

**5. platform.py:**

```python
from sglang.srt.platforms.interface import SRTPlatform
from my_platform_plugin.device import MyDeviceMixin

class MySRTPlatform(SRTPlatform, MyDeviceMixin):
    def get_default_attention_backend(self) -> str: ...
    def support_cuda_graph(self) -> bool: ...
```

### General Plugin Hook Types

| Hook Type | Signature | Description |
|-----------|-----------|-------------|
| BEFORE | `fn(*args, **kwargs) -> (args, kwargs) or None` | Runs before original. Return modified args or None. |
| AFTER | `fn(result, *args, **kwargs) -> new_result or None` | Runs after original. Return modified result or None. |
| AROUND | `fn(original_fn, *args, **kwargs) -> result` | Wraps original. Must call `original_fn` yourself. |
| REPLACE | `fn(*args, **kwargs) -> result` or class | Replace the original function or class entirely. |

### Common Hook Targets

| Target | Description |
|--------|-------------|
| `sglang.srt.server_args.ServerArgs.add_cli_args` | Add custom CLI arguments |
| `sglang.srt.server_args.ServerArgs.__post_init__` | Modify ServerArgs after parsing |
| `sglang.srt.managers.scheduler.Scheduler.__init__` | Custom scheduler state |
| `sglang.srt.managers.scheduler.Scheduler.get_next_batch_to_run` | Custom scheduling policy |
| `sglang.srt.managers.scheduler.Scheduler.run_batch` | Profiling/inspection |
| `sglang.srt.managers.tp_worker.TpModelWorker.forward_batch_generation` | Forward pass wrapping |

### Key Files

| File | Description |
|------|-------------|
| `sglang/srt/platforms/device_mixin.py` | PlatformEnum + DeviceMixin base class |
| `sglang/srt/platforms/interface.py` | SRTPlatform base class |
| `sglang/srt/platforms/__init__.py` | current_platform lazy singleton + discovery logic |
| `sglang/srt/plugins/__init__.py` | load_plugins() + load_plugins_by_group() |
| `sglang/srt/plugins/hook_registry.py` | HookRegistry, HookType, plugin_hook decorator |
