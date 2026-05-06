# vLLM Installation and Setup

## Table of Contents

- [Hardware Requirements](#hardware-requirements)
- [Installation Methods](#installation-methods)
- [Dependencies and Version Requirements](#dependencies-and-version-requirements)
- [Docker Images and Build Instructions](#docker-images-and-build-instructions)
- [Environment Variables and Configuration](#environment-variables-and-configuration)
- [Platform-Specific Setup](#platform-specific-setup)
- [Verification Steps](#verification-steps)
- [Troubleshooting](#troubleshooting)

---

## Hardware Requirements

### NVIDIA GPU (CUDA)

| Requirement | Details |
|-------------|---------|
| **GPU** | NVIDIA GPU with compute capability >= 7.0 (V100, A100, H100, L40S, etc.) |
| **CUDA** | CUDA 12.x (primary support for CUDA 13.0) |
| **Driver** | NVIDIA driver supporting CUDA 12.x+ |
| **VRAM** | Minimum 16 GB recommended (varies by model size) |
| **CPU** | Modern multi-core CPU |
| **RAM** | At least 2x GPU VRAM |

### AMD GPU (ROCm)

| Requirement | Details |
|-------------|---------|
| **GPU** | AMD Instinct MI300X, MI250, MI210, or compatible |
| **ROCm** | ROCm 6.x+ |
| **VRAM** | Minimum 32 GB recommended |
| **Linux** | Required (ROCm not available on Windows/macOS) |

### Google TPU

| Requirement | Details |
|-------------|---------|
| **TPU** | TPU v4, v5p, v5e, v6e |
| **Runtime** | libtpu, JAX/XLA runtime |
| **Python** | 3.10 - 3.14 |
| **TPU Inference** | tpu-inference >= 0.18.0 |

### Intel XPU

| Requirement | Details |
|-------------|---------|
| **Device** | Intel Gaudi (Habana), Intel Arc, or Intel Data Center GPU |
| **oneAPI** | Intel oneAPI toolkit |
| **Level Zero** | Level Zero loader and driver |
| **vLLM XPU Kernels** | vllm_xpu_kernels >= 0.1.7 |

### CPU

| Requirement | Details |
|-------------|---------|
| **Architecture** | x86_64, ARM64/aarch64, PowerPC (ppc64le), s390x, RISC-V |
| **SIMD** | AVX2 or AVX-512 recommended (x86_64) |
| **Cores** | 16+ cores recommended |
| **RAM** | 32 GB+ recommended |
| **tcmalloc** | Recommended for best performance (`libtcmalloc-minimal4`) |
| **OS** | Linux (recommended), macOS (supported) |

---

## Installation Methods

### Method 1: pip install (Recommended)

```bash
# Basic install
pip install vllm

# Or using uv (recommended by vLLM)
uv pip install vllm
```

This installs pre-built wheels for CUDA (x86_64, manylinux).

### Method 2: Install with extras

```bash
# Audio processing support
pip install vllm[audio]

# Benchmark tools
pip install vllm[bench]

# Tensorizer model loading
pip install vllm[tensorizer]

# Fast safetensors loading
pip install vllm[fastsafetensors]

# RunAI model streaming
pip install vllm[runai]

# gRPC server support
pip install vllm[grpc]

# OpenTelemetry tracing
pip install vllm[otel]

# AMD Zen CPU optimizations
pip install vllm[zen]

# Helion kernel development
pip install vllm[helion]

# Multiple extras
pip install vllm[audio,tensorizer,otel]
```

### Method 3: Install from Source (Development)

```bash
# Clone the repository
git clone https://github.com/vllm-project/vllm.git
cd vllm

# Create virtual environment
uv venv --python 3.12
source .venv/bin/activate

# Install for development (Python changes only, uses precompiled C++ extensions)
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto

# Full build from source (includes C/C++ compilation)
uv pip install -e . --torch-backend=auto
```

#### Build from Source with Custom Options

```bash
# Set target device
VLLM_TARGET_DEVICE=cuda pip install -e .

# For CPU
VLLM_TARGET_DEVICE=cpu pip install -e .

# For ROCm
VLLM_TARGET_DEVICE=rocm pip install -e .

# For XPU
VLLM_TARGET_DEVICE=xpu pip install -e .

# Control parallelism
MAX_JOBS=8 NVCC_THREADS=2 pip install -e .

# Build type
CMAKE_BUILD_TYPE=Release pip install -e .

# Verbose build
VERBOSE=1 pip install -e .
```

### Method 4: Precompiled Wheel with Nightly Build

```bash
# Install using precompiled nightly wheel
VLLM_USE_PRECOMPILED=1 pip install -e .

# Specify wheel commit
VLLM_PRECOMPILED_WHEEL_COMMIT=<40-char-sha> VLLM_USE_PRECOMPILED=1 pip install -e .

# Specify wheel variant
VLLM_PRECOMPILED_WHEEL_VARIANT=cu129 VLLM_USE_PRECOMPILED=1 pip install -e .

# Use local wheel file
VLLM_PRECOMPILED_WHEEL_LOCATION=/path/to/vllm.whl VLLM_USE_PRECOMPILED=1 pip install -e .
```

### Method 5: Conda

```bash
# From conda-forge (may lag behind pip)
conda install -c conda-forge vllm
```

---

## Dependencies and Version Requirements

### Common Dependencies (All Platforms)

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.11.0 (platform-specific) | Deep learning framework |
| `transformers` | >= 4.56.0 | HuggingFace model loading |
| `tokenizers` | >= 0.21.1 | Fast tokenization |
| `numpy` | any | Array operations |
| `pydantic` | >= 2.12.0 | Data validation |
| `fastapi[standard]` | >= 0.115.0 | API server |
| `aiohttp` | >= 3.13.3 | Async HTTP |
| `openai` | >= 2.0.0 | OpenAI compatibility |
| `pyzmq` | >= 25.0.0 | ZMQ IPC |
| `msgspec` | any | Serialization |
| `prometheus_client` | >= 0.18.0 | Metrics |
| `protobuf` | >= 5.29.6 | Protocol buffers |
| `xgrammar` | >= 0.2.0, < 1.0.0 | Structured outputs |
| `outlines_core` | == 0.2.14 | Structured outputs |
| `lm-format-enforcer` | == 0.11.3 | Format enforcement |
| `gguf` | >= 0.17.0 | GGUF model support |
| `compressed-tensors` | == 0.15.0.1 | Compressed tensors |
| `tiktoken` | >= 0.6.0 | DBRX tokenizer |
| `mistral_common[image]` | >= 1.11.2 | Mistral models |
| `opencv-python-headless` | >= 4.13.0 | Video IO |
| `pillow` | any | Image processing |
| `regex` | any | Regex matching |
| `blake3` | any | Hashing |
| `tqdm` | any | Progress bars |
| `filelock` | >= 3.16.1 | File locking |
| `typing_extensions` | >= 4.10 | Type hints |
| `pyyaml` | any | YAML config |
| `depyf` | == 0.20.0 | Profiling/debugging |
| `cloudpickle` | any | Lambda pickling |
| `setproctitle` | any | Process naming |
| `anthropic` | >= 0.71.0 | Anthropic API |
| `mcp` | any | Model Context Protocol |

### CUDA-Specific Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.11.0 | CUDA-enabled PyTorch |
| `torchaudio` | 2.11.0 | Audio processing |
| `torchvision` | 0.26.0 | Image processing |
| `flashinfer-python` | 0.6.8.post1 | FlashInfer attention |
| `flashinfer-cubin` | 0.6.8.post1 | FlashInfer CUDA binaries |
| `numba` | 0.65.0 | N-gram speculative decoding |
| `nvidia-cudnn-frontend` | >= 1.13.0, < 1.19.0 | cuDNN frontend |
| `fastsafetensors` | >= 0.2.2 | Fast model loading |
| `nvidia-cutlass-dsl` | >= 4.4.2 | Cutlass DSL (FA4) |
| `quack-kernels` | >= 0.3.3 | QuACK kernels |
| `apache-tvm-ffi` | 0.1.9 | TVM FFI |
| `tilelang` | 0.1.9 | TileLang kernels |

### CPU-Specific Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.11.0+cpu | CPU-only PyTorch |
| `intel-openmp` | 2024.2.1 | OpenMP (x86_64 only) |
| `numba` | 0.65.0 | N-gram speculative decoding |

### ROCm-Specific Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | via ROCm PyTorch | ROCm-enabled |
| `grpcio` | 1.78.0 | gRPC communication |
| `conch-triton-kernels` | 1.2.1 | Triton kernels for ROCm |
| `amd-quark` | >= 0.8.99 | Quark quantization |
| `tilelang` | 0.1.9 | TileLang kernels |

### TPU-Specific Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `ray[default]` | any | Distributed runtime |
| `ray[data]` | any | Ray data |
| `nixl` | 0.3.0 | NIXL connector |
| `tpu-inference` | 0.18.0 | TPU inference library |
| `setuptools` | 78.1.0 | Build system |

### XPU-Specific Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.11.0+xpu | XPU-enabled PyTorch |
| `ray` | >= 2.9 | Distributed runtime |
| `vllm_xpu_kernels` | 0.1.7 | XPU custom kernels |

### Optional Extras

| Extra | Package | Version |
|-------|---------|---------|
| `zen` | `zentorch-weekly` | 5.2.1.dev20260408 |
| `bench` | `pandas, matplotlib, seaborn, datasets, scipy, plotly` | - |
| `tensorizer` | `tensorizer` | 2.10.1 |
| `fastsafetensors` | `fastsafetensors` | >= 0.2.2 |
| `instanttensor` | `instanttensor` | >= 0.1.5 |
| `runai` | `runai-model-streamer[s3,gcs,azure]` | >= 0.15.7 |
| `audio` | `av, scipy, soundfile, mistral_common[audio]` | - |
| `video` | (kept for backwards compatibility) | - |
| `flashinfer` | (kept for backwards compatibility) | - |
| `helion` | `helion` | 1.0.0 |
| `grpc` | `smg-grpc-servicer[vllm]` | >= 0.5.2 |
| `otel` | `opentelemetry-*` | >= 1.26.0 |

---

## Docker Images and Build Instructions

### Available Dockerfiles

| File | Target | Base Image |
|------|--------|-----------|
| `Dockerfile` | CUDA GPU | `nvidia/cuda:13.0.2-devel-ubuntu22.04` |
| `Dockerfile.rocm` | AMD ROCm | ROCm base |
| `Dockerfile.cpu` | CPU | Ubuntu |
| `Dockerfile.tpu` | Google TPU | TPU base |
| `Dockerfile.xpu` | Intel XPU | Intel base |
| `Dockerfile.ppc64le` | PowerPC | PowerPC base |
| `Dockerfile.s390x` | IBM s390x | s390x base |
| `Dockerfile.nightly_torch` | Nightly PyTorch | CUDA base |

### CUDA Docker Build

```bash
# Build from source
docker build -t vllm-cuda -f docker/Dockerfile .

# Build with custom CUDA version
docker build \
  --build-arg CUDA_VERSION=12.9.1 \
  -t vllm-cuda12 \
  -f docker/Dockerfile .

# Build with custom Python version
docker build \
  --build-arg PYTHON_VERSION=3.11 \
  -t vllm-cuda \
  -f docker/Dockerfile .

# Build with KV connector dependencies
docker build \
  --build-arg INSTALL_KV_CONNECTORS=true \
  -t vllm-cuda-kv \
  -f docker/Dockerfile .

# Using docker bake
docker buildx bake -f docker/docker-bake.hcl -f docker/versions.json
```

#### CUDA Docker Build Arguments

| ARG | Default | Description |
|-----|---------|-------------|
| `CUDA_VERSION` | `13.0.2` | CUDA toolkit version |
| `PYTHON_VERSION` | `3.12` | Python version |
| `UBUNTU_VERSION` | `22.04` | Ubuntu version |
| `BUILD_BASE_IMAGE` | `nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04` | Build base image |
| `FINAL_BASE_IMAGE` | `nvidia/cuda:${CUDA_VERSION}-base-ubuntu${UBUNTU_VERSION}` | Final base image |
| `BUILD_OS` | `ubuntu` | OS family for package manager |
| `INSTALL_KV_CONNECTORS` | `false` | Include KV connector dependencies |
| `PIP_INDEX_URL` | (empty) | Custom pip index |
| `UV_INDEX_URL` | (same as PIP) | Custom uv index |

### ROCm Docker Build

```bash
docker build -t vllm-rocm -f docker/Dockerfile.rocm .
```

### CPU Docker Build

```bash
docker build -t vllm-cpu -f docker/Dockerfile.cpu .
```

### TPU Docker Build

```bash
docker build -t vllm-tpu -f docker/Dockerfile.tpu .
```

### XPU Docker Build

```bash
docker build -t vllm-xpu -f docker/Dockerfile.xpu .
```

### Running Docker Containers

```bash
# CUDA
docker run --runtime nvidia --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  vllm-cuda \
  --model meta-llama/Llama-3.1-8B

# ROCm
docker run --device /dev/kfd --device /dev/dri \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  vllm-rocm \
  --model meta-llama/Llama-3.1-8B

# CPU
docker run \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  vllm-cpu \
  --model meta-llama/Llama-3.1-8B
```

---

## Environment Variables and Configuration

### Installation-Time Variables

These must be set **before** `pip install` or `docker build`.

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_TARGET_DEVICE` | `cuda` (auto-detected) | Target hardware: cuda, rocm, cpu, tpu, xpu |
| `VLLM_MAIN_CUDA_VERSION` | `13.0` | Main CUDA version for wheel naming |
| `MAX_JOBS` | CPU count | Max parallel compilation jobs |
| `NVCC_THREADS` | `1` | Threads per nvcc invocation |
| `VLLM_USE_PRECOMPILED` | `0` | Use precompiled .so files |
| `VLLM_PRECOMPILED_WHEEL_LOCATION` | None | Path or URL to precompiled wheel |
| `VLLM_PRECOMPILED_WHEEL_VARIANT` | auto-detected | Wheel variant (e.g., cu129, cu130) |
| `VLLM_PRECOMPILED_WHEEL_COMMIT` | main HEAD | Git commit for nightly wheel |
| `VLLM_DOCKER_BUILD_CONTEXT` | `0` | Force precompiled in Docker |
| `CMAKE_BUILD_TYPE` | `RelWithDebInfo` | CMake build type |
| `VERBOSE` | `0` | Verbose build output |
| `VLLM_VERSION_OVERRIDE` | None | Override version string |

### Runtime Variables

#### Core Engine

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_HOST_IP` | `""` | Node IP for distributed |
| `VLLM_PORT` | None | Base communication port |
| `VLLM_RPC_BASE_PATH` | `/tmp` | IPC path for engine communication |
| `VLLM_ENGINE_ITERATION_TIMEOUT_S` | `60` | Per-iteration timeout |
| `VLLM_ENGINE_READY_TIMEOUT_S` | `600` | Engine startup timeout |
| `VLLM_RPC_TIMEOUT` | `10000` | RPC timeout in ms |
| `VLLM_MAX_N_SEQUENCES` | `16384` | Max concurrent sequences |
| `VLLM_ENABLE_V1_MULTIPROCESSING` | `1` | V1 multiprocessing mode |
| `VLLM_WORKER_MULTIPROC_METHOD` | `fork` | Worker start method (fork/spawn) |
| `VLLM_HTTP_TIMEOUT_KEEP_ALIVE` | `5` | HTTP keep-alive timeout |
| `VLLM_KEEP_ALIVE_ON_ENGINE_DEATH` | `0` | Keep alive when engine dies |
| `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` | `300` | Model execution timeout |

#### Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_CONFIGURE_LOGGING` | `1` | Auto-configure logging |
| `VLLM_LOGGING_LEVEL` | `INFO` | Log level |
| `VLLM_LOGGING_PREFIX` | `""` | Log message prefix |
| `VLLM_LOGGING_STREAM` | `ext://sys.stdout` | Log output stream |
| `VLLM_LOGGING_CONFIG_PATH` | None | Custom logging config JSON |
| `VLLM_LOGGING_COLOR` | `auto` | Color mode |
| `VLLM_LOG_STATS_INTERVAL` | `10.0` | Stats interval (seconds) |
| `VLLM_LOG_BATCHSIZE_INTERVAL` | `-1` | Batch size log interval |
| `VLLM_TRACE_FUNCTION` | `0` | Function call tracing |

#### Compilation

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_USE_AOT_COMPILE` | auto | AOT compilation |
| `VLLM_USE_BYTECODE_HOOK` | `1` | Bytecode hook |
| `VLLM_FORCE_AOT_LOAD` | `0` | Force AOT load |
| `VLLM_USE_MEGA_AOT_ARTIFACT` | auto | Mega AOT artifact |
| `VLLM_DISABLE_COMPILE_CACHE` | `0` | Disable compile cache |
| `VLLM_USE_STANDALONE_COMPILE` | `1` | Standalone compile |
| `VLLM_ENABLE_PREGRAD_PASSES` | `1` | Pre-grad passes |
| `VLLM_DEBUG_DUMP_PATH` | None | Dump fx graphs |
| `VLLM_PATTERN_MATCH_DEBUG` | None | Debug pattern matching |
| `VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE` | `1` | Inductor autotune |
| `VLLM_COMPILE_CACHE_SAVE_FORMAT` | `binary` | Compile cache format |

#### CUDA / GPU

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_USE_FLASHINFER_SAMPLER` | `1` | FlashInfer sampler |
| `VLLM_FLOAT32_MATMUL_PRECISION` | `highest` | Float32 matmul precision |
| `VLLM_SKIP_P2P_CHECK` | `0` | Skip GPU P2P check |
| `VLLM_ENABLE_CUDAGRAPH_GC` | `0` | CUDA graph GC |
| `VLLM_USE_DEEP_GEMM` | `1` | DeepGEMM kernels |
| `VLLM_MOE_USE_DEEP_GEMM` | `1` | DeepGEMM for MoE |
| `CUDA_VISIBLE_DEVICES` | None | GPU device visibility |
| `CUDA_HOME` | None | CUDA toolkit path |
| `VLLM_NCCL_SO_PATH` | None | Custom NCCL path |

#### CPU

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_CPU_KVCACHE_SPACE` | `0` | KV cache space in bytes |
| `VLLM_CPU_OMP_THREADS_BIND` | `auto` | OpenMP binding pattern |
| `VLLM_CPU_NUM_OF_RESERVED_CPU` | `0` | Reserved CPU cores |
| `VLLM_CPU_SGL_KERNEL` | `0` | SGL kernels |
| `VLLM_CPU_ATTN_SPLIT_KV` | `1` | Attention split KV |
| `VLLM_ZENTORCH_WEIGHT_PREPACK` | `1` | ZenDNN prepack |

#### ROCm

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE` | `256` | Sleep memory chunk size (MB) |
| `VLLM_ROCM_USE_AITER` | `0` | AITER kernels |
| `VLLM_ROCM_USE_AITER_PAGED_ATTN` | `0` | AITER paged attention |
| `VLLM_ROCM_USE_AITER_LINEAR` | `1` | AITER linear |
| `VLLM_ROCM_USE_AITER_MOE` | `1` | AITER MoE |
| `VLLM_ROCM_FP8_PADDING` | `1` | FP8 padding |
| `VLLM_ROCM_MOE_PADDING` | `1` | MoE padding |

#### TPU

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_XLA_CACHE_PATH` | `~/.cache/vllm/xla_cache` | XLA cache path |
| `VLLM_XLA_CHECK_RECOMPILATION` | `0` | Assert on XLA recompilation |
| `VLLM_XLA_USE_SPMD` | `0` | SPMD mode |
| `VLLM_TPU_BUCKET_PADDING_GAP` | `0` | Bucket padding gap |
| `VLLM_TPU_USING_PATHWAYS` | `0` | Pathways proxy |

#### XPU

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_XPU_ENABLE_XPU_GRAPH` | `0` | XPU graph support |
| `VLLM_XPU_USE_SAMPLER_KERNEL` | `1` | XPU sampler kernel |

#### Multimodal / Media

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_IMAGE_FETCH_TIMEOUT` | `5` | Image fetch timeout (s) |
| `VLLM_VIDEO_FETCH_TIMEOUT` | `30` | Video fetch timeout (s) |
| `VLLM_AUDIO_FETCH_TIMEOUT` | `10` | Audio fetch timeout (s) |
| `VLLM_MEDIA_CACHE_MAX_SIZE_MB` | `5120` | Media cache max size |
| `VLLM_MEDIA_CACHE_TTL_HOURS` | `24` | Cache TTL |
| `VLLM_MEDIA_FETCH_MAX_RETRIES` | `3` | Fetch retries |
| `VLLM_VIDEO_LOADER_BACKEND` | `opencv` | Video loader |
| `VLLM_MM_HASHER_ALGORITHM` | `blake3` | Hash algorithm |

#### Distributed

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_DP_RANK` | `0` | Data parallel rank |
| `VLLM_DP_SIZE` | `1` | Data parallel size |
| `LOCAL_RANK` | `0` | Local GPU rank |
| `VLLM_USE_RAY_V2_EXECUTOR_BACKEND` | `1` | Ray V2 executor |
| `VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE` | `auto` | Ray channel type |

#### Cache / Storage

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_CACHE_ROOT` | `~/.cache/vllm` | Cache root |
| `VLLM_CONFIG_ROOT` | `~/.config/vllm` | Config root |
| `VLLM_ASSETS_CACHE` | `~/.cache/vllm/assets` | Assets cache |
| `VLLM_USE_MODELSCOPE` | `False` | Use ModelScope |

---

## Platform-Specific Setup

### CUDA Setup

```bash
# 1. Install NVIDIA driver (535+)
sudo apt-get install nvidia-driver-535

# 2. Verify CUDA
nvidia-smi
nvcc --version

# 3. Install vLLM
pip install vllm

# 4. Verify
python -c "import vllm; print(vllm.__version__)"
```

### ROCm Setup

```bash
# 1. Install ROCm (6.x+)
# Follow AMD ROCm installation guide

# 2. Verify
rocminfo
hipcc --version

# 3. Install vLLM for ROCm
pip install vllm
# Or from AMD's PyPI index:
pip install vllm -f https://pypi.amd.com/vllm-rocm/simple

# 4. Verify
python -c "import torch; print(torch.version.hip)"
```

### CPU Setup

```bash
# 1. Install system dependencies
sudo apt-get install -y libtcmalloc-minimal4

# 2. Install vLLM for CPU
VLLM_TARGET_DEVICE=cpu pip install -e .

# Or use pre-built CPU wheel
pip install vllm

# 3. Configure CPU settings
export VLLM_CPU_KVCACHE_SPACE=4  # 4 GB KV cache
export VLLM_CPU_OMP_THREADS_BIND="0-31"

# 4. Verify
python -c "import vllm; print(vllm.__version__)"
```

#### CPU Architecture-Specific Notes

**x86_64:**
- Requires AVX2 or AVX-512
- Builds custom C extensions with AVX2 and AVX-512 variants
- Intel OpenMP recommended (`intel-openmp`)

**ARM64/aarch64:**
- Supported on ARM Neoverse cores
- Uses NEON SIMD instructions

**PowerPC (ppc64le):**
- Supported but may require building from source

**s390x (IBM Z):**
- Supported for IBM Z mainframes
- No numba dependency

**RISC-V:**
- Community-supported
- Chunked prefill and prefix caching disabled

### TPU Setup

```bash
# 1. Install TPU dependencies
pip install -r requirements/tpu.txt

# 2. Set environment
export VLLM_TARGET_DEVICE=tpu

# 3. Install vLLM
pip install -e .

# 4. Configure XLA cache
export VLLM_XLA_CACHE_PATH=~/.cache/vllm/xla_cache
```

### XPU Setup

```bash
# 1. Install Intel oneAPI toolkit
# Follow Intel installation guide

# 2. Install vLLM for XPU
pip install -r requirements/xpu.txt
pip install -e .

# 3. Verify
python -c "import torch; print(torch.xpu.is_available())"
```

### AMD Zen CPU with zentorch

```bash
# Install with Zen optimizations
pip install vllm[zen]
```

---

## Verification Steps

### Step 1: Check Installation

```python
import vllm
print(f"vLLM version: {vllm.__version__}")
```

### Step 2: Collect Environment Info

```bash
python -m vllm.collect_env
```

This outputs:
- OS, GCC, CMake versions
- PyTorch version and CUDA availability
- GPU models and driver versions
- vLLM version and build flags
- All relevant environment variables

### Step 3: Basic Inference Test

```python
from vllm import LLM, SamplingParams

# Create LLM instance
llm = LLM(model="meta-llama/Llama-3.1-8B")

# Generate
prompts = ["Hello, my name is"]
sampling_params = SamplingParams(max_tokens=50, temperature=0.8)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}")
```

### Step 4: API Server Test

```bash
# Start server
vllm serve meta-llama/Llama-3.1-8B

# Test with curl
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B",
    "prompt": "Hello",
    "max_tokens": 50
  }'
```

### Step 5: Platform Detection

```python
from vllm.platforms import current_platform

print(f"Platform: {current_platform.platform_name}")
print(f"Device type: {current_platform.device_type}")
if current_platform.is_cuda():
    print(f"Device: {current_platform.get_device_name()}")
    print(f"Memory: {current_platform.get_device_total_memory() / 1e9:.1f} GB")
```

---

## Troubleshooting

### Common Issues

#### CUDA Out of Memory

```
Solution: Reduce gpu_memory_utilization or use quantization
--gpu-memory-utilization 0.8
--quantization awq
```

#### Build Errors: CMake not found

```bash
pip install cmake ninja
```

#### Build Errors: NVCC version mismatch

```bash
# Check CUDA versions
nvcc --version
nvidia-smi  # Shows driver-supported CUDA version

# Ensure consistency
export CUDA_HOME=/usr/local/cuda
```

#### ROCm: Library not found

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib
export ROCM_PATH=/opt/rocm
```

#### CPU: Slow inference

```bash
# Install tcmalloc
sudo apt-get install -y libtcmalloc-minimal4
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4

# Increase KV cache
export VLLM_CPU_KVCACHE_SPACE=8  # 8 GB

# Bind CPU threads
export VLLM_CPU_OMP_THREADS_BIND="0-63"
```

#### ImportError: vllm._C

```bash
# Reinstall from source without precompiled
pip install -e . --no-build-isolation --torch-backend=auto
```

#### HuggingFace Offline Mode

```bash
export HF_HUB_OFFLINE=True
export TRANSFORMERS_OFFLINE=1
# vLLM auto-detects offline mode and uses local paths
```

#### FlashInfer Installation Issues

```bash
# FlashInfer is included as a dependency for CUDA
# If issues arise, ensure correct version:
pip install flashinfer-python==0.6.8.post1 flashinfer-cubin==0.6.8.post1
```

#### Environment Variable Validation

```bash
# vLLM can validate environment variables at startup
vllm serve model_name --fail-on-environ-validation
```

#### NCCL Issues

```bash
# Specify custom NCCL path
export VLLM_NCCL_SO_PATH=/path/to/libnccl.so

# Disable custom all-reduce
--disable-custom-all-reduce
```

#### Model Download Issues

```bash
# Use ModelScope instead of HuggingFace
export VLLM_USE_MODELSCOPE=True

# Use custom download directory
--download-dir /path/to/cache
```

#### Performance Tuning

```bash
# Enable throughput mode
--performance-mode throughput

# Increase max tokens per batch
--max-num-batched-tokens 16384

# Enable prefix caching
--enable-prefix-caching

# Enable chunked prefill (default on supported models)
--enable-chunked-prefill
```
