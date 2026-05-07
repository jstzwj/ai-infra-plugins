# SGLang Installation and Setup Reference

This document provides comprehensive installation instructions for SGLang across all supported
platforms, hardware configurations, and deployment methods.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation Methods](#installation-methods)
3. [Hardware Support Matrix](#hardware-support-matrix)
4. [Docker Image Variants](#docker-image-variants)
5. [Platform-Specific Guides](#platform-specific-guides)
6. [Post-Installation Verification](#post-installation-verification)
7. [Common Troubleshooting](#common-troubleshooting)
8. [Upgrading SGLang](#upgrading-sglang)

---

## Prerequisites

### Software Requirements

| Requirement | Minimum Version | Recommended | Notes |
|-------------|----------------|-------------|-------|
| Python | 3.10 | 3.10+ | Required |
| CUDA | 12.x or 13.x | 13.x | For NVIDIA GPUs (sm80+) |
| PyTorch | 2.x (auto-installed) | Latest | Auto-installed as dependency |
| FlashInfer | auto-installed | Latest | Default attention kernel backend |
| Git | 2.x | Latest | For source installation |
| Docker | 20.x+ | Latest | For containerized deployment |

### Hardware Requirements

#### NVIDIA GPUs

| GPU Series | Compute Capability | Supported | Notes |
|------------|-------------------|-----------|-------|
| B200 / B300 / GB300 | sm100 / sm103a | Yes | Latest Blackwell, full support |
| H100 / H200 | sm90 | Yes | Hopper, full support |
| A100 | sm80 | Yes | Ampere, full support |
| L40S | sm8.9 | Yes | Ampere, full support |
| L4 | sm8.9 | Yes | Ampere, full support |
| A10 | sm80 | Yes | Ampere, full support |
| RTX 5090 | sm120 | Yes | Ada Lovelace successor |
| RTX 4090 | sm8.9 | Yes | Ada Lovelace |
| T4 | sm75 | Partial | FlashInfer may require fallback backends |
| V100 | sm70 | No | Not supported by FlashInfer |

#### Other Accelerators

| Platform | Supported | Backend |
|----------|-----------|---------|
| AMD MI300 / MI355 | Yes | ROCm / HIP |
| Intel Xeon CPU | Yes | AMX instructions |
| Google TPU | Yes | SGLang-Jax backend |
| NVIDIA Jetson | Yes | CUDA (aarch64) |
| Ascend NPUs | Yes | Ascend CANN |
| Intel XPU | Yes | Intel XPU runtime |
| NVIDIA DGX Spark | Yes | CUDA |

### Operating System

| OS | Support Level |
|----|--------------|
| Linux (x86_64) | Fully supported, recommended |
| Linux (aarch64) | Supported (Jetson) |
| macOS (MPS) | Partial (stubs available, no GPU inference) |
| Windows | Not officially supported |

---

## Installation Methods

### Method 1: pip / uv (Recommended)

This is the simplest and recommended installation method for most users.

#### Using uv (faster, recommended)

```bash
# Upgrade pip
pip install --upgrade pip

# Install uv
pip install uv

# Install SGLang
uv pip install sglang
```

#### Using pip

```bash
pip install --upgrade pip
pip install sglang
```

#### Installing specific version

```bash
pip install sglang==0.5.9
```

#### Installing with optional dependencies

```bash
# For HTTP/2 support (Granian server)
pip install "sglang[http2]"

# For development
pip install -e "python[dev]"
```

### Method 2: From Source

Install the latest development version from the GitHub repository.

```bash
# Clone the repository (use the latest release branch)
git clone -b v0.5.9 https://github.com/sgl-project/sglang.git
cd sglang

# Install Python packages
pip install --upgrade pip
pip install -e "python"
```

#### Installing from main branch (bleeding edge)

```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang
pip install --upgrade pip
pip install -e "python"
```

### Method 3: Docker

Docker images are available on Docker Hub at `lmsysorg/sglang`. Images are built from the
official Dockerfile in the repository.

#### Standard Docker Image

```bash
docker run --gpus all \
    --shm-size 32g \
    -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=<your-hf-token>" \
    --ipc=host \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server \
        --model-path meta-llama/Llama-3.1-8B-Instruct \
        --host 0.0.0.0 \
        --port 30000
```

#### Runtime Docker Image (Smaller, Recommended for Production)

The runtime variant is approximately 40% smaller by excluding build tools and development
dependencies:

```bash
docker run --gpus all \
    --shm-size 32g \
    -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=<your-hf-token>" \
    --ipc=host \
    lmsysorg/sglang:latest-runtime \
    python3 -m sglang.launch_server \
        --model-path meta-llama/Llama-3.1-8B-Instruct \
        --host 0.0.0.0 \
        --port 30000
```

#### Key Docker Flags Explained

| Flag | Value | Purpose |
|------|-------|---------|
| `--gpus all` | - | Expose all NVIDIA GPUs to the container |
| `--shm-size 32g` | 32 GB | Shared memory size (required for tensor parallelism) |
| `-p 30000:30000` | Host:Container | Port mapping |
| `-v ~/.cache/huggingface:...` | Host:Container | HuggingFace model cache mount |
| `--env HF_TOKEN` | - | HuggingFace authentication token |
| `--ipc=host` | - | Use host IPC namespace (required for NCCL) |

#### Using Docker Compose

Copy the `compose.yml` from the repository to your local machine and run:

```bash
docker compose up -d
```

This is recommended for production service deployments. For even better orchestration, use the
Kubernetes manifests.

### Method 4: Kubernetes

#### Using OME (Recommended)

[OME](https://github.com/sgl-project/ome) is a Kubernetes operator for enterprise-grade
management and serving of large language models.

#### Single-Node Serving

For models that fit on a single GPU node:

```bash
kubectl apply -f docker/k8s-sglang-service.yaml
```

#### Multi-Node Distributed Serving

For large models requiring multiple GPU nodes (e.g., DeepSeek-R1):

1. Modify the LLM model path and arguments in the manifest
2. Apply the stateful set configuration:

```bash
kubectl apply -f docker/k8s-sglang-distributed-sts.yaml
```

This creates a two-node Kubernetes StatefulSet with a serving Service.

### Method 5: SkyPilot (Cloud and Kubernetes)

Deploy on any of 12+ cloud providers or Kubernetes using [SkyPilot](https://github.com/skypilot-org/skypilot).

#### Setup

```bash
# Install SkyPilot
pip install skypilot

# Configure cloud access
sky check
```

#### Create SkyPilot YAML

```yaml
# sglang.yaml
envs:
  HF_TOKEN: null

resources:
  image_id: docker:lmsysorg/sglang:latest
  accelerators: A100
  ports: 30000

run: |
  conda deactivate
  python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 30000
```

#### Deploy

```bash
# Deploy on any cloud or Kubernetes
HF_TOKEN=<your-token> sky launch -c sglang --env HF_TOKEN sglang.yaml

# Get the HTTP API endpoint
sky status --endpoint 30000 sglang
```

#### Auto-scaling with SkyServe

For auto-scaling and failure recovery, use [SkyServe](https://github.com/skypilot-org/skypilot/tree/master/llm/sglang).

### Method 6: AWS SageMaker

#### Using AWS SGLang DLC

AWS provides pre-built SGLang Deep Learning Containers with security patching. Check
[AWS SGLang DLCs](https://github.com/aws/deep-learning-containers/blob/master/available_images.md#sglang-containers)
for available images.

#### Custom Container Deployment

1. Build a custom Docker image:

```bash
docker build -t sglang-sagemaker -f docker/sagemaker.Dockerfile .
```

2. Push to AWS ECR:

```bash
AWS_ACCOUNT="<your-account>"
AWS_REGION="<your-region>"
ECR_REGISTRY="${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com"

aws ecr get-login-password --region ${AWS_REGION} | \
    docker login --username AWS --password-stdin ${ECR_REGISTRY}

docker tag sglang-sagemaker ${ECR_REGISTRY}/sglang:latest
docker push ${ECR_REGISTRY}/sglang:latest
```

3. Deploy using SageMaker Python SDK:

```python
import sagemaker
from sagemaker.model import Model

model = Model(
    image_uri=f"{ECR_REGISTRY}/sglang:latest",
    model_data="s3://your-bucket/model.tar.gz",
    role="your-sagemaker-role",
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.12xlarge",
)
```

#### SageMaker Environment Variable Configuration

The SageMaker serve script automatically converts environment variables with the prefix
`SM_SGLANG_` to CLI arguments. For example:

| Environment Variable | CLI Flag |
|---------------------|----------|
| `SM_SGLANG_MODEL_PATH=Qwen/Qwen3-0.6B` | `--model-path Qwen/Qwen3-0.6B` |
| `SM_SGLANG_REASONING_PARSER=qwen3` | `--reasoning-parser qwen3` |
| `SM_SGLANG_TP_SIZE=4` | `--tp-size 4` |

Default serving command:
```
python3 -m sglang.launch_server --model-path opt/ml/model --host 0.0.0.0 --port 8080
```

---

## Hardware Support Matrix

### Complete Platform Support

| Platform | Hardware | Backend | Installation Method | Special Notes |
|----------|----------|---------|-------------------|---------------|
| NVIDIA GPU (CUDA) | H100, A100, L40S, L4, A10, B200, etc. | CUDA 12/13 | pip, Docker, source | Default platform |
| NVIDIA GPU (DGX Spark) | DGX Spark | CUDA | Docker | Specialized guide available |
| AMD GPU (ROCm) | MI300X, MI355 | ROCm/HIP | Docker, source | Use ROCm images |
| Intel Xeon CPU | Xeon w/ AMX | CPU/AMX | pip, source | AMX instructions required |
| Google TPU | TPU v4, v5 | JAX | Source, Docker | SGLang-Jax backend |
| NVIDIA Jetson | Jetson AGX, Orin | CUDA (aarch64) | Docker, source | ARM64 build |
| Ascend NPU | Ascend 910B | CANN | Docker, source | Ascend platform plugin |
| Intel XPU | Intel GPU Max | XPU runtime | Docker, source | XPU platform plugin |

### GPU Memory Requirements by Model Size

The following table provides approximate GPU memory requirements for common model configurations.

These are rough estimates and actual usage depends on context length, batch size, quantization,
and KV-cache settings.

| Model Size | Precision | Min GPU Memory | Recommended | Example GPU |
|------------|-----------|----------------|-------------|-------------|
| 0.5B params | FP16/BF16 | 2 GB | 4 GB | RTX 3090 |
| 1.5B params | FP16/BF16 | 4 GB | 8 GB | RTX 3090 |
| 7B params | FP16/BF16 | 16 GB | 24 GB | RTX 4090, L4 |
| 7B params | FP8/INT8 | 8 GB | 16 GB | L4, A10 |
| 8B params | FP16/BF16 | 18 GB | 24 GB | L40S, RTX 4090 |
| 13B params | FP16/BF16 | 28 GB | 48 GB | A6000, A100-40G |
| 70B params | FP16/BF16 | 140 GB | 160 GB | 2x H100, 2x A100-80G |
| 70B params | FP8/INT8 | 72 GB | 80 GB | 1x H100, 1x A100-80G |
| 405B params | FP16/BF16 | 810 GB | 900 GB | 8x H100 |
| 405B params | FP8 | 410 GB | 480 GB | 8x H100 |
| DeepSeek-V3 (671B MoE) | FP8 | ~160 GB active | 8x H100 | 8x H100 (with EP) |
| DeepSeek-R1 (671B MoE) | FP8 | ~160 GB active | 8x H100 | 8x H100 (with EP) |

---

## Docker Image Variants

SGLang provides several Docker image variants on Docker Hub (`lmsysorg/sglang`).

### Image Tags

| Tag | Description | Approximate Size | Use Case |
|-----|-------------|------------------|----------|
| `latest` | Standard image with all dependencies | ~15 GB | Development, testing |
| `latest-runtime` | Production image without build tools | ~9 GB | Production deployment |
| `latest-cu12` | Standard image with CUDA 12 | ~14 GB | CUDA 12 environments |
| `latest-cu129` | Standard image with CUDA 12.9 | ~14 GB | CUDA 12.9 environments |
| `nightly` | Nightly build from main branch | ~15 GB | Testing latest features |
| `nightly-runtime` | Nightly runtime build | ~9 GB | Testing latest with smaller image |
| `dev` | Development image with build tools | ~18 GB | SGLang development |

### CUDA Version Notes

SGLang ships with CUDA 13 by default. For CUDA 12 environments, use images with the `-cu12` or
`-cu129` suffix:

```bash
# CUDA 12 environment
lmsysorg/sglang:latest-cu129

# CUDA 12 development
lmsysorg/sglang:dev-cu12
```

### Building Custom Docker Images

You can build custom Docker images from the repository Dockerfiles:

```bash
# Standard image
docker build -t sglang-custom -f docker/Dockerfile .

# Runtime image
docker build -t sglang-custom-runtime -f docker/Dockerfile.runtime .

# SageMaker image
docker build -t sglang-sagemaker -f docker/sagemaker.Dockerfile .
```

---

## Platform-Specific Guides

### AMD GPUs (ROCm)

For AMD GPU support (MI300X, MI355, etc.), refer to the dedicated AMD GPU guide at
`docs/hardware-platforms/amd_gpu`.

Key considerations:
- Use ROCm-compatible Docker images
- Set `--attention-backend aiter` or `--attention-backend wave` for AMD-optimized kernels
- The `--sampling-backend` may need adjustment
- FlashInfer is not available on AMD; alternative backends are used automatically

### Intel Xeon CPUs

For CPU-only inference on Intel Xeon processors:

- Requires AMX instruction support (4th Gen Intel Xeon or newer)
- Set `--device cpu` when launching the server
- Use `--attention-backend intel_amx` for optimized CPU attention
- Performance is significantly lower than GPU inference

### Google TPUs

SGLang runs natively on Google TPUs using the SGLang-Jax backend:

- Requires a Google Cloud TPU instance
- Uses the JAX framework instead of PyTorch
- Refer to `docs/hardware-platforms/tpu` for setup instructions

### NVIDIA Jetson

For NVIDIA Jetson devices (AGX Orin, etc.):

- Use ARM64-compatible Docker images
- CUDA is supported but with limited compute capability
- Memory constraints may limit model size

### Ascend NPUs

For Huawei Ascend NPU devices:

- Requires CANN toolkit installation
- Use Ascend-specific Docker images
- Refer to `docs/hardware-platforms/ascend-npus/ascend_npu`

### Intel XPU

For Intel GPU Max and other XPU devices:

- Requires Intel XPU runtime
- Refer to `docs/hardware-platforms/xpu`

---

## Post-Installation Verification

After installing SGLang, verify the installation is working correctly.

### Step 1: Check Version

```python
import sglang
print(sglang.__version__)
```

### Step 2: Launch a Test Server

```bash
# Launch server with a small model
python3 -m sglang.launch_server \
    --model-path qwen/qwen2.5-0.5b-instruct \
    --host 0.0.0.0 \
    --port 30000
```

Wait for the output: `The server is fired up and ready to roll!`

### Step 3: Send a Test Request

```bash
curl -s http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen/qwen2.5-0.5b-instruct",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Step 4: Verify API Documentation

Open a browser or curl the following endpoints:
- Swagger UI: `http://localhost:30000/docs`
- ReDoc: `http://localhost:30000/redoc`
- OpenAPI Spec: `http://localhost:30000/openapi.json`

### Step 5: Verify Health Check

```bash
curl http://localhost:30000/health
curl http://localhost:30000/health_generate
```

### Step 6: Test Offline Engine (No Server Required)

```python
import sglang as sgl

llm = sgl.Engine(model_path="qwen/qwen2.5-0.5b-instruct")
outputs = llm.generate(["Hello, world!"], {"temperature": 0.7, "max_new_tokens": 32})
print(outputs)
llm.shutdown()
```

---

## Common Troubleshooting

### CUDA_HOME Not Set

**Error:** `OSError: CUDA_HOME environment variable is not set`

**Solution:**

```bash
export CUDA_HOME=/usr/local/cuda-<your-cuda-version>
```

Alternatively, install FlashInfer first following the
[FlashInfer installation doc](https://docs.flashinfer.ai/installation.html), then install SGLang.

### FlashInfer Issues on sm75+ Devices

**Error:** FlashInfer-related errors on T4, A10, A100, L4, L40S, or H100 GPUs.

**Solution:** Switch to alternative backends:

```bash
python3 -m sglang.launch_server \
    --model-path <model> \
    --attention-backend triton \
    --sampling-backend pytorch
```

### Reinstalling FlashInfer

If FlashInfer is corrupted or outdated:

```bash
pip3 install --upgrade flashinfer-python --force-reinstall --no-deps
rm -rf ~/.cache/flashinfer
```

### ptxas Error on B300/GB300 (sm_103a)

**Error:** ptxas compilation errors on Blackwell B300/GB300 GPUs.

**Solution:**

```bash
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
```

### Out of Memory (OOM) Errors

**Error:** CUDA out of memory during model loading or inference.

**Solutions:**

1. Reduce memory fraction: `--mem-fraction-static 0.8`
2. Use quantization: `--quantization fp8` or `--quantization awq`
3. Reduce context length: `--context-length 4096`
4. Reduce chunked prefill size: `--chunked-prefill-size 2048`
5. Disable CUDA graphs: `--disable-cuda-graph`
6. Use multiple GPUs: `--tp-size 2` (or more)

### Model Download Issues

**Error:** Cannot download model from HuggingFace.

**Solutions:**

1. Set HF_TOKEN: `export HF_TOKEN=<your-token>`
2. Use a mirror: `export HF_ENDPOINT=https://hf-mirror.com`
3. Pre-download and specify local path

### Port Already in Use

**Error:** `Address already in use` when launching server.

**Solution:** Use a different port:

```bash
python3 -m sglang.launch_server --model-path <model> --port 30001
```

### Shared Memory Issues in Docker

**Error:** NCCL communication errors in Docker.

**Solution:** Increase shared memory:

```bash
docker run --shm-size 32g ...
```

### Import Errors

**Error:** `ModuleNotFoundError` or `ImportError` for SGLang modules.

**Solutions:**

1. Reinstall: `pip install --force-reinstall sglang`
2. Check Python version: `python --version` (must be 3.10+)
3. Verify installation: `python -c "import sglang; print(sglang.__version__)"`

### Slow First Request

The first request after server launch may be slow due to CUDA graph warmup. This is normal
behavior. Subsequent requests will be faster.

To skip warmup during development:

```bash
python3 -m sglang.launch_server --model-path <model> --skip-server-warmup
```

---

## Upgrading SGLang

### pip Upgrade

```bash
pip install --upgrade sglang
```

### uv Upgrade

```bash
uv pip install --upgrade sglang
```

### Source Upgrade

```bash
cd sglang
git pull
pip install -e "python"
```

### Docker Upgrade

Pull the latest image:

```bash
docker pull lmsysorg/sglang:latest
# or
docker pull lmsysorg/sglang:latest-runtime
```

### Version Pinning

For production deployments, pin to a specific version:

```bash
pip install sglang==0.5.9
# or in Docker
docker pull lmsysorg/sglang:v0.5.9
```

---

## Related Documentation

- [Overview and Architecture](./01-overview-architecture.md)
- [Server Configuration Reference](./03-server-configuration.md)
- [API Reference](./04-api-reference.md)
