# SGLang Diffusion Models Reference

This document provides a comprehensive reference for SGLang's diffusion model inference
framework, covering image and video generation architectures, supported models, installation,
server configuration, API endpoints, quantization, disaggregation, performance optimization,
compatibility matrices, development guides, benchmarking, and contributing workflows.

---

## Table of Contents

1. [Overview of SGLang Diffusion](#overview-of-sglang-diffusion)
2. [Architecture](#architecture)
3. [Supported Diffusion Models](#supported-diffusion-models)
4. [Installation and Setup](#installation-and-setup)
5. [Server Configuration](#server-configuration)
6. [Usage Guide](#usage-guide)
7. [API Reference](#api-reference)
8. [Quantization Support](#quantization-support)
9. [Disaggregation](#disaggregation)
10. [Performance Optimization](#performance-optimization)
11. [Caching Acceleration](#caching-acceleration)
12. [Attention Backends](#attention-backends)
13. [Post-Processing](#post-processing)
14. [Compatibility Matrix](#compatibility-matrix)
15. [Environment Variables](#environment-variables)
16. [Development Guide: Adding New Models](#development-guide-adding-new-models)
17. [Benchmarking](#benchmarking)
18. [Contributing](#contributing)

---

## Overview of SGLang Diffusion

### What It Is

SGLang Diffusion is a high-performance inference framework for image and video generation.
It extends the SGLang serving system with native pipelines for diffusion transformer (DiT)
models, a diffusers backend for compatibility, an OpenAI-compatible HTTP server, and an
optimized kernel stack built on both precompiled `sgl-kernel` operators and JIT kernels for
key inference paths.

### Key Features

- **Broad model support** across Wan, HunyuanVideo, Qwen-Image, FLUX, Z-Image, GLM-Image,
  LTX-Video, SANA, Stable Diffusion 3, Helios, ERNIE-Image, FireRed-Image-Edit, and more.
- **Fast inference** with `sgl-kernel`, JIT kernels, scheduler improvements, and caching
  acceleration (Cache-DiT and TeaCache).
- **Multiple interfaces**: `sglang generate` for one-off jobs, `sglang serve` for persistent
  HTTP serving, and an OpenAI-compatible API.
- **Multi-platform support** for NVIDIA, AMD (ROCm), Intel XPU, Ascend NPU, Apple Silicon
  (MPS), and Moore Threads (MUSA).
- **Quantization**: FP8, NVFP4, SVDQuant (Nunchaku), and msmodelslim for compressed inference.
- **Disaggregation**: Split the pipeline into Encoder, Denoiser, and Decoder roles running on
  separate GPUs or machines.
- **Advanced parallelism**: Tensor parallelism (TP), sequence parallelism (SP), Ulysses/Ring
  attention, and per-role parallelism in disaggregated mode.
- **Post-processing**: Frame interpolation (RIFE) and upscaling (Real-ESRGAN).

### Project Metadata

| Field              | Value                                                |
|--------------------|------------------------------------------------------|
| Module Path        | `sglang.multimodal_gen`                              |
| GitHub Repository  | [sgl-project/sglang](https://github.com/sgl-project/sglang) |
| Install Extra      | `sglang[diffusion]`                                  |
| License            | Apache 2.0                                           |

### Quick Start

```bash
# Install
uv pip install "sglang[diffusion]" --prerelease=allow

# One-off image generation
sglang generate --model-path Qwen/Qwen-Image \
  --prompt "A beautiful sunset over the mountains" \
  --save-output

# Start HTTP server
sglang serve --model-path Qwen/Qwen-Image --port 30010
```

---

## Architecture

### Pipeline Architecture

SGLang Diffusion is built upon a pipeline architecture consisting of two core abstractions:

- **`ComposedPipeline`**: Orchestrates a series of `PipelineStage`s to define the complete
  generation process for a specific model. Acts as the main entry point and manages data flow
  between stages.
- **`PipelineStage`**: Each stage encapsulates a function within the diffusion process. Stages
  include prompt encoding, latent preparation, timestep preparation, the denoising loop, and
  VAE decoding.

### Two Pipeline Styles

#### Style A: Hybrid Monolithic Pipeline (Recommended Default)

Uses a three-stage structure where pre-processing is consolidated into a single model-specific
stage:

```
{Model}BeforeDenoisingStage  -->  DenoisingStage  -->  DecodingStage
     (model-specific)            (standard)           (standard)
```

| Stage                          | Ownership       | Responsibility                                                     |
|--------------------------------|-----------------|---------------------------------------------------------------------|
| `{Model}BeforeDenoisingStage`  | Model-specific  | Input validation, text/image encoding, latent preparation, timestep computation |
| `DenoisingStage`               | Framework-standard | Denoising loop (DiT/UNet forward passes)                          |
| `DecodingStage`                | Framework-standard | VAE decoding from latent to pixel space                            |

This style is recommended when the model has unique or complex pre-processing (VLM captioning,
AR token generation, custom latent packing, etc.).

#### Style B: Modular Composition Style

Uses fine-grained standard stages to build the pipeline by composition. Convenience methods
like `add_standard_t2i_stages()` and `add_standard_ti2i_stages()` make this concise. Appropriate
when the model's pre-processing can largely reuse existing stages.

### Core Stages Reference

| Stage Class                | Description                                                              |
|----------------------------|--------------------------------------------------------------------------|
| `DenoisingStage`           | Executes the main denoising loop, iteratively applying the DiT/UNet     |
| `DecodingStage`            | Decodes latent tensor back into pixel space using the VAE               |
| `DmdDenoisingStage`        | Specialized denoising for DMD model architectures                       |
| `CausalDMDDenoisingStage`  | Specialized causal denoising for specific video models                  |
| `InputValidationStage`     | Validates user-provided `SamplingParams`                                 |
| `TextEncodingStage`        | Encodes text prompts into embeddings using text encoders                 |
| `ImageEncodingStage`       | Encodes input images into embeddings                                     |
| `ImageVAEEncodingStage`    | Encodes an input image into latent space using the VAE                   |
| `TimestepPreparationStage` | Prepares the scheduler's timesteps for the diffusion process             |
| `LatentPreparationStage`   | Creates the initial noisy latent tensor                                  |

### Model Components

Each pipeline references modules loaded from the model repository:

| Component           | Description                                                       |
|---------------------|-------------------------------------------------------------------|
| `text_encoder`      | Encodes text prompts into embeddings                              |
| `tokenizer`         | Tokenizes raw text input for the text encoder(s)                  |
| `processor`         | Preprocesses images and extracts features                         |
| `image_encoder`     | Specialized image feature extractor                               |
| `dit`/`transformer` | Core denoising network (DiT/UNet) operating in latent space       |
| `scheduler`         | Controls the timestep schedule and denoising dynamics             |
| `vae`               | Variational Autoencoder for encoding/decoding between pixel and latent space |

---

## Supported Diffusion Models

### Video Generation Models

| Model Name                   | Hugging Face Model ID                                     | Resolutions                    |
|:-----------------------------|:----------------------------------------------------------|:-------------------------------|
| FastWan2.1 T2V 1.3B          | `FastVideo/FastWan2.1-T2V-1.3B-Diffusers`                 | 480p                           |
| FastWan2.2 TI2V 5B Full Attn | `FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers`         | 720p                           |
| Wan2.2 TI2V 5B               | `Wan-AI/Wan2.2-TI2V-5B-Diffusers`                         | 720p                           |
| Wan2.2 T2V A14B              | `Wan-AI/Wan2.2-T2V-A14B-Diffusers`                        | 480p, 720p                     |
| Wan2.2 I2V A14B              | `Wan-AI/Wan2.2-I2V-A14B-Diffusers`                        | 480p, 720p                     |
| HunyuanVideo                 | `hunyuanvideo-community/HunyuanVideo`                     | 720x1280, 544x960              |
| FastHunyuan                  | `FastVideo/FastHunyuan-diffusers`                         | 720x1280, 544x960              |
| Wan2.1 T2V 1.3B              | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`                        | 480p                           |
| Wan2.1 T2V 14B               | `Wan-AI/Wan2.1-T2V-14B-Diffusers`                         | 480p, 720p                     |
| Wan2.1 I2V 480P              | `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers`                    | 480p                           |
| Wan2.1 I2V 720P              | `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers`                    | 720p                           |
| TurboWan2.1 T2V 1.3B         | `IPostYellow/TurboWan2.1-T2V-1.3B-Diffusers`              | 480p                           |
| TurboWan2.1 T2V 14B          | `IPostYellow/TurboWan2.1-T2V-14B-Diffusers`               | 480p                           |
| TurboWan2.1 T2V 14B 720P     | `IPostYellow/TurboWan2.1-T2V-14B-720P-Diffusers`          | 720p                           |
| TurboWan2.2 I2V A14B         | `IPostYellow/TurboWan2.2-I2V-A14B-Diffusers`              | 720p                           |
| Wan2.1 Fun 1.3B InP          | `weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers`                 | 480p                           |
| Helios Base                  | `BestWishYsh/Helios-Base`                                 | 720p                           |
| Helios Mid                   | `BestWishYsh/Helios-Mid`                                  | 720p                           |
| Helios Distilled             | `BestWishYsh/Helios-Distilled`                            | 720p                           |
| LTX-2                        | `Lightricks/LTX-2`                                        | 768x512, 1536x1024             |
| LTX-2.3                      | `Lightricks/LTX-2.3`                                      | 768x512, 1536x1024, 1920x1088  |

**Notes on LTX pipeline selection**:
- One-stage: `--pipeline-class-name LTX2Pipeline`
- Two-stage: `--pipeline-class-name LTX2TwoStagePipeline`
- Two-stage HQ: `--pipeline-class-name LTX2TwoStageHQPipeline`
- LTX-2 and LTX-2.3 support both T2V and TI2V (`--image-path`) on one-stage and two-stage pipelines.
- LTX-2/2.3 two-stage supports `--ltx2-two-stage-device-mode {original,snapshot,resident}`:
  - `snapshot` (default, recommended)
  - `resident` (best latency/throughput, higher VRAM)
  - `original` (official semantics without premerged stage-2)
  - Example performance: `original` 154.67s, `snapshot` 114.05s, `resident` 75.71s.

### Image Generation Models

| Model Name                | HuggingFace Model ID                                     |
|:--------------------------|:---------------------------------------------------------|
| FLUX.1-dev                | `black-forest-labs/FLUX.1-dev`                           |
| FLUX.2-dev                | `black-forest-labs/FLUX.2-dev`                           |
| FLUX.2-dev-NVFP4          | `black-forest-labs/FLUX.2-dev-NVFP4`                     |
| FLUX.2-Klein-4B           | `black-forest-labs/FLUX.2-klein-4B`                      |
| FLUX.2-Klein-9B           | `black-forest-labs/FLUX.2-klein-9B`                      |
| Z-Image                   | `Tongyi-MAI/Z-Image`                                    |
| Z-Image-Turbo             | `Tongyi-MAI/Z-Image-Turbo`                              |
| GLM-Image                 | `zai-org/GLM-Image`                                     |
| Qwen Image                | `Qwen/Qwen-Image`                                       |
| Qwen Image 2512           | `Qwen/Qwen-Image-2512`                                  |
| Qwen Image Edit           | `Qwen/Qwen-Image-Edit`                                  |
| Qwen Image Edit 2509      | `Qwen/Qwen-Image-Edit-2509`                             |
| Qwen Image Edit 2511      | `Qwen/Qwen-Image-Edit-2511`                             |
| Qwen Image Layered        | `Qwen/Qwen-Image-Layered`                               |
| SD3 Medium                | `stabilityai/stable-diffusion-3-medium-diffusers`        |
| SD3.5 Medium              | `stabilityai/stable-diffusion-3.5-medium-diffusers`      |
| SD3.5 Large               | `stabilityai/stable-diffusion-3.5-large-diffusers`       |
| Hunyuan3D-2               | `tencent/Hunyuan3D-2`                                    |
| SANA 1.5 1.6B             | `Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers`   |
| SANA 1.5 4.8B             | `Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers`   |
| SANA 1600M 1024px         | `Efficient-Large-Model/Sana_1600M_1024px_diffusers`     |
| SANA 600M 1024px          | `Efficient-Large-Model/Sana_600M_1024px_diffusers`      |
| SANA 1600M 512px          | `Efficient-Large-Model/Sana_1600M_512px_diffusers`      |
| SANA 600M 512px           | `Efficient-Large-Model/Sana_600M_512px_diffusers`       |
| FireRed-Image-Edit 1.0    | `FireRedTeam/FireRed-Image-Edit-1.0`                     |
| FireRed-Image-Edit 1.1    | `FireRedTeam/FireRed-Image-Edit-1.1`                     |
| ERNIE-Image               | `baidu/ERNIE-Image`                                      |
| ERNIE-Image-Turbo         | `baidu/ERNIE-Image-Turbo`                                |

---

## Installation and Setup

### Standard Installation (NVIDIA GPUs)

#### Method 1: pip/uv

```bash
pip install --upgrade pip
pip install uv
uv pip install "sglang[diffusion]" --prerelease=allow
```

#### Method 2: From Source

```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang

pip install --upgrade pip
pip install -e "python[diffusion]"

# Or with uv:
uv pip install -e "python[diffusion]" --prerelease=allow
```

#### Method 3: Docker

Docker images are available at [lmsysorg/sglang](https://hub.docker.com/r/lmsysorg/sglang).
Replace `<secret>` with your HuggingFace Hub token.

```bash
docker run --gpus all \
    --shm-size 32g \
    -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=<secret>" \
    --ipc=host \
    lmsysorg/sglang:dev \
    zsh -c '\
        echo "Installing diffusion dependencies..." && \
        pip install -e "python[diffusion]" && \
        echo "Starting SGLang-Diffusion..." && \
        sglang generate \
            --model-path black-forest-labs/FLUX.1-dev \
            --prompt "A logo With Bold Large text: SGL Diffusion" \
            --save-output \
    '
```

### Platform-Specific Installations

#### ROCm (AMD GPUs)

For AMD Instinct GPUs (e.g., MI300X):

```bash
docker run --device=/dev/kfd --device=/dev/dri --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env HF_TOKEN=<secret> \
  lmsysorg/sglang:v0.5.5.post2-rocm700-mi30x \
  sglang generate --model-path black-forest-labs/FLUX.1-dev \
    --prompt "A logo With Bold Large text: SGL Diffusion" --save-output
```

#### MUSA (Moore Threads GPUs)

```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang
pip install --upgrade pip
rm -f python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml
pip install -e "python[all_musa]"
```

#### Intel XPU

Follow the XPU installation guide for the base environment, then:

```bash
pip install -e "python[diffusion]"
```

#### Ascend NPU

Follow the NPU installation guide. Quick test:

```bash
sglang generate --model-path black-forest-labs/FLUX.1-dev \
    --prompt "A logo With Bold Large text: SGL Diffusion" \
    --save-output
```

#### Apple MPS

```bash
brew install ffmpeg
brew install uv

git clone https://github.com/sgl-project/sglang.git
cd sglang

uv venv -p 3.11 sglang-diffusion
source sglang-diffusion/bin/activate

uv pip install --upgrade pip
rm -f python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml
uv pip install -e "python[all_mps]"
```

---

## Server Configuration

### Diffusion-Specific Server Arguments

SGLang Diffusion exposes a rich set of CLI arguments for both `sglang generate` and
`sglang serve`. Use `sglang generate --help` or `sglang serve --help` for the exhaustive
flag list.

#### Model and Runtime Arguments

| Argument                     | Description                                                        |
|------------------------------|--------------------------------------------------------------------|
| `--model-path {MODEL}`       | Model path or Hugging Face model ID                                |
| `--lora-path {PATH}`         | Load a LoRA adapter                                                |
| `--lora-nickname {NAME}`     | Nickname for the LoRA adapter                                      |
| `--num-gpus {N}`             | Number of GPUs to use                                              |
| `--tp-size {N}`              | Tensor parallelism size (mainly for encoders)                      |
| `--sp-degree {N}`            | Sequence parallelism size                                          |
| `--ulysses-degree {N}`       | Ulysses parallelism control                                        |
| `--ring-degree {N}`          | Ring attention parallelism control                                 |
| `--attention-backend {BACKEND}` | Attention backend for native SGLang pipelines                   |
| `--component-attention-backends {MAP}` | Per-component attention backend overrides, e.g. `text_encoder=torch_sdpa,transformer=fa` |
| `--attention-backend-config {CONFIG}` | Attention backend configuration (JSON/YAML file, JSON string, or key=value pairs) |

#### Sampling and Output Arguments

| Argument                     | Description                                                        |
|------------------------------|--------------------------------------------------------------------|
| `--prompt {PROMPT}`          | Text prompt for generation                                         |
| `--negative-prompt {PROMPT}` | Negative text prompt                                               |
| `--image-path {PATH} [...]`  | Input image(s) for I2V or I2I generation                           |
| `--num-inference-steps {N}`  | Number of denoising steps                                          |
| `--seed {SEED}`              | Random seed for reproducibility                                    |
| `--height {HEIGHT}`          | Output height in pixels                                            |
| `--width {WIDTH}`            | Output width in pixels                                             |
| `--num-frames {N}`           | Number of video frames                                             |
| `--fps {FPS}`                | Frames per second for video                                        |
| `--output-path {PATH}`       | Directory to save output files                                     |
| `--output-file-name {NAME}`  | Custom output filename                                             |
| `--save-output`              | Save generated output to disk                                      |
| `--return-frames`            | Return raw frames in response                                      |

#### Quantized Transformer Arguments

| Argument                       | Description                                                      |
|--------------------------------|------------------------------------------------------------------|
| `--transformer-path {PATH}`    | Quantized transformers-style transformer component directory     |
| `--transformer-weights-path`   | Quantized safetensors file, directory, or repo ID               |
| `--enable-svdquant`            | Enable SVDQuant (Nunchaku) quantization                          |
| `--quantization-precision`     | Quantization precision (`int4`, `nvfp4`)                        |
| `--quantization-rank`          | Quantization rank for SVDQuant                                   |

#### Backend and Diffusers Arguments

| Argument                      | Values                                | Description                               |
|-------------------------------|---------------------------------------|-------------------------------------------|
| `--backend`                   | `auto`, `sglang`, `diffusers`         | Choose backend                            |
| `--diffusers-attention-backend` | `flash`, `_flash_3_hub`, `sage`, `xformers`, `native` | Diffusers attention backend |
| `--trust-remote-code`         | flag                                  | Required for custom pipeline classes      |
| `--vae-tiling`                | flag                                  | Lower memory usage for VAE decode         |
| `--vae-slicing`               | flag                                  | Lower memory usage for VAE decode         |
| `--dit-precision`             | `fp16`, `bf16`, `fp32`                | DiT/transformer precision                 |
| `--vae-precision`             | `fp16`, `bf16`, `fp32`                | VAE precision                             |
| `--enable-torch-compile`      | flag                                  | Enable `torch.compile`                    |
| `--cache-dit-config`          | `{PATH}`                              | Cache-DiT configuration YAML/JSON         |

#### Component Path Overrides

Override individual pipeline components using `--<component>-path`:

| Component Type      | Supported Keys                                                                        |
|:-------------------|:--------------------------------------------------------------------------------------|
| VAE                | `--vae-path`, `--video-vae-path`, `--audio-vae-path`                                  |
| Transformer / DiT  | `--transformer-path`, `--transformer-2-path`, `--video-dit-path`, `--audio-dit-path`  |
| Text / Preprocess  | `--text-encoder-path`, `--text-encoder-2-path`, `--tokenizer-path`, `--processor-path`, `--image-processor-path` |
| Auxiliary          | `--scheduler-path`, `--spatial-upsampler-path`, `--vocoder-path`, `--connectors-path`, `--dual-tower-bridge-path`, `--image-encoder-path`, `--vision-language-encoder-path` |

Example:

```bash
sglang serve \
  --model-path black-forest-labs/FLUX.2-dev \
  --vae-path fal/FLUX.2-Tiny-AutoEncoder
```

#### Configuration Files

Load JSON or YAML configuration with `--config`. CLI flags override values from the config file.

```bash
sglang generate --config config.yaml
```

Example YAML configuration:

```yaml
model_path: FastVideo/FastHunyuan-diffusers
prompt: A beautiful woman in a red dress walking down a street
output_path: outputs/
num_gpus: 2
sp_size: 2
tp_size: 1
num_frames: 45
height: 720
width: 1280
num_inference_steps: 6
seed: 1024
fps: 24
precision: bf16
vae_precision: fp16
vae_tiling: true
vae_sp: true
enable_torch_compile: false
```

Config files also support component overrides:

```yaml
model_path: black-forest-labs/FLUX.2-dev
component_paths:
  vae: black-forest-labs/FLUX.2-small-decoder
  transformer: /models/flux2/transformer
```

#### Overlay Repos for Non-Diffusers Models

For supported non-diffusers source repos, SGLang resolves them through self-hosted overlay
repos. Override the built-in registry:

```bash
export SGLANG_DIFFUSION_MODEL_OVERLAY_REGISTRY='{
  "Wan-AI/Wan2.2-S2V-14B": {
    "overlay_repo_id": "your-org/Wan2.2-S2V-14B-overlay",
    "overlay_revision": "main"
  }
}'

sglang generate \
  --model-path Wan-AI/Wan2.2-S2V-14B \
  --config configs/wan_s2v.yaml
```

On first load, SGLang downloads overlay metadata and original source files, materializing
a local standard component repo under `~/.cache/sgl_diffusion/materialized_models/`.

---

## Usage Guide

### Text-to-Image Generation

```bash
sglang generate \
  --model-path Qwen/Qwen-Image \
  --prompt "A beautiful sunset over the mountains" \
  --save-output
```

With specific dimensions and steps:

```bash
sglang generate \
  --model-path black-forest-labs/FLUX.2-dev \
  --prompt "A calico cat playing a piano on stage" \
  --height 1024 \
  --width 1024 \
  --num-inference-steps 30 \
  --save-output \
  --output-path outputs/ \
  --output-file-name "cat_piano.png"
```

### Text-to-Video Generation

```bash
sglang generate \
  --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --text-encoder-cpu-offload \
  --pin-cpu-memory \
  --num-gpus 4 \
  --ulysses-degree 2 \
  --ring-degree 2 \
  --prompt "A curious raccoon" \
  --save-output \
  --output-path outputs \
  --output-file-name "a-curious-raccoon.mp4"
```

Multi-GPU video generation with specific resolution:

```bash
sglang generate \
  --model-path hunyuanvideo-community/HunyuanVideo \
  --transformer-path lmsys/hunyuanvideo-modelopt-fp8-sglang-transformer \
  --height 544 --width 960 --num-frames 17 \
  --prompt "A cinematic shot of a red sports car driving through rain at night" \
  --save-output
```

### Image-to-Image / Image-to-Video

Image-to-video with an input image:

```bash
sglang generate \
  --model-path Wan-AI/Wan2.2-I2V-A14B-Diffusers \
  --image-path /path/to/input.png \
  --prompt "A fox walking through a snowy forest" \
  --save-output
```

Image editing:

```bash
sglang generate \
  --model-path Qwen/Qwen-Image-Edit-2511 \
  --image-path /path/to/input.png \
  --prompt "Turn the scene into a warm watercolor illustration" \
  --save-output
```

### Batch Generation via Server

Start the server:

```bash
sglang serve \
  --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --text-encoder-cpu-offload \
  --pin-cpu-memory \
  --num-gpus 4 \
  --ulysses-degree=2 \
  --ring-degree=2 \
  --port 30010
```

Then send multiple requests to the HTTP server (see API Reference below).

### Diffusers Backend

Force vanilla diffusers pipelines when no native SGLang implementation exists:

```bash
sglang generate \
  --model-path AIDC-AI/Ovis-Image-7B \
  --backend diffusers \
  --trust-remote-code \
  --diffusers-attention-backend flash \
  --prompt "A serene Japanese garden with cherry blossoms" \
  --height 1024 \
  --width 1024 \
  --num-inference-steps 30 \
  --save-output
```

---

## API Reference

The SGLang diffusion HTTP server implements an OpenAI-compatible API for image and video
generation, plus LoRA adapter management. Requires Python 3.11+ for the OpenAI Python SDK.

### Get Model Information

**Endpoint:** `GET /models`

```bash
curl -sS -X GET "http://localhost:30010/models"
```

Response:

```json
{
  "model_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
  "task_type": "T2V",
  "pipeline_name": "wan_pipeline",
  "pipeline_class": "WanPipeline",
  "num_gpus": 4,
  "dit_precision": "bf16",
  "vae_precision": "fp16"
}
```

### Image Generation

#### Create an Image

**Endpoint:** `POST /v1/images/generations`

**Parameters:**

| Parameter          | Type    | Description                                      |
|--------------------|---------|--------------------------------------------------|
| `prompt`           | string  | Text prompt for image generation                  |
| `size`             | string  | Image size as `WIDTHxHEIGHT` (e.g., `1024x1024`) |
| `n`                | integer | Number of images to generate                      |
| `response_format`  | string  | `b64_json` or `url`                               |
| `seed`             | integer | Random seed for reproducibility                   |
| `num_inference_steps` | integer | Number of denoising steps                      |

**Python SDK Example:**

```python
import base64
from openai import OpenAI

client = OpenAI(api_key="sk-proj-1234567890", base_url="http://localhost:30010/v1")

img = client.images.generate(
    prompt="A calico cat playing a piano on stage",
    size="1024x1024",
    n=1,
    response_format="b64_json",
)

image_bytes = base64.b64decode(img.data[0].b64_json)
with open("output.png", "wb") as f:
    f.write(image_bytes)
```

**Curl Example:**

```bash
curl -sS -X POST "http://localhost:30010/v1/images/generations" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-proj-1234567890" \
  -d '{
        "prompt": "A calico cat playing a piano on stage",
        "size": "1024x1024",
        "n": 1,
        "response_format": "b64_json"
      }'
```

#### Edit an Image

**Endpoint:** `POST /v1/images/edits`

Accepts multipart form upload with input images and a text prompt.

**Parameters:**

| Parameter          | Type    | Description                                  |
|--------------------|---------|----------------------------------------------|
| `image`            | file    | Input image file (multipart upload)           |
| `url`              | string  | URL to an input image                         |
| `prompt`           | string  | Text prompt describing the edit               |
| `size`             | string  | Output size as `WIDTHxHEIGHT`                 |
| `response_format`  | string  | `b64_json` or `url`                           |

```bash
curl -sS -X POST "http://localhost:30010/v1/images/edits" \
  -H "Authorization: Bearer sk-proj-1234567890" \
  -F "image=@local_input_image.png" \
  -F "prompt=A calico cat playing a piano on stage" \
  -F "size=1024x1024" \
  -F "response_format=b64_json"
```

#### Download Image Content

**Endpoint:** `GET /v1/images/{image_id}/content`

When `response_format=url` is used, the API returns a relative URL. Use this endpoint to
download the actual content.

```bash
curl -sS -L "http://localhost:30010/v1/images/<IMAGE_ID>/content" \
  -H "Authorization: Bearer sk-proj-1234567890" \
  -o output.png
```

### Video Generation

#### Create a Video (Text-to-Video)

**Endpoint:** `POST /v1/videos`

**Parameters:**

| Parameter          | Type    | Description                                      |
|--------------------|---------|--------------------------------------------------|
| `prompt`           | string  | Text prompt for video generation                  |
| `size`             | string  | Video size as `WIDTHxHEIGHT` (e.g., `1280x720`)  |
| `num_frames`       | integer | Number of frames to generate                      |
| `fps`              | integer | Frames per second                                 |
| `seed`             | integer | Random seed                                       |

**Python SDK Example:**

```python
from openai import OpenAI

client = OpenAI(api_key="sk-proj-1234567890", base_url="http://localhost:30010/v1")

video = client.videos.create(
    prompt="A calico cat playing a piano on stage",
    size="1280x720"
)
print(f"Video ID: {video.id}, Status: {video.status}")
```

**Curl Example:**

```bash
curl -sS -X POST "http://localhost:30010/v1/videos" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-proj-1234567890" \
  -d '{
        "prompt": "A calico cat playing a piano on stage",
        "size": "1280x720"
      }'
```

#### Create a Video (Image-to-Video)

For I2V or TI2V models, pass an input image via multipart upload or reference URL.

**Multipart upload:**

```bash
curl -sS -X POST "http://localhost:30010/v1/videos" \
  -H "Authorization: Bearer sk-proj-1234567890" \
  -F "prompt=A cat playing a piano" \
  -F "input_reference=@input_image.png" \
  -F "size=1280x720"
```

**Reference URL:**

```bash
curl -sS -X POST "http://localhost:30010/v1/videos" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-proj-1234567890" \
  -d '{
        "prompt": "A cat playing a piano",
        "reference_url": "https://example.com/input_image.png",
        "size": "1280x720"
      }'
```

#### List Videos

**Endpoint:** `GET /v1/videos`

```python
videos = client.videos.list()
for item in videos.data:
    print(item.id, item.status)
```

#### Download Video Content

**Endpoint:** `GET /v1/videos/{video_id}/content`

```python
import time

# Poll for completion
while True:
    page = client.videos.list()
    item = next((v for v in page.data if v.id == video_id), None)
    if item and item.status == "completed":
        break
    time.sleep(5)

# Download content
resp = client.videos.download_content(video_id=video_id)
with open("output.mp4", "wb") as f:
    f.write(resp.read())
```

### LoRA Management

#### Set LoRA Adapter

**Endpoint:** `POST /v1/set_lora`

Loads and merges LoRA adapter(s). Supports single or multiple LoRAs.

**Parameters:**

| Parameter        | Type                      | Description                                                |
|------------------|---------------------------|------------------------------------------------------------|
| `lora_nickname`  | string or list of strings | Unique identifier for the LoRA adapter(s)                  |
| `lora_path`      | string or list of strings | Path to `.safetensors` file(s) or HF repo ID(s)            |
| `target`         | string or list of strings | Which transformer(s) to apply: `all`, `transformer`, `transformer_2`, `critic` |
| `strength`       | float or list of floats   | LoRA merge strength (default 1.0)                          |

**Single LoRA:**

```bash
curl -X POST http://localhost:30010/v1/set_lora \
  -H "Content-Type: application/json" \
  -d '{
        "lora_nickname": "lora_name",
        "lora_path": "/path/to/lora.safetensors",
        "target": "all",
        "strength": 0.8
      }'
```

**Multiple LoRAs:**

```bash
curl -X POST http://localhost:30010/v1/set_lora \
  -H "Content-Type: application/json" \
  -d '{
        "lora_nickname": ["lora_1", "lora_2"],
        "lora_path": ["/path/to/lora1.safetensors", "/path/to/lora2.safetensors"],
        "target": ["transformer", "transformer_2"],
        "strength": [0.8, 1.0]
      }'
```

#### Merge LoRA Weights

**Endpoint:** `POST /v1/merge_lora_weights`

Manually merges the currently set LoRA weights. Typically only needed after an unmerge.

```bash
curl -X POST http://localhost:30010/v1/merge_lora_weights \
  -H "Content-Type: application/json" \
  -d '{"strength": 0.8}'
```

#### Unmerge LoRA Weights

**Endpoint:** `POST /v1/unmerge_lora_weights`

Restores the model to its original state. Must be called before setting a different LoRA.

```bash
curl -X POST http://localhost:30010/v1/unmerge_lora_weights \
  -H "Content-Type: application/json"
```

#### List LoRA Adapters

**Endpoint:** `GET /v1/list_loras`

```bash
curl -sS -X GET "http://localhost:30010/v1/list_loras"
```

Response:

```json
{
  "loaded_adapters": [
    { "nickname": "lora_a", "path": "/weights/lora_a.safetensors" },
    { "nickname": "lora_b", "path": "/weights/lora_b.safetensors" }
  ],
  "active": {
    "transformer": [
      {
        "nickname": "lora2",
        "path": "tarn59/pixel_art_style_lora_z_image_turbo",
        "merged": true,
        "strength": 1.0
      }
    ]
  }
}
```

#### Switching LoRAs Workflow

```bash
# 1. Set LoRA A
curl -X POST http://localhost:30010/v1/set_lora -d '{"lora_nickname": "lora_a", "lora_path": "path/to/A"}'
# 2. Generate with LoRA A...
# 3. Unmerge LoRA A
curl -X POST http://localhost:30010/v1/unmerge_lora_weights
# 4. Set LoRA B
curl -X POST http://localhost:30010/v1/set_lora -d '{"lora_nickname": "lora_b", "lora_path": "path/to/B"}'
# 5. Generate with LoRA B...
```

### Output Quality Parameters

| Parameter             | Type    | Values | Description                                   |
|-----------------------|---------|--------|-----------------------------------------------|
| `output-quality`      | string  | `maximum`, `high`, `medium`, `low`, `default` | Preset quality level            |
| `output-compression`  | integer | 0-100  | Direct compression override (takes precedence) |

Quality presets map to compression values: `maximum`=100, `high`=90, `medium`=55, `low`=35,
`default`=50 for video / 75 for image.

---

## Quantization Support

SGLang Diffusion supports quantized transformer checkpoints. In most cases, keep the base
model and the quantized transformer override separate.

### Quick Reference Paths

| Flag                        | Purpose                                                       |
|-----------------------------|---------------------------------------------------------------|
| `--model-path`              | The base or original model                                    |
| `--transformer-path`        | Quantized transformers-style transformer directory with `config.json` |
| `--transformer-weights-path`| Quantized safetensors file, sharded directory, or HF repo ID  |

### Quantization Families

| quant_family      | Checkpoint Form                                                                 | Canonical CLI                      | Supported Models              | Extra Dependency | Notes                                              |
|-------------------|---------------------------------------------------------------------------------|------------------------------------|-------------------------------|------------------|-----------------------------------------------------|
| `fp8`             | Quantized transformer component folder, or safetensors with quantization_config | `--transformer-path` or `--transformer-weights-path` | ALL            | None             | Component-folder and single-file flows supported    |
| `modelopt-fp8`    | Converted ModelOpt FP8 transformer directory or repo with `config.json`         | `--transformer-path`               | FLUX.1/2, Wan2.2, HunyuanVideo, Qwen Image | None | `dit_layerwise_offload` supported; `dit_cpu_offload` disabled |
| `modelopt-nvfp4`  | Mixed transformer directory/repo, or raw NVFP4 safetensors                      | `--transformer-path` or `--transformer-weights-path` | FLUX.1/2, Wan2.2 | None | Mixed overrides keep base separate; raw exports use weights-path flow |
| `nunchaku-svdq`   | Pre-quantized Nunchaku weights, named `svdq-{int4|fp4}_r{rank}-...`             | `--transformer-weights-path`       | Qwen-Image, FLUX, Z-Image     | `nunchaku`       | SGLang infers precision and rank from filename      |
| `msmodelslim`     | Pre-quantized msmodelslim transformer weights                                   | `--model-path`                     | Wan2.2 family                 | None             | Ascend NPU only; supports `w8a8` and `w4a4`         |

### ModelOpt FP8 Usage

#### Validated Checkpoints

| Quant Algo | Base Model | HF Repo | Notes |
|:-----------|:-----------|:--------|:------|
| FP8 | FLUX.1-dev | `lmsys/flux1-dev-modelopt-fp8-sglang-transformer` | Validated BF16 fallback for modulation and FF projection layers |
| FP8 | FLUX.2-dev | `lmsys/flux2-dev-modelopt-fp8-sglang-transformer` | Published SGLang-ready transformer override |
| FP8 | Wan2.2-T2V-A14B | `lmsys/wan22-t2v-a14b-modelopt-fp8-sglang-transformer` | Primary `transformer` quantized, `transformer_2` kept BF16 |
| FP8 | HunyuanVideo | `lmsys/hunyuanvideo-modelopt-fp8-sglang-transformer` | Converter maps diffusers module names to runtime names |
| FP8 | Qwen-Image | `lmsys/qwen-image-modelopt-fp8-sglang-transformer` | BF16 fallback for `img_in`, `txt_in`, timestep embedder, `norm_out`, `proj_out`, etc. |
| FP8 | Qwen-Image-Edit-2511 | `lmsys/qwen-image-edit-modelopt-fp8-sglang-transformer` | Shares `QwenImageTransformer2DModel` with Qwen Image |
| NVFP4 | FLUX.1-dev | `lmsys/flux1-dev-modelopt-nvfp4-sglang-transformer` | Mixed BF16+NVFP4; validated builder |
| NVFP4 | FLUX.2-dev | `black-forest-labs/FLUX.2-dev-NVFP4` | Official raw export repo |
| NVFP4 | Wan2.2-T2V-A14B | `lmsys/wan22-t2v-a14b-modelopt-nvfp4-sglang-transformer` | Primary transformer quantized; uses `SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND=cudnn` on Blackwell |

#### FP8 Examples

```bash
# FLUX.2 with FP8
sglang generate \
  --model-path black-forest-labs/FLUX.2-dev \
  --transformer-path lmsys/flux2-dev-modelopt-fp8-sglang-transformer \
  --prompt "A Logo With Bold Large Text: SGL Diffusion" \
  --save-output

# Wan2.2 with FP8
sglang generate \
  --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --transformer-path lmsys/wan22-t2v-a14b-modelopt-fp8-sglang-transformer \
  --prompt "a fox walking through neon rain" \
  --save-output

# HunyuanVideo with FP8
sglang generate \
  --model-path hunyuanvideo-community/HunyuanVideo \
  --transformer-path lmsys/hunyuanvideo-modelopt-fp8-sglang-transformer \
  --height 544 --width 960 --num-frames 17 \
  --prompt "A cinematic shot of a red sports car" \
  --save-output

# Qwen-Image with FP8
sglang generate \
  --model-path Qwen/Qwen-Image \
  --transformer-path lmsys/qwen-image-modelopt-fp8-sglang-transformer \
  --prompt "A tiny astronaut reading a book" \
  --save-output

# Qwen-Image-Edit with FP8
sglang generate \
  --model-path Qwen/Qwen-Image-Edit-2511 \
  --transformer-path lmsys/qwen-image-edit-modelopt-fp8-sglang-transformer \
  --image-path /path/to/input.png \
  --prompt "Turn the scene into a warm watercolor illustration" \
  --save-output
```

#### NVFP4 Examples

```bash
# FLUX.1 with NVFP4 mixed override
sglang generate \
  --model-path black-forest-labs/FLUX.1-dev \
  --transformer-path lmsys/flux1-dev-modelopt-nvfp4-sglang-transformer \
  --prompt "A Logo With Bold Large Text: SGL Diffusion" \
  --save-output

# FLUX.2 with raw NVFP4 export
sglang generate \
  --model-path black-forest-labs/FLUX.2-dev \
  --transformer-weights-path black-forest-labs/FLUX.2-dev-NVFP4 \
  --prompt "A Logo With Bold Large Text: SGL Diffusion" \
  --save-output

# Wan2.2 with NVFP4 on Blackwell
SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND=cudnn \
sglang generate \
  --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --transformer-path lmsys/wan22-t2v-a14b-modelopt-nvfp4-sglang-transformer \
  --prompt "a fox walking through neon rain" \
  --save-output
```

#### FP8 Notes

- `--transformer-path` is the canonical flag for converted ModelOpt FP8 transformer repos
  that already carry `config.json`.
- `dit_layerwise_offload` is supported for ModelOpt FP8 checkpoints; `dit_cpu_offload`
  stays disabled.
- The HunyuanVideo converter maps diffusers module names to SGLang runtime module names
  for fused QKV and fused QKV+MLP layers.
- Qwen Image FP8 conversion should write explicit BF16 fallback tensors before honoring
  ModelOpt ignored weights.
- To build converted checkpoints from a ModelOpt diffusers export:
  `python -m sglang.multimodal_gen.tools.build_modelopt_fp8_transformer`

### Nunchaku (SVDQuant)

#### Installation

```bash
pip install nunchaku
```

#### File Naming and Auto-Detection

If the basename of `--transformer-weights-path` contains the pattern `svdq-(int4|fp4)_r{rank}`,
SGLang automatically enables SVDQuant and infers precision and rank.

| Checkpoint Name Fragment | Inferred Precision | Inferred Rank |
|--------------------------|--------------------|---------------|
| `svdq-int4_r32`          | `int4`             | `32`          |
| `svdq-int4_r128`         | `int4`             | `128`         |
| `svdq-fp4_r32`           | `nvfp4`            | `32`          |
| `svdq-fp4_r128`          | `nvfp4`            | `128`         |

#### Usage

```bash
# Auto-detected flow
sglang generate \
  --model-path Qwen/Qwen-Image \
  --transformer-weights-path /path/to/svdq-int4_r32-qwen-image.safetensors \
  --prompt "a beautiful sunset" \
  --save-output

# Manual override
sglang generate \
  --model-path Qwen/Qwen-Image \
  --transformer-weights-path /path/to/custom_nunchaku_checkpoint.safetensors \
  --enable-svdquant \
  --quantization-precision int4 \
  --quantization-rank 128 \
  --prompt "a beautiful sunset" \
  --save-output
```

**Note**: Nunchaku runtime is currently validated only on NVIDIA CUDA Ampere (SM8x) or
SM12x GPUs. Hopper (SM90) is currently rejected.

### msmodelslim (Ascend NPU)

MindStudio-ModelSlim is a quantization compression tool optimized for Ascend hardware.

```bash
# Install
git clone https://gitcode.com/Ascend/msmodelslim.git
cd msmodelslim && bash install.sh

# Quantize
msmodelslim quant \
  --model_path /path/to/wan2_2_float_weights \
  --save_path /path/to/wan2_2_quantized_weights \
  --device npu --model_type Wan2_2 --quant_type w8a8 \
  --trust_remote_code True

# Run with auto-detected quantization
sglang generate \
  --model-path Eco-Tech/Wan2.2-T2V-A14B-Diffusers-w8a8 \
  --prompt "a beautiful sunset" \
  --save-output
```

Available quantization methods:
- `W4A4_DYNAMIC` -- linear with online quantization of activations
- `W8A8` -- linear with offline quantization of activations
- `W8A8_DYNAMIC` -- linear with online quantization of activations
- `mxfp8` -- in progress

---

## Disaggregation

Split the monolithic pipeline into independent Encoder, Denoiser, and Decoder roles, each
running on its own GPU(s). A central DiffusionServer routes requests through the pipeline.

### Role Definitions

| `--disagg-role` | What It Runs                                                    |
|-----------------|-----------------------------------------------------------------|
| `monolithic`    | (Default) Standard single-server mode                           |
| `encoder`       | InputValidation, TextEncoding, ImageEncoding/VAEEncoding, LatentPreparation, TimestepPreparation, model-specific before-denoising stages |
| `denoiser`      | `DenoisingStage` and subclasses (the DiT forward loop + scheduler) |
| `decoder`       | `DecodingStage` (VAE decode) and subclasses                     |
| `server`        | DiffusionServer head node + HTTP server (no GPU)                |

Each stage declares its role via the `role_affinity` property on `PipelineStage`.

### Single-Machine Example

Tested on 8xH200 with `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`:

```bash
# Terminal 1: Encoder (GPU 0)
sglang serve --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --disagg-role encoder \
    --disagg-server-addr tcp://127.0.0.1:19655 \
    --scheduler-port 19000 \
    --num-gpus 1 --base-gpu-id 0

# Terminal 2: Denoiser (GPU 1)
sglang serve --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --disagg-role denoiser \
    --disagg-server-addr tcp://127.0.0.1:19655 \
    --scheduler-port 19001 \
    --num-gpus 1 --base-gpu-id 1

# Terminal 3: Decoder (GPU 2)
sglang serve --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --disagg-role decoder \
    --disagg-server-addr tcp://127.0.0.1:19655 \
    --scheduler-port 19002 \
    --num-gpus 1 --base-gpu-id 2

# Terminal 4: DiffusionServer head (no GPU)
sglang serve --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --disagg-role server \
    --encoder-urls  "tcp://127.0.0.1:19000" \
    --denoiser-urls "tcp://127.0.0.1:19001" \
    --decoder-urls  "tcp://127.0.0.1:19002" \
    --host 0.0.0.0 --port 22000 \
    --scheduler-port 19655

# Send request
curl http://127.0.0.1:22000/v1/videos \
    -H "Content-Type: application/json" \
    -d '{"model": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "prompt": "A curious raccoon exploring a garden, cinematic", "size": "832x480"}'
```

Tested result (8xH200): Encoder 2.3s -> Denoiser 312.8s (50 steps) -> Decoder 7.1s.
Total ~322s for 81-frame 1024x1024 video.

### Multi-Machine Example

Replace `127.0.0.1` with actual IPs and add RDMA flags:

```bash
# Machine A (10.0.0.1): Encoder
sglang serve --model-path Wan-AI/Wan2.1-T2V-14B-Diffusers \
    --disagg-role encoder \
    --disagg-server-addr tcp://10.0.0.4:19655 \
    --scheduler-port 19000 \
    --num-gpus 1 \
    --disagg-p2p-hostname 10.0.0.1 --disagg-ib-device mlx5_0

# Machine B (10.0.0.2): Denoiser (4 GPUs with SP)
sglang serve --model-path Wan-AI/Wan2.1-T2V-14B-Diffusers \
    --disagg-role denoiser \
    --disagg-server-addr tcp://10.0.0.4:19655 \
    --scheduler-port 19001 \
    --num-gpus 4 --denoiser-sp 4 --denoiser-ulysses 2 --denoiser-ring 2 \
    --disagg-p2p-hostname 10.0.0.2 --disagg-ib-device mlx5_0

# Machine C (10.0.0.3): Decoder
sglang serve --model-path Wan-AI/Wan2.1-T2V-14B-Diffusers \
    --disagg-role decoder \
    --disagg-server-addr tcp://10.0.0.4:19655 \
    --scheduler-port 19002 \
    --num-gpus 1 \
    --disagg-p2p-hostname 10.0.0.3 --disagg-ib-device mlx5_0

# Machine D (10.0.0.4): DiffusionServer head
sglang serve --model-path Wan-AI/Wan2.1-T2V-14B-Diffusers \
    --disagg-role server \
    --encoder-urls  "tcp://10.0.0.1:19000" \
    --denoiser-urls "tcp://10.0.0.2:19001" \
    --decoder-urls  "tcp://10.0.0.3:19002" \
    --host 0.0.0.0 --port 30000 \
    --scheduler-port 19655 \
    --disagg-dispatch-policy max_free_slots
```

ZMQ handles startup order gracefully -- instances and head can start in any order.

### Multiple Instances per Role

Use semicolons in `--*-urls` to register multiple instances:

```bash
sglang serve --model-path ... --disagg-role server \
    --encoder-urls  "tcp://10.0.0.1:35000;tcp://10.0.0.2:35000" \
    --denoiser-urls "tcp://10.0.0.3:35000;tcp://10.0.0.4:35000" \
    --decoder-urls  "tcp://10.0.0.5:35000"
```

### Transfer Mechanism

Tensor data flows directly between instances via mooncake-transfer-engine (RDMA):

```bash
pip install mooncake-transfer-engine
```

**Transfer Flow:**

1. Sender stages tensors (async copy to transfer buffer, overlapped with metadata serialization).
2. Sender sends `transfer_staged` control message to DiffusionServer (metadata only).
3. DiffusionServer sends `transfer_alloc` to receiver; receiver allocates buffer and replies.
4. DiffusionServer sends `transfer_push` to receiver with sender's address info.
5. Receiver pulls data via transfer engine, sends `transfer_ready`.
6. Receiver loads tensors async on a dedicated transfer stream.

### Port Convention

Derived from `--scheduler-port` (default: 5555):

| Socket                    | Port                  |
|---------------------------|-----------------------|
| DS frontend (ROUTER)      | `scheduler_port`      |
| Encoder result (PULL)     | `scheduler_port + 1`  |
| Denoiser result (PULL)    | `scheduler_port + 2`  |
| Decoder result (PULL)     | `scheduler_port + 3`  |

### Per-Role Parallelism

| Flag                                     | Description                       |
|------------------------------------------|-----------------------------------|
| `--encoder-tp`                           | Encoder tensor parallelism        |
| `--denoiser-tp` / `--denoiser-sp` / `--denoiser-ulysses` / `--denoiser-ring` | Denoiser parallelism |
| `--decoder-tp`                           | Decoder tensor parallelism        |

### RDMA Flags

| Flag                        | Default     | Description                              |
|-----------------------------|-------------|------------------------------------------|
| `--disagg-p2p-hostname`     | `127.0.0.1` | RDMA-reachable hostname/IP of instance   |
| `--disagg-ib-device`        | None        | InfiniBand device (e.g., `mlx5_0`)       |
| `--disagg-transfer-pool-size` | 256 MiB   | Pinned memory pool per instance          |

### Other Disaggregation Options

| Flag                      | Default       | Description                              |
|---------------------------|---------------|------------------------------------------|
| `--disagg-timeout`        | `600`         | Timeout (seconds) for pending requests   |
| `--disagg-dispatch-policy`| `round_robin` | `round_robin` or `max_free_slots`        |

### Python API

```python
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.launch_server import launch_pool_disagg_server

server_args = ServerArgs.from_kwargs(
    model_path="Wan-AI/Wan2.1-T2V-14B-Diffusers",
    denoiser_sp=4, denoiser_ulysses=2, denoiser_ring=2,
    disagg_ib_device="mlx5_0",
)

launch_pool_disagg_server(
    server_args,
    encoder_gpus=[[0]],
    denoiser_gpus=[[1, 2, 3, 4], [5, 6, 7, 8]],
    decoder_gpus=[[0]],
)
```

### Disaggregation Architecture

```
Client --- HTTP (port 30000) --> FastAPI Server
                                      |
                                      v
                              DiffusionServer (ROUTER, scheduler_port)
                              +-------|-------+
                   PUSH work  |       |       |  PUSH work
                              v       |       v
                    Encoder[0..N]     |    Decoder[0..K]
                              |       |       ^
                   P2P tensor |       |       | P2P tensor
                   transfer   v       |       | transfer
                          Denoiser[0..M] -----+
                                      |
                    PULL results <-----+  (decoder -> DS -> client)
```

### Request State Machine

```
PENDING -> ENCODER_WAITING -> ENCODER_RUNNING -> ENCODER_DONE
                                                      |
                          DENOISING_WAITING -> DENOISING_RUNNING -> DENOISING_DONE
                                                                         |
                                  DECODER_WAITING -> DECODER_RUNNING -> DONE
```

Any state can transition to `FAILED` or `TIMED_OUT`.

---

## Performance Optimization

### Overview

| Optimization         | Type       | Description                                                  |
|----------------------|------------|--------------------------------------------------------------|
| Cache-DiT            | Caching    | Block-level caching with DBCache, TaylorSeer, and SCM       |
| TeaCache             | Caching    | Timestep-level caching based on temporal similarity          |
| Attention Backends   | Kernel     | Optimized attention implementations (FA, SageAttention, etc.) |
| sgl-kernel           | Kernel     | Precompiled optimized operators                              |
| JIT Kernels          | Kernel     | Just-in-time compiled kernels for key inference paths        |
| torch.compile        | Compiler   | PyTorch compilation for fused operations                     |
| Profiling            | Diagnostic | PyTorch Profiler and Nsight Systems integration              |

### Profiling

#### PyTorch Profiler

Profile the denoising stage:

```bash
sglang generate \
  --model-path Qwen/Qwen-Image \
  --prompt "A Logo With Bold Large Text: SGL Diffusion" \
  --seed 0 \
  --profile
```

Profile all pipeline stages:

```bash
sglang generate \
  --model-path Qwen/Qwen-Image \
  --prompt "A Logo With Bold Large Text: SGL Diffusion" \
  --seed 0 \
  --profile \
  --profile-all-stages
```

**Parameters:**

| Argument                  | Description                                          |
|---------------------------|------------------------------------------------------|
| `--profile`               | Enable profiling for the denoising stage             |
| `--num-profiled-timesteps N` | Number of timesteps to profile (default: 5)       |
| `--profile-all-stages`    | Profile all pipeline stages                          |

Trace files are saved to `./logs/` and can be viewed at https://ui.perfetto.dev/.

#### Stage/Step Timing Dump

```bash
sglang generate \
  --model-path <MODEL_PATH_OR_ID> \
  --prompt "<PROMPT>" \
  --perf-dump-path perf.json
```

The JSON output contains stage-level timing breakdown and per-step denoising timing.

#### Nsight Systems

```bash
nsys profile \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --force-overwrite=true \
  -o QwenImage \
  sglang generate \
    --model-path Qwen/Qwen-Image \
    --prompt "A Logo With Bold Large Text: SGL Diffusion" \
    --seed 0
```

Targeted stage profiling with delay/duration:

```bash
nsys profile \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --force-overwrite=true \
  --delay 10 \
  --duration 30 \
  -o QwenImage_denoising \
  sglang generate \
    --model-path Qwen/Qwen-Image \
    --prompt "A Logo With Bold Large Text: SGL Diffusion" \
    --seed 0
```

### Ring SP Performance Benchmark

Benchmark of `Wan2.2-TI2V-5B-Diffusers` on 2x RTX 40-series (48GB):

**Stage Time Breakdown:**

| Stage              | u1r2 (s) | u1r1 Baseline (s) | Speedup |
|--------------------|----------|--------------------|---------|
| InputValidation    | 0.1060   | 0.1029             | 0.97x   |
| TextEncoding       | 1.3965   | 2.2261             | 1.59x   |
| LatentPreparation  | 0.0002   | 0.0002             | 1.00x   |
| TimestepPreparation| 0.0003   | 0.0004             | 1.33x   |
| Denoising          | 52.6358  | 71.6785            | 1.36x   |
| Decoding           | 7.6708   | 13.4314            | 1.75x   |
| **Total**          | **63.74**| **90.63**          | **1.42x**|

**Memory Usage:**

| Memory Metric     | u1r2 (GB) | u1r1 Baseline (GB) | Delta  |
|-------------------|-----------|--------------------|--------|
| Peak GPU Memory   | 20.07     | 27.40              | -7.33  |
| Peak Allocated    | 13.35     | 20.40              | -7.05  |

---

## Caching Acceleration

SGLang provides two complementary caching strategies for DiT models:

| Strategy   | Scope          | Mechanism                                   | Best For              |
|------------|----------------|---------------------------------------------|-----------------------|
| Cache-DiT  | Block-level    | Skip individual transformer blocks           | Advanced, higher speedup |
| TeaCache   | Timestep-level | Skip entire denoising steps via L1 similarity | Simple, built-in      |

### Cache-DiT

Integrates [Cache-DiT](https://github.com/vipshop/cache-dit) for up to **1.69x speedup**.

#### Quick Start

```bash
SGLANG_CACHE_DIT_ENABLED=true \
sglang generate --model-path Qwen/Qwen-Image \
    --prompt "A beautiful sunset over the mountains"
```

#### Diffusers Backend with Config File

Define `cache.yaml`:

```yaml
cache_config:
  max_warmup_steps: 8
  warmup_interval: 2
  max_cached_steps: -1
  max_continuous_cached_steps: 2
  Fn_compute_blocks: 1
  Bn_compute_blocks: 0
  residual_diff_threshold: 0.12
  enable_taylorseer: true
  taylorseer_order: 1
```

Apply:

```bash
sglang generate \
  --backend diffusers \
  --model-path Qwen/Qwen-Image \
  --cache-dit-config cache.yaml \
  --prompt "A beautiful sunset over the mountains"
```

#### DBCache Parameters

| Parameter | Environment Variable        | Default | Description                              |
|-----------|-----------------------------|---------|------------------------------------------|
| Fn        | `SGLANG_CACHE_DIT_FN`       | 1       | Number of first blocks to always compute |
| Bn        | `SGLANG_CACHE_DIT_BN`       | 0       | Number of last blocks to always compute  |
| W         | `SGLANG_CACHE_DIT_WARMUP`   | 4       | Warmup steps before caching starts       |
| R         | `SGLANG_CACHE_DIT_RDT`      | 0.24    | Residual difference threshold            |
| MC        | `SGLANG_CACHE_DIT_MC`       | 3       | Maximum continuous cached steps          |

#### TaylorSeer Configuration

| Parameter | Environment Variable            | Default | Description                    |
|-----------|---------------------------------|---------|--------------------------------|
| Enable    | `SGLANG_CACHE_DIT_TAYLORSEER`   | false   | Enable TaylorSeer calibrator   |
| Order     | `SGLANG_CACHE_DIT_TS_ORDER`     | 1       | Taylor expansion order (1 or 2)|

#### SCM (Step Computation Masking) Presets

| Preset   | Compute Ratio | Speed    | Quality    |
|----------|---------------|----------|------------|
| `none`   | 100%          | Baseline | Best       |
| `slow`   | ~75%          | ~1.3x    | High       |
| `medium` | ~50%          | ~2x      | Good       |
| `fast`   | ~35%          | ~3x      | Acceptable |
| `ultra`  | ~25%          | ~4x      | Lower      |

#### Combined Configuration Example

```bash
SGLANG_CACHE_DIT_ENABLED=true \
SGLANG_CACHE_DIT_FN=2 \
SGLANG_CACHE_DIT_BN=1 \
SGLANG_CACHE_DIT_WARMUP=4 \
SGLANG_CACHE_DIT_RDT=0.4 \
SGLANG_CACHE_DIT_MC=4 \
SGLANG_CACHE_DIT_TAYLORSEER=true \
SGLANG_CACHE_DIT_TS_ORDER=2 \
sglang generate --model-path black-forest-labs/FLUX.1-dev \
    --prompt "A curious raccoon in a forest"
```

#### Distributed Inference with Cache-DiT

Define `parallel.yaml`:

```yaml
parallelism_config:
  ulysses_size: auto
  attention_backend: native
```

```bash
sglang generate \
  --backend diffusers \
  --num-gpus 4 \
  --model-path Qwen/Qwen-Image \
  --cache-dit-config parallel.yaml \
  --prompt "A futuristic cityscape at sunset"
```

2D, 3D parallelism and hybrid cache+parallel configs are also supported through the YAML
configuration. See Cache-DiT documentation for full details.

#### Supported Models for Cache-DiT

| Model Family | Example Models              |
|--------------|-----------------------------|
| Wan          | Wan2.1, Wan2.2              |
| Flux         | FLUX.1-dev, FLUX.2-dev      |
| Z-Image      | Z-Image-Turbo               |
| Qwen         | Qwen-Image, Qwen-Image-Edit |
| Hunyuan      | HunyuanVideo                |

#### Cache-DiT Limitations

- SGLang-native pipelines: distributed support (TP/SP) not yet validated; Cache-DiT is
  auto-disabled when `world_size > 1`.
- SCM requires >= 8 inference steps.
- Only models registered in Cache-DiT's BlockAdapterRegister are supported.

### TeaCache

TeaCache accelerates diffusion inference by detecting when consecutive denoising steps are
similar enough to skip computation entirely. It tracks L1 distance between modulated inputs
across timesteps.

#### How It Works

1. At each denoising step, compute the relative L1 distance between current and previous
   modulated inputs.
2. Accumulate the rescaled L1 distance using polynomial coefficients.
3. If accumulated distance is below threshold, reuse the cached residual.
4. Support CFG with separate positive/negative caches.

#### Configuration

```python
from sglang.multimodal_gen.configs.sample.teacache import TeaCacheParams

params = TeaCacheParams(
    teacache_thresh=0.1,
    coefficients=[1.0, 0.0, 0.0],
)
```

| Parameter          | Type         | Description                                          |
|--------------------|--------------|------------------------------------------------------|
| `teacache_thresh`  | float        | Threshold for accumulated L1 distance (lower = more caching, faster) |
| `coefficients`     | list[float]  | Polynomial coefficients for L1 rescaling (model-specific) |

#### Supported Models for TeaCache

| Model Family | CFG Cache Separation | Notes        |
|--------------|---------------------|--------------|
| Wan (wan2.1, wan2.2) | Yes      | Full support |
| Hunyuan (HunyuanVideo) | Yes    | To be supported |
| Z-Image      | Yes                  | To be supported |
| Flux         | No                   | To be supported |
| Qwen         | No                   | To be supported |

---

## Attention Backends

Attention backends are defined by `AttentionBackendEnum` and selected via `--attention-backend`.

### Backend Options

| CLI value                    | Enum                     | Notes                                         |
|------------------------------|--------------------------|-----------------------------------------------|
| `fa` / `fa3` / `fa4`        | `FA`                     | FlashAttention. `fa3/fa4` normalized to `fa`. |
| `torch_sdpa`                 | `TORCH_SDPA`             | PyTorch `scaled_dot_product_attention`.        |
| `sliding_tile_attn`          | `SLIDING_TILE_ATTN`      | Requires `st_attn`. Configure via backend config. |
| `sage_attn`                  | `SAGE_ATTN`              | Requires `sageattention`. SM80/86/89/90/120.  |
| `sage_attn_3`                | `SAGE_ATTN_3`            | Requires SageAttention3.                      |
| `video_sparse_attn`          | `VIDEO_SPARSE_ATTN`      | Requires `vsa`. Configure `sparsity`.         |
| `vmoba_attn`                 | `VMOBA_ATTN`             | Requires `kernel.attn.vmoba_attn.vmoba`.      |
| `aiter`                      | `AITER`                  | Requires `aiter` (ROCm).                      |
| `aiter_sage`                 | `AITER_SAGE`             | Requires `aiter` (ROCm).                      |
| `sla_attn`                   | `SLA_ATTN`               | Sparse Linear Attention. Requires `SpargeAttn`.|
| `sage_sla_attn`              | `SAGE_SLA_ATTN`          | SageAttention + SLA. Requires `SpargeAttn`.   |
| `sparse_video_gen_2_attn`    | `SPARSE_VIDEO_GEN_2_ATTN`| Requires `svg`.                               |

### Platform Support Matrix

| Backend                     | CUDA | ROCm | XPU | MUSA | MPS | NPU |
|-----------------------------|:----:|:----:|:---:|:----:|:---:|:---:|
| `fa`                        | Yes  | Yes  | Yes | Yes  | No  | Yes |
| `torch_sdpa`                | Yes  | Yes  | Yes | Yes  | Yes | Yes |
| `sliding_tile_attn`         | Yes  | No   | No  | No   | No  | No  |
| `sage_attn`                 | Yes  | No   | No  | No   | No  | No  |
| `sage_attn_3`               | Yes  | No   | No  | No   | No  | No  |
| `video_sparse_attn`         | Yes  | No   | No  | No   | No  | No  |
| `sla_attn`                  | Yes  | No   | No  | No   | No  | No  |
| `sage_sla_attn`             | Yes  | No   | No  | No   | No  | No  |
| `vmoba_attn`                | Yes  | No   | No  | No   | No  | No  |
| `aiter`                     | No   | Yes  | No  | No   | No  | No  |
| `aiter_sage`                | No   | Yes  | No  | No   | No  | No  |
| `sparse_video_gen_2_attn`   | Yes  | No   | No  | No   | No  | No  |

### Selection Priority

1. `global_force_attn_backend(...)` / context manager
2. Component override from `--component-attention-backends`
3. CLI `--attention-backend`
4. Auto selection (platform capability, dtype, installed packages)

### Configuration Parameters

Pass via `--attention-backend-config` (JSON/YAML file, JSON string, or key=value pairs).

**Sliding Tile Attention:**

| Parameter             | Type   | Description                           | Default          |
|-----------------------|--------|---------------------------------------|------------------|
| `mask_strategy_file_path` | str | Required. Path to mask strategy JSON. | -                |
| `sta_mode`            | str    | Mode of STA.                          | `STA_inference`  |
| `skip_time_steps`     | int    | Steps of full attention before STA.   | `15`             |

**Video Sparse Attention:**

| Parameter   | Type  | Description                  | Default |
|-------------|-------|------------------------------|---------|
| `sparsity`  | float | Validation sparsity (0.0-1.0)| `0.0`   |

**V-MoBA Attention:**

| Parameter           | Type       | Description                            | Default     |
|---------------------|------------|----------------------------------------|-------------|
| `temporal_chunk_size` | int      | Temporal chunk size                    | -           |
| `temporal_topk`     | int        | Top-K tokens in temporal dimension     | -           |
| `spatial_chunk_size` | list[int] | Spatial chunk size (H, W)              | -           |
| `spatial_topk`      | int        | Top-K tokens in spatial dimension      | -           |
| `st_chunk_size`     | list[int]  | Spatiotemporal chunk size (T, H, W)    | -           |
| `st_topk`           | int        | Top-K tokens in spatiotemporal dimension| -          |
| `moba_select_mode`  | str        | Selection mode                         | `threshold` |
| `moba_threshold`    | float      | Threshold value                        | `0.25`      |
| `first_full_step`   | int        | Initial steps with full attention      | `12`        |

### Usage Examples

```bash
# Select backend via CLI
sglang generate \
  --model-path <MODEL_PATH_OR_ID> \
  --prompt "..." \
  --attention-backend fa

# Override per component
sglang generate \
  --model-path <MODEL_PATH_OR_ID> \
  --prompt "..." \
  --attention-backend fa \
  --component-attention-backends text_encoder=torch_sdpa

# Sliding Tile Attention
sglang generate \
  --model-path <MODEL_PATH_OR_ID> \
  --prompt "..." \
  --attention-backend sliding_tile_attn \
  --attention-backend-config "mask_strategy_file_path=/abs/path/to/mask_strategy.json"
```

---

## Post-Processing

### Frame Interpolation (Video Only)

Uses [RIFE](https://github.com/hzwer/Practical-RIFE) (Real-Time Intermediate Flow Estimation).
Only RIFE 4.22.lite is supported.

Output frame count: **(N - 1) x 2^exp + 1**

| Argument                          | Description                                                   |
|-----------------------------------|---------------------------------------------------------------|
| `--enable-frame-interpolation`    | Enable frame interpolation                                    |
| `--frame-interpolation-exp {EXP}` | Interpolation exponent (default: `1`)                         |
| `--frame-interpolation-scale`     | RIFE inference scale (default: `1.0`, use `0.5` for high-res) |
| `--frame-interpolation-model-path`| Local directory or HF repo for RIFE weights (default: `elfgum/RIFE-4.22.lite`) |

```bash
sglang generate \
  --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --prompt "A dog running through a park" \
  --num-frames 5 \
  --enable-frame-interpolation \
  --frame-interpolation-exp 1 \
  --save-output
```

### Upscaling (Image and Video)

Uses [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN). Architecture is auto-detected
from checkpoint keys.

| Architecture         | Example Weights             | Description                          |
|----------------------|-----------------------------|--------------------------------------|
| RRDBNet              | `RealESRGAN_x4plus.pth`     | Heavier, higher quality for photos   |
| SRVGGNetCompact      | `RealESRGAN_x4.pth` (default)| Lightweight, faster, good for video |

| Argument                    | Description                                              |
|-----------------------------|----------------------------------------------------------|
| `--enable-upscaling`        | Enable post-generation upscaling                         |
| `--upscaling-scale {SCALE}` | Upscaling factor (default: `4`)                          |
| `--upscaling-model-path`    | Local `.pth`, HF repo ID, or `repo_id:filename`          |

```bash
# Image upscaling 4x
sglang generate \
  --model-path black-forest-labs/FLUX.2-dev \
  --prompt "A cat sitting on a windowsill" \
  --output-size 1024x1024 \
  --enable-upscaling \
  --save-output

# Video + interpolation + upscaling
sglang generate \
  --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --prompt "A curious raccoon" \
  --num-frames 5 \
  --enable-frame-interpolation \
  --frame-interpolation-exp 1 \
  --enable-upscaling \
  --upscaling-scale 4 \
  --save-output
```

When both are enabled, frame interpolation runs first, then upscaling runs on every frame.

---

## Compatibility Matrix

### Video Generation Models x Optimization

Legend: Yes = Full compatibility, No = No compatibility, N/A = Does not apply.

| Model Name                   | TeaCache | Sliding Tile Attn | Sage Attn | VSA | SLA | SageSLA | SVG2 |
|:-----------------------------|:--------:|:-----------------:|:---------:|:---:|:---:|:-------:|:----:|
| FastWan2.1 T2V 1.3B          | N/A      | N/A               | N/A       | Yes | No  | No      | No   |
| FastWan2.2 TI2V 5B Full Attn | N/A      | N/A               | N/A       | Yes | No  | No      | No   |
| Wan2.2 TI2V 5B               | N/A      | N/A               | Yes       | N/A | No  | No      | No   |
| Wan2.2 T2V A14B              | No       | No                | Yes       | N/A | No  | No      | No   |
| Wan2.2 I2V A14B              | No       | No                | Yes       | N/A | No  | No      | No   |
| HunyuanVideo                 | No       | Yes               | Yes       | N/A | No  | No      | Yes  |
| FastHunyuan                  | No       | Yes               | Yes       | N/A | No  | No      | Yes  |
| Wan2.1 T2V 1.3B              | Yes      | Yes               | Yes       | N/A | No  | No      | Yes  |
| Wan2.1 T2V 14B               | Yes      | Yes               | Yes       | N/A | No  | No      | Yes  |
| Wan2.1 I2V 480P              | Yes      | Yes               | Yes       | N/A | No  | No      | Yes  |
| Wan2.1 I2V 720P              | Yes      | Yes               | Yes       | N/A | No  | No      | Yes  |
| TurboWan2.1 T2V 1.3B         | Yes      | No                | No        | No  | Yes | Yes     | N/A  |
| TurboWan2.1 T2V 14B          | Yes      | No                | No        | No  | Yes | Yes     | N/A  |
| TurboWan2.1 T2V 14B 720P     | Yes      | No                | No        | No  | Yes | Yes     | N/A  |
| TurboWan2.2 I2V A14B         | Yes      | No                | No        | No  | Yes | Yes     | N/A  |
| Wan2.1 Fun 1.3B InP          | Yes      | Yes               | Yes       | N/A | No  | No      | Yes  |
| Helios Base/Mid/Distilled    | No       | No                | No        | No  | No  | No      | No   |
| LTX-2 / LTX-2.3              | No       | No                | No        | No  | No  | No      | No   |

### Verified LoRA Examples

| Base Model      | Supported LoRAs                                                                                                                                    |
|:----------------|:---------------------------------------------------------------------------------------------------------------------------------------------------|
| Wan2.2          | `lightx2v/Wan2.2-Distill-Loras`, `Cseti/wan2.2-14B-Arcane_Jinx-lora-v1`                                                                            |
| Wan2.1          | `lightx2v/Wan2.1-Distill-Loras`                                                                                                                    |
| Z-Image-Turbo   | `tarn59/pixel_art_style_lora_z_image_turbo`, `wcde/Z-Image-Turbo-DeJPEG-Lora`                                                                      |
| Qwen-Image      | `lightx2v/Qwen-Image-Lightning`, `flymy-ai/qwen-image-realism-lora`, `prithivMLmods/Qwen-Image-HeadshotX`, `starsfriday/Qwen-Image-EVA-LoRA` |
| Qwen-Image-Edit | `ostris/qwen_image_edit_inpainting`, `lightx2v/Qwen-Image-Edit-2511-Lightning`                                                                      |
| Flux            | `dvyio/flux-lora-simple-illustration`, `XLabs-AI/flux-furry-lora`, `XLabs-AI/flux-RealismLora`                                                      |

### Special Requirements

- **Sliding Tile Attention**: Currently only supported on Hopper GPUs (H100).
- **SageSLA**: Based on SpargeAttn. Install: `pip install git+https://github.com/thu-ml/SpargeAttn.git --no-build-isolation`.

---

## Environment Variables

### Runtime

| Variable                                     | Default  | Description                                       |
|----------------------------------------------|----------|---------------------------------------------------|
| `SGLANG_DIFFUSION_TARGET_DEVICE`             | `cuda`   | Target device (`cuda`, `rocm`, `xpu`, `npu`, `musa`, `mps`, `cpu`) |
| `SGLANG_DIFFUSION_ATTENTION_BACKEND`         | not set  | Override attention backend (`fa`, `torch_sdpa`, `sage_attn`) |
| `SGLANG_DIFFUSION_ATTENTION_CONFIG`          | not set  | Path to attention backend configuration file      |
| `SGLANG_DIFFUSION_STAGE_LOGGING`             | false    | Enable per-stage timing logs                      |
| `SGLANG_DIFFUSION_SERVER_DEV_MODE`           | false    | Enable dev-only HTTP endpoints                    |
| `SGLANG_DIFFUSION_TORCH_PROFILER_DIR`        | not set  | Directory for torch profiler traces               |
| `SGLANG_DIFFUSION_CACHE_ROOT`                | `~/.cache/sgl_diffusion` | Root for cache files                    |
| `SGLANG_DIFFUSION_CONFIG_ROOT`               | `~/.config/sgl_diffusion`| Root for config files                   |
| `SGLANG_DIFFUSION_LOGGING_LEVEL`             | `INFO`   | Default logging level                             |
| `SGLANG_DIFFUSION_WORKER_MULTIPROC_METHOD`   | `fork`   | Multiprocess context (`fork` or `spawn`)          |
| `SGLANG_USE_RUNAI_MODEL_STREAMER`            | true     | Use Run:AI model streamer for loading             |

### Platform-Specific

| Variable                            | Default | Description                                        |
|-------------------------------------|---------|----------------------------------------------------|
| `SGLANG_USE_MLX`                    | not set | Set to `1` for MLX fused Metal kernels on MPS      |
| `SGLANG_USE_ROCM_VAE`               | false   | Use AITer GroupNorm in VAE for ROCm                |
| `SGLANG_USE_ROCM_CUDNN_BENCHMARK`   | false   | Enable MIOpen auto-tuning for VAE conv layers      |

### Quantization

| Variable                                     | Default  | Description                               |
|----------------------------------------------|----------|-------------------------------------------|
| `SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND` | not set | FlashInfer FP4 GEMM backend (`flashinfer_cudnn`, `flashinfer_cutlass`, `flashinfer_trtllm`) |

### Cache-DiT

| Variable                              | Default | Description                              |
|---------------------------------------|---------|------------------------------------------|
| `SGLANG_CACHE_DIT_ENABLED`            | false   | Enable Cache-DiT acceleration            |
| `SGLANG_CACHE_DIT_FN`                 | 1       | First N blocks to always compute         |
| `SGLANG_CACHE_DIT_BN`                 | 0       | Last N blocks to always compute          |
| `SGLANG_CACHE_DIT_WARMUP`             | 4       | Warmup steps before caching              |
| `SGLANG_CACHE_DIT_RDT`                | 0.24    | Residual difference threshold            |
| `SGLANG_CACHE_DIT_MC`                 | 3       | Max continuous cached steps              |
| `SGLANG_CACHE_DIT_TAYLORSEER`         | false   | Enable TaylorSeer calibrator             |
| `SGLANG_CACHE_DIT_TS_ORDER`           | 1       | TaylorSeer order (1 or 2)                |
| `SGLANG_CACHE_DIT_SCM_PRESET`         | none    | SCM preset (none/slow/medium/fast/ultra) |
| `SGLANG_CACHE_DIT_SCM_POLICY`         | dynamic | SCM caching policy                       |
| `SGLANG_CACHE_DIT_SCM_COMPUTE_BINS`   | not set | Custom SCM compute bins                  |
| `SGLANG_CACHE_DIT_SCM_CACHE_BINS`     | not set | Custom SCM cache bins                    |

### Cloud Storage

| Variable                          | Default   | Description                                   |
|-----------------------------------|-----------|-----------------------------------------------|
| `SGLANG_CLOUD_STORAGE_TYPE`       | not set   | Set to `s3` to enable cloud storage           |
| `SGLANG_S3_BUCKET_NAME`           | not set   | S3 bucket name                                |
| `SGLANG_S3_ENDPOINT_URL`          | not set   | Custom endpoint URL (MinIO, OSS, etc.)        |
| `SGLANG_S3_REGION_NAME`           | us-east-1 | AWS region name                               |
| `SGLANG_S3_ACCESS_KEY_ID`         | not set   | AWS Access Key ID                             |
| `SGLANG_S3_SECRET_ACCESS_KEY`     | not set   | AWS Secret Access Key                         |

### CUDA Crash Debugging

| Variable                        | Default                    | Description                                    |
|---------------------------------|----------------------------|------------------------------------------------|
| `SGLANG_KERNEL_API_LOGLEVEL`    | `0`                        | Crash-debug kernel API logging level           |
| `SGLANG_KERNEL_API_LOGDEST`     | `stdout`                   | Log destination (`stdout`, `stderr`, or file)  |
| `SGLANG_KERNEL_API_DUMP_DIR`    | `sglang_kernel_api_dumps`  | Output directory for level-10 dumps            |
| `SGLANG_KERNEL_API_DUMP_INCLUDE`| not set                    | Comma-separated wildcard patterns to include   |
| `SGLANG_KERNEL_API_DUMP_EXCLUDE`| not set                    | Comma-separated wildcard patterns to exclude   |

---

## Development Guide: Adding New Models

### Key Components for Implementation

1. **`PipelineConfig`**: Dataclass holding static configurations for the model pipeline
   (precision, architecture parameters, callback methods for `DenoisingStage` and
   `DecodingStage`).
2. **`SamplingParams`**: Dataclass defining runtime generation parameters (`prompt`,
   `guidance_scale`, `num_inference_steps`, `seed`, `height`, `width`, etc.).
3. **Pre-processing stage(s)**: Either a single `{Model}BeforeDenoisingStage` (Hybrid) or
   a combination of standard stages (Modular).
4. **`ComposedPipeline`**: Class wiring together pre-processing, denoising, and decoding.
5. **Modules (model components)**: Components loaded from the model repository.

### Implementation Steps

#### Step 1: Study the Reference Implementation

- Find the model's `model_index.json` to identify required modules.
- Read the Diffusers pipeline's `__call__` method to understand text encoding, latent
  preparation, timestep computation, conditioning kwargs, denoising loop, and VAE decoding.

#### Step 2: Evaluate Reuse

Compare against existing pipelines (Flux, Wan, Qwen-Image, GLM-Image, HunyuanVideo, LTX).
If the model shares most structure, prefer extending an existing implementation.

#### Step 3: Implement Model Components

Place implementations in the appropriate directories:

- DiT/Transformer: `runtime/models/dits/`
- Encoders: `runtime/models/encoders/`
- VAEs: `runtime/models/vaes/`
- Schedulers: `runtime/models/schedulers/`

Use SGLang's fused kernels where possible (`LayerNormScaleShift`, `RMSNormScaleShift`,
`apply_qk_norm`, etc.).

For multi-GPU: add TP/SP support to the DiT model. Reference implementations:
- Wan model (`runtime/models/dits/wanvideo.py`) -- Full TP + SP
- Qwen-Image model (`runtime/models/dits/qwen_image.py`) -- SP via `USPAttention`

#### Step 4: Create Configs

- DiT Config: `configs/models/dits/{model_name}.py`
- VAE Config: `configs/models/vaes/{model_name}.py`
- SamplingParams: `configs/sample/{model_name}.py`

#### Step 5: Create PipelineConfig

```python
# configs/pipeline_configs/my_model.py

@dataclass
class MyModelPipelineConfig(ImagePipelineConfig):
    task_type: ModelTaskType = ModelTaskType.T2I
    vae_precision: str = "bf16"
    should_use_guidance: bool = True
    dit_config: DiTConfig = field(default_factory=MyModelDitConfig)
    vae_config: VAEConfig = field(default_factory=MyModelVAEConfig)

    def get_freqs_cis(self, batch, device, rotary_emb, dtype):
        """Prepare rotary position embeddings for the DiT."""
        ...

    def prepare_pos_cond_kwargs(self, batch, latent_model_input, t, **kwargs):
        """Build positive conditioning kwargs for each denoising step."""
        return {
            "hidden_states": latent_model_input,
            "encoder_hidden_states": batch.prompt_embeds[0],
            "timestep": t,
        }

    def prepare_neg_cond_kwargs(self, batch, latent_model_input, t, **kwargs):
        """Build negative conditioning kwargs for CFG."""
        return {
            "hidden_states": latent_model_input,
            "encoder_hidden_states": batch.negative_prompt_embeds[0],
            "timestep": t,
        }

    def get_decode_scale_and_shift(self):
        """Return (scale, shift) for latent denormalization before VAE decode."""
        ...
```

#### Step 6: Implement Pre-processing

**Option A: BeforeDenoisingStage (Hybrid Style)**

```python
# runtime/pipelines_core/stages/model_specific_stages/my_model.py

class MyModelBeforeDenoisingStage(PipelineStage):
    """Monolithic pre-processing stage for MyModel."""

    def __init__(self, vae, text_encoder, tokenizer, transformer, scheduler):
        super().__init__()
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.transformer = transformer
        self.scheduler = scheduler

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        # 1. Encode prompt
        prompt_embeds, negative_prompt_embeds = self._encode_prompt(...)
        # 2. Prepare latents
        latents = self._prepare_latents(...)
        # 3. Prepare timesteps
        timesteps, sigmas = self._prepare_timesteps(...)
        # 4. Populate batch for DenoisingStage
        batch.prompt_embeds = [prompt_embeds]
        batch.negative_prompt_embeds = [negative_prompt_embeds]
        batch.latents = latents
        batch.timesteps = timesteps
        batch.num_inference_steps = len(timesteps)
        batch.sigmas = sigmas.tolist()
        batch.generator = generator
        batch.raw_latent_shape = latents.shape
        return batch
```

**Key batch fields that `DenoisingStage` expects:**

| Field                        | Type             | Description                                   |
|------------------------------|------------------|-----------------------------------------------|
| `batch.latents`              | `torch.Tensor`   | Initial noisy latent tensor                   |
| `batch.timesteps`            | `torch.Tensor`   | Timestep schedule                             |
| `batch.num_inference_steps`  | `int`            | Number of denoising steps                     |
| `batch.sigmas`               | `list[float]`    | Sigma schedule (Python list, not numpy)       |
| `batch.prompt_embeds`        | `list[Tensor]`   | Positive prompt embeddings (wrapped in list)  |
| `batch.negative_prompt_embeds` | `list[Tensor]` | Negative prompt embeddings (wrapped in list)  |
| `batch.generator`            | `torch.Generator` | RNG generator                                |
| `batch.raw_latent_shape`     | `tuple`          | Original latent shape before packing           |

#### Step 7: Define the Pipeline Class

**Hybrid Style:**

```python
# runtime/pipelines/my_model.py

class MyModelPipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "MyModelPipeline"  # Must match model_index.json _class_name

    _required_config_modules = [
        "text_encoder", "tokenizer", "vae", "transformer", "scheduler",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs):
        self.add_stage(MyModelBeforeDenoisingStage(...))
        self.add_stage(DenoisingStage(...))
        self.add_standard_decoding_stage()

EntryClass = [MyModelPipeline]
```

**Modular Style:**

```python
class MyModelPipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "MyModelPipeline"

    _required_config_modules = [
        "text_encoder", "tokenizer", "vae", "transformer", "scheduler",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs):
        self.add_standard_t2i_stages(
            prepare_extra_timestep_kwargs=[prepare_mu],
        )

EntryClass = [MyModelPipeline]
```

#### Step 8: Register the Model

```python
# registry.py
register_configs(
    model_family="my_model",
    sampling_param_cls=MyModelSamplingParams,
    pipeline_config_cls=MyModelPipelineConfig,
    hf_model_paths=["org/my-model-name"],
)
```

The `EntryClass` in the pipeline file is automatically discovered by the registry.

#### Step 9: Verify Output Quality

Verify generated output is not noise. Common causes of incorrect output:
- Incorrect latent scale/shift factors
- Wrong timestep/sigma schedule
- Mismatched conditioning kwargs
- Rotary embedding style mismatch (`is_neox_style`)

Debug by comparing intermediate tensor values against the Diffusers reference with the same seed.

### Reference Implementations

**Hybrid Style:**

| Model              | Pipeline                    | BeforeDenoisingStage                    |
|--------------------|-----------------------------|-----------------------------------------|
| GLM-Image          | `runtime/pipelines/glm_image.py` | `stages/model_specific_stages/glm_image.py` |
| Qwen-Image-Layered | `runtime/pipelines/qwen_image.py` | `stages/model_specific_stages/qwen_image_layered.py` |

**Modular Style:**

| Model              | Pipeline                    | Notes                                 |
|--------------------|-----------------------------|---------------------------------------|
| Qwen-Image (T2I)   | `runtime/pipelines/qwen_image.py` | Uses `add_standard_t2i_stages()` |
| Qwen-Image-Edit    | `runtime/pipelines/qwen_image.py` | Uses `add_standard_ti2i_stages()`|
| Flux               | `runtime/pipelines/flux.py`      | With custom `prepare_mu`         |
| Wan                | `runtime/pipelines/wan_pipeline.py` | Uses `add_standard_ti2v_stages()`|

### Implementation Checklist

**Common (both styles):**
- Pipeline file at `runtime/pipelines/{model_name}.py` with `EntryClass`
- PipelineConfig at `configs/pipeline_configs/{model_name}.py`
- SamplingParams at `configs/sample/{model_name}.py`
- DiT model at `runtime/models/dits/{model_name}.py`
- Model configs at `configs/models/dits/` and `configs/models/vaes/`
- Registry entry in `registry.py` via `register_configs()`
- `pipeline_name` matches Diffusers `model_index.json` `_class_name`
- `_required_config_modules` lists all modules from `model_index.json`
- `PipelineConfig` callbacks match the DiT's `forward()` signature
- Uses framework-standard `DenoisingStage` and `DecodingStage`
- TP/SP support considered for DiT model
- Output quality verified against Diffusers reference

**Hybrid style only:**
- BeforeDenoisingStage at `stages/model_specific_stages/{model_name}.py`
- `forward()` populates all batch fields required by `DenoisingStage`

---

## Benchmarking

### Performance Baseline Generation

The script at `python/sglang/multimodal_gen/test/scripts/gen_perf_baselines.py` starts a
local diffusion server, issues requests for selected test cases, aggregates stage/denoise-step/
E2E timings, and writes results to `perf_baselines.json`.

#### Usage

Update a single case:

```bash
python python/sglang/multimodal_gen/test/scripts/gen_perf_baselines.py --case qwen_image_t2i
```

Select by regex:

```bash
python python/sglang/multimodal_gen/test/scripts/gen_perf_baselines.py --match 'qwen_image_.*'
```

Run all keys:

```bash
python python/sglang/multimodal_gen/test/scripts/gen_perf_baselines.py --all-from-baseline
```

Specify paths and timeout:

```bash
python python/sglang/multimodal_gen/test/scripts/gen_perf_baselines.py \
  --baseline python/sglang/multimodal_gen/test/server/perf_baselines.json \
  --out /tmp/perf_baselines.json \
  --timeout 600
```

### Performance Comparison Report

For PRs that impact latency, throughput, or memory:

```bash
# 1. Run baseline
sglang generate --model-path <model> --prompt "A benchmark prompt" --perf-dump-path baseline.json

# 2. Run new
sglang generate --model-path <model> --prompt "A benchmark prompt" --perf-dump-path new.json

# 3. Compare
python python/sglang/multimodal_gen/benchmarks/compare_perf.py baseline.json new.json [new2.json ...]
```

---

## Contributing

### Commit Message Convention

Format: `[diffusion] <scope>: <subject>`

Examples:
- `[diffusion] cli: add --perf-dump-path argument`
- `[diffusion] scheduler: fix deadlock in batch processing`
- `[diffusion] model: support Stable Diffusion 3.5`

Rules:
- Always start with `[diffusion]`.
- Scope (optional): `cli`, `scheduler`, `model`, `pipeline`, `docs`, etc.
- Subject: imperative mood, short and clear.

### On AI-Assisted ("Vibe Coding") PRs

Vibe-coded PRs are welcome. The bar is the same for all PRs:

- No over-commenting. If the name says it all, skip the docstring.
- No over-catching. Do not guard against errors that virtually never happen.
- Test before submitting. AI-generated code can be subtly wrong.

### Performance Reporting

For PRs impacting latency, throughput, or memory, provide a performance comparison report
using the benchmarking tools described above.

### CI-Based Change Protection

Add tests to the `pr-test` or `nightly-test` suites for PRs that:
- Support a new model (add testcase to `testcase_configs.py`)
- Support or fix important features
- Significantly improve performance

Run the testcase and update/add the baseline to `perf_baselines.json`.

Test examples are in `python/sglang/multimodal_gen/test/`.
