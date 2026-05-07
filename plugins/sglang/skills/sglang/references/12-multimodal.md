# SGLang Multimodal Reference

This document provides a comprehensive reference for multimodal inference in SGLang, covering image, video, and audio input support, vision-language models, encoder optimizations, and feature processing.

## Table of Contents

- [Overview](#overview)
- [Supported Vision Language Models](#supported-vision-language-models)
- [Image Input Support](#image-input-support)
- [Video Input Support](#video-input-support)
- [Audio Input Support](#audio-input-support)
- [OpenAI-Compatible Vision API](#openai-compatible-vision-api)
- [Offline Engine API](#offline-engine-api)
- [Input Formats for VLMs](#input-formats-for-vlms)
- [Data Parallelism for Encoder](#data-parallelism-for-encoder)
- [CUDA Graph for Multimodal Encoder](#cuda-graph-for-multimodal-encoder)
- [Image Processing Configuration](#image-processing-configuration)
- [Feature Caching and Precomputed Embeddings](#feature-caching-and-precomputed-embeddings)
- [Source Code Structure](#source-code-structure)

---

## Overview

SGLang supports a wide range of multimodal models including vision-language models (VLMs), audio models, and models that combine multiple modalities. The multimodal pipeline in SGLang handles image/video/audio preprocessing, feature extraction through dedicated encoders (e.g., Vision Transformers), and integration with the language model decoder.

A typical VLM architecture involves two main components:
1. **Multimodal encoder** (e.g., Vision Transformer / ViT): Processes visual data, extracts features, and transforms them into a format the language model can understand.
2. **Text decoder** (LLM): Processes textual data and generates output based on the encoded visual features.

---

## Supported Vision Language Models

SGLang supports a wide variety of vision-language and multimodal models through dedicated processors:

| Model | Processor | Description |
|-------|-----------|-------------|
| Qwen2.5-VL | `qwen_vl.py` | Qwen 2.5 Vision-Language model |
| Qwen3-VL | `qwen_vl.py` | Qwen 3 Vision-Language model |
| LLaVA / LLaVA-OneVision | `llava.py` | Large Language and Vision Assistant |
| InternVL | `internvl.py` | InternVision-Language model |
| Gemma 3 | `gemma3.py` | Google Gemma 3 multimodal |
| Gemma 3N | `gemma3n.py` | Google Gemma 3N |
| Gemma 4 | `gemma4.py` | Google Gemma 4 |
| GLM-4V / GLM-4.5V / GLM-4.6V | `glm4v.py` | ChatGLM Vision models |
| Llama 3.2 Vision | `mlama.py` | Meta Llama 3.2 multimodal |
| Llama 4 Scout/Maverick | `mllama4.py` | Meta Llama 4 with vision |
| DeepSeek VL V2 | `deepseek_vl_v2.py` | DeepSeek Vision-Language V2 |
| DeepSeek OCR | `deepseek_ocr.py` | DeepSeek OCR model |
| Janus Pro | `janus_pro.py` | DeepSeek Janus Pro multimodal |
| MiniCPM-V | `minicpm.py` | MiniCPM Vision model |
| Pixtral | `pixtral.py` | Mistral Pixtral |
| Phi-4-MM | `phi4mm.py` | Microsoft Phi-4 Multimodal |
| NVILA | `nvila.py` | NVIDIA VILA |
| Kimi VL | `kimi_vl.py` | Moonshot Kimi Vision-Language |
| Kimi K2.5 | `kimi_k25.py` | Moonshot Kimi K2.5 |
| MiMo V2 | `mimo_v2.py` | Xiaomi MiMo V2 |
| Step3 VL | `step3_vl.py` | Step3 Vision-Language |
| Ernie 4.5 VL | `ernie45_vl.py` | Baidu Ernie 4.5 Vision |
| Dots VLM | `dots_vlm.py` | Dots Vision-Language model |
| Moss VL | `moss_vl.py` | Moss Vision-Language model |
| LightOnOCR | `lightonocr.py` | LightOn OCR model |
| PaddleOCR VLM | `paddleocr_vlm.py` | PaddleOCR-based VLM |
| Nano Nemotron VL | `nano_nemotron_vl.py` | NVIDIA Nano Nemotron Vision |
| LFM2 VL | `lfm2_vl.py` | Liquid Foundation Models V2 VL |
| Points V1.5 Chat | `points_v15_chat.py` | Points V1.5 model |
| Sarashina2 Vision | `sarashina2_vision.py` | Sarashina2 Vision model |
| Voxtral | `voxtral.py` | Mistral Voxtral audio model |
| Qwen Audio | `qwen_audio.py` | Qwen audio model |
| Qwen3 ASR | `qwen3_asr.py` | Qwen3 Automatic Speech Recognition |
| Whisper | `whisper.py` | OpenAI Whisper ASR |
| GLM ASR | `glmasr.py` | ChatGLM ASR model |
| InternS1 Pro | `interns1pro.py` | InternS1 Pro model |
| MidaShengLM | `midashenglm.py` | MidaShengLM model |
| CLIP | `clip.py` | CLIP vision model |
| Transformers Auto | `transformers_auto.py` | Generic transformers processor |

---

## Image Input Support

SGLang supports multiple ways to provide image input to vision-language models:

### 1. URL (HTTP/HTTPS)

Pass an image URL directly:

```python
{
    "type": "image_url",
    "image_url": {
        "url": "https://example.com/image.png"
    }
}
```

### 2. Base64 Encoded

Encode the image as base64:

```python
import base64

with open("image.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode("utf-8")

{
    "type": "image_url",
    "image_url": {
        "url": f"data:image/png;base64,{b64}"
    }
}
```

### 3. File Path

Provide a local file path:

```python
{
    "type": "image_url",
    "image_url": {
        "url": "file:///path/to/image.png"
    }
}
```

### 4. PIL Image (Offline Engine)

Pass a PIL Image object directly to the offline engine:

```python
from PIL import Image

image = Image.open("image.png")
out = llm.generate(prompt=conv.get_prompt(), image_data=[image])
```

### Multiple Images

SGLang supports multiple images in a single request if the model supports it:

```python
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": url1}},
                {"type": "image_url", "image_url": {"url": url2}},
                {"type": "text", "text": "Compare these two images."},
            ],
        }
    ],
)
```

---

## Video Input Support

Several SGLang VLM models support video input, including Qwen2.5-VL, Qwen3-VL, and others. Video is typically provided as a file path or URL, and the processor extracts frames for the vision encoder.

Video input handling varies by model. Refer to the specific model's documentation for supported video formats and any configuration options.

---

## Audio Input Support

SGLang supports audio input through dedicated models:

| Model | Type | Processor |
|-------|------|-----------|
| Voxtral | Audio understanding | `voxtral.py` |
| Qwen Audio | Audio-language | `qwen_audio.py` |
| Qwen3 ASR | Speech recognition | `qwen3_asr.py` |
| Whisper | Speech recognition | `whisper.py` |
| GLM ASR | Speech recognition | `glmasr.py` |
| Gemma 3N | Multimodal (audio+vision) | `gemma3n.py` |
| Phi-4-MM | Multimodal (audio+vision) | `phi4mm.py` |

Audio can also be extracted from video files for processing by compatible models (see `audio_from_video.py`).

---

## OpenAI-Compatible Vision API

SGLang provides OpenAI-compatible APIs for vision-language models, enabling smooth transition from OpenAI services to self-hosted models.

### Launch the Server

```bash
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-VL-7B-Instruct \
    --log-level warning
```

### Using cURL

```bash
curl -s http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-VL-7B-Instruct",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "What is in this image?"},
          {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}
        ]
      }
    ],
    "max_tokens": 300
  }'
```

### Using Python Requests

```python
import requests

url = "http://localhost:30000/v1/chat/completions"
data = {
    "model": "Qwen/Qwen2.5-VL-7B-Instruct",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
            ],
        }
    ],
    "max_tokens": 300,
}
response = requests.post(url, json=data)
print(response.text)
```

### Using OpenAI Python Client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="None")

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
            ],
        }
    ],
    max_tokens=300,
)
print(response.choices[0].message.content)
```

---

## Offline Engine API

The offline Engine API provides three ways to pass visual data to VLMs:

### 1. Raw Images (Simplest)

Pass PIL Images, file paths, URLs, or base64 strings directly. SGLang handles all preprocessing automatically.

```python
from sglang import Engine
from PIL import Image

llm = Engine(model_path="Qwen/Qwen2.5-VL-3B-Instruct", log_level="warning")
image = Image.open("image.png")
out = llm.generate(prompt=conv.get_prompt(), image_data=[image])
```

**Best for**: Quick prototyping, simple applications.

### 2. Processor Output (Custom Preprocessing)

Use a HuggingFace processor for data preprocessing and pass the complete processor output dict.

```python
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
processor_output = processor(
    images=[image], text=conv.get_prompt(), return_tensors="pt"
)

out = llm.generate(
    input_ids=processor_output["input_ids"][0].detach().cpu().tolist(),
    image_data=[dict(processor_output, format="processor_output")],
)
```

**Requirements**: Must use `input_ids` instead of text prompt.

**Best for**: Custom image transformations, integration with existing pipelines.

### 3. Precomputed Embeddings (Maximum Performance)

Pre-calculate visual embeddings using the vision encoder to avoid redundant computation. Provides 30-50% speedup for repeated queries on the same images.

```python
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path).eval()
vision = model.model.visual.cuda()

processor_output = processor(
    images=[image], text=conv.get_prompt(), return_tensors="pt"
)
input_ids = processor_output["input_ids"][0].detach().cpu().tolist()

precomputed_embeddings = vision(
    processor_output["pixel_values"].cuda(),
    processor_output["image_grid_thw"].cuda()
)
precomputed_embeddings = precomputed_embeddings.pooler_output

multi_modal_item = dict(
    processor_output,
    format="precomputed_embedding",
    feature=precomputed_embeddings,
)

out = llm.generate(input_ids=input_ids, image_data=[multi_modal_item])
```

**Best for**: Repeated queries on same images, caching, high-throughput serving.

### Llama 4 Vision Example

Llama 4 requires multi-GPU parallelism:

```python
llm = Engine(
    model_path="meta-llama/Llama-4-Scout-17B-16E-Instruct",
    enable_multimodal=True,
    attention_backend="fa3",
    tp_size=4,
    context_length=65536,
)

out = llm.generate(prompt=conv.get_prompt(), image_data=[image])
```

### Key Rule

Within a single request, use only one format for all images. Do not mix formats.

---

## Input Formats for VLMs

### Format Summary

| Format | How to Pass | Preprocessing | Best For |
|--------|-------------|---------------|----------|
| Raw image | PIL Image, URL, file path, base64 | SGLang handles automatically | Quick prototyping |
| Processor output | `dict(processor_output, format="processor_output")` | HuggingFace processor | Custom transformations |
| Precomputed embedding | `dict(processor_output, format="precomputed_embedding", feature=tensor)` | Manual pre-computation | Caching, high throughput |

---

## Data Parallelism for Encoder

Since the size of ViT is small compared to language decoders, there is relatively little gain from tensor parallelism (TP) for the encoder. TP also incurs significant communication overhead due to all-reduce operations after every layer.

Placing the ViT in data parallelism (DP) while keeping the LLM in tensor parallelism consistently lowers TTFT and boosts end-to-end throughput. In this hybrid layout:
- The vision front-end becomes parallel and lightweight
- Scarce interconnect bandwidth is reserved for the LLM

### Enabling DP Encoder

```bash
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-VL-7B-Instruct \
    --tp 2 \
    --mm-enable-dp-encoder
```

### Supported Models

| Model | PR |
|-------|----|
| Qwen2.5-VL | [#13126](https://github.com/sgl-project/sglang/pull/13126) |
| Qwen3-VL | [#13724](https://github.com/sgl-project/sglang/pull/13724) |
| InternVL | [#13925](https://github.com/sgl-project/sglang/pull/13925) |
| GLM-4.5V & GLM-4.6V | [#14097](https://github.com/sgl-project/sglang/pull/14097) |

---

## CUDA Graph for Multimodal Encoder

The visual encoder (ViT) has characteristics that benefit significantly from CUDA graph optimization:
- Many layers with fragmented operators (LN, QKV, attention, MLP, residuals)
- Small batch sizes with high kernel launch overhead
- Variable sequence lengths (different image resolutions, different batch compositions)

### Enabling CUDA Graph for ViT

Set the environment variable:

```bash
SGLANG_VIT_ENABLE_CUDA_GRAPH=1 \
python3 -m sglang.launch_server \
    --model Qwen/Qwen3-VL-8B-Instruct
```

### Combined with Piecewise CUDA Graph

```bash
SGLANG_VIT_ENABLE_CUDA_GRAPH=1 \
python3 -m sglang.launch_server \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --piecewise-cuda-graph-max-tokens 4096 \
    --enable-piecewise-cuda-graph \
    --piecewise-cuda-graph-compiler eager
```

### Design Details

The CUDA graph for ViT is built on `ViTCudaGraphRunner`, which captures the "blocks + merger + deepstack merger (optional)" part of the vision transformer.

**Dynamic inputs**: Variable sequence length (S) is handled by building a graph cache keyed by S. The first time a new S is encountered, a new graph is captured; subsequent requests with the same S replay the existing graph.

**Stable addresses**: All parameter-like data uses static buffers:
- `block_input`, `block_ws`, `block_output`
- `cu_full_len`, `cu_window_len` and their kk variants
- `sin_cos_ws` (rotary buffer)

**Attention backend**: Arguments are fixed inside the graph. For the same graph_key = S, the segmentation pattern in `cu_seqlens` must be identical.

**Rotary buffer**: Reallocated to a larger size when `seq_len` increases. `max_content_len` ensures the maximum size of the allocated buffer.

### Supported Models

| Model | PR |
|-------|----|
| Qwen2.5-VL | [#14422](https://github.com/sgl-project/sglang/pull/14422) |
| Qwen3-VL | [#15320](https://github.com/sgl-project/sglang/pull/15320) |

### Memory Considerations

Many distinct S values increase VRAM usage due to graph-private memory pools. For workloads with highly variable image resolutions, monitor memory usage when enabling this feature.

---

## Image Processing Configuration

Image processing in SGLang is handled by model-specific processors located in `python/sglang/srt/multimodal/processors/`. Each processor extends `base_processor.py` and implements the appropriate preprocessing pipeline for its model.

### Processing Pipeline

1. **Input parsing**: Accept image URL, base64, file path, or PIL Image
2. **Preprocessing**: Model-specific transforms (resize, normalize, etc.)
3. **Feature extraction**: Run through the vision encoder
4. **Integration**: Combine visual features with text embeddings for the LLM

### Resolution Handling

Different models handle image resolution differently:
- **Fixed resolution**: Resize to a standard size (e.g., 224x224 or 336x336)
- **Dynamic resolution**: Adapt to the original image aspect ratio (e.g., Qwen2.5-VL, InternVL)
- **AnyRes**: Use a grid of crops at multiple resolutions

---

## Feature Caching and Precomputed Embeddings

### Precomputed Embeddings

For maximum performance, especially with repeated queries on the same images, precompute visual embeddings:

```python
# Compute once
precomputed_embeddings = vision(pixel_values, image_grid_thw)

# Use multiple times
for prompt in prompts:
    out = llm.generate(
        input_ids=input_ids,
        image_data=[dict(processor_output, format="precomputed_embedding",
                        feature=precomputed_embeddings)],
    )
```

**Performance gain**: 30-50% speedup by avoiding redundant vision encoder computation.

### Feature Caching Strategy

The precomputed embedding approach is particularly effective for:
- Image analysis applications where the same image is queried with different prompts
- Batch processing where images appear in multiple requests
- Caching layers in production deployments

---

## Source Code Structure

The multimodal implementation is located in `python/sglang/srt/multimodal/`:

| File | Description |
|------|-------------|
| `processors/base_processor.py` | Base processor interface for all multimodal models |
| `processors/qwen_vl.py` | Qwen2.5-VL and Qwen3-VL processor |
| `processors/llava.py` | LLaVA processor |
| `processors/internvl.py` | InternVL processor |
| `processors/gemma3.py` | Gemma 3 processor |
| `processors/glm4v.py` | GLM-4V processor |
| `processors/mlama.py` | Meta Llama 3.2 vision processor |
| `processors/mllama4.py` | Meta Llama 4 processor |
| `processors/deepseek_vl_v2.py` | DeepSeek VL V2 processor |
| `vit_cuda_graph_runner.py` | CUDA graph runner for ViT |
| `internvl_vit_cuda_graph_runner.py` | CUDA graph runner for InternVL ViT |
| `mm_utils.py` | Multimodal utility functions |
| `audio_from_video.py` | Audio extraction from video |
| `internvl_utils.py` | InternVL utility functions |
| `customized_mm_processor_utils.py` | Custom processor utilities |
| `evs/` | EVS (Enhanced Video Streaming) support |

---

## References

- [SGLang Vision API Documentation](https://github.com/sgl-project/sglang/blob/main/docs/references/vision_api.md)
- [Qwen2.5-VL Model](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [DP Encoder PR](https://github.com/sgl-project/sglang/pull/13126)
- [ViT CUDA Graph PR](https://github.com/sgl-project/sglang/pull/14422)
- [Offline Batch Inference VLM Example](https://github.com/sgl-project/sglang/blob/main/examples/runtime/engine/offline_batch_inference_vlm.py)
