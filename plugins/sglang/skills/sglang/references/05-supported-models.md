# SGLang Supported Models Reference

This document provides a comprehensive reference for all model architectures supported by SGLang. Models are organized by task category: generative language models, multimodal language models, embedding models, rerank models, reward models, classification models, and diffusion language models.

## Table of Contents

- [Generative Language Models](#generative-language-models)
- [Multimodal Language Models](#multimodal-language-models)
- [Embedding Models](#embedding-models)
- [Rerank Models](#rerank-models)
- [Reward Models](#reward-models)
- [Classification Models](#classification-models)
- [Diffusion Language Models](#diffusion-language-models)
- [Model Implementation Files](#model-implementation-files)

---

## Generative Language Models

These models accept text input and produce text output (e.g., chat completions). They are primarily large language models (LLMs), some with mixture-of-experts (MoE) architectures for scaling.

### Example Launch Command

```bash
python3 -m sglang.launch_server \
  --model-path meta-llama/Llama-3.2-1B-Instruct \
  --host 0.0.0.0 \
  --port 30000
```

### Supported Models

#### DeepSeek (v1, v2, v3/R1)

- **Example**: `deepseek-ai/DeepSeek-R1`
- **Architecture**: DeepSeekV2ForCausalLM, DeepseekV3ForCausalLM
- **Implementation**: `deepseek_v2.py`, `deepseek.py`
- **Parameters**: Up to 671B (MoE)
- **Features**: Advanced reasoning-optimized models trained with reinforcement learning; top performance on complex reasoning, math, and code tasks. SGLang provides DeepSeek v3/R1 model-specific optimizations and a Reasoning Parser for separating reasoning from output.
- **Launch Example**:
  ```bash
  python3 -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-R1 \
    --tp 8 --trust-remote-code \
    --host 0.0.0.0 --port 30000
  ```

#### Kimi K2 (Thinking, Instruct)

- **Example**: `moonshotai/Kimi-K2-Instruct`
- **Architecture**: KimiK25ForCausalLM
- **Implementation**: `kimi_k25.py`
- **Parameters**: 1 trillion parameter MoE model (32B active) with 128K-256K context
- **Features**: State-of-the-art agentic intelligence with stable long-horizon agency across 200-300 sequential tool calls. Features MLA attention and native INT4 quantization. Supports Reasoning Parser.
- **Launch Example**:
  ```bash
  python3 -m sglang.launch_server \
    --model-path moonshotai/Kimi-K2-Instruct \
    --tp 8 --trust-remote-code \
    --host 0.0.0.0 --port 30000
  ```

#### Kimi Linear (48B-A3B)

- **Example**: `moonshotai/Kimi-Linear-48B-A3B-Instruct`
- **Architecture**: BailingMoeLinearForCausalLM
- **Implementation**: `bailing_moe_linear.py`
- **Parameters**: 48B total, 3B active
- **Features**: Hybrid linear attention model with 1M token context; features Kimi Delta Attention (KDA) for up to 6x faster decoding and 75% KV cache reduction.

#### GPT-OSS

- **Example**: `openai/gpt-oss-20b`, `openai/gpt-oss-120b`
- **Architecture**: GptOssForCausalLM
- **Implementation**: `gpt_oss.py`
- **Features**: OpenAI's GPT-OSS series for complex reasoning, agentic tasks, and versatile developer use cases.

#### Qwen (3.5, 3, 3MoE, 3Next, 2.5, 2 series)

- **Example**: `Qwen/Qwen3.5-397B-A17B`, `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-30B-A3B`, `Qwen/Qwen3-Next-80B-A3B-Instruct`
- **Architecture**: Qwen2ForCausalLM, Qwen3ForCausalLM, Qwen3MoeForCausalLM, Qwen3NextForCausalLM, Qwen35ForCausalLM
- **Implementation**: `qwen2.py`, `qwen3.py`, `qwen3_moe.py`, `qwen3_next.py`, `qwen3_5.py`
- **Parameters**: 0.6B to 397B (MoE variants with various active parameter counts)
- **Features**: Alibaba's latest Qwen3 series for complex reasoning, language understanding, and generation; support for MoE variants. SGLang provides Qwen3 specific reasoning parser.
- **Launch Example**:
  ```bash
  python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-30B-A3B-Instruct \
    --trust-remote-code \
    --host 0.0.0.0 --port 30000
  ```

#### Llama (2, 3.x, 4 series)

- **Example**: `meta-llama/Llama-4-Scout-17B-16E-Instruct`
- **Architecture**: LlamaForCausalLM, Llama4ForCausalLM
- **Implementation**: `llama.py`, `llama4.py`
- **Parameters**: 7B to 400B+
- **Features**: Meta's open LLM series, spanning 7B to 400B parameters (Llama 2, 3, and Llama 4) with well-recognized performance. SGLang provides Llama-4 model-specific optimizations.
- **Launch Example**:
  ```bash
  python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-4-Scout-17B-16E-Instruct \
    --host 0.0.0.0 --port 30000
  ```

#### Mistral (Mixtral, NeMo, Small3)

- **Example**: `mistralai/Mistral-7B-Instruct-v0.2`
- **Architecture**: MistralForCausalLM, MixtralForCausalLM
- **Implementation**: `mistral.py`, `mixtral.py`
- **Parameters**: 7B to large MoE variants
- **Features**: Open 7B LLM by Mistral AI with strong performance; extended into MoE ("Mixtral") and NeMo Megatron variants for larger scale.

#### Gemma (v1, v2, v3, v4)

- **Example**: `google/gemma-3-1b-it`
- **Architecture**: GemmaForCausalLM, Gemma2ForCausalLM, Gemma3ForCausalLM, Gemma4ForCausalLM
- **Implementation**: `gemma.py`, `gemma2.py`, `gemma3_causal.py`, `gemma4_causal.py`
- **Parameters**: 1B to 27B
- **Features**: Google's family of efficient multilingual models; Gemma 3 offers a 128K context window, and larger (4B+) variants support vision input.

#### Phi (Phi-1.5, Phi-2, Phi-3, Phi-4, Phi-MoE series)

- **Example**: `microsoft/Phi-4-multimodal-instruct`, `microsoft/Phi-3.5-MoE-instruct`
- **Architecture**: PhiForCausalLM, PhiMoEForCausalLM
- **Implementation**: `phi.py`, `phimoe.py`
- **Parameters**: 1.3B to 5.6B
- **Features**: Microsoft's Phi family of small models; Phi-4-multimodal (5.6B) processes text, images, and speech, Phi-4-mini is a high-accuracy text model and Phi-3.5-MoE is a mixture-of-experts model.

#### MiniCPM (v3, 4B)

- **Example**: `openbmb/MiniCPM3-4B`
- **Architecture**: MiniCPM3ForCausalLM
- **Implementation**: `minicpm3.py`
- **Parameters**: 4B
- **Features**: OpenBMB's series of compact LLMs for edge devices; MiniCPM 3 (4B) achieves GPT-3.5-level results in text tasks.

#### OLMo (2, 3)

- **Example**: `allenai/OLMo-3-1125-32B`, `allenai/OLMo-2-1124-7B-Instruct`
- **Architecture**: OlmoForCausalLM
- **Implementation**: `olmo.py`, `olmo2.py`
- **Parameters**: 7B to 32B
- **Features**: Allen AI's series of Open Language Models designed to enable the science of language models.

#### OLMoE (Open MoE)

- **Example**: `allenai/OLMoE-1B-7B-0924`
- **Architecture**: OlmoeForCausalLM
- **Implementation**: `olmoe.py`
- **Parameters**: 7B total, 1B active parameters
- **Features**: Allen AI's open Mixture-of-Experts model delivering state-of-the-art results with sparse expert activation.

#### MiniMax-M2 (M2, M2.1, M2.5)

- **Example**: `MiniMaxAI/MiniMax-M2.5`
- **Architecture**: MiniMaxM2ForCausalLM
- **Implementation**: `minimax_m2.py`
- **Features**: MiniMax's SOTA LLM for coding and agentic workflows.

#### StableLM (3B, 7B)

- **Example**: `stabilityai/stablelm-tuned-alpha-7b`
- **Architecture**: StableLmForCausalLM
- **Implementation**: `stablelm.py`
- **Parameters**: 3B and 7B
- **Features**: StabilityAI's early open-source LLM for general text generation.

#### Command-(R,A) (Cohere)

- **Example**: `CohereLabs/c4ai-command-r-v01`, `CohereLabs/c4ai-command-a-03-2025`
- **Architecture**: CohereForCausalLM
- **Implementation**: `commandr.py`
- **Parameters**: 7B to large
- **Features**: Cohere's open conversational LLM optimized for long context, retrieval-augmented generation, and tool use.

#### DBRX (Databricks)

- **Example**: `databricks/dbrx-instruct`
- **Architecture**: DbrxForCausalLM
- **Implementation**: `dbrx.py`
- **Parameters**: 132B total, 36B active (MoE)
- **Features**: Databricks' MoE model trained on 12T tokens; competes with GPT-3.5 quality as a fully open foundation model.

#### Grok (xAI)

- **Example**: `xai-org/grok-1`
- **Architecture**: GrokForCausalLM
- **Implementation**: `grok.py`
- **Parameters**: 314B
- **Features**: xAI's grok-1 model known for vast size and high quality.

#### ChatGLM (GLM-130B family)

- **Example**: `THUDM/chatglm2-6b`
- **Architecture**: ChatGLMModel
- **Implementation**: `chatglm.py`
- **Parameters**: 6B
- **Features**: Zhipu AI's bilingual chat model excelling at Chinese-English dialogue.

#### InternLM 2 (7B, 20B)

- **Example**: `internlm/internlm2-7b`
- **Architecture**: InternLM2ForCausalLM
- **Implementation**: `internlm2.py`
- **Parameters**: 7B and 20B
- **Features**: Next-gen InternLM from SenseTime, offering strong reasoning and ultra-long context support (up to 200K tokens).

#### ExaONE 3 (Korean-English)

- **Example**: `LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct`
- **Architecture**: Exaone4ForCausalLM
- **Implementation**: `exaone4.py`
- **Parameters**: 7.8B
- **Features**: LG AI Research's Korean-English model trained on 8T tokens; provides high-quality bilingual understanding and generation.

#### Baichuan 2 (7B, 13B)

- **Example**: `baichuan-inc/Baichuan2-13B-Chat`
- **Architecture**: BaichuanForCausalLM
- **Implementation**: `baichuan.py`
- **Parameters**: 7B and 13B
- **Features**: BaichuanAI's second-generation Chinese-English LLM with improved performance and an open commercial license.

#### XVERSE (MoE)

- **Example**: `xverse/XVERSE-MoE-A36B`
- **Architecture**: XverseMoeForCausalLM
- **Implementation**: `xverse_moe.py`
- **Parameters**: 255B total, 36B active (MoE)
- **Features**: Yuanxiang's open MoE LLM supporting ~40 languages.

#### SmolLM (135M-1.7B)

- **Example**: `HuggingFaceTB/SmolLM-1.7B`
- **Parameters**: 135M to 1.7B
- **Features**: Hugging Face's ultra-small LLM series offering surprisingly strong results for mobile/edge devices.

#### GLM-4 (Multilingual 9B)

- **Example**: `ZhipuAI/glm-4-9b-chat`
- **Architecture**: Glm4MoeForCausalLM
- **Implementation**: `glm4.py`, `glm4_moe.py`
- **Parameters**: Up to 9B
- **Features**: Zhipu's GLM-4 series open multilingual models with support for 1M-token context.

#### MiMo (7B series)

- **Example**: `XiaomiMiMo/MiMo-7B-RL`
- **Architecture**: MiMoForCausalLM
- **Implementation**: `mimo.py`
- **Parameters**: 7B
- **Features**: Xiaomi's reasoning-optimized model series, leverages Multiple-Token Prediction for faster inference.

#### ERNIE-4.5 (4.5, 4.5MoE series)

- **Example**: `baidu/ERNIE-4.5-21B-A3B-PT`
- **Architecture**: Ernie4ForCausalLM
- **Implementation**: `ernie4.py`
- **Parameters**: MoE with 47B and 3B active parameters; largest model has 424B total
- **Features**: Baidu's ERNIE-4.5 series MoE and dense models.

#### Additional Supported Models

| Model Family | Example HuggingFace ID | Parameters | Notes |
|---|---|---|---|
| Arcee AFM-4.5B | `arcee-ai/AFM-4.5B-Base` | 4.5B | Foundational model for edge deployments |
| Persimmon | `adept/persimmon-8b-chat` | 8B | 16K context, Apache 2.0 licensed |
| Solar | `upstage/SOLAR-10.7B-Instruct-v1.0` | 10.7B | Depth-up scaling methodology |
| Tele FLM | `CofeAI/Tele-FLM` | 52B-1T | Multilingual decoder-only transformer |
| Ling | `inclusionAI/Ling-lite` | 16.8B-290B (MoE) | Open MoE models for NLP and reasoning |
| Granite 3.0/3.1 | `ibm-granite/granite-3.1-8b-instruct` | 3B-8B | IBM's open dense foundation models |
| Granite 3.0 MoE | `ibm-granite/granite-3.0-3b-a800m-instruct` | 3B total/800M active | IBM MoE for enterprise deployment |
| GPT-J | `EleutherAI/gpt-j-6b` | 6B | GPT-2-like causal model trained on The Pile |
| Orion | `OrionStarAI/Orion-14B-Base` | 14B | Multilingual model pretrained on 2.5T tokens |
| Llama Nemotron Super | `nvidia/Llama-3_3-Nemotron-Super-49B-v1` | 49B | NVIDIA enterprise-ready reasoning models |
| Llama Nemotron Ultra | `nvidia/Llama-3_1-Nemotron-Ultra-253B-v1` | 253B | NVIDIA's largest Nemotron model |
| Nemotron Nano 2.0 | `nvidia/NVIDIA-Nemotron-Nano-9B-v2` | 9B | Hybrid Mamba-Transformer for reasoning |
| Nemotron 3 Super | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | 120B/12B active | MoE for enterprise AI agents |
| Nemotron 3 Nano | `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` | 4B | Compact model for edge deployment |
| StarCoder2 | `bigcode/starcoder2-7b` | 3B-15B | Specialized for code generation |
| Jet-Nemotron | `jet-ai/Jet-Nemotron-2B` | 2B | Hybrid-architecture surpassing full-attention models |
| Trinity | `arcee-ai/Trinity-Mini` | MoE | Arcee's foundational MoE family |
| LFM2 | `LiquidAI/LFM2.5-1.2B-Instruct` | 350M-1.2B | Hybrid attention + short convolution |
| LFM2-MoE | `LiquidAI/LFM2-8B-A1B` | 8B-A1B, 24B-A2B | MoE with sigmoid routing and top-k selection |
| Falcon-H1 | `tiiuae/Falcon-H1-34B-Instruct` | 0.5B-34B | Hybrid Mamba-Transformer |
| Hunyuan-Large | `tencent/Tencent-Hunyuan-Large` | 389B/52B active (MoE) | Cross-Layer Attention (CLA) for efficiency |
| IBM Granite 4.0 | `ibm-granite/granite-4.0-h-micro` | Micro | Hybrid Mamba-MoE and dense variants |
| Sarvam 2 | `sarvamai/sarvam-2` | 30B-A2B, 105B-A10B | MoE with MLA or GQA, 128 routed experts |

---

## Multimodal Language Models

These models accept multi-modal inputs (e.g., images, audio, video, and text) and generate text output. They augment language models with multimodal encoders.

### Example Launch Command

```bash
python3 -m sglang.launch_server \
  --model-path meta-llama/Llama-3.2-11B-Vision-Instruct \
  --host 0.0.0.0 \
  --port 30000
```

### Supported Models

| Model Family | Example Model | Features |
|---|---|---|
| Qwen-VL | `Qwen/Qwen3-VL-235B-A22B-Instruct` | Vision-language extension of Qwen; analyzes and converses about image content |
| DeepSeek-VL2 | `deepseek-ai/deepseek-vl2` | Vision-language variant with dedicated image processor |
| DeepSeek-OCR / OCR-2 | `deepseek-ai/DeepSeek-OCR-2` | OCR-focused models for document understanding. Use `--trust-remote-code` |
| Janus-Pro (1B, 7B) | `deepseek-ai/Janus-Pro-7B` | Multimodal model for both image understanding and generation; decoupled architecture |
| MiniCPM-V / MiniCPM-o | `openbmb/MiniCPM-V-2_6` | Supports image inputs; MiniCPM-o adds audio/video; optimized for edge deployment |
| Llama 3.2 Vision (11B) | `meta-llama/Llama-3.2-11B-Vision-Instruct` | Vision-enabled Llama 3 variant for visual question answering |
| LLaVA (v1.5 & v1.6) | `liuhaotian/llava-v1.5-13b` | Open vision-chat models adding image encoder to LLaMA/Vicuna |
| LLaVA-NeXT (8B, 72B) | `lmms-lab/llava-next-72b` | Enhanced visual instruction-following and accuracy |
| LLaVA-OneVision | `lmms-lab/llava-onevision-qwen2-7b-ov` | Integrates Qwen backbone; supports multiple images and video frames |
| Gemma 3 (Multimodal) | `google/gemma-3-4b-it` | 4B+ variants accept images (256 tokens/image) in 128K context |
| Kimi-VL (A3B) | `moonshotai/Kimi-VL-A3B-Instruct` | Multimodal model understanding text from images |
| Mistral-Small-3.1-24B | `mistralai/Mistral-Small-3.1-24B-Instruct-2503` | Multimodal with tool calling and structured output |
| Phi-4-multimodal-instruct | `microsoft/Phi-4-multimodal-instruct` | Supports text, vision and audio modalities |
| MiMo-VL (7B) | `XiaomiMiMo/MiMo-VL-7B-RL` | Native resolution ViT encoder for fine-grained visual details |
| GLM-4.5V (106B) / GLM-4.1V (9B) | `zai-org/GLM-4.5V` | Versatile multimodal reasoning. Use `--chat-template glm-4v` |
| GLM-OCR | `zai-org/GLM-OCR` | Fast and accurate general OCR model |
| DotsVLM (General/OCR) | `rednote-hilab/dots.vlm1.inst` | 1.2B vision encoder + DeepSeek V3 LLM with NaViT dynamic resolution |
| DotsVLM-OCR | `rednote-hilab/dots.ocr` | Specialized OCR variant. Do not use `--trust-remote-code` |
| NVILA | `Efficient-Large-Model/NVILA-8B` | Full stack efficiency multi-modal design; multiple sizes available |
| Nemotron Nano 2.0 VL | `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16` | Multi-image reasoning, video understanding, document intelligence. Use `--trust-remote-code` |
| Ernie4.5-VL | `baidu/ERNIE-4.5-VL-28B-A3B-PT` | Baidu's VLMs (28B, 424B) supporting image, video, and thinking |
| Step3-VL (10B) | `stepfun-ai/Step3-VL-10B` | Lightweight 10B parameter VLM for multimodal intelligence |
| Qwen3-ASR (0.6B, 1.7B) | `Qwen/Qwen3-ASR-1.7B` | ASR models supporting 52 languages; served via `/v1/audio/transcriptions` |
| Qwen3-Omni | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | Omni-modal MoE model (Thinker: text/images/audio/video; Talker: not yet supported) |
| LFM2-VL | `LiquidAI/LFM2.5-VL-1.6B` | SigLip2 vision encoder + LFM2 hybrid attention model; multi-image inputs |

### Audio Transcription

SGLang supports audio-only ASR models via the OpenAI-compatible `/v1/audio/transcriptions` endpoint.

```bash
# Launch ASR server
sglang serve \
  --model-path Qwen/Qwen3-ASR-1.7B \
  --served-model-name qwen3-asr \
  --trust-remote-code \
  --host 0.0.0.0 --port 30000

# Send request
curl http://localhost:30000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=qwen3-asr \
  -F response_format=verbose_json
```

| Model Family | Example Model | Notes |
|---|---|---|
| Whisper | `openai/whisper-large-v3` | OpenAI's speech recognition model |
| Qwen3-ASR (0.6B, 1.7B) | `Qwen/Qwen3-ASR-1.7B` | Use `--trust-remote-code`. Supports 52 languages |

### Video Input Support

SGLang supports video input for Vision-Language Models. Video clips are decoded, key frames are sampled, and the resulting tensors are batched with the text prompt.

| Model Family | Example Model | Video Notes |
|---|---|---|
| Qwen-VL | `Qwen/Qwen3-VL-235B-A22B-Instruct` | Runs Qwen's frame sampler, merges features with text tokens |
| GLM-4v | `zai-org/GLM-4.5V` | Decord for frame extraction with rotary-position handling |
| NVILA | `Efficient-Large-Model/NVILA-8B` | Samples 8 frames per clip |
| LLaVA video variants | `lmms-lab/LLaVA-NeXT-Video-7B` | Routes video prompts to LlavaVid architecture |
| Nemotron Nano 2.0 VL | `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16` | Samples at 2 FPS, max 128 frames, with EVS pruning |

### Performance Optimization

- Use `--keep-mm-feature-on-device` to keep multimodal feature tensors on GPU for lower latency at the cost of more GPU memory.
- Use `--mm-process-config` to set image, video, and audio input limits for memory management:
  ```bash
  --mm-process-config '{"image":{"max_pixels":1048576},"video":{"fps":3,"max_pixels":602112,"max_frames":60}}'
  ```

### Bidirectional Attention

Gemma-3 employs bidirectional attention between image tokens during prefill. To enable this, use the Triton attention backend with CUDA Graph and Chunked Prefill disabled:

```bash
python -m sglang.launch_server \
  --model-path google/gemma-3-4b-it \
  --enable-multimodal \
  --dtype bfloat16 --triton-attention-reduce-in-fp32 \
  --attention-backend triton \
  --disable-cuda-graph \
  --chunked-prefill-size -1
```

---

## Embedding Models

Embedding models generate vector representations of text (and optionally images) for semantic search and retrieval. Launch with `--is-embedding` flag.

### Example Launch Command

```bash
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-Embedding-4B \
  --is-embedding \
  --host 0.0.0.0 \
  --port 30000
```

### Client Request

```python
import requests

url = "http://127.0.0.1:30000"
payload = {
    "model": "Qwen/Qwen3-Embedding-4B",
    "input": "What is the capital of France?",
    "encoding_format": "float"
}
response = requests.post(url + "/v1/embeddings", json=payload).json()
print("Embedding:", response["data"][0]["embedding"])
```

### Supported Models

| Model Family | Example Model | Chat Template | Description |
|---|---|---|---|
| E5 (Llama/Mistral based) | `intfloat/e5-mistral-7b-instruct` | N/A | High-quality text embeddings based on Mistral/Llama |
| GTE-Qwen2 | `Alibaba-NLP/gte-Qwen2-7B-instruct` | N/A | Alibaba's text embedding model with multilingual support |
| Qwen3-Embedding | `Qwen/Qwen3-Embedding-4B` | N/A | Latest Qwen3-based text embedding model |
| BGE | `BAAI/bge-large-en-v1.5` | N/A | BAAI's text embeddings (requires `attention-backend` triton/torch_native) |
| GME (Multimodal) | `Alibaba-NLP/gme-Qwen2-VL-2B-Instruct` | `gme-qwen2-vl` | Multimodal embedding for text and image cross-modal tasks |
| CLIP | `openai/clip-vit-large-patch14-336` | N/A | OpenAI's CLIP for image and text embeddings |

### Matryoshka Embedding

Matryoshka Embeddings (MRL) allow trading off between performance and cost by truncating embedding dimensions.

```bash
# Launch with Matryoshka dimensions
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-Embedding-0.6B \
  --is-embedding \
  --json-model-override-args '{"matryoshka_dimensions": [128, 256, 512, 1024, 1536]}'
```

Request a truncated embedding by specifying `dimensions`:
```python
payload = {
    "model": "Qwen/Qwen3-Embedding-0.6B",
    "input": "Explain diffusion models simply.",
    "dimensions": 512  # or 128, 1024, or omit for full size
}
```

---

## Rerank Models

Rerank models reorder search results based on semantic relevance. SGLang supports two categories:

- **Cross-encoder rerank models**: Run with `--is-embedding` (embedding runner)
- **Decoder-only rerank models**: Run WITHOUT `--is-embedding` and use next-token logprob scoring (yes/no)

### Supported Models

| Model Family | Example Model | Chat Template | Description |
|---|---|---|---|
| BGE-Reranker | `BAAI/bge-reranker-v2-m3` | N/A | Cross-encoder reranker. Requires `attention-backend` triton or torch_native |
| Qwen3-Reranker | `Qwen/Qwen3-Reranker-8B` | `examples/chat_template/qwen3_reranker.jinja` | Decoder-only yes/no logprob scoring. Launch WITHOUT `--is-embedding` |
| Qwen3-VL-Reranker | `Qwen/Qwen3-VL-Reranker-2B` | `examples/chat_template/qwen3_vl_reranker.jinja` | Multimodal decoder-only reranker. Launch WITHOUT `--is-embedding` |

### Cross-Encoder Rerank Launch

```bash
python3 -m sglang.launch_server \
  --model-path BAAI/bge-reranker-v2-m3 \
  --disable-radix-cache \
  --chunked-prefill-size -1 \
  --attention-backend triton \
  --is-embedding \
  --host 0.0.0.0 --port 30000
```

### Decoder-Only Rerank Launch

```bash
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-Reranker-0.6B \
  --trust-remote-code \
  --disable-radix-cache \
  --chat-template examples/chat_template/qwen3_reranker.jinja \
  --host 0.0.0.0 --port 30000
```

### Request Parameters

- `query` (required): Query text to rank documents against. For multimodal models, can be a list of content parts.
- `documents` (required): List of documents to be ranked. Each can be a string or list of content parts.
- `model` (required): Model to use for reranking.
- `instruct` (optional): Instruction text for the reranker.
- `top_n` (optional): Maximum number of documents to return.
- `return_documents` (optional): Whether to return documents in the response (default: true).

### Response Format

Returns a list of objects sorted by descending score:
```json
[
  {"score": 0.99, "document": "...", "index": 0},
  {"score": 0.01, "document": "...", "index": 1}
]
```

---

## Reward Models

Reward models output scalar reward scores, often used in RLHF and content moderation. Launch with `--is-embedding`.

### Example Launch Command

```bash
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-Math-RM-72B \
  --is-embedding \
  --tp-size 4 \
  --host 0.0.0.0 --port 30000
```

### Supported Models

| Model Family | Architecture | Example Model | Description |
|---|---|---|---|
| Llama 3.1 Reward | LlamaForSequenceClassification | `Skywork/Skywork-Reward-Llama-3.1-8B-v0.2` | Reward model based on Llama 3.1 (8B) for RLHF scoring |
| Gemma 2 Reward | Gemma2ForSequenceClassification | `Skywork/Skywork-Reward-Gemma-2-27B-v0.2` | Human preference scoring for RLHF and multilingual tasks |
| InternLM 2 Reward | InternLM2ForRewardModel | `internlm/internlm2-7b-reward` | Reward model used in alignment pipelines |
| Qwen2.5 Math Reward | Qwen2ForRewardModel | `Qwen/Qwen2.5-Math-RM-72B` | 72B math-specialized RLHF reward model |
| Qwen2.5 Sequence Classification | Qwen2ForSequenceClassification | `jason9693/Qwen2.5-1.5B-apeach` | Sequence classification for RLHF scoring |

---

## Classification Models

The `/v1/classify` endpoint classifies text inputs. Compatible with vLLM's 0.7.0 classification API format.

### Supported Architecture Classes

| Architecture | Type | Notes |
|---|---|---|
| LlamaForSequenceClassification | Multi-class | Automatic `id2label` mapping from config |
| Qwen2ForSequenceClassification | Multi-class | Automatic label mapping |
| Qwen3ForSequenceClassification | Multi-class | Automatic label mapping |
| BertForSequenceClassification | Multi-class | Automatic label mapping |
| Gemma2ForSequenceClassification | Multi-class | Automatic label mapping |
| InternLM2ForRewardModel | Single reward score | Special reward model |
| Qwen2ForRewardModel | Single reward score | Special reward model |

### Request Format

```bash
curl "http://127.0.0.1:8000/v1/classify" \
  -H "Content-Type: application/json" \
  -d '{"model": "jason9693/Qwen2.5-1.5B-apeach", "input": "Loved the new cafe."}'
```

### Response Format

```json
{
  "id": "classify-9bf17f2847b046c7b2d5495f4b4f9682",
  "object": "list",
  "created": 1745383213,
  "model": "jason9693/Qwen2.5-1.5B-apeach",
  "data": [
    {
      "index": 0,
      "label": "Default",
      "probs": [0.566, 0.434],
      "num_classes": 2
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "total_tokens": 10,
    "completion_tokens": 0
  }
}
```

---

## Diffusion Language Models

Diffusion language models enable non-autoregressive text generation with parallel decoding capabilities. Unlike autoregressive models, they require different decoding strategies.

### Example Launch Command

```bash
python3 -m sglang.launch_server \
  --model-path inclusionAI/LLaDA2.0-mini \
  --dllm-algorithm LowConfidence \
  --dllm-algorithm-config ./config.yaml \
  --host 0.0.0.0 --port 30000
```

### DLLM Algorithms

#### LowConfidence

Accepts predicted tokens when confidence exceeds a threshold.

```yaml
# Config
threshold: 0.95       # Confidence threshold (0.0-1.0). Higher = more conservative
block_size: 32        # Default: 32, for LLaDA2MoeModelLM
```

#### JointThreshold

Two-phase decoding: Mask-to-Token (M2T) and Token-to-Token (T2T).

```yaml
# Config
threshold: 0.5            # M2T phase threshold (0.0-1.0)
edit_threshold: 0.0       # T2T phase threshold (0.0-1.0). 0.0 allows full editing
max_post_edit_steps: 16   # Max extra T2T steps after all masks removed
penalty_lambda: 0         # 2-gram repetition penalty (default 0; try 3 for mitigating repetitions)
```

### Supported Models

| Model Family | Example Model | Architecture | Description |
|---|---|---|---|
| LLaDA2.0 (mini, flash) | `inclusionAI/LLaDA2.0-flash` | Llada2ForCausalLM | 100B MoE diffusion language model |
| SDAR (JetLM) | `JetLM/SDAR-8B-Chat` | SdarForCausalLM | Dense architecture diffusion model |
| SDAR MoE (JetLM) | `JetLM/SDAR-30B-A3B-Chat` | SdarMoeForCausalLM | MoE architecture diffusion model |

### Client Usage

```python
import sglang as sgl

llm = sgl.Engine(
    model_path="inclusionAI/LLaDA2.0-mini",
    dllm_algorithm="LowConfidence",
    max_running_requests=1,
    trust_remote_code=True
)

prompts = ["Write a brief introduction of the great wall"]
sampling_params = {"temperature": 0, "max_new_tokens": 1024}
outputs = llm.generate(prompts, sampling_params)
print(outputs)
```

---

## Model Implementation Files

All model implementations are located in `python/sglang/srt/models/`. Key files include:

### Text Generation
`llama.py`, `llama4.py`, `qwen2.py`, `qwen3.py`, `qwen3_moe.py`, `qwen3_next.py`, `qwen3_5.py`, `deepseek.py`, `deepseek_v2.py`, `mistral.py`, `mixtral.py`, `gemma.py`, `gemma2.py`, `gemma3_causal.py`, `gemma4_causal.py`, `phi.py`, `phimoe.py`, `grok.py`, `dbrx.py`, `chatglm.py`, `glm4.py`, `glm4_moe.py`, `commandr.py`, `internlm2.py`, `minicpm3.py`, `stablelm.py`, `baichuan.py`, `olmo.py`, `olmoe.py`, `solar.py`, `starcoder2.py`, `xverse_moe.py`, `gpt_j.py`, `gpt_oss.py`, `kimi_k25.py`, `kimi_linear.py`, `jet_nemotron.py`, `falcon_h1.py`, `lfm2.py`, `lfm2_moe.py`, `hunyuan.py`, `granite.py`, `granitemoe.py`, `ernie4.py`, `mimo.py`, `minimax_m2.py`, `sarvam_moe.py`

### Multimodal
`mllama.py`, `mllama4.py`, `qwen2_vl.py`, `qwen2_5_vl.py`, `qwen3_vl.py`, `qwen3_vl_moe.py`, `deepseek_vl2.py`, `deepseek_janus_pro.py`, `deepseek_ocr.py`, `llava.py`, `llavavid.py`, `minicpmv.py`, `minicpmo.py`, `gemma3_mm.py`, `gemma4_mm.py`, `gemma4_vision.py`, `kimi_vl.py`, `nvila.py`, `nvila_lite.py`, `phi4mm.py`, `dots_vlm.py`, `dots_ocr.py`, `glm4v.py`, `glm4v_moe.py`, `glm_ocr.py`, `step3_vl.py`, `step3_vl_10b.py`, `lfm2_vl.py`, `mimo_vl.py`

### Embedding / Reward / Classification
`llama_embedding.py`, `llama_classification.py`, `llama_reward.py`, `gemma2_reward.py`, `qwen2_classification.py`, `qwen3_classification.py`, `qwen2_rm.py`, `qwen3_rm.py`, `internlm2_reward.py`, `bert.py`

### Diffusion
`llada2.py`, `sdar.py`, `sdar_moe.py`

### Speculative Decoding (Eagle)
`llama_eagle.py`, `llama_eagle3.py`, `qwen2_eagle.py`, `mistral_eagle.py`, `deepseek_nextn.py`, `ernie4_eagle.py`

### To search for a specific architecture

Use GitHub search:
```
repo:sgl-project/sglang path:/^python/sglang/srt/models// Qwen3ForCausalLM
```
