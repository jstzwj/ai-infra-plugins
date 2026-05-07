# SGLang API Reference

This document provides the complete reference for all SGLang APIs including OpenAI-compatible
endpoints, native SGLang endpoints, Anthropic and Ollama compatibility layers, the offline
engine API, gRPC protocol, and all management and debugging APIs.

---

## Table of Contents

1. [API Overview](#api-overview)
2. [OpenAI-Compatible APIs](#openai-compatible-apis)
   - [Chat Completions](#chat-completions)
   - [Text Completions](#text-completions)
   - [Embeddings](#embeddings)
   - [Rerank](#rerank)
   - [Score](#score)
   - [Classify](#classify-v1)
   - [Tokenize](#tokenize)
   - [Detokenize](#detokenize)
   - [Audio Transcriptions](#audio-transcriptions)
   - [Responses API](#responses-api)
   - [List Models](#list-models)
   - [Retrieve Model](#retrieve-model)
3. [Native SGLang APIs](#native-sglang-apis)
   - [Generate](#generate)
   - [Encode](#encode)
   - [Classify (Native)](#classify-native)
   - [Model Info](#model-info)
   - [Server Info](#server-info)
   - [Health Check](#health-check)
   - [Load Metrics](#load-metrics)
   - [Flush Cache](#flush-cache)
   - [Abort Request](#abort-request)
   - [Update Weights from Disk](#update-weights-from-disk)
   - [Tokenize / Detokenize (Native)](#tokenize--detokenize-native)
   - [Expert Distribution](#expert-distribution)
   - [Parse Function Call](#parse-function-call)
   - [Separate Reasoning](#separate-reasoning)
4. [LoRA Management API](#lora-management-api)
   - [Load LoRA Adapter](#load-lora-adapter)
   - [Load LoRA Adapter from Tensors](#load-lora-adapter-from-tensors)
   - [Unload LoRA Adapter](#unload-lora-adapter)
5. [Weight Management API](#weight-management-api)
   - [Update Weights from Disk](#update-weights-from-disk-detailed)
   - [Update Weights from Tensor](#update-weights-from-tensor)
   - [Update Weights from Distributed](#update-weights-from-distributed)
   - [Update Weights from IPC](#update-weights-from-ipc)
   - [Update Weight Version](#update-weight-version)
   - [Init Weights Update Group](#init-weights-update-group)
   - [Destroy Weights Update Group](#destroy-weights-update-group)
   - [Init Weights Send Group for Remote Instance](#init-weights-send-group-for-remote-instance)
   - [Send Weights to Remote Instance](#send-weights-to-remote-instance)
   - [Get Weights by Name](#get-weights-by-name)
   - [Weights Checker](#weights-checker)
   - [Release / Resume Memory Occupation](#release--resume-memory-occupation)
   - [Slow Down](#slow-down)
6. [HiCache Management API](#hicache-management-api)
   - [Clear HiCache Storage Backend](#clear-hicache-storage-backend)
   - [Attach HiCache Storage Backend](#attach-hicache-storage-backend)
   - [Detach HiCache Storage Backend](#detach-hicache-storage-backend)
   - [HiCache Storage Backend Status](#hicache-storage-backend-status)
7. [Session Management API](#session-management-api)
   - [Open Session](#open-session)
   - [Close Session](#close-session)
   - [Pause Generation](#pause-generation)
   - [Continue Generation](#continue-generation)
8. [Ngram Speculative Decoding API](#ngram-speculative-decoding-api)
   - [Add External Corpus](#add-external-corpus)
   - [Remove External Corpus](#remove-external-corpus)
   - [List External Corpora](#list-external-corpora)
9. [Profiling and Debugging API](#profiling-and-debugging-api)
   - [Start Profile](#start-profile)
   - [Stop Profile](#stop-profile)
   - [Set Trace Level](#set-trace-level)
   - [Freeze GC](#freeze-gc)
   - [Configure Logging](#configure-logging)
   - [Set Internal State](#set-internal-state)
10. [Ollama-Compatible API](#ollama-compatible-api)
    - [Ollama Chat](#ollama-chat)
    - [Ollama Generate](#ollama-generate)
    - [Ollama Tags (List Models)](#ollama-tags-list-models)
    - [Ollama Show (Model Info)](#ollama-show-model-info)
    - [Ollama Root](#ollama-root)
11. [Anthropic-Compatible API](#anthropic-compatible-api)
    - [Anthropic Messages](#anthropic-messages)
    - [Anthropic Count Tokens](#anthropic-count-tokens)
12. [SageMaker Compatibility](#sagemaker-compatibility)
    - [SageMaker Health (Ping)](#sagemaker-health-ping)
    - [SageMaker Invocations](#sagemaker-invocations)
13. [Vertex AI Compatibility](#vertex-ai-compatibility)
    - [Vertex Generate](#vertex-generate)
14. [gRPC API](#grpc-api)
15. [Offline Engine API (Python)](#offline-engine-api-python)
16. [Sampling Parameters Reference](#sampling-parameters-reference)
17. [Structured Outputs](#structured-outputs)
18. [Custom Logit Processor](#custom-logit-processor)
19. [Streaming Protocol](#streaming-protocol)
20. [LoRA Adapters in Requests](#lora-adapters-in-requests)
21. [Model Thinking / Reasoning](#model-thinking--reasoning)
22. [Request / Response Formats](#request--response-formats)
23. [Error Handling](#error-handling)
24. [Authentication](#authentication)

---

## API Overview

SGLang provides seven API interfaces:

| Interface | Purpose | Protocol | Best For |
|-----------|---------|----------|----------|
| OpenAI API | Chat, completions, embeddings, vision, responses | HTTP/REST | Drop-in OpenAI replacement, production use |
| Native API | Generation, scoring, management, debugging | HTTP/REST | Advanced control, non-OpenAI tasks |
| Offline Engine | Batch inference | Python API | Script-based batch processing, RL training |
| Ollama API | CLI and Python client compatibility | HTTP/REST | Local development, Ollama ecosystem |
| Anthropic API | Anthropic SDK compatibility | HTTP/REST | Applications using Anthropic SDK |
| SageMaker API | AWS SageMaker endpoint compatibility | HTTP/REST | SageMaker deployments |
| gRPC API | High-performance RPC | gRPC/protobuf | Low-latency production systems, typed clients |

### Base URLs

| API | Base URL | Default Port |
|-----|----------|-------------|
| OpenAI API | `http://host:port/v1/` | 30000 |
| Native API | `http://host:port/` | 30000 |
| Ollama API | `http://host:port/` | 30000 |
| Anthropic API | `http://host:port/v1/` | 30000 |
| gRPC API | `http://host:port/` | 50051 |

### Interactive Documentation

When the server is running, interactive API documentation is available at:
- **Swagger UI**: `http://localhost:30000/docs`
- **ReDoc**: `http://localhost:30000/redoc`
- **OpenAPI Spec**: `http://localhost:30000/openapi.json`

---

## OpenAI-Compatible APIs

SGLang implements the OpenAI API specification, enabling drop-in replacement for OpenAI services.
You can use the official OpenAI Python SDK, any OpenAI-compatible client, or direct HTTP requests.

### Client Setup

```python
import openai

client = openai.Client(
    base_url="http://127.0.0.1:30000/v1",
    api_key="None"  # API key is optional unless configured
)
```

### Authentication

If the server was launched with `--api-key`, include it in requests:

```python
client = openai.Client(
    base_url="http://127.0.0.1:30000/v1",
    api_key="your-api-key"
)
```

Management endpoints that are marked `ADMIN_OPTIONAL` require the admin API key
(set via `--admin-api-key`) if that flag was provided at launch. Requests to
these endpoints must include the header:

```
Authorization: Bearer <admin-api-key>
```

---

## Chat Completions

### Endpoint

```
POST /v1/chat/completions
```

### Description

Generate chat completions given a conversation history. The server automatically applies the
chat template from the HuggingFace tokenizer (or a custom template specified with
`--chat-template`).

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `string` | `"default"` | Model name or `model:adapter` for LoRA |
| `messages` | `array` | (required) | Array of message objects |
| `temperature` | `float` | `None` | Sampling temperature (0 = greedy). Falls back to model config, then 1.0 |
| `top_p` | `float` | `None` | Top-p (nucleus) sampling. Falls back to model config, then 1.0 |
| `top_k` | `int` | `None` | Top-k sampling. Falls back to model config, then -1 (disabled) |
| `min_p` | `float` | `None` | Min-p sampling threshold. Falls back to model config, then 0.0 |
| `max_tokens` | `int` | `None` | Maximum tokens to generate (deprecated, use `max_completion_tokens`) |
| `max_completion_tokens` | `int` | `None` | Maximum completion tokens including reasoning tokens |
| `stream` | `bool` | `false` | Enable streaming (SSE) response |
| `stream_options` | `object` | `null` | Streaming options (`include_usage`, `continuous_usage_stats`) |
| `stop` | `string or array` | `null` | Stop sequences |
| `stop_token_ids` | `array of ints` | `null` | Stop sequences as token IDs |
| `stop_regex` | `string or array` | `null` | Stop when matching any regex pattern |
| `presence_penalty` | `float` | `0.0` | Presence penalty (-2 to 2) |
| `frequency_penalty` | `float` | `0.0` | Frequency penalty (-2 to 2) |
| `repetition_penalty` | `float` | `None` | Scale logits of previous tokens. Falls back to 1.0 |
| `n` | `int` | `1` | Number of completions to generate |
| `seed` | `int` | `null` | Random seed for reproducibility |
| `logit_bias` | `object` | `null` | Token ID to bias mapping (-100 to 100) |
| `logprobs` | `bool` | `false` | Return log probabilities |
| `top_logprobs` | `int` | `null` | Number of top logprobs to return (0-20) |
| `response_format` | `object` | `null` | Output format: `{"type": "text"}`, `{"type": "json_object"}`, `{"type": "json_schema", "json_schema": {...}}` |
| `tools` | `array` | `null` | Tool definitions for function calling |
| `tool_choice` | `string or object` | `"auto"` | Tool choice: `"auto"`, `"required"`, `"none"`, or `{"type":"function","function":{"name":"..."}}` |
| `parallel_tool_calls` | `bool` | `true` | Allow parallel tool calls |
| `user` | `string` | `null` | User identifier |
| `reasoning_effort` | `string` | `null` | Reasoning effort: `"none"`, `"low"`, `"medium"`, `"high"`. `"none"` disables reasoning |
| `min_tokens` | `int` | `0` | Force minimum generation length |
| `regex` | `string` | `null` | Regex pattern for constrained output |
| `ebnf` | `string` | `null` | EBNF grammar for constrained output |
| `no_stop_trim` | `bool` | `false` | Do not trim stop words from output |
| `ignore_eos` | `bool` | `false` | Do not stop on EOS token |
| `continue_final_message` | `bool` | `false` | Continue the final assistant message |
| `skip_special_tokens` | `bool` | `true` | Remove special tokens from output |

### Extra Body Parameters (SGLang-specific)

These are passed via the `extra_body` parameter in the OpenAI SDK:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chat_template_kwargs` | `object` | `{}` | Arguments passed to chat template processor |
| `separate_reasoning` | `bool` | `true` | Separate reasoning content in response |
| `stream_reasoning` | `bool` | `true` | Stream reasoning tokens separately |
| `custom_logit_processor` | `string` | `null` | Serialized custom logit processor |
| `custom_params` | `object` | `null` | Parameters for custom logit processor |
| `return_hidden_states` | `bool` | `false` | Return hidden states |
| `return_routed_experts` | `bool` | `false` | Return MoE expert routing data |
| `return_cached_tokens_details` | `bool` | `false` | Return cached token details |
| `lora_path` | `string` | `null` | LoRA adapter name (deprecated, use `model:adapter`) |
| `session_params` | `object` | `null` | Session parameters for continual prompting |
| `routed_dp_rank` | `int` | `null` | Route to specific data-parallel worker |
| `cache_salt` | `string` | `null` | Cache salt for request caching |
| `priority` | `int` | `null` | Request priority |
| `rid` | `string` | `null` | Custom request ID |
| `extra_key` | `string` | `null` | Extra key for request classification |

### Message Format

Standard roles: `system`, `developer`, `user`, `assistant`, `tool`, `function`.

Text-only message:

```json
{
  "role": "user",
  "content": "What is the capital of France?"
}
```

Vision message with image:

```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "What is in this image?"},
    {"type": "image_url", "image_url": {"url": "https://example.com/image.png", "detail": "auto"}}
  ]
}
```

Vision message with video:

```json
{
  "role": "user",
  "content": [
    {"type": "video_url", "video_url": {"url": "https://example.com/video.mp4"}}
  ]
}
```

Audio message:

```json
{
  "role": "user",
  "content": [
    {"type": "audio_url", "audio_url": {"url": "https://example.com/audio.mp3"}}
  ]
}
```

Tool reference message (GLM-specific):

```json
{
  "role": "assistant",
  "content": [
    {"type": "tool_reference", "name": "get_weather"}
  ]
}
```

### Image URL Formats

The `url` field in `image_url` supports:
- HTTP/HTTPS URL: `"https://example.com/image.png"`
- Local file path: `"file:///path/to/image.png"`
- Base64 data URI: `"data:image/png;base64,iVBOR..."`

The `detail` field supports: `"auto"` (default), `"low"`, `"high"`.

### Example: Basic Chat Completion

```python
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    temperature=0,
    max_tokens=128,
)

print(response.choices[0].message.content)
```

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "temperature": 0,
    "max_tokens": 128
  }'
```

### Example: Multi-turn Conversation

```python
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a knowledgeable historian."},
        {"role": "user", "content": "Tell me about ancient Rome"},
        {"role": "assistant", "content": "Ancient Rome was a civilization centered in Italy."},
        {"role": "user", "content": "What were their major achievements?"},
    ],
    temperature=0.3,
    max_tokens=256,
)
```

### Example: Streaming

```python
stream = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Example: Logit Bias

```python
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Complete: The weather is"}],
    logit_bias={
        "12345": 50,   # Increase likelihood of token 12345
        "67890": -100, # Block token 67890 entirely
    },
    max_tokens=20,
)
```

### Example: JSON Schema Output

```python
import json

json_schema = {
    "type": "json_schema",
    "json_schema": {
        "name": "city_info",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "population": {"type": "integer"},
                "country": {"type": "string"},
            },
            "required": ["name", "population", "country"],
            "additionalProperties": False,
        },
    },
}

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Tell me about Paris"}],
    response_format=json_schema,
)
```

### Example: Tool Calling

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "What is the weather in Paris?"}],
    tools=tools,
    tool_choice="auto",
)

# Check if model wants to call a tool
if response.choices[0].finish_reason == "tool_calls":
    tool_call = response.choices[0].message.tool_calls[0]
    print(f"Function: {tool_call.function.name}")
    print(f"Arguments: {tool_call.function.arguments}")
```

### Example: LoRA Adapter

```python
# Using model:adapter syntax (recommended)
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct:my_adapter",
    messages=[{"role": "user", "content": "Hello!"}],
)

# Using extra_body (backward compatible)
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
    extra_body={"lora_path": "my_adapter"},
)
```

### Response Format

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Paris is the capital of France."
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 8,
    "total_tokens": 18
  }
}
```

### Response with Tool Calls

```json
{
  "id": "chatcmpl-456",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_abc123",
        "type": "function",
        "index": 0,
        "function": {
          "name": "get_weather",
          "arguments": "{\"location\": \"Paris\", \"unit\": \"celsius\"}"
        }
      }]
    },
    "finish_reason": "tool_calls"
  }],
  "usage": {
    "prompt_tokens": 50,
    "completion_tokens": 20,
    "total_tokens": 70
  }
}
```

### Response with Reasoning Content

```json
{
  "id": "chatcmpl-789",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "Qwen/Qwen3-4B",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "reasoning_content": "Let me think about this. The user is asking about...",
      "content": "The answer is 42."
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 100,
    "total_tokens": 110,
    "reasoning_tokens": 80
  }
}
```

### Response with SGLang Extensions

When `return_routed_experts` or `return_cached_tokens_details` is enabled:

```json
{
  "id": "chatcmpl-101",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "deepseek-ai/DeepSeek-V3",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Hello"},
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 1,
    "total_tokens": 6,
    "prompt_tokens_details": {"cached_tokens": 3}
  },
  "sglext": {
    "routed_experts": "<base64-encoded int32 array>",
    "cached_tokens_details": {
      "device": 3,
      "host": 0
    }
  }
}
```

### Streaming Response Format

Each chunk in the stream:

```
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"meta-llama/Llama-3.1-8B-Instruct","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"meta-llama/Llama-3.1-8B-Instruct","choices":[{"index":0,"delta":{"content":"Paris"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"meta-llama/Llama-3.1-8B-Instruct","choices":[{"index":0,"delta":{"content":" is"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"meta-llama/Llama-3.1-8B-Instruct","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

Streaming with reasoning content:

```
data: {"id":"chatcmpl-789","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"reasoning_content":"Let me think"},"finish_reason":null}]}

data: {"id":"chatcmpl-789","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"reasoning_content":" about this."},"finish_reason":null}]}

data: {"id":"chatcmpl-789","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"The answer"},"finish_reason":null}]}

data: [DONE]
```

---

## Text Completions

### Endpoint

```
POST /v1/completions
```

### Description

Generate text completions given a prompt. Unlike chat completions, this endpoint does not apply
chat templates.

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `string` | `"default"` | Model name or `model:adapter` for LoRA |
| `prompt` | `string, array of strings, array of ints, or array of array of ints` | (required) | Input prompt(s) or token ID(s) |
| `temperature` | `float` | `1.0` | Sampling temperature |
| `top_p` | `float` | `1.0` | Top-p sampling |
| `top_k` | `int` | `-1` | Top-k sampling (-1 = disabled) |
| `min_p` | `float` | `0.0` | Min-p sampling threshold |
| `max_tokens` | `int` | `16` | Maximum tokens to generate. Must be positive |
| `stream` | `bool` | `false` | Enable streaming response |
| `stream_options` | `object` | `null` | Streaming options (`include_usage`, `continuous_usage_stats`) |
| `stop` | `string or array` | `null` | Stop sequences |
| `stop_token_ids` | `array of ints` | `null` | Stop sequences as token IDs |
| `stop_regex` | `string or array` | `null` | Stop when matching any regex pattern |
| `presence_penalty` | `float` | `0.0` | Presence penalty (-2 to 2) |
| `frequency_penalty` | `float` | `0.0` | Frequency penalty (-2 to 2) |
| `repetition_penalty` | `float` | `1.0` | Scale logits of previous tokens |
| `n` | `int` | `1` | Number of completions per prompt |
| `best_of` | `int` | `null` | Not fully supported; kept for API compatibility |
| `seed` | `int` | `null` | Random seed |
| `logit_bias` | `object` | `null` | Token ID to bias mapping |
| `logprobs` | `int` | `null` | Number of top logprobs to return |
| `echo` | `bool` | `false` | Echo the prompt in the response |
| `suffix` | `string` | `null` | Suffix to append to completion |
| `user` | `string` | `null` | User identifier |
| `json_schema` | `string` | `null` | JSON schema for structured output |
| `regex` | `string` | `null` | Regex for structured output |
| `ebnf` | `string` | `null` | EBNF grammar for structured output |
| `min_tokens` | `int` | `0` | Force minimum generation length |
| `ignore_eos` | `bool` | `false` | Do not stop on EOS token |
| `skip_special_tokens` | `bool` | `true` | Remove special tokens from output |
| `no_stop_trim` | `bool` | `false` | Do not trim stop words from output |
| `response_format` | `object` | `null` | Output format (JSON schema, etc.) |
| `lora_path` | `string` | `null` | LoRA adapter path (deprecated) |
| `custom_logit_processor` | `string` | `null` | Serialized custom logit processor |
| `custom_params` | `object` | `null` | Parameters for custom logit processor |
| `session_params` | `object` | `null` | Session parameters for continual prompting |
| `return_hidden_states` | `bool` | `false` | Return hidden states |
| `return_routed_experts` | `bool` | `false` | Return MoE expert routing data |
| `return_cached_tokens_details` | `bool` | `false` | Return cached token details |

### Example: Basic Completion

```python
response = client.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    prompt="The capital of France is",
    temperature=0,
    max_tokens=64,
)

print(response.choices[0].text)
```

```bash
curl http://localhost:30000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "The capital of France is",
    "temperature": 0,
    "max_tokens": 64
  }'
```

### Example: Batch Completions

```python
response = client.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    prompt=["The capital of France is", "The capital of Germany is"],
    temperature=0,
    max_tokens=32,
)

for choice in response.choices:
    print(f"Prompt {choice.index}: {choice.text}")
```

### Example: Streaming Completion

```python
stream = client.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    prompt="Once upon a time",
    stream=True,
    max_tokens=100,
)

for chunk in stream:
    if chunk.choices[0].text:
        print(chunk.choices[0].text, end="", flush=True)
```

### Response Format

```json
{
  "id": "cmpl-123",
  "object": "text_completion",
  "created": 1677652288,
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "choices": [{
    "text": " Paris, the largest city in France.",
    "index": 0,
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 8,
    "total_tokens": 13
  }
}
```

---

## Embeddings

### Endpoint

```
POST /v1/embeddings
```

### Description

Generate embeddings for text input. Requires the server to be launched with `--is-embedding`.

### Server Setup

```bash
python3 -m sglang.launch_server \
    --model-path Alibaba-NLP/gte-Qwen2-1.5B-instruct \
    --is-embedding
```

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `string` | `"default"` | Model name |
| `input` | `string, array of strings, array of ints, array of array of ints, or array of MultimodalEmbeddingInput` | (required) | Text(s), token ID(s), or multimodal input to embed |
| `encoding_format` | `string` | `"float"` | Response format: `"float"` or `"base64"` |
| `dimensions` | `int` | `null` | Output dimensions (if supported by model) |
| `user` | `string` | `null` | User identifier |
| `rid` | `string or array` | `null` | Request ID(s) |
| `priority` | `int` | `null` | Request priority |
| `lora_path` | `string or array` | `null` | LoRA adapter path(s) |

### Multimodal Embedding Input

For multimodal embedding models, input can be a list of objects:

```json
{
  "input": [
    {"text": "a photo of a cat", "image": "base64_encoded_image"}
  ]
}
```

### Example: Using OpenAI SDK

```python
client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")

response = client.embeddings.create(
    model="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    input="Once upon a time",
)

embedding = response.data[0].embedding
print(f"Embedding dimension: {len(embedding)}")
print(f"First 10 values: {embedding[:10]}")
```

### Example: Using cURL

```bash
curl http://localhost:30000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    "input": "Once upon a time"
  }'
```

### Example: Batch Embeddings

```python
response = client.embeddings.create(
    model="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    input=["Once upon a time", "In a galaxy far away"],
)

for item in response.data:
    print(f"Index {item.index}: dimension {len(item.embedding)}")
```

### Response Format

```json
{
  "object": "list",
  "data": [{
    "object": "embedding",
    "embedding": [0.0023, -0.0094, ...],
    "index": 0
  }],
  "model": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
  "usage": {
    "prompt_tokens": 4,
    "total_tokens": 4
  }
}
```

---

## Rerank

### Endpoint

```
POST /v1/rerank
```

### Description

Rerank documents given a query using a cross-encoder model. Supports both text-only and
multimodal (image/video) reranking.

### Server Setup

```bash
python3 -m sglang.launch_server \
    --model-path BAAI/bge-reranker-v2-m3 \
    --disable-radix-cache \
    --chunked-prefill-size -1 \
    --attention-backend triton \
    --is-embedding
```

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `string or array of content parts` | (required) | Query text or multimodal content parts |
| `documents` | `array of strings or array of content part arrays` | (required) | Documents to rerank |
| `instruct` | `string` | `null` | Instruction to the reranker model |
| `top_n` | `int` | `null` | Maximum documents to return (null = all) |
| `return_documents` | `bool` | `true` | Whether to return document text in response |

### Multimodal Content Parts

Each content part can be:

```json
{"type": "text", "text": "description text"}
{"type": "image_url", "image_url": {"url": "https://..."}}
{"type": "video_url", "video_url": {"url": "https://..."}}
```

### Example: Text Reranking

```python
import requests

response = requests.post(
    "http://localhost:30000/v1/rerank",
    json={
        "query": "what is panda?",
        "documents": [
            "hi",
            "The giant panda is a bear species endemic to China.",
            "Pandas eat bamboo.",
        ],
        "top_n": 2,
    },
)

for item in response.json():
    print(f"Score: {item['score']:.2f} - Index: {item['index']} - Document: '{item['document']}'")
```

### Example: Multimodal Reranking

```python
response = requests.post(
    "http://localhost:30000/v1/rerank",
    json={
        "query": [
            {"type": "text", "text": "Find images of pandas"},
            {"type": "image_url", "image_url": {"url": "https://example.com/panda.png"}}
        ],
        "documents": [
            [{"type": "text", "text": "A panda eating bamboo"}],
            [{"type": "text", "text": "A bear in the forest"}],
        ],
    },
)
```

### Response Format

```json
[
  {
    "score": 0.95,
    "document": "The giant panda is a bear species endemic to China.",
    "index": 1
  },
  {
    "score": 0.85,
    "document": "Pandas eat bamboo.",
    "index": 2
  }
]
```

When `return_documents` is `false`, the `document` field is omitted.

---

## Score

### Endpoint

```
POST /v1/score
```

### Description

Compute token probabilities for specified tokens given a query and items. Useful for
classification, response scoring, and log-probability computation. Supports both
CausalLM (logprob-based) and SequenceClassification (class logit-based) models.

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `string` | `"default"` | Model name |
| `query` | `string or array of ints` | (required) | Query text or pre-tokenized token IDs |
| `items` | `string, array of strings, or array of array of ints` | (required) | Item texts or pre-tokenized token IDs to score |
| `label_token_ids` | `array of ints` | (required) | Token IDs to compute probabilities for |
| `apply_softmax` | `bool` | `false` | Apply softmax for normalized probabilities |
| `item_first` | `bool` | `false` | Items come first in concatenation |
| `return_pooled_hidden_states` | `bool` | `false` | Return pooled hidden states |

### Example

```python
response = requests.post(
    "http://localhost:30000/v1/score",
    json={
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "query": "The capital of France is",
        "items": ["Paris", "London", "Berlin"],
        "label_token_ids": [9454, 2753],  # "Yes" and "No" token IDs
        "apply_softmax": True,
    },
)

for item, scores in zip(["Paris", "London", "Berlin"], response.json()["scores"]):
    print(f"Item '{item}': probabilities = {[f'{s:.4f}' for s in scores]}")
```

### Response Format

```json
{
  "scores": [
    [0.82, 0.18],
    [0.45, 0.55],
    [0.30, 0.70]
  ],
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "usage": {
    "prompt_tokens": 15,
    "total_tokens": 15
  },
  "object": "scoring"
}
```

---

## Classify V1

### Endpoint

```
POST /v1/classify
```

### Description

Classify text using a SequenceClassification model. Returns label probabilities per input.

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `string` | `"default"` | Model name |
| `input` | `string or array of strings or array of ints` | (required) | Text(s) or token IDs to classify |
| `user` | `string` | `null` | User identifier |
| `rid` | `string or array` | `null` | Request ID(s) |
| `priority` | `int` | `null` | Request priority |

### Response Format

```json
{
  "id": "classify-123",
  "object": "list",
  "created": 1677652288,
  "model": "model-name",
  "data": [
    {
      "index": 0,
      "label": "POSITIVE",
      "probs": [0.15, 0.85],
      "num_classes": 2
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "total_tokens": 10
  }
}
```

---

## Tokenize

### Endpoint

```
POST /v1/tokenize
POST /tokenize
```

### Description

Tokenize text into token IDs. The `/tokenize` endpoint (without `/v1` prefix) is also
supported but hidden from the OpenAPI schema.

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `string` | `"default"` | Model name |
| `prompt` | `string or array of strings` | (required) | Text to tokenize |
| `add_special_tokens` | `bool` | `true` | Add model-specific special tokens (e.g. BOS/EOS) |

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `tokens` | `array of ints or array of array of ints` | Token IDs |
| `count` | `int or array of ints` | Number of tokens |
| `max_model_len` | `int` | Maximum model context length |

### Example

```python
response = requests.post(
    "http://localhost:30000/v1/tokenize",
    json={"model": "model-name", "prompt": "Hello world", "add_special_tokens": False},
)
print(f"Token IDs: {response.json()['tokens']}")
print(f"Token count: {response.json()['count']}")
print(f"Max model length: {response.json()['max_model_len']}")
```

### Response Format

```json
{
  "tokens": [15496, 995],
  "count": 2,
  "max_model_len": 131072
}
```

---

## Detokenize

### Endpoint

```
POST /v1/detokenize
POST /detokenize
```

### Description

Convert token IDs back to text.

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `string` | `"default"` | Model name |
| `tokens` | `array of ints or array of array of ints` | (required) | Token IDs to decode |
| `skip_special_tokens` | `bool` | `true` | Skip special tokens in output |

### Example

```python
# Tokenize then detokenize
tokenize_response = requests.post(
    "http://localhost:30000/v1/tokenize",
    json={"prompt": "Hello world"},
)
token_ids = tokenize_response.json()["tokens"]

detokenize_response = requests.post(
    "http://localhost:30000/v1/detokenize",
    json={"tokens": token_ids},
)
print(detokenize_response.json()["text"])
```

### Response Format

```json
{
  "text": "Hello world"
}
```

---

## Audio Transcriptions

### Endpoint

```
POST /v1/audio/transcriptions
```

### Description

Transcribe audio files to text. Supports JSON, text, and verbose JSON response formats.
Uses multipart form data (not JSON).

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | `UploadFile` | (required) | Audio file (multipart form upload) |
| `model` | `string` | `"default"` | Model name |
| `language` | `string` | `null` | Language code (e.g., "en") |
| `response_format` | `string` | `"json"` | Response format: `"json"`, `"text"`, or `"verbose_json"` |
| `temperature` | `float` | `0.0` | Sampling temperature |
| `stream` | `bool` | `false` | Enable streaming response |
| `timestamp_granularities[]` | `array of strings` | `null` | Timestamp granularities for verbose_json |

### Example: cURL

```bash
curl http://localhost:30000/v1/audio/transcriptions \
  -F "file=@recording.wav" \
  -F "model=default" \
  -F "language=en" \
  -F "response_format=json"
```

### Response Format (json)

```json
{
  "text": "Hello, this is a transcription of the audio file."
}
```

### Response Format (verbose_json)

```json
{
  "task": "transcribe",
  "language": "en",
  "duration": 5.2,
  "text": "Hello, this is a transcription.",
  "segments": [
    {"id": 0, "start": 0.0, "end": 2.5, "text": "Hello, this is"},
    {"id": 1, "start": 2.5, "end": 5.2, "text": " a transcription."}
  ],
  "usage": {"type": "duration", "seconds": 6}
}
```

---

## Responses API

### Endpoints

```
POST /v1/responses
GET /v1/responses/{response_id}
POST /v1/responses/{response_id}/cancel
```

### Description

OpenAI-compatible Responses API for generating responses with reasoning support. Supports
both synchronous and background (async) processing modes.

### Request Parameters (POST /v1/responses)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | `string or array` | (required) | Input text or structured input items |
| `model` | `string` | `null` | Model name |
| `instructions` | `string` | `null` | System-level instructions |
| `max_output_tokens` | `int` | `null` | Maximum output tokens |
| `max_tool_calls` | `int` | `null` | Maximum tool calls per turn |
| `metadata` | `object` | `null` | Custom metadata |
| `parallel_tool_calls` | `bool` | `true` | Allow parallel tool calls |
| `previous_response_id` | `string` | `null` | Previous response ID for conversation continuity |
| `reasoning` | `object` | `null` | Reasoning configuration: `{"effort": "low"|"medium"|"high"}` |
| `service_tier` | `string` | `"auto"` | Service tier |
| `store` | `bool` | `true` | Whether to store the response |
| `stream` | `bool` | `false` | Enable streaming response |
| `temperature` | `float` | `null` | Sampling temperature (default 0.7) |
| `tool_choice` | `string` | `"auto"` | Tool choice: `"auto"`, `"required"`, `"none"` |
| `tools` | `array` | `[]` | Tool definitions |
| `top_logprobs` | `int` | `0` | Number of top logprobs |
| `top_p` | `float` | `null` | Top-p sampling |
| `truncation` | `string` | `"disabled"` | Truncation mode: `"auto"` or `"disabled"` |
| `user` | `string` | `null` | User identifier |
| `background` | `bool` | `false` | Run in background, return immediately |
| `include` | `array` | `null` | Additional data to include in response |

SGLang-specific extra parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `request_id` | `string` | auto-generated | Custom request ID |
| `priority` | `int` | `0` | Request priority |
| `extra_key` | `string` | `null` | Extra key for request classification |
| `cache_salt` | `string` | `null` | Cache salt |
| `frequency_penalty` | `float` | `0.0` | Frequency penalty |
| `presence_penalty` | `float` | `0.0` | Presence penalty |
| `stop` | `string or array` | `null` | Stop sequences |
| `top_k` | `int` | `-1` | Top-k sampling |
| `min_p` | `float` | `0.0` | Min-p threshold |
| `repetition_penalty` | `float` | `1.0` | Repetition penalty |

### Example: Basic Response

```python
response = requests.post(
    "http://localhost:30000/v1/responses",
    json={
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "input": "What is the capital of France?",
    },
)
print(response.json()["output"])
```

### Example: Background Response

```python
# Start a background response
response = requests.post(
    "http://localhost:30000/v1/responses",
    json={
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "input": "Write a long essay about AI",
        "background": True,
    },
)
response_id = response.json()["id"]

# Poll for completion
status = requests.get(f"http://localhost:30000/v1/responses/{response_id}").json()

# Cancel if needed
requests.post(f"http://localhost:30000/v1/responses/{response_id}/cancel")
```

### Example: With Reasoning

```python
response = requests.post(
    "http://localhost:30000/v1/responses",
    json={
        "model": "Qwen/Qwen3-4B",
        "input": "Solve: What is 15 * 37?",
        "reasoning": {"effort": "high"},
    },
)
```

### Response Format

```json
{
  "id": "resp_abc123",
  "object": "response",
  "created_at": 1677652288,
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "status": "completed",
  "output": [
    {
      "type": "message",
      "role": "assistant",
      "content": [
        {"type": "output_text", "text": "The capital of France is Paris."}
      ]
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 8,
    "total_tokens": 18
  },
  "parallel_tool_calls": true,
  "tool_choice": "auto",
  "tools": [],
  "instructions": null,
  "max_output_tokens": null,
  "temperature": null,
  "top_p": null,
  "truncation": "disabled",
  "text": {"format": {"type": "text"}},
  "metadata": {}
}
```

### Retrieve Response (GET /v1/responses/{response_id})

Returns the current state of a background response. Uses the same response format as above.

### Cancel Response (POST /v1/responses/{response_id}/cancel)

Cancels a background response. Returns the response with `status: "cancelled"`.

---

## List Models

### Endpoint

```
GET /v1/models
```

### Description

Show available models including the base model and any loaded LoRA adapters.

### Example

```python
response = client.models.list()
for model in response.data:
    print(f"Model: {model.id}")
```

```bash
curl http://localhost:30000/v1/models
```

### Response Format

```json
{
  "object": "list",
  "data": [
    {
      "id": "meta-llama/Llama-3.1-8B-Instruct",
      "object": "model",
      "created": 1677652288,
      "owned_by": "sglang",
      "root": "meta-llama/Llama-3.1-8B-Instruct",
      "max_model_len": 131072
    },
    {
      "id": "my_adapter",
      "object": "model",
      "created": 1677652288,
      "owned_by": "sglang",
      "root": "/path/to/adapter",
      "parent": "meta-llama/Llama-3.1-8B-Instruct"
    }
  ]
}
```

---

## Retrieve Model

### Endpoint

```
GET /v1/models/{model}
```

### Description

Retrieves a model instance, providing basic information.

### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `string` | Model name (must match the served model name) |

### Example

```bash
curl http://localhost:30000/v1/models/meta-llama/Llama-3.1-8B-Instruct
```

### Response Format

```json
{
  "id": "meta-llama/Llama-3.1-8B-Instruct",
  "object": "model",
  "created": 1677652288,
  "owned_by": "sglang",
  "root": "meta-llama/Llama-3.1-8B-Instruct",
  "max_model_len": 131072
}
```

### Error Response (Model Not Found)

```json
{
  "error": {
    "message": "The model 'nonexistent-model' does not exist",
    "type": "invalid_request_error",
    "param": "model",
    "code": "model_not_found"
  }
}
```

---

## Native SGLang APIs

These endpoints provide lower-level access to the SGLang runtime beyond the OpenAI API.

---

## Generate

### Endpoint

```
POST /generate
PUT /generate
```

### Description

Generate text completions with fine-grained control over sampling and output. Similar to
`/v1/completions` but with more SGLang-specific options including multimodal input,
input embeddings, and advanced logprob controls.

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `string or array of strings` | `null` | Input prompt(s). Mutually exclusive with `input_ids` and `input_embeds`. |
| `input_ids` | `list of ints or list of list of ints` | `null` | Token ID input(s). |
| `input_embeds` | `list of list of floats or list of list of list of floats` | `null` | Embedding input(s). |
| `image_data` | `string, list, or nested list` | `null` | Image input(s). Supports file path, URL, base64, processor output, or precomputed embeddings. |
| `video_data` | `string, list, or nested list` | `null` | Video input(s). |
| `audio_data` | `string or list` | `null` | Audio input(s). |
| `use_audio_in_video` | `bool` | `false` | Extract and process audio from video inputs. |
| `sampling_params` | `object or list of objects` | `{}` | Sampling parameters (see Sampling Parameters Reference). |
| `rid` | `string or list of strings` | `null` | Request ID(s). |
| `return_logprob` | `bool or list of bools` | `false` | Return log probabilities. |
| `logprob_start_len` | `int or list of ints` | `-1` | Start position for logprobs. `-1` = output tokens only. |
| `top_logprobs_num` | `int or list of ints` | `null` | Number of top logprobs per position. |
| `token_ids_logprob` | `list of ints or list of list of ints` | `null` | Specific token IDs to get logprobs for. |
| `return_text_in_logprobs` | `bool` | `false` | Include text in logprob output. |
| `stream` | `bool` | `false` | Enable streaming (SSE) response. |
| `lora_path` | `string or list` | `null` | LoRA adapter path or name. |
| `custom_logit_processor` | `string or list` | `null` | Serialized custom logit processor. |
| `return_hidden_states` | `bool or list of bools` | `false` | Return hidden states. |
| `return_routed_experts` | `bool` | `false` | Return MoE expert routing data. |
| `routed_experts_start_len` | `int` | `0` | Start position for routed experts. |
| `modalities` | `list of strings` | `null` | Modalities for image data: `["image"]`, `["multi-images"]`, `["video"]` |
| `session_params` | `object or list of objects` | `null` | Session parameters for continual prompting. |
| `bootstrap_host` | `string or list` | `null` | Bootstrap host for PD disaggregation. |
| `bootstrap_port` | `int or list` | `null` | Bootstrap port for PD disaggregation. |
| `bootstrap_room` | `int or list` | `null` | Bootstrap room for PD disaggregation. |
| `routed_dp_rank` | `int` | `null` | Route to specific DP worker. |
| `disagg_prefill_dp_rank` | `int` | `null` | Hint for decode about prefill DP worker. |
| `require_reasoning` | `bool` | `false` | Require reasoning for hybrid reasoning models. |
| `background` | `bool` | `false` | Run in background mode. |
| `conversation_id` | `string` | `null` | Conversation ID for tracking. |
| `priority` | `int` | `null` | Request priority. |
| `extra_key` | `string or list` | `null` | Extra key for request classification. |
| `routing_key` | `string` | `null` | Routing key for schedule policy. |
| `return_entropy` | `bool` | `false` | Return entropy. |

### Example: Basic Generation

```python
import requests

response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "The capital of France is",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 32,
        },
    },
)
print(response.json())
```

### Example: Streaming Generation

```python
import requests
import json

response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "The capital of France is",
        "sampling_params": {"temperature": 0, "max_new_tokens": 32},
        "stream": True,
    },
    stream=True,
)

prev = 0
for chunk in response.iter_lines(decode_unicode=False):
    chunk = chunk.decode("utf-8")
    if chunk and chunk.startswith("data:"):
        if chunk == "data: [DONE]":
            break
        data = json.loads(chunk[5:].strip("\n"))
        output = data["text"]
        print(output[prev:], end="", flush=True)
        prev = len(output)
```

### Example: Multimodal Generation

```python
response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "<|im_start|>user\n<image>\nDescribe this image.<|im_end|>\n<|im_start|>assistant\n",
        "image_data": "example_image.png",
        "sampling_params": {"temperature": 0, "max_new_tokens": 32},
    },
)
```

### Example: Batch Generation

```python
response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": ["Hello, my name is", "The capital of France is"],
        "sampling_params": {"temperature": 0.7, "max_new_tokens": 64},
    },
)
```

### Example: Generation with Logprobs

```python
response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "The capital of France is",
        "sampling_params": {"temperature": 0, "max_new_tokens": 16},
        "return_logprob": True,
        "logprob_start_len": -1,
        "top_logprobs_num": 5,
        "return_text_in_logprobs": True,
    },
)
meta_info = response.json()["meta_info"]
print("Token logprobs:", meta_info.get("input_token_logprobs"))
print("Output logprobs:", meta_info.get("output_token_logprobs"))
```

### Response Format (Non-streaming)

```json
{
  "text": " Paris, the largest city in France and one of the most important cities in Europe.",
  "meta_info": {
    "id": "req-123",
    "finish_reason": {
      "type": "stop",
      "matched": null
    },
    "prompt_tokens": 5,
    "completion_tokens": 18,
    "cached_tokens": 0
  }
}
```

### Response Format (Streaming)

```
data: {"text":" Paris","meta_info":{"id":"req-123","finish_reason":null,"prompt_tokens":5,"completion_tokens":1,"cached_tokens":0}}

data: {"text":" Paris, the largest city","meta_info":{"id":"req-123","finish_reason":null,"prompt_tokens":5,"completion_tokens":4,"cached_tokens":0}}

data: {"text":" Paris, the largest city in France.","meta_info":{"id":"req-123","finish_reason":{"type":"stop","matched":"\n"},"prompt_tokens":5,"completion_tokens":18,"cached_tokens":0}}

data: [DONE]
```

---

## Encode

### Endpoint

```
POST /encode
PUT /encode
```

### Description

Encode text into embeddings. Only available when the server is launched with `--is-embedding`.

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `string or list of strings` | (required) | Text to encode |
| `model` | `string` | `null` | Model name |

### Example

```python
response = requests.post(
    "http://localhost:30000/encode",
    json={"model": "Alibaba-NLP/gte-Qwen2-1.5B-instruct", "text": "Once upon a time"},
)
print(response.json()["embedding"][:10])
```

### Response Format

```json
{
  "embedding": [0.0023, -0.0094, 0.0012, ...]
}
```

---

## Classify Native

### Endpoint

```
POST /classify
PUT /classify
```

### Description

Classify text using a reward model. The request format is the same as the embedding endpoint.
Useful for reward scoring, quality assessment, and pairwise comparison.

### Server Setup

```bash
python3 -m sglang.launch_server \
    --model-path Skywork/Skywork-Reward-Llama-3.1-8B-v0.2 \
    --is-embedding
```

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `string or list of strings` | (required) | Formatted prompt(s) from chat template |
| `model` | `string` | `null` | Model name |

### Example

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Skywork/Skywork-Reward-Llama-3.1-8B-v0.2")

CONVS = [
    [{"role": "user", "content": "What is sigmoid?"},
     {"role": "assistant", "content": "Output is between -1 and 1."}],
    [{"role": "user", "content": "What is sigmoid?"},
     {"role": "assistant", "content": "Output is between 0 and 1."}],
]

prompts = tokenizer.apply_chat_template(CONVS, tokenize=False)

response = requests.post(
    "http://localhost:30000/classify",
    json={"model": "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2", "text": prompts},
)

for r in response.json():
    print(f"Reward: {r['embedding'][0]}")
```

### Response Format

```json
[
  {"embedding": [-1.234]},
  {"embedding": [2.567]}
]
```

The reward score is the first element of the embedding array.

---

## Model Info

### Endpoint

```
GET /model_info
GET /get_model_info
```

> Note: `/get_model_info` is deprecated. Use `/model_info` instead.

### Description

Returns information about the currently loaded model.

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `model_path` | `string` | Path/name of the loaded model |
| `is_generation` | `bool` | Whether the model is a generation model |
| `tokenizer_path` | `string` | Path/name of the tokenizer |
| `preferred_sampling_params` | `object or null` | Default sampling params from `--preferred-sampling-params` |
| `weight_version` | `string` | Model weight version |
| `has_image_understanding` | `bool` | Whether the model supports image input |
| `has_audio_understanding` | `bool` | Whether the model supports audio input |
| `model_type` | `string` | Model type from HuggingFace config (e.g., "qwen2", "llama") |
| `architectures` | `list of strings` | Model architecture names |

### Example

```python
response = requests.get("http://localhost:30000/model_info")
print(response.json())
```

```bash
curl http://localhost:30000/model_info
```

### Response Format

```json
{
  "model_path": "meta-llama/Llama-3.1-8B-Instruct",
  "tokenizer_path": "meta-llama/Llama-3.1-8B-Instruct",
  "is_generation": true,
  "preferred_sampling_params": null,
  "weight_version": "default",
  "has_image_understanding": false,
  "has_audio_understanding": false,
  "model_type": "llama",
  "architectures": ["LlamaForCausalLM"]
}
```

---

## Server Info

### Endpoint

```
GET /server_info
GET /get_server_info
```

> Note: `/get_server_info` is deprecated. Use `/server_info` instead.

### Description

Returns server information including CLI arguments, token limits, memory pool sizes,
internal states, and SGLang version.

### Example

```python
response = requests.get("http://localhost:30000/server_info")
print(response.text)
```

### Response Fields

The response includes all `ServerArgs` fields plus:

| Field | Type | Description |
|-------|------|-------------|
| `version` | `string` | SGLang version |
| `internal_states` | `list` | Per-DP-rank internal states |
| `max_total_num_tokens` | `int` | Maximum total tokens across all DP ranks |
| `max_req_input_len` | `int` | Maximum request input length |

---

## Health Check

### Endpoints

```
GET /health
GET /health_generate
```

### Description

- **`/health`**: Basic health check. By default returns 200 immediately if server is running
  (no inference). Set `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=1` to also generate a token.
- **`/health_generate`**: Extended health check. Generates one token to verify the full inference
  pipeline. Times out after `SGLANG_HEALTH_CHECK_TIMEOUT` seconds (default 20).

Both endpoints return:
- **200**: Server is healthy
- **503**: Server is unhealthy (starting up, shutting down, or health check timed out)

### Example

```python
# Basic health
response = requests.get("http://localhost:30000/health")
assert response.status_code == 200

# Full pipeline health
response = requests.get("http://localhost:30000/health_generate")
assert response.status_code == 200
```

```bash
# Basic health
curl -s -o /dev/null -w "%{http_code}" http://localhost:30000/health

# Full pipeline health
curl -s -o /dev/null -w "%{http_code}" http://localhost:30000/health_generate
```

---

## Load Metrics

### Endpoint

```
GET /v1/loads
```

### Description

Returns comprehensive load metrics for all DP ranks, useful for load balancing and monitoring.

### Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dp_rank` | `int` | `null` | Filter to specific DP rank |
| `include` | `string` | `null` | Comma-separated sections: `core`, `memory`, `spec`, `lora`, `disagg`, `queues`, `all` |
| `format` | `string` | `"json"` | Response format: `"json"` or `"prometheus"` |

### Example

```bash
# All metrics
curl http://localhost:30000/v1/loads

# Memory metrics only
curl "http://localhost:30000/v1/loads?include=core,memory"

# Prometheus format
curl "http://localhost:30000/v1/loads?format=prometheus"

# Specific DP rank
curl "http://localhost:30000/v1/loads?dp_rank=0"
```

### Response Format (JSON)

```json
{
  "timestamp": "2024-01-15T10:30:00+00:00",
  "version": "0.4.0",
  "dp_rank_count": 1,
  "loads": [
    {
      "dp_rank": 0,
      "num_running_reqs": 5,
      "num_waiting_reqs": 2,
      "num_total_reqs": 7,
      "num_used_tokens": 5000,
      "num_total_tokens": 100000,
      "token_usage": 0.05,
      "gen_throughput": 120.5,
      "utilization": 0.75
    }
  ],
  "aggregate": {
    "total_running_reqs": 5,
    "total_waiting_reqs": 2,
    "total_reqs": 7,
    "total_used_tokens": 5000,
    "total_tokens": 100000,
    "avg_token_usage": 0.05,
    "avg_throughput": 120.5,
    "avg_utilization": 0.75
  }
}
```

---

## Flush Cache

### Endpoint

```
GET /flush_cache
POST /flush_cache
```

### Description

Flush the radix cache. Automatically triggered when model weights are updated.
Requires admin API key if `--admin-api-key` was set at launch.

### Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `timeout` | `float` | `0` | Wait time (seconds) for idle state before flushing. `0` = fail fast. |

### Example

```bash
# Immediate flush (fail if not idle)
curl -s -X POST "http://localhost:30000/flush_cache"

# Wait up to 30s for idle state
curl -s -X POST "http://localhost:30000/flush_cache?timeout=30"
```

### Response

Success:
```
Cache flushed.
Please check backend logs for more details. (When there are running or waiting requests, the operation will not be performed.)
```

Failure (400):
```
Flush cache failed.
```

---

## Abort Request

### Endpoint

```
POST /abort_request
```

### Description

Abort a specific request or all in-flight requests. Requires admin API key if
`--admin-api-key` was set at launch.

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rid` | `string` | `""` | Request ID to abort (ignored if `abort_all` is true) |
| `abort_all` | `bool` | `false` | Abort all in-flight requests |

### Example

```bash
# Abort specific request
curl -s -X POST http://localhost:30000/abort_request \
  -H "Content-Type: application/json" \
  -d '{"rid": "req-123"}'

# Abort all requests
curl -s -X POST http://localhost:30000/abort_request \
  -H "Content-Type: application/json" \
  -d '{"abort_all": true}'
```

---

## Update Weights from Disk

### Endpoint

```
POST /update_weights_from_disk
```

### Description

Update model weights from disk without restarting the server. Only works for models with the
same architecture and parameter size. Requires admin API key if `--admin-api-key` was set.

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | `string` | (required) | Path to new model weights |
| `load_format` | `string` | `null` | Format to load the weights |
| `abort_all_requests` | `bool` | `false` | Abort all requests before updating |
| `weight_version` | `string` | `null` | Update weight version along with weights |
| `is_async` | `bool` | `false` | Update weights asynchronously |
| `torch_empty_cache` | `bool` | `false` | Empty torch cache after update |
| `keep_pause` | `bool` | `false` | Keep scheduler paused after update |
| `recapture_cuda_graph` | `bool` | `false` | Recapture CUDA graph after update |
| `token_step` | `int` | `0` | Trainer step ID |
| `flush_cache` | `bool` | `true` | Flush cache after updating weights |

### Example

```python
response = requests.post(
    "http://localhost:30000/update_weights_from_disk",
    json={"model_path": "meta-llama/Llama-3.1-8B-Instruct"},
)
print(response.json())
# {"success": true, "message": "Succeeded to update model weights."}
```

### Response Format

Success:
```json
{
  "success": true,
  "message": "Succeeded to update model weights.",
  "num_paused_requests": 3
}
```

Failure:
```json
{
  "success": false,
  "message": "Error description",
  "num_paused_requests": 0
}
```

---

## Tokenize / Detokenize Native

These are the native versions of the tokenize and detokenize endpoints (without the `/v1`
prefix). They share the same request/response format as the OpenAI versions documented above.

### Endpoints

```
POST /tokenize
POST /detokenize
```

See [Tokenize](#tokenize) and [Detokenize](#detokenize) for full documentation.

---

## Expert Distribution

### Endpoints

```
GET /start_expert_distribution_record
POST /start_expert_distribution_record
GET /stop_expert_distribution_record
POST /stop_expert_distribution_record
GET /dump_expert_distribution_record
POST /dump_expert_distribution_record
```

### Description

Record expert selection distribution in MoE models for analysis and optimization. Requires
`--expert-distribution-recorder-mode` server flag. Requires admin API key if set.

### Example

```python
# Start recording
requests.post("http://localhost:30000/start_expert_distribution_record")

# Generate some requests to collect data
requests.post("http://localhost:30000/generate",
    json={"text": "What is the capital of France?"})

# Stop recording
requests.post("http://localhost:30000/stop_expert_distribution_record")

# Dump results
response = requests.post("http://localhost:30000/dump_expert_distribution_record")
print(response.text)
```

---

## Parse Function Call

### Endpoint

```
POST /parse_function_call
```

### Description

Parse function calls from text. Useful for extracting tool calls from model output
without running inference.

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `string` | (required) | Text to parse for function calls |
| `tools` | `array` | (required) | Tool definitions |
| `tool_call_parser` | `string` | `null` | Parser type (e.g., "pythonic", "mistral", etc.) |

### Example

```python
response = requests.post(
    "http://localhost:30000/parse_function_call",
    json={
        "text": '<tool_call({"name": "get_weather", "arguments": {"city": "Paris"}})',
        "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}],
        "tool_call_parser": "pythonic",
    },
)
print(response.json())
```

### Response Format

```json
{
  "normal_text": "",
  "calls": [
    {
      "id": "call_0",
      "type": "function",
      "function": {
        "name": "get_weather",
        "arguments": "{\"city\": \"Paris\"}"
      }
    }
  ]
}
```

---

## Separate Reasoning

### Endpoint

```
POST /separate_reasoning
```

### Description

Separate reasoning content from normal text. Useful for post-processing model output.

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `string` | (required) | Text containing mixed reasoning and content |
| `reasoning_parser` | `string` | (required) | Parser type: `"deepseek-r1"`, `"deepseek-v3"`, `"qwen3"`, `"qwen3-thinking"`, `"kimi"`, `"gpt-oss"` |

### Example

```python
response = requests.post(
    "http://localhost:30000/separate_reasoning",
    json={
        "text": "<think\>Let me think about this.</think\>The answer is 42.",
        "reasoning_parser": "qwen3",
    },
)
print(response.json())
```

### Response Format

```json
{
  "reasoning_text": "Let me think about this.",
  "text": "The answer is 42."
}
```

---

## LoRA Management API

### Server Setup

```bash
python -m sglang.launch_server \
    --model-path base-model \
    --enable-lora \
    --lora-paths adapter_a=/path/to/a adapter_b=/path/to/b
```

---

## Load LoRA Adapter

### Endpoint

```
POST /load_lora_adapter
```

### Description

Load a new LoRA adapter dynamically without re-launching the server. Requires admin API
key if `--admin-api-key` was set.

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lora_name` | `string` | (required) | Name identifier for the adapter |
| `lora_path` | `string` | (required) | Path to the LoRA adapter |
| `pinned` | `bool` | `false` | Pin the adapter in memory (prevent eviction) |

### Example

```bash
curl -s -X POST http://localhost:30000/load_lora_adapter \
  -H "Content-Type: application/json" \
  -d '{
    "lora_name": "my_adapter",
    "lora_path": "/path/to/lora/adapter",
    "pinned": false
  }'
```

### Response Format

Success:
```json
{
  "success": true,
  "loaded_adapters": {
    "my_adapter": {"lora_name": "my_adapter", "lora_path": "/path/to/lora/adapter", "pinned": false}
  }
}
```

---

## Load LoRA Adapter from Tensors

### Endpoint

```
POST /load_lora_adapter_from_tensors
```

### Description

Load a LoRA adapter from serialized tensor data without re-launching the server.

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lora_name` | `string` | (required) | Name identifier for the adapter |
| `config_dict` | `object` | (required) | LoRA configuration dictionary |
| `serialized_tensors` | `string` | (required) | Serialized tensor data (base64 encoded) |
| `pinned` | `bool` | `false` | Pin the adapter in memory |
| `added_tokens_config` | `object` | `null` | Configuration for added tokens |
| `load_format` | `string` | `null` | Format specification |

### Example

```python
import base64
import json

response = requests.post(
    "http://localhost:30000/load_lora_adapter_from_tensors",
    json={
        "lora_name": "tensor_adapter",
        "config_dict": {"r": 16, "lora_alpha": 32, "target_modules": ["q_proj", "v_proj"]},
        "serialized_tensors": base64.b64encode(tensor_bytes).decode(),
        "pinned": False,
    },
)
```

---

## Unload LoRA Adapter

### Endpoint

```
POST /unload_lora_adapter
```

### Description

Unload a LoRA adapter. Requires admin API key if `--admin-api-key` was set.

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lora_name` | `string` | (required) | Name of the adapter to unload |

### Example

```bash
curl -s -X POST http://localhost:30000/unload_lora_adapter \
  -H "Content-Type: application/json" \
  -d '{"lora_name": "my_adapter"}'
```

### Response Format

```json
{
  "success": true,
  "loaded_adapters": {}
}
```

---

## Weight Management API

These endpoints provide fine-grained control over model weight updates, enabling
online weight updates for RL training and dynamic model switching.

---

## Update Weights from Disk Detailed

See [Update Weights from Disk](#update-weights-from-disk) above for the basic usage.
Below are the remaining weight management endpoints.

---

## Update Weights from Tensor

### Endpoint

```
POST /update_weights_from_tensor
```

### Description

Update model weights from tensor input. The HTTP request transmits only tensor metadata;
the tensor data is directly copied to the model. Binary data should be base64 encoded.
Requires admin API key if set.

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `serialized_named_tensors` | `array` | (required) | List of serialized named tensors (base64) |
| `load_format` | `string` | `null` | Format specification |
| `flush_cache` | `bool` | `true` | Flush cache after updating |
| `abort_all_requests` | `bool` | `false` | Abort all requests before updating |
| `weight_version` | `string` | `null` | Update weight version |
| `disable_draft_model` | `bool` | `null` | Disable updating the draft model |
| `torch_empty_cache` | `bool` | `false` | Empty torch cache during flush |

### Notes

1. Ensure the model is on the correct device (GPU) before calling.
2. HTTP transmits only tensor metadata; the tensor itself is directly copied to the model.
3. Binary data in named tensors should be base64 encoded.

---

## Update Weights from Distributed

### Endpoint

```
POST /update_weights_from_distributed
```

### Description

Update model weights from a distributed source. Used for online weight updates in
distributed training setups.

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `names` | `array of strings` | (required) | Weight tensor names |
| `dtypes` | `array of strings` | (required) | Data types for each tensor |
| `shapes` | `array of array of ints` | (required) | Shapes for each tensor |
| `group_name` | `string` | `"weight_update_group"` | Communication group name |
| `flush_cache` | `bool` | `true` | Flush cache after updating |
| `abort_all_requests` | `bool` | `false` | Abort all requests before updating |
| `weight_version` | `string` | `null` | Update weight version |
| `load_format` | `string` | `null` | Format specification |
| `torch_empty_cache` | `bool` | `false` | Empty torch cache during flush |

---

## Update Weights from IPC

### Endpoint

```
POST /update_weights_from_ipc
```

### Description

Update weights from IPC (Inter-Process Communication) for checkpoint-engine integration.

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `zmq_handles` | `object` | (required) | ZMQ socket paths for each device UUID |
| `flush_cache` | `bool` | `true` | Flush cache after weight update |
| `weight_version` | `string` | `null` | Update weight version |
| `torch_empty_cache` | `bool` | `false` | Empty torch cache during flush |

---

## Update Weight Version

### Endpoint

```
POST /update_weight_version
```

### Description

Update the weight version string. This operation requires no active requests unless
`abort_all_requests` is set.

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `new_version` | `string` | (required) | New weight version string |
| `abort_all_requests` | `bool` | `true` | Abort all running requests before updating |

### Example

```bash
curl -s -X POST http://localhost:30000/update_weight_version \
  -H "Content-Type: application/json" \
  -d '{"new_version": "v2.0", "abort_all_requests": true}'
```

### Response Format

```json
{
  "success": true,
  "message": "Weight version updated to v2.0",
  "new_version": "v2.0"
}
```

---

## Init Weights Update Group

### Endpoint

```
POST /init_weights_update_group
```

### Description

Initialize a communication group for distributed weight updates.

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `master_address` | `string` | (required) | Master node address |
| `master_port` | `int` | (required) | Master port |
| `rank_offset` | `int` | (required) | Rank offset |
| `world_size` | `int` | (required) | Total world size |
| `group_name` | `string` | `"weight_update_group"` | Group name |
| `backend` | `string` | `"nccl"` | Communication backend |

---

## Destroy Weights Update Group

### Endpoint

```
POST /destroy_weights_update_group
```

### Description

Destroy the weight update communication group.

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `group_name` | `string` | `"weight_update_group"` | Group name to destroy |

---

## Init Weights Send Group for Remote Instance

### Endpoint

```
POST /init_weights_send_group_for_remote_instance
```

### Description

Initialize a communication group for sending weights to a remote instance.

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `master_address` | `string` | (required) | Master address |
| `ports` | `string` | (required) | Ports for each rank |
| `group_rank` | `int` | (required) | Rank in the group |
| `world_size` | `int` | (required) | World size |
| `group_name` | `string` | `"weight_send_group"` | Group name |
| `backend` | `string` | `"nccl"` | Communication backend |

---

## Send Weights to Remote Instance

### Endpoint

```
POST /send_weights_to_remote_instance
```

### Description

Send model weights to a remote instance.

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `master_address` | `string` | (required) | Master address |
| `ports` | `string` | (required) | Ports for each rank |
| `group_name` | `string` | `"weight_send_group"` | Group name |

---

## Get Weights by Name

### Endpoint

```
GET /get_weights_by_name
POST /get_weights_by_name
```

### Description

Retrieve model parameter values by name.

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `string` | (required) | Parameter name |
| `truncate_size` | `int` | `100` | Truncate output to this many elements |

---

## Weights Checker

### Endpoint

```
POST /weights_checker
```

### Description

Check model weights integrity.

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `action` | `string` | (required) | Check action to perform |

### Response Format

```json
{
  "success": true,
  "message": "All weights checked.",
  "ranks": [0, 1, 2, 3]
}
```

---

## Release / Resume Memory Occupation

### Endpoints

```
GET /release_memory_occupation
POST /release_memory_occupation
GET /resume_memory_occupation
POST /resume_memory_occupation
```

### Description

Temporarily release or resume GPU memory occupation. Useful for RL training when GPU memory
needs to be shared with training processes.

### Request Parameters (Release and Resume)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tags` | `array of strings` | `null` | Memory region tags: `"weights"`, `"kv_cache"`, or both |

### Example

```python
# Release GPU memory for training
requests.post("http://localhost:30000/release_memory_occupation",
    json={"tags": ["weights", "kv_cache"]})

# ... run training ...

# Resume GPU memory for inference
requests.post("http://localhost:30000/resume_memory_occupation",
    json={"tags": ["weights", "kv_cache"]})
```

---

## Slow Down

### Endpoint

```
GET /slow_down
POST /slow_down
```

### Description

Slow down the system deliberately. Only for testing purposes. Example: when testing PD
disaggregation performance without enough prefill nodes, slow down decode to accumulate
running sequences.

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `forward_sleep_time` | `float` | `null` | Sleep time in seconds between forward passes. `null` to disable. |

---

## HiCache Management API

Hierarchical Cache (HiCache) management endpoints for configuring the storage backend
at runtime.

---

## Clear HiCache Storage Backend

### Endpoint

```
POST /hicache/storage-backend/clear
```

### Description

Clear the hierarchical cache storage backend.

### Example

```bash
curl -s -X POST http://localhost:30000/hicache/storage-backend/clear
```

### Response

```
Hierarchical cache storage backend cleared.
```

---

## Attach HiCache Storage Backend

### Endpoint

```
PUT /hicache/storage-backend
```

### Description

Attach (enable) HiCache storage backend at runtime. Only allowed when there are NO running
or queued requests. Requires admin API key.

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hicache_storage_backend` | `string` | (required) | Storage backend type (e.g., `"file"`) |
| `hicache_storage_backend_extra_config_json` | `string` | `"{}"` | Extra configuration as JSON string |
| `hicache_storage_prefetch_policy` | `string` | `"timeout"` | Prefetch policy |
| `hicache_write_policy` | `string` | (required) | Write policy (e.g., `"write_through"`) |

### Example

```bash
curl -s -X PUT http://localhost:30000/hicache/storage-backend \
  -H 'Content-Type: application/json' \
  -d '{
    "hicache_storage_backend": "file",
    "hicache_storage_backend_extra_config_json": "{}",
    "hicache_storage_prefetch_policy": "timeout",
    "hicache_write_policy": "write_through"
  }'
```

---

## Detach HiCache Storage Backend

### Endpoint

```
DELETE /hicache/storage-backend
```

### Description

Detach (disable) HiCache storage backend at runtime. Only allowed when there are NO running
or queued requests. Requires admin API key.

### Example

```bash
curl -s -X DELETE http://localhost:30000/hicache/storage-backend
```

---

## HiCache Storage Backend Status

### Endpoint

```
GET /hicache/storage-backend
```

### Description

Get current HiCache storage backend status. Requires admin API key.

### Response Format

```json
{
  "hicache_storage_backend": "file",
  "hicache_storage_backend_extra_config": "{}",
  "hicache_storage_prefetch_policy": "timeout",
  "hicache_write_policy": "write_through"
}
```

---

## Session Management API

Sessions enable stateful multi-turn conversations with KV cache persistence across requests.

---

## Open Session

### Endpoint

```
GET /open_session
POST /open_session
```

### Description

Open a new session for stateful interaction. Returns a unique session ID that can be used
in subsequent requests via `session_params`.

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `capacity_of_str_len` | `int` | (required) | Maximum string length capacity for the session |
| `session_id` | `string` | `null` | Custom session ID (auto-generated if null) |
| `streaming` | `bool` | `null` | Whether to enable streaming |
| `timeout` | `float` | `null` | Session timeout in seconds |

### Example

```python
session_id = requests.post(
    "http://localhost:30000/open_session",
    json={"capacity_of_str_len": 4096},
).json()
print(f"Session ID: {session_id}")
```

---

## Close Session

### Endpoint

```
GET /close_session
POST /close_session
```

### Description

Close and clean up a session.

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `session_id` | `string` | (required) | Session ID to close |

### Example

```python
requests.post(
    "http://localhost:30000/close_session",
    json={"session_id": session_id},
)
```

---

## Pause Generation

### Endpoint

```
POST /pause_generation
```

### Description

Pause generation on the server. Supports three modes:

| Mode | Description |
|------|-------------|
| `abort` | Abort and return all currently processed requests |
| `in_place` | Pause scheduler inference; requests stay in event loop; KV cache preserved |
| `retract` | Pause scheduler; retract running requests back to waiting queue; KV cache flushed |

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | `string` | `"abort"` | Pause mode: `"abort"`, `"retract"`, or `"in_place"` |

### Example

```python
# Pause and retract all requests
requests.post("http://localhost:30000/pause_generation",
    json={"mode": "retract"})

# Pause in place (KV cache preserved)
requests.post("http://localhost:30000/pause_generation",
    json={"mode": "in_place"})
```

### Response Format

```json
{"message": "Generation paused successfully.", "status": "ok"}
```

---

## Continue Generation

### Endpoint

```
POST /continue_generation
```

### Description

Resume generation after a pause. If paused in `in_place` mode, continues with existing
KV cache. If paused in `retract` mode, recomputes KV cache as needed.

### Example

```python
requests.post("http://localhost:30000/continue_generation")
```

### Response Format

```json
{"message": "Generation continued successfully.", "status": "ok"}
```

---

## Ngram Speculative Decoding API

These endpoints manage external corpora for NGRAM speculative decoding.

---

## Add External Corpus

### Endpoint

```
POST /add_external_corpus
```

### Description

Add an external corpus for NGRAM speculative decoding. Requires admin API key if set.

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `corpus_id` | `string` | `null` | Custom corpus identifier |
| `content` | `string` | (required) | Corpus text content |
| `tokenizer` | `string` | `null` | Tokenizer to use |

### Response Format

```json
{
  "success": true,
  "corpus_id": "corpus_123",
  "message": "Corpus loaded successfully",
  "loaded_token_count": 50000
}
```

---

## Remove External Corpus

### Endpoint

```
POST /remove_external_corpus
```

### Description

Remove an external corpus by ID.

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `corpus_id` | `string` | (required) | Corpus identifier to remove |

### Response Format

```json
{
  "success": true,
  "message": "Corpus removed successfully"
}
```

---

## List External Corpora

### Endpoint

```
GET /list_external_corpora
```

### Description

List all active external corpora.

### Response Format

```json
{
  "success": true,
  "corpus_token_counts": {
    "corpus_123": 50000,
    "corpus_456": 30000
  },
  "message": null
}
```

---

## Profiling and Debugging API

These endpoints provide profiling, tracing, and debugging capabilities.

---

## Start Profile

### Endpoint

```
GET /start_profile
POST /start_profile
```

### Description

Start PyTorch profiling. Requires admin API key if set.

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | `string` | `null` | Output directory for profile files |
| `start_step` | `int` | `null` | Step at which to start profiling |
| `num_steps` | `int` | `null` | Number of steps to profile (auto-stops after) |
| `activities` | `array of strings` | `null` | Activities: `"CPU"`, `"GPU"`, `"MEM"`, `"RPD"` |
| `profile_by_stage` | `bool` | `false` | Profile prefill and decode stages separately |
| `with_stack` | `bool` | `null` | Record source file/line info |
| `record_shapes` | `bool` | `null` | Record operator input shapes |
| `merge_profiles` | `bool` | `false` | Merge profiles from all ranks |
| `profile_prefix` | `string` | `null` | Prefix for profile filenames |
| `profile_stages` | `array of strings` | `null` | Only profile these stages |

### Example

```bash
# Start profiling for 10 steps
curl -s -X POST http://localhost:30000/start_profile \
  -H "Content-Type: application/json" \
  -d '{"num_steps": 10, "activities": ["CPU", "GPU"]}'
```

---

## Stop Profile

### Endpoint

```
GET /stop_profile
POST /stop_profile
```

### Description

Stop profiling and save results.

### Example

```bash
curl -s -X POST http://localhost:30000/stop_profile
```

---

## Set Trace Level

### Endpoint

```
GET /set_trace_level
POST /set_trace_level
```

### Description

Set the global OpenTelemetry trace level.

### Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `level` | `int` | (required) | Trace level (>= 0). Higher = more verbose. |

### Example

```bash
curl -s -X POST "http://localhost:30000/set_trace_level?level=5"
```

---

## Freeze GC

### Endpoint

```
GET /freeze_gc
POST /freeze_gc
```

### Description

Freeze Python garbage collection. Useful for performance-critical operations where GC
pauses are undesirable.

### Example

```bash
curl -s -X POST http://localhost:30000/freeze_gc
```

---

## Configure Logging

### Endpoint

```
GET /configure_logging
POST /configure_logging
```

### Description

Configure request logging options at runtime.

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_requests` | `bool` | `null` | Enable/disable request logging |
| `log_requests_level` | `int` | `null` | Logging level |
| `log_requests_format` | `string` | `null` | Log format |
| `dump_requests_folder` | `string` | `null` | Folder for request dumps |
| `dump_requests_threshold` | `int` | `null` | Threshold for dumping |
| `crash_dump_folder` | `string` | `null` | Folder for crash dumps |

---

## Set Internal State

### Endpoint

```
POST /set_internal_state
PUT /set_internal_state
```

### Description

Update internal server state parameters at runtime.

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `server_args` | `object` | (required) | Server args to update (e.g., `{"pp_max_micro_batch_size": 8}`) |

### Example

```bash
curl -s -X POST http://localhost:30000/set_internal_state \
  -H "Content-Type: application/json" \
  -d '{"server_args": {"pp_max_micro_batch_size": 8}}'
```

---

## Ollama-Compatible API

SGLang provides Ollama API compatibility, allowing the Ollama CLI and Python library to use
SGLang as the inference backend.

### Prerequisites

```bash
pip install ollama
```

Note: You do not need the Ollama server installed; SGLang acts as the backend. Only the Ollama
client library is needed.

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` or custom root | GET, HEAD | Health check for Ollama CLI |
| `/api/tags` | GET | List available models |
| `/api/chat` | POST | Chat completions (streaming and non-streaming) |
| `/api/generate` | POST | Text generation (streaming and non-streaming) |
| `/api/show` | POST | Model information |

The route paths can be customized via environment variables:
- `SGLANG_OLLAMA_ROOT_ROUTE`
- `SGLANG_OLLAMA_CHAT_ROUTE` (default: `/api/chat`)
- `SGLANG_OLLAMA_GENERATE_ROUTE` (default: `/api/generate`)
- `SGLANG_OLLAMA_TAGS_ROUTE` (default: `/api/tags`)
- `SGLANG_OLLAMA_SHOW_ROUTE` (default: `/api/show`)

### Launch Server

```bash
python -m sglang.launch_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --port 30001 \
    --host 0.0.0.0
```

---

## Ollama Chat

### Endpoint

```
POST /api/chat
```

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `string` | (required) | Model name (must match `--model` argument exactly) |
| `messages` | `array` | (required) | Array of message objects with `role` and `content` |
| `stream` | `bool` | `true` | Enable streaming |
| `format` | `string or object` | `null` | Output format: `"json"` or JSON schema object |
| `options` | `object` | `null` | Ollama-style options (temperature, top_p, etc.) |
| `keep_alive` | `float or string` | `null` | Keep model loaded duration |
| `think` | `bool or string` | `null` | Enable thinking/reasoning: `true`, `false`, `"low"`, `"medium"`, `"high"` |

### Message Format

```json
{
  "role": "system" | "user" | "assistant",
  "content": "message text",
  "images": ["base64_encoded_image", ...]
}
```

### Example: Using Ollama CLI

```bash
# List models
OLLAMA_HOST=http://localhost:30001 ollama list

# Interactive chat
OLLAMA_HOST=http://localhost:30001 ollama run "Qwen/Qwen2.5-1.5B-Instruct"
```

### Example: Using Ollama Python Library

```python
import ollama

client = ollama.Client(host='http://localhost:30001')

# Non-streaming chat
response = client.chat(
    model='Qwen/Qwen2.5-1.5B-Instruct',
    messages=[{'role': 'user', 'content': 'Hello!'}]
)
print(response['message']['content'])

# Streaming chat
stream = client.chat(
    model='Qwen/Qwen2.5-1.5B-Instruct',
    messages=[{'role': 'user', 'content': 'Tell me a story'}],
    stream=True
)
for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)
```

### Non-streaming Response Format

```json
{
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "created_at": "2024-01-15T10:30:00Z",
  "message": {
    "role": "assistant",
    "content": "Hello! How can I help you?"
  },
  "done": true,
  "done_reason": "stop",
  "total_duration": 500000000,
  "load_duration": 100000000,
  "prompt_eval_count": 10,
  "prompt_eval_duration": 50000000,
  "eval_count": 8,
  "eval_duration": 200000000
}
```

---

## Ollama Generate

### Endpoint

```
POST /api/generate
```

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `string` | (required) | Model name |
| `prompt` | `string` | (required) | Input prompt |
| `suffix` | `string` | `null` | Suffix to append |
| `system` | `string` | `null` | System prompt |
| `template` | `string` | `null` | Prompt template |
| `context` | `array of ints` | `null` | Context token IDs for continuation |
| `stream` | `bool` | `true` | Enable streaming |
| `raw` | `bool` | `false` | Raw mode (no formatting) |
| `format` | `string or object` | `null` | Output format |
| `options` | `object` | `null` | Ollama-style options |
| `keep_alive` | `float or string` | `null` | Keep model loaded duration |
| `images` | `array of strings` | `null` | Base64-encoded images |
| `think` | `bool` | `null` | Enable thinking |

---

## Ollama Tags List Models

### Endpoint

```
GET /api/tags
```

### Description

List available models in Ollama format.

### Response Format

```json
{
  "models": [
    {
      "name": "Qwen/Qwen2.5-1.5B-Instruct",
      "model": "Qwen/Qwen2.5-1.5B-Instruct",
      "modified_at": "2024-01-15T10:30:00Z",
      "size": 3000000000,
      "digest": "abc123",
      "details": {}
    }
  ]
}
```

---

## Ollama Show Model Info

### Endpoint

```
POST /api/show
```

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `string` | (required) | Model name |

### Response Format

```json
{
  "license": "",
  "modelfile": "",
  "parameters": "",
  "template": "",
  "modified_at": "",
  "details": {},
  "model_info": {},
  "capabilities": []
}
```

---

## Ollama Root

### Endpoint

```
GET /
HEAD /
```

### Description

Returns `"Ollama is running"` for Ollama CLI compatibility (when `SGLANG_OLLAMA_ROOT_ROUTE`
is set), or `"SGLang is running"` otherwise.

---

## Anthropic-Compatible API

SGLang provides compatibility with the Anthropic Messages API, allowing applications that
use the Anthropic SDK to work with SGLang as a backend.

---

## Anthropic Messages

### Endpoint

```
POST /v1/messages
```

### Description

Anthropic-compatible Messages API endpoint. Converts Anthropic-format requests to SGLang
internal format and generates responses.

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `string` | (required) | Model name |
| `messages` | `array` | (required) | Array of Anthropic-format message objects |
| `max_tokens` | `int` | (required) | Maximum tokens to generate. Must be positive. |
| `metadata` | `object` | `null` | Custom metadata |
| `stop_sequences` | `array of strings` | `null` | Custom stop sequences |
| `stream` | `bool` | `false` | Enable streaming |
| `system` | `string or array` | `null` | System prompt (string or content blocks) |
| `temperature` | `float` | `null` | Sampling temperature |
| `tool_choice` | `object` | `null` | Tool choice: `{"type": "auto"|"any"|"tool"|"none", "name": "..."}` |
| `tools` | `array` | `null` | Tool definitions |
| `top_k` | `int` | `null` | Top-k sampling |
| `top_p` | `float` | `null` | Top-p sampling |

### Message Format

```json
{
  "role": "user" | "assistant",
  "content": "string or array of content blocks"
}
```

Content block types: `text`, `image`, `tool_use`, `tool_result`, `tool_reference`, `thinking`, `redacted_thinking`.

### Example

```python
import requests

response = requests.post(
    "http://localhost:30000/v1/messages",
    json={
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "max_tokens": 256,
        "messages": [
            {"role": "user", "content": "What is the capital of France?"}
        ],
    },
)
print(response.json())
```

### Non-streaming Response Format

```json
{
  "id": "msg_abc123",
  "type": "message",
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "The capital of France is Paris."
    }
  ],
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 15,
    "output_tokens": 10
  }
}
```

### Streaming Response Format

```
event: message_start
data: {"type":"message_start","message":{"id":"msg_abc123","type":"message","role":"assistant","content":[],"model":"model","usage":{"input_tokens":15,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"The"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" capital"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":10}}

event: message_stop
data: {"type":"message_stop"}
```

---

## Anthropic Count Tokens

### Endpoint

```
POST /v1/messages/count_tokens
```

### Description

Count the number of tokens for a given set of messages.

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `string` | (required) | Model name |
| `messages` | `array` | (required) | Array of message objects |
| `system` | `string or array` | `null` | System prompt |
| `tool_choice` | `object` | `null` | Tool choice |
| `tools` | `array` | `null` | Tool definitions |

### Response Format

```json
{
  "input_tokens": 42
}
```

---

## SageMaker Compatibility

SGLang provides compatibility with AWS SageMaker endpoints.

---

## SageMaker Health Ping

### Endpoint

```
GET /ping
```

### Description

SageMaker health check endpoint. Returns 200 if the HTTP server is running.

### Example

```bash
curl http://localhost:30000/ping
```

---

## SageMaker Invocations

### Endpoint

```
POST /invocations
```

### Description

SageMaker invocation endpoint. Accepts OpenAI chat completion format requests.

### Example

```bash
curl http://localhost:30000/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

---

## Vertex AI Compatibility

---

## Vertex Generate

### Endpoint

```
POST /vertex_generate
```

> Note: The route can be customized via the `AIP_PREDICT_ROUTE` environment variable.

### Description

Vertex AI prediction endpoint. Accepts a list of instances with generation parameters.

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `instances` | `array` | (required) | Array of instance objects |
| `parameters` | `object` | `null` | Generation parameters applied to all instances |

Each instance can contain:

| Field | Type | Description |
|-------|------|-------------|
| `text` | `string` | Input prompt (one of `text`, `input_ids`, `input_embeds`) |
| `input_ids` | `array of ints` | Token IDs |
| `input_embeds` | `array of floats` | Embedding input |
| `image_data` | `string` | Image data |

### Example

```bash
curl http://localhost:30000/vertex_generate \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      {"text": "The capital of France is"}
    ],
    "parameters": {
      "sampling_params": {"temperature": 0, "max_new_tokens": 32}
    }
  }'
```

### Response Format

```json
{
  "predictions": [
    {
      "text": " Paris, the capital and largest city of France.",
      "meta_info": {"id": "req-123", "finish_reason": {"type": "stop"}}
    }
  ]
}
```

---

## gRPC API

SGLang provides a gRPC interface defined in `proto/sglang/runtime/v1/sglang.proto`.

### Service Definition

```protobuf
service SglangService {
  // SGLang-native RPCs
  rpc TextGenerate(TextGenerateRequest) returns (stream TextGenerateResponse);
  rpc Generate(GenerateRequest) returns (stream GenerateResponse);
  rpc TextEmbed(TextEmbedRequest) returns (TextEmbedResponse);
  rpc Embed(EmbedRequest) returns (EmbedResponse);
  rpc Classify(ClassifyRequest) returns (ClassifyResponse);
  rpc Tokenize(TokenizeRequest) returns (TokenizeResponse);
  rpc Detokenize(DetokenizeRequest) returns (DetokenizeResponse);
  rpc HealthCheck(HealthCheckRequest) returns (HealthCheckResponse);
  rpc GetModelInfo(GetModelInfoRequest) returns (GetModelInfoResponse);
  rpc GetServerInfo(GetServerInfoRequest) returns (GetServerInfoResponse);
  rpc ListModels(ListModelsRequest) returns (ListModelsResponse);
  rpc GetLoad(GetLoadRequest) returns (GetLoadResponse);
  rpc Abort(AbortRequest) returns (AbortResponse);
  rpc FlushCache(FlushCacheRequest) returns (FlushCacheResponse);
  rpc PauseGeneration(PauseGenerationRequest) returns (PauseGenerationResponse);
  rpc ContinueGeneration(ContinueGenerationRequest) returns (ContinueGenerationResponse);

  // OpenAI-compatible RPCs (JSON pass-through)
  rpc ChatComplete(OpenAIRequest) returns (stream OpenAIStreamChunk);
  rpc Complete(OpenAIRequest) returns (stream OpenAIStreamChunk);
  rpc OpenAIEmbed(OpenAIRequest) returns (OpenAIResponse);
  rpc OpenAIClassify(OpenAIRequest) returns (OpenAIResponse);
  rpc Score(OpenAIRequest) returns (OpenAIResponse);
  rpc Rerank(OpenAIRequest) returns (OpenAIResponse);

  // Admin/Ops RPCs
  rpc StartProfile(StartProfileRequest) returns (StartProfileResponse);
  rpc StopProfile(StopProfileRequest) returns (StopProfileResponse);
  rpc UpdateWeightsFromDisk(UpdateWeightsRequest) returns (UpdateWeightsResponse);
}
```

### Key Message Types

#### SamplingParams

```protobuf
message SamplingParams {
  optional float temperature = 1;
  optional float top_p = 2;
  optional int32 top_k = 3;
  optional float min_p = 4;
  optional float frequency_penalty = 5;
  optional float presence_penalty = 6;
  optional float repetition_penalty = 7;
  optional int32 max_new_tokens = 8;
  optional int32 min_new_tokens = 9;
  repeated string stop = 10;
  repeated int32 stop_token_ids = 11;
  optional bool ignore_eos = 12;
  optional int32 n = 13;
  optional string json_schema = 14;
  optional string regex = 15;
}
```

#### TextGenerateRequest / Response

```protobuf
message TextGenerateRequest {
  string text = 1;
  optional SamplingParams sampling_params = 2;
  optional bool stream = 3;
  optional bool return_logprob = 4;
  optional int32 top_logprobs_num = 5;
  optional int32 logprob_start_len = 6;
  optional bool return_text_in_logprobs = 7;
  optional string rid = 8;
  optional string lora_path = 9;
  optional string routing_key = 10;
  optional int32 routed_dp_rank = 11;
  map<string, string> trace_headers = 12;
}

message TextGenerateResponse {
  string text = 1;
  map<string, string> meta_info = 2;
  bool finished = 3;
}
```

#### OpenAI Pass-through

```protobuf
message OpenAIRequest {
  bytes json_body = 1;
  map<string, string> trace_headers = 2;
}

message OpenAIStreamChunk {
  bytes json_chunk = 1;
  bool finished = 2;
}

message OpenAIResponse {
  bytes json_body = 1;
  int32 status_code = 2;
}
```

---

## Offline Engine API (Python)

The offline engine provides direct inference without HTTP server overhead, ideal for batch
processing, RL training, and custom server implementations.

### Creating an Engine

```python
import sglang as sgl

llm = sgl.Engine(model_path="meta-llama/Llama-3.1-8B-Instruct")
```

All `ServerArgs` parameters can be passed as keyword arguments:

```python
llm = sgl.Engine(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    tp_size=4,
    mem_fraction_static=0.85,
    quantization="fp8",
)
```

### Engine.generate()

```python
def generate(
    prompt: Optional[Union[List[str], str]] = None,
    sampling_params: Optional[Union[List[Dict], Dict]] = None,
    input_ids: Optional[Union[List[List[int]], List[int]]] = None,
    image_data: Optional[MultimodalDataInputFormat] = None,
    audio_data: Optional[MultimodalDataInputFormat] = None,
    video_data: Optional[MultimodalDataInputFormat] = None,
    return_logprob: Optional[Union[List[bool], bool]] = False,
    logprob_start_len: Optional[Union[List[int], int]] = None,
    top_logprobs_num: Optional[Union[List[int], int]] = None,
    token_ids_logprob: Optional[Union[List[List[int]], List[int]]] = None,
    lora_path: Optional[List[Optional[str]]] = None,
    custom_logit_processor: Optional[Union[List[str], str]] = None,
    return_hidden_states: bool = False,
    return_routed_experts: bool = False,
    stream: bool = False,
    bootstrap_host: Optional[Union[List[str], str]] = None,
    bootstrap_port: Optional[Union[List[int], int]] = None,
    bootstrap_room: Optional[Union[List[int], int]] = None,
    routed_dp_rank: Optional[int] = None,
    disagg_prefill_dp_rank: Optional[int] = None,
    external_trace_header: Optional[Dict] = None,
    rid: Optional[Union[List[str], str]] = None,
    session_params: Optional[Dict] = None,
    priority: Optional[int] = None,
) -> Union[Dict, Iterator[Dict]]
```

### Non-Streaming Synchronous Generation

```python
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
]

sampling_params = {"temperature": 0.8, "top_p": 0.95}

outputs = llm.generate(prompts, sampling_params)

for prompt, output in zip(prompts, outputs):
    print(f"Prompt: {prompt}")
    print(f"Generated: {output['text']}")
```

### Streaming Synchronous Generation

```python
from sglang.utils import stream_and_merge

for prompt in prompts:
    merged_output = stream_and_merge(llm, prompt, sampling_params)
    print(f"Prompt: {prompt}")
    print(f"Generated: {merged_output}")
```

### Engine.async_generate()

Same parameters as `generate()` but returns an async generator.

```python
async def async_generate(
    prompt: Optional[Union[List[str], str]] = None,
    sampling_params: Optional[Union[List[Dict], Dict]] = None,
    ...
) -> Union[Dict, AsyncIterator[Dict]]
```

### Non-Streaming Asynchronous Generation

```python
import asyncio

async def main():
    outputs = await llm.async_generate(prompts, sampling_params)
    for prompt, output in zip(prompts, outputs):
        print(f"Prompt: {prompt}")
        print(f"Generated: {output['text']}")

asyncio.run(main())
```

### Streaming Asynchronous Generation

```python
from sglang.utils import async_stream_and_merge

async def main():
    for prompt in prompts:
        async for chunk in async_stream_and_merge(llm, prompt, sampling_params):
            print(chunk, end="", flush=True)

asyncio.run(main())
```

### Engine.encode()

```python
def encode(
    prompt: Optional[Union[List[str], str]] = None,
    sampling_params: Optional[Union[List[Dict], Dict]] = None,
    ...
) -> Union[Dict, Iterator[Dict]]
```

### Using in Jupyter/IPython

In environments with existing event loops (Jupyter, IPython), use nest_asyncio:

```python
import nest_asyncio
nest_asyncio.apply()
```

### Engine.shutdown()

```python
llm.shutdown()
```

Releases all GPU resources and stops worker processes.

---

## Sampling Parameters Reference

These parameters control text generation behavior. They can be specified in the `sampling_params`
field of the `/generate` endpoint or as request parameters in OpenAI API calls.

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_new_tokens` | `int` | `128` | Maximum output length in tokens |
| `stop` | `string or list` | `null` | Stop sequences (words that end generation) |
| `stop_token_ids` | `list of ints` | `null` | Stop sequences as token IDs |
| `stop_regex` | `string or list` | `null` | Stop when matching any regex pattern |
| `temperature` | `float` | model default / 1.0 | Sampling temperature. 0 = greedy, higher = more diverse |
| `top_p` | `float` | model default / 1.0 | Top-p (nucleus) sampling. 1.0 = unrestricted |
| `top_k` | `int` | model default / -1 | Top-k sampling. -1 = disabled |
| `min_p` | `float` | model default / 0.0 | Min-p sampling threshold |

### Penalizers

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `frequency_penalty` | `float` | `0.0` | Penalize tokens by frequency (-2 to 2). Positive = less repetition. |
| `presence_penalty` | `float` | `0.0` | Penalize tokens if appeared (-2 to 2). Positive = more diversity. |
| `repetition_penalty` | `float` | `1.0` | Scale logits of previous tokens (0-2). >1 = less repetition, <1 = more. |
| `min_new_tokens` | `int` | `0` | Force minimum generation length |

### Constrained Decoding

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `json_schema` | `string` | `null` | JSON schema for structured output |
| `regex` | `string` | `null` | Regex pattern for structured output |
| `ebnf` | `string` | `null` | EBNF grammar for structured output |
| `structural_tag` | `string` | `null` | Structural tag for structured output |

### Other Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n` | `int` | `1` | Number of output sequences per request |
| `ignore_eos` | `bool` | `false` | Do not stop on EOS token |
| `skip_special_tokens` | `bool` | `true` | Remove special tokens from output |
| `spaces_between_special_tokens` | `bool` | `true` | Add spaces between special tokens |
| `no_stop_trim` | `bool` | `false` | Do not trim stop words from output |
| `custom_params` | `list of dicts` | `null` | Parameters for custom logit processor |

### Default Parameter Source

By default, SGLang initializes sampling parameters from the model's `generation_config.json`.
To use constant OpenAI defaults instead:

```bash
# Use model-provided defaults (default behavior)
python -m sglang.launch_server --model-path <MODEL> --sampling-defaults model

# Use SGLang/OpenAI constant defaults
python -m sglang.launch_server --model-path <MODEL> --sampling-defaults openai
```

---

## Structured Outputs

SGLang supports constraining model output to specific formats using grammar backends.

### Grammar Backend Options

| Backend | Flag | Supports |
|---------|------|----------|
| XGrammar (default) | `--grammar-backend xgrammar` | JSON schema, regex, EBNF |
| Outlines | `--grammar-backend outlines` | JSON schema, regex |
| LLGuidance | `--grammar-backend llguidance` | JSON schema, regex, EBNF |

### JSON Schema

```python
import json

json_schema = json.dumps({
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "population": {"type": "integer"},
    },
    "required": ["name", "population"],
})

response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "The capital of France in JSON format.\n",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 64,
            "json_schema": json_schema,
        },
    },
)
```

### Regex

```python
response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "Paris is the capital of",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 64,
            "regex": "(France|England)",
        },
    },
)
```

### EBNF (XGrammar only)

```python
response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "Write a greeting.",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 64,
            "ebnf": 'root ::= "Hello" | "Hi" | "Hey"',
        },
    },
)
```

---

## Custom Logit Processor

Custom logit processors allow modifying the logits distribution before sampling.

### Server Setup

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3-8B-Instruct \
    --enable-custom-logit-processor
```

### Define a Custom Processor

```python
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor

class DeterministicLogitProcessor(CustomLogitProcessor):
    """Always sample a specific token ID."""

    def __call__(self, logits, custom_param_list):
        assert logits.shape[0] == len(custom_param_list)
        key = "token_id"

        for i, param_dict in enumerate(custom_param_list):
            logits[i, :] = -float("inf")
            logits[i, param_dict[key]] = 0.0
        return logits
```

### Use with Native API

```python
response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "The capital of France is",
        "custom_logit_processor": DeterministicLogitProcessor().to_str(),
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": 32,
            "custom_params": {"token_id": 5},
        },
    },
)
```

### Use with OpenAI API

```python
response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[{"role": "user", "content": "Hello"}],
    extra_body={
        "custom_logit_processor": DeterministicLogitProcessor().to_str(),
        "custom_params": {"token_id": 5},
    },
)
```

---

## Streaming Protocol

### OpenAI API Streaming

Set `stream=True` in the request:

```python
# Chat completions
stream = client.chat.completions.create(
    model="model-name",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)

# Completions
stream = client.completions.create(
    model="model-name",
    prompt="Hello",
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].text:
        print(chunk.choices[0].text, end="", flush=True)
```

### Native API Streaming

Set `stream=True` in the request body:

```python
response = requests.post(
    "http://localhost:30000/generate",
    json={"text": "Hello", "stream": True},
    stream=True,
)

prev = 0
for chunk in response.iter_lines(decode_unicode=False):
    chunk = chunk.decode("utf-8")
    if chunk and chunk.startswith("data:"):
        if chunk == "data: [DONE]":
            break
        data = json.loads(chunk[5:].strip("\n"))
        output = data["text"]
        print(output[prev:], end="", flush=True)
        prev = len(output)
```

### SSE Protocol

Streaming uses Server-Sent Events (SSE):
- Each event is prefixed with `data: `
- Events are separated by newlines
- The stream ends with `data: [DONE]`
- Each event contains a JSON payload with incremental output

---

## LoRA Adapters in Requests

SGLang supports LoRA (Low-Rank Adaptation) adapters with all API endpoints.

### Specify Adapter in Request

Recommended method -- use `model:adapter` syntax:

```python
response = client.chat.completions.create(
    model="base-model:adapter_a",
    messages=[{"role": "user", "content": "Hello"}],
)
```

Backward compatible method using `extra_body`:

```python
response = client.chat.completions.create(
    model="base-model",
    messages=[{"role": "user", "content": "Hello"}],
    extra_body={"lora_path": "adapter_a"},
)
```

When both `model:adapter` and `extra_body["lora_path"]` are specified, `model:adapter` takes
precedence.

---

## Model Thinking / Reasoning

SGLang provides unified support for reasoning models that expose internal thinking processes.

### Server Configuration

Launch with the appropriate `--reasoning-parser`:

```bash
python -m sglang.launch_server --model Qwen/Qwen3-4B --reasoning-parser qwen3
```

### Enabling Reasoning Output

```python
response = client.chat.completions.create(
    model="Qwen/Qwen3-4B",
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    extra_body={
        "chat_template_kwargs": {"enable_thinking": True},
        "separate_reasoning": True,
    },
)

reasoning = response.choices[0].message.reasoning_content
answer = response.choices[0].message.content
```

### Using reasoning_effort Parameter

```python
response = client.chat.completions.create(
    model="Qwen/Qwen3-4B",
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    reasoning_effort="high",
)
```

### Supported Models and Configuration

| Model Family | Template Parameter | Reasoning Parser | Notes |
|-------------|-------------------|-----------------|-------|
| DeepSeek-R1 | `enable_thinking` | `--reasoning-parser deepseek-r1` | Standard reasoning |
| DeepSeek-V3.1 | `thinking` | `--reasoning-parser deepseek-v3` | Hybrid thinking/non-thinking |
| Qwen3 | `enable_thinking` | `--reasoning-parser qwen3` | Hybrid thinking/non-thinking |
| Qwen3-Thinking | N/A (always) | `--reasoning-parser qwen3-thinking` | Always generates reasoning |
| Kimi | N/A (always) | `--reasoning-parser kimi` | Always generates reasoning |
| Gpt-Oss | N/A (always) | `--reasoning-parser gpt-oss` | Always generates reasoning |

### Disabling Reasoning

Set `reasoning_effort="none"` or set the template parameter to `false`:

```python
response = client.chat.completions.create(
    model="Qwen/Qwen3-4B",
    messages=[{"role": "user", "content": "Hello"}],
    reasoning_effort="none",
)
```

---

## Request / Response Formats

### Common Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | `string` | Unique request/response ID |
| `object` | `string` | Object type (e.g., `chat.completion`, `text_completion`) |
| `created` | `int` | Unix timestamp of creation |
| `model` | `string` | Model name used |
| `choices` | `array` | Array of completion choices |
| `usage` | `object` | Token usage statistics |

### Usage Object

| Field | Type | Description |
|-------|------|-------------|
| `prompt_tokens` | `int` | Number of tokens in the prompt |
| `completion_tokens` | `int` | Number of tokens in the completion |
| `total_tokens` | `int` | Total tokens (prompt + completion) |
| `prompt_tokens_details` | `object` | Detailed prompt token breakdown (optional) |
| `reasoning_tokens` | `int` | Reasoning tokens (optional) |

### Prompt Tokens Details

| Field | Type | Description |
|-------|------|-------------|
| `cached_tokens` | `int` | Number of cached tokens |

### Cached Tokens Details

| Field | Type | Description |
|-------|------|-------------|
| `device` | `int` | Tokens from device cache (GPU) |
| `host` | `int` | Tokens from host cache (CPU memory) |
| `storage` | `int` | Tokens from L3 storage backend (optional) |
| `storage_backend` | `string` | Type of storage backend used (optional) |

### Finish Reasons

| Value | Description |
|-------|-------------|
| `stop` | Natural stop or stop sequence hit |
| `length` | Max tokens reached |
| `tool_calls` | Model requested tool calls |
| `content_filter` | Content filtered (if configured) |
| `function_call` | Function call (legacy) |
| `abort` | Request was aborted |

### Model Card Object

| Field | Type | Description |
|-------|------|-------------|
| `id` | `string` | Model identifier |
| `object` | `string` | Always `"model"` |
| `created` | `int` | Unix timestamp |
| `owned_by` | `string` | Always `"sglang"` |
| `root` | `string` | Model path |
| `parent` | `string` | Parent model (for LoRA) |
| `max_model_len` | `int` | Maximum context length |

---

## Error Handling

### Error Response Format

Standard OpenAI format:

```json
{
  "object": "error",
  "message": "Error description",
  "type": "Bad Request",
  "param": null,
  "code": 400
}
```

Responses API format (for `/v1/responses`):

```json
{
  "error": {
    "message": "Error description",
    "type": "Bad Request",
    "param": null,
    "code": 400
  }
}
```

### Common Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad request (invalid parameters, validation error) |
| 404 | Model or resource not found |
| 429 | Rate limited (if configured) |
| 500 | Internal server error |
| 503 | Server not ready or health check failed |

### Validation Errors

When request validation fails, SGLang returns a 400 (not 422) with details:

```json
{
  "object": "error",
  "message": "validation error details",
  "type": "Bad Request",
  "code": 400
}
```

---

## Authentication

### API Key (--api-key)

When the server is launched with `--api-key`, all requests must include:

```
Authorization: Bearer <api-key>
```

### Admin API Key (--admin-api-key)

Management endpoints marked as `ADMIN_OPTIONAL` require the admin API key:

```
Authorization: Bearer <admin-api-key>
```

If `--admin-api-key` is not set, admin endpoints are accessible without authentication
(unless `--api-key` is set, in which case the regular API key is used).

### Endpoints Requiring Admin Key (when set)

- `/flush_cache`
- `/start_profile`, `/stop_profile`
- `/set_trace_level`, `/freeze_gc`
- `/update_weights_from_disk`, `/update_weights_from_tensor`, `/update_weights_from_distributed`
- `/update_weights_from_ipc`, `/update_weight_version`
- `/init_weights_update_group`, `/destroy_weights_update_group`
- `/init_weights_send_group_for_remote_instance`, `/send_weights_to_remote_instance`
- `/get_weights_by_name`, `/weights_checker`
- `/release_memory_occupation`, `/resume_memory_occupation`
- `/slow_down`
- `/load_lora_adapter`, `/unload_lora_adapter`
- `/pause_generation`, `/continue_generation`
- `/abort_request`
- `/configure_logging`
- `/set_internal_state`
- Expert distribution recording endpoints
- HiCache management endpoints
- Ngram corpus management endpoints

---

## Related Documentation

- [Overview and Architecture](./01-overview-architecture.md)
- [Installation and Setup](./02-installation-setup.md)
- [Server Configuration Reference](./03-server-configuration.md)
- [Supported Models](./05-supported-models.md)
- [Sampling and Decoding](./07-sampling-decoding.md)
- [LoRA Adapters](./11-lora-adapters.md)
- [Observability and Profiling](./15-observability-profiling.md)
