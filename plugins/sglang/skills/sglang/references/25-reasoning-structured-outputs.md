# SGLang Reasoning and Structured Outputs Reference

This document provides a comprehensive reference for SGLang's reasoning model support, structured output generation, function calling / tool use, custom logit processors, and the complete sampling parameter API. These features allow fine-grained control over model outputs, from constraining output formats to managing thinking/reasoning tokens in models like DeepSeek-R1 and Qwen3.

---

## Table of Contents

1. [Overview](#overview)
2. [Reasoning Parsers](#reasoning-parsers)
3. [Structured Outputs](#structured-outputs)
4. [Function Calling / Tool Use](#function-calling--tool-use)
5. [Custom Logit Processors](#custom-logit-processors)
6. [Sampling Parameters (Complete Reference)](#sampling-parameters-complete-reference)
7. [Advanced Sampling](#advanced-sampling)

---

## Overview

### Reasoning Models

Reasoning models (such as DeepSeek-R1, Qwen3, GLM-4.5, and others) produce an internal "thinking" or "reasoning" trace before emitting their final answer. This trace is typically enclosed in special tokens (e.g., `<think...</think->`) and can be optionally streamed or hidden from the end user. SGLang provides a pluggable reasoning parser framework that:

- Detects and separates reasoning content from normal output text
- Supports both streaming and non-streaming scenarios
- Handles model-specific formatting variations
- Integrates with grammar backends to allow structured outputs after reasoning completes

### Structured Outputs

SGLang supports constrained decoding to guarantee that model outputs conform to specified formats:

- **JSON Schema validation**: Enforce outputs match a given JSON schema
- **Regular expressions**: Constrain output to match a regex pattern
- **EBNF grammars**: Specify arbitrary context-free grammars
- **Structural Tags**: Model-native tool call format constraints
- **Response formats**: OpenAI-compatible `text`, `json_object`, and `json_schema` modes

### Architecture

The constrained decoding pipeline works as follows:

```
Request (with constraints)
  -> GrammarManager (compiles/queues grammar objects)
    -> GrammarBackend (xgrammar | outlines | llguidance)
      -> GrammarObject (per-request, applies vocab masks)
        -> SamplingBatchInfo (applies masks to logits)
          -> Token selection (constrained)
```

When reasoning is active, `ReasonerGrammarBackend` wraps the base grammar backend to defer constraint application until after the reasoning section ends.

---

## Reasoning Parsers

### Supported Reasoning Formats

SGLang includes built-in detectors for multiple reasoning model families. Each detector knows the specific tokens and patterns used by its model family.

#### Detector Map

| Model Type Key       | Detector Class                | Think Start Token           | Think End Token             | `force_reasoning` | Notes                                            |
|----------------------|-------------------------------|-----------------------------|-----------------------------|-------------------|--------------------------------------------------|
| `deepseek-r1`        | `DeepSeekR1Detector`          | `<think\>` (optional)       | `</think\>`                 | `True`            | R1 always reasons; R1-0528 emits `<think\>`      |
| `deepseek-v3`        | `Qwen3Detector`               | `<think\>`                  | `</think\>`                 | `False`           | Uses Qwen3-style format                          |
| `qwen3`              | `Qwen3Detector`               | `<think\>`                  | `</think\>`                 | `False`           | Supports `enable_thinking` toggle                |
| `qwen3-thinking`     | `Qwen3Detector`               | `<think\>`                  | `</think\>`                 | `True`            | Always-reasoning variant                         |
| `glm45`              | `Glm45Detector`               | `<think\>`                  | `</think\>`                 | `False`           | Uses `<tool_call\>` as tool start                |
| `hunyuan`            | `HunyuanDetector`             | `<think\>`                  | `</think\>`                 | `False`           | Uses `<tool_calls\>` as tool start               |
| `gpt-oss`            | `GptOssDetector`              | `<\|channel\|>analysis<\|message\|>` | `<\|end\|>`          | `True`            | T4-style Harmony format                          |
| `kimi`               | `KimiDetector`                | `◁think▷`                   | `◁/think▷`                  | `False`           | Kimi Thinking model                              |
| `kimi_k2`            | `KimiK2Detector`              | `<think\>`                  | `</think\>`                 | `False`           | Tool call section begin before `</think\>`       |
| `mimo`               | `Qwen3Detector`               | `<think\>`                  | `</think\>`                 | `False`           | Uses Qwen3-style format                          |
| `minimax`            | `Qwen3Detector`               | `<think\>`                  | `</think\>`                 | `True`            | Always-reasoning variant                         |
| `minimax-append-think` | `MiniMaxAppendThinkDetector` | `<think\>`                  | `</think\>`                 | `False`           | Appends `<think\>` token to beginning            |
| `step3`              | `DeepSeekR1Detector`          | `<think\>` (optional)       | `</think\>`                 | `True`            | Uses DeepSeek-R1 format                          |
| `step3p5`            | `DeepSeekR1Detector`          | `<think\>` (optional)       | `</think\>`                 | `True`            | Uses DeepSeek-R1 format                          |
| `mistral`            | `MistralDetector`             | `[THINK]`                   | `[/THINK]`                  | `False`           | Mistral-Small-4; reasoning optional              |
| `nemotron_3`         | `Nemotron3Detector`           | `<think\>`                  | `</think\>`                 | `False`           | Supports force_nonempty_content                  |
| `interns1`           | `Qwen3Detector`               | `<think\>`                  | `</think\>`                 | `False`           | Uses Qwen3-style format                          |
| `gemma4`             | `Gemma4Detector`              | `<\|channel>`               | `<channel\|>`               | `False`           | Uses `thought\n` self-label                      |

Source: `python/sglang/srt/parser/reasoning_parser.py`

### ReasoningParser API

The `ReasoningParser` class is the main entry point for reasoning content extraction.

```python
from sglang.srt.parser.reasoning_parser import ReasoningParser

# Create a parser for a specific model type
parser = ReasoningParser(
    model_type="deepseek-r1",
    stream_reasoning=True,       # Stream reasoning tokens as they arrive
    force_reasoning=None,        # Override per-detector default
    request=None,                # ChatCompletionRequest for context
)

# Non-streaming: parse full text at once
reasoning_text, normal_text = parser.parse_non_stream(full_text)

# Streaming: parse incremental chunks
reasoning_chunk, normal_chunk = parser.parse_stream_chunk(chunk_text)
```

#### Constructor Parameters

| Parameter          | Type                          | Default   | Description                                                       |
|--------------------|-------------------------------|-----------|-------------------------------------------------------------------|
| `model_type`       | `str`                         | required  | One of the keys from the DetectorMap above                        |
| `stream_reasoning` | `bool`                        | `True`    | If `False`, accumulates reasoning until end tag; if `True`, streams immediately |
| `force_reasoning`  | `Optional[bool]`              | `None`    | Override whether model is assumed to be reasoning at start         |
| `request`          | `ChatCompletionRequest`       | `None`    | Used for `continue_final_message` support                         |

#### Methods

| Method                 | Signature                                           | Description                                     |
|------------------------|-----------------------------------------------------|-------------------------------------------------|
| `parse_non_stream`     | `(full_text: str) -> Tuple[Optional[str], Optional[str]]` | One-time parsing; returns `(reasoning_text, normal_text)` |
| `parse_stream_chunk`   | `(chunk_text: str) -> Tuple[Optional[str], Optional[str]]` | Incremental parsing; returns `(reasoning_text, normal_text)` |

### Configuration Options

#### `separate_reasoning` Parameter

The `separate_reasoning` parameter on `ChatCompletionRequest` (default: `True`) controls whether reasoning content is separated from normal content in the API response.

When `True` (default), the response includes a separate `reasoning_content` field:

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "The answer is 42.",
      "reasoning_content": "I need to think about this step by step..."
    }
  }]
}
```

When `False`, reasoning tokens are included inline with the content.

```python
response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1",
    messages=[{"role": "user", "content": "What is 6 * 7?"}],
    extra_body={"separate_reasoning": False},
)
```

#### `stream_reasoning` Parameter

The `stream_reasoning` parameter on `ChatCompletionRequest` (default: `True`) controls how reasoning content is delivered in streaming mode:

- **`True`**: Reasoning tokens are streamed as they arrive, appearing in `delta.reasoning_content` chunks.
- **`False`**: Reasoning content is accumulated and only delivered once the `</think\>` end tag is encountered.

```python
# Stream reasoning tokens as they arrive
response = client.chat.completions.create(
    model="Qwen/Qwen3-235B-A22B",
    messages=[{"role": "user", "content": "Explain quantum entanglement"}],
    stream=True,
    extra_body={"stream_reasoning": True},
)

for chunk in response:
    delta = chunk.choices[0].delta
    if delta.reasoning_content:
        print(f"[THINK] {delta.reasoning_content}", end="")
    if delta.content:
        print(f"[ANSWER] {delta.content}", end="")
```

```bash
# curl example
curl -s http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "stream": true,
    "stream_reasoning": true,
    "separate_reasoning": true
  }'
```

#### `reasoning_effort` Levels

The `reasoning_effort` parameter constrains how much computational effort the model spends on reasoning.

| Level    | Description                                                                 |
|----------|-----------------------------------------------------------------------------|
| `none`   | Disables reasoning entirely. Sets `thinking=False` and `enable_thinking=False` in `chat_template_kwargs`. |
| `low`    | Minimal reasoning effort. Fastest response, least reasoning tokens.         |
| `medium` | Balanced reasoning effort.                                                  |
| `high`   | Maximum reasoning effort. Most thorough reasoning, most tokens used.        |

```python
# No reasoning (direct answer)
response = client.chat.completions.create(
    model="Qwen/Qwen3-235B-A22B",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    extra_body={"reasoning_effort": "none"},
)

# Maximum reasoning effort
response = client.chat.completions.create(
    model="Qwen/Qwen3-235B-A22B",
    messages=[{"role": "user", "content": "Prove the Riemann hypothesis"}],
    extra_body={"reasoning_effort": "high"},
)
```

The `reasoning_effort` parameter can also be passed inside a `reasoning` object:

```python
response = client.chat.completions.create(
    model="Qwen/Qwen3-235B-A22B",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    extra_body={
        "reasoning": {
            "effort": "none",
        }
    },
)
```

The `reasoning` object also supports an `enabled` / `enable` boolean that sets `thinking=True` in `chat_template_kwargs`:

```python
response = client.chat.completions.create(
    model="Qwen/Qwen3-235B-A22B",
    messages=[{"role": "user", "content": "Explain gravity"}],
    extra_body={
        "reasoning": {
            "enabled": True,
        }
    },
)
```

```bash
# curl example for reasoning_effort
curl -s http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-235B-A22B",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "reasoning_effort": "none"
  }'
```

#### `continue_final_message`

When `continue_final_message` is `True` and the last message has role `assistant`, the parser initializes its state from the previous content. This allows resuming generation from an incomplete response, properly tracking whether the model was in a reasoning block.

```python
response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1",
    messages=[
        {"role": "user", "content": "Write a poem"},
        {"role": "assistant", "content": "Let me think about this...\n<think\nSo far I have"},
    ],
    extra_body={"continue_final_message": True},
)
```

### Reasoning + Structured Outputs

When both reasoning and structured outputs are active, SGLang uses the `ReasonerGrammarBackend` to wrap the base grammar backend. This wrapper defers grammar constraint application until after the reasoning section ends:

- During reasoning (before `</think\>`): No grammar constraints are applied; the model reasons freely.
- After reasoning (after `</think\>`): Grammar constraints activate to enforce the output format.

This is configured automatically when the server is launched with `--reasoning-parser`.

```bash
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-R1 \
  --reasoning-parser deepseek-r1 \
  --grammar-backend xgrammar
```

---

## Structured Outputs

### JSON Schema Validation

SGLang can guarantee that model outputs conform to a JSON schema. This is useful for building reliable APIs, data extraction pipelines, and structured data generation.

#### Basic JSON Schema

```python
import json
import requests

json_schema = json.dumps({
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "population": {"type": "integer"},
        "is_capital": {"type": "boolean"},
    },
    "required": ["name", "population"],
})

response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "Give me info about Paris in JSON format.",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 128,
            "json_schema": json_schema,
        },
    },
)
print(response.json())
```

#### OpenAI-Compatible JSON Schema

```python
from openai import OpenAI
import json

client = OpenAI(base_url="http://localhost:30000/v1", api_key="None")

# Using json_schema response format
response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[
        {"role": "user", "content": "List 3 countries with their capitals."}
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "countries",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "countries": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "capital": {"type": "string"},
                            },
                            "required": ["name", "capital"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["countries"],
                "additionalProperties": False,
            },
        },
    },
)

data = json.loads(response.choices[0].message.content)
print(data)
```

```bash
# curl example
curl -s http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Tell me about France"}],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "country_info",
        "strict": true,
        "schema": {
          "type": "object",
          "properties": {
            "name": {"type": "string"},
            "capital": {"type": "string"},
            "population": {"type": "integer"}
          },
          "required": ["name", "capital", "population"],
          "additionalProperties": false
        }
      }
    }
  }'
```

### Response Format Types

SGLang supports three response format types through the OpenAI-compatible API:

| Type           | Description                                      | Constraint Applied                         |
|----------------|--------------------------------------------------|--------------------------------------------|
| `text`         | No constraint; free-form text output             | None                                       |
| `json_object`  | Output must be valid JSON (any structure)        | `json_schema = '{"type": "object"}'`      |
| `json_schema`  | Output must match the specified JSON schema      | Schema compiled to grammar                 |

#### `json_object` Format

```python
response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[
        {"role": "user", "content": "Generate a recipe for chocolate cake"}
    ],
    response_format={"type": "json_object"},
)
```

#### `text` Format (default)

```python
response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[
        {"role": "user", "content": "Write a poem about the ocean"}
    ],
    response_format={"type": "text"},
)
```

### StructuralTag Format

The StructuralTag format provides model-native tool call format constraints. It wraps begin/end markers around a JSON schema to guide structured generation of tool calls in the model's native format.

#### Legacy Structural Tag Format

```json
{
  "type": "structural_tag",
  "structures": [
    {
      "begin": "<tool_call\n",
      "schema": {"type": "object", "properties": {"name": {"type": "string"}, "arguments": {"type": "object"}}},
      "end": "\n</tool_call"
    }
  ],
  "triggers": ["<tool_call"],
  "at_least_one": true
}
```

#### Legacy Format Fields

| Field          | Type                           | Description                                           |
|----------------|--------------------------------|-------------------------------------------------------|
| `type`         | `Literal["structural_tag"]`    | Must be `"structural_tag"`                            |
| `structures`   | `List[StructuresResponseFormat]` | List of begin/schema/end structures                 |
| `triggers`     | `List[str]`                    | Trigger tokens that activate the structural tag        |
| `at_least_one` | `bool`                         | If `True`, at least one tool call is required          |

### Grammar Backends

SGLang supports multiple grammar backends for constrained decoding. The backend is selected via the `--grammar-backend` server flag.

#### Backend Comparison

| Feature           | xgrammar (default) | outlines          | llguidance         | none  |
|-------------------|--------------------|--------------------|--------------------|-------|
| JSON Schema       | Yes                | Yes                | Yes                | No    |
| Regular Expression| Yes                | Yes                | Yes                | No    |
| EBNF Grammar      | Yes                | No                  | Yes                | No    |
| Structural Tag    | Yes                | No                  | Yes                | No    |
| Jump Forward      | Yes                | Yes (limited)      | Yes (fast-forward) | N/A   |
| Bitmask Engine    | Triton/CUDA kernel | Boolean tensor     | llguidance torch   | N/A   |

#### XGrammar Backend (Default)

The XGrammar backend is the default and most feature-complete option.

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --grammar-backend xgrammar
```

Configuration options:
- `--constrained-json-disable-any-whitespace`: Disable any-whitespace matching in JSON schemas (default: disabled, meaning any-whitespace is enabled)
- `--constrained-json-whitespace-pattern`: Custom whitespace pattern for JSON schemas

#### Outlines Backend

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --grammar-backend outlines \
  --constrained-json-whitespace-pattern '( |\n)+'
```

Limitations:
- Does not support EBNF grammars
- Does not support Structural Tag format
- Regex support via interegular FSM compilation

#### Llguidance Backend

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --grammar-backend llguidance
```

Supports all constraint types (JSON, regex, EBNF, structural tag) using the `llguidance` library.

#### Disabling Grammar Backend

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --grammar-backend none
```

When disabled, structured output parameters (`json_schema`, `regex`, `ebnf`, `structural_tag`) will cause errors.

#### Grammar Caching

All grammar backends cache compiled grammar objects. When the same schema is requested again, the cached object is copied and reused. Cache hits avoid the compilation overhead entirely.

The `GrammarManager` handles compilation asynchronously using a thread pool and polls for completion to avoid blocking the main scheduling loop.

### Regex Constrained Output

```python
# Constrain output to match a regex pattern
response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "Paris is the capital of",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 64,
            "regex": "(France|England|Germany)",
        },
    },
)
```

```bash
curl -s http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "regex": "Paris|London|Berlin"
  }'
```

Common regex patterns:

```python
# Phone number
regex = "\\d{3}-\\d{3}-\\d{4}"

# Email (simplified)
regex = "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}"

# Choice from list
regex = "(yes|no|maybe)"

# Date
regex = "\\d{4}-\\d{2}-\\d{2}"
```

### EBNF Constrained Output

EBNF (Extended Backus-Naur Form) grammars provide the most flexible constraint mechanism. Only supported with xgrammar and llguidance backends.

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

More complex EBNF example:

```python
ebnf_grammar = '''
root ::= sentence
sentence ::= subject " " verb " " object "."
subject ::= "The cat" | "The dog" | "The bird"
verb ::= "chased" | "caught" | "watched"
object ::= "the mouse" | "the ball" | "the fish"
'''

response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "Write a simple sentence about animals.",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 64,
            "ebnf": ebnf_grammar,
        },
    },
)
```

```bash
curl -s http://localhost:30000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Write a greeting.",
    "sampling_params": {
      "temperature": 0,
      "max_new_tokens": 64,
      "ebnf": "root ::= \"Hello\" | \"Hi\" | \"Hey\""
    }
  }'
```

Note: XGrammar uses the GGML BNF format for EBNF grammars. See the [llama.cpp grammars README](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md) for format details.

### Constraint Mutual Exclusion

Only one of `json_schema`, `regex`, or `ebnf` can be specified per request. Setting multiple will raise a `ValueError`:

```python
# ERROR: Only one constraint allowed
sampling_params = {
    "json_schema": '{"type": "object"}',
    "regex": ".*",          # Conflict!
}
```

---

## Function Calling / Tool Use

### Tool Definitions

Tools are defined using the OpenAI-compatible schema. Each tool has a function with a name, description, and JSON schema for parameters.

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g., 'San Francisco'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current time for a timezone",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "IANA timezone, e.g., 'America/New_York'"
                    }
                },
                "required": ["timezone"],
            },
        },
    },
]

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[
        {"role": "user", "content": "What is the weather in Tokyo?"}
    ],
    tools=tools,
    tool_choice="auto",
)
```

#### Tool Definition Schema

| Field                      | Type                | Description                                       |
|----------------------------|---------------------|---------------------------------------------------|
| `type`                     | `str`               | Must be `"function"`                              |
| `function.name`            | `str`               | Function name                                     |
| `function.description`     | `Optional[str]`     | Human-readable description                        |
| `function.parameters`      | `Optional[object]`  | JSON Schema for function parameters               |
| `function.strict`          | `bool`              | If `True`, enforce strict schema validation       |
| `function.defer_loading`   | `Optional[bool]`    | If `True`, defer tool schema loading              |
| `defer_loading`            | `Optional[bool]`    | Top-level defer loading flag; propagates to function |

### `tool_choice` Options

The `tool_choice` parameter controls when and how the model calls tools.

| Value              | Type                    | Description                                                         |
|--------------------|-------------------------|---------------------------------------------------------------------|
| `"auto"`           | `str`                   | Model decides whether to call a tool or respond directly (default)  |
| `"required"`       | `str`                   | Model must call at least one tool                                    |
| `"none"`           | `str`                   | Model must not call any tools; disables tool use                     |
| `{"type": "function", "function": {"name": "..."}}` | `ToolChoice` | Model must call the specified function                |

```python
# Auto: model decides
response = client.chat.completions.create(
    model="...",
    messages=[...],
    tools=tools,
    tool_choice="auto",
)

# Required: must call at least one tool
response = client.chat.completions.create(
    model="...",
    messages=[...],
    tools=tools,
    tool_choice="required",
)

# None: disable tool calling
response = client.chat.completions.create(
    model="...",
    messages=[...],
    tools=tools,
    tool_choice="none",
)

# Named: must call specific function
response = client.chat.completions.create(
    model="...",
    messages=[...],
    tools=tools,
    tool_choice={"type": "function", "function": {"name": "get_weather"}},
)
```

Default behavior: If `tools` is provided but `tool_choice` is not explicitly set, it defaults to `"auto"`. If no `tools` are provided, `tool_choice` defaults to `"none"`.

### Parallel Tool Calls

The `parallel_tool_calls` parameter (default: `True`) controls whether the model can call multiple tools in a single response.

```python
response = client.chat.completions.create(
    model="...",
    messages=[{"role": "user", "content": "Get weather for Tokyo and London"}],
    tools=tools,
    parallel_tool_calls=True,  # Allow multiple tool calls in one response
)
```

When `parallel_tool_calls=False`, the model will make at most one tool call per response.

### Function Call Parser Types

SGLang supports multiple function call formats through pluggable detectors. The parser type is usually auto-detected from the model configuration, but can be specified explicitly.

#### Supported Parser Types

| Parser Key      | Detector Class          | Format Description                         |
|-----------------|-------------------------|--------------------------------------------|
| `deepseekv3`    | `DeepSeekV3Detector`    | DeepSeek V3 tool call format               |
| `deepseekv31`   | `DeepSeekV31Detector`   | DeepSeek V3.1 tool call format             |
| `deepseekv32`   | `DeepSeekV32Detector`   | DeepSeek V3.2 tool call format             |
| `glm`           | `Glm4MoeDetector`       | GLM-4/4.5 tool call format                 |
| `glm45`         | `Glm4MoeDetector`       | GLM-4.5 tool call format                   |
| `glm47`         | `Glm47MoeDetector`      | GLM-4.7 tool call format                   |
| `gpt-oss`       | `GptOssDetector`        | T4-style GPT-OSS format                    |
| `kimi_k2`       | `KimiK2Detector`        | Kimi K2 format                             |
| `lfm2`          | `Lfm2Detector`          | LFM2 format                                |
| `llama3`        | `Llama32Detector`       | Llama 3.2 function calling format          |
| `mimo`          | `MiMoDetector`          | MiMo format                                |
| `mistral`       | `MistralDetector`       | Mistral function calling format            |
| `pythonic`      | `PythonicDetector`      | Python-style function calls                |
| `qwen`          | `Qwen25Detector`        | Qwen 2.5 tool call format                  |
| `qwen25`        | `Qwen25Detector`        | Qwen 2.5 tool call format                  |
| `qwen3_coder`   | `Qwen3CoderDetector`    | Qwen3 Coder tool call format               |
| `step3`         | `Step3Detector`         | Step3 format                               |
| `step3p5`       | `Qwen3CoderDetector`    | Step3.5 uses Qwen3 Coder format            |
| `minimax-m2`    | `MinimaxM2Detector`     | MiniMax M2 format                          |
| `trinity`       | `TrinityDetector`       | Trinity format                             |
| `interns1`      | `InternlmDetector`      | Intern-S1 format                           |
| `hermes`        | `HermesDetector`        | Hermes format                              |
| `hunyuan`       | `HunyuanDetector`       | Hunyuan format                             |
| `gigachat3`     | `GigaChat3Detector`     | GigaChat 3 format                          |
| `gemma4`        | `Gemma4Detector`        | Gemma 4 format                             |

Source: `python/sglang/srt/function_call/function_call_parser.py`

### FunctionCallParser API

```python
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.entrypoints.openai.protocol import Tool

parser = FunctionCallParser(
    tools=[...],              # List of Tool objects
    tool_call_parser="hermes",  # Parser type
)

# Check if text contains a tool call
has_call = parser.has_tool_call(text)

# Non-streaming: parse full text
normal_text, tool_calls = parser.parse_non_stream(full_text)

# Streaming: parse incremental chunks
normal_text, tool_calls = parser.parse_stream_chunk(chunk_text)

# Get structural tag for constrained generation
structural_tag = parser.get_legacy_structural_tag(at_least_one=False)

# Get structure constraint based on tool_choice
constraint = parser.get_structure_constraint(
    tool_choice="required",
    parallel_tool_calls=True,
    thinking_mode=False,
)
```

### Custom Tool Servers

SGLang supports custom tool servers that can be invoked by the model during generation. This enables integration with external APIs and services.

#### Tool Reference Blocks

The `defer_loading` flag on tools allows lazy loading of tool schemas. When a tool has `defer_loading=True`, its full schema is not included in the initial prompt. Instead, the model can reference the tool by name using a `tool_reference` content block:

```python
# Tool reference in message content
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Check the weather in Tokyo"},
            {"type": "tool_reference", "name": "get_weather"},
        ],
    },
]
```

The chat template looks up `tools[*].function.name == tr.name` and renders the referenced tool schemas inline for the current turn.

### MCP Tool Server Integration

SGLang can integrate with Model Context Protocol (MCP) tool servers, enabling dynamic tool discovery and invocation through a standardized protocol.

### Strict Mode

The `strict` flag on tool functions controls whether the grammar enforces the parameter schema:

- `strict=True`: The grammar only allows parameters that exactly match the schema
- `strict=False`: The grammar accepts any JSON, allowing more flexible output

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                },
                "required": ["location"],
            },
            "strict": True,  # Enforce exact schema
        },
    },
]
```

The global strictness level can also be controlled via the `SGLANG_TOOL_STRICT_LEVEL` environment variable, which has two levels:

- **PARAMETER level**: Strict when individual function has `strict=True`
- **FUNCTION level**: Apply structural tags even for `tool_choice="auto"` when any tool has `strict=True`

### Tool Call Response Handling

```python
# Complete tool use flow
response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "What is the weather in Tokyo?"}],
    tools=tools,
    tool_choice="auto",
)

# Check if model wants to call a tool
if response.choices[0].finish_reason == "tool_calls":
    tool_calls = response.choices[0].message.tool_calls

    # Execute the tool calls and add results
    messages = [
        {"role": "user", "content": "What is the weather in Tokyo?"},
        response.choices[0].message,  # Include the assistant's tool call
    ]

    for tool_call in tool_calls:
        # Execute the function
        result = execute_tool(tool_call.function.name, tool_call.function.arguments)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": str(result),
        })

    # Get the final response
    final_response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        messages=messages,
    )
```

```bash
# curl example for tool calling
curl -s http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string"}
          },
          "required": ["location"]
        }
      }
    }],
    "tool_choice": "auto"
  }'
```

---

## Custom Logit Processors

Custom logit processors allow users to implement arbitrary modifications to the logits tensor before token sampling. This enables advanced use cases like token banning, custom sampling strategies, and thinking budget control.

### Enabling Custom Logit Processors

The server must be started with the `--enable-custom-logit-processor` flag:

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3-8B-Instruct \
  --port 30000 \
  --enable-custom-logit-processor
```

### CustomLogitProcessor API

```python
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor
import torch
from typing import Any, Dict, List, Optional

class MyCustomProcessor(CustomLogitProcessor):
    """Custom logit processor example."""

    def __call__(
        self,
        logits: torch.Tensor,       # Shape: [batch_size, vocab_size]
        custom_param_list: Optional[List[Dict[str, Any]]] = None,
    ) -> torch.Tensor:
        """
        Modify logits before sampling.

        Args:
            logits: Raw logits tensor of shape [batch_size, vocab_size]
            custom_param_list: Per-request parameters from sampling_params.custom_params

        Returns:
            Modified logits tensor of the same shape
        """
        assert logits.shape[0] == len(custom_param_list)

        for i, params in enumerate(custom_param_list):
            # Apply custom logic per request
            pass

        return logits

# Serialize for transport
processor_str = MyCustomProcessor.to_str()
```

#### Serialization

Custom logit processors are serialized using `dill` and transmitted as JSON strings:

```python
# Serialize
serialized = DeterministicLogitProcessor.to_str()
# Returns: {"callable": "<hex-encoded dill bytes>"}

# Deserialize
processor = CustomLogitProcessor.from_str(serialized)
```

The serialization uses a cached `_cache_from_str` function with `@lru_cache(maxsize=None)` for efficiency.

### Built-in Processors

#### DisallowedTokensLogitsProcessor

Bans specific token IDs from being generated:

```python
from sglang.srt.sampling.custom_logit_processor import DisallowedTokensLogitsProcessor

response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "Write a sentence",
        "custom_logit_processor": DisallowedTokensLogitsProcessor().to_str(),
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": 64,
            "custom_params": {"token_ids": [13, 14, 15]},  # Token IDs to ban
        },
    },
)
```

#### ThinkingBudgetLogitProcessor

Controls the maximum length of the thinking/reasoning section. This is useful for limiting how long a reasoning model can think.

Built-in variants for specific models:

| Processor Class                        | Model           | THINKING_START_TOKEN_ID | THINKING_END_TOKEN_ID | NEW_LINE_TOKEN_ID |
|----------------------------------------|-----------------|-------------------------|------------------------|-------------------|
| `Glm4MoeThinkingBudgetLogitProcessor`  | GLM-4.5/4.6     | 151350                  | 151351                 | 198               |
| `Qwen3ThinkingBudgetLogitProcessor`    | Qwen3           | 151667                  | 151668                 | 198               |
| `DeepSeekR1ThinkingBudgetLogitProcessor`| DeepSeek-R1    | 128798                  | 128799                 | 201               |

```python
from sglang.srt.sampling.custom_logit_processor import Qwen3ThinkingBudgetLogitProcessor

response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "...",
        "custom_logit_processor": Qwen3ThinkingBudgetLogitProcessor().to_str(),
        "sampling_params": {
            "max_new_tokens": 2048,
            "custom_params": {"thinking_budget": 500},  # Max thinking tokens
        },
    },
)
```

#### DeepseekOCRNoRepeatNGramLogitProcessor

Prevents n-gram repetitions within a sliding window, useful for OCR and structured text generation:

```python
from sglang.srt.sampling.custom_logit_processor import DeepseekOCRNoRepeatNGramLogitProcessor

response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "...",
        "custom_logit_processor": DeepseekOCRNoRepeatNGramLogitProcessor().to_str(),
        "sampling_params": {
            "max_new_tokens": 1024,
            "custom_params": {
                "ngram_size": 3,           # Size of n-grams to check
                "window_size": 100,         # Sliding window size
                "whitelist_token_ids": [1, 2],  # Tokens exempt from banning
            },
        },
    },
)
```

### Writing a Custom Processor

#### Example: Temperature Per Token

```python
import torch
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor
from typing import Any, Dict, List, Optional

class TemperatureScheduleProcessor(CustomLogitProcessor):
    """Apply different temperatures based on generation progress."""

    def __call__(
        self,
        logits: torch.Tensor,
        custom_param_list: Optional[List[Dict[str, Any]]] = None,
    ) -> torch.Tensor:
        for i, params in enumerate(custom_param_list):
            if params is None:
                continue

            # Get request object for token counting
            req = params.get("__req__")
            if req is None:
                continue

            num_generated = len(req.output_ids)
            max_tokens = params.get("max_tokens", 100)

            # Progress ratio: 0.0 at start, 1.0 at end
            progress = min(num_generated / max_tokens, 1.0)

            # Start with high temperature, decrease over time
            temp = max(0.1, 2.0 * (1.0 - progress))

            # Apply temperature
            logits[i] = logits[i] / temp

        return logits
```

#### Example: Category-Constrained Output

```python
class CategoryConstrainedProcessor(CustomLogitProcessor):
    """Only allow tokens that form valid category names."""

    def __call__(
        self,
        logits: torch.Tensor,
        custom_param_list: Optional[List[Dict[str, Any]]] = None,
    ) -> torch.Tensor:
        for i, params in enumerate(custom_param_list):
            if params is None:
                continue

            allowed_token_ids = params.get("allowed_token_ids", [])
            if not allowed_token_ids:
                continue

            # Mask all tokens except allowed ones
            logits[i, :] = -float("inf")
            logits[i, allowed_token_ids] = 0.0

        return logits
```

### Using Custom Processors via OpenAI API

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="None")

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0.0,
    max_tokens=32,
    extra_body={
        "custom_logit_processor": DeterministicLogitProcessor().to_str(),
        "custom_params": {"token_id": 5},
    },
)
```

### Integration with Sampling Pipeline

Custom logit processors are integrated into the `SamplingBatchInfo`:

1. Requests with custom logit processors are detected during batch preparation
2. Processors are deserialized and grouped by their serialized string (identical processors are batched)
3. A boolean mask tensor indicates which requests in the batch use which processor
4. Custom params are collected per request
5. During sampling, processors are applied after penalties and grammar masks but before final token selection

The batch merging and filtering logic correctly handles custom logit processors when batches are split or combined during continuous batching.

---

## Sampling Parameters (Complete Reference)

### Overview

SGLang's sampling parameters control how tokens are selected during generation. They are specified via the `sampling_params` field in the native `/generate` API or mapped from OpenAI API fields.

Two sources for default values:
- **`model` defaults** (default): Read from the model's `generation_config.json`
- **`openai` defaults**: Use constant SGLang/OpenAI default values

```bash
# Use model-provided defaults (default)
python -m sglang.launch_server --model-path <MODEL> --sampling-defaults model

# Use SGLang/OpenAI constant defaults
python -m sglang.launch_server --model-path <MODEL> --sampling-defaults openai
```

### Core Parameters

#### `temperature`

| Attribute     | Value                         |
|---------------|-------------------------------|
| Type          | `float`                       |
| Default       | Model config; fallback `1.0`  |
| Valid Range   | `[0, +inf)`                   |
| API Field     | `temperature`                 |

Controls randomness in token selection. `temperature=0` enables greedy sampling (always pick the highest-probability token). Higher values increase diversity.

Internally, when `0 <= temperature < 1e-6`, SGLang sets `temperature=1.0` and `top_k=1` to achieve greedy sampling.

```python
# Greedy (deterministic)
sampling_params = {"temperature": 0}

# Balanced
sampling_params = {"temperature": 0.7}

# Creative
sampling_params = {"temperature": 1.5}
```

#### `top_p` (Nucleus Sampling)

| Attribute     | Value                         |
|---------------|-------------------------------|
| Type          | `float`                       |
| Default       | Model config; fallback `1.0`  |
| Valid Range   | `(0, 1]`                      |
| API Field     | `top_p`                       |

Selects tokens from the smallest sorted set whose cumulative probability exceeds `top_p`. When `top_p=1`, all tokens are considered (no filtering).

```python
# Conservative (top 50% of probability mass)
sampling_params = {"top_p": 0.5}

# Default (no filtering)
sampling_params = {"top_p": 1.0}
```

#### `top_k`

| Attribute     | Value                         |
|---------------|-------------------------------|
| Type          | `int`                         |
| Default       | Model config; fallback `-1`   |
| Valid Range   | `-1` (disable) or `>= 1`      |
| API Field     | `top_k`                       |

Randomly selects from the `k` highest-probability tokens. `-1` disables top-k filtering (the internal value is set to `2^30`, effectively the whole vocabulary).

```python
# Top-10 sampling
sampling_params = {"top_k": 10}

# Top-1 (greedy)
sampling_params = {"top_k": 1}

# No top-k filtering
sampling_params = {"top_k": -1}
```

#### `min_p`

| Attribute     | Value                         |
|---------------|-------------------------------|
| Type          | `float`                       |
| Default       | Model config; fallback `0.0`  |
| Valid Range   | `[0, 1]`                      |
| API Field     | `min_p`                       |

Filters out tokens whose probability is less than `min_p * highest_token_probability`. This dynamically adapts to the confidence of the model.

```python
# Only consider tokens within 10% of the highest probability
sampling_params = {"min_p": 0.1}

# No min-p filtering
sampling_params = {"min_p": 0.0}
```

### Penalizers

#### `frequency_penalty`

| Attribute     | Value          |
|---------------|----------------|
| Type          | `float`        |
| Default       | `0.0`          |
| Valid Range   | `[-2, 2]`      |
| API Field     | `frequency_penalty` |

Penalizes tokens based on their frequency in the generation so far. The penalty grows linearly with each appearance of a token. Positive values discourage repetition; negative values encourage it.

```python
# Discourage repetition
sampling_params = {"frequency_penalty": 0.5}

# Encourage repetition
sampling_params = {"frequency_penalty": -0.5}
```

#### `presence_penalty`

| Attribute     | Value          |
|---------------|----------------|
| Type          | `float`        |
| Default       | `0.0`          |
| Valid Range   | `[-2, 2]`      |
| API Field     | `presence_penalty` |

Penalizes tokens if they have appeared at all in the generation so far. Unlike frequency_penalty, the penalty is constant regardless of how many times the token appeared.

```python
# Encourage new topics
sampling_params = {"presence_penalty": 0.6}
```

#### `repetition_penalty`

| Attribute     | Value          |
|---------------|----------------|
| Type          | `float`        |
| Default       | `1.0`          |
| Valid Range   | `[0, 2]`       |
| API Field     | `repetition_penalty` |

Scales the logits of previously generated tokens. Values `> 1` discourage repetition; values `< 1` encourage it. `1.0` leaves probabilities unchanged.

```python
# Reduce repetition
sampling_params = {"repetition_penalty": 1.2}
```

#### `min_new_tokens`

| Attribute     | Value          |
|---------------|----------------|
| Type          | `int`          |
| Default       | `0`            |
| Valid Range   | `[0, max_new_tokens]` |
| API Field     | `min_tokens`   |

Forces the model to generate at least `min_new_tokens` before a stop word or EOS token can terminate generation. The penalty is implemented by suppressing EOS and stop tokens until the minimum is reached.

```python
# Ensure at least 100 tokens are generated
sampling_params = {"min_tokens": 100, "max_new_tokens": 500}
```

### Output Length Parameters

#### `max_new_tokens` / `max_completion_tokens`

| Attribute           | Value                   |
|---------------------|-------------------------|
| Type                | `int`                   |
| Default             | `128`                   |
| Valid Range         | `>= 0`                  |
| Native API Field    | `max_new_tokens`        |
| OpenAI API Field    | `max_completion_tokens` or `max_tokens` |

The maximum number of tokens to generate. In the OpenAI API, `max_completion_tokens` is preferred; `max_tokens` is deprecated but still supported.

```python
# Native API
response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "Tell me a story",
        "sampling_params": {"max_new_tokens": 500},
    },
)

# OpenAI API
response = client.chat.completions.create(
    model="...",
    messages=[...],
    max_completion_tokens=500,  # Preferred
    # max_tokens=500,           # Deprecated but works
)
```

#### `ignore_eos`

| Attribute     | Value     |
|---------------|-----------|
| Type          | `bool`    |
| Default       | `False`   |
| API Field     | `ignore_eos` |

When `True`, the model continues generating even after the EOS token is sampled. Useful for generating text that naturally contains EOS-like sequences.

```python
sampling_params = {"ignore_eos": True, "max_new_tokens": 500}
```

#### `no_stop_trim`

| Attribute     | Value     |
|---------------|-----------|
| Type          | `bool`    |
| Default       | `False`   |
| API Field     | `no_stop_trim` |

When `True`, stop words and EOS tokens are not trimmed from the generated text. The raw output including stop tokens is returned.

```python
sampling_params = {"no_stop_trim": True}
```

### Stop Sequences

#### `stop` (String Stop Words)

| Attribute     | Value                              |
|---------------|------------------------------------|
| Type          | `Optional[Union[str, List[str]]]`  |
| Default       | `None`                             |
| API Field     | `stop`                             |

One or more stop strings. Generation stops when one of these strings is sampled.

```python
# Single stop word
sampling_params = {"stop": "\n"}

# Multiple stop words
sampling_params = {"stop": ["\n", "END", "```"]}
```

```python
# OpenAI API
response = client.chat.completions.create(
    model="...",
    messages=[...],
    stop=["\n\n", "###"],
)
```

#### `stop_token_ids`

| Attribute     | Value              |
|---------------|--------------------|
| Type          | `Optional[List[int]]` |
| Default       | `None`             |
| API Field     | `stop_token_ids`   |

Stop words specified as token IDs. Generation stops when one of these token IDs is sampled.

```python
sampling_params = {"stop_token_ids": [128000, 128001]}
```

#### `stop_regex`

| Attribute     | Value                              |
|---------------|------------------------------------|
| Type          | `Optional[Union[str, List[str]]]`  |
| Default       | `None`                             |
| API Field     | `stop_regex`                       |

Regular expression patterns. Generation stops when the output matches any of these patterns.

```python
sampling_params = {"stop_regex": ["\n\n\n", "---+"]}
```

### Seed for Deterministic Inference

#### `seed` / `sampling_seed`

| Attribute          | Value              |
|--------------------|--------------------|
| Type               | `Optional[int]`    |
| Default            | `None`             |
| OpenAI API Field   | `seed`             |
| Native API Field   | `sampling_seed`    |

When set, enables deterministic sampling. The same seed with the same input and parameters produces the same output. Requires `--enable-deterministic-inference` server flag.

```bash
python -m sglang.launch_server \
  --model-path <MODEL> \
  --enable-deterministic-inference
```

```python
response = client.chat.completions.create(
    model="...",
    messages=[...],
    seed=42,
)
```

### Logprobs

#### `return_logprob` (Native API)

| Attribute     | Value     |
|---------------|-----------|
| Type          | `bool`    |
| Default       | `False`   |
| API Field     | `return_logprob` |

Whether to return log probabilities for generated tokens.

#### `logprob_start_len`

| Attribute     | Value     |
|---------------|-----------|
| Type          | `int`     |
| Default       | `-1`      |
| API Field     | `logprob_start_len` |

The start position in the prompt for returning logprobs. `-1` returns logprobs for output tokens only.

#### `top_logprobs_num`

| Attribute     | Value     |
|---------------|-----------|
| Type          | `int`     |
| Default       | N/A       |
| API Field     | `top_logprobs_num` |

Number of top logprobs to return at each position.

#### `logprobs` / `top_logprobs` (OpenAI API)

```python
response = client.chat.completions.create(
    model="...",
    messages=[...],
    logprobs=True,
    top_logprobs=5,
)

# Access logprobs
for token_info in response.choices[0].logprobs.content:
    print(f"Token: {token_info.token}, LogProb: {token_info.logprob}")
    for top in token_info.top_logprobs:
        print(f"  {top.token}: {top.logprob}")
```

### `n` (Multiple Completions)

| Attribute     | Value     |
|---------------|-----------|
| Type          | `int`     |
| Default       | `1`       |
| API Field     | `n`       |

Specifies the number of output sequences to generate per request. Note: generating multiple outputs in one request (`n > 1`) is discouraged; repeating the same prompts several times offers better control and efficiency.

```python
response = client.chat.completions.create(
    model="...",
    messages=[...],
    n=3,  # Generate 3 completions
    temperature=0.7,
)

for i, choice in enumerate(response.choices):
    print(f"Completion {i}: {choice.message.content}")
```

### `continue_final_message`

| Attribute     | Value     |
|---------------|-----------|
| Type          | `bool`    |
| Default       | `False`   |
| API Field     | `continue_final_message` |

When `True` and the last message has role `assistant`, generation continues from the last assistant message instead of starting a new one.

```python
response = client.chat.completions.create(
    model="...",
    messages=[
        {"role": "user", "content": "Write a story"},
        {"role": "assistant", "content": "Once upon a time, in a land far away,"},
    ],
    extra_body={"continue_final_message": True},
)
```

### `logit_bias`

| Attribute     | Value                        |
|---------------|------------------------------|
| Type          | `Optional[Dict[str, float]]` |
| Default       | `None`                       |
| API Field     | `logit_bias`                 |

Modifies the likelihood of specific tokens appearing in the output. Keys are token IDs (as strings), and values are the bias to add to the logits (-100 to 100). A bias of -100 effectively bans the token.

```python
response = client.chat.completions.create(
    model="...",
    messages=[...],
    logit_bias={"1234": -100, "5678": 5},  # Ban token 1234, boost 5678
)
```

### Text Processing Options

#### `skip_special_tokens`

| Attribute     | Value     |
|---------------|-----------|
| Type          | `bool`    |
| Default       | `True`    |
| API Field     | `skip_special_tokens` |

Remove special tokens during decoding.

#### `spaces_between_special_tokens`

| Attribute     | Value     |
|---------------|-----------|
| Type          | `bool`    |
| Default       | `True`    |
| API Field     | `spaces_between_special_tokens` |

Whether to add spaces between special tokens during detokenization. Can be overridden via `chat_template_kwargs`.

### Complete Parameter Summary Table

#### Native `/generate` Endpoint

| Argument                   | Type/Default                                              | Description                                           |
|----------------------------|-----------------------------------------------------------|-------------------------------------------------------|
| `text`                     | `Optional[Union[List[str], str]]` = `None`               | Input prompt                                          |
| `input_ids`                | `Optional[Union[List[List[int]], List[int]]]` = `None`   | Token IDs for input                                   |
| `input_embeds`             | `Optional[...]` = `None`                                  | Precomputed embeddings                                |
| `image_data`               | `Optional[...]` = `None`                                  | Image input (file path, URL, or base64)               |
| `audio_data`               | `Optional[...]` = `None`                                  | Audio input                                           |
| `sampling_params`          | `Optional[Union[List[Dict], Dict]]` = `None`             | Sampling parameters (see below)                       |
| `rid`                      | `Optional[Union[List[str], str]]` = `None`               | Request ID                                            |
| `return_logprob`           | `Optional[Union[List[bool], bool]]` = `None`             | Return log probabilities                              |
| `logprob_start_len`        | `Optional[Union[List[int], int]]` = `None`               | Start position for logprobs                           |
| `top_logprobs_num`         | `Optional[Union[List[int], int]]` = `None`               | Number of top logprobs per position                   |
| `token_ids_logprob`        | `Optional[...]` = `None`                                  | Specific token IDs to return logprobs for             |
| `return_text_in_logprobs`  | `bool` = `False`                                          | Detokenize tokens in logprobs                         |
| `stream`                   | `bool` = `False`                                          | Stream output                                         |
| `lora_path`                | `Optional[...]` = `None`                                  | Path to LoRA adapter                                  |
| `custom_logit_processor`   | `Optional[...]` = `None`                                  | Serialized custom logit processor                     |
| `return_hidden_states`     | `Union[List[bool], bool]` = `False`                       | Return hidden states                                  |
| `return_routed_experts`    | `bool` = `False`                                          | Return routed experts for MoE models                  |

#### SamplingParams Object

| Argument                    | Type/Default                    | Description                                    |
|-----------------------------|---------------------------------|------------------------------------------------|
| `max_new_tokens`            | `int` = `128`                   | Maximum output length in tokens                |
| `stop`                      | `Optional[Union[str, List[str]]]` = `None` | Stop words                            |
| `stop_token_ids`            | `Optional[List[int]]` = `None`  | Stop token IDs                                 |
| `stop_regex`                | `Optional[Union[str, List[str]]]` = `None` | Stop regex patterns                  |
| `temperature`               | `float` (model default; fb `1.0`) | Sampling temperature                        |
| `top_p`                     | `float` (model default; fb `1.0`) | Nucleus sampling threshold                   |
| `top_k`                     | `int` (model default; fb `-1`)  | Top-k sampling                                 |
| `min_p`                     | `float` (model default; fb `0.0`) | Minimum probability threshold                |
| `frequency_penalty`         | `float` = `0.0`                 | Frequency-based repetition penalty             |
| `presence_penalty`          | `float` = `0.0`                 | Presence-based repetition penalty              |
| `repetition_penalty`        | `float` = `1.0`                 | Logit scaling for repetition                   |
| `min_new_tokens`            | `int` = `0`                     | Minimum tokens before stopping                 |
| `n`                         | `int` = `1`                     | Number of completions per request              |
| `json_schema`               | `Optional[str]` = `None`        | JSON schema for structured output              |
| `regex`                     | `Optional[str]` = `None`        | Regex for structured output                    |
| `ebnf`                      | `Optional[str]` = `None`        | EBNF grammar for structured output             |
| `structural_tag`            | `Optional[str]` = `None`        | Structural tag for constrained generation      |
| `ignore_eos`                | `bool` = `False`                | Don't stop on EOS token                        |
| `skip_special_tokens`       | `bool` = `True`                 | Remove special tokens in output                |
| `spaces_between_special_tokens` | `bool` = `True`             | Add spaces between special tokens              |
| `no_stop_trim`              | `bool` = `False`                | Don't trim stop words from output              |
| `custom_params`             | `Optional[Dict]` = `None`       | Parameters for custom logit processor          |
| `stream_interval`           | `Optional[int]` = `None`        | Streaming interval                             |
| `logit_bias`                | `Optional[Dict[str, float]]` = `None` | Per-token logit bias                      |
| `sampling_seed`             | `Optional[int]` = `None`        | Seed for deterministic sampling                |

---

## Advanced Sampling

### Session-Based Generation

Session parameters allow maintaining state across multiple requests, enabling multi-turn conversations with prefix caching.

```python
response = requests.post(
    "http://localhost:30000/v1/chat/completions",
    json={
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "messages": [...],
        "session_params": {
            "id": "my-session-123",  # Session identifier
        },
    },
)
```

### Priority Scheduling

Requests can be assigned a priority level. Higher-priority requests are scheduled before lower-priority ones.

```python
response = requests.post(
    "http://localhost:30000/v1/chat/completions",
    json={
        "model": "...",
        "messages": [...],
        "priority": 10,  # Higher number = higher priority
    },
)
```

```bash
curl -s http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "...",
    "messages": [{"role": "user", "content": "Hello"}],
    "priority": 10
  }'
```

### Cache Salt for Prefix Caching

The `cache_salt` parameter adds a salt key to the prefix cache, allowing different cache entries for the same prefix but different contexts. This is useful for isolating cache entries between different tenants or use cases.

```python
response = requests.post(
    "http://localhost:30000/v1/chat/completions",
    json={
        "model": "...",
        "messages": [...],
        "cache_salt": "tenant-abc-123",  # Cache isolation key
    },
)
```

```bash
curl -s http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "...",
    "messages": [{"role": "user", "content": "Hello"}],
    "cache_salt": "my-unique-salt"
  }'
```

### Batch Inference Patterns

SGLang supports multiple batch inference patterns for high-throughput scenarios.

#### Multiple Prompts in One Request

```python
# Native API: batch of prompts
response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": [
            "The capital of France is",
            "The capital of Germany is",
            "The capital of Italy is",
        ],
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 32,
        },
    },
)
```

#### Per-Prompt Sampling Parameters

```python
response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": [
            "Write a creative story",
            "What is 2+2?",
        ],
        "sampling_params": [
            {"temperature": 1.0, "max_new_tokens": 256},
            {"temperature": 0, "max_new_tokens": 32},
        ],
    },
)
```

### Data Parallel Routing

Requests can be routed to specific data-parallel workers using the `routed_dp_rank` parameter:

```python
response = requests.post(
    "http://localhost:30000/v1/chat/completions",
    json={
        "model": "...",
        "messages": [...],
        "routed_dp_rank": 2,  # Route to DP worker 2
    },
)
```

### PD Disaggregation Bootstrap

For prefill-decode disaggregation, bootstrap parameters allow the decode worker to connect to the prefill worker's KV cache:

```python
response = requests.post(
    "http://localhost:30000/v1/chat/completions",
    json={
        "model": "...",
        "messages": [...],
        "bootstrap_host": "10.0.0.1",
        "bootstrap_port": 30001,
        "bootstrap_room": 42,
        "disagg_prefill_dp_rank": 0,
    },
)
```

### Complete Server Launch Example

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 30000 \
  --host 0.0.0.0 \
  --grammar-backend xgrammar \
  --reasoning-parser deepseek-r1 \
  --enable-custom-logit-processor \
  --enable-deterministic-inference \
  --sampling-defaults model
```

### Complete Request Example with All Features

```python
import json
import requests
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor
import torch
from typing import Any, Dict, List, Optional

# Define a custom processor
class QualityBoostProcessor(CustomLogitProcessor):
    def __call__(self, logits, custom_param_list=None):
        for i, params in enumerate(custom_param_list or []):
            if params is None:
                continue
            # Apply quality-boosting logic
            boost_tokens = params.get("boost_tokens", [])
            for tid in boost_tokens:
                logits[i, tid] += params.get("boost_amount", 1.0)
        return logits

# Build the request
response = requests.post(
    "http://localhost:30000/v1/chat/completions",
    json={
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me about Paris"},
        ],
        # Sampling parameters
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "min_p": 0.05,
        "max_completion_tokens": 256,
        "frequency_penalty": 0.3,
        "presence_penalty": 0.2,
        "repetition_penalty": 1.1,
        "seed": 42,
        "stop": ["\n\n\n", "==="],

        # Structured output
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "city_info",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "country": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["name", "country", "description"],
                    "additionalProperties": False,
                },
            },
        },

        # Custom logit processor
        "custom_logit_processor": QualityBoostProcessor().to_str(),
        "custom_params": {"boost_tokens": [1234, 5678], "boost_amount": 0.5},

        # Advanced options
        "cache_salt": "my-session",
        "priority": 5,
        "separate_reasoning": True,
        "stream_reasoning": True,
    },
)

result = response.json()
print(json.dumps(result, indent=2))
```

---

## Source Files

| Component                  | Path                                                              |
|----------------------------|-------------------------------------------------------------------|
| SamplingParams             | `python/sglang/srt/sampling/sampling_params.py`                  |
| SamplingBatchInfo          | `python/sglang/srt/sampling/sampling_batch_info.py`              |
| CustomLogitProcessor       | `python/sglang/srt/sampling/custom_logit_processor.py`           |
| Penalty Library            | `python/sglang/srt/sampling/penaltylib/`                         |
| ReasoningParser            | `python/sglang/srt/parser/reasoning_parser.py`                   |
| BaseGrammarBackend         | `python/sglang/srt/constrained/base_grammar_backend.py`          |
| XGrammarBackend            | `python/sglang/srt/constrained/xgrammar_backend.py`              |
| OutlinesBackend            | `python/sglang/srt/constrained/outlines_backend.py`              |
| LlguidanceBackend          | `python/sglang/srt/constrained/llguidance_backend.py`            |
| ReasonerGrammarBackend     | `python/sglang/srt/constrained/reasoner_grammar_backend.py`      |
| GrammarManager             | `python/sglang/srt/constrained/grammar_manager.py`               |
| FunctionCallParser         | `python/sglang/srt/function_call/function_call_parser.py`        |
| BaseFormatDetector         | `python/sglang/srt/function_call/base_format_detector.py`        |
| Core Types                 | `python/sglang/srt/function_call/core_types.py`                  |
| OpenAI Protocol            | `python/sglang/srt/entrypoints/openai/protocol.py`               |
| Sampling Params Docs       | `docs/basic_usage/sampling_params.md`                             |
