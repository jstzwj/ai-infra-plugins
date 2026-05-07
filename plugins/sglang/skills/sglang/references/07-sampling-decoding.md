# SGLang Sampling and Decoding Reference

This document provides a comprehensive reference for all sampling parameters, structured outputs, grammar backends, custom logit processors, and tool calling parsers in SGLang.

## Table of Contents

- [Sampling Parameters Overview](#sampling-parameters-overview)
- [Generate Endpoint Parameters](#generate-endpoint-parameters)
- [Core Sampling Parameters](#core-sampling-parameters)
- [Penalty Parameters](#penalty-parameters)
- [Constrained Decoding Parameters](#constrained-decoding-parameters)
- [Other Sampling Options](#other-sampling-options)
- [Structured Outputs](#structured-outputs)
- [Grammar Backends](#grammar-backends)
- [Custom Logit Processors](#custom-logit-processors)
- [Reasoning/Thinking Model Support](#reasoningthinking-model-support)
- [Tool Calling Parsers](#tool-calling-parsers)
- [Sampling Defaults Configuration](#sampling-defaults-configuration)
- [Sampling Backend Architecture](#sampling-backend-architecture)

---

## Sampling Parameters Overview

SGLang provides fine-grained control over text generation through sampling parameters. These parameters can be specified per-request via the `/generate` endpoint's `sampling_params` field or through the OpenAI-compatible API.

Parameters are defined in `python/sglang/srt/sampling/sampling_params.py` (SamplingParams class) and `python/sglang/srt/sampling/sampling_batch_info.py` (batch-level sampling metadata).

### Note on Defaults

By default, SGLang initializes several sampling parameters from the model's `generation_config.json` (when launched with `--sampling-defaults model`, which is the default). To use constant SGLang/OpenAI defaults instead:

```bash
# Use model-provided defaults (default behavior)
python -m sglang.launch_server --model-path <MODEL> --sampling-defaults model

# Use SGLang/OpenAI constant defaults
python -m sglang.launch_server --model-path <MODEL> --sampling-defaults openai
```

You can always override any parameter per request via `sampling_params`.

---

## Generate Endpoint Parameters

The `/generate` endpoint accepts the following top-level parameters:

| Argument | Type/Default | Description |
|---|---|---|
| text | `Optional[Union[List[str], str]] = None` | The input prompt. Single or batch. |
| input_ids | `Optional[Union[List[List[int]], List[int]]] = None` | Token IDs; specify text or input_ids. |
| input_embeds | `Optional[Union[List[List[List[float]]], List[List[float]]]] = None` | Embeddings for input_ids. |
| image_data | `Optional[...] = None` | Image input. Supports raw images (PIL, path, URL, base64), processor output dicts, or precomputed embeddings. |
| audio_data | `Optional[...] = None` | Audio input. File name, URL, or base64. |
| sampling_params | `Optional[Union[List[Dict], Dict]] = None` | Sampling parameters. |
| rid | `Optional[Union[List[str], str]] = None` | Request ID. |
| return_logprob | `Optional[Union[List[bool], bool]] = None` | Whether to return log probabilities. |
| logprob_start_len | `Optional[Union[List[int], int]] = None` | Start location for returning logprobs. Default -1 (output tokens only). |
| top_logprobs_num | `Optional[Union[List[int], int]] = None` | Number of top logprobs per position. |
| token_ids_logprob | `Optional[Union[List[List[int]], List[int]]] = None` | Specific token IDs to return logprob for. |
| return_text_in_logprobs | `bool = False` | Whether to detokenize tokens in logprobs. |
| stream | `bool = False` | Whether to stream output. |
| lora_path | `Optional[...] = None` | Path to LoRA adapter. |
| custom_logit_processor | `Optional[...] = None` | Custom logit processor (serialized). |
| return_hidden_states | `Union[List[bool], bool] = False` | Whether to return hidden states. |
| return_routed_experts | `bool = False` | Whether to return routed experts for MoE models. Requires `--enable-return-routed-experts`. |

---

## Core Sampling Parameters

| Argument | Type/Default | Description |
|---|---|---|
| **max_new_tokens** | `int = 128` | Maximum output length in tokens. |
| **stop** | `Optional[Union[str, List[str]]] = None` | Stop words. Generation stops when one is sampled. |
| **stop_token_ids** | `Optional[List[int]] = None` | Stop words as token IDs. |
| **stop_regex** | `Optional[Union[str, List[str]]] = None` | Stop when hitting any regex pattern. |
| **temperature** | `float (model default; fallback 1.0)` | Sampling temperature. 0 = greedy; higher = more diverse. |
| **top_p** | `float (model default; fallback 1.0)` | Top-p (nucleus) sampling. Selects from smallest sorted set whose cumulative probability exceeds top_p. 1.0 = unrestricted. |
| **top_k** | `int (model default; fallback -1)` | Top-k sampling. Randomly selects from k highest-probability tokens. -1 = disabled. |
| **min_p** | `float (model default; fallback 0.0)` | Min-p sampling. Samples from tokens with probability > min_p * highest_token_probability. |

### Temperature Behavior

- `temperature = 0`: Greedy decoding (always selects highest probability token)
- `temperature = 0 < t < 1`: Less random, more focused
- `temperature = 1`: Standard sampling
- `temperature > 1`: More random, more diverse

### Top-p vs Top-k

- **Top-p (nucleus sampling)**: Dynamically selects from the smallest set of tokens whose cumulative probability exceeds the threshold. More adaptive than top-k.
- **Top-k**: Always samples from the k highest-probability tokens. Fixed number regardless of probability distribution.
- Both can be used together; top-k is applied first, then top-p filters within the top-k set.

---

## Penalty Parameters

| Argument | Type/Default | Description |
|---|---|---|
| **frequency_penalty** | `float = 0.0` | Penalizes tokens based on frequency in generation. Range [-2, 2]. Negative encourages repetition; positive encourages novelty. Penalty scales linearly with appearance count. |
| **presence_penalty** | `float = 0.0` | Penalizes tokens if they appeared at all. Range [-2, 2]. Negative encourages repetition; positive encourages novelty. Penalty is constant regardless of frequency. |
| **repetition_penalty** | `float = 1.0` | Scales logits of previously generated tokens. Range [0, 2]. >1 discourages repetition; <1 encourages it; 1.0 = unchanged. |
| **min_new_tokens** | `int = 0` | Forces at least min_new_tokens before stopping. Note: may cause unintended behavior with highly skewed distributions. |

### Penalty Comparison

| Penalty Type | When Applied | Scaling | Effect |
|---|---|---|---|
| frequency_penalty | Each occurrence adds penalty | Linear with count | Strong effect on frequent tokens |
| presence_penalty | Once if token appeared | Constant | Same penalty regardless of frequency |
| repetition_penalty | Multiplies logit | Multiplicative | Affects probability ratio |

---

## Constrained Decoding Parameters

These parameters control structured outputs via grammar-based constraints. Only one constraint parameter can be specified per request.

| Argument | Type/Default | Description |
|---|---|---|
| **json_schema** | `Optional[str] = None` | JSON schema for structured JSON output. |
| **regex** | `Optional[str] = None` | Regular expression for constrained output. |
| **ebnf** | `Optional[str] = None` | EBNF grammar for constrained output. |
| **structural_tag** | `Optional[str] = None` | Structural tag for function call constrained output. |

---

## Other Sampling Options

| Argument | Type/Default | Description |
|---|---|---|
| **n** | `int = 1` | Number of output sequences per request. Generating n > 1 in one request is discouraged; repeating the same prompt several times offers better control and efficiency. |
| **ignore_eos** | `bool = False` | Don't stop generation when EOS token is sampled. |
| **skip_special_tokens** | `bool = True` | Remove special tokens during decoding. |
| **spaces_between_special_tokens** | `bool = True` | Add spaces between special tokens during detokenization. |
| **no_stop_trim** | `bool = False` | Don't trim stop words or EOS token from generated text. |
| **custom_params** | `Optional[List[Optional[Dict[str, Any]]]] = None` | Custom parameters for CustomLogitProcessor. |

---

## Structured Outputs

SGLang supports constraining model output to follow specific formats through three grammar backends.

### JSON Schema

Constrain output to valid JSON matching a schema. Works with all three grammar backends.

```python
# Using Pydantic
from pydantic import BaseModel, Field
import json

class CapitalInfo(BaseModel):
    name: str = Field(..., pattern=r"^\w+$")
    population: int

# Via /generate endpoint
response = requests.post("http://localhost:30000/generate", json={
    "text": "Capital of France in JSON:",
    "sampling_params": {
        "temperature": 0,
        "max_new_tokens": 64,
        "json_schema": json.dumps(CapitalInfo.model_json_schema()),
    },
})

# Via OpenAI API
response = client.chat.completions.create(
    model="...",
    messages=[...],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "capital_info",
            "schema": CapitalInfo.model_json_schema(),
        },
    },
)
```

### Regular Expression

Constrain output to match a regex pattern. Works with Outlines and XGrammar backends.

```python
response = requests.post("http://localhost:30000/generate", json={
    "text": "Paris is the capital of",
    "sampling_params": {
        "temperature": 0,
        "max_new_tokens": 64,
        "regex": "(France|England)",
    },
})
```

### EBNF (Extended Backus-Naur Form)

Constrain output to match a context-free grammar. Works with XGrammar and Llguidance backends only.

```python
ebnf_grammar = """
root ::= city | description
city ::= "London" | "Paris" | "Berlin" | "Rome"
description ::= city " is " status
status ::= "the capital of " country
country ::= "England" | "France" | "Germany" | "Italy"
"""

response = requests.post("http://localhost:30000/generate", json={
    "text": "Give me the information of the capital of France.",
    "sampling_params": {
        "temperature": 0,
        "max_new_tokens": 128,
        "ebnf": ebnf_grammar,
    },
})
```

### Structural Tag

Constrain output to embed JSON within tagged sections, useful for function calling. Two formats are supported:

#### Format 1: Simple structures

```python
response = client.chat.completions.create(
    model="...",
    messages=[...],
    response_format={
        "type": "structural_tag",
        "structures": [
            {
                "begin": "<function=get_weather>",
                "schema": weather_schema,
                "end": "</function>",
            },
        ],
        "triggers": ["<function="],
    },
)
```

#### Format 2: XGrammar triggered_tags format

```python
response = client.chat.completions.create(
    model="...",
    messages=[...],
    response_format={
        "type": "structural_tag",
        "format": {
            "type": "triggered_tags",
            "triggers": ["<function="],
            "tags": [
                {
                    "begin": "<function=get_weather>",
                    "content": {"type": "json_schema", "json_schema": schema},
                    "end": "</function>",
                },
            ],
            "at_least_one": False,
            "stop_after_first": False,
        },
    },
)
```

---

## Grammar Backends

SGLang supports three grammar backends for structured outputs:

### XGrammar (Default)

- **Repository**: https://github.com/mlc-ai/xgrammar
- **Supports**: JSON schema, regex, EBNF constraints
- **Format**: GGML BNF format
- **Recommended for**: Best performance and utility

```bash
python -m sglang.launch_server --model-path <MODEL> --grammar-backend xgrammar
```

### Outlines

- **Repository**: https://github.com/dottxt-ai/outlines
- **Supports**: JSON schema, regex constraints
- **Does NOT support**: EBNF

```bash
python -m sglang.launch_server --model-path <MODEL> --grammar-backend outlines
```

### Llguidance

- **Repository**: https://github.com/guidance-ai/llguidance
- **Supports**: JSON schema, regex, EBNF constraints

```bash
python -m sglang.launch_server --model-path <MODEL> --grammar-backend llguidance
```

### Implementation Files

| File | Description |
|---|---|
| `python/sglang/srt/constrained/base_grammar_backend.py` | Base class for grammar backends |
| `python/sglang/srt/constrained/xgrammar_backend.py` | XGrammar implementation |
| `python/sglang/srt/constrained/outlines_backend.py` | Outlines implementation |
| `python/sglang/srt/constrained/llguidance_backend.py` | Llguidance implementation |
| `python/sglang/srt/constrained/grammar_manager.py` | Grammar manager for caching |
| `python/sglang/srt/constrained/reasoner_grammar_backend.py` | Grammar backend for reasoning models |

---

## Custom Logit Processors

Custom logit processors allow advanced sampling control by modifying logits before token selection.

### Enabling

Launch the server with `--enable-custom-logit-processor`:

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3-8B-Instruct \
  --port 30000 \
  --enable-custom-logit-processor
```

### Defining a Custom Processor

```python
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor

class DeterministicLogitProcessor(CustomLogitProcessor):
    """Always samples a specific token ID."""

    def __call__(self, logits, custom_param_list):
        assert logits.shape[0] == len(custom_param_list)
        for i, param_dict in enumerate(custom_param_list):
            logits[i, :] = -float("inf")
            logits[i, param_dict["token_id"]] = 0.0
        return logits
```

### Using with /generate

```python
response = requests.post("http://localhost:30000/generate", json={
    "text": "The capital of France is",
    "custom_logit_processor": DeterministicLogitProcessor().to_str(),
    "sampling_params": {
        "temperature": 0.0,
        "max_new_tokens": 32,
        "custom_params": {"token_id": 5},
    },
})
```

### Using with OpenAI API

```python
import openai

client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")
response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[{"role": "user", "content": "List 3 countries."}],
    temperature=0.0,
    max_tokens=32,
    extra_body={
        "custom_logit_processor": DeterministicLogitProcessor().to_str(),
        "custom_params": {"token_id": 5},
    },
)
```

### Implementation

Custom logit processors are defined in `python/sglang/srt/sampling/custom_logit_processor.py`. The `CustomLogitProcessor` base class provides serialization via `to_str()`.

---

## Reasoning/Thinking Model Support

SGLang supports reasoning models (e.g., DeepSeek R1, Qwen3, Kimi K2) that produce thinking/reasoning tokens before the final output. The Reasoning Parser separates reasoning content from the final answer.

### Supported Models

- DeepSeek R1 / V3 (deepseekv3 reasoning parser)
- Qwen3 (qwen3 reasoning parser)
- Kimi K2 (kimi_k2 reasoning parser)
- GPT-OSS

### Usage

When using reasoning models, the `reasoning_content` field in the response contains the model's thinking process, separate from the main `content` field.

---

## Tool Calling Parsers

SGLang provides tool calling (function calling) support via tool parsers that interpret model output for structured function invocations.

### Supported Parsers

| Parser | Supported Models | Notes |
|---|---|---|
| `deepseekv3` | DeepSeek-v3 (`deepseek-ai/DeepSeek-V3-0324`) | Use with `--chat-template ./examples/chat_template/tool_chat_template_deepseekv3.jinja` |
| `deepseekv31` | DeepSeek-V3.1, V3.2-Exp (`deepseek-ai/DeepSeek-V3.1`) | Use with corresponding jinja template |
| `deepseekv32` | DeepSeek-V3.2 (`deepseek-ai/DeepSeek-V3.2`) | |
| `glm` | GLM series (`zai-org/GLM-4.6`) | |
| `gpt-oss` | GPT-OSS (`openai/gpt-oss-120b`, `openai/gpt-oss-20b`) | Filters analysis channel events; use `no_stop_trim: True` |
| `kimi_k2` | `moonshotai/Kimi-K2-Instruct` | |
| `llama3` | Llama 3.1/3.2/3.3 | |
| `llama4` | Llama 4 (`meta-llama/Llama-4-Scout-17B-16E-Instruct`) | |
| `mistral` | Mistral (`mistralai/Mistral-7B-Instruct-v0.3`, Nemo) | |
| `pythonic` | Llama-3.2/3.3/4 | Model outputs function calls as Python code. Use with `--tool-call-parser pythonic` |
| `qwen` | Qwen series (except Qwen3-Coder) | |
| `qwen3_coder` | Qwen3-Coder (`Qwen/Qwen3-Coder-30B-A3B-Instruct`) | |
| `step3` | Step-3 | |

### Launch Example

```bash
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --tool-call-parser qwen25 \
  --host 0.0.0.0 --port 30000
```

### OpenAI-Compatible Tool Calling

```python
import openai

client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city", "unit"],
            },
        },
    }
]

# Non-streaming
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "What's the weather in Boston?"}],
    tools=tools,
    stream=False,
)

# Access tool calls
tool_call = response.choices[0].message.tool_calls[0]
print(f"Function: {tool_call.function.name}")
print(f"Arguments: {tool_call.function.arguments}")

# Execute tool and send results back
messages.append(response.choices[0].message)
messages.append({
    "role": "tool",
    "tool_call_id": tool_call.id,
    "content": "The weather in Boston, MA is 72F and sunny.",
    "name": "get_current_weather",
})
final_response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=messages,
    tools=tools,
)
```

### Tool Choice Mode

SGLang supports OpenAI's `tool_choice` parameter:

- `tool_choice="required"`: Forces at least one tool call
- `tool_choice={"type": "function", "function": {"name": "func_name"}}`: Forces a specific function

**Backend compatibility**: Fully supported with XGrammar (default). May not be fully supported with Outlines.

```python
response = client.chat.completions.create(
    model="...",
    messages=[...],
    tools=tools,
    tool_choice="required",  # Force tool call
)
```

### Pythonic Tool Call Format (Llama-3.2/3.3/4)

Some Llama models output function calls as Python code:

```python
[get_current_weather(city="San Francisco", state="CA", unit="celsius")]
```

Launch with `--tool-call-parser pythonic` and optionally a chat template:
```bash
python3 -m sglang.launch_server \
  --model-path meta-llama/Llama-3.2-1B-Instruct \
  --tool-call-parser pythonic \
  --chat-template examples/chat_template/tool_chat_template_llama4_pythonic.jinja
```

### Native API Tool Calling

```python
from transformers import AutoTokenizer
import requests

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
input_text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, tools=tools
)

# Generate
gen_response = requests.post("http://localhost:30000/generate", json={
    "text": input_text,
    "sampling_params": {"skip_special_tokens": False, "max_new_tokens": 1024},
}).json()["text"]

# Parse function call
parse_response = requests.post("http://localhost:30000/parse_function_call", json={
    "text": gen_response,
    "tool_call_parser": "qwen25",
    "tools": tools,
}).json()

print("Text:", parse_response["normal_text"])
print("Calls:", parse_response["calls"])
```

---

## Sampling Defaults Configuration

### Model Defaults vs OpenAI Defaults

When `--sampling-defaults model` (default), SGLang reads from `generation_config.json`:
- `temperature`: from model config
- `top_p`: from model config
- `top_k`: from model config
- `repetition_penalty`: from model config

When `--sampling-defaults openai`:
- `temperature`: 1.0
- `top_p`: 1.0
- `top_k`: -1 (disabled)
- `repetition_penalty`: 1.0

---

## Sampling Backend Architecture

### Implementation Files

| File | Description |
|---|---|
| `python/sglang/srt/sampling/sampling_params.py` | SamplingParams class definition |
| `python/sglang/srt/sampling/sampling_batch_info.py` | Batch-level sampling metadata |
| `python/sglang/srt/sampling/custom_logit_processor.py` | Custom logit processor base class |
| `python/sglang/srt/sampling/penaltylib/` | Frequency/presence/repetition penalty implementations |

### Sampling Flow

1. **Logits computation**: Model produces logits for the next token
2. **Logit processing**: Apply repetition/frequency/presence penalties, custom logit processors
3. **Grammar masking**: Apply constrained decoding masks (JSON, regex, EBNF)
4. **Sampling**: Select token based on temperature, top_p, top_k, min_p
5. **Stopping criteria**: Check for stop tokens, max_new_tokens, EOS

### Sampling Backends

SGLang supports multiple sampling computation backends (selected automatically or via flags):

- **flashinfer**: Uses FlashInfer sampling kernels
- **pytorch**: Pure PyTorch sampling implementation
- **ascend**: Ascend NPU sampling implementation
