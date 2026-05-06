# vLLM API Server Reference

This document provides comprehensive reference documentation for vLLM's API server,
including all endpoints, request/response formats, sampling parameters, pooling parameters,
logprobs handling, logits processors, chat template processing, gRPC server, SSL/TLS configuration,
server launch options, and command-line arguments.

---

## Table of Contents

1. [OpenAI-Compatible REST API Endpoints](#openai-compatible-rest-api-endpoints)
2. [Simple API Server Endpoints](#simple-api-server-endpoints)
3. [SamplingParams](#samplingparams)
4. [BeamSearchParams](#beamsearchparams)
5. [PoolingParams](#poolingparams)
6. [Logprobs Handling](#logprobs-handling)
7. [Logits Processors](#logits-processors)
8. [Chat Template Processing](#chat-template-processing)
9. [Multi-Modal Content Parsing](#multi-modal-content-parsing)
10. [gRPC Server](#grpc-server)
11. [SSL/TLS Configuration](#ssltls-configuration)
12. [Server Launch Options](#server-launch-options)
13. [Command-Line Arguments](#command-line-arguments)
14. [LLM Offline API](#llm-offline-api)
15. [Error Handling](#error-handling)
16. [Constants and Defaults](#constants-and-defaults)

---

## OpenAI-Compatible REST API Endpoints

The primary production API server is located at `vllm/entrypoints/openai/api_server.py`.
It provides an OpenAI-compatible REST API with the following endpoint groups:

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/completions` | POST | Text completions (generate text from a prompt) |
| `/v1/chat/completions` | POST | Chat completions (generate text from messages) |
| `/v1/embeddings` | POST | Generate embeddings for input text |
| `/v1/score` | POST | Score similarity between text pairs |
| `/v1/classify` | POST | Classification logits for input text |
| `/v1/tokenize` | POST | Tokenize input text (returns token IDs) |
| `/v1/detokenize` | POST | Detokenize token IDs back to text |
| `/v1/models` | GET | List available models |
| `/v1/audio/transcriptions` | POST | Speech-to-text transcription |
| `/health` | GET | Health check endpoint |
| `/tokenize` | POST | Alternative tokenization endpoint |
| `/detokenize` | POST | Alternative detokenization endpoint |
| `/version` | GET | Return vLLM version |

### Generation Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/generate` | POST | Direct generation with raw prompts |
| `/v1/render` | POST | Render chat templates to prompts |

### RLHF / Weight Transfer Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/init_weight_transfer` | POST | Initialize weight transfer for RL training |
| `/v1/update_weights` | POST | Update model weights during RL training |

### Disaggregated Serving Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/disagg_prefill` | POST | Submit prefill requests in disaggregated setup |
| `/v1/tokens_in_out` | POST | Token input/output for disaggregated setup |

### Elastic Expert Parallelism Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/elastic_ep/scale_up` | POST | Scale up expert parallelism |
| `/v1/elastic_ep/scale_down` | POST | Scale down expert parallelism |

### Request Format: Completions (`/v1/completions`)

```json
{
  "model": "meta-llama/Llama-2-7b-hf",
  "prompt": "Once upon a time",
  "max_tokens": 128,
  "temperature": 0.7,
  "top_p": 1.0,
  "n": 1,
  "stream": false,
  "logprobs": null,
  "stop": ["\n"],
  "suffix": null,
  "echo": false
}
```

**CompletionsRequest Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `str` | required | Model name or ID |
| `prompt` | `Union[str, List[str]]` | required | The prompt(s) to generate from |
| `best_of` | `int` | `None` | Number of best completions to return (deprecated) |
| `echo` | `bool` | `False` | Echo the prompt in the output |
| `frequency_penalty` | `float` | `0.0` | Penalize tokens by frequency (-2.0 to 2.0) |
| `logit_bias` | `Dict[int, float]` | `None` | Bias specific token IDs |
| `logprobs` | `int` | `None` | Number of logprobs per token |
| `max_tokens` | `int` | `16` | Maximum tokens to generate |
| `n` | `int` | `1` | Number of completions to return |
| `presence_penalty` | `float` | `0.0` | Penalize tokens by presence (-2.0 to 2.0) |
| `seed` | `int` | `None` | Random seed |
| `stop` | `Union[str, List[str]]` | `None` | Stop sequences |
| `stream` | `bool` | `False` | Stream results via SSE |
| `stream_options` | `dict` | `None` | Streaming options |
| `suffix` | `str` | `None` | Suffix to append |
| `temperature` | `float` | `1.0` | Sampling temperature |
| `top_p` | `float` | `1.0` | Top-p (nucleus) sampling |
| `user` | `str` | `None` | User identifier |

### Response Format: Completions

```json
{
  "id": "cmpl-abc123",
  "object": "text_completion",
  "created": 1234567890,
  "model": "meta-llama/Llama-2-7b-hf",
  "choices": [
    {
      "index": 0,
      "text": " there was a brave knight...",
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "total_tokens": 20,
    "completion_tokens": 15
  }
}
```

### Request Format: Chat Completions (`/v1/chat/completions`)

```json
{
  "model": "meta-llama/Llama-2-7b-hf",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "max_tokens": 128,
  "temperature": 0.7,
  "stream": false
}
```

**ChatCompletionRequest Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `str` | required | Model name or ID |
| `messages` | `List[ChatCompletionMessageParam]` | required | List of messages |
| `add_generation_prompt` | `bool` | `True` | Add generation prompt |
| `audio` | `dict` | `None` | Audio parameters |
| `chat_template` | `str` | `None` | Override chat template |
| `chat_template_kwargs` | `dict` | `None` | Chat template kwargs |
| `continue_final_message` | `bool` | `False` | Continue the final message |
| `frequency_penalty` | `float` | `0.0` | Frequency penalty |
| `guided_choice` | `List[str]` | `None` | Constrain to choices |
| `guided_decoding_backend` | `str` | `None` | Guided decoding backend |
| `guided_grammar` | `str` | `None` | Context-free grammar |
| `guided_json` | `Union[str, dict]` | `None` | JSON schema constraint |
| `guided_json_object` | `bool` | `None` | Output valid JSON |
| `guided_regex` | `str` | `None` | Regex constraint |
| `guided_whitespace_pattern` | `str` | `None` | Whitespace pattern |
| `include_stop_str_in_output` | `bool` | `False` | Include stop strings in output |
| `logit_bias` | `Dict[int, float]` | `None` | Logit bias |
| `logprobs` | `bool` | `False` | Return logprobs |
| `max_completion_tokens` | `int` | `None` | Max tokens for completion |
| `max_tokens` | `int` | `None` | Max tokens (deprecated) |
| `media_kwargs` | `dict` | `None` | Media processing kwargs |
| `messages` | `List` | required | Chat messages |
| `metadata` | `dict` | `None` | Request metadata |
| `min_tokens` | `int` | `0` | Minimum tokens |
| `model` | `str` | required | Model name |
| `n` | `int` | `1` | Number of completions |
| `presence_penalty` | `float` | `0.0` | Presence penalty |
| `reasoning_effort` | `str` | `None` | Reasoning effort level |
| `seed` | `int` | `None` | Random seed |
| `stop` | `Union[str, List[str]]` | `None` | Stop sequences |
| `stream` | `bool` | `False` | Stream results |
| `stream_options` | `dict` | `None` | Stream options |
| `streaming` | `bool` | `False` | Streaming alias |
| `temperature` | `float` | `None` | Sampling temperature |
| `tool_choice` | `Union[str, dict]` | `None` | Tool choice mode |
| `tools` | `List[dict]` | `None` | Available tools |
| `top_logprobs` | `int` | `None` | Top logprobs count |
| `top_p` | `float` | `1.0` | Top-p sampling |
| `user` | `str` | `None` | User identifier |

### Response Format: Chat Completions

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "meta-llama/Llama-2-7b-hf",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "total_tokens": 30,
    "completion_tokens": 10
  }
}
```

### Request Format: Embeddings (`/v1/embeddings`)

```json
{
  "model": "BAAI/bge-base-en-v1.5",
  "input": "Hello world"
}
```

**EmbeddingsRequest Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `str` | required | Model name |
| `input` | `Union[str, List[str], List[int], List[List[int]]]` | required | Input text or token IDs |
| `encoding_format` | `str` | `"float"` | Encoding format (float, base64) |
| `dimensions` | `int` | `None` | Output dimensions (for Matryoshka models) |
| `user` | `str` | `None` | User identifier |
| `truncate_prompt_tokens` | `int` | `None` | Truncate to this many tokens |

### Response Format: Embeddings

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.1, -0.2, 0.3, ...],
      "index": 0
    }
  ],
  "model": "BAAI/bge-base-en-v1.5",
  "usage": {
    "prompt_tokens": 2,
    "total_tokens": 2
  }
}
```

### Request Format: Scoring (`/v1/score`)

```json
{
  "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
  "query": "What is deep learning?",
  "documents": ["Deep learning is a subset of machine learning.", "The weather is nice today."]
}
```

### Realtime API

The realtime endpoint provides WebSocket support for real-time interactions:
- Path: `/v1/realtime`
- Protocol: WebSocket with JSON messages

---

## Simple API Server Endpoints

The simple API server at `vllm/entrypoints/api_server.py` provides a minimal API for
demonstration and benchmarking purposes.

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check, returns HTTP 200 |
| `/generate` | POST | Generate completion from a prompt |

### Generate Endpoint (`/generate`)

**Request Body:**
```json
{
  "prompt": "Once upon a time",
  "stream": false,
  "max_tokens": 128,
  "temperature": 0.7,
  "top_p": 1.0,
  "top_k": 0,
  "n": 1,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "repetition_penalty": 1.0,
  "stop": [],
  "stop_token_ids": [],
  "ignore_eos": false,
  "min_tokens": 0,
  "logprobs": null,
  "prompt_logprobs": null
}
```

All fields from `SamplingParams` are accepted directly in the request body alongside `prompt` and `stream`.

**Response (non-streaming):**
```json
{
  "text": ["Once upon a time there was a brave knight..."]
}
```

**Response (streaming):**
Each line is a JSON object:
```json
{"text": ["Once upon a time there"]}
{"text": ["Once upon a time there was a brave"]}
```

### Server Functions

#### `build_app(args: Namespace) -> FastAPI`

Builds the FastAPI application with the given arguments.

**Parameters:**
- `args` (`Namespace`): Command-line arguments including `root_path`

**Returns:** `FastAPI` application instance

#### `init_app(args: Namespace, llm_engine: AsyncLLMEngine | None = None) -> FastAPI`

Initialize the FastAPI application and engine.

**Parameters:**
- `args` (`Namespace`): Command-line arguments
- `llm_engine` (`AsyncLLMEngine | None`): Optional pre-created engine. If `None`, creates one from args.

**Returns:** Initialized `FastAPI` application

#### `run_server(args: Namespace, llm_engine: AsyncLLMEngine | None = None, **uvicorn_kwargs: Any) -> None`

Run the API server.

**Parameters:**
- `args` (`Namespace`): Command-line arguments
- `llm_engine` (`AsyncLLMEngine | None`): Optional pre-created engine
- `**uvicorn_kwargs`: Additional keyword arguments passed to uvicorn

---

## SamplingParams

The `SamplingParams` class (`vllm.sampling_params.py`) controls text generation behavior.
It follows the OpenAI text completion API conventions with additional vLLM-specific features.

### Class: `SamplingParams`

```python
class SamplingParams(PydanticMsgspecMixin, msgspec.Struct, omit_defaults=True, dict=True)
```

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `n` | `int` | `1` | Number of output sequences. Max controlled by `VLLM_MAX_N_SEQUENCES` env var (default 16384). |
| `presence_penalty` | `float` | `0.0` | Penalizes tokens based on presence in generated text. Range: [-2.0, 2.0]. Values > 0 encourage new tokens. |
| `frequency_penalty` | `float` | `0.0` | Penalizes tokens based on frequency in generated text. Range: [-2.0, 2.0]. Values > 0 encourage new tokens. |
| `repetition_penalty` | `float` | `1.0` | Penalizes tokens based on presence in prompt AND generated text. Values > 1 encourage new tokens. Must be > 0. |
| `temperature` | `float` | `1.0` | Controls sampling randomness. 0 = greedy. Lower = more deterministic, higher = more random. Auto-clamped to 0.01 minimum for numerical stability. |
| `top_p` | `float` | `1.0` | Cumulative probability of top tokens. Range: (0, 1]. Set to 1 to consider all tokens. |
| `top_k` | `int` | `0` | Number of top tokens to consider. 0 or -1 = consider all tokens. Must be >= -1. |
| `min_p` | `float` | `0.0` | Minimum probability for a token relative to most likely token. Range: [0, 1]. 0 = disabled. |
| `seed` | `int | None` | `None` | Random seed for reproducible generation. -1 is treated as None. |
| `stop` | `str | list[str] | None` | `None` | Stop strings that halt generation. Output will not contain stop strings (unless `include_stop_str_in_output=True`). |
| `stop_token_ids` | `list[int] | None` | `None` | Token IDs that halt generation. Output will contain stop tokens unless they are special tokens. |
| `ignore_eos` | `bool` | `False` | Whether to ignore the EOS token and continue generating. |
| `max_tokens` | `int | None` | `16` | Maximum tokens to generate per output sequence. Must be >= 1. |
| `min_tokens` | `int` | `0` | Minimum tokens before EOS or stop_token_ids can be generated. Must be >= 0 and <= max_tokens. |
| `logprobs` | `int | None` | `None` | Number of log probabilities per output token. -1 = all vocab_size. When set, up to logprobs+1 elements may be returned (includes sampled token). |
| `prompt_logprobs` | `int | None` | `None` | Number of log probabilities per prompt token. -1 = all vocab_size. |
| `logprob_token_ids` | `list[int] | None` | `None` | Specific token IDs to return logprobs for. Max length: 128. More efficient than logprobs=-1 for small token sets. |
| `flat_logprobs` | `bool` | `False` | Return logprobs in flat format (FlatLogprob) for better performance. Reduces GC overhead. |
| `detokenize` | `bool` | `True` | Whether to detokenize the output. Must be True when using stop strings. |
| `skip_special_tokens` | `bool` | `True` | Whether to skip special tokens in the output text. |
| `spaces_between_special_tokens` | `bool` | `True` | Whether to add spaces between special tokens in the output. |
| `include_stop_str_in_output` | `bool` | `False` | Whether to include stop strings in the output text. |
| `output_kind` | `RequestOutputKind` | `CUMULATIVE` | Output return mode: CUMULATIVE (return all so far), DELTA (return only changes), or FINAL_ONLY (no intermediate). |
| `skip_clone` | `bool` | `False` | Internal: skip deep copy in clone() for single-request use. |
| `structured_outputs` | `StructuredOutputsParams | None` | `None` | Parameters for structured output generation. |
| `logit_bias` | `dict[int, float] | None` | `None` | Bias values for specific token IDs. Values clamped to [-100, 100]. |
| `allowed_token_ids` | `list[int] | None` | `None` | If provided, only these token IDs will have non-zero scores. |
| `extra_args` | `dict[str, Any] | None` | `None` | Arbitrary additional args for custom sampling implementations. |
| `bad_words` | `list[str] | None` | `None` | Words that are not allowed to be generated. |
| `skip_reading_prefix_cache` | `bool | None` | `None` | Skip reading prefix cache. Auto-set when prompt_logprobs is not None. |
| `thinking_token_budget` | `int | None` | `None` | Maximum tokens allowed for thinking operations. |
| `repetition_detection` | `RepetitionDetectionParams | None` | `None` | Parameters for detecting repetitive N-gram patterns. |

### Methods

#### `from_optional(**kwargs) -> SamplingParams`

Static factory that accepts `None` values and substitutes defaults.

```python
@staticmethod
def from_optional(
    n: int | None = 1,
    presence_penalty: float | None = 0.0,
    frequency_penalty: float | None = 0.0,
    repetition_penalty: float | None = 1.0,
    temperature: float | None = 1.0,
    top_p: float | None = 1.0,
    top_k: int = 0,
    min_p: float = 0.0,
    seed: int | None = None,
    stop: str | list[str] | None = None,
    stop_token_ids: list[int] | None = None,
    bad_words: list[str] | None = None,
    thinking_token_budget: int | None = None,
    include_stop_str_in_output: bool = False,
    ignore_eos: bool = False,
    max_tokens: int | None = 16,
    min_tokens: int = 0,
    logprobs: int | None = None,
    prompt_logprobs: int | None = None,
    detokenize: bool = True,
    skip_special_tokens: bool = True,
    spaces_between_special_tokens: bool = True,
    output_kind: RequestOutputKind = RequestOutputKind.CUMULATIVE,
    structured_outputs: StructuredOutputsParams | None = None,
    logit_bias: dict[int, float] | dict[str, float] | None = None,
    allowed_token_ids: list[int] | None = None,
    extra_args: dict[str, Any] | None = None,
    skip_clone: bool = False,
    repetition_detection: RepetitionDetectionParams | None = None,
) -> SamplingParams
```

#### `update_from_generation_config(generation_config: dict, eos_token_id: int | None = None) -> None`

Updates sampling params from a HuggingFace generation config. Handles `eos_token_id`
which can be int or list of int.

#### `update_from_tokenizer(tokenizer: TokenizerLike) -> None`

Encodes `bad_words` into token IDs for use by the bad words logits processor.

#### `clone() -> SamplingParams`

Returns a deep copy of the SamplingParams. If `skip_clone=True`, returns a shallow copy instead.

#### `verify(model_config, speculative_config, structured_outputs_config, tokenizer) -> None`

Validates all parameters against model capabilities. Checks logprobs limits, logit bias,
allowed token IDs, speculative decoding compatibility, and structured outputs.

#### `for_sampler_warmup() -> SamplingParams`

Static factory that creates params exercising all sampler logic paths.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `sampling_type` | `SamplingType` | GREEDY, RANDOM, or RANDOM_SEED (cached property) |
| `eos_token_id` | `int | None` | The EOS token ID for this request |
| `all_stop_token_ids` | `set[int]` | Combined set of stop_token_ids + EOS |
| `bad_words_token_ids` | `list[list[int]] | None` | Token ID sequences for bad words |
| `num_logprobs` | `int | None` | Effective number of logprobs to return |

### Class: `StructuredOutputsParams`

```python
@dataclass
class StructuredOutputsParams
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `json` | `str | dict | None` | `None` | JSON schema for structured output |
| `regex` | `str | None` | `None` | Regex pattern for structured output |
| `choice` | `list[str] | None` | `None` | List of allowed choices |
| `grammar` | `str | None` | `None` | Context-free grammar |
| `json_object` | `bool | None` | `None` | Output valid JSON (no schema) |
| `disable_any_whitespace` | `bool` | `False` | Disable any whitespace in output |
| `disable_additional_properties` | `bool` | `False` | Disable additional properties |
| `whitespace_pattern` | `str | None` | `None` | Custom whitespace pattern |
| `structural_tag` | `str | None` | `None` | Structural tag for guided generation |

**Note:** Only one of `json`, `regex`, `choice`, `grammar`, `json_object`, or `structural_tag` can be set at a time.

### Class: `RepetitionDetectionParams`

```python
@dataclass
class RepetitionDetectionParams
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_pattern_size` | `int` | `0` | Maximum N-gram size to detect. 0 = disabled. |
| `min_pattern_size` | `int` | `0` | Minimum N-gram size. Must be <= max_pattern_size. |
| `min_count` | `int` | `0` | Minimum repetitions to trigger detection. Must be >= 2. |

### Enum: `SamplingType`

| Value | Description |
|-------|-------------|
| `GREEDY = 0` | Greedy decoding (temperature < epsilon) |
| `RANDOM = 1` | Random sampling |
| `RANDOM_SEED = 2` | Random sampling with fixed seed |

### Enum: `RequestOutputKind`

| Value | Description |
|-------|-------------|
| `CUMULATIVE = 0` | Return entire output so far in every RequestOutput |
| `DELTA = 1` | Return only deltas in each RequestOutput |
| `FINAL_ONLY = 2` | Do not return intermediate RequestOutput |

---

## BeamSearchParams

```python
class BeamSearchParams(msgspec.Struct, omit_defaults=True, dict=True)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `beam_width` | `int` | required | Number of beams |
| `max_tokens` | `int` | required | Maximum tokens per beam |
| `ignore_eos` | `bool` | `False` | Whether to ignore EOS token |
| `temperature` | `float` | `0.0` | Temperature for beam search sampling |
| `length_penalty` | `float` | `1.0` | Length penalty for beam scoring |
| `include_stop_str_in_output` | `bool` | `False` | Whether to include stop strings |

---

## PoolingParams

The `PoolingParams` class (`vllm/pooling_params.py`) controls pooling model behavior.

### Class: `PoolingParams`

```python
class PoolingParams(msgspec.Struct, omit_defaults=True, array_like=True)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `use_activation` | `bool | None` | `None` | Apply activation function to pooler outputs. Default depends on task. |
| `dimensions` | `int | None` | `None` | Reduce dimensions (for Matryoshka models). |
| `step_tag_id` | `int | None` | `None` | Step tag token ID for step pooling. |
| `returned_token_ids` | `list[int] | None` | `None` | Token IDs to return from step pooling. |
| `task` | `PoolingTask | None` | `None` | Internal: pooling task type. |
| `requires_token_ids` | `bool` | `False` | Internal: whether token IDs are required. |
| `skip_reading_prefix_cache` | `bool | None` | `None` | Internal: skip prefix cache reading. |
| `late_interaction_params` | `LateInteractionParams | None` | `None` | Internal: late interaction scoring params. |
| `extra_kwargs` | `dict[str, Any] | None` | `None` | Internal: extra keyword arguments. |
| `output_kind` | `RequestOutputKind` | `FINAL_ONLY` | Output return mode. Must be FINAL_ONLY for pooling. |

### Methods

#### `clone() -> PoolingParams`
Returns a deep copy.

#### `verify(model_config: ModelConfig) -> None`
Validates parameters against model configuration. Merges defaults from `PoolerConfig`.

#### `all_parameters -> list[str]`
Returns `["dimensions", "use_activation"]`.

#### `valid_parameters -> dict`
Returns valid parameters per task:
- `"embed"`: `["dimensions", "use_activation"]`
- `"classify"`: `["use_activation"]`
- `"token_embed"`: `["dimensions", "use_activation"]`
- `"token_classify"`: `["use_activation"]`

### Class: `LateInteractionParams`

```python
class LateInteractionParams(msgspec.Struct, omit_defaults=True, array_like=True)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | `str` | required | "cache_query" or "score_doc" |
| `query_key` | `str` | required | Stable key for DP routing and cache lookup |
| `query_uses` | `int | None` | `None` | Expected number of document requests |

---

## Logprobs Handling

### Class: `Logprob`

```python
@dataclass
class Logprob
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `logprob` | `float` | required | The log probability of the token |
| `rank` | `int | None` | `None` | The vocab rank of the token (>=1) |
| `decoded_token` | `str | None` | `None` | The decoded token string |

### Class: `FlatLogprobs`

```python
@dataclass
class FlatLogprobs(MutableSequence[LogprobsOnePosition | None])
```

A flat, GC-efficient representation of logprobs across all positions. Instead of creating
individual dict objects for each position, it stores parallel arrays of primitive types.

| Field | Type | Description |
|-------|------|-------------|
| `start_indices` | `list[int]` | Start index in flat arrays for each position |
| `end_indices` | `list[int]` | End index in flat arrays for each position |
| `token_ids` | `list[int]` | Flat array of token IDs |
| `logprobs` | `list[float]` | Flat array of log probabilities |
| `ranks` | `list[int | None]` | Flat array of token ranks |
| `decoded_tokens` | `list[str | None]` | Flat array of decoded token strings |

#### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `append` | `(logprobs_one_position: LogprobsOnePosition | None) -> None` | Append logprobs for the next position |
| `append_fast` | `(token_ids, logprobs, ranks, decoded_tokens) -> None` | Append without creating intermediate dict |
| `extend` | `(logprobs_multi_positions) -> None` | Extend with multiple positions |
| `__len__` | `() -> int` | Number of positions |
| `__getitem__` | `(index: int | slice) -> LogprobsOnePosition | FlatLogprobs` | Extract by position or slice |
| `__iter__` | `() -> Iterator[LogprobsOnePosition]` | Iterate over positions |

### Type Aliases

```python
LogprobsOnePosition = dict[int, Logprob]
PromptLogprobs = FlatLogprobs | list[LogprobsOnePosition | None]
SampleLogprobs = FlatLogprobs | list[LogprobsOnePosition]
```

### Helper Functions

#### `create_prompt_logprobs(flat_logprobs: bool) -> PromptLogprobs`
Creates a prompt logprobs container. First position is always None.

#### `create_sample_logprobs(flat_logprobs: bool) -> SampleLogprobs`
Creates a sample logprobs container.

#### `append_logprobs_for_next_position(request_logprobs, token_ids, logprobs, decoded_tokens, rank, num_logprobs) -> None`
Appends logprobs for the next position to either flat or regular container.

---

## Logits Processors

### Type: `LogitsProcessor`

```python
LogitsProcessor: TypeAlias = (
    Callable[[list[int], torch.Tensor], torch.Tensor]
    | Callable[[list[int], list[int], torch.Tensor], torch.Tensor]
)
```

A function that takes previously generated tokens (and optionally prompt tokens),
the logits tensor, and returns modified logits.

### Class: `NoBadWordsLogitsProcessor`

Prevents generation of specified bad words.

```python
class NoBadWordsLogitsProcessor:
    _SMALLEST_LOGIT = float("-inf")
    _NEUTRAL_LOGIT = 0.0
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(bad_words_ids: list[list[int]])` | Initialize with bad word token ID sequences |
| `__call__` | `(past_tokens_ids: Sequence[int], logits: torch.FloatTensor) -> torch.Tensor` | Apply bad word suppression |

### Function: `get_bad_words_logits_processors`

```python
def get_bad_words_logits_processors(
    bad_words: list[str],
    tokenizer: TokenizerLike
) -> list[LogitsProcessor]
```

Creates logits processors for bad word suppression. Handles both start-of-text
and mid-text word occurrences.

---

## Chat Template Processing

### Class: `ChatTemplateConfig`

```python
@dataclass
class ChatTemplateConfig
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `chat_template` | `str | None` | `None` | The chat template string or path |
| `chat_template_content_format` | `ChatTemplateContentFormatOption` | `"auto"` | How to render message content: "auto", "string", or "openai" |
| `trust_request_chat_template` | `bool` | `False` | Whether to trust request-provided chat templates |

### Types

```python
ChatTemplateContentFormatOption = Literal["auto", "string", "openai"]
ChatTemplateContentFormat = Literal["string", "openai"]
```

- `"string"`: Render content as plain string. Example: `"Who are you?"`
- `"openai"`: Render content as list of dicts. Example: `[{"type": "text", "text": "Who are you?"}]`

### Functions

#### `validate_chat_template(chat_template: Path | str | None)`

Validates the chat template. Raises:
- `FileNotFoundError` if path does not exist
- `ValueError` if string looks like a path but file does not exist
- `TypeError` for unsupported types

#### `load_chat_template(chat_template: Path | str | None, *, is_literal: bool = False) -> str | None`

Loads the chat template. Resolution order:
1. If `None`, returns `None`
2. If `is_literal=True`, returns the string directly
3. Tries to open as a file
4. Falls back to searching built-in templates directory

### Content Part Types

| Type | Description |
|------|-------------|
| `AudioURL` | Audio URL with optional base64 data |
| `ChatCompletionContentPartAudioParam` | Audio content part |
| `ChatCompletionContentPartImageEmbedsParam` | Image embedding content part |
| `ChatCompletionContentPartAudioEmbedsParam` | Audio embedding content part |
| `ChatCompletionContentPartPromptEmbedsParam` | Prompt embedding content part |
| `VideoURL` | Video URL with optional base64 data |
| `ChatCompletionContentPartVideoParam` | Video content part |
| `PILImage` | PIL Image wrapper |
| `CustomThinkCompletionContentParam` | Thinking content with closed flag |
| `CustomChatCompletionContentToolReferenceParam` | Tool reference content |

### Modality Placeholder Map

```python
MODALITY_PLACEHOLDERS_MAP = {
    "image": "<##IMAGE##>",
    "audio": "<##AUDIO##>",
    "video": "<##VIDEO##>",
    "prompt_embeds": "<##PROMPT_EMBEDS##>",
}
```

---

## Multi-Modal Content Parsing

### Class: `BaseMultiModalItemTracker`

Abstract base class for tracking multi-modal items in requests.

```python
class BaseMultiModalItemTracker(ABC, Generic[_T])
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(model_config: ModelConfig, media_io_kwargs: dict | None)` | Initialize with model config |
| `use_unified_vision_chunk_modality` | `-> bool` (cached_property) | Check if model uses unified vision_chunk |
| `model_config` | `-> ModelConfig` (property) | Get model config |
| `model_cls` | `-> type[SupportsMultiModal]` (cached_property) | Get model class |
| `media_io_kwargs` | `-> dict | None` (property) | Get media IO kwargs |
| `allowed_local_media_path` | `-> str` (property) | Get allowed local media path |
| `allowed_media_domains` | `-> list | None` (property) | Get allowed media domains |
| `mm_registry` | `->` registry (property) | Get multi-modal registry |
| `mm_processor` | (cached_property) | Get/create multi-modal processor |
| `add` | `(modality: ModalityStr, item: _T) -> str | None` | Add a multi-modal item |
| `create_parser` | `(mm_processor_kwargs) -> BaseMultiModalContentParser` | Create content parser (abstract) |

### Class: `MultiModalItemTracker`

Sync version of the tracker.

```python
class MultiModalItemTracker(BaseMultiModalItemTracker[tuple[object, str | None]])
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `resolve_items` | `() -> (MultiModalDataDict | None, MultiModalUUIDDict | None)` | Resolve tracked items to mm_data and mm_uuids |

### Class: `AsyncMultiModalItemTracker`

Async version of the tracker.

```python
class AsyncMultiModalItemTracker(BaseMultiModalItemTracker[Awaitable[tuple[object, str | None]]])
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `resolve_items` | `async () -> (MultiModalDataDict | None, MultiModalUUIDDict | None)` | Async resolve tracked items |

### Class: `BaseMultiModalContentParser`

Abstract base class for parsing multi-modal content.

| Method | Signature | Description |
|--------|-----------|-------------|
| `parse_image` | `(image_url: str | None, uuid: str | None) -> None` | Parse image URL |
| `parse_image_embeds` | `(image_embeds, uuid) -> None` | Parse image embeddings |
| `parse_image_pil` | `(image_pil: Image | None, uuid) -> None` | Parse PIL image |
| `parse_audio` | `(audio_url: str | None, uuid) -> None` | Parse audio URL |
| `parse_input_audio` | `(input_audio: InputAudio | None, uuid) -> None` | Parse input audio |
| `parse_audio_embeds` | `(audio_embeds, uuid) -> None` | Parse audio embeddings |
| `parse_prompt_embeds` | `(data: str) -> None` | Parse prompt embeddings |
| `parse_video` | `(video_url: str | None, uuid) -> None` | Parse video URL |
| `mm_placeholder_storage` | `() -> dict[str, list]` | Get placeholder storage |

### Message Parsing Functions

#### `parse_chat_messages(messages, model_config, content_format, media_io_kwargs, mm_processor_kwargs) -> tuple`

Parses chat messages into conversation messages, multi-modal data, and UUIDs.

```python
def parse_chat_messages(
    messages: list[ChatCompletionMessageParam],
    model_config: ModelConfig,
    content_format: ChatTemplateContentFormat,
    media_io_kwargs: dict[str, dict[str, Any]] | None = None,
    mm_processor_kwargs: dict[str, Any] | None = None,
) -> tuple[list[ConversationMessage], MultiModalDataDict | None, MultiModalUUIDDict | None]
```

#### `parse_chat_messages_async(...) -> tuple`

Async version of `parse_chat_messages`.

### Helper Functions

#### `get_history_tool_calls_cnt(conversation: list[ConversationMessage]) -> int`

Returns count of tool calls in assistant messages.

#### `get_tool_call_id_type(model_config: ModelConfig) -> str`

Returns "kimi_k2" for Kimi models, "random" otherwise.

#### `make_tool_call_id(id_type: str, func_name=None, idx=None) -> str`

Generates a tool call ID based on type.

---

## gRPC Server

The gRPC server is located at `vllm/entrypoints/grpc_server.py` and provides an alternative
to the HTTP REST API.

### Command-Line Usage

```bash
python -m vllm.entrypoints.grpc_server \
    --model meta-llama/Llama-2-7b-hf \
    --host 0.0.0.0 \
    --port 50051
```

### gRPC Server Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--host` | `str` | `"0.0.0.0"` | Host to bind to |
| `--port` | `int` | `50051` | Port to bind to |

Plus all `AsyncEngineArgs` arguments.

### gRPC Configuration

The gRPC server sets the following channel options:

```python
options = [
    ("grpc.max_send_message_length", -1),      # Unlimited
    ("grpc.max_receive_message_length", -1),    # Unlimited
    ("grpc.http2.min_recv_ping_interval_without_data_ms", 10000),
    ("grpc.keepalive_permit_without_calls", True),
]
```

### Services

- **VllmEngine**: Main inference service
- **Health**: Standard gRPC health service for Kubernetes probes
- **Reflection**: gRPC reflection for grpcurl and other tools

### Function: `serve_grpc(args: argparse.Namespace)`

Main gRPC serving coroutine. Creates AsyncLLM, sets up servicer, health checks,
reflection, and signal handlers.

---

## SSL/TLS Configuration

### Class: `SSLCertRefresher`

Monitors SSL certificate files and reloads them on change.

```python
class SSLCertRefresher
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(ssl_context: SSLContext, key_path=None, cert_path=None, ca_path=None)` | Initialize with SSL context and file paths |
| `stop` | `() -> None` | Stop watching files |

The refresher uses `watchfiles.awatch` to asynchronously monitor certificate file changes.
When changes are detected:
- Certificate chain: calls `ssl.load_cert_chain(cert_path, key_path)`
- CA certificates: calls `ssl.load_verify_locations(ca_path)`

---

## Server Launch Options

### Function: `serve_http`

```python
async def serve_http(
    app: FastAPI,
    sock: socket.socket | None,
    enable_ssl_refresh: bool = False,
    **uvicorn_kwargs: Any,
) -> asyncio.Task
```

Starts a FastAPI app using Uvicorn. Sets up:
- H11 header limit defaults (4MB incomplete event size, 256 max headers)
- SSL certificate refresher
- Watchdog loop (5-second interval)
- Signal handlers (SIGINT, SIGTERM)
- Graceful shutdown with configurable timeout

### Function: `watchdog_loop`

```python
async def watchdog_loop(server: uvicorn.Server, engine: EngineClient)
```

Background task checking engine health every 5 seconds. Triggers shutdown if engine
is errored and not running (unless `VLLM_KEEP_ALIVE_ON_ENGINE_DEATH` is set).

### Function: `terminate_if_errored`

```python
def terminate_if_errored(server: uvicorn.Server, engine: EngineClient)
```

Checks engine error state and sets `server.should_exit = True` if engine has errored.

---

## Command-Line Arguments

### OpenAI Server CLI Arguments

#### Server Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--host` | `str` | `None` | Host address |
| `--port` | `int` | `8000` | Port number |
| `--uds` | `str` | `None` | Unix domain socket path |
| `--root-path` | `str` | `None` | FastAPI root_path for proxy |
| `--log-level` | `str` | `"debug"` | Log level |
| `--uvicorn-log-level` | `str` | `"info"` | Uvicorn log level |

#### SSL Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--ssl-keyfile` | `str` | `None` | SSL key file path |
| `--ssl-certfile` | `str` | `None` | SSL certificate file path |
| `--ssl-ca-certs` | `str` | `None` | CA certificates file |
| `--ssl-cert-reqs` | `int` | `ssl.CERT_NONE` | Client certificate requirement |
| `--enable-ssl-refresh` | `bool` | `False` | Enable SSL certificate refresh |
| `--ssl-ciphers` | `str` | `None` | SSL cipher suites |

#### CORS Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--allowed-origins` | `list[str]` | `["*"]` | Allowed CORS origins |
| `--allowed-methods` | `list[str]` | `["*"]` | Allowed CORS methods |
| `--allowed-headers` | `list[str]` | `["*"]` | Allowed CORS headers |
| `--allow-credentials` | `bool` | `False` | Allow CORS credentials |

#### API Key Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--api-key` | `str | list[str]` | `None` | API key(s). Falls back to `VLLM_API_KEY` env var. |

#### Chat Template Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--chat-template` | `str` | `None` | Chat template path or string |
| `--chat-template-content-format` | `str` | `"auto"` | Content format: "string" or "openai" |
| `--trust-request-chat-template` | `bool` | `False` | Trust request chat templates |
| `--default-chat-template-kwargs` | `dict` | `None` | Default kwargs for chat template |

#### Tool Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--enable-auto-tool-choice` | `bool` | `False` | Enable auto tool choice |
| `--tool-call-parser` | `str` | `None` | Tool call parser name |
| `--tool-parser-plugin` | `str` | `""` | Tool parser plugin path |
| `--exclude-tools-when-tool-choice-none` | `bool` | `False` | Exclude tools when tool_choice=none |

#### LoRA Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--lora-modules` | `list` | `None` | LoRA module configs (name=path or JSON) |

#### Logging Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--max-log-len` | `int` | `None` | Max prompt chars/IDs to log |
| `--enable-log-requests` | `bool` | `False` | Enable request logging |
| `--enable-log-outputs` | `bool` | `False` | Enable output logging |
| `--log-config-file` | `str` | `None` | Logging config JSON file |

#### HTTP Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--h11-max-incomplete-event-size` | `int` | `4194304` (4MB) | Max incomplete H11 event size |
| `--h11-max-header-count` | `int` | `256` | Max H11 header count |
| `--disable-uvicorn-access-log` | `bool` | `False` | Disable uvicorn access log |

#### Response Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--return-tokens-as-token-ids` | `bool` | `False` | Represent tokens as "token_id:{id}" strings |
| `--enable-prompt-tokens-details` | `bool` | `False` | Enable prompt_tokens_details in usage |
| `--fingerprint-mode` | `str` | `"full"` | Fingerprint mode: full, hash, custom, none |
| `--fingerprint-value` | `str` | `None` | Custom fingerprint value |

---

## LLM Offline API

The `LLM` class (`vllm/entrypoints/llm.py`) provides an offline inference API.

### Class: `LLM`

```python
class LLM
```

#### Constructor

```python
def __init__(
    self,
    model: str,
    *,
    runner: RunnerOption = "auto",
    convert: ConvertOption = "auto",
    tokenizer: str | None = None,
    tokenizer_mode: TokenizerMode | str = "auto",
    skip_tokenizer_init: bool = False,
    trust_remote_code: bool = False,
    allowed_local_media_path: str = "",
    allowed_media_domains: list[str] | None = None,
    tensor_parallel_size: int = 1,
    dtype: ModelDType = "auto",
    quantization: QuantizationMethods | None = None,
    revision: str | None = None,
    tokenizer_revision: str | None = None,
    chat_template: Path | str | None = None,
    seed: int = 0,
    gpu_memory_utilization: float = 0.92,
    cpu_offload_gb: float = 0,
    offload_group_size: int = 0,
    offload_num_in_group: int = 1,
    offload_prefetch_step: int = 1,
    offload_params: set[str] | None = None,
    enforce_eager: bool = False,
    enable_return_routed_experts: bool = False,
    disable_custom_all_reduce: bool = False,
    hf_token: bool | str | None = None,
    hf_overrides: HfOverrides | None = None,
    mm_processor_kwargs: dict[str, Any] | None = None,
    pooler_config: PoolerConfig | None = None,
    structured_outputs_config: dict | StructuredOutputsConfig | None = None,
    profiler_config: dict | ProfilerConfig | None = None,
    attention_config: dict | AttentionConfig | None = None,
    kv_cache_memory_bytes: int | None = None,
    compilation_config: int | dict | CompilationConfig | None = None,
    quantization_config: dict | OnlineQuantizationConfigArgs | None = None,
    logits_processors: list[str | type[LogitsProcessor]] | None = None,
    **kwargs: Any,
) -> None
```

#### Methods

##### `generate(prompts, sampling_params, *, use_tqdm, lora_request, priority, tokenization_kwargs, mm_processor_kwargs) -> list[RequestOutput]`

Generate completions for prompts.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompts` | `PromptType | Sequence[PromptType]` | required | Input prompts |
| `sampling_params` | `SamplingParams | Sequence[SamplingParams] | None` | `None` | Sampling parameters |
| `use_tqdm` | `bool | Callable` | `True` | Progress bar |
| `lora_request` | `LoRARequest | Sequence[LoRARequest] | None` | `None` | LoRA request |
| `priority` | `list[int] | None` | `None` | Request priorities |
| `tokenization_kwargs` | `dict | None` | `None` | Tokenizer overrides |
| `mm_processor_kwargs` | `dict | None` | `None` | Processor overrides |

##### `chat(messages, sampling_params, *, use_tqdm, lora_request, chat_template, ...) -> list[RequestOutput]`

Generate responses for chat conversations.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `messages` | `list[ChatCompletionMessageParam] | Sequence[list[...]]` | required | Chat messages |
| `sampling_params` | `SamplingParams | Sequence | None` | `None` | Sampling parameters |
| `chat_template` | `str | None` | `None` | Override chat template |
| `chat_template_content_format` | `str` | `"auto"` | Content format |
| `add_generation_prompt` | `bool` | `True` | Add generation prompt |
| `continue_final_message` | `bool` | `False` | Continue final message |
| `tools` | `list[dict] | None` | `None` | Available tools |
| `chat_template_kwargs` | `dict | None` | `None` | Template kwargs |

##### `encode(prompts, pooling_params, *, use_tqdm, lora_request, pooling_task, tokenization_kwargs) -> list[PoolingRequestOutput]`

Apply pooling to hidden states.

##### `embed(prompts, *, use_tqdm, pooling_params, lora_request, tokenization_kwargs) -> list[EmbeddingRequestOutput]`

Generate embeddings.

##### `classify(prompts, *, pooling_params, use_tqdm, lora_request, tokenization_kwargs) -> list[ClassificationRequestOutput]`

Generate classification logits.

##### `score(data_1, data_2, /, *, use_tqdm, pooling_params, ...) -> list[ScoringRequestOutput]`

Generate similarity scores.

##### `beam_search(prompts, params, lora_request, use_tqdm, concurrency_limit) -> list[BeamSearchOutput]`

Generate sequences using beam search.

##### `enqueue(prompts, sampling_params, ...) -> list[str]`

Enqueue prompts without waiting for completion.

##### `wait_for_completion(output_type, *, use_tqdm) -> list`

Wait for all enqueued requests.

##### `sleep(level, mode)` / `wake_up(tags)`

Put engine to sleep / wake it up. Sleep levels:
- 0: Pause scheduling
- 1: Offload weights to CPU, discard KV cache
- 2: Discard all GPU memory

##### `start_profile(profile_prefix)` / `stop_profile()`

Start/stop profiling.

##### `reset_prefix_cache(reset_running_requests, reset_connector) -> bool`

Reset prefix cache.

##### `collective_rpc(method, timeout, args, kwargs) -> list`

Execute RPC on all workers.

##### `apply_model(func) -> list`

Run a function on the model in each worker.

##### `get_tokenizer() -> TokenizerLike`

Get the tokenizer.

##### `get_world_size(include_dp) -> int`

Get world size (TP * PP * DP optionally).

##### `get_metrics() -> list[Metric]`

Get Prometheus metrics snapshot.

##### `init_weight_transfer_engine(request) -> None`

Initialize weight transfer for RL training.

##### `update_weights(request) -> None`

Update model weights during RL training.

---

## Error Handling

### Function: `create_error_response`

```python
def create_error_response(
    message: str | Exception,
    err_type: str = "BadRequestError",
    status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
    param: str | None = None,
) -> ErrorResponse
```

Creates standardized error responses. Automatically maps exception types to error types:

| Exception | Error Type | Status Code |
|-----------|-----------|-------------|
| `VLLMValidationError` | `BadRequestError` | 400 |
| `VLLMNotFoundError` | `NotFoundError` | 404 |
| `ValueError`, `TypeError`, `OverflowError` | `BadRequestError` | 400 |
| `NotImplementedError` | `NotImplementedError` | 501 |
| `GenerationError` | `InternalServerError` | Varies |
| `TemplateError` (jinja2) | `BadRequestError` | 400 |
| Other exceptions | `InternalServerError` | 500 |

### ErrorResponse Format

```python
@dataclass
class ErrorResponse:
    error: ErrorInfo

@dataclass
class ErrorInfo:
    message: str
    type: str
    code: int
    param: str | None
```

---

## Constants and Defaults

### HTTP Constants (`vllm/entrypoints/constants.py`)

| Constant | Value | Description |
|----------|-------|-------------|
| `H11_MAX_INCOMPLETE_EVENT_SIZE_DEFAULT` | `4194304` (4 MB) | Default max incomplete H11 event size |
| `H11_MAX_HEADER_COUNT_DEFAULT` | `256` | Default max header count |
| `MCP_PREFIX` | `"mcp_"` | MCP prefix for tool names |

### Sampling Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `_SAMPLING_EPS` | `1e-5` | Epsilon for greedy detection |
| `_MAX_TEMP` | `1e-2` | Minimum temperature before clamping |
| `MAX_LOGPROB_TOKEN_IDS` | `128` | Max length of logprob_token_ids list |

### Request Logger

The `RequestLogger` class (`vllm/entrypoints/logger.py`) logs request details.

```python
class RequestLogger:
    def __init__(self, *, max_log_len: int | None)
    def log_inputs(self, request_id, prompt, prompt_token_ids, prompt_embeds, params, lora_request)
    def log_outputs(self, request_id, outputs, output_token_ids, finish_reason, is_streaming, delta)
```

### Helper Functions

#### `listen_for_disconnect(request: Request) -> None`

Monitors request for HTTP disconnect messages.

#### `with_cancellation(handler_func)`

Decorator allowing route handler cancellation on client disconnect.

#### `load_aware_call(func)`

Decorator that tracks server load metrics.

#### `cli_env_setup()`

Sets `VLLM_WORKER_MULTIPROC_METHOD` to "spawn" for safety.

#### `get_max_tokens(max_model_len, max_tokens, input_length, default_sampling_params, override_max_tokens) -> int`

Calculates effective max tokens respecting model limits, platform limits, and user settings.

#### `log_non_default_args(args)`

Logs non-default arguments at startup.

#### `should_include_usage(stream_options, enable_force_include_usage) -> tuple[bool, bool]`

Determines whether to include usage stats in streaming responses.

#### `sanitize_message(message: str) -> str`

Removes memory addresses from error messages.

#### `log_version_and_model(logger, version, model_name)`

Logs vLLM version and model with ASCII art logo.
