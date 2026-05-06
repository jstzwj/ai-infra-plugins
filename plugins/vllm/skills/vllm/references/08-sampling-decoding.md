# vLLM V1 Sampling and Decoding Reference

This document provides a comprehensive reference for the token sampling, decoding,
and output processing pipeline in vLLM's V1 architecture. It covers sampling parameters,
all sampling strategies, logits processors, logprobs calculation, beam search,
structured output, speculative decoding, detokenization, and stop conditions.

---

## Table of Contents

1. [SamplingParams Full Reference](#1-samplingparams-full-reference)
2. [Sampling Pipeline](#2-sampling-pipeline)
3. [Sampling Strategies](#3-sampling-strategies)
4. [Sampling Metadata](#4-sampling-metadata)
5. [Sampler Module](#5-sampler-module)
6. [Top-K Top-P Sampler](#6-top-k-top-p-sampler)
7. [Logits Processors](#7-logits-processors)
8. [Penalties](#8-penalties)
9. [Logprobs Calculation](#9-logprobs-calculation)
10. [Beam Search](#10-beam-search)
11. [Structured Output](#11-structured-output)
12. [Reasoning / Thinking Output](#12-reasoning--thinking-output)
13. [Speculative Decoding](#13-speculative-decoding)
14. [Output Processing](#14-output-processing)
15. [Detokenization Pipeline](#15-detokenization-pipeline)
16. [Stop Conditions](#16-stop-conditions)
17. [Rejection Sampler](#17-rejection-sampler)
18. [Bad Words Filtering](#18-bad-words-filtering)
19. [Parallel Sampling](#19-parallel-sampling)

---

## 1. SamplingParams Full Reference

**File:** `vllm/sampling_params.py`

### SamplingType

```python
class SamplingType(IntEnum):
    GREEDY = 0
    RANDOM = 1
    RANDOM_SEED = 2
```

### RequestOutputKind

```python
class RequestOutputKind(Enum):
    CUMULATIVE = 0   # Return entire output so far in every RequestOutput
    DELTA = 1         # Return only deltas in each RequestOutput
    FINAL_ONLY = 2    # Do not return intermediate RequestOutput
```

### StructuredOutputsParams

```python
@dataclass
class StructuredOutputsParams:
    json: str | dict | None = None
    regex: str | None = None
    choice: list[str] | None = None
    grammar: str | None = None
    json_object: bool | None = None
    disable_any_whitespace: bool = False
    disable_additional_properties: bool = False
    whitespace_pattern: str | None = None
    structural_tag: str | None = None
    _backend: str | None = None
    _backend_was_auto: bool = False
```

### RepetitionDetectionParams

```python
@dataclass
class RepetitionDetectionParams:
    max_pattern_size: int = 0    # Max N-gram pattern size (0 = disabled)
    min_pattern_size: int = 0    # Min N-gram pattern size
    min_count: int = 0           # Min repetition count (>= 2)
```

### SamplingParams

```python
class SamplingParams(PydanticMsgspecMixin, msgspec.Struct):
```

#### Core Sampling Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n` | `int` | `1` | Number of output sequences. Max controlled by `VLLM_MAX_N_SEQUENCES` (default 16384) |
| `temperature` | `float` | `1.0` | Controls randomness. 0 = greedy, higher = more random |
| `top_p` | `float` | `1.0` | Cumulative probability threshold for nucleus sampling. Range: (0, 1] |
| `top_k` | `int` | `0` | Number of top tokens to consider. 0 or -1 = all tokens |
| `min_p` | `float` | `0.0` | Minimum probability relative to most likely token. Range: [0, 1] |
| `seed` | `int \| None` | `None` | Random seed for reproducibility |

#### Penalty Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `presence_penalty` | `float` | `0.0` | Penalizes tokens that appear in generated text. Range: [-2, 2] |
| `frequency_penalty` | `float` | `0.0` | Penalizes tokens based on frequency in generated text. Range: [-2, 2] |
| `repetition_penalty` | `float` | `1.0` | Penalizes tokens from prompt + generated text. >1 encourages new tokens |

#### Stop/EOS Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `stop` | `str \| list[str] \| None` | `None` | Strings that stop generation |
| `stop_token_ids` | `list[int] \| None` | `None` | Token IDs that stop generation |
| `ignore_eos` | `bool` | `False` | Continue past EOS token |
| `include_stop_str_in_output` | `bool` | `False` | Include stop strings in output |

#### Length Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_tokens` | `int \| None` | `16` | Maximum tokens to generate |
| `min_tokens` | `int` | `0` | Minimum tokens before EOS/stop allowed |

#### Logprobs Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `logprobs` | `int \| None` | `None` | Number of top logprobs per token. -1 = all |
| `prompt_logprobs` | `int \| None` | `None` | Number of prompt logprobs per token. -1 = all |
| `logprob_token_ids` | `list[int] \| None` | `None` | Specific token IDs to get logprobs for |
| `flat_logprobs` | `bool` | `False` | Return flat logprobs format for performance |

#### Output Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `detokenize` | `bool` | `True` | Whether to detokenize output |
| `skip_special_tokens` | `bool` | `True` | Skip special tokens in output |
| `spaces_between_special_tokens` | `bool` | `True` | Add spaces between special tokens |
| `output_kind` | `RequestOutputKind` | `CUMULATIVE` | Output return mode |

#### Structured Output Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `structured_outputs` | `StructuredOutputsParams \| None` | `None` | Structured output configuration |
| `logit_bias` | `dict[int, float] \| None` | `None` | Token ID to bias mapping. Range: [-100, 100] |
| `allowed_token_ids` | `list[int] \| None` | `None` | Whitelist of allowed token IDs |

#### Advanced Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bad_words` | `list[str] \| None` | `None` | Words that cannot be generated |
| `thinking_token_budget` | `int \| None` | `None` | Max thinking tokens budget |
| `repetition_detection` | `RepetitionitionDetectionParams \| None` | `None` | N-gram repetition detection |
| `extra_args` | `dict[str, Any] \| None` | `None` | Custom args for plugins |
| `skip_clone` | `bool` | `False` | Skip cloning when safe to reuse |
| `skip_reading_prefix_cache` | `bool \| None` | `None` | Skip prefix cache reading |

#### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `from_optional` | `(n=None, presence_penalty=None, ..., repetition_detection=None) -> SamplingParams` | Factory with all optional params |
| `clone` | `() -> SamplingParams` | Deep copy of params |
| `update` | `(**kwargs) -> SamplingParams` | Create copy with updates |

---

## 2. Sampling Pipeline

The sampling pipeline processes model output logits through multiple stages:

```
Model Output (logits)
    |
    v
[1] Compute raw logprobs (if requested)
    |
    v
[2] Convert logits to float32
    |
    v
[3] Apply allowed_token_ids whitelist
    |
    v
[4] Apply bad_words exclusion
    |
    v
[5] Apply non-argmax-invariant processors
    |   - Min tokens processor
    |   - Logit bias processor
    v
[6] Apply penalties
    |   - Repetition penalty
    |   - Frequency penalty
    |   - Presence penalty
    v
[7] Sample next tokens
    |   a) Greedy sample (if applicable)
    |   b) Apply temperature
    |   c) Apply argmax-invariant processors (min_p)
    |   d) Apply top_k and/or top_p
    |   e) Random sample
    |   f) Select greedy or random based on temperature
    v
[8] Gather logprobs for top-k and sampled token
    |
    v
[9] Return SamplerOutput
```

---

## 3. Sampling Strategies

### Greedy Sampling

- Triggered when `temperature < 1e-5`
- Uses `torch.argmax()` on logits
- Deterministic and reproducible
- `top_p`, `top_k`, `min_p` are all ignored

### Random Sampling

- Triggered when `temperature >= 1e-5`
- Logits are divided by temperature
- Optional `min_p` filtering applied
- `top_k` and/or `top_p` filtering applied
- Final sampling from filtered distribution

### Top-K Sampling

- `top_k > 0` retains only the top K highest probability tokens
- Implemented in `TopKTopPSampler`
- Can use FlashInfer sampler, Triton kernel, or PyTorch native
- Selected via `VLLM_USE_FLASHINFER_SAMPLER` env var

### Top-P (Nucleus) Sampling

- `top_p < 1.0` retains smallest set of tokens whose cumulative probability >= top_p
- Applied after top_k filtering
- Efficient implementation using sorted probabilities

### Min-P Sampling

- `min_p > 0` sets minimum probability threshold relative to most likely token
- Tokens with `prob < min_p * max_prob` are filtered out
- Applied as an argmax-invariant logits processor
- Default processor is `MinPLogitsProcessor`

### Temperature Sampling

- Logits divided by temperature before softmax
- `temperature = 0` -> greedy (argmax)
- `temperature = 1` -> standard sampling
- `temperature > 1` -> more random/flatter distribution
- Values below `_MAX_TEMP` (0.01) are clamped to avoid numerical issues

### Seed-Based Sampling

- `SamplingType.RANDOM_SEED` when `seed` is specified
- Uses per-request `torch.Generator` with the given seed
- Ensures reproducible outputs

### Beam Search

- `n > 1` with `use_beam_search=True`
- Explores multiple candidate sequences
- Selects top sequences by cumulative log probability
- See [Beam Search](#10-beam-search) section for details

---

## 4. Sampling Metadata

**File:** `vllm/v1/sample/metadata.py`

### SamplingMetadata

```python
@dataclass
class SamplingMetadata:
    temperature: torch.Tensor | None
    all_greedy: bool
    all_random: bool

    top_p: torch.Tensor | None
    top_k: torch.Tensor | None

    generators: dict[int, torch.Generator]

    max_num_logprobs: int | None
    """None = no logprobs, 0 = sampled only, >0 = top-k logprobs, -1 = all"""

    no_penalties: bool
    prompt_token_ids: torch.Tensor | None
    frequency_penalties: torch.Tensor
    presence_penalties: torch.Tensor
    repetition_penalties: torch.Tensor

    output_token_ids: list[list[int]]

    allowed_token_ids_mask: torch.Tensor | None
    """Shape: (max_batch_size, vocab_size), bool mask"""

    bad_words_token_ids: dict[int, list[list[int]]]
    """req_index -> list of bad word token ID sequences"""

    logitsprocs: LogitsProcessors
    """Loaded logits processors"""

    logprob_token_ids: dict[int, list[int]] | None = None
    """req_index -> specific token IDs for logprobs"""

    spec_token_ids: list[list[int]] | None = None
    """Speculative token IDs"""

    thinking_budget_state_holder: ThinkingBudgetStateHolder | None = None
    """Thinking token budget state"""
```

---

## 5. Sampler Module

**File:** `vllm/v1/sample/sampler.py`

### Sampler

```python
class Sampler(nn.Module):
    def __init__(
        self,
        logprobs_mode: LogprobsMode = "raw_logprobs",
    ): ...
    topk_topp_sampler: TopKTopPSampler
    pin_memory: bool
    logprobs_mode: LogprobsMode
```

#### forward

```python
def forward(
    self,
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    predict_bonus_token: bool = False,
    logprobs_mode_override: LogprobsMode | None = None,
) -> SamplerOutput:
```

#### apply_logits_processors

```python
def apply_logits_processors(
    self,
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    predict_bonus_token: bool = False,
) -> torch.Tensor:
```

Applies processors in order:
1. Allowed token IDs mask
2. Bad words filtering
3. Min tokens processor (non-argmax-invariant)
4. Logit bias processor (non-argmax-invariant)
5. All penalties

#### sample

```python
def sample(
    self,
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    logprobs_mode_override: LogprobsMode | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
```

Returns `(sampled_token_ids, processed_logprobs)`.

#### Static Methods

```python
@staticmethod
def apply_temperature(
    logits: torch.Tensor,
    temp: torch.Tensor,
    all_random: bool,
) -> torch.Tensor: ...

@staticmethod
def greedy_sample(logits: torch.Tensor) -> torch.Tensor: ...

@staticmethod
def compute_logprobs(logits: torch.Tensor) -> torch.Tensor: ...

@staticmethod
def gather_logprobs(
    logprobs: torch.Tensor,
    num_logprobs: int,
    token_ids: torch.Tensor,
) -> LogprobsTensors: ...
```

---

## 6. Top-K Top-P Sampler

**File:** `vllm/v1/sample/ops/topk_topp_sampler.py`

### TopKTopPSampler

```python
class TopKTopPSampler(nn.Module):
    def __init__(
        self,
        logprobs_mode: LogprobsMode = "raw_logprobs",
    ) -> None: ...
```

Dispatches to platform-specific implementations:
- **CUDA**: FlashInfer sampler (default) or PyTorch native
- **CPU**: Optimized CPU sampler or native fallback
- **XPU**: XPU sampler or native fallback
- **ROCm**: Aiter sampler or native fallback

#### Forward Methods

| Method | Platform | Description |
|--------|----------|-------------|
| `forward_cuda` | CUDA | Uses FlashInfer sampling ops |
| `forward_cpu` | CPU | Optimized CPU sampling |
| `forward_xpu` | XPU | Intel GPU sampling |
| `forward_native` | Any | PyTorch native implementation |

### Triton Top-K/Top-P

**File:** `vllm/v1/sample/ops/topk_topp_triton.py`

Triton kernel for fused top-k and top-p sampling on GPU.

---

## 7. Logits Processors

### V1 Logits Processor Interface

**File:** `vllm/v1/sample/logits_processor/`

| File | Description |
|------|-------------|
| `interface.py` | LogitsProcessor abstract interface |
| `builtin.py` | Built-in processor implementations |
| `state.py` | LogitsProcessors container state |

### LogitsProcessor Types

#### MinPLogitsProcessor

Filters tokens whose probability is below `min_p * max_prob`:
```python
# Applied as argmax-invariant processor
logits[logits < min_p * max_logit] = -inf
```

#### MinTokensLogitsProcessor

Prevents EOS and stop tokens before `min_tokens`:
```python
# Bans EOS and stop_token_ids until min_tokens is reached
logits[:, eos_token_id] = -inf
```

#### LogitBiasLogitsProcessor

Applies per-token bias:
```python
logits[:, token_id] += bias
```

### V0 Logits Processors

**File:** `vllm/logits_process.py`

#### LogitsProcessor Type Alias

```python
LogitsProcessor: TypeAlias = (
    Callable[[list[int], torch.Tensor], torch.Tensor]
    | Callable[[list[int], list[int], torch.Tensor], torch.Tensor]
)
```

#### NoBadWordsLogitsProcessor

```python
class NoBadWordsLogitsProcessor:
    _SMALLEST_LOGIT = float("-inf")
    _NEUTRAL_LOGIT = 0.0

    def __init__(self, bad_words_ids: list[list[int]]): ...
    def __call__(
        self,
        past_tokens_ids: Sequence[int],
        logits: torch.FloatTensor,
    ) -> torch.Tensor: ...
```

---

## 8. Penalties

**File:** `vllm/v1/sample/ops/penalties.py`

### apply_all_penalties

```python
def apply_all_penalties(
    logits: torch.Tensor,
    prompt_token_ids: torch.Tensor,
    presence_penalties: torch.Tensor,
    frequency_penalties: torch.Tensor,
    repetition_penalties: torch.Tensor,
    output_token_ids: list[list[int]],
) -> torch.Tensor:
```

Applies three types of penalties:

1. **Repetition Penalty**: Penalizes tokens that appear in prompt + generated text.
   - `logit = logit / repetition_penalty` (if logit > 0)
   - `logit = logit * repetition_penalty` (if logit < 0)

2. **Frequency Penalty**: Penalizes based on token frequency in generated text.
   - `logit = logit - frequency_penalty * count(token)`

3. **Presence Penalty**: Penalizes tokens that appear at all in generated text.
   - `logit = logit - presence_penalty * (count(token) > 0)`

### Helper Functions

```python
def _convert_to_tensors(
    output_token_ids: list[list[int]],
    vocab_size: int,
    device: torch.device,
) -> torch.Tensor:
```

Converts variable-length output token ID lists to padded tensors for batch processing.

---

## 9. Logprobs Calculation

### LogprobsTensors

```python
@dataclass
class LogprobsTensors:
    logprob_token_ids: torch.Tensor  # [num_tokens, num_logprobs+1] int32
    logprobs: torch.Tensor           # [num_tokens, num_logprobs+1] float32
    selected_token_ranks: torch.Tensor  # [num_tokens] int32
```

### Logprobs Computation

The sampler supports multiple logprobs modes:

| Mode | Description |
|------|-------------|
| `raw_logprobs` | Log softmax of original logits (before penalties/temperature) |
| `raw_logits` | Clone of original logits |
| `processed_logits` | Logits after all processing |
| `processed_logprobs` | Log softmax of processed logits |

### Logprobs by Token IDs

```python
def gather_specific_token_logprobs(
    self,
    logits: torch.Tensor,
    logprob_token_ids: dict[int, list[int]],
    sampled: torch.Tensor,
) -> LogprobsTensors | None:
```

Efficiently computes logprobs for specific token IDs using a fused Triton kernel
for `log_softmax + gather`. This is ~1.4x faster than sparse gather for batch sizes > 1.

### Logprobs Engine

**File:** `vllm/v1/engine/logprobs.py`

Engine-level logprobs processing including:
- Prompt logprobs computation
- Sample logprobs extraction
- Flat logprobs format support

### Logprobs Ops

**File:** `vllm/v1/sample/ops/logprobs.py`

```python
def batched_count_greater_than(
    values: torch.Tensor,
    threshold: torch.Tensor,
) -> torch.Tensor:
```

Computes rank of each token's logprob (how many tokens have higher logprob).

---

## 10. Beam Search

**File:** `vllm/beam_search.py`

Beam search explores multiple candidate sequences simultaneously, keeping the top-k
best sequences at each step based on cumulative log probability.

### Key Concepts

- **Beam Width**: Number of parallel sequences (controlled by `n` parameter)
- **Beam Hypotheses**: Tracks top sequences with their cumulative scores
- **Beam Search Scorer**: Scores and ranks candidate sequences

### Sequence Scoring

Sequences are scored by:
1. Summing log probabilities of all tokens
2. Optionally applying length penalty
3. Selecting top-k sequences by adjusted score

### Implementation Details

- Each beam maintains its own KV cache state
- Beams can diverge and converge during search
- Early stopping when all beams reach EOS
- Final selection of best sequence from completed beams

---

## 11. Structured Output

**Directory:** `vllm/v1/structured_output/`

### Backends

| Backend | File | Description |
|---------|------|-------------|
| `backend_xgrammar.py` | xgrammar | XGrammar-based structured output |
| `backend_outlines.py` | outlines | Outlines-based structured output |
| `backend_guidance.py` | guidance | Guidance-based structured output |
| `backend_lm_format_enforcer.py` | LM Format Enforcer | Format enforcer backend |

### Supported Formats

1. **JSON Schema**: `structured_outputs.json` - Validates output against JSON schema
2. **Regex**: `structured_outputs.regex` - Conforms output to regex pattern
3. **Choice**: `structured_outputs.choice` - Output must be one of given choices
4. **Grammar**: `structured_outputs.grammar` - Custom grammar (CFG)
5. **JSON Object**: `structured_outputs.json_object` - Any valid JSON object
6. **Structural Tag**: `structured_outputs.structural_tag` - Tag-based structure

### Request Handling

**File:** `vllm/v1/structured_output/request.py`

```python
class StructuredOutputRequest:
    """Wraps structured output parameters for processing."""
```

### Backend Types

**File:** `vllm/v1/structured_output/backend_types.py`

Defines the structured output backend interface and types.

### Utils

**File:** `vllm/v1/structured_output/utils.py`

Utility functions for structured output processing.

---

## 12. Reasoning / Thinking Output

### Thinking Budget State

**File:** `vllm/v1/sample/thinking_budget_state.py`

```python
class ThinkingBudgetStateHolder:
    """Manages thinking token budget for reasoning models."""
```

Tracks:
- Number of thinking tokens used per request
- Maximum thinking token budget (`thinking_token_budget` from SamplingParams)
- Whether to force end thinking phase

### Reasoning Module

**Directory:** `vllm/v1/reasoning/` (or `vllm/reasoning/`)

Handles reasoning/thinking output for models that support chain-of-thought:
- Separates thinking content from final output
- Manages thinking token budget
- Detects thinking start/end markers
- Streams thinking tokens separately

---

## 13. Speculative Decoding

**Directory:** `vllm/v1/spec_decode/`

Speculative decoding improves throughput by guessing future tokens with a smaller
draft model, then verifying them with the target model.

### Speculative Decoding Methods

| Method | File | Description |
|--------|------|-------------|
| **EAGLE** | `eagle.py` | Feature-level speculative decoding using hidden states |
| **Medusa** | `medusa.py` | Multi-head speculative decoding |
| **NGram** | `ngram_proposer.py` | N-gram based token prediction |
| **NGram GPU** | `ngram_proposer_gpu.py` | GPU-accelerated N-gram proposer |
| **Suffix Decoding** | `suffix_decoding.py` | Suffix-based speculative decoding |
| **DFlash** | `dflash.py` | Draft Flash attention |
| **Draft Model** | `draft_model.py` | Small draft model proposer |
| **LLM Base** | `llm_base_proposer.py` | LLM-based proposer base |

### EAGLE (Extraction from Abstract Grid LEarning)

```python
class EagleProposer:
    """Feature-level speculative decoding using hidden states."""
```

- Uses hidden states from the target model to predict future tokens
- Tree-based speculation with multiple draft paths
- Verification against target model probabilities
- Supports `num_speculative_tokens` configuration

### Medusa

```python
class MedusaProposer:
    """Multi-head speculative decoding with Medusa heads."""
```

- Adds multiple prediction heads on top of the model
- Each head predicts a token at a different future position
- Tree-based verification of draft tokens

### NGram Proposer

```python
class NGramProposer:
    """N-gram based speculative token prediction."""
```

- Looks up N-gram patterns from a lookup table
- GPU-accelerated version available
- No additional model parameters needed

### Metadata

**File:** `vllm/v1/spec_decode/metadata.py`

```python
class SpecDecodeMetadata:
    """Metadata for speculative decoding."""
```

### Metrics

**File:** `vllm/v1/spec_decode/metrics.py`

Tracks speculative decoding metrics:
- Draft acceptance rate
- Number of draft tokens per step
- Verification latency

### Hidden State Extraction

**File:** `vllm/v1/spec_decode/extract_hidden_states.py`

Extracts hidden states from the target model for use by draft models.

### Utilities

**File:** `vllm/v1/spec_decode/utils.py`

Helper functions for speculative decoding operations.

---

## 14. Output Processing

### Output Processor

**File:** `vllm/v1/engine/output_processor.py`

Processes model outputs after sampling:
- Handles speculative decoding verification
- Manages logprobs processing
- Handles stop conditions
- Tracks request completion

### SamplerOutput

```python
@dataclass
class SamplerOutput:
    sampled_token_ids: torch.Tensor  # [num_requests, 1] int32
    logprobs_tensors: LogprobsTensors | None
```

---

## 15. Detokenization Pipeline

### Detokenizer

**File:** `vllm/v1/engine/detokenizer.py`

Handles conversion of token IDs back to text:

1. **Incremental Decoding**: Only detokenizes new tokens (delta)
2. **Stop String Detection**: Checks for stop strings in decoded text
3. **Special Token Handling**: Skip/add spaces for special tokens
4. **Buffer Management**: Holds back text for stop string matching

#### Key Operations

```python
class Detokenizer:
    def decode_step(
        self,
        output_token_ids: list[int],
        new_token_ids: list[int],
        skip_special_tokens: bool,
        spaces_between_special_tokens: bool,
    ) -> str: ...
```

#### Stop String Detection

- When `include_stop_str_in_output=False`, text is buffered up to `max(len(stop)) - 1`
  characters to detect stop strings before they appear in output
- Stop strings are matched incrementally as tokens are generated

---

## 16. Stop Conditions

### Stop Triggers

A request stops generating when any of these conditions are met:

1. **EOS Token**: The model generates an end-of-sequence token
   - Unless `ignore_eos=True`
2. **Stop Strings**: Generated text matches a string in `stop`
   - Output may or may not include the stop string based on `include_stop_str_in_output`
3. **Stop Token IDs**: Generated token ID is in `stop_token_ids`
4. **Max Tokens**: Number of generated tokens reaches `max_tokens`
5. **Repetition Detection**: Repetitive N-gram pattern detected (if configured)

### Min Tokens Enforcement

Before `min_tokens` are generated:
- EOS token is banned (logit set to -inf)
- Stop token IDs are banned
- Generation continues regardless

---

## 17. Rejection Sampler

**File:** `vllm/v1/sample/rejection_sampler.py`

```python
class RejectionSampler:
    """Verifies speculative decoding draft tokens."""
```

### Verification Process

For each draft token:
1. Compare draft probability with target probability
2. Accept if `random() < min(1, p_target / p_draft)`
3. Reject if probability ratio is too low
4. When rejected, resample from modified distribution

### Output

- Accepted tokens are kept
- First rejected token is replaced with resampled token
- All subsequent draft tokens are discarded

---

## 18. Bad Words Filtering

### Bad Words Processing

**File:** `vllm/v1/sample/ops/bad_words.py`

```python
def apply_bad_words(
    logits: torch.Tensor,
    bad_words_token_ids: dict[int, list[list[int]]],
) -> torch.Tensor:
```

Sets logits to `-inf` for tokens that would complete a bad word sequence.

### V0 Bad Words

**File:** `vllm/logits_process.py`

```python
def get_bad_words_logits_processors(
    bad_words: list[str],
    tokenizer: TokenizerLike,
) -> list[LogitsProcessor]:
```

Tokenizes bad words and creates a `NoBadWordsLogitsProcessor`.

---

## 19. Parallel Sampling

### Parallel Sampling

**File:** `vllm/v1/engine/parallel_sampling.py`

Handles generation of multiple output sequences (`n > 1`) for a single request:

- Each sequence gets its own KV cache state
- All sequences share the prompt KV cache
- Sequences are processed in parallel during prefill
- During decode, sequences may diverge

### Multi-N Sequence Management

For `n > 1`:
1. Prefill phase: All sequences share the same prompt computation
2. Decode phase: Each sequence maintains its own:
   - Generated token IDs
   - Logprobs
   - KV cache blocks (shared prefix, separate suffix)
3. Output: All `n` sequences are returned with their respective completions

---

## Appendix: Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_MAX_N_SEQUENCES` | `16384` | Maximum value for `n` parameter |
| `VLLM_USE_FLASHINFER_SAMPLER` | `1` | Use FlashInfer for top-k/top-p sampling |
| `VLLM_XPU_USE_SAMPLER_KERNEL` | varies | Use XPU sampler kernel |
| `Q_SCALE_CONSTANT` | varies | Q scale constant for dynamic quantization |
| `K_SCALE_CONSTANT` | varies | K scale constant for dynamic quantization |
| `V_SCALE_CONSTANT` | varies | V scale constant for dynamic quantization |

## Appendix: Logprobs Constants

```python
MAX_LOGPROB_TOKEN_IDS = 128
"""Upper bound on SamplingParams.logprob_token_ids list length."""

_SAMPLING_EPS = 1e-5
_MAX_TEMP = 1e-2
```
