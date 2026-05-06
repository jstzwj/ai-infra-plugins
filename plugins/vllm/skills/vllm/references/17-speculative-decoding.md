# vLLM Speculative Decoding Reference

This document provides a comprehensive reference for vLLM's speculative decoding system, covering all algorithms, configuration, proposers, token trees, metrics, and CUDA kernel utilities.

---

## Table of Contents

1. [Overview](#overview)
2. [Speculative Configuration](#speculative-configuration-speculativeconfig)
3. [Speculative Methods](#speculative-methods)
4. [Proposer Classes](#proposer-classes)
5. [SpecDecodeBaseProposer](#specdecodebaseproposer)
6. [EagleProposer](#eagleproposer)
7. [DFlashProposer](#dflashproposer)
8. [DraftModelProposer](#draftmodelproposer)
9. [MedusaProposer](#medusaproposer)
10. [NgramProposer (CPU)](#ngramproposer-cpu)
11. [NgramProposerGPU](#ngramproposergpu)
12. [SuffixDecodingProposer](#suffixdecodingproposer)
13. [ExtractHiddenStatesProposer](#extracthiddenstatesproposer)
14. [Speculative Decoding Metadata](#speculative-decoding-metadata)
15. [Metrics and Logging](#metrics-and-logging)
16. [Triton CUDA Kernels](#triton-cuda-kernels)
17. [Token Trees](#token-trees)
18. [Parallel Drafting](#parallel-drafting)
19. [Padded Drafter Batch](#padded-drafter-batch)

---

## Overview

Speculative decoding accelerates inference by having a smaller, faster "draft" model propose candidate tokens that the larger "target" model then verifies in a single forward pass. When the draft tokens are correct, this effectively generates multiple tokens per step, increasing throughput without sacrificing quality.

vLLM supports multiple speculative decoding methods:
- **EAGLE / EAGLE3**: Uses hidden states from the target model for high-quality draft proposals
- **MTP (Multi-Token Prediction)**: Uses the target model's own MTP heads
- **DFlash**: Parallel drafting with cross-attention between context and query states
- **Medusa**: Multiple LM heads on top of the model for parallel token prediction
- **N-gram**: CPU or GPU-based lookup of matching n-grams in the prompt
- **Suffix Decoding**: Pattern matching using suffix trees built from prompt history
- **Draft Model**: Standalone smaller draft model
- **Extract Hidden States**: Caches hidden states without actual speculation (for KV transfer)
- **MLP Speculator**: MLP-based prediction heads (currently disabled)

---

## Speculative Configuration (SpeculativeConfig)

**Source:** `vllm/config/speculative.py`

```python
@config
class SpeculativeConfig:
    ...
```

### Configuration Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_speculative_tokens` | `int` | required | Number of tokens to speculate per step |
| `model` | `str \| None` | `None` | Draft model name or path |
| `method` | `str \| None` | `None` | Speculative method (auto-detected if None) |
| `draft_tensor_parallel_size` | `int \| None` | `None` | Tensor parallel size for draft model |
| `quantization` | `str \| None` | `None` | Quantization for draft model |
| `moe_backend` | `str \| None` | `None` | MoE backend for draft model |
| `attention_backend` | `str \| None` | `None` | Attention backend override |
| `max_model_len` | `int \| None` | `None` | Maximum model length for draft model |
| `disable_padded_drafter_batch` | `bool` | `False` | Disable padded drafter batch |
| `use_local_argmax_reduction` | `bool` | `False` | Use local argmax reduction |
| `prompt_lookup_max` | `int` | `None` | Maximum n-gram size for prompt lookup |
| `prompt_lookup_min` | `int` | `None` | Minimum n-gram size for prompt lookup |
| `speculative_token_tree` | `str \| None` | `None` | Token tree configuration |
| `parallel_drafting` | `bool \| None` | `None` | Enable parallel drafting mode |
| `suffix_decoding_max_tree_depth` | `int` | `64` | Maximum tree depth for suffix decoding |
| `suffix_decoding_max_spec_factor` | `float` | `1.0` | Maximum speculation factor for suffix decoding |
| `suffix_decoding_min_token_prob` | `float` | `0.0` | Minimum token probability for suffix decoding |
| `suffix_decoding_max_cached_requests` | `int` | `1000` | Maximum cached requests for suffix decoding |
| `rejection_sample_method` | `str` | `None` | Rejection sampling method |
| `synthetic_acceptance_rates` | `list[float] \| None` | `None` | Synthetic acceptance rates for benchmarking |
| `synthetic_acceptance_length` | `int \| None` | `None` | Fixed acceptance length for benchmarking |
| `draft_sample_method` | `str \| None` | `None` | Draft token sampling method |

### SpeculativeMethod Enum

```python
class SpeculativeMethod(str, Enum):
    EAGLE = "eagle"
    EAGLE3 = "eagle3"
    MTP = "mtp"
    DFLASH = "dflash"
    MEDUSA = "medusa"
    NGRAM = "ngram"
    NGRAM_GPU = "ngram_gpu"
    SUFFIX = "suffix"
    DRAFT_MODEL = "draft_model"
    EXTRACT_HIDDEN_STATES = "extract_hidden_states"
    MLP_SPECULATOR = "mlp_speculator"
```

### MTPModelTypes

```python
class MTPModelTypes(str, Enum):
    DEEPSEEK_MTP = "DeepSeekMTPModel"
    DEEPSEEK_V4_MTP = "DeepSeekV4MTPModel"
    ERNIE_MTP = "ErnieMTPModel"
    EXAONE_MOE_MTP = "ExaoneMoeMTP"
    EXAONE4_5_MTP = "Exaone4_5_MTP"
    NEMOTRON_H_MTP = "NemotronHMTPModel"
    LONGCAT_FLASH_MTP = "LongCatFlashMTPModel"
    GLM4_MOE_MTP = "Glm4MoeMTPModel"
    GLM4_MOE_LITE_MTP = "Glm4MoeLiteMTPModel"
    GLM_OCR_MTP = "GlmOcrMTPModel"
    MIMO_MTP = "MiMoMTPModel"
    MIMO_V2_MTP = "MiMoV2MTPModel"
    MIMO_V2_OMNI_MTP = "MiMoV2OmniMTPModel"
    OPENPANGU_MTP = "OpenPanguMTPModel"
    QWEN3_NEXT_MTP = "Qwen3NextMTP"
    STEP3P5_MTP = "Step3p5MTP"
    QWEN3_5_MTP = "Qwen3_5MTP"
    QWEN3_5_MOE_MTP = "Qwen3_5MoeMTP"
    HYV3_MTP = "HYV3MTPModel"
```

### EagleModelTypes

```python
class EagleModelTypes(str, Enum):
    EAGLE_LLAMA = "EagleLlamaForCausalLM"
    EAGLE_LLAMA4 = "EagleLlama4ForCausalLM"
    EAGLE_MINICPM = "EagleMiniCPMForCausalLM"
    EAGLE3_LLAMA = "Eagle3LlamaForCausalLM"
    EAGLE3_MINIMAX_M2 = "Eagle3MiniMaxM2ForCausalLM"
    EAGLE_MISTRAL = "EagleMistralForCausalLM"
    EAGLE_MISTRAL_LARGE3 = "EagleMistralLarge3ForCausalLM"
    EAGLE3_DEEPSEEK_V2 = "Eagle3DeepseekV2ForCausalLM"
    EAGLE3_DEEPSEEK_V3 = "Eagle3DeepseekV3ForCausalLM"
    EAGLE_DEEPSEEK_MTP = "EagleDeepSeekMTPModel"
    DFLASH_QWEN3 = "DFlashDraftModel"
    EAGLE_LLAMA3 = "LlamaForCausalLMEagle3"
    EAGLE3_QWEN2_5VL = "Eagle3Qwen2_5vlForCausalLM"
    EAGLE3_QWEN3VL = "Eagle3Qwen3vlForCausalLM"
```

### RejectionSampleMethod

```python
class RejectionSampleMethod(str, Enum):
    COMPARE_DRAFT_TOKENS = "compare_draft_tokens"
    SAMPLING = "sampling"
```

### Key Properties

- `max_num_new_slots_for_drafting: int` - Maximum number of new KV slots needed for one speculative decoding step. Equals `num_speculative_tokens * num_speculative_drafts + 1` for tree speculation, or `num_speculative_tokens + 1` otherwise.

- `use_eagle: bool` - True if method is EAGLE or EAGLE3
- `use_dflash: bool` - True if method is DFLASH
- `use_medusa: bool` - True if method is MEDUSA
- `use_ngram: bool` - True if method is NGRAM or NGRAM_GPU
- `use_suffix: bool` - True if method is SUFFIX
- `use_extract_hidden_states: bool` - True if method is EXTRACT_HIDDEN_STATES

### Method Auto-Detection

When `method` is None, the speculative method is auto-detected from the draft model's architecture:

1. **EAGLE/EAGLE3**: If architecture is in `EagleModelTypes`
2. **MTP**: If architecture is in `MTPModelTypes`
3. **DFlash**: If architecture is `DFlashDraftModel`
4. **Medusa**: If architecture is `MedusaModel`
5. **Extract Hidden States**: If architecture is `ExtractHiddenStatesModel`
6. **Draft Model**: For all other architectures

### hf_config_override

```python
@staticmethod
def hf_config_override(config: PretrainedConfig) -> PretrainedConfig
```

Detects and configures MTP model types by reading the HF config's `model_type` field. Sets the architecture appropriately for known MTP model types.

---

## Speculative Methods

### Method Summary Table

| Method | Model Required | Hidden States | Parallel Drafting | Tree Spec | Verification |
|--------|---------------|---------------|-------------------|-----------|-------------|
| `eagle` | Yes | Yes | No | Yes | Rejection sampling |
| `eagle3` | Yes | Yes + aux | No | Yes | Rejection sampling |
| `mtp` | Yes (MTP heads) | No | No | Optional | Rejection sampling |
| `dflash` | Yes | Yes | Yes | No | Rejection sampling |
| `medusa` | Yes (Medusa heads) | No | Yes | No | Rejection sampling |
| `ngram` | No | No | No | No | Rejection sampling |
| `ngram_gpu` | No | No | No | No | Rejection sampling |
| `suffix` | No | No | No | No | Rejection sampling |
| `draft_model` | Yes | No | No | No | Rejection sampling |
| `extract_hidden_states` | Yes (cache-only) | Yes | No | No | Always accepts |
| `mlp_speculator` | Yes (MLP heads) | No | Yes | No | Rejection sampling |

---

## Proposer Classes

All proposers share a common interface for proposing draft tokens. The hierarchy is:

```
SpecDecodeBaseProposer (abstract base)
  +-- EagleProposer
  +-- DFlashProposer
  +-- DraftModelProposer

MedusaProposer (standalone)
NgramProposer (standalone, CPU)
NgramProposerGPU (standalone, GPU)
SuffixDecodingProposer (standalone)
ExtractHiddenStatesProposer (standalone)
```

---

## SpecDecodeBaseProposer

**Source:** `vllm/v1/spec_decode/llm_base_proposer.py`

The base class for LLM-based proposers (EAGLE, DFlash, standalone draft models). Provides buffer management, input preparation, CUDA graph support, and the main proposal loop.

### Initialization

```python
class SpecDecodeBaseProposer:
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        pass_hidden_states_to_model: bool = False,
        runner=None,
    )
```

Parameters:
- `vllm_config` - Full vLLM configuration
- `device` - Device for tensor buffers
- `pass_hidden_states_to_model` - Whether to pass hidden states to the draft model (True for EAGLE)
- `runner` - Reference to the model runner

### Buffer Management

The proposer pre-allocates GPU buffers for:

| Buffer | Shape | Dtype | Purpose |
|--------|-------|-------|---------|
| `input_ids` | `(max_num_tokens,)` | `torch.int32` | Token IDs for draft model input |
| `positions` | `(max_num_tokens,)` | `torch.int64` | Position indices |
| `hidden_states` | `(max_num_tokens, hidden_size)` | `dtype` | Hidden states from target model |
| `_slot_mapping_buffer` | `(max_num_tokens,)` | `torch.int64` | Slot mapping for KV cache |
| `token_arange_np` | `(max_batch_size + 1,)` | `numpy` | Arange for query_start_loc construction |

### Key Methods

#### `propose()`

```python
def propose(
    self,
    target_token_ids: torch.Tensor,
    next_token_ids: torch.Tensor,
    target_positions: torch.Tensor,
    target_hidden_states: torch.Tensor,
    common_attn_metadata: CommonAttentionMetadata,
    sampled_token_ids: torch.Tensor | None = None,
    token_indices_to_sample: torch.Tensor | None = None,
    num_rejected_tokens_gpu: torch.Tensor | None = None,
) -> torch.Tensor
```

Main entry point for proposing draft tokens. Implements autoregressive drafting with the following flow:

1. Call `set_inputs_first_pass()` to prepare inputs for the first draft step
2. Run draft model forward pass
3. Sample next tokens from draft model logits
4. For subsequent steps, rotate inputs and run additional forward passes
5. Return proposed token IDs

If `speculative_token_tree` is configured, delegates to `propose_tree()` instead.

#### `set_inputs_first_pass()`

```python
def set_inputs_first_pass(
    self,
    target_token_ids: torch.Tensor,
    next_token_ids: torch.Tensor,
    target_positions: torch.Tensor,
    target_hidden_states: torch.Tensor,
    token_indices_to_sample: torch.Tensor | None,
    cad: CommonAttentionMetadata,
    num_rejected_tokens_gpu: torch.Tensor | None,
) -> tuple[int, torch.Tensor, CommonAttentionMetadata]
```

Prepares inputs for the first draft forward pass. Uses Triton kernel `eagle_prepare_inputs_padded_kernel` (or `copy_and_expand_eagle_inputs_kernel` for non-padded mode) to:
- Copy and expand input IDs, positions, and slot mappings for each request
- Handle rejected tokens by adjusting sequences
- Build new attention metadata for the draft model

Returns `(num_input_tokens, token_indices_to_sample, updated_common_attn_metadata)`.

#### `propose_tree()`

```python
def propose_tree(
    self,
    target_token_ids: torch.Tensor,
    next_token_ids: torch.Tensor,
    target_positions: torch.Tensor,
    target_hidden_states: torch.Tensor,
    common_attn_metadata: CommonAttentionMetadata,
    sampled_token_ids: torch.Tensor | None,
) -> torch.Tensor
```

Tree-based speculative decoding. Instead of linear autoregressive drafting, generates a tree of candidate tokens. This allows exploring multiple token paths in parallel.

#### `prepare_inputs()`

```python
def prepare_inputs(
    self,
    step: int,
    num_input_tokens: int,
    token_indices_to_sample: torch.Tensor | None,
    cad: CommonAttentionMetadata,
) -> tuple[dict[str, Any], int]
```

Builds model input dictionary for a given drafting step. Handles:
- EAGLE-style hidden state stacking
- Position increment per step
- Slot mapping construction

#### `prepare_inputs_padded()`

```python
def prepare_inputs_padded(
    self,
    step: int,
    num_input_tokens: int,
    common_attn_metadata: CommonAttentionMetadata,
) -> tuple[dict[str, Any], int]
```

Similar to `prepare_inputs` but for padded mode. Uses `eagle_prepare_next_token_padded_kernel` Triton kernel for input preparation.

#### `load_model()`

```python
def load_model(target_model: nn.Module) -> None
```

Loads the draft model. For EAGLE models, handles embedding and LM head sharing with the target model:
- If `tie_embeddings` in HF config, shares embedding weights
- If `tie_lm_head` in HF config or both have same vocab and embed dimension, shares LM head

Also initializes attention backend and metadata builder.

#### `dummy_run()`

```python
@torch.inference_mode()
def dummy_run(
    self,
    num_tokens: int,
    use_cudagraphs: bool = True,
    is_graph_capturing: bool = False,
    slot_mappings: dict[str, torch.Tensor] | None = None,
) -> None
```

Runs a dummy forward pass for CUDA graph capture and memory profiling.

#### `build_per_group_and_layer_attn_metadata()`

```python
def build_per_group_and_layer_attn_metadata(
    self,
    cad: CommonAttentionMetadata,
    draft_index: int = 0,
) -> tuple[list[object], dict[str, object]]
```

Builds attention metadata for each KV cache group and layer. Handles multiple draft indices for multi-step speculation.

### Data Parallel Support

The proposer supports data parallelism via:
- `_determine_batch_execution_and_padding()` - Coordinates batch sizes across DP ranks
- DP-aware CUDA graph dispatching
- Per-rank attention metadata construction

### CUDA Graph Support

- `initialize_cudagraph_keys()` - Initializes CUDA graph keys for the dispatcher
- `cudagraph_dispatcher` - Manages CUDA graph capture and replay
- Piecewise CUDA graphs for speculative decoding phases

---

## EagleProposer

**Source:** `vllm/v1/spec_decode/eagle.py`

```python
class EagleProposer(SpecDecodeBaseProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    )
```

EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency) proposer. Key characteristics:

- **Hidden state passing**: Passes target model's hidden states to the draft model
- **Embedding concatenation**: Concatenates token embeddings with hidden states for richer input
- **Tree speculation support**: Can use speculative token trees for exploring multiple paths
- **EAGLE3 support**: Uses auxiliary hidden states from intermediate layers

### EAGLE3 Configuration

For EAGLE3 models, the `use_aux_hidden_state` flag is read from the draft model's HF config. When enabled, the proposer uses hidden states from multiple intermediate layers of the target model.

### Token Embedding Mode

EAGLE uses a special `parallel_drafting_token_id` (typically a mask token) to embed draft tokens alongside hidden states.

---

## DFlashProposer

**Source:** `vllm/v1/spec_decode/dflash.py`

```python
class DFlashProposer(SpecDecodeBaseProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    )
```

DFlash (Draft Flash) proposer for parallel drafting with non-causal cross-attention.

### Key Differences from EAGLE

1. **Parallel Drafting**: All speculative tokens are drafted in a single forward pass (not autoregressive)
2. **Non-Causal Attention**: Uses non-causal attention for cross-attention between context and query states
3. **Context/Query Separation**: Separates context K/V from query Q states
4. **Precomputed Context KVs**: Context key/values are pre-inserted into cache via `model.precompute_and_store_context_kv()`

### Buffer Architecture

| Buffer | Purpose |
|--------|---------|
| `_context_slot_mapping_buffer` | Slot mapping for context (target) K/V states |
| `_slot_mapping_buffer` | Slot mapping for query (draft) tokens |
| `_context_positions_buffer` | Positions for context states |
| `positions` | Positions for query states |
| `arange` | Arange tensor for query_start_loc construction |

### set_inputs_first_pass()

```python
@override
def set_inputs_first_pass(
    self,
    target_token_ids: torch.Tensor,
    next_token_ids: torch.Tensor,
    target_positions: torch.Tensor,
    target_hidden_states: torch.Tensor,
    token_indices_to_sample: torch.Tensor | None,
    cad: CommonAttentionMetadata,
    num_rejected_tokens_gpu: torch.Tensor | None,
) -> tuple[int, torch.Tensor, CommonAttentionMetadata]
```

Prepares DFlash inputs using `copy_and_expand_dflash_inputs_kernel` Triton kernel. Creates:
- Context positions and slot mappings (from target model)
- Query positions and slot mappings (for draft tokens)
- `token_indices_to_sample` for sampling from draft logits

Constructs new `CommonAttentionMetadata` with `causal=False` for non-causal attention.

### build_model_inputs_first_pass()

```python
@override
def build_model_inputs_first_pass(
    self,
    num_tokens: int,
    num_input_tokens: int,
    mm_embed_inputs: tuple | None,
) -> tuple[dict[str, Any], int]
```

Pre-inserts context K/V into cache via `model.precompute_and_store_context_kv()`, then returns model inputs for query tokens only.

### dummy_run()

```python
@override
@torch.inference_mode()
def dummy_run(
    self,
    num_tokens: int,
    use_cudagraphs: bool = True,
    is_graph_capturing: bool = False,
    slot_mappings: dict[str, torch.Tensor] | None = None,
) -> None
```

Runs context KV precomputation followed by query model forward. Only profiles query tokens for CUDA graphs since context precomputation is not graph-captured.

### Attention Metadata Validation

```python
@override
def build_per_group_and_layer_attn_metadata(
    self, cad: CommonAttentionMetadata, draft_index: int = 0
) -> tuple[list[object], dict[str, object]]
```

Validates that all attention metadata has `causal=False`, raising an error if the attention backend does not support non-causal attention.

---

## DraftModelProposer

**Source:** `vllm/v1/spec_decode/draft_model.py`

```python
class DraftModelProposer(SpecDecodeBaseProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    )
```

Standalone draft model proposer. Uses a completely separate model as the draft model. Key validations:

- **Vocabulary size check**: Draft model's vocab size must match target model's vocab size
- **Tensor parallel check**: Draft model's TP degree must be compatible with the target model

---

## MedusaProposer

**Source:** `vllm/v1/spec_decode/medusa.py`

```python
class MedusaProposer:
    def __init__(self, vllm_config: VllmConfig, device: torch.device)
```

Medusa-style speculative decoding uses multiple LM heads on top of the model's hidden states to predict multiple future tokens in parallel.

### Key Methods

#### `propose()`

```python
def propose(
    self,
    target_hidden_states: torch.Tensor,
    common_attn_metadata: CommonAttentionMetadata,
    sampled_token_ids: torch.Tensor | None = None,
) -> torch.Tensor
```

Takes target model hidden states and generates draft tokens using Medusa heads. Each head predicts a token at a different future position.

#### `load_model()`

```python
def load_model(target_model: nn.Module) -> None
```

Loads the Medusa model. The Medusa heads are separate from the target model and produce predictions from the target model's hidden states.

---

## NgramProposer (CPU)

**Source:** `vllm/v1/spec_decode/ngram_proposer.py`

```python
class NgramProposer:
    def __init__(self, vllm_config: VllmConfig)
```

CPU-based n-gram proposer using numba JIT compilation. Uses the Longest Previous Sequence (LPS) algorithm to find the longest matching n-gram in the prompt history.

### Key Properties

- `prompt_lookup_max: int` - Maximum n-gram size
- `prompt_lookup_min: int` - Minimum n-gram size

### `propose()`

```python
def propose(
    self,
    input_ids: torch.Tensor,
    num_tokens: int,
) -> list[int]
```

Given the input token sequence, finds the longest matching n-gram and returns the tokens following that match as draft tokens. Uses numba-accelerated `_find_longest_ngram_match` function.

### LPS Algorithm

The `_find_longest_ngram_match` function implements a longest prefix suffix matching algorithm:
1. For each n-gram length from `max` down to `min`
2. Search backward through the input for matching n-grams
3. Return the longest match found
4. Speculate tokens following the match

---

## NgramProposerGPU

**Source:** `vllm/v1/spec_decode/ngram_proposer_gpu.py`

```python
class NgramProposerGPU:
    def __init__(self, vllm_config: VllmConfig, device: torch.device)
```

GPU-accelerated n-gram proposer using vectorized matching kernels.

### NgramGPUKernel

```python
class NgramGPUKernel(nn.Module):
    def __init__(
        self,
        max_num_seqs: int,
        max_model_len: int,
        device: torch.device,
        max_ngram: int,
        min_ngram: int,
    )
```

Core GPU matching kernel. Maintains pre-allocated buffers for vectorized n-gram matching:

| Buffer | Shape | Purpose |
|--------|-------|---------|
| `token_ids_buf` | `(max_num_seqs, max_model_len)` | Token ID buffer |
| `ngram_search_buf` | `(max_ngram - min_ngram + 1, max_num_seqs, max_model_len)` | Search result buffer |
| `matched_length_buf` | `(max_num_seqs,)` | Matched length per sequence |
| `match_buf` | `(max_num_seqs, max_model_len)` | Match result buffer |

### Key Methods

- `propose(input_ids_cpu, seq_lens, num_speculative_tokens) -> list[list[int]]` - Main proposal entry point
- `_vectorized_ngram_match(seq_lens, num_speculative_tokens) -> torch.Tensor` - GPU-accelerated n-gram matching

### Helper Functions

- `update_tensor_buf(buf, new_data, lengths)` - Updates buffer with new data at specified lengths
- `async_d2h_copy(tensor, event)` - Asynchronous device-to-host copy with CUDA event synchronization

---

## SuffixDecodingProposer

**Source:** `vllm/v1/spec_decode/suffix_decoding.py`

```python
class SuffixDecodingProposer:
    def __init__(self, vllm_config: VllmConfig)
```

Implements suffix decoding from "Suffix Decoding: A Model-Free Approach to Speeding Up Large Language Model Inference" (arXiv:2411.04975). Uses the Arctic Inference library's suffix tree implementation.

### Key Properties

| Property | Type | Description |
|----------|------|-------------|
| `num_speculative_tokens` | `int` | Max tokens to speculate per request |
| `max_tree_depth` | `int` | Max suffix tree search depth |
| `max_spec_factor` | `float` | Max speculation factor |
| `min_token_prob` | `float` | Minimum token probability threshold |
| `max_model_len` | `int` | Maximum model length |
| `suffix_cache` | `SuffixDecodingCache` | Arctic Inference suffix tree cache |

### `propose()`

```python
def propose(
    self,
    input_batch: InputBatch,
    sampled_token_ids: list[list[int]],
    slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None = None,
) -> list[list[int]]
```

For each request in the batch:
1. Check if request has reached max model length
2. Manage the suffix cache (start/evict/stop requests)
3. Add newly sampled tokens to the cache
4. Extract a pattern from recent tokens (up to `max_tree_depth`)
5. Use `suffix_cache.speculate()` to get draft tokens
6. Returns variable-length draft token lists per request

### Cache Management

- `start_request(req_id, prompt_token_ids)` - Builds suffix tree for a new prompt
- `add_active_response(req_id, sampled_ids)` - Appends new tokens to cache
- `evict_cached_response(req_id)` - Resets cache for reused request IDs
- `stop_request(req_id)` - Stops tracking a request

---

## ExtractHiddenStatesProposer

**Source:** `vllm/v1/spec_decode/extract_hidden_states.py`

```python
class ExtractHiddenStatesProposer:
    def __init__(self, vllm_config: VllmConfig, device: torch.device)
```

Special proposer that doesn't perform actual speculation. Instead, it caches hidden states in the KV cache for later use (e.g., KV transfer between nodes). Always returns the target model's sampled tokens as "draft" tokens, ensuring 100% verification rate.

### Key Properties

| Property | Type | Description |
|----------|------|-------------|
| `model` | `nn.Module \| None` | ExtractHiddenStatesModel |
| `attn_layer_names` | `list[str]` | Names of attention layers |
| `attn_metadata_builder` | `AttentionMetadataBuilder \| None` | Attention metadata builder |
| `hidden_states` | `torch.Tensor` | Buffer `[max_num_tokens, num_hidden_states, hidden_size]` |
| `num_hidden_states` | `int` | Number of aux hidden state layers |
| `cudagraph_dispatcher` | `CudagraphDispatcher` | CUDA graph dispatcher |

### Initialization Requirements

- `num_speculative_tokens` must be 1
- `disable_padded_drafter_batch` must be False
- `eagle_aux_hidden_state_layer_ids` must be set in draft model config

### `propose()`

```python
def propose(
    self,
    sampled_token_ids: torch.Tensor,
    target_hidden_states: list[torch.Tensor],
    common_attn_metadata: CommonAttentionMetadata,
    slot_mappings: dict | list | None = None,
) -> torch.Tensor
```

1. Stacks hidden states from multiple layers: `[num_tokens, num_hidden_states, hidden_size]`
2. Copies to pre-allocated buffer
3. Runs the ExtractHiddenStatesModel forward (cache-only, no computation)
4. Returns sampled tokens as "draft" (guaranteed acceptance)

### `load_model()`

```python
def load_model(target_model: nn.Module) -> None
```

1. Records target model's attention layers
2. Loads draft model with `set_model_tag("extract_hidden_states")`
3. Identifies draft-specific attention layers (difference from target)
4. Asserts exactly one attention layer in draft model
5. Builds attention metadata builder for that layer

### Data Parallel Support

Supports DP coordination via `coordinate_batch_across_dp()`. Raises error if DBO ubatching is needed (not implemented for this proposer).

---

## Speculative Decoding Metadata

**Source:** `vllm/v1/spec_decode/metadata.py`

```python
@dataclass
class SpecDecodeMetadata:
    draft_token_ids: torch.Tensor          # [num_requests, max_num_spec_tokens]
    num_draft_tokens: list[int]            # Per-request draft token count
    cu_num_draft_tokens: list[int]         # Cumulative sum of draft tokens
    target_logits_indices: torch.Tensor    # Indices into target logits for verification
    bonus_logits_indices: torch.Tensor     # Indices for bonus token logits
    logits_indices: torch.Tensor           # Combined indices for all logits needed
```

This metadata is constructed during the proposal phase and used during verification to:
- Extract the correct logits from the target model's output
- Verify draft tokens against target model predictions
- Determine bonus tokens for accepted sequences

---

## Metrics and Logging

**Source:** `vllm/v1/spec_decode/metrics.py`

### SpecDecodingStats

```python
@dataclass
class SpecDecodingStats:
    num_drafts: int = 0                              # Total draft steps
    num_draft_tokens: int = 0                        # Total tokens drafted
    num_accepted_tokens: int = 0                     # Total tokens accepted
    acceptance_rates: list[float] | None = None      # Per-step acceptance rates
    num_accepted_tokens_per_pos: list[int] | None = None  # Per-position acceptance counts
```

### SpecDecodingLogging

```python
class SpecDecodingLogging:
    @staticmethod
    def observe(stats: SpecDecodingStats, spec_config: SpeculativeConfig) -> None
```

Logs speculative decoding statistics including:
- Draft acceptance rate
- Average draft length
- Per-position acceptance distribution
- Comparison with theoretical acceptance rates

### SpecDecodingProm (Prometheus Metrics)

```python
class SpecDecodingProm:
    # Prometheus counter metric names
    vllm_spec_decode_num_drafts = "vllm:spec_decode_num_drafts_total"
    vllm_spec_decode_num_draft_tokens = "vllm:spec_decode_num_draft_tokens_total"
    vllm_spec_decode_num_accepted_tokens = "vllm:spec_decode_num_accepted_tokens_total"
    vllm_spec_decode_num_accepted_tokens_per_pos = "vllm:spec_decode_num_accepted_tokens_per_pos"
```

Prometheus metrics exposed:

| Metric Name | Type | Description |
|-------------|------|-------------|
| `vllm:spec_decode_num_drafts_total` | Counter | Total number of draft steps |
| `vllm:spec_decode_num_draft_tokens_total` | Counter | Total number of drafted tokens |
| `vllm:spec_decode_num_accepted_tokens_total` | Counter | Total number of accepted tokens |
| `vllm:spec_decode_num_accepted_tokens_per_pos` | Counter vector | Accepted tokens per draft position |

The per-position metric uses a `position` label to track acceptance at each speculative token position (0, 1, 2, ...).

---

## Triton CUDA Kernels

**Source:** `vllm/v1/spec_decode/utils.py`

vLLM uses Triton JIT-compiled CUDA kernels for high-performance input preparation during speculative decoding.

### eagle_step_slot_mapping_metadata_kernel

```python
@triton.jit
def eagle_step_slot_mapping_metadata_kernel(
    slot_mapping_in_ptr, slot_mapping_out_ptr,
    num_draft_tokens_ptr, cu_num_draft_tokens_ptr,
    num_seqs, num_spec_tokens, max_num_tokens: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
)
```

Computes slot mapping metadata for EAGLE drafting steps. Expands slot mappings from target model to draft model tokens.

### eagle_prepare_inputs_padded_kernel

```python
@triton.jit
def eagle_prepare_inputs_padded_kernel(
    next_token_ids_ptr, target_positions_ptr,
    out_input_ids_ptr, out_positions_ptr,
    out_slot_mapping_ptr, out_token_indices_ptr,
    block_table_ptr, block_table_stride,
    query_start_loc_ptr, num_rejected_tokens_ptr,
    parallel_drafting_token_id, block_size,
    num_query_per_req, num_speculative_tokens,
    total_input_tokens, BLOCK_SIZE: tl.constexpr,
    HAS_NUM_REJECTED: tl.constexpr,
)
```

Fused kernel for preparing EAGLE inputs in padded mode. Handles:
- Input ID expansion with parallel drafting token
- Position computation for draft tokens
- Slot mapping construction from block tables
- Token index computation for sampling
- Rejected token adjustment

### eagle_prepare_next_token_padded_kernel

```python
@triton.jit
def eagle_prepare_next_token_padded_kernel(
    next_token_ids_ptr, positions_ptr,
    out_input_ids_ptr, out_positions_ptr,
    out_slot_mapping_ptr, out_token_indices_ptr,
    block_table_ptr, block_table_stride,
    query_start_loc_ptr, num_rejected_tokens_ptr,
    parallel_drafting_token_id, block_size,
    num_query_per_req, num_speculative_tokens,
    total_input_tokens, BLOCK_SIZE: tl.constexpr,
    HAS_NUM_REJECTED: tl.constexpr,
)
```

Prepares next-token inputs for subsequent EAGLE drafting steps in padded mode.

### copy_and_expand_eagle_inputs_kernel

```python
@triton.jit
def copy_and_expand_eagle_inputs_kernel(
    next_token_ids_ptr, target_positions_ptr,
    out_input_ids_ptr, out_positions_ptr,
    out_slot_mapping_ptr,
    block_table_ptr, block_table_stride,
    query_start_loc_ptr,
    parallel_drafting_token_id, block_size,
    num_query_per_req, total_input_tokens,
    BLOCK_SIZE: tl.constexpr,
)
```

Non-padded version of input expansion for EAGLE. Simpler than padded kernel - no rejected token handling.

### copy_and_expand_dflash_inputs_kernel

```python
@triton.jit
def copy_and_expand_dflash_inputs_kernel(
    next_token_ids_ptr, target_positions_ptr,
    out_input_ids_ptr, out_context_positions_ptr,
    out_query_positions_ptr, out_context_slot_mapping_ptr,
    out_query_slot_mapping_ptr, out_token_indices_ptr,
    block_table_ptr, block_table_stride,
    query_start_loc_ptr, num_rejected_tokens_ptr,
    parallel_drafting_token_id, block_size,
    num_query_per_req, num_speculative_tokens,
    total_input_tokens, BLOCK_SIZE: tl.constexpr,
    HAS_NUM_REJECTED: tl.constexpr,
)
```

DFlash-specific input expansion. Separates context and query states:
- Context positions and slot mappings (from target model)
- Query positions and slot mappings (for draft tokens)
- Token indices for sampling

### Utility Functions

#### compute_new_slot_mapping

```python
def compute_new_slot_mapping(
    block_table: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_reqs: int,
    num_prev_tokens: int,
    num_new_tokens: int,
    block_size: int,
) -> torch.Tensor
```

Computes new slot mappings for draft tokens based on block tables.

#### extend_all_queries_by_N

```python
def extend_all_queries_by_N(
    cad: CommonAttentionMetadata,
    N: int,
    device: torch.device,
    block_size: int,
    block_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, CommonAttentionMetadata]
```

Extends all queries by N tokens, updating query_start_loc, seq_lens, and slot_mapping.

#### update_num_computed_tokens_for_batch_change

```python
def update_num_computed_tokens_for_batch_change(
    common_attn_metadata: CommonAttentionMetadata,
    num_draft_tokens_accepted: list[int],
) -> CommonAttentionMetadata
```

Updates computed token counts when the batch changes due to accepted/rejected draft tokens.

---

## Token Trees

Token trees allow exploring multiple token paths during speculative decoding. Instead of a single linear sequence of draft tokens, a tree of candidate sequences is generated.

### Configuration

Token trees are enabled via `speculative_token_tree` in `SpeculativeConfig`. The tree structure defines:
- Branching factor at each level
- Which draft tokens to verify together
- How to score and select the best path

### Tree-Based Proposal Flow

1. `propose_tree()` generates multiple candidate tokens at each position
2. The tree structure determines which tokens to include
3. All tree tokens are verified in a single target model forward pass
4. The best (longest matching) path is selected

---

## Parallel Drafting

Parallel drafting generates all speculative tokens in a single forward pass instead of autoregressively. This reduces drafting latency significantly.

### Methods Supporting Parallel Drafting

- **DFlash**: Native parallel drafting via cross-attention
- **Medusa**: Multiple heads predict tokens at different positions
- **MLP Speculator**: MLP heads predict future tokens

### How It Works

In parallel drafting, the draft model processes all speculative token positions simultaneously:
1. Context K/V states are pre-inserted into the cache
2. Query tokens (one per speculative position per request) are processed in one forward pass
3. The draft model predicts all positions in parallel

---

## Padded Drafter Batch

The padded drafter batch is a technique for efficient batching in speculative decoding. When different requests have different numbers of accepted/rejected tokens, the batch is padded to maintain uniform tensor shapes.

### Configuration

- `disable_padded_drafter_batch: bool = False` in `SpeculativeConfig` - Set to True to disable padding
- Disabled padding is not supported by EAGLE or ExtractHiddenStates methods

### Padding Logic

The padded batch approach:
1. After verification, some requests may have rejected tokens
2. Rejected tokens are removed but the batch size stays constant
3. Padding tokens are added to maintain consistent tensor shapes
4. This allows CUDA graph capture with fixed-size inputs

### Slot Mapping Padding

The `PADDING_SLOT_ID = -1` is used for padding slots that should not be written to in the KV cache.
