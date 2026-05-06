# 19 - Utils and Helpers Reference

This document provides exhaustive documentation for all utility modules, helper functions, and supporting classes in FlashAttention.

---

## Table of Contents

1. [benchmark.py - Benchmarking Utilities](#benchmarkpy)
2. [distributed.py - Distributed Training Utilities](#distributedpy)
3. [generation.py - Text Generation Utilities](#generationpy)
4. [pretrained.py - Pretrained Model Loading](#pretrainedpy)
5. [testing.py - Testing Utilities](#testingpy)
6. [torch.py - PyTorch Compatibility](#torchpy)
7. [library.py - Custom Op Registration](#librarypy)
8. [bert_padding.py - Padding Utilities](#bert_paddingpy)
9. [Rotary Embedding Layers](#rotary-embedding-layers)
10. [Patch Embedding Layers](#patch-embedding-layers)
11. [Cross Entropy Loss](#cross-entropy-loss)

---

## benchmark.py

File: `flash_attn/utils/benchmark.py`

### `benchmark_forward`

```python
def benchmark_forward(fn, *inputs, repeats=10, desc="", verbose=True,
                      amp=False, amp_dtype=torch.float16, **kwinputs):
```

Benchmarks the forward pass of an arbitrary function using PyTorch's `torch.utils.benchmark.Timer`.

**Parameters**:
- `fn` (Callable): Function to benchmark
- `*inputs`: Positional arguments passed to `fn`
- `repeats` (int): Number of timing repetitions. Default: 10
- `desc` (str): Description string printed before results
- `verbose` (bool): Whether to print results. Default: True
- `amp` (bool): Whether to use automatic mixed precision. Default: False
- `amp_dtype` (torch.dtype): AMP dtype. Default: `torch.float16`
- `**kwinputs`: Keyword arguments passed to `fn`

**Returns**: `(timer, measurement)` tuple

**Example**:
```python
from flash_attn.utils.benchmark import benchmark_forward
t, m = benchmark_forward(flash_attn_func, q, k, v, causal=True, desc="FA2 fwd")
print(f"Median time: {m.median * 1000:.3f} ms")
```

### `benchmark_backward`

```python
def benchmark_backward(fn, *inputs, grad=None, repeats=10, desc="",
                       verbose=True, amp=False, amp_dtype=torch.float16,
                       **kwinputs):
```

Benchmarks the backward pass. Runs forward first to build the computation graph, then measures backward only.

**Parameters**:
- `grad` (Tensor, optional): Gradient tensor for the output. If None, random gradients are generated.
- All other parameters same as `benchmark_forward`

**Returns**: `(timer, measurement)` tuple

### `benchmark_combined`

```python
def benchmark_combined(fn, *inputs, grad=None, repeats=10, desc="",
                       verbose=True, amp=False, amp_dtype=torch.float16,
                       **kwinputs):
```

Benchmarks the combined forward + backward pass in a single measurement.

**Returns**: `(timer, measurement)` tuple

### `benchmark_fwd_bwd`

```python
def benchmark_fwd_bwd(fn, *inputs, grad=None, repeats=10, desc="",
                       verbose=True, amp=False, amp_dtype=torch.float16,
                       **kwinputs):
```

Returns separate forward and backward measurements as a tuple.

**Returns**: `((t_fwd, m_fwd), (t_bwd, m_bwd))` tuple

### `benchmark_all`

```python
def benchmark_all(fn, *inputs, grad=None, repeats=10, desc="",
                  verbose=True, amp=False, amp_dtype=torch.float16,
                  **kwinputs):
```

Returns forward, backward, and combined measurements.

**Returns**: `((t_fwd, m_fwd), (t_bwd, m_bwd), (t_comb, m_comb))` tuple

### `pytorch_profiler`

```python
def pytorch_profiler(fn, *inputs, trace_filename=None, backward=False,
                     amp=False, amp_dtype=torch.float16, cpu=False,
                     verbose=True, **kwinputs):
```

Wraps benchmark functions with PyTorch profiler for detailed CUDA analysis.

**Parameters**:
- `trace_filename` (str, optional): Path to export Chrome trace file
- `backward` (bool): Whether to profile backward pass
- `cpu` (bool): Whether to include CPU activities
- Warmup: 30 iterations before profiling

**Example**:
```python
pytorch_profiler(flash_attn_func, q, k, v, causal=True,
                 backward=True, trace_filename="fa2_trace.json")
```

### `benchmark_memory`

```python
def benchmark_memory(fn, *inputs, desc="", verbose=True, **kwinputs):
```

Measures peak GPU memory usage during function execution.

**Returns**: Maximum memory allocated in GB (float)

---

## distributed.py

File: `flash_attn/utils/distributed.py`

### `all_gather_raw`

```python
def all_gather_raw(input_: Tensor, process_group: ProcessGroup,
                   async_op: bool = False) -> Tuple[Tensor, Handle]:
```

Raw all-gather operation without autograd support.

**Parameters**:
- `input_` (Tensor): Input tensor of shape `(local_dim, ...)`
- `process_group` (ProcessGroup): Distributed process group
- `async_op` (bool): Whether to launch asynchronously

**Returns**: `(output_tensor, work_handle)` where output has shape `(world_size * local_dim, ...)`

### `reduce_scatter_raw`

```python
def reduce_scatter_raw(input_: Tensor, process_group: ProcessGroup,
                       async_op: bool = False) -> Tuple[Tensor, Handle]:
```

Raw reduce-scatter operation. The first dimension must be divisible by `world_size`.

**Returns**: `(output_tensor, work_handle)` where output has shape `(input_dim / world_size, ...)`

### `all_reduce_raw`

```python
def all_reduce_raw(input_: Tensor, process_group: ProcessGroup,
                   async_op: bool = False) -> Tuple[Tensor, Handle]:
```

Raw all-reduce (sum) operation. Modifies input in-place.

**Returns**: `(input_tensor, work_handle)`

### `all_gather`

```python
all_gather = AllGatherFunc.apply
```

Autograd-compatible all-gather. Forward: all-gather; Backward: reduce-scatter.

**Usage**:
```python
output = all_gather(local_input, process_group)
```

### `reduce_scatter`

```python
reduce_scatter = ReduceScatterFunc.apply
```

Autograd-compatible reduce-scatter. Forward: reduce-scatter; Backward: all-gather.

### `all_reduce`

```python
all_reduce = AllReduceFunc.apply
```

Autograd-compatible all-reduce. Forward: all-reduce; Backward: pass-through.

### `sync_shared_params`

```python
def sync_shared_params(model: torch.nn.Module, process_group: ProcessGroup):
```

Broadcasts parameters marked with `_shared_params=True` from rank 0 to all ranks. Iterates in sorted order to ensure consistent ordering across ranks.

**Usage**:
```python
# Mark parameters as shared
for p in model.layer_norm.parameters():
    p._shared_params = True

sync_shared_params(model, process_group)
```

### `allreduce_sequence_parallel_grad`

```python
def allreduce_sequence_parallel_grad(model: torch.nn.Module,
                                     process_group: ProcessGroup):
```

All-reduces gradients for parameters marked with `_sequence_parallel=True`. Uses coalesced all-reduce for efficiency.

**Usage**:
```python
# After backward pass
allreduce_sequence_parallel_grad(model, tp_group)
```

### `get_dim_for_local_rank`

```python
def get_dim_for_local_rank(dim: int, world_size: int, local_rank: int,
                           multiple_of: int = 1) -> int:
```

Computes the local dimension for the given rank when splitting `dim` across `world_size` processes. Handles uneven splits.

**Parameters**:
- `dim` (int): Total dimension to split
- `world_size` (int): Number of processes
- `local_rank` (int): Local rank index
- `multiple_of` (int): Ensure result is a multiple of this value

**Returns**: Local dimension for this rank

---

## generation.py

File: `flash_attn/utils/generation.py`

### `InferenceParams`

```python
@dataclass
class InferenceParams:
    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)
    lengths_per_sample: Optional[Tensor] = None

    def reset(self, max_seqlen, max_batch_size):
        """Reset for a new generation."""
        self.max_seqlen = max_seqlen
        self.max_batch_size = max_batch_size
        self.seqlen_offset = 0
        if self.lengths_per_sample is not None:
            self.lengths_per_sample.zero_()
```

### `modify_logits_for_top_k_filtering`

```python
def modify_logits_for_top_k_filtering(logits, top_k):
```

Sets logits for non-top-k values to `-inf` in-place. Used before sampling.

### `modify_logits_for_top_p_filtering`

```python
def modify_logits_for_top_p_filtering(logits, top_p):
```

Sets logits for tokens exceeding cumulative `top_p` probability to `-inf` in-place.

### `sample`

```python
def sample(logits, top_k=1, top_p=0.0, temperature=1.0):
```

Samples tokens from logits with top-k and top-p filtering.

**Parameters**:
- `logits` (Tensor): Shape `(batch_size, vocab_size)`
- `top_k` (int): Number of top candidates. 0 = no limit. 1 = greedy
- `top_p` (float): Cumulative probability threshold. 0.0 = disabled
- `temperature` (float): Sampling temperature

**Returns**: Sampled token indices, shape `(batch_size,)`

### `decode`

```python
@torch.inference_mode()
def decode(input_ids, model, max_length, top_k=1, top_p=0.0, temperature=1.0,
           eos_token_id=None, teacher_outputs=None, vocab_size=None,
           tensor_parallel=1, cg=False, enable_timing=False):
```

Autoregressive decoding with optional CUDA graph capture.

**Parameters**:
- `input_ids` (Tensor): Shape `(batch, seq_len)`, input token IDs
- `model`: Model with `forward` method accepting `inference_params`
- `max_length` (int): Maximum generation length
- `top_k` (int): Top-k sampling parameter
- `top_p` (float): Top-p sampling parameter
- `temperature` (float): Sampling temperature
- `eos_token_id` (int, optional): End-of-sequence token ID
- `teacher_outputs` (Tensor, optional): Teacher forcing outputs
- `vocab_size` (int, optional): Limit vocabulary size
- `tensor_parallel` (int): Tensor parallelism degree
- `cg` (bool): Enable CUDA graph capture for decoding
- `enable_timing` (bool): Print timing information

**Returns**: `GreedySearchDecoderOnlyOutput` or `SampleDecoderOnlyOutput` with `sequences` and `scores` fields

### `sample_speculative`

```python
def sample_speculative(logits, logits_draft, tokens_draft,
                       top_k=1, top_p=0.0, temperature=1.0):
```

Speculative decoding sampling (Algorithm 1 from Leviathan et al., 2022).

**Parameters**:
- `logits` (Tensor): `(batch, seqlen+1, vocab_size)` main model logits
- `logits_draft` (Tensor): `(batch, seqlen, vocab_size)` draft model logits
- `tokens_draft` (Tensor): `(batch, seqlen)` draft tokens

**Returns**: `(tokens, num_generated_tokens)` tuple

### `decode_speculative`

```python
@torch.inference_mode()
def decode_speculative(input_ids, model, model_draft, max_length,
                       speculative_lookahead=3, top_k=1, top_p=0.0,
                       temperature=1.0, eos_token_id=None, vocab_size=None,
                       tensor_parallel=1, cg=False, enable_timing=False,
                       debug=False):
```

Speculative decoding with draft model. Currently only supports `batch_size=1`.

### `GenerationMixin`

```python
class GenerationMixin:
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        raise NotImplementedError

    def generate(self, input_ids, max_length, top_k=1, top_p=0.0,
                 temperature=1.0, return_dict_in_generate=False,
                 output_scores=False, **kwargs):
```

Mixin class for models to add generation capability.

### `allocate_inference_cache`

```python
def allocate_inference_cache(max_batch_size, max_seqlen, nheads, headdim,
                             layers, device, dtype=torch.float16):
```

Allocates KV cache for all layers.

**Parameters**:
- `max_batch_size` (int): Maximum batch size
- `max_seqlen` (int): Maximum sequence length
- `nheads` (int): Number of attention heads
- `headdim` (int): Head dimension
- `layers` (int or Sequence): Layer indices
- `device` (torch.device): Device to allocate on
- `dtype` (torch.dtype): Data type. Default: `torch.float16`

**Returns**: Dict mapping layer index to KV cache tensor of shape `(max_batch_size, max_seqlen, 2, nheads, headdim)`

### `DecodingCGCache`

```python
@dataclass
class DecodingCGCache:
    max_batch_size: int = 0
    max_seqlen: int = 0
    device = None
    dtype = None
    callables: dict = field(default_factory=dict)
    mempool = None
    inference_params: Optional[InferenceParams] = None
    run: Optional[Callable] = None
```

### `update_graph_cache`

```python
@torch.inference_mode()
def update_graph_cache(model, cache, batch_size, seqlen_og, max_seqlen,
                       decoding_seqlens=(1,), tensor_parallel=1, dtype=None,
                       n_warmups=2):
```

Updates or creates CUDA graph cache for autoregressive decoding.

**Parameters**:
- `model`: The model
- `cache`: Existing `DecodingCGCache` or None
- `batch_size` (int): Current batch size
- `seqlen_og` (int): Original prompt sequence length
- `max_seqlen` (int): Maximum generation length
- `decoding_seqlens` (tuple): Sequence lengths to capture graphs for
- `n_warmups` (int): Number of warmup iterations before capture

**Returns**: Updated `DecodingCGCache`

### `capture_graph`

```python
def capture_graph(model, inference_params, batch_size, max_seqlen,
                  decoding_seqlen=1, mempool=None, n_warmups=2):
```

Captures a CUDA graph for a specific decoding sequence length. Called internally by `update_graph_cache`.

---

## pretrained.py

File: `flash_attn/utils/pretrained.py`

### `state_dict_from_pretrained`

```python
def state_dict_from_pretrained(model_name, device=None, dtype=None):
```

Loads a pretrained model's state dict from local files or Hugging Face Hub.

**Parameters**:
- `model_name` (str): Model name or local path
- `device` (torch.device, optional): Target device. Default: CPU (for non-fp32)
- `dtype` (torch.dtype, optional): Target dtype. Converted after loading

**Returns**: Dict of parameter name to tensor

**Features**:
- Supports `pytorch_model.bin` and `model.safetensors` formats
- Supports sharded checkpoints (`pytorch_model.bin.index.json`)
- Searches local directory first, then Hugging Face Hub
- Handles both safe tensors and pickle formats

---

## testing.py

File: `flash_attn/utils/testing.py`

### `generate_random_padding_mask`

```python
def generate_random_padding_mask(max_seqlen, batch_size, device,
                                 mode="random", zero_lengths=False):
```

Generates random padding masks for testing variable-length sequences.

**Parameters**:
- `max_seqlen` (int): Maximum sequence length
- `batch_size` (int): Number of sequences
- `device` (torch.device): Device for the mask
- `mode` (str): One of "full" (all max length), "random", or "third" (between 1/3 and max)
- `zero_lengths` (bool): Whether to include zero-length sequences (every 5th batch + last)

**Returns**: Boolean tensor of shape `(batch_size, max_seqlen)`

### `generate_qkv`

```python
def generate_qkv(q, k, v, query_padding_mask=None, key_padding_mask=None,
                  qv=None, kvpacked=False, qkvpacked=False,
                  query_unused_mask=None, key_unused_mask=None):
```

Generates Q, K, V tensors in various packed formats for testing.

**Parameters**:
- `q` (Tensor): `(batch, seqlen_q, nheads, d)`
- `k` (Tensor): `(batch, seqlen_k, nheads_k, d)`
- `v` (Tensor): `(batch, seqlen_k, nheads_k, d_v)`
- `query_padding_mask` (Tensor, optional): `(batch, seqlen_q)` bool
- `key_padding_mask` (Tensor, optional): `(batch, seqlen_k)` bool
- `qv` (Tensor, optional): For MLA-style attention
- `kvpacked` (bool): Return K,V stacked as `(batch, seqlen, 2, nheads, d)`
- `qkvpacked` (bool): Return Q,K,V stacked as `(batch, seqlen, 3, nheads, d)`
- `query_unused_mask` (Tensor, optional): Mask for allocated-but-unused elements
- `key_unused_mask` (Tensor, optional): Same for keys

**Returns**: Tuple of tensors in the requested format. For the default (no packing):
```
(q_unpad, k_unpad, v_unpad, qv_unpad,
 cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k,
 max_seqlen_q, max_seqlen_k,
 q, k, v, qv,
 output_pad_fn, dq_pad_fn, dk_pad_fn)
```

### `construct_local_mask`

```python
def construct_local_mask(seqlen_q, seqlen_k, window_size=(None, None),
                          sink_token_length=0, query_padding_mask=None,
                          key_padding_mask=None, key_leftpad=None, device=None):
```

Constructs a local (sliding window) attention mask.

**Parameters**:
- `seqlen_q` (int): Query sequence length
- `seqlen_k` (int): Key sequence length
- `window_size` (tuple): `(left, right)` window sizes. None means no bound
- `sink_token_length` (int): Number of "sink" tokens that are always visible
- `query_padding_mask` (Tensor, optional): For computing effective lengths
- `key_padding_mask` (Tensor, optional): Same for keys
- `key_leftpad` (Tensor, optional): Left padding per batch element

**Returns**: Boolean mask tensor where True means "masked out"

### `construct_chunk_mask`

```python
def construct_chunk_mask(seqlen_q, seqlen_k, attention_chunk,
                          query_padding_mask=None, key_padding_mask=None,
                          key_leftpad=None, device=None):
```

Constructs a chunk attention mask where attention is restricted to chunks of `attention_chunk` size.

### `attention_ref`

```python
def attention_ref(q, k, v, query_padding_mask=None, key_padding_mask=None,
                   key_leftpad=None, attn_bias=None, dropout_p=0.0,
                   dropout_mask=None, causal=False, qv=None,
                   q_descale=None, k_descale=None, v_descale=None,
                   window_size=(None, None), attention_chunk=0,
                   sink_token_length=0, learnable_sink=None,
                   softcap=0.0, upcast=True, reorder_ops=False,
                   intermediate_dtype=None):
```

Reference attention implementation for testing. Computes attention using standard PyTorch operations.

**Parameters**:
- `q, k, v` (Tensor): `(batch, seqlen, nheads, head_dim)` or `(batch, seqlen, nheads, head_dim_v)` for V
- `query_padding_mask` (Tensor, optional): `(batch, seqlen_q)` bool
- `key_padding_mask` (Tensor, optional): `(batch, seqlen_k)` bool
- `attn_bias` (Tensor, optional): Broadcastable to `(batch, nheads, seqlen_q, seqlen_k)`
- `dropout_p` (float): Dropout probability
- `dropout_mask` (Tensor, optional): `(batch, nheads, seqlen_q, seqlen_k)` bool
- `causal` (bool): Apply causal masking
- `qv` (Tensor, optional): For MLA attention
- `q_descale, k_descale, v_descale` (Tensor, optional): FP8 descale factors
- `window_size` (tuple): Sliding window size
- `attention_chunk` (int): Chunk attention size
- `softcap` (float): Softcap value
- `upcast` (bool): Cast to fp32 for computation
- `reorder_ops` (bool): Change operation order for numerical error estimation

**Returns**: `(output, attention)` tuple
- `output`: `(batch, seqlen_q, nheads, head_dim_v)`
- `attention`: `(batch, nheads, seqlen_q, seqlen_k)` softmax probabilities

---

## torch.py

File: `flash_attn/utils/torch.py`

### `custom_fwd` / `custom_bwd`

```python
# Handles PyTorch version differences in AMP decorators
from flash_attn.utils.torch import custom_fwd, custom_bwd

@custom_fwd
def forward(ctx, ...):
    ...

@custom_bwd
def backward(ctx, ...):
    ...
```

Automatically selects between `torch.amp.custom_fwd/custom_bwd` (PyTorch >= 2.0) and `torch.cuda.amp.custom_fwd/custom_bwd` (PyTorch 1.x).

---

## library.py

File: `flash_attn/utils/library.py`

### `triton_op`

```python
def triton_op(name, fn=None, /, *, mutates_args, schema=None,
              allow_decomposition=True):
```

Registers a function as a `torch.library` custom op with optional decomposition for `torch.compile()`.

**Parameters**:
- `name` (str): Operator name (e.g., `"flash_attn::flash_attn_func"`)
- `fn` (Callable, optional): The function to wrap
- `mutates_args` (str or Iterable[str]): Arguments that are mutated
- `schema` (str, optional): Operator schema string
- `allow_decomposition` (bool): If True, allows Inductor to decompose the op

**Returns**: `CustomOpDef` that can be called as a function

**Usage**:
```python
@triton_op("my_lib::my_op", mutates_args=())
def my_op(x, y):
    return x + y

# Can be used with torch.compile
compiled_fn = torch.compile(my_op)
```

**Key difference from `torch.library.triton_op`**: The `schema` parameter is passed through to `custom_op`, allowing explicit type specification.

---

## bert_padding.py

File: `flash_attn/bert_padding.py`

### `IndexFirstAxis`

```python
class IndexFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, indices):
        """Selects elements from input at the given indices.
        Args:
            input: (total_tokens, ...) where total_tokens >= len(indices)
            indices: (num_selected,) integer indices
        Returns: (num_selected, ...)
        """

    @staticmethod
    def backward(ctx, grad_output):
        """Scatter gradients back to original positions."""
```

Autograd-compatible indexing along the first axis. Uses `torch.gather` for performance.

### `IndexPutFirstAxis`

```python
class IndexPutFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, indices, first_axis_dim):
        """Places values at the given indices in a zero-initialized tensor.
        Args:
            values: (num_selected, ...)
            indices: (num_selected,) integer indices
            first_axis_dim: Size of the first axis in the output
        Returns: (first_axis_dim, ...)
        """

    @staticmethod
    def backward(ctx, grad_output):
        """Gathers gradients from the indexed positions."""
```

### `IndexFirstAxisResidual`

```python
class IndexFirstAxisResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, indices):
        """Same as IndexFirstAxis but also returns the input for residual gradient.
        Returns: (indexed_output, input_detached)
        """

    @staticmethod
    def backward(ctx, grad_output, grad_residual):
        """Uses scatter_add for gradient accumulation (residual path)."""
```

### `unpad_input`

```python
def unpad_input(hidden_states, attention_mask, unused_mask=None):
    """Remove padding from a padded input tensor.
    Args:
        hidden_states: (batch, seqlen, ...)
        attention_mask: (batch, seqlen), bool (1=valid, 0=pad)
        unused_mask: (batch, seqlen), bool (1=allocated but unused)
    Returns:
        output: (total_nnz, ...)
        indices: (total_nnz,)
        cu_seqlens: (batch + 1,)
        max_seqlen_in_batch: int
        seqused: (batch,) number of valid tokens per batch
    """
```

### `unpad_input_for_concatenated_sequences`

```python
def unpad_input_for_concatenated_sequences(hidden_states, attention_mask_in_length):
    """Unpad for concatenated short samples within a single sequence.
    Args:
        hidden_states: (batch, seqlen, ...)
        attention_mask_in_length: (batch, seqlen), int
            Nonzero = length of concatenated sub-sequence at that position
            0 = no sub-sequence starts here
    Returns: Same as unpad_input
    """
```

Supports packing multiple short sequences into one long sequence for efficient fine-tuning (e.g., SFT in LLMs).

### `pad_input`

```python
def pad_input(hidden_states, indices, batch, seqlen):
    """Reverse of unpad_input.
    Args:
        hidden_states: (total_nnz, ...)
        indices: (total_nnz,) original positions
        batch: int, batch size
        seqlen: int, max sequence length
    Returns: (batch, seqlen, ...)
    """
```

---

## Rotary Embedding Layers

File: `flash_attn/layers/rotary.py`

### `RotaryEmbedding`

```python
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000.0, device=None):
        """Rotary position embedding.
        Args:
            dim (int): Dimension of the rotary embedding (typically head_dim)
            base (float): Base for the frequency computation. Default: 10000
        """
```

#### Methods

**`forward`**:
```python
def forward(self, max_seqlen, device=None):
    """Compute cos and sin for rotary embedding.
    Returns: (cos, sin) tensors of shape (max_seqlen, dim)
    """
```

**`apply_rotary_emb`**:
```python
def apply_rotary_emb(q, k, cos, sin, interleaved=False, inplace=True):
    """Apply rotary embedding to query and key tensors.
    Args:
        q, k: (batch, seqlen, nheads, headdim)
        cos, sin: (seqlen, headdim) or (1, seqlen, 1, headdim)
        interleaved: True for interleaved format, False for half-rotation
        inplace: Whether to modify tensors in place
    Returns: (q_rotated, k_rotated)
    """
```

---

## Patch Embedding Layers

File: `flash_attn/layers/patch_embed.py`

### `PatchEmbed`

```python
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=768, norm_layer=None, flatten=True):
        """2D Image to Patch Embedding.
        Args:
            img_size (int): Image size. Default: 224
            patch_size (int): Patch size. Default: 16
            in_chans (int): Number of input channels. Default: 3
            embed_dim (int): Embedding dimension. Default: 768
            norm_layer (nn.Module, optional): Normalization layer
            flatten (bool): Flatten spatial dimensions. Default: True
        """
```

#### Forward

```python
def forward(self, x):
    """Args: x: (batch, channels, height, width)
    Returns: (batch, num_patches, embed_dim) if flatten else (batch, embed_dim, h', w')
    """
```

---

## Cross Entropy Loss

File: `flash_attn/losses/cross_entropy.py`

### `CrossEntropyLoss`

```python
class CrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=-100, reduction='mean', label_smoothing=0.0):
        """Cross entropy loss with optional label smoothing.
        Args:
            ignore_index (int): Index to ignore in loss computation. Default: -100
            reduction (str): 'mean' or 'sum'. Default: 'mean'
            label_smoothing (float): Label smoothing factor. Default: 0.0
        """
```

This is typically a reimplementation that may be optimized for use with FlashAttention's internal tensor layouts or fused kernels.
