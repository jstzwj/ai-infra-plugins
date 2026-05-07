# Chapter 23: CUDA Graphs

## Source Files
- `megatron/core/transformer/cuda_graphs.py` - Core CUDA graph implementation
- `megatron/core/full_cuda_graph.py` - Full CUDA graph wrapper
- `megatron/core/transformer/enums.py` - CudaGraphScope enum

## Overview

CUDA graphs capture a sequence of GPU operations (kernels, memcpy, etc.) into a single graph object that can be replayed with minimal CPU overhead. Megatron-LM provides comprehensive CUDA graph support for both training and inference.

CUDA graphs are particularly beneficial for:
- Small micro-batch sizes where kernel launch overhead dominates
- Transformer layers with many small operations (layer norm, bias add, activations)
- RL inference rollouts with repeated generation passes
- Optimizer steps with many small all-gather/reduce-scatter operations

## CudaGraphScope

The scope of CUDA graph capture is controlled by `CudaGraphScope`:

| Scope | Description |
|-------|-------------|
| `CudaGraphScope.unspecified` | Default, no graphs |
| `CudaGraphScope.training` | Capture forward and backward of transformer layers |
| `CudaGraphScope.training_inference` | Both training and inference graphs |
| `CudaGraphScope.inference` | Only inference graphs |

```bash
--cuda-graph-impl local    # Megatron-native CUDA graphs
--cuda-graph-impl te        # Transformer Engine CUDA graphs
```

## Architecture

### Key Classes

```
_CudagraphGlobalRecord     -- Global singleton managing graph creation order
  ├── cudagraph_record     -- List of (runner, graph_type, args...) tuples
  ├── tensor_reuse_pool    -- Pool for reusing input/output buffers across graphs
  └── cudagraph_created    -- Flag indicating graphs have been captured

_CudaGraphRunner           -- Per-module graph runner
  ├── create_fwd_graph()   -- Capture forward graph
  ├── create_bwd_graph()   -- Capture backward graph
  ├── replay_fwd()         -- Replay forward graph
  └── replay_bwd()         -- Replay backward graph

TECudaGraphHelper           -- Transformer Engine integration helper
  ├── capture()            -- Setup TE for graph capture
  └── replay()             -- Replay TE-graphed operations

TensorReusePool             -- Buffer reuse across graphs
  ├── get(meta)            -- Get or allocate a buffer
  ├── insert(tensor)       -- Return a buffer to the pool
  └── owns(tensor)         -- Check if tensor is from this pool

ArgMetadata                 -- Metadata for graph arguments
  ├── type, shape, dtype, device
  └── zeros_like()         -- Create zero tensor with same properties

CudagraphBufferMetadata     -- Per-buffer metadata for reuse tracking
  ├── is_cudagraph_input/output
  ├── input_use_count
  └── fwd/bwd_cudagraph_buffer
```

## Graph Capture Process

### Phase 1: Warmup

During the first training step, graph-eligible modules record their execution order:

```python
_set_warmup_start()
# During forward/backward, each _CudaGraphRunner records:
_CudagraphGlobalRecord.record_fwd_graph(runner, args, kwargs, out)
_CudagraphGlobalRecord.record_bwd_graph(runner)
_set_warmup_end()
```

The warmup phase uses `is_graph_warmup()` to return True, which signals to modules that they should record rather than immediately capture.

### Phase 2: Capture

At the end of the first step (in pipeline schedule functions), `create_cudagraphs()` is called:

```python
capture_stats = _CudagraphGlobalRecord.create_cudagraphs()
```

This iterates through the recorded operations in execution order and:

1. **Preparation:**
   - Freezes Python garbage collection (`gc.collect()`, `torch.cuda.empty_cache()`)
   - Sets global capture flag (`_IS_GRAPH_CAPTURING = True`)
   - If TE modules present, calls `te_set_capture_start()`

2. **Per-graph capture:**
   - For forward graphs: `runner.create_fwd_graph(args, kwargs, out, clone_inputs=True)`
   - For backward graphs: `runner.create_bwd_graph()`
   - Progress tracking with memory stats

3. **Finalization:**
   - Sets `_IS_GRAPH_CAPTURING = False`
   - If TE modules, calls `te_set_capture_end()`
   - Reports capture time and memory usage

### Phase 3: Replay

On subsequent steps, captured graphs are replayed:

```python
# In module forward:
if self.cudagraph_created:
    return self.replay_fwd(*args, **kwargs)
else:
    # Normal eager execution
    return super().forward(*args, **kwargs)
```

Replay copies inputs into the captured input buffers and launches the graph, which writes outputs to captured output buffers.

## Graph Types

### Local CUDA Graphs (Megatron-native)

The default implementation when `--cuda-graph-impl local`:

```bash
--cuda-graph-impl local
```

Features:
- Captures forward and backward passes of each `GraphableMegatronModule`
- Uses `TensorReusePool` to share buffers across graphs
- Manages RNG state for reproducibility
- Supports `StaticInferenceContext` and `DynamicInferenceContext`

### TE CUDA Graphs

When `--cuda-graph-impl te` and Transformer Engine is available:

```bash
--cuda-graph-impl te
```

Uses TE's `make_graphed_callables` for capturing TE-wrapped modules. Benefits:
- TE-specific optimizations for FP8 kernels
- Weak reference buffers reduce memory overhead
- Automatic FP8 tensor save/restore across graph boundaries

### Full CUDA Graph

The `FullCudaGraphWrapper` captures the entire forward-backward-optimizer sequence as a single graph:

```bash
--full-cuda-graph
```

This eliminates all kernel launch overhead but requires:
- Fixed batch sizes and sequence lengths
- No dynamic control flow
- Careful memory management

## Buffer Management

### TensorReusePool

The `TensorReusePool` manages buffer reuse across multiple CUDA graphs to minimize memory overhead:

```python
pool = TensorReusePool()

# During capture: get buffer for graph output
buffer = pool.get(ArgMetadata(output_tensor))

# During next capture: return buffer for reuse
pool.insert(buffer)

# Check ownership
if pool.owns(some_tensor):
    pool.insert(some_tensor)
```

The pool maintains:
- `tensor_strong_refs`: Strong references preventing deallocation between captures
- `tensor_strong_refs_dataptrs`: Set of data pointers for ownership checks
- `pool`: Available buffers for reuse

### CudagraphBufferMetadata

Each tensor can carry metadata about its role in graph execution:

```python
tensor.cg_buffer_metadata = CudagraphBufferMetadata(
    is_cudagraph_input=True,
    input_use_count=1,
    cudagraph_reuse_ref_count=0,
)
```

This metadata enables:
- Tracking whether a tensor is a graph input or output
- Reference counting for safe buffer reuse
- Enabling cross-graph buffer sharing (output of one graph = input of next)

## Memory Considerations

### Memory Overhead

CUDA graphs require additional memory for:
1. Input buffers (copied from real inputs before each replay)
2. Output buffers (read after each replay)
3. Intermediate activations (captured during graph recording)
4. Workspace memory for kernels

The capture process reports memory usage:
```
> built 96 cuda graph(s) in 12.34 sec, with total memory usage:
  allocated 2.1 GB, reserved 2.5 GB.
```

### Memory Optimization Strategies

1. **Shared memory pools:** Multiple graphs share the same CUDA memory pool (`_CudagraphGlobalRecord` ensures graphs are created in execution order, enabling pool sharing)

2. **Tensor reuse:** The `TensorReusePool` reuses buffers between consecutive graphs

3. **TE weak references:** When using TE graphs, weak reference buffers allow memory to be reclaimed between graph replays while remaining valid during replay

4. **GC freezing:** Python garbage collection is frozen during capture to prevent interference with CUDA memory allocation

### First/Last Layer Detection

The `_determine_if_first_last_layer_of_this_vp_chunk()` function identifies pipeline stage boundaries, allowing special handling for layers that need different graph configurations at VP chunk boundaries.

## MoE Support

CUDA graphs are compatible with Mixture of Experts layers:

- Expert routing and top-k selection are captured in the graph
- The `transition_moe_cudagraphs` utility manages MoE-specific graph transitions
- Expert parallel parameters must be consistent across replays
- Token-to-expert routing must use deterministic patterns for graph compatibility

## RNG State Management

CUDA graphs capture RNG state at capture time. For reproducibility:

```python
gen = _ensure_generator_state_is_cudagraph_safe(gen)
```

This function:
1. Gets the current generator state
2. Clones it outside inference mode (to avoid inference tensor issues)
3. Sets the cloned state back on the generator

## Inference CUDA Graphs

For inference workloads (including RL rollouts):

```bash
--inference-cuda-graphs
```

Inference graphs:
- Capture only the forward pass
- Can be used with decode-only generation
- Support variable-length sequences via padding/unpadding
- Are tracked separately in `cudagraph_inference_record`

### RL Training with CUDA Graphs

In RL training, CUDA graphs are used for both:
1. **Training step graphs**: Capture the training forward-backward-optimizer
2. **Inference rollout graphs**: Capture the inference generation pass

```bash
--rl-training-cuda-graphs
```

The RL training flow with CUDA graphs:
1. Build inference CUDA graphs before first rollout collection
2. Replay inference graphs for each rollout batch
3. Build training CUDA graphs during first training step
4. Replay training graphs for subsequent training steps

## Configuration Examples

### Basic Training CUDA Graphs
```bash
python pretrain_gpt.py \
    --cuda-graph-impl local \
    --recompute-granularity selective \
    --recompute-method uniform \
    ...
```

### Training with TE CUDA Graphs
```bash
python pretrain_gpt.py \
    --cuda-graph-impl te \
    --fp8 hybrid \
    --fp8-recipe delayed \
    ...
```

### Full CUDA Graph (Maximum Performance)
```bash
python pretrain_gpt.py \
    --full-cuda-graph \
    --no-async-gradient-reduction \
    ...
```

### RL Training with CUDA Graphs
```bash
python train_rl.py \
    --cuda-graph-impl local \
    --rl-training-cuda-graphs \
    ...
```

## Limitations

1. **Fixed shapes:** Graph inputs must have the same shape across all replays. Variable-length sequences require padding.

2. **No dynamic control flow:** Conditional branches, dynamic loops, or data-dependent shapes cannot be captured.

3. **Memory overhead:** Each captured graph holds references to all intermediate tensors.

4. **Warmup cost:** The first step is slower due to warmup + capture.

5. **Debugging difficulty:** Errors inside captured graphs are harder to diagnose.

6. **Pipeline parallelism:** Graphs are captured per-pipeline-stage, with the pipeline schedule calling `create_cudagraphs()` at the end of each stage's first step.

## Deleting Graphs

```python
from megatron.core.transformer.cuda_graphs import delete_cuda_graphs
delete_cuda_graphs()
```

This resets all graph runners, clears the global record, and frees captured graph memory. Useful when transitioning between training phases (e.g., from training to inference in RL).
