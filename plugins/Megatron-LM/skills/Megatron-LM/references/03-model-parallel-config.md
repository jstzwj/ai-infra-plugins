# Chapter 03: ModelParallelConfig Reference

## Source Files
- `sources/Megatron-LM/megatron/core/model_parallel_config.py`

## Overview

`ModelParallelConfig` is the base configuration dataclass for Megatron Core. It controls all parallelism strategies, training parameters, optimizations, and pipeline parallel settings. It is extended by `TransformerConfig` which adds model architecture parameters.

```python
from megatron.core import ModelParallelConfig

config = ModelParallelConfig(
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=2,
    bf16=True,
    sequence_parallel=True,
)
```

## Model Parallelism Parameters

### tensor_model_parallel_size
- **Type**: `int`
- **Default**: `1`
- **Description**: Intra-layer model parallelism. Splits weight tensors across GPU ranks. Each GPU holds 1/TP of each weight matrix.

### pipeline_model_parallel_comm_backend
- **Type**: `Optional[Literal["nccl", "ucc"]]`
- **Default**: `None`
- **Description**: Backend for pipeline parallel communication. If None, uses the default backend.

### pipeline_model_parallel_size
- **Type**: `int`
- **Default**: `1`
- **Description**: Inter-layer model parallelism. Splits transformer layers across GPU ranks.

### virtual_pipeline_model_parallel_size
- **Type**: `Optional[int]`
- **Default**: `None`
- **Description**: Number of virtual pipeline stages per PP rank for interleaved pipeline parallelism. Reduces pipeline bubble by interleaving model chunks. See [Efficient Large-Scale Language Model Training on GPU Clusters](https://arxiv.org/pdf/2104.04473.pdf).

### sequence_parallel
- **Type**: `bool`
- **Default**: `False`
- **Description**: Makes tensor parallelism more memory efficient by parallelizing LayerNorm and dropout along the sequence dimension. Recommended for 20B+ models. Requires `tensor_model_parallel_size > 1`.

### context_parallel_size
- **Type**: `int`
- **Default**: `1`
- **Description**: Splits network input along sequence dimension across GPU ranks for long-context training.

### hierarchical_context_parallel_sizes
- **Type**: `Optional[list[int]]`
- **Default**: `None`
- **Description**: Degrees of hierarchical context parallelism. For `a2a+p2p` CP, the first value is the A2A group size, the second is the P2P group size.

### max_seqlen_per_dp_cp_rank
- **Type**: `Optional[int]`
- **Default**: `None`
- **Description**: Maximum sequence length per DP×CP rank. Used with hybrid context parallelism to balance workload.

### hybrid_context_parallel
- **Type**: `bool`
- **Default**: `False`
- **Description**: Enables hybrid context parallel for variable-length packed samples. Set `max_seqlen_per_dp_cp_rank` when using this.

### expert_model_parallel_size
- **Type**: `int`
- **Default**: `1`
- **Description**: Distributes MoE experts across sub data parallel dimension.

### expert_tensor_parallel_size
- **Type**: `Optional[int]`
- **Default**: `None` (inherits `tensor_model_parallel_size`)
- **Description**: Intra-layer tensor parallelism for expert layers. Defaults to the global TP size.

## Initialization Parameters

### perform_initialization
- **Type**: `bool`
- **Default**: `True`
- **Description**: Controls weight initialization. Set to False when loading from checkpoint.

### use_cpu_initialization
- **Type**: `bool`
- **Default**: `False`
- **Description**: Initialize weights on CPU (same across TP ranks) then transfer to GPU. GPU initialization is faster but differs across TP ranks.

## Training Parameters

### fp16
- **Type**: `bool`
- **Default**: `False`
- **Description**: Enable fp16 mixed precision training.

### bf16
- **Type**: `bool`
- **Default**: `False`
- **Description**: Enable bf16 mixed precision training. Mutually exclusive with fp16.

### params_dtype
- **Type**: `torch.dtype`
- **Default**: `torch.float32`
- **Description**: Data type used when initializing weights.

### timers
- **Type**: `Optional[Callable]`
- **Default**: `None`
- **Description**: Timer object for timing functions. See `megatron.core.timers.Timers`.

### finalize_model_grads_func
- **Type**: `Optional[Callable]`
- **Default**: `None`
- **Description**: Function to finalize gradients across DP, PP, and SP dimensions.

### grad_scale_func
- **Type**: `Optional[Callable]`
- **Default**: `None`
- **Description**: Function for loss scaling (e.g., for fp16 training). Takes loss, returns scaled loss.

### no_sync_func
- **Type**: `Optional[Callable]`
- **Default**: `None`
- **Description**: Context manager to suppress async data-parallel communication. Default uses `DistributedDataParallel.no_sync`.

### grad_sync_func
- **Type**: `Optional[Callable]`
- **Default**: `None`
- **Description**: Function to launch async gradient reductions. Takes an iterable of parameters.

### param_sync_func
- **Type**: `Optional[Callable]`
- **Default**: `None`
- **Description**: Function to launch async parameter synchronizations. Takes an iterable of parameters.

### deterministic_mode
- **Type**: `bool`
- **Default**: `False`
- **Description**: Use deterministic execution for debugging/testing. Usually slower.

### enable_autocast
- **Type**: `bool`
- **Default**: `False`
- **Description**: Run forward step inside `torch.autocast` context.

### autocast_dtype
- **Type**: `Optional[torch.dtype]`
- **Default**: `None` (uses `pipeline_dtype`)
- **Description**: dtype for `torch.amp.autocast`.

### num_microbatches_with_partial_activation_checkpoints
- **Type**: `Optional[int]`
- **Default**: `None`
- **Description**: Number of microbatches with partial activation checkpointing. The remaining microbatches recompute all layers.

## Optimization Parameters

### gradient_accumulation_fusion
- **Type**: `bool`
- **Default**: `False`
- **Description**: Fuses weight gradient accumulation into GEMMs. Requires custom CUDA extension from APEX with `--cpp_ext` and `--cuda_ext`. Requires CUDA >= 11.

### use_te_rng_tracker
- **Type**: `bool`
- **Default**: `False`
- **Description**: Use TransformerEngine RNG state tracker. Required for CUDA graphs support.

### tp_comm_overlap
- **Type**: `bool`
- **Default**: `False`
- **Description**: Overlap Linear layer execution with TP communication (AllGather/ReduceScatter).

### tp_comm_bulk_wgrad
- **Type**: `bool`
- **Default**: `True`
- **Description**: All-Gather overlap with Bprop activation gradient GEMM. Only effective with `tp_comm_overlap=True`.

### tp_comm_bulk_dgrad
- **Type**: `bool`
- **Default**: `True`
- **Description**: Reduce-Scatter overlap with Bprop weight gradient GEMM. Only effective with `tp_comm_overlap=True`.

### tp_comm_overlap_ag
- **Type**: `bool`
- **Default**: `True`
- **Description**: All-Gather overlap with GEMM via pipelining. Only effective with `tp_comm_overlap=True`.

### tp_comm_overlap_rs
- **Type**: `bool`
- **Default**: `True`
- **Description**: Reduce-Scatter overlap with GEMM via pipelining. Only effective with `tp_comm_overlap=True`.

### tp_comm_overlap_rs_dgrad
- **Type**: `bool`
- **Default**: `False`
- **Description**: Reduce-Scatter overlap with DGRAD GEMM via pipelining. Only effective with `tp_comm_overlap=True`.

### tp_comm_bootstrap_backend
- **Type**: `Literal['nccl', 'mpi', 'gloo']`
- **Default**: `'nccl'`
- **Description**: Bootstrap backend for TP communications.

### cross_entropy_loss_fusion
- **Type**: `bool`
- **Default**: `False`
- **Description**: Use fused cross entropy implementation.

### cross_entropy_fusion_impl
- **Type**: `Literal['native', 'te']`
- **Default**: `'native'`
- **Description**: `'native'` uses MCore CE loss fusion, `'te'` uses TransformerEngine Parallel CE loss.

### overlap_moe_expert_parallel_comm
- **Type**: `bool`
- **Default**: `False`
- **Description**: Overlap EP All-to-All communications with independent computations in 1F1B pipeline schedule.

### delay_wgrad_compute
- **Type**: `bool`
- **Default**: `False`
- **Description**: Delay weight gradient computation for batch-level communication overlapping.

### overlap_dispatch_backward_with_experts_wgrad
- **Type**: `bool`
- **Default**: `False`
- **Description**: Delay MoE expert wgrad computation. Overlaps with EP A2A. Requires TE >= 2.3.0 with GroupedLinear support.

### ep_overlap_early_attn_memory_release
- **Type**: `bool`
- **Default**: `False`
- **Description**: Reorder attention backward to execute before MLP forward during EP overlap, releasing memory earlier.

## Pipeline Parallel Parameters

### pipeline_dtype
- **Type**: `torch.dtype`
- **Default**: `None`
- **Description**: dtype for P2P communication between pipeline stages. Must be set when PP > 1.

### variable_seq_lengths
- **Type**: `bool`
- **Default**: `False`
- **Description**: Support variable sequence lengths across microbatches in PP. Adds communication overhead.

### overlap_p2p_comm
- **Type**: `bool`
- **Default**: `False`
- **Description**: Overlap P2P communication with computation in PP. Must be False if `batch_p2p_comm=True`.

### batch_p2p_comm
- **Type**: `bool`
- **Default**: `True`
- **Description**: Use `batch_isend_irecv` instead of individual `isend/irecv`. Must be False if `overlap_p2p_comm=True`.

### batch_p2p_sync
- **Type**: `bool`
- **Default**: `True`
- **Description**: Do `cuda.device.synchronize` after batch P2P. Workaround for older PyTorch bugs.

### use_ring_exchange_p2p
- **Type**: `bool`
- **Default**: `False`
- **Description**: Use custom `ring_exchange` kernel. Requires custom-built PyTorch.

### deallocate_pipeline_outputs
- **Type**: `bool`
- **Default**: `False`
- **Description**: Deallocate output tensors after sending to next pipeline stage. Saves memory.

### defer_embedding_wgrad_compute
- **Type**: `bool`
- **Default**: `False`
- **Description**: Defer embedding weight gradient GEMMs during pipeline flush. Hides flush latency. Requires PP > 1 and `gradient_accumulation_fusion=True`.

### wgrad_deferral_limit
- **Type**: `int`
- **Default**: `0` (all microbatches deferred)
- **Description**: Number of microbatches for which to defer embedding wgrad. Only valid with `defer_embedding_wgrad_compute=True`.

### overlap_p2p_comm_warmup_flush
- **Type**: `bool`
- **Default**: `False`
- **Description**: Overlap communication and computation in warmup/flush phase. Only valid with `overlap_p2p_comm=True` and `batch_p2p_comm=False`.

### microbatch_group_size_per_vp_stage
- **Type**: `Optional[int]`
- **Default**: `None` (defaults to `pipeline_model_parallel_size`)
- **Description**: Number of microbatches executed per virtual stage at a time. Controls depth-first vs breadth-first scheduling.

### mtp_standalone
- **Type**: `bool`
- **Default**: `False`
- **Description**: Automatically set based on pipeline layout. True if MTP is in a separate virtual pipeline stage.

## CPU Offloading Parameters

### cpu_offloading
- **Type**: `bool`
- **Default**: `False`
- **Description**: Offload activations to CPU asynchronously.

### cpu_offloading_num_layers
- **Type**: `int`
- **Default**: `0`
- **Description**: Number of transformer layers for which to offload activations.

### cpu_offloading_activations
- **Type**: `bool`
- **Default**: `True`
- **Description**: Offload activations to CPU (when `cpu_offloading=True`).

### cpu_offloading_weights
- **Type**: `bool`
- **Default**: `False`
- **Description**: Offload weights to CPU (when `cpu_offloading=True`).

### cpu_offloading_double_buffering
- **Type**: `bool`
- **Default**: `False`
- **Description**: Enable double buffering across layers when reloading activations from CPU.

### cpu_offloading_retain_pinned_cpu_buffers
- **Type**: `bool`
- **Default**: `False`
- **Description**: Retain pinned CPU buffers for reuse. Useful for CUDA graphs.

## Timing Parameters

### barrier_with_L1_time
- **Type**: `bool`
- **Default**: `True`
- **Description**: Call barrier with level 1 time measurements. Can cause hangs if not all ranks call the same timer.

## Post-Init Validation

The `__post_init__` method performs extensive validation:

1. `sequence_parallel=True` requires `tensor_model_parallel_size > 1`
2. `expert_tensor_parallel_size` defaults to `tensor_model_parallel_size`
3. PP > 1 requires `pipeline_dtype` to be set
4. `defer_embedding_wgrad_compute` requires PP > 1 and `gradient_accumulation_fusion`
5. EP + TP without sequence parallelism generates a warning
6. `microbatch_group_size_per_vp_stage` defaults to `pipeline_model_parallel_size`
7. `overlap_p2p_comm_warmup_flush` requires `overlap_p2p_comm` and not `batch_p2p_comm`
