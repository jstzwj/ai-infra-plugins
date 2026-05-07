# Chapter 24: Activation Checkpointing and Recomputation

## Source Files
- `megatron/core/transformer/transformer_config.py` - Config fields for recompute
- `megatron/core/transformer/transformer_layer.py` - Layer-level recompute hooks
- `megatron/core/transformer/transformer_block.py` - Block-level recompute orchestration
- `megatron/core/transformer/enums.py` - Recompute method enums

## Overview

Activation checkpointing (also called gradient checkpointing or activation recomputation) is a memory optimization technique that trades computation for memory. Instead of storing all intermediate activations for the backward pass, selected activations are discarded during forward and recomputed during backward.

This is critical for training large models where activation memory can exceed model parameter memory.

## Recompute Granularities

### Full Recomputation

Recomputes the entire transformer layer during backward:

```bash
--recompute-granularity full
```

All intermediate activations within a layer are discarded after the forward pass and recomputed from the layer input during the backward pass. This provides the maximum memory savings but requires re-running the entire forward computation for each layer.

**Memory savings:** Approximately 70-80% of activation memory per checkpointed layer.

**Trade-off:** ~33% increase in compute time (forward pass runs twice for checkpointed layers).

### Selective Recomputation

Discards only the activations that are cheap to recompute (attention and MLP intermediate results) while keeping the expensive-to-compute values:

```bash
--recompute-granularity selective
```

Selective recompute targets:
- MLP intermediate activations (after the first linear, before the second)
- Attention score matrices (Q*K^T results)
- Attention output projections

Values that are kept:
- Layer norm outputs (needed for residual connections)
- Linear layer weight outputs that are needed for both attention and MLP paths

**Memory savings:** Approximately 40-60% of activation memory per layer.

**Trade-off:** Minimal compute overhead (~5-10%) since only cheap operations are recomputed.

### No Recomputation

All activations are stored for backward:

```bash
--recompute-granularity None
```

Default behavior. Maximum memory usage, minimum compute.

## Recompute Methods

### Uniform Method

Applies recompute uniformly to all eligible layers:

```bash
--recompute-granularity selective
--recompute-method uniform
```

Every transformer layer within the recompute scope is checkpointed. This is the simplest method and provides consistent memory savings across all layers.

### Block Method

Applies recompute to groups (blocks) of layers:

```bash
--recompute-granularity selective
--recompute-method block
--recompute-num-layers 4    # Checkpoint every Nth group
```

With block recompute:
- Layers are divided into blocks of `recompute_num_layers` size
- Only the input to each block is stored; the entire block is recomputed during backward
- Within a block, intermediate activations may or may not be stored depending on the granularity

**Example with 24 layers and `recompute_num_layers=4`:**
- Layers 0-3: Block 0 (store input, recompute layers during backward)
- Layers 4-7: Block 1 (store input, recompute layers during backward)
- Layers 8-11: Block 2 (store input, recompute layers during backward)
- Layers 12-15: Block 3 (store input, recompute layers during backward)
- Layers 16-19: Block 4 (store input, recompute layers during backward)
- Layers 20-23: Block 5 (store input, recompute layers during backward)

## Recomputable Modules

The following module types support activation recompute:

| Module | Selective Recompute Details |
|--------|---------------------------|
| `core_attn` (core attention) | Attention scores and output projection recomputed |
| `mlp` (MLP block) | Intermediate activation after fc1 recomputed |
| `moe` (Mixture of Experts) | Expert MLP intermediates recomputed |
| `layer_norm` / `rms_norm` | Generally kept (cheap to store, expensive to recompute) |
| `q_layernorm` / `k_layernorm` | Kept for attention computation |

### Custom Recompute Modules

The recompute configuration can be customized per-module using the TransformerConfig:

```python
TransformerConfig(
    recompute_granularity='selective',
    recompute_method='uniform',
    recompute_num_layers=1,
    # Per-module recompute control:
    recompute_core_attention=True,
    recompute_mlp=True,
)
```

## Pipeline Parallel Integration

Activation checkpointing interacts with pipeline parallelism in important ways:

### With Pipeline Parallelism

In pipeline parallelism, each pipeline stage independently manages its own activation checkpointing. Activations that need to be communicated between stages (at pipeline stage boundaries) are always preserved.

```bash
--pipeline-model-parallel-size 4
--recompute-granularity selective
--recompute-method uniform
```

Each PP stage checkpoints its own layers independently. The communication buffers between stages are never checkpointed.

### With Virtual Pipeline Parallelism (VPP)

With interleaved pipeline scheduling (VPP), each model chunk manages recompute independently:

```bash
--pipeline-model-parallel-size 4
--num-layers-in-virtual-pipeline-stage 3
--recompute-granularity selective
```

The first and last layers of each VP chunk are handled specially to ensure correct gradient flow at chunk boundaries.

### With Context Parallelism

When using context parallelism, activation checkpointing preserves the necessary context for sequence splitting:

- Context parallelism splits the sequence across CP ranks
- Activation recompute must preserve the attention context needed for cross-rank attention
- The `hybrid_context_parallel` configuration handles the interaction

## Fine-Grained Activation Offloading

Beyond traditional in-GPU activation checkpointing, Megatron supports offloading activations to CPU memory:

### CPU Offloading of Activations

During forward, selected activations can be moved to CPU memory and brought back during backward:

```bash
--activation-offloading CPU
```

This is beneficial when:
- GPU memory is extremely constrained
- The model cannot fit even with selective recompute
- The CPU-GPU bandwidth is sufficient (NVLink-connected systems)

### How CPU Offloading Works

1. During forward pass:
   - Compute activations as normal
   - Copy selected activations to pinned CPU memory
   - Release GPU memory

2. During backward pass:
   - Copy activations back from CPU to GPU
   - Recompute any non-stored intermediates
   - Continue backward computation

### Offloading Candidates

Typical candidates for CPU offloading:
- Large attention score matrices (S^2 per head)
- MLP intermediate activations (batch * seq_len * 4*hidden)
- Long-sequence activations where the size justifies the transfer cost

## Distributed Checkpointing of Activations

When using distributed checkpointing, activation states are handled consistently:

- Activation checkpointing is a runtime optimization, not persisted in checkpoints
- The recompute configuration is stored in the checkpoint metadata
- Upon checkpoint reload, the same recompute configuration is applied

## Memory Savings Analysis

For a transformer layer with hidden_size H, sequence_length S, batch_size B, and MLP expansion factor 4:

| Component | Full Storage | Full Recompute | Selective Recompute |
|-----------|-------------|----------------|-------------------|
| Attention QKV | 3*B*S*H | 0 | 3*B*S*H |
| Attention scores | B*S^2*heads | B*S^2*heads | 0 |
| Attention output | B*S*H | 0 | B*S*H |
| MLP fc1 output | B*S*4H | 0 | 0 |
| MLP fc2 output | B*S*H | 0 | B*S*H |
| **Total per layer** | ~B*S*(8H + S*heads) | ~B*S^2*heads | ~B*S*(4H) |

Selective recompute saves the MLP intermediate (4H per token) and attention scores (S*heads per token), which are the largest tensors, while keeping the relatively small QKV and attention output tensors.

## Configuration Examples

### Maximum Memory Savings
```bash
--recompute-granularity full
--recompute-method uniform
```
All layers fully recomputed. Maximum memory savings, ~33% compute overhead.

### Balanced Memory-Compute Trade-off
```bash
--recompute-granularity selective
--recompute-method uniform
```
All layers selectively recomputed. Good memory savings, ~5-10% compute overhead.

### Block-wise Recompute
```bash
--recompute-granularity selective
--recompute-method block
--recompute-num-layers 4
```
Groups of 4 layers checkpointed together. Balances memory savings with recompute granularity.

### No Recompute (Maximum Throughput)
```bash
--recompute-granularity None
```
All activations stored. Maximum throughput, maximum memory usage.

### With FP8 and Distributed Optimizer
```bash
--recompute-granularity selective
--recompute-method uniform
--fp8 hybrid --fp8-recipe delayed
--use-distributed-optimizer
```
Combining selective recompute with FP8 and distributed optimizer provides the best overall memory savings.

### With CPU Activation Offloading
```bash
--recompute-granularity selective
--recompute-method block
--recompute-num-layers 8
--activation-offloading CPU
```
Block recompute with CPU offloading for extreme memory constraints.

## Interaction with CUDA Graphs

When using CUDA graphs with activation checkpointing:

```bash
--recompute-granularity selective
--cuda-graph-impl local
```

The recomputation is captured as part of the backward graph. The graph captures both the "normal" backward operations and the recompute operations, replaying them together.

For RL training with CUDA graphs:
```bash
--rl-training-cuda-graphs
--recompute-granularity selective
```

Note: When using `--full-cuda-graph`, activation recompute may not be supported since the entire forward-backward sequence is captured as a single graph, and recompute requires running forward operations during backward.

## Best Practices

1. **Start with selective recompute:** It provides the best trade-off for most workloads.

2. **Use full recompute only when memory-bound:** The 33% compute overhead is significant at scale.

3. **Combine with distributed optimizer:** The distributed optimizer saves optimizer state memory while recompute saves activation memory.

4. **Use block recompute for very deep models:** When there are many layers, block recompute reduces the overhead of entering/exiting checkpoint regions.

5. **Profile memory first:** Use `--report-theoretical-memory` to understand where memory is allocated before choosing a recompute strategy.

6. **Consider FP8:** FP8 reduces both parameter memory and activation memory, which may reduce the need for aggressive recompute.
