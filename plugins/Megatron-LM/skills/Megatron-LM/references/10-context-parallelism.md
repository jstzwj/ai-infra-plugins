# Context Parallelism (CP) Reference

## Overview

Context Parallelism (CP) is a sequence-dimension parallelization scheme in Megatron-LM that
partitions network inputs and all activations along the sequence dimension across multiple GPUs.
Unlike prior Sequence Parallelism (SP), which only splits the sequence of Dropout and LayerNorm
activations, CP partitions every activation along the sequence dimension throughout the entire
model.

### When to Use Context Parallelism

Use CP when training with long sequences (8K+ tokens) where activation memory causes OOM errors.
CP scales down both compute and communication per GPU by the CP degree, reducing activation
memory footprint proportionally. It is preferable over increasing Tensor Parallelism (TP) because
TP can make compute in layers such as Linear too short to hide communication latency.

Key scenarios:
- Sequence lengths >= 8K tokens
- Models hitting OOM on long contexts
- When full activation recomputation overhead (~30%) is unacceptable
- When scaling TP further degrades compute-communication overlap

### How CP Works

All modules except attention (Linear, LayerNorm, etc.) operate on their sequence chunk without
changes because they have no inter-token operations. For attention, each token's Query (Q) must
combine with the Key/Value (KV) of all tokens in the sequence. CP requires additional
communication to collect the full KV sequence across GPUs.

Each GPU stores only its KV chunk during forward and gathers KV again during backward. KV
communication happens between a GPU and its counterparts in other TP groups. With MQA or GQA,
communication volume is reduced because fewer attention heads are used for KV.

## Communication Types

The `cp_comm_type` configuration controls how CP communication is performed. It can be set as a
single string (all layers share the same type) or as a list (per-layer communication type).

### p2p (Point-to-Point Ring)

Default communication type. Exchanges KV chunks with P2P communications in a ring topology.

```bash
--cp-comm-type p2p
```

Characteristics:
- P2P is asynchronous and can be overlapped with attention compute
- Uses cuDNN flash attention kernels
- Avoids extra work from lower-triangle causal masking
- Keeps load balanced across GPUs
- Best for: general long-sequence training where compute-communication overlap is desired

### all_gather

All-gather to get the full sequence of KV before attention computation.

```bash
--cp-comm-type all_gather
```

Characteristics:
- The all-gather is synchronous and cannot be overlapped with computation
- Simpler communication pattern than p2p
- Best for: short sequences where communication latency is small, or when
  overlap is not critical

### a2a (All-to-All / DeepSpeed Ulysses Style)

Scatters attention heads across the CP group and gathers to get the full sequence of QKV.

```bash
--cp-comm-type a2a
```

Characteristics:
- Similar to DeepSpeed Ulysses approach
- Each GPU holds a subset of attention heads with the full sequence
- Best for: models with many attention heads where head distribution is efficient

### a2a+p2p (Hierarchical)

A hierarchical implementation that uses A2A communications in low-level CP groups (via NVLink)
and P2P communications in high-level CP groups (via IBLink).

```bash
--cp-comm-type a2a+p2p
```

Characteristics:
- Designed for multi-node setups with different interconnect speeds
- Combines intra-node (NVLink) and inter-node (IBLink) communication optimally
- Best for: large-scale multi-node training with hierarchical network topology

## Ring Attention Mechanism

Megatron-LM's CP is based on the Ring Attention concept (https://arxiv.org/abs/2310.01889) but
targets higher performance through two key optimizations:

1. Uses current open-source and cuDNN flash attention kernels directly
2. Avoids extra work from lower-triangle causal masking while keeping load balanced across GPUs

### Ring Attention Forward Pass

In a ring of CP GPUs, the forward pass works as follows:

```
Step 1: Each GPU processes its local sequence chunk
Step 2: KV chunks are exchanged in a ring topology
  - GPU0 sends its KV to GPU1, receives KV from GPU_N-1
  - Each GPU accumulates attention results for all KV chunks
Step 3: Output is reduce-scattered back to local chunks
```

For bidirectional attention, both halves of the ring operate simultaneously for balanced load.
For causal (unidirectional) attention, Megatron avoids the naive lower-triangle masking approach
by reordering chunks so each GPU performs equal work.

## Hierarchical Context Parallelism

For multi-node training, Megatron-LM supports hierarchical CP that creates separate NVLink and
IB groups.

### Configuration

```bash
# Example: CP=8 with 2-level hierarchy [4 NVLink, 2 IB]
--context-parallel-size 8
--hierarchical-context-parallel-sizes 4 2
```

The product of `hierarchical_context_parallel_sizes` must equal `context_parallel_size`.

### How It Works

1. **Inner group (NVLink)**: High-bandwidth intra-node communication for frequent KV exchanges
2. **Outer group (IB)**: Lower-bandwidth inter-node communication for less frequent exchanges

```python
# From parallel_state.py initialization
if hierarchical_context_parallel_sizes:
    assert np.prod(hierarchical_context_parallel_sizes) == context_parallel_size
    # Creates separate groups for NVLink and IB communication
```

Access hierarchical CP groups:

```python
from megatron.core import parallel_state

# Get hierarchical CP groups (NVLink inner, IB outer)
inner_cp_groups = parallel_state.get_hierarchical_context_parallel_groups()
```

## Hybrid Context Parallelism

Hybrid CP combines Context Parallelism and Data Parallelism into a single unified group for
variable-length sequence training.

### Configuration

```bash
--hybrid-context-parallel
--context-parallel-size 8
```

When enabled, the DPxCP group is treated as a single hybrid group. This is useful when:
- Sequences have variable lengths
- You want to dynamically adjust the effective CP/DP split based on sequence length
- Different micro-batches may benefit from different CP configurations

### Process Group Access

```python
from megatron.core import parallel_state

# Get hybrid data-context parallel groups
hybrid_groups = parallel_state.get_hybrid_data_context_parallel_groups()

# Get data parallel group with context parallelism
dp_cp_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
```

## Integration with Other Parallelism Dimensions

CP works seamlessly with all other parallelism dimensions. The total GPU count is:

```
Total GPUs = TP * PP * CP * DP
```

### CP + Tensor Parallelism (TP)

- CP and TP are orthogonal: CP splits sequence, TP splits heads/hidden
- KV communication happens between GPUs in different TP groups
- The TP-CP combined group enables efficient cross-group communication

```bash
--tensor-model-parallel-size 4
--context-parallel-size 2
--sequence-parallel  # Required for optimal CP+TP integration
```

### CP + Pipeline Parallelism (PP)

- CP operates within each pipeline stage
- No additional complexity when combining CP and PP

```bash
--pipeline-model-parallel-size 4
--context-parallel-size 2
```

### CP + Data Parallelism (DP)

- CP reduces activation memory, potentially allowing larger DP
- Hybrid CP can merge CP and DP groups for dynamic allocation

```bash
--context-parallel-size 2
# DP is automatically computed: DP = Total / (TP * PP * CP)
```

### CP + Expert Parallelism (MoE)

- CP is supported in MoE models (MoE Parallel Folding)
- CP applies to attention layers; MoE layers use their own EP configuration
- Attention uses TP x CP x DP x PP; MoE uses ETP x EP x EDP x PP

## Performance Analysis

### Memory Savings

Activation memory scales approximately linearly with sequence length. CP divides activation
memory by the CP degree:

| Configuration | Relative Activation Memory | Notes |
|--------------|---------------------------|-------|
| TP8, CP1     | 1x (baseline)             | Full recompute needed for long seq |
| TP4, CP2     | ~0.5x                     | Better compute-communication balance |
| TP2, CP4     | ~0.25x                    | Excellent memory savings |
| TP1, CP8     | ~0.125x                   | Minimal TP communication |

Weight and optimizer memory are NOT reduced by CP alone (weights are replicated).
Use distributed optimizer (`--use-distributed-optimizer`) for optimizer state sharding.

### Communication Overhead

| Communication Type | Volume (per attention layer) | Overlap Support |
|-------------------|------------------------------|-----------------|
| p2p               | 2 * KV_size / CP_degree      | Yes (async)     |
| all_gather        | KV_size * (CP-1)/CP          | No (sync)       |
| a2a               | Head_size * seq_len          | Limited         |
| a2a+p2p           | Hierarchical (mixed)         | Partial         |

With MQA/GQA, KV communication is reduced because fewer KV heads are used:
```bash
--group-query-attention
--num-query-groups 8  # Reduces KV heads, lowering CP communication
```

### Benchmark Results (175B GPT)

From NVIDIA benchmarks, TP+CP combinations significantly outperform full recomputation:

| Configuration | Speedup vs Full Recomputation |
|--------------|-------------------------------|
| TP4 CP1 (full recompute) | 1.0x baseline |
| TP4 CP2 | ~1.2x |
| TP2 CP4 | ~1.4x |
| TP1 CP8 | ~1.3x (communication bound) |

The optimal configuration balances compute against communication. Very high CP with very low TP
can hit the communication overlap limit.

## Configuration Examples

### 8K Sequence Length (GPT Training)

```bash
# Balanced TP+CP for 8K sequences
--tensor-model-parallel-size 4
--context-parallel-size 2
--sequence-parallel
--cp-comm-type p2p
```

### 32K Sequence Length (Long Context Training)

```bash
# Higher CP for 32K sequences
--tensor-model-parallel-size 2
--context-parallel-size 4
--sequence-parallel
--cp-comm-type p2p
```

### 128K Sequence Length (Ultra-Long Context)

```bash
# Maximum CP for 128K sequences
--tensor-model-parallel-size 1
--context-parallel-size 8
--sequence-parallel
--cp-comm-type p2p
```

### Multi-Node 32K with Hierarchical CP

```bash
# 8 GPUs per node, 4 nodes
--tensor-model-parallel-size 2
--context-parallel-size 8
--hierarchical-context-parallel-sizes 4 2
--sequence-parallel
--cp-comm-type a2a+p2p
```

### Variable-Length Sequences with Hybrid CP

```bash
--tensor-model-parallel-size 2
--context-parallel-size 8
--hybrid-context-parallel
--sequence-parallel
```

### MoE Model with CP

```bash
# DeepSeek-V3 style with CP
--tensor-model-parallel-size 1
--pipeline-model-parallel-size 4
--context-parallel-size 4
--num-experts 256
--expert-model-parallel-size 64
--sequence-parallel
--cp-comm-type p2p
```

## Key Configuration Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--context-parallel-size` | Number of GPUs to split sequence across | 1 |
| `--cp-comm-type` | CP communication type: p2p, all_gather, a2a, a2a+p2p | None (auto) |
| `--hierarchical-context-parallel-sizes` | Hierarchical CP group sizes (e.g., `4 2`) | None |
| `--hybrid-context-parallel` | Enable hybrid CP+DP for variable-length | False |
| `--sequence-parallel` | Required for CP to work optimally | False |

## Requirements

- Megatron Core >= 0.5.0
- Transformer Engine >= 1.1
- Works with MHA, MQA, and GQA attention variants
- Supports both unidirectional (causal) and bidirectional masking
- Compatible with FP8 training, selective recomputation, and CUDA graphs

## API Reference

### Process Group Functions

```python
from megatron.core import parallel_state

# Get CP world size and rank
cp_size = parallel_state.get_context_parallel_world_size()
cp_rank = parallel_state.get_context_parallel_rank()

# Get the CP process group
cp_group = parallel_state.get_context_parallel_group()

# Get combined TP+CP group
tp_cp_group = parallel_state.get_tensor_and_context_parallel_group()
tp_cp_size = parallel_state.get_tensor_and_context_parallel_world_size()

# Get DP group that includes CP dimension
dp_cp_group = parallel_state.get_data_parallel_group(with_context_parallel=True)

# Get hierarchical CP groups (returns inner and outer groups)
hier_groups = parallel_state.get_hierarchical_context_parallel_groups()

# Get global ranks in CP group
cp_ranks = parallel_state.get_context_parallel_global_ranks()
```

### TransformerConfig CP Fields

```python
from megatron.core.transformer import TransformerConfig

config = TransformerConfig(
    # CP basic configuration
    context_parallel_size=4,           # Number of CP partitions
    cp_comm_type="p2p",               # Communication type

    # CP with attention
    # MQA/GQA reduces CP communication volume
    num_query_groups=8,                # Fewer KV heads = less CP comm
)
```

## Troubleshooting

### OOM Despite CP

1. Verify CP is actually active: check `context_parallel_size > 1`
2. Enable distributed optimizer: `--use-distributed-optimizer`
3. Consider selective recomputation: `--recompute-granularity selective`
4. Check if sequence parallelism is enabled: `--sequence-parallel`

### Slow Training with CP

1. Ensure TP x CP is not too small (causes short compute)
2. Use `p2p` communication type for async overlap
3. Reduce TP and increase CP if compute-communication overlap is poor
4. For multi-node, use hierarchical CP with `a2a+p2p`

### CP Communication Errors

1. Verify Transformer Engine version >= 1.1
2. Verify Megatron Core version >= 0.5.0
3. Check that `--sequence-parallel` is enabled
4. For hierarchical CP, verify sizes multiply to CP size
