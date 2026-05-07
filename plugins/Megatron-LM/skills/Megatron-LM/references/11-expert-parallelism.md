# Expert Parallelism (EP) and Mixture-of-Experts Reference

## Overview

Expert Parallelism distributes MoE expert parameters across GPUs, where each GPU hosts a subset
of experts. Tokens are routed to the appropriate expert's GPU, processed, and sent back.
Megatron-LM provides a production-ready MoE stack with advanced routing, token dispatching,
load balancing, and performance optimizations.

The total GPU parallelism for an MoE model is:

```
Total GPUs = TP * PP * EP * DP (or ETP * EP * EDP with MoE Parallel Folding)
```

## Token Dispatchers

Token dispatchers handle the movement of tokens between GPUs based on routing decisions.
Megatron-LM provides four dispatcher types controlled by `--moe-token-dispatcher-type`.

### AllGather Dispatcher

Gathers all tokens from the TP*EP domain to each GPU, then locally selects tokens for local
experts. No inter-GPU token movement beyond the initial gather.

```bash
--moe-token-dispatcher-type allgather
```

**Workflow:**
1. AllGather across TP*EP ranks: `[S/TP*B, H] -> [S*B*EP, H]`
2. Local token permutation for local experts
3. Expert computation
4. Unpermute tokens
5. Reduce-scatter across TP*EP ranks

**Best for:** TP-only setups, small EP, or large Top-K routing.

### AlltoAll Dispatcher

NCCL-based All-to-All communication for token exchange across EP ranks. The standard dispatcher
for most EP > 1 setups.

```bash
--moe-token-dispatcher-type alltoall
```

**Workflow:**
1. Preprocess: Calculate metadata for communication and permutation
2. Permutation 1: Permute tokens for AlltoAll input
3. Token dispatch: A2A across EP ranks
4. Dispatch postprocess: AllGather(TP) then sort by local expert
5. Expert computation
6. Combine preprocess: Unsort and Reduce-scatter(TP)
7. Token combine: A2A across EP ranks
8. Combine postprocess: Unpermute tokens

**Best for:** Standard EP > 1 setups, most production workloads.

### Flex Dispatcher with DeepEP Backend

Uses DeepEP's fused dispatch/combine kernels that combine permutation and AlltoAll communication
into a single optimized operation. Removes redundant tokens during cross-node communication.

```bash
--moe-token-dispatcher-type flex --moe-flex-dispatcher-backend deepep
```

**Key features:**
- Fused permutation + communication reduces memory bandwidth
- Better overlap between computation and communication
- Supports async communication completion
- Only supports float32 router probabilities

**Best for:** Cross-node EP, fine-grained MoE models (DeepSeek-V3 style).

### Flex Dispatcher with HybridEP Backend

NVIDIA's optimized dispatcher using TMA (Tensor Memory Accelerator) and IBGDA (InfiniBand
GPUDirect Async). Supports native MNNVL (Multi-Node NVLink).

```bash
--moe-token-dispatcher-type flex --moe-flex-dispatcher-backend hybridep
```

**Key features:**
- Fewer SMs required for communication
- Native MNNVL support for GB200 NVL72
- TMA-based data movement
- Supports permute fusion into the communication kernel

**Best for:** GB200 NVL72, multi-node NVLink systems, H100/B200 clusters.

### Dispatcher Comparison

| Dispatcher | Communication | Cross-Node | Fused Ops | Best For |
|-----------|---------------|------------|-----------|----------|
| allgather | AllGather + ReduceScatter | Inefficient | No | TP-only, small EP |
| alltoall | AlltoAll (NCCL) | Yes | No | Standard EP setups |
| flex+deepep | Fused A2A | Optimized | Yes | DeepSeek-V3, cross-node EP |
| flex+hybridep | TMA + IBGDA | MNNVL | Yes | GB200, B200 clusters |

## Router

The router determines which expert(s) handle each token. Megatron-LM implements a `TopKRouter`
that scores every token against all experts and selects the top-K experts.

### Top-K Routing

```bash
--moe-router-topk 2  # Route each token to top 2 experts
```

Each token is scored against all experts via a gating linear layer, then the top-K experts are
selected. The output is a probability distribution (`probs`) and a routing map (`routing_map`).

### Score Functions

Control how logits are converted to routing probabilities:

| Score Function | Description | Config |
|---------------|-------------|--------|
| `softmax` | Standard softmax over expert logits | `--moe-router-score-function softmax` |
| `sigmoid` | Independent sigmoid per expert | `--moe-router-score-function sigmoid` |
| `sqrtsoftplus` | Square root softplus (internal) | Used in specific configurations |

**Recommendation:** Use `sigmoid` for fine-grained MoE models and `softmax` for standard MoE.

```bash
--moe-router-score-function sigmoid
```

### Pre-Softmax

Apply softmax before top-K selection instead of after:

```bash
--moe-router-pre-softmax
```

This changes the routing distribution and can improve load balancing in some cases.

### Group-Limited Routing (DeepSeek-V3 Style)

Select top-K expert groups, then route to experts within selected groups:

```bash
--moe-router-num-groups 8       # Number of expert groups
--moe-router-group-topk 4       # Number of groups to select
```

This is the routing strategy used by DeepSeek-V3. It first selects the top groups based on
accumulated scores, then picks top-K experts within those groups. This reduces routing noise
and improves expert specialization.

### Router Precision

```bash
--moe-router-dtype fp32  # or fp64
```

**Critical:** Router logits should remain in FP32 or FP64 rather than BF16. At high expert counts,
FP32 precision yields better accuracy because expert outputs are multiplied by router scores and
accumulated. With many experts, BF16 precision in the router can cause significant accuracy
degradation.

### Router Fusion

Fuse router projection, top-K selection, softmax, and auxiliary loss into fewer GPU kernels:

```bash
--moe-router-fusion
```

This reduces kernel launch overhead and improves small-operation efficiency.

### Expert Bias (Aux-Loss-Free Load Balancing)

Dynamic per-expert bias that updates without auxiliary loss:

```bash
--moe-router-enable-expert-bias
--moe-router-bias-update-rate 1e-3
```

Maintains a running count of tokens per expert and adjusts expert bias to encourage load
balancing. This avoids adding any loss term to the training objective.

### Z-Loss

Encourages the router's logits to remain small for training stability:

```bash
--moe-z-loss-coeff 0.01
```

Refer to the ST-MoE paper (https://arxiv.org/pdf/2202.08906.pdf) for details. Only applied
during training with gradients enabled.

### Input Jitter

Add noise to router input for regularization:

```bash
--moe-input-jitter-eps 0.01
```

Refer to https://arxiv.org/abs/2101.03961 for the jittering technique.

## Load Balancing Strategies

Load balancing ensures even expert utilization across training. Without it, a few experts receive
disproportionately many tokens while others are underutilized.

### aux_loss (Micro-Batch Level)

Auxiliary loss for balancing expert usage within a micro-batch:

```bash
--moe-router-load-balancing-type aux_loss
--moe-aux-loss-coeff 1e-2
```

Uses the Switch Transformer load balancing loss:
```
loss = E * sum_i(f_i * P_i)
```
where `f_i` is the fraction of tokens dispatched to expert i and `P_i` is the average router
probability for expert i. The loss is reduced across the TP and CP groups.

### seq_aux_loss (Sequence Level)

Sequence-level auxiliary loss that computes balance per individual sequence:

```bash
--moe-router-load-balancing-type seq_aux_loss
--moe-aux-loss-coeff 1e-2
```

Reshapes the batch dimension into the experts dimension, computing aux loss per sequence. The
result is averaged across the batch. This is useful when different sequences have very different
routing patterns.

### global_aux_loss (Global Batch Level)

Global auxiliary loss that balances expert usage across all ranks and the full global batch:

```bash
--moe-router-load-balancing-type global_aux_loss
--moe-aux-loss-coeff 1e-2
```

Tracks a running average of tokens per expert across training steps, providing smoother load
balancing. The token count is reduced across the TP, DP, and CP groups.

### Combined Load Balancing

Multiple load balancing strategies can be combined:

```bash
--moe-router-load-balancing-type aux_loss seq_aux_loss global_aux_loss
--moe-aux-loss-coeff 1e-2 1e-3 1e-3
```

When using a list, each coefficient corresponds to its respective loss type.

### sinkhorn

Optimal transport formulation for balanced expert assignment:

```bash
--moe-router-load-balancing-type sinkhorn
```

Uses the Sinkhorn algorithm to compute an approximately balanced routing assignment. Note:
Sinkhorn routing is incompatible with auxiliary loss (`--moe-aux-loss-coeff` must be 0).

### none

No load balancing. Tokens are routed purely based on expert affinity:

```bash
--moe-router-load-balancing-type none
```

## Shared Experts

Shared experts process ALL tokens regardless of routing decisions, providing a stable baseline
computation alongside the routed experts.

### Configuration

```bash
--moe-shared-expert-intermediate-size 2048  # FFN hidden size for shared expert
```

### Shared Expert Overlap

Overlap shared expert computation with EP token transfer to hide latency:

```bash
--moe-shared-expert-overlap
```

When enabled, the shared expert forward pass is split into stages that run concurrently with
the routed expert token dispatch:

```
Timeline:
  [Routed Expert Dispatch A2A] [Shared Expert FC1+Act] [Routed Expert Probs A2A]
                               [Shared Expert FC2]      [Shared Expert RS]
  [Routed Expert Combine A2A]  [Shared Expert Output]
```

The shared expert runs on a separate CUDA stream and its operations are interleaved with the
routed expert communication.

### Shared Expert Gate

A learned gate that controls the shared expert's contribution:

```python
# Internal: SharedExpertMLP automatically creates a gate when configured
gate_score = sigmoid(linear(hidden_states, gate_weight))
output = shared_expert_output * gate_score
```

## Expert Parallelism Communication Overlap

### EP A2A Overlap

Overlaps EP All-to-All communication with computation by merging forward-backward passes of
adjacent microbatches:

```bash
--overlap-moe-expert-parallel-comm
--delay-wgrad-compute
```

**Requirements:**
- `expert_model_parallel_size > 1`
- `CUDA_DEVICE_MAX_CONNECTIONS > 1`

This optimization can reduce EP communication overhead from 30-40% of training time to nearly
zero by hiding it behind computation from adjacent microbatches.

### Batch-Level Overlapping

The 1F1B (one forward, one backward) pipeline schedule naturally creates opportunities for
overlapping EP communication:

```
Microbatch 1: FWD dispatch A2A -> [computation] -> FWD combine A2A
Microbatch 2:                FWD dispatch A2A -> [computation] -> FWD combine A2A
Microbatch 1:                                                                  BWD ...
```

## Grouped GEMM

Batch multiple expert GEMM operations into a single kernel call:

```bash
--moe-grouped-gemm
```

Without Grouped GEMM, each expert is a separate GEMM call, causing high kernel launch overhead.
With Grouped GEMM, all local experts are processed in a single batched kernel, significantly
improving GPU utilization.

### TEGroupedMLP

The `TEGroupedMLP` class implements Grouped GEMM using TransformerEngine's GroupedLinear:

```python
# From megatron/core/transformer/moe/experts.py
class TEGroupedMLP(MegatronModule):
    def __init__(self, num_local_experts, config, submodules, pg_collection):
        self.linear_fc1 = submodules.linear_fc1(
            num_local_experts,
            input_size,
            ffn_hidden_size,
            ...
        )
        self.linear_fc2 = submodules.linear_fc2(
            num_local_experts,
            ffn_hidden_size,
            output_size,
            ...
        )

    def forward(self, permuted_local_hidden_states, tokens_per_expert, permuted_probs):
        # FP8 padding if needed
        tokens_per_expert = tokens_per_expert.tolist()
        # Grouped FC1
        fc1_output, bias = self.linear_fc1(permuted_local_hidden_states, tokens_per_expert)
        # Activation function (e.g., SwiGLU)
        act_output = self.bias_act_func(fc1_output, bias, permuted_probs)
        # Grouped FC2
        output, output_bias = self.linear_fc2(act_output, tokens_per_expert)
        return output, output_bias
```

### SequentialMLP

For debugging or when Grouped GEMM is not available, `SequentialMLP` runs experts one at a time:

```bash
# Don't set --moe-grouped-gemm to use SequentialMLP
```

## Capacity Factor and Token Dropping

### Token Capacity

Control how many tokens each expert can process:

```bash
--moe-expert-capacity-factor 1.0   # Experts can handle 1x average load
--moe-pad-expert-input-to-capacity  # Pad inputs to fixed capacity
```

The capacity per expert is calculated as:
```
capacity = num_tokens * topk * capacity_factor / num_experts
```

### Token Drop Policies

When capacity is exceeded, tokens can be dropped:

```bash
--moe-token-drop-policy probs      # Drop lowest-probability tokens first
--moe-token-drop-policy position   # Drop based on position
```

**Recommendation:** For production training, use dropless MoE (do not set capacity factor).
Megatron-LM's token dispatcher handles variable token counts efficiently without dropping.

### FP8 Alignment Padding

Pad the routing map (not tokens) to align dimensions for FP8 GEMM efficiency:

```bash
--moe-router-padding-for-fp8
```

Pads to multiples of 16 or 32 to avoid per-tensor padding overhead during quantized GEMM.

## Integration with FP8 Quantization

### FP8 Training with MoE

```bash
--fp8-format e4m3
--fp8-recipe blockwise
--moe-grouped-gemm
--moe-router-padding-for-fp8
--fp8-param-gather
```

### FP8 Benefits for MoE

| Aspect | Benefit |
|--------|---------|
| Memory | 50% activation reduction (FP8 instead of BF16) |
| Communication | 50% EP dispatch volume (FP8 tokens) |
| Compute | Faster FP8 Tensor Core GEMMs |
| Parameters | 50% parameter all-gather reduction with FP8 primary weights |

### FP8 Recipes

| Recipe | Granularity | Format | Platform | Use Case |
|--------|------------|--------|----------|----------|
| Per-tensor | Whole tensor | E4M3/E5M2 | Hopper+ | Conservative experimentation |
| Blockwise | 1x128 act, 128x128 weight | E4M3 | Hopper | Production-proven (DeepSeek-V3) |
| MXFP8 | 1x32 | E4M3 + E8M0 | Blackwell | Native hardware on GB200 |

### FP8 in Grouped GEMM

The `TEGroupedMLP` class handles FP8 padding and unpadding automatically:

```python
# From experts.py TEGroupedMLP.forward()
if self.config.fp8 or self.config.fp4:
    actual_tokens_per_expert = tokens_per_expert
    permuted_local_hidden_states, tokens_per_expert = self.quantization_padding(
        permuted_local_hidden_states, tokens_per_expert
    )
    permuted_probs, _ = self.quantization_padding(
        permuted_probs.unsqueeze(-1), actual_tokens_per_expert
    )

# After expert computation:
if self.config.fp8 or self.config.fp4:
    output = self.quantization_unpadding(output, actual_tokens_per_expert)
```

## MoE Parallel Folding

MoE Parallel Folding decouples attention and MoE parallelism, allowing independent optimization
of each.

### Traditional Approach Problems

1. EP <= DP constraint limits scalability
2. Same TP/CP for attention and MoE is suboptimal
3. High TP benefits attention but hurts MoE (small per-expert dimensions)

### Folding Solution

| Layer Type | Parallelism Dimensions |
|-----------|----------------------|
| Attention | TP x CP x DP x PP |
| MoE | ETP x EP x EDP x PP |

Key benefits:
- Breaks EP <= DP constraint (8x more expert parallelism possible)
- Reduces minimum GPU requirements
- ETP=1 for MoE gives better GEMM efficiency
- High-bandwidth communication stays in NVLink domain

## Process Groups for MoE

The `ProcessGroupCollection` dataclass manages all process groups needed for MoE:

```python
@dataclass
class ProcessGroupCollection:
    # Attention process groups
    tp: torch.distributed.ProcessGroup          # Tensor parallel
    cp: Optional[ProcessGroup] = None            # Context parallel
    tp_cp: Optional[ProcessGroup] = None         # Combined TP+CP
    tp_dp_cp: Optional[ProcessGroup] = None      # Combined TP+DP+CP

    # Expert process groups
    ep: Optional[ProcessGroup] = None            # Expert parallel
    expt_tp: Optional[ProcessGroup] = None       # Expert tensor parallel
    tp_ep: Optional[ProcessGroup] = None         # Combined TP+EP
    expt_dp: Optional[ProcessGroup] = None       # Expert data parallel
```

## Complete Configuration Examples

### Mixtral 8x7B

```bash
GPUS_PER_NODE=8
NNODES=4

--num-experts 8
--expert-model-parallel-size 8
--moe-router-load-balancing-type aux_loss
--moe-router-topk 2
--moe-aux-loss-coeff 1e-2
--moe-grouped-gemm
--moe-permute-fusion
--moe-token-dispatcher-type alltoall

--tensor-model-parallel-size 1
--pipeline-model-parallel-size 4
--num-layers-per-virtual-pipeline-stage 8
--sequence-parallel
--use-distributed-optimizer
```

### DeepSeek-V3 Style (Fine-Grained MoE)

```bash
--num-experts 256
--expert-model-parallel-size 64
--expert-tensor-parallel-size 1
--moe-router-topk 8
--moe-router-num-groups 8
--moe-router-group-topk 4
--moe-router-score-function sigmoid
--moe-router-load-balancing-type aux_loss seq_aux_loss
--moe-aux-loss-coeff 1e-2 1e-3

--moe-token-dispatcher-type flex
--moe-flex-dispatcher-backend deepep
--moe-grouped-gemm
--moe-permute-fusion
--moe-router-fusion

--moe-shared-expert-intermediate-size 2048
--moe-shared-expert-overlap

--overlap-moe-expert-parallel-comm
--delay-wgrad-compute
```

### Production MoE with FP8

```bash
--num-experts 64
--expert-model-parallel-size 8

--moe-token-dispatcher-type alltoall
--moe-grouped-gemm
--moe-permute-fusion
--moe-router-fusion
--moe-router-padding-for-fp8

--fp8-format e4m3
--fp8-recipe blockwise
--fp8-param-gather

--overlap-grad-reduce
--overlap-param-gather
--tp-comm-overlap
--use-distributed-optimizer
```

## Argument Reference

### Core MoE Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--num-experts` | Total number of experts | None |
| `--expert-model-parallel-size` | Expert parallel degree | 1 |
| `--expert-tensor-parallel-size` | Expert tensor parallel degree | Same as TP |
| `--moe-ffn-hidden-size` | Expert FFN hidden size | Model FFN size |
| `--moe-layer-freq` | MoE layer frequency (1=all layers) | 1 |

### Router Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--moe-router-topk` | Experts per token | 2 |
| `--moe-router-score-function` | Score: softmax, sigmoid | softmax |
| `--moe-router-pre-softmax` | Softmax before top-K | False |
| `--moe-router-num-groups` | Groups for group-limited routing | None |
| `--moe-router-group-topk` | Selected groups | None |
| `--moe-router-load-balancing-type` | Balancing: aux_loss, seq_aux_loss, global_aux_loss, sinkhorn, none | aux_loss |
| `--moe-aux-loss-coeff` | Aux loss coefficient(s) | 0.0 |
| `--moe-router-enable-expert-bias` | Enable bias-based balancing | False |
| `--moe-router-bias-update-rate` | Bias update rate | 1e-3 |
| `--moe-router-fusion` | Fuse router kernels | False |
| `--moe-router-dtype` | Router precision: fp32, fp64 | None |
| `--moe-z-loss-coeff` | Z-loss coefficient | None |

### Token Dispatcher Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--moe-token-dispatcher-type` | Dispatcher: allgather, alltoall, flex | allgather |
| `--moe-flex-dispatcher-backend` | Flex backend: deepep, hybridep | deepep |
| `--moe-expert-capacity-factor` | Capacity factor for token dropping | None |
| `--moe-pad-expert-input-to-capacity` | Pad to capacity | False |
| `--moe-token-drop-policy` | Drop policy: probs, position | probs |
| `--moe-permute-fusion` | Fuse permutation ops | False |

### Performance Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--moe-grouped-gemm` | Use GroupedGEMM | False |
| `--overlap-moe-expert-parallel-comm` | EP A2A overlap | False |
| `--delay-wgrad-compute` | Split dgrad/wgrad | False |
| `--moe-shared-expert-intermediate-size` | Shared expert FFN size | None |
| `--moe-shared-expert-overlap` | Overlap shared expert | False |
| `--moe-router-padding-for-fp8` | Pad for FP8 alignment | False |
