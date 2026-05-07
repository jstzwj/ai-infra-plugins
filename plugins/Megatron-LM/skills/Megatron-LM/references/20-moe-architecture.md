# 20 - Mixture of Experts (MoE) Architecture Reference

This document provides an exhaustive reference for Megatron-LM's Mixture of Experts
implementation, covering the MoE layer architecture, router, token dispatchers,
expert types, shared experts, load balancing, FP8 quantization, and memory optimization.

## Overview

Megatron-LM's MoE implementation is located at `megatron/core/transformer/moe/` and
implements a modular architecture with these key components:

1. **Router**: Determines which experts each token is sent to
2. **Token Dispatcher**: Handles communication and token permutation across devices
3. **Experts**: The actual expert networks that process tokens
4. **Shared Experts**: Optional always-active experts that process all tokens
5. **MoE Layer**: Orchestrates all components in the forward/backward pass

### Data Flow

```
Hidden States [S, B, H]
    |
    v
Router (gating + top-k) -> probs [num_tokens, num_experts], routing_map [num_tokens, num_experts]
    |
    v
Dispatch Preprocess (reshape, permute tokens by expert assignment)
    |
    v
Token Dispatch (all-to-all / all-gather communication across EP ranks)
    |
    v
Dispatch Postprocess (all-gather TP, sort by local expert)
    |
    v
Expert Compute (GroupedMLP or SequentialMLP)
    |
    v
Combine Preprocess (unsort, reduce-scatter TP)
    |
    v
Token Combine (all-to-all / reduce-scatter communication)
    |
    v
Combine Postprocess (unpermute, reshape, add shared expert output)
    |
    v
Output [S, B, H]
```

## MoE Layer Architecture

### BaseMoELayer

`BaseMoELayer` is the abstract base class for all MoE layers:

```python
class BaseMoELayer(MegatronModule, ABC):
    def __init__(self, config, layer_number, pg_collection, is_mtp_layer):
        self.num_local_experts = config.num_moe_experts // ep_size
        self.local_expert_indices = [offset + i for i in range(num_local_experts)]
        self.use_shared_expert = config.moe_shared_expert_intermediate_size is not None
        self.shared_expert_overlap = config.moe_shared_expert_overlap
```

### MoELayer

`MoELayer` is the concrete implementation with four forward phases:

1. **Route**: Compute token routing via the router
2. **Preprocess**: Apply latent projections, prepare for dispatch
3. **Dispatch + Expert Compute + Combine**: Token communication and expert processing
4. **Postprocess**: Combine outputs, add shared expert

#### MoESubmodules

```python
@dataclass
class MoESubmodules:
    experts: ExpertsBuilder
    shared_experts: SharedExpertsBuilder | None = None
    router: RouterBuilder = TopKRouter
```

#### Forward Execution Map

```python
self.fwd_execution_map = ["route", "expert_compute", "postprocess"]
```

The forward method executes through `custom_forward()` which runs:

```python
# Phase 1: Route
shared_expert_output = self.shared_experts_compute(hidden_states)
probs, routing_map = self.route(hidden_states, padding_mask)
hidden_states, probs = self.preprocess(hidden_states, probs, routing_map)

# Phase 2: Expert Compute
dispatched_input, probs = self.dispatch(hidden_states, probs)
output, mlp_bias = self.routed_experts_compute(dispatched_input, probs)
output = self.combine(output)

# Phase 3: Postprocess
output = self.postprocess(output, shared_expert_output)
```

### MoE with Latent Projections

When `config.moe_latent_size` is set, the MoE layer projects hidden states to a lower
latent dimension before expert processing, reducing computation:

```python
# Project down
hidden_states, _ = self.fc1_latent_proj(hidden_states)  # [S, B, H] -> [S, B, L]

# ... expert processing in latent dimension ...

# Project back up
output, _ = self.fc2_latent_proj(output)  # [S, B, L] -> [S, B, H]
```

## Router

### TopKRouter

**File**: `megatron/core/transformer/moe/router.py`

The `TopKRouter` routes each token to the top-k experts.

#### Routing Workflow

1. Compute logits via gating network: `logits = x @ weight.T + bias`
2. Apply z-loss (optional, for training stability)
3. Compute scores and routing map via score function
4. Apply token dropping (optional)
5. Apply auxiliary load balancing losses (optional)
6. Apply expert bias (optional)

#### Router Weight

```python
self.weight = torch.nn.Parameter(
    torch.empty((num_experts, hidden_size), dtype=torch.float32)
)
```

The weight is always stored in float32 for numerical stability, regardless of the
model's compute dtype.

#### Router Data Type

Controlled by `config.moe_router_dtype`:
- `"fp32"` (default): Router computation in float32
- `"fp64"`: Router computation in float64 (maximum precision)
- Model dtype: Router computation in the model's native dtype

#### Score Functions

The score function is configured via `config.moe_router_score_function`:

| Score Function | Formula | Use Case |
|---|---|---|
| `"softmax"` | `softmax(logits)` | Standard MoE (Switch Transformer) |
| `"sigmoid"` | `sigmoid(logits) / sum(topk_sigmoid)` | Token-level routing (DeepSeek-V2) |
| `"sqrtsoftplus"` | `sqrt(softplus(logits)) / sum(topk)` | Alternative normalization |

**Softmax score function modes**:
- `use_pre_softmax=False` (default): top-k first, then softmax on selected scores
- `use_pre_softmax=True`: softmax first, then top-k (preserves probability distribution)

#### Group-Limited Routing

When `config.moe_router_num_groups` and `config.moe_router_group_topk` are set, the
router implements group-limited routing (DeepSeek-V2/V3 style):

1. Divide experts into `num_groups` equal-sized groups
2. For each token, select `group_topk` groups based on the sum of top expert scores
3. From selected groups, choose top-k individual experts

```python
def group_limited_topk(scores, topk, num_tokens, num_experts, num_groups, group_topk):
    group_scores = scores.view(num_tokens, num_groups, -1).topk(topk // group_topk, dim=-1)[0].sum(dim=-1)
    group_idx = torch.topk(group_scores, k=group_topk, dim=-1, sorted=False)[1]
    # Mask experts not in selected groups
    masked_scores = scores.masked_fill(~score_mask.bool(), float('-inf'))
    probs, top_indices = torch.topk(masked_scores, k=topk, dim=-1)
```

#### Top-k Scaling

`config.moe_router_topk_scaling_factor` applies a scaling factor to routing scores:

```python
if scaling_factor:
    probs = probs * scaling_factor
```

#### Expert Bias

When `config.moe_router_enable_expert_bias=True`, the router maintains a running count
of tokens per expert and uses it to bias routing decisions for load balancing.

#### Input Jitter

Optional noise injection controlled by `config.moe_input_jitter_eps`:

```python
x = x * Uniform(1-eps, 1+eps).rsample(x.shape)
```

### InferenceTopKRouter

A stripped-down router for inference that:
- Skips z-loss, auxiliary losses, token dropping, and expert bias updates
- Uses `@torch.compile` on the routing function
- Returns dense `[num_tokens, topk]` tensors instead of sparse `[num_tokens, num_experts]`
- Only supports `num_groups=None` and score functions `"sigmoid"` or `"softmax"`

## Token Dispatchers

### MoETokenDispatcher (Base Class)

All dispatchers implement a six-method interface:

| Method | Phase | Description |
|---|---|---|
| `dispatch_preprocess` | Pre-dispatch | Reshape, compute metadata, permute tokens |
| `token_dispatch` | Communication | All-to-all or all-gather across ranks |
| `dispatch_postprocess` | Post-dispatch | AG(TP), sort by local expert |
| `combine_preprocess` | Pre-combine | Unsort, RS(TP) |
| `token_combine` | Communication | Reverse communication |
| `combine_postprocess` | Post-combine | Unpermute, reshape, add shared expert |

### AllGather Token Dispatcher

**Class**: `MoEAllGatherTokenDispatcher`

Communication pattern: **AllGather** (TP*EP domain) -> **ReduceScatter** (TP*EP domain)

**Workflow**:
1. `dispatch_preprocess`: Reshape to `[num_local_tokens, H]`
2. `token_dispatch`: Gather tokens from all TP*EP ranks
3. `dispatch_postprocess`: Permute to local experts, extract local probs
4. `combine_preprocess`: Unpermute expert outputs
5. `token_combine`: Reduce-scatter across TP*EP ranks
6. `combine_postprocess`: Reshape to original shape

**Best for**: Small EP sizes, when allgather bandwidth is available.

### AlltoAll Token Dispatcher

**Class**: `MoEAlltoAllTokenDispatcher`

Communication pattern: **Permutation 1** -> **A2A(EP)** -> **AG(TP)** -> Sort -> Experts ->
Unsort -> **RS(TP)** -> **A2A(EP)** -> **Unpermutation 1**

**Key features**:
- Separates EP communication from TP communication
- Two-level permutation: tokens-to-EP-ranks, then tokens-to-local-experts
- Supports dynamic token counts per expert
- DtoH stream for overlapping CPU metadata computation with GPU operations
- CUDA graph compatible with drop-and-pad mode

**Drop and Pad Mode**: When `moe_pad_expert_input_to_capacity=True`, tokens are
dropped to a fixed capacity per expert and padded, enabling static shapes for CUDA
graphs.

### Flex Token Dispatcher

**Class**: `MoEFlexTokenDispatcher`

A flexible dispatcher that abstracts TP and EP using a single communication group.
Supports two backends selected by `config.moe_flex_dispatcher_backend`:

#### DeepEP Backend (`"deepep"`)

Uses fused dispatch/combine kernels from the DeepEP package. Combines permutation and
communication into a single optimized operation.

- Token indices format: `[num_tokens, topk]` (dense)
- Fused operations reduce memory bandwidth requirements
- Requires `moe-router-dtype=fp32` for float32 probability tensors

#### HybridEP Backend (`"hybridep"`)

Uses fused kernels from the HybridEP package (DeepSeek-V3 style). Similar to DeepEP
but with different communication patterns optimized for hybrid parallelism.

- Routing map format: `[num_tokens, num_experts]` (sparse multi-hot)
- Supports configurable number of SMs for dispatch/combine
- Supports block-level permutation/unpermutation

## Expert Types

### SequentialMLP

**File**: `megatron/core/transformer/moe/experts.py`

Executes experts sequentially, one at a time:

```python
for expert, tokens, probs in zip(self.local_experts, tokens_list, probs_list):
    output, output_bias = expert(tokens, probs)
    output_local_list.append(output)
```

- Each expert is a full `MLP` module with its own parameters
- Supports FP8/FP4 with padding for quantization alignment
- Simple but less efficient than grouped implementations
- Supports different FFN hidden sizes via `config.moe_ffn_hidden_size`

### TEGroupedMLP

Executes all experts in parallel using Transformer Engine's grouped GEMM:

```python
fc1_output, bias = self.linear_fc1(permuted_input, tokens_per_expert)
bias_act_output = self.bias_act_func(fc1_output, bias, permuted_probs)
output, output_bias = self.linear_fc2(bias_act_output, tokens_per_expert)
```

Key features:
- Uses `GroupedLinear` for efficient batched expert computation
- Supports FP8/FP4 quantization with padding/unpadding
- Supports activation offloading (`expert_fc1`, `moe_act`)
- Supports activation recomputation for MoE activations
- Supports bias-activation fusion (SwiGLU, QuickGELU)
- Supports `moe_apply_probs_on_input` for top-1 routing optimization

### InferenceGroupedMLP

Inference-optimized version that extends `TEGroupedMLP`:

- Builds concatenated weight tensors on first forward pass for efficient access
- Three inference backends:
  - **FlashInfer**: `cutlass_fused_moe` kernel for CUDA-graphed inference
  - **Torch**: `torch.nn.functional.grouped_mm` with GPU-resident offsets
  - **vLLM**: Triton-based fused MoE kernel
- Supports MXFP8 quantization with stacked weight tensors
- Concatenated weights share storage with TE's per-expert parameters

## Shared Experts

### SharedExpertMLP

**File**: `megatron/core/transformer/moe/shared_experts.py`

A shared expert that processes all tokens unconditionally:

```python
class SharedExpertMLP(MLP):
    def __init__(self, config, submodules, gate, pg_collection):
        config.ffn_hidden_size = config.moe_shared_expert_intermediate_size
        # No bias supported in shared experts
        assert config.add_bias_linear == False
```

### Shared Expert Gate

When `config.moe_shared_expert_gate=True`, a learnable gate controls the shared
expert's contribution:

```python
logits = F.linear(hidden_states, self.gate_weight)  # [S*B, 1]
gate_score = torch.sigmoid(logits)
output = output * gate_score
```

### Shared Expert Overlap

When `config.moe_shared_expert_overlap=True`, the shared expert computation is
overlapped with token dispatch communication using a separate CUDA stream:

```
Main stream:   [Dispatch A2A]          [Combine A2A]  [Add shared output]
Shared stream: [AG(SP)] [FC1] [Act] [FC2] [RS(SP)]
```

The overlap uses a state machine (`SharedExpertState`) to enforce correct execution order:

```python
class SharedExpertState(Enum):
    IDLE = 0
    PRE_FORWARD_COMM_DONE = 1     # After all-gather for SP
    FC1_FORWARD_DONE = 2           # After linear_fc1 + activation
    FC2_FORWARD_DONE = 3           # After linear_fc2
    POST_FORWARD_COMM_DONE = 4     # After reduce-scatter for SP
```

Each state transition is validated via the `@overlap_state_check` decorator.

## Load Balancing

### Auxiliary Loss Types

Three types of auxiliary load balancing losses are supported, configured via
`config.moe_router_load_balancing_type`:

#### aux_loss (Switch Transformer)

Standard load balancing loss from the Switch Transformer paper:

```
loss = E * sum_i(f_i * P_i)
```

Where:
- `f_i` = fraction of tokens dispatched to expert i (across TP*CP ranks)
- `P_i` = average router probability for expert i
- `E` = total number of experts

#### seq_aux_loss (Sequence-Level)

Per-sequence load balancing loss. The batch dimension is reshaped to create
per-sequence expert counts, and the loss is averaged over the batch:

```
loss = (sum over sequences of aux_loss) / batch_size
```

#### global_aux_loss (Global)

Uses a running average of global token counts across all ranks (TP*DP*CP):

```python
self.global_tokens_per_expert += global_tokens_per_expert
self.ga_steps += 1
averaged_tokens = self.global_tokens_per_expert / self.ga_steps
```

This provides a more stable estimate of load imbalance across training steps.

#### Sinkhorn Routing

When `config.moe_router_load_balancing_type="sinkhorn"`, Sinkhorn balancing is applied
to routing logits instead of aux loss:

```python
norm_logits = sinkhorn(logits.to(float32))  # Iterative normalization
_, indices = torch.topk(norm_logits, k=topk, dim=1)
```

Sinkhorn routing produces near-perfect load balancing but is incompatible with
auxiliary loss.

### Z-Loss

From the ST-MoE paper, encourages router logits to remain small for training stability:

```python
z_loss = mean(logsumexp(logits)^2) * moe_z_loss_coeff
```

Configured via `config.moe_z_loss_coeff`.

### Fused Auxiliary Loss Computation

When `config.moe_router_fusion=True` and Transformer Engine >= 2.6.0, the aux loss
computation uses fused kernels for improved performance.

## Router Replay

**File**: `megatron/core/transformer/moe/router_replay.py`

Router replay enables deterministic routing for debugging and development:

```python
class RouterReplayAction(Enum):
    RECORD = "record"             # Record top-k indices
    REPLAY_FORWARD = "replay_forward"  # Use recorded indices for forward
    REPLAY_BACKWARD = "replay_backward"  # Use recorded indices for backward recompute
```

Global control:
```python
RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)
# ... forward pass ...
recorded_data = RouterReplay.get_recorded_data()

RouterReplay.set_replay_data(recorded_data)
RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
# ... forward pass with same routing ...
```

Static buffers for CUDA graph compatibility:
```python
RouterReplay.set_global_static_buffers(buffer)  # [max_tokens, num_layers, topk]
```

## MoE with FP8 Quantization

### Quantization Padding

FP8 and FP4 require token counts to be multiples of alignment sizes:

```python
def get_align_size_for_quantization(config):
    if config.fp8:
        return get_fp8_align_size(config.fp8_recipe)  # 16 or 128
    elif config.fp4:
        return get_fp4_align_size(config.fp4_recipe)
    return 0
```

The experts pad token counts to alignment multiples and unpad after computation.

### Router Padding for Quantization

When `config.moe_router_padding_for_quantization=True`, the routing map is padded
to ensure each expert receives a multiple of the alignment size tokens:

```python
routing_map = pad_routing_map(routing_map, pad_multiple)
```

### FP8 in GroupedMLP

`TEGroupedMLP` handles FP8 padding/unpadding automatically:

```python
if config.fp8 or config.fp4:
    actual_tokens_per_expert = tokens_per_expert
    permuted_input, tokens_per_expert = self.quantization_padding(input, tokens)
    # ... expert computation ...
    output = self.quantization_unpadding(output, actual_tokens_per_expert)
```

## MoE Layer Frequency Patterns

MoE layers can be interleaved with dense layers using the hybrid layer pattern:

```
"--hybrid-layer-pattern" "------E------E"   # MoE every 7 layers
"--hybrid-layer-pattern" "MM*EMM*EMM*E"     # Custom pattern
```

Alternatively, use `config.moe_layer_freq` to control MoE layer placement:
- Integer `N`: MoE layer every N layers
- List: Boolean mask per layer (1=MoE, 0=dense)

## Memory Optimization

### MoE Layer Recompute

When `config.recompute_granularity='selective'` and `"moe"` is in
`config.recompute_modules`, the entire MoE forward is wrapped in a checkpoint:

```python
if self.moe_layer_recompute and self.training:
    outputs = te_checkpoint(custom_forward, False, ...)
```

### Shared Expert Recompute

When `"shared_experts"` is in `config.recompute_modules`, shared expert computation
is checkpointed separately:

```python
if self.shared_experts_recompute:
    shared_expert_output = te_checkpoint(self.shared_experts, False, ...)
```

### Activation Offloading

Fine-grained activation offloading for expert layers:

| Module | Offload Key | Description |
|---|---|---|
| `expert_fc1` | `self.offload_expert_fc1` | Offload fc1 input to CPU |
| `moe_act` | `self.offload_moe_act` | Offload activation output to CPU |

Configured via `config.fine_grained_activation_offloading` and `config.offload_modules`.

### Delayed Weight Gradient Computation

When `config.overlap_dispatch_backward_with_experts_wgrad=True`, expert weight
gradients are computed on a separate CUDA stream, overlapped with dispatch backward:

```python
class _RecordExpertDgradCompletion(torch.autograd.Function):
    # Records event when expert data gradients complete

class _RegisterDelayedWgradForExperts(torch.autograd.Function):
    # Waits for dgrad event, then runs wgrad on separate stream
```

## Expert Capacity and Token Dropping

### Capacity Calculation

```python
capacity = ceil((num_tokens / num_experts) * capacity_factor)
```

### Token Dropping

When `config.moe_expert_capacity_factor` is set, tokens exceeding expert capacity
are dropped:

| Drop Policy | Description |
|---|---|
| `"probs"` | Drop tokens with lowest routing probability |
| `"position"` | Drop tokens based on position (later tokens dropped first) |

### Pad to Capacity

When `config.moe_pad_expert_input_to_capacity=True`, tokens are padded (not dropped)
to fill expert capacity. This enables static shapes for CUDA graphs.

## Configuration Quick Reference Table

| Parameter | Type | Default | Description |
|---|---|---|---|
| `num_moe_experts` | int | None | Total number of experts |
| `moe_router_topk` | int | 2 | Top-k experts per token |
| `moe_router_score_function` | str | `"softmax"` | Score: softmax, sigmoid, sqrtsoftplus |
| `moe_router_load_balancing_type` | str/list | `"aux_loss"` | Load balancing: aux_loss, seq_aux_loss, global_aux_loss, sinkhorn |
| `moe_aux_loss_coeff` | float/list | `0.01` | Auxiliary loss coefficient(s) |
| `moe_router_dtype` | str | `"fp32"` | Router compute dtype: fp32, fp64 |
| `moe_router_pre_softmax` | bool | False | Softmax before top-k |
| `moe_router_num_groups` | int | None | Number of expert groups |
| `moe_router_group_topk` | int | None | Groups to select per token |
| `moe_router_topk_scaling_factor` | float | None | Score scaling factor |
| `moe_router_fusion` | bool | False | Use fused TE kernels |
| `moe_router_enable_expert_bias` | bool | False | Expert bias for load balancing |
| `moe_token_dispatcher_type` | str | `"alltoall"` | Dispatcher: allgather, alltoall, flex |
| `moe_flex_dispatcher_backend` | str | `"deepep"` | Flex backend: deepep, hybridep |
| `moe_grouped_gemm` | bool | False | Use grouped GEMM for experts |
| `moe_ffn_hidden_size` | int | None | Expert FFN hidden size (default = ffn_hidden_size) |
| `moe_shared_expert_intermediate_size` | int | None | Shared expert FFN size |
| `moe_shared_expert_overlap` | bool | False | Overlap shared expert with dispatch |
| `moe_shared_expert_gate` | bool | False | Gated shared expert |
| `moe_expert_capacity_factor` | float | None | Expert capacity factor |
| `moe_pad_expert_input_to_capacity` | bool | False | Pad tokens to capacity |
| `moe_token_drop_policy` | str | `"probs"` | Token drop policy |
| `moe_apply_probs_on_input` | bool | False | Apply probs to input (top-1 only) |
| `moe_permute_fusion` | bool | False | Use fused permute kernels |
| `moe_z_loss_coeff` | float | None | Z-loss coefficient |
| `moe_input_jitter_eps` | float | None | Input jitter epsilon |
| `moe_enable_routing_replay` | bool | False | Enable router replay |
| `moe_layer_recompute` | via recompute | False | Recompute MoE activations |
| `moe_latent_size` | int | None | Latent dimension for expert projections |
| `moe_router_padding_for_quantization` | bool | False | Pad routing map for FP8/FP4 alignment |
| `overlap_dispatch_backward_with_experts_wgrad` | bool | False | Overlap wgrad with dispatch backward |
| `moe_hybridep_num_sms` | int | - | SM count for HybridEP dispatch |
| `moe_deepep_num_sms` | int | - | SM count for DeepEP dispatch |
| `inference_grouped_gemm_backend` | str | `"auto"` | Inference GEMM: auto, torch, te, vllm |

## Complete Code Example

### Training an MoE Model

```python
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.experts import TEGroupedMLP, GroupedMLPSubmodules
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer import TransformerConfig

# Configure the model
config = TransformerConfig(
    num_layers=32,
    hidden_size=4096,
    ffn_hidden_size=14336,
    num_moe_experts=8,
    moe_router_topk=2,
    moe_router_score_function="softmax",
    moe_router_load_balancing_type="aux_loss",
    moe_aux_loss_coeff=0.01,
    moe_token_dispatcher_type="alltoall",
    moe_grouped_gemm=True,
    moe_ffn_hidden_size=14336,
    moe_shared_expert_intermediate_size=14336,  # Enable shared expert
    moe_shared_expert_overlap=True,
    moe_shared_expert_gate=True,
    expert_model_parallel_size=4,
    tensor_model_parallel_size=2,
)

# Build MoE layer submodules
submodules = MoESubmodules(
    experts=GroupedMLPSubmodules(
        linear_fc1=TEGroupedMLP.linear_fc1,
        linear_fc2=TEGroupedMLP.linear_fc2,
    ),
    shared_experts=SharedExpertMLP,
    router=TopKRouter,
)

# Create MoE layer
moe_layer = MoELayer(config=config, submodules=submodules, layer_number=1)

# Forward pass
output, loss_mask = moe_layer(hidden_states)
```

### Hybrid Model with MoE Layers

```python
# Use hybrid_layer_pattern with 'E' symbol for MoE layers
# Example: 12-layer model with MoE every 4 layers
pattern = "---E---E---E"

# Or mix all layer types
pattern = "M*EM*EM*E"  # Mamba, Attention, MoE interleaved

# With MTP (Multi-Token Prediction)
pattern = "M*EM*EM*E/*E/*E"  # Main + 2 MTP depths
```

### Router Replay for Debugging

```python
from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction

# Record routing decisions
RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)
loss = model(inputs)  # Forward pass records routing
recorded = RouterReplay.get_recorded_data()

# Replay with same routing
RouterReplay.set_replay_data(recorded)
RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
loss = model(inputs)  # Uses recorded routing decisions

# Clean up
RouterReplay.clear_global_router_replay_action()
RouterReplay.clear_global_indices()
```

## Key Process Groups

MoE uses several process groups for different communication patterns:

| Group | Usage |
|---|---|
| `ep_group` | Expert parallel communication (all-to-all) |
| `tp_group` | Tensor parallel within experts |
| `expt_tp_group` | Tensor parallel for expert computation |
| `tp_ep_group` | Combined TP*EP domain for allgather/reduce-scatter |
| `tp_cp_group` | TP*CP domain for aux loss reduction |
| `tp_dp_cp_group` | TP*DP*CP domain for global aux loss |

These groups are configured via `ProcessGroupCollection` and accessed through
`pg_collection` in each MoE component.
