# DeepSpeed Mixture of Experts (MoE) Reference

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [ShardedMoE Class](#shardedmoe-class)
5. [Expert Module](#expert-module)
6. [Token Routing and Mappings](#token-routing-and-mappings)
7. [Router and Gating Mechanism](#router-and-gating-mechanism)
8. [DeepSpeedMoEConfig](#deepspeedmoeconfig)
9. [Expert Parallelism](#expert-parallelism)
10. [MoE Layer Integration](#moe-layer-integration)
11. [Load Balancing](#load-balancing)
12. [MoE Inference](#moe-inference)
13. [Configuration Examples](#configuration-examples)
14. [Performance Tuning](#performance-tuning)
15. [Code Examples](#code-examples)
16. [Troubleshooting](#troubleshooting)

---

## Overview

DeepSpeed provides a comprehensive Mixture of Experts (MoE) implementation that enables training sparse models with billions or trillions of parameters at the cost of dense models with far fewer parameters. The key idea is to route each input token to a subset of "expert" neural networks, activating only a fraction of the total parameters for any given input.

Key features of DeepSpeed MoE:
- **Expert parallelism**: Distribute experts across GPUs with all-to-all communication for token dispatch
- **Top-k gating**: Route each token to the top-k most relevant experts (typically k=2)
- **Load balancing**: Auxiliary loss to ensure even distribution of tokens across experts
- **Residual MoE**: Optional residual connections for improved gradient flow
- **Efficient all-to-all**: Optimized token dispatch and gather via NCCL all-to-all
- **MoE inference**: Optimized inference path for sparse expert evaluation
- **Integration with ZeRO**: Combine expert parallelism with ZeRO stages for maximum memory savings

---

## Architecture

### Directory Structure

```
deepspeed/moe/
  __init__.py
  sharded_moe.py          # ShardedMoE: main MoE layer
  expert.py               # Expert module implementation
  mappings.py             # Token routing (dispatch/gather)
  router.py               # Top-k gating router
  layers.py               # MoE layer wrappers
  reshape.py              # Tensor reshaping utilities
```

### MoE Architecture Diagram

```
                        Input Tokens
                        [B * S, D]
                            │
                    ┌───────▼───────┐
                    │   Gate/Router  │
                    │  (Linear +     │
                    │   Softmax +    │
                    │   Top-k)       │
                    └───┬───────┬───┘
                        │       │
                  top-1 gate  top-2 gate
                        │       │
                    ┌───▼───────▼───┐
                    │ Token Dispatch │  All-to-All
                    │ (dispatch to   │  (tokens -> experts)
                    │  expert GPUs)  │
                    └───┬───────┬───┘
                        │       │
              ┌─────────▼─┐ ┌──▼─────────┐
              │  Expert 0  │ │  Expert 1  │  ...
              │  (FFN)     │ │  (FFN)     │
              └─────────┬─┘ └──┬─────────┘
                        │       │
                    ┌───▼───────▼───┐
                    │ Token Gather   │  All-to-All
                    │ (combine from │  (experts -> tokens)
                    │  expert GPUs) │
                    └───┬───────┬───┘
                        │       │
                    ┌───▼───────▼───┐
                    │ Weighted Sum   │  gate_weight * expert_output
                    │ (combine top-1 │
                    │  and top-2)    │
                    └───────┬───────┘
                            │
                      Output Tokens
                      [B * S, D]
```

---

## Core Components

### Component Overview

| Component | File | Description |
|-----------|------|-------------|
| `ShardedMoE` | `sharded_mobe.py` | Main MoE layer with sharded experts |
| `Expert` | `expert.py` | Individual expert module (typically FFN) |
| `TopKGate` | `router.py` | Top-k gating mechanism with load balancing |
| `MOELayer` | `layers.py` | Wrapper combining gate, experts, and dispatch |
| `dispatch`, `combine` | `mappings.py` | Token routing between experts |
| `UniformPipeExpert` | `reshape.py` | Expert with uniform input/output shapes |

---

## ShardedMoE Class

The `ShardedMoE` class is the primary entry point for MoE in DeepSpeed. It manages a set of experts distributed across GPUs and handles the routing of tokens to the appropriate experts.

### Class Definition

```python
class ShardedMoE(nn.Module):
    """Mixture of Experts layer with expert sharding across GPUs.

    Distributes E experts across ep_size GPUs, where each GPU holds
    E/ep_size experts. Tokens are dispatched to the appropriate GPU
    via all-to-all communication.

    Args:
        hidden_size: Input and output dimension.
        expert: Expert module class or instance.
        num_experts: Total number of experts.
        ep_size: Expert parallel size (number of GPUs sharing experts).
        ep_group: Expert parallel process group.
        k: Number of experts to route each token to (default: 1).
        capacity_factor: Expert capacity as a multiplier of the ideal
            capacity (tokens_per_expert = total_tokens / num_experts).
            Default: 1.0 (no capacity factor).
        eval_capacity_factor: Capacity factor during evaluation.
        min_capacity: Minimum expert capacity.
        noisy_gate_policy: Noise policy for gating ("Jitter" or "RSample").
        drop_tokens: Whether to drop tokens that exceed expert capacity.
        use_rts: Use Random Token Selection for dropped tokens.
        capacity_factor_test: Capacity factor for test mode.
    """

    def __init__(
        self,
        hidden_size: int,
        expert: nn.Module,
        num_experts: int = 1,
        ep_size: int = 1,
        ep_group: Optional[dist.ProcessGroup] = None,
        k: int = 1,
        capacity_factor: float = 1.0,
        eval_capacity_factor: float = 1.0,
        min_capacity: int = 4,
        noisy_gate_policy: Optional[str] = None,
        drop_tokens: bool = True,
        use_rts: bool = True,
        capacity_factor_test: Optional[float] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.ep_size = ep_size
        self.ep_group = ep_group
        self.k = k

        # Number of experts per GPU
        self.num_local_experts = num_experts // ep_size

        # Create local experts
        self.experts = nn.ModuleList([
            copy.deepcopy(expert) for _ in range(self.num_local_experts)
        ])

        # Gate (router)
        self.gate = TopKGate(
            hidden_size=hidden_size,
            num_experts=num_experts,
            k=k,
            noisy_gate_policy=noisy_gate_policy,
            capacity_factor=capacity_factor,
            eval_capacity_factor=eval_capacity_factor,
            min_capacity=min_capacity,
            drop_tokens=drop_tokens,
            use_rts=use_rts,
        )

        # Token mapping
        self.moe_layer = MOELayer(
            gate=self.gate,
            experts=self.experts,
            ep_group=ep_group,
            ep_size=ep_size,
            num_local_experts=self.num_local_experts,
        )
```

### forward()

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    used_token: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward pass through MoE layer.

    Args:
        hidden_states: Input tensor of shape [B * S, D] or [B, S, D].
        used_token: Optional mask indicating valid tokens.

    Returns:
        Tuple of:
        - output: Combined expert outputs, shape [B * S, D]
        - balance_loss: Load balancing auxiliary loss
        - gate_logits: Raw gating logits for debugging
    """
    # Reshape to 2D: [B * S, D]
    orig_shape = hidden_states.shape
    if hidden_states.dim() == 3:
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_size)

    # Compute gating and dispatch tokens
    output, balance_loss = self.moe_layer(hidden_states, used_token)

    # Reshape back to original shape
    if len(orig_shape) == 3:
        output = output.reshape(orig_shape)

    return output, balance_loss
```

### all_to_all_communication()

The all-to-all communication dispatches tokens from all GPUs to the GPUs that hold the relevant experts, and then gathers the results back.

```python
def all_to_all_communication(self, tokens, is_dispatch=True):
    """Perform all-to-all token routing.

    Dispatch phase:
      - Input: tokens sorted by origin GPU, shape [num_tokens_per_gpu, D]
      - Output: tokens sorted by destination expert GPU, shape [tokens_for_local_experts, D]

    Combine phase (is_dispatch=False):
      - Input: tokens sorted by expert GPU, shape [tokens_from_local_experts, D]
      - Output: tokens sorted by origin GPU, shape [num_tokens_per_gpu, D]
    """
    # Split tokens into chunks for each GPU
    tokens_list = tokens.split(tokens.shape[0] // self.ep_size, dim=0)

    # Perform all-to-all
    output_list = [torch.empty_like(chunk) for chunk in tokens_list]
    dist.all_to_all(output_list, tokens_list, group=self.ep_group)

    return torch.cat(output_list, dim=0)
```

---

## Expert Module

The `Expert` class represents a single expert network. In most transformer MoE implementations, each expert is a feed-forward network (FFN).

### Expert Class

```python
class Expert(nn.Module):
    """A single expert module.

    Typically implemented as a feed-forward network:
      Expert(x) = W2(activation(W1(x) + b1) + b2)

    Supports both standard FFN and SwiGLU variants.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: Optional[int] = None,
        activation: nn.Module = nn.GELU(),
        use_swiglu: bool = False,
    ):
        super().__init__()
        if intermediate_size is None:
            intermediate_size = 4 * hidden_size

        if use_swiglu:
            self.w_gate = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.w_up = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.w_down = nn.Linear(intermediate_size, hidden_size, bias=False)
        else:
            self.w1 = nn.Linear(hidden_size, intermediate_size)
            self.activation = activation
            self.w2 = nn.Linear(intermediate_size, hidden_size)

        self.use_swiglu = use_swiglu

    def forward(self, x):
        if self.use_swiglu:
            return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
        else:
            return self.w2(self.activation(self.w1(x)))
```

### Expert Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_size` | `int` | required | Input/output dimension |
| `intermediate_size` | `int` | `4 * hidden_size` | FFN intermediate dimension |
| `activation` | `nn.Module` | `nn.GELU()` | Activation function |
| `use_swiglu` | `bool` | `False` | Use SwiGLU activation variant |

### Expert Types

DeepSpeed supports multiple expert types specified via the `type` configuration:

| Type | Description | Architecture |
|------|-------------|-------------|
| `standard` | Standard FFN expert | `W2(act(W1(x)))` |
| `residual` | Residual connection expert | `x + W2(act(W1(x)))` |

```python
class ResidualExpert(Expert):
    """Expert with residual connection."""

    def forward(self, x):
        return x + super().forward(x)
```

---

## Token Routing and Mappings

The `mappings.py` module implements the token dispatch (send tokens to experts) and combine (gather expert outputs) operations.

### Dispatch Operation

```python
def dispatch(tokens, gate_indices, gate_weights, capacity, num_experts, ep_size):
    """Dispatch tokens to their assigned experts.

    Args:
        tokens: [num_tokens, hidden_size]
        gate_indices: [num_tokens, k] - expert assignment for each token
        gate_weights: [num_tokens, k] - gate probability for each assignment
        capacity: int - maximum tokens per expert
        num_experts: int - total number of experts
        ep_size: int - expert parallel size

    Returns:
        dispatch_output: [num_experts, capacity, hidden_size] - tokens organized by expert
        mask: [num_experts, capacity] - valid token mask
        position: [num_tokens] - position index for each token in its expert's buffer
    """
    num_local_experts = num_experts // ep_size
    batch_size = tokens.shape[0]

    # Initialize expert buffers
    expert_input = torch.zeros(
        num_local_experts, capacity, tokens.shape[-1],
        device=tokens.device, dtype=tokens.dtype,
    )
    expert_mask = torch.zeros(
        num_local_experts, capacity,
        device=tokens.device, dtype=torch.bool,
    )

    # Place tokens into expert buffers
    for i in range(batch_size):
        for j in range(gate_indices.shape[1]):  # k experts
            expert_idx = gate_indices[i, j]
            local_expert_idx = expert_idx % num_local_experts
            # Find next available slot in expert's buffer
            slot = expert_mask[local_expert_idx].sum().item()
            if slot < capacity:
                expert_input[local_expert_idx, slot] = tokens[i] * gate_weights[i, j]
                expert_mask[local_expert_idx, slot] = True

    return expert_input, expert_mask
```

### Combine Operation

```python
def combine(expert_output, gate_indices, gate_weights, mask, num_tokens, num_experts, ep_size):
    """Combine expert outputs back to original token order.

    Args:
        expert_output: [num_local_experts, capacity, hidden_size] - expert outputs
        gate_indices: [num_tokens, k] - expert assignments
        gate_weights: [num_tokens, k] - gate probabilities
        mask: [num_experts, capacity] - valid token mask
        num_tokens: int - total number of tokens
        num_experts: int - total number of experts
        ep_size: int - expert parallel size

    Returns:
        output: [num_tokens, hidden_size] - combined expert outputs
    """
    num_local_experts = num_experts // ep_size
    output = torch.zeros(
        num_tokens, expert_output.shape[-1],
        device=expert_output.device, dtype=expert_output.dtype,
    )

    # Build reverse mapping: for each token, find its expert output
    # Expert outputs are scattered back to their original token positions
    expert_positions = mask.cumsum(dim=1) - 1  # Position within expert buffer

    for i in range(num_tokens):
        for j in range(gate_indices.shape[1]):
            expert_idx = gate_indices[i, j]
            local_expert_idx = expert_idx % num_local_experts
            pos = expert_positions[local_expert_idx][i] if mask[local_expert_idx][i] else -1
            if pos >= 0:
                output[i] += expert_output[local_expert_idx, pos]

    return output
```

### All-to-All Token Routing

```python
class AllToAllTokenRouter:
    """Routes tokens between GPUs using NCCL all-to-all."""

    def __init__(self, ep_group, ep_size, num_local_experts):
        self.ep_group = ep_group
        self.ep_size = ep_size
        self.num_local_experts = num_local_experts

    def dispatch(self, tokens_by_expert):
        """Send tokens to expert GPUs via all-to-all.

        Input shape:  [ep_size, num_local_experts, capacity, hidden_size]
        Output shape: [ep_size, num_local_experts, capacity, hidden_size]

        All-to-all reorganizes the first dimension (source GPU -> destination GPU).
        """
        # Flatten for all-to-all
        input_flat = tokens_by_expert.flatten(0, 1)  # [ep_size * num_local_experts, capacity, D]
        split_tensors = list(input_flat.chunk(self.ep_size * self.num_local_experts, dim=0))

        output_flat = [torch.empty_like(t) for t in split_tensors]
        dist.all_to_all(output_flat, split_tensors, group=self.ep_group)

        return torch.cat(output_flat, dim=0).reshape(
            self.ep_size, self.num_local_experts, -1, tokens_by_expert.shape[-1]
        )

    def combine(self, expert_outputs):
        """Gather expert outputs back via all-to-all (inverse of dispatch)."""
        return self.dispatch(expert_outputs)  # Symmetric operation
```

---

## Router and Gating Mechanism

The router determines which experts should process each token. DeepSpeed implements a top-k gating mechanism with optional noise for load balancing.

### TopKGate

```python
class TopKGate(nn.Module):
    """Top-k gating mechanism with load balancing loss.

    Routes each token to the top-k experts based on learned gate scores.
    Includes an auxiliary load balancing loss to encourage even token
    distribution across experts.

    Args:
        hidden_size: Input dimension.
        num_experts: Total number of experts.
        k: Number of experts to select per token (default: 1).
        noisy_gate_policy: Noise injection strategy:
            - "Jitter": Add uniform noise to gate scores during training
            - "RSample": Use reparameterized sampling
            - None: No noise
        capacity_factor: Multiplier for expert capacity during training.
        eval_capacity_factor: Capacity factor during evaluation.
        min_capacity: Minimum expert capacity (default: 4).
        drop_tokens: Whether to drop tokens exceeding capacity.
        use_rts: Use Random Token Selection for capacity management.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        k: int = 1,
        noisy_gate_policy: Optional[str] = None,
        capacity_factor: float = 1.0,
        eval_capacity_factor: float = 1.0,
        min_capacity: int = 4,
        drop_tokens: bool = True,
        use_rts: bool = True,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.noisy_gate_policy = noisy_gate_policy
        self.capacity_factor = capacity_factor
        self.eval_capacity_factor = eval_capacity_factor
        self.min_capacity = min_capacity
        self.drop_tokens = drop_tokens
        self.use_rts = use_rts

        # Learnable gate weights
        self.wg = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, x):
        """Compute gating decisions.

        Args:
            x: Input tensor [B * S, D]

        Returns:
            gate_scores: [B * S, k] - normalized gate weights for selected experts
            gate_indices: [B * S, k] - indices of selected experts
            balance_loss: scalar - auxiliary load balancing loss
        """
        logits = self.wg(x)  # [B * S, num_experts]

        # Add noise during training
        if self.training and self.noisy_gate_policy == "Jitter":
            noise = torch.randn_like(logits) * 1.0 / self.num_experts
            logits = logits + noise

        # Compute top-k
        gate_scores, gate_indices = torch.topk(logits, self.k, dim=-1)
        gate_scores = F.softmax(gate_scores, dim=-1)

        # Compute load balancing loss
        balance_loss = self._load_balancing_loss(logits, gate_indices)

        return gate_scores, gate_indices, balance_loss

    def _load_balancing_loss(self, logits, indices):
        """Compute auxiliary load balancing loss.

        Loss = num_experts * sum_i(f_i * P_i)
        where:
          f_i = fraction of tokens dispatched to expert i
          P_i = fraction of gate probability allocated to expert i

        This encourages uniform distribution of tokens across experts.
        """
        num_tokens = logits.shape[0]

        # f_i: fraction of tokens assigned to expert i
        mask = F.one_hot(indices, self.num_experts).sum(dim=1).float()  # [B*S, E]
        f = mask.mean(dim=0)  # [E]

        # P_i: mean gate probability for expert i
        P = F.softmax(logits, dim=-1).mean(dim=0)  # [E]

        # Balance loss
        balance_loss = self.num_experts * (f * P).sum()

        return balance_loss
```

### Top-2 Gating Detail

The standard configuration uses top-2 gating (k=2), where each token is routed to the two highest-scoring experts:

```python
# Gating computation for k=2
logits = self.wg(x)                              # [B*S, E]
top2_scores, top2_indices = logits.topk(2, dim=-1)  # [B*S, 2]

# Softmax over top-2 only
top2_scores = F.softmax(top2_scores, dim=-1)     # [B*S, 2], sums to 1

# Final output = score_0 * expert_0(x) + score_1 * expert_1(x)
```

### Capacity Factor

The capacity factor controls how many tokens each expert can process:

```python
# Expert capacity calculation
tokens_per_expert = total_tokens / num_experts  # Ideal capacity
capacity = int(tokens_per_expert * capacity_factor)
capacity = max(capacity, min_capacity)

# If capacity_factor = 1.0:
#   Each expert processes exactly tokens/num_experts tokens
#   Some tokens may be dropped if distribution is uneven

# If capacity_factor = 1.5:
#   Each expert reserves 50% extra capacity
#   Fewer tokens dropped, but more memory usage

# If capacity_factor = 2.0:
#   Each expert reserves 2x the ideal capacity
#   Almost no tokens dropped, but significant memory overhead
```

---

## DeepSpeedMoEConfig

The MoE configuration is specified within the DeepSpeed configuration JSON.

### Configuration Structure

```json
{
    "moe": {
        "enabled": true,
        "ep_size": 8,
        "moe_experts": 8,
        "type": "residual",
        "ep_mp_group": null,
        "ep_group": null
    }
}
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | `bool` | `False` | Enable MoE |
| `ep_size` | `int` | `1` | Expert parallel size (number of GPUs for expert distribution) |
| `moe_experts` | `int` | `1` | Total number of experts |
| `type` | `str` | `"standard"` | Expert type: `"standard"` or `"residual"` |
| `ep_mp_group` | `ProcessGroup` | `None` | Expert-model-parallel group for combined EP+MP |
| `ep_group` | `ProcessGroup` | `None` | Expert parallel group (auto-created if None) |

### Python API

```python
from deepspeed.moe.sharded_moe import ShardedMoE

moe_layer = ShardedMoE(
    hidden_size=4096,
    expert=Expert(hidden_size=4096, intermediate_size=11008),
    num_experts=8,
    ep_size=4,       # 8 experts across 4 GPUs (2 experts per GPU)
    k=2,             # Top-2 gating
    capacity_factor=1.25,
    drop_tokens=True,
    use_rts=True,
)
```

---

## Expert Parallelism

Expert parallelism distributes experts across multiple GPUs, with each GPU holding a subset of the total experts. Tokens are routed to the appropriate GPU via all-to-all communication.

### Expert Distribution

```
Total experts: 8, EP size: 4

GPU 0: Expert 0, Expert 1
GPU 1: Expert 2, Expert 3
GPU 2: Expert 4, Expert 5
GPU 3: Expert 6, Expert 7

Token routing:
  Token A -> Expert 3 -> dispatched to GPU 1
  Token B -> Expert 6 -> dispatched to GPU 3
  Token C -> Expert 0 -> stays on GPU 0
```

### EP Group Management

```python
# DeepSpeed automatically creates expert parallel groups
def create_expert_parallel_groups(ep_size, world_size):
    """Create process groups for expert parallelism."""
    num_ep_groups = world_size // ep_size
    ep_groups = []

    for i in range(num_ep_groups):
        ranks = list(range(i * ep_size, (i + 1) * ep_size))
        group = dist.new_group(ranks)
        ep_groups.append(group)

    return ep_groups
```

### EP + TP Combination

Expert parallelism can be combined with tensor parallelism:

```
Total GPUs: 16
TP size: 2, EP size: 8

For each TP group (2 GPUs):
  - All 8 experts are distributed across 8 EP GPUs
  - Each expert's weights are further split by TP (column/row parallel)

Layout:
  TP Group 0 (ranks 0,1):   EP split across ranks 0,2,4,6,8,10,12,14
  TP Group 1 (ranks 1,3):   EP split across ranks 1,3,5,7,9,11,13,15
```

---

## MoE Layer Integration

### Using MoE in a Transformer Model

```python
class MoETransformerBlock(nn.Module):
    """Transformer block with MoE FFN."""

    def __init__(self, hidden_size, num_heads, intermediate_size,
                 num_experts=8, ep_size=4, k=2):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, num_heads)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        # Replace standard FFN with MoE
        expert = Expert(hidden_size, intermediate_size)
        self.moe = ShardedMoE(
            hidden_size=hidden_size,
            expert=expert,
            num_experts=num_experts,
            ep_size=ep_size,
            k=k,
            capacity_factor=1.25,
        )

    def forward(self, x):
        # Attention
        residual = x
        x = self.ln1(x)
        x = self.attention(x)
        x = residual + x

        # MoE FFN
        residual = x
        x = self.ln2(x)
        x, balance_loss, _ = self.moe(x)
        x = residual + x

        return x, balance_loss
```

### MoE with HuggingFace Models

DeepSpeed can replace FFN layers in HuggingFace models with MoE layers:

```python
from transformers import AutoModelForCausalLM
from deepspeed.moe.layers import MoELayer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Replace FFN layers with MoE
for layer in model.model.layers:
    original_ffn = layer.mlp
    layer.mlp = ShardedMoE(
        hidden_size=4096,
        expert=Expert(4096, 11008),
        num_experts=8,
        ep_size=4,
        k=2,
    )
```

---

## Load Balancing

Load balancing ensures that tokens are distributed evenly across experts, preventing some experts from being overloaded while others sit idle.

### Auxiliary Loss

The standard load balancing loss from the Switch Transformer paper:

```
L_balance = N * sum_i(f_i * P_i)

where:
  N = number of experts
  f_i = fraction of tokens dispatched to expert i
  P_i = mean gate probability for expert i
```

The loss is minimized when tokens are uniformly distributed (f_i = 1/N for all i) and gate probabilities are uniform (P_i = 1/N for all i).

### Loss Weight

```python
# The balance loss is typically weighted by a small coefficient
balance_loss_weight = 0.01

total_loss = task_loss + balance_loss_weight * balance_loss
```

### Capacity Management

When an expert receives more tokens than its capacity allows:

1. **Drop tokens** (`drop_tokens=True`): Excess tokens are dropped and their output is zero.
2. **Random Token Selection** (`use_rts=True`): Randomly select which tokens to keep, providing better load balancing over time.

```python
# Capacity calculation
total_tokens = batch_size * seq_length
tokens_per_expert = total_tokens / num_experts
capacity = max(int(tokens_per_expert * capacity_factor), min_capacity)
```

### Capacity Factor Guidelines

| Capacity Factor | Token Drop Rate | Memory Overhead | Recommendation |
|----------------|----------------|-----------------|----------------|
| 1.0 | ~5-10% | None | Tight memory, may lose tokens |
| 1.25 | ~1-3% | 25% | Good balance |
| 1.5 | <1% | 50% | Safe choice for most cases |
| 2.0 | ~0% | 100% | No drops, maximum memory |

---

## MoE Inference

DeepSpeed supports optimized inference for MoE models, including efficient expert evaluation and sparse computation.

### Inference Configuration

```json
{
    "moe": {
        "enabled": true,
        "ep_size": 4,
        "moe_experts": 8
    },
    "inference": {
        "enabled": true,
        "dtype": "fp16"
    }
}
```

### Inference Optimizations

1. **Expert Batching**: Batch tokens assigned to the same expert for efficient GPU utilization.
2. **Sparse Activation**: Only compute forward pass through selected experts (k out of E).
3. **Expert Caching**: Cache expert weights in GPU memory for low-latency inference.
4. **Dynamic Batching**: Dynamically adjust batch sizes based on token routing.

### Inference Engine Integration

```python
import deepspeed
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1")

# Initialize DeepSpeed inference engine with MoE support
ds_engine = deepspeed.init_inference(
    model=model,
    mp_size=4,      # Model parallel for each expert
    dtype=torch.float16,
    moe_config={
        "enabled": True,
        "ep_size": 4,
        "moe_experts": 8,
    },
)

# Run inference
outputs = ds_engine.generate(input_ids, max_new_tokens=100)
```

---

## Configuration Examples

### Example 1: 8-Expert MoE with Expert Parallelism

```json
{
    "moe": {
        "enabled": true,
        "ep_size": 8,
        "moe_experts": 8,
        "type": "standard"
    },
    "train_batch_size": 32,
    "gradient_accumulation_steps": 4,
    "optimizer": {
        "type": "AdamW",
        "params": {"lr": 1e-4}
    },
    "fp16": {"enabled": true}
}
```

### Example 2: Mixtral-style 8x7B with Top-2 Gating

```json
{
    "moe": {
        "enabled": true,
        "ep_size": 4,
        "moe_experts": 8,
        "type": "standard"
    },
    "tensor_parallel": {
        "enabled": true,
        "autotp_size": 2,
        "preset_model": "mixtral"
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        }
    },
    "train_batch_size": 16,
    "bf16": {"enabled": true},
    "optimizer": {
        "type": "AdamW",
        "params": {"lr": 5e-5}
    }
}
```

### Example 3: 64-Expert MoE with Residual Connections

```json
{
    "moe": {
        "enabled": true,
        "ep_size": 8,
        "moe_experts": 64,
        "type": "residual"
    },
    "zero_optimization": {
        "stage": 1
    },
    "train_batch_size": 64,
    "bf16": {"enabled": true},
    "optimizer": {
        "type": "AdamW",
        "params": {"lr": 1e-4}
    }
}
```

### Example 4: MoE + ZeRO-3 + Offload

```json
{
    "moe": {
        "enabled": true,
        "ep_size": 16,
        "moe_experts": 128,
        "type": "standard"
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "nvme",
            "nvme_path": "/local_nvme"
        },
        "offload_param": {
            "device": "nvme",
            "nvme_path": "/local_nvme"
        }
    },
    "train_batch_size": 32,
    "bf16": {"enabled": true},
    "optimizer": {
        "type": "AdamW",
        "params": {"lr": 1e-4}
    }
}
```

---

## Performance Tuning

### Expert Count Selection

| Model Size | Recommended Experts | Active Parameters | Total Parameters |
|-----------|--------------------|--------------------|-----------------|
| 1B | 8 | ~250M | ~2B |
| 7B | 8 | ~2B | ~14B |
| 7B | 64 | ~250M | ~56B |
| 13B | 8 | ~3.5B | ~26B |
| 70B | 8 | ~18B | ~140B |

### Capacity Factor Tuning

1. Start with `capacity_factor=1.25` for training.
2. Monitor token drop rate (should be < 1%).
3. Increase to 1.5 if drops are too frequent.
4. Use `capacity_factor=1.0` for memory-constrained scenarios.

### Communication Optimization

Expert parallelism introduces all-to-all communication. Optimize with:

```bash
# NCCL tuning for all-to-all
export NCCL_ALGO=Ring                  # Ring algorithm for all-to-all
export NCCL_PROTO=Simple               # Simple protocol
export NCCL_MIN_NCHANNELS=16           # Minimum channels
export NCCL_MAX_NCHANNELS=32           # Maximum channels
export NCCL_IB_DISABLE=0               # Enable InfiniBand
```

### Gradient Accumulation with MoE

MoE models benefit from larger effective batch sizes for load balancing:

```json
{
    "gradient_accumulation_steps": 8,
    "train_micro_batch_size_per_gpu": 4
}
```

---

## Code Examples

### Example 1: Training a Custom MoE Model

```python
import torch
import torch.nn as nn
import deepspeed
from deepspeed.moe.sharded_moe import ShardedMoE
from deepspeed.moe.expert import Expert

# Define model with MoE
class MoEModel(nn.Module):
    def __init__(self, vocab_size=32000, hidden_size=4096,
                 num_layers=32, num_experts=8, ep_size=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            self._make_layer(hidden_size, num_experts, ep_size)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

    def _make_layer(self, hidden_size, num_experts, ep_size):
        return nn.ModuleDict({
            "attention": nn.MultiheadAttention(hidden_size, 32),
            "ln1": nn.LayerNorm(hidden_size),
            "moe": ShardedMoE(
                hidden_size=hidden_size,
                expert=Expert(hidden_size, 11008),
                num_experts=num_experts,
                ep_size=ep_size,
                k=2,
                capacity_factor=1.25,
            ),
            "ln2": nn.LayerNorm(hidden_size),
        })

    def forward(self, input_ids):
        x = self.embed(input_ids)
        total_balance_loss = 0.0
        for layer in self.layers:
            # Attention
            residual = x
            x = layer["ln1"](x)
            x, _ = layer["attention"](x, x, x)
            x = residual + x
            # MoE
            residual = x
            x = layer["ln2"](x)
            x, balance_loss, _ = layer["moe"](x)
            x = residual + x
            total_balance_loss += balance_loss
        x = self.ln_f(x)
        logits = self.head(x)
        return logits, total_balance_loss

# DeepSpeed configuration
ds_config = {
    "moe": {
        "enabled": True,
        "ep_size": 4,
        "moe_experts": 8,
    },
    "train_batch_size": 32,
    "fp16": {"enabled": True},
    "optimizer": {
        "type": "AdamW",
        "params": {"lr": 1e-4},
    },
}

# Initialize
model = MoEModel()
ds_engine = deepspeed.initialize(
    model=model,
    config=ds_config,
    model_parameters=model.parameters(),
)

# Training loop
for batch in dataloader:
    logits, balance_loss = ds_engine(batch["input_ids"])
    task_loss = F.cross_entropy(logits[:, :-1].contiguous().view(-1, vocab_size),
                                 batch["labels"][:, 1:].contiguous().view(-1))
    total_loss = task_loss + 0.01 * balance_loss
    ds_engine.backward(total_loss)
    ds_engine.step()
```

### Example 2: Converting Dense Model to MoE

```python
import torch
import deepspeed
from transformers import AutoModelForCausalLM
from deepspeed.moe.sharded_moe import ShardedMoE
from deepspeed.moe.expert import Expert

def convert_to_moe(model, num_experts=8, ep_size=4):
    """Convert a dense transformer model to MoE by replicating FFN layers."""
    for name, module in model.named_modules():
        if isinstance(module, LlamaMLP):  # Or appropriate FFN class
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]

            # Create MoE with copies of the original FFN as experts
            expert = Expert(
                hidden_size=module.hidden_size,
                intermediate_size=module.intermediate_size,
                use_swiglu=True,
            )
            moe = ShardedMoE(
                hidden_size=module.hidden_size,
                expert=expert,
                num_experts=num_experts,
                ep_size=ep_size,
                k=2,
            )

            # Replace the FFN with MoE
            parent = model.get_submodule(parent_name)
            setattr(parent, child_name, moe)

    return model

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = convert_to_moe(model, num_experts=8, ep_size=4)
```

---

## Troubleshooting

### Common Issues

**1. Token dropping during training**

```
Warning: Expert 3 dropped 15.2% of tokens due to capacity overflow
```

Increase `capacity_factor` or decrease batch size:
```json
{"moe": {"capacity_factor": 1.5}}
```

**2. Load imbalance**

```
Expert utilization: [0.12, 0.08, 0.35, 0.15, 0.06, 0.09, 0.10, 0.05]
```

Ensure the balance loss weight is appropriate (typically 0.01). Check if the gate weights are learning properly.

**3. All-to-all communication bottleneck**

Profile the all-to-all time:
```python
import torch.distributed as dist
torch.cuda.synchronize()
start = time.time()
dist.all_to_all(output, input, group=ep_group)
torch.cuda.synchronize()
print(f"All-to-all time: {time.time() - start:.3f}s")
```

**4. OOM with large expert count**

Use ZeRO-3 with offloading:
```json
{
    "moe": {"moe_experts": 128, "ep_size": 16},
    "zero_optimization": {
        "stage": 3,
        "offload_param": {"device": "cpu"}
    }
}
```

**5. Gradient flow issues in residual MoE**

Reduce the residual coefficient:
```python
# In residual expert
self.residual_coefficient = 0.1  # Scale down residual contribution
```
