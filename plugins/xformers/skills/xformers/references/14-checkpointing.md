# 14 - Selective Activation Checkpointing

## Overview

xFormers provides selective activation checkpointing that uses Mixed Integer Linear Programming (MILP) to find the optimal set of operators to store (vs recompute) given a memory budget. This is more efficient than PyTorch's all-or-nothing checkpointing.

**Source**: `xformers/checkpoint.py`

## Key Idea

Standard activation checkpointing either stores all activations or recomputes all of them. xFormers' selective approach:
1. Profiles each operator's runtime and memory usage
2. Uses MILP optimization to find the subset of operators to store that:
   - Minimizes total recomputation time
   - Fits within the given memory budget

## API Reference

### `checkpoint`

```python
from xformers import checkpoint

output = checkpoint(
    function,                      # Function to checkpoint
    *args,                         # Arguments to function
    preserve_rng_state=True,       # Whether to save/restore RNG state
    policy_fn=None,               # Checkpoint policy
    **kwargs,                     # Keyword arguments
) -> Any                          # Function output
```

Wraps `torch.utils.checkpoint.checkpoint` with custom policy support.

**Parameters:**
- `function` - The function to run (typically a model's forward pass or a sub-module)
- `preserve_rng_state` - Whether to save and restore RNG state during checkpointing
- `policy_fn` - Either:
  - `None` (default policy: store mm, addmm, FMHA ops)
  - `List[Op]` - List of operators to store (e.g., `["aten.mm.default", ...]`)
  - `Callable` - Custom policy function `(ctx, func, *args, **kwargs) -> bool`

### `get_optimal_checkpoint_policy`

```python
from xformers import get_optimal_checkpoint_policy

policy = get_optimal_checkpoint_policy(
    function,          # Function to optimize (usually forward pass)
    *args,             # Example arguments
    memory_budget: float,  # 0.0 to 1.0
) -> Callable         # Policy function for checkpoint()
```

Automatically finds the optimal checkpointing policy given a memory budget.

**Parameters:**
- `function` - The forward pass function to optimize
- `*args` - Example inputs (used for profiling)
- `memory_budget` - Float between 0 and 1:
  - 0 = recompute everything (like standard checkpointing)
  - 1 = store everything (like no checkpointing)
  - 0.5 = store up to 50% of activation memory

**Raises:**
- `RuntimeError` if scipy is not available
- `ValueError` if memory_budget is not in [0, 1]

### `list_operators`

```python
from xformers import list_operators

operators = list_operators(function, *args, **kwargs) -> List[Op]
```

Returns the list of operators used inside a function with the given arguments. Useful for understanding what operators your model uses.

### `selective_checkpoint_wrapper`

```python
from xformers import selective_checkpoint_wrapper

wrapped_module = selective_checkpoint_wrapper(
    module: torch.nn.Module,
    memory_budget: Optional[float] = None,
    policy_fn: Optional[Callable] = None,
) -> SelectiveCheckpointWrapper
```

Wraps a module with selective activation checkpointing. Either `memory_budget` or `policy_fn` must be specified (not both).

### `SelectiveCheckpointWrapper`

```python
from xformers.checkpoint import SelectiveCheckpointWrapper

wrapper = SelectiveCheckpointWrapper(
    module,                   # Module to wrap
    memory_budget=None,      # Memory budget (0-1)
    policy_fn=None,          # Or explicit policy
)
```

An `ActivationWrapper` subclass that:
1. On first forward pass, profiles operators and computes optimal policy
2. In distributed settings, broadcasts the policy from rank 0
3. Uses the computed policy for subsequent forward passes

## Policy Functions

### Default Policy

Stores these operators (recomputes everything else):

```python
_default_allow_list = [
    "xformers.efficient_attention_forward_cutlass.default",
    "xformers_flash.flash_fwd.default",
    "aten.addmm.default",
    "aten.mm.default",
]
```

These are expensive matrix operations that benefit most from being stored.

### Custom Policy (List of Operators)

```python
# Store only specific operators
policy = checkpoint(
    model_block, x,
    policy_fn=["aten.mm.default", "aten.addmm.default"],
)
```

### Custom Policy (Function)

```python
def my_policy(ctx, func, *args, **kwargs):
    # Store expensive ops, recompute cheap ones
    return str(func) in expensive_ops

output = checkpoint(model_block, x, policy_fn=my_policy)
```

## MILP Optimization

The `get_optimal_checkpoint_policy` uses scipy's MILP solver:

```python
# Minimize: sum(runtime_i * x_i)  (total recomputation time)
# Subject to: sum(memory_i * x_i) <= budget  (memory constraint)
# Where x_i = 0 means store, x_i = 1 means recompute
```

**Additional constraints:**
1. **View-like ops**: Always recomputed (no memory cost)
2. **In-place ops**: Must be stored/recomputed together with their parent
3. **Random ops**: Always stored (for determinism)
4. **Last op**: Always stored (it's the output, memory cost set to 0)

## Profiling

### `ProfileOperatorsTorchDispatchMode`

Profiles each operator's runtime and memory usage:

```python
profiler = ProfileOperatorsTorchDispatchMode(num_runs=10)
with profiler:
    model(input)

for op_data in profiler.data:
    print(f"{op_data.name}: {op_data.time_taken:.3f}s, {op_data.memory_used:.1f}MB")
```

**ProfileMetadata fields:**
- `name` - Operator handle
- `time_taken` - Average runtime over num_runs
- `memory_used` - Peak memory during operator
- `curr_idx` - Operator index in execution order
- `output_ids` - Storage data pointers (for in-place detection)
- `inplace_info` - Tuple of (op_id, parent_id) for in-place ops
- `is_view_like` - Whether the op is a view
- `is_rand_op` - Whether the op uses random state

## Usage Examples

### Basic Selective Checkpointing

```python
import torch
import torch.nn as nn
from xformers import checkpoint

class TransformerBlock(nn.Module):
    def __init__(self, dim):
        self.attn = Attention(dim)
        self.mlp = MLP(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

block = TransformerBlock(768)
x = torch.randn(32, 128, 768, device="cuda")

# With default policy
output = checkpoint(block, x)
```

### Optimal Policy

```python
from xformers import checkpoint, get_optimal_checkpoint_policy

block = TransformerBlock(768)
example_input = torch.randn(32, 128, 768, device="cuda")

# Find optimal policy for 30% memory budget
policy = get_optimal_checkpoint_policy(block, example_input, memory_budget=0.3)

# Use the policy
output = checkpoint(block, x, policy_fn=policy)
```

### Module Wrapper

```python
from xformers import selective_checkpoint_wrapper

# Wrap each transformer block
model = nn.Sequential(*[
    selective_checkpoint_wrapper(TransformerBlock(768), memory_budget=0.5)
    for _ in range(12)
])

# Normal forward pass - checkpointing is handled automatically
output = model(x)
output.sum().backward()
```

### Discover Operators

```python
from xformers import list_operators

# See what operators a function uses
ops = list_operators(block, x)
for op in ops:
    print(op)
```

## torch.compile Compatibility

Selective checkpointing is compatible with `torch.compile`:

```python
import torch

# The policy is computed before compilation
policy = get_optimal_checkpoint_policy(block, example_input, memory_budget=0.5)

@torch.compile
def forward(x):
    return checkpoint(block, x, policy_fn=policy)

output = forward(x)
```

Note: `SelectiveCheckpointWrapper._get_policy_fn` is decorated with `@torch.compiler.disable` to prevent the policy computation from being compiled.

## Distributed Training

In distributed settings, the policy is broadcast from rank 0 to ensure all ranks use the same policy:

```python
class SelectiveCheckpointWrapper:
    def _get_policy_fn(self, *args, **kwargs):
        policy_fn = get_optimal_checkpoint_policy(...)

        if distributed and world_size > 1:
            objects = [policy_fn]
            torch.distributed.broadcast_object_list(objects, src=0)
            policy_fn = objects[0]

        return policy_fn
```

## Comparison with Standard Checkpointing

| Feature | PyTorch checkpoint | xFormers selective |
|---------|--------------------|--------------------|
| Memory control | All or nothing | Fine-grained (0-100%) |
| Policy | Always recompute all | Optimal subset |
| Optimization | None | MILP (scipy) |
| Overhead | None | One-time profiling |
| torch.compile | Yes | Yes |
| Distributed | Yes | Yes (broadcast policy) |
