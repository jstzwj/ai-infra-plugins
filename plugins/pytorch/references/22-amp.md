# Automatic Mixed Precision (AMP) in PyTorch

This chapter provides a comprehensive reference for automatic mixed precision training in PyTorch, covering `torch.amp.autocast`, `torch.amp.GradScaler`, BFloat16 vs Float16, custom autocast behavior, and complete training loop examples.

---

## 1. Overview of Mixed Precision Training

Mixed precision training uses a combination of float32 (FP32) and lower-precision formats (float16 or bfloat16) to:

- **Reduce memory usage:** FP16/BF16 use 2 bytes vs 4 bytes for FP32
- **Accelerate training:** Modern GPUs have specialized hardware for lower-precision math (Tensor Cores)
- **Maintain accuracy:** Critical operations remain in FP32

### 1.1 Precision Formats

| Format | Bits | Exponent | Mantissa | Range |
|---|---|---|---|---|
| FP32 | 32 | 8 | 23 | ~1e-38 to ~3e38 |
| FP16 | 16 | 5 | 10 | ~6e-8 to ~65504 |
| BF16 | 16 | 8 | 7 | ~1e-38 to ~3e38 |

**FP16**: Same dynamic range as FP32 but with reduced precision (7 mantissa bits vs 23). It is fully supported on NVIDIA GPUs with Ampere architecture and later. BF16 avoids the overflow/underflow issues of FP16 at the cost of slightly lower precision.

**FP16 Limitations**: The maximum representable value is 65504. Values above this overflow to `inf`. The smallest normal value is ~6e-8; values below this may be flushed to zero (denormal numbers are not always supported efficiently).

### 1.2 How AMP Works

AMP has two components:

1. **`torch.amp.autocast`**: Automatically casts eligible operations to a lower-precision dtype (FP16 or BF16). Operations are classified into categories based on their numerical safety.

2. **`torch.amp.GradScaler`** (FP16 only): Handles loss scaling to prevent gradient underflow. Gradients in FP16 can be too small to represent, so the loss is scaled up before backpropagation and the gradients are scaled back down.

---

## 2. torch.amp.autocast

### 2.1 API Signature

```python
torch.amp.autocast(
    device_type: str,
    dtype: Optional[torch.dtype] = None,
    enabled: bool = True,
    cache_enabled: bool = True
) -> _AutocastContextManager
```

**Parameters:**

- **`device_type`** (str, required): The device type to use. Must be `"cuda"` for GPU mixed precision, or `"cpu"` for CPU mixed precision. Other valid values include `"xpu"` (Intel GPU), `"hpu"` (Habana), etc.

- **`dtype`** (torch.dtype, optional): The target data type for operations within the autocast context. If `None`, uses the default dtype for the device type:
  - For `"cuda"`: `torch.float16`
  - For `"cpu"`: `torch.bfloat16`
  
  Common values: `torch.float16`, `torch.bfloat16`.

- **`enabled`** (bool): Whether autocast is enabled. Default: `True`. When `False`, disables the autocast context. This is useful for conditionally enabling mixed precision.

- **`cache_enabled`** (bool): Whether to cache the weight cast decisions. Default: `True`. When `True`, autocast caches whether each parameter should be cast, reducing overhead on subsequent iterations.

**Returns:** A context manager or decorator.

### 2.2 Basic Usage

#### As a Context Manager

```python
import torch

model = torch.nn.Linear(512, 256).cuda()
optimizer = torch.optim.Adam(model.parameters())
data = torch.randn(32, 512, device='cuda')

# Using autocast as context manager
with torch.amp.autocast('cuda'):
    output = model(data)
    loss = output.sum()

loss.backward()
optimizer.step()
```

#### As a Decorator

```python
@torch.amp.autocast('cuda')
def forward_pass(model, data):
    return model(data)

output = forward_pass(model, data)
```

#### FP16 (Default for CUDA)

```python
with torch.amp.autocast('cuda', dtype=torch.float16):
    # Operations will be performed in FP16 where safe
    output = model(data)
```

#### BFloat16

```python
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    # Operations will be performed in BF16 where safe
    output = model(data)
```

### 2.3 Disabling Autocast Conditionally

```python
use_amp = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):
    output = model(data)
    loss = criterion(output, target)
```

### 2.4 Nested Autocast

```python
# Outer autocast in BF16
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    output = model(data)

    # Inner autocast overrides to FP16 for specific operations
    with torch.amp.autocast('cuda', dtype=torch.float16):
        special_output = special_op(output)
```

### 2.5 Local Disabling with torch.amp.custom_fwd and custom_bwd

```python
# Disable autocast in a custom autograd function's forward pass
class MyCustomOp(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd  # Default: disables autocast in forward
    def forward(ctx, x):
        # This runs in whatever dtype x is, no autocast
        return x.float().matmul(x.float().t())

    @staticmethod
    @torch.amp.custom_bwd  # Default: disables autocast in backward
    def backward(ctx, grad_output):
        return grad_output.float()

# Or explicitly control the behavior:
class MyCustomOp2(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(cast_inputs=torch.float32)  # Cast inputs to FP32
    def forward(ctx, x):
        return x.matmul(x.t())

    @staticmethod
    @torch.amp.custom_bwd
    def backward(ctx, grad_output):
        return grad_output
```

---

## 3. Autocast Operation Categories

Autocast classifies operations into three categories based on their numerical safety:

### 3.1 Whitelist (Cast to Lower Precision)

These operations are numerically safe in FP16/BF16 and are automatically cast:

**Linear Algebra:**
- `torch.mm`, `torch.bmm`, `torch.addmm`
- `torch.baddbmm`, `torch.btrilinear`
- `torch.nn.functional.linear`, `torch.nn.functional.bilinear`

**Convolution:**
- `torch.nn.functional.conv1d`, `conv2d`, `conv3d`
- `torch.nn.functional.conv_transpose1d`, `conv_transpose2d`, `conv_transpose3d`

**Recurrent Layers:**
- `torch.nn.functional.gru_cell`, `lstm_cell`, `rnn_cell`

**Attention:**
- `torch.nn.functional.scaled_dot_product_attention`

**Pointwise Operations:**
- `torch.nn.functional.relu`, `leaky_relu`, `elu`, `selu`, `gelu`, `silu`, `mish`
- `torch.nn.functional.tanh`, `sigmoid`
- `torch.abs`, `torch.clamp`, `torch.neg`

**Reductions (safe in lower precision):**
- `torch.norm` (some variants)

### 3.2 Blacklist (Keep in FP32)

These operations are numerically unsafe in lower precision and always remain in FP32:

**Loss Functions:**
- `torch.nn.functional.cross_entropy`
- `torch.nn.functional.nll_loss`
- `torch.nn.functional.binary_cross_entropy`
- `torch.nn.functional.binary_cross_entropy_with_logits`
- `torch.nn.functional.kl_div`
- `torch.nn.functional.mse_loss`, `l1_loss`, `smooth_l1_loss`, `huber_loss`
- `torch.nn.functional.cosine_similarity`
- `torch.nn.functional.triplet_margin_loss`
- `torch.log_softmax`

**Exp/Log Operations:**
- `torch.exp`, `torch.log`, `torch.log10`, `torch.log2`, `torch.log1p`
- `torch.pow` (when exponent is not an integer)

**Normalization:**
- `torch.nn.functional.layer_norm`, `group_norm`, `instance_norm`
- `torch.nn.functional.batch_norm` (some implementations)

**Other Numerically Sensitive:**
- `torch.norm` (certain variants)
- `torch.cumsum`
- `torch.softmax` (usually computed in FP32 internally even when inputs are FP16)

### 3.3 Graylist (Op-Specific Behavior)

These operations may or may not be cast depending on the specific function and context:

- `torch.matmul` (cast for 2D tensors, may stay FP32 for batched operations)
- `torch.nn.functional.embedding` (cast output to lower precision)
- `torch.cat`, `torch.stack` (preserve input dtype)
- Binary operations: output dtype matches the higher-precision input

### 3.4 Operation Behavior Summary

```python
import torch

x_fp16 = torch.randn(10, 10, device='cuda', dtype=torch.float16)
x_fp32 = torch.randn(10, 10, device='cuda', dtype=torch.float32)

with torch.amp.autocast('cuda'):
    # Whitelist ops: cast inputs to FP16, output is FP16
    y = torch.mm(x_fp32, x_fp32)  # FP16 computation, FP16 result
    print(y.dtype)  # torch.float16

    # Blacklist ops: force FP32 computation
    z = torch.nn.functional.cross_entropy(
        x_fp32, torch.randint(0, 10, (10,)), reduction='mean'
    )
    print(z.dtype)  # torch.float32

    # Pointwise: preserve input dtype
    w = torch.relu(x_fp32)
    print(w.dtype)  # torch.float16 (input was autocast to FP16)
```

---

## 4. torch.amp.GradScaler

### 4.1 Why GradScaler Is Needed (FP16 Only)

When using FP16, gradients can be very small (e.g., 1e-6) and may fall below FP16's minimum representable value (~6e-8 for normal numbers). These tiny gradients become zero, leading to training stagnation.

GradScaler addresses this by:
1. Scaling up the loss before backpropagation (e.g., by 2^16 = 65536)
2. Computing gradients at the scaled-up level
3. Unscaling the gradients before the optimizer step
4. Adjusting the scale factor dynamically

**Note:** GradScaler is NOT needed for BFloat16, because BF16 has the same dynamic range as FP32 (8 exponent bits).

### 4.2 API Signature

```python
torch.amp.GradScaler(
    device: str = "cuda",
    init_scale: float = 2.**16,
    growth_factor: float = 2.0,
    backoff_factor: float = 0.5,
    growth_interval: int = 2000,
    enabled: bool = True
)
```

**Parameters:**

- **`device`** (str): The device type. Default: `"cuda"`. Must match the device used in `autocast`.

- **`init_scale`** (float): The initial scale factor. Default: `65536.0` (2^16). This value is a reasonable starting point for most models.

- **`growth_factor`** (float): Multiplier for increasing the scale when no infs/NaNs are found. Default: `2.0`. Each growth interval, if no overflow occurs, the scale is multiplied by this factor.

- **`backoff_factor`** (float): Multiplier for decreasing the scale when infs/NaNs are found. Default: `0.5`. When overflow is detected, the scale is multiplied by this factor.

- **`growth_interval`** (int): Number of steps between scale increases. Default: `2000`. Every `growth_interval` consecutive steps without overflow, the scale increases by `growth_factor`.

- **`enabled`** (bool): Whether the scaler is active. Default: `True`. When `False`, all scaler operations become no-ops, which is useful for easy enable/disable.

### 4.3 GradScaler Methods

#### scale

```python
scaler.scale(outputs: Union[torch.Tensor, Iterable[torch.Tensor]]) -> Union[torch.Tensor, Iterable[torch.Tensor]]
```

Multiplies the loss (or outputs) by the current scale factor. Returns scaled loss.

```python
scaler = torch.amp.GradScaler('cuda')
with torch.amp.autocast('cuda'):
    output = model(data)
    loss = criterion(output, target)

# Scale the loss
scaled_loss = scaler.scale(loss)
```

#### unscale_

```python
scaler.unscale_(optimizer: torch.optim.Optimizer) -> None
```

Unscales the gradients held by the optimizer's parameters. Divides gradients by the current scale factor. This is an in-place operation on the gradients.

**Raises:** `RuntimeError` if `unscale_` is called twice on the same optimizer before `update()`.

```python
scaler.unscale_(optimizer)

# After unscaling, you can clip gradients normally
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

#### step

```python
scaler.step(optimizer: torch.optim.Optimizer, *args, **kwargs) -> Optional[torch.Tensor]
```

Attempts to call `optimizer.step()`. If unscaled gradients contain `inf` or `NaN`, this skips the optimizer step entirely (avoiding corrupted model weights).

```python
scaler.step(optimizer)
```

#### update

```python
scaler.update(new_scale: Optional[Union[float, torch.Tensor]] = None) -> None
```

Updates the scale factor based on whether infs/NaNs were found during the step. Must be called after `scaler.step()`.

- If no overflow occurred and `growth_interval` steps have passed, scale increases by `growth_factor`.
- If overflow occurred, scale decreases by `backoff_factor`.

```python
scaler.update()
```

#### get_scale

```python
scaler.get_scale() -> float
```

Returns the current scale factor as a float.

```python
current_scale = scaler.get_scale()
print(f"Current scale: {current_scale}")
```

#### get_growth_factor

```python
scaler.get_growth_factor() -> float
```

Returns the growth factor.

#### get_backoff_factor

```python
scaler.get_backoff_factor() -> float
```

Returns the backoff factor.

#### get_growth_interval

```python
scaler.get_growth_interval() -> int
```

Returns the growth interval.

#### is_enabled

```python
scaler.is_enabled() -> bool
```

Returns whether the scaler is enabled.

#### set_backoff_factor

```python
scaler.set_backoff_factor(new_factor: float) -> None
```

Sets a new backoff factor.

#### set_growth_factor

```python
scaler.set_growth_factor(new_factor: float) -> None
```

Sets a new growth factor.

#### set_growth_interval

```python
scaler.set_growth_interval(new_interval: int) -> None
```

Sets a new growth interval.

#### scale_, unscale_grad_

```python
# Internal methods
scaler._scale: torch.Tensor  # The current scale as a tensor
scaler._growth_tracker: int  # Steps since last scale change
```

### 4.4 GradScaler State Dict

```python
# Save scaler state (important for resuming training)
scaler_state = scaler.state_dict()
torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scaler': scaler_state,
}, 'checkpoint.pt')

# Load scaler state
checkpoint = torch.load('checkpoint.pt')
scaler.load_state_dict(checkpoint['scaler'])
```

---

## 5. BFloat16 vs Float16 Comparison

### 5.1 Detailed Comparison

| Feature | Float16 | BFloat16 |
|---------|---------|----------|
| Total bits | 16 | 16 |
| Exponent bits | 5 | 8 |
| Mantissa bits | 10 | 7 |
| Dynamic range | ~6e-8 to 65504 | ~1e-38 to ~3e38 |
| Precision | Higher | Lower |
| Requires GradScaler | Yes | No |
| Tensor Core support | Ampere+ | Ampere+ |
| Memory savings | Same | Same |
| Numerical stability | Lower | Higher |
| Hardware support | Broad | Ampere+ (A100, RTX 30xx) |

### 5.2 When to Use Each

**Use Float16 when:**
- You need maximum performance on older GPUs (pre-Ampere)
- Your model is well-tested with FP16 and stable
- You want slightly higher precision for small values

**Use BFloat16 when:**
- You have Ampere or later GPUs (A100, H100, RTX 30xx/40xx)
- Your model has numerical stability issues with FP16
- You want simpler code (no GradScaler needed)
- You are training large language models or transformers

### 5.3 BFloat16 Training (No GradScaler Needed)

```python
model = MyModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for data, target in dataloader:
    data, target = data.cuda(), target.cuda()

    # Only autocast needed for BF16 - no GradScaler
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        output = model(data)
        loss = criterion(output, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 5.4 FP16 Training (With GradScaler)

```python
model = MyModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = torch.amp.GradScaler('cuda')

for data, target in dataloader:
    data, target = data.cuda(), target.cuda()

    with torch.amp.autocast('cuda', dtype=torch.float16):
        output = model(data)
        loss = criterion(output, target)

    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

## 6. Custom Autocast for Custom Operations

### 6.1 Registering Custom Autocast Implementations

```python
import torch

# Define a custom operation
class MyMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a @ b

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        return grad_output @ b.T, a.T @ grad_output

# Register autocast implementation
@torch.amp.custom_fwd
def my_matmul(a, b):
    return MyMatMul.apply(a, b)
```

### 6.2 Custom Forward/Backward Casting

```python
class MyFP32SafeOp(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):
        # Inputs are automatically cast to FP32
        # Do numerically sensitive computation in FP32
        result = torch.logsumexp(x, dim=-1)
        ctx.save_for_backward(x)
        return result

    @staticmethod
    @torch.amp.custom_bwd
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        # Backward also in FP32
        softmax = torch.softmax(x, dim=-1)
        return grad_output.unsqueeze(-1) * softmax
```

### 6.3 Registering Autocast for Existing Functions

```python
# Register an autocast implementation for a specific op
torch.amp.autocast.define_autocast_hook(
    device_type='cuda',
    dtype=torch.float16
)

# More commonly, use the decorator pattern
class CustomLinear(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight, bias)
        output = input @ weight.T
        if bias is not None:
            output += bias
        return output

    @staticmethod
    @torch.amp.custom_bwd
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_output @ weight
        grad_weight = grad_output.T @ input
        grad_bias = grad_output.sum(0) if bias is not None else None
        return grad_input, grad_weight, grad_bias
```

### 6.4 Disabling Autocast for Specific Operations

```python
with torch.amp.autocast('cuda'):
    output = model(data)

    # Force FP32 for a specific operation
    with torch.amp.autocast('cuda', enabled=False):
        sensitive_result = numerically_sensitive_op(output.float())

    # Autocast resumes here
    more_results = model.more_layers(sensitive_result)
```

---

## 7. Complete Training Loop Examples

### 7.1 FP16 Training Loop with GradScaler

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def train_fp16(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    learning_rate: float = 1e-3,
    device: str = 'cuda',
):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')  # FP16 requires GradScaler

    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # Forward pass with autocast
            with torch.amp.autocast('cuda', dtype=torch.float16):
                output = model(data)
                loss = criterion(output, target)

            # Backward pass with scaled gradients
            optimizer.zero_grad()
            scaler.scale(loss).backward()

            # Unscale before gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step with skip on overflow
            scaler.step(optimizer)

            # Update scale factor
            scaler.update()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                scale = scaler.get_scale()
                print(f"Epoch {epoch}, Batch {batch_idx}, "
                      f"Loss: {loss.item():.4f}, Scale: {scale:.0f}")

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)

                with torch.amp.autocast('cuda', dtype=torch.float16):
                    output = model(data)
                    val_loss += criterion(output, target).item()
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()

        val_loss /= len(val_loader)
        accuracy = 100.0 * correct / len(val_loader.dataset)

        print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return model
```

### 7.2 BFloat16 Training Loop (No GradScaler)

```python
def train_bf16(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int,
    learning_rate: float = 1e-3,
    device: str = 'cuda',
):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                output = model(data)
                loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return model
```

### 7.3 Generic AMP Training with Enable/Disable

```python
def train_with_amp(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int,
    use_amp: bool = True,
    dtype: torch.dtype = torch.float16,
    device: str = 'cuda',
):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    criterion = nn.CrossEntropyLoss()

    # GradScaler only needed for FP16
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp and dtype == torch.float16)

    for epoch in range(num_epochs):
        model.train()

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            with torch.amp.autocast('cuda', enabled=use_amp, dtype=dtype):
                output = model(data)
                loss = criterion(output, target)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

    return model
```

### 7.4 Distributed Training with AMP

```python
import torch.distributed as dist

def train_distributed_amp(
    rank: int,
    world_size: int,
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int,
):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    model = model.to(rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(model.parameters())
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(num_epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(rank), target.to(rank)

            with torch.amp.autocast('cuda'):
                output = model(data)
                loss = criterion(output, target)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    dist.destroy_process_group()
```

### 7.5 Gradient Accumulation with AMP

```python
def train_with_gradient_accumulation(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int,
    accum_steps: int = 4,
):
    model = model.cuda()
    optimizer = torch.optim.AdamW(model.parameters())
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()

            with torch.amp.autocast('cuda'):
                output = model(data)
                loss = criterion(output, target) / accum_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
```

---

## 8. Advanced Topics

### 8.1 Monitoring Scale Factor

```python
# Track scale factor over training
scale_history = []

for batch in dataloader:
    # ... training step ...
    scaler.step(optimizer)
    scaler.update()
    scale_history.append(scaler.get_scale())

# If scale drops frequently, consider:
# 1. Reducing learning rate
# 2. Using BF16 instead of FP16
# 3. Increasing init_scale
```

### 8.2 Saving and Resuming Training

```python
def save_checkpoint(model, optimizer, scaler, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
    }, path)

def load_checkpoint(model, optimizer, scaler, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    return checkpoint['epoch']
```

### 8.3 Debugging AMP Issues

```python
# Check if overflow is occurring
scaler = torch.amp.GradScaler('cuda')

with torch.amp.autocast('cuda'):
    output = model(data)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)

# Check for inf/nan in gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        if torch.isnan(param.grad).any():
            print(f"NaN gradient in {name}")
        if torch.isinf(param.grad).any():
            print(f"Inf gradient in {name}")
```

### 8.4 CPU Autocast (BFloat16)

```python
# CPU mixed precision is also supported
with torch.amp.autocast('cpu', dtype=torch.bfloat16):
    x = torch.randn(100, 100)  # CPU tensor
    y = torch.mm(x, x.T)       # Computed in BF16

# Useful for CPU-bound inference or training on CPUs with BF16 support
# (Intel Cooper Lake, Sapphire Rapids, etc.)
```

### 8.5 Inference with AMP

```python
def inference_with_amp(model, dataloader):
    model.eval()
    results = []

    with torch.no_grad():
        for data in dataloader:
            data = data.cuda()
            with torch.amp.autocast('cuda', dtype=torch.float16):
                output = model(data)
            results.append(output.cpu())

    return torch.cat(results, dim=0)

# For production inference, consider torch.compile with AMP
model_compiled = torch.compile(model)
with torch.amp.autocast('cuda', dtype=torch.float16):
    output = model_compiled(input_tensor)
```

---

## 9. Performance Considerations

### 9.1 Expected Speedups

| Model Type | GPU | FP16 Speedup | BF16 Speedup |
|-----------|-----|-------------|-------------|
| CNN (ResNet) | A100 | 2-3x | 2-3x |
| Transformer | A100 | 2-4x | 2-4x |
| RNN/LSTM | V100 | 1.5-2x | N/A |

### 9.2 Best Practices

1. **Always use autocast for the entire forward pass**, not just parts of it. This ensures consistent casting behavior.

2. **Use GradScaler for FP16** but not for BF16.

3. **Monitor the scale factor** -- frequent drops indicate instability.

4. **Use `non_blocking=True`** for data transfers to overlap with computation.

5. **Consider `torch.compile`** combined with AMP for additional speedups.

6. **Test accuracy** -- compare FP32 baseline with AMP results to ensure no degradation.

7. **Use `cache_enabled=True`** (default) to reduce autocast overhead.

### 9.3 Common Pitfalls

1. **Mismatched dtypes**: Some operations may produce FP32 outputs inside autocast. Be careful with operations outside the autocast context.

2. **Skipping GradScaler for FP16**: This will likely lead to gradient underflow and training failure.

3. **Using FP16 on pre-Ampere GPUs**: Tensor Core FP16 is much less efficient on V100 and earlier compared to Ampere+.

4. **Not saving scaler state**: When resuming training, forgetting to restore the scaler state can lead to suboptimal performance.
