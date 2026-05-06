# Quantization

## Overview

Quantization reduces the precision of neural network weights and activations from 32-bit floating point (FP32) to lower precision formats (INT8, INT4, etc.), trading a small amount of accuracy for significant improvements in model size, memory bandwidth, and inference speed. PyTorch provides a comprehensive quantization toolkit supporting post-training quantization (PTQ), quantization-aware training (QAT), and dynamic quantization.

**Source location**: `torch/quantization/`, `torch/ao/quantization/`

### Why Quantize?

| Metric | FP32 | INT8 | Improvement |
|--------|------|------|-------------|
| Model size | 4 bytes/param | 1 byte/param | 4x smaller |
| Memory bandwidth | 4x | 1x | 4x less |
| Compute throughput | 1x | 2-4x (INT8 hardware) | 2-4x faster |
| Accuracy | Baseline | ~1% loss (typically) | Acceptable for most tasks |

---

## Quantized Data Types

```python
import torch

# Supported quantized dtypes
torch.qint8      # 8-bit signed integer (quantized)
torch.quint8     # 8-bit unsigned integer (quantized)
torch.qint32     # 32-bit signed integer (quantized)
torch.quint4x2   # 4-bit unsigned (packed in pairs)
torch.quint2x4   # 2-bit unsigned (packed in quads)
torch.float8_e4m3fn   # 8-bit float (E4M3)
torch.float8_e5m2     # 8-bit float (E5M2)
```

### Quantization Parameters

```python
# Quantization maps float values to integers using:
# Q(x) = round(x / scale + zero_point)
# Dequantization: x = (Q(x) - zero_point) * scale

# scale: floating point factor that determines the quantization step size
# zero_point: integer offset that maps to 0.0 in float

# For qint8: values range from -128 to 127
# For quint8: values range from 0 to 255
```

---

## torch.quantization.quantize

### Post-Training Static Quantization

```python
import torch
import torch.quantization as quant

# Full static quantization workflow
model = torchvision.models.resnet18(pretrained=True)
model.eval()

# Step 1: Set quantization configuration
model.qconfig = quant.get_default_qconfig('fbgemm')  # x86
# model.qconfig = quant.get_default_qconfig('qnnpack')  # ARM

# Step 2: Prepare model for quantization
# This inserts observers into the model
quant.prepare(model, inplace=True)

# Step 3: Calibrate with representative data
with torch.no_grad():
    for images, _ in calibration_dataloader:
        model(images)
    # Observers record activation statistics (min, max, histogram)

# Step 4: Convert to quantized model
quant.convert(model, inplace=True)

# Now model uses INT8 operations
output = model(test_input)
```

### Signature

```python
torch.quantization.quantize(
    model: torch.nn.Module,
    qconfig: Optional[QConfig] = None,
    mapping: Optional[Dict[type, type]] = None,
    inplace: bool = False,
) -> torch.nn.Module
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `nn.Module` | Model to quantize |
| `qconfig` | `QConfig` | Quantization configuration |
| `mapping` | `Dict` | Module type mapping for quantization |
| `inplace` | `bool` | Modify model in-place |

---

## torch.quantization.quantize_dynamic

### Dynamic Quantization

Dynamic quantization quantizes only the weights statically but quantizes activations dynamically during inference. It requires no calibration data.

```python
import torch
import torch.quantization as quant

model = torch.nn.Sequential(
    torch.nn.Linear(784, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 10),
)

# Dynamic quantization
quantized_model = quant.quantize_dynamic(
    model,
    qconfig_spec={torch.nn.Linear},  # quantize only Linear layers
    dtype=torch.qint8,                # quantized dtype for weights
)

# The quantized model:
# - Weights are stored as INT8 (static)
# - Activations are quantized on-the-fly during inference
# - No calibration needed

output = quantized_model(test_input)
```

### Signature

```python
torch.quantization.quantize_dynamic(
    model: torch.nn.Module,
    qconfig_spec: Optional[Union[Set[type], Dict[str, type]]] = None,
    dtype: torch.dtype = torch.qint8,
    mapping: Optional[Dict[type, type]] = None,
    inplace: bool = False,
) -> torch.nn.Module
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `nn.Module` | Model to quantize |
| `qconfig_spec` | `Set` or `Dict` | Which layer types to quantize |
| `dtype` | `torch.dtype` | Quantized dtype (qint8, float16) |
| `mapping` | `Dict` | Module type mapping |
| `inplace` | `bool` | Modify in-place |

### qconfig_spec Options

```python
# Quantize specific layer types
qconfig_spec = {torch.nn.Linear}  # only Linear layers
qconfig_spec = {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU}
qconfig_spec = {torch.nn.Embedding, torch.nn.Linear}

# Use a dictionary for per-type configuration
qconfig_spec = {
    torch.nn.Linear: torch.quantization.default_dynamic_qconfig,
    torch.nn.LSTM: torch.quantization.default_dynamic_qconfig,
}

# None means quantize all quantizable layers
qconfig_spec = None
```

---

## torch.quantization.quantize_qat

### Quantization-Aware Training (QAT)

QAT simulates quantization during training so the model learns to be robust to quantization noise.

```python
import torch
import torch.quantization as quant

model = torchvision.models.resnet18(pretrained=True)
model.train()

# Step 1: Set QAT configuration
model.qconfig = quant.get_default_qat_qconfig('fbgemm')

# Step 2: Prepare for QAT
# This inserts fake quantization modules
quant.prepare_qat(model, inplace=True)

# Step 3: Fine-tune with fake quantization
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for images, targets in train_dataloader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

    # Optionally freeze observers after some epochs
    if epoch > 3:
        model.apply(torch.quantization.disable_observer)

    # Optionally freeze batch norm stats
    if epoch > 3:
        model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

# Step 4: Convert to quantized model
quantized_model = quant.convert(model, inplace=False)

# Evaluate quantized model
quantized_model.eval()
output = quantized_model(test_input)
```

### Signature

```python
torch.quantization.quantize_qat(
    model: torch.nn.Module,
    qconfig: Optional[QConfig] = None,
    mapping: Optional[Dict[type, type]] = None,
    inplace: bool = False,
) -> torch.nn.Module
```

---

## QConfig

`QConfig` pairs an activation observer with a weight observer.

### Default Configurations

```python
import torch.quantization as quant

# For x86 CPUs (FBGEMM backend)
default_qconfig = quant.default_qconfig
# activation: MinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine)
# weight: MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)

# For QAT on x86
default_qat_qconfig = quant.default_qat_qconfig
# activation: FakeQuantize.with_args(observer=MinMaxObserver, ...)
# weight: FakeQuantize.with_args(observer=MinMaxObserver, ...)

# For dynamic quantization
default_dynamic_qconfig = quant.default_dynamic_qconfig
# weight: MinMaxObserver.with_args(dtype=torch.qint8)

# For ARM CPUs (QNNPACK backend)
# Use 'qnnpack' instead of 'fbgemm' in get_default_qconfig

# Float16 dynamic quantization
float16_dynamic_qconfig = quant.float16_dynamic_qconfig
# weight: dtype=torch.float16

# Per-channel quantization (better accuracy for LLMs)
per_channel_qconfig = quant.QConfig(
    activation=quant.MinMaxObserver.with_args(
        dtype=torch.quint8, qscheme=torch.per_tensor_affine
    ),
    weight=quant.MinMaxObserver.with_args(
        dtype=torch.qint8, qscheme=torch.per_channel_symmetric
    ),
)
```

### Custom QConfig

```python
# Create custom QConfig
custom_qconfig = quant.QConfig(
    activation=quant.HistogramObserver.with_args(
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=True,
    ),
    weight=quant.PerChannelMinMaxObserver.with_args(
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        ch_axis=0,  # output channel axis
    ),
)
```

---

## Observers

Observers collect statistics about tensor values during calibration to determine optimal quantization parameters (scale and zero_point).

### MinMaxObserver

```python
# Records min and max values
observer = torch.quantization.MinMaxObserver(
    dtype=torch.quint8,
    qscheme=torch.per_tensor_affine,
    reduce_range=False,    # reduce range for better accuracy on some hardware
    quant_min=None,        # override minimum quantized value
    quant_max=None,        # override maximum quantized value
    eps=1e-5,              # minimum scale
)

# Usage
x = torch.randn(100)
observer(x)                 # observe (record statistics)
scale, zero_point = observer.calculate_qparams()
print(f"Scale: {scale}, Zero point: {zero_point}")
```

### MovingAverageMinMaxObserver

```python
# Exponential moving average of min/max
observer = torch.quantization.MovingAverageMinMaxObserver(
    dtype=torch.quint8,
    qscheme=torch.per_tensor_affine,
    averaging_constant=0.01,  # EMA smoothing factor
)

# Better for streaming/online calibration
for batch in dataloader:
    observer(batch)
```

### HistogramObserver

```python
# Collects histogram of values for more accurate quantization
observer = torch.quantization.HistogramObserver(
    dtype=torch.quint8,
    qscheme=torch.per_tensor_affine,
    bins=2048,              # number of histogram bins
    upsample_rate=128,      # upsampling rate for calibration
)

# More accurate but slower than MinMaxObserver
# Best for post-training static quantization
```

### PerChannelMinMaxObserver

```python
# Per-channel (per-row/column) min/max observation
observer = torch.quantization.PerChannelMinMaxObserver(
    dtype=torch.qint8,
    qscheme=torch.per_channel_symmetric,
    ch_axis=0,              # channel axis (typically output dim for weights)
    reduce_range=False,
)

# Returns one scale and zero_point per channel
# Better accuracy for weights with different value ranges per channel
```

### MovingAveragePerChannelMinMaxObserver

```python
# EMA per-channel observer
observer = torch.quantization.MovingAveragePerChannelMinMaxObserver(
    dtype=torch.qint8,
    qscheme=torch.per_channel_symmetric,
    ch_axis=0,
    averaging_constant=0.01,
)
```

### NoopObserver

```python
# Pass-through observer (no observation)
# Used for layers that should not be quantized
observer = torch.quantization.NoopObserver(dtype=torch.float32)
```

### Observer Comparison

| Observer | Speed | Accuracy | Use Case |
|----------|-------|----------|----------|
| MinMaxObserver | Fast | Good | Quick calibration |
| MovingAverageMinMaxObserver | Fast | Good | Online/streaming |
| HistogramObserver | Slow | Best | Offline calibration |
| PerChannelMinMaxObserver | Medium | Better | Per-channel weights |
| NoopObserver | N/A | N/A | Skip quantization |

---

## FakeQuantize

FakeQuantize modules simulate quantization during QAT by quantizing and dequantizing tensors (fake quantization), allowing the model to learn to compensate for quantization error.

### FakeQuantizeMinMaxObserver

```python
fake_quant = torch.quantization.FakeQuantize(
    observer=torch.quantization.MinMaxObserver,
    quant_min=None,
    quant_max=None,
    observer_kwargs={},
)

# During training, this:
# 1. Observes the tensor (collects min/max)
# 2. Quantizes: Q(x) = clamp(round(x/scale + zero_point), quant_min, quant_max)
# 3. Dequantizes: x_hat = (Q(x) - zero_point) * scale
# 4. Uses STE (Straight-Through Estimator) for backward: grad(x_hat) = grad(x)

# Usage
x = torch.randn(4, requires_grad=True)
x_hat = fake_quant(x)  # fake quantized, but still float
loss = x_hat.sum()
loss.backward()  # gradients pass through (STE)
```

### FixedQParamsFakeQuantize

```python
# Fake quantize with fixed (pre-determined) scale and zero_point
fake_quant = torch.quantization.FixedQParamsFakeQuantize(
    scale=0.1,
    zero_point=128,
    dtype=torch.quint8,
)
```

---

## torch.quantization.prepare

```python
torch.quantization.prepare(
    model: torch.nn.Module,
    qconfig: Optional[QConfig] = None,
    inplace: bool = False,
    allow_list: Optional[Set[type]] = None,
    observer_non_leaf_module_list: Optional[List[type]] = None,
    prepare_custom_config_dict: Optional[Dict] = None,
) -> torch.nn.Module
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `nn.Module` | Model to prepare |
| `qconfig` | `QConfig` | Quantization config (overrides model.qconfig) |
| `inplace` | `bool` | Modify in-place |
| `allow_list` | `Set` | Additional module types to quantize |
| `prepare_custom_config_dict` | `Dict` | Custom configuration |

### What prepare() Does

```python
# Before prepare():
model = nn.Sequential(
    nn.Conv2d(3, 64, 3),
    nn.ReLU(),
    nn.Linear(64, 10),
)

# After prepare():
# Observers are inserted after each quantizable module
model_prepared = nn.Sequential(
    nn.Conv2d(3, 64, 3),
    torch.quantization.MinMaxObserver(),  # inserted
    nn.ReLU(),
    torch.quantization.MinMaxObserver(),  # inserted
    nn.Linear(64, 10),
    torch.quantization.MinMaxObserver(),  # inserted
)
```

---

## torch.quantization.convert

```python
torch.quantization.convert(
    model: torch.nn.Module,
    mapping: Optional[Dict[type, type]] = None,
    inplace: bool = False,
    remove_qconfig: bool = True,
    convert_custom_config_dict: Optional[Dict] = None,
) -> torch.nn.Module
```

### What convert() Does

```python
# After calibration with prepare():
# model has observers with collected statistics

# After convert():
# Observers are replaced with quantized modules
model_quantized = nn.Sequential(
    torch.nn.quantized.Conv2d(3, 64, 3),  # quantized conv
    torch.nn.quantized.ReLU(),             # quantized relu
    torch.nn.quantized.Linear(64, 10),     # quantized linear
    # QuantStub/DeQuantStub inserted at input/output
)
```

### Module Mapping

```python
# Default mapping from float to quantized modules:
{
    nn.Linear: nnq.Linear,           # -> quantized Linear
    nn.Conv2d: nnq.Conv2d,           # -> quantized Conv2d
    nn.ReLU: nnq.ReLU,               # -> quantized ReLU
    nn.BatchNorm2d: nn.Identity,     # fused into conv
    QuantStub: nnq.Quantize,         # -> quantize input
    DeQuantStub: nnq.DeQuantize,     # -> dequantize output
}
```

---

## Quantization Stubs

### QuantStub and DeQuantStub

```python
import torch.quantization as quant

class QuantizableModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = quant.QuantStub()    # marks input quantization point
        self.conv = nn.Conv2d(3, 64, 3)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(64, 10)
        self.dequant = quant.DeQuantStub()  # marks output dequantization point

    def forward(self, x):
        x = self.quant(x)      # quantize input (during inference)
        x = self.conv(x)
        x = self.relu(x)
        x = self.fc(x)
        x = self.dequant(x)    # dequantize output (during inference)
        return x

model = QuantizableModel()
model.qconfig = quant.get_default_qconfig('fbgemm')
quant.prepare(model, inplace=True)
# ... calibrate ...
quant.convert(model, inplace=True)
```

---

## torch.quantization.fuse_modules

Module fusing combines multiple modules into a single optimized module, reducing quantization error and improving inference speed.

### Signature

```python
torch.quantization.fuse_modules(
    model: torch.nn.Module,
    modules_to_fuse: List[str],      # module names to fuse
    inplace: bool = False,
    fuser_func: Optional[Callable] = None,
    fuse_custom_config_dict: Optional[Dict] = None,
) -> torch.nn.Module
```

### Supported Fusions

```python
# Common fusions:
# Conv2d + BatchNorm2d -> ConvBn2d
# Conv2d + BatchNorm2d + ReLU -> ConvBnReLU2d
# Conv2d + ReLU -> ConvReLU2d
# Linear + ReLU -> LinearReLU
# Linear + BatchNorm1d -> LinearBn1d
# Linear + BatchNorm1d + ReLU -> LinearBnReLU1d

# Example: fuse Conv+BN+ReLU
model = torchvision.models.resnet18(pretrained=True)

# Fuse specific modules by name
fused_model = quant.fuse_modules(model, [
    ['conv1', 'bn1', 'relu'],          # first layer
    ['layer1.0.conv1', 'layer1.0.bn1'],
    ['layer1.0.conv2', 'layer1.0.bn2'],
    ['layer2.0.conv1', 'layer2.0.bn1'],
])

# Fusing replaces the original modules:
# conv1 + bn1 + relu -> nn.intrinsic.ConvBnReLU2d
# The fused module combines batch norm into conv weights
```

### Automatic Fusing

```python
# Fuse all eligible modules automatically
model = torch.quantization.fuse_modules(model, [
    ['conv1', 'bn1', 'relu'],
    # list all conv-bn-relu patterns
], inplace=True)

# Or use the automatic fuser
from torch.quantization import fuse_modules
model.eval()
fuse_modules(model, inplace=True)  # auto-detect and fuse
```

---

## Per-Channel vs Per-Tensor Quantization

### Per-Tensor Quantization

```python
# Single scale and zero_point for entire tensor
# Simpler, faster, but less accurate for tensors with wide value ranges
qconfig_per_tensor = quant.QConfig(
    activation=quant.MinMaxObserver.with_args(dtype=torch.quint8),
    weight=quant.MinMaxObserver.with_args(dtype=torch.qint8),
)
# All elements in a weight tensor share the same scale/zero_point
```

### Per-Channel Quantization

```python
# Separate scale and zero_point for each output channel
# More accurate, especially for models with varying channel magnitudes
qconfig_per_channel = quant.QConfig(
    activation=quant.MinMaxObserver.with_args(dtype=torch.quint8),
    weight=quant.PerChannelMinMaxObserver.with_args(
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        ch_axis=0,
    ),
)
# Each output channel has its own scale/zero_point
```

### Comparison

| Aspect | Per-Tensor | Per-Channel |
|--------|------------|-------------|
| Granularity | Single scale for entire tensor | Scale per output channel |
| Accuracy | Lower | Higher |
| Speed | Faster (simpler) | Slightly slower |
| Supported backends | All | FBGEMM, QNNPACK |
| Recommended for | Simple models | LLMs, Transformers, modern CNNs |

---

## Static vs Dynamic Quantization

### Static Quantization

```python
# Both weights AND activations are quantized
# Requires calibration data to determine activation quantization parameters
# Best inference speed

model.qconfig = quant.get_default_qconfig('fbgemm')
quant.prepare(model, inplace=True)
# Calibrate with representative data
for batch in calibration_loader:
    model(batch)
quant.convert(model, inplace=True)
```

### Dynamic Quantization

```python
# Weights are quantized statically
# Activations are quantized dynamically during inference
# No calibration needed
# Good for NLP models (LSTM, Transformer)

quantized_model = quant.quantize_dynamic(
    model,
    qconfig_spec={nn.Linear, nn.LSTM},
    dtype=torch.qint8,
)
```

### Comparison

| Aspect | Static | Dynamic |
|--------|--------|---------|
| Weight quantization | Pre-quantized | Pre-quantized |
| Activation quantization | Pre-quantized (calibrated) | On-the-fly |
| Calibration required | Yes | No |
| Inference speed | Fastest | Fast |
| Accuracy | Can be higher with good calibration | Robust (adapts to input) |
| Best for | CNNs, vision models | NLP models, RNNs |

---

## Post-Training Quantization Workflow

```python
import torch
import torch.quantization as quant

# Step 1: Get a pretrained model
model = torchvision.models.mobilenet_v2(pretrained=True)
model.eval()

# Step 2: Fuse modules
model = quant.fuse_modules(model, [
    ['features.0.0', 'features.0.1', 'features.0.2'],
    ['features.1.conv.0', 'features.1.conv.1'],
    # ... fuse all conv-bn-relu patterns
])

# Step 3: Set QConfig
model.qconfig = quant.get_default_qconfig('fbgemm')

# Step 4: Insert stubs (if not already present)
# For models without explicit quant/dequant:
# Add QuantStub at input, DeQuantStub at output

# Step 5: Prepare (insert observers)
quant.prepare(model, inplace=True)

# Step 6: Calibrate
with torch.no_grad():
    for i, (images, _) in enumerate(calibration_loader):
        model(images)
        if i >= 100:  # 100 batches usually sufficient
            break

# Step 7: Convert
quant.convert(model, inplace=True)

# Step 8: Evaluate
accuracy = evaluate(model, test_loader)
print(f"Quantized model accuracy: {accuracy:.2f}%")

# Step 9: Compare size
original_size = get_model_size(original_model)
quantized_size = get_model_size(model)
print(f"Size reduction: {original_size/quantized_size:.1f}x")
```

---

## Quantization-Aware Training Workflow

```python
import torch
import torch.quantization as quant

# Step 1: Get a pretrained model
model = torchvision.models.resnet18(pretrained=True)

# Step 2: Fuse modules
model = quant.fuse_modules(model, [
    ['conv1', 'bn1', 'relu'],
    # ... fuse all patterns
])

# Step 3: Set QAT QConfig (with FakeQuantize)
model.qconfig = quant.get_default_qat_qconfig('fbgemm')

# Step 4: Prepare QAT (insert fake quantization)
quant.prepare_qat(model, inplace=True)

# Step 5: Fine-tune
model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(10):
    for images, targets in train_loader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

    # Freeze observers and BN after warmup
    if epoch >= 3:
        model.apply(quant.disable_observer)
        model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

    # Evaluate
    model.eval()
    acc = evaluate(model, test_loader)
    print(f"Epoch {epoch}: accuracy = {acc:.2f}%")

# Step 6: Convert to quantized model
model.eval()
quant.convert(model, inplace=True)

# Step 7: Final evaluation
accuracy = evaluate(model, test_loader)
```

---

## torch.ao.ns: Numeric Suite

The numeric suite compares float and quantized model outputs to identify where quantization causes the most error.

```python
import torch.ao.ns.numeric_suite as ns

# Compare float and quantized models
float_model = torchvision.models.resnet18(pretrained=True)
float_model.eval()

quantized_model = ...  # quantized version

# Create comparison
comp = ns.FloatFunctional()

# Compare module-level outputs
compare_dict = ns.compare_model_outputs(
    float_model,
    quantized_model,
    input_data,  # representative input
)

# Get per-layer statistics
for name, (float_out, quant_out) in compare_dict.items():
    # Compute error metrics
    error = ns.compute_error(float_out, quant_out)
    print(f"{name}: SQNR = {error:.2f} dB")

# Compare weights
wt_compare_dict = ns.compare_model_weights(float_model, quantized_model)
for name, (float_w, quant_w) in wt_compare_dict.items():
    print(f"Weight {name}: {ns.compute_error(float_w, quant_w):.2f} dB")
```

---

## Quantized Tensor Operations

```python
# Create a quantized tensor
x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

# Quantize
x_q = torch.quantize_per_tensor(x, scale=0.1, zero_point=128, dtype=torch.quint8)
print(x_q)           # quantized tensor
print(x_q.int_repr())  # integer representation: tensor([138, 148, 158, 168, 178])

# Dequantize
x_dq = x_q.dequantize()
print(x_dq)  # tensor([1.0, 2.0, 3.0, 4.0, 5.0]) approximately

# Per-channel quantization
x = torch.randn(4, 8)
scales = torch.tensor([0.1, 0.05, 0.2, 0.15])
zero_points = torch.tensor([128, 100, 128, 120])
x_q = torch.quantize_per_channel(x, scales, zero_points, axis=0, dtype=torch.qint8)

# Quantized tensor operations
q_tensor1 = torch.quantize_per_tensor(torch.randn(3, 4), 0.1, 128, torch.quint8)
q_tensor2 = torch.quantize_per_tensor(torch.randn(3, 4), 0.1, 128, torch.quint8)

# Operations on quantized tensors
result = torch.ops.quantized.add(q_tensor1, q_tensor2, scale=0.1, zero_point=128)
result = torch.ops.quantized.mul(q_tensor1, q_tensor2, scale=0.1, zero_point=128)
result = torch.ops.quantized.linear(torch.quantize_per_tensor(torch.randn(3, 4), 0.1, 128, torch.quint8),
                                     torch.quantize_per_tensor(torch.randn(5, 4), 0.05, 0, torch.qint8),
                                     torch.quantize_per_tensor(torch.randn(5), 0.02, 128, torch.quint8),
                                     scale=0.1, zero_point=128)
```

---

## Backend Configuration

```python
# Different hardware backends have different quantization capabilities

# x86 CPUs (FBGEMM)
torch.backends.quantized.engine = 'fbgemm'

# ARM CPUs (QNNPACK)
torch.backends.quantized.engine = 'qnnpack'

# Get default qconfig for backend
qconfig = torch.quantization.get_default_qconfig('fbgemm')
qconfig = torch.quantization.get_default_qconfig('qnnpack')

# Backend-specific qconfigs
# FBGEMM: supports per-channel weight quantization
# QNNPACK: optimized for mobile CPUs
```

---

## Summary

PyTorch's quantization provides:

1. **Quantized dtypes**: qint8, quint8, qint32, quint4x2, float8 variants
2. **Post-training static quantization**: Best speed, requires calibration
3. **Dynamic quantization**: No calibration needed, good for NLP
4. **Quantization-aware training**: Best accuracy, simulates quantization during training
5. **Observers**: MinMax, MovingAverage, Histogram, PerChannel variants
6. **FakeQuantize**: Simulates quantization with STE gradients
7. **Module fusing**: Combines Conv+BN+ReLU for better quantization
8. **Per-channel quantization**: Separate scales per output channel
9. **Numeric suite**: Compare float vs quantized accuracy per layer
10. **Backend support**: FBGEMM (x86), QNNPACK (ARM), and custom backends
