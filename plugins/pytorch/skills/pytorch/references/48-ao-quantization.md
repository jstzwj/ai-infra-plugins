# torch.ao Quantization (PyTorch 2.0+ API)

## Overview

The `torch.ao` (Architecture and Optimization) module is the modern home for PyTorch's quantization APIs. Starting from PyTorch 2.0, the recommended quantization flow uses the PT2E (PyTorch 2 Export) API with `torch.compile`, which provides better integration with the compiler, supports dynamic shapes, and enables advanced quantization techniques like GPTQ-style weight quantization.

**Source location**: `torch/ao/quantization/`, `torch/ao/quantization/pt2e/`

---

## torch.ao Module Overview

```
torch/ao/
  quantization/
    pt2e/               # PyTorch 2 Export quantization (recommended)
      quantize_pt2e.py  # Main PT2E quantization API
      prepare_pt2e.py   # Prepare for PT2E quantization
      convert_pt2e.py   # Convert to quantized model
    quantizer/          # Custom quantizer framework
      x86inductor.py    # x86 Inductor quantizer
      arm.py            # ARM quantizer
    observer/           # Observer modules
    fake_quantize/      # Fake quantize modules
    qconfig_mapping.py  # QConfig mapping utilities
    backend_config/     # Backend configuration
    fx/                 # FX-based quantization (PyTorch 1.x style)
  nn/
    quantized/          # Quantized nn modules
    qat/                # QAT modules
    intrinsic/          # Fused modules
  ns/                   # Numeric suite
  pruning/              # Model pruning (if present)
```

---

## torch.ao.quantization.quantize_pt2e (PyTorch 2.0 Export Quantization)

The PT2E API quantizes models captured via `torch.export` or `torch.compile`, enabling quantization of models with dynamic shapes, custom ops, and complex control flow.

### Basic PT2E Workflow

```python
import torch
import torch.ao.quantization as quant

# Step 1: Create or load a model
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))

model = M()
model.eval()

# Step 2: Export the model (capture the computation graph)
exported_model = torch.export.export(model, (torch.randn(1, 10),))

# Step 3: Prepare for quantization
# Create a quantizer
from torch.ao.quantization.quantizer.x86inductor_quantizer import X86InductorQuantizer
quantizer = X86InductorQuantizer()

# Prepare inserts observers
prepared_model = quant.prepare_pt2e(exported_model, quantizer)

# Step 4: Calibrate
with torch.no_grad():
    for _ in range(10):
        prepared_model(torch.randn(1, 10))

# Step 5: Convert to quantized model
quantized_model = quant.convert_pt2e(prepared_model)

# Step 6: Run inference
output = quantized_model(torch.randn(1, 10))
```

### prepare_pt2e

```python
torch.ao.quantization.quantize_pt2e.prepare_pt2e(
    model: torch.export.ExportedProgram,
    quantizer: torch.ao.quantization.quantizer.Quantizer,
) -> torch.export.ExportedProgram
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `ExportedProgram` | Exported model from `torch.export` |
| `quantizer` | `Quantizer` | Backend-specific quantizer |

### convert_pt2e

```python
torch.ao.quantization.quantize_pt2e.convert_pt2e(
    model: torch.export.ExportedProgram,
    use_reference_representation: bool = False,
) -> torch.export.ExportedProgram
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `ExportedProgram` | Prepared model with observers |
| `use_reference_representation` | `bool` | Use reference quantized ops (for testing) |

---

## torch.ao.quantization.quantizer: Custom Quantizers

Quantizers define how to quantize a model for a specific backend. They specify which ops to quantize, what quantization parameters to use, and how to insert observers.

### Quantizer Base Class

```python
from torch.ao.quantization.quantizer import Quantizer, QuantizationSpec, QuantizationAnnotation

class Quantizer(ABC):
    """Base class for all quantizers."""

    @abstractmethod
    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """Annotate the model with quantization specifications.
        This decides which nodes should be quantized and how."""
        pass

    @abstractmethod
    def validate(self, model: torch.fx.GraphModule) -> None:
        """Validate that the annotated model is quantizable."""
        pass
```

### X86InductorQuantizer

```python
from torch.ao.quantization.quantizer.x86inductor_quantizer import (
    X86InductorQuantizer,
    X86InductorQuantizerConfig,
)

# Default configuration
quantizer = X86InductorQuantizer()

# Custom configuration
config = X86InductorQuantizerConfig(
    dtype=torch.int8,               # quantized dtype
    is_per_channel=True,            # per-channel weight quantization
    is_dynamic=False,               # static quantization
    is_qat=False,                   # not quantization-aware training
)
quantizer = X86InductorQuantizer().set_config(config)
```

### ARMQuantizer

```python
from torch.ao.quantization.quantizer.arm_quantizer import ARMQuantizer

quantizer = ARMQuantizer()
# Optimized for ARM CPU inference (mobile devices)
```

### Custom Quantizer

```python
from torch.ao.quantization.quantizer import (
    Quantizer,
    QuantizationSpec,
    QuantizationAnnotation,
    SharedQuantizationSpec,
)
import torch.fx

class MyQuantizer(Quantizer):
    def annotate(self, gm: torch.fx.GraphModule):
        # Define quantization specs
        act_spec = QuantizationSpec(
            dtype=torch.int8,
            quant_min=-128,
            quant_max=127,
            qscheme=torch.per_tensor_symmetric,
            observer_class=torch.ao.quantization.MinMaxObserver,
        )

        weight_spec = QuantizationSpec(
            dtype=torch.int8,
            quant_min=-128,
            quant_max=127,
            qscheme=torch.per_channel_symmetric,
            observer_class=torch.ao.quantization.PerChannelMinMaxObserver,
            ch_axis=0,
        )

        # Annotate nodes
        for node in gm.graph.nodes:
            if node.op == 'call_function':
                if node.target == torch.nn.functional.linear:
                    # Annotate input activation
                    input_node = node.args[0]
                    input_node.meta['quantization'] = QuantizationAnnotation(
                        input_qspec_map={input_node: act_spec},
                        output_qspec=act_spec,
                    )

                    # Annotate weight
                    weight_node = node.args[1]
                    weight_node.meta['quantization'] = QuantizationAnnotation(
                        input_qspec_map={weight_node: weight_spec},
                    )

        return gm

    def validate(self, gm: torch.fx.GraphModule):
        # Check that all annotated nodes are valid
        pass
```

---

## torch.ao.quantization.observer: All Observer Types

```python
from torch.ao.quantization.observer import (
    ObserverBase,
    MinMaxObserver,
    MovingAverageMinMaxObserver,
    HistogramObserver,
    PerChannelMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    NoopObserver,
    PlaceholderObserver,
    RecordingObserver,
    ReLUOutputObserver,
    DebugObserver,
    PartialQuantStubObserver,
)

# MinMaxObserver: records min and max tensor values
obs = MinMaxObserver(
    dtype=torch.quint8,
    qscheme=torch.per_tensor_affine,
    reduce_range=False,
    quant_min=None,
    quant_max=None,
)

# MovingAverageMinMaxObserver: EMA of min/max
obs = MovingAverageMinMaxObserver(
    dtype=torch.quint8,
    averaging_constant=0.01,
)

# HistogramObserver: histogram-based calibration
obs = HistogramObserver(
    dtype=torch.quint8,
    bins=2048,
    upsample_rate=128,
)

# PerChannelMinMaxObserver: per-channel min/max
obs = PerChannelMinMaxObserver(
    dtype=torch.qint8,
    qscheme=torch.per_channel_symmetric,
    ch_axis=0,
)

# RecordingObserver: records all observed values
obs = RecordingObserver()
for _ in range(10):
    obs(torch.randn(5))
print(obs.get_tensor_value())  # all recorded values

# PlaceholderObserver: does nothing
obs = PlaceholderObserver()

# DebugObserver: prints observed values
obs = DebugObserver(custom_processor=lambda x: print(f"observed: {x.shape}"))
```

---

## torch.ao.quantization.fake_quantize: All Fake Quantize Modules

```python
from torch.ao.quantization.fake_quantize import (
    FakeQuantize,
    FixedQParamsFakeQuantize,
    FusedMovingAvgObsFakeQuantize,
    default_fused_act_fake_quant,
    default_fused_wt_fake_quant,
    default_weight_fake_quant,
    default_dynamic_fake_quant,
)

# Standard FakeQuantize (uses observer for scale/zp)
fq = FakeQuantize(
    observer=MinMaxObserver,
    quant_min=-128,
    quant_max=127,
    dtype=torch.qint8,
    qscheme=torch.per_tensor_symmetric,
)

# FixedQParamsFakeQuantize (predetermined scale/zp)
fq = FixedQParamsFakeQuantize(
    scale=0.1,
    zero_point=0,
    dtype=torch.qint8,
)

# FusedMovingAvgObsFakeQuantize (fused observer + fake quant)
fq = FusedMovingAvgObsFakeQuantize(
    observer=MovingAverageMinMaxObserver,
    quant_min=0,
    quant_max=255,
    dtype=torch.quint8,
    qscheme=torch.per_tensor_affine,
    eps=1e-5,
    reduce_range=False,
)

# Usage in QAT
x = torch.randn(4, 4, requires_grad=True)
x_fq = fq(x)        # fake quantized (still float, but with quantization simulation)
loss = x_fq.sum()
loss.backward()      # gradients pass through using STE
```

---

## torch.ao.quantization.qconfig_mapping

`QConfigMapping` allows per-module quantization configuration.

```python
from torch.ao.quantization.qconfig_mapping import QConfigMapping
from torch.ao.quantization import QConfig, MinMaxObserver, HistogramObserver

# Create a mapping
qconfig_mapping = QConfigMapping()
qconfig_mapping.set_global(torch.ao.quantization.default_qconfig)

# Per-module-type configuration
qconfig_mapping.set_module_type(torch.nn.Linear, QConfig(
    activation=HistogramObserver.with_args(dtype=torch.quint8),
    weight=MinMaxObserver.with_args(dtype=torch.qint8),
))

# Per-module-name configuration
qconfig_mapping.set_module_name("head", QConfig(
    activation=MinMaxObserver.with_args(dtype=torch.quint8),
    weight=MinMaxObserver.with_args(dtype=torch.qint8),
))

# Per-module-name-object configuration
qconfig_mapping.set_module_name_regex("conv.*", custom_conv_qconfig)

# Per-layer configuration
qconfig_mapping.set_object_type(torch.nn.functional.linear, custom_linear_qconfig)
```

---

## torch.ao.quantization.backend_config

Backend configuration defines what operations are supported by a specific backend.

```python
from torch.ao.quantization.backend_config import (
    BackendConfig,
    BackendPatternConfig,
    DTypeConfig,
    ObservationType,
)

# Pre-built backend configs
from torch.ao.quantization.backend_config.fbgemm import get_fbgemm_backend_config
from torch.ao.quantization.backend_config.qnnpack import get_qnnpack_backend_config
from torch.ao.quantization.backend_config.x86 import get_x86_backend_config
from torch.ao.quantization.backend_config.arm import get_arm_backend_config

# Get a backend config
backend_config = get_fbgemm_backend_config()

# Custom backend config
dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float32,
)

pattern_config = BackendPatternConfig(
    pattern=torch.nn.Linear,
    observation_type=ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
    dtype_configs=[dtype_config],
    root_module=torch.nn.Linear,
    qat_module=torch.ao.nn.qat.Linear,
    reference_quantized_module=torch.ao.nn.quantized.reference.Linear,
)

backend_config = BackendConfig()
backend_config.set_backend_pattern_config(pattern_config)
```

---

## torch.ao.nn: Quantized NN Modules

### Quantized Modules

```python
from torch.ao.nn import quantized as nnq

# Quantized Linear
linear = nnq.Linear(10, 5)
# Must set quantized weights explicitly or via convert()

# Quantized Conv2d
conv = nnq.Conv2d(3, 16, 3, padding=1)

# Quantized ReLU
relu = nnq.ReLU()

# Quantized Functional
import torch.ao.nn.quantized.functional as Fq
output = Fq.relu(input, scale=0.1, zero_point=128)
```

### QAT Modules

```python
from torch.ao.nn import qat

# QAT Linear (with fake quantization)
linear = qat.Linear(10, 5, qconfig=torch.ao.quantization.get_default_qat_qconfig('fbgemm'))

# QAT Conv2d
conv = qat.Conv2d(3, 16, 3, qconfig=torch.ao.quantization.get_default_qat_qconfig('fbgemm'))

# QAT Embedding
embedding = qat.Embedding(10000, 300)
```

### Reference Quantized Modules

```python
from torch.ao.nn import quantized as nnq_ref

# Reference modules for testing and debugging
# They implement quantized operations using floating-point arithmetic
# Useful for verifying correctness of optimized quantized implementations
linear_ref = nnq_ref.Linear(10, 5)
conv_ref = nnq_ref.Conv2d(3, 16, 3)
```

---

## torch.ao.ns: Numeric Suite

```python
import torch.ao.ns.numeric_suite as ns

# Compare model outputs at each layer
results = ns.compare_model_outputs(
    float_model,
    quantized_model,
    input_tensor,
    # atol=1e-5, rtol=1e-5,
)

# Compare model weights
weight_results = ns.compare_model_weights(float_model, quantized_model)

# Compute Signal-to-Quantization-Noise Ratio
for name, (float_val, quant_val) in results.items():
    sqnr = ns.compute_error(float_val, quant_val)
    print(f"{name}: SQNR = {sqnr:.2f} dB")

# Shadow model: attach quantized modules alongside float modules
shadow_model = ns.ShadowModel(
    float_model,
    quantized_model,
)
```

---

## torch.ao.quantization.fx: FX-Based Quantization

The FX-based quantization flow uses `torch.fx` symbolic tracing to prepare, calibrate, and convert models. This is the PyTorch 1.x approach, still supported but superseded by PT2E for new work.

```python
import torch
import torch.ao.quantization as quant
from torch.ao.quantization.fx import prepare_fx, convert_fx

# FX-based static quantization
model = torchvision.models.resnet18(pretrained=True)
model.eval()

# Prepare
qconfig_mapping = quant.QConfigMapping()
qconfig_mapping.set_global(quant.get_default_qconfig('fbgemm'))

prepared_model = prepare_fx(model, qconfig_mapping, example_inputs=(torch.randn(1, 3, 224, 224),))

# Calibrate
with torch.no_grad():
    for _ in range(100):
        prepared_model(torch.randn(1, 3, 224, 224))

# Convert
quantized_model = convert_fx(prepared_model)

# Inference
output = quantized_model(torch.randn(1, 3, 224, 224))
```

### FX QAT

```python
from torch.ao.quantization.fx import prepare_qat_fx, convert_fx

model = torchvision.models.resnet18(pretrained=True)
model.train()

qconfig_mapping = quant.QConfigMapping()
qconfig_mapping.set_global(quant.get_default_qat_qconfig('fbgemm'))

prepared_model = prepare_qat_fx(model, qconfig_mapping, example_inputs=(torch.randn(1, 3, 224, 224),))

# Train with fake quantization
for epoch in range(10):
    for images, targets in train_loader:
        output = prepared_model(images)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Convert
prepared_model.eval()
quantized_model = convert_fx(prepared_model)
```

---

## GPTQ-Style Quantization

GPTQ (Generative Pre-trained Transformer Quantization) is a post-training quantization method specifically designed for large language models. It quantizes weights based on the Hessian (second-order information) of the loss.

```python
# GPTQ-style quantization concept
# PyTorch does not have built-in GPTQ, but supports the building blocks

import torch
import torch.ao.quantization as quant

# Conceptual GPTQ process:
# 1. For each layer, compute the Hessian of the loss w.r.t. weights
# 2. Quantize weights column-by-column, updating remaining columns
#    based on the Hessian to compensate for quantization error
# 3. This minimizes the layer-wise reconstruction error

# Building blocks available in PyTorch:
# 1. Per-channel quantization
qconfig = quant.QConfig(
    activation=quant.NoopObserver,
    weight=quant.PerChannelMinMaxObserver.with_args(
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
    ),
)

# 2. Custom quantization with specific scale/zp
def gptq_quantize_weight(weight, scale, zero_point, hessian_inv):
    # Quantize using GPTQ formula
    quantized = torch.quantize_per_channel(
        weight, scale, zero_point, axis=0, dtype=torch.qint8
    )
    # Compensate remaining weights
    # error = (quantized.dequantize() - weight)
    # weight_remaining -= (error @ hessian_inv) / diag(hessian_inv)
    return quantized
```

---

## 4-Bit Quantization

```python
import torch

# PyTorch supports sub-byte quantization types
# quint4x2: 4-bit unsigned (packed in pairs of 8 bits)
# quint2x4: 2-bit unsigned (packed in quads of 8 bits)

# 4-bit quantization (NF4 format used in QLoRA)
def quantize_4bit(weight, quant_type="nf4"):
    """Quantize weight to 4-bit format."""
    if quant_type == "nf4":
        # NF4: NormalFloat 4-bit
        # Quantiles of a normal distribution for optimal information density
        nf4_values = torch.tensor([
            -1.0, -0.6962, -0.5251, -0.3949, -0.2844,
            -0.1861, -0.0961, 0.0, 0.0961, 0.1861,
            0.2844, 0.3949, 0.5251, 0.6962, 1.0,
        ])

        # Normalize weight
        weight_normalized = weight / weight.abs().max()

        # Find nearest NF4 value
        indices = torch.argmin(
            torch.abs(weight_normalized.unsqueeze(-1) - nf4_values),
            dim=-1
        )
        return indices.to(torch.uint8), nf4_values

    elif quant_type == "fp4":
        # FP4: 4-bit floating point (E2M1 format)
        pass

# Packing 4-bit values into 8-bit storage
def pack_4bit(values):
    """Pack two 4-bit values into one uint8."""
    assert values.dtype == torch.uint8
    even = values[::2] & 0x0F
    odd = (values[1::2] & 0x0F) << 4
    return even | odd

def unpack_4bit(packed):
    """Unpack uint8 into two 4-bit values."""
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    return torch.stack([low, high], dim=-1).flatten()
```

---

## Custom Quantization Backends

### Creating a Custom Backend

```python
from torch.ao.quantization.quantizer import Quantizer, QuantizationSpec, QuantizationAnnotation
from torch.ao.quantization.backend_config import BackendConfig, BackendPatternConfig, DTypeConfig

class MyBackendQuantizer(Quantizer):
    """Custom quantizer for a specific hardware backend."""

    def annotate(self, gm):
        # Define quantization specifications
        act_spec = QuantizationSpec(
            dtype=torch.int8,
            quant_min=-128,
            quant_max=127,
            qscheme=torch.per_tensor_symmetric,
        )

        weight_spec = QuantizationSpec(
            dtype=torch.int8,
            quant_min=-128,
            quant_max=127,
            qscheme=torch.per_channel_symmetric,
            ch_axis=0,
        )

        # Annotate graph nodes
        for node in gm.graph.nodes:
            if node.op == 'call_function':
                if node.target in self.supported_ops:
                    self._annotate_node(node, act_spec, weight_spec)

        return gm

    def validate(self, gm):
        pass

    @property
    def supported_ops(self):
        return {torch.nn.functional.linear, torch.nn.functional.conv2d}

    def _annotate_node(self, node, act_spec, weight_spec):
        input_node = node.args[0]
        weight_node = node.args[1]

        node.meta['quantization'] = QuantizationAnnotation(
            input_qspec_map={
                input_node: act_spec,
                weight_node: weight_spec,
            },
            output_qspec=act_spec,
        )
```

---

## Example: Quantize a Model with PT2E API

```python
import torch
import torch.ao.quantization as quant
from torch.ao.quantization.quantizer.x86inductor_quantizer import (
    X86InductorQuantizer,
    X86InductorQuantizerConfig,
)

# Model
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.relu = torch.nn.ReLU()
        self.fc = torch.nn.Linear(16 * 32 * 32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

model = SimpleModel()
model.eval()

# Step 1: Export the model
example_input = torch.randn(1, 3, 32, 32)
exported_model = torch.export.export(model, (example_input,))

# Step 2: Configure quantizer
quantizer = X86InductorQuantizer().set_config(
    X86InductorQuantizerConfig(
        dtype=torch.int8,
        is_per_channel=True,
        is_dynamic=False,
    )
)

# Step 3: Prepare
prepared_model = quant.prepare_pt2e(exported_model, quantizer)

# Step 4: Calibrate
calibration_data = torch.randn(100, 3, 32, 32)
with torch.no_grad():
    for i in range(100):
        prepared_model(calibration_data[i:i+1])

# Step 5: Convert
quantized_model = quant.convert_pt2e(prepared_model)

# Step 6: Run inference
test_input = torch.randn(1, 3, 32, 32)
output = quantized_model(test_input)

# Step 7: Compile for optimized execution
compiled_model = torch.compile(quantized_model)
output = compiled_model(test_input)

# Step 8: Compare sizes
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

print(f"Float model size: {count_parameters(model) * 4 / 1e6:.1f} MB")
print(f"Quantized model size: ~{count_parameters(model) * 1 / 1e6:.1f} MB (INT8)")
```

---

## PT2E with Dynamic Shapes

```python
import torch
from torch.ao.quantization.quantizer.x86inductor_quantizer import X86InductorQuantizer

# Model with dynamic batch dimension
class DynamicModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(64, 128)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

model = DynamicModel().eval()

# Export with dynamic batch dim
exported = torch.export.export(
    model,
    (torch.randn(1, 64),),
    dynamic_shapes={"x": {0: torch.export.Dim("batch")}},
)

# Quantize with PT2E (supports dynamic shapes)
from torch.ao.quantization import prepare_pt2e, convert_pt2e

quantizer = X86InductorQuantizer()
prepared = prepare_pt2e(exported, quantizer)

# Calibrate with various batch sizes
for batch_size in [1, 4, 8, 16]:
    prepared(torch.randn(batch_size, 64))

quantized = convert_pt2e(prepared)

# Inference with any batch size
for batch_size in [1, 5, 10, 32]:
    output = quantized(torch.randn(batch_size, 64))
    print(f"Batch {batch_size}: output shape {output.shape}")
```

---

## Summary

The modern `torch.ao` quantization API provides:

1. **PT2E API**: Quantization integrated with `torch.export` and `torch.compile`
2. **Quantizer framework**: `X86InductorQuantizer`, `ARMQuantizer`, custom quantizers
3. **Observers**: Complete set for min/max, histogram, per-channel observation
4. **FakeQuantize**: Modules for quantization-aware training simulation
5. **QConfigMapping**: Per-module quantization configuration
6. **BackendConfig**: Backend-specific quantization support definitions
7. **FX-based quantization**: Legacy flow still supported via `prepare_fx`/`convert_fx`
8. **Numeric Suite**: Float vs quantized comparison at each layer
9. **Sub-byte quantization**: 4-bit and 2-bit support
10. **Dynamic shapes**: Full support through PT2E API
