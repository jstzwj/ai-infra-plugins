# PyTorch - Chapter 51: Mobile Deployment

This reference covers deploying PyTorch models on mobile devices.

---

## 51.1 Model Preparation

```python
# Script the model
scripted = torch.jit.script(model)
traced = torch.jit.trace(model, example_input)

# Optimize for mobile
from torch.utils.mobile_optimizer import optimize_for_mobile
optimized = optimize_for_mobile(scripted)

# Save
optimized.save("model.ptl")  # .ptl for mobile
optimized._save_for_lite_interpreter("model.ptl")
```

---

## 51.2 Optimization Options

```python
# Selective build: only include needed operators
# Reduces binary size significantly

# Quantization for mobile
quantized = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
```

---

## 51.3 Platform Integration

**Android**: Use PyTorch Mobile library (org.pytorch:pytorch_mobile_lite)
**iOS**: Use LibTorch Lite pod

```java
// Android example
Module module = Module.load("model.ptl");
Tensor input = Tensor.fromBlob(floatData, shape);
Tensor output = module.forward(IValue.from(input)).toTensor();
```

---

## 51.4 Lite Interpreter

The Lite Interpreter is a reduced version of the TorchScript interpreter:
- Smaller binary size (~50% reduction)
- Lower memory overhead
- Fewer operator registrations
