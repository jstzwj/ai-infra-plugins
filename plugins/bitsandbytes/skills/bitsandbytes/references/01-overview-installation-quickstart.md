# bitsandbytes: Overview, Installation, and Quick Start

## Version

**0.50.0.dev0**

## License

MIT License -- Copyright (c) Facebook, Inc. and its affiliates.

---

## What is bitsandbytes?

bitsandbytes is a library that provides **k-bit quantization primitives for PyTorch**. It integrates seamlessly into PyTorch models to reduce memory usage for both inference and training with minimal accuracy loss. The library is the foundational building block for efficient large language model (LLM) deployment and fine-tuning workflows used by Hugging Face Transformers, PEFT, Accelerate, TGI, and vLLM.

The library exposes three major capabilities:

### 1. 8-bit Optimizers

Drop-in replacements for standard PyTorch optimizers (Adam, AdamW, Lion, SGD, etc.) that store optimizer states in 8-bit precision using block-wise quantization. This reduces optimizer memory by roughly 75% (from 32-bit float32 states to 8-bit uint8 states) with negligible impact on training dynamics.

Available optimizers:
- `Adam8bit`, `AdamW8bit`, `PagedAdam8bit`, `PagedAdamW8bit`
- `Lion8bit`, `PagedLion8bit`
- `SGD8bit`, `Adagrad8bit`, `RMSprop8bit`
- `LAMB8bit`, `LARS8bit`
- `AdEMAMix8bit`, `PagedAdEMAMix8bit`

Each also has a 32-bit variant (`Adam32bit`, `AdamW32bit`, etc.) for selective per-parameter precision control.

### 2. LLM.int8() -- 8-bit Inference

The `Linear8bitLt` module implements the **LLM.int8()** algorithm ([paper](https://arxiv.org/abs/2208.07339)) for 8-bit quantized matrix multiplication. It quantizes weights to int8 offline and quantizes activations on the fly during forward passes. A mixed-precision decomposition (controlled by a threshold parameter) handles outlier activation features in fp16 to preserve accuracy.

### 3. QLoRA -- 4-bit Quantization

The `Linear4bit`, `LinearFP4`, and `LinearNF4` modules implement the **QLoRA** algorithm ([paper](https://arxiv.org/abs/2305.14314)) for 4-bit quantization of model weights. Two 4-bit data types are supported:
- **NF4** (NormalFloat4): optimal for normally-distributed neural network weights
- **FP4** (4-bit floating point): IEEE 754-like encoding with sign, exponent, and mantissa

An optional "double quantization" (compress_statistics) further quantizes the absmax scaling factors themselves.

---

## System Requirements

### Software Requirements

| Requirement | Minimum Version |
|-------------|-----------------|
| Python      | 3.10+           |
| PyTorch     | 2.3+            |
| NumPy       | Any compatible  |

### Hardware Accelerator Support Matrix

| Platform          | Backend  | Minimum HW                        | Recommended HW                            | Notes                                      |
|-------------------|----------|-----------------------------------|-------------------------------------------|--------------------------------------------|
| NVIDIA CUDA       | `cuda`   | SM 6.0 (Pascal)                   | SM 7.5+ (Turing, Ampere, Hopper, Blackwell) | Full feature support                       |
| AMD ROCm          | `rocm`   | CDNA: gfx90a, gfx942, gfx950      | RDNA: gfx1100+                            | Via HIP compatibility layer                |
| Intel XPU         | `xpu`    | Data Center Max, Arc A/B-Series   | Data Center Max                           | SYCL-based, uses MatMul8bitFp path        |
| Intel Gaudi       | `hpu`    | Gaudi2, Gaudi3                    | Gaudi3                                    | Requires habana_frameworks                 |
| Apple Metal       | `mps`    | Apple M1+                         | Apple M2+                                 | Limited feature set                        |
| CPU               | `cpu`    | x86-64 with AVX2                  | x86-64 with AVX512 (esp. AVX512_BF16)     | aarch64 supported; packed format for AVX512BF16 |

### Feature Support by Platform

| Feature            | CUDA | ROCm | XPU | HPU | MPS | CPU       |
|--------------------|------|------|-----|-----|-----|-----------|
| LLM.int8()         | Yes  | Yes  | Yes | Yes | No  | Partial*  |
| QLoRA 4-bit (NF4/FP4) | Yes | Yes | Yes | Yes | No  | Yes (AVX512BF16) |
| 8-bit Optimizers   | Yes  | Yes  | Yes | Yes | No  | Yes       |
| gemv_4bit fast kernel | Yes | Yes | No  | No  | No  | Yes (packed) |
| Paged optimizers   | Yes  | Yes  | No  | No  | No  | No        |

*CPU int8 uses dequantize+matmul fallback (MatMul8bitFp) rather than native int8 tensor cores.

### Supported Device Strings

bitsandbytes recognizes the following PyTorch device types:

```python
supported_torch_devices = {
    "cpu",    # CPU (x86-64, aarch64)
    "cuda",   # NVIDIA/AMD GPU
    "xpu",    # Intel GPU
    "hpu",    # Intel Gaudi
    "npu",    # Ascend NPU
    "mps",    # Apple Silicon
}
```

---

## Installation

### Standard Installation (pip)

```bash
# Basic install (CPU-only or auto-detect GPU)
pip install bitsandbytes

# Install with test dependencies (includes scipy for create_normal_map)
pip install bitsandbytes[test]
```

### CUDA-Specific Installation

bitsandbytes ships pre-compiled binaries for multiple CUDA versions. The library auto-detects the CUDA version from the installed PyTorch. If needed, override with:

```bash
# Override CUDA version detection
export BNB_CUDA_VERSION=121  # e.g., CUDA 12.1
pip install bitsandbytes
```

Supported CUDA versions in pre-built wheels typically include: 11.8, 12.1, 12.4, 12.5, etc.

### ROCm Installation

```bash
# For AMD ROCm, install the ROCm-built wheel
pip install bitsandbytes

# Override ROCm version if needed
export BNB_ROCM_VERSION=602  # e.g., ROCm 6.0.2
```

### Intel XPU Installation

```bash
# Install with Intel XPU support
pip install bitsandbytes
# The library auto-detects XPU via torch.xpu.is_available()
```

### CPU-Only Installation

```bash
# CPU-only build (no GPU required)
pip install bitsandbytes
```

### Building from Source

```bash
# Standard build (auto-detects backend)
pip install . -v

# Skip CMake build (e.g., for packaging)
BNB_SKIP_CMAKE=1 pip install . -v

# Build with specific compute backend
cmake -DCOMPUTE_BACKEND=cpu -S . && make
```

### Verifying Installation

```bash
python -m bitsandbytes
```

This prints diagnostic information including:
- bitsandbytes version
- PyTorch version
- CUDA/ROCm/XPU version detected
- Available pre-compiled binaries
- Device detection results

---

## Quick Start Examples

### 1. 8-bit Optimizers

```python
import torch
import bitsandbytes as bnb

# Create a simple model
model = torch.nn.Linear(1024, 1024).cuda()

# Drop-in replacement: torch.optim.AdamW -> bnb.optim.AdamW8bit
optimizer = bnb.optim.AdamW8bit(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.01,
)

# Training loop works identically to standard optimizer
for batch in dataloader:
    loss = model(batch).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

#### Per-Parameter Precision Override

```python
import bitsandbytes as bnb

mng = bnb.optim.GlobalOptimManager.get_instance()

model = MyModel().cuda()

# Register parameters before creating optimizer
mng.register_parameters(model.parameters())

# Use 8-bit for all, but override specific layers to 32-bit
optimizer = bnb.optim.Adam(model.parameters(), lr=1e-3, optim_bits=8)
mng.override_config(model.embed_tokens.weight, "optim_bits", 32)
```

#### Paged Optimizers (for large models)

```python
# Paged optimizer uses unified memory to avoid OOM on large state tensors
optimizer = bnb.optim.PagedAdamW8bit(
    model.parameters(),
    lr=1e-4,
)
```

### 2. LLM.int8() 8-bit Inference

```python
import torch
import torch.nn as nn
import bitsandbytes as bnb
from bitsandbytes.nn import Linear8bitLt

# Step 1: Create the quantized model
fp16_model = nn.Sequential(
    nn.Linear(4096, 4096),
    nn.Linear(4096, 4096),
)

int8_model = nn.Sequential(
    Linear8bitLt(4096, 4096, has_fp16_weights=False),
    Linear8bitLt(4096, 4096, has_fp16_weights=False),
)

# Step 2: Load fp16 weights into the quantized model
int8_model.load_state_dict(fp16_model.state_dict())

# Step 3: Move to GPU -- quantization happens here
int8_model = int8_model.to("cuda")

# Step 4: Run inference
x = torch.randn(1, 4096, device="cuda", dtype=torch.float16)
output = int8_model(x)
```

#### With Mixed-Precision Decomposition (Outlier Handling)

```python
from bitsandbytes.nn import Linear8bitLt

# threshold > 0.0 enables mixed-precision decomposition
# Columns in activations where any value exceeds threshold
# are computed in fp16 instead of int8
linear = Linear8bitLt(
    4096, 4096,
    has_fp16_weights=False,
    threshold=6.0,  # default is 0.0 (disabled)
)
```

#### Loading Pre-Quantized 8-bit Checkpoints

```python
# Load a pre-quantized checkpoint directly
model = nn.Sequential(
    Linear8bitLt(4096, 4096, has_fp16_weights=False),
)
model = model.to("cuda")  # Must move to GPU first to initialize quantization
model.load_state_dict(torch.load("quantized_checkpoint.pt"))
```

### 3. QLoRA 4-bit Quantization

```python
import torch
import torch.nn as nn
import bitsandbytes as bnb
from bitsandbytes.nn import LinearNF4

# Step 1: Create the quantized model with NF4 data type
fp16_model = nn.Sequential(
    nn.Linear(4096, 4096),
    nn.Linear(4096, 4096),
)

quantized_model = nn.Sequential(
    LinearNF4(4096, 4096, compute_dtype=torch.bfloat16),
    LinearNF4(4096, 4096, compute_dtype=torch.bfloat16),
)

# Step 2: Load fp16 weights
quantized_model.load_state_dict(fp16_model.state_dict())

# Step 3: Move to GPU -- quantization happens here
quantized_model = quantized_model.to("cuda")

# Step 4: Run inference
x = torch.randn(1, 4096, device="cuda", dtype=torch.bfloat16)
output = quantized_model(x)
```

#### With Double Quantization (compress_statistics)

```python
from bitsandbytes.nn import LinearNF4

# compress_statistics=True enables double quantization:
# the absmax values themselves are quantized to save additional memory
linear = LinearNF4(
    4096, 4096,
    compute_dtype=torch.bfloat16,
    compress_statistics=True,
)
```

#### With Different 4-bit Types

```python
from bitsandbytes.nn import LinearFP4, LinearNF4

# FP4: IEEE 754-like 4-bit floating point
linear_fp4 = LinearFP4(4096, 4096, compute_dtype=torch.float16)

# NF4: NormalFloat4, optimal for normally-distributed weights
linear_nf4 = LinearNF4(4096, 4096, compute_dtype=torch.bfloat16)

# Generic Linear4bit with explicit quant_type
linear = bnb.nn.Linear4bit(
    4096, 4096,
    compute_dtype=torch.bfloat16,
    quant_type="nf4",          # or "fp4"
    compress_statistics=True,
    quant_storage=torch.uint8,  # default
)
```

#### Replace All Linear Layers in a Model

```python
import bitsandbytes as bnb
from bitsandbytes.nn import LinearNF4

def replace_with_4bit(model):
    """Replace all nn.Linear layers with LinearNF4."""
    return bnb.utils.replace_linear(
        model,
        linear_replacement=lambda inf, outf, bias: LinearNF4(
            inf, outf, bias,
            compute_dtype=torch.bfloat16,
            compress_statistics=True,
        ),
        skip_modules=("lm_head",),
        copy_weights=True,
    )

model = replace_with_4bit(my_llm)
model = model.to("cuda")  # Quantization happens here
```

---

## Academic Citations

### QLoRA (4-bit quantization, NF4 data type)

```bibtex
@article{dettmers2023qlora,
  title={QLoRA: Efficient Finetuning of Quantized LLMs},
  author={Dettmers, Tim and Pagnoni, Artidoro and Holtzman, Ari and Zettlemoyer, Luke},
  journal={arXiv preprint arXiv:2305.14314},
  year={2023}
}
```

### LLM.int8() (8-bit inference with mixed-precision decomposition)

```bibtex
@article{dettmers2022llmint8,
  title={LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale},
  author={Dettmers, Tim and Lewis, Mike and Belkada, Younes and Zettlemoyer, Luke},
  journal={arXiv preprint arXiv:2208.07339},
  year={2022}
}
```

### 8-bit Optimizers (block-wise dynamic quantization for optimizer states)

```bibtex
@article{dettmers20218bit,
  title={8-Bit Approximations for Parallelism in Deep Learning},
  author={Dettmers, Tim},
  journal={arXiv preprint arXiv:1511.04561},
  year={2021}
}
```

---

## Compatibility Signal

The library exposes a `features` set that downstream integrations (Transformers, Diffusers) check:

```python
import bitsandbytes
bitsandbytes.features  # {"multi_backend"}
```

This signals that bitsandbytes supports multiple hardware backends (CUDA, ROCm, XPU, HPU, MPS, CPU) through a unified interface.

---

## Module Overview

```
bitsandbytes/
  __init__.py                  -- Package init, version, backend loading
  functional.py                -- Core quantization functions (quantize_4bit, quantize_blockwise, etc.)
  _ops.py                      -- torch.library op definitions (custom ops)
  cextension.py                -- Native library loading (CUDA, XPU, CPU)
  cuda_specs.py                -- CUDA/ROCm version detection
  utils.py                     -- Utility functions (replace_linear, OutlierTracer, pack_dict_to_tensor)
  autograd/
    _functions.py              -- MatMul8bitLt, MatMul4Bit, MatmulLtState, matmul(), matmul_4bit()
  nn/
    modules.py                 -- Linear8bitLt, Linear4bit, LinearFP4, LinearNF4, Int8Params, Params4bit
  optim/
    optimizer.py               -- Optimizer8bit, Optimizer2State, Optimizer1State, GlobalOptimManager
    adam.py, adamw.py, ...     -- Concrete optimizer implementations
  backends/
    cuda/ops.py                -- CUDA kernel registrations
    cpu/ops.py                 -- CPU kernel registrations
    xpu/ops.py                 -- XPU kernel registrations
    hpu/ops.py                 -- HPU kernel registrations
    mps/ops.py                 -- MPS kernel registrations
    default/ops.py             -- Default (PyTorch-native) fallback kernels
```
