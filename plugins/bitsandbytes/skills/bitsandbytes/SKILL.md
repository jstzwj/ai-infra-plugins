---
name: bitsandbytes
description: >
  Comprehensive reference documentation and skill for bitsandbytes, the k-bit quantization library
  for PyTorch enabling accessible large language models. Use this skill whenever the user mentions
  bitsandbytes, LLM.int8(), QLoRA, 4-bit quantization, 8-bit quantization, NF4, FP4, 8-bit optimizers,
  block-wise quantization, Int8Params, Params4bit, Linear8bitLt, Linear4bit, StableEmbedding,
  quantize_blockwise, dequantize_blockwise, quantize_4bit, dequantize_4bit, quantize_nf4, quantize_fp4,
  GlobalOptimManager, paged optimizers, FSDP integration with quantization, Triton kernels for quantization,
  CPU/XPU/MPS/HPU backends, CUDA kernels for quantized matmul, or bitsandbytes internals.
version: 0.50.0.dev0
---

# bitsandbytes - k-bit Quantization for PyTorch

## How to use this skill

Use this skill as a code-aware bitsandbytes reference manual. When a user asks about bitsandbytes usage, debugging,
performance, quantization, or internals, first identify which layer they are working at:

1. **Quantization fundamentals**: block-wise quantization, dynamic types, NF4/FP4 data types, QuantState.
2. **8-bit inference (LLM.int8())**: Linear8bitLt, Int8Params, MatMul8bitLt, outlier detection, mixed-precision.
3. **4-bit inference/training (QLoRA)**: Linear4bit, Params4bit, MatMul4Bit, gemv_4bit kernel, FSDP integration.
4. **8-bit optimizers**: Adam, AdamW, SGD, Lion, LAMB, LARS, Adagrad, RMSprop, AdEMAMix - all with 8-bit/32-bit/paged variants.
5. **Backend architecture**: CUDA, CPU, Triton, MPS, XPU, HPU backend implementations.
6. **Integration**: Transformers, PEFT, Accelerate, Diffusers, TGI, vLLM.
7. **Advanced topics**: Paged memory management, parametrize API, custom autograd functions, state_dict serialization.

When the answer needs precise behavior, cite the source file path from `sources/bitsandbytes` and, if you read
current files, cite `path:line`. If a reference here names a symbol and the user is about to modify code,
verify the current source first.

## bitsandbytes in one page

bitsandbytes is a PyTorch library providing three main features for dramatically reducing memory consumption:

1. **8-bit Optimizers** - Block-wise quantized optimizer states (Adam, AdamW, SGD, etc.) that maintain 32-bit
   performance at a fraction of the memory cost. Uses dynamic quantization with block-wise scaling.
2. **LLM.int8() / 8-bit Quantization** - Large language model inference with half the memory and no performance
   degradation. Vector-wise quantization quantizes most features to 8-bit with separate 16-bit treatment of outliers.
3. **QLoRA / 4-bit Quantization** - Large language model training via 4-bit quantization with NF4/FP4 data types,
   double quantization (compressing quantization statistics), and paged optimizers.

**Core module hierarchy:**
- `bitsandbytes/functional.py` - Quantization/dequantization primitives, QuantState, optimizer update kernels
- `bitsandbytes/nn/modules.py` - Linear8bitLt, Linear4bit, LinearFP4, LinearNF4, Int8Params, Params4bit, embeddings
- `bitsandbytes/nn/parametrize.py` - Bnb4bitParametrization for MoE and non-Linear quantization
- `bitsandbytes/autograd/_functions.py` - MatMul8bitLt, MatMul4Bit, MatMul8bitFp, MatmulLtState
- `bitsandbytes/optim/` - 8-bit optimizers (Adam, AdamW, SGD, Lion, LAMB, LARS, Adagrad, RMSprop, AdEMAMix)
- `bitsandbytes/backends/` - Platform-specific implementations (CUDA, CPU, Triton, MPS, XPU, HPU)
- `bitsandbytes/_ops.py` - PyTorch custom op definitions and fake tensor implementations
- `bitsandbytes/csrc/` - CUDA/C++ kernel implementations

## Quick usage patterns

### 8-bit inference (LLM.int8())
```python
import torch
import torch.nn as nn
import bitsandbytes as bnb
from bitsandbytes.nn import Linear8bitLt

fp16_model = nn.Sequential(
    nn.Linear(64, 64),
    nn.Linear(64, 64)
)

int8_model = nn.Sequential(
    Linear8bitLt(64, 64, has_fp16_weights=False),
    Linear8bitLt(64, 64, has_fp16_weights=False)
)

int8_model.load_state_dict(fp16_model.state_dict())
int8_model = int8_model.to(0)  # Quantization happens here
```

### 4-bit quantization (QLoRA)
```python
import torch
import torch.nn as nn
import bitsandbytes as bnb
from bitsandbytes.nn import Linear4bit

fp16_model = nn.Sequential(
    nn.Linear(64, 64),
    nn.Linear(64, 64)
)

quantized_model = nn.Sequential(
    Linear4bit(64, 64, quant_type="nf4"),
    Linear4bit(64, 64, quant_type="nf4")
)

quantized_model.load_state_dict(fp16_model.state_dict())
quantized_model = quantized_model.to(0)  # Quantization happens here
```

### 8-bit optimizer
```python
import bitsandbytes as bnb

# Drop-in replacement for torch.optim.AdamW
optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-5, weight_decay=0.01)

# Or with paged memory for large models
optimizer = bnb.optim.PagedAdamW8bit(model.parameters(), lr=1e-5)
```

### QLoRA with Transformers
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained("model_name", quantization_config=bnb_config)
```

### Per-parameter optimizer override
```python
import bitsandbytes as bnb

mng = bnb.optim.GlobalOptimManager.get_instance()
model = MyModel()
mng.register_parameters(model.parameters())

model = model.cuda()
adam = bnb.optim.Adam(model.parameters(), lr=0.001, optim_bits=8)

# Override specific parameter to use 32-bit Adam states
mng.override_config(model.fc1.weight, 'optim_bits', 32)
```

## Documentation map

### Quantization fundamentals
- [01 - Overview, Installation, and Quick Start](references/01-overview-installation-quickstart.md): system requirements, platform support, installation, basic examples.
- [02 - Quantization Fundamentals](references/02-quantization-fundamentals.md): block-wise quantization, dynamic data types, NF4/FP4, QuantState, quantization maps.

### 8-bit quantization (LLM.int8())
- [03 - LLM.int8() 8-bit Quantization](references/03-llm-int8-quantization.md): Linear8bitLt, Int8Params, MatMul8bitLt, outlier detection, mixed-precision decomposition, int8_scaled_mm.

### 4-bit quantization (QLoRA)
- [04 - QLoRA 4-bit Quantization](references/04-qlora-4bit-quantization.md): Linear4bit, Params4bit, MatMul4Bit, gemv_4bit, double quantization, CPU packed format, FSDP compatibility.

### Neural network modules
- [05 - NN Modules and Layers](references/05-nn-modules-layers.md): Linear8bitLt, Linear4bit, LinearFP4, LinearNF4, Embedding8bit, Embedding4bit, EmbeddingFP4, EmbeddingNF4, StableEmbedding, OutlierAwareLinear.

### Functional API
- [06 - Functional API Reference](references/06-functional-api-reference.md): quantize_blockwise, dequantize_blockwise, quantize_4bit, dequantize_4bit, quantize_nf4/fp4, int8_vectorwise_quant, int8_linear_matmul, int8_mm_dequant, optimizer_update_32bit, optimizer_update_8bit_blockwise, gemv_4bit, igemm.

### 8-bit optimizers
- [07 - Optimizer Architecture and Base Classes](references/07-optimizer-architecture.md): Optimizer8bit, Optimizer2State, Optimizer1State, GlobalOptimManager, paged memory, state_dict FSDP compatibility.
- [08 - Optimizer Variants](references/08-optimizer-variants.md): Adam, AdamW, SGD, Lion, LAMB, LARS, Adagrad, RMSprop, AdEMAMix - all with 8bit/32bit/Paged variants.

### Backend and kernel architecture
- [09 - Backend Architecture](references/09-backend-architecture.md): multi-backend system, CUDA, CPU, Triton, MPS, XPU, HPU backends, backend autoloading, ops dispatch.
- [10 - CUDA/C++ Kernels and Custom Ops](references/10-cuda-kernels-custom-ops.md): csrc/ kernels, PyTorch custom op definitions, fake tensor impls, CUBLAS context, igemm/batched_igemm.

### Advanced topics
- [11 - Autograd Functions and Custom Backward](references/11-autograd-functions.md): MatMul8bitLt forward/backward, MatMul4Bit forward/backward, MatMul8bitFp, MatmulLtState, gradient computation.
- [12 - Parametrize API and MoE Support](references/12-parametrize-api.md): Bnb4bitParametrization, replace_parameter_4bit, replace_parameter_4bit_prequantized, state dict hooks.
- [13 - FSDP Integration and Distributed Training](references/13-fsdp-integration.md): QuantState serialization, Params4bit properties, state_dict wrapping, fsdp_qlora patterns.
- [14 - Paged Memory Management](references/14-paged-memory.md): GlobalPageManager, get_paged, prefetch_tensor, paged optimizer states, unified memory.

### Integration and development
- [15 - Integration with Transformers, PEFT, and Accelerate](references/15-integration-ecosystem.md): BitsAndBytesConfig, from_pretrained integration, PEFT/LoRA on quantized models, Diffusers.
- [16 - Diagnostics, Troubleshooting, and FAQs](references/16-diagnostics-troubleshooting.md): common errors, CUDA kernel issues, fatbinwrap errors, device compatibility, performance tuning.
- [17 - Development Guide and Contributing](references/17-development-contributing.md): worktree workflow, pre-commit, testing, linting, CI/CD, build system.
- [18 - Source File Index](references/18-source-file-index.md): complete source-code map by module for fast navigation.

## Answering rules for bitsandbytes tasks

- For quantization questions, explain both **the quantization algorithm** and **the data flow through modules**. Quantization happens lazily on `.to(device)` calls.
- For LLM.int8() questions, distinguish between `has_fp16_weights=True` (on-the-fly quantization during training) and `has_fp16_weights=False` (pre-quantized for inference).
- For QLoRA/4-bit questions, explain NF4 vs FP4 data types, double quantization (`compress_statistics`), and the compute_dtype flow.
- For optimizer questions, clarify the `min_8bit_size` threshold (default 4096) below which parameters use 32-bit states regardless of `optim_bits`.
- For FSDP questions, note that `QuantState.as_dict(packed=True)` is critical for safetensors serialization and that Params4bit has @property proxies for FSDP traversal.
- For performance, distinguish between gemv_4bit (single-batch inference, fast path) and MatMul4Bit (training/multi-batch, dequantize-then-matmul path).
- For source changes, start from `bitsandbytes/functional.py`, `bitsandbytes/nn/modules.py`, `bitsandbytes/autograd/_functions.py`, and `bitsandbytes/_ops.py`.
- For backend questions, note the dispatch hierarchy: CUDA > Triton > default (CPU fallback), with backend-specific ops in `backends/<name>/ops.py`.
