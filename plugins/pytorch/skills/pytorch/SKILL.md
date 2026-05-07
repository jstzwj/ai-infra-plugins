---
name: pytorch
description: >
  Comprehensive reference documentation and skill for PyTorch - the GPU-accelerated tensor computation
  and deep learning framework. Covers tensor operations, automatic differentiation, neural network modules
  (nn), optimization, distributed training, CUDA support, automatic mixed precision (AMP), torch.compile/Dynamo,
  TorchScript, FX graph transformation, Inductor backend, ONNX export, quantization, profiling, data loading,
  probability distributions, FFT, linear algebra, sparse tensors, C++ API (libtorch), operator dispatch,
  custom operators, and deployment. Based on PyTorch source code analysis.
version: 2.7
---

# PyTorch - Tensor Computation & Deep Learning Framework

## Overview

PyTorch is a GPU-accelerated tensor computation framework with a tape-based automatic differentiation system and a comprehensive deep learning library. It provides two high-level features:

1. **Tensor computation** with strong GPU acceleration (via CUDA, XPU, MPS backends)
2. **Deep neural networks** built on a tape-based autograd system

**Supported Hardware:** NVIDIA GPUs (CUDA), Intel GPUs (XPU), Apple Silicon (MPS), AMD GPUs (ROCm), Meta MTIA, CPU

**Supported Platforms:** Linux, macOS, Windows, Android, iOS

**PyTorch Version:** 2.7

## Key Architecture Concepts

- **Tensor**: Multi-dimensional matrix containing elements of a single data type
- **Autograd**: Automatic differentiation engine that powers neural network training
- **nn.Module**: Base class for all neural network components
- **Optimizer**: Algorithms for updating model parameters (Adam, SGD, etc.)
- **Dispatcher**: Routes operator calls to backend-specific implementations
- **Dynamo**: Just-in-time compiler for PyTorch (`torch.compile`)
- **Inductor**: Triton-based code generation backend for Dynamo
- **FX**: Python-to-Python program transformation toolkit
- **TorchScript**: Static typing and compilation for production deployment
- **ATen**: A Tensor Library - C++ core tensor operations
- **c10**: Core library providing foundational abstractions

## Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│                   Python Frontend                     │
│  torch.nn  │  torch.optim  │  torch.autograd  │ ...  │
├──────────────┼──────────────┼─────────────────────────┤
│            torch._C (Pybind11 Bindings)               │
├──────────────┼──────────────┼─────────────────────────┤
│           ATen (A Tensor Library)                     │
│  native ops │  CPU kernels  │  CUDA kernels  │ ...    │
├──────────────┼──────────────┼─────────────────────────┤
│           c10 (Core Library)                          │
│  TensorImpl  │  Device  │  ScalarType  │  Storage     │
└──────────────────────────────────────────────────────┘
```

## Quick Reference

### Tensor Creation
```python
import torch

# From data
t = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, device='cuda')

# Factory functions
z = torch.zeros(3, 4)
o = torch.ones(2, 3)
r = torch.randn(5, 5)        # Standard normal
u = torch.rand(3, 3)         # Uniform [0, 1)
a = torch.arange(0, 10, 2)   # [0, 2, 4, 6, 8]
l = torch.linspace(0, 1, 5)  # [0, 0.25, 0.5, 0.75, 1.0]
e = torch.empty(2, 3)        # Uninitialized
f = torch.full((2, 3), 7.0)  # Filled with 7.0
i = torch.eye(3)             # Identity matrix

# From existing tensor
like = torch.zeros_like(t)
new_t = t.new_zeros(3, 4)
```

### Training Loop
```python
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
).cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
```

### torch.compile
```python
model = torch.compile(model)  # JIT compile for speed
output = model(input)
```

### Automatic Mixed Precision
```python
scaler = torch.amp.GradScaler('cuda')
with torch.amp.autocast('cuda'):
    output = model(input)
    loss = criterion(output, target)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Distributed Training
```python
import torch.distributed as dist

dist.init_process_group("nccl")
model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
```

### Custom Autograd Function
```python
class MyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * (input > 0).float()
```

### C++ API (LibTorch)
```cpp
#include <torch/torch.h>
auto tensor = torch::rand({2, 3});
auto result = torch::matmul(tensor, tensor.transpose(0, 1));
```

## Code Examples

Official examples from [pytorch/examples](https://github.com/pytorch/examples) (219 files), covering key deep learning patterns:

| Category | Description |
|----------|-------------|
| `mnist/` | MNIST classification — basic training loop |
| `mnist_hogwild/` | Hogwild multi-process training |
| `mnist_rnn/` | RNN on MNIST |
| `mnist_forward_forward/` | Forward-Forward algorithm |
| `imagenet/` | ImageNet training with distributed data parallel |
| `dcgan/` | DCGAN generative adversarial network |
| `vae/` | Variational Autoencoder |
| `word_language_model/` | LSTM/Transformer language modeling |
| `time_sequence_prediction/` | Sequence prediction with RNN |
| `regression/` | Polynomial regression |
| `reinforcement_learning/` | DQN, policy gradient, actor-critic, REINFORCE |
| `distributed/` | DDP, FSDP, pipeline parallel, RPC, minGPT-ddp |
| `super_resolution/` | SRCNN super-resolution with AMP |
| `fast_neural_style/` | Neural style transfer |
| `language_translation/` | Seq2seq translation with attention |
| `siamese_network/` | Siamese network for similarity |
| `gat/` / `gcn/` | Graph neural networks (GAT, GCN) |
| `fx/` | torch.fx graph manipulation examples |
| `cpp/` | LibTorch C++ API examples |
| `legacy/` | Legacy examples (AlephBet, SNLI, etc.) |

## Reference Chapters

### Core

1. [Overview and Architecture](references/01-overview-and-architecture.md) - Design philosophy, system architecture, code layout
2. [Tensor Fundamentals](references/02-tensor-fundamentals.md) - Creation, indexing, slicing, reshaping, views
3. [Tensor Operations](references/03-tensor-operations.md) - Math, comparison, reduction, BLAS, spectral ops
4. [Automatic Differentiation](references/04-autograd.md) - Autograd engine, Function, grad modes, forward-mode AD
5. [Tensor Types and Device Management](references/05-tensor-types-and-device.md) - dtypes, devices, layouts, memory formats

### Neural Network Modules (torch.nn)

6. [nn.Module System](references/06-nn-module-system.md) - Module, Parameter, hooks, serialization, state_dict
7. [Linear and Convolution Layers](references/07-nn-linear-conv.md) - Linear, Conv1d/2d/3d, ConvTranspose, Lazy variants
8. [Recurrent and Transformer Layers](references/08-nn-rnn-transformer.md) - RNN, LSTM, GRU, Transformer, MultiheadAttention
9. [Normalization, Pooling, and Activation](references/09-nn-norm-pool-activation.md) - BatchNorm, LayerNorm, GroupNorm, MaxPool, ReLU, GELU, etc.
10. [Loss Functions](references/10-nn-loss-functions.md) - CrossEntropy, MSE, L1, NLL, BCE, KLDiv, CosineEmbedding, etc.
11. [nn.functional](references/11-nn-functional.md) - Functional API for all nn operations
12. [nn.init and Utilities](references/12-nn-init-utils.md) - Weight initialization, RNN utils, clip_grad_norm, data utilities

### Optimization

13. [Optimizers](references/13-optimizers.md) - SGD, Adam, AdamW, RMSprop, Adagrad, LBFGS, etc.
14. [Learning Rate Schedulers](references/14-lr-schedulers.md) - StepLR, CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau, etc.

### Distributed Training

15. [Distributed Overview](references/15-distributed-overview.md) - Architecture, backends (NCCL, Gloo, MPI), initialization
16. [DistributedDataParallel](references/16-distributed-ddp.md) - DDP design, gradient sync, bucketing, comm hooks
17. [Collective Communications](references/17-distributed-collectives.md) - all_reduce, all_gather, broadcast, reduce_scatter, etc.
18. [Pipeline Parallelism](references/18-distributed-pipeline.md) - PipelineParallel, GPipe, virtual stages
19. [FSDP](references/19-distributed-fsdp.md) - FullyShardedDataParallel, sharding strategies, mixed precision
20. [RPC Framework](references/20-distributed-rpc.md) - Remote procedure calls, RRef, distributed autograd

### GPU and Performance

21. [CUDA Support](references/21-cuda-support.md) - CUDA tensors, memory management, pinned memory, device operations
22. [Automatic Mixed Precision](references/22-amp.md) - autocast, GradScaler, bfloat16, float16
23. [Memory Management](references/23-memory-management.md) - Memory allocation, caching allocator,OutOfMemoryError handling
24. [Streams and Events](references/24-streams-events.md) - CUDA streams, events, synchronization, graph capture

### Compilation and Export

25. [torch.compile](references/25-torch-compile.md) - Dynamo, graph breaks, backends, config
26. [TorchScript](references/26-torchscript.md) - Scripting, tracing, optimization, deployment
27. [FX Graph](references/27-fx-graph.md) - Graph representation, Node, symbolic trace, transformation passes
28. [Inductor Backend](references/28-inductor.md) - Triton codegen, CPU backend, memory planning, scheduling
29. [Model Export](references/29-export.md) - torch.export, dynamo export, ExportedProgram
30. [ONNX Export](references/30-onnx.md) - ONNX graph conversion, opset versions, custom ops

### Data Pipeline

31. [DataLoader](references/31-dataloader.md) - Batching, shuffling, multiprocessing, prefetching, worker init
32. [Datasets](references/32-datasets.md) - Dataset, IterableDataset, TensorDataset, ConcatDataset, random_split
33. [Transforms](references/33-transforms.md) - torchvision transforms, custom transforms, functional API

### Mathematical Libraries

34. [Probability Distributions](references/34-distributions.md) - Normal, Uniform, Bernoulli, Categorical, etc., KL divergence
35. [FFT Operations](references/35-fft.md) - FFT, IFFT, RFFT, IRFFT, FFT2, FFTN, hann/hamming windows
36. [Linear Algebra](references/36-linalg.md) - Matrix decompositions (SVD, LU, QR, cholesky), solve, eig, norm
37. [Sparse Tensors](references/37-sparse.md) - COO, CSR, CSC formats, sparse operations, semi-structured sparsity
38. [Special Math Functions](references/38-special-functions.md) - Bessel, gamma, erf, digamma, softmax, log_softmax
39. [Masked and Nested Tensors](references/39-masked-nested-tensors.md) - Masked operations, nested/jagged tensors

### C++ Backend

40. [c10 Core Library](references/40-c10-core.md) - Device, ScalarType, TensorImpl, Storage, ArrayRef, DispatchKey
41. [ATen Operations](references/41-aten-operations.md) - Native functions, operator schema, backend implementations
42. [LibTorch C++ API](references/42-libtorch-cpp-api.md) - C++ frontend, nn::Module, optim::Optimizer, data loaders
43. [Operator Dispatch](references/43-operator-dispatch.md) - Dispatch keys, dispatch table, fallthrough, functionalization
44. [Custom Operators](references/44-custom-operators.md) - TORCH_LIBRARY, torch.library, custom CUDA kernels
45. [Code Generation (torchgen)](references/45-torchgen.md) - Operator generation, native_functions.yaml, build system
46. [Autograd Engine (C++)](references/46-autograd-engine.md) - Eval queue, graph execution, compiled autograd

### Deployment and Optimization

47. [Quantization](references/47-quantization.md) - Post-training quantization, quantization-aware training, observer, fake quant
48. [Advanced Quantization (torch.ao)](references/48-ao-quantization.md) - torch.ao.ns, GPTQ, quantized operators, custom quant
49. [Profiling](references/49-profiling.md) - torch.profiler, TensorBoard, Kineto, memory profiling, flame graphs
50. [Package and Hub](references/50-package-hub.md) - torch.package, torch.hub, model packaging, dependency management
51. [Mobile Deployment](references/51-mobile-deployment.md) - Model optimization for mobile, quantization, selective build
52. [Inference Optimization](references/52-inference-optimization.md) - Inference mode, graph optimization, kernel fusion, torch._C

### Advanced Topics

53. [functorch (JAX-like transforms)](references/53-functorch.md) - vmap, grad, jacfwd, jacrev, hessian, functional calls
54. [Storage and Serialization](references/54-storage-serialization.md) - Storage types, save/load, safetensors, format details
55. [Multiprocessing](references/55-multiprocessing.md) - torch.multiprocessing, shared memory, CUDA IPC
56. [Backend System](references/56-backends.md) - Backend modules (mkl, mkldnn, cudnn, etc.), flags, context managers
57. [Control Flow and Gradient](references/57-control-flow-grad.md) - torch.cond, torch.while_loop, checkpoint, gradient checkpointing
58. [Type System and dtypes](references/58-type-system-dtypes.md) - All dtypes, type promotion, casting rules, SymInt/SymFloat
59. [Device Management](references/59-device-management.md) - Multi-device, device context, XPU, MPS, MTIA backends
60. [Advanced Features](references/60-advanced-features.md) - Named tensors, compiler hints, tracing, debugging utilities
