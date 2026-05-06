# PyTorch Reference - Chapter 1: Overview and Architecture

This chapter provides a comprehensive overview of PyTorch's design philosophy, system architecture, source code layout, dispatch mechanism, compilation pipeline, key abstractions, and build system. Understanding these foundational concepts is essential for navigating, extending, and debugging PyTorch at any level.

---

## 1.1 Design Philosophy and History

### 1.1.1 Origins

PyTorch was developed by Facebook's AI Research lab (FAIR, now Meta AI) and released as an open-source project in September 2016. Its primary creators -- Soumith Chintala, Adam Paszke, and Sam Gross -- designed it as a Python-first deep learning framework that prioritized flexibility, debuggability, and rapid experimentation over the static graph paradigm dominant at the time (exemplified by TensorFlow 1.x and Theano).

PyTorch evolved from two earlier projects:

- **Torch** (2002): A scientific computing framework based on LuaJIT, providing N-dimensional arrays and GPU acceleration. Torch was powerful but suffered from Lua's limited ecosystem.
- **Chainer** (2015): A Python-based deep learning framework that pioneered the "define-by-run" approach, where the computation graph is constructed dynamically during forward execution.

PyTorch combined Torch's mature C backend with Chainer's dynamic graph philosophy, wrapped in a Pythonic API.

### 1.1.2 Core Design Principles

PyTorch's architecture is guided by several key principles:

1. **Python-First**: The primary interface is Python. Users interact with Python objects and use standard Python debugging tools (pdb, print, IDE breakpoints). There is no separate "graph definition" language.

2. **Eager Execution by Default**: Operations execute immediately when called. There is no separate compilation step required for basic usage. This makes debugging straightforward -- you can print intermediate values, use conditional logic, and mix Python control flow with tensor operations.

3. **Tape-Based Autograd**: Gradients are computed via reverse-mode automatic differentiation. During the forward pass, PyTorch records operations in a dynamic computation graph. During `backward()`, it traverses this graph in reverse to compute gradients. This approach naturally supports dynamic control flow (loops, conditionals, recursion).

4. **Deep Integration with the Python Ecosystem**: PyTorch tensors can interoperate with NumPy arrays (zero-copy conversion), are compatible with standard Python scientific computing tools (SciPy, scikit-learn, Matplotlib), and support standard Python protocols (buffer protocol, iteration, indexing).

5. **Extensibility**: PyTorch is designed to be extended at multiple levels:
   - Python-level: Custom modules, autograd functions, loss functions
   - C++ level: Custom operators via the dispatcher, custom backends
   - Compiler level: Custom FX passes, custom Dynamo backends

6. **Performance Through Compilation (PyTorch 2.x)**: Starting with PyTorch 2.0, the framework added `torch.compile()` as a first-class feature. Rather than requiring users to learn a separate compilation API, PyTorch compiles standard eager-mode code under the hood using Dynamo (bytecode analysis) and generates optimized kernels via Inductor (Triton for GPU, C++ for CPU).

### 1.1.3 Version History (Key Milestones)

| Version | Date | Key Features |
|---------|------|-------------|
| 0.1 | Sep 2016 | Initial release, basic tensor ops + autograd |
| 0.2 | Dec 2016 | Advanced indexing, distributed training |
| 0.3 | Apr 2017 | torch.nn.DataParallel, torch.utils.data |
| 0.4 | Apr 2018 | Merged Tensor and Variable, zero-dimensional tensors |
| 1.0 | Dec 2018 | TorchScript (JIT), C++ frontend (libtorch) |
| 1.5 | Apr 2020 | torch.nn.Transformer, torch.fft |
| 1.6 | Jul 2020 | torch.cuda.amp (automatic mixed precision) |
| 1.7 | Oct 2020 | Complex tensor support, torch.fft module |
| 1.8 | Mar 2021 | torch.fx, torch.linalg |
| 1.11 | Mar 2022 | functorch stable |
| 1.12 | Jun 2022 | torch.compile (beta), Dynamo, Inductor |
| 2.0 | Mar 2023 | torch.compile stable, torch.export |
| 2.1 | Oct 2023 | SDPA, dynamic shapes improvements |
| 2.2 | Jan 2024 | Compiled autograd, FlexAttention |
| 2.5 | Oct 2024 | Compiled autograd stable |
| 2.7 | 2025 | Continued compiler optimization, expanded backend support |

---

## 1.2 System Architecture Layers

PyTorch's architecture is organized as a layered stack, with higher-level abstractions built on top of lower-level foundations:

```
Layer 5: Application Code
         User Python scripts, training loops, model definitions
           |
Layer 4: High-Level Python API
         torch.nn, torch.optim, torch.utils.data, torch.distributed
         torch.autograd, torch.amp, torch.profiler
           |
Layer 3: Core Python API
         torch.Tensor, torch factory functions, torch._C bindings
         torch._dynamo, torch.fx, torch._inductor
           |
Layer 2: C++ Tensor Library (ATen)
         aten/src/ATen/native/ - Operator implementations
         aten/src/ATen/core/ - Tensor class, operator registration
           |
Layer 1: Foundation Library (c10)
         c10/core/ - TensorImpl, Storage, Device, ScalarType, DispatchKey
         c10/util/ - ArrayRef, Optional, intrusive_ptr
           |
Layer 0: Backend Kernels
         CPU: SIMD (AVX2, AVX512, NEON), MKL, MKLDNN
         GPU: CUDA, cuDNN, cuBLAS, Triton (via Inductor)
         Accelerator: XPU (oneDNN), MPS (Metal), ROCm
```

### 1.2.1 Python Frontend (Layers 3-4)

The Python frontend is what users interact with directly. Key modules:

| Module | Location | Purpose |
|--------|----------|---------|
| `torch` | `torch/__init__.py` | Top-level namespace, factory functions |
| `torch._tensor` | `torch/_tensor.py` | Tensor class definition |
| `torch.nn` | `torch/nn/` | Neural network modules, layers, loss functions |
| `torch.nn.functional` | `torch/nn/functional.py` | Functional interface to nn operations |
| `torch.optim` | `torch/optim/` | Optimizers (SGD, Adam, etc.) |
| `torch.autograd` | `torch/autograd/` | Automatic differentiation engine |
| `torch.utils.data` | `torch/utils/data/` | DataLoader, Dataset, Sampler |
| `torch.distributed` | `torch/distributed/` | Distributed training (DDP, FSDP, RPC) |
| `torch.jit` | `torch/jit/` | TorchScript (scripting and tracing) |
| `torch.fft` | `torch/fft/` | Fast Fourier Transform operations |
| `torch.linalg` | `torch/linalg/` | Linear algebra operations |
| `torch.sparse` | `torch/sparse/` | Sparse tensor operations |
| `torch.special` | `torch/special/` | Special mathematical functions |
| `torch._dynamo` | `torch/_dynamo/` | Dynamo bytecode compiler |
| `torch._inductor` | `torch/_inductor/` | Inductor code generation backend |
| `torch.fx` | `torch/fx/` | Python-to-Python transformation toolkit |
| `torch.export` | `torch/_export/` | Model export infrastructure |
| `torch.compiler` | `torch/compiler/` | torch.compile public API |
| `torch.func` | `torch/_functorch/` | vmap, grad, jacfwd, jacrev transforms |

### 1.2.2 PyBind11 Bindings (`torch._C`)

The `torch._C` module is the Python extension module that bridges Python and C++. It is compiled from `torch/csrc/` and exposes C++ classes and functions to Python.

Key files in `torch/csrc/`:

```
torch/csrc/
  autograd/           # Python bindings for autograd (Variable, Function)
  distributed/        # Bindings for distributed training
  jit/                # Bindings for TorchScript
  Module.cpp          # Main module initialization (defines torch._C)
  Storage.cpp         # Storage Python bindings
  Tensor.cpp          # Tensor Python bindings
  utils.cpp           # Utility functions
  dynamic_dispatch.cpp # Dynamic dispatch helpers
  generic/            # Template-generated type-specific methods
```

The `torch._C` module provides:

- `_C.TensorBase`: The base C++ tensor object
- `_C._TensorBase`: Python-level tensor class (inherits from TensorBase)
- `_C._AutogradFunction`: Base class for custom autograd functions
- `_C._NodeBase`: Base class for autograd graph nodes
- `_C.Storage`: Underlying storage object
- Various internal functions for operator dispatch

### 1.2.3 ATen (A Tensor Library) - Layer 2

ATen is the C++ tensor library that implements all tensor operations. It provides:

- The `at::Tensor` class (the C++ tensor)
- Operator implementations for all backends (CPU, CUDA, Sparse, etc.)
- The operator registration and dispatch infrastructure
- Type dispatch and promotion logic

ATen source layout:

```
aten/
  src/
    ATen/
      core/               # Core types: Tensor, TensorImpl, op registration
        Tensor.cpp        # Tensor method implementations
        TensorImpl.cpp    # TensorImpl reference-counted tensor body
        OperatorEntry.cpp # Operator dispatch table entries
        interned_strings.cpp # String interning for dispatch keys
      native/             # "Native" operator implementations
        cpu/              # CPU kernel implementations
        cuda/             # CUDA kernel implementations
        QuantizedCPU.cpp  # Quantized CPU implementations
        QuantizedCUDA.cpp # Quantized CUDA implementations
        SparseCPU.cpp     # Sparse CPU implementations
        SparseCUDA.cpp    # Sparse CUDA implementations
        Math.cpp          # Shared math implementations
        Transformer.cpp   # Attention kernels
        BatchLinearAlgebra.cpp # BLAS/LAPACK wrappers
      ops/                # Per-operator organized files
        core_generated/   # Auto-generated operator files
      transforms/         # Type reduction, batching rules
      cpu/                # CPU-specific utilities
        vec/              # SIMD vector abstractions (AVX2, AVX512, NEON)
      cuda/               # CUDA-specific utilities
      blas_openblas.cpp   # OpenBLAS backend
      blas_mkl.cpp        # MKL backend
      mkldnn/             # oneDNN backend integration
    TH/                   # Legacy C tensor library (being phased out)
    THC/                  # Legacy CUDA tensor library (being phased out)
    THNN/                 # Legacy neural network ops (being phased out)
    THCUNN/               # Legacy CUDA neural network ops (being phased out)
```

### 1.2.4 c10 (Core Library) - Layer 1

c10 is the foundational library that provides the most basic abstractions used throughout PyTorch. "c10" stands for "Core 10" (a reference to the 10 foundational abstractions).

Key components:

```
c10/
  core/
    TensorImpl.cpp/h          # Reference-counted tensor implementation
    Storage.h/cpp             # Data storage (pointer + size + allocator)
    StorageImpl.h             # Storage implementation
    Device.h/cpp              # Device representation (type + index)
    DeviceType.h              # Device type enums (CPU, CUDA, XPU, MPS, etc.)
    Scalar.h/cpp              # Unified scalar type (int, float, complex)
    ScalarType.h              # Enum for all tensor element types
    DispatchKey.h/cpp         # Dispatch key definitions
    DispatchKeySet.h          # Set of dispatch keys for operator routing
    Allocator.h/cpp           # Memory allocator interface
    CPUAllocator.cpp          # CPU memory allocator
    SymInt.h/cpp              # Symbolic integer for dynamic shapes
    SymFloat.h/cpp            # Symbolic float for dynamic shapes
    TensorOptions.h           # Tensor configuration (dtype, device, layout)
    Layout.h                  # Tensor layout enum (Strided, Sparse, etc.)
    MemoryFormat.h            # Memory format enum (Contiguous, ChannelsLast)
  util/
    ArrayRef.h                # Non-owning array reference
    Optional.h                # C++17-like optional
    intrusive_ptr.h           # Intrusive reference-counted pointer
    Exception.h               # Exception types
    Half.h                    # float16 implementation
    BFloat16.h                # bfloat16 implementation
    Float8.h                  # float8 implementations
  cuda/
    CUDAStream.h/cpp          # CUDA stream abstraction
    CUDAEvent.h               # CUDA event abstraction
    CUDACachingAllocator.cpp  # CUDA caching allocator implementation
  xpu/
    XPUStream.h               # XPU stream abstraction
  macros/
    Macros.h                  # Platform and compiler macros
    Export.h                  # DLL export macros
```

### 1.2.5 Relationship Between Layers

```
User calls: torch.add(a, b)
    |
    v
Python __torch_function__ dispatch (torch/_tensor.py)
    |
    v
torch._C._ops.add (PyBind11 binding)
    |
    v
at::native::add (ATen C++ function)
    |
    v
c10::Dispatcher::call (Dispatch to backend)
    |
    +---> CPU: at::native::add_cpu_kernel (via SIMD intrinsics or MKL)
    +---> CUDA: at::native::add_cuda_kernel (via CUDA kernel or cuBLAS)
    +---> Sparse: at::native::add_sparse_kernel
    +---> Autograd: add_autograd (records for backward)
    +---> CompositeExplicitAutograd: fallback implementation
```

---

## 1.3 Directory Structure of the Source Code

The PyTorch source repository is large and complex. Here is a detailed map:

```
pytorch/                            # Repository root
|
|-- torch/                          # Python package (the main user-facing API)
|   |-- __init__.py                 # Top-level namespace, factory functions
|   |-- _tensor.py                  # torch.Tensor class
|   |-- _C/                         # Compiled C++ extension module
|   |   |-- _*.pyi                  # Type stubs for C++ functions
|   |-- _subclasses/                # Tensor subclasses
|   |   |-- fake_tensor.py          # FakeTensor for compilation/tracing
|   |   |-- functional_tensor.py    # FunctionalTensor for functionalization
|   |-- autograd/                   # Automatic differentiation
|   |   |-- __init__.py             # Public API: backward, grad, no_grad, etc.
|   |   |-- function.py             # torch.autograd.Function base class
|   |   |-- variable.py             # Variable (now merged with Tensor)
|   |   |-- gradcheck.py            # Gradient checking utilities
|   |   |-- grad_mode.py            # Gradient mode context managers
|   |   |-- forward_ad.py           # Forward-mode AD
|   |   |-- graph.py                # Computation graph manipulation
|   |   |-- profiler.py             # Autograd profiler
|   |   |-- anomaly_mode.py         # Anomaly detection
|   |-- nn/                         # Neural network modules
|   |   |-- __init__.py             # Module, Parameter, containers
|   |   |-- modules/                # Individual layer implementations
|   |   |   |-- linear.py           # Linear, Bilinear, LazyLinear
|   |   |   |-- conv.py             # Conv1d, Conv2d, Conv3d, ConvTranspose*
|   |   |   |-- rnn.py              # RNN, LSTM, GRU
|   |   |   |-- transformer.py      # Transformer, TransformerEncoder
|   |   |   |-- normalization.py    # BatchNorm, LayerNorm, GroupNorm
|   |   |   |-- pooling.py          # MaxPool, AvgPool, AdaptivePool
|   |   |   |-- activation.py       # ReLU, GELU, SiLU, Mish, etc.
|   |   |   |-- loss.py             # CrossEntropy, MSE, L1, etc.
|   |   |   |-- sparse.py           # Embedding, EmbeddingBag
|   |   |   |-- pixelshuffle.py     # PixelShuffle, PixelUnshuffle
|   |   |-- functional.py           # Functional API: F.relu, F.conv2d, etc.
|   |   |-- init.py                 # Weight initialization
|   |   |-- utils/                  # clip_grad_norm, rnn.pack_padded_sequence
|   |   |-- parallel/               # DataParallel, DistributedDataParallel
|   |-- optim/                      # Optimizers
|   |   |-- optimizer.py            # Base Optimizer class
|   |   |-- adam.py                 # Adam, AdamW
|   |   |-- sgd.py                  # SGD
|   |   |-- lr_scheduler.py         # Learning rate schedulers
|   |-- distributed/                # Distributed training
|   |   |-- __init__.py             # init_process_group, etc.
|   |   |-- rpc/                    # Remote procedure calls
|   |   |-- fsdp/                   # FullyShardedDataParallel
|   |   |-- pipeline/               # Pipeline parallelism
|   |   |-- _shard/                 # Tensor sharding
|   |   |-- tensor/                 # DTensor (distributed tensor)
|   |   |-- c10d/                   # Process group implementations
|   |-- utils/                      # Utility modules
|   |   |-- data/                   # DataLoader, Dataset, Sampler
|   |   |-- checkpoint.py           # Gradient checkpointing
|   |   |-- tensorboard.py          # TensorBoard integration
|   |-- _dynamo/                    # Dynamo compiler
|   |   |-- convert_frame.py        # Frame conversion (Python frame -> FX graph)
|   |   |-- bytecode_transformation.py  # Bytecode rewriting
|   |   |-- output_graph.py         # FX graph construction
|   |   |-- resume_execution.py     # Resume after graph breaks
|   |   |-- config.py               # Dynamo configuration
|   |   |-- guards.py               # Input validation guards
|   |   |-- variables/              # Variable tracking
|   |   |-- compiled_autograd.py    # Compiled autograd support
|   |-- _inductor/                  # Inductor code generation
|   |   |-- codegen/                # Code generation backends
|   |   |   |-- triton/             # Triton GPU kernel generation
|   |   |   |-- cpp/                # C++ CPU kernel generation
|   |   |-- scheduler.py            # Operator scheduling
|   |   |-- ir.py                   # Intermediate representation
|   |   |-- lowering.py             # FX node to IR lowering
|   |   |-- memory_planning.py      # Memory allocation planning
|   |   |-- decomposition/          # Operator decompositions
|   |-- fx/                         # FX graph transformation
|   |   |-- graph.py                # FX Graph class
|   |   |-- node.py                 # FX Node class
|   |   |-- tracer.py               # Symbolic tracer
|   |   |-- interpreter.py          # Graph interpreter
|   |   |-- passes/                 # Transformation passes
|   |-- _export/                    # torch.export
|   |-- jit/                        # TorchScript
|   |   |-- frontend/               # Python-to-TorchScript compiler
|   |   |-- runtime/                # TorchScript interpreter
|   |   |-- passes/                 # Optimization passes
|   |-- amp/                        # Automatic Mixed Precision
|   |-- compiler/                   # torch.compile public API
|   |-- _functorch/                 # functorch (torch.func)
|   |-- backends/                   # Backend integrations
|   |-- csrc/                       # C++ source for Python bindings
|
|-- aten/                           # A Tensor Library (C++ core)
|   |-- src/
|   |   |-- ATen/
|   |       |-- core/               # Core tensor types
|   |       |-- native/             # Operator kernel implementations
|   |       |-- ops/                # Per-operator organized files
|   |       |-- cpu/vec/            # SIMD vectorization
|   |       |-- transforms/         # Tensor transformations
|
|-- c10/                            # Core library (lowest level)
|   |-- core/                       # Fundamental types
|   |-- util/                       # Utility templates
|   |-- cuda/                       # CUDA abstractions
|   |-- xpu/                        # XPU abstractions
|
|-- torchgen/                       # Code generation system
|   |-- gen.py                      # Main code generator
|   |-- gen_backend_stubs.py        # Backend stub generation
|   |-- gen_functionalization_type.py # Functionalization generation
|   |-- gen_vmap_plumbing.py        # vmap plumbing generation
|   |-- model.py                    # Operator model
|
|-- caffe2/                         # Legacy Caffe2 (being integrated/removed)
|
|-- functorch/                      # functorch source
|   |-- _src/                       # Core functorch implementation
|   |-- dim/                        # torch.dim (named dimensions)
|
|-- test/                           # Test suite
|-- docs/                           # Documentation source
|-- cmake/                          # CMake modules
|-- tools/                          # Build and development tools
```

---

## 1.4 How the Dispatch System Works

The dispatch system is the heart of PyTorch's extensibility. It determines, for each operation call, which implementation (kernel) should be executed based on the properties of the input tensors.

### 1.4.1 Dispatch Keys

A **dispatch key** identifies a particular "aspect" or "mode" that modifies how an operation should behave. Dispatch keys include:

**Backend dispatch keys** (determine which hardware-specific implementation to use):

| Dispatch Key | Description |
|-------------|-------------|
| `CPU` | CPU backend |
| `CUDA` | NVIDIA GPU backend |
| `XPU` | Intel GPU backend |
| `MPS` | Apple Metal backend |
| `IPU` | Graphcore IPU backend |
| `XLA` | Google TPU / XLA backend |
| `MTIA` | Meta Training and Inference Accelerator |
| `Lazy` | Lazy tensor (deferred execution) |
| `Meta` | Shape/dtype inference without data computation |

**Functionality dispatch keys** (modify how operations behave regardless of backend):

| Dispatch Key | Description |
|-------------|-------------|
| `Autograd` | Records operation for automatic differentiation |
| `Autocast` | Automatic type casting for mixed precision |
| `Functionalize` | Transforms in-place ops to functional form |
| `Batched` | vmap batching rule |
| `Sparse` | Sparse tensor implementation |
| `SparseCsr` | CSR sparse tensor implementation |
| `QuantizedCPU` | Quantized CPU implementation |
| `QuantizedCUDA` | Quantized CUDA implementation |
| `NestedTensor` | Nested/jagged tensor implementation |
| `Python` | Python fallback implementation |
| `CompositeExplicitAutograd` | Generic implementation requiring autograd |
| `CompositeImplicitAutograd` | Generic implementation with implicit autograd |
| `ZeroTensor` | Zero-filled tensor optimization |

### 1.4.2 Dispatch Key Set

Each tensor carries a **DispatchKeySet** that represents the set of dispatch keys applicable to it. For example, a CUDA tensor that requires gradients would have a key set containing `{CUDA, AutogradCUDA}`. The dispatcher computes the intersection of all input tensor key sets and uses it to determine which kernel to invoke.

### 1.4.3 Dispatch Table

For each registered operator, the dispatcher maintains a **dispatch table** -- a mapping from dispatch keys to function pointers (kernels). When an operator is called:

1. The dispatcher collects the dispatch key set from all input tensors.
2. It looks up the highest-priority dispatch key in the dispatch table.
3. It calls the registered kernel for that key.

Priority is determined by the order of dispatch keys in the `DispatchKeySet`:

```
Functionalize > Python > Autocast > Batched > VmapMode > ... >
Autograd > Sparse > BackendSpecific > Composite > ...
```

### 1.4.4 Operator Registration

Operators are registered using macros and functions in the C++ code:

```cpp
// In native_functions.yaml (operator schema)
- func: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor

// Dispatch key registration (generated by torchgen)
TORCH_IMPL_FUNC(add_out)(const Tensor& self, const Tensor& other,
                          const Scalar& alpha, const Tensor& result) {
    // ... implementation ...
}

at::native::ADD_DISPATCH(CPU, add_cpu);
at::native::ADD_DISPATCH(CUDA, add_cuda);
at::native::ADD_DISPATCH(SparseCPU, add_sparse_cpu);
at::native::ADD_DISPATCH(SparseCUDA, add_sparse_cuda);
```

### 1.4.5 The Dispatcher Class

The core dispatch logic lives in `c10/core/Dispatcher.h`:

```cpp
class Dispatcher {
    // Singleton instance
    static Dispatcher& singleton();

    // Register an operator schema
    RegistrationHandleURL def(const FunctionSchema& schema);

    // Register a kernel for a specific dispatch key
    RegistrationHandleURL impl(
        const OperatorHandle& op,
        DispatchKey key,
        KernelFunction kernel
    );

    // Call an operator (performs dispatch)
    template<class Return, class... Args>
    Return call(const OperatorHandle& op, Args&&... args);

    // Redispatch (used within kernels to fall through to next key)
    template<class Return, class... Args>
    Return redispatch(const OperatorHandle& op,
                       DispatchKeySet keys, Args&&... args);
};
```

### 1.4.6 Dispatch Flow Example

When `torch.add(a, b)` is called where `a` is a CUDA tensor with `requires_grad=True` and `b` is a CUDA tensor:

```
1. Python: torch.add(a, b)
   --> Calls torch._C._ops.add.Tensor(a, b)

2. PyBind11: Routes to C++ at::add(a, b)

3. Dispatcher:
   a. Collects dispatch key set from inputs:
      a: {CUDA, AutogradCUDA}
      b: {CUDA}
      Intersection: {CUDA, AutogradCUDA}
   b. Looks up operator "add.Tensor" in dispatch table
   c. Highest priority key: AutogradCUDA
   d. Calls add_autograd kernel

4. Autograd kernel:
   a. Records the operation for backward
   b. Redispatches with key set minus AutogradCUDA: {CUDA}
   c. Calls add_cuda kernel

5. CUDA kernel:
   a. Launches CUDA kernel: add_kernel<<<blocks, threads>>>(...)
   b. Returns result tensor

6. Autograd kernel:
   a. Wraps result in autograd::Node
   b. Returns result with requires_grad=True
```

---

## 1.5 The Compilation Pipeline

PyTorch 2.x introduces a compilation pipeline that transforms standard Python code into optimized GPU/CPU kernels. The pipeline has three main stages: Dynamo, FX Graph, and Inductor.

### 1.5.1 Pipeline Overview

```
Python Code (eager)
    |
    v
[Dynamo] (torch._dynamo)
    |-- Bytecode analysis and transformation
    |-- Extracts computation subgraphs
    |-- Handles graph breaks (unsupported Python features)
    |-- Outputs: FX Graph
    |
    v
[FX Graph] (torch.fx)
    |-- Python-level intermediate representation
    |-- Nodes represent operations
    |-- Graph can be transformed, optimized, analyzed
    |-- Outputs: Optimized FX Graph
    |
    v
[Inductor] (torch._inductor)
    |-- Lowers FX graph to Inductor IR
    |-- Fuses operations into subgraphs
    |-- Generates optimized code:
    |   |-- GPU: Triton kernels
    |   |-- CPU: C++ with SIMD intrinsics
    |-- Memory planning and scheduling
    |-- Outputs: Compiled function
    |
    v
Compiled Function (callable from Python)
```

### 1.5.2 Dynamo (`torch._dynamo`)

Dynamo is a bytecode-level compiler that intercepts Python function execution and captures computation graphs.

**How it works:**

1. **Frame interception**: Dynamo hooks into Python's frame evaluation API (PEP 523) to intercept function calls.
2. **Bytecode analysis**: It reads and analyzes Python bytecode, tracking tensor operations and data flow.
3. **Graph extraction**: It constructs an FX graph from the analyzed bytecode, identifying pure tensor operations.
4. **Graph breaks**: When it encounters unsupported Python constructs (e.g., data-dependent control flow, certain C extensions), it "breaks" the graph -- the compiled subgraph ends, the unsupported code runs eagerly, and a new subgraph begins.
5. **Caching**: Compiled graphs are cached based on input tensor properties (shape, dtype, device).

**Key Dynamo components:**

```
torch/_dynamo/
    convert_frame.py      # Frame conversion (Python frame -> FX graph)
    bytecode_transformation.py  # Bytecode rewriting
    output_graph.py       # FX graph construction
    resume_execution.py   # Resume after graph breaks
    config.py             # Configuration
    guards.py             # Input validation guards
    variables/            # Variable tracking (TensorVariable, ConstantVariable, etc.)
    compiled_autograd.py  # Compiled autograd support
```

### 1.5.3 FX Graph (`torch.fx`)

FX is the intermediate representation used between Dynamo and Inductor.

**Key classes:**

- **`torch.fx.Graph`**: A container of `Node` objects representing a computation graph.
- **`torch.fx.Node`**: A single operation in the graph:
  - `op`: Operation type (`placeholder`, `get_attr`, `call_function`, `call_method`, `call_module`, `output`)
  - `target`: The function/method/module being called
  - `args`: Positional arguments
  - `kwargs`: Keyword arguments
  - `name`: Unique name for the node's output
  - `meta`: Metadata (tensor shape, dtype, etc.)
- **`torch.fx.GraphModule`**: An `nn.Module` whose forward method is defined by a `Graph`.

**Example FX graph:**

```python
import torch.fx

class MyModule(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x + 1)

gm = torch.fx.symbolic_trace(MyModule())
print(gm.graph)
# graph():
#     %x : [num_users=1] = placeholder[target=x]
#     %add : [num_users=1] = call_function[target=torch.add](args = (%x, 1), kwargs = {})
#     %relu : [num_users=1] = call_function[target=torch.relu](args = (%add,), kwargs = {})
#     return relu
```

### 1.5.4 Inductor (`torch._inductor`)

Inductor is the code generation backend that converts FX graphs into optimized machine code.

**Key stages:**

1. **Decomposition**: Complex operations are decomposed into simpler primitives.
2. **Lowering**: FX nodes are lowered to Inductor IR (buffer operations, loops, reductions).
3. **Scheduling**: Operations are grouped into "scheduling nodes" that determine fusion opportunities.
4. **Fusion**: Compatible operations are fused into a single kernel to minimize memory bandwidth.
5. **Memory planning**: Temporary buffers are planned to minimize memory allocation.
6. **Code generation**: Fused operations are compiled to:
   - **Triton** (for GPU): Generates Triton kernel code JIT-compiled to PTX.
   - **C++** (for CPU): Generates C++ code with SIMD intrinsics.

**Inductor directory structure:**

```
torch/_inductor/
    codegen/
        triton/              # Triton GPU code generation
        cpp/                 # C++ CPU code generation
    ir.py                    # Inductor IR definitions
    lowering.py              # FX node -> IR lowering
    scheduler.py             # Operation scheduling and fusion
    memory_planning.py       # Memory allocation planning
    graph.py                 # Inductor graph wrapper
    config.py                # Inductor configuration
    decomposition/           # Operator decompositions
```

### 1.5.5 Using torch.compile

```python
import torch

# Basic usage
compiled_model = torch.compile(model)
output = compiled_model(input)

# Backend selection
compiled_model = torch.compile(model, backend='inductor')  # default
compiled_model = torch.compile(model, backend='eager')     # tracing only
compiled_model = torch.compile(model, backend='cudagraphs') # CUDA graphs

# Mode selection
compiled_model = torch.compile(model, mode='default')
compiled_model = torch.compile(model, mode='reduce-overhead')
compiled_model = torch.compile(model, mode='max-autotune')

# Dynamic shapes
compiled_model = torch.compile(model, dynamic=True)

# Full graph (no graph breaks)
compiled_model = torch.compile(model, fullgraph=True)

# Disable for specific functions
@torch.compiler.disable
def my_function(x):
    return x

# Debugging
torch._dynamo.explain(model, *inputs)
```

---

## 1.6 Key Abstractions

### 1.6.1 Tensor

`torch.Tensor` is the central data structure in PyTorch. In C++, it is represented as `at::Tensor`, which wraps an `intrusive_ptr<TensorImpl>`.

```python
class torch.Tensor:
    # Properties
    shape: torch.Size          # Tensor dimensions
    dtype: torch.dtype         # Element type
    device: torch.device       # Device (cpu, cuda:0, etc.)
    layout: torch.layout       # Memory layout (strided, sparse)
    requires_grad: bool        # Whether gradients are tracked
    grad: Optional[Tensor]     # Gradient tensor
    grad_fn: Optional[Node]    # Autograd graph node
    ndim: int                  # Number of dimensions
    T: Tensor                  # Transpose (2D only)
    is_contiguous: bool        # Memory contiguity
    data_ptr: int              # Memory address
```

### 1.6.2 Storage

A `Storage` represents a contiguous block of memory that holds tensor data. Multiple tensors can share the same storage (via views).

```python
class torch.Storage:
    data_ptr() -> int          # Memory address
    size() -> int              # Number of elements
    element_size() -> int      # Bytes per element
    is_cuda() -> bool          # Whether on CUDA
    device: torch.device       # Device
    dtype: torch.dtype         # Element type
```

### 1.6.3 TensorImpl (C++)

`TensorImpl` is the C++ class that holds the actual tensor data and metadata:

```cpp
class TensorImpl : public c10::intrusive_ptr_target {
    c10::StorageImpl* storage_;           // Data storage
    c10::SymIntArrayRef sizes_;           // Tensor dimensions (may be symbolic)
    c10::SymIntArrayRef strides_;         // Strides for each dimension
    int64_t storage_offset_;              // Offset into storage
    c10::DispatchKeySet dispatch_key_set_; // Dispatch keys
    c10::ScalarType dtype_;               // Element type
    c10::Device device_;                  // Device

    const Storage& storage() const;
    IntArrayRef sizes() const;
    IntArrayRef strides() const;
    bool is_contiguous() const;
    void* data_ptr() const;
    int64_t numel() const;
    int64_t dim() const;
};
```

### 1.6.4 Dispatcher

The `Dispatcher` is the global singleton that routes operator calls to the appropriate kernel implementations. See Section 1.4 for full details.

### 1.6.5 DispatchKey

A `DispatchKey` is an enum value that identifies a particular dispatch behavior. See Section 1.4.1 for the complete list.

### 1.6.6 TensorOptions

`TensorOptions` bundles the configuration for creating a tensor:

```cpp
struct TensorOptions {
    c10::optional<ScalarType> dtype_;
    c10::optional<Device> device_;
    c10::optional<Layout> layout_;
    c10::optional<bool> requires_grad_;
    c10::optional<MemoryFormat> memory_format_;
};
```

### 1.6.7 Scalar

`Scalar` represents a single numerical value that can be of any supported type:

```cpp
class Scalar {
    Scalar(int64_t v);
    Scalar(double v);
    Scalar(bool v);
    Scalar(c10::complex<double> v);

    bool isIntegral(bool includeBool) const;
    bool isFloatingPoint() const;
    bool isComplex() const;
    bool isBoolean() const;

    int64_t toInt() const;
    double toDouble() const;
    bool toBool() const;
};
```

---

## 1.7 Build System Overview

### 1.7.1 CMake Build

PyTorch uses CMake as its primary build system:

```
1. CMake configuration (cmake ...)
   |-- Detects platform, compilers, CUDA/cuDNN, MKL, etc.
   |-- Generates build files (Makefile, Ninja, etc.)
   |-- Processes cmake/ dependencies

2. Code generation (torchgen)
   |-- Parses aten/src/ATen/native/native_functions.yaml
   |-- Generates C++ code for operator registration,
   |   dispatch tables, Python bindings, type-derived methods
   |-- Output: build/aten/src/ATen/ generated files

3. C++ compilation
   |-- Compiles c10/ sources
   |-- Compiles aten/ sources (native ops, CPU/CUDA kernels)
   |-- Compiles torch/csrc/ (Python bindings)
   |-- Links into libtorch.so and _C.so

4. Python packaging
   |-- Copies compiled libraries to torch/lib/
   |-- Creates wheel package
```

### 1.7.2 Key CMake Variables

```cmake
BUILD_PYTHON=ON              # Build Python bindings
BUILD_CAFFE2=ON              # Build Caffe2 (legacy)
BUILD_TEST=ON                # Build tests
USE_CUDA=ON                  # Enable CUDA support
USE_CUDNN=ON                 # Enable cuDNN support
USE_MKL=ON                   # Enable MKL support
USE_MKLDNN=ON                # Enable oneDNN support
USE_NCCL=ON                  # Enable NCCL support
USE_DISTRIBUTED=ON           # Enable distributed training
CMAKE_BUILD_TYPE=Release     # Release, Debug, RelWithDebInfo
```

### 1.7.3 Operator Definition: native_functions.yaml

All ATen operators are defined in `aten/src/ATen/native/native_functions.yaml`:

```yaml
- func: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
  dispatch:
    CPU: add_cpu
    CUDA: add_cuda
    SparseCPU: add_sparse_cpu
    SparseCUDA: add_sparse_cuda
  structured: True
  structured_delegate: add.out

- func: add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
  structured: True
  dispatch:
    CPU, CUDA: add_structured
```

### 1.7.4 setup.py

```bash
python setup.py install       # Build and install
python setup.py develop       # Develop mode (editable install)
python setup.py bdist_wheel   # Build wheel
python setup.py clean         # Clean build artifacts
```

---

## 1.8 Data Flow for a Typical Training Step

### 1.8.1 Complete Training Step Data Flow

```
Step 1: DATA LOADING
  DataLoader yields (inputs, targets)
  ├── Dataset.__getitem__(idx) -> (tensor, label)
  ├── Collate function stacks samples into batch
  └── Tensor pin_memory() for async CPU->GPU transfer

Step 2: DEVICE TRANSFER
  inputs = inputs.to('cuda')
  ├── If pinned memory: async DMA transfer
  └── Current CUDA stream synchronizes

Step 3: FORWARD PASS
  outputs = model(inputs)
  ├── nn.Module.__call__(inputs)
  │   ├── forward_pre_hooks (modify input)
  │   ├── module.forward(inputs)
  │   │   ├── Linear: F.linear(input, weight, bias)
  │   │   │   └── torch.addmm -> Dispatcher -> CUDA kernel (cuBLAS)
  │   │   ├── ReLU: torch.relu(hidden)
  │   │   │   └── Dispatcher -> CUDA kernel
  │   │   └── ... more layers ...
  │   └── forward_hooks (modify output)
  └── Autograd: Builds computation graph
      ├── Each op creates an autograd::Node
      ├── Node stores: saved tensors, backward function, edges
      └── Output tensor has grad_fn pointing to last Node

Step 4: LOSS COMPUTATION
  loss = criterion(outputs, targets)
  ├── CrossEntropyLoss:
  │   ├── F.log_softmax(outputs)
  │   └── F.nll_loss(log_probs, targets)
  └── Loss is a scalar tensor with grad_fn

Step 5: BACKWARD PASS (Reverse-Mode AD)
  loss.backward()
  ├── Initialize: grad_output = 1.0 (for scalar loss)
  ├── Topological sort of computation graph
  ├── For each Node (reverse order):
  │   ├── Node.backward(grad_output)
  │   │   └── Returns grad_input for each input
  │   ├── Accumulate: parameter.grad += grad_input
  │   └── Propagate grad to next nodes
  ├── Engine: CPU thread pool executes backward tasks
  └── Result: All parameters have .grad populated

Step 6: GRADIENT SYNCHRONIZATION (if DDP/FSDP)
  For DDP:
  ├── Buckets of gradients are AllReduce'd across GPUs
  └── NCCL: all_reduce(bucket.grad) -> averaged gradients
  For FSDP:
  ├── AllGather shard of parameter for forward
  └── ReduceScatter gradients for backward

Step 7: PARAMETER UPDATE
  optimizer.step()
  └── For each parameter:
      ├── Adam: param -= lr * m_hat / (sqrt(v_hat) + eps)
      └── All operations are in-place on parameter data

Step 8: GRADIENT RESET
  optimizer.zero_grad()
  └── Sets param.grad = None for all parameters
```

### 1.8.2 Memory Lifecycle During Training

```
GPU MEMORY LAYOUT:
  +--------------------------------------------------------+
  | Model Parameters (persistent)                          |
  | [weight1][weight2][bias1][weight3]...                  |
  +--------------------------------------------------------+
  | Optimizer State (persistent)                           |
  | [m1][v1][m2][v2]... (Adam moment estimates)           |
  +--------------------------------------------------------+
  | Activations (forward pass) -> Freed after backward     |
  | [input_batch][hidden1][hidden2][...][output]           |
  +--------------------------------------------------------+
  | Gradients (accumulated during backward)                |
  | [grad_w1][grad_b1][grad_w2]... Freed by zero_grad()   |
  +--------------------------------------------------------+
  | CUDA Caching Allocator Free Blocks                     |
  | [free][free][free]... Reused for next iteration        |
  +--------------------------------------------------------+
```

---

## 1.9 Component Relationships

### 1.9.1 Component Dependency Graph

```
torch (Python package)
  +-- torch._C (C extension, built from torch/csrc/)
  |     +-- ATen (C++ tensor library)
  |           +-- c10 (core library)
  |                 +-- CUDA runtime / CPU runtime
  +-- torch._dynamo (bytecode compiler)
  |     +-- torch.fx (graph IR)
  +-- torch._inductor (code generation)
  |     +-- Triton (GPU kernel generation)
  |     +-- C++ compiler (CPU kernel generation)
  +-- torch.autograd (automatic differentiation)
  |     +-- torch._C._autograd (C++ autograd engine)
  +-- torch.nn (neural network modules)
  |     +-- torch.autograd + ATen
  +-- torch.optim (optimizers)
  |     +-- torch.nn.Parameter + torch.autograd
  +-- torch.distributed (distributed training)
        +-- ProcessGroupNCCL / ProcessGroupGloo
```

### 1.9.2 Threading Model

```
Main Thread (Python)
  +-- Forward pass (Python -> C++ -> CUDA kernel launch)
  +-- Backward pass
  |     +-- Autograd engine thread pool (C++)
  |     |     |-- Worker threads execute backward tasks
  |     |     +-- Ready queue manages dependencies
  |     +-- CUDA kernels execute asynchronously on GPU
  +-- DataLoader worker processes (multiprocessing)
  |     +-- Prefetch queue feeds main thread
  +-- NCCL background threads (for distributed)
```

---

## 1.10 Version and Compatibility

- **PyTorch Version**: 2.7
- **Python Support**: 3.9+
- **CUDA Support**: CUDA 11.8, 12.x
- **C++ Compiler**: GCC 9+, Clang 12+, MSVC 2019+
- **Binary Size**: ~2GB for full CUDA package
- **Key Dependencies**: numpy, typing_extensions, filelock, jinja2, networkx, sympy

---

## 1.11 Summary

PyTorch's architecture is a carefully designed layered system where:

1. **c10** provides the foundational types (TensorImpl, Storage, Device, DispatchKey) that everything else builds on.
2. **ATen** implements all tensor operations with backend-specific kernels registered through the dispatcher.
3. **The Dispatcher** routes each operator call to the correct implementation based on tensor properties.
4. **The Python Frontend** (`torch`) wraps C++ operations in a Pythonic API.
5. **Autograd** automatically tracks operations during forward and computes gradients during backward.
6. **Dynamo + Inductor** provide just-in-time compilation that transforms eager Python code into optimized fused kernels.

This layered design allows PyTorch to be both easy to use (Python-first, eager execution) and highly performant (compiled fused kernels, efficient dispatch). The same operator call `torch.add()` seamlessly works across CPU, GPU, sparse, quantized, and other tensor types thanks to the dispatch system.
