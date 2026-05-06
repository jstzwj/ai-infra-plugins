# 01 — Overview and Architecture

## What is Apache TVM

Apache TVM is an open-source machine learning compilation framework that follows the principles of **Python-first development** and **universal deployment**. It takes pre-trained machine learning models, compiles them, and generates deployable modules that can be embedded and run everywhere — from cloud GPUs to mobile devices to bare-metal embedded systems.

### Key Principles

- **Python-first**: The optimization process is fully customizable in Python. It is easy to customize the optimization pipeline without recompiling the TVM stack.
- **Composable**: The optimization process is composable. It is easy to compose new optimization passes, libraries, and codegen into the existing pipeline.

### Key Goals

- **Optimize** performance of ML workloads by composing libraries and codegen.
- **Deploy** ML workloads to a diverse set of new environments, including new runtimes and new hardware.
- **Continuously improve and customize** the ML deployment pipeline in Python by quickly customizing library dispatching, bringing in customized operators and code generation.

---

## Overall Compilation Flow

The typical flow of using TVM to deploy a machine learning model consists of four stages:

### 1. Model Creation

Create the IRModule to be optimized and compiled. An IRModule contains a collection of functions that internally represent the model. Users can:

- Manually construct IRModule via **NNModule** (Relax NN frontend)
- Write IR using **TVMScript** DSL
- Import a pre-trained model from **Relax frontend** (PyTorch, ONNX, TFLite)

### 2. Transformation

The compiler transforms an IRModule to another functionally equivalent (or approximately equivalent, e.g., in quantization) IRModule. Many transformations are target (backend) independent. TVM also allows targets to affect the configuration of the transformation pipeline.

Transformations serve one of two purposes:
- **Optimization**: Transform a program to an equivalent, possibly more optimized version.
- **Lowering**: Transform a program to a lower-level representation closer to the target.

### 3. Target Translation

The compiler translates (codegen) the IRModule to an executable format specified by the target. The target translation result is encapsulated as a `runtime.Module` that can be exported, loaded, and executed on the target runtime environment.

Supported code generation paths:
- **LLVM**: In-memory LLVM IR for x86, ARM, RISC-V
- **Source-level**: CUDA C, OpenCL C, Metal shaders, Vulkan SPIR-V, WebGPU WGSL
- **External**: CUTLASS, TensorRT, cuBLAS, cuDNN via BYOC

### 4. Runtime Execution

The user loads a `runtime.Module` and runs the compiled functions in the supported runtime environment.

```python
import tvm
from tvm import relax

# Load the compiled artifact
mod = tvm.runtime.load_module("compiled_artifact.so")

# Create a VM instance on cuda(0)
vm = relax.VirtualMachine(mod, tvm.cuda(0))

# Run the model
result = vm["main"](input_data).numpy()
```

---

## Key Data Structures

### IRModule

**IRModule** is the primary data structure used across the entire stack. An IRModule (intermediate representation module) contains a collection of functions. TVM supports two primary variants:

- **`relax::Function`** — A high-level functional program representation. A `relax.Function` represents high-level graph structure, usually corresponding to an end-to-end model or a sub-graph. It can be viewed as a computational graph with additional support for control flow and complex data structures.

- **`tirx::PrimFunc`** — A low-level program representation containing loop-nest choices, multi-dimensional load/store, threading, and vector/tensor instructions. Usually represents an operator program that executes a (possibly-fused) layer in a model.

During compilation, all relax operators are lowered to `tirx::PrimFunc` or `TVM PackedFunc`, which can be executed directly on the target device.

### runtime.Module

**`runtime.Module`** encapsulates the result of compilation. It contains a `GetFunction` method to obtain `PackedFunc` instances by name.

### PackedFunc

**`PackedFunc`** is a type-erased function interface. It can take arguments and return values of the following types:
- POD types (int, float)
- String
- `PackedFunc`
- `runtime.Module`
- `runtime.Tensor` (NDArray)
- Other subclasses of `runtime.Object`

---

## Transformations

### Relax Transformations

Relax transformations contain a collection of passes that apply to relax functions. The optimizations include common graph-level optimizations such as:

- **Constant folding** and **dead-code elimination**
- **Operator fusion** (FuseOps, FuseTIR, FuseOpsByPattern)
- **Backend-specific optimizations** such as library dispatch
- **Legalization** — lowering relax operators to TIR PrimFunc (LegalizeOps)
- **Decomposition** — decomposing complex ops into simpler ones

### TensorIR Transformations

- **TensorIR Schedule**: TensorIR schedules optimize TensorIR functions for a specific target with user-guided instructions. For CPU targets, a TensorIR PrimFunc can generate valid code without scheduling but with very low performance. For GPU targets, scheduling is essential for generating valid code with thread bindings.
- **Lowering Passes**: These passes usually perform after the schedule is applied, transforming a PrimFunc into another functionally equivalent PrimFunc closer to the target-specific representation.

### Cross-level Transformations

Apache TVM enables cross-level optimization of end-to-end models. Since IRModule includes both Relax and TensorIR functions, cross-level transformations mutate the IRModule by applying different transformations to both types:

- **`relax.LegalizeOps`** — lowers relax operators, adds corresponding TensorIR PrimFunc, and replaces relax operators with calls to lowered PrimFunc.
- **Operator fusion pipeline** (`relax.FuseOps` + `relax.FuseTIR`) — fuses multiple consecutive tensor operations into a single kernel.
- **MetaSchedule** — automates the search of optimal TensorIR schedules.

---

## Target Translation

The target translation phase transforms an IRModule to the corresponding target executable format:

- For backends such as x86 and ARM, TVM uses the **LLVM IRBuilder** to build in-memory LLVM IR.
- TVM can also generate **source-level languages** such as CUDA C and OpenCL.
- TVM supports direct translations of a Relax function (sub-graph) to specific targets via **external code generators** (BYOC).

The **Target** structure specifies the compilation target:

```python
from tvm.target import Target

# From a registered tag
target = Target("nvidia/nvidia-a100")

# From a config dictionary
target = Target({"kind": "cuda", "arch": "sm_80"})

# From a tag with attribute overrides
target = Target({"tag": "nvidia/nvidia-a100", "l2_cache_size_bytes": 12345})
```

---

## Runtime Execution

TVM's runtime provides a minimal API for loading and executing compiled artifacts in multiple languages:

```python
import tvm

# Simple function execution
mod = tvm.runtime.load_module("compiled_artifact.so")
arr = tvm.runtime.ndarray.array([1, 2, 3], device=tvm.cuda(0))
fun = mod["addone"]
fun(arr)
print(arr.numpy())

# End-to-end model execution via VM
from tvm import relax
vm = relax.VirtualMachine(mod, tvm.cuda(0))
result = vm["main"](input_data).numpy()
```

### Runtime Architecture

- **`runtime.Module`** and **`PackedFunc`** are sufficient to encapsulate both operator-level programs and end-to-end models.
- **`runtime.Object`** is a reference-counted base class with a type index for runtime type checking and downcasting.
- **DLPack support** enables zero-copy exchange with PyTorch, TensorFlow, and other ecosystems.

---

## Logical Module Overview

### tvm/support
Contains common infrastructure utilities: generic arena allocator, socket, and logging.

### tvm/runtime
The foundation of the TVM stack. Provides the mechanism to load and execute compiled artifacts. Defines a stable set of C APIs to interface with frontend languages.

Key components:
- **`runtime::Object`** — reference-counted base class with type index
- **`PackedFunc`** — type-erased function interface
- **`runtime::Module`** — compilation result encapsulation
- **Device APIs** — CUDA, ROCm, Metal, OpenCL, Vulkan, WebGPU, Hexagon
- **RPC** — remote procedure call support for cross-compilation
- **VM** — Relax Virtual Machine
- **Disco** — distributed runtime

### tvm/node
Adds features on top of `runtime::Object` for IR data structures:
- **Reflection** — access any field of an IRNode by name in Python
- **Serialization** — serialize arbitrary IR nodes to JSON and back
- **Structural equivalence** and **hashing**

```python
x = tvm.tirx.Var("x", "int32")
y = tvm.tirx.Add(x, x)
# Access IR fields directly by name
assert y.a == x
```

### tvm/ir
Contains unified data structures and interfaces across all IR function variants:
- **IRModule** — the primary data structure
- **Type** — unified type system
- **PassContext** and **Pass** — pass infrastructure
- **Op** — common class for system-defined primitive operators

### tvm/relax
Relax is the high-level IR used to represent the computational graph of a model. Various optimizations are defined in `relax.transform`. Relax usually works closely with TensorIR — most transformations apply to both Relax and TensorIR functions in the IRModule.

Subdirectories:
- `analysis/` — analysis passes
- `backend/` — backend implementations (Adreno, CPU, CUDA, GPU, Metal, ROCm)
- `distributed/` — distributed computing support
- `frontend/` — model converters (ONNX, TFLite, PyTorch, StableHLO)
- `op/` — operator definitions (nn, math, image, vision, memory, etc.)
- `transform/` — transformation passes
- `training/` — training support
- `dpl/` — Dataflow Pattern Language

### tvm/tirx
Contains the core IR definitions and lowering infrastructure for TensorIR:
- IR data structures: PrimFunc, Buffer, SBlock, expressions, statements
- Analysis passes in `tirx/analysis`
- Transformation and lowering passes in `tirx/transform`

### tvm/s_tir (Schedulable TIR)
Contains schedule primitives and auto-tuning tools that operate on `tirx::PrimFunc`:
- Schedule primitives: `s_tir/schedule`
- Tensor intrinsics: `s_tir/tensor_intrin`
- MetaSchedule: automated performance tuning
- DLight: pre-defined high-performance schedules

### tvm/arith
Closely tied to TensorIR. Provides tools for (primarily integer) analysis of index arithmetic properties — positiveness, variable bounds, and integer sets describing iterator spaces. TIR passes use these analyses to simplify and optimize code.

### tvm/te (Tensor Expression)
A domain-specific language (DSL) for describing tensor computations. A tensor expression is not a self-contained function — use `te.create_prim_func` to convert to `tirx::PrimFunc`.

### tvm/topi (Tensor Operator Inventory)
Provides pre-defined operators found in common deep learning workloads. Saves the effort of constructing operators directly via TensorIR or TE for each use case.

### tvm/target
Contains all code generators that translate an IRModule to a target `runtime.Module`. Provides a common `Target` class that describes the target.

### tvm/script (TVMScript)
A Python-based DSL for writing TVM IR. Uses Python syntax with three import aliases: `I` (module-level), `T` (TIR), and `R` (Relax). Supports **roundtrip**: any IRModule can be printed back to TVMScript via `mod.script()` and re-parsed.

---

## Source Code Layout

### C++ Source (src/) — 640 files
```
src/
├── arith/          (22 files) — Arithmetic analysis
├── ir/             (23 files) — Core IR infrastructure
├── relax/          (95 files) — Relax IR + transforms
│   ├── analysis/
│   ├── backend/
│   ├── distributed/
│   ├── ir/
│   ├── op/
│   ├── transform/   (58 files — most active area)
│   └── ...
├── s_tir/          (156 files) — TensorIR scheduling + MetaSchedule
│   ├── analysis/
│   ├── backend/
│   ├── meta_schedule/
│   ├── schedule/
│   └── transform/
├── tirx/           (67 files) — Extended TIR
├── runtime/        — Runtime system
│   ├── cuda, rocm, metal, opencl, vulkan, webgpu, hexagon
│   ├── rpc, minrpc
│   ├── vm, disco
│   └── contrib
├── target/         — Code generation
│   ├── llvm, cuda, opencl, vulkan, metal, rocm, hexagon, webgpu
│   └── source
├── te/             — Tensor Expression
├── topi/           — Operator library
├── script/         — TVMScript printer/builder
└── support/        — Utilities
```

### Python Source (python/tvm/) — 745 files
```
python/tvm/
├── relax/          — Relax IR Python bindings
├── s_tir/          — Schedulable TIR (138 files)
│   ├── meta_schedule/
│   ├── schedule/
│   ├── dlight/
│   └── transform/
├── tirx/           — Extended TIR
├── runtime/        — Runtime (177 files)
│   └── disco/
├── topi/           — Operator library (210 files)
├── target/         — Target configuration
├── ir/             — Core IR
├── arith/          — Arithmetic analysis
├── te/             — Tensor Expression
├── script/         — TVMScript
├── contrib/        — External integrations
├── driver/         — Build driver
└── rpc/            — RPC support
```

### Other Directories
- **include/** — C++ headers
- **docs/** — RST documentation (87 files)
- **tests/** — Test suites
- **apps/** — Applications (RPC servers, mobile apps)
- **3rdparty/** — Dependencies (cutlass, tensorrt_llm, libflash_attn, compiler-rt, OpenCL-Headers)
- **ci/** — CI scripts
- **cmake/** — Build configuration
- **docker/** — Docker files
- **web/** — Web frontend
- **jvm/** — Java bindings

---

## Summary

Apache TVM is a sophisticated ML compiler with:

- **3,000+ source files** (640 C++, 745 Python)
- **Comprehensive optimization**: graph-level, tensor-level, and cross-level
- **Multi-backend support**: CPU (x86, ARM, RISC-V), GPU (NVIDIA, AMD, Apple), DSP (Qualcomm), Web
- **Python-first design**: full pipeline customization without C++ recompilation
- **Universal runtime**: deploy anywhere with minimal runtime support
- **Extensible**: BYOC, custom patterns, custom passes, custom targets
