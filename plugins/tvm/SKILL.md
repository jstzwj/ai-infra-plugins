# Apache TVM — Machine Learning Compilation Framework

Apache TVM is an open-source machine learning compilation framework that optimizes and deploys ML models across diverse hardware backends. It follows the principles of **Python-first development** and **universal deployment**.

## Activation

Activate this skill when the user asks about:
- Apache TVM, TVM compiler, ML compilation, model optimization, model deployment
- TensorIR, Relax IR, TVMScript, MetaSchedule, DLight
- Model import from PyTorch, ONNX, TensorFlow Lite
- GPU/CPU/DSP code generation, operator fusion, auto-tuning
- Cross-compilation, RPC, runtime deployment
- BYOC (Bring Your Own Codegen), external library dispatch
- Tensor Expression (TE), TOPI operator library
- Dataflow Pattern Language (DPL), graph pattern matching and rewriting
- LLM optimization, paged KV cache, distributed inference
- Relax Virtual Machine, Disco distributed runtime

## Overview

TVM takes pre-trained ML models, compiles them, and generates deployable modules that can run everywhere — from cloud GPUs to mobile devices to bare-metal embedded systems. The optimization pipeline is fully customizable in Python without recompiling the TVM stack.

### Key Compilation Flow

1. **Import/Construct** — Import models from PyTorch, ONNX, TFLite, or construct directly using Relax frontend / TVMScript
2. **Optimize** — Apply composable transformations via pipelines (graph optimizations, tensor program optimization, library dispatch)
3. **Build** — Translate IRModule to target-specific executable format
4. **Deploy** — Load and execute via minimal runtime (Python, C++, Java, Rust, JavaScript)

### Core Architecture

```
IRModule
├── relax::Function     — High-level graph IR (computational graph + control flow)
├── tirx::PrimFunc      — Low-level tensor IR (loop nests, buffers, threading)
└── PackedFunc          — Runtime-callable functions

Key Modules:
├── tvm/relax           — Graph-level IR, transformations, frontends, VM
├── tvm/tirx            — Core TIR IR definitions and lowering
├── tvm/s_tir           — Schedulable TIR (schedule primitives, MetaSchedule, DLight)
├── tvm/target          — Hardware target abstractions and code generation
├── tvm/runtime         — Runtime system (PackedFunc, Module, VM, Disco)
├── tvm/script          — TVMScript parser and printer
├── tvm/te              — Tensor Expression DSL
├── tvm/topi            — Tensor Operator Inventory (pre-defined operators)
├── tvm/arith           — Arithmetic analysis (bounds, simplification)
├── tvm/ir              — Unified IR infrastructure (IRModule, Pass, Op, Type)
└── tvm/contrib         — External library integrations (cuBLAS, cuDNN, CUTLASS)
```

## Quick Reference

### Importing Models
```python
# From PyTorch
from tvm.relax.frontend.torch import from_exported_program
mod = from_exported_program.exported_program(model, args)

# From ONNX
from tvm.relax.frontend.onnx import from_onnx
mod = from_onnx(onnx_model, shape_info, dtype_dict)

# From TFLite
from tvm.relax.frontend.tflite import from_tflite
mod = from_tflite(tflite_model, shape_dict, dtype_dict)
```

### Building and Deploying
```python
import tvm
from tvm import relax

# Apply optimization pipeline
mod = relax.get_pipeline("zero")(mod)

# Build for target
exec = relax.build(mod, target="nvidia/nvidia-a100")

# Deploy
vm = relax.VirtualMachine(exec, tvm.cuda(0))
result = vm["main"](input_data)
```

### TVMScript
```python
from tvm.script import ir as I, tirx as T, relax as R

@I.ir_module
class MyModule:
    @T.prim_func
    def my_kernel(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32")):
        for i in range(128):
            with T.sblock("B"):
                vi = T.axis.spatial(128, i)
                B[vi] = A[vi] * T.float32(2.0)

    @R.function
    def main(x: R.Tensor((128,), "float32")) -> R.Tensor((128,), "float32"):
        with R.dataflow():
            lv = R.call_tir(cls.my_kernel, (x,), out_sinfo=R.Tensor((128,), "float32"))
            R.output(lv)
        return lv
```

### Schedule Primitives
```python
sch = tvm.s_tir.Schedule(mod)
block = sch.get_block("Y")
i, j, k = sch.get_loops(block)
sch.tile(i, j, factors=[32, 32])  # tiling
sch.vectorize(...)                  # vectorization
sch.bind(..., "threadIdx.x")       # GPU thread binding
sch.reorder(...)                    # loop reordering
```

## Reference Chapters

See the `references/` directory for comprehensive documentation:

| # | Chapter | Description |
|---|---------|-------------|
| 01 | [Overview & Architecture](references/01-overview-and-architecture.md) | Compilation flow, key data structures, module overview |
| 02 | [Getting Started](references/02-getting-started.md) | Installation, quick start tutorial, basic concepts |
| 03 | [IRModule & Core IR](references/03-irmodule-and-core-ir.md) | IRModule, unified type system, Pass infrastructure |
| 04 | [Relax IR](references/04-relax-ir.md) | Graph abstraction, struct info, dataflow blocks, R.call_tir |
| 05 | [Relax Transformations](references/05-relax-transformations.md) | All relax transform passes (FuseOps, LegalizeOps, etc.) |
| 06 | [Relax Frontend](references/06-relax-frontend.md) | Importing from PyTorch, ONNX, TFLite |
| 07 | [Relax Operators](references/07-relax-ops.md) | Complete operator reference (nn, math, image, vision, etc.) |
| 08 | [Relax Virtual Machine](references/08-relax-vm.md) | VM architecture, instruction set, execution model |
| 09 | [Relax Distributed](references/09-relax-distributed.md) | Disco distributed runtime, device mesh, collective ops |
| 10 | [TensorIR Abstraction](references/10-tensor-ir-abstraction.md) | PrimFunc, Buffer, SBlock, axis annotations |
| 11 | [TensorIR Expressions & Statements](references/11-tir-expressions-and-statements.md) | TIR expressions, statements, operators |
| 12 | [TIR Transformations](references/12-tir-transformations.md) | Lowering passes, optimization passes |
| 13 | [TIR Analysis](references/13-tir-analysis.md) | Analysis passes (dependence, bound, var touch) |
| 14 | [Schedule Primitives](references/14-schedule-primitives.md) | Complete scheduling API (tile, vectorize, bind, etc.) |
| 15 | [MetaSchedule](references/15-meta-schedule.md) | Auto-tuning framework, search strategy, database |
| 16 | [DLight](references/16-dlight.md) | Pre-defined high-performance schedule rules |
| 17 | [Tensor Intrinsics](references/17-tensor-intrinsics.md) | Hardware-specific tensor intrinsics |
| 18 | [TVMScript](references/18-tvmscript.md) | DSL syntax, parser, printer, roundtrip |
| 19 | [Target System](references/19-target-system.md) | Target configuration, code generation backends |
| 20 | [Runtime System](references/20-runtime-system.md) | PackedFunc, Module, Object system, device APIs |
| 21 | [TE & TOPI](references/21-te-and-topi.md) | Tensor Expression DSL and operator inventory |
| 22 | [Arithmetic Module](references/22-arith-module.md) | Bounds analysis, integer sets, simplification |
| 23 | [DPL Pattern Language](references/23-dpl-pattern-language.md) | Pattern matching, rewriting, FusionPattern |
| 24 | [BYOC & External Dispatch](references/24-byoc-and-external-dispatch.md) | Bring Your Own Codegen pipeline |
| 25 | [Operator Fusion](references/25-operator-fusion.md) | Fusion algorithm, pattern classification, FuseTIR |
| 26 | [Cross Compilation & RPC](references/26-cross-compilation-and-rpc.md) | Remote execution, cross-compilation workflow |
| 27 | [LLM Optimization](references/27-llm-optimization.md) | LLaMA, paged KV cache, attention optimization |
| 28 | [Module Serialization](references/28-module-serialization.md) | Export/import compiled artifacts |
| 29 | [Code Generation](references/29-code-generation.md) | LLVM, CUDA C, OpenCL, Vulkan codegen |
| 30 | [Installation Guide](references/30-installation-guide.md) | Building from source, dependencies, platform notes |
| 31 | [Pass Infrastructure](references/31-pass-infrastructure.md) | PassContext, Pass base classes, pipeline composition |
| 32 | [Device & Target Interactions](references/32-device-target-interactions.md) | Device APIs, target attributes, compilation flow |
| 33 | [Error Handling & Debugging](references/33-error-handling-and-debugging.md) | Error types, debugging strategies, BasePyModule |
| 34 | [Testing & Benchmarking](references/34-testing-and-benchmarking.md) | Testing framework, pytest integration, benchmarks |
| 35 | [Contributing Guide](references/35-contributing-guide.md) | Code style, PR process, CI, documentation |

## Important Notes

- TVM uses **destination-passing style** for low-level tensor functions (output allocated externally)
- **Symbolic shapes** are first-class in Relax (`"n"` in tensor shapes)
- Dataflow blocks (`R.dataflow()`) mark pure computation regions for optimization
- MetaSchedule automates schedule search; DLight provides pre-defined high-performance rules
- TVMScript is parsed from Python AST — it is NOT executed by the Python interpreter
- The runtime is language-agnostic: C API enables bindings for Python, Rust, Go, Java, JS
