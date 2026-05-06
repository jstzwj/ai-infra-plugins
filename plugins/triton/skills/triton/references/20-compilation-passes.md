# Chapter 20: Compilation Passes

## Pass Pipeline Overview

```
TTIR Passes → TritonToTritonGPU → TTGIR Passes → TritonGPUToLLVM → LLVM Passes
```

## Common Passes

| Pass | Description |
|------|-------------|
| `SCCP` | Sparse Conditional Constant Propagation |
| `SymbolDCE` | Dead code elimination for symbols |
| `Inliner` | Function inlining |
| `Canonicalizer` | Operation canonicalization |
| `CSE` | Common Subexpression Elimination |
| `LICM` | Loop Invariant Code Motion |

## TTIR Passes (Triton IR)

| Pass | Description |
|------|-------------|
| `CombineOps` | Combine multiple operations |
| `ReorderBroadcast` | Reorder broadcast operations |
| `RewriteTensorDescriptor` | Rewrite tensor descriptor ops |
| `LoopUnroll` | Unroll loops |
| `TritonLICM` | Loop invariant code motion for Triton |
| `LoopAwareCSE` | CSE aware of loop structures |
| `ConvertTritonToTritonGPU` | Main conversion to GPU dialect |

## TTGIR Passes (Triton GPU IR)

### Memory Optimization
| Pass | Description |
|------|-------------|
| `Coalesce` | Optimize memory access coalescing |
| `OptimizeThreadLocality` | Improve data locality across threads |
| `RemoveLayoutConversions` | Remove unnecessary layout changes |
| `OptimizeDotOperands` | Optimize dot product memory access |
| `ReduceDataDuplication` | Eliminate redundant data copies |

### Memory Layout
| Pass | Description |
|------|-------------|
| `AccelerateMatmul` | Detect and optimize matmul patterns |
| `Pipeline` | Software pipelining for loops |
| `Prefetch` | Memory prefetching |
| `AllocateSharedMemory` | Shared memory allocation |

### Scheduling
| Pass | Description |
|------|-------------|
| `AssignLatencies` | Assign operation latencies |
| `ScheduleLoad` | Schedule memory loads |
| `WarpSpecialize` | Warp-level specialization |
| `ReorderInstructions` | Instruction reordering |

### Conversion
| Pass | Description |
|------|-------------|
| `DecomposeUnsupportedConversions` | Break down complex conversions |
| `FinalizeMembar` | Insert memory barriers |
| `ConvertLayout` | Layout conversion |

### Memory Management
| Pass | Description |
|------|-------------|
| `AllocateSharedMemory` | Allocate shared memory buffers |
| `Barrier` | Insert synchronization barriers |
| `HoistTMemAlloc` | Move tensor memory allocations |

## Conversion Passes

### TritonToTritonGPU
Converts generic Triton IR to GPU-specific IR with layout encodings.

```
tt.load → ttg.local_load (with #blocked layout)
tt.dot  → ttg.mma (with #dot_op layout)
```

### TritonGPUToLLVM
Converts GPU operations to LLVM IR with target-specific intrinsics.

| Sub-pass | Description |
|----------|-------------|
| `DotOpToLLVM` | Matrix multiply → NVVM/ROCDL intrinsics |
| `MemoryOpToLLVM` | Load/store → LLVM memory operations |
| `ElementwiseOpToLLVM` | Arithmetic → LLVM vector operations |
| `FuncOpToLLVM` | Functions → LLVM kernel functions |
| `ConvertLayoutOpToLLVM` | Layout conversions → memory copies |
| `AssertOpToLLVM` | Assertions → __assertfail calls |
| `PrintOpToLLVM` | Printing → printf calls |
| `HistogramOpToLLVM` | Histogram → atomic operations |

## LLVM IR Passes

| Pass | Description |
|------|-------------|
| `DebugScope` | Debug scope information |
| `DebugLocalVariable` | Local variable debug info |
| `BreakStructPhiNodes` | Break struct phi nodes |
| `CanonicalizeLLVMIR` | Canonicalize LLVM IR |

## NVIDIA-Specific Passes

Located in `third_party/nvidia/`:

| Pass | Description |
|------|-------------|
| `F32DotTC` | TF32 tensor core conversion |
| `OptimizeAccConversion` | Optimize accumulator conversion |
| `OptimizeDotOperands` | NVIDIA-specific dot optimization |
| `RemoveLayoutConversions` | Layout conversion removal |
| `WarpSpecialization` | Hopper warp specialization |

## AMD-Specific Passes

Located in `third_party/amd/`:

| Pass | Description |
|------|-------------|
| `AccelerateMatmul` | AMD MFMA matmul optimization |
| `OptimizeDotOperands` | AMD-specific dot optimization |
| `RemoveLayoutConversions` | Layout conversion removal |
| `BufferOps` | Buffer operation optimization |
| `InThreadTranspose` | In-thread transpose optimization |
| `BlockPingpong` | Ping-pong scheduling |
| `StreamPipeline` | Stream pipeline optimization |

## Plugin Passes

Triton supports out-of-tree plugin passes:

```python
def inspect_stages(self, stages, options, language, capability):
    # Add custom passes
    stages["custom"] = (my_custom_pass, True)
    return stages

triton.knobs.runtime.add_stages_inspection_hook = inspect_stages
```
