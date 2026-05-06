# Chapter 1: Introduction

## 1.1 Overview

Tile IR is a portable, low-level tile virtual machine and instruction set designed to
provide a first-class abstraction for tile-based programming on GPU hardware. It
represents a fundamental shift in how developers reason about and express data-parallel
computation on accelerators.

Traditional GPU programming models such as PTX (Parallel Thread Execution) model the GPU
as a data-parallel SIMT (Single Instruction, Multiple Thread) processor. In the SIMT
model, the programmer writes code from the perspective of a single scalar thread, and the
hardware executes these threads in lockstep groups called warps. While this model has
served the CUDA ecosystem well for general-purpose GPU programming, it creates
significant friction when expressing tile-based workloads that operate on multi-
dimensional array fragments (tiles).

Tile IR takes a different approach: it models the GPU as a **tile-based processor** where
each logical thread -- called a **tile block** -- computes over partial fragments
(**tiles**) of multi-dimensional arrays (**tensors**). This alignment between the
programming abstraction and the underlying hardware execution model (tensor cores,
shared memory tiles, register-level fragment storage) enables both higher programmer
productivity and improved performance.

### Key Characteristics

- **Tile-native abstraction.** The fundamental unit of computation is a tile, not a
  scalar thread. Operations work directly on multi-dimensional array fragments.
- **Portable bytecode.** Tile IR defines a versioned bytecode format that can be
  compiled by the CUDA driver or a standalone toolkit, ensuring forward compatibility
  with future hardware.
- **Hardware agnostic at the tile level.** The Tile IR specification abstracts away the
  exact mapping of tiles to hardware resources such as warps, registers, and shared
  memory, while still providing hints for performance-critical scenarios.
- **Complementary to PTX.** Tile IR is not a replacement for PTX; rather, it is a
  complementary instruction set that coexists within the CUDA platform. Programs can
  interoperate between Tile IR and PTX / CUDA C++ seamlessly.

> **Note:** Tile IR is primarily intended as a compilation target for domain-specific
> languages (DSLs), compilers, and frameworks. While it is possible to write Tile IR
> programs by hand, the primary audience consists of toolchain authors and performance
> engineers.

---

## 1.2 Scalable Data Parallel Computing with Tiles on GPU

### 1.2.1 The Evolution of GPU Programming Models

The CUDA platform has historically provided portability through two primary mechanisms:

1. **CUDA C++**, a high-level programming language that compiles to GPU machine code.
2. **PTX**, a low-level intermediate instruction set that provides a stable, forward-
   compatible abstraction over GPU hardware.

Both of these are built on the SIMT execution model. A CUDA program specifies the
behavior of a single thread; the hardware and compiler are responsible for mapping many
such threads onto the physical execution units of the GPU. This model maps well to
scalar and vector workloads but becomes increasingly cumbersome for workloads that
naturally operate on matrix and tensor fragments.

### 1.2.2 The Rise of Tensor Cores and Tile-Based Execution

Modern NVIDIA GPUs introduced tensor cores -- specialized hardware units that perform
matrix-multiply-accumulate (MMA) operations on fixed-size matrix fragments. The rapid
evolution of tensor core capabilities across GPU generations has dramatically increased
programming complexity:

- **Volta (V100):** Introduced mixed-precision MMA with 4x4x4 or 8x8x4 fragments.
- **Turing (T4):** Added INT8, INT4, and binary MMA operations.
- **Ampere (A100):** Expanded to support TF32, BF16, and larger fragment sizes.
- **Hopper (H100):** Introduced the TMA (Tensor Memory Accelerator) unit and the
  distributed shared memory model with cluster-level tile operations.
- **Blackwell and beyond:** Further extensions to tile sizes, data types, and memory
  hierarchy.

Each generation has introduced new instructions, data types, and fragment layouts. PTX
exposes these capabilities through specialized warp-level MMA instructions that require
the programmer to manage register layouts, fragment swizzling, and cross-lane data
movement explicitly. This places a significant burden on developers and compiler
authors.

### 1.2.3 The Tile IR Solution

Tile IR introduces a virtual instruction set that enables **native programming in terms
of tiles**. Rather than requiring the programmer to reason about individual CUDA threads
and their register assignments, Tile IR provides:

| Aspect | Traditional (PTX/CUDA C++) | Tile IR |
|--------|---------------------------|---------|
| Unit of computation | Scalar thread | Tile block |
| Data granularity | Per-thread scalar / vector | Multi-dimensional tile fragment |
| Tensor core access | Explicit warp-level MMA instructions | Abstracted tile-level operations |
| Memory hierarchy mapping | Manual shared memory / register management | Automated tile placement and layout |
| Hardware evolution burden | Programmer must adapt per-generation | Tile IR compiler handles mapping |

The key design principle is **separation of concerns**: the developer focuses on
partitioning a data-parallel program into tiles and tile blocks, while the Tile IR
compiler handles mapping these onto hardware resources such as threads, the memory
hierarchy, and tensor cores.

### 1.2.4 Raising the Abstraction Level

By operating at the tile level, Tile IR raises the abstraction level sufficiently to
enable new categories of tooling:

- **Domain-specific languages (DSLs).** Language authors can compile directly to Tile IR
  without implementing low-level hardware resource management.
- **Optimizing compilers.** Compiler frameworks can target Tile IR as a portable
  intermediate representation, delegating final code generation to the Tile IR compiler.
- **High-performance frameworks.** Framework developers can express tile-based kernels
  without the boilerplate associated with warp-level programming in PTX or CUDA C++.
- **Research prototypes.** Researchers investigating tile-based programming models can
  use Tile IR as a stable compilation target that evolves with hardware.

```
+-------------------+     +-------------------+     +-------------------+
|   DSL / Compiler  |     |   Triton / MLIR   |     |  Framework / App  |
+--------+----------+     +--------+----------+     +--------+----------+
         |                         |                         |
         v                         v                         v
   +-----+-------------------------+-------------------------+-----+
   |                        Tile IR Bytecode                        |
   +-----+-------------------------+-------------------------+-----+
         |                         |                         |
         v                         v                         v
   +-----+----------+     +--------+----------+     +--------+------+
   |  CUDA Driver   |     | Standalone Toolkit |     |   AOT Compiler |
   | (JIT)          |     | (Optimization)     |     |   (Offline)    |
   +----------------+     +--------------------+     +----------------+
         |                         |                         |
         v                         v                         v
   +------------------------------------------------------------------+
   |                     GPU Machine Code (SASS)                       |
   +------------------------------------------------------------------+
```

*Figure 1.1: Tile IR compilation pipeline. Multiple frontends target Tile IR bytecode,
which is then compiled to GPU machine code by the driver, toolkit, or an ahead-of-time
compiler.*

---

## 1.3 Goals and Scope

### 1.3.1 Core Goals

Tile IR is designed around the following core goals, listed in order of priority:

**Goal 1: Introduce a data-parallel tile programming abstraction aligned with programmer
intent.**

The primary goal is to provide a programming abstraction where the constructs available
to the programmer directly correspond to the concepts they are trying to express. When a
programmer thinks in terms of "load a tile of matrix A, load a tile of matrix B, compute
a matrix multiply-accumulate, store the result tile," the Tile IR instruction set should
mirror this mental model with minimal impedance mismatch.

**Goal 2: Abstract tensor cores and their programming model for hardware innovation.**

Tensor core programming interfaces have changed significantly across GPU generations and
will continue to evolve. Tile IR aims to abstract these details so that programs written
today can take advantage of future tensor core improvements without modification. The
Tile IR compiler is responsible for mapping abstract tile operations onto the specific
tensor core instructions available on each GPU generation.

**Goal 3: Abstract low-level architecture-specific details.**

CUDA threads, warps, thread blocks, shared memory bank conflicts, register allocation,
and fragment layout are all architecture-specific implementation details that Tile IR
aims to abstract. The programmer specifies *what* tiles to compute, and the compiler
determines *how* to map them onto hardware resources.

**Goal 4: Minimize abstraction overhead.**

Abstraction often comes at a performance cost. A key design constraint for Tile IR is
that the performance overhead of the tile abstraction should be modest -- ideally zero
for well-optimized programs. This is achieved by designing the abstraction around
operations that map cleanly onto hardware capabilities and by providing escape hatches
for performance-critical code paths.

**Goal 5: Provide user controls and optimization hints for peak performance.**

While the Tile IR compiler makes best-effort decisions for tile placement and scheduling,
some workloads require explicit control for peak performance. Tile IR provides a set of
user controls and optimization hints that allow advanced users to guide compilation
decisions without sacrificing portability. These hints are advisory; the compiler may
ignore them if they are inapplicable to the target hardware.

**Goal 6: Provide seamless interoperability with CUDA C++ and PTX.**

Tile IR does not exist in isolation. Real-world applications will need to combine Tile IR
kernels with existing CUDA C++ and PTX code. The specification ensures that Tile IR
programs can:

- Call functions written in CUDA C++ or PTX.
- Be called from CUDA C++ host code.
- Share memory and synchronization primitives with CUDA C++ / PTX code within the same
  kernel launch.
- Link with existing CUDA libraries and runtime infrastructure.

### 1.3.2 Key Components

The Tile IR ecosystem consists of the following components, listed in order of
importance:

1. **Versioned specification of the Tile IR abstract machine with portable bytecode.**

   This is the foundational artifact. The specification defines:
   - The abstract machine model (tile blocks, tile memory, execution semantics).
   - The instruction set (tile load, store, compute, reduce, etc.).
   - The type system (tile types, tensor types, element types).
   - The binary format (bytecode encoding, section layout, metadata).
   - Stability guarantees and versioning policy.

   The bytecode format is designed to be forward-compatible: bytecode compiled for an
   earlier version of the specification can be executed on any future GPU that supports
   the required operations, with the compiler handling any necessary adaptation.

2. **Optimizing compiler (part of CUDA driver and standalone toolkit).**

   The Tile IR compiler translates Tile IR bytecode into optimized GPU machine code
   (SASS). It is available in two forms:
   - **Driver-integrated compiler:** Enables just-in-time (JIT) compilation of Tile IR
     bytecode at kernel launch time, similar to how PTX is JIT-compiled by the CUDA
     driver.
   - **Standalone toolkit compiler:** Provides ahead-of-time (AOT) compilation for
     deployment scenarios where JIT compilation is undesirable, as well as optimization
     passes that can be run offline.

3. **MLIR dialect for existing compilers.**

   An MLIR (Multi-Level Intermediate Representation) dialect provides integration with
   the broader compiler ecosystem. The Tile IR MLIR dialect enables:
   - Direct emission of Tile IR from MLIR-based compiler pipelines.
   - Interoperation with other MLIR dialects (e.g., `linalg`, `affine`, `scf`).
   - Reuse of MLIR infrastructure for analysis, transformation, and code generation.

### 1.3.3 What Tile IR Is Not

To set appropriate expectations, it is worth clarifying what Tile IR does not aim to be:

- **Tile IR is not a high-level programming language.** It is an instruction set and
  virtual machine specification. Users are expected to write Tile IR through compilers,
  DSLs, or framework-level APIs, not by hand-coding bytecode.
- **Tile IR is not a replacement for CUDA C++ or PTX.** It is a complementary
  abstraction that coexists alongside these technologies. Simd-style workloads that do
  not benefit from tile abstractions should continue to use CUDA C++ or PTX.
- **Tile IR is not a runtime system.** It does not define kernel launch mechanics,
  stream management, or device management. These responsibilities remain with the CUDA
  runtime and driver APIs.
- **Tile IR is not hardware-specific.** While it is initially designed with NVIDIA GPU
  hardware in mind, the specification is intended to be portable to any hardware that
  supports tile-based execution.

---

## 1.4 Document Structure

This document is organized into the following chapters:

| Chapter | Title | Description |
|---------|-------|-------------|
| 1 | **Introduction** | Overview, motivation, goals, and scope (this chapter). |
| 2 | **Programming Model** | Describes the Tile IR programming model including tile blocks, tiles, execution model, and synchronization semantics. |
| 3 | **Syntax** | Defines the textual syntax of Tile IR, including instruction mnemonics, operand notation, and directive syntax. |
| 4 | **Binary Format** | Describes the Tile IR bytecode binary encoding, section layout, header format, and metadata tables. |
| 5 | **Type System** | Defines the Tile IR type system including element types, tile types, tensor types, and memory space qualifiers. |
| 6 | **Semantics** | Provides semi-formal operational semantics for each Tile IR operation category. |
| 7 | **Memory Model** | Describes the Tile IR memory model including tile memory, global memory, shared memory, and consistency guarantees. |
| 8 | **Operations** | Full listing of all Tile IR operations organized by category (arithmetic, memory, control flow, synchronization, etc.). |
| 9 | **Debug Info** | Defines the debug information format for Tile IR, including source mapping, variable tracking, and stack unwinding. |
| 10 | **Stability** | Stability guarantees, versioning policy, and deprecation rules for Tile IR. |
| A | **Appendix** | Reference materials including full program examples, quick reference cards, and errata. |

> **Note:** Chapters are designed to be read in order for a first pass, but can also be
> used as standalone references. Cross-references are provided throughout.

---

## 1.5 How Tile IR Differs from PTX

Tile IR and PTX are both low-level, portable instruction sets that target NVIDIA GPUs.
However, they differ fundamentally in their programming models, abstractions, and target
workloads. The following table provides a detailed comparison.

### 1.5.1 Detailed Comparison

| Dimension | PTX | Tile IR |
|-----------|-----|---------|
| **Programming model** | SIMT (Single Instruction, Multiple Thread). Programs describe the behavior of a single scalar thread. | Tile-based. Programs describe the behavior of a tile block operating on multi-dimensional tile fragments. |
| **Unit of computation** | Individual CUDA thread executing scalar or vector instructions. | Tile block executing tile-level operations (load tile, compute on tile, store tile). |
| **Thread abstraction** | Explicit CUDA threads organized into warps (32 threads) and thread blocks. Programmer manages thread IDs, lane masks, and warp-level primitives. | Tile blocks. The concept of individual CUDA threads is abstracted away; the programmer works with tile blocks and tile indices. |
| **Memory model** | Per-thread private memory (registers, local memory), per-block shared memory, global memory. Programmer manually manages data movement and layout. | Tile memory with automatic placement. The compiler determines whether a tile resides in registers, shared memory, or global memory based on usage patterns and hints. |
| **Tensor core access** | Explicit warp-level MMA instructions (`mma.sync`, `wmma.load`, `wmma.store`). Programmer must manage register fragment layouts, swizzling, and data type conversions. | Abstracted tile-level operations (`tile.load`, `tile.mma`, `tile.store`). The compiler maps these to appropriate tensor core instructions and manages fragment layouts. |
| **Data types** | Scalar types (`f32`, `f64`, `s32`, `u32`, `b16`, etc.) and limited vector types (`.v2`, `.v4`). | Tile types parameterized by element type and tile dimensions (`tile<16x16xf32>`, `tile<8x16xbf16>`). |
| **Portability mechanism** | PTX ISA versioning. PTX programs are JIT-compiled by the CUDA driver for the target GPU architecture. | Tile IR bytecode versioning. Tile IR bytecode is compiled by the CUDA driver or standalone toolkit for the target architecture. |
| **Compilation model** | JIT compilation (driver) and AOT compilation (`ptxas`). | JIT compilation (driver-integrated compiler) and AOT compilation (standalone toolkit compiler). |
| **Synchronization** | Explicit barrier instructions (`bar.sync`, `bar.arrive`), fence instructions (`membar.cta`, `membar.gpu`), and warp-level vote/shuffle primitives. | Tile-level synchronization primitives (`tile.barrier`, `tile.fence`) that operate at tile block granularity. |
| **Control flow** | Per-thread divergence with reconvergence (`@p` predication, `bra`, `call`). | Tile block-level control flow with limited divergence support. |
| **Optimization hints** | Limited (`.maxnreg`, `.minnctapersm`, pragma directives). | Rich set of hints for tile placement, scheduling, and resource allocation (`tile.hint`, `tile.layout`). |
| **Interoperability** | Native CUDA C++ interop; PTX can be embedded in CUDA C++ via inline assembly. | Seamless interop with CUDA C++ and PTX; Tile IR kernels can coexist in the same program. |
| **Target workloads** | General-purpose GPU computing, graphics, ray tracing, scalar/vector algorithms. | Tile-based data-parallel workloads: matrix multiplication, convolution, attention, reduction, scan, and other linear algebra operations. |

### 1.5.2 Conceptual Mapping

The following table maps key PTX concepts to their Tile IR equivalents:

| PTX Concept | Tile IR Equivalent | Notes |
|-------------|-------------------|-------|
| Thread | Tile block | A tile block replaces the concept of a cooperative group of threads. |
| Warp | (Internal) | Warps are an implementation detail managed by the Tile IR compiler. |
| Thread block | Tile block group | A group of tile blocks that share a memory hierarchy level. |
| Register | Tile register | Storage for a tile fragment; the compiler manages register allocation. |
| Shared memory | Tile memory | The compiler manages shared memory allocation for inter-tile-block communication. |
| Global memory | Tile global | Direct access to global memory through tile load/store operations. |
| `mma.sync` | `tile.mma` | Abstracted matrix multiply-accumulate on tile fragments. |
| `ld.global` / `st.global` | `tile.load` / `tile.store` | Load/store operations on tile objects. |
| `bar.sync` | `tile.barrier` | Synchronization between tile blocks in the same group. |
| `shfl.sync` | `tile.shuffle` | Cross-tile-block data exchange. |
| `atom` | `tile.atomic` | Atomic operations on tile-level data. |

### 1.5.3 When to Use Which

| Scenario | Recommended Technology |
|----------|----------------------|
| General-purpose GPU kernels (scans, sorts, graph algorithms) | CUDA C++ / PTX |
| Tile-based linear algebra (matmul, conv, attention) | Tile IR |
| Mixed workload with both tile and scalar phases | Tile IR + CUDA C++ interop |
| Maximum control over register allocation and warp scheduling | PTX |
| Portable, future-proof tile-based kernels | Tile IR |
| Research on tile-based programming models | Tile IR (as compilation target) |

---

## 1.6 Target Audience

Tile IR is designed for several distinct audiences, each with different needs and levels
of interaction with the instruction set.

### 1.6.1 DSL and Compiler Authors Targeting NVIDIA Hardware

This is the **primary audience** for Tile IR. Developers building domain-specific
languages (e.g., for deep learning, scientific computing, or signal processing) or
compiler backends that target NVIDIA GPUs can use Tile IR as a portable compilation
target.

**What they need from Tile IR:**
- A well-specified, stable instruction set with clear semantics.
- A versioned bytecode format that guarantees forward compatibility.
- Sufficient expressiveness to represent common tile-based computation patterns.
- Interoperability with existing CUDA ecosystem tooling (profilers, debuggers, etc.).

**How they interact with Tile IR:**
- Emit Tile IR bytecode or textual IR from their compiler pipeline.
- Use the Tile IR MLIR dialect as part of an MLIR-based compilation flow.
- Link Tile IR bytecode with CUDA C++ / PTX code.

### 1.6.2 Performance Engineers Writing Tile-Based Kernels

Performance engineers who write highly optimized tile-based kernels (e.g., custom
attention kernels, fused linear algebra operations, or specialized reduction patterns)
can use Tile IR to express their algorithms at the tile level without managing low-level
hardware details.

**What they need from Tile IR:**
- Direct control over tile sizes, data layouts, and computation ordering.
- Optimization hints for guiding the compiler's resource allocation decisions.
- Performance predictability and minimal abstraction overhead.
- Escape hatches for cases where the abstraction is too restrictive.

**How they interact with Tile IR:**
- Write Tile IR programs using a textual representation or a framework-level API.
- Profile and tune using CUDA profiling tools.
- Apply optimization hints and measure their impact.

### 1.6.3 Framework Developers Building on the CUDA Platform

Developers building high-performance frameworks (e.g., deep learning frameworks,
numerical libraries, or graph processing engines) that run on NVIDIA GPUs can use Tile IR
as a backend for tile-based operations within their framework.

**What they need from Tile IR:**
- Reliable performance across GPU generations.
- Seamless integration with existing CUDA C++ codebases.
- Support for the data types and operations commonly used in their domain.
- Toolchain support (debuggers, profilers, error reporting).

**How they interact with Tile IR:**
- Use Tile IR through a framework-level API or code generation pipeline.
- Manage kernel launch, memory allocation, and data movement at the framework level.
- Compose Tile IR kernels with existing CUDA library calls.

### 1.6.4 Researchers Investigating Tile-Based Programming Models

Academic and industrial researchers studying tile-based programming models, compiler
optimizations, or hardware-software co-design can use Tile IR as a stable platform for
experimentation.

**What they need from Tile IR:**
- A clear, formally specified abstract machine model.
- Flexibility to define custom tile operations and scheduling strategies.
- Reproducibility across experiments and hardware generations.
- The ability to extend the instruction set for research purposes.

**How they interact with Tile IR:**
- Build experimental frontends that emit Tile IR.
- Modify the MLIR dialect to prototype new operations or optimizations.
- Use Tile IR as a baseline for comparison with alternative programming models.

---

## 1.7 Relationship to Other Technologies

Tile IR exists within a rich ecosystem of GPU programming technologies. Understanding
its relationship to these technologies clarifies its role and helps developers choose
the right tool for their use case.

### 1.7.1 CUDA C++

**Relationship:** Tile IR interoperates with CUDA C++ programs at the binary level. Tile
IR kernels can be launched from CUDA C++ host code using the standard CUDA kernel launch
API. Within a single kernel, Tile IR tile blocks can coexist with CUDA C++ thread-level
code, sharing access to global memory, constant memory, and texture memory.

**Key points:**
- Tile IR kernels are launched using `<<<grid, block>>>` syntax from CUDA C++ host code.
- Tile IR and CUDA C++ code can share pointers to global memory allocations.
- The Tile IR compiler generates code that is compatible with the CUDA runtime and driver
  APIs.
- Existing CUDA libraries (cuBLAS, cuDNN, cuFFT, etc.) can be called alongside Tile IR
  kernels.

**Example interop pattern:**
```
// CUDA C++ host code
void launch_tile_kernel(float* d_A, float* d_B, float* d_C, int M, int N, int K) {
    dim3 grid(M / TILE_M, N / TILE_N);
    dim3 block(WARP_SIZE * NUM_WARPS);

    // Launch a Tile IR kernel
    tile_ir_matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
}
```

### 1.7.2 PTX

**Relationship:** Tile IR is a **complementary ISA**, not a replacement for PTX. Both
instruction sets target NVIDIA GPUs but serve different purposes and programming models.

**Key points:**
- PTX excels at expressing SIMT-style workloads where per-thread control flow and scalar
  operations dominate.
- Tile IR excels at expressing tile-based workloads where operations on multi-dimensional
  fragments are the primary concern.
- In some implementations, the Tile IR compiler may internally generate PTX as an
  intermediate step before final compilation to SASS.
- The two ISAs can coexist in the same program; a kernel can contain both PTX and Tile IR
  components.

**When to choose PTX over Tile IR:**
- The workload is inherently scalar or requires complex per-thread control flow.
- Maximum control over warp-level primitives (shuffle, vote, match) is needed.
- The workload does not benefit from tile-level abstractions (e.g., graph algorithms,
  irregular data structures).

**When to choose Tile IR over PTX:**
- The workload naturally operates on matrix or tensor fragments.
- Portability across tensor core generations is important.
- The programmer wants to avoid managing register layouts and fragment swizzling.

### 1.7.3 Triton

**Relationship:** Triton is a higher-level tile programming language and compiler that
provides a Python-based DSL for writing tile-based GPU kernels. Tile IR can serve as a
backend for Triton, providing a lower-level compilation target.

**Key points:**
- Triton operates at a higher level of abstraction than Tile IR, providing Python syntax,
  automatic shared memory management, and higher-level control flow constructs.
- Tile IR provides a lower-level, more explicit abstraction that is suitable as a
  compilation target for Triton and similar languages.
- A potential compilation path is: Triton source -> Triton IR -> Tile IR bytecode ->
  GPU machine code.
- Researchers and DSL authors who find Triton too high-level but PTX too low-level may
  find Tile IR to be the right abstraction level.

**Comparison:**

| Aspect | Triton | Tile IR |
|--------|--------|---------|
| Language level | High-level DSL (Python-based) | Low-level instruction set |
| Primary users | Kernel developers, ML engineers | Compiler authors, framework developers |
| Abstraction | Automatic memory management, implicit tiling | Explicit tile operations, user-controlled placement |
| Compilation | Triton compiler -> LLVM IR -> PTX | Tile IR compiler -> SASS (or via PTX) |
| Control flow | Python-like loops and conditionals | Tile block-level control flow |
| Optimization | Compiler-driven | Compiler-driven with user hints |

### 1.7.4 CUTLASS

**Relationship:** CUTLASS is a C++ template library for writing high-performance tile-
based GPU kernels. It provides abstractions similar to Tile IR but at the C++ library
level rather than the ISA level.

**Key points:**
- CUTLASS provides tile-based abstractions through C++ templates, requiring explicit
  management of data layouts, pipeline stages, and tensor core instructions.
- Tile IR provides similar abstractions at the instruction set level, enabling
  cross-language, cross-compiler portability.
- CUTLASS kernels are compiled through the standard CUDA C++ toolchain (NVCC -> PTX ->
  SASS), while Tile IR kernels use the Tile IR compiler (Tile IR bytecode -> SASS).
- CUTLASS is well-suited for developers who prefer a C++ library interface; Tile IR is
  better suited for compiler authors who need a lower-level, language-agnostic target.

**Comparison:**

| Aspect | CUTLASS | Tile IR |
|--------|---------|---------|
| Form | C++ template library | Instruction set + bytecode |
| Portability | C++ source-level | Binary-level (bytecode) |
| Compilation | NVCC -> PTX -> SASS | Tile IR compiler -> SASS |
| Hardware adaptation | Template specialization per architecture | Compiler handles automatically |
| Optimization control | Template parameters | Optimization hints in IR |
| Interoperability | Native C++ interop | Binary-level interop with CUDA C++ |

### 1.7.5 MLIR

**Relationship:** Tile IR provides an MLIR dialect (`tile-ir`) for integration with
MLIR-based compiler pipelines.

**Key points:**
- The `tile-ir` MLIR dialect provides operations, types, and attributes that correspond
  directly to the Tile IR specification.
- Compiler pipelines can lower high-level dialects (e.g., `linalg`, `tosa`) to the
  `tile-ir` dialect and then emit Tile IR bytecode.
- The dialect supports progressive lowering: operations can be refined from abstract
  tile operations to more concrete, hardware-specific forms.
- MLIR infrastructure (pattern rewriting, analysis passes, verification) is available
  for Tile IR programs.

**Example MLIR dialect usage:**
```
// Conceptual MLIR using the tile-ir dialect
%0 = tile_ir.load %A[%i, %j] : !tile_ir.tile<16x16xf32> from memref<1024x1024xf32>
%1 = tile_ir.load %B[%k, %j] : !tile_ir.tile<16x16xf32> from memref<1024x1024xf32>
%2 = tile_ir.mma %0, %1 : !tile_ir.tile<16x16xf32>
%3 = tile_ir.load %C[%i, %k] : !tile_ir.tile<16x16xf32> from memref<1024x1024xf32>
%4 = tile_ir.add %2, %3 : !tile_ir.tile<16x16xf32>
tile_ir.store %4, %D[%i, %k] : !tile_ir.tile<16x16xf32> to memref<1024x1024xf32>
```

### 1.7.6 Technology Stack Summary

The following diagram illustrates how Tile IR fits within the broader GPU programming
technology stack:

```
+-----------------------------------------------------------+
|                  Application / Framework                    |
+-----------------------------------------------------------+
|   CUDA C++   |   Triton    |   CUTLASS   |   Custom DSL    |
+-----------------------------------------------------------+
|     PTX       |            Tile IR           |   MLIR       |
+-----------------------------------------------------------+
|                  CUDA Driver / Runtime                      |
+-----------------------------------------------------------+
|                   GPU Hardware (SASS)                       |
+-----------------------------------------------------------+
```

*Figure 1.2: Tile IR sits between high-level programming models and the CUDA driver,
providing a tile-native abstraction layer that complements PTX.*

---

## 1.8 Summary

Tile IR represents a new approach to programming NVIDIA GPUs for tile-based workloads.
By modeling the GPU as a tile-based processor rather than a SIMT processor, Tile IR
aligns the programming abstraction with both the programmer's mental model and the
underlying hardware execution model. This alignment delivers:

- **Improved productivity** for developers of tile-based kernels, who can express their
  algorithms directly in terms of tiles rather than individual threads.
- **Improved portability** across GPU generations, as the Tile IR compiler handles the
  mapping of tile operations to generation-specific tensor core instructions.
- **Improved performance** for well-structured tile-based workloads, as the Tile IR
  compiler can apply whole-program optimizations that are difficult to express at the
  CUDA C++ or PTX level.

The following chapters provide a detailed specification of the Tile IR programming
model, type system, instruction set, and binary format. Readers are encouraged to start
with Chapter 2 (Programming Model) for a conceptual understanding before diving into the
formal specification in later chapters.
