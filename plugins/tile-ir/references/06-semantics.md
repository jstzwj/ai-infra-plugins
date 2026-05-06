# 6. Semantics

This section provides a written English presentation of the operational semantics of Tile IR. These semantics are intended to provide an understanding of Tile IR for: 1) those interested in generating Tile IR as a code generation target, or 2) those interested in reading Tile IR programs produced by others.

It does not attempt to formalize every possible behavior that might be admitted by an axiomatic formulation or a small step operational semantics. For understanding an even more informal presentation of the language and its core concepts see the Programming Model section.

We first introduce the abstract machine state and language definitions before describing semantics of individual kernels and programs.

We then discuss the semantics of broad classes of operations as well, for more detailed descriptions of individual operations and their behavior see Operations for a complete listing.

## 6.1. The Abstract Machine

The Tile IR abstract machine state `S` is a tuple consisting of the following components, each explained below:

```
S = (M, B, R, G, P)
```

Where:

- `M` -- A well-formed module which stores one or more items, discussed below in Modules.
- `B` -- A grid of tile blocks (or logical tile threads) each representing a single tile-kernel instance.
- `R` -- A per-tile-block infinite register file, that maps named "registers" to values.
- `G` -- A global memory store, that maps addresses to scalar values.
- `P` -- A set of pending memory accesses, that make progress asynchronously to the execution of Tile IR operations.

These five components fully characterize the observable state of a Tile IR program at any point during execution. The abstract machine provides a formal framework for reasoning about program behavior without committing to any particular hardware implementation or compiler optimization strategy.

The separation between register file state (`R`) and global memory state (`G`) is fundamental to Tile IR's memory model: tiles and other values reside in registers, and only explicit memory operations transfer data between registers and global memory. The pending memory operations set (`P`) captures the asynchronous nature of memory accesses -- operations may be issued by the program but not yet completed at the hardware level.

## 6.2. Modules

A program in Tile IR is represented as a module. A module is a single translation unit which contains zero or more items. An item may either be a:

- a global variable definition
- a tile kernel
- a tile function

For those familiar with CUDA, a tile kernel is the global entry point for a tile program, much like a kernel is in CUDA C++ or PTX. Tile functions represent device side functions that can be called from the tile kernel currently with some restrictions.

A module serves as the top-level container for all Tile IR program elements. The module is the unit of compilation and defines the scope within which names must be unique. All items within a module are visible to each other without explicit imports or declarations.

### 6.2.1. Global Variable

A global is a named global variable that is stored in global device memory and accessible to all tile blocks. Global variables are declared using the `cuda_tile.global` operation. A global variable must be initialized upon declaration and will be initialized exactly once.

A global variable must contain a value of Tile Type.

A global variable can be modified by using the `cuda_tile.get_global` operation to obtain a pointer which can be used to read and write to the global variable.

Global variables provide a mechanism for sharing mutable state across all tile blocks executing within a kernel launch. Because global variables reside in global device memory, accesses to them are subject to the same memory ordering constraints as other global memory operations.

Example of declaring and using a global variable:

```
// Declare a global variable initialized to 0
%global = cuda_tile.global <i32: 0> : tile<i32>

// In a tile kernel, obtain a pointer to the global
%ptr = cuda_tile.get_global %global : tile<i32> -> tile<ptr<i32>>

// Use the pointer for load/store operations
```

### 6.2.2. Tile Kernel

```
entry @tile_func(%A0: T0, ..., %AN: TN) {
     %0 = op %P0, %P1, ... %PN -> R0
     ...
     return
}
```

The basic unit of execution in Tile IR is the tile kernel. A tile kernel is a tile function that acts as the entry point of a tile program. A tile kernel represents a function parameterized by a set of grid coordinates. At kernel runtime, each unique grid coordinate is available to each kernel instance (tile block). A tile kernel can query its grid coordinates via `cuda_tile.get_tile_block_id` and the coordinates can be one-, two-, or three-dimensional depending on the grid the kernel is launched with. A kernel may also query the total number of tile blocks along each dimension via `cuda_tile.get_num_tile_blocks`.

A tile kernel is a tile function with additional restrictions:

- can only have parameters with scalar (i.e., 0-rank) tensor types
- requires all input tensors to be provided as scalar pointers (i.e. `tile<ptr<E>>`)
- produces no return value
- the kernel is only executed for its effect on global device memory

A tile kernel is otherwise a tile function and all properties of tile functions also apply to tile kernels.

The restriction to scalar pointer parameters reflects the kernel launch model: the host passes base pointers to device allocations, and each tile block uses its grid coordinates to determine which portion of the data to process. This is analogous to how CUDA kernels receive pointers to device memory.

### 6.2.3. Tile Function

A tile function consists of a name, a list of formal parameters, a return type, and a body. A tile function's body contains a single threaded tile program (referred to as tile block) parameterized by formal parameters.

A tile function has `N` formal parameters and produces `M` return values. The type of the parameters can be one of the valid types described in Type System.

> **Note:** Currently defining non-kernel tile functions is disabled with support planned for a future release.

### 6.2.4. Function Bodies

A function body consists of a sequence of statements that are in static-single-assignment (SSA) form.

Each statement assigns the result of a single operation to a set of unique result variables. All operations in Tile IR are represented uniformly in this way, including control flow and memory operations.

The SSA form requirement means that each variable is defined exactly once, and every use of a variable must be dominated by its definition. This property simplifies dataflow analysis and enables many compiler optimizations. For example:

```
entry @example(%A: tile<ptr<f32>>, %B: tile<ptr<f32>>, %N: tile<i32>) {
    // Each variable (%0, %1, %2, ...) is assigned exactly once
    %0 = cuda_tile.get_tile_block_id x : tile<i32>
    %1 = cuda_tile.constant <i32: 128> : tile<i32>
    %2 = cuda_tile.mul %0, %1 : tile<i32>, tile<i32> -> tile<i32>
    // ... operations continue ...
    cuda_tile.return
}
```

### 6.2.5. Well-Formedness

A well-formed module is a module that satisfies the following properties:

1. The module contains at least one item.
2. Each item is uniquely named within the module.
3. Each Tile Kernel and Tile Function has a body that is a sequence of statements in valid static-single-assignment (SSA) form.
4. The program type checks according to the rules specified in Type System and the operator type signatures are specified in Operations.

Program well-formedness is required as a pre-condition and post-condition of both optimizations and operational semantic rules.

The well-formedness guarantees ensure that the Tile IR program is syntactically and semantically valid before execution or optimization begins. Any transformation applied to the program (by an optimizer, for example) must preserve these properties.

## 6.3. Values

Tile IR has a small set of types as described in Type System but we only have three types of values.

- **Pointers**, which represent a memory address.
- **Tiles**, or an N-dimensional array of scalars.
- **Views**, which represent a structured view of memory.

### 6.3.1. Pointers

A pointer is a 64-bit integer memory address that references a location in global device memory. Pointers are typed as `ptr<E>` where `E` is the type of the memory location it references. Pointers are required to be aligned to the size of the underlying datatype they point to see Element Type Encoding for specific encodings and information about allocation layout.

A pointer is a memory address that points to a location in global memory.

**Data Layout**

Allocations pointed to by input pointer values, and by extension views (see below), must conform to the specified data layout.

We expect that the allocation pointed to by `ptr<E>` is a sized contiguous allocation of scalar values of element type `E`.

There is no padding between elements of the allocation, and we expect that for an allocation of size `N` will be equivalent to `N * sizeof(E)` bytes. The size and encoding of an element is determined by its type `E` and is defined in the table below.

As an aside the datatype encoding is compatible with DLPack a standard adopted by most deep learning frameworks and array libraries. We provide the equivalent PyTorch and NumPy encodings for each datatype.

For NumPy low-precision types we provide the equivalent in terms of the ml_dtypes library a standard collection of low-precision NumPy data types.

> **Warning:** Tile IR layouts are currently restricted to be contiguous for sub-byte types.

**Element Type Encoding**

| Tile IR Type | DLPack Type Code | DLPack Bits | DLPack Lanes | NumPy Type | PyTorch Type |
|---|---|---|---|---|---|
| i1 | kDLInt, kDLUInt | 8 | 1 | numpy.uint8 (unpacked) | N/A |
| i8 | kDLInt, kDLUInt | 8 | 1 | numpy.uint8, numpy.int8 | torch.bool |
| i16 | kDLInt, kDLUInt | 16 | 1 | numpy.int16, numpy.uint16 | torch.int16, torch.uint16 |
| i32 | kDLInt, kDLUInt | 32 | 1 | numpy.int32, numpy.uint32 | torch.int32, torch.uint32 |
| i64 | kDLInt, kDLUInt | 64 | 1 | numpy.int64, numpy.uint64 | torch.int64, torch.uint64 |
| f16 | kDLFloat | 16 | 1 | numpy.float16 | torch.float16 |
| f32 | kDLFloat | 32 | 1 | numpy.float32 | torch.float32 |
| f64 | kDLFloat | 64 | 1 | numpy.float64 | torch.float64 |
| bf16 | kDLBfloat | 16 | 1 | ml_dtypes.bfloat16 | torch.bfloat16 |
| fp8 (E4M3) | kDLFloat8_e4m3 | 8 | 1 | ml_dtypes.float8_e4m3fn | torch.float8_e4m3fn |
| fp8 (E5M2) | kDLFloat8_e5m2 | 8 | 1 | ml_dtypes.float8_e5m2 | torch.float8_e5m2 |

The alignment requirements for each type are as follows:

| Tile IR Type | Size (bits) | Size (bytes) | Required Alignment (bytes) |
|---|---|---|---|
| i1 | 8 | 1 | 1 |
| i8 | 8 | 1 | 1 |
| i16 | 16 | 2 | 2 |
| i32 | 32 | 4 | 4 |
| i64 | 64 | 8 | 8 |
| f16 | 16 | 2 | 2 |
| f32 | 32 | 4 | 4 |
| f64 | 64 | 8 | 8 |
| bf16 | 16 | 2 | 2 |
| fp8 (E4M3) | 8 | 1 | 1 |
| fp8 (E5M2) | 8 | 1 | 1 |

For example, a pointer of type `ptr<f32>` must point to an allocation that is 4-byte aligned. Each element in the allocation occupies exactly 4 bytes, and an allocation of `N` elements occupies `N * 4 = 4N` bytes with no padding between elements.

> **Note:** Allocations are the only values where memory layout is specified in Tile IR.

### 6.3.2. Tiles

A tile is an **immutable** N-dimensional array of scalars characterized by:

- Its **rank** (number of dimensions)
- Its **shape** (extent along each dimension)
- Its **primitive element type**

These properties are part of both the tile's type and the size and shape of the value.

A tile may have any non-negative rank, where:

- **Rank-0** tiles represent scalar values. For example, `tile<f32>` is a scalar f32 value.
- **Rank-1** tiles represent vectors. For example, `tile<128xf32>` is a 1D vector of 128 f32 elements.
- **Rank-2** tiles represent matrices. For example, `tile<64x128xf32>` is a 2D matrix with 64 rows and 128 columns of f32 elements.
- And **Rank-N** tiles represent higher-order N-d arrays. For example, `tile<2x4x8xf32>` is a 3D array.

For a given tile of type `tile<NxKxE>`, where the tile has shape `(N, K)` with elements of type `E`, results in a value of size `N * K`, containing `N * K` individual elements.

A tile value of this type is abstractly represented as a tuple of an array with `N * K` elements of type `E`, and an opaque layout which provides a mapping from the index space of elements to a linear index.

The immutability of tiles is a key design choice: once created, a tile value cannot be modified in place. Instead, operations produce new tile values. This property simplifies reasoning about data flow and enables the compiler to make aggressive optimization decisions about register allocation and memory placement.

Concrete examples of tile values:

```
// A scalar tile (rank-0) containing a single i32 value
%scalar = cuda_tile.constant <i32: 42> : tile<i32>

// A vector tile (rank-1) of 8 f32 elements
%vector = cuda_tile.constant <f32: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]> : tile<8xf32>

// A matrix tile (rank-2) of 2x4 f32 elements
%matrix = cuda_tile.constant <f32: [[1.0, 2.0, 3.0, 4.0],
                                     [5.0, 6.0, 7.0, 8.0]]> : tile<2x4xf32>

// A tile of pointers (rank-1) of 4 elements
%ptrs = cuda_tile.constant <ptr<f32>: [%p0, %p1, %p2, %p3]> : tile<4xptr<f32>>
```

> **Note:** The physical layout and memory representation of a given tile is not visible to the program and is not specified by the language semantics.
>
> The compiler will choose to represent a tile in memory in a way that is most efficient for the target architecture, and specific program and tiles sizes.

### 6.3.3. Views

Tile IR provides a set of view types as described in Tensor View.

Views provide structured views of memory by enriching a pointer with additional data. Views have their own set of memory operations `cuda_tile.load_view_tko` and `cuda_tile.store_view_tko` which vary in behavior based on the view type and make use of the additional metadata. Due to the fact that views are not a singular type, but a family of types each concrete view type has its own value representation.

**Tensor View** is the first-order view type. A tensor view value is logically a tuple of `(ptr<E>, shape)`, and load and store operate on the entire tensor view. The index space of this view is rank-0 (i.e., there is only a single element in the index space).

For example, a tensor view of type `tensor_view<128x64xf32>` consists of a base pointer `ptr<f32>` pointing to a `128 x 64` allocation of f32 values. Loading from this view reads the entire `128 x 64` block of data into a `tile<128x64xf32>`.

**Partition View** is the primary second-order (or "sub-view") view type.

The partition view type Partition View is a subview type which represents a tensor view partitioned into tiles. A partition view value is logically a tuple of `(ptr<E>, shape, tile_shape)`, where load and store operate over tile values of the given size. The index space of this view is:

```
shape / tile_shape
```

For example a tensor view with shape `(M, N)` may be partitioned into a grid of `(TM, TN)` tiles, resulting in a partition view with shape `(M/TM, N/TN)`. See Type System for more details on the different types of views.

Concrete example: Given a partition view over a `512 x 256` tensor with tile shape `(64, 32)`:

- The partition view index space is `(512/64, 256/32) = (8, 8)`
- Loading at index `(i, j)` produces a `tile<64x32xf32>` containing the `(i, j)`-th tile block of the original tensor
- The base pointer is offset by `(i * 64 * 256 + j * 32) * sizeof(f32)` to reach the start of the tile

> **Note:** As with tiles, the physical layout and memory representation of a view is not visible to the program and is not specified by the language semantics.

> **Warning:** Reading or writing out of bounds of any allocation is undefined behavior and bounds checking must be performed by the programmer if desired.

## 6.4. Tile Grid

During execution, the abstract machine instantiates a grid of tile blocks. A tile grid is a grid of tile blocks arranged in a 1-, 2-, or 3-dimensional array.

Each position of the grid corresponds to a single independent tile kernel instance.

The abstract machine stores a sequence of tile blocks, `B`, that are indexed by a grid of coordinates, `G(x, y, z)`. Each tile block is assigned a unique tile block id based on the grid size and the tile block's position within the grid.

The bijective mapping between 1-, 2-, or 3-dimensional grid coordinates and flattened tile block ids is computed as follows:

```
id = x + y * gridDim.x + z * gridDim.x * gridDim.y
```

Where:
- `x` is the coordinate along dimension 0 (range: `[0, gridDim.x)`)
- `y` is the coordinate along dimension 1 (range: `[0, gridDim.y)`)
- `z` is the coordinate along dimension 2 (range: `[0, gridDim.z)`)
- `gridDim.x`, `gridDim.y`, `gridDim.z` are the total number of tile blocks along each dimension

For a 1D grid with `gridDim.x = 4`, the four tile blocks have ids:
- Block at (0): id = 0
- Block at (1): id = 1
- Block at (2): id = 2
- Block at (3): id = 3

For a 2D grid with `gridDim.x = 2, gridDim.y = 3`, the six tile blocks have ids:
- Block at (0,0): id = 0 + 0 * 2 = 0
- Block at (1,0): id = 1 + 0 * 2 = 1
- Block at (0,1): id = 0 + 1 * 2 = 2
- Block at (1,1): id = 1 + 1 * 2 = 3
- Block at (0,2): id = 0 + 2 * 2 = 4
- Block at (1,2): id = 1 + 2 * 2 = 5

This is identical to the CUDA thread block indexing scheme and is designed for compatibility with the CUDA launch model.

## 6.5. Register File

The register file, `R`, maps named registers to values. Each assignment (i.e., SSA variable) in the tile function's body is assigned to a unique register which eventually holds the value of the operation's result. Registers are local to a tile block and are not visible to other tile blocks. As stated previously, the memory representation of values in registers are not visible to the program and is not specified by the language semantics.

Values will only be fetched or persisted to global memory by memory operations (see Memory).

The register file is indexed by the tile block's coordinates, `G(x, y, z)`, and the register's name, `r`, and produces a value `v`.

```
R[G(x, y, z)][r] = v
```

The register file is described as "infinite" in the abstract machine, meaning there is no limit on the number of named registers that can be held simultaneously. In practice, the compiler maps the abstract registers to physical registers or spills them to local memory as needed. The infinite register file abstraction simplifies the semantics by removing concerns about register pressure from the language specification.

Key properties of the register file:

- **Local scope**: Each tile block has its own independent register file. Tile blocks cannot directly access each other's registers.
- **SSA-based**: Registers correspond to SSA variables. Each register is written exactly once (when its defining operation executes) and may be read zero or more times.
- **Opaque representation**: The physical memory representation of values stored in registers (e.g., whether a tile is stored in registers, shared memory, or some other location) is not specified by the language and is left to the compiler.
- **Tile values**: Registers can hold tiles of arbitrary rank and shape, not just scalar values. A single register might hold a `tile<64x128xf32>` value.

## 6.6. Global Memory

The global memory, `G`, is a mapping from addresses to scalar values.

The global memory is used to store the values of the tile block's global variables.

The heap is abstractly modeled as a map from addresses to scalar values, not tile values. This distinction is essential to describe the memory effect of tile operations as a sequence of individual scalar memory operations. A fine-grained model enables both reasoning about aggregate operations granularly as well as a straightforward denotation into the existing PTX memory model.

For example, when a `cuda_tile.store_ptr_tko` operation stores a `tile<4x4xf32>` tile, the memory model describes this as 16 individual f32 stores, one for each element of the tile. This granularity enables precise reasoning about memory ordering, atomicity, and potential data races between tile blocks.

> **Note:** Global memory is the same global device memory that is used by CUDA programs and described in the PTX Specification.

The specification elaborates the intricacies of the Tile IR memory model in Memory Model.

## 6.7. Tile Block

A tile block is a single thread of execution that is assigned a unique coordinate in the tile grid.

Abstractly its state consists of:

- The tile kernel under execution.
- A register file, `R`, that maps named registers to values.
- A statement under evaluation represented by an integer index into the sequence of SSA statements in the tile function's body.

Each tile block executes independently and in isolation from other tile blocks. The only mechanism for inter-block communication is through global memory. This isolation property means that the compiler is free to schedule tile blocks onto hardware threads in any order, or even to overlap their execution, without affecting the semantics of correctly synchronized programs.

A tile block can be thought of as a lightweight CUDA thread block, but where the entire block is a single logical thread of execution operating on tile-granularity data, rather than a collection of scalar CUDA threads.

## 6.8. Execution Semantics

Tile IR program execution starts with a kernel launch. The kernel launch API is uniform for all CUDA kernels and is specified in the CUDA Runtime API.

### 6.8.1. Initialization

A launch of tile kernel initializes the abstract machine with:

- The module representing the complete program.
- A grid of tile blocks where each tile block is instantiated using the same tile kernel, begins at statement 0, assigned a unique grid coordinate, and assigned a unique empty register file.
- A reference to global memory, where its state is the state of global memory prior to the kernel launch.
- The set of pending memory operations is initialized to be empty.

In more detail, the initialization process for a kernel launch with grid dimensions `(gridDim.x, gridDim.y, gridDim.z)` creates `gridDim.x * gridDim.y * gridDim.z` tile blocks. For each tile block at coordinate `(x, y, z)`:

1. The tile block's program counter is set to statement 0 (the first statement in the kernel body).
2. The tile block's register file is initialized as empty.
3. The formal parameters of the kernel are bound to the register file using the arguments passed at launch time.
4. The tile block's unique id is computed as `x + y * gridDim.x + z * gridDim.x * gridDim.y`.

All tile blocks share the same global memory state, which is whatever was present before the kernel was launched.

### 6.8.2. Forward Progress

Execution proceeds with unspecified scheduling of tile blocks. Each tile block will be executed in some order which is non-deterministic and not specified by the language semantics. We guarantee forward progress of the execution that is all tile blocks will be guaranteed to eventually be scheduled for execution. It is possible that all tile blocks run completely in parallel, completely serially, or anything in between.

The forward progress guarantee means that no tile block can be starved indefinitely. Every tile block that has been launched will eventually make progress toward completion. However, the language does not guarantee:

- That tile blocks execute in any particular order.
- That tile blocks make progress at the same rate.
- That tile blocks that start earlier finish earlier.

This is the same forward progress guarantee provided by CUDA for thread blocks.

### 6.8.3. Tile Block Execution

Execution of a single tile block is isolated from other tile blocks. Tile blocks can only observe effects of other blocks via global memory which can be used to implement forms of cooperation or communication.

Function bodies are a series of static-single-assignment (SSA) statements which assign the result of each operation to a unique variable. Each variable is mapped to a register in the abstract machine's register file. A function body executes statements sequentially, in order. The compiler is free to reorder statements as long as there is no effect on the program visible effects or violated program semantics.

For more detailed example programs and explanations of their execution see Programming Model or Appendix.

The one unique semantic of Tile IR is the partitioning of memory operations into **program ordered** and **token ordered** operations. All memory operations produce their result values immediately but the order in which these operations effect memory is more subtle.

**Program Ordered Operations**

For program ordered operations, the order between any pair of memory operations acting on the same address is defined by the operation's position in program. Intuitively the effect of all prior memory operations on the same address will be visible to all subsequent memory operations on the same address.

**Token Ordered Operations**

In contrast, the order between any pair of token ordered operations is undefined, and has no relation to program order. The order of a pair of any two token ordered operations (`A` and `B`) is only defined if established by a direct or transitive relationship between `A`'s output token and `B`'s input token.

This choice importantly allows a producer of Tile IR to induce different memory ordering semantics by inserting the appropriate memory ordering tokens.

**Token Threading**

For example starting with a single fresh token depended upon by the first operation with the result token of the first operation being depended upon by the second operation and so on threading the tokens through each operation in program order.

```
%token0 = cuda_tile.make_token : token
%result1, %token1 = cuda_tile.load_ptr_tko ... token=%token0 : ...
%result2, %token2 = cuda_tile.load_ptr_tko ... token=%token1 : ...
%token3 = cuda_tile.store_ptr_tko ... token=%token2 : ...
```

In this example:
- `%result1`'s load is ordered before `%result2`'s load (via `%token1`).
- `%result2`'s load is ordered before the store (via `%token2`).
- The transitive relationship means `%result1`'s load is also ordered before the store.

Token threading like this establishes an ordering of the memory operations which is consistent with the same program with each token ordered operation being replaced by a program ordered memory operation.

**Joining Tokens**

When multiple independent memory operations need to be ordered before a subsequent operation, the `cuda_tile.join_tokens` operation can merge multiple dependency tokens into one:

```
%token0 = cuda_tile.make_token : token
%result1, %token1 = cuda_tile.load_ptr_tko ... token=%token0 : ...
%result2, %token2 = cuda_tile.load_ptr_tko ... token=%token0 : ...
%joined = cuda_tile.join_tokens %token1, %token2 : token
%token3 = cuda_tile.store_ptr_tko ... token=%joined : ...
```

In this example, both loads must complete before the store begins, but the two loads can proceed concurrently with respect to each other.

For a detailed discussion of the memory model and memory operations see Memory Model.

### 6.8.4. Termination

A tile block will terminate when the tile block's function body reaches the final statement. Tile kernels must terminate with a return operation `cuda_tile.return` which signals the end of the execution.

A kernel launch is considered complete when all tile blocks have terminated. After all tile blocks have terminated, any remaining pending memory operations (`P`) are guaranteed to have completed, and the final state of global memory reflects the combined effects of all tile blocks.

The termination semantics ensure that the host can safely read the results from global memory after the kernel launch returns. There is no need for explicit synchronization between tile blocks and the host for the purpose of memory visibility at kernel termination -- this is guaranteed by the CUDA execution model.
