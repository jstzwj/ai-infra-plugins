# Operations: Core

This section describes the core operations in Tile IR -- the fundamental building blocks for constructing tile programs.

## Meta Types

Operations have arguments which are Tile IR values with Tile IR types but many operations have immediate or static arguments which correspond to attributes. These meta types are not representable in the Tile IR type system but are used to construct Tile IR programs and only present at compile time.

| Meta Type | Description |
|-----------|-------------|
| Symbol | A symbol in the program, begins with `@`, uniquely identifies a symbol |
| Flag | A boolean value that controls operation behavior |
| Token | Represents a memory ordering token |
| Variadic | A statically sized but variable number of arguments |
| Any | A value of any valid Tile IR type |
| Name | A name in the program, begins with `#` |
| Type | A Tile IR type attached as an attribute |
| Array | A statically sized array of values |
| String | A string value passed as an attribute |
| bool | A boolean value passed as an attribute |
| DenseConstant | A dense constant value passed as an attribute |

## Design Considerations

### Explicit Broadcast

There are no implicit broadcasts performed by operations in the Tile IR dialect. All operations that require operands of the same shape must be explicitly broadcasted using `cuda_tile.reshape` or `cuda_tile.broadcast` operations.

### Distinct Floating-Point and Integer Operations

Numeric operations are split across integer and floating-point types due to differences in flags such as rounding modes, NaN handling, and fast math. For example, `cuda_tile.addf` supports a rounding attribute, but `addi` does not.

### Explicit Overflow Annotations

Some operations such as `cuda_tile.addi` support an explicit overflow annotation that expresses the expected overflow behavior. These attributes serve as assumptions that an implementation may use to reason about the operation. It is the responsibility of the code generator to ensure that the operation respects these assumptions dynamically during execution.

---

## `cuda_tile.broadcast`

Broadcast tile to new shape.

```
cuda_tile.broadcast %source
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| source | `tile` | The tile to broadcast |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile` | The broadcasted tile |

**Description:**

The broadcast operation expands each unary (1) dimension in the input tile by duplicating the data along that dimension. Expansion happens only for dimensions of size one that are stretched or "copied" to match the size of the dimension implied by the result type. The operation does not change the rank of the source tile. Any change to the rank must be made using reshape-like operations before broadcasting.

**Constraints:**

- source and result must have the same element type
- source and result must have the same rank

---

## `cuda_tile.cat`

Concatenate tiles along specified dimension.

```
cuda_tile.cat %lhs %rhs %dim
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| lhs | `tile` | The left hand side operand |
| rhs | `tile` | The right hand side operand |
| dim | `i64` | The dimension along which to concatenate |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile` | The concatenated result tile |

**Description:**

The cat operation concatenates the two input tiles. The input tiles must have the same shape in all but the concatenating dimension. Concatenation happens along the dimension specified by the attribute dim; the resulting dimension is the sum of the two input tiles' concatenating dimension.

**Constraints:**

- lhs, rhs and result must have the same rank
- lhs, rhs and result must have the same element type

**Examples:**

```cuda_tile
// A valid invocation of cat.
%0 = cat %arg0, %arg1 dim = 1
  : tile<2x4xf32>, tile<2x4xf32> -> tile<2x8xf32>

// >>> %arg0 = tile([[ A, B, C ],
//                   [ D, E, F ]])
// >>> %arg1 = tile([[ 1, 2, 3 ],
//                   [ 4, 5, 6 ]])
// >>> %0 = tile([[ A, B, C, 1, 2, 3 ],
//                [ D, E, F, 4, 5, 6 ]])

// A valid invocation of cat.
%1 = cat %arg0, %arg1 dim = 0
  : tile<2x4xf32>, tile<2x4xf32> -> tile<4x4xf32>
```

---

## `cuda_tile.constant`

Construct a constant tile.

```
cuda_tile.constant %value
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| value | `DenseConstant` | The constant value to create |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile<i1 \| i8 \| i16 \| i32 \| i64 \| f16 \| bf16 \| f32 \| f64 \| fp8e4m3fn \| fp8e5m2 \| tf32>` | The constant tile |

**Description:**

The constant operation creates a tile initialized by value. There are two main forms:

1. **Scalar fill:** `<D: c>` -- tile is filled with identical values for all elements with element type D.
2. **Dense fill:** `dense<D: [c0, c1, c2, ...]>` -- the constant value's shape must match the tile's shape with element type D.

**Examples:**

```cuda_tile
%c0 = constant <i32: 0> : tile<i32>
%c1 = constant <i64: 1> : tile<i64>
%c2 = constant <i32: [0, 1, 2, 3]> : tile<4xi32>
%c3 = constant <f32: 0.0> : tile<2x4xf32>
%c4 = constant <f64: [0.0, 1.0, 2.0, 3.0]> : tile<4xf64>
```

---

## `cuda_tile.entry`

Define a tile kernel.

```
cuda_tile.entry %sym_name %function_type %arg_attrs %res_attrs %optimization_hints
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| sym_name | `Symbol` | The name of the function |
| function_type | `Type` | The type of the function |
| arg_attrs | `Attributes` | Argument attributes (currently unsupported) |
| res_attrs | `Attributes` | Result attributes (currently unsupported) |
| optimization_hints | `OptimizationHints` | Architecture-specific compiler hints |

**Description:**

The entry operation defines a tile kernel; a kernel is a function that can serve as the program entry point. It has a unique name per-module. A kernel cannot return any value. It must be launched from the host side using `cuLaunchKernel` or similar CUDA runtime API functions.

The optimization_hints attribute provides architecture-specific compiler hints:

- `num_cta_in_cga` - suggest the number of CTAs in a CGA
- `allow_tma` - suggest whether to use TMA for view loads/stores
- `latency` - latency hint for view loads/stores

**Constraints:**

- Must be a symbol in the global symbol table
- Must implement callable target interface
- Each region must contain exactly one block

---

## `cuda_tile.extract`

Extract a subtile from a tile.

```
cuda_tile.extract %source %indices
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| source | `tile` | The source tile to extract from |
| indices | `Variadic<tile<i32>>` | The indices of the slice to extract |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile` | The extracted subtile |

**Description:**

The extract operation extracts a subtile from the given source tile. The shape of the result tile must divide the shape of the source tile evenly. The indices indicate the number of the slice to extract (not the offsets). Slices with the same shape are non-overlapping for unique indices. The indices operands are interpreted as unsigned integers.

> **Warning:** If the indices specify a non-existent (out-of-bounds) slice, the behavior is undefined.

**Constraints:**

- source and result must have the same rank

**Examples:**

```cuda_tile
%c1 = constant <i32: 1> : tile<i32>
%c2 = constant <i32: 2> : tile<i32>
%t = constant <f32: 0.0> : tile<32x8xf32>
// Valid indices are: [ {0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3} ]
%0 = extract %t[%c1, %c2]
    : tile<32x8xf32> -> tile<4x2xf32>
```

---

## `cuda_tile.get_global`

Get a pointer to a global variable.

```
cuda_tile.get_global %name
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| name | `Symbol` | The name of the global variable |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile<ptr>` | A pointer to the global variable |

**Description:**

Returns a pointer to the specified global variable. The element type of the returned pointer will be of the same type as the element type of the declared global variable.

**Examples:**

```cuda_tile
global @val <f32: [0.1, 0.2, 0.3, 0.4]> : tile<4xf32>

entry @example() {
  %ptr = get_global @val : tile<ptr<f32>>
  return
}
```

---

## `cuda_tile.get_num_tile_blocks`

Get total number of tile blocks.

```
cuda_tile.get_num_tile_blocks
```

**Parameters:** None.

**Results:**

| Name | Type | Description |
|------|------|-------------|
| gridSize_x | `tile<i32>` | Number of tile blocks in dimension x |
| gridSize_y | `tile<i32>` | Number of tile blocks in dimension y |
| gridSize_z | `tile<i32>` | Number of tile blocks in dimension z |

**Description:**

Queries the total number of tile blocks in the form of a 3-tuple specifying the extent of each grid dimension. When launching 1- or 2-dimensional grids, the unspecified dimensions will have a cardinality of 1.

> **Note:** Grid dimensions are limited to 2^24-1 (16,777,215) per axis.

**Examples:**

```cuda_tile
entry @example() {
  %x, %y, %z = get_num_tile_blocks : tile<i32>
}
```

---

## `cuda_tile.get_tile_block_id`

Get the currently executing tile block coordinates.

```
cuda_tile.get_tile_block_id
```

**Parameters:** None.

**Results:**

| Name | Type | Description |
|------|------|-------------|
| blockId_x | `tile<i32>` | The tile block ID for dimension x |
| blockId_y | `tile<i32>` | The tile block ID for dimension y |
| blockId_z | `tile<i32>` | The tile block ID for dimension z |

**Description:**

Returns a 3-d tile block coordinates of the currently executing tile block. The value of each dimension is between 0 (inclusive) and the value returned by `get_num_tile_blocks` for the respective axis (exclusive). Grid dimensions unspecified at launch will always be 0.

> **Note:** Grid dimensions are limited to 2^24-1 (16,777,215) per axis.

---

## `cuda_tile.global`

Allocate static global memory.

```
cuda_tile.global %sym_name %value %alignment
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| sym_name | `Symbol` | The name of the global variable |
| value | `DenseConstant` | The value to initialize the allocation with |
| alignment | `i64` | The alignment of the buffer |

**Results:** None.

**Description:**

Statically allocates a mutable 1-dimensional location in global memory and initializes it using value. The initialization is performed at CUDA module load time. The lifetime of the allocation is the same as the lifetime of the module. The allocation may be read or written to by first using `cuda_tile.get_global` to obtain a pointer.

Global operations must be directly nested within the Tile IR module. They cannot be defined inside functions.

**Examples:**

```cuda_tile
global @val alignment = 128 <f32: [0.1, 0.2, 0.3, 0.4]> : tile<4xf32>
entry @example() {}
```

---

## `cuda_tile.iota`

Generate a 1-d tile range from 0 to n-1.

```
cuda_tile.iota
```

**Parameters:** None.

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile<i1 \| i8 \| i16 \| i32 \| i64>` | The result of the iota operation |

**Description:**

Generates a 1-d tile with a sequence of integer values. The starting value is 0 and the stride is 1. If the shape of the result tile is (n), then the generated values are [0, n - 1]. The result values should be interpreted as unsigned integers.

> **Note:** The number of elements must not exceed the maximum value expressible by the element type.

---

## `cuda_tile.module`

Top-level module containing a series of defined items.

```
cuda_tile.module %sym_name
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| sym_name | `Symbol` | The name of the module |

**Results:** None.

**Description:**

Represents a single compilation unit and contains zero or more items (global variables, functions, or kernels). The module operation is the top-level operation in a Tile IR module.

**Constraints:**

- Must contain only Tile IR operations
- All regions must have zero arguments
- Each region must contain exactly one block
- Must define a symbol scope

---

## `cuda_tile.offset`

Offsets a tile of pointers.

```
cuda_tile.offset %ptr %offset
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| ptr | `ptr` | The base pointer tile to advance |
| offset | `tile<i1 \| i8 \| i16 \| i32 \| i64>` | The offset tile to add to the pointer |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `ptr` | The resulting pointer tile after advancement |

**Description:**

Advances a tile of pointers. It takes ptr as base and offset as increment, and performs element-wise addition:

```
result[i,j] = ptr[i,j] + offset[i,j] * bitwidth
```

ptr is interpreted as an unsigned integer. offset is interpreted as a signed integer. bitwidth is the storage bitwidth of the pointee type. In case of overflow, the result is undefined.

**Constraints:**

- ptr, offset and result must have the same shape
- result and ptr must have the same element type (ptr)

---

## `cuda_tile.permute`

Permute tile dimensions.

```
cuda_tile.permute %source %permutation
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| source | `tile` | The input tile |
| permutation | `Array<i32>` | The permutation of the dimensions |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile` | The permuted tile |

**Description:**

Permutes the dimensions of the input tile according to the permutation array. For example, if the input tile has shape [2, 4, 8], and the permutation is [2, 0, 1], the output tile will have shape [8, 2, 4].

**Constraints:**

- source and result must have the same element type
- source and result must have the same rank

**Examples:**

```cuda_tile
%arg0 = constant <f16: 0.0> : tile<2x4x8xf16>
%0 = permute %arg0 [2, 0, 1] : tile<2x4x8xf16> -> tile<8x2x4xf16>
```

---

## `cuda_tile.reduce`

Variadic tile reduction across dimensions.

```
cuda_tile.reduce %operands %dim %identities
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| operands | `Variadic<tile>` | The set of tiles to reduce |
| dim | `i32` | The dimension to perform reduction on |
| identities | `Array` | The reduction identities for each operand |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| results | `Variadic<tile>` | The set of reduced tiles |

**Description:**

Applies a custom reduction function along a specified dimension of one or more input tiles, producing the same number of output tiles. The reduction function must be an associative operation defined within the reduce operation's region.

All input tiles must have the same shape. The output tiles will have a matching shape in every dimension except the one being reduced, which is removed. Only pure operations are allowed in the body of reduce.

> **Note:** There are no guarantees on the order of element reduction. However, the result is deterministic across runs.

**Examples:**

```cuda_tile
%input = constant <f32: 0.0> : tile<8xf32>
%0 = reduce %input dim=0 identities=[0.000000e+0 : f32] : tile<8xf32> -> tile<f32>
  (%input_arg: tile<f32>, %input_accum: tile<f32>) {
    %add_result = addf %input_arg, %input_accum : tile<f32>
    yield %add_result : tile<f32>
  }
```

---

## `cuda_tile.reshape`

Reshape tile dimensions.

```
cuda_tile.reshape %source
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| source | `tile` | The source tile to reshape |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile` | The reshaped tile |

**Description:**

Changes the shape of the source operand. Reshape is only a change in the indexing of the tile. The number of elements and element type must remain unchanged. 0-d tiles (scalars) contain precisely one element and thus are the one exception where a 0-d tile can be reshaped to a shape where the size(shape) == 1.

Conceptually reshaping a tile is equivalent to first creating a 1-d tile from the data of the source assuming a row-major layout and then converting the 1-d tile into the new shape in a row-major layout.

**Constraints:**

- source and result must have the same element type

**Examples:**

```cuda_tile
%cst = constant <i32: [[0, 1, 2, 3], [4, 5, 6, 7]]> : tile<2x4xi32>
%r0 = reshape %cst : tile<2x4xi32> -> tile<2x2x2xi32>
```

---

## `cuda_tile.scan`

A parallel prefix sum operation.

```
cuda_tile.scan %operands %dim %reverse %identities
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| operands | `Variadic<tile>` | The set of tiles to scan |
| dim | `i32` | The dimension along which to scan |
| reverse | `bool` | Whether to scan in reverse order |
| identities | `Array` | The identities of the scan operation |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| results | `Variadic<tile>` | The resulting tiles from the scan |

**Description:**

Computes an inclusive parallel prefix along a given dimension using a binary associative function and an identity. The scan preserves all intermediate accumulator values. Only pure operations are allowed in the body of scan.

> **Warning:** The scan operation is restricted to only support single tile input.

**Examples:**

```cuda_tile
%input = constant <f32: 0.0> : tile<8x16xf32>
%result = scan %input dim=1 reverse=false identities=[1.0 : f32] : tile<8x16xf32> -> tile<8x16xf32>
(%acc: tile<f32>, %elem: tile<f32>) {
  %prod = mulf %acc, %elem rounding<nearest_even>: tile<f32>
  yield %prod : tile<f32>
}
```

---

## `cuda_tile.select`

Select values based on condition.

```
cuda_tile.select %cond %val_if_true %val_if_false
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| cond | `tile<i1>` | The condition tile |
| val_if_true | `tile` | The value if true tile |
| val_if_false | `tile` | The value if false tile |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile` | The tile of selected values |

**Description:**

Chooses values based on the binary conditions supplied as the cond operand. The val_if_true operand contains the value(s) to use if the condition is 1. The val_if_false operand contains the value(s) to use if the condition is 0. The choice is made element-wise.

All tiles must have the same shape. The tiles val_if_true, val_if_false, and the result must have the same element type. The cond tile must be a tile of i1 values.

**Constraints:**

- val_if_true, val_if_false and result must have the same shape and element type
