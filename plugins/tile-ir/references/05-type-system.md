# 5. Type System

This chapter provides a comprehensive specification of the Tile IR type system, including element types, pointer types, tensor types (tiles, tensor views, and partition views), type equivalence rules, and data layout conventions. The Tile IR type system is foundational to the language's correctness guarantees and directly shapes how programs interact with GPU hardware.

---

## 5.1 Overview

The Tile IR type system is built on the following core principles:

**Static typing.** All values in a Tile IR program are statically typed. The type of every register, operation result, kernel parameter, and global variable is known at compile time. There are no dynamically typed values, no type erasure, and no runtime type dispatch. This enables the Tile IR compiler to emit efficient machine code without any type-checking overhead at execution time.

**Tensor-valued computation.** Tile IR is a tensor-oriented instruction set: every value is a tensor. There are no bare scalar values in the system. Even what programmers think of as a "scalar" is, in Tile IR's type system, a rank-0 tensor (a tile with an empty shape). This design reflects the hardware reality that GPU computation is fundamentally structured around multi-dimensional data movement and transformation.

**Two tensor kinds.** There are exactly two tensor types in Tile IR:

1. **Tile** (`tile<...>`) -- a pure tensor value that lives in registers. Tiles carry actual data. All arithmetic, logical, and comparison operations produce tiles. A tile has a fully static shape known at compile time.

2. **Tensor view** (`tensor_view<...>`) -- a structured pointer that describes how to interpret a region of global memory as a multi-dimensional array. Views do not hold data themselves; they describe how to access data in memory. A derived form, **partition view** (`partition_view<...>`), describes a tiled partitioning of a tensor view.

**Element types are not standalone.** An element type (such as `f32` or `i16`) does not describe a value on its own. Element types exist only as constituents of tensor types. You cannot declare a register of type `f32`; you must declare it as `tile<f32>` (a rank-0 tile containing one f32 element) or `tile<MxNxf32>` (a rank-2 tile). This is analogous to how a lane type in SIMD programming only has meaning within the context of a vector register.

```
// Valid: element types appear only within tensor types
%x : tile<f32>              // rank-0 tile (scalar tensor)
%y : tile<128xf32>          // rank-1 tile (vector tensor)
%z : tile<128x64xf32>       // rank-2 tile (matrix tensor)

// Invalid: bare element types are not value types
// %x : f32                 // ERROR: element type is not a value type
```

---

## 5.2 Element Types

Element types define the representation of individual data elements within tensors. Tile IR provides two categories of element types: **fundamental types** (IEEE-standard integers and floating-point numbers) and **alternative types** (specialized low-precision formats for machine learning workloads).

### 5.2.1 Fundamental Types

Fundamental types comprise signless integers and IEEE 754 floating-point numbers. These types are universally supported across all Tile IR operations and target architectures.

| Type | Size (bits) | Size (bytes) | Category | Description |
|------|-------------|---------------|----------|-------------|
| `i1` | 1 | -- | Boolean | Predicate / boolean value. Stored as a bit within a container type; does not occupy a full byte on its own. |
| `i8` | 8 | 1 | Integer | 8-bit signless integer. |
| `i16` | 16 | 2 | Integer | 16-bit signless integer. |
| `i32` | 32 | 4 | Integer | 32-bit signless integer. Most common integer type. |
| `i64` | 64 | 8 | Integer | 64-bit signless integer. |
| `f16` | 16 | 2 | Float | IEEE 754 half-precision (binary16): 1 sign bit, 5 exponent bits, 10 mantissa bits. |
| `f32` | 32 | 4 | Float | IEEE 754 single-precision (binary32): 1 sign bit, 8 exponent bits, 23 mantissa bits. |
| `f64` | 64 | 8 | Float | IEEE 754 double-precision (binary64): 1 sign bit, 11 exponent bits, 52 mantissa bits. |

> **Note: Signless integers.** Tile IR integers are signless. The bit pattern `0xFF` in an `i8` register does not inherently represent -1 (signed) or 255 (unsigned). Instead, the interpretation is determined by the operation that consumes the value. For example, `cmpi` (integer comparison) takes an explicit signedness flag:
>
> ```
> // Signed comparison
> %r = cmpi sgt %a, %b : tile<i32>   // treat operands as signed
>
> // Unsigned comparison
> %r = cmpi ugt %a, %b : tile<i32>   // treat operands as unsigned
> ```
>
> This design avoids the need for separate signed and unsigned integer types and eliminates the need for signedness conversion operations. Operations that care about signedness include `cmpi`, `divi`, `remi`, `shri` (arithmetic vs. logical right shift), and conversion operations (`itof`, `ftoi`).

> **Note: The `i1` type.** The `i1` type is used exclusively for predicate and boolean values. It is produced by comparison operations (`cmpi`, `cmpf`) and consumed by the `select` operation. It cannot be used in arithmetic operations (`addi`, `muli`, etc.). The `i1` type has a storage size of 1 bit, but when stored in memory it occupies 1 byte with the value in the least significant bit.

### 5.2.2 Alternative Types

Alternative types are specialized numeric formats designed primarily for machine learning workloads. They trade full IEEE compliance for improved computational density, memory bandwidth, or dynamic range characteristics.

| Type | Total Bits | Sign | Exponent | Mantissa | Description |
|------|------------|------|----------|----------|-------------|
| `tf32` | 32 | 1 | 8 | 10 | TensorFloat-32. Same range as f32 (8 exponent bits) but reduced precision (10 mantissa bits vs. 23). Designed for transparent use in deep learning training without hyperparameter tuning. |
| `bf16` | 16 | 1 | 8 | 7 | Brain Float 16. Same dynamic range as f32 (8 exponent bits) but greatly reduced precision (7 mantissa bits vs. 23). Popular in deep learning for its f32-like range. |
| `e4m3` | 8 | 1 | 4 | 3 | FP8 E4M3FN (NVIDIA/OCP). Higher precision than e5m2 but smaller dynamic range. The "FN" suffix denotes that NaN is not supported (the NaN encoding is reused for additional normal values). |
| `e5m2` | 8 | 1 | 5 | 2 | FP8 E5M2 (NVIDIA/OCP). Larger dynamic range than e4m3 but lower precision. Supports NaN and Inf representations. |

**Bit layout diagrams:**

```
tf32 (32 bits):
+---+----------+-------------------+-----------+
| S | EEEEEEEE | MMMMMMMMmm        | (padded)  |
+---+----------+-------------------+-----------+
  31  30 .. 23   22 .. 13           12 .. 0 (implicit zeros)

bf16 (16 bits):
+---+----------+---------+
| S | EEEEEEEE | MMMMMMM |
+---+----------+---------+
  15  14 ..  7   6 ..  0

e4m3 (8 bits):
+---+------+---+
| S | EEEE | MMM |
+---+------+---+
  7  6..3   2..0

e5m2 (8 bits):
+---+-------+--+
| S | EEEEE | MM |
+---+-------+--+
  7  6..2   1..0
```

**Representable ranges:**

| Type | Min Subnormal | Min Normal | Max Normal | Max Exponent |
|------|---------------|------------|------------|--------------|
| `tf32` | ~1.4e-45 | ~1.2e-38 | ~3.4e+38 | 127 |
| `bf16` | ~9.2e-41 | ~1.2e-38 | ~3.4e+38 | 127 |
| `e4m3` | ~1.6e-6 | ~1.5e-2 | ~448.0 | 7 |
| `e5m2` | ~2.0e-6 | ~6.1e-5 | ~57344.0 | 15 |

**Restrictions on alternative types:**

Alternative types have restricted support in Tile IR operations. The following table summarizes which operation categories support each alternative type:

| Operation Category | `tf32` | `bf16` | `e4m3` | `e5m2` | Notes |
|--------------------|--------|--------|--------|--------|-------|
| Arithmetic (`addf`, `subf`, `mulf`, `divf`) | No | No | No | No | Alternative types are generally not supported element-wise. |
| Matrix multiply (`mmaf`) | Yes (input) | No | Yes (input) | Yes (input) | Alternative types serve as MMA inputs; accumulator is typically f32 or f16. |
| Comparison (`cmpf`) | No | No | No | No | Use conversion to a fundamental type first. |
| Conversion (`ftof`) | Yes | Yes | Yes | Yes | Conversion to/from fundamental types is fully supported. |
| Memory operations (load/store) | Yes | Yes | Yes | Yes | All alternative types can be loaded from and stored to memory. |
| Bitcast | Yes | Yes | Yes | Yes | Reinterpretation of bits is always supported. |

> **Design rationale.** Alternative types exist primarily as memory storage formats and matrix multiply accumulator inputs. They are not intended for general-purpose computation. The typical workflow is: load alternative-type data from memory, convert to a fundamental type (f16 or f32) via `ftof`, perform element-wise computation, convert back if needed, and store. For matrix multiplication, alternative types can be used directly as `mmaf` inputs without element-wise conversion.

### 5.2.3 Floating-Point Conversion Semantics

Tile IR defines two distinct conversion semantics for floating-point values, depending on the source and destination types. These semantics govern what happens when a value cannot be exactly represented in the target type.

#### IEEE Rounding Semantics

Conversions between IEEE types (`f16`, `f32`, `f64`) and `bf16`/`tf32` use **standard IEEE rounding**:

- **Nearest representable value**: The result is the value in the destination type that is closest to the original value. When the original value falls exactly between two representable values, the tie-breaking rule specified by the operation's rounding mode attribute is used (e.g., `rounding<nearest_even>`).
- **NaN preserved**: If the input is NaN, the output is NaN. The specific NaN payload is not guaranteed to be preserved; a canonical NaN may be substituted.
- **Inf produced**: If the input value exceeds the largest finite representable value in the destination type, the result is positive or negative infinity, as appropriate.
- **Subnormals**: Subnormal (denormalized) values are preserved when the destination format supports them. If the destination format does not support subnormals, the result is flushed to zero (FTZ).

#### Saturation-to-Finite (satfinite) Semantics

Conversions involving the low-precision FP8 types (`e4m3`, `e5m2`) use **saturation-to-finite** (satfinite) semantics:

- **Saturation**: If the input value exceeds the largest finite representable value in the destination type, the result saturates to that maximum finite value (not infinity). For example, converting a large f32 value to e4m3 yields `448.0` (the maximum e4m3 value), not infinity.
- **No Inf output**: Infinity is never produced as a conversion result. If the input is Inf, the result is the maximum finite value of the destination type.
- **NaN handling**: NaN inputs produce NaN outputs when the destination type supports NaN (e5m2). If the destination type does not support NaN (e4m3), NaN inputs map to a defined fallback value.

#### Special Value Behavior Table

The following table specifies the behavior for converting special values between types:

| Source Value | f16 -> f32 | f32 -> f16 | f32 -> bf16 | f32 -> tf32 | f32 -> e4m3 | f32 -> e5m2 | e4m3 -> f32 | e5m2 -> f32 |
|---|---|---|---|---|---|---|---|---|
| **Normal (in range)** | Exact | Nearest | Nearest | Nearest (loss of low mantissa bits) | Nearest | Nearest | Exact | Exact |
| **Normal (out of range)** | N/A | +/-Inf | +/-Inf | N/A | Saturate to +/-448.0 | Saturate to +/-57344.0 | N/A | N/A |
| **+Inf** | +Inf | +Inf | +Inf | +Inf | +448.0 (satfinite) | +57344.0 (satfinite) | +Inf | +Inf |
| **-Inf** | -Inf | -Inf | -Inf | -Inf | -448.0 (satfinite) | -57344.0 (satfinite) | -Inf | -Inf |
| **NaN** | NaN | NaN | NaN | NaN | Fallback (no NaN) | NaN | NaN | NaN |
| **+0.0** | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 |
| **-0.0** | -0.0 | -0.0 | -0.0 | -0.0 | -0.0 | -0.0 | -0.0 | -0.0 |
| **Subnormal** | Exact | Preserve | Flush to zero | Preserve | Flush to zero | Flush to zero | Exact | Exact |

> **Important note on e4m3 and NaN.** The e4m3 format does not support NaN. The bit patterns that would represent NaN in other floating-point formats are instead used to represent additional finite values in e4m3. This means:
>
> - Converting NaN to e4m3 produces a defined finite value, not NaN.
> - There is no way to represent or propagate NaN through e4m3 data.
> - Debugging code that relies on NaN propagation will not work correctly when e4m3 is involved.

#### Conversion Examples

```
// f32 -> e4m3: saturation semantics
// 1000.0 exceeds e4m3 range (max = 448.0)
%a = ftof %val : tile<f32> -> tile<e4m3>    // result: 448.0

// f32 -> f16: IEEE semantics
// 70000.0 exceeds f16 range (max = 65504.0)
%b = ftof %big : tile<f32> -> tile<f16>     // result: +Inf

// e4m3 -> f32: exact expansion
%c = ftof %fp8 : tile<e4m3> -> tile<f32>    // result: exact representation

// bf16 -> f32: exact expansion (bf16 is a truncated f32)
%d = ftof %bf : tile<bf16> -> tile<f32>     // result: exact (zero-extended mantissa)
```

---

## 5.3 Pointers

Tile IR provides a typed pointer abstraction for referencing elements in the global address space.

### 5.3.1 Pointer Type Syntax

```
ptr<E>
```

Where `E` is an element type. This denotes a 64-bit pointer that points to a memory location containing a value of element type `E`.

```
ptr<f32>     // pointer to a 32-bit float
ptr<i32>     // pointer to a 32-bit integer
ptr<bf16>    // pointer to a brain-float-16 value
ptr<e4m3>    // pointer to an FP8 e4m3 value
```

### 5.3.2 Pointer Properties

| Property | Value |
|----------|-------|
| Size | 64 bits (8 bytes) |
| Address space | Global memory only |
| Alignment | Naturally aligned to the element type |
| Null value | Supported (all-zeros bit pattern) |

### 5.3.3 Pointer Arithmetic

Pointer arithmetic in Tile IR is **element-scaled**. Incrementing a pointer by an integer offset `n` advances the address by `n * sizeof(E)` bytes, where `E` is the pointer's element type. This is consistent with C/C++ pointer semantics.

```
// Offset a base pointer by a tile of integer offsets
// Each pointer advances by offset * sizeof(f32) = offset * 4 bytes
%offsets = iota : tile<128xi32>
%ptrs = offset %base_ptr, %offsets : tile<ptr<f32>>, tile<128xi32> -> tile<128xptr<f32>>
// Result: ptrs[i] = base_ptr + offsets[i] * sizeof(f32)
```

**Element type storage sizes used for pointer arithmetic:**

| Element Type | `sizeof(E)` (bytes) | Offset multiplier |
|---|---|---|
| `i8`, `e4m3`, `e5m2` | 1 | 1 |
| `i16`, `f16`, `bf16` | 2 | 2 |
| `i32`, `f32`, `tf32` | 4 | 4 |
| `i64`, `f64` | 8 | 8 |
| `i1` | 1 | 1 |
| `ptr<E>` | 8 | 8 |

### 5.3.4 Restrictions

- **No nested pointers.** The type `ptr<ptr<E>>` is not valid. Tile IR does not support pointers to pointers. If indirection is needed, use `ptr<i64>` and interpret the stored integers as addresses via `int_to_ptr`.
- **No function pointers.** Pointers can only reference data, not code.
- **No pointer-to-tile.** Pointers reference individual elements, not tile values. To reference a block of elements, use a tensor view (Section 5.4.2).
- **No address-space annotation.** All pointers implicitly reference global memory. There are no shared-memory or constant-memory pointers in Tile IR's type system.

### 5.3.5 Pointer Type Conversion

Type conversion between pointer types uses the `ptr_to_ptr` operation (for reinterpretation) or the `bitcast` operation. Neither operation modifies the address value; they only change the element type interpretation.

```
// Reinterpret a float pointer as an integer pointer
%iptr = ptr_to_ptr %fptr : tile<ptr<f32>> -> tile<ptr<i32>>

// Same address, different element type interpretation
// ptr<f32> with value 0x7f000000 -> ptr<i32> with value 0x7f000000
// Both point to the same byte address
```

---

## 5.4 Tensor Types

Tile IR defines three tensor types: **tile**, **tensor view**, and **partition view**. Each serves a distinct purpose in the programming model.

### 5.4.1 Tile Type

A tile is a multi-dimensional array of elements with a **statically known shape**. Tiles are the fundamental computational unit in Tile IR: all data processing operates on tiles, and all operation results are tiles.

#### Syntax

```
tile<shape x element_type>
tile<shape x ptr<element_type>>
```

Where `shape` is a sequence of dimension sizes separated by `x`, and each dimension size is a compile-time constant that is a **power of 2**.

#### Shape Rules

1. **All dimensions must be powers of 2.** This is a hard constraint enforced by the Tile IR validator. Valid dimension sizes are: 1, 2, 4, 8, 16, 32, 64, 128, 256, etc. The value `3` is not a valid dimension size.

2. **Rank can be 0 or more.** A rank-0 tile has no dimensions and contains exactly one element (a scalar tensor). A rank-1 tile is a vector. A rank-2 tile is a matrix. Higher-rank tiles are supported but less common.

3. **The total number of elements is the product of all dimensions.** For `tile<MxNxKxE>`, the tile contains `M * N * K` elements of type `E`.

4. **No dynamic dimensions.** All dimensions are compile-time constants. Tiles cannot have dynamic extents (use tensor views for dynamic shapes).

#### Examples

```
// Scalar tiles (rank-0)
tile<f32>                  // single f32 value
tile<i32>                  // single i32 value
tile<ptr<f32>>             // single pointer to f32

// Vector tiles (rank-1)
tile<128xi32>              // 128 i32 values
tile<64xf16>               // 64 f16 values
tile<256xptr<f32>>         // 256 pointers to f32

// Matrix tiles (rank-2)
tile<128x64xf32>           // 128 x 64 matrix of f32 values (8192 elements)
tile<16x16xf16>            // 16 x 16 matrix of f16 values (256 elements)
tile<64x128xptr<f32>>      // 64 x 128 grid of pointers to f32

// Higher-rank tiles (rank-3)
tile<4x8x16xf32>           // 4 x 8 x 16 tensor of f32 values (512 elements)
```

#### Tiles of Pointers

A tile of pointers (`tile<MxNxptr<E>>`) is used for **scatter/gather** memory access patterns. Each element is an independent pointer, allowing non-contiguous memory accesses to be expressed naturally.

```
// Create 128 pointers starting from a base address
%offsets = iota : tile<128xi32>                    // offsets: [0, 1, 2, ..., 127]
%base = reshape %base_ptr : tile<ptr<f32>> -> tile<1xptr<f32>>
%base_bc = broadcast %base : tile<1xptr<f32>> -> tile<128xptr<f32>>
%ptrs = offset %base_bc, %offsets : tile<128xptr<f32>>, tile<128xi32> -> tile<128xptr<f32>>

// Gather: load from 128 potentially non-contiguous addresses
%vals, %tok = load_ptr_tko weak %ptrs : tile<128xptr<f32>> -> tile<128xf32>, token

// Scatter: store to 128 potentially non-contiguous addresses
%tok2 = store_ptr_tko weak %ptrs, %vals : tile<128xptr<f32>>, tile<128xf32> -> token
```

#### Tile Size Considerations

The hardware implementation of tiles maps to physical register file space. Large tiles consume more registers, which can limit occupancy. The Tile IR compiler manages register allocation, but programmers should be aware of tile sizes:

| Tile Type | Elements | Bytes (f32) | Register Count (32-bit registers) |
|---|---|---|---|
| `tile<f32>` | 1 | 4 | 1 |
| `tile<32xf32>` | 32 | 128 | 32 |
| `tile<64xf32>` | 64 | 256 | 64 |
| `tile<128xf32>` | 128 | 512 | 128 |
| `tile<128x64xf32>` | 8,192 | 32,768 | 8,192 |
| `tile<128x64xf16>` | 8,192 | 16,384 | 4,096 |

> **Note: Register pressure.** A single `tile<128x128xf32>` requires 16,384 32-bit registers, which exceeds the maximum register file size per SM on current architectures. The Tile IR compiler may spill such tiles to local memory or split operations into smaller tiles. Programmers should prefer tile sizes that fit within hardware constraints. Typical MMA tile sizes (e.g., 128x64, 64x128) are designed to map efficiently to tensor core operations.

### 5.4.2 Tensor View

A tensor view is a **structured pointer** that describes a multi-dimensional array in global memory. Unlike a tile, a tensor view does not hold data -- it describes how to access data stored at a base address using shape and stride information.

#### Syntax

```
tensor_view<shape x element_type, strides=[stride_list]>
```

Where:
- `shape` is a list of dimension extents, which may be **dynamic** (written as `?`) or static literals.
- `element_type` is the type of each element.
- `stride_list` is a list of strides, which may be dynamic (`?`) or static literals.

#### Creation

Tensor views are created using the `make_tensor_view` operation:

```
%view = make_tensor_view %base_ptr, shape=[%M, %N], strides=[%s0, %s1]
    : tile<i32> -> tensor_view<?x?xf32, strides=[?, ?]>
```

The parameters `shape` and `strides` are provided as tile values (typically `tile<i32>` for dynamic dimensions). The result type records whether each dimension/stride is dynamic or static.

#### Address Computation Formula

For a tensor view with shape `[S0, S1, ..., Sn]` and strides `[s0, s1, ..., sn]`, the address of element at indices `[i0, i1, ..., in]` is computed as:

```
address(base, i0, i1, ..., in) = base + sum(ik * sk for k = 0..n)
```

Where `base` is the pointer value used to create the view, `ik` is the index in dimension `k`, and `sk` is the stride in dimension `k`.

```
// For a 2D view with shape [M, N] and strides [stride_m, 1]:
ptr[i][j] = base + i * stride_m + j * 1

// This represents a row-major MxN matrix where stride_m >= N
```

#### Dynamic Extents

The `?` character denotes a dynamic dimension or stride whose value is not known at compile time. Dynamic values are provided at kernel launch time through the `make_tensor_view` operation's operands.

```
// Fully dynamic: all extents and strides are runtime values
tensor_view<?x?xf32, strides=[?, ?]>

// Mixed: first extent is dynamic, second is static; strides are dynamic
tensor_view<?x64xf32, strides=[?, 1]>

// Fully static: all extents and strides are known at compile time
tensor_view<128x64xf32, strides=[64, 1]>
```

> **Note: Dynamic vs. static in types.** A `tensor_view<?x?xf32, strides=[?, ?]>` and a `tensor_view<128x64xf32, strides=[64, 1]>` are **different types**, even if the dynamic values happen to be 128, 64, 64, and 1 at runtime. Type equivalence in Tile IR is purely structural and syntactic (see Section 5.5).

#### Examples of Different Views

```
// Row-major matrix M x N
%rm = make_tensor_view %ptr, shape=[%M, %N], strides=[%N, 1]
    : tile<i32> -> tensor_view<?x?xf32, strides=[?, 1]>

// Column-major matrix M x N
%cm = make_tensor_view %ptr, shape=[%M, %N], strides=[1, %M]
    : tile<i32> -> tensor_view<?x?xf32, strides=[1, ?]>

// Strided access: every other element in a 1D array
%strided = make_tensor_view %ptr, shape=[%N], strides=[2]
    : tile<i32> -> tensor_view<?xf32, strides=[?]>

// 3D tensor M x N x K in row-major order
%vol = make_tensor_view %ptr, shape=[%M, %N, %K], strides=[%Nk, %K, 1]
    : tile<i32> -> tensor_view<?x?x?f32, strides=[?, ?, 1]>

// Transposed view of a matrix (same data, different logical layout)
%trans = make_tensor_view %ptr, shape=[%N, %M], strides=[1, %N]
    : tile<i32> -> tensor_view<?x?xf32, strides=[1, ?]>
```

### 5.4.3 Subview Types (Partition View)

A partition view describes a **tiled partitioning** of a tensor view. It maps from an index space (defined by the grid of tile blocks) to statically-sized tiles of the underlying tensor view. Partition views are the primary mechanism for dividing work across tile blocks in Tile IR.

#### Syntax

```
partition_view<tile=tile_shape, view_type, dim_map=map, padding_value=val>
```

Where:
- `tile_shape` is the static shape of each tile partition (e.g., `(128x64)`). All dimensions must be powers of 2.
- `view_type` is the underlying tensor view type.
- `dim_map` (optional) specifies the mapping from index space dimensions to tensor view dimensions.
- `padding_value` (optional) specifies the value used for out-of-bounds elements during loads.

#### Creation

Partition views are created from tensor views using `make_partition_view`:

```
%pview = make_partition_view %view
    : tensor_view<?x?xf32, strides=[?, 1]> ->
      partition_view<tile=(128x64), tensor_view<?x?xf32, strides=[?, 1]>, dim_map=[0, 1]>
```

#### Index Space Shape

Given a partition view with tile shape `[T0, T1, ..., Tn]` over a tensor view with shape `[S0, S1, ..., Sn]`, the **index space shape** is:

```
Pk = ceil(Sk / Tk)
```

Where `ceil` rounds up to the nearest integer. The index space defines the grid of tiles that cover the tensor view. Each tile block is assigned coordinates in this index space.

```
// Example: tensor view shape [256, 128], tile shape [64, 32]
// Index space shape: [ceil(256/64), ceil(128/32)] = [4, 4]
// There are 4 x 4 = 16 tile partitions

// Example: tensor view shape [300, 100], tile shape [128, 64]
// Index space shape: [ceil(300/128), ceil(100/64)] = [3, 2]
// There are 3 x 2 = 6 tile partitions
// Note: some partitions will partially extend beyond the tensor view
```

The index space shape can be queried at runtime using `get_index_space_shape`:

```
%ishape = get_index_space_shape %pview
    : partition_view<tile=(128x64), tensor_view<?x?xf32, strides=[?, 1]>> -> tile<i32>
```

#### Dimension Mapping

The `dim_map` attribute controls which index space dimensions correspond to which tensor view dimensions. This enables sophisticated partitioning strategies:

```
// dim_map=[0, 1]: index space dimension 0 maps to tensor dimension 0,
//                  index space dimension 1 maps to tensor dimension 1
// Standard row-major partitioning

// dim_map=[1, 0]: index space dimension 0 maps to tensor dimension 1,
//                  index space dimension 1 maps to tensor dimension 0
// Transposed partitioning

// This is used for GEMM where A is partitioned along M,
// B is partitioned along N, and both iterate over K
```

#### Out-of-Bounds Handling

When the tensor view's extent is not evenly divisible by the tile shape, some tile partitions will extend beyond the tensor's logical boundary. The behavior differs between loading and storing:

| Operation | Out-of-bounds behavior |
|---|---|
| **Loading** (`load_view_tko`) | Out-of-bounds elements are filled with the partition view's `padding_value`. The padding value defaults to `0` for integer types and `0.0` for floating-point types. A custom padding value can be specified. The in-bounds elements are loaded normally. |
| **Storing** (`store_view_tko`) | Out-of-bounds elements are **not written**. Only the in-bounds elements within the tile partition are stored. This prevents corrupting adjacent memory. |

```
// Tensor view: shape [200, 50], tile shape: [128, 64]
// Index space: [ceil(200/128), ceil(50/64)] = [2, 1]
//
// Tile block (0,0): covers rows [0,127], cols [0,49]
//   - Loading: rows 0-127 cols 0-49 are loaded; cols 50-63 are padded
//   - Storing: only rows 0-127 cols 0-49 are stored
//
// Tile block (1,0): covers rows [128,255], cols [0,49]
//   - Loading: rows 128-199 are loaded; rows 200-255 are padded; cols 50-63 are padded
//   - Storing: only rows 128-199 cols 0-49 are stored
```

#### Partition View Examples

**Example 1: Simple 1D partition**

```
// Partition a 1D array into chunks of 128 elements
%view = make_tensor_view %ptr, shape=[%N], strides=[1]
    : tile<i32> -> tensor_view<?xf32, strides=[1]>

%pview = make_partition_view %view
    : tensor_view<?xf32, strides=[1]> ->
      partition_view<tile=(128), tensor_view<?xf32, strides=[1]>, dim_map=[0]>

// Index space: [ceil(N/128)]
// Tile block i covers elements [i*128, i*128+127]
```

**Example 2: 2D partition for matrix operations**

```
// Partition a 2D matrix into 128x64 blocks
%view = make_tensor_view %ptr, shape=[%M, %N], strides=[%stride, 1]
    : tile<i32> -> tensor_view<?x?xf32, strides=[?, 1]>

%pview = make_partition_view %view
    : tensor_view<?x?xf32, strides=[?, 1]> ->
      partition_view<tile=(128, 64), tensor_view<?x?xf32, strides=[?, 1]>, dim_map=[0, 1]>

// Index space: [ceil(M/128), ceil(N/64)]
// Tile block (i, j) covers rows [i*128, i*128+127], cols [j*64, j*64+63]
```

**Example 3: GEMM partitioning with transposed dimension mapping**

```
// Matrix A: shape [K, M], partitioned along K and M
// We want tile blocks indexed by (m_idx, k_idx) to load tile A[k_idx, m_idx]
%A_view = make_tensor_view %A_ptr, shape=[%K, %M], strides=[%stride_ak, 1]
    : tile<i32> -> tensor_view<?x?xf16, strides=[?, 1]>

%A_block = make_partition_view %A_view
    : tensor_view<?x?xf16, strides=[?, 1]> ->
      partition_view<tile=(128, 64), tensor_view<?x?xf16, strides=[?, 1]>, dim_map=[1, 0]>

// Note dim_map=[1, 0]: first index space dim -> tensor dim 1 (M),
//                        second index space dim -> tensor dim 0 (K)
// This means A_block[m, k] loads tile at rows [k*128, k*128+127], cols [m*64, m*64+63]
```

**Example 4: Partition view with custom padding**

```
// Partition with custom padding value for out-of-bounds elements
%pview = make_partition_view %view, padding_value=-1.0
    : tensor_view<?x?xf32, strides=[?, 1]> ->
      partition_view<tile=(128, 64), tensor_view<?x?xf32, strides=[?, 1]>,
                     dim_map=[0, 1], padding_value=-1.0>

// Loading out-of-bounds elements will yield -1.0 instead of the default 0.0
```

---

## 5.5 Type Equivalence

Type equivalence in Tile IR is **structural equality**: two types are equal if and only if they have the identical textual representation. There is no notion of type aliasing, subtyping, or structural compatibility beyond exact match.

### 5.5.1 Rules

1. **Element types are equal** if they have the same name: `f32 == f32`, `i32 == i32`, `bf16 == bf16`.

2. **Tile types are equal** if they have the same shape and element type: `tile<128x64xf32> == tile<128x64xf32>`.

3. **Tensor view types are equal** if they have the same shape, strides, and element type, including whether dimensions/strides are dynamic or static.

4. **Partition view types are equal** if they have the same tile shape, underlying view type, dim_map, and padding_value.

5. **Pointer types are equal** if their element types are equal: `ptr<f32> == ptr<f32>`.

### 5.5.2 Dynamic vs. Static Distinction

Dynamic dimensions (`?`) and static dimensions (integer literals) are **never equivalent**, even if the runtime value matches. This is because the type system encodes compile-time information that the compiler uses for optimization and validation.

```
// These are DIFFERENT types:
tensor_view<?x?xf32, strides=[?, 1]>     // dynamic extents, dynamic first stride
tensor_view<128x64xf32, strides=[64, 1]>  // static extents, static strides

// These are DIFFERENT types:
tile<128xf32>                              // rank-1 tile
tile<1x128xf32>                            // rank-2 tile (even though both have 128 elements)

// These are EQUAL types:
tile<f32>                                   // rank-0 tile
tile<f32>                                   // identical
```

### 5.5.3 Type Compatibility in Operations

Operations require exact type matches for their operands and results. There is no implicit type coercion. If an operation expects a `tile<128xf32>` operand, providing a `tile<64xf32>` operand is a type error, even if the program logic would otherwise be correct.

```
// Type error: shape mismatch
%a : tile<128xf32>
%b : tile<64xf32>
%c = addf %a, %b : tile<128xf32>    // ERROR: operand type tile<64xf32> does not match expected tile<128xf32>

// Correct: matching shapes
%d : tile<128xf32>
%e : tile<128xf32>
%f = addf %d, %e : tile<128xf32>    // OK: all types match
```

Shape transformations (broadcast, reshape, etc.) must be performed explicitly before operations that require specific shapes.

---

## 5.6 Data Layout

This section describes how Tile IR element types map to standard data interchange formats, and the layout requirements that Tile IR imposes on memory.

### 5.6.1 Element Type Encoding Table

The following table maps Tile IR element types to their equivalents in common frameworks and specifications:

| Tile IR Type | Size (bits) | DLPack Code | NumPy dtype | PyTorch dtype | CUDA C++ type |
|---|---|---|---|---|---|
| `i1` | 1 | `kBool` | `np.bool_` | `torch.bool` | `bool` |
| `i8` | 8 | `kInt8` | `np.int8` | `torch.int8` | `int8_t` |
| `i16` | 16 | `kInt16` | `np.int16` | `torch.int16` | `int16_t` |
| `i32` | 32 | `kInt32` | `np.int32` | `torch.int32` | `int32_t` |
| `i64` | 64 | `kInt64` | `np.int64` | `torch.int64` | `int64_t` |
| `f16` | 16 | `kFloat16` | `np.float16` | `torch.float16` | `__half` |
| `f32` | 32 | `kFloat32` | `np.float32` | `torch.float32` | `float` |
| `f64` | 64 | `kFloat64` | `np.float64` | `torch.float64` | `double` |
| `bf16` | 16 | `kBfloat16` | -- (not in NumPy) | `torch.bfloat16` | `__nv_bfloat16` |
| `tf32` | 32 | -- (no DLPack code) | -- | -- | -- (internal format) |
| `e4m3` | 8 | `kFloat8_e4m3fn` | -- | `torch.float8_e4m3fn` | `__nv_fp8_e4m3` |
| `e5m2` | 8 | `kFloat8_e5m2` | -- | `torch.float8_e5m2` | `__nv_fp8_e5m2` |

> **Note: tf32 as a storage format.** The `tf32` type is unusual in that it occupies 32 bits in memory but only uses 19 bits of information (1 sign + 8 exponent + 10 mantissa). When stored to memory from an f32 value, the lower 13 mantissa bits are silently discarded. When loaded from memory, those 13 bits are zero-filled. This means `tf32` can be stored in the same memory as `f32` without changing the memory footprint, but with reduced precision.

### 5.6.2 Contiguous Layout Requirements

Tile IR memory operations have specific layout requirements depending on the operation type:

| Operation | Layout Requirement | Notes |
|---|---|---|
| `load_ptr_tko` / `store_ptr_tko` | None (gather/scatter) | Each pointer is independent; no contiguity required. |
| `load_view_tko` / `store_view_tko` | Described by tensor view strides | The view's strides define the layout; any stride pattern is valid, including non-contiguous layouts. |
| `mmaf` (matrix multiply) | Architecture-specific | Tensor core operations may require specific leading dimensions or alignment. The Tile IR compiler handles these requirements automatically. |

**Vectorized loads.** While Tile IR does not require contiguous access, performance is significantly better when the innermost dimension of a tensor view has stride 1 (contiguous in the innermost dimension). The Tile IR compiler may generate optimized memory instructions (e.g., vector loads) when it can prove contiguity.

### 5.6.3 Alignment Requirements

All memory accesses through Tile IR must satisfy the following alignment requirements:

| Element Type | Required Alignment (bytes) | Notes |
|---|---|---|
| `i8`, `e4m3`, `e5m2` | 1 | No alignment constraint. |
| `i16`, `f16`, `bf16` | 2 | 2-byte aligned. |
| `i32`, `f32`, `tf32` | 4 | 4-byte aligned. |
| `i64`, `f64` | 8 | 8-byte aligned. |
| `ptr<E>` | 8 | 8-byte aligned (64-bit pointers). |

**Base pointer alignment.** The base pointer provided to `make_tensor_view` must be aligned to the natural alignment of the element type. Unaligned pointers result in undefined behavior.

**Tile alignment.** When loading a tile from a tensor view partition, the starting address of the tile (computed as `base + sum(ik * sk)`) should be aligned to the element type's alignment for optimal performance. The Tile IR compiler may insert padding or use slower unaligned access instructions for misaligned tiles.

**MMA alignment.** Operations involving tensor cores (`mmaf`, `mmai`) may impose stricter alignment requirements on the leading dimension of the tensor view. These requirements are architecture-dependent:

| Architecture | MMA Alignment Requirement |
|---|---|
| sm_80 (Ampere) | Leading dimension must be a multiple of 8 bytes for f16 MMA |
| sm_89 (Ada) | Leading dimension must be a multiple of 8 bytes for f16 MMA |
| sm_100 (Blackwell) | Leading dimension must be a multiple of 16 bytes for FP8 MMA |

The Tile IR compiler validates these requirements at compile time when the strides are static, and at runtime when they are dynamic.

### 5.6.4 Byte Ordering and Bit Layout

All Tile IR types use **little-endian** byte ordering, consistent with NVIDIA GPU hardware:

- Multi-byte values are stored with the least significant byte at the lowest address.
- Floating-point values follow IEEE 754 bit layout with sign bit in the most significant position.
- Pointer values are stored as 64-bit little-endian unsigned integers.

### 5.6.5 Padding and Storage of Tile Types in Memory

When tiles are stored to memory (via `store_ptr_tko`, `store_view_tko`, or other mechanisms), the elements are stored in **row-major order** with no padding between elements or between rows:

```
// tile<2x3xf32> stored in memory (row-major, no padding):
// Address:   base+0  base+4  base+8  base+12  base+16  base+20
// Element:   [0,0]   [0,1]   [0,2]   [1,0]    [1,1]    [1,2]

// tile<2x2x3xf32> stored in memory (depth > row > column):
// Elements in order: [0,0,0], [0,0,1], [0,0,2], [0,1,0], [0,1,1], [0,1,2],
//                    [1,0,0], [1,0,1], [1,0,2], [1,1,0], [1,1,1], [1,1,2]
```

> **Note: No implicit padding.** Unlike some GPU programming models that pad shared memory accesses to avoid bank conflicts, Tile IR never inserts implicit padding. The memory layout of a tile is exactly the contiguous row-major sequence of its elements. If padding is needed for performance reasons, it must be explicitly expressed through tensor view strides.

---

## 5.7 Type System Summary

The following table provides a complete summary of all types in the Tile IR type system:

| Category | Type | Syntax | Static/Dynamic | Description |
|---|---|---|---|---|
| Element | Integer | `i1`, `i8`, `i16`, `i32`, `i64` | N/A | Signless integer |
| Element | IEEE Float | `f16`, `f32`, `f64` | N/A | IEEE 754 floating-point |
| Element | Alt Float | `bf16`, `tf32`, `e4m3`, `e5m2` | N/A | Specialized ML format |
| Pointer | Typed pointer | `ptr<E>` | N/A | 64-bit pointer to element type E |
| Tensor | Tile | `tile<Sx...xE>` | Fully static | Register-resident tensor value |
| Tensor | Tile of pointers | `tile<Sx...xptr<E>>` | Fully static | Register-resident tensor of pointers |
| Tensor | Tensor view | `tensor_view<Sx...xE, strides=[...]>` | Mixed | Structured pointer to global memory |
| Tensor | Partition view | `partition_view<tile=..., ...>` | Mixed | Tiled partitioning of a tensor view |

**Type formation rules:**

1. An element type `E` is valid if it is one of: `i1`, `i8`, `i16`, `i32`, `i64`, `f16`, `f32`, `f64`, `bf16`, `tf32`, `e4m3`, `e5m2`.

2. A pointer type `ptr<E>` is valid if `E` is a valid element type. The type `ptr<ptr<E>>` is not valid (no nested pointers).

3. A tile type `tile<S0 x S1 x ... x Sn x E>` is valid if:
   - Each `Si` is a positive integer that is a power of 2.
   - `E` is a valid element type or pointer type.

4. A tensor view type `tensor_view<S0 x ... x Sn x E, strides=[s0, ..., sn]>` is valid if:
   - The number of shape entries equals the number of stride entries.
   - Each `Si` is a positive integer or `?`.
   - Each `si` is a positive integer or `?`.
   - `E` is a valid element type.

5. A partition view type is valid if:
   - It has a valid tile shape (all dimensions positive powers of 2).
   - The underlying type is a valid tensor view type.
   - The `dim_map` maps each tile dimension to a tensor view dimension.

---

## 5.8 Appendix: Type System Quick Reference

### Common Type Patterns

| Pattern | Type | Use Case |
|---|---|---|
| Scalar parameter | `tile<i32>` or `tile<f32>` | Kernel arguments for scalar values |
| Base pointer | `tile<ptr<f32>>` | Pointer to start of a memory buffer |
| Pointer vector | `tile<Nxptr<f32>>` | Gather/scatter N addresses |
| Data vector | `tile<Nxf32>` | N float values in registers |
| Matrix tile | `tile<MxNxf32>` | MxN matrix in registers |
| 1D tensor view | `tensor_view<?xf32, strides=[?]>` | View of a 1D array with dynamic extent |
| 2D tensor view | `tensor_view<?x?xf32, strides=[?, 1]>` | View of a 2D matrix with row-major layout |
| 2D partition | `partition_view<tile=(M,N), tensor_view<?x?xf32, strides=[?, 1]>>` | Partition a 2D matrix into MxN tiles |

### Type Sizes Quick Reference

| Type | Bit Width | Byte Width |
|---|---|---|
| `i1` | 1 (stored as 8) | 1 |
| `i8` | 8 | 1 |
| `i16` | 16 | 2 |
| `i32` | 32 | 4 |
| `i64` | 64 | 8 |
| `f16` | 16 | 2 |
| `f32` | 32 | 4 |
| `f64` | 64 | 8 |
| `bf16` | 16 | 2 |
| `tf32` | 32 | 4 |
| `e4m3` | 8 | 1 |
| `e5m2` | 8 | 1 |
| `ptr<E>` | 64 | 8 |
