# Chapter 3: Syntax Reference

## 3.1 Overview

Tile IR is intended to be constructed using the Tile IR MLIR dialect and stored as bytecode. A textual representation based on the MLIR dialect is provided for human readability but has **no stability guarantees** and is not intended for writing Tile IR programs directly.

---

## 3.2 Module

A Tile IR program consists of a Tile IR module containing a series of items.

```
symbol_name := `@` identifier

cuda_tile.module @symbol_name {
    <items>*
}
```

### Module Properties
- A module represents a single compilation unit
- Contains zero or more items (globals, functions, kernels)
- Must contain only Tile IR operations (no other dialects)
- The module operation is the top-level operation

---

## 3.3 Items

An item is either a kernel definition or a global variable definition.

```
<items> ::= <kernel_definition> | <global_variable_definition>
```

---

## 3.4 Globals

A global variable definition creates a named variable defined outside of any kernel.

```
global_variable_definition ::= `global` <symbol_name> `:` <type> `=` <value>
```

### Example

```
global @val alignment = 128 <f32: [0.1, 0.2, 0.3, 0.4]> : tile<4xf32>
```

### Global Variable Properties
- Stored in global device memory, accessible to all tile blocks
- Must be initialized upon declaration (initialized exactly once at module load time)
- Must contain a value of Tile Type
- Names are globally unique symbols within the module
- Cannot be defined inside functions
- Mutable: can be read/written via `cuda_tile.get_global` + `cuda_tile.load_ptr_tko`/`cuda_tile.store_ptr_tko`

---

## 3.5 Kernels

A kernel definition is a function defined inside a Tile IR module.

```
ssa_name := `%` identifier

function_signature ::= <function_parameter>*

function_parameter ::= <ssa_name> `:` <type>

<kernel_definition> ::= `entry` @kernel_name `(` <function_signature> `)` {
    <kernel_body>
}
```

A kernel body is a sequence of operations in static-single-assignment (SSA) form:

```
kernel_body ::= <operation>*

operation ::= (ssa_name `,`?)* `=` <operation_name> <ssa_name>* attribute=attribute_value : type ...
```

### Kernel Restrictions
- Can only have parameters with scalar (0-rank) tensor types
- All input tensors must be scalar pointers (`tile<ptr<E>>`)
- Produces no return value
- Executed only for effects on global device memory
- Launched from host using `cuLaunchKernel` or similar CUDA runtime API

### Entry Operation Attributes

The entry operation supports optimization hints:

```
entry @kernel_name(...) optimization_hints = {
    sm_100 = { num_cta_in_cga = 4 },
    sm_120 = { allow_tma = true, latency = 100 }
}
```

Available hints:
| Hint | Description | Operations |
|------|-------------|------------|
| `num_cta_in_cga` | Number of CTAs in CGA (power of 2, <= 16) | `cuda_tile.entry` |
| `allow_tma` | Whether to use TMA | `cuda_tile.load_view_tko`, `cuda_tile.store_view_tko` |
| `latency` | Latency hint | `cuda_tile.load_view_tko`, `cuda_tile.store_view_tko` |

---

## 3.6 Types

### Element Types

```
element_type ::= `f32` | `f64` | `i8` | `i16` | `i32` | `i64`
              | `f16` | `bf16` | `tf32` | `e4m3` | `e5m2`
```

### Tile Type

```
type ::= `tile` `<` shape `x` element_type `>`
       | `tile` `<` element_type `>`           // rank-0 (scalar)

shape ::= integer_literal (`x` integer_literal)*
```

All dimensions must be powers of two.

### Pointer Type

```
ptr_type ::= `ptr` `<` element_type `>`
```

### Tensor View Type

```
tensor_view ::= `tensor_view` `<` shape `x` element_type `,` `strides` `=` `[` stride_list `]` `>`

stride_list ::= integer_literal | `?` (`,` integer_literal | `?`)*
```

Dynamic dimensions/strides are denoted with `?`.

### Partition View Type

```
partition_view ::= `partition_view` `<`
    `tile` `=` `(` tile_shape `)` `,`
    `tensor_view_type` `,`
    `dim_map` `=` `[` int_list `]` `,`
    `padding_value` `=` padding_kind
    `>`
```

---

## 3.7 Operations Syntax

### General Operation Form

```
%result1, %result2 = operation_name operand1, operand2 attribute1=value1 : type1, type2
```

### Common Attribute Syntax

| Attribute | Syntax | Example |
|-----------|--------|---------|
| rounding_mode | `rounding<mode>` | `rounding<nearest_even>` |
| signedness | `signed` or `unsigned` | `addi %a, %b signed` |
| overflow | `overflow` keyword | `no_wrap` |
| memory ordering | `weak` or `relaxed` etc. | `load_ptr_tko weak` |
| memory scope | `device` or `sys` or `tl_blk` | `relaxed device` |
| token | `token=%t` | `load_ptr_tko weak %ptr token=%t` |

### Control Flow Syntax

```
// For loop
%result = for %iv in (%lo to %hi, step %step) : tile<i32>
    iter_values(%acc = %init) -> (tile<type>) {
  // body
  continue %new_val : tile<type>
}

// If-then-else
%x, %y = if %cond -> (tile<f32>, tile<i32>) {
  yield %x_then, %y_then : tile<f32>, tile<i32>
} else {
  yield %x_else, %y_else : tile<f32>, tile<i32>
}

// Loop (while-like)
%result = loop iter_values(%v = %init) : tile<type> -> tile<type> {
  if %cond {
    continue %new_v : tile<type>
  }
  break %v : tile<type>
}
```

### Memory Operation Syntax

```
// Pointer-based load
%val, %token = load_ptr_tko weak %ptrs : tile<128xptr<f32>> -> tile<128xf32>, token

// Pointer-based load with mask and padding
%val, %token = load_ptr_tko weak %ptrs, %mask, %pad : tile<128xptr<f32>>, tile<128xi1>, tile<128xf32> -> tile<128xf32>, token

// View-based load
%tile, %token = load_view_tko weak %partition[%x, %y] :
    partition_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128,1]>>, tile<i32> -> tile<64x64xf32>, token

// Store
store_ptr_tko weak %ptrs, %val : tile<128xptr<f32>>, tile<128xf32> -> token

// View store
%token = store_view_tko weak %val, %partition[%x, %y] :
    tile<64x64xf32>, partition_view<...>, tile<i32> -> token
```

### View Construction Syntax

```
// Tensor view
%view = make_tensor_view %ptr, shape=[%M, %N], strides=[%M, 1] :
    tile<i32> -> tensor_view<?x?xf32, strides=[?,1]>

// Partition view
%partition = make_partition_view %view :
    partition_view<tile=(128x64), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]>
```

### Assume Syntax

```
%ptr_assume = assume #cuda_tile.div_by<16>, %ptr : tile<ptr<f16>>
%stride_assume = assume #cuda_tile.div_by<8>, %stride : tile<i32>
```

---

## 3.8 Naming Conventions

| Item | Syntax | Example |
|------|--------|---------|
| Symbol name | `@identifier` | `@my_kernel` |
| SSA value | `%identifier` | `%result` |
| Attribute name | `#identifier` | `#cuda_tile.div_by<16>` |

---

## 3.9 Type Annotation Convention

Type annotations follow MLIR convention with `:` separator:

```
operation operands : operand_types -> result_types
```

For operations producing multiple results:

```
%a, %b, %c = operation %x : tile<i32> -> (tile<i32>, tile<i32>, token)
```

---

## 3.10 Location Annotations

Operations may carry source location information:

```
%result = operation operands {loc("file.py":10:5)} : types
```

Or with fused scope metadata:

```
%result = operation operands loc(#cuda_tile.di_loc<loc("file.py":10:5) in #scope>) : types
```
