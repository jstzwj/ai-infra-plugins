# MLIR Types & Attributes Reference

## Builtin Types

The builtin dialect defines core types available to all MLIR dialects.

### Integer Types

**Syntax:** `i<N>` (signless), `si<N>` (signed), `ui<N>` (unsigned)

Where N is the bit width (1 to infinity).

```
i1       // 1-bit signless integer (boolean)
i8       // 8-bit signless integer
i32      // 32-bit signless integer
i64      // 64-bit signless integer
si8      // 8-bit signed integer
ui8      // 8-bit unsigned integer
```

**API:**
```c++
IntegerType::get(MLIRContext *ctx, unsigned width);
IntegerType::get(MLIRContext *ctx, unsigned width, SignednessSemantics);
unsigned getWidth();
SignednessSemantics getSignedness(); // Signless, Signed, Unsigned
bool isSignless();
bool isSigned();
bool isUnsigned();
bool isSignlessInteger();
bool isSignedInteger();
bool isUnsignedInteger();
```

### Float Types

| Type | Description | Bit Width |
|------|-------------|-----------|
| `f16` | Half precision (IEEE 754) | 16 |
| `bf16` | BFloat16 (brain float) | 16 |
| `f32` | Single precision (IEEE 754) | 32 |
| `f64` | Double precision (IEEE 754) | 64 |
| `f80` | Extended precision (x87) | 80 |
| `f128` | Quadruple precision | 128 |
| `float8_e5m2` | FP8 (E5M2 format) | 8 |
| `float8_e4m3fn` | FP8 (E4M3 format) | 8 |
| `float8_e5m2fnuz` | FP8 (E5M2 FNUZ) | 8 |
| `float8_e4m3fnuz` | FP8 (E4M3 FNUZ) | 8 |
| `float8_e8m0fnu` | FP8 (E8M0 FNU) | 8 |

**API:**
```c++
FloatType::getFloat8E5M2(MLIRContext *ctx);
FloatType::getFloat8E4M3FN(MLIRContext *ctx);
FloatType::getBF16(MLIRContext *ctx);
FloatType::getF16(MLIRContext *ctx);
FloatType::getF32(MLIRContext *ctx);
FloatType::getF64(MLIRContext *ctx);
unsigned getWidth();
const llvm::fltSemantics &getFloatSemantics();
```

### Index Type

**Syntax:** `index`

Target-specific unsigned integer type used for array indices, sizes, and dimensions. Maps to the target pointer size.

```c++
IndexType::get(MLIRContext *ctx);
```

### None Type

**Syntax:** `none`

Represents the absence of a value.

```c++
NoneType::get(MLIRContext *ctx);
```

### Complex Type

**Syntax:** `complex<ELEMENT_TYPE>`

Represents a complex number with the given element type.

```mlir
complex<f32>   // Complex number with f32 real and imaginary parts
complex<f64>
```

```c++
ComplexType::get(Type elementType);
Type getElementType();
```

### Vector Types

#### VectorType

**Syntax:** `vector<DIMS x ELEMENT_TYPE>` or `vector<DIMS x COUNT x ELEMENT_TYPE>`

```mlir
vector<4xf32>              // 1D vector of 4 floats
vector<2x4xf32>            // 2D vector (2 rows, 4 columns)
vector<4x4x4xf32>          // 3D vector
vector<[4]xf32>            // Scalable vector
vector<4x[8]xf32>          // Partially scalable
```

```c++
VectorType::get(ArrayRef<int64_t> shape, Type elementType);
VectorType::get(ArrayRef<int64_t> shape, Type elementType, ArrayRef<bool> scalableDims);
ArrayRef<int64_t> getShape();
Type getElementType();
unsigned getRank();
int64_t getNumElements();
bool isScalable();
```

### Tensor Types

#### RankedTensorType

**Syntax:** `tensor<DIMS x ELEMENT_TYPE>` or `tensor<DIMS x ELEMENT_TYPE, ENCODING>`

```mlir
tensor<4x4xf32>                  // Static 4x4 tensor
tensor<?xf32>                     // Dynamic 1D tensor
tensor<?x?xf32>                   // Dynamic 2D tensor
tensor<4x?xf32>                   // Mixed static/dynamic
tensor<*xf32>                     // Unranked tensor
tensor<4x4xf32, #enc>             // With encoding attribute
```

```c++
RankedTensorType::get(ArrayRef<int64_t> shape, Type elementType);
RankedTensorType::get(ArrayRef<int64_t> shape, Type elementType, Attribute encoding);
ArrayRef<int64_t> getShape();
Type getElementType();
Attribute getEncoding();
bool hasStaticShape();
int64_t getNumElements();
```

#### UnrankedTensorType

**Syntax:** `tensor<*xELEMENT_TYPE>`

```c++
UnrankedTensorType::get(Type elementType);
Type getElementType();
```

### MemRef Types

#### MemRefType

**Syntax:** `memref<DIMS x ELEMENT_TYPE>` or `memref<DIMS x ELEMENT_TYPE, LAYOUT, SPACE>`

```mlir
memref<4x4xf32>                              // Static, default layout
memref<?xf32>                                 // Dynamic dimension
memref<4x?xf32>                               // Mixed
memref<4x4xf32, strided<[4, 1]>>             // Strided layout
memref<4x4xf32, affine_map<(i,j)->(i*4+j)>>  // Affine layout
memref<4x4xf32, #layout, 0>                  // With memory space
memref<4x4xf32, #layout, "global">           // Named memory space
```

```c++
MemRefType::get(ArrayRef<int64_t> shape, Type elementType);
MemRefType::get(ArrayRef<int64_t> shape, Type elementType, MemRefLayoutAttrInterface layout, Attribute memorySpace);
ArrayRef<int64_t> getShape();
Type getElementType();
MemRefLayoutAttrInterface getLayout();
Attribute getMemorySpace();
bool hasStaticShape();
bool getStridesAndOffset(SmallVectorImpl<int64_t> &strides, int64_t &offset);
```

#### UnrankedMemRefType

**Syntax:** `memref<*xELEMENT_TYPE>` or `memref<*xELEMENT_TYPE, SPACE>`

```c++
UnrankedMemRefType::get(Type elementType, Attribute memorySpace);
Type getElementType();
Attribute getMemorySpace();
```

### Tuple Type

**Syntax:** `tuple<TYPE1, TYPE2, ...>`

```mlir
tuple<i32, f32>
tuple<memref<4xf32>, i1>
```

```c++
TupleType::get(MLIRContext *ctx, ArrayRef<Type> elements);
Type getType(unsigned index);
ArrayRef<Type> getTypes();
size_t size();
```

### Function Type

**Syntax:** `(TYPE_LIST) -> TYPE_LIST`

```mlir
(i32, f32) -> i64
() -> i32
(i32) -> ()
(i32, f32) -> (i64, f64)
```

```c++
FunctionType::get(MLIRContext *ctx, TypeRange inputs, TypeRange results);
TypeRange getInputs();
TypeRange getResults();
unsigned getNumInputs();
unsigned getNumResults();
```

### Opaque Type

**Syntax:** `!dialect<body>`

Represents an unregistered dialect type for round-tripping.

## Builtin Attributes

### Integer Attributes

```mlir
42 : i32
-1 : si32
255 : ui8
0x1a : i32          // Hexadecimal
```

```c++
IntegerAttr::get(Type type, int64_t value);
IntegerAttr::get(Type type, const APInt &value);
int64_t getInt();
int64_t getSInt();
uint64_t getUInt();
APInt getValue();
```

### Float Attributes

```mlir
42.0 : f32
1.5 : f64
0.0 : f16
```

```c++
FloatAttr::get(Type type, double value);
FloatAttr::get(Type type, const APFloat &value);
APFloat getValue();
double getValueAsDouble();
```

### String Attributes

```mlir
"hello world"
"foo"
```

```c++
StringAttr::get(MLIRContext *ctx, StringRef bytes);
StringAttr::get(MLIRContext *ctx, StringRef bytes, Type type);
StringRef getValue();
StringRef strref();
```

### Type Attributes

```mlir
i32          // As attribute
memref<4xf32>
```

```c++
TypeAttr::get(Type value);
Type getValue();
```

### Array Attributes

```mlir
[42 : i32, "hello", 3.14 : f32]
```

```c++
ArrayAttr::get(MLIRContext *ctx, ArrayRef<Attribute> value);
Attribute operator[](unsigned index);
size_t size();
iterator begin();
iterator end();
```

### Dictionary Attributes

```mlir
{name = "foo", value = 42 : i32, nested = {x = 1 : i32}}
```

```c++
DictionaryAttr::get(MLIRContext *ctx, ArrayRef<NamedAttribute> value);
Attribute get(StringRef key);
NamedAttribute getNamed(StringRef key);
bool contains(StringRef key);
size_t size();
```

### Dense Elements Attributes

```mlir
// Dense array of integers
dense<[1, 2, 3, 4]> : tensor<4xi32>

// Dense array of floats
dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>

// Splat value
dense<1.0> : tensor<4xf32>

// Dense resource
dense_resource<"resource_handle"> : tensor<4xi32>
```

```c++
DenseElementsAttr::get(ShapedType type, ArrayRef<Attribute> values);
DenseElementsAttr::get(ShapedType type, ArrayRef<int64_t> values);
DenseElementsAttr::get(ShapedType type, ArrayRef<float> values);
DenseElementsAttr::get(ShapedType type, APFloat value);  // splat
bool isSplat();
int64_t getNumElements();
Type getElementType();
```

### Sparse Elements Attributes

```mlir
sparse<[[0, 0], [1, 1]], [1.0, 2.0]> : tensor<2x2xf32>
```

### Symbol Reference Attributes

```mlir
@function_name
@module::@nested_function
@module::@nested::@deep
```

```c++
SymbolRefAttr::get(MLIRContext *ctx, StringRef value);
SymbolRefAttr::get(MLIRContext *ctx, StringRef value, ArrayRef<FlatSymbolRefAttr> nestedRefs);
StringRef getRootReference();
FlatSymbolRefAttr getLeafReference();
```

### Affine Map Attributes

```mlir
#map = affine_map<(d0, d1) -> (d0, d1)>
#map = affine_map<(d0) -> (d0 + 10)>
#map = affine_map<(d0, d1)[s0] -> (d0 * s0 + d1)>
```

### Affine Expression Attributes

```mlir
#expr = affine_expr<d0 * 2 + 1>
```

### Dense Int Or FP Elements Attributes

Specialized dense attributes for integer and float types:

```c++
DenseIntElementsAttr::get(ShapedType type, ArrayRef<int64_t> values);
DenseFPElementsAttr::get(ShapedType type, ArrayRef<APFloat> values);
```

### Unit Attribute

```mlir
unit
```

Represents a valueless attribute (like a flag).

```c++
UnitAttr::get(MLIRContext *ctx);
```

### Location Attributes

- **FileLineColLoc**: `loc("file.mlir":4:12)`
- **FusedLoc**: `loc(fused<"name">[loc0, loc1])`
- **CallSiteLoc**: `loc(callsite(loc("a.mlir":1:0) at loc("b.mlir":5:3)))`
- **UnknownLoc**: `loc(unknown)`

### Distinct Attribute

```mlir
#distinct = distinct[0]<42.0 : f32>
```

Unique identifier for grouping operations sharing a common property.

### Opaque Attribute

```mlir
#dialect<"raw data">
```

For round-tripping unregistered dialect attributes.

## ShapedType Interface

Common interface for types with shape (tensor, memref, vector):

```c++
// Available on TensorType, MemRefType, VectorType
Type getElementType();
ArrayRef<int64_t> getShape();
unsigned getRank();
int64_t getDimSize(unsigned i);
bool hasStaticShape();
bool hasRank();
int64_t getNumElements();
```

## Type Utilities

```c++
// Check type categories
llvm::isa<IntegerType>(type)
llvm::isa<FloatType>(type)
llvm::isa<MemRefType>(type)
llvm::isa<TensorType>(type)
llvm::isa<VectorType>(type)
llvm::isa<FunctionType>(type)
llvm::isa<ShapedType>(type)

// Get element type from shaped type
getElementTypeOrSelf(Type type)
getElementTypeOrSelf(OpResult result)
getElementTypeOrSelf(Value value)
```
