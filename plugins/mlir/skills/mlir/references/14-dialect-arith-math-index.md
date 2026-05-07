# MLIR Arith, Math, Index & Complex Dialects

## Arith Dialect

The arithmetic dialect provides standard integer and floating-point operations.

### Integer Arithmetic

| Operation | Syntax | Description |
|-----------|--------|-------------|
| `arith.addi` | `%r = arith.addi %a, %b : i32` | Integer addition |
| `arith.subi` | `%r = arith.subi %a, %b : i32` | Integer subtraction |
| `arith.muli` | `%r = arith.muli %a, %b : i32` | Integer multiplication |
| `arith.divsi` | `%r = arith.divsi %a, %b : i32` | Signed division |
| `arith.divui` | `%r = arith.divui %a, %b : i32` | Unsigned division |
| `arith.remsi` | `%r = arith.remsi %a, %b : i32` | Signed remainder |
| `arith.remui` | `%r = arith.remui %a, %b : i32` | Unsigned remainder |
| `arith.ceildivsi` | `%r = arith.ceildivsi %a, %b : i32` | Signed ceil division |
| `arith.ceildivui` | `%r = arith.ceildivui %a, %b : i32` | Unsigned ceil division |
| `arith.floordivsi` | `%r = arith.floordivsi %a, %b : i32` | Signed floor division |
| `arith.negi` | `%r = arith.negi %a : i32` | Integer negation |
| `arith.maxsi` | `%r = arith.maxsi %a, %b : i32` | Signed maximum |
| `arith.maxui` | `%r = arith.maxui %a, %b : i32` | Unsigned maximum |
| `arith.minsi` | `%r = arith.minsi %a, %b : i32` | Signed minimum |
| `arith.minui` | `%r = arith.minui %a, %b : i32` | Unsigned minimum |

### Bitwise Operations

| Operation | Syntax | Description |
|-----------|--------|-------------|
| `arith.andi` | `%r = arith.andi %a, %b : i32` | Bitwise AND |
| `arith.ori` | `%r = arith.ori %a, %b : i32` | Bitwise OR |
| `arith.xori` | `%r = arith.xori %a, %b : i32` | Bitwise XOR |
| `arith.shli` | `%r = arith.shli %a, %b : i32` | Shift left |
| `arith.shrsi` | `%r = arith.shrsi %a, %b : i32` | Shift right (signed/arithmetic) |
| `arith.shrui` | `%r = arith.shrui %a, %b : i32` | Shift right (unsigned/logical) |

### Floating-Point Arithmetic

| Operation | Syntax | Description |
|-----------|--------|-------------|
| `arith.addf` | `%r = arith.addf %a, %b : f32` | Float addition |
| `arith.subf` | `%r = arith.subf %a, %b : f32` | Float subtraction |
| `arith.mulf` | `%r = arith.mulf %a, %b : f32` | Float multiplication |
| `arith.divf` | `%r = arith.divf %a, %b : f32` | Float division |
| `arith.remf` | `%r = arith.remf %a, %b : f32` | Float remainder |
| `arith.negf` | `%r = arith.negf %a : f32` | Float negation |
| `arith.maxf` | `%r = arith.maxf %a, %b : f32` | Float maximum (NaN-propagating) |
| `arith.minf` | `%r = arith.minf %a, %b : f32` | Float minimum (NaN-propagating) |
| `arith.maximumf` | `%r = arith.maximumf %a, %b : f32` | Float maximum (NaN-propagating) |
| `arith.minimumf` | `%r = arith.minimumf %a, %b : f32` | Float minimum (NaN-propagating) |
| `arith.copysign` | `%r = arith.copysign %a, %b : f32` | Copy sign |

### Comparison Operations

```mlir
// Integer comparison
%r = arith.cmpi "eq", %a, %b : i32
%r = arith.cmpi "ne", %a, %b : i32
%r = arith.cmpi "slt", %a, %b : i32   // signed less than
%r = arith.cmpi "sle", %a, %b : i32
%r = arith.cmpi "sgt", %a, %b : i32
%r = arith.cmpi "sge", %a, %b : i32
%r = arith.cmpi "ult", %a, %b : i32   // unsigned less than
%r = arith.cmpi "ule", %a, %b : i32
%r = arith.cmpi "ugt", %a, %b : i32
%r = arith.cmpi "uge", %a, %b : i32

// Float comparison
%r = arith.cmpf "oeq", %a, %b : f32   // ordered equals
%r = arith.cmpf "one", %a, %b : f32
%r = arith.cmpf "olt", %a, %b : f32
%r = arith.cmpf "ole", %a, %b : f32
%r = arith.cmpf "ogt", %a, %b : f32
%r = arith.cmpf "oge", %a, %b : f32
%r = arith.cmpf "ueq", %a, %b : f32   // unordered equals
%r = arith.cmpf "une", %a, %b : f32
%r = arith.cmpf "ult", %a, %b : f32
%r = arith.cmpf "ule", %a, %b : f32
%r = arith.cmpf "ugt", %a, %b : f32
%r = arith.cmpf "uge", %a, %b : f32
%r = arith.cmpf "ord", %a, %b : f32   // ordered (neither NaN)
%r = arith.cmpf "uno", %a, %b : f32   // unordered (either NaN)
%r = arith.cmpf "true", %a, %b : f32  // always true
%r = arith.cmpf "false", %a, %b : f32 // always false
```

### Type Conversion

```mlir
// Integer to integer
%r = arith.trunci %a : i64 to i32
%r = arith.extsi %a : i32 to i64      // signed extend
%r = arith.extui %a : i32 to i64      // unsigned extend

// Float to float
%r = arith.truncf %a : f64 to f32
%r = arith.extf %a : f32 to f64

// Integer to float
%r = arith.sitofp %a : i32 to f32     // signed int to float
%r = arith.uitofp %a : i32 to f32     // unsigned int to float

// Float to integer
%r = arith.fptosi %a : f32 to i32     // float to signed int
%r = arith.fptoui %a : f32 to i32     // float to unsigned int

// Index conversion
%r = arith.index_cast %a : index to i32
%r = arith.index_cast %a : i32 to index
%r = arith.index_castui %a : index to i32
%r = arith.index_castui %a : i32 to index

// Bitwise cast (preserves bit pattern)
%r = arith.bitcast %a : i32 to f32

// Select
%r = arith.select %cond, %true_val, %false_val : i32
```

### Constants

```mlir
%zero = arith.constant 0 : i32
%one = arith.constant 1 : i64
%float = arith.constant 3.14 : f32
%tensor = arith.constant dense<[1, 2, 3]> : tensor<3xi32>
%splat = arith.constant dense<1.0> : tensor<4xf32>
%bool = arith.constant true
```

## Math Dialect

Mathematical operations on floating-point types:

| Operation | Syntax | Description |
|-----------|--------|-------------|
| `math.absf` | `%r = math.absf %a : f32` | Absolute value |
| `math.ceil` | `%r = math.ceil %a : f32` | Ceiling |
| `math.floor` | `%r = math.floor %a : f32` | Floor |
| `math.round` | `%r = math.round %a : f32` | Round to nearest |
| `math.roundeven` | `%r = math.roundeven %a : f32` | Round to nearest even |
| `math.cos` | `%r = math.cos %a : f32` | Cosine |
| `math.sin` | `%r = math.sin %a : f32` | Sine |
| `math.tan` | `%r = math.tan %a : f32` | Tangent |
| `math.tanh` | `%r = math.tanh %a : f32` | Hyperbolic tangent |
| `math.cosh` | `%r = math.cosh %a : f32` | Hyperbolic cosine |
| `math.sinh` | `%r = math.sinh %a : f32` | Hyperbolic sine |
| `math.atan` | `%r = math.atan %a : f32` | Arc tangent |
| `math.atan2` | `%r = math.atan2 %a, %b : f32` | Two-argument arc tangent |
| `math.acos` | `%r = math.acos %a : f32` | Arc cosine |
| `math.asin` | `%r = math.asin %a : f32` | Arc sine |
| `math.sqrt` | `%r = math.sqrt %a : f32` | Square root |
| `math.rsqrt` | `%r = math.rsqrt %a : f32` | Reciprocal square root |
| `math.exp` | `%r = math.exp %a : f32` | Exponential |
| `math.exp2` | `%r = math.exp2 %a : f32` | Base-2 exponential |
| `math.expm1` | `%r = math.expm1 %a : f32` | exp(x) - 1 |
| `math.log` | `%r = math.log %a : f32` | Natural logarithm |
| `math.log2` | `%r = math.log2 %a : f32` | Base-2 logarithm |
| `math.log10` | `%r = math.log10 %a : f32` | Base-10 logarithm |
| `math.log1p` | `%r = math.log1p %a : f32` | log(1 + x) |
| `math.powf` | `%r = math.powf %a, %b : f32` | Power |
| `math.erf` | `%r = math.erf %a : f32` | Error function |
| `math.cbrt` | `%r = math.cbrt %a : f32` | Cube root |
| `math.isfinite` | `%r = math.isfinite %a : f32` | Is finite |
| `math.isinf` | `%r = math.isinf %a : f32` | Is infinity |
| `math.isnan` | `%r = math.isnan %a : f32` | Is NaN |
| `math.copysign` | `%r = math.copysign %a, %b : f32` | Copy sign |
| `math.fma` | `%r = math.fma %a, %b, %c : f32` | Fused multiply-add |
| `math.ctlz` | `%r = math.ctlz %a : i32` | Count leading zeros |
| `math.cttz` | `%r = math.cttz %a : i32` | Count trailing zeros |
| `math.ctpop` | `%r = math.ctpop %a : i32` | Population count |

### Integer Math Operations

| Operation | Syntax | Description |
|-----------|--------|-------------|
| `math.absi` | `%r = math.absi %a : i32` | Integer absolute value |
| `math.ctlz` | `%r = math.ctlz %a : i32` | Count leading zeros |
| `math.cttz` | `%r = math.cttz %a : i32` | Count trailing zeros |
| `math.ctpop` | `%r = math.ctpop %a : i32` | Population count (bit count) |

## Index Dialect

Platform-independent index computations:

```mlir
// Sizeof
%s = index.sizeof

// Constants
%c = index.constant 42

// Arithmetic
%sum = index.add %a, %b
%dif = index.sub %a, %b
%prod = index.mul %a, %b
%quot = index.divu %a, %b      // unsigned division
%rem = index.remu %a, %b       // unsigned remainder

// Floor division and ceiling division
%fd = index.floordiv %a, %b
%cd = index.ceildiv %a, %b

// Comparison
%r = index.cmp "eq", %a, %b
%r = index.cmp "ne", %a, %b
%r = index.cmp "slt", %a, %b
%r = index.cmp "sle", %a, %b
%r = index.cmp "sgt", %a, %b
%r = index.cmp "sge", %a, %b
%r = index.cmp "ult", %a, %b
%r = index.cmp "ule", %a, %b
%r = index.cmp "ugt", %a, %b
%r = index.cmp "uge", %a, %b

// Cast
%i = index.castu %a : index to i64
%x = index.casts %a : index to i64
```

## Complex Dialect

Operations on complex numbers:

| Operation | Syntax | Description |
|-----------|--------|-------------|
| `complex.create` | `%r = complex.create %re, %im : complex<f32>` | Create complex number |
| `complex.re` | `%r = complex.re %c : complex<f32>` | Real part |
| `complex.im` | `%r = complex.im %c : complex<f32>` | Imaginary part |
| `complex.add` | `%r = complex.add %a, %b : complex<f32>` | Addition |
| `complex.sub` | `%r = complex.sub %a, %b : complex<f32>` | Subtraction |
| `complex.mul` | `%r = complex.mul %a, %b : complex<f32>` | Multiplication |
| `complex.div` | `%r = complex.div %a, %b : complex<f32>` | Division |
| `complex.abs` | `%r = complex.abs %a : complex<f32>` | Absolute value |
| `complex.sqrt` | `%r = complex.sqrt %a : complex<f32>` | Square root |
| `complex.conj` | `%r = complex.conj %a : complex<f32>` | Conjugate |
| `complex.angle` | `%r = complex.angle %a : complex<f32>` | Phase angle |
| `complex.cos` | `%r = complex.cos %a : complex<f32>` | Cosine |
| `complex.sin` | `%r = complex.sin %a : complex<f32>` | Sine |
| `complex.tanh` | `%r = complex.tanh %a : complex<f32>` | Hyperbolic tangent |
| `complex.exp` | `%r = complex.exp %a : complex<f32>` | Exponential |
| `complex.log` | `%r = complex.log %a : complex<f32>` | Logarithm |
| `complex.pow` | `%r = complex.pow %a, %b : complex<f32>` | Power |
| `complex.rsqrt` | `%r = complex.rsqrt %a : complex<f32>` | Reciprocal sqrt |
| `complex.neg` | `%r = complex.neg %a : complex<f32>` | Negation |
| `complex.constant` | `%r = complex.constant [1.0, 0.0] : complex<f32>` | Constant |
