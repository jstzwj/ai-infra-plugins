# Chapter 10: Stability and Compatibility

## Table of Contents

1. [Definitions](#1-definitions)
2. [Platform and Compatibility Guarantees](#2-platform-and-compatibility-guarantees)
3. [Supported Architectures](#3-supported-architectures)
4. [Feature Availability and Emulation](#4-feature-availability-and-emulation)
5. [Hardware Support Matrix](#5-hardware-support-matrix)
6. [Emulation of Unsupported Operations](#6-emulation-of-unsupported-operations)
7. [Execution and Numerical Guarantees](#7-execution-and-numerical-guarantees)
8. [Release Notes 13.2](#8-release-notes-132)
9. [Known Issues](#9-known-issues)

---

## 1. Definitions

The following terms are used throughout this chapter with precise meanings. Understanding
these definitions is critical for interpreting the guarantees that Tile IR provides.

### Stability

**Stability** refers to the property that a Tile IR program, once compiled to bytecode,
produces the same observable results across different versions of the Tile IR toolchain
(compiler, runtime, driver) and across different conforming implementations. There are
several categories of stability:

| Stability Category | Scope | Guarantee |
|--------------------|-------|-----------|
| **Bytecode stability** | Binary format | A bytecode file produced by version N can be loaded and executed by any conforming driver version >= N |
| **Semantic stability** | Program behavior | A conforming program produces the same result for the same input on the same hardware with the same toolchain version |
| **API stability** | Compiler interface | The textual and binary representations of Tile IR are backward-compatible within a major version |

### Portability

**Portability** refers to the ability of a Tile IR program to execute correctly across
different hardware architectures and GPU compute capabilities. A program is **portable**
if it can be loaded, compiled, and executed on any conforming target without
modification.

Tile IR distinguishes two levels of portability:

- **Full portability**: The program uses only operations and types that are supported
  natively on all target architectures. No emulation is required.

- **Portable with emulation**: The program may use operations not natively supported
  on all architectures, but the compiler can emulate them using available primitives.
  The program remains functionally correct but may have reduced performance.

### Compatibility

**Compatibility** refers to the relationship between Tile IR and the broader CUDA
platform, including CUDA driver versions, CUDA runtime versions, and GPU driver
support. Tile IR inherits CUDA's compatibility model and extends it with Tile IR-
specific guarantees.

### Toolchain

The **toolchain** encompasses all software components involved in translating a Tile IR
program from source (or bytecode) to execution on a GPU:

| Component | Description |
|-----------|-------------|
| **Tile IR Compiler** | Translates Tile IR bytecode to PTX or native GPU code. Available as part of the CUDA driver (JIT) and as a standalone tool. |
| **CUDA Driver** | The GPU driver that manages device memory, kernel launches, and the JIT compilation pipeline. |
| **CUDA Runtime** | The user-space runtime library (libcudart) that provides host-side API for launching kernels. |
| **PTX Compiler** | The PTX-to-SASS compiler (part of the CUDA driver) that produces native GPU instructions. |
| **GPU Hardware** | The physical GPU device with a specific compute capability (e.g., sm_80, sm_89). |

---

## 2. Platform and Compatibility Guarantees

Tile IR provides the following platform and compatibility guarantees for conforming
programs.

### 2.1 Bytecode Stability

**Guarantee**: A Tile IR bytecode file produced by a conforming toolchain version N can
be loaded, validated, and executed by any conforming toolchain version M where M >= N
within the same major version.

**Implications**:

- Bytecode files are **forward-compatible**: newer drivers can always load older
  bytecode.

- Bytecode files are **NOT backward-compatible**: an older driver may not be able to
  load bytecode produced by a newer toolchain if the bytecode uses features introduced
  after the older driver was released.

- The bytecode format includes a version header (`major.minor.tag`) that the loader
  checks before attempting to parse. If the version is newer than the loader supports,
  it rejects the file with a descriptive error.

**Bytecode Versioning**:

```
bytecode_header {
    magic:   "\x7FTileIR\x00"    // 8 bytes, fixed
    version: {
        major: u16,               // Breaking changes only
        minor: u16,               // New features, backward compatible
        tag:   u16                // Patch/bugfix level
    }
}
```

- **Major version change**: Indicates breaking changes. Old bytecode may not load.
- **Minor version change**: New opcodes, types, or sections added. Old bytecode
  continues to work.
- **Tag change**: Bug fixes only. No format changes.

### 2.2 Program Portability

**Guarantee**: A conforming Tile IR program is **syntactically portable**: it can be
parsed, validated, and loaded by any conforming Tile IR implementation regardless of
target architecture.

**Portable Execution**:

Whether a program can also be **executed** on a given target depends on the operations
and types it uses:

1. If all operations and types in the program are natively supported on the target
   architecture, the program executes natively at full performance.

2. If some operations or types are not natively supported, the Tile IR compiler
   attempts to **emulate** them using available primitives. See Section 4 for details.

3. If emulation is not possible (e.g., an atomic operation on an unsupported data
   type with no software fallback), the compiler rejects the program with a
   diagnostic error.

**Compiler Fallback Behavior**:

```
Program --> [Load & Validate] --> [Select Target] --> [Check Features]
                                                         |
                                       +------------------+------------------+
                                       |                  |                  |
                                  [Native]           [Emulate]          [Reject]
                                  (fastest)     (correct, slower)    (error message)
```

### 2.3 CUDA Compatibility

Tile IR respects CUDA's forward and backward compatibility model:

- **CUDA Forward Compatibility**: A Tile IR program compiled for compute capability
  X can run on a GPU with compute capability Y where Y >= X, provided Y is in the
  same architecture family or a later family.

- **CUDA Minor Version Compatibility**: Within a major CUDA version, Tile IR programs
  maintain compatibility. A program compiled with CUDA 13.x can run on any CUDA 13.y
  driver where y >= x.

- **CUDA Enhanced Compatibility**: When the CUDA enhanced compatibility driver is
  installed, Tile IR programs can target newer GPU architectures on older CUDA driver
  versions, as long as the driver supports the minimum required CUDA version for that
  architecture.

**Important Note**: Tile IR version 13.2 requires CUDA 13.0 or later. Programs that
use features introduced in 13.2 require CUDA 13.2 or later on the target system.

---

## 3. Supported Architectures

Tile IR 13.2 supports the following GPU architectures and compute capabilities:

### Architecture Support Table

| Architecture Family | Compute Capability | Example GPUs | Supported Since | Native Support Level |
|---------------------|--------------------|---------------|-----------------|---------------------|
| **Ampere** | `sm_80` | NVIDIA A100, A30 | 13.2 | Full |
| **Ampere** | `sm_86` | NVIDIA A40, RTX 3090, RTX 3080 | 13.2 | Full |
| **Ampere** | `sm_87` | NVIDIA A10, A16 | 13.2 | Full |
| **Ampere** | `sm_88` | NVIDIA A2 | 13.2 | Full |
| **Ada Lovelace** | `sm_89` | NVIDIA L40, L40S, RTX 4090, RTX 4080 | 13.2 | Full |
| **Blackwell** | `sm_100` | NVIDIA B200, B100 | 13.1 | Full |
| **Blackwell** | `sm_120` | NVIDIA RTX 5090, RTX 5080 | 13.1 | Full |

### Architecture NOT Supported in 13.2

| Architecture Family | Compute Capability | Example GPUs | Status |
|---------------------|--------------------|---------------|--------|
| **Hopper** | `sm_90` | NVIDIA H100, H200 | **Not supported in 13.2** |

**Note on Hopper (sm_90)**: Hopper architecture GPUs are not supported in Tile IR 13.2.
Programs targeting `sm_90` will be rejected by the compiler. Support for Hopper is
planned for a future release. Users with H100/H200 hardware should use PTX or CUDA C++
for GPU programming until Tile IR support is available.

### Compute Capability Selection

The Tile IR compiler accepts a target compute capability via the `-arch` flag or
equivalent API:

```
# Compile for specific architecture
tileir-compile -arch=sm_89 -o kernel.tileir kernel.tile

# Compile for multiple architectures (fat binary)
tileir-compile -arch=sm_80,sm_86,sm_89 -o kernel.tileir kernel.tile
```

When compiling a fat binary, the compiler includes optimized code for each specified
architecture. At runtime, the driver selects the best-matching version for the
installed GPU.

---

## 4. Feature Availability and Emulation

### 4.1 Target-Specific Features

Some Tile IR operations and types have hardware-specific behavior or availability.
The following table summarizes which features are available on which architectures:

| Feature | sm_80 | sm_86 | sm_87 | sm_88 | sm_89 | sm_100 | sm_120 |
|---------|-------|-------|-------|-------|-------|--------|--------|
| `mmaf` f16xf16->f32 | Native | Native | Native | Native | Native | Native | Native |
| `mmaf` bf16xbf16->f32 | Native | Native | Native | Native | Native | Native | Native |
| `mmaf` tf32xtf32->f32 | Native | Native | Native | Native | Native | Native | Native |
| `mmaf` e4m3xe4m3->f32 | -- | -- | -- | -- | Native | Native | Native |
| `mmaf` e5m2xe5m2->f32 | -- | -- | -- | -- | Native | Native | Native |
| `mmaf` e4m3xe5m2->f32 | -- | -- | -- | -- | Native | Native | Native |
| `mmai` s8xs8->s32 | Native | Native | Native | Native | Native | Native | Native |
| `mmai` u8xu8->s32 | Native | Native | Native | Native | Native | Native | Native |
| `atomic_rmw_tko` f16 | Native | Native | Native | Native | Native | Native | Native |
| `atomic_rmw_tko` bf16 | -- | -- | -- | -- | Native | Native | Native |
| `atan2` f32/f64 | Native | Native | Native | Native | Native | Native | Native |
| `unsignedCmp` | Emulated | Emulated | Emulated | Emulated | Emulated | Emulated | Emulated |

### 4.2 Fallback Behavior

When a program uses a feature that is not natively supported on the target
architecture, the Tile IR compiler applies one of the following fallback strategies:

| Strategy | Description | Performance Impact |
|----------|-------------|-------------------|
| **Emulation via software** | The operation is implemented using multiple native operations | Moderate to severe |
| **Type promotion** | Operands are promoted to a supported type, computed, then demoted | Moderate |
| **Rejection** | No feasible emulation exists; compilation fails | N/A |

**Example: FP8 MMA on Ampere**

```
// Original: uses e4m3 MMA (not supported on sm_80)
%result = mmaf %A, %B, %C : tile<128x64xe4m3>, tile<64x128xe4m3>, tile<128x128xf32>

// Compiler emulation on sm_80:
// 1. Promote e4m3 -> f16 (with precision loss)
// 2. Execute mmaf with f16
// 3. Result is tile<128x128xf32> (same as original)
```

### 4.3 Warning About 13.x "Cold Start" Period

Tile IR 13.x is the first major release of the Tile IR specification and toolchain.
During this initial release period (the "cold start"), users should be aware of the
following:

1. **Performance may not be optimal**: The compiler may not yet have all architecture-
   specific optimizations for every supported GPU. Performance will improve in
   subsequent minor releases.

2. **Emulation paths may be less tested**: Fallback code paths for features not
   natively supported on older architectures may have edge cases. Report any
   discrepancies between emulated and native results.

3. **API surface may evolve**: While bytecode stability is guaranteed within 13.x,
   the compiler command-line interface, error messages, and diagnostic output may
   change between minor releases.

4. **Documentation may lag**: Some edge cases in the interaction between Tile IR
   operations and specific GPU behaviors may not yet be fully documented.

---

## 5. Hardware Support Matrix

### 5.1 Data Types per Architecture

The following matrix shows which element types are natively supported for arithmetic
operations on each architecture:

| Element Type | Size | sm_80 | sm_86 | sm_89 | sm_100 | sm_120 |
|-------------|------|-------|-------|-------|--------|--------|
| `i1` | 1 bit | Yes | Yes | Yes | Yes | Yes |
| `i8` | 8 bits | Yes | Yes | Yes | Yes | Yes |
| `i16` | 16 bits | Yes | Yes | Yes | Yes | Yes |
| `i32` | 32 bits | Yes | Yes | Yes | Yes | Yes |
| `i64` | 64 bits | Yes | Yes | Yes | Yes | Yes |
| `f16` | 16 bits | Yes | Yes | Yes | Yes | Yes |
| `f32` | 32 bits | Yes | Yes | Yes | Yes | Yes |
| `f64` | 64 bits | Yes | Yes | Yes | Yes | Yes |
| `bf16` | 16 bits | Yes | Yes | Yes | Yes | Yes |
| `tf32` | 19 bits | Yes | Yes | Yes | Yes | Yes |
| `e4m3` | 8 bits | Emulated | Emulated | Yes | Yes | Yes |
| `e5m2` | 8 bits | Emulated | Emulated | Yes | Yes | Yes |

### 5.2 Memory Operation Support

| Memory Feature | sm_80 | sm_86 | sm_89 | sm_100 | sm_120 |
|---------------|-------|-------|-------|--------|--------|
| `load_ptr_tko` weak | Yes | Yes | Yes | Yes | Yes |
| `load_ptr_tko` relaxed | Yes | Yes | Yes | Yes | Yes |
| `load_ptr_tko` release/acquire | Yes | Yes | Yes | Yes | Yes |
| `load_ptr_tko` acq_rel | Yes | Yes | Yes | Yes | Yes |
| `store_ptr_tko` weak | Yes | Yes | Yes | Yes | Yes |
| `store_ptr_tko` relaxed | Yes | Yes | Yes | Yes | Yes |
| `store_ptr_tko` release | Yes | Yes | Yes | Yes | Yes |
| `atomic_cas_tko` | Yes | Yes | Yes | Yes | Yes |
| `atomic_rmw_tko` | Yes | Yes | Yes | Yes | Yes |
| Tensor view (dynamic shape) | Yes | Yes | Yes | Yes | Yes |
| Partition view (tiling) | Yes | Yes | Yes | Yes | Yes |

### 5.3 Maximum Tile Dimensions

The maximum tile dimensions supported by the hardware tensor cores vary by
architecture. The Tile IR compiler validates tile dimensions at compile time.

| MMA Tile Configuration | sm_80 | sm_86 | sm_89 | sm_100 | sm_120 |
|----------------------|-------|-------|-------|--------|--------|
| 16x8x16 (f16) | Yes | Yes | Yes | Yes | Yes |
| 16x8x8 (tf32) | Yes | Yes | Yes | Yes | Yes |
| 16x8x32 (e4m3) | -- | -- | Yes | Yes | Yes |
| 16x8x32 (e5m2) | -- | -- | Yes | Yes | Yes |
| 8x8x16 (bf16) | Yes | Yes | Yes | Yes | Yes |
| 128x128x64 (f16, batched) | -- | -- | -- | Yes | Yes |

---

## 6. Emulation of Unsupported Operations

When the Tile IR compiler encounters an operation that is not natively supported on
the target architecture, it applies an emulation strategy. This section documents the
specific emulation approaches used for common cases.

### 6.1 FP8 Emulation on Pre-Ada Architectures

FP8 types (`e4m3`, `e5m2`) are not supported in hardware before Ada (sm_89). The
compiler emulates them as follows:

**Conversion**: FP8 values are promoted to `f16` or `f32` for computation, then
converted back to FP8 for storage.

**MMA operations**: `mmaf` with FP8 inputs is emulated by:
1. Converting both input tiles from FP8 to f16 (with rounding)
2. Executing the f16 MMA operation
3. The accumulator remains in f32 (same as the native operation)

**Precision note**: FP8 emulation via f16 promotion loses the quantization behavior
of the original FP8 format. Results from emulated execution will differ slightly from
native FP8 execution on Ada/Blackwell GPUs.

### 6.2 Unsigned Comparison Emulation

The `unsignedCmp` modifier on the `cmpi` operation is emulated on all architectures
by:

1. XORing the sign bit of both operands with `1` (flipping the sign bit)
2. Performing a signed comparison on the modified values
3. The result is the correct unsigned comparison

This emulation works because flipping the sign bit transforms unsigned ordering into
signed ordering for two's-complement integers.

### 6.3 Atomic Operation Emulation

Most atomic operations are natively supported. When an atomic operation on an
unsupported type is encountered (e.g., `atomic_rmw_tko` on `bf16` for sm_80), the
compiler emulates it using `atomic_cas_tko` in a compare-and-swap loop:

```
// Emulated atomic_rmw_tko add on bf16:
loop {
    %old = load_ptr_tko acq_rel %ptr : ... -> tile<bf16>, token
    %old_f32 = ftof %old : tile<bf16> -> tile<f32>
    %new_f32 = addf %old_f32, %val_f32 rounding<nearest_even> : tile<f32>
    %new = ftof %new_f32 : tile<f32> -> tile<bf16>
    %success, %expected = atomic_cas_tko acq_rel %ptr, %old, %new : ...
    %done = cmpi %success, %one cmp<eq> : tile<i32>
    breakif %done : tile<i1>
}
```

### 6.4 Emulation Performance Impact

| Emulated Feature | Approximate Slowdown vs Native |
|-----------------|-------------------------------|
| FP8 MMA -> f16 MMA | 2-3x (due to conversion overhead) |
| unsignedCmp -> signed + XOR | 1.1x (nearly free) |
| bf16 atomic -> CAS loop | 5-20x (depends on contention) |
| e4m3 arithmetic -> f32 arithmetic | 2-4x (per-element conversion) |

---

## 7. Execution and Numerical Guarantees

### 7.1 Execution Determinism

**Guarantee**: A Tile IR program produces deterministic results when all of the
following are held constant:

1. **Toolchain version**: The same Tile IR compiler version (major.minor.tag)
2. **Compiler configuration**: The same optimization flags and target architecture
3. **Hardware**: The same GPU model and driver version
4. **Input**: The same input data and parameters

**Non-determinism sources**: The following factors can cause non-deterministic results:

| Factor | Description | Affected Operations |
|--------|-------------|-------------------|
| **Tile block scheduling** | Order of tile block execution is unspecified | Operations that depend on inter-block communication via global memory |
| **Floating-point reduction order** | The order of floating-point additions in reductions is unspecified | `reduce` with `addf` accumulator, `scan` with `addf` |
| **Atomic operations** | The order of concurrent atomic updates is unspecified | `atomic_rmw_tko`, `atomic_cas_tko` with concurrent access |
| **Floating-point rounding** | Different hardware may apply different internal rounding | `mmaf` with mixed precision |

**Ensuring determinism**:

- For reductions: use the same reduction algorithm and ensure tile dimensions are
  consistent across runs.

- For atomics: use only a single tile block, or use atomic operations only where the
  order does not affect the final result (e.g., computing a maximum).

- For cross-block communication: use explicit synchronization via host code between
  kernel launches.

### 7.2 Numerical Stability

**Guarantee**: Tile IR does NOT guarantee bit-identical results across different
compiler versions, optimization levels, or hardware architectures.

**Reasons for numerical variation**:

1. **FMA contraction**: The compiler may fuse a multiply and add into a fused
   multiply-add (FMA), which produces a different (more accurate) result than separate
   multiply and add operations.

2. **Reduction reassociation**: The compiler may reorder floating-point reductions,
   changing the accumulation order and producing slightly different results.

3. **Operation substitution**: The compiler may replace one operation with a
   mathematically equivalent but numerically different sequence (e.g., replacing
   `x * 0.5` with `x / 2.0`).

4. **Tensor core precision**: `mmaf` operations on tensor cores use reduced-precision
   internal arithmetic. Different architectures may produce slightly different results
   for the same MMA operation.

5. **Rounding mode interactions**: Different sequences of rounding operations (even
   with the same rounding mode) can produce different final results.

### 7.3 Floating-Point Semantics

Tile IR follows IEEE 754 floating-point semantics with the following clarifications:

**Default rounding mode**: All floating-point operations in Tile IR require an explicit
`rounding<mode>` attribute. The supported modes are:

| Mode | Description |
|------|-------------|
| `nearest_even` | Round to nearest, ties to even (IEEE 754 default) |
| `toward_zero` | Round toward zero (truncation) |
| `toward_positive` | Round toward positive infinity |
| `toward_negative` | Round toward negative infinity |

**Compiler transformations**: The Tile IR compiler is permitted to perform the
following floating-point transformations that are NOT value-preserving:

| Transformation | Example | Condition |
|---------------|---------|-----------|
| FMA contraction | `a * b + c -> fma(a, b, c)` | Always permitted |
| Reassociation | `(a + b) + c -> a + (b + c)` | Permitted for reductions |
| Reciprocal optimization | `a / b -> a * (1/b)` | Only when `rounding<nearest_even>` |
| Division by constant | `a / 2.0 -> a * 0.5` | Permitted for exact powers of 2 |

**NaN and Infinity**: Tile IR follows IEEE 754 rules for NaN and Infinity propagation:
- Operations involving NaN produce NaN (signaling NaNs are quieted)
- Division by zero produces Infinity (not an exception)
- Comparisons with NaN follow the IEEE 754 totalOrdering rules

**Subnormal (denormal) numbers**: Tile IR preserves subnormal numbers in all
operations. The compiler does NOT flush subnormals to zero unless explicitly requested
via a compiler flag.

**Note on `tanh` rounding**: As of Tile IR 13.2, the `tanh` operation supports the
`rounding_mode` attribute. Previously, `tanh` was specified with undefined rounding.
See Release Notes for details.

---

## 8. Release Notes 13.2

This section documents the changes introduced in Tile IR version 13.2.

### 8.1 New Architectures

Tile IR 13.2 adds support for the following architectures (previously supported in
13.1 for Blackwell, newly added in 13.2 for Ampere and Ada):

| Architecture | Compute Capability | Status |
|-------------|--------------------|--------|
| Ampere | sm_80, sm_86, sm_87, sm_88 | **New in 13.2** |
| Ada Lovelace | sm_89 | **New in 13.2** |
| Blackwell | sm_100 | Supported since 13.1 |
| Blackwell | sm_120 | Supported since 13.1 |

### 8.2 New Operations

#### `atan2`

```
%result = atan2 %y, %x rounding<nearest_even> : tile<Nxf32>
```

Computes the arc tangent of `y/x` using the signs of both arguments to determine the
quadrant. Supported for `f32` and `f64` element types.

- **Opcode**: Assigned in 13.2
- **Behavior**: Follows IEEE 754 semantics for `atan2`
- **Supported architectures**: All (sm_80 and above)
- **Rounding**: Supports all rounding modes

### 8.3 Updated Operations

#### `negi` Overflow Behavior Change

The `negi` operation (integer negation) now has defined overflow behavior:

```
// Before 13.2: negi of INT_MIN was undefined behavior
// In 13.2: negi of INT_MIN wraps around (produces INT_MIN for two's complement)
%result = negi %val : tile<i32>
```

- **Change**: `negi` of the minimum signed integer value (e.g., `INT_MIN` for `i32`)
  now produces the minimum value itself (i.e., `negi(-2147483648) = -2147483648` for
  `i32`). This is consistent with two's-complement wrap-around semantics.
- **Impact**: Programs that relied on undefined behavior for `negi` of `INT_MIN` may
  produce different results. This change makes the behavior deterministic and portable.

#### `tanh` `rounding_mode` Attribute

The `tanh` operation now supports an explicit `rounding_mode` attribute:

```
// New syntax (13.2):
%result = tanh %val rounding<nearest_even> : tile<Nxf32>

// Old syntax (pre-13.2, still accepted but deprecated):
%result = tanh %val : tile<Nxf32>
```

- **Change**: `tanh` now accepts the standard `rounding<mode>` attribute. The old
  syntax (without rounding) is still accepted and defaults to `nearest_even`.
- **Impact**: Existing programs continue to work without modification. New programs
  should specify the rounding mode explicitly.

#### `print_tko` Token Output

The `print` operation (used for debug output) now produces a token:

```
// New syntax (13.2):
%tok = print_tko "value = %f\n", %val : tile<Nxf32> -> token

// Old syntax (pre-13.2):
print "value = %f\n", %val : tile<Nxf32>
```

- **Change**: `print_tko` now returns a token value, allowing it to be ordered with
  respect to memory operations. The old `print` syntax without token return is
  deprecated but still accepted.
- **Impact**: Programs using `print` in token chains should switch to `print_tko`.
  Programs using standalone `print` continue to work but should be migrated.

#### `cmpi` `unsignedCmp` Modifier

The `cmpi` operation now supports an `unsignedCmp` modifier for unsigned integer
comparison:

```
// Signed comparison (default):
%result = cmpi %a, %b cmp<gt> : tile<i32>

// Unsigned comparison (new in 13.2):
%result = cmpi %a, %b cmp<gt> unsignedCmp : tile<i32>
```

- **Change**: `cmpi` now accepts an `unsignedCmp` modifier that treats operands as
  unsigned integers for comparison purposes. Without the modifier, comparison is
  signed.
- **Impact**: This is a new feature; existing programs are unaffected.
- **Emulation**: On all currently supported architectures, `unsignedCmp` is emulated
  via sign-bit flipping and signed comparison (see Section 6.2).

### 8.4 Other Changes

- **Bytecode version**: Updated to 13.2.0 (major=13, minor=2, tag=0)
- **String section deduplication**: The string section now deduplicates identical
  strings, reducing bytecode size for programs with repeated string constants.
- **Improved error messages**: The validator now provides more descriptive error
  messages for common mistakes, including the specific operation and location
  information when available.

---

## 9. Known Issues

The following known issues exist in Tile IR version 13.2:

### Issue 1: Incorrect `scan` Results for Large Tiles

**Description**: The `scan` operation may produce incorrect results when applied to
tiles with dimensions larger than 1024 elements in the scan dimension, specifically
when using the `addf` accumulator with `f32` element type on Ampere (sm_80, sm_86)
architectures.

**Workaround**: Limit scan tile dimensions to 1024 or fewer elements. For larger
scans, split the operation into multiple passes.

**Affected architectures**: sm_80, sm_86, sm_87, sm_88
**Affected operations**: `scan` with `addf` accumulator, `f32` element type,
dimension > 1024

### Issue 2: Token Ordering Not Enforced for `store_view_tko` After `load_view_tko`

**Description**: In certain optimization configurations, the compiler may reorder a
`store_view_tko` that writes to the same partition view location as a preceding
`load_view_tko`, even when the two operations are connected by a token chain. This
violates the token ordering guarantee.

**Workaround**: Insert a `make_token` barrier between the load and store to force
ordering.

```
// Workaround:
%val, %t1 = load_view_tko weak %pv[%x, %y] : ... -> tile<f32>, token
%barrier = make_token : token                            // Force barrier
%t2 = store_view_tko weak %val, %pv[%x, %y] : ... -> token
%t3 = join_tokens %t1, %barrier, %t2 : token, token, token -> token
```

**Affected architectures**: All
**Affected configurations**: `-O2` and `-O3` optimization levels

### Issue 3: `mmai` with Mixed Sign Operands on Blackwell sm_120

**Description**: The `mmai` (integer matrix multiply-accumulate) operation produces
incorrect results on Blackwell sm_120 when the two input tiles have different sign
types (e.g., one `s8` and one `u8`). This occurs only when the accumulator is non-zero.

**Workaround**: Ensure both input tiles have the same sign type. Use sign extension
(`exti`) or zero-padding to convert operands to a common sign before `mmai`.

**Affected architectures**: sm_120 only
**Affected operations**: `mmai` with mismatched signed/unsigned input types

### Issue 4: Compiler Crash on Deeply Nested `for` Loops

**Description**: The Tile IR compiler may crash with an out-of-memory error when
compiling programs with `for` loops nested more than 8 levels deep. The compiler's
internal representation allocates exponential space for deeply nested loop
structures.

**Workaround**: Limit loop nesting to 8 levels or fewer. Refactor deeply nested loops
into separate kernel launches where possible.

**Affected architectures**: All (compiler issue, not hardware-specific)
**Affected configurations**: All optimization levels

---

## Appendix: Version History Summary

| Version | Date | Major Changes |
|---------|------|---------------|
| 13.0 | 2025-09-01 | Initial Tile IR release. Blackwell support. |
| 13.1 | 2025-12-15 | Bug fixes. Improved emulation. Blackwell sm_100/sm_120. |
| 13.2 | 2026-03-11 | Ampere/Ada support. New: `atan2`, `unsignedCmp`. Updated: `negi`, `tanh`, `print_tko`. |
