# Stability

Tile IR provides a set of guarantees regarding portability, stability, and compatibility to ensure predictable behavior across different platforms, toolchains, and hardware targets. These guarantees are documented below.

## Definitions

The following terms are used throughout the stability and compatibility documentation. Understanding these definitions is essential for interpreting the guarantees Tile IR provides.

- **Stability:** An unchanging property of a program or interface. A stable interface is one that will not change incompatibly across versions. For example, the Tile IR bytecode format is stable: bytecode produced by an older compiler can be read by a newer driver.

- **Portability:** A property of a program to be transferred to a different hardware or toolchain version with the same behavior. A portable program produces correct results regardless of the target architecture. Portability does not guarantee identical numerical results across targets, only correct semantic behavior.

- **Compatibility:** A property of a program to be executed on a different platform or toolchain with the same behavior. Tile IR distinguishes between forward compatibility (old programs on new toolchains) and backward compatibility (new programs on old toolchains, with caveats).

- **Toolchain:** Either the compiler and CTK (CUDA Toolkit) version used to perform ahead-of-time compilation, or the driver and CTK version used to perform JIT compilation of a Tile IR program. Different toolchain versions may produce different but semantically equivalent programs.

- **Target:** The specific GPU architecture for which a Tile IR program is compiled (e.g., sm_80 for NVIDIA A100). The target determines which operations and data types are natively supported.

- **Feature:** A capability of the Tile IR language or runtime, such as a specific data type, operation, or optimization. Features may be target-specific (only available on certain architectures) or universal (available everywhere).

- **Lowering:** The process of converting a Tile IR operation into one or more operations that are supported by the target hardware. This may involve emulation when the target lacks native support.

- **Emulation:** A software implementation of a feature that is not natively supported by the target hardware. Emulation preserves semantics but may have different performance characteristics.

## Platform & Compatibility Guarantees

### Bytecode Stability

The Tile IR bytecode format ensures that programs can be interpreted and loaded by all conforming drivers (see Binary Format). The bytecode format is designed to be stable across versions, providing the following guarantees:

**Backward Compatibility (old bytecode on new driver):**
- A new driver can load and execute bytecode produced by any older Tile IR compiler.
- All previously defined opcodes, types, sections, and attributes remain supported.
- The semantics of existing operations are preserved across versions.

**Forward Compatibility (new bytecode on old driver):**
- An old driver encountering newer bytecode will skip unknown optional sections gracefully.
- The old driver will fail with a clear error message if it encounters unknown required features.
- The old driver will only process bytecode up to its supported version.

**Version Targeting:**
- A compiler can target a specific older Tile IR version by restricting output to features available in that version.
- If a program uses features unavailable in the target version, the compiler must diagnose the incompatibility.

### Program Portability

A program conforming to Tile IR vX.Y is syntactically portable to any platform that advertises support for vX.Y or newer.

Portability does not imply availability of target-specific features on all targets: if a program uses a feature that the selected hardware target does not support, the compiler will either:

- **Diagnose** the incompatibility: The compiler reports an error indicating that the feature is not supported on the selected target.
- **Apply a lowering** that preserves the semantics defined by the specification: The compiler replaces the unsupported operation with an equivalent sequence of supported operations, potentially at the cost of performance.

**Portability Example:**

```
// This program uses f8E4M3FN (FP8) data types:
%result = cuda_tile.add %a, %b : tensor<128xf8E4M3FN>

// On Blackwell (sm_100): Native support, direct hardware instructions.
// On Ampere (sm_80): No native FP8 support. The compiler will either:
//   - Emulate using F16/F32 instructions, OR
//   - Diagnose an error if emulation is not available for this operation.
```

### CUDA Compatibility

Tile IR respects the CUDA platform's forward and backward minor-version compatibility rules for toolchain and driver integration (see CUDA Minor Version Compatibility Rules).

**CUDA Compatibility Rules:**

| Compatibility Type | Rule |
|---|---|
| Forward (new driver, old CTK) | Tile IR programs compiled with older CTK versions can run on newer drivers |
| Backward (old driver, new CTK) | Limited; depends on CUDA minor version compatibility |
| Minor version compatibility | Within a major CUDA version, newer drivers support programs compiled with older minor versions |

**Example: CUDA Version Compatibility**

```
Tile IR program compiled with CUDA 13.1:
  - Runs on CUDA 13.2 driver: YES (forward compatible)
  - Runs on CUDA 13.0 driver: MAYBE (depends on features used)
  - Runs on CUDA 12.x driver: NO (major version mismatch)
```

## Supported Architectures

Tile IR bytecode programs are portable across all supported architectures. A single bytecode file can be compiled to any supported target or JIT-compiled by the driver at load time.

For ahead-of-time compilation, the target architecture is specified using the `--gpu-name` flag with a supported NVIDIA GPU architecture compute capability (CC) number (e.g., `tileiras --gpu-name sm_80`). For JIT compilation, the driver automatically selects the architecture of the target device.

### Supported Architectures Table

| Family | Compute Capability | Example GPUs | Tile IR Support Since | Key Features |
|--------|-------------------|--------------|----------------------|--------------|
| Ampere | `sm_80` | A100, A30 | Tile IR 13.2 | Third-gen Tensor Cores, BF16, TF32 |
| Ampere | `sm_86` | A40, RTX 3090, RTX 3080 | Tile IR 13.2 | Third-gen Tensor Cores, BF16, TF32 |
| Ampere | `sm_87` | Jetson Orin NX | Tile IR 13.2 | Edge variant of Ampere |
| Ampere | `sm_88` | Jetson AGX Orin | Tile IR 13.2 | Edge variant of Ampere |
| Ada | `sm_89` | L40, L40S, RTX 4090, RTX 4080 | Tile IR 13.2 | Fourth-gen Tensor Cores, DPX |
| Hopper | `sm_90` | H100, H200 | Not supported (13.2) | Planned for future release |
| Blackwell | `sm_100` | B200, B100 | Tile IR 13.1 | Fifth-gen Tensor Cores, FP8 native |
| Blackwell | `sm_120` | RTX 5090, RTX PRO 6000 | Tile IR 13.1 | Consumer Blackwell, FP8 native |

> **Note:** Hopper (`sm_90`) is not supported in the 13.2 release. Support is planned for a future release.

### Architecture Details

**Ampere Architecture (sm_80, sm_86, sm_87, sm_88):**

The Ampere architecture introduced third-generation Tensor Cores with support for BF16 and TF32 data types. Key characteristics:

- Tensor Core operations: MMA with mixed precision (F16, BF16, TF32, INT8, INT4, binary)
- Shared memory: Up to 164 KB per SM (configurable)
- L2 cache: Up to 40 MB (A100)
- Async copy from global to shared memory
- Cooperative groups support

**Ada Architecture (sm_89):**

The Ada Lovelace architecture provides fourth-generation Tensor Cores with improved throughput. Key characteristics:

- Enhanced Tensor Core operations with improved FP16 and BF16 performance
- DPX instructions for accelerated dynamic programming
- Higher clock speeds compared to Ampere
- Improved power efficiency

**Blackwell Architecture (sm_100, sm_120):**

The Blackwell architecture introduces fifth-generation Tensor Cores with native FP8 support. Key characteristics:

- Native FP8 (E4M3FN and E5M2) data type support in Tensor Cores
- Second-generation Transformer Engine
- Fifth-gen NVLink and NVSwitch
- Confidential computing support (sm_100)

### Feature Availability Matrix

The following matrix shows feature availability per architecture family:

| Feature | Ampere (sm_80-88) | Ada (sm_89) | Hopper (sm_90) | Blackwell (sm_100) | Blackwell (sm_120) |
|---------|-------------------|-------------|----------------|--------------------|--------------------|
| Integer types (i1-i64) | Supported | Supported | n/a | Supported | Supported |
| FP16 operations | Supported | Supported | n/a | Supported | Supported |
| BF16 operations | Supported | Supported | n/a | Supported | Supported |
| TF32 operations | Supported | Supported | n/a | Supported | Supported |
| FP32 operations | Supported | Supported | n/a | Supported | Supported |
| FP64 operations | Supported | Supported | n/a | Supported | Supported |
| FP8 (E4M3FN) | Not Supported | Not Supported | n/a | Supported | Supported |
| FP8 (E5M2) | Not Supported | Not Supported | n/a | Supported | Supported |
| Tensor Core MMA | Supported | Supported | n/a | Supported | Supported |
| Tile loads/stores | Supported | Supported | n/a | Supported | Supported |
| Token-ordered ops | Supported | Supported | n/a | Supported | Supported |
| Atomic RMW | Supported | Supported | n/a | Supported | Supported |
| Memory model (scopes) | Supported | Supported | n/a | Supported | Supported |
| Debug info in bytecode | Supported | Supported | n/a | Supported | Supported |

## Feature Availability & Emulation

### Target-specific Features

Tile IR may introduce new target-specific features (e.g., new datatypes, new operations) over time.

- **Availability:** A feature introduced in vX.Y becomes usable on a hardware target starting with the first platform release that declares support for it.
- **Fallback:** If a program uses a feature unsupported by the selected hardware target, the compiler will either diagnose the incompatibility or apply a lowering (emulation) that preserves semantics as defined by the specification.

Note that certain types have more restricted usage than others. See Element Types for details.

> **Warning:** During the 13.x release cycle, we are bringing up existing hardware targets which may introduce new features on old targets. This "cold start" period is an exception; normally, new features will only appear in new targets.

> **Note:** Today the only target-specific features are specific datatypes.

### Hardware Support Matrix

Detailed hardware support for each data type across all supported architectures:

| Data Type | Size (bits) | Ampere (sm_80+) | Ada (sm_89) | Hopper (sm_90) | Blackwell (sm_100+) |
|-----------|------------|-----------------|-------------|----------------|---------------------|
| `i1` | 1 | Supported | Supported | n/a | Supported |
| `i8` | 8 | Supported | Supported | n/a | Supported |
| `i16` | 16 | Supported | Supported | n/a | Supported |
| `i32` | 32 | Supported | Supported | n/a | Supported |
| `i64` | 64 | Supported | Supported | n/a | Supported |
| `f16` | 16 | Supported | Supported | n/a | Supported |
| `bf16` | 16 | Supported | Supported | n/a | Supported |
| `f32` | 32 | Supported | Supported | n/a | Supported |
| `tf32` | 19 | Supported | Supported | n/a | Supported |
| `f64` | 64 | Supported | Supported | n/a | Supported |
| `f8E4M3FN` | 8 | Not Supported | Not Supported | n/a | Supported |
| `f8E5M2` | 8 | Not Supported | Not Supported | n/a | Supported |

**Data Type Descriptions:**

| Type | Description | Typical Use Case |
|------|------------|-----------------|
| `i1` | 1-bit boolean | Masks, predicates, conditions |
| `i8` | 8-bit signed/unsigned integer | Quantized inference, character data |
| `i16` | 16-bit signed/unsigned integer | Half-precision integer computation |
| `i32` | 32-bit signed/unsigned integer | General-purpose integer computation, indexing |
| `i64` | 64-bit signed/unsigned integer | Large integer computation, wide addressing |
| `f16` | IEEE 754 half-precision (binary16) | Mixed-precision training, inference |
| `bf16` | Brain float (1 sign, 8 exp, 7 mantissa) | Deep learning training |
| `tf32` | TensorFloat-32 (1 sign, 8 exp, 10 mantissa) | Tensor Core operations |
| `f64` | IEEE 754 double-precision (binary64) | High-precision scientific computing |
| `f8E4M3FN` | 8-bit float (4 exp, 3 mantissa) | FP8 inference on Blackwell |
| `f8E5M2` | 8-bit float (5 exp, 2 mantissa) | FP8 training on Blackwell |

### Emulation Strategies

To maintain portability, Tile IR may emulate operations on hardware targets that lack native support. The following emulation strategies are available:

**Type Emulation:**

When a data type is not natively supported, Tile IR may emulate it using a wider type:

| Unsupported Type | Emulation Strategy | Notes |
|-----------------|-------------------|-------|
| FP8 on Ampere/Ada | Promote to FP16 or BF16, perform operation, convert back | Loss of dynamic range; semantics preserved within FP8 representable range |
| BF16 on legacy | Promote to FP32, perform operation, convert back | May differ in edge cases |
| TF32 on non-Tensor-Core | Use FP32 with reduced mantissa | Tensor Core specific; falls back to FP32 math |

**Operation Emulation:**

When an operation is not natively supported on the target:

| Scenario | Strategy | Notes |
|----------|----------|-------|
| Native hardware op unavailable | Decompose into sequence of supported ops | Preserves semantics; may be slower |
| Tensor Core MMA on non-supporting target | Emulate using scalar/vector operations | Significant performance impact |
| Advanced atomics | Decompose into CAS loop | Correct but slower |

**Emulation Behavior Summary:**

```
For a program using an unsupported feature:
  1. Compiler checks if the feature is available on the target
  2. If not available:
     a. If emulation exists: Apply lowering, emit performance warning
     b. If no emulation: Diagnose error, fail compilation
  3. Emulated operations preserve semantic correctness
  4. Performance may differ significantly from native execution
```

## Execution & Numerical Guarantees

### Execution Determinism

For a fixed toolchain, configuration, and hardware target, compilation and execution are deterministic within a single tile-block thread.

**What IS deterministic:**

- Within a single tile-block thread, for a fixed toolchain version, fixed configuration flags, and fixed target hardware, the sequence of operations and their results are deterministic.
- The same bytecode, compiled with the same toolchain to the same target, will produce the same results on the same hardware.

**What is NOT guaranteed to be deterministic:**

- Results across different toolchain versions (the compiler may choose different instruction sequences).
- Results across different hardware targets (different hardware may implement operations differently).
- Results across different configuration settings (e.g., different optimization levels).
- Inter-tile-block thread execution ordering (the order in which different tile blocks execute relative to each other is unspecified).

**Version Changes:** Using a different toolchain version may produce a different program and thus different results; this is expected behavior, not non-determinism.

**Example:**

```
// Same bytecode, different toolchains:
// Toolchain 13.1: compiles add -> hardware_add_instruction_v1
// Toolchain 13.2: compiles add -> hardware_add_instruction_v2 (optimized)

// Both produce correct results, but bit patterns may differ
// due to different instruction selection, scheduling, etc.
```

### Numerical Stability

Tile IR does not guarantee bit-identical numerical results across different toolchain versions, configurations, or targets, except where explicitly documented.

**Scope:** Stability guarantees are scoped to specific versions and targets. When a guarantee is documented for a specific operation on a specific target, it holds only for that combination.

**Updates:** Changes are not retroactive; compiling/executing with an earlier toolchain retains the guarantees published for that version.

**Numerical Stability Scope Table:**

| Aspect | Guaranteed | Not Guaranteed |
|--------|-----------|----------------|
| Same toolchain + config + target | Deterministic results | -- |
| Different toolchain version | -- | Bit-identical results |
| Different target architecture | -- | Bit-identical results |
| Different optimization level | -- | Bit-identical results |
| MMA operations | -- | Bit-identical results (unless documented) |
| Scalar arithmetic | IEEE-compliant for evaluation order | Bit-identical across targets |

### Floating-point Semantics

Floating-point operations follow applicable IEEE semantics for the order in which they are actually evaluated.

**IEEE Compliance:**

- Floating-point operations in Tile IR follow IEEE 754 semantics for the precision and rounding mode in effect.
- The order of evaluation is determined by the compiler and may differ from the source order.
- Intermediate results may be computed at a higher precision than the final result type.

**Transformations:** Compiler transformations (e.g., reordering, fusion) can change numeric results across versions. The compiler may:

- Reorder floating-point operations (e.g., `(a + b) + c` -> `a + (b + c)`).
- Fuse operations (e.g., multiply-add -> fused multiply-add).
- Convert between precision representations for intermediate results.
- Apply algebraic simplifications that are not strictly IEEE-equivalent.

These transformations preserve overall correctness but may change bit-level results.

**Precision:** Operations like MMA (Matrix Multiply-Accumulate) may have weaker or no guarantees of bit-identical numerical results unless explicitly documented. MMA operations are typically performed at reduced precision internally (e.g., TF32 accumulates in FP32 but only uses 10 mantissa bits for the multiplication).

**Floating-point Operation Behavior:**

| Operation Type | Guarantee |
|---------------|-----------|
| Scalar add/sub/mul/div | IEEE 754 compliant at evaluation precision |
| FMA (fused multiply-add) | Single rounding at target precision |
| MMA (Tensor Core) | Precision determined by input/output types; may not be bit-identical across implementations |
| Type conversions | IEEE 754 rounding applied; overflow/underflow behavior defined |
| Reduction operations | Order of reduction unspecified; may not be bit-identical across toolchains |
| Transcendental functions | Implementation-defined accuracy; typically within 1-2 ULP |

**Example: Non-associativity of floating-point arithmetic**

```
// These may produce different results depending on evaluation order:
%a = add %x, %y    // x + y
%b = add %a, %z    // (x + y) + z

// vs.
%c = add %y, %z    // y + z
%d = add %x, %c    // x + (y + z)

// IEEE 754 guarantees each individual add is correct,
// but the final results may differ due to rounding.
// Tile IR does not guarantee which order is chosen.
```

## Release Notes

### Known Issues

The following known issues exist in the current Tile IR release:

1. **Cross-tile block kernel support:** The programming model is missing a section on a cross-tile block kernel such as split-k. This means that kernel patterns requiring coordination across multiple tile blocks (beyond what the memory model provides) are not yet documented or fully supported in the programming model.

2. **Operation encoding detail:** The bytecode section does not provide exact encoding of each individual operation. While the general encoding format and common operation examples are documented, the precise binary layout for every operation opcode is not yet specified. Expect this to be introduced in a future release.

3. **Memory model examples:** The semi-formal memory model section is written but does not provide detailed examples of how to utilize it. The axioms and relations are defined, but practical usage patterns and complete working examples are not yet available.

4. **Limited atomics:** Atomics are currently limited in Tile IR and will be expanded in a future release. The current atomic operations support basic read-modify-write patterns (add, sub, and, or, xor, min, max) and compare-and-swap, but more advanced atomic patterns and wider atomic operation support are planned.

5. **Hopper support gap:** Hopper (`sm_90`) architecture is not supported in the 13.2 release. Programs targeting H100/H200 GPUs cannot be compiled with the current Tile IR release.

6. **Emulation limitations:** Not all unsupported features can be emulated on all targets. In some cases, the compiler must diagnose the incompatibility rather than providing a fallback path.

### Changelog

#### Spec 13.2 (2026-03-11)

The Tile IR 13.2 release adds support for additional GPU architectures and introduces several new and updated operations.

**Supported Architectures**

- Added support for Ampere (sm_80, sm_86, sm_87, sm_88) architectures. This enables Tile IR programs to run on NVIDIA A100, A30, A40, RTX 3090, RTX 3080, and Jetson Orin platforms.
- Added support for Ada (sm_89) architecture. This enables Tile IR programs to run on NVIDIA L40, L40S, RTX 4090, and RTX 4080 platforms.

**New Operations**

- **`cuda_tile.atan2`**: Added a new operation for element-wise two-argument arctangent. This operation computes `atan2(y, x)` for each element in the input tiles, returning the angle in radians between the positive x-axis and the point `(x, y)`. The operation supports all floating-point data types and follows the IEEE 754 semantics for `atan2`, including proper handling of special cases (zero, infinity, NaN).

  Syntax:
  ```
  %result = cuda_tile.atan2 %y, %x : tensor<MxNxf32>
  ```

**Updated Operations**

- **`cuda_tile.negi`**: Added `overflow` attribute to control integer overflow behavior. When `overflow` is set to `nsw` (no signed wrap), the compiler can assume that negation does not overflow, enabling additional optimizations. When `overflow` is not set, the operation follows standard two's complement wrapping behavior.

  Syntax:
  ```
  %result = cuda_tile.negi %input {overflow = nsw} : tensor<MxNi32>
  ```

- **`cuda_tile.tanh`**: Added `rounding_mode` attribute to control floating-point rounding behavior. This allows programmers to specify the rounding behavior for the hyperbolic tangent operation, which can be important for numerical reproducibility.

  Syntax:
  ```
  %result = cuda_tile.tanh %input {rounding_mode = "nearest"} : tensor<MxNxf32>
  ```

- **`cuda_tile.print_tko`**: Added token result for memory ordering support. The print operation now produces a token output that can be used to order the print relative to other token-ordered memory operations. This ensures that printed values reflect the correct state of memory when used in conjunction with the memory model.

  Syntax:
  ```
  %tok_out = cuda_tile.print_tko %value token(%tok_in)
  ```

- **`cuda_tile.for`**: Added `unsignedCmp` flag to support unsigned integer comparison for loop termination. When `unsignedCmp` is set to true, the loop comparison treats the induction variable and bounds as unsigned integers, which changes the behavior for negative values and large positive values near the maximum representable value.

  Syntax:
  ```
  %result = cuda_tile.for %lb to %ub step %step {unsignedCmp = true} ...
  ```

- **`cuda_tile.print` -> `cuda_tile.print_tko`**: Renamed `cuda_tile.print` to `cuda_tile.print_tko` in the textual format to reflect the token-ordered nature of the operation. The bytecode encoding is unchanged and remains backward compatible; existing bytecode files with the old encoding continue to work without modification. Only the textual (assembly) representation has changed.

**Backward Compatibility Notes:**

- All existing bytecode files remain compatible with the 13.2 driver.
- The `cuda_tile.print` to `cuda_tile.print_tko` rename only affects the textual format. Binary encoding is unchanged.
- New operations (`atan2`) use new opcodes that older drivers will skip or reject gracefully.
- New attributes (`overflow`, `rounding_mode`, `unsignedCmp`) are optional; existing programs that do not use them remain compatible.

**Breaking Changes:**

- None. The 13.2 release is fully backward compatible with 13.1.

**Deprecations:**

- The textual format name `cuda_tile.print` is deprecated in favor of `cuda_tile.print_tko`. The old name may still be accepted by the assembler in 13.2 but will be removed in a future release.

#### Spec 13.1 (2025-XX-XX)

The Tile IR 13.1 release introduced initial support for the Blackwell architecture.

**Supported Architectures**

- Initial support for Blackwell (sm_100) architecture, enabling Tile IR programs on NVIDIA B200 and B100 platforms.
- Initial support for Blackwell (sm_120) architecture, enabling Tile IR programs on NVIDIA RTX 5090 and RTX PRO 6000 platforms.

**Features:**

- Native FP8 (f8E4M3FN and f8E5M2) data type support on Blackwell architectures.
- Fifth-generation Tensor Core support.
- Initial Tile IR bytecode format specification.

### Version History Summary

| Version | Date | Major Changes |
|---------|------|---------------|
| 13.2 | 2026-03-11 | Ampere + Ada support, atan2, updated negi/tanh/for/print_tko |
| 13.1 | 2025 | Initial Blackwell support, FP8 native, bytecode format |

### Upgrade Guide

**Upgrading from 13.1 to 13.2:**

1. **Ampere/Ada targets:** You can now compile for sm_80, sm_86, sm_87, sm_88, and sm_89 targets using `--gpu-name`.
2. **New operation `atan2`:** Available on all supported targets. Add to your programs as needed.
3. **`cuda_tile.negi` overflow attribute:** Use `{overflow = nsw}` for additional optimization opportunities when negation is guaranteed not to wrap.
4. **`cuda_tile.tanh` rounding_mode:** Use `{rounding_mode = "nearest"}` or other modes for explicit rounding control.
5. **`cuda_tile.for` unsignedCmp:** Use `{unsignedCmp = true}` when loop bounds should be compared as unsigned integers.
6. **`cuda_tile.print` renamed to `cuda_tile.print_tko`:** Update your textual Tile IR sources. Bytecode is unchanged.

**Compatibility when upgrading:**

- All 13.1 bytecode files run without modification on 13.2 drivers.
- No recompilation required for existing programs.
- New features are opt-in; existing programs are unaffected.

### Frequently Asked Questions

**Q: Will my Tile IR program produce identical results on Ampere and Blackwell?**

A: Not necessarily. While the program semantics are preserved, different architectures may use different instruction sequences, different precisions for intermediate results, and different rounding behaviors. Tile IR guarantees correctness, not bit-identical results across targets. See the Numerical Stability section for details.

**Q: What happens if I use FP8 types on an Ampere GPU?**

A: The compiler will attempt to emulate FP8 operations using wider types (typically FP16 or BF16). If emulation is available, the program will run correctly but with lower performance than native FP8 on Blackwell. If no emulation path exists, the compiler will diagnose an error.

**Q: Can I target multiple architectures with a single Tile IR bytecode file?**

A: Yes. Tile IR bytecode is architecture-independent. You can compile the same bytecode to different targets using `--gpu-name`. The driver will JIT-compile for the appropriate architecture at load time.

**Q: Is Hopper (sm_90) support coming?**

A: Hopper support is planned for a future release. It is not available in the 13.2 release cycle.

**Q: What if a new Tile IR version introduces a required change to the bytecode format?**

A: Required changes are always introduced with a new bytecode version number. Older drivers will detect the version mismatch and produce a clear error message. You can always target an older version using the version targeting feature.

**Q: How do I ensure my program is portable across all supported architectures?**

A: Stick to universal features (integer types, FP16, BF16, TF32, FP32, FP64) that are supported on all architectures. Avoid target-specific types like FP8 unless you have a fallback path. Use the hardware support matrix to check availability.

### Compatibility Guarantee Summary

The following table summarizes all compatibility guarantees provided by Tile IR:

| Guarantee | Scope | Details |
|-----------|-------|---------|
| Bytecode backward compat | All versions | New drivers read old bytecode |
| Bytecode forward compat | Minor versions | Old drivers gracefully handle new bytecode (skip unknown sections) |
| Program portability | All supported targets | Same bytecode compiles to any supported architecture |
| Semantic correctness | All targets | Operations produce correct results (within type precision) |
| Execution determinism | Single toolchain + target + config | Same inputs produce same outputs |
| Numerical stability | Per-version + per-target | Documented per operation; not guaranteed across versions/targets |
| CUDA compatibility | Minor versions | Respects CUDA forward/backward minor-version rules |

### Reporting Issues

When reporting compatibility or stability issues, please include the following information:

1. Tile IR version (major.minor.tag)
2. Toolchain version (compiler + CTK version)
3. Target architecture (`--gpu-name` value)
4. GPU device and driver version
5. Minimal reproducer program
6. Expected vs. actual behavior
7. Whether the issue is reproducible across different toolchain versions
