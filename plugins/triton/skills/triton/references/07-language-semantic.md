# Chapter 7: Language Semantic Module

The semantic module (`triton.language.semantic`) is the internal implementation layer that translates Triton operations into MLIR IR. Users typically don't interact with it directly, but understanding it helps with debugging.

## TritonSemantic Class

```python
class TritonSemantic(Generic[TensorTy]):
    def __init__(self, tensor, lang, builder):
        self.tensor = tensor    # Tensor class (triton.language.tensor or gluon)
        self.lang = lang        # Language module reference
        self.builder = builder  # MLIR builder (ir.builder)
```

## Type Promotion Rules

### Integer Promotion
When mixing integer types:
1. If any operand is unsigned, result is unsigned
2. Result width is max of operand widths
3. Signedness follows the wider type

### Computation Type (for binary ops)
1. **Kind hierarchy:** `{bool} < {integral} < {floating point}`
2. **Cross-kind:** If one operand is float and other is int, int is promoted to float
3. **Same-kind, different width:** Wider type wins
4. **float16 vs bfloat16:** Prefer float16
5. **signed vs unsigned (same width):** Prefer unsigned

### Broadcasting Rules
1. Pad shorter dimensions with ones on the left
2. Dimensions must be equal or one of them is 1
3. Dimension of 1 expands to match

```python
# Broadcasting examples in Triton
a = tl.full([128, 1], 1.0, tl.float32)
b = tl.full([1, 64], 2.0, tl.float32)
c = a + b  # Shape: [128, 64]
```

## Operation Implementation Details

### Binary Arithmetic
Each binary operation follows these steps:
1. Check type compatibility
2. Determine computation type via promotion rules
3. Cast operands to computation type
4. Emit MLIR operation
5. Return result tensor

### Memory Operations
The `load` and `store` operations support:
- **Masking:** Out-of-bounds access prevention
- **Boundary check:** Automatic boundary handling
- **Padding options:** Zero, NaN, or custom padding
- **Cache modifiers:** Cache hint for memory access
- **Eviction policy:** LRU, evict_first, evict_last, evict_normal
- **Volatile:** Volatile memory access

### Atomic Operations
Memory semantics:
- `ACQUIRE_RELEASE` (default)
- `ACQUIRE`
- `RELEASE`
- `RELAXED`

Scope:
- `GPU` (default) - System-wide visibility
- `CTA` - Thread-block only
- `DEVICE` - Device-wide

### Dot Product
Precision options for `tl.dot`:
- `"ieee"` - IEEE 754 compliant (slowest)
- `"tf32"` - TF32 tensor cores (Ampere+)
- `"tf32x3"` - Triple-TF32 for higher accuracy

FP8 formats for `tl.dot_scaled`:
- `"e4m3"` (float8e4nv) - Normal range
- `"e5m2"` (float8e5) - Extended range

## Target Information

### `tl.target_info.current_target()`
Returns current GPU target (GPUTarget or None).

### `tl.target_info.is_cuda() -> constexpr[bool]`
True if running on NVIDIA CUDA.

### `tl.target_info.is_hip() -> constexpr[bool]`
True if running on AMD ROCm/HIP.

### `tl.target_info.is_hip_cdna3() -> constexpr[bool]`
True if running on AMD CDNA3 (gfx942).

### `tl.target_info.is_hip_cdna4() -> constexpr[bool]`
True if running on AMD CDNA4 (gfx950).

### `tl.target_info.cuda_capability_geq(major, minor=0) -> constexpr[bool]`
Check CUDA compute capability.

```python
@triton.jit
def kernel(ptr, n, BLOCK_SIZE: tl.constexpr):
    if tl.target_info.cuda_capability_geq(8, 0):
        # Use Ampere+ features
        pass
    else:
        # Fallback path
        pass
```

## NumPy Differences

Triton differs from NumPy in key areas:

1. **Integer division:** Follows C semantics (rounds toward zero)
   - Triton: `7 // 2 == 3`, `-7 // 2 == -3`
   - NumPy: `-7 // 2 == -4`

2. **Modulo:** Follows C semantics
   - Triton: `-7 % 2 == -1`
   - NumPy: `-7 % 2 == 1`

3. **Type promotion:** Simpler rules, no value-dependent promotion
