# Bitwise and Comparison Operations

## Overview

Bitwise and comparison operations are essential for creating masks, implementing conditional logic, and performing low-level bit manipulation in GPU kernels. cuTile provides a complete set of these operations that work element-wise on tiles.

This chapter covers:
- **Bitwise operations**: AND, OR, XOR, shifts, and NOT for integer types
- **Comparison operations**: Relational operators that return boolean results
- **Selection operations**: Conditional element selection using masks
- **Practical patterns**: Common use cases with complete examples

## Bitwise Operations

Bitwise operations work on the binary representation of integers. These are particularly useful for:
- Creating and manipulating boolean masks
- Implementing custom data encoding/decoding
- Performing efficient flag operations
- Implementing certain mathematical optimizations

### ct.bitwise_and() — Bitwise AND

#### Syntax
```python
ct.bitwise_and(a, b) -> Tile
# Or using operator:
a & b
```

#### Description
Performs element-wise bitwise AND operation. For each bit position, the result is 1 if both operands have 1, otherwise 0.

#### Parameters
- `a` (`Tile`): First operand (integer type)
- `b` (`Tile`): Second operand (integer type)

#### Returns
- `Tile`: Result of bitwise AND (same type as inputs)

#### Supported Data Types
`int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`

#### Truth Table
| A | B | A & B |
|---|---|-------|
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

#### Example: Checking Even Numbers

```python
@ct.kernel
def is_even_kernel(X: ct.Buffer, is_even: ct.Buffer):
    """Check if each number is even using bitwise AND."""
    i = ct.tid()
    
    x = X.load(i)
    
    # Least significant bit = 0 for even numbers
    even_mask = x & 1
    is_even_result = even_mask == 0
    
    is_even.store(i, is_even_result)
```

#### Example: Masking Bits

```python
@ct.kernel
def extract_byte_kernel(X: ct.Buffer, byte_idx: ct.Constant[int], result: ct.Buffer):
    """Extract a specific byte from a 32-bit integer."""
    i = ct.tid()
    
    x = X.load(i)
    
    # Shift right to target byte, then mask with 0xFF
    shift_amount = byte_idx * 8
    masked_byte = (x >> shift_amount) & 0xFF
    
    result.store(i, masked_byte)
```

### ct.bitwise_or() — Bitwise OR

#### Syntax
```python
ct.bitwise_or(a, b) -> Tile
# Or using operator:
a | b
```

#### Description
Performs element-wise bitwise OR operation. For each bit position, the result is 1 if either operand has 1.

#### Parameters
- `a` (`Tile`): First operand (integer type)
- `b` (`Tile`): Second operand (integer type)

#### Returns
- `Tile`: Result of bitwise OR (same type as inputs)

#### Truth Table
| A | B | A \| B |
|---|---|-------|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 1 |

#### Example: Setting Flags

```python
@ct.kernel
def set_flag_kernel(flags: ct.Buffer, flag_mask: ct.Constant[int]):
    """Set a specific flag in a flags register."""
    i = ct.tid()
    
    f = flags.load(i)
    
    # Set the flag (OR with mask)
    f_with_flag = f | flag_mask
    
    flags.store(i, f_with_flag)
```

#### Example: Combining Masks

```python
@ct.kernel
def combine_masks_kernel(
    mask1: ct.Buffer,
    mask2: ct.Buffer,
    combined: ct.Buffer
):
    """Combine two boolean masks using OR."""
    i = ct.tid()
    
    m1 = mask1.load(i)
    m2 = mask2.load(i)
    
    # Combine: true if either mask is true
    combined_mask = m1 | m2
    
    combined.store(i, combined_mask)
```

### ct.bitwise_xor() — Bitwise XOR

#### Syntax
```python
ct.bitwise_xor(a, b) -> Tile
# Or using operator:
a ^ b
```

#### Description
Performs element-wise bitwise XOR (exclusive OR) operation. For each bit position, the result is 1 if operands differ.

#### Parameters
- `a` (`Tile`): First operand (integer type)
- `b` (`Tile`): Second operand (integer type)

#### Returns
- `Tile`: Result of bitwise XOR (same type as inputs)

#### Truth Table
| A | B | A ^ B |
|---|---|-------|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

#### Example: Toggling Bits

```python
@ct.kernel
def toggle_flag_kernel(flags: ct.Buffer, flag_mask: ct.Constant[int]):
    """Toggle a specific flag (XOR with mask)."""
    i = ct.tid()
    
    f = flags.load(i)
    
    # Toggle the flag
    f_toggled = f ^ flag_mask
    
    flags.store(i, f_toggled)
```

#### Example: Parity Check

```python
@ct.kernel
def parity_kernel(X: ct.Buffer, parity: ct.Buffer):
    """Compute parity bit (XOR of all bits)."""
    i = ct.tid()
    
    x = X.load(i)
    
    # Compute parity by XOR-ing halves repeatedly
    x ^= x >> 32
    x ^= x >> 16
    x ^= x >> 8
    x ^= x >> 4
    x ^= x >> 2
    x ^= x >> 1
    
    # LSB now contains parity
    parity_bit = x & 1
    
    parity.store(i, parity_bit)
```

### ct.bitwise_lshift() — Left Shift

#### Syntax
```python
ct.bitwise_lshift(a, b) -> Tile
# Or using operator:
a << b
```

#### Description
Performs element-wise left shift operation. Shifts bits to the left, filling with zeros.

#### Parameters
- `a` (`Tile`): Value to shift (integer type)
- `b` (`Tile`): Shift amount (integer type)

#### Returns
- `Tile`: Shifted value (same type as `a`)

#### Example: Multiply by Powers of 2

```python
@ct.kernel
def fast_multiply_kernel(X: ct.Buffer, power: ct.Constant[int], result: ct.Buffer):
    """Fast multiply by 2^power using left shift."""
    i = ct.tid()
    
    x = X.load(i)
    
    # Multiply by 2^power
    result_val = x << power
    
    result.store(i, result_val)
```

#### Example: Packing Values

```python
@ct.kernel
def pack_4x8_kernel(
    a: ct.Buffer,  # 8-bit values
    b: ct.Buffer,
    c: ct.Buffer,
    d: ct.Buffer,
    packed: ct.Buffer  # 32-bit packed values
):
    """Pack four 8-bit values into a 32-bit integer."""
    i = ct.tid()
    
    # Load 8-bit values
    av = a.load(i) & 0xFF
    bv = b.load(i) & 0xFF
    cv = c.load(i) & 0xFF
    dv = d.load(i) & 0xFF
    
    # Pack into 32 bits: [a|b|c|d]
    packed_val = (av << 24) | (bv << 16) | (cv << 8) | dv
    
    packed.store(i, packed_val)
```

### ct.bitwise_rshift() — Right Shift

#### Syntax
```python
ct.bitwise_rshift(a, b) -> Tile
# Or using operator:
a >> b
```

#### Description
Performs element-wise right shift operation. Shifts bits to the right. For signed types, performs arithmetic shift (sign-extended). For unsigned types, performs logical shift (zero-filled).

#### Parameters
- `a` (`Tile`): Value to shift (integer type)
- `b` (`Tile`): Shift amount (integer type)

#### Returns
- `Tile`: Shifted value (same type as `a`)

#### Example: Divide by Powers of 2

```python
@ct.kernel
def fast_divide_kernel(X: ct.Buffer, power: ct.Constant[int], result: ct.Buffer):
    """Fast divide by 2^power using right shift."""
    i = ct.tid()
    
    x = X.load(i)
    
    # Divide by 2^power
    result_val = x >> power
    
    result.store(i, result_val)
```

#### Example: Unpacking Values

```python
@ct.kernel
def unpack_4x8_kernel(
    packed: ct.Buffer,  # 32-bit packed values
    a: ct.Buffer,  # 8-bit values
    b: ct.Buffer,
    c: ct.Buffer,
    d: ct.Buffer
):
    """Unpack four 8-bit values from a 32-bit integer."""
    i = ct.tid()
    
    # Load packed value
    packed_val = packed.load(i)
    
    # Extract each byte
    av = (packed_val >> 24) & 0xFF
    bv = (packed_val >> 16) & 0xFF
    cv = (packed_val >> 8) & 0xFF
    dv = packed_val & 0xFF
    
    # Store extracted values
    a.store(i, av)
    b.store(i, bv)
    c.store(i, cv)
    d.store(i, dv)
```

### ct.bitwise_not() — Bitwise NOT

#### Syntax
```python
ct.bitwise_not(a) -> Tile
# Or using operator:
~a
```

#### Description
Performs element-wise bitwise NOT (complement) operation. Flips all bits.

#### Parameters
- `a` (`Tile`): Input tile (integer type)

#### Returns
- `Tile`: Bitwise complement (same type as input)

#### Truth Table
| A | ~A |
|---|----|
| 0 | 1 |
| 1 | 0 |

#### Example: Two's Complement Negation

```python
@ct.kernel
def negate_kernel(X: ct.Buffer, result: ct.Buffer):
    """Compute two's complement negation: -x = ~x + 1."""
    i = ct.tid()
    
    x = X.load(i)
    
    # Two's complement
    negated = ~x + 1
    
    result.store(i, negated)
```

#### Example: Inverting Mask

```python
@ct.kernel
def invert_mask_kernel(mask: ct.Buffer, inverted: ct.Buffer):
    """Invert a boolean mask."""
    i = ct.tid()
    
    m = mask.load(i)
    
    # Invert all bits
    inv = ~m
    
    inverted.store(i, inv)
```

## Comparison Operations

Comparison operations compare two tiles element-wise and return boolean tiles. These are fundamental for creating masks and implementing conditional logic.

### ct.greater() — Greater Than

#### Syntax
```python
ct.greater(a, b) -> Tile
# Or using operator:
a > b
```

#### Description
Performs element-wise greater-than comparison.

#### Parameters
- `a` (`Tile`): Left operand
- `b` (`Tile`): Right operand

#### Returns
- `Tile`: Boolean tile with `True` where `a > b`

#### Example: Thresholding

```python
@ct.kernel
def threshold_kernel(X: ct.Buffer, threshold: ct.Constant[float], mask: ct.Buffer):
    """Create mask where X > threshold."""
    i = ct.tid()
    
    x = X.load(i)
    
    # Create boolean mask
    above_threshold = x > threshold
    
    mask.store(i, above_threshold)
```

### ct.greater_equal() — Greater Than or Equal

#### Syntax
```python
ct.greater_equal(a, b) -> Tile
# Or using operator:
a >= b
```

#### Description
Performs element-wise greater-than-or-equal comparison.

#### Parameters
- `a` (`Tile`): Left operand
- `b` (`Tile`): Right operand

#### Returns
- `Tile`: Boolean tile with `True` where `a >= b`

#### Example: ReLU Activation

```python
@ct.kernel
def relu_kernel(X: ct.Buffer, Y: ct.Buffer):
    """ReLU: max(0, x) = x if x >= 0 else 0."""
    i = ct.tid()
    
    x = X.load(i)
    
    # Create condition
    is_positive = x >= 0
    
    # Select output based on condition
    y = ct.where(is_positive, x, 0.0)
    
    Y.store(i, y)
```

### ct.less() — Less Than

#### Syntax
```python
ct.less(a, b) -> Tile
# Or using operator:
a < b
```

#### Description
Performs element-wise less-than comparison.

#### Parameters
- `a` (`Tile`): Left operand
- `b` (`Tile`): Right operand

#### Returns
- `Tile`: Boolean tile with `True` where `a < b`

#### Example: Upper Clipping

```python
@ct.kernel
def clip_upper_kernel(X: ct.Buffer, max_val: ct.Constant[float], Y: ct.Buffer):
    """Clip values above max_val."""
    i = ct.tid()
    
    x = X.load(i)
    
    # Clip to max
    y = ct.where(x < max_val, x, max_val)
    
    Y.store(i, y)
```

### ct.less_equal() — Less Than or Equal

#### Syntax
```python
ct.less_equal(a, b) -> Tile
# Or using operator:
a <= b
```

#### Description
Performs element-wise less-than-or-equal comparison.

#### Parameters
- `a` (`Tile`): Left operand
- `b` (`Tile`): Right operand

#### Returns
- `Tile`: Boolean tile with `True` where `a <= b`

#### Example: Lower Clipping

```python
@ct.kernel
def clip_lower_kernel(X: ct.Buffer, min_val: ct.Constant[float], Y: ct.Buffer):
    """Clip values below min_val."""
    i = ct.tid()
    
    x = X.load(i)
    
    # Clip to min
    y = ct.where(x <= min_val, min_val, x)
    
    Y.store(i, y)
```

### ct.equal() — Equality

#### Syntax
```python
ct.equal(a, b) -> Tile
# Or using operator:
a == b
```

#### Description
Performs element-wise equality comparison.

#### Parameters
- `a` (`Tile`): Left operand
- `b` (`Tile`): Right operand

#### Returns
- `Tile`: Boolean tile with `True` where `a == b`

#### Example: Find Matching Indices

```python
@ct.kernel
def find_matches_kernel(
    values: ct.Buffer,
    target: ct.Constant[int],
    matches: ct.Buffer
):
    """Find indices where values == target."""
    i = ct.tid()
    
    v = values.load(i)
    
    # Check for match
    is_match = v == target
    
    matches.store(i, is_match)
```

#### Example: Floating Point Comparison with Epsilon

```python
@ct.kernel
def approx_equal_kernel(
    A: ct.Buffer,
    B: ct.Buffer,
    C: ct.Buffer,
    eps: ct.Constant[float] = 1e-5
):
    """Check if A and B are approximately equal."""
    i = ct.tid()
    
    a = A.load(i)
    b = B.load(i)
    
    # Check |a - b| < eps
    diff = ct.abs(a - b)
    is_close = diff < eps
    
    C.store(i, is_close)
```

### ct.not_equal() — Inequality

#### Syntax
```python
ct.not_equal(a, b) -> Tile
# Or using operator:
a != b
```

#### Description
Performs element-wise inequality comparison.

#### Parameters
- `a` (`Tile`): Left operand
- `b` (`Tile`): Right operand

#### Returns
- `Tile`: Boolean tile with `True` where `a != b`

#### Example: Find Changed Elements

```python
@ct.kernel
def find_changes_kernel(
    old_values: ct.Buffer,
    new_values: ct.Buffer,
    changed: ct.Buffer
):
    """Find indices where values changed."""
    i = ct.tid()
    
    old = old_values.load(i)
    new = new_values.load(i)
    
    # Check for change
    has_changed = old != new
    
    changed.store(i, has_changed)
```

## Selection Operations

Selection operations use boolean conditions to choose between values.

### ct.where() — Conditional Selection

#### Syntax
```python
ct.where(condition, x, y) -> Tile
```

#### Description
Selects elements from `x` or `y` based on `condition`. For each element:
- If `condition` is `True`, select from `x`
- If `condition` is `False`, select from `y`

#### Parameters
- `condition` (`Tile`): Boolean mask
- `x` (`Tile`): Values to select when condition is True
- `y` (`Tile`): Values to select when condition is False

#### Returns
- `Tile`: Selected values

#### Broadcasting
`condition` is broadcast to match the shape of `x` and `y`. `x` and `y` must be broadcastable to the same shape.

#### Example: Conditional Assignment

```python
@ct.kernel
def conditional_assign_kernel(
    X: ct.Buffer,
    Y: ct.Buffer,
    threshold: ct.Constant[float]
):
    """Y = X if X > threshold else 0."""
    i = ct.tid()
    
    x = X.load(i)
    
    # Conditional assignment
    y = ct.where(x > threshold, x, 0.0)
    
    Y.store(i, y)
```

#### Example: Piecewise Function

```python
@ct.kernel
def piecewise_kernel(X: ct.Buffer, Y: ct.Buffer):
    """Implement piecewise function."""
    i = ct.tid()
    
    x = X.load(i)
    
    # Piecewise: x^2 if x < 0, x if 0 <= x < 1, sqrt(x) if x >= 1
    condition1 = x < 0
    condition2 = (x >= 0) & (x < 1)
    
    y = ct.where(condition1, x ** 2,
         ct.where(condition2, x, ct.sqrt(x)))
    
    Y.store(i, y)
```

### ct.extract() — Extract Subtile

#### Syntax
```python
ct.extract(tile, offset, shape) -> Tile
```

#### Description
Extracts a smaller tile from a larger tile at the specified offset.

#### Parameters
- `tile` (`Tile`): Source tile
- `offset` (tuple of int): Starting offset for extraction
- `shape` (tuple of int): Shape of extracted tile

#### Returns
- `Tile`: Extracted subtile

#### Example: Extract ROI

```python
@ct.kernel
def extract_roi_kernel(
    image: ct.Buffer,
    roi: ct.Buffer,
    roi_x: ct.Constant[int],
    roi_y: ct.Constant[int],
    roi_width: ct.Constant[int],
    roi_height: ct.Constant[int]
):
    """Extract region of interest from image."""
    i, j = ct.bid(0), ct.bid(1)
    
    # Load tile from image
    image_tile = image.load((i + roi_x, j + roi_y), shape=(roi_width, roi_height))
    
    # Store to ROI buffer
    roi.store((i, j), image_tile)
```

## Complete Examples

### Example 1: Boolean Mask Operations

```python
@ct.kernel
def mask_operations_kernel(
    mask1: ct.Buffer,
    mask2: ct.Buffer,
    and_result: ct.Buffer,
    or_result: ct.Buffer,
    xor_result: ct.Buffer,
    not_result: ct.Buffer
):
    """Demonstrate boolean mask operations."""
    i = ct.tid()
    
    m1 = mask1.load(i)
    m2 = mask2.load(i)
    
    # Boolean operations (bitwise operations on boolean masks)
    and_result.store(i, m1 & m2)
    or_result.store(i, m1 | m2)
    xor_result.store(i, m1 ^ m2)
    not_result.store(i, ~m1)
```

### Example 2: Clamping to Range

```python
@ct.kernel
def clamp_kernel(
    X: ct.Buffer,
    Y: ct.Buffer,
    min_val: ct.Constant[float],
    max_val: ct.Constant[float]
):
    """Clamp values to [min_val, max_val]."""
    i = ct.tid()
    
    x = X.load(i)
    
    # Clamp using comparisons
    x_clipped = ct.where(x < min_val, min_val, x)
    x_clipped = ct.where(x_clipped > max_val, max_val, x_clipped)
    
    Y.store(i, x_clipped)
```

### Example 3: Sign Function

```python
@ct.kernel
def sign_kernel(X: ct.Buffer, Y: ct.Buffer):
    """Compute sign: -1 if x < 0, 0 if x == 0, 1 if x > 0."""
    i = ct.tid()
    
    x = X.load(i)
    
    # Compute sign
    is_positive = x > 0
    is_negative = x < 0
    is_zero = x == 0
    
    # Convert boolean to int and combine
    sign = ct.where(is_positive, 1,
            ct.where(is_negative, -1, 0))
    
    Y.store(i, sign)
```

### Example 4: Absolute Value using Bitwise Operations

```python
@ct.kernel
def abs_bitwise_kernel(X: ct.Buffer, Y: ct.Buffer):
    """Compute absolute value using bitwise operations (for integers)."""
    i = ct.tid()
    
    x = X.load(i)
    
    # For 32-bit signed integer: abs(x) = (x ^ mask) - mask
    # where mask = x >> 31 (arithmetic shift)
    mask = x >> 31
    abs_val = (x ^ mask) - mask
    
    Y.store(i, abs_val)
```

### Example 5: Min/Max of Three Values

```python
@ct.kernel
def min_of_three_kernel(
    A: ct.Buffer,
    B: ct.Buffer,
    C: ct.Buffer,
    result: ct.Buffer
):
    """Find minimum of three values element-wise."""
    i = ct.tid()
    
    a = A.load(i)
    b = B.load(i)
    c = C.load(i)
    
    # Min of three: min(min(a, b), c)
    min_ab = ct.where(a < b, a, b)
    min_abc = ct.where(min_ab < c, min_ab, c)
    
    result.store(i, min_abc)
```

### Example 6: Median of Three Values

```python
@ct.kernel
def median_of_three_kernel(
    A: ct.Buffer,
    B: ct.Buffer,
    C: ct.Buffer,
    result: ct.Buffer
):
    """Find median of three values element-wise."""
    i = ct.tid()
    
    a = A.load(i)
    b = B.load(i)
    c = C.load(i)
    
    # Median formula: a + b + c - min(a,b,c) - max(a,b,c)
    min_ab = ct.where(a < b, a, b)
    min_abc = ct.where(min_ab < c, min_ab, c)
    
    max_ab = ct.where(a > b, a, b)
    max_abc = ct.where(max_ab > c, max_ab, c)
    
    median = a + b + c - min_abc - max_abc
    
    result.store(i, median)
```

### Example 7: Range Check

```python
@ct.kernel
def in_range_kernel(
    X: ct.Buffer,
    lower: ct.Constant[float],
    upper: ct.Constant[float],
    in_range: ct.Buffer
):
    """Check if values are in [lower, upper]."""
    i = ct.tid()
    
    x = X.load(i)
    
    # Check if in range
    is_in_range = (x >= lower) & (x <= upper)
    
    in_range.store(i, is_in_range)
```

### Example 8: Binary Search Step

```python
@ct.kernel
def binary_search_step_kernel(
    array: ct.Buffer,
    target: ct.Buffer,
    indices: ct.Buffer,
    step: ct.Constant[int]
):
    """Perform one step of binary search."""
    i = ct.tid()
    
    idx = indices.load(i)
    offset = 1 << step  # 2^step
    
    # Load current and offset values
    current_val = array.load(idx)
    offset_val = array.load(idx + offset)
    target_val = target.load(i)
    
    # Decide to jump forward or stay
    should_jump = (offset_val <= target_val) & (offset_val != 0)
    new_idx = ct.where(should_jump, idx + offset, idx)
    
    indices.store(i, new_idx)
```

### Example 9: Color Thresholding

```python
@ct.kernel
def color_threshold_kernel(
    image: ct.Buffer,  # RGB image
    threshold: ct.Constant[int],
    mask: ct.Buffer
):
    """Create mask where pixel brightness > threshold."""
    i, j = ct.bid(0), ct.bid(1)
    
    # Load RGB pixel
    r = image.load((i, j, 0))
    g = image.load((i, j, 1))
    b = image.load((i, j, 2))
    
    # Compute brightness
    brightness = (r + g + b) // 3
    
    # Create mask
    is_bright = brightness > threshold
    
    mask.store((i, j), is_bright)
```

### Example 10: Bit Counting (Population Count)

```python
@ct.kernel
def popcount_kernel(X: ct.Buffer, count: ct.Buffer):
    """Count number of set bits in each integer."""
    i = ct.tid()
    
    x = X.load(i)
    
    # Parallel population count algorithm
    # First step: count pairs of bits
    x = x - ((x >> 1) & 0x55555555)
    
    # Second step: count groups of 4 bits
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    
    # Third step: count groups of 8 bits
    x = (x + (x >> 4)) & 0x0F0F0F0F
    
    # Fourth step: add all byte counts
    x = x + (x >> 8)
    x = x + (x >> 16)
    
    # Extract count (lower 8 bits)
    bit_count = x & 0x7F
    
    count.store(i, bit_count)
```

### Example 11: Swap Based on Condition

```python
@ct.kernel
def conditional_swap_kernel(
    A: ct.Buffer,
    B: ct.Buffer,
    condition: ct.Buffer
):
    """Swap A[i] and B[i] if condition[i] is True."""
    i = ct.tid()
    
    a = A.load(i)
    b = B.load(i)
    cond = condition.load(i)
    
    # Conditional swap using xor
    # If cond is True: swap, otherwise keep as-is
    new_a = ct.where(cond, b, a)
    new_b = ct.where(cond, a, b)
    
    A.store(i, new_a)
    B.store(i, new_b)
```

### Example 12: Find Indices with Multiple Conditions

```python
@ct.kernel
def multi_condition_mask_kernel(
    data: ct.Buffer,
    mask: ct.Buffer,
    min_val: ct.Constant[float],
    max_val: ct.Constant[float],
    must_be_positive: ct.Constant[bool]
):
    """Create mask with multiple conditions."""
    i = ct.tid()
    
    x = data.load(i)
    
    # Combine multiple conditions
    in_range = (x >= min_val) & (x <= max_val)
    is_positive = x > 0
    
    # Final mask
    if must_be_positive:
        final_mask = in_range & is_positive
    else:
        final_mask = in_range
    
    mask.store(i, final_mask)
```

### Example 13: Integer Division with Rounding

```python
@ct.kernel
def rounded_divide_kernel(
    numerator: ct.Buffer,
    denominator: ct.Buffer,
    result: ct.Buffer
):
    """Integer division with rounding to nearest."""
    i = ct.tid()
    
    num = numerator.load(i)
    den = denominator.load(i)
    
    # Rounded division: (num + den/2) / den
    # Equivalent to: (num * 2 + den) // (den * 2)
    adjusted_num = (num << 1) + den
    adjusted_den = den << 1
    rounded = adjusted_num // adjusted_den
    
    result.store(i, rounded)
```

### Example 14: Next Power of 2

```python
@ct.kernel
def next_power_of_2_kernel(X: ct.Buffer, result: ct.Buffer):
    """Find smallest power of 2 >= x (for x > 0)."""
    i = ct.tid()
    
    x = X.load(i)
    
    # Subtract 1 to handle exact powers of 2
    x = x - 1
    
    # Fill all bits to the right of MSB
    x |= x >> 1
    x |= x >> 2
    x |= x >> 4
    x |= x >> 8
    x |= x >> 16
    
    # Add 1 to get next power of 2
    next_pow2 = x + 1
    
    result.store(i, next_pow2)
```

### Example 15: Safe Division

```python
@ct.kernel
def safe_divide_kernel(
    numerator: ct.Buffer,
    denominator: ct.Buffer,
    result: ct.Buffer,
    default_value: ct.Constant[float] = 0.0
):
    """Division that handles divide-by-zero safely."""
    i = ct.tid()
    
    num = numerator.load(i)
    den = denominator.load(i)
    
    # Check for divide-by-zero
    is_zero = den == 0
    
    # Perform division or use default
    result_val = ct.where(is_zero, default_value, num / den)
    
    result.store(i, result_val)
```

## Summary

Bitwise and comparison operations provide powerful tools for creating masks and implementing conditional logic:

**Bitwise Operations:**
- `ct.bitwise_and(a, b)` or `a & b`: Bitwise AND
- `ct.bitwise_or(a, b)` or `a | b`: Bitwise OR
- `ct.bitwise_xor(a, b)` or `a ^ b`: Bitwise XOR
- `ct.bitwise_lshift(a, b)` or `a << b`: Left shift
- `ct.bitwise_rshift(a, b)` or `a >> b`: Right shift
- `ct.bitwise_not(a)` or `~a`: Bitwise NOT

**Comparison Operations:**
- `ct.greater(a, b)` or `a > b`: Greater than
- `ct.greater_equal(a, b)` or `a >= b`: Greater than or equal
- `ct.less(a, b)` or `a < b`: Less than
- `ct.less_equal(a, b)` or `a <= b`: Less than or equal
- `ct.equal(a, b)` or `a == b`: Equality
- `ct.not_equal(a, b)` or `a != b`: Inequality

**Selection Operations:**
- `ct.where(condition, x, y)`: Conditional element selection
- `ct.extract(tile, offset, shape)`: Extract subtile

**Key takeaways:**
1. Comparison operations return boolean masks that can be combined with bitwise operators
2. Use `ct.where()` for conditional selection without branching
3. Bitwise operations are efficient for integer manipulation and flag operations
4. Always handle edge cases like divide-by-zero with conditional logic
5. Combine multiple conditions with `&` (AND) and `|` (OR) operators

These operations are fundamental for implementing complex kernels with conditional logic and data-dependent behavior.
