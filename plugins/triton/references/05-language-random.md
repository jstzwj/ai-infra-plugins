# Chapter 5: Language Random Module (`triton.language.random`)

Triton provides random number generation using the **Philox** counter-based PRNG algorithm, which is well-suited for parallel GPU execution.

## Philox PRNG

### `tl.philox(seed, c0, c1, c2, c3, n_rounds=10) -> tuple[int32, int32, int32, int32]`

Core Philox pseudo-random number generator. Takes a seed and 4 counter values, returns 4 updated int32 values.

```python
@triton.jit
def kernel(seed, offsets, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_offs < n

    # Generate random integers using Philox
    r0, r1, r2, r3 = tl.philox(seed, block_offs, 0, 0, 0)
    tl.store(output_ptr + block_offs * 4 + 0, r0, mask=mask)
    tl.store(output_ptr + block_offs * 4 + 1, r1, mask=mask)
    tl.store(output_ptr + block_offs * 4 + 2, r2, mask=mask)
    tl.store(output_ptr + block_offs * 4 + 3, r3, mask=mask)
```

**Parameters:**
- `seed` (int32/int64): Random seed value
- `c0, c1, c2, c3` (tensor of int32/int64): Counter values (often block offsets)
- `n_rounds` (int, constexpr): Number of Philox rounds (default: 10)

**Returns:** Tuple of 4 int32 tensors

## Uniform Random

### `tl.rand(seed, offset, n_rounds=10) -> tensor`
Generate uniform random floats in [0, 1).

```python
@triton.jit
def dropout_kernel(x_ptr, out_ptr, n, p, seed, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n

    x = tl.load(x_ptr + offs, mask=mask)
    random = tl.rand(seed, offs)           # U(0, 1)
    keep = random > p
    out = tl.where(keep, x / (1 - p), 0.0)
    tl.store(out_ptr + offs, out, mask=mask)
```

### `tl.rand4x(seed, offset, n_rounds=10) -> tuple`
Generate 4 uniform random floats. Most efficient when you need multiple random values.

## Normal Random

### `tl.randn(seed, offset, n_rounds=10) -> tensor`
Generate standard normal random floats N(0, 1) using Box-Muller transform.

```python
@triton.jit
def init_kernel(ptr, n, seed, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    # Initialize with normal distribution
    val = tl.randn(seed, offs)  # N(0, 1)
    tl.store(ptr + offs, val, mask=mask)
```

### `tl.randn4x(seed, offset, n_rounds=10) -> tuple`
Generate 4 standard normal random floats.

## Integer Random

### `tl.randint(seed, offset, n_rounds=10) -> tensor`
Generate random int32 value.

### `tl.randint4x(seed, offset, n_rounds=10) -> tuple`
Generate 4 random int32 values. Most efficient entry point to Philox.

## Utility Functions

### `tl.uint_to_uniform_float(x) -> tensor`
Convert unsigned integer to uniform float in [0, 1).

```python
# Manual conversion
int_val = tl.randint(seed, offs)
float_val = tl.uint_to_uniform_float(int_val)  # Maps to U(0,1)
```

### `tl.pair_uniform_to_normal(u1, u2) -> tuple`
Box-Muller transform: convert two uniform random numbers to two standard normal numbers.

```python
# Manual normal generation
u1 = tl.rand(seed, offs)
u2 = tl.rand(seed, offs + 1)
n1, n2 = tl.pair_uniform_to_normal(u1, u2)  # Both N(0,1)
```

## Seeded Dropout Pattern

A common pattern using RNG for memory-efficient dropout:

```python
@triton.jit
def seeded_dropout_kernel(
    x_ptr, output_ptr, n_elements, p, seed,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask)

    # Generate random mask using seed (no need to store mask!)
    random = tl.rand(seed, offsets)
    keep_mask = random > p

    # Apply dropout with scaling
    output = tl.where(keep_mask, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)

# Forward pass: use seed=42
seeded_dropout_kernel[grid](x, y, n, p=0.1, seed=42, BLOCK_SIZE=1024)

# Backward pass: use SAME seed to reconstruct same mask
seeded_dropout_kernel[grid](grad_x, grad_y, n, p=0.1, seed=42, BLOCK_SIZE=1024)
```

**Benefits of seeded approach:**
- No need to store dropout mask (saves memory)
- Deterministic: same seed produces same mask
- Efficient: only requires storing a single int32 seed
