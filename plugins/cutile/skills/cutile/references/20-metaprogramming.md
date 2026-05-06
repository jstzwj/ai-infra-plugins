# 20. Metaprogramming

This chapter covers cuTile's metaprogramming features, enabling compile-time computation, code generation, and static assertions. These capabilities allow you to write more efficient, flexible, and type-safe kernels by leveraging the compiler's ability to evaluate code at compile time.

## 20.1 Compile-Time vs Runtime Execution

### 20.1.1 Understanding the Distinction

cuTile kernels execute in two distinct phases:

**Compile-Time (Host):**
- Executes on the CPU during kernel compilation
- Uses Python's full standard library and capabilities
- Results are embedded as constants in the generated GPU code
- No runtime overhead
- Must be deterministic and side-effect free

**Runtime (Device):**
- Executes on the GPU during kernel launch
- Limited to cuTile's device-side operations
- Works with dynamic data and buffers
- Subject to GPU constraints (no arbitrary Python code)

**Key Differences:**

| Aspect | Compile-Time | Runtime |
|--------|-------------|---------|
| Execution Context | CPU (Python interpreter) | GPU (Device code) |
| Available Features | Full Python standard library | cuTile device operations only |
| Performance | No runtime cost | Executed on GPU |
| Flexibility | Static, fixed at compile | Dynamic, varies per launch |
| Use Cases | Constants, code generation | Data processing |

### 20.1.2 When Values Must Be Known at Compile Time

Certain operations require compile-time constants:

1. **Array/Buffer Shape Dimensions:**
   ```python
   @ct.kernel
   def kernel(x: ct.Buffer[float]):
       # Shape must be known at compile time
       tile = ct.load(x, index=(0,), shape=(128,))  # ✓ OK
       dynamic_shape = 128  # ✓ OK (constant expression)
       
       # NOT OK: Runtime value for shape
       # tile = ct.load(x, index=(0,), shape=(runtime_dim,))
   ```

2. **Loop Unrolling Bounds:**
   ```python
   @ct.kernel
   def kernel(x: ct.Buffer[float]):
       # Unroll count must be compile-time constant
       for i in range(8):  # ✓ OK
           val = x[i]
       
       # NOT OK: Runtime loop bound
       # for i in range(runtime_count):
       #     val = x[i]
   ```

3. **Type Annotations:**
   ```python
   @ct.kernel
   def kernel(x: ct.Buffer[float], SIZE: ct.Constant[int]):
       # SIZE must be compile-time constant
       tile = ct.load(x, index=(0,), shape=(SIZE,))
   ```

## 20.2 Static Assertions

### 20.2.1 ct.static_assert()

The `ct.static_assert()` function validates conditions at compile time, ensuring invariants are enforced before the kernel runs.

```python
ct.static_assert(condition)
```

**Parameters:**
- `condition`: A compile-time boolean expression
- Can be a simple boolean value
- Can be a comparison of constants
- Can be any expression that evaluates to `bool` at compile time

**Behavior:**
- Evaluates `condition` during kernel compilation
- If `condition` is `True`: compilation proceeds normally
- If `condition` is `False`: raises `TileValueError` and halts compilation
- Error message includes the failed condition

**Purpose:**
- Validate kernel parameters at compile time
- Enforce constraints on constant values
- Provide earlier error detection than runtime checks
- Document assumptions in the code
- Enable compiler optimizations based on proven invariants

### 20.2.2 Static Assertion Examples

**Basic Parameter Validation:**

```python
@ct.kernel
def vector_add(
    x: ct.Buffer[float],
    y: ct.Buffer[float],
    result: ct.Buffer[float],
    TILE_SIZE: ct.Constant[int]
):
    # Ensure TILE_SIZE is positive
    ct.static_assert(TILE_SIZE > 0)
    
    # Ensure TILE_SIZE is power of 2
    ct.static_assert(TILE_SIZE & (TILE_SIZE - 1) == 0, 
                     "TILE_SIZE must be power of 2")
    
    # Ensure TILE_SIZE doesn't exceed reasonable limit
    ct.static_assert(TILE_SIZE <= 1024, 
                     "TILE_SIZE must be <= 1024")
    
    bid = ct.program_id(0)
    tx = ct.load(x, index=(bid * TILE_SIZE,), shape=(TILE_SIZE,))
    ty = ct.load(y, index=(bid * TILE_SIZE,), shape=(TILE_SIZE,))
    result_tile = tx + ty
    ct.store(result, index=(bid * TILE_SIZE,), tile=result_tile)
```

**Architecture-Specific Constraints:**

```python
@ct.kernel
def arch_specific_kernel(
    x: ct.Buffer[float],
    SHMEM_SIZE: ct.Constant[int]
):
    # Validate shared memory size based on architecture
    # A100: 164KB shared memory, leave room for other uses
    ct.static_assert(SHMEM_SIZE <= 160 * 1024,
                     "SHMEM_SIZE exceeds A100 capacity")
    
    # H100: 228KB shared memory
    ct.static_assert(SHMEM_SIZE <= 220 * 1024,
                     "SHMEM_SIZE exceeds H100 capacity")
    
    # Use shared memory
    # ...
```

**Type Compatibility Checks:**

```python
@ct.kernel
def typed_kernel(
    x: ct.Buffer[float],
    BLOCK_SIZE: ct.Constant[int],
    ELEMENTS_PER_THREAD: ct.Constant[int]
):
    # Ensure BLOCK_SIZE is divisible by elements per thread
    ct.static_assert(BLOCK_SIZE % ELEMENTS_PER_THREAD == 0,
                     "BLOCK_SIZE must be divisible by ELEMENTS_PER_THREAD")
    
    # Ensure total work is reasonable
    total_elements = BLOCK_SIZE * ELEMENTS_PER_THREAD
    ct.static_assert(total_elements <= 1024,
                     "Total elements per block must be <= 1024")
    
    tid = ct.tid()
    for i in range(ELEMENTS_PER_THREAD):
        idx = tid * ELEMENTS_PER_THREAD + i
        val = x[idx]
```

**Complex Compile-Time Logic:**

```python
@ct.kernel
def matrix_multiply(
    A: ct.Buffer[float],
    B: ct.Buffer[float],
    C: ct.Buffer[float],
    TILE_M: ct.Constant[int],
    TILE_N: ct.Constant[int],
    TILE_K: ct.Constant[int]
):
    # Validate tile sizes
    ct.static_assert(TILE_M > 0 and TILE_N > 0 and TILE_K > 0,
                     "All tile dimensions must be positive")
    
    # Ensure tile sizes are powers of 2 (for efficient indexing)
    ct.static_assert((TILE_M & (TILE_M - 1)) == 0,
                     "TILE_M must be power of 2")
    ct.static_assert((TILE_N & (TILE_N - 1)) == 0,
                     "TILE_N must be power of 2")
    ct.static_assert((TILE_K & (TILE_K - 1)) == 0,
                     "TILE_K must be power of 2")
    
    # Ensure tile sizes are reasonable for shared memory
    # Each tile: TILE_M * TILE_K * sizeof(float) bytes
    tile_a_bytes = TILE_M * TILE_K * 4
    tile_b_bytes = TILE_K * TILE_N * 4
    total_bytes = tile_a_bytes + tile_b_bytes
    
    ct.static_assert(total_bytes <= 48 * 1024,
                     "Tile sizes exceed shared memory capacity (48KB)")
    
    # Matrix multiplication logic
    # ...
```

## 20.3 Static Evaluation

### 20.3.1 ct.static_eval()

The `ct.static_eval()` function evaluates Python expressions at compile time using the host's Python interpreter.

```python
result = ct.static_eval(expression)
```

**Parameters:**
- `expression` (str): A string containing valid Python code

**Returns:**
- The result of evaluating the expression
- Type depends on the expression

**Behavior:**
- Executes the expression string during kernel compilation
- Has access to Python's full standard library
- Can import modules, perform calculations, call functions
- Result is embedded as a constant in the generated code
- Expression must be deterministic and side-effect free

**Use Cases:**
1. Import and use Python math constants
2. Compute complex constant expressions
3. Generate lookup tables
4. Perform compile-time calculations
5. Access host-side configuration

### 20.3.2 Static Evaluation Examples

**Mathematical Constants:**

```python
@ct.kernel
def math_kernel(x: ct.Buffer[float]):
    # Import and use mathematical constants
    PI = ct.static_eval("import math; math.pi")
    E = ct.static_eval("import math; math.e")
    
    # Use constants in computation
    tid = ct.tid()
    angle = x[tid]
    sin_val = ct.static_eval(f"import math; math.sin({angle})")
    cos_val = ct.static_eval(f"import math; math.cos({angle})")
    
    # Note: The above doesn't work as expected because angle is runtime
    # Instead, precompute at compile time:
    result = sin_val * PI
```

**Compute Complex Constants:**

```python
@ct.kernel
def lookup_table_kernel(x: ct.Buffer[float]):
    # Generate lookup table at compile time
    TABLE_SIZE = 256
    table = ct.static_eval(f"[{i} * {i} for i in range({TABLE_SIZE})]")
    
    tid = ct.tid()
    idx = tid % TABLE_SIZE
    result = table[idx]  # Use precomputed table
```

**Compile-Time Configuration:**

```python
@ct.kernel
def configured_kernel(x: ct.Buffer[float], CONFIG_PATH: ct.Constant[str]):
    # Load configuration at compile time
    tile_size = ct.static_eval(f"""
import json
with open('{CONFIG_PATH}', 'r') as f:
    config = json.load(f)
config['tile_size']
""")
    
    # Use loaded configuration
    bid = ct.program_id(0)
    tx = ct.load(x, index=(bid * tile_size,), shape=(tile_size,))
```

**Generate Specialized Code:**

```python
@ct.kernel
def specialized_kernel(x: ct.Buffer[float], N: ct.Constant[int]):
    # Generate unrolled code based on N
    # Note: This is a conceptual example
    unroll_factor = ct.static_eval(f"min({N}, 8)")
    
    # Compiler will unroll based on unroll_factor
    for i in range(unroll_factor):
        val = x[i]
```

## 20.4 Static Iteration

### 20.3.1 ct.static_iter()

The `ct.static_iter()` function creates a compile-time iterator that unrolls loops at compile time.

```python
for i in ct.static_iter(iterable):
    # Loop body is unrolled at compile time
    pass
```

**Parameters:**
- `iterable`: An iterable object (typically a range or list)

**Behavior:**
- Iterates over the provided iterable during compilation
- Unrolls the loop body for each iteration
- Each unrolled iteration can have different constant values
- Equivalent to writing out each iteration manually

**Benefits:**
1. Zero runtime loop overhead
2. Enables constant propagation in loop body
3. Allows compiler to optimize each iteration independently
4. Reduces branch divergence

### 20.4.2 Static Iteration Examples

**Basic Loop Unrolling:**

```python
@ct.kernel
def unrolled_kernel(x: ct.Buffer[float], y: ct.Buffer[float]):
    tid = ct.tid()
    
    # This loop is completely unrolled at compile time
    for i in ct.static_iter(range(4)):
        # Each iteration becomes separate code
        offset = i * 16
        tile = ct.load(x, index=(tid * 64 + offset,), shape=(16,))
        ct.store(y, index=(tid * 64 + offset,), tile=tile)
    
    # Equivalent to:
    # tile0 = ct.load(x, index=(tid * 64 + 0,), shape=(16,))
    # ct.store(y, index=(tid * 64 + 0,), tile=tile0)
    # tile1 = ct.load(x, index=(tid * 64 + 16,), shape=(16,))
    # ct.store(y, index=(tid * 64 + 16,), tile=tile1)
    # tile2 = ct.load(x, index=(tid * 64 + 32,), shape=(16,))
    # ct.store(y, index=(tid * 64 + 32,), tile=tile2)
    # tile3 = ct.load(x, index=(tid * 64 + 48,), shape=(16,))
    # ct.store(y, index=(tid * 64 + 48,), tile=tile3)
```

**Compile-Time Tile Size Specialization:**

```python
@ct.kernel
def specialized_tile_kernel(
    x: ct.Buffer[float],
    y: ct.Buffer[float],
    TILE_SIZE: ct.Constant[int]
):
    bid = ct.program_id(0)
    num_chunks = TILE_SIZE // 16
    
    # Unroll based on compile-time TILE_SIZE
    for i in ct.static_iter(range(num_chunks)):
        offset = i * 16
        tile = ct.load(
            x,
            index=(bid * TILE_SIZE + offset,),
            shape=(16,)
        )
        ct.store(
            y,
            index=(bid * TILE_SIZE + offset,),
            tile=tile
        )
```

**Generate Specialized Computation:**

```python
@ct.kernel
def polynomial_kernel(x: ct.Buffer[float], DEGREE: ct.Constant[int]):
    tid = ct.tid()
    x_val = x[tid]
    
    # Compute polynomial: a0 + a1*x + a2*x^2 + ...
    coefficients = ct.static_eval(f"[{i} for i in range({DEGREE + 1})]")
    
    result = 0.0
    for i in ct.static_iter(range(DEGREE + 1)):
        # Each iteration uses different constant coefficient
        coeff = coefficients[i]
        term = coeff * (x_val ** i)
        result = result + term
    
    x[tid] = result
```

**Multi-Dimensional Unrolling:**

```python
@ct.kernel
def stencil_kernel(
    input: ct.Buffer[float],
    output: ct.Buffer[float],
    STENCIL_SIZE: ct.Constant[int]
):
    # Get thread ID
    gid_x = ct.program_id(0) * 16 + ct.tid() % 16
    gid_y = ct.program_id(1) * 16 + ct.tid() // 16
    
    # Apply stencil by unrolling both dimensions
    result = 0.0
    for dy in ct.static_iter(range(-STENCIL_SIZE, STENCIL_SIZE + 1)):
        for dx in ct.static_iter(range(-STENCIL_SIZE, STENCIL_SIZE + 1)):
            # Each iteration accesses different offset
            val = ct.load(
                input,
                index=(gid_y + dy, gid_x + dx),
                shape=(1,)
            )
            result = result + val
    
    ct.store(output, index=(gid_y, gid_x), tile=result)
```

## 20.5 Advanced Metaprogramming Patterns

### 20.5.1 Architecture-Specific Code Generation

Generate different code paths for different GPU architectures:

```python
@ct.kernel
def adaptive_kernel(
    x: ct.Buffer[float],
    y: ct.Buffer[float],
    ARCH: ct.Constant[str]
):
    tid = ct.tid()
    
    # Validate architecture
    ct.static_assert(
        ARCH in ["sm_80", "sm_86", "sm_90"],
        f"Unsupported architecture: {ARCH}"
    )
    
    # Architecture-specific tile sizes
    if ARCH == "sm_90":
        # Hopper: Larger tiles for higher bandwidth
        TILE_SIZE = ct.static_eval("128")
    elif ARCH == "sm_80":
        # A100: Medium tiles
        TILE_SIZE = ct.static_eval("64")
    else:
        # Others: Smaller tiles
        TILE_SIZE = ct.static_eval("32")
    
    ct.static_assert(TILE_SIZE > 0, "Invalid tile size")
    
    # Use architecture-specific tile size
    tx = ct.load(x, index=(tid * TILE_SIZE,), shape=(TILE_SIZE,))
    ct.store(y, index=(tid * TILE_SIZE,), tile=tx)
```

### 20.5.2 Compile-Time Lookup Tables

Generate and use lookup tables at compile time:

```python
@ct.kernel
def trig_kernel(x: ct.Buffer[float]):
    # Generate sine lookup table at compile time
    TABLE_SIZE = 256
    sine_table = ct.static_eval(f"""
import math
[math.sin(2 * math.pi * i / {TABLE_SIZE}) for i in range({TABLE_SIZE})]
""")
    
    tid = ct.tid()
    angle = x[tid]
    
    # Normalize angle to [0, 2π)
    normalized = ct.static_eval(f"lambda x: x % (2 * 3.14159265359)")(angle)
    
    # Map to table index
    idx = ct.cast((normalized / (2 * 3.14159265359)) * TABLE_SIZE, int)
    idx = idx % TABLE_SIZE
    
    # Use precomputed value
    result = sine_table[idx]
    x[tid] = result
```

### 20.5.3 Compile-Time Loop Unrolling with Specialization

Combine static assertions, evaluation, and iteration:

```python
@ct.kernel
def optimized_convolution(
    input: ct.Buffer[float],
    output: ct.Buffer[float],
    kernel: ct.Buffer[float],
    KERNEL_SIZE: ct.Constant[int],
    TILE_SIZE: ct.Constant[int]
):
    # Validate parameters
    ct.static_assert(KERNEL_SIZE > 0, "KERNEL_SIZE must be positive")
    ct.static_assert(KERNEL_SIZE % 2 == 1, "KERNEL_SIZE must be odd")
    ct.static_assert(TILE_SIZE % KERNEL_SIZE == 0,
                     "TILE_SIZE must be divisible by KERNEL_SIZE")
    
    # Compute half kernel size
    HALF_KERNEL = KERNEL_SIZE // 2
    
    # Get global thread ID
    bid_x = ct.program_id(0)
    bid_y = ct.program_id(1)
    tid_x = ct.tid() % 16
    tid_y = ct.tid() // 16
    
    gid_x = bid_x * TILE_SIZE + tid_x
    gid_y = bid_y * TILE_SIZE + tid_y
    
    # Process tile with unrolled convolution
    for ty in ct.static_iter(range(0, TILE_SIZE, 16)):
        for tx in ct.static_iter(range(0, TILE_SIZE, 16)):
            # Compute convolution for this position
            result = 0.0
            
            # Unroll kernel loop
            for ky in ct.static_iter(range(-HALF_KERNEL, HALF_KERNEL + 1)):
                for kx in ct.static_iter(range(-HALF_KERNEL, HALF_KERNEL + 1)):
                    # Compute kernel index
                    kidx_x = kx + HALF_KERNEL
                    kidx_y = ky + HALF_KERNEL
                    kidx = kidx_y * KERNEL_SIZE + kidx_x
                    
                    # Load kernel weight
                    weight = ct.load(
                        kernel,
                        index=(kidx,),
                        shape=(1,)
                    )
                    
                    # Load input pixel
                    input_y = gid_y + ty + ky
                    input_x = gid_x + tx + kx
                    pixel = ct.load(
                        input,
                        index=(input_y, input_x),
                        shape=(1,)
                    )
                    
                    # Accumulate
                    result = result + weight * pixel
            
            # Store result
            output_y = gid_y + ty
            output_x = gid_x + tx
            ct.store(output, index=(output_y, output_x), tile=result)
```

### 20.5.4 Type-Safe Kernel Wrappers

Use metaprogramming for type safety:

```python
@ct.kernel
def type_safe_kernel(
    x: ct.Buffer[float],
    y: ct.Buffer[float],
    DTYPE_SIZE: ct.Constant[int]
):
    # Validate dtype size
    ct.static_assert(DTYPE_SIZE in [2, 4, 8],
                     "DTYPE_SIZE must be 2, 4, or 8 bytes")
    
    # Compute maximum tile size based on dtype
    MAX_TILE_SIZE = 1024 // DTYPE_SIZE
    
    tid = ct.tid()
    
    # Use safe tile size
    tx = ct.load(x, index=(tid,), shape=(MAX_TILE_SIZE,))
    ct.store(y, index=(tid,), tile=tx)
```

### 20.5.5 Compile-Time Optimization Selection

Choose algorithms at compile time:

```python
@ct.kernel
def adaptive_sort(
    data: ct.Buffer[float],
    SIZE: ct.Constant[int],
    ALGORITHM: ct.Constant[str]
):
    # Validate algorithm choice
    ct.static_assert(
        ALGORITHM in ["bubble", "insertion", "quick"],
        f"Unknown algorithm: {ALGORITHM}"
    )
    
    # Choose algorithm based on size
    if SIZE < 32:
        # Use insertion sort for small arrays
        USE_ALGORITHM = ct.static_eval('"insertion"')
    elif SIZE < 1024:
        # Use bubble sort for medium arrays
        USE_ALGORITHM = ct.static_eval('"bubble"')
    else:
        # Use quicksort for large arrays
        USE_ALGORITHM = ct.static_eval('"quick"')
    
    ct.static_assert(
        USE_ALGORITHM == ALGORITHM or ALGORITHM == "auto",
        "Algorithm mismatch"
    )
    
    # Implement sorting based on choice
    # ...
```

## 20.6 Performance Implications

### 20.6.1 Benefits of Metaprogramming

**Compile-Time Computation:**
- **Zero Runtime Cost:** Calculations performed once during compilation
- **Better Optimization:** Compiler sees all constants
- **Smaller Code:** No runtime computation logic

**Loop Unrolling:**
- **Eliminates Branch Overhead:** No loop control instructions
- **Enables SIMD:** Compiler can vectorize operations
- **Reduces Divergence:** Each iteration independent

**Static Assertions:**
- **Early Error Detection:** Catch bugs before runtime
- **Better Diagnostics:** Compile-time errors more descriptive
- **Enables Optimizations:** Compiler can prove invariants

### 20.6.2 When to Use Metaprogramming

**Use Metaprogramming When:**
- Values are truly constant (not runtime variables)
- Code generation reduces runtime overhead
- You need architecture-specific optimizations
- You want to enforce compile-time constraints
- Generating lookup tables or constants

**Avoid Metaprogramming When:**
- Values vary at runtime
- Code generation increases compile time significantly
- Runtime value is just as efficient
- Code becomes harder to understand

### 20.6.3 Compile-Time vs Runtime Trade-offs

**Example: Fixed vs Runtime Tile Size**

```python
# Compile-time (better if tile size is constant)
@ct.kernel
def compile_time_tile(x: ct.Buffer[float], TILE_SIZE: ct.Constant[int]):
    bid = ct.program_id(0)
    tx = ct.load(x, index=(bid * TILE_SIZE,), shape=(TILE_SIZE,))
    # Compiler can optimize based on known TILE_SIZE

# Runtime (more flexible, potentially slower)
@ct.kernel
def runtime_tile(x: ct.Buffer[float], tile_size: int):
    bid = ct.program_id(0)
    # Must handle variable tile size
    for i in range(tile_size):
        val = x[bid * tile_size + i]
```

**Guideline:** Use compile-time values when the value doesn't change between kernel launches. Use runtime values only when necessary.

## 20.7 Debugging Metaprogramming

### 20.7.1 Common Errors

**Error: Non-constant expression:**
```python
# WRONG: Runtime value used where constant required
@ct.kernel
def bad_kernel(x: ct.Buffer[float], runtime_size: int):
    # Error: runtime_size is not a compile-time constant
    tile = ct.load(x, index=(0,), shape=(runtime_size,))

# CORRECT: Use ct.Constant
@ct.kernel
def good_kernel(x: ct.Buffer[float], SIZE: ct.Constant[int]):
    tile = ct.load(x, index=(0,), shape=(SIZE,))
```

**Error: Static assertion failure:**
```python
# Error: TILE_SIZE must be power of 2
@ct.kernel
def invalid_kernel(x: ct.Buffer[float], TILE_SIZE: ct.Constant[int]):
    ct.static_assert(TILE_SIZE > 0 and (TILE_SIZE & (TILE_SIZE - 1)) == 0)
    # Fails if TILE_SIZE = 24 (not power of 2)
```

**Error: Invalid static_eval expression:**
```python
# WRONG: Syntax error in expression
PI = ct.static_eval("import math; math.pi")  # Missing semicolon

# WRONG: Non-deterministic expression
random_val = ct.static_eval("import random; random.random()")

# CORRECT: Valid, deterministic expression
PI = ct.static_eval("import math; math.pi")
```

### 20.7.2 Debugging Techniques

**Use Static Assertions for Validation:**
```python
@ct.kernel
def debug_kernel(x: ct.Buffer[float], PARAM: ct.Constant[int]):
    # Add assertions to catch issues early
    ct.static_assert(PARAM > 0, "PARAM must be positive")
    ct.static_assert(PARAM <= 1024, "PARAM must be <= 1024")
    ct.static_assert(PARAM % 32 == 0, "PARAM must be multiple of 32")
    
    # Now use PARAM with confidence
    tile = ct.load(x, index=(0,), shape=(PARAM,))
```

**Print Compile-Time Values:**
```python
@ct.kernel
def debug_eval_kernel(x: ct.Buffer[float]):
    # Evaluate and print at compile time
    value = ct.static_eval("print('Compile-time check'); 42")
    # value is now 42
```

**Incremental Development:**
```python
# Start simple
@ct.kernel
def simple_kernel(x: ct.Buffer[float], SIZE: ct.Constant[int]):
    ct.static_assert(SIZE > 0)
    tile = ct.load(x, index=(0,), shape=(SIZE,))

# Add complexity gradually
@ct.kernel
def complex_kernel(x: ct.Buffer[float], SIZE: ct.Constant[int]):
    ct.static_assert(SIZE > 0 and SIZE <= 1024)
    ct.static_assert(SIZE % 32 == 0)
    
    for i in ct.static_iter(range(SIZE // 32)):
        offset = i * 32
        tile = ct.load(x, index=(offset,), shape=(32,))
```

## 20.8 Summary

This chapter covered cuTile's metaprogramming capabilities:

1. **Static Assertions (`ct.static_assert`):** Validate conditions at compile time
2. **Static Evaluation (`ct.static_eval`):** Execute Python code during compilation
3. **Static Iteration (`ct.static_iter`):** Unroll loops at compile time
4. **Compile-Time vs Runtime:** Understand when to use each approach
5. **Advanced Patterns:** Architecture-specific code, lookup tables, specialized algorithms

Metaprogramming enables you to write more efficient, flexible, and type-safe kernels by leveraging compile-time computation. Use these features to optimize performance while maintaining code clarity and correctness.

Key takeaways:
- Use compile-time features when values are truly constant
- Validate assumptions with static assertions
- Generate specialized code for different architectures
- Unroll loops to eliminate runtime overhead
- Balance metaprogramming benefits against code complexity
