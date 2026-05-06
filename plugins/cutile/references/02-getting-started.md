# Getting Started with cuTile

This chapter provides a comprehensive guide to installing, configuring, and using cuTile for GPU programming. We'll cover everything from system requirements to writing and optimizing your first kernels, with practical examples and troubleshooting tips throughout.

## Prerequisites and System Requirements

Before installing cuTile, ensure your system meets all the necessary hardware and software requirements.

### Hardware Requirements

**GPU:**
- **Minimum**: Any NVIDIA GPU with Compute Capability 8.0 or higher
  - RTX 3080, RTX 3080 Ti, RTX 3090 (Ampere, SM 8.6)
  - RTX 4080, RTX 4090 (Ada Lovelace, SM 8.9)
  - A100, A40, A30 (Ampere, SM 8.0)
  - H100, H200 (Hopper, SM 9.0)
  - B100, B200, GB200 (Blackwell, SM 10.0)
- **Recommended**: RTX 4080+ or A100+ for development
- **Production**: H100 or Blackwell GPUs for optimal performance

**System Memory:**
- **Minimum**: 16 GB RAM
- **Recommended**: 32 GB RAM or more
- GPU memory requirements depend on your workload size

**Storage:**
- **Minimum**: 10 GB free space for CUDA Toolkit and cuTile
- **Recommended**: 50 GB+ for development with multiple CUDA versions

### Software Requirements

**Operating System:**
- **Linux** (Primary support):
  - Ubuntu 20.04 LTS or later
  - Ubuntu 22.04 LTS (recommended)
  - Ubuntu 24.04 LTS
  - RHEL 8 or later
  - CentOS Stream 8 or later
  - Rocky Linux 8 or later
- **Windows** (Limited support):
  - Windows 10/11 with WSL2
  - Native Windows support is limited
- **macOS**: Not supported (no NVIDIA GPU support)

**Python:**
- **Supported versions**: 3.10, 3.11, 3.12, 3.13
- **Recommended**: Python 3.11 or 3.12
- **Installation**: 64-bit version required
- **Package manager**: pip 23.0 or later recommended

**NVIDIA Driver:**
- **Minimum**: r580 or later
- **Recommended**: Latest production driver
- **How to check**: `nvidia-smi` command
```bash
$ nvidia-smi
Driver Version: 560.35.03    # Should be 580.00 or higher
CUDA Version: 12.6           # Displayed CUDA version
```

**CUDA Toolkit:**
- **Minimum**: CUDA 13.1
- **Supported versions**: 13.x, 14.x
- **Required for**: nvcc compiler, runtime libraries
- **Installation**: Can be installed with cuTile or use system installation

### Verifying Your Setup

Before installing cuTile, verify your system configuration:

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA installation (if installed)
nvcc --version

# Check Python version
python --version

# Check pip version
pip --version
```

## Installation Methods

cuTile offers two installation methods depending on your needs and preferences.

### Method 1: Bundled Installation (Recommended for Development)

This method installs cuTile with bundled CUDA Toolkit dependencies, ideal for development environments where you want everything managed together.

```bash
# Install cuTile with bundled dependencies
pip install cuda-tile[tileiras]
```

**What this installs:**
- `cuda-tile`: Core cuTile package
- `cuda-tileir`: TileIR intermediate representation
- `cuda-tileiras`: Optimized IR and compiler
- CUDA runtime libraries (bundled)
- All Python dependencies

**Advantages:**
- Complete, self-contained installation
- Version compatibility guaranteed
- No system CUDA Toolkit required
- Easy to manage and upgrade

**Disadvantages:**
- Larger download size (~500 MB)
- May conflict with system CUDA installation
- Less control over CUDA version

### Method 2: System CUDA Installation (Recommended for Production)

This method uses your system's CUDA Toolkit installation, ideal for production environments where you want precise control over CUDA versions.

```bash
# Install cuTile core package only
pip install cuda-tile
```

**Requirements:**
- CUDA Toolkit 13.1+ must be installed on system
- CUDA libraries must be in PATH/LD_LIBRARY_PATH
- Compatible driver for CUDA version

**Setting up CUDA environment:**
```bash
# Add CUDA to PATH (add to ~/.bashrc or ~/.zshrc)
export PATH=/usr/local/cuda-13.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:$LD_LIBRARY_PATH

# Verify CUDA installation
nvcc --version
```

**Advantages:**
- Smaller installation size
- Uses system CUDA libraries
- Better for production deployment
- More control over CUDA version

**Disadvantages:**
- Requires manual CUDA installation
- Potential version conflicts
- More complex setup

### Version Compatibility

cuTile versions must align with CUDA Toolkit versions:

| cuTile Version | CUDA Toolkit | Driver Version |
|----------------|--------------|----------------|
| 0.1.x          | 13.1         | r580+          |
| 0.2.x          | 13.1-14.0    | r580+          |
| 0.3.x          | 14.x         | r550+          |

**Important notes:**
- Always check compatibility before installing
- Downgrading cuTile may require reinstalling dependencies
- Keep CUDA Toolkit updated for latest features

### Verifying Installation

After installation, verify cuTile is working correctly:

```bash
# Check cuTile installation
python -c "import cuda_tile as ct; print(ct.__version__)"

# Run test suite (if installed)
python -m pytest cuda_tile/tests/

# Verify GPU detection
python -c "import cuda_tile as ct; print(ct.get_device_properties())"
```

## Installing Companion Packages

cuTile integrates with several popular Python packages for GPU computing. Install these based on your workflow:

### CuPy (Recommended)

CuPy provides NumPy-compatible GPU arrays and integrates seamlessly with cuTile:

```bash
# Install CuPy for CUDA 13.x
pip install cupy-cuda13x

# Or install specific version
pip install cupy-cuda13x==13.0.0
```

**Version matching:**
- `cupy-cuda13x` for CUDA 13.x
- `cupy-cuda14x` for CUDA 14.x

### PyTorch

PyTorch tensors can be passed to cuTile kernels via CUDA Array Interface:

```bash
# Install PyTorch with CUDA support
pip install torch

# Or install specific CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### NumPy

NumPy arrays can be used with cuTile (will be transferred to GPU):

```bash
pip install numpy
```

### Development Tools

For development and testing:

```bash
# Testing framework
pip install pytest

# Code formatting
pip install black

# Type checking
pip install mypy

# Performance profiling
pip install nvsmi  # GPU monitoring
```

## Your First cuTile Kernel

Let's write your first cuTile kernel: vector addition. This example demonstrates the complete workflow from kernel definition to execution.

### Complete Vector Addition Example

Create a file named `vector_add.py`:

```python
import cuda_tile as ct
import cupy as cp
import numpy as np


@ct.kernel
def vector_add(
    a: ct.Array[float32],
    b: ct.Array[float32],
    c: ct.Array[float32],
    n: int
):
    """
    Add two vectors element-wise: c[i] = a[i] + b[i]
    
    Args:
        a: First input vector
        b: Second input vector
        c: Output vector
        n: Number of elements
    """
    # Get block index (which element this block processes)
    i = ct.bid(0)
    
    # Check bounds
    if i < n:
        # Load elements from global memory
        a_val = ct.load(a, (i,))
        b_val = ct.load(b, (i,))
        
        # Compute addition
        c_val = a_val + b_val
        
        # Store result to global memory
        ct.store(c, (i,), c_val)


def main():
    # Problem size
    n = 1024 * 1024  # 1M elements
    
    # Create input arrays on GPU using CuPy
    a = cp.random.randn(n).astype(cp.float32)
    b = cp.random.randn(n).astype(cp.float32)
    c = cp.zeros(n, dtype=cp.float32)
    
    # Create CUDA stream
    stream = ct.Stream()
    
    # Calculate grid dimensions
    # Each block processes one element in this simple example
    grid_size = ct.cdiv(n, 1)  # Ceiling division
    
    # Launch kernel
    ct.launch(
        stream=stream,
        grid=(grid_size,),
        kernel=vector_add,
        args=[a, b, c, n]
    )
    
    # Wait for kernel to complete
    stream.synchronize()
    
    # Verify results
    expected = a + b
    if cp.allclose(c, expected):
        print("✓ Vector addition successful!")
        print(f"  Processed {n} elements")
    else:
        print("✗ Results mismatch!")
        print(f"  Max error: {cp.max(cp.abs(c - expected))}")


if __name__ == "__main__":
    main()
```

### Understanding the Kernel Structure

Let's break down the key components:

#### 1. Kernel Decorator

```python
@ct.kernel
def vector_add(...):
    ...
```

The `@ct.kernel` decorator marks a function as a cuTile kernel. This decorator:
- Parses the function and validates cuTile Python subset
- Performs type inference on array arguments
- Generates TileIR intermediate representation
- Enables JIT/AOT compilation

#### 2. Type Annotations

```python
a: ct.Array[float32]
```

Type annotations specify:
- **Array**: Indicates GPU array argument
- **float32**: Element data type (float32, int32, etc.)

Type annotations are required for all array arguments and enable compiler optimizations.

#### 3. Block Index

```python
i = ct.bid(0)  # Block index in dimension 0
```

`ct.bid(dimension)` returns the block index:
- `ct.bid(0)`: Index in first dimension
- `ct.bid(1)`: Index in second dimension
- Multi-dimensional grids use multiple indices

#### 4. Load Operations

```python
a_val = ct.load(a, (i,))
```

`ct.load(array, index)` loads data from global memory:
- **array**: Source array
- **index**: Tuple of indices (one per dimension)
- **Returns**: Scalar or tile value

#### 5. Store Operations

```python
ct.store(c, (i,), c_val)
```

`ct.store(array, index, value)` stores data to global memory:
- **array**: Destination array
- **index**: Tuple of indices
- **value**: Data to store

#### 6. Launch Configuration

```python
stream = ct.Stream()
grid_size = ct.cdiv(n, 1)
ct.launch(stream, grid=(grid_size,), kernel=vector_add, args=[a, b, c, n])
```

Kernel launch parameters:
- **stream**: CUDA stream for execution ordering
- **grid**: Grid dimensions (tuple of block counts)
- **kernel**: Kernel function to execute
- **args**: Kernel arguments (arrays, scalars)

#### 7. Ceiling Division

```python
grid_size = ct.cdiv(n, 1)
```

`ct.cdiv(numerator, denominator)` computes ceiling division:
- Ensures all elements are processed
- Equivalent to: `(numerator + denominator - 1) // denominator`
- Critical for correct grid dimension calculation

### Running the Example

Execute the vector addition example:

```bash
# Run the script
python vector_add.py

# Expected output:
# ✓ Vector addition successful!
#   Processed 1048576 elements
```

## Second Example: Elementwise ReLU Operation

Let's create a more practical example implementing the ReLU (Rectified Linear Unit) activation function commonly used in neural networks.

Create `relu_example.py`:

```python
import cuda_tile as ct
import cupy as cp
import time


@ct.kernel
def relu_kernel(
    input: ct.Array[float32],
    output: ct.Array[float32],
    n: int
):
    """
    Apply ReLU activation: output[i] = max(0, input[i])
    
    Args:
        input: Input array
        output: Output array
        n: Number of elements
    """
    i = ct.bid(0)
    
    if i < n:
        # Load input value
        val = ct.load(input, (i,))
        
        # Apply ReLU: max(0, val)
        if val > 0:
            result = val
        else:
            result = 0.0
        
        # Store result
        ct.store(output, (i,), result)


def benchmark_relu(size: int, num_iterations: int = 100):
    """Benchmark ReLU kernel performance."""
    
    # Create random input data
    input_array = cp.random.randn(size).astype(cp.float32)
    output_array = cp.zeros(size, dtype=cp.float32)
    
    # Warm-up run
    stream = ct.Stream()
    grid = (ct.cdiv(size, 1),)
    ct.launch(stream, grid, relu_kernel, [input_array, output_array, size])
    stream.synchronize()
    
    # Benchmark runs
    start_time = time.time()
    for _ in range(num_iterations):
        ct.launch(stream, grid, relu_kernel, [input_array, output_array, size])
        stream.synchronize()
    end_time = time.time()
    
    # Calculate statistics
    avg_time_ms = (end_time - start_time) * 1000 / num_iterations
    bandwidth_gb_s = (size * 4 * 2) / (avg_time_ms * 1e-3) / 1e9  # Read + Write
    
    print(f"ReLU Performance (size={size}):")
    print(f"  Average time: {avg_time_ms:.3f} ms")
    print(f"  Bandwidth: {bandwidth_gb_s:.2f} GB/s")
    print(f"  Throughput: {size/1e6:.2f} M elements/sec")
    
    return avg_time_ms


def verify_correctness():
    """Verify ReLU kernel against NumPy implementation."""
    
    # Test data
    input_data = cp.array([-1.5, -0.5, 0.0, 0.5, 1.5], dtype=cp.float32)
    output_data = cp.zeros_like(input_data)
    
    # Run kernel
    stream = ct.Stream()
    grid = (ct.cdiv(len(input_data), 1),)
    ct.launch(stream, grid, relu_kernel, [input_data, output_data, len(input_data)])
    stream.synchronize()
    
    # Expected result
    expected = cp.maximum(input_data, 0)
    
    print("ReLU Correctness Test:")
    print(f"  Input:    {input_data}")
    print(f"  Output:   {output_data}")
    print(f"  Expected: {expected}")
    print(f"  Correct:  {cp.allclose(output_data, expected)}")
    
    return cp.allclose(output_data, expected)


def main():
    print("=" * 60)
    print("cuTile ReLU Example")
    print("=" * 60)
    
    # Verify correctness
    verify_correctness()
    print()
    
    # Benchmark different sizes
    sizes = [1024*1024, 10*1024*1024, 100*1024*1024]
    for size in sizes:
        benchmark_relu(size)
        print()


if __name__ == "__main__":
    main()
```

### Key Features Demonstrated

This example shows several important concepts:

**1. Conditional Logic in Kernels:**
```python
if val > 0:
    result = val
else:
    result = 0.0
```
cuTile supports conditional statements, enabling element-wise operations with branching logic.

**2. Performance Benchmarking:**
```python
start_time = time.time()
for _ in range(num_iterations):
    ct.launch(stream, grid, relu_kernel, [...])
    stream.synchronize()
end_time = time.time()
```
Standard Python timing works for GPU kernels when properly synchronized.

**3. Bandwidth Calculation:**
```python
bandwidth_gb_s = (size * 4 * 2) / (avg_time_ms * 1e-3) / 1e9
```
Calculates effective memory bandwidth considering both read and write operations.

## Third Example: Vector Reduction

Reductions are fundamental parallel operations where multiple elements are combined into a single result. Let's implement a sum reduction.

Create `reduction_example.py`:

```python
import cuda_tile as ct
import cupy as cp
import numpy as np


@ct.kernel
def reduce_sum_kernel(
    input: ct.Array[float32],
    partial_sums: ct.Array[float32],
    n: int
):
    """
    Compute partial sums of input array.
    Each block computes sum of its portion.
    
    Args:
        input: Input array
        partial_sums: Output array for partial sums
        n: Number of elements in input
    """
    # Block index determines which portion this block processes
    block_idx = ct.bid(0)
    
    # Block size (elements per block)
    block_size = 256
    
    # Start index for this block
    start_idx = block_idx * block_size
    
    # Initialize accumulator
    block_sum = 0.0
    
    # Accumulate elements in this block
    for i in range(block_size):
        global_idx = start_idx + i
        if global_idx < n:
            val = ct.load(input, (global_idx,))
            block_sum = block_sum + val
    
    # Store partial sum
    ct.store(partial_sums, (block_idx,), block_sum)


def parallel_sum(input_array: cp.ndarray) -> float:
    """
    Compute sum of array using parallel reduction.
    
    Args:
        input_array: Input array on GPU
        
    Returns:
        Sum of all elements
    """
    n = len(input_array)
    block_size = 256
    num_blocks = ct.cdiv(n, block_size)
    
    # Allocate array for partial sums
    partial_sums = cp.zeros(num_blocks, dtype=cp.float32)
    
    # Launch reduction kernel
    stream = ct.Stream()
    grid = (num_blocks,)
    ct.launch(stream, grid, reduce_sum_kernel, [input_array, partial_sums, n])
    stream.synchronize()
    
    # Final reduction on CPU (or could do another GPU pass)
    total_sum = float(cp.sum(partial_sums))
    
    return total_sum


def test_reduction():
    """Test reduction implementation."""
    
    print("Testing Parallel Reduction")
    print("=" * 60)
    
    # Test with known values
    test_cases = [
        (cp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=cp.float32), 15.0, "Small array"),
        (cp.ones(1000, dtype=cp.float32), 1000.0, "Ones array"),
        (cp.random.randn(10000).astype(cp.float32), None, "Random array"),
    ]
    
    for test_input, expected, description in test_cases:
        result = parallel_sum(test_input)
        
        if expected is not None:
            correct = np.isclose(result, expected, rtol=1e-5)
            status = "✓" if correct else "✗"
            print(f"{status} {description}:")
            print(f"  Expected: {expected}")
            print(f"  Got:      {result:.6f}")
        else:
            # Compare with NumPy
            numpy_result = float(cp.sum(test_input))
            correct = np.isclose(result, numpy_result, rtol=1e-5)
            status = "✓" if correct else "✗"
            print(f"{status} {description}:")
            print(f"  NumPy result: {numpy_result:.6f}")
            print(f"  cuTile result: {result:.6f}")
        
        print()
    
    # Performance test
    print("Performance Test:")
    sizes = [10**6, 10**7, 10**8]
    for size in sizes:
        data = cp.random.randn(size).astype(cp.float32)
        
        # Time cuTile reduction
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        result = parallel_sum(data)
        end.record()
        end.synchronize()
        elapsed_ms = cp.cuda.get_elapsed_time(start, end)
        
        print(f"  Size: {size:10d}, Time: {elapsed_ms:6.3f} ms, "
              f"Bandwidth: {size*4/elapsed_ms/1e6:6.2f} GB/s")


def main():
    test_reduction()


if __name__ == "__main__":
    main()
```

### Reduction Concepts

This example demonstrates several important reduction patterns:

**1. Block-Wise Accumulation:**
```python
block_sum = 0.0
for i in range(block_size):
    global_idx = start_idx + i
    if global_idx < n:
        val = ct.load(input, (global_idx,))
        block_sum = block_sum + val
```
Each block accumulates a partial sum independently.

**2. Partial Results:**
```python
partial_sums = cp.zeros(num_blocks, dtype=cp.float32)
ct.store(partial_sums, (block_idx,), block_sum)
```
Blocks store partial results for final combination.

**3. Multi-Pass Reduction:**
For very large arrays, you might need multiple passes:
- First pass: Reduce array to partial sums
- Second pass: Reduce partial sums to final result

## Profiling with Nsight Compute

Nsight Compute (NCU) is NVIDIA's powerful GPU profiler. Let's use it to analyze our kernels.

### Basic Profiling

```bash
# Profile vector addition kernel
ncu --set full python vector_add.py

# Profile specific kernel only
ncu --kernel-name base_name python vector_add.py

# Export to file for analysis
ncu -o profile_report python vector_add.py
```

### Useful NCU Options

```bash
# Detailed profiling
ncu --set full --force-overhead true python vector_add.py

# Focus on memory operations
ncu --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.sum \
    python vector_add.py

# Focus on compute operations
ncu --metrics sm__pipe_tensor_cycles_active.avg.pct_of_session \
    python vector_add.py

# Stalls analysis
ncu --metrics gpu__time_duration.sum,smsp__warp_issue_stalled \
    python vector_add.py
```

### Interpreting Results

Key metrics to examine:

**Memory Bandwidth:**
```
DRAM Throughput: 500 GB/s (max: 1000 GB/s)
```
Target: 80%+ of theoretical maximum

**Occupancy:**
```
Achieved Occupancy: 75%
```
Target: 70%+ for good performance

**Warp Efficiency:**
```
Warp Execution Efficiency: 98%
```
Target: 95%+ indicates good divergence handling

## Testing and Validation

Proper testing is crucial for GPU kernels. Let's set up a comprehensive testing framework.

### Creating Test Suite

Create `test_kernels.py`:

```python
import pytest
import cuda_tile as ct
import cupy as cp
import numpy as np


class TestVectorAdd:
    """Test suite for vector addition kernel."""
    
    @pytest.fixture
    def kernel(self):
        """Import and return the kernel."""
        from vector_add import vector_add
        return vector_add
    
    @pytest.fixture
    def stream(self):
        """Create CUDA stream for testing."""
        return ct.Stream()
    
    def test_small_arrays(self, kernel, stream):
        """Test with small arrays."""
        n = 128
        a = cp.random.randn(n).astype(cp.float32)
        b = cp.random.randn(n).astype(cp.float32)
        c = cp.zeros(n, dtype=cp.float32)
        
        grid = (ct.cdiv(n, 1),)
        ct.launch(stream, grid, kernel, [a, b, c, n])
        stream.synchronize()
        
        expected = a + b
        assert cp.allclose(c, expected, rtol=1e-5)
    
    def test_large_arrays(self, kernel, stream):
        """Test with large arrays."""
        n = 10 * 1024 * 1024
        a = cp.random.randn(n).astype(cp.float32)
        b = cp.random.randn(n).astype(cp.float32)
        c = cp.zeros(n, dtype=cp.float32)
        
        grid = (ct.cdiv(n, 1),)
        ct.launch(stream, grid, kernel, [a, b, c, n])
        stream.synchronize()
        
        expected = a + b
        assert cp.allclose(c, expected, rtol=1e-5)
    
    def test_edge_cases(self, kernel, stream):
        """Test edge cases."""
        # Single element
        a = cp.array([1.0], dtype=cp.float32)
        b = cp.array([2.0], dtype=cp.float32)
        c = cp.zeros(1, dtype=cp.float32)
        
        grid = (1,)
        ct.launch(stream, grid, kernel, [a, b, c, 1])
        stream.synchronize()
        
        assert cp.allclose(c, [3.0], rtol=1e-5)
    
    def test_zeros(self, kernel, stream):
        """Test with zero arrays."""
        n = 1024
        a = cp.zeros(n, dtype=cp.float32)
        b = cp.zeros(n, dtype=cp.float32)
        c = cp.zeros(n, dtype=cp.float32)
        
        grid = (ct.cdiv(n, 1),)
        ct.launch(stream, grid, kernel, [a, b, c, n])
        stream.synchronize()
        
        assert cp.allclose(c, 0.0, rtol=1e-5)
    
    def test_performance(self, kernel, stream):
        """Test performance meets minimum threshold."""
        import time
        
        n = 1024 * 1024
        a = cp.random.randn(n).astype(cp.float32)
        b = cp.random.randn(n).astype(cp.float32)
        c = cp.zeros(n, dtype=cp.float32)
        
        grid = (ct.cdiv(n, 1),)
        
        # Warm-up
        for _ in range(10):
            ct.launch(stream, grid, kernel, [a, b, c, n])
        stream.synchronize()
        
        # Timed run
        start = time.time()
        for _ in range(100):
            ct.launch(stream, grid, kernel, [a, b, c, n])
        stream.synchronize()
        elapsed = time.time() - start
        
        # Should process > 1M elements per second
        throughput = n * 100 / elapsed
        assert throughput > 1e6, f"Performance too low: {throughput:.0f} elements/sec"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### Running Tests

```bash
# Run all tests
pytest test_kernels.py -v

# Run specific test
pytest test_kernels.py::TestVectorAdd::test_small_arrays -v

# Run with coverage
pytest test_kernels.py --cov=vector_add --cov-report=html

# Run with performance profiling
pytest test_kernels.py -k performance --durations=10
```

## Common Pitfalls and Troubleshooting

### Common Issues

**1. Incorrect Grid Dimensions**

```python
# WRONG: May miss elements
grid = (n // block_size,)

# CORRECT: Ceiling division
grid = (ct.cdiv(n, block_size),)
```

**2. Forgetting Synchronization**

```python
# WRONG: Results may not be ready
ct.launch(stream, grid, kernel, args)
result = output_array.copy()  # May contain old data!

# CORRECT: Wait for kernel completion
ct.launch(stream, grid, kernel, args)
stream.synchronize()
result = output_array.copy()  # Guaranteed complete
```

**3. Type Mismatches**

```python
# WRONG: Array types don't match kernel
a = cp.array([1, 2, 3], dtype=cp.float64)  # float64
# Kernel expects float32

# CORRECT: Match types to kernel
a = cp.array([1, 2, 3], dtype=cp.float32)  # float32
```

**4. Array Aliasing**

```python
# WRONG: Input and output overlap
a = cp.array([1, 2, 3, 4])
ct.launch(stream, grid, kernel, [a, a, c])  # Undefined behavior!

# CORRECT: Use separate arrays
a = cp.array([1, 2, 3, 4])
b = a.copy()  # Create separate copy
ct.launch(stream, grid, kernel, [a, b, c])
```

**5. Missing Bounds Checks**

```python
# WRONG: May access out of bounds
i = ct.bid(0) * block_size + ct.tid(0)
val = ct.load(a, (i,))  # May exceed array size!

# CORRECT: Check bounds
i = ct.bid(0) * block_size + ct.tid(0)
if i < n:
    val = ct.load(a, (i,))
```

### Debugging Tips

**1. Enable Verbose Output**

```python
import cuda_tile as ct
ct.set_log_level(ct.LogLevel.DEBUG)
```

**2. Use Small Test Cases**

```python
# Start with small, verifiable cases
test_n = 128  # Not 1M!
a = cp.arange(test_n, dtype=cp.float32)
```

**3. Verify with Reference Implementation**

```python
# Compare against known-good implementation
my_result = my_kernel(a)
expected = cp.linalg.norm(a)
assert cp.allclose(my_result, expected)
```

**4. Check GPU Properties**

```python
import cuda_tile as ct
props = ct.get_device_properties()
print(f"Device: {props['name']}")
print(f"Compute Capability: {props['compute_capability']}")
print(f"Memory: {props['total_memory'] / 1e9:.2f} GB")
```

## Environment Setup Tips

### Virtual Environment Setup

```bash
# Create virtual environment
python -m venv cutile_env
source cutile_env/bin/activate  # On Windows: cutile_env\Scripts\activate

# Install cuTile
pip install --upgrade pip
pip install cuda-tile[tileiras]

# Install development tools
pip install pytest black mypy jupyter
```

### Jupyter Notebook Setup

```bash
# Install Jupyter with GPU support
pip install jupyter cupy-cuda13x

# Launch Jupyter
jupyter notebook

# In notebook:
import cuda_tile as ct
import cupy as cp
```

### IDE Configuration

**VS Code:**
```json
{
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "python.analysis.typeCheckingMode": "basic"
}
```

**PyCharm:**
- Enable type checking in Settings
- Configure pytest as test runner
- Set up remote interpreter for GPU systems

## Best Practices

### 1. Always Use Type Annotations

```python
# GOOD
@ct.kernel
def my_kernel(a: ct.Array[float32], b: ct.Array[float32]):
    ...

# AVOID
@ct.kernel
def my_kernel(a, b):  # No type info
    ...
```

### 2. Validate Input Sizes

```python
def run_kernel(a, b):
    assert a.shape == b.shape, "Array shapes must match"
    assert a.dtype == b.dtype, "Array dtypes must match"
    ...
```

### 3. Use Context Managers for Streams

```python
# Stream is automatically cleaned up
with ct.Stream() as stream:
    ct.launch(stream, grid, kernel, args)
    # Auto-synchronize on exit
```

### 4. Profile Before Optimizing

```bash
# Always profile first
ncu --set full python my_kernel.py

# Identify bottlenecks
# Then optimize what matters
```

### 5. Keep Kernels Simple

```python
# GOOD: Single, clear operation
@ct.kernel
def add_arrays(a, b, c):
    c[i] = a[i] + b[i]

# AVOID: Multiple complex operations
@ct.kernel
def do_everything(a, b, c, d, e):
    # 100 lines of complex logic
    ...
```

## Conclusion

This chapter covered the essential steps to get started with cuTile, from installation and setup to writing, testing, and optimizing your first kernels. The examples demonstrated key concepts like kernel structure, memory operations, and parallel reduction patterns.

Key takeaways:
- Use `pip install cuda-tile[tileiras]` for development
- Always include type annotations in kernel signatures
- Use `ct.cdiv()` for calculating grid dimensions
- Remember to synchronize streams before reading results
- Profile with Nsight Compute before optimizing
- Write comprehensive tests to verify correctness

With these fundamentals, you're ready to explore more advanced cuTile features and tackle complex GPU computing problems. The next chapter will dive deep into cuTile's data model and array operations.