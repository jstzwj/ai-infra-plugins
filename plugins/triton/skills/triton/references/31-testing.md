# Chapter 31: Testing

## Test Structure

```
test/                   # MLIR lit tests
├── Triton/             # Triton dialect tests
├── TritonGPU/          # TritonGPU dialect tests
├── Conversion/         # Conversion pass tests
├── Analysis/           # Analysis pass tests
├── Gluon/              # Gluon dialect tests
├── Hopper/             # Hopper-specific tests
└── Proton/             # Proton profiler tests

python/test/            # Python tests
├── unit/
│   ├── language/       # Language unit tests
│   ├── runtime/        # Runtime unit tests
│   ├── cuda/           # CUDA-specific tests
│   ├── instrumentation/# Instrumentation tests
│   ├── plugins/        # Plugin tests
│   └── tools/          # Tool tests
├── backend/            # Backend tests
├── gluon/              # Gluon tests
├── gsan/               # GSAN tests
├── kernel_comparison/  # Kernel comparison tests
├── microbenchmark/     # Microbenchmarks
└── regression/         # Regression tests

unittest/               # C++ unit tests
├── Analysis/           # Analysis tests
├── Dialect/            # Dialect tests
└── Tools/              # Tool tests
```

## Running Tests

### pytest Tests

```bash
# Run all Python tests
make test

# Run specific test file
pytest python/test/unit/language/test_core.py -s --tb=short

# Run specific test function
pytest python/test/unit/language/test_core.py::test_add -s --tb=short

# Run with keyword filter
pytest python/test/unit/ -k "test_softmax" -s

# Run tests without GPU
make test-nogpu

# Run in interpreter mode
TRITON_INTERPRET=1 pytest python/test/unit/language/test_core.py
```

### MLIR lit Tests

```bash
# Build triton-opt
cd BUILD_DIR && ninja triton-opt

# Run specific lit test
lit -v test/TritonGPU/some_test.mlir

# Run all lit tests in a directory
lit -v test/Triton/
```

### C++ Unit Tests

```bash
cd BUILD_DIR
ninja triton-unittest
./unittest/triton-unittest
```

## Testing Utilities

### `triton.testing`

```python
from triton import testing

# Benchmark functions
ms = testing.do_bench(fn, warmup=25, rep=100)
ms = testing.do_bench_cudagraph(fn, rep=20)
ms = testing.do_bench_proton(fn, warmup=25, rep=100)

# Assert numerical closeness
testing.assert_close(actual, expected, atol=1e-2, rtol=0)

# Check hardware capabilities
testing.is_cuda()
testing.is_hip()
testing.is_hopper()
testing.is_blackwell()
testing.is_ampere_or_newer()

# TMA support check
testing.supports_tma()
testing.requires_tma  # pytest mark

# Random tensor generation
x = testing.numpy_random((128, 128), "float32")

# Conversion utilities
t = testing.to_triton(numpy_array, device="cuda")
n = testing.to_numpy(triton_tensor)
```

### Performance Reporting

```python
@testing.perf_report([
    testing.Benchmark(
        x_names=["N"],
        x_vals=[128, 256, 512, 1024, 2048],
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "PyTorch"],
        plot_name="vector-add-performance",
        args={"M": 1024},
        xlabel="N",
        ylabel="ms",
    )
])
def benchmark(N, M, provider):
    x = torch.randn(N, M, device="cuda")
    if provider == "triton":
        ms = testing.do_bench(lambda: triton_add(x))
    else:
        ms = testing.do_bench(lambda: x + x)
    return ms

# Run and display results
benchmark.run(print_data=True)
```

### Hardware Detection

```python
# GPU architecture checks
testing.is_cuda()           # Running on CUDA
testing.is_hip()            # Running on HIP
testing.is_ampere_or_newer()  # CC >= 8.0
testing.is_hopper_or_newer()  # CC >= 9.0
testing.is_hopper()          # CC == 9.0
testing.is_blackwell()       # CC == 10.x or 11.x
testing.is_hip_cdna3()       # gfx942
testing.is_hip_cdna4()       # gfx950
```

### Data Types

```python
# Predefined dtype lists
testing.int_dtypes         # ['int8', 'int16', 'int32', 'int64']
testing.uint_dtypes        # ['uint8', 'uint16', 'uint32', 'uint64']
testing.float_dtypes       # ['float16', 'float32', 'float64']
testing.dtypes             # All standard dtypes
testing.tma_dtypes         # TMA-compatible dtypes
```

## FileCheck Tests

```python
from triton._filecheck import filecheck_test

@filecheck_test
@triton.jit
def test_my_kernel():
    # CHECK: tt.func @test_my_kernel
    # CHECK: tt.load
    # CHECK: tt.store
    x = tl.load(ptr)
    tl.store(ptr, x + 1)
```

## Process Isolation

```python
from triton.testing import run_in_process

# Run test in subprocess (for crash isolation)
result = run_in_process(
    my_test_fn,
    args=(arg1, arg2),
    env={"CUDA_VISIBLE_DEVICES": "0"},
)
```

## Continuous Integration

Triton uses GitHub Actions for CI:
- Build verification
- Lit test execution
- pytest execution
- Documentation build
- Wheel packaging
