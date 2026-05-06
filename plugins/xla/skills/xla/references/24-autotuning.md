# XLA Autotuning

This document provides comprehensive documentation about XLA's autotuning system, which automatically selects optimal parameters for compiled operations to maximize performance on target hardware.

## Table of Contents

- [Overview](#overview)
- [Persisted Autotuning](#persisted-autotuning)
- [Autotuning Workflow](#autotuning-workflow)
- [GPU Autotuner](#gpu-autotuner)
- [Example Autotune Result Entry](#example-autotune-result-entry)

## Overview

### Automatic Optimization Parameter Selection

XLA compiles High-Level Optimizer (HLO) programs into optimized executables for target hardware. Many compilation decisions involve tradeoffs that depend on the specific hardware, input shapes, and data types. For example:

- **Tiling parameters**: How to tile computations across the hardware's compute units.
- **Block sizes**: The dimensions of thread blocks in GPU kernels (block_m, block_n, block_k).
- **Pipeline stages**: How many pipeline stages to use for overlapping computation and memory access.
- **Warp counts**: How many warps to launch per thread block.
- **Split-K factor**: Whether to split reduction dimensions across multiple thread groups.

Manually tuning these parameters for every operation in every model is impractical. XLA's autotuning system automates this process by:

1. **Generating candidate configurations**: Creating a set of possible parameter combinations.
2. **Benchmarking each candidate**: Running the compiled operation with each configuration and measuring execution time.
3. **Selecting the best configuration**: Choosing the configuration with the lowest execution time.
4. **Caching results**: Storing the optimal configuration for reuse in future compilations.

### Triton Autotuning Parameters

For Triton-based GPU kernels (used in XLA's GPU backend for certain operations), the key autotuning parameters are:

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `BLOCK_M` | Tile size along the M dimension | 32, 64, 128, 256 |
| `BLOCK_N` | Tile size along the N dimension | 32, 64, 128, 256 |
| `BLOCK_K` | Tile size along the K dimension | 32, 64, 128 |
| `num_stages` | Number of pipeline stages for overlapping compute and memory | 2, 3, 4, 5 |
| `num_warps` | Number of warps per thread block | 2, 4, 8, 16 |
| `split_k` | Split-K factor for parallelizing reductions | 1, 2, 4, 8 |

These parameters interact in complex ways:

- Larger `BLOCK_M`, `BLOCK_N` values increase compute intensity but require more shared memory.
- Larger `BLOCK_K` values improve data reuse but increase register pressure.
- More pipeline stages (`num_stages`) overlap more memory accesses with computation but increase shared memory usage.
- More warps (`num_warps`) increase occupancy but may reduce resources per warp.
- `split_k` parallelizes reductions across thread blocks but adds synchronization overhead.

### Autotuning in the Compilation Pipeline

Autotuning occurs during the compilation phase, after HLO optimization but before final code generation:

```
HLO Module
    |
    | HLO Optimization Passes
    v
Optimized HLO Module
    |
    | Kernel Thunk Assignment (identify operations that need kernels)
    v
Operations to Compile
    |
    | Autotuning (for each operation)
    |   - Generate candidate configurations
    |   - Compile each candidate
    |   - Benchmark on target hardware
    |   - Select best configuration
    v
Optimized Executable
```

## Persisted Autotuning

### Saving and Loading Autotuning Results

Autotuning is expensive because it requires compiling and benchmarking multiple configurations for each operation. To avoid repeating this cost, XLA supports persisting autotuning results to disk and loading them in future runs.

### Autotune Results Protobuf Format

Autotuning results are stored in a protobuf format defined in XLA:

```protobuf
// xla/autotuning.proto

message AutotuneResult {
  // The device this result was generated on
  DeviceDescription device_description = 1;

  // The HLO instruction text (used for matching)
  string hlo = 2;

  // The chosen configuration
  oneof result {
    TritonGemmAutotuneResult triton_gemm_result = 3;
    CudaConvAutotuneResult cuda_conv_result = 4;
    // ... other operation types
  }

  // Performance measurement
  int64 run_time_nanos = 10;

  // Version of the autotuning format
  int32 version = 20;
}

message AutotuneResults {
  repeated AutotuneResult results = 1;
  int32 version = 2;
}

message DeviceDescription {
  string device_vendor = 1;
  string device_model = 2;
  string device_cc = 3;  // Compute capability for GPUs
  int64 device_memory = 4;  // Total device memory in bytes
}

message TritonGemmAutotuneResult {
  int32 block_m = 1;
  int32 block_n = 2;
  int32 block_k = 3;
  int32 num_stages = 4;
  int32 num_warps = 5;
  int32 split_k = 6;
}

message CudaConvAutotuneResult {
  int32 block_dim_x = 1;
  int32 block_dim_y = 2;
  int32 block_dim_z = 3;
  int32 tile_size = 4;
  int32 num_warps = 5;
  // ... additional conv-specific parameters
}
```

### Version Field

The `version` field in `AutotuneResults` ensures forward and backward compatibility:

- **Version 1**: Initial format with basic Triton GEMM results.
- **Version 2**: Added support for convolution autotuning results.
- **Version 3**: Added device description for cross-device validation.

When loading autotune results, XLA checks the version and rejects results with an incompatible version. This prevents loading results that were generated with a different autotuning format.

### Matching Logic

When loading persisted autotune results, XLA matches the current operation against stored results using:

1. **Device description**: The target device must match the device that generated the results (same vendor, model, and compute capability).

2. **HLO text**: The HLO instruction text must match. This is a string comparison of the operation, its operands' shapes, and any relevant attributes.

3. **Operation kind**: The type of operation (GEMM, convolution, etc.) must match.

If no matching result is found, XLA falls back to autotuning the operation from scratch (or using default parameters if autotuning is disabled).

## Autotuning Workflow

### Dumping Autotune Results

To generate autotune results for persistence:

```bash
XLA_FLAGS="--xla_gpu_dump_autotune_results_to=/path/to/autotune_results.pbtxt" \
    python my_model.py
```

This flag tells XLA to:
1. Run autotuning as normal during compilation.
2. After autotuning completes, serialize the results to the specified file in text protobuf format.

The output file will contain entries like:

```
results {
  device_description {
    device_vendor: "NVIDIA"
    device_model: "NVIDIA A100-SXM4-80GB"
    device_cc: "8.0"
    device_memory: 85899345920
  }
  hlo: "dot.123 = f32[4096,4096] dot(f32[4096,4096] %p0, f32[4096,4096] %p1), ..."
  triton_gemm_result {
    block_m: 128
    block_n: 128
    block_k: 32
    num_stages: 3
    num_warps: 8
    split_k: 1
  }
  run_time_nanos: 12345
  version: 3
}
```

### Loading Autotune Results

To use previously generated autotune results:

```bash
XLA_FLAGS="--xla_gpu_load_autotune_results_from=/path/to/autotune_results.pbtxt" \
    python my_model.py
```

This flag tells XLA to:
1. Load the autotune results from the specified file at startup.
2. During compilation, look up each operation in the loaded results.
3. If a matching result is found, use the stored configuration directly (no benchmarking needed).
4. If no matching result is found, fall back to default parameters or autotuning.

### Disabling Autotuning

To disable autotuning entirely (use default parameters for all operations):

```bash
XLA_FLAGS="--xla_gpu_autotune_level=0" python my_model.py
```

The autotune level values are:

| Level | Behavior |
|-------|----------|
| 0 | Autotuning disabled. Use default parameters. |
| 1 | Basic autotuning. Only autotune critical operations (GEMMs, convolutions). |
| 2 | Standard autotuning. Autotune most kernel types. |
| 3 | Aggressive autotuning. Autotune all kernel types, including elementwise and reduction operations. |

### Combining Flags

You can combine dump and load flags to update an existing autotune results file:

```bash
XLA_FLAGS="\
--xla_gpu_load_autotune_results_from=/path/to/existing_results.pbtxt \
--xla_gpu_dump_autotune_results_to=/path/to/updated_results.pbtxt" \
    python my_model.py
```

This will:
1. Load existing results for known operations.
2. Autotune any operations not covered by the loaded results.
3. Dump the complete set of results (loaded + newly generated) to the output file.

### Autotuning in Production

For production deployments, the recommended workflow is:

1. **Calibration phase**: Run the model with autotuning enabled on representative inputs:
   ```bash
   XLA_FLAGS="--xla_gpu_dump_autotune_results_to=autotune.pbtxt" \
       python my_model.py --calibration_data
   ```

2. **Deployment phase**: Use the persisted autotune results in production:
   ```bash
   XLA_FLAGS="--xla_gpu_load_autotune_results_from=autotune.pbtxt" \
       python my_model.py --production_data
   ```

This eliminates autotuning overhead in production while ensuring optimal performance.

## GPU Autotuner

### Backend-Specific Autotuning

The GPU autotuner is the primary implementation of autotuning in XLA. It handles:

1. **GEMM (dot) operations**: Autotunes Triton-based GEMM kernels for matrix multiplications.
2. **Convolution operations**: Autotunes cuDNN convolution algorithms and Triton-based convolution kernels.
3. **Reduction operations**: Autotunes reduction kernel parameters.
4. **Elementwise operations**: May autotune tiling and vectorization parameters for elementwise kernels.

### Kernel Benchmarking

The autotuner benchmarks kernels using the following process:

1. **Compile**: Generate the kernel binary for each candidate configuration.

2. **Warm up**: Run the kernel a few times to warm up the GPU (ensuring clocks are stable and caches are primed).

3. **Measure**: Run the kernel multiple times and measure execution time using GPU timers (high-precision hardware timers on the device).

4. **Statistical analysis**: Compute the median (or minimum) execution time across runs to reduce noise.

5. **Select**: Choose the configuration with the lowest measured execution time.

```cpp
// Simplified autotuner benchmarking logic
StatusOr<AutotuneResult> AutotuneGemm(const HloInstruction& gemm,
                                        StreamExecutor* executor) {
  std::vector<TritonGemmConfig> candidates = GenerateCandidates(gemm);

  AutotuneResult best_result;
  int64_t best_time_ns = INT64_MAX;

  for (const auto& config : candidates) {
    // Compile with this configuration
    auto kernel = CompileGemmKernel(gemm, config);

    // Benchmark
    auto elapsed_ns = BenchmarkKernel(kernel, executor,
                                       /*warmup_runs=*/3,
                                       /*measure_runs=*/10);

    if (elapsed_ns < best_time_ns) {
      best_time_ns = elapsed_ns;
      best_result = MakeResult(config, elapsed_ns);
    }
  }

  return best_result;
}
```

### Candidate Generation

The autotuner generates candidate configurations based on:

1. **Operation properties**: The shapes, data types, and dimensions of the operation influence which configurations are viable.

2. **Hardware constraints**: Shared memory size, register file size, maximum threads per block, and other hardware limits constrain the search space.

3. **Heuristic filtering**: Some configurations are known to perform poorly for certain operation patterns and are excluded from the search.

4. **Exhaustive or sampled search**: For small search spaces, all configurations are tested. For large search spaces, a representative sample is tested.

### Multi-GPU Considerations

When running on multiple GPUs:

- Autotuning results are per-device. Each GPU may have different optimal configurations.
- When loading persisted results, XLA matches the device description.
- For homogeneous multi-GPU setups (all same model), results can be shared across devices.
- For heterogeneous setups, each device type needs its own autotune results.

## Example Autotune Result Entry

### Complete Autotune Result for a GEMM Operation

Here is a complete example of an autotune result entry for a matrix multiplication operation:

```
results {
  # Device that generated this result
  device_description {
    device_vendor: "NVIDIA"
    device_model: "NVIDIA A100-SXM4-80GB"
    device_cc: "8.0"
    device_memory: 85899345920  # 80 GB
  }

  # The HLO instruction that was autotuned
  hlo: "dot.42 = f16[4096,4096]{1,0} dot(f16[4096,4096]{1,0} %p0, \
        f16[4096,4096]{1,0} %p1), lhs_contracting_dims={1}, \
        rhs_contracting_dims={0}, operand_precision={highest,highest}"

  # The optimal Triton configuration
  triton_gemm_result {
    block_m: 128        # 128 elements along M per tile
    block_n: 128        # 128 elements along N per tile
    block_k: 32         # 32 elements along K per tile
    num_stages: 4       # 4 pipeline stages
    num_warps: 8        # 8 warps per thread block (256 threads)
    split_k: 1          # No split-K
  }

  # Measured performance
  run_time_nanos: 12345  # 12.3 microseconds

  # Format version
  version: 3
}
```

### Interpreting the Entry

- **Device**: This result is for an NVIDIA A100 GPU with 80GB memory and compute capability 8.0.

- **HLO**: The operation is a matrix multiplication of two 4096x4096 f16 matrices. The `operand_precision={highest,highest}` indicates that both operands use the highest available precision for accumulation.

- **Configuration**:
  - `block_m: 128, block_n: 128`: Each thread block computes a 128x128 tile of the output matrix.
  - `block_k: 32`: Each iteration of the inner loop processes 32 elements along the K dimension.
  - `num_stages: 4`: The kernel uses 4 pipeline stages, meaning it overlaps 4 iterations of memory loads with computation.
  - `num_warps: 8`: Each thread block has 8 warps (256 threads), providing good occupancy on the A100's SMs.
  - `split_k: 1`: The K dimension is not split, meaning each output tile is computed by a single thread block.

- **Performance**: The kernel executes in approximately 12.3 microseconds for this matrix size.

### Autotune Result for a Convolution Operation

```
results {
  device_description {
    device_vendor: "NVIDIA"
    device_model: "NVIDIA A100-SXM4-80GB"
    device_cc: "8.0"
    device_memory: 85899345920
  }

  hlo: "convolution.123 = f16[1,224,224,64]{3,2,1,0} \
        convolution(f16[1,224,224,3]{3,2,1,0} %p0, \
                    f16[7,7,3,64]{3,2,1,0} %p1), \
        window={size=7x7 stride=2x2 pad=3_3x3_3}, \
        dim_labels=b01f_01io->b01f"

  cuda_conv_result {
    algorithm: 1          # ALGO_IMPLICIT_GEMM
    block_dim_x: 128
    block_dim_y: 8
    block_dim_z: 1
    tile_size: 8
    num_warps: 4
    split_k_factor: 1
  }

  run_time_nanos: 89432   # 89.4 microseconds
  version: 3
}
```

### Autotune Result with Split-K

```
results {
  device_description {
    device_vendor: "NVIDIA"
    device_model: "NVIDIA H100-SXM5-80GB"
    device_cc: "9.0"
    device_memory: 85899345920
  }

  # Large reduction dimension (K=8192) benefits from split-K
  hlo: "dot.78 = f32[1024,1024]{1,0} dot(f32[1024,8192]{1,0} %p0, \
        f32[8192,1024]{1,0} %p1), lhs_contracting_dims={1}, \
        rhs_contracting_dims={0}"

  triton_gemm_result {
    block_m: 64
    block_n: 64
    block_k: 64
    num_stages: 3
    num_warps: 4
    split_k: 4            # Split K across 4 thread blocks
  }

  run_time_nanos: 234567  # 234 microseconds
  version: 3
}
```

The `split_k: 4` means that the K=8192 dimension is split into 4 chunks of K=2048, with each chunk processed by a separate thread block. The partial results are then reduced to produce the final output. This can improve performance for large K dimensions by increasing parallelism, at the cost of a small reduction overhead.
