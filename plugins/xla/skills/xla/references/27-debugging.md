# Debugging XLA

This document provides comprehensive documentation about debugging XLA compilation and runtime issues, including HLO dump analysis, memory debugging, error codes, and determinism.

## Table of Contents

- [HLO Dumps](#hlo-dumps)
- [OOM Debugging](#oom-debugging)
- [Error Codes](#error-codes)
- [Determinism](#determinism)
- [Flags Guidance](#flags-guidance)

## HLO Dumps

### XLA_FLAGS=--xla_dump_to

The most powerful debugging tool in XLA is the HLO dump facility. It captures the state of the HLO module at every stage of compilation:

```bash
# Dump all compilation stages to a directory
XLA_FLAGS="--xla_dump_to=/tmp/xla_dump" python my_model.py
```

This creates files in the specified directory, one for each compilation stage.

### Before/After Optimization Dumps

The dump directory typically contains files like:

```
/tmp/xla_dump/
  module_0000.before_optimizations.hlo
  module_0001.after_constant_folding.hlo
  module_0002.after_algebraic_simplifier.hlo
  module_0003.after_cse.hlo
  module_0004.after_dce.hlo
  module_0005.after_fusion.hlo
  module_0006.after_layout_assignment.hlo
  module_0007.before_backend_optimizations.hlo
  module_0008.after_backend_optimizations.hlo
  module_0009.buffer_assignment.hlo
  module_0010.llvm_ir.ll
  module_0011.ptx          # GPU only
  module_0012.cubin        # GPU only
```

To compare the HLO before and after a specific pass:

```bash
# Compare before and after fusion
diff /tmp/xla_dump/module_0004.after_dce.hlo \
     /tmp/xla_dump/module_0005.after_fusion.hlo
```

### Understanding Dump File Names

Dump file names follow the pattern:

```
module_<sequence_number>.<stage_description>.<format_extension>
```

Key components:

1. **Sequence number**: Zero-padded number indicating the order in the pipeline. This makes it easy to sort files chronologically.

2. **Stage description**: A human-readable name for the compilation stage. Common stages include:
   - `before_optimizations`: The initial HLO as received from the framework.
   - `after_<pass_name>`: After a specific optimization pass.
   - `before_backend_optimizations`: Before backend-specific passes.
   - `after_backend_optimizations`: After backend-specific passes.
   - `buffer_assignment`: Shows the memory layout assignment.
   - `llvm_ir`: The generated LLVM IR.
   - `ptx`: The generated PTX (GPU only).
   - `cubin`: The compiled GPU binary (GPU only).

3. **Format extension**: Indicates the file format:
   - `.hlo` or `.txt`: Human-readable HLO text.
   - `.pb`: Binary protobuf.
   - `.ll`: LLVM IR text.
   - `.ptx`: PTX assembly text.

### Additional Dump Options

```bash
# Dump per-pass HLO (verbose, creates many files)
XLA_FLAGS="--xla_dump_to=/tmp/xla_dump --xla_dump_per_pass_hlo" python my_model.py

# Dump only HLO text (skip LLVM IR and PTX)
XLA_FLAGS="--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_as_text" python my_model.py

# Dump as short text (compact, fewer details)
XLA_FLAGS="--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_as_short_text" python my_model.py

# Dump with HTML graph visualization
XLA_FLAGS="--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_as_html" python my_model.py

# Dump module fingerprints for deduplication
XLA_FLAGS="--xla_dump_to=/tmp/xla_dump --xla_dump_module_fingerprint" python my_model.py
```

### Analyzing HLO Dumps

#### Checking Operation Count

```bash
# Count operations by type
grep -oP '\w+(?=\s*=\s*\w+\s)' /tmp/xla_dump/module_0000.before_optimizations.hlo | \
    sort | uniq -c | sort -rn
```

#### Finding Specific Operations

```bash
# Find all dot (matmul) operations
grep "dot(" /tmp/xla_dump/module_*.hlo

# Find all custom calls
grep "custom-call" /tmp/xla_dump/module_*.hlo

# Find all fusion operations
grep "fusion" /tmp/xla_dump/module_*.hlo
```

#### Checking Shapes and Layouts

```bash
# Check shapes of all parameters
grep "parameter" /tmp/xla_dump/module_0000.before_optimizations.hlo

# Check layout assignment
diff <(grep -E "parameter|ROOT" /tmp/xla_dump/module_0006.after_layout_assignment.hlo) \
     <(grep -E "parameter|ROOT" /tmp/xla_dump/module_0005.before_layout_assignment.hlo)
```

## OOM Debugging

### Memory Profiling

Out-of-memory (OOM) errors are common when working with large models on GPUs. XLA provides several tools for debugging memory issues.

#### XLA Memory Profiling

```bash
# Enable memory profiling
XLA_FLAGS="--xla_dump_to=/tmp/xla_dump --xla_dump_buffer_assignment" python my_model.py
```

The buffer assignment dump shows:
- Total allocated memory.
- Per-buffer sizes and lifetimes.
- Buffer aliasing information.

#### Python Memory Profiling

```python
# JAX memory profiling
import jax
import jax.profiler

# Get memory stats
devices = jax.devices()
for device in devices:
    stats = device.memory_stats()
    print(f"Device {device.id}:")
    print(f"  Used: {stats['bytes_in_use'] / 1e9:.2f} GB")
    print(f"  Limit: {stats['bytes_limit'] / 1e9:.2f} GB")
    print(f"  Peak: {stats.get('peak_bytes_in_use', 0) / 1e9:.2f} GB")
```

### Buffer Assignment Analysis

The buffer assignment dump provides detailed information about how XLA maps tensors to device memory:

```
# Buffer Assignment Dump (simplified)

# Total allocations: 42
# Total memory: 2.4 GB
# Peak memory: 1.8 GB

Buffer #0: 512.0 MB, f32[128, 1024, 1024]
  Allocated at: %parameter.1
  Used by: %dot.2, %add.3
  Freed after: %add.3

Buffer #1: 256.0 MB, f32[64, 1024, 1024]
  Allocated at: %dot.2
  Used by: %add.3, %multiply.4
  Freed after: %multiply.4

Buffer #2: 512.0 MB, f32[128, 1024, 1024]
  Aliased with: Buffer #0  # Memory reuse!
  Allocated at: %add.3
  Used by: %multiply.4, %dot.5
  Freed after: %dot.5
```

#### Analyzing Buffer Assignment

```bash
# Find the largest buffers
grep "Buffer #" /tmp/xla_dump/module_0009.buffer_assignment.txt | \
    sort -t',' -k2 -rn | head -20

# Find buffers that are not aliased (potential optimization targets)
grep -v "Aliased with" /tmp/xla_dump/module_0009.buffer_assignment.txt | \
    grep "Buffer #"
```

### Reducing Memory Usage

1. **Enable buffer donation**: Allow XLA to reuse input buffers for outputs.
   ```python
   # JAX: Enable donation
   @jax.jit(donate_argnums=(0, 1))
   def f(x, y):
       return jnp.dot(x, y)
   ```

2. **Use gradient checkpointing**: Reduce peak memory for large models.
   ```python
   # JAX: Gradient checkpointing
   import jax.checkpoint as checkpoint

   @checkpoint
   def layer(x, weights):
       return jnp.dot(x, weights)

   def model(x, all_weights):
       for weights in all_weights:
               x = layer(x, weights)
       return x
   ```

3. **Reduce batch size**: The most straightforward way to reduce memory usage.

4. **Use bfloat16**: Half the memory compared to float32.
   ```python
   # JAX: Use bfloat16
   x = jnp.ones((1024, 1024), dtype=jnp.bfloat16)
   ```

## Error Codes

### Overview of Error Code System

XLA uses a structured error code system that provides information about the nature and location of errors. Error codes follow a hierarchical numbering scheme:

```
XXYY
||  |
||  +-- Specific error within the category
+----- Error category
```

Categories include resource errors, compilation errors, runtime errors, memory errors, communication errors, and hardware errors.

### Common Error Codes

#### 0100, 0101, 0102: Resource Errors

These errors indicate problems with hardware resources or resource allocation.

| Code | Name | Description |
|------|------|-------------|
| 0100 | `RESOURCE_EXHAUSTED` | A resource (GPU memory, file descriptors, etc.) has been exhausted. Common causes: OOM on GPU, too many open files. |
| 0101 | `RESOURCE_UNAVAILABLE` | A requested resource is not available. Common causes: GPU is in use by another process, driver not loaded. |
| 0102 | `RESOURCE_NOT_FOUND` | A requested resource does not exist. Common causes: invalid device ordinal, missing CUDA library. |

**Troubleshooting**:
```bash
# Check GPU memory usage
nvidia-smi

# Check for zombie processes using GPU
fuser -v /dev/nvidia*

# Free GPU memory
sudo fuser -k /dev/nvidia*
```

#### 0200: Compilation Errors

| Code | Name | Description |
|------|------|-------------|
| 0200 | `COMPILATION_FAILED` | The XLA compiler failed to compile the HLO module. Common causes: unsupported operation, invalid shapes, compiler bug. |

**Troubleshooting**:
```bash
# Dump HLO to identify the problematic instruction
XLA_FLAGS="--xla_dump_to=/tmp/xla_dump" python my_model.py

# Check the dump for error indicators
grep -i "error\|failed\|unsupported" /tmp/xla_dump/*.hlo

# Try compiling with fewer optimizations
XLA_FLAGS="--xla_disable_hlo_passes=fusion" python my_model.py
```

#### 1000, 1001: Runtime Errors

| Code | Name | Description |
|------|------|-------------|
| 1000 | `RUNTIME_FAILED` | A generic runtime error occurred during execution. Common causes: kernel crash, illegal memory access. |
| 1001 | `RUNTIME_LAUNCH_FAILED` | A kernel launch failed. Common causes: invalid kernel arguments, insufficient GPU resources for launch configuration. |

**Troubleshooting**:
```bash
# Enable GPU error checking (may slow down execution)
XLA_FLAGS="--xla_gpu_check_llvm_ir" python my_model.py

# Run with CUDA error checking
CUDA_LAUNCH_BLOCKING=1 python my_model.py
```

#### 1200: Memory Errors

| Code | Name | Description |
|------|------|-------------|
| 1200 | `OUT_OF_MEMORY` | The device ran out of memory during execution. This is different from compilation-time OOM; it means the compiled executable needs more memory than available at runtime. |

**Troubleshooting**:
```bash
# Reduce batch size
# Enable gradient checkpointing
# Check for memory leaks (growing memory usage across iterations)
```

#### 2001, 2002, 2003: Communication Errors

| Code | Name | Description |
|------|------|-------------|
| 2001 | `COMMUNICATION_UNAVAILABLE` | Distributed communication infrastructure is not available. Common causes: NCCL not installed, network not configured. |
| 2002 | `COMMUNICATION_FAILED` | A distributed communication operation failed. Common causes: network timeout, peer unreachable. |
| 2003 | `COMMUNICATION_MISMATCH` | A communication operation received unexpected data. Common causes: shape mismatch between sender and receiver, different sharding configurations. |

**Troubleshooting**:
```bash
# Check NCCL configuration
nccl-test

# Enable NCCL debug logging
NCCL_DEBUG=INFO python my_model.py

# Check network connectivity
ping <peer_host>
```

#### 3000, 3001: Hardware Errors

| Code | Name | Description |
|------|------|-------------|
| 3000 | `HARDWARE_ERROR` | A hardware error occurred. Common causes: GPU thermal shutdown, PCIe error, GPU memory ECC error. |
| 3001 | `HARDWARE_UNSTABLE` | The hardware is behaving unreliably. Common causes: overheating, power supply issues, overclocking instability. |

**Troubleshooting**:
```bash
# Check GPU health
nvidia-smi -q -d ECC

# Check GPU temperature
nvidia-smi -q -d TEMPERATURE

# Run GPU stress test
gpu-burn 60
```

## Determinism

### Deterministic Execution Options

By default, XLA aims for deterministic execution, but some operations may produce non-deterministic results for performance reasons. XLA provides flags to enforce determinism.

#### Enabling Deterministic Mode

```bash
# Enable deterministic mode (may reduce performance)
XLA_FLAGS="--xla_gpu_deterministic_ops" python my_model.py
```

In JAX:
```python
# Enable deterministic flag
jax.config.update("jax_default_matmul_precision", "highest")

# Enable deterministic reduction
jax.config.update("jax_distributed_debug", True)
```

### Non-Deterministic Operations

The following operations may produce non-deterministic results when determinism is not enforced:

1. **Reductions on GPUs**: Parallel reductions may accumulate values in different orders depending on the scheduling. The result may differ by a small amount due to floating-point non-associativity.

2. **Convolutions**: cuDNN may choose different algorithms for the same convolution depending on the input. Different algorithms may produce slightly different results.

3. **Matrix multiplication**: The order of accumulation in tiled matrix multiplication can vary, leading to small floating-point differences.

4. **Random number generation**: The default PRNG may produce different sequences on different runs unless seeded explicitly.

5. **Collective operations**: The order of contributions in `all-reduce` may vary across runs, leading to floating-point differences.

#### Ensuring Determinism

```python
# JAX: Force deterministic reductions
from jax import lax

# Use deterministic reduction
result = lax.reduce(x, init_value, lax.add, dimensions=(0,))

# Force deterministic matmul precision
result = jnp.dot(x, y, precision=jax.lax.Precision.HIGHEST)
```

```bash
# Force deterministic cuDNN algorithms
XLA_FLAGS="--xla_gpu_deterministic_ops" python my_model.py

# Disable autotuning (which may select different algorithms across runs)
XLA_FLAGS="--xla_gpu_autotune_level=0" python my_model.py
```

## Flags Guidance

### Important XLA Flags

#### Compilation Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--xla_dump_to=<path>` | Dump compilation artifacts to a directory | (none) |
| `--xla_dump_hlo_as_text` | Dump HLO as text | true |
| `--xla_dump_hlo_as_proto` | Dump HLO as binary protobuf | false |
| `--xla_dump_hlo_as_short_text` | Dump HLO in short text format | false |
| `--xla_dump_per_pass_hlo` | Dump HLO after each pass (verbose) | false |
| `--xla_disable_hlo_passes=<passes>` | Disable specific HLO passes | (none) |
| `--xla_enable_hlo_passes=<passes>` | Enable specific HLO passes | (none) |
| `--xla_gpu_deterministic_ops` | Force deterministic GPU operations | false |
| `--xla_gpu_autotune_level=<level>` | Autotuning aggressiveness (0-3) | 2 |

#### GPU-Specific Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--xla_gpu_cuda_data_dir=<path>` | Path to CUDA toolkit | auto-detect |
| `--xla_gpu_num_repetitions=<n>` | Number of repetitions for benchmarking | 10 |
| `--xla_gpu_force_compilation_parallelism=<n>` | Parallel compilation threads | auto |
| `--xla_gpu_target_config_filename=<path>` | GPU target config for cross-compilation | (none) |
| `--xla_gpu_dump_autotune_results_to=<path>` | Dump autotune results | (none) |
| `--xla_gpu_load_autotune_results_from=<path>` | Load autotune results | (none) |
| `--xla_gpu_check_llvm_ir` | Verify LLVM IR during compilation | false |
| `--xla_gpu_enable_triton_gemm` | Enable Triton-based GEMM kernels | true |

#### Debug Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--xla_hlo_profile` | Enable HLO-level profiling | false |
| `--xla_gpu_per_fusion_kernel_output` | Separate output per fusion kernel | false |
| `--xla_log_hlo_text` | Log HLO text during compilation | false |
| `--xla_gpu_dump_ir_to=<path>` | Dump LLVM IR to path | (none) |
| `--xla_dump_fusion_clusters` | Dump fusion cluster information | false |
| `--xla_dump_ir` | Dump LLVM IR during compilation | false |

### Debug Options for Common Issues

#### Compilation Too Slow

```bash
# Disable autotuning
XLA_FLAGS="--xla_gpu_autotune_level=0" python my_model.py

# Disable expensive passes
XLA_FLAGS="--xla_disable_hlo_passes=fusion" python my_model.py

# Reduce compilation parallelism to reduce memory pressure
XLA_FLAGS="--xla_gpu_force_compilation_parallelism=1" python my_model.py
```

#### Numerical Differences Between Platforms

```bash
# Enable deterministic operations
XLA_FLAGS="--xla_gpu_deterministic_ops" python my_model.py

# Disable autotuning (may select different algorithms)
XLA_FLAGS="--xla_gpu_autotune_level=0" python my_model.py

# Use highest precision for matmul
# (In JAX) jax.config.update("jax_default_matmul_precision", "highest")
```

#### Crashes During Execution

```bash
# Enable synchronous execution for better error messages
CUDA_LAUNCH_BLOCKING=1 python my_model.py

# Dump HLO for post-mortem analysis
XLA_FLAGS="--xla_dump_to=/tmp/xla_dump" python my_model.py

# Check LLVM IR
XLA_FLAGS="--xla_gpu_check_llvm_ir" python my_model.py
```

#### Performance Regression

```bash
# Dump autotune results to compare
XLA_FLAGS="--xla_gpu_dump_autotune_results_to=/tmp/after.pbtxt" python my_model.py

# Compare with baseline autotune results
diff /tmp/before.pbtxt /tmp/after.pbtxt

# Dump HLO before and after optimization to identify regressions
XLA_FLAGS="--xla_dump_to=/tmp/xla_dump" python my_model.py
```
