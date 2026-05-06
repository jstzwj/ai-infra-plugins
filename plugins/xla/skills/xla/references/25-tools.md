# XLA Tools

This document provides comprehensive documentation about the tools available for working with XLA, including HLO module runners, optimizers, and debugging utilities.

## Table of Contents

- [run_hlo_module](#run_hlo_module)
- [multihost_hlo_runner (hlo_runner_main)](#multihost_hlo_runner)
- [hlo-opt](#hlo-opt)
- [ptx-opt](#ptx-opt)
- [isolate_hlo](#isolate_hlo)
- [Getting HLO Dumps](#getting-hlo-dumps)

## run_hlo_module

`run_hlo_module` is a command-line tool for compiling and executing HLO modules. It is useful for testing, debugging, and benchmarking individual HLO computations.

### Usage

```bash
bazel run //xla/tools:run_hlo_module -- [flags] <hlo_file>
```

### Key Flags

#### --platform

Specifies the target platform for execution:

```bash
# Run on CPU
bazel run //xla/tools:run_hlo_module -- \
    --platform=CPU path/to/module.hlo

# Run on GPU
bazel run //xla/tools:run_hlo_module -- \
    --platform=CUDA path/to/module.hlo
```

Available platforms:
- `CPU`: Execute on the host CPU.
- `CUDA`: Execute on an NVIDIA GPU.
- `ROCM`: Execute on an AMD GPU.
- `TPU`: Execute on a Google TPU.

#### --reference_platform

Specifies a reference platform for correctness checking. When set, `run_hlo_module` executes the module on both the target platform and the reference platform and compares the results:

```bash
# Run on GPU, verify against CPU
bazel run //xla/tools:run_hlo_module -- \
    --platform=CUDA \
    --reference_platform=CPU \
    path/to/module.hlo
```

This is extremely useful for:
- Verifying new backend implementations.
- Debugging numerical differences between platforms.
- Regression testing after compiler changes.

The comparison uses configurable tolerances for floating-point values:
- `--relaxed_tolerance`: Use relaxed tolerance for floating-point comparison.
- `--f32_comparison_bit_error`: Maximum allowed bit error for f32 values.

#### --input_format

Specifies the input format:

```bash
# Text HLO format (default)
bazel run //xla/tools:run_hlo_module -- \
    --input_format=text path/to/module.hlo

# Binary protobuf format
bazel run //xla/tools:run_hlo_module -- \
    --input_format=proto path/to/module.pb

# Binary protobuf with Snappy compression
bazel run //xla/tools:run_hlo_module -- \
    --input_format=snappy path/to/module.snappy.pb
```

#### Running Multiple Modules

To run multiple HLO modules sequentially:

```bash
bazel run //xla/tools:run_hlo_module -- \
    --platform=CUDA \
    module1.hlo module2.hlo module3.hlo
```

Each module is compiled and executed independently. This is useful for running a batch of test cases.

### Example Usage

```bash
# Run a simple HLO module on GPU with CPU reference
cat > add.hlo << 'EOF'
HloModule add_module

ENTRY main {
  %p0 = f32[4] parameter(0)
  %p1 = f32[4] parameter(1)
  ROOT %add = f32[4] add(%p0, %p1)
}
EOF

bazel run //xla/tools:run_hlo_module -- \
    --platform=CUDA \
    --reference_platform=CPU \
    add.hlo
```

### Output

The tool outputs:
1. **Compilation status**: Whether the module compiled successfully.
2. **Execution status**: Whether execution completed without errors.
3. **Results**: The output values (for small tensors).
4. **Timing**: Compilation and execution time.
5. **Comparison result**: If a reference platform is specified, whether the outputs match.

## multihost_hlo_runner (hlo_runner_main)

`multihost_hlo_runner` (also known as `hlo_runner_main`) is a tool for running HLO modules in a multi-host (distributed) setting. It supports SPMD (Single Program, Multiple Data) execution across multiple hosts.

### SPMD Support

The multihost runner supports XLA's SPMD mode, where the same HLO module runs on all hosts with sharded inputs and outputs:

```bash
# Run on 4 hosts with SPMD
bazel run //xla/tools:multihost_hlo_runner -- \
    --num_hosts=4 \
    --host_addresses=host1:1234,host2:1234,host3:1234,host4:1234 \
    --spmd \
    path/to/spmd_module.hlo
```

### Cross-Host Communication

The runner handles cross-host communication for distributed operations:

- **Collective operations**: `all-reduce`, `all-gather`, `all-to-all`, `collective-permute`.
- **Send/recv**: Point-to-point communication between hosts.
- **SPMD partitioning**: Automatically partitions the computation based on sharding annotations.

### Usage

```bash
# Start the coordinator on host1
bazel run //xla/tools:multihost_hlo_runner -- \
    --mode=coordinator \
    --num_hosts=2 \
    --port=1234 \
    path/to/module.hlo

# Start the worker on host2
bazel run //xla/tools:multihost_hlo_runner -- \
    --mode=worker \
    --coordinator_address=host1:1234 \
    path/to/module.hlo
```

### Flags

| Flag | Description |
|------|-------------|
| `--num_hosts` | Number of participating hosts |
| `--host_addresses` | Comma-separated list of host:port pairs |
| `--mode` | `coordinator` or `worker` |
| `--port` | Port for the coordinator |
| `--coordinator_address` | Address of the coordinator (for workers) |
| `--spmd` | Enable SPMD mode |
| `--platform` | Target platform (CPU, CUDA, etc.) |
| `--input_format` | Input format (text, proto, snappy) |

### Example: SPMD All-Reduce

```
HloModule spmd_all_reduce

ENTRY main {
  %p0 = f32[1024] parameter(0), sharding={replicated}
  %all_reduce = f32[1024] all-reduce(%p0), replica_groups={{0,1,2,3}},
      to_apply=add
  ROOT %root = f32[1024] add(%all_reduce, %p0)
}
```

```bash
# Run across 4 GPUs
bazel run //xla/tools:multihost_hlo_runner -- \
    --num_hosts=4 \
    --spmd \
    --platform=CUDA \
    spmd_all_reduce.hlo
```

## hlo-opt

`hlo-opt` is the primary tool for analyzing, transforming, and debugging HLO modules. It provides access to XLA's entire compilation pipeline and individual optimization passes.

### Compilation Stages

#### --list-stages

List all available compilation stages:

```bash
bazel run //xla/tools:hlo-opt -- --list-stages
```

Output includes stages like:
```
buffer-assignment
hlo
hlo-backend
llvm
ptx
...
```

#### --stage

Run the compilation pipeline up to a specific stage and dump the output:

```bash
# Dump the HLO after all optimization passes
bazel run //xla/tools:hlo-opt -- \
    --stage=hlo \
    --platform=CUDA \
    input.hlo

# Dump the LLVM IR
bazel run //xla/tools:hlo-opt -- \
    --stage=llvm \
    --platform=CUDA \
    input.hlo

# Dump the PTX
bazel run //xla/tools:hlo-opt -- \
    --stage=ptx \
    --platform=CUDA \
    input.hlo
```

#### Stage Types

| Stage | Description |
|-------|-------------|
| `hlo` | HLO after target-independent optimization |
| `hlo-backend` | HLO after backend-specific optimization |
| `buffer-assignment` | Buffer assignment (memory layout) |
| `llvm` | LLVM IR after code generation |
| `ptx` | PTX assembly (GPU only) |
| `cubin` | Compiled GPU binary |

### Pass Development

#### --passes

Run specific optimization passes:

```bash
# Run algebraic simplifier only
bazel run //xla/tools:hlo-opt -- \
    --passes=algebraic-simplifier \
    input.hlo

# Run multiple passes
bazel run //xla/tools:hlo-opt -- \
    --passes=algebraic-simplifier,cse,dce \
    input.hlo

# Run a custom pass pipeline
bazel run //xla/tools:hlo-opt -- \
    --passes=fusion,algebraic-simplifier,dce \
    input.hlo
```

#### --list-passes

List all available passes:

```bash
bazel run //xla/tools:hlo-opt -- --list-passes
```

Output includes passes like:
```
algebraic-simplifier
all-reduce-simplifier
batchnorm-expander
broadcast-simplifier
cholesky_expander
conditional-simplifier
constant_folding
conv-grad-var-update-expander
copy-removal
cse
dce
defuser
dot-merger
dot_merger
dynamic-index-split
fft-expander
fusion
hlo-verification
layout-assignment
log-merger
multi_output_fusion
reshape-mover
scatter-expander
sort-expander
topk_splitter
triangular-solve-expander
while-loop-constant-sink
while-loop-simplifier
...
```

### Custom Pipelines

You can create custom optimization pipelines by combining passes:

```bash
# Aggressive fusion pipeline
bazel run //xla/tools:hlo-opt -- \
    --passes=dot-merger,algebraic-simplifier,cse,fusion,layout-assignment \
    input.hlo

# Debugging pipeline (minimal optimization, preserve structure)
bazel run //xla/tools:hlo-opt -- \
    --passes=dce,constant_folding \
    input.hlo
```

### Format Conversion

#### --emit-proto

Convert between HLO text and protobuf formats:

```bash
# Convert text to binary protobuf
bazel run //xla/tools:hlo-opt -- \
    --emit-proto \
    input.hlo > output.pb

# Convert binary protobuf to text
bazel run //xla/tools:hlo_opt -- \
    input.pb > output.hlo
```

### Deviceless GPU Compilation

#### --xla_gpu_target_config_filename

Compile for a specific GPU without needing the GPU present. This is useful for:
- Cross-compilation (building on a machine without a GPU).
- CI/CD pipelines.
- Reproducing compilation issues on different hardware.

```bash
# Compile for A100 without an A100 present
bazel run //xla/tools:hlo-opt -- \
    --platform=CUDA \
    --xla_gpu_target_config_filename=a100_config.pbtxt \
    input.hlo
```

The target config file is a text protobuf containing GPU device information:

```
# a100_config.pbtxt
device_description {
  device_vendor: "NVIDIA"
  device_model: "NVIDIA A100-SXM4-80GB"
  device_memory_limit: 85899345920
  gpu_compute_capability {
    major: 8
    minor: 0
  }
}
```

#### GPU Specs for Popular GPUs

| GPU | Compute Capability | Target Config Name |
|-----|-------------------|-------------------|
| NVIDIA A100 | 8.0 | `sm_80` |
| NVIDIA A100 (80GB) | 8.0 | `sm_80` |
| NVIDIA H100 | 9.0 | `sm_90` |
| NVIDIA H200 | 9.0 | `sm_90` |
| NVIDIA L40S | 8.9 | `sm_89` |
| NVIDIA RTX 4090 | 8.9 | `sm_89` |
| NVIDIA V100 | 7.0 | `sm_70` |
| NVIDIA T4 | 7.5 | `sm_75` |
| AMD MI250 | gfx90a | `gfx90a` |
| AMD MI300X | gfx942 | `gfx942` |

### Autotuning with hlo-opt

The `hlo-opt` tool supports autotuning directly:

```bash
# Autotune GEMM operations in the HLO module
bazel run //xla/tools:hlo-opt -- \
    --platform=CUDA \
    --xla_gpu_autotune_level=2 \
    --xla_gpu_dump_autotune_results_to=autotune.pbtxt \
    input.hlo

# Use persisted autotune results
bazel run //xla/tools:hlo-opt -- \
    --platform=CUDA \
    --xla_gpu_load_autotune_results_from=autotune.pbtxt \
    input.hlo
```

### Example: Debugging a Compilation Issue

```bash
# Step 1: List stages to understand the pipeline
bazel run //xla/tools:hlo-opt -- --list-stages

# Step 2: Check the HLO after optimization
bazel run //xla/tools:hlo-opt -- \
    --platform=CUDA \
    --stage=hlo \
    problematic.hlo

# Step 3: Check buffer assignment
bazel run //xla/tools:hlo-opt -- \
    --platform=CUDA \
    --stage=buffer-assignment \
    problematic.hlo

# Step 4: Check generated LLVM IR
bazel run //xla/tools:hlo-opt -- \
    --platform=CUDA \
    --stage=llvm \
    problematic.hlo

# Step 5: Try disabling specific passes to isolate the issue
bazel run //xla/tools:hlo-opt -- \
    --platform=CUDA \
    --passes=algebraic-simplifier,cse,dce \
    problematic.hlo
```

## ptx-opt

`ptx-opt` is a tool for compiling LLVM IR to PTX (Parallel Thread Execution) for NVIDIA GPUs. It provides a way to inspect and optimize the GPU code generation pipeline.

### LLVM to PTX Compilation

```bash
# Compile LLVM IR to PTX
bazel run //xla/tools:ptx-opt -- input.ll
```

### --arch Flag

Specify the target GPU architecture:

```bash
# Target A100 (sm_80)
bazel run //xla/tools:ptx-opt -- \
    --arch=sm_80 \
    input.ll

# Target H100 (sm_90)
bazel run //xla/tools:ptx-opt -- \
    --arch=sm_90 \
    input.ll
```

### LLVMIR Dump

Dump the LLVM IR at various stages of the PTX compilation pipeline:

```bash
# Dump LLVM IR before PTX generation
bazel run //xla/tools:ptx-opt -- \
    --dump-llvmir \
    input.ll

# Dump optimized LLVM IR
bazel run //xla/tools:ptx-opt -- \
    --dump-llvmir-after-opt \
    input.ll
```

### Usage with hlo-opt

A typical workflow combines `hlo-opt` and `ptx-opt`:

```bash
# Step 1: Generate LLVM IR from HLO
bazel run //xla/tools:hlo-opt -- \
    --platform=CUDA \
    --stage=llvm \
    input.hlo > output.ll

# Step 2: Compile LLVM IR to PTX
bazel run //xla/tools:ptx-opt -- \
    --arch=sm_80 \
    output.ll > output.ptx
```

## isolate_hlo

`isolate_hlo` is a tool for extracting problematic HLO instructions from a larger module and creating minimal reproducers. This is essential for filing bug reports and debugging complex compilation issues.

### Extracting Problematic Instructions

```bash
# Extract a specific instruction by name
bazel run //xla/tools:isolate_hlo -- \
    --instruction_name=dot.42 \
    input.hlo > isolated.hlo
```

### Creating Minimal Reproducers

The tool creates a minimal HLO module that contains only the instruction of interest and its transitive dependencies:

```bash
# Create a minimal reproducer for a failing instruction
bazel run //xla/tools:isolate_hlo -- \
    --instruction_name=failing_op \
    --extract_scope=computation_name \
    input.hlo > reproducer.hlo
```

The output module will:
1. Include only the instructions needed to compute the target instruction.
2. Replace intermediate values with `parameter` instructions where possible.
3. Preserve the original shapes and data types.
4. Be much smaller and easier to analyze than the original module.

### Flags

| Flag | Description |
|------|-------------|
| `--instruction_name` | Name of the instruction to extract |
| `--extract_scope` | Name of the computation to extract from |
| `--extract_all_gte` | Extract all get-tuple-element dependencies |

### Example Workflow for Bug Reporting

```bash
# Step 1: Get the HLO dump from a failing compilation
XLA_FLAGS="--xla_dump_to=/tmp/xla_dump" python failing_model.py

# Step 2: Find the problematic instruction
cat /tmp/xla_dump/module_before_optimizations.txt | grep "dot\."

# Step 3: Extract the instruction
bazel run //xla/tools:isolate_hlo -- \
    --instruction_name=dot.42 \
    /tmp/xla_dump/module_before_optimizations.hlo > reproducer.hlo

# Step 4: Verify the reproducer triggers the issue
bazel run //xla/tools:run_hlo_module -- \
    --platform=CUDA \
    reproducer.hlo

# Step 5: Report the bug with the reproducer
```

## Getting HLO Dumps

### XLA_FLAGS=--xla_dump_to

The primary mechanism for getting HLO dumps is the `--xla_dump_to` flag:

```bash
# Dump all compilation artifacts to a directory
XLA_FLAGS="--xla_dump_to=/tmp/xla_dump" python my_model.py
```

### Dump File Types

The dump directory will contain files at various stages of compilation:

| File Pattern | Description |
|-------------|-------------|
| `module_0000.before_optimizations.txt` | HLO before any optimization |
| `module_0001.after_optimizations.txt` | HLO after all optimizations |
| `module_0002.before_backend_optimizations.txt` | HLO before backend-specific passes |
| `module_0003.after_backend_optimizations.txt` | HLO after backend passes |
| `module_0004.buffer_assignment.txt` | Buffer assignment details |
| `module_0005.llvm_ir.ll` | Generated LLVM IR |
| `module_0006.ptx` | Generated PTX (GPU) |
| `module_0007.cubin` | Compiled GPU binary |

### Understanding Dump File Names

File names follow the pattern:

```
module_<NNNN>.<stage_name>.<extension>
```

Where:
- `<NNNN>`: A sequential number indicating the order of the stage in the pipeline.
- `<stage_name>`: The name of the compilation stage.
- `<extension>`: The file format:
  - `.txt`: Human-readable HLO text
  - `.hlo`: HLO text (same as .txt)
  - `.pb`: Binary protobuf
  - `.ll`: LLVM IR text
  - `.ptx`: PTX assembly
  - `.cubin`: GPU binary

### Additional Dump Flags

```bash
# Dump only HLO text (not LLVM IR or PTX)
XLA_FLAGS="--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_as_text" python my_model.py

# Dump as protobuf
XLA_FLAGS="--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_as_proto" python my_model.py

# Dump as short text (compact format)
XLA_FLAGS="--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_as_short_text" python my_model.py

# Dump with HTML visualization
XLA_FLAGS="--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_as_html" python my_model.py

# Dump module fingerprints for deduplication
XLA_FLAGS="--xla_dump_to=/tmp/xla_dump --xla_dump_fusion_clusters" python my_model.py

# Dump HLO snapshots (before/after each pass)
XLA_FLAGS="--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_snapshots" python my_model.py

# Include pass timings in dumps
XLA_FLAGS="--xla_dump_to=/tmp/xla_dump --xla_dump_enable_passthrough_metrics" python my_model.py

# Dump per-pass HLO (verbose)
XLA_FLAGS="--xla_dump_to=/tmp/xla_dump --xla_dump_per_pass_hlo" python my_model.py
```

### Using Dumps with Other Tools

HLO dumps can be fed into other XLA tools for analysis:

```bash
# Run a dumped HLO module
bazel run //xla/tools:run_hlo_module -- \
    --platform=CUDA \
    /tmp/xla_dump/module_0001.after_optimizations.hlo

# Optimize a dumped HLO module
bazel run //xla/tools:hlo-opt -- \
    --passes=algebraic-simplifier,cse,dce \
    /tmp/xla_dump/module_0000.before_optimizations.hlo

# Extract a problematic instruction
bazel run //xla/tools:isolate_hlo -- \
    --instruction_name=dot.42 \
    /tmp/xla_dump/module_0001.after_optimizations.hlo
```

### Dumping from JAX

```python
import jax
import jax.numpy as jnp

# Enable HLO dumping
jax.config.update("jax_dump_ir_to", "/tmp/jax_dump")

# Or via environment variable
# XLA_FLAGS="--xla_dump_to=/tmp/jax_dump" python script.py

# Your JAX program
@jax.jit
def f(x):
    return jnp.dot(x, x)

x = jnp.ones((1024, 1024))
result = f(x)
# Check /tmp/jax_dump for the HLO dumps
```

### Dumping from TensorFlow

```python
import tensorflow as tf

# Enable HLO dumping via environment variable
# XLA_FLAGS="--xla_dump_to=/tmp/tf_dump" python script.py

@tf.function(jit_compile=True)
def f(x):
    return tf.matmul(x, x)

x = tf.ones((1024, 1024))
result = f(x)
# Check /tmp/tf_dump for the HLO dumps
```
