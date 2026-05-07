# XLA JIT Compilation

This document provides a comprehensive reference for XLA just-in-time (JIT) compilation
in TensorFlow. JIT compilation enables TensorFlow to compile subgraphs of TF operations
into optimized machine code at runtime using XLA.

## Table of Contents

1. [XLA JIT Overview](#xla-jit-overview)
2. [MarkForCompilationPass](#markforcompilationpass)
3. [EncapsulateXlaComputationsPass](#encapsulatexlacomputationspass)
4. [BuildXlaOpsPass](#buildxlaopspass)
5. [DeviceCompiler](#devicecompiler)
6. [Explicit JIT: tf.function(jit_compile=True)](#explicit-jit-tffunctionjit_compiletrue)
7. [Auto-Clustering](#auto-clustering)
8. [Cluster Properties](#cluster-properties)
9. [XlaLaunch Op](#xlaunch-op)
10. [Compilation Cache](#compilation-cache)
11. [XLA Scope](#xla-scope)
12. [Debugging](#debugging)
13. [Common Issues](#common-issues)
14. [Performance Tips](#performance-tips)
15. [Environment Variables](#environment-variables)

---

## XLA JIT Overview

XLA JIT compilation translates TensorFlow subgraphs into XLA HLO (High Level Operations)
at runtime, compiles them through the XLA compiler pipeline, and executes the resulting
optimized code. This can provide significant performance improvements by:

- Fusing multiple operations into single kernels
- Eliminating intermediate buffer allocations
- Optimizing memory layouts for the target device
- Enabling platform-specific instruction generation

### Compilation Flow

```
TF Graph
   |
   v
MarkForCompilationPass  -- identifies XLA-compilable clusters
   |
   v
EncapsulateSubgraphsPass  -- wraps clusters into functions
   |
   v
BuildXlaOpsPass  -- converts to XLA computation ops
   |
   v
XlaCompiler  -- converts TF graph to HLO
   |
   v
XLA Compiler Pipeline  -- optimizes and generates code
   |
   v
Executable  -- cached and executed via XlaLaunch op
```

### Key Components

| Component | Header | Description |
|-----------|--------|-------------|
| `MarkForCompilationPass` | `compiler/jit/mark_for_compilation_pass.h` | Identifies XLA-compilable clusters |
| `EncapsulateSubgraphsPass` | `compiler/jit/encapsulate_subgraphs_pass.h` | Wraps clusters into TF functions |
| `DeviceCompiler` | `compiler/jit/device_compiler.h` | Manages compilation and caching |
| `DeviceCompilationCache` | `compiler/jit/device_compilation_cache.h` | Caches compiled executables |
| `XlaCompiler` | `compiler/tf2xla/xla_compiler.h` | Converts TF graph to HLO |
| `XlaLaunchOp` | `compiler/jit/xla_launch_util.h` | Launches XLA executables |

---

## MarkForCompilationPass

The `MarkForCompilationPass` is a `GraphOptimizationPass` that identifies subsets of
nodes in the TensorFlow graph that can be compiled with XLA. It assigns each cluster
a unique ID via the `_XlaCluster` attribute.

### Cluster Formation Algorithm

1. **Operation filtering**: Determine which operations are XLA-compilable
2. **Deadness analysis**: Ensure operations have compatible deadness (control flow)
3. **Resource safety analysis**: Ensure no unsafe resource variable access
4. **Cluster merging**: Merge adjacent compatible operations into clusters
5. **Cluster size limits**: Enforce minimum and maximum cluster sizes

### Declaration

```cpp
// From: tensorflow/compiler/jit/mark_for_compilation_pass.h

// The attribute that marks nodes to be grouped into functions by the
// encapsulate subgraphs pass.
extern const char* const kXlaClusterAttr;  // "_XlaCluster"

// Marks a subset of nodes in the graph which are to be clustered
// with an attribute _XlaCluster=<cluster id> so they are picked up by the
// EncapsulateSubgraphsPass.
class MarkForCompilationPass : public GraphOptimizationPass {
 public:
  MarkForCompilationPass() = default;
  absl::Status Run(const GraphOptimizationPassOptions& options) override;
};
```

### Cluster Identification Rules

An operation is eligible for XLA clustering if:

1. **Op has an XLA kernel**: The operation has a registered XLA translation
2. **Same device**: All operations in a cluster must be on the same device
3. **No unsupported resource ops**: Operations that modify resource variables
   in ways XLA cannot handle are excluded
4. **Deadness compatible**: Operations must have consistent deadness (all live
   or all dead at the same time)
5. **No side effects that XLA cannot handle**: Certain stateful operations
   are excluded

### Allowlist

The pass maintains an allowlist of operations known to be XLA-compatible:

```cpp
// Get the table of allowed operations
absl::flat_hash_map<std::string, std::vector<std::string>>* GetAllowlistTable();

// Get the known set of XLA-allowed operations
absl::flat_hash_set<std::string> GetKnownXLAAllowlistOp();
```

### Compilation Pass Ordering

The `MarkForCompilationPass` runs at the `POST_REWRITE_FOR_EXEC` optimization pass
level (level 2), after graph optimization but before execution.

---

## EncapsulateXlaComputationsPass

After marking, the `EncapsulateSubgraphsPass` wraps each cluster into a TensorFlow
function and replaces the original nodes with a call to that function.

### Declaration

```cpp
// From: tensorflow/compiler/jit/encapsulate_subgraphs_pass.h

class EncapsulateSubgraphsPass : public GraphOptimizationPass {
 public:
  absl::Status Run(const GraphOptimizationPassOptions& options) override;
};

// Transformation that finds subgraphs whose nodes are marked with
// 'group_attribute', splits those subgraphs into functions, and replaces
// the originals with function calls.
absl::Status EncapsulateSubgraphsInFunctions(
    std::string group_attribute, const Graph& graph_in,
    const RewriteSubgraphFn& rewrite_subgraph_fn,
    bool reuse_existing_functions,
    std::unique_ptr<Graph>* graph_out,
    FunctionLibraryDefinition* library);
```

### Key Attributes

The pass adds several attributes to the generated function call:

| Attribute | Description |
|-----------|-------------|
| `kXlaCompiledKernelAttr` (`_XlaCompiledKernel`) | Marks the function as XLA-compiled |
| `kXlaNumConstantArgsAttr` | Number of compile-time constant arguments |
| `kXlaNumResourceArgsAttr` | Number of resource variable arguments |
| `kXlaHasReferenceVarsAttr` | Whether the cluster has reference variables |

### Argument Ordering

Functions produced by the encapsulation pass order their arguments as:
1. Compile-time constant arguments (host memory)
2. Other arguments (device memory)
3. Resource variable arguments (host memory for the resource Tensor)

### Subgraph Rewriting

The `RewriteSubgraphFn` callback allows custom rewriting of each subgraph before
function conversion:

```cpp
typedef std::function<absl::Status(
    const std::vector<OutputTensor>& arg_source_tensors,
    std::unique_ptr<Graph>* graph,
    std::vector<int>* input_permutation,
    std::vector<int>* output_permutation,
    NodeDef* node_def)>
    RewriteSubgraphFn;
```

---

## BuildXlaOpsPass

The `BuildXlaOpsPass` converts the encapsulated function calls into XLA computation
operations that can be executed by the XLA runtime.

This pass:
1. Identifies function calls marked with `kXlaCompiledKernelAttr`
2. Converts each function into an XLA computation
3. Creates `XlaLaunch` ops to execute the compiled computations

---

## DeviceCompiler

The `DeviceCompiler` manages the full lifecycle of XLA compilation for JIT-compiled
TensorFlow functions. It coordinates:
- Converting TF functions to XLA computations
- Invoking the XLA compiler
- Caching compiled executables
- Managing device streams for execution

### Template Instantiation

`DeviceCompiler` is templated on the executable type and client type:

```cpp
template <typename ExecutableType, typename ClientType>
class DeviceCompiler {
 public:
  // Compiles a TF function into an XLA executable
  absl::Status Compile(
      const NameAttrList& function,
      const std::vector<XlaArgument>& args,
      const XlaCompiler::Options& options,
      DeviceCompileState* compile_state,
      ExecutableType** executable);

  // Retrieves a cached executable
  absl::Status GetExecutable(
      const DeviceCompilationClusterSignature& signature,
      ExecutableType** executable);
};
```

The two main instantiations are:
- `DeviceCompiler<xla::LocalExecutable, xla::LocalClient>` for standard XLA
- `DeviceCompiler<xla::PjRtLoadedExecutable, xla::PjRtClient>` for PjRt-based execution

### Compiler Options Generation

```cpp
// From: tensorflow/compiler/jit/xla_compiler_options_util.h

// Returns created options for the XLA compiler
XlaCompiler::Options GenerateCompilerOptions(
    const DeviceCompiler<xla::LocalExecutable, xla::LocalClient>&
        xla_device_compiler,
    const FunctionLibraryRuntime& function_library,
    DeviceBase* device,
    se::Stream* stream,
    const XlaPlatformInfo& platform_info,
    bool has_ref_vars);

// Returns created CompileOptions for XLA compiler
XlaCompiler::CompileOptions GenerateCompileOptions(
    bool has_ref_vars,
    bool may_alias_resource_update);
```

---

## Explicit JIT: tf.function(jit_compile=True)

TensorFlow provides explicit JIT compilation through the `jit_compile=True` parameter
on `tf.function`:

```python
@tf.function(jit_compile=True)
def my_function(x):
    return tf.matmul(x, x) + tf.reduce_mean(x)

# This function will always be compiled with XLA
result = my_function(tf.random.normal([100, 100]))
```

### How Explicit JIT Works

1. The `tf.function` decorator creates a `ConcreteFunction` from the Python function
2. With `jit_compile=True`, the function is unconditionally compiled through XLA
3. The compilation happens on the first call with specific input shapes
4. Subsequent calls with the same shapes reuse the cached executable

### Differences from Auto-Clustering

| Aspect | Auto-Clustering | jit_compile=True |
|--------|----------------|------------------|
| Trigger | Automatic optimization pass | Explicit decorator |
| Scope | Individual ops or subgraphs | Entire function |
| Recompilation | Per-shape changes | Per-shape changes |
| Fallback | Native TF for unsupported ops | Error on unsupported ops |
| Control | Limited via jit_scope | Full control |

### Compilation Errors

With `jit_compile=True`, if any operation in the function cannot be compiled by XLA,
an error is raised immediately. With auto-clustering, unsupported operations simply
remain as native TF ops.

---

## Auto-Clustering

Auto-clustering is the process by which TensorFlow automatically identifies and groups
operations that can benefit from XLA compilation, without explicit user annotation.

### Enabling Auto-Clustering

Auto-clustering can be enabled through:

```python
# Enable for CPU
tf.config.optimizer.set_jit(True)

# Enable for GPU
tf.config.optimizer.set_jit(True)

# Via environment variable
# TF_XLA_FLAGS=--tf_xla_auto_jit=2
```

### Clustering Levels

| Level | Behavior |
|-------|----------|
| 0 | Auto-clustering disabled |
| 1 | Conservative clustering (CPU only) |
| 2 | Aggressive clustering (CPU and GPU) |

### Cluster Scoping

The `ClusterScopingPass` assigns scope IDs to operations to prevent overly large
clusters. Operations in different scopes will not be merged into the same cluster:

```cpp
// From: tensorflow/compiler/jit/cluster_scoping_pass.h
class ClusterScopingPass : public GraphOptimizationPass {
 public:
  absl::Status Run(const GraphOptimizationPassOptions& options) override;
};
```

### Partial Declustering

The `PartiallyDeclusterPass` handles cases where only some operations in a cluster
should be executed via XLA:

```cpp
// From: tensorflow/compiler/jit/partially_decluster_pass.h
class PartiallyDeclusterPass : public GraphOptimizationPass {
 public:
  absl::Status Run(const GraphOptimizationPassOptions& options) override;
};
```

### Increase Dynamism Pass

The `IncreaseDynamismForAutoJitPass` makes auto-clustered computations more robust
to dynamic shapes by replacing static shape operations with dynamic equivalents:

```cpp
// From: tensorflow/compiler/jit/increase_dynamism_for_auto_jit_pass.h
class IncreaseDynamismForAutoJitPass : public GraphOptimizationPass {
 public:
  absl::Status Run(const GraphOptimizationPassOptions& options) override;
};
```

---

## Cluster Properties

For a set of operations to form a valid XLA cluster, they must satisfy several
invariant properties.

### Same Device Requirement

All operations in a cluster must be placed on the same device. XLA compiles each
cluster as a single unit targeting a specific device type.

### Supported Operations

Each operation in the cluster must have a registered XLA translation. Operations
without XLA support break the cluster.

### Resource-Free (with exceptions)

XLA has limited support for resource variables. Clusters can contain resource variable
reads and writes, but certain patterns are disallowed:
- Resource variables with non-trivial control flow
- Resource variables shared across clusters in unsafe ways

### Deadness Compatibility

All operations in a cluster must have the same "deadness" -- they must all be live
or all be dead at any point during execution. This is determined through deadness
analysis:

```cpp
// From: tensorflow/compiler/jit/deadness_analysis.h
// Analyzes which operations have the same deadness predicate
class DeadnessAnalysis {
 public:
  // Returns true if the outputs of two nodes always have the same deadness
  bool HasSameDeadness(const OutputTensor& a, const OutputTensor& b);
};
```

### Resource Operation Safety

```cpp
// From: tensorflow/compiler/jit/compilability_check_util.h
// Checks if operations can be safely compiled by XLA
```

Resource operations are analyzed for safety:
- `AssignVariableOp` is allowed within XLA clusters
- `ReadVariableOp` is always allowed
- Operations that create or destroy resources are not allowed

---

## XlaLaunch Op

The `XlaLaunch` op (also known as `_XlaLaunch`) is the TensorFlow operation that
executes a compiled XLA computation.

### Op Signature

```
_XlaLaunch(
    constants...,       // Compile-time constant inputs
    args...,            // Runtime arguments
    resource_args...,   // Resource variable arguments
    device_ordinal,     // Target device
    compilation_key,    // Cache key for the compiled executable
    program_shape       // Shape of the computation
) -> (results...)
```

### Execution Flow

1. **Lookup executable**: Use the `compilation_key` to find the cached executable
2. **Transfer inputs**: Copy input tensors to the device
3. **Execute**: Run the XLA executable on the device stream
4. **Transfer outputs**: Copy results back if needed
5. **Update resources**: Write back resource variable updates

### Launch Utilities

```cpp
// From: tensorflow/compiler/jit/xla_launch_util.h

// Manages the launch of XLA computations
class XlaComputationLaunchContext {
 public:
  // Builds XLA inputs from TF tensors
  void PopulateInputs(
      xla::LocalExecutable* executable,
      OpKernelContext* ctx,
      const XlaCompiler::CompilationResult& compilation_result,
      int device_ordinal);

  // Extracts TF tensors from XLA outputs
  void PopulateOutputs(
      OpKernelContext* ctx,
      const XlaCompiler::CompilationResult& compilation_result,
      absl::Span<xla::ShapedBuffer> output_buffers);
};
```

---

## Compilation Cache

The compilation cache stores compiled XLA executables keyed by the computation's
signature (operation types, shapes, layouts).

### DeviceCompilationCache

```cpp
// From: tensorflow/compiler/jit/device_compilation_cache.h

template <typename ExecutableType>
class DeviceCompilationCache {
 public:
  using Key = DeviceCompilationClusterSignature;

  struct Value {
    DeviceCompileState compile_state = DeviceCompileState::kUncompiled;
    absl::Status compilation_status;
    int64_t request_count = 0;
    const XlaCompiler::CompilationResult* compilation_result = nullptr;
    ExecutableType* executable = nullptr;
  };

  // Lookup a cached entry
  std::optional<Value> Lookup(const Key& key) const;

  // Lookup or create an entry
  Value LookupOrCreate(const Key& key);

  // Store a compiled result
  void Store(const Key& key,
             std::optional<DeviceCompileState> compile_state,
             std::optional<absl::Status> compilation_status,
             std::optional<std::unique_ptr<XlaCompiler::CompilationResult>>
                 compilation_result,
             std::optional<std::unique_ptr<ExecutableType>> executable);

  // Debug string
  std::string DebugString() const;

  // Finalize: release XlaComputation references
  void Finalize();
};
```

### Cache Entry States

```cpp
enum class DeviceCompileState {
  kUncompiled,        // Not yet compiled
  kCompiled,          // Successfully compiled
  kQueued,            // Compilation queued
  kCompiling,         // Currently being compiled
};
```

### Key Generation

The cache key (`DeviceCompilationClusterSignature`) is generated from:

1. **Op types**: The types of all operations in the cluster
2. **Input shapes**: The static shapes of all inputs
3. **Input types**: The data types of all inputs
4. **Attributes**: Any relevant operation attributes
5. **Device**: The target device

```cpp
// From: tensorflow/compiler/jit/device_compilation_cluster_signature.h

class DeviceCompilationClusterSignature {
 public:
  // Build from a function and its inputs
  static absl::StatusOr<DeviceCompilationClusterSignature> Build(
      const NameAttrList& function,
      const std::vector<XlaArgument>& args,
      const XlaCompiler::Options& options);

  std::string HumanString() const;

  struct Hash {
    size_t operator()(const DeviceCompilationClusterSignature& s) const;
  };
};
```

### Thread Safety

The cache is thread-safe, using fine-grained locking:
- An outer lock (`compile_cache_mu_`) protects the existence of cache entries
- Each entry has its own lock (`Entry::mu`) protecting the entry contents
- This allows concurrent lookups of different entries

### Cache Growth

The cache currently has no eviction policy and grows without bound. Entries are
only freed when the cache is destroyed or `Finalize()` is called.

---

## XLA Scope

The XLA scope API allows fine-grained control over which operations are compiled
with XLA.

### tf.xla.experimental.jit_scope

```python
import tensorflow as tf

# Compile all supported ops in scope with XLA
with tf.xla.experimental.jit_scope(True):
    result = tf.matmul(a, b)  # Will be compiled with XLA

# Disable XLA compilation for specific ops
with tf.xla.experimental.jit_scope(False):
    result = tf.matmul(a, b)  # Will NOT be compiled with XLA
```

### Compile/No-Compile Directives

The scope can be used to explicitly include or exclude operations:

```python
# Mixed usage
with tf.xla.experimental.jit_scope(True):
    x = tf.matmul(a, b)       # XLA compiled
    with tf.xla.experimental.jit_scope(False):
        y = custom_op(x)      # Not XLA compiled
    z = tf.matmul(y, c)       # XLA compiled
```

### On/Off Optional

The scope parameter can be a callable for conditional compilation:

```python
# Only compile if the operation is on GPU
with tf.xla.experimental.jit_scope(
    lambda op: op.device.startswith("/device:GPU")):
    # Only GPU ops will be compiled
    result = tf.matmul(a, b)
```

---

## Debugging

XLA JIT compilation provides several debugging mechanisms.

### xla_dump_to

The `xla_dump_to` flag directs XLA to dump intermediate representations during
compilation:

```bash
# Dump HLO and LLVM IR to a directory
TF_XLA_FLAGS="--xla_dump_to=/tmp/xla_dumps" python my_model.py

# Dump only HLO
TF_XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=/tmp/xla_dumps" python my_model.py
```

### Dumped Files

When dumping is enabled, the following files are generated:

| File | Description |
|------|-------------|
| `module_*.before_optimizations.txt` | HLO before optimization |
| `module_*.after_optimizations.txt` | HLO after optimization |
| `module_*.before_backend.txt` | HLO before backend compilation |
| `module_*.llvm_ir.ll` | LLVM IR (CPU backend) |
| `module_*.ptx` | PTX code (GPU backend) |
| `module_*.thinlto.bc` | LLVM bitcode |

### XLA Debugging Flags

```bash
# Enable verbose logging
TF_XLA_FLAGS="--xla_log_hlo_text"

# Dump HLO as protobuf
TF_XLA_FLAGS="--xla_dump_hlo_as_proto"

# Dump HLO with snapshots (includes constant values)
TF_XLA_FLAGS="--xla_dump_hlo_snapshots"

# Dump fuzzer input (for reproducing compilation issues)
TF_XLA_FLAGS="--xla_dump_fuzzer_input"
```

### Per-Compilation Dumping

To dump only specific compilations:

```bash
# Dump only compilations matching a regex
TF_XLA_FLAGS="--xla_dump_to=/tmp/dumps --xla_dump_hlo_module_regex=my_cluster.*"
```

### TF XLA Flags

```bash
# General TF XLA flags
TF_XLA_FLAGS="--tf_xla_auto_jit=2"            # Enable auto-clustering
TF_XLA_FLAGS="--tf_xla_min_cluster_size=2"     # Minimum cluster size
TF_XLA_FLAGS="--tf_xla_max_cluster_size=5000"  # Maximum cluster size
TF_XLA_FLAGS="--tf_xla_disable_deadness_analysis"  # Disable deadness checks
```

### Compilation Statistics

```python
# Enable XLA compilation statistics
tf.debugging.set_log_device_placement(True)

# Get XLA activity information
# From: tensorflow/compiler/jit/xla_activity_listener.h
```

---

## Common Issues

### Unsupported Operations

Some TensorFlow operations do not have XLA translations:

```python
@tf.function(jit_compile=True)
def problematic(x):
    # This will fail because tf.py_function has no XLA translation
    return tf.py_function(my_python_func, [x], tf.float32)
```

**Solution**: Use `tf.raw_ops` or restructure the computation to use supported ops.

### Dynamic Shapes

XLA compiles for specific input shapes. Dynamic shapes cause recompilation:

```python
@tf.function(jit_compile=True)
def my_func(x):
    return tf.reduce_sum(x)

# First call compiles for shape [100]
my_func(tf.ones([100]))

# Second call recompiles for shape [200]
my_func(tf.ones([200]))  # Recompilation!
```

**Solution**: Use `tf.TensorSpec` with `None` dimensions for dynamic shapes,
or pad to fixed sizes.

### Resource Variables

Resource variables can cause issues in XLA clusters:

```python
var = tf.Variable(0.0)

@tf.function(jit_compile=True)
def update_var():
    var.assign_add(1.0)  # May fail if variable capture doesn't work
    return var.read_value()
```

**Solution**: Pass variables as explicit arguments:

```python
@tf.function(jit_compile=True)
def update_var(var):
    return var + 1.0
```

### Compilation Time

Initial compilation can be slow (seconds to minutes for large models):

```python
# First call triggers compilation
result = compiled_func(input_data)  # Slow (compiling)

# Subsequent calls are fast
result = compiled_func(input_data)  # Fast (cached)
```

### Shape Mismatch Recompilation

Each unique input shape triggers a new compilation:

```python
# Compiles 100 times if batch sizes vary
for batch_size in range(1, 101):
    result = compiled_func(tf.ones([batch_size, 128]))
```

---

## Performance Tips

### Use Static Shapes

Static shapes allow XLA to generate optimal code:

```python
# Good: static batch size
@tf.function(input_signature=[tf.TensorSpec([128, 256], tf.float32)])
def good_func(x):
    return tf.matmul(x, x)

# Bad: fully dynamic batch size (causes recompilation or less optimization)
@tf.function
def bad_func(x):
    return tf.matmul(x, x)
```

### Avoid Recompilation

Use `tf.function` with explicit input signatures to bound the number of compilations:

```python
# Specify allowed shapes
@tf.function(input_signature=[
    tf.TensorSpec([None, 256], tf.float32)  # Dynamic batch, fixed features
])
def bounded_func(x):
    return tf.matmul(x, weights)
```

### Enable Compilation Cache Warmup

Pre-compile for expected input shapes:

```python
# Warmup: compile for common shapes before serving
for shape in [(32, 256), (64, 256), (128, 256)]:
    _ = compiled_func(tf.zeros(shape))
```

### Profile Compilation Time

```python
# Enable profiling
tf.profiler.experimental.start('/tmp/profile')
result = compiled_func(input_data)
tf.profiler.experimental.stop()
```

### Use XLA-Friendly Operations

Some operations are more XLA-friendly than others:

| Prefer | Avoid |
|--------|-------|
| `tf.matmul` | `tf.einsum` (with complex patterns) |
| `tf.gather` with fixed indices | `tf.gather` with dynamic indices |
| `tf.while_loop` with static bounds | Python `for` loops |
| `tf.cond` | Python `if` statements |
| `tf.reduce_sum` | `tf.reduce_sum` with `axis=None` |

### Fusion-Friendly Patterns

XLA achieves maximum performance when operations can be fused:

```python
# Good: chain of elementwise ops (will be fused)
@tf.function(jit_compile=True)
def good(x):
    x = tf.matmul(x, w1)
    x = tf.nn.relu(x)      # Fused with matmul output
    x = tf.matmul(x, w2)
    x = tf.nn.relu(x)      # Fused with matmul output
    return x

# Bad: materializing intermediates prevents fusion
def bad(x):
    intermediates = []
    for w in weights:
        x = tf.matmul(x, w)
        x = tf.nn.relu(x)
        intermediates.append(x)  # Materializing prevents fusion
    return intermediates
```

---

## Environment Variables

### TF_XLA_FLAGS

General TensorFlow XLA configuration:

```bash
# Enable auto-clustering
TF_XLA_FLAGS="--tf_xla_auto_jit=2"

# Disable deadness analysis
TF_XLA_FLAGS="--tf_xla_disable_deadness_analysis"

# Set minimum cluster size
TF_XLA_FLAGS="--tf_xla_min_cluster_size=2"

# Disable compilation cache
TF_XLA_FLAGS="--tf_xla_disable_constant_folding"
```

### XLA_FLAGS

XLA compiler configuration:

```bash
# Enable fast math (may reduce accuracy)
XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"

# Set GPU CUDA data directory
XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"

# Enable LLVM-based compilation for CPU
XLA_FLAGS="--xla_cpu_use_thunk_runtime=true"
```

### Dumping Flags

```bash
# Dump to directory
XLA_FLAGS="--xla_dump_to=/tmp/xla_dumps"

# Dump as text
XLA_FLAGS="--xla_dump_hlo_as_text"

# Dump as protobuf
XLA_FLAGS="--xla_dump_hlo_as_proto"

# Dump fuzzer reproducer
XLA_FLAGS="--xla_dump_fuzzer_input"

# Dump module matching regex
XLA_FLAGS="--xla_dump_hlo_module_regex=.*my_cluster.*"
```

### Compilation Flags

```bash
# Force compilation to fail after N seconds
XLA_FLAGS="--xla_compilation_timeout_secs=60"

# Enable XLA GPU autotuning
XLA_FLAGS="--xla_gpu_autotune_level=4"

# Disable XLA GPU autotuning
XLA_FLAGS="--xla_gpu_autotune_level=0"
```

### Debugging Flags

```bash
# Enable verbose XLA logging
TF_CPP_MIN_LOG_LEVEL=0

# Enable VLOG for JIT-related modules
TF_CPP_VMODULE="xla_compilation_cache=1,jit_compilation_pass=1"

# Log placement
TF_CPP_VMODULE="device_compiler=1"
```

---

## JIT Compilation and Device Compilers

### XLA CPU Device

```cpp
// From: tensorflow/compiler/jit/xla_cpu_device.cc
// Registers the XLA CPU device and kernel
```

The XLA CPU device allows running JIT-compiled computations on the CPU.

### XLA GPU Device

```cpp
// From: tensorflow/compiler/jit/xla_gpu_device.cc
// Registers the XLA GPU device and kernel
```

### XLA TPU Device

```cpp
// From: tensorflow/compiler/jit/xla_tpu_device.cc
// Registers the XLA TPU device and kernel
```

### Compilation Flow for On-Demand Compilation

When an operation is encountered that has not been compiled:

```cpp
// From: tensorflow/compiler/jit/xla_compile_on_demand_op.cc
// Handles on-demand compilation of single operations
```

This path is used when `jit_compile=True` is set but the function hasn't been
compiled for the current input shapes yet.

---

## Device Compilation Profiler

The `DeviceCompilationProfiler` tracks compilation metrics:

```cpp
// From: tensorflow/compiler/jit/device_compilation_profiler.h
class DeviceCompilationProfiler {
 public:
  // Record a compilation event
  void RecordCompilation(const string& cluster_name,
                         int64_t compile_time_us);

  // Get profiling statistics
  std::string GetStatsString() const;
};
```

### Metrics Tracked

- Number of compilations
- Total compilation time
- Per-cluster compilation time
- Cache hit rate
- Recompilation count

---

## XLA Activity Listener

The activity listener provides hooks for monitoring XLA compilation events:

```cpp
// From: tensorflow/compiler/jit/xla_activity_listener.h
class XlaActivityListener {
 public:
  virtual void Listen(const XlaActivity& activity) = 0;
};
```

Activity types include:
- Compilation started
- Compilation completed
- Compilation failed
- Cache hit
- Executable launched

---

## Compilation Utilities

### XlaCompileUtil

```cpp
// From: tensorflow/compiler/jit/xla_compile_util.h

// Creates a single-op graph for on-demand compilation
absl::StatusOr<std::unique_ptr<Graph>> CreateSingleOpGraph(
    const NodeDef& node_def,
    absl::Span<const XlaArgument> args,
    absl::Span<const DataType> result_types);
```

### Variable Info

```cpp
// From: tensorflow/compiler/jit/variable_info.h
struct VariableInfo {
  int index;
  Var* variable;
};
```

### Shape Inference

```cpp
// From: tensorflow/compiler/jit/shape_inference.h
// Infers output shapes for XLA-compiled computations
```

---

## Key Source Files Reference

| File Path | Description |
|-----------|-------------|
| `compiler/jit/mark_for_compilation_pass.h` | Cluster identification |
| `compiler/jit/encapsulate_subgraphs_pass.h` | Subgraph encapsulation |
| `compiler/jit/device_compiler.h` | Compilation management |
| `compiler/jit/device_compilation_cache.h` | Compilation cache |
| `compiler/jit/device_compilation_cluster_signature.h` | Cache key generation |
| `compiler/jit/xla_compiler_options_util.h` | Compiler option generation |
| `compiler/jit/xla_launch_util.h` | Launch utilities |
| `compiler/jit/xla_compile_util.h` | Compilation utilities |
| `compiler/jit/cluster_scoping_pass.h` | Cluster scoping |
| `compiler/jit/partially_decluster_pass.h` | Partial declustering |
| `compiler/jit/deadness_analysis.h` | Deadness analysis |
| `compiler/jit/xla_activity_listener.h` | Activity monitoring |
| `compiler/jit/flags.cc` | JIT flags definition |
| `compiler/jit/xla_cpu_device.cc` | XLA CPU device |
| `compiler/jit/xla_gpu_device.cc` | XLA GPU device |
| `compiler/jit/xla_tpu_device.cc` | XLA TPU device |
