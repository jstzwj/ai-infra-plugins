# XLA Ahead-of-Time (AOT) Compilation

This document provides a comprehensive reference for XLA ahead-of-time compilation
via `tfcompile`. AOT compilation converts TensorFlow graphs into native object files
that can be linked directly into applications, eliminating the need for the TensorFlow
runtime and XLA JIT infrastructure.

## Table of Contents

1. [tfcompile Overview](#tfcompile-overview)
2. [Compilation Pipeline](#compilation-pipeline)
3. [tfcompile Flags](#tfcompile-flags)
4. [Build Configuration](#build-configuration)
5. [Generated Header](#generated-header)
6. [Config Proto](#config-proto)
7. [ShapeInfo](#shapeinfo)
8. [Target Configuration](#target-configuration)
9. [Linking](#linking)
10. [Thread Safety](#thread-safety)
11. [AOT vs JIT Comparison](#aot-vs-jit-comparison)
12. [XlaCompiledCpuFunction](#xlacompiledcpufunction)
13. [Warmup](#warmup)

---

## tfcompile Overview

`tfcompile` is a standalone tool that compiles TensorFlow computation graphs into
executable object code. It takes a `GraphDef` (serialized TensorFlow graph) and a
configuration proto, and produces:

1. An object file (`.o`) containing the compiled function
2. A C++ header file providing the API to call the compiled function
3. An optional metadata object file

### Key Benefits

- **No runtime dependency**: Eliminates TensorFlow and XLA runtime overhead
- **Smaller binary**: Only includes the compiled computation, not the full TF runtime
- **Predictable performance**: No JIT compilation latency at runtime
- **Embeddable**: Can be used in environments where TensorFlow cannot run (mobile,
  embedded, constrained environments)
- **Cross-compilation**: Compile on a build machine for a different target architecture

### When to Use AOT

- Deployment to resource-constrained environments
- Latency-sensitive serving with predictable compilation
- Mobile or embedded inference
- Integration with non-Python codebases
- Security-sensitive environments where JIT is not desired

---

## Compilation Pipeline

The AOT compilation pipeline transforms a TensorFlow graph through multiple stages:

```
TensorFlow GraphDef
      |
      v
  Config Proto (tf2xla::Config)
      |
      v
  +---+---+
  | TF -> |  tf2xla conversion
  | XLA   |  (identifies feed/fetch nodes)
  +---+---+
      |
      v
  XLA HLO Module
      |
      v
  +---+---+
  | HLO   |  XLA optimization passes
  | Opt   |  (fusion, layout, simplification)
  +---+---+
      |
      v
  +---+---+
  | Layout |  Layout assignment
  | Assign |  (CPU target layouts)
  +---+---+
      |
      v
  +---+---+
  | LLVM  |  LLVM IR generation
  | IR    |  (HLO -> LLVM IR)
  +---+---+
      |
      v
  +---+---+
  | LLVM  |  LLVM optimization
  | Opt   |  (vectorization, inlining)
  +---+---+
      |
      v
  +---+---+
  | Code  |  Object code generation
  | Gen   |  (target-specific machine code)
  +---+---+
      |
      v
  Object File (.o)
```

### CompileGraph Function

The core compilation function:

```cpp
// From: tensorflow/compiler/aot/compile.h

struct CompileResult {
  // Contains object file and meta-info
  std::unique_ptr<xla::cpu::CpuAotCompilationResult> aot;
  xla::ProgramShapeProto program_shape;  // Static shape of args and results
  std::string entry_point;               // Name of generated function
  int pointer_size = 0;                  // Size of a pointer in bytes
};

// CompileGraph compiles the graph_def into an object file containing a function
// that performs the graph operations
absl::Status CompileGraph(
    GraphDef graph_def,
    const tf2xla::Config& config,
    const MainFlags& flags,
    CompileResult* compile_result);

// The full compilation method, for reuse in a library setting
absl::Status Main(const MainFlags& flags);
```

### CompileResult Structure

The `CompileResult` holds the output of compilation:

| Field | Type | Description |
|-------|------|-------------|
| `aot` | `unique_ptr<CpuAotCompilationResult>` | Object file data and metadata |
| `program_shape` | `ProgramShapeProto` | Static shapes of parameters and results |
| `entry_point` | `string` | Name of the generated C-callable function |
| `pointer_size` | `int` | Size of a pointer on the target platform |

---

## tfcompile Flags

`tfcompile` accepts command-line flags that control the compilation process.

### MainFlags Structure

```cpp
// From: tensorflow/compiler/aot/flags.h

struct MainFlags {
  std::string graph;                     // Input GraphDef file path
  std::string debug_info;                // Debug info file path
  std::string debug_info_path_begin_marker;  // Debug info path marker
  std::string config;                    // Config proto file path
  bool dump_fetch_nodes = false;         // Dump fetch node names
  std::string target_triple;             // LLVM target triple
  std::string target_cpu;                // Target CPU name
  std::string target_features;           // Target CPU features
  std::string entry_point;               // Generated function name
  std::string cpp_class;                 // C++ class name for header
  std::string out_function_object;       // Output object file path
  std::string out_metadata_object;       // Output metadata object file path
  std::string out_header;                // Output header file path
  std::string out_constant_buffers_object;  // Output constant buffers object
  std::string out_session_module;         // Output session module
  std::string mlir_components;           // MLIR components to use
  bool experimental_quantize = false;    // Enable experimental quantization

  // Sanitizer pass options
  bool sanitize_dataflow = false;
  std::string sanitize_abilists_dataflow;

  // C++ codegen options
  bool gen_name_to_index = false;        // Generate name-to-index mapping
  bool gen_program_shape = false;        // Generate program shape data
  bool use_xla_nanort_runtime = false;   // Use XLA nanort runtime
};
```

### Common Flags

| Flag | Required | Description |
|------|----------|-------------|
| `--graph` | Yes | Path to the input GraphDef file |
| `--config` | Yes | Path to the config proto file |
| `--entry_point` | Yes | Name of the generated entry point function |
| `--cpp_class` | No | C++ class name (e.g., `my::namespace::MyClass`) |
| `--out_header` | No | Output C++ header file path |
| `--out_function_object` | No | Output function object file path |
| `--out_metadata_object` | No | Output metadata object file path |
| `--target_triple` | No | LLVM target triple |
| `--target_cpu` | No | Target CPU architecture |
| `--target_features` | No | Target CPU features (e.g., "+avx2") |

### Example Usage

```bash
tfcompile \
  --graph=my_graph.pb \
  --config=my_config.pbtxt \
  --entry_point=my_computation \
  --cpp_class="myns::MyComputation" \
  --out_header=my_computation.h \
  --out_function_object=my_computation.o \
  --target_triple=x86_64-unknown-linux-gnu \
  --target_cpu=haswell \
  --target_features="+avx2,+fma"
```

---

## Build Configuration

### tf_library() Bazel Macro

The recommended way to use tfcompile is through the `tf_library()` Bazel macro:

```python
# BUILD file
load("@org_tensorflow//tensorflow/compiler/aot:tf_library.bzl", "tf_library")

tf_library(
    name = "my_computation",
    config = "my_computation.config.pbtxt",
    graph = "my_computation.graph.pb",
    cpp_class = "myns::MyComputation",
    target_triple = "x86_64-unknown-linux-gnu",
)
```

### tf_library Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | string | Rule name |
| `graph` | label | Input GraphDef file |
| `config` | label | Config proto file |
| `cpp_class` | string | C++ class name with optional namespaces |
| `target_triple` | string | LLVM target triple |
| `target_cpu` | string | Target CPU name |
| `target_features` | string | Target CPU features |
| `entry_point` | string | Entry point function name |
| `gen_name_to_index` | bool | Generate name-to-index mapping |
| `gen_program_shape` | bool | Generate program shape data |
| `experimental_enable_fast_math` | bool | Enable fast math optimizations |

### Makefile Integration

For non-Bazel builds:

```makefile
# Compile the graph
my_computation.o: my_graph.pb my_config.pbtxt
    tfcompile \
        --graph=my_graph.pb \
        --config=my_config.pbtxt \
        --entry_point=my_computation \
        --out_function_object=my_computation.o \
        --out_header=my_computation.h

# Link with application
my_app: my_app.o my_computation.o
    $(CXX) -o $@ $^ -lm -lpthread
```

---

## Generated Header

The generated header file provides a C++ interface to the compiled computation.

### Header Structure

```cpp
// Generated header example for class "myns::MyComputation"
#ifndef MYPREFIX_MY_COMPUTATION_H_
#define MYPREFIX_MY_COMPUTATION_H_

#include "tensorflow/compiler/aot/aot_only_var_handle_op.h"
#include "tensorflow/compiler/xla/cpu_function_runtime.h"

namespace myns {

class MyComputation {
 public:
  // Input/Output structures
  struct Args {
    float* arg0;  // Shape: [batch, features]
    float* arg1;  // Shape: [features, hidden]
  };

  struct Result {
    float* result0;  // Shape: [batch, hidden]
  };

  // Run the computation
  static void Run(const Args& args, Result* result);

  // Program shape information
  static xla::ProgramShapeProto ProgramShape();

  // Name-to-index mapping (if gen_name_to_index=true)
  static int LookupArgIndex(const std::string& name);
  static int LookupResultIndex(const std::string& name);

  // Buffer information
  static const xla::cpu_function_runtime::BufferInfo* buffer_infos();
  static int buffer_info_count();
  static const void* buffer_infos_data();
  static int temp_buffer_size();

 private:
  MyComputation();
};

}  // namespace myns

#endif  // MYPREFIX_MY_COMPUTATION_H_
```

### CodegenOpts

```cpp
// From: tensorflow/compiler/aot/codegen.h

struct CodegenOpts {
  std::string class_name;              // C++ class name
  std::string target_triple;           // Target architecture triple
  std::vector<std::string> namespaces; // C++ namespaces
  bool gen_name_to_index = false;      // Generate name-to-index data
  bool gen_program_shape = false;      // Generate program shape data
  bool gen_hlo_profile_printer_data = false;  // Generate profile data
  bool use_xla_runtime = false;        // Use XLA Runtime
  bool use_xla_nanort_runtime = false;  // Use XLA nanort runtime
};
```

### Function Signatures

The generated header includes:

1. **Run function**: Executes the compiled computation
2. **ProgramShape**: Returns the static shape information
3. **BufferInfo**: Provides raw buffer metadata
4. **LookupArgIndex/LookupResultIndex**: Maps names to tensor indices (optional)

---

## Config Proto

The config proto (`tf2xla::Config`) specifies the inputs, outputs, and variables
of the computation to compile.

### Config Structure

```protobuf
// tensorflow/compiler/tf2xla/tf2xla.proto

message Config {
  repeated Feed feed = 1;           // Input tensors
  repeated Fetch fetch = 2;         // Output tensors
  repeated Variable variable = 3;   // Resource variables
  repeated TensorShape shape = 4;   // Shape information
  ProgrammingConfig programming_config = 5;
}
```

### Feed

```protobuf
message Feed {
  string id = 1;                    // Tensor name (e.g., "input:0")
  TensorShape shape = 2;            // Expected shape
  int64 parameter = 3;             // Parameter number in the compiled function
  string name = 4;                  // Human-readable name
}
```

### Fetch

```protobuf
message Fetch {
  string id = 1;                    // Tensor name (e.g., "output:0")
  int64 parameter = 2;             // Result number in the compiled function
  string name = 3;                  // Human-readable name
}
```

### Variable

```protobuf
message Variable {
  string node_name = 1;             // Variable node name
  int64 parameter = 2;             // Parameter number
  TensorShape shape = 3;           // Variable shape
  string name = 4;                  // Human-readable name
  DataType type = 5;               // Variable data type
  float min = 6;                   // Min value (for quantization)
  float max = 7;                   // Max value (for quantization)
}
```

### Example Config

```
# Text format config proto
feed {
  id: "input:0"
  shape { dim { size: 1 } dim { size: 784 } }
  name: "input"
}
feed {
  id: "weights:0"
  shape { dim { size: 784 } dim { size: 10 } }
  name: "weights"
}
fetch {
  id: "output:0"
  name: "output"
}
```

### Programming Config

```protobuf
message ProgrammingConfig {
  // Additional compilation options
}
```

---

## ShapeInfo

Static shape specification is required for AOT compilation. All tensor shapes must
be known at compile time.

### Shape Specification

```protobuf
message TensorShape {
  repeated Dim dim = 1;
}

message Dim {
  int64 size = 1;    // -1 for dynamic dimensions (limited support)
  string name = 2;   // Optional dimension name
}
```

### Shape Inference During Compilation

The AOT compiler infers shapes for intermediate tensors based on:
- Input shapes (specified in `Config.feed`)
- Operation shape inference rules
- Constant folding of shape computations

### Dynamic Dimensions

Limited support for dynamic dimensions is available:
- Batch dimension can be dynamic in some cases
- Dynamic dimensions may prevent certain optimizations
- The compiled function assumes the maximum expected size

---

## Target Configuration

### Target Triple

The target triple specifies the target architecture:

| Triple | Architecture |
|--------|-------------|
| `x86_64-unknown-linux-gnu` | x86-64 Linux |
| `x86_64-apple-darwin` | x86-64 macOS |
| `aarch64-unknown-linux-gnu` | AArch64 (ARM64) Linux |
| `aarch64-apple-darwin` | AArch64 macOS (Apple Silicon) |
| `armv7-unknown-linux-gnueabihf` | ARMv7 Linux |
| `wasm32-unknown-unknown` | WebAssembly (32-bit) |
| `wasm64-unknown-unknown` | WebAssembly (64-bit) |

### Target CPU

The target CPU enables CPU-specific optimizations:

| CPU | Optimizations |
|-----|--------------|
| `generic` | Generic (no CPU-specific optimizations) |
| `haswell` | AVX2, FMA, BMI |
| `skylake` | AVX2, FMA, BMI, ADX |
| `skylake-avx512` | AVX-512F, AVX-512BW, AVX-512VL |
| `cannonlake` | AVX-512F, AVX-512BW, AVX-512VL, AVX-512IFMA |
| `cortex-a53` | ARM NEON |
| `cortex-a57` | ARM NEON, CRC |
| `cortex-a76` | ARM NEON, LSE |
| `apple-a14` | Apple M1 features |

### Target Features

Additional CPU features can be enabled/disabled:

```bash
--target_features="+avx2,+fma,-avx512f"
```

Common features:

| Feature | Description |
|---------|-------------|
| `+avx` | Advanced Vector Extensions |
| `+avx2` | AVX2 256-bit integer operations |
| `+fma` | Fused multiply-add |
| `+avx512f` | AVX-512 foundation |
| `+sse4.2` | SSE 4.2 |
| `+neon` | ARM NEON SIMD |

---

## Linking

### Static Linking

Link the compiled object file directly into your application:

```makefile
# Compile the application with the AOT object
my_app: main.o my_computation.o
    $(CXX) -o $@ $^ -lm -lpthread
```

### Dynamic Linking

Alternatively, create a shared library:

```bash
# Create shared library from compiled objects
g++ -shared -o libmy_computation.so my_computation.o my_metadata.o

# Link application against shared library
g++ -o my_app main.o -L. -lmy_computation
```

### Required Runtime Libraries

The compiled code requires minimal runtime support:

- **xla::cpu_function_runtime**: Buffer allocation and metadata
- **Standard C library**: For basic operations (`memcpy`, `memset`, etc.)
- **libm**: Math functions (if the computation uses transcendentals)

### Linking with Bazel

```python
# BUILD file
cc_binary(
    name = "my_app",
    srcs = ["main.cc"],
    deps = [
        ":my_computation",           # The tf_library target
        "@org_tensorflow//tensorflow/compiler/xla/cpu_function_runtime",
    ],
)
```

---

## Thread Safety

### Compiled Executable Thread Safety

The compiled computation is **not thread-safe** by default. Each thread should
use its own buffer allocations:

```cpp
// Thread-safe usage: separate buffers per thread
void worker_thread(float* input, float* output) {
    myns::MyComputation::Args args;
    args.arg0 = input;
    args.arg1 = weights;

    myns::MyComputation::Result result;
    result.result0 = output;

    myns::MyComputation::Run(args, &result);
}
```

### Buffer Allocation

Temporary buffers used by the compiled function must not be shared between threads.
The generated header provides buffer size information:

```cpp
// Get the required temporary buffer size
int temp_size = myns::MyComputation::temp_buffer_size();

// Allocate per-thread temp buffer
char* temp_buffer = new char[temp_size];
```

### Static Data

The compiled function's code and constant data are thread-safe (read-only). Multiple
threads can execute the same compiled function simultaneously with different input/output
buffers.

---

## AOT vs JIT Comparison

| Aspect | AOT (tfcompile) | JIT (XLA JIT) |
|--------|----------------|----------------|
| **Compilation time** | Build time | Runtime (first call) |
| **Runtime overhead** | None | Compilation latency on first call |
| **Binary size** | Small (computation only) | Large (full TF + XLA runtime) |
| **Dependencies** | Minimal runtime | Full TensorFlow runtime |
| **Dynamic shapes** | Limited support | Full support with recompilation |
| **Flexibility** | Fixed graph | Dynamic graph construction |
| **Debugging** | Limited | Full TF debugging tools |
| **Deployment** | Simple (single .o) | Complex (full TF installation) |
| **Cross-compilation** | Supported (target triple) | Same architecture only |
| **Performance** | Predictable | Variable (depends on JIT warmup) |
| **Use case** | Production serving, embedded | Development, research |

### When to Choose AOT

- Production deployment with strict latency requirements
- Mobile or embedded targets
- Environments where TensorFlow runtime is unavailable
- Regulatory requirements that prohibit JIT compilation

### When to Choose JIT

- Research and development
- Models with dynamic shapes
- Rapid prototyping
- Environments with full TensorFlow available

---

## XlaCompiledCpuFunction

The `XlaCompiledCpuFunction` class provides the runtime interface for AOT-compiled
functions. It manages buffer allocation and execution.

### Usage Pattern

```cpp
#include "my_computation.h"

int main() {
    // Prepare inputs
    float input[1 * 784];
    float weights[784 * 10];
    float output[1 * 10];

    // Set up arguments
    myns::MyComputation::Args args;
    args.arg0 = input;
    args.arg1 = weights;

    myns::MyComputation::Result result;
    result.result0 = output;

    // Run the computation
    myns::MyComputation::Run(args, &result);

    return 0;
}
```

### Buffer Management

```cpp
// Buffer info provides metadata about all buffers
const xla::cpu_function_runtime::BufferInfo* infos =
    myns::MyComputation::buffer_infos();
int info_count = myns::MyComputation::buffer_info_count();

// Temporary buffer size
int temp_size = myns::MyComputation::temp_buffer_size();
```

### Program Shape Access

```cpp
// Get the program shape (when gen_program_shape=true)
xla::ProgramShapeProto shape = myns::MyComputation::ProgramShape();

// Access parameter shapes
for (const auto& param : shape.parameters()) {
    std::cout << "Parameter: " << param.ShortDebugString() << std::endl;
}

// Access result shape
std::cout << "Result: " << shape.result().ShortDebugString() << std::endl;
```

### Name-to-Index Mapping

```cpp
// When gen_name_to_index=true
int arg_index = myns::MyComputation::LookupArgIndex("input");
int result_index = myns::MyComputation::LookupResultIndex("output");
```

---

## Warmup

### Compilation Cache Warmup

For JIT usage, compilation cache warmup pre-compiles for expected input shapes:

```python
# Pre-compile for expected shapes
@tf.function(jit_compile=True, input_signature=[
    tf.TensorSpec([32, 784], tf.float32)
])
def warmup_func(x):
    return x

# Trigger compilation
_ = warmup_func(tf.zeros([32, 784]))
```

### AOT "Warmup"

AOT compilation eliminates the need for runtime warmup since compilation happens
at build time. The generated object file is ready for immediate execution with
no startup latency.

### Memory Warmup

For production systems, pre-allocating buffers can reduce first-request latency:

```cpp
// Pre-allocate buffers
struct ComputationState {
    float input_buffer[BATCH_SIZE * INPUT_SIZE];
    float output_buffer[BATCH_SIZE * OUTPUT_SIZE];
    char temp_buffer[TEMP_SIZE];  // From temp_buffer_size()
};

// Use pre-allocated state
ComputationState state;
myns::MyComputation::Args args = {state.input_buffer, weights};
myns::MyComputation::Result result = {state.output_buffer};
myns::MyComputation::Run(args, &result);
```

---

## Codegen API

### GenerateMetadata

Generates a metadata object file containing serialized program shape and profiling
information:

```cpp
// From: tensorflow/compiler/aot/codegen.h

struct MetadataResult {
  std::vector<std::string> header_variable_decls;
  std::string program_shape_access_shim;
  std::string hlo_profile_printer_data_access_shim;
  std::string cpu_executable_access_shim;
  std::string object_file_data;
};

absl::Status GenerateMetadata(
    const CodegenOpts& opts,
    const CompileResult& compile_result,
    MetadataResult* metadata_result);
```

### GenerateHeader

Generates the C++ header file for the compiled computation:

```cpp
absl::Status GenerateHeader(
    const CodegenOpts& opts,
    const tf2xla::Config& config,
    const CompileResult& compile_result,
    const MetadataResult& metadata_result,
    const xla::EmbeddedConstantBuffers& embedded_constant_buffers,
    std::string* header);
```

### GenerateConstantBuffersData

Generates embedded constant buffers for weights that are compile-time constants:

```cpp
absl::StatusOr<xla::EmbeddedConstantBuffers> GenerateConstantBuffersData(
    const CodegenOpts& opts,
    const CompileResult& compile_result);
```

### ParseCppClass

Parses the C++ class specification into components:

```cpp
// Parses "myns::inner::MyClass" into:
//   class_name = "MyClass"
//   namespaces = ["myns", "inner"]
absl::Status ParseCppClass(
    const std::string& cpp_class,
    std::string* class_name,
    std::vector<std::string>* namespaces);
```

### ValidateCppIdent

Validates that a string is a valid C++ identifier:

```cpp
absl::Status ValidateCppIdent(
    absl::string_view ident,
    absl::string_view msg);
```

---

## AOT-Only Variable Handle Op

AOT compilation uses a special variable handle operation that does not require
the full TensorFlow resource manager:

```cpp
// From: tensorflow/compiler/aot/aot_only_var_handle_op.h
// A specialized variable handle for AOT-compiled functions
```

This allows resource variables to be passed as simple pointers rather than
requiring the TensorFlow resource management infrastructure.

---

## Quantization Support

### Experimental Quantization

tfcompile supports experimental quantization:

```bash
tfcompile \
  --graph=my_graph.pb \
  --config=my_config.pbtxt \
  --experimental_quantize \
  --entry_point=my_quantized_computation
```

### Quantization in Config Proto

Variables can specify quantization ranges:

```protobuf
variable {
  node_name: "my_weight"
  shape { dim { size: 784 } dim { size: 10 } }
  type: DT_FLOAT
  min: -1.0
  max: 1.0
}
```

---

## Embedded Protocol Buffers

The `EmbeddedProtocolBuffers` utility handles embedding serialized protocol buffers
into the generated object file:

```cpp
// From: tensorflow/compiler/aot/embedded_protocol_buffers.h
// Handles embedding protocol buffer data in AOT-compiled objects
```

---

## Thunk Proto Execution Deserializer

For thunk-based AOT compilation, the deserializer reconstructs the execution plan
from the serialized representation:

```cpp
// From: tensorflow/compiler/aot/thunk_proto_execution_deserializer.h
// Deserializes thunk execution plans for AOT execution
```

---

## Benchmark Support

tfcompile includes benchmarking utilities for AOT-compiled functions:

```cpp
// From: tensorflow/compiler/aot/benchmark.h

// Runs benchmarks on AOT-compiled functions
void RunBenchmark(
    const CompileResult& compile_result,
    int num_iterations,
    double* total_time_ms);
```

---

## Key Source Files Reference

| File Path | Description |
|-----------|-------------|
| `compiler/aot/compile.h` | Core compilation API |
| `compiler/aot/compile.cc` | Compilation implementation |
| `compiler/aot/codegen.h` | Header/code generation API |
| `compiler/aot/codegen.cc` | Code generation implementation |
| `compiler/aot/flags.h` | Command-line flag definitions |
| `compiler/aot/flags.cc` | Flag parsing |
| `compiler/aot/tfcompile_main.cc` | Main entry point for tfcompile |
| `compiler/aot/aot_only_var_handle_op.h` | AOT variable handling |
| `compiler/aot/embedded_protocol_buffers.h` | Embedded protobuf utilities |
| `compiler/aot/benchmark.h` | Benchmarking utilities |
| `compiler/aot/thunk_proto_execution_deserializer.h` | Thunk deserialization |

---

## Advanced AOT Patterns

### Multi-Function Compilation

A single tfcompile invocation can produce multiple functions from the same graph
by specifying multiple entry points:

```bash
tfcompile \
  --graph=my_graph.pb \
  --config=my_config.pbtxt \
  --entry_point=compute_features \
  --entry_point=compute_logits \
  --cpp_class="myns::MyModel"
```

### AOT with TPU

While the primary AOT target is CPU, similar principles apply to TPU compilation:

1. The graph is converted to HLO (same as CPU AOT)
2. HLO is compiled for the TPU target
3. The compiled program is serialized for TPU execution

### AOT Compilation for WebAssembly

tfcompile can target WebAssembly for browser-based inference:

```bash
tfcompile \
  --graph=my_graph.pb \
  --config=my_config.pbtxt \
  --target_triple=wasm32-unknown-unknown \
  --entry_point=my_computation \
  --out_function_object=my_computation.o
```

The generated object file is then compiled to WebAssembly using Emscripten:

```bash
emcc my_computation.o -o my_computation.js \
  -s WASM=1 \
  -s EXPORTED_FUNCTIONS='["_my_computation"]'
```

### AOT with Custom Call Targets

AOT-compiled functions can include custom call targets for operations not natively
supported by XLA:

```cpp
// Register a custom call target
extern "C" void my_custom_call(void* output, const void* input, int64_t size) {
    // Custom implementation
}

// The AOT-compiled function will call this at runtime
```

The custom call must be linked into the final binary alongside the AOT-compiled
object file.

### Memory Layout Optimization

AOT compilation allows precise control over memory layout:

- **Input buffer layout**: Controlled via the `ShapeLayout` in the config proto
- **Output buffer layout**: Determined by the compilation result
- **Temporary buffers**: Managed by the XLA runtime

### AOT Compilation Diagnostics

When compilation fails, tfcompile provides diagnostic information:

```
ERROR: Compilation failed for entry point "my_computation"
  - Unsupported operation: MyCustomOp at node "custom_node_1"
  - Shape mismatch: expected tensor<10xf32>, got tensor<20xf32>
```

Common diagnostic steps:
1. Verify the config proto matches the graph structure
2. Check that all feed/fetch nodes exist in the graph
3. Ensure all shapes in the config are correct
4. Verify the target triple is valid for the host platform

### AOT Compilation Performance

AOT compilation performance characteristics:

| Model Size | Compilation Time | Object File Size |
|-----------|-----------------|------------------|
| Small (<1M params) | 5-30 seconds | 1-10 MB |
| Medium (1-10M params) | 30-120 seconds | 10-100 MB |
| Large (>10M params) | 2-10 minutes | 100 MB - 1 GB |

Factors affecting compilation time:
- Number of operations in the graph
- Target architecture complexity
- Optimization level
- Fusion opportunities

### AOT and Quantization

AOT compilation can be combined with quantization for further optimization:

```bash
tfcompile \
  --graph=my_quantized_graph.pb \
  --config=my_config.pbtxt \
  --experimental_quantize \
  --entry_point=my_quantized_computation
```

Quantized AOT models:
- Use INT8 arithmetic where possible
- Maintain FP32 fallback for non-quantizable operations
- Include quantization parameters in the generated code
- Can achieve 2-4x speedup over FP32 AOT

### Integration with Build Systems

#### CMake Integration

```cmake
# CMakeLists.txt
add_custom_command(
    OUTPUT my_computation.o my_computation.h
    COMMAND tfcompile
        --graph ${CMAKE_SOURCE_DIR}/models/my_graph.pb
        --config ${CMAKE_SOURCE_DIR}/configs/my_config.pbtxt
        --entry_point my_computation
        --cpp_class "myns::MyComputation"
        --out_function_object my_computation.o
        --out_header my_computation.h
    DEPENDS my_graph.pb my_config.pbtxt
)

add_executable(my_app main.cc my_computation.o)
target_include_directories(my_app PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
```

#### Makefile Integration

```makefile
# Makefile
TF_COMPILE = tfcompile
GRAPH_DIR = models
CONFIG_DIR = configs

my_computation.o: $(GRAPH_DIR)/my_graph.pb $(CONFIG_DIR)/my_config.pbtxt
	$(TF_COMPILE) \
		--graph=$< \
		--config=$(word 2,$^) \
		--entry_point=my_computation \
		--cpp_class="myns::MyComputation" \
		--out_function_object=$@ \
		--out_header=my_computation.h

my_app: main.o my_computation.o
	$(CXX) -o $@ $^ -lm -lpthread
```

### AOT Error Handling

When AOT-compiled functions encounter runtime errors:

```cpp
// Check for null pointers before calling
if (args.arg0 == nullptr || result.result0 == nullptr) {
    // Handle error
    return -1;
}

// The compiled function does not throw exceptions
// All errors are handled through return values or output buffers
myns::MyComputation::Run(args, &result);
```

### AOT Security Considerations

AOT compilation offers security advantages:

- **No JIT at runtime**: Eliminates JIT-related attack surfaces
- **Deterministic execution**: No recompilation based on input
- **Static analysis**: The compiled code can be statically analyzed
- **Sandboxing**: Easier to sandbox since there is no TF runtime

However, consider:
- The object file contains the full computation graph structure
- Constant weights are embedded in the binary
- The generated code is architecture-specific
