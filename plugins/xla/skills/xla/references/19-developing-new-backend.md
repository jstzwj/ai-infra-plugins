# Developing a New XLA Backend

This document provides comprehensive guidance for developing a new XLA backend to target hardware that is not currently supported. XLA's modular architecture makes it possible to add support for new hardware accelerators, CPUs, or other compute devices by implementing a well-defined set of interfaces.

## Table of Contents

- [Overview](#overview)
- [Three Scenarios for Backend Development](#three-scenarios-for-backend-development)
- [Required Classes to Implement](#required-classes-to-implement)
- [Build System Integration](#build-system-integration)
- [Reference Implementations](#reference-implementations)
- [Tips and Best Practices](#tips-and-best-practices)

## Overview

XLA (Accelerated Linear Algebra) is a domain-specific compiler for linear algebra that can produce optimized executables for various hardware backends. Adding a new backend involves implementing several key abstract interfaces that allow XLA to compile High-Level Optimizer (HLO) instructions into executables that run on your target hardware.

The backend development process varies significantly depending on the nature of the target hardware and the existing compiler infrastructure available for it. XLA already ships with CPU and GPU (NVIDIA) backends that serve as reference implementations, and studying these is essential when developing a new backend.

## Three Scenarios for Backend Development

### Scenario 1: Existing CPU Architecture Not Yet Supported (With/Without LLVM Backend)

This scenario applies when you want to target a CPU architecture that XLA does not currently support (e.g., RISC-V, older ARM variants, or custom processor architectures).

#### With an Existing LLVM Backend

If your CPU architecture already has an LLVM backend, the process is relatively straightforward:

1. **Leverage the existing XLA CPU backend**: XLA's CPU backend already uses LLVM for code generation. If LLVM supports your target architecture, you primarily need to configure the target triple and data layout.

2. **Configuration changes**:
   - Update the LLVM target triple in the CPU compiler configuration.
   - Ensure the data layout matches your architecture (endianness, pointer size, register width).
   - Configure vectorization parameters appropriate for your architecture's SIMD capabilities.

3. **Build configuration**:
   - Add your architecture to the Bazel build configuration for the CPU backend.
   - Ensure the LLVM build includes the target backend (e.g., `LLVM_TARGETS_TO_BUILD`).

4. **Testing**: Run the existing XLA CPU test suite on your target hardware to validate correctness.

#### Without an Existing LLVM Backend

If your CPU architecture does not have an LLVM backend, you have two options:

1. **Write an LLVM backend**: This is a substantial undertaking but provides the best long-term integration. You would need to:
   - Implement instruction selection (ISel) for your architecture.
   - Implement register allocation.
   - Implement scheduling and code emission.
   - This approach allows you to reuse all of XLA's LLVM-based optimization passes.

2. **Write a custom XLA backend from scratch**: If writing an LLVM backend is not feasible, you can implement the XLA compiler interface directly, translating HLO instructions to your own machine code or intermediate representation.

### Scenario 2: Non-CPU Hardware With LLVM Backend

This scenario applies to hardware accelerators, DSPs, or other non-CPU devices that have an LLVM backend available. Examples might include certain AI accelerators with LLVM-based compilers.

The development process involves:

1. **Implement StreamExecutor**: Create a `StreamExecutor` implementation that manages your device's compute streams, memory, and execution queues. This is your primary device abstraction layer.

2. **Implement xla::Compiler**: Create a compiler that translates HLO into your device's executable format. Since you have an LLVM backend, the compilation pipeline would be:
   - HLO optimization passes (reuse XLA's generic passes)
   - HLO-to-LLVM IR lowering
   - LLVM optimization passes
   - LLVM backend code generation for your target

3. **Implement xla::Executable**: Create an executable class that can load and run the compiled code on your device, managing kernel launches and synchronization.

4. **Implement xla::TransferManager**: Handle host-to-device and device-to-host data transfers, accounting for any differences in memory layout or data representation.

5. **Key considerations**:
   - Memory addressing may differ between host and device.
   - Alignment requirements may be different.
   - You may need custom intrinsics for hardware-specific operations.
   - Threading and synchronization models may differ.

### Scenario 3: Non-CPU Hardware Without LLVM Backend

This is the most complex scenario. It applies to custom AI accelerators, FPGAs, or other specialized hardware without an LLVM backend.

The development process requires:

1. **Full custom compilation pipeline**: You must implement the entire HLO-to-executable compilation flow:
   - HLO optimization (can reuse XLA's generic passes)
   - HLO analysis and scheduling for your hardware
   - Custom code generation (HLO directly to your ISA or intermediate representation)
   - Binary generation and packaging

2. **Comprehensive StreamExecutor implementation**: Your device abstraction must handle:
   - Multiple compute units or cores
   - Device memory management (potentially with custom allocators)
   - Stream scheduling and dependency tracking
   - Event synchronization primitives

3. **Custom transfer mechanisms**: Data transfer between host and device may require:
   - Custom serialization/deserialization
   - DMA engine programming
   - Potentially custom data format conversion

4. **Verification strategy**: Without LLVM's existing test infrastructure, you need:
   - Comprehensive unit tests for each HLO operation
   - Integration tests comparing against a reference (CPU) backend
   - Performance regression tests

## Required Classes to Implement

### StreamExecutor: Device Abstraction and Stream Management

`StreamExecutor` is the primary abstraction for interacting with a hardware device. It provides a platform-independent interface for managing device memory, launching computations, and synchronizing execution.

#### Key Responsibilities

```cpp
class StreamExecutor {
 public:
  // Device memory management
  virtual DeviceMemoryBase Allocate(uint64_t size) = 0;
  virtual void Deallocate(DeviceMemoryBase* memory) = 0;
  virtual bool HostMemoryToDeviceMemory(
      const void* host_memory, DeviceMemoryBase* device_memory,
      uint64_t size) = 0;
  virtual bool DeviceMemoryToHostMemory(
      const DeviceMemoryBase& device_memory, void* host_memory,
      uint64_t size) = 0;

  // Stream management
  virtual Stream* AllocateStream() = 0;
  virtual void DeallocateStream(Stream* stream) = 0;

  // Synchronization
  virtual bool SynchronizeAllActivity() = 0;

  // Device information
  virtual int device_ordinal() const = 0;
  virtual const DeviceDescription& GetDeviceDescription() const = 0;
};
```

#### Stream Class

The `Stream` class represents an ordered sequence of operations on a device:

```cpp
class Stream {
 public:
  // Enqueue operations
  virtual void ThenLaunch(ThreadDim thread_dim,
                          BlockDim block_dim,
                          const KernelBase& kernel,
                          const Args& args) = 0;

  // Synchronization
  virtual void ThenWaitFor(Stream* other) = 0;
  virtual Status BlockHostUntilDone() = 0;
};
```

#### Implementation Notes

- Your `StreamExecutor` implementation should handle device initialization, including loading firmware or microcode if needed.
- Stream management should map to your device's native queue or command buffer mechanism.
- Memory allocation should use your device's memory allocator, potentially with memory pooling for performance.
- Consider implementing `StreamExecutorMemoryAllocator` for integration with XLA's buffer allocation system.

### xla::Compiler: HLO to Executable Compilation

The `xla::Compiler` class is responsible for translating HLO modules into `Executable` objects that can run on your hardware.

#### Key Interface

```cpp
class Compiler {
 public:
  // Main compilation entry point
  virtual StatusOr<std::unique_ptr<Executable>> Compile(
      const HloModule& module,
      const CompilationEnvironments& compile_env) = 0;

  // Run HLO passes on the module (optimization)
  virtual Status RunHloPasses(
      std::unique_ptr<HloModule> module,
      const CompilationEnvironments& compile_env) = 0;

  // Backend-specific compilation (after HLO passes)
  virtual StatusOr<std::unique_ptr<Executable>> RunBackend(
      std::unique_ptr<HloModule> module,
      const CompilationEnvironments& compile_env) = 0;

  // Shape inference for the target platform
  virtual StatusOr<std::vector<std::pair<Shape, Shape>>> GetDefaultShapeFn(
      HloInstruction* instruction) const = 0;

  // Hlo cost analysis for the target platform
  virtual HloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction() const = 0;

  // Target-specific configurations
  virtual se::Platform::Id PlatformId() const = 0;
};
```

#### Compilation Pipeline

A typical compilation pipeline for a new backend includes:

1. **HLO Verification**: Validate the HLO module structure and ensure all operations are supported.

2. **Target-independent HLO optimizations**: Run XLA's built-in optimization passes:
   - Algebraic simplifier
   - Constant folding
   - Dead code elimination
   - Fusion (various fusion strategies)
   - Layout assignment

3. **Target-specific HLO optimizations**: Custom passes for your hardware:
   - Operation legalization (replacing unsupported ops with supported alternatives)
   - Tiling for your hardware's memory hierarchy
   - Vectorization or parallelization for your compute units

4. **Code generation**: Translate the optimized HLO into your target format:
   - If using LLVM: HLO -> LLVM IR -> LLVM optimization -> machine code
   - If custom: HLO -> your intermediate representation -> machine code/binary

5. **Executable creation**: Package the generated code with metadata needed for execution.

### xla::Executable: Launching Compiled Computation

The `xla::Executable` class encapsulates a compiled computation that can be executed on the device.

#### Key Interface

```cpp
class Executable {
 public:
  // Execute the computation
  virtual StatusOr<ExecutionOutput> Execute(
      const ServiceExecutableRunOptions& run_options,
      std::vector<ExecutionInput> arguments) = 0;

  // Execute asynchronously (returning a future)
  virtual StatusOr<std::future<StatusOr<ExecutionOutput>>> ExecuteAsync(
      const ServiceExecutableRunOptions& run_options,
      std::vector<ExecutionInput> arguments) = 0;

  // Get the HLO module (for debugging)
  virtual const HloModule& module() const = 0;

  // Get shaping information
  virtual StatusOr<Shape> GetResultShape() const = 0;

  // Resource information
  virtual int64_t SizeOfGeneratedCodeInBytes() = 0;
};
```

#### Implementation Notes

- The `Execute` method must handle argument binding (mapping input buffers to the expected locations), kernel launch, and output collection.
- For asynchronous execution, use your device's native event/notification mechanism.
- Consider caching compiled executables if your hardware supports persistent code storage.
- Profile data can be attached to the executable for performance analysis.

### xla::TransferManager: Host-Device Data Transfer

The `TransferManager` class handles data movement between the host and device, including any necessary format conversion.

#### Key Interface

```cpp
class TransferManager {
 public:
  // Transfer host data to device
  virtual Status TransferLiteralToDeviceAsync(
      StreamExecutor* executor,
      se::Stream* stream,
      const LiteralSlice& literal,
      const ShapedBuffer& device_buffer) = 0;

  // Transfer device data to host
  virtual Status TransferLiteralFromDeviceAsync(
      StreamExecutor* executor,
      se::Stream* stream,
      const ShapedBuffer& device_buffer,
      MutableLiteralSlice* literal) = 0;

  // Buffer management
  virtual StatusOr<std::unique_ptr<ShapedBuffer>> AllocateShapedBuffer(
      const Shape& shape,
      StreamExecutor* executor) = 0;

  // Shape information
  virtual StatusOr<Shape> ChooseCompactLayoutForShape(
      const Shape& shape) const = 0;
};
```

#### Implementation Notes

- Handle endianness conversion if your device has a different byte order from the host.
- Handle alignment requirements for your device's memory accesses.
- Consider using DMA or your device's native transfer mechanism for large data.
- Implement efficient transfer for common tensor layouts (row-major, column-major).
- Support for non-contiguous tensors may require special handling.

## Build System Integration

### Bazel Build Rules

XLA uses Bazel as its build system. Integrating a new backend requires adding appropriate build rules.

#### Directory Structure

```
xla/
  service/
    my_backend/          # Your backend implementation
      BUILD              # Build rules
      my_backend_compiler.h
      my_backend_compiler.cc
      my_backend_executable.h
      my_backend_executable.cc
      my_transfer_manager.h
      my_transfer_manager.cc
  stream_executor/
    my_platform/         # Your StreamExecutor implementation
      BUILD
      my_executor.h
      my_executor.cc
      my_stream.h
      my_stream.cc
```

#### BUILD File Example

```python
# xla/service/my_backend/BUILD

load("//xla:xla.bzl", "xla_cc_library", "xla_cc_test")

package(default_visibility = ["//visibility:public"])

xla_cc_library(
    name = "my_backend_compiler",
    srcs = ["my_backend_compiler.cc"],
    hdrs = ["my_backend_compiler.h"],
    deps = [
        "//xla:hlo",
        "//xla:compiler",
        "//xla:executable",
        "//xla/service:compilation_environments",
        "//xla/service:hlo_pass_pipeline",
        "//xla/stream_executor/my_platform:my_executor",
        "@llvm-project//llvm:support",
    ],
)

xla_cc_library(
    name = "my_backend_executable",
    srcs = ["my_backend_executable.cc"],
    hdrs = ["my_backend_executable.h"],
    deps = [
        "//xla:executable",
        "//xla/stream_executor/my_platform:my_executor",
    ],
)

xla_cc_library(
    name = "my_transfer_manager",
    srcs = ["my_transfer_manager.cc"],
    hdrs = ["my_transfer_manager.h"],
    deps = [
        "//xla:transfer_manager",
        "//xla/stream_executor/my_platform:my_executor",
    ],
)

xla_cc_test(
    name = "my_backend_compiler_test",
    srcs = ["my_backend_compiler_test.cc"],
    deps = [
        ":my_backend_compiler",
        "//xla:test_helpers",
        "@com_google_googletest//:gtest_main",
    ],
)
```

### Platform Registration

Your backend must register itself with XLA's platform mechanism so that it can be discovered at runtime.

#### StreamExecutor Platform Registration

```cpp
// xla/stream_executor/my_platform/my_platform.cc

#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform/initialize.h"

namespace stream_executor {
namespace my_platform {

class MyPlatform : public Platform {
 public:
  MyPlatform() : Platform("MyPlatform", /*id=*/kMyPlatformId) {}

  std::string Name() const override { return "MyPlatform"; }
  int64_t VisibleDeviceCount() const override;
  StatusOr<std::unique_ptr<StreamExecutor>> ExecutorForDevice(
      int ordinal) override;
  StatusOr<std::unique_ptr<StreamExecutor>> ExecutorForDeviceWithPluginConfig(
      int ordinal, const PluginConfig& config) override;

 private:
  static void Register();
};

SE_REGISTER_PLATFORM(MyPlatform);

}  // namespace my_platform
}  // namespace stream_executor
```

#### Compiler Registration

```cpp
// xla/service/my_backend/my_backend_compiler.cc

#include "xla/service/compiler.h"
#include "xla/service/my_backend/my_backend_compiler.h"

namespace xla {
namespace my_backend {

class MyBackendCompiler : public Compiler {
  // ... implementation ...
};

REGISTER_COMPILER(my_platform::kMyPlatformId, MyBackendCompiler);

}  // namespace my_backend
}  // namespace xla
```

#### Transfer Manager Registration

```cpp
// xla/service/my_backend/my_transfer_manager.cc

#include "xla/service/transfer_manager.h"
#include "xla/service/my_backend/my_transfer_manager.h"

namespace xla {
namespace my_backend {

class MyTransferManager : public TransferManager {
  // ... implementation ...
};

REGISTER_TRANSFER_MANAGER(my_platform::kMyPlatformId, MyTransferManager);

}  // namespace my_backend
}  // namespace xla
```

## Reference Implementations

### CPUCompiler

The CPU backend (`xla/service/cpu/`) is the simplest reference implementation. Key files to study:

- **`cpu_compiler.cc`**: Shows the complete compilation pipeline for CPU, including LLVM IR generation, optimization, and JIT compilation.
- **`cpu_executable.cc`**: Demonstrates how to wrap JIT-compiled LLVM functions as XLA executables.
- **`cpu_transfer_manager.cc`**: Shows data transfer for the (relatively simple) case where host and device are the same.
- **`llvm_ir_gen.cc`**: The core code generation logic that translates HLO to LLVM IR.

#### CPU Compilation Pipeline

```cpp
StatusOr<std::unique_ptr<Executable>> CpuCompiler::Compile(
    const HloModule& module,
    const CompilationEnvironments& compile_env) {
  // 1. Run HLO-level optimization passes
  TF_ASSIGN_OR_RETURN(auto optimized_module,
                       RunHloPasses(module.Clone(), compile_env));

  // 2. Assign memory layout
  TF_ASSIGN_OR_RETURN(auto buffer_assignment,
                       BufferAssigner::Run(optimized_module.get(), ...));

  // 3. Generate LLVM IR
  llvm::Module llvm_module("xla_cpu_module", llvm_context);
  IrEmitter ir_emitter(...);
  TF_RETURN_IF_ERROR(ir_emitter.EmitModule(optimized_module.get(),
                                             buffer_assignment.get()));

  // 4. Optimize LLVM IR
  OptimizeLLVMIR(&llvm_module);

  // 5. JIT compile
  auto executable = std::make_unique<CpuExecutable>(...);
  return executable;
}
```

### GPUCompiler

The GPU backend (`xla/service/gpu/`) is a more complex reference implementation. Key files:

- **`gpu_compiler.cc`**: Shows the GPU compilation pipeline, including HLO optimization, LLVM IR generation, PTX emission, and cubin compilation.
- **`gpu_executable.cc`**: Demonstrates GPU kernel loading and execution using CUDA or ROCm streams.
- **`gpu_transfer_manager.cc`**: Shows host-to-GPU data transfer with proper memory pinning and DMA.
- **`nvptx_compiler.cc`** or **`amdgpu_compiler.cc`**: Target-specific compilation steps.

#### GPU Compilation Pipeline

```cpp
StatusOr<std::unique_ptr<Executable>> GpuCompiler::Compile(
    const HloModule& module,
    const CompilationEnvironments& compile_env) {
  // 1. Run HLO-level optimization passes (including GPU-specific ones)
  TF_ASSIGN_OR_RETURN(auto optimized_module,
                       RunHloPasses(module.Clone(), compile_env));

  // 2. Assign buffers
  TF_ASSIGN_OR_RETURN(auto buffer_assignment,
                       BufferAssigner::Run(optimized_module.get(), ...));

  // 3. Generate LLVM IR for GPU kernels
  llvm::Module llvm_module("xla_gpu_module", llvm_context);
  // ... emit kernel IR ...

  // 4. Lower LLVM IR to PTX (NVIDIA) or AMDGCN (AMD)
  std::string ptx = EmitPTX(&llvm_module, gpu_device_info);

  // 5. Compile PTX to cubin using ptxas or driver API
  std::string cubin = CompilePTXToCubin(ptx, gpu_device_info);

  // 6. Create GPU executable
  auto executable = std::make_unique<GpuExecutable>(
      std::move(ptx), std::move(cubin), ...);
  return executable;
}
```

## Tips and Best Practices

### Start with the CPU Backend as a Template

The CPU backend is the simplest and most self-contained. Clone it and modify it incrementally:

1. Copy `xla/service/cpu/` to `xla/service/my_backend/`.
2. Rename all classes and files.
3. Get it to compile with the same functionality as the CPU backend.
4. Gradually replace the LLVM code generation with your target-specific code.

### Implement HLO Operations Incrementally

Do not try to implement all HLO operations at once. Start with a minimal subset:

1. **Core operations**: `add`, `multiply`, `subtract`, `divide` (element-wise).
2. **Shape operations**: `reshape`, `transpose`, `broadcast`, `slice`.
3. **Reduction**: `reduce` (used by many higher-level operations).
4. **MatMul**: `dot` or `convolution` (depending on your hardware's focus).
5. **Control flow**: `while`, `conditional` (can be lowered to basic operations initially).

Use HLO legalization passes to convert unsupported operations into sequences of supported ones. For example, if your hardware does not support `log` directly, you can legalize it to a polynomial approximation.

### Leverage XLA's Built-in Optimization Passes

XLA includes many target-independent optimization passes that you can reuse:

- **AlgebraicSimplifier**: Simplifies arithmetic expressions.
- **ConstantFolding**: Evaluates constant expressions at compile time.
- **DeadCodeElimination**: Removes unused computations.
- **HloCSE**: Common subexpression elimination.
- **LayoutAssignment**: Assigns memory layouts.
- **Fusion**: Combines operations to reduce memory bandwidth.

Use `HloPassPipeline` to compose these passes:

```cpp
Status RunHloPasses(std::unique_ptr<HloModule> module, ...) override {
  HloPassPipeline pipeline("my_backend_hlo_passes");

  // Target-independent passes
  pipeline.AddPass<AlgebraicSimplifier>(...);
  pipeline.AddPass<ConstantFolding>();
  pipeline.AddPass<DeadCodeElimination>();

  // Target-specific passes
  pipeline.AddPass<MyBackendLegalizer>();
  pipeline.AddPass<MyBackendTilingPass>();

  return pipeline.Run(module).status();
}
```

### Use Buffer Assignment Carefully

Buffer assignment determines how tensors are mapped to device memory. This has a major impact on performance:

- Study the `BufferAssigner` and `BufferAssignment` classes.
- Consider implementing a custom buffer allocator if your hardware has memory constraints.
- Pay attention to buffer aliasing opportunities for in-place operations.
- Implement proper buffer sharing for operations like `reshape` and `transpose`.

### Implement Comprehensive Testing

Testing a new backend requires multiple layers:

1. **Unit tests**: Test each component (compiler, executor, transfer manager) in isolation.
2. **HLO operation tests**: Test each supported HLO operation with various shapes and data types.
3. **Integration tests**: Test end-to-end compilation and execution of real models.
4. **Correctness tests**: Compare outputs against the CPU reference backend.

```python
# Example test using XLA's test infrastructure
class MyBackendTest : public HloTestBase {
 protected:
  void RunTest(const std::string& hlo_text,
               const Literal& expected) {
    auto module = ParseAndReturnVerifiedModule(hlo_text).value();
    auto executable = CompileToExecutable(std::move(module));
    auto result = Execute(std::move(executable), {});
    EXPECT_EQ(result, expected);
  }
};
```

### Handle Errors Gracefully

Provide clear error messages when operations are not supported or when hardware constraints are violated:

- Use `Unimplemented()` status for unsupported operations with a clear message explaining what is not supported and suggesting alternatives.
- Use `InvalidArgument()` for invalid parameter values.
- Include operation details in error messages (shape, data type, etc.).

### Consider Performance from the Start

While correctness comes first, keep performance in mind:

- Profile your backend early to identify bottlenecks.
- Focus optimization effort on the most frequently executed operations.
- Consider operation fusion to reduce memory traffic.
- Use your hardware's native memory layout to avoid unnecessary transpose operations.
- Implement efficient kernel launch mechanisms to minimize overhead.

### Document Your Backend

Maintain clear documentation for:

- Supported HLO operations and any limitations.
- Hardware-specific behavior (rounding modes, overflow handling, etc.).
- Performance characteristics and tuning options.
- Known issues and workarounds.

### Interact with the XLA Community

XLA is an open-source project with an active community:

- Engage with the XLA team early if you plan to contribute your backend upstream.
- Follow the XLA development mailing list and issue tracker.
- Contribute fixes for generic issues you encounter during backend development.
- Consider making your backend a plugin (using PJRT) rather than forking XLA, if that fits your use case.

### PJRT Plugin Alternative

For many use cases, implementing a PJRT plugin is a more practical approach than developing a full XLA backend. PJRT provides a stable C API that abstracts the device interface, allowing you to integrate with JAX, TensorFlow, and other frameworks without modifying XLA itself. See the PJRT API documentation (File 20) for details.
