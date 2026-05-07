# TFRT Runtime Reference

This document provides a comprehensive reference for TFRT (TensorFlow Runtime),
the next-generation TensorFlow runtime designed for improved performance,
modularity, and extensibility.

## Table of Contents

1. [TFRT Overview](#tfrt-overview)
2. [BEF (Binary Execution Format)](#bef-binary-execution-format)
3. [AsyncValue System](#asyncvalue-system)
4. [ExecutionContext and HostContext](#executioncontext-and-hostcontext)
5. [Kernel Registration](#kernel-registration)
6. [BEFExecutor](#befexecutor)
7. [DenseTensor Representation](#densortensor-representation)
8. [Conversion from GraphDef](#conversion-from-graphdef)
9. [Fallback Mechanism](#fallback-mechanism)
10. [Thread Pool and Work Queue](#thread-pool-and-work-queue)
11. [Cost Analysis](#cost-analysis)
12. [Comparison with Classic Runtime](#comparison-with-classic-runtime)
13. [Device Abstraction](#device-abstraction)
14. [Error Handling](#error-handling)
15. [Resource Management](#resource-management)

---

## TFRT Overview

TFRT is a new runtime for TensorFlow that addresses several limitations of
the classic TensorFlow runtime:

### Design Goals

1. **Performance**: Reduce overhead through efficient async execution and
   minimal synchronization
2. **Modularity**: Clear separation between runtime components (kernels,
   devices, execution)
3. **Extensibility**: Easy to add new devices, kernels, and optimizations
4. **Non-Graph Execution**: Support both graph-based and immediate (eager)
   execution modes
5. **Unified Runtime**: Single runtime for training and inference

### Architecture

```
+--------------------------------------------------+
|                   Client API                      |
|  (Python/C++ bindings, eager context)             |
+--------------------------------------------------+
|                   TFRT Core                       |
|  +----------+  +-----------+  +---------------+  |
|  | BEF      |  | AsyncValue|  | Kernel        |  |
|  | Executor |  | System    |  | Registry      |  |
|  +----------+  +-----------+  +---------------+  |
|  +----------+  +-----------+  +---------------+  |
|  | Host     |  | Device    |  | Work Queue    |  |
|  | Context  |  | Context   |  | (ThreadPool)  |  |
|  +----------+  +-----------+  +---------------+  |
+--------------------------------------------------+
|                   Device Layer                    |
|  +----------+  +-----------+  +---------------+  |
|  | CPU      |  | GPU       |  | TPU           |  |
|  | Device   |  | Device    |  | Device        |  |
|  +----------+  +-----------+  +---------------+  |
+--------------------------------------------------+
```

### Key Differences from Classic Runtime

| Aspect | Classic Runtime | TFRT |
|--------|----------------|------|
| Execution model | Executor + OpKernel | BEFExecutor + Kernels |
| Async values | Tensor/Status callbacks | AsyncValue/AsyncValueRef |
| Graph format | GraphDef (protobuf) | BEF (binary) |
| Device abstraction | DeviceBase | DeviceContext |
| Kernel interface | OpKernelContext | KernelContext |
| Error handling | Status/StatusCallback | Error in AsyncValue |
| Memory management | Allocator | Allocator (shared interface) |

---

## BEF (Binary Execution Format)

BEF is the serialized graph representation used by TFRT. It is a compact
binary format designed for fast loading and execution.

### BEF Structure

```
+-------------------+
| BEF Header        |
|  - Magic number   |
|  - Version        |
|  - Section offsets|
+-------------------+
| String Section    |
|  - Kernel names   |
|  - Attribute keys |
|  - Type names     |
+-------------------+
| Attribute Section |
|  - Dense attributes|
|  - Aggregate attrs|
|  - Function attrs |
+-------------------+
| Kernel Section    |
|  - Kernel entries |
|  - Kernel indices |
+-------------------+
| Register Section  |
|  - Register types |
|  - Register counts|
+-------------------+
| Instruction Section|
|  - Kernels        |
|  - Jump/Condition |
|  - Return         |
+-------------------+
| Function Section  |
|  - Function headers|
|  - Function bodies|
+-------------------+
```

### BEF File Components

#### Header

```
struct BefHeader {
  uint32_t magic;           // Magic number for validation
  uint32_t version;         // BEF format version
  uint32_t num_sections;    // Number of sections
  uint32_t section_offsets[]; // Offset of each section
};
```

#### Instructions

BEF instructions include:

| Instruction | Description |
|-------------|-------------|
| `kernel` | Execute a registered kernel |
| `jump` | Unconditional jump |
| `jump_if` | Conditional jump based on AsyncValue |
| `return` | Return values from a function |

#### Attributes

BEF supports several attribute types:

1. **Dense Attributes**: Raw byte arrays (for tensors, arrays)
2. **Aggregate Attributes**: Named collections of attributes
3. **Function Attributes**: References to BEF functions
4. **Type Attributes**: Data type specifications
5. **String Attributes**: String values

### BEF Encoding

- Uses little-endian byte order
- All offsets are relative to section start
- Strings are null-terminated
- Attributes are aligned to 8-byte boundaries
- Register indices are 4-byte unsigned integers

### BEF Creation Flow

```
1. MLIR Module
   |
2. TFRT Dialect Lowering
   |
3. BEF Translation (mlir::tfrt::translateModuleToBef)
   |
4. BEF Binary
   |
5. BEFExecutor loads and executes
```

---

## AsyncValue System

The AsyncValue system is TFRT's core abstraction for managing asynchronously
computed values. It replaces TensorFlow's traditional callback-based async
model.

### AsyncValue

`AsyncValue<T>` represents a value of type T that may not yet be available.
It is the fundamental building block for all data flow in TFRT.

#### State Machine

```
              +-----------+
              | Created   |
              +-----+-----+
                    |
         +----------+----------+
         |                     |
   Set value/            Set error/
   emplace               SetError
         |                     |
         v                     v
  +------+------+       +------+------+
  | Available   |       | Error       |
  | (has value) |       | (has error) |
  +------+------+       +------+------+
```

#### Key Operations

```cpp
// Check state
bool IsAvailable() const;    // True if value or error is set
bool IsError() const;        // True if error is set
bool IsValid() const;        // True if value is set (not error)

// Get value (blocks if not available)
T& get();                    // Get reference to value (undefined if error)
const T& get() const;

// Wait for availability
void AndThen(FrozenFunctionRef<void()> callback);  // Register callback
void BlockUntilAvailable();                        // Block current thread

// Construct
static AsyncValue* Create(HostContext* host);                      // Unconstructed
static AsyncValue* Create(T value, HostContext* host);             // With value
static AsyncValue* CreateError(HostContext* host, Error error);    // With error
```

#### Error Propagation

When an AsyncValue contains an error:
- `IsError()` returns true
- `GetError()` returns the error message
- Consumers that depend on this value also receive the error
- `AndThen` callbacks are still invoked (to propagate errors)

### AsyncValueRef

`AsyncValueRef<T>` is a reference-counted smart pointer to an `AsyncValue<T>`.

```cpp
template <typename T>
class AsyncValueRef {
 public:
  // Default construction (null)
  AsyncValueRef() = default;

  // Construction from AsyncValue
  explicit AsyncValueRef(AsyncValue* async_value);

  // Copy and move
  AsyncValueRef(const AsyncValueRef& other);
  AsyncValueRef(AsyncValueRef&& other);
  AsyncValueRef& operator=(const AsyncValueRef& other);
  AsyncValueRef& operator=(AsyncValueRef&& other);

  // Access
  T& get() const;
  T* operator->() const;
  T& operator*() const;

  // State queries
  bool IsAvailable() const;
  bool IsError() const;
  bool IsValid() const;

  // Callback registration
  void AndThen(FunctionRef<void()> callback) const;

  // Value construction
  void emplace(Args&&... args);     // Construct value in-place
  void SetError(Error error);       // Set error

  // Factory methods
  static AsyncValueRef<T> Make(T value, HostContext* host);
  static AsyncValueRef<T> MakeError(Error error, HostContext* host);
  static AsyncValueRef<T> MakeUnconstructed(HostContext* host);
};
```

### AsyncValue Types

TFRT supports several AsyncValue variants:

1. **Concrete AsyncValue**: Holds a specific C++ type
2. **Indirect AsyncValue**: Points to another AsyncValue (for forwarding)
3. **Chain AsyncValue**: Represents execution ordering (no data)
4. **Error AsyncValue**: Contains only an error

### Chain Values

Chain values are special AsyncValues used for sequencing:

```cpp
// A chain represents completion of an operation
using Chain = AsyncValueRef<ChainValue>;

// Operations that produce chains are sequenced
Chain input_chain = ...;
auto [result, output_chain] = ExecuteOp(input_chain, ...);
```

---

## ExecutionContext and HostContext

### HostContext

`HostContext` represents the host CPU device and provides shared infrastructure.

```cpp
class HostContext {
 public:
  // Construction
  explicit HostContext(std::unique_ptr<HostAllocator> allocator,
                       std::unique_ptr<WorkQueue> work_queue);

  // Memory allocation
  HostAllocator* allocator();
  void* AllocateBytes(size_t size, size_t alignment);
  void DeallocateBytes(void* ptr, size_t size);

  // AsyncValue factory
  template <typename T>
  AsyncValueRef<T> MakeAvailableAsyncValueRef(T value);
  template <typename T>
  AsyncValueRef<T> MakeUnconstructedAsyncValueRef();
  AsyncValueRef<Error> MakeErrorAsyncValueRef(Error error);

  // Work scheduling
  void EnqueueWork(FunctionRef<void()> work);
  void RunBlockingWork(FunctionRef<void()> work);

  // Blocking work
  bool InFlightWorkIsEmpty();

  // Kernel registry
  const KernelRegistry& GetKernelRegistry() const;

 private:
  std::unique_ptr<HostAllocator> allocator_;
  std::unique_ptr<WorkQueue> work_queue_;
  KernelRegistry kernel_registry_;
};
```

#### HostAllocator

```cpp
class HostAllocator {
 public:
  virtual void* AllocateBytes(size_t size, size_t alignment) = 0;
  virtual void DeallocateBytes(void* ptr, size_t size) = 0;
};
```

Implementations:
- `MallocAllocator`: Uses standard malloc/free
- `TcmallocAllocator`: Uses tcmalloc for improved performance
- `AlignedAllocator`: Ensures specific alignment requirements

### DeviceContext

Abstracts a specific device (CPU, GPU, TPU):

```cpp
class DeviceContext {
 public:
  virtual ~DeviceContext() = default;

  // Device identification
  virtual string_view device_type() const = 0;
  virtual int device_id() const = 0;

  // Memory management
  virtual HostAllocator* allocator() = 0;

  // Work scheduling
  virtual void EnqueueWork(FunctionRef<void()> work) = 0;

  // Data transfer
  virtual AsyncValueRef<void> TransferToHost(AsyncValueRef<void> device_data) = 0;
  virtual AsyncValueRef<void> TransferToDevice(AsyncValueRef<void> host_data) = 0;
};
```

### ExecutionContext

The context for a single kernel execution:

```cpp
class ExecutionContext {
 public:
  ExecutionContext(HostContext* host, Location loc);

  HostContext* host() const;
  Location location() const;

  // Convenience methods
  AsyncValueRef<Error> EmitError(string_view message);
};
```

The `ExecutionContext` is passed to every kernel and provides:
- Access to the `HostContext`
- Error reporting via `Location`
- Request ID for tracing

---

## Kernel Registration

### Kernel Traits

Kernels in TFRT are registered with compile-time type information:

```cpp
// Define kernel signature
using MyKernelSignature = void(Argument<int>, Argument<float>,
                               Result<int>, Result<float>,
                               Attribute<int>);

// Register kernel
TFRT_KERNEL(MyKernel, "my_kernel");
```

### Kernel Signature Components

| Component | Description |
|-----------|-------------|
| `Argument<T>` | Input tensor or value of type T |
| `Result<T>` | Output tensor or value of type T |
| `Attribute<T>` | Compile-time constant attribute |
| `Chain` | Execution chain for sequencing |
| `RemainingResults` | Variable number of results |
| `RemainingArguments` | Variable number of arguments |

### Kernel Implementation

```cpp
// Simple kernel
void AddKernel(Argument<int> a, Argument<int> b, Result<int> result) {
  result.Emplace(*a + *b);
}

// Kernel with attributes
void MatMulKernel(Argument<Tensor> a, Argument<Tensor> b,
                  Attribute<bool> transpose_a, Attribute<bool> transpose_b,
                  Result<Tensor> result) {
  // Compute matrix multiplication
}

// Kernel with chain
void PrintKernel(Chain chain, Argument<Tensor> input,
                 Attribute<string> message, Result<Chain> output_chain) {
  LOG(INFO) << message << ": " << input->DebugString();
  output_chain.Emplace();
}

// Kernel with error handling
void DivideKernel(Argument<float> a, Argument<float> b,
                  ExecutionContext exec_ctx, Result<float> result) {
  if (*b == 0.0f) {
    result.SetError(exec_ctx.EmitError("Division by zero"));
    return;
  }
  result.Emplace(*a / *b);
}
```

### Kernel Registry

```cpp
class KernelRegistry {
 public:
  // Register a kernel
  void AddKernel(string_view name, KernelImplementation kernel);

  // Lookup a kernel
  KernelImplementation LookupKernel(string_view name) const;
};
```

### Kernel Registration Macros

```cpp
// Register a simple kernel
TFRT_KERNEL(my_kernel_fn, "my.kernel.name");

// Register with specific device
TFRT_KERNEL_DEVICE(my_kernel_fn, "my.kernel.name", "cpu");

// Register with multiple overload signatures
TFRT_REGISTER_KERNEL(my_kernel_fn, "my.kernel.name",
                     Signature<int, int, int>,
                     Signature<float, float, float>);
```

---

## BEFExecutor

The `BEFExecutor` executes BEF programs. It interprets the BEF binary and
dispatches kernels.

### Execution Model

```
BEF Binary
    |
BEFExecutor::Execute()
    |
    +-- Parse instructions
    +-- Allocate registers (AsyncValueRef array)
    +-- Schedule ready kernels
    |
    +-- For each ready kernel:
    |   +-- Read input AsyncValues
    |   +-- Execute kernel function
    |   +-- Write output AsyncValues
    |   +-- Mark dependent kernels as ready
    |
    +-- Await all results
```

### Execution Flow

```cpp
class BEFExecutor {
 public:
  // Execute a BEF function
  void Execute(const ExecutionContext& exec_ctx,
               ArrayRef<AsyncValueRef<void>> arguments,
               MutableArrayRef<AsyncValueRef<void>> results);

  // Execute and await all results
  void ExecuteSync(const ExecutionContext& exec_ctx,
                   ArrayRef<AsyncValueRef<void>> arguments,
                   MutableArrayRef<AsyncValueRef<void>> results);
};
```

### Register Allocation

The executor allocates a fixed array of `AsyncValueRef` registers:

- Each instruction output is assigned a register
- Input arguments occupy the first N registers
- Results are read from designated registers

### Ready Queue

Kernels are scheduled when all their inputs are available:

```
for each instruction in BEF:
  count unavailable inputs
  when input becomes available:
    decrement count
    if count == 0:
      add to ready queue
```

### Async Dispatch

Kernels are dispatched to the work queue:

```cpp
// When a kernel becomes ready
host_context->EnqueueWork([kernel, inputs, results, exec_ctx]() {
  kernel.Execute(inputs, results, exec_ctx);
});
```

### Sequential vs Concurrent Execution

The executor supports both:
- **Sequential**: Kernels execute one at a time on a single thread
- **Concurrent**: Independent kernels execute in parallel on the work queue

---

## DenseTensor Representation

TFRT uses its own tensor representation optimized for the runtime.

### DenseTensor

```cpp
class DenseTensor {
 public:
  // Construction
  DenseTensor(TensorShape shape, ElementType element_type,
              RCReference<Buffer> buffer);

  // Shape and type
  const TensorShape& shape() const;
  ElementType element_type() const;

  // Data access
  void* data();
  const void* data() const;
  size_t size_in_bytes() const;

  // Buffer management
  RCReference<Buffer> buffer();
};
```

### TensorShape

```cpp
class TensorShape {
 public:
  TensorShape();                          // Scalar
  explicit TensorShape(ArrayRef<ssize_t> dims);

  ssize_t GetDimensionSize(int dim) const;
  int GetRank() const;
  ssize_t GetNumElements() const;
  ArrayRef<ssize_t> GetDimensions() const;
};
```

### ElementType

```cpp
enum class ElementType {
  // Standard types
  INT8, INT16, INT32, INT64,
  UINT8, UINT16, UINT32, UINT64,
  FLOAT16, FLOAT32, FLOAT64,
  BFLOAT16,
  BOOL,
  COMPLEX64, COMPLEX128,
  // Extended types
  STRING, RESOURCE, VARIANT
};
```

### Buffer

```cpp
class Buffer : public ReferenceCounted<Buffer> {
 public:
  void* data();
  size_t size() const;

  // Allocation
  static RCReference<Buffer> Create(HostAllocator* allocator,
                                    size_t size, size_t alignment);
};
```

---

## Conversion from GraphDef

TFRT can convert TensorFlow GraphDef programs to BEF for execution.

### Conversion Pipeline

```
1. GraphDef (TensorFlow)
   |
2. Import to MLIR (tf_dialect)
   |
3. Lower to TFRT dialect (tfrt_dialect)
   |
4. Optimize (tfrt optimizations)
   |
5. Translate to BEF
   |
6. Execute with BEFExecutor
```

### Conversion Steps

1. **GraphDef Import**: Parse the protobuf GraphDef into MLIR tf dialect
2. **Shape Inference**: Run shape inference on the MLIR graph
3. **Type Lowering**: Convert TensorFlow types to TFRT types
4. **Device Placement**: Assign operations to devices
5. **Kernel Mapping**: Map TensorFlow ops to TFRT kernels
6. **BEF Translation**: Generate BEF binary from MLIR

### Supported Conversions

- Most common TensorFlow ops are directly supported
- Unsupported ops fall back to the classic runtime (see Fallback Mechanism)
- Function inlining can be performed during conversion
- Control flow (while, if) is converted to BEF jump instructions

---

## Fallback Mechanism

TFRT supports falling back to the classic TensorFlow runtime for operations
that are not yet implemented in TFRT.

### Fallback Architecture

```
TFRT Execution
    |
    +-- Native TFRT kernels (fast path)
    |
    +-- Fallback kernels (when native not available)
        |
        +-- Create classic TensorFlow OpKernelContext
        +-- Execute classic OpKernel::Compute()
        +-- Convert results back to TFRT AsyncValues
```

### Fallback Kernel

```cpp
// A fallback kernel wraps a classic TensorFlow OpKernel
class FallbackKernel {
 public:
  FallbackKernel(std::unique_ptr<OpKernel> op_kernel);

  void Execute(ArrayRef<AsyncValueRef<void>> inputs,
               MutableArrayRef<AsyncValueRef<void>> results,
               const ExecutionContext& exec_ctx);
};
```

### Fallback Process

1. **Detect Unsupported Op**: During BEF conversion, ops without TFRT
   kernels are marked for fallback
2. **Create Fallback Kernel**: Wrap the classic OpKernel in a fallback adapter
3. **Convert Inputs**: Convert TFRT DenseTensors to classic TensorFlow Tensors
4. **Execute**: Run the classic OpKernel::Compute()
5. **Convert Outputs**: Convert results back to TFRT DenseTensors

### Performance Impact

- Fallback execution has overhead from data conversion
- Memory may be copied between TFRT and classic allocators
- Performance-critical ops should have native TFRT implementations

---

## Thread Pool and Work Queue

### WorkQueue Interface

```cpp
class WorkQueue {
 public:
  virtual ~WorkQueue() = default;

  // Enqueue a work item
  virtual void EnqueueTask(FunctionRef<void()> task) = 0;

  // Enqueue a blocking work item (uses separate thread pool)
  virtual void EnqueueBlockingTask(FunctionRef<void()> task) = 0;

  // Wait for all pending tasks
  virtual void AwaitPendingTasks() = 0;

  // Check if queue is empty
  virtual bool IsEmpty() const = 0;
};
```

### Thread Pool Implementations

1. **DefaultThreadPool**: Standard thread pool with configurable thread count
2. **FifoWorkQueue**: First-in-first-out scheduling
3. **PriorityWorkQueue**: Priority-based scheduling
4. **MultiThreadedWorkQueue**: Multiple thread pools for different task types

### Thread Configuration

```cpp
// Configure thread pool
auto work_queue = CreateMultiThreadedWorkQueue(
    /*num_threads=*/4,
    /*num_blocking_threads=*/2);

auto host = std::make_unique<HostContext>(
    std::make_unique<MallocAllocator>(),
    std::move(work_queue));
```

### Work Scheduling

- **EnqueueWork**: Non-blocking task, runs on any available thread
- **EnqueueBlockingWork**: May block (I/O, synchronization), uses separate pool
- **RunBlockingWork**: Runs blocking work on the caller's thread

---

## Cost Analysis

### Operation Cost Estimation

TFRT provides cost analysis for scheduling decisions:

```cpp
class CostAnalysis {
 public:
  // Estimated compute time in microseconds
  virtual int64_t EstimateComputeCost(string_view kernel_name,
                                      ArrayRef<TensorShape> input_shapes) = 0;

  // Estimated memory bytes accessed
  virtual int64_t EstimateMemoryCost(string_view kernel_name,
                                     ArrayRef<TensorShape> input_shapes) = 0;

  // Estimated data transfer time
  virtual int64_t EstimateTransferCost(size_t bytes,
                                       string_view src_device,
                                       string_view dst_device) = 0;
};
```

### Cost-Based Scheduling

The executor can use cost analysis to:
1. **Prioritize Expensive Ops**: Schedule compute-heavy ops first
2. **Overlap Transfer and Compute**: Schedule data transfers alongside compute
3. **Batch Small Ops**: Combine small operations to amortize scheduling overhead

---

## Comparison with Classic Runtime

### Performance Characteristics

| Metric | Classic Runtime | TFRT |
|--------|----------------|------|
| Graph loading | Protobuf parsing (slow) | BEF binary (fast) |
| Kernel dispatch | Virtual function + context setup | Direct function call |
| Async handling | Callback chains | AsyncValue + AndThen |
| Memory allocation | Per-op allocation | Batched allocation |
| Thread scheduling | Per-session thread pool | Shared work queue |
| Error propagation | Status return + callbacks | AsyncValue error state |

### Execution Overhead Reduction

1. **No protobuf serialization during execution**: BEF is loaded once
2. **Fewer allocations**: AsyncValues are reference-counted, not copied
3. **Better caching**: Kernel lookups use compile-time type information
4. **Reduced synchronization**: AsyncValue state machine is lock-free

### Feature Comparison

| Feature | Classic Runtime | TFRT |
|---------|----------------|------|
| Graph execution | Yes | Yes |
| Eager execution | Yes | Yes |
| AutoGraph | Yes | Partial |
| tf.function | Yes | Yes |
| Distributed training | Yes | Partial |
| SavedModel | Yes | Partial |
| Profiling | Yes (TensorBoard) | Partial |
| Debugging | Yes (tfdbg) | Partial |
| XLA compilation | Yes | Via fallback |

---

## Device Abstraction

### Device Interface

```cpp
class Device {
 public:
  virtual ~Device() = default;

  // Device identification
  virtual string_view name() const = 0;
  virtual string_view type() const = 0;
  virtual int id() const = 0;

  // Memory management
  virtual Allocator* GetAllocator() = 0;

  // Execution
  virtual void Execute(KernelInvocation invocation) = 0;

  // Data transfer
  virtual AsyncValueRef<void> TransferFrom(Device* src,
                                            AsyncValueRef<void> data) = 0;
};
```

### CPU Device

The default device for host operations:

```cpp
class CpuDevice : public Device {
  string_view type() const override { return "cpu"; }
  void Execute(KernelInvocation invocation) override {
    host_context_->EnqueueWork(std::move(invocation));
  }
};
```

### GPU Device

Integrates with GPU runtimes (CUDA, ROCm):

```cpp
class GpuDevice : public Device {
  string_view type() const override { return "gpu"; }
  void Execute(KernelInvocation invocation) override {
    // Submit to GPU stream
    gpu_stream_->Submit(std::move(invocation));
  }
};
```

---

## Error Handling

### Error Model

Errors in TFRT are represented as data within AsyncValues rather than as
separate Status objects:

```cpp
// Error is stored in the AsyncValue
auto async_value = host->MakeErrorAsyncValueRef(
    Error{"Division by zero"});

// Check for errors
if (async_value.IsError()) {
  auto error = async_value.GetError();
  LOG(ERROR) << error.message();
}
```

### Error Propagation

When a kernel produces an error:
1. The output AsyncValue is set to an error state
2. Downstream kernels that depend on this value skip execution
3. Their outputs are also set to the same error
4. The error propagates to the final results

### Location Tracking

```cpp
class Location {
 public:
  string_view filename() const;
  int line() const;
  string_view function_name() const;
};
```

Locations provide source-level error messages:

```
Error at my_model.py:42 in function 'train_step':
  Division by zero in kernel 'div'
```

---

## Resource Management

### Reference Counting

TFRT uses reference counting for resource management:

```cpp
template <typename T>
class RCReference {
 public:
  RCReference() = default;
  RCReference(T* ptr);  // Takes ownership

  T* get() const;
  T& operator*() const;
  T* operator->() const;

  void reset();          // Release reference
  T* release();          // Release without destroying

  // Copy (increments refcount)
  RCReference(const RCReference& other);
  RCReference& operator=(const RCReference& other);

  // Move (transfers ownership)
  RCReference(RCReference&& other);
  RCReference& operator=(RCReference&& other);
};
```

### Resource Cleanup

Resources are cleaned up when:
1. The last `RCReference` is destroyed
2. The reference count drops to zero
3. The destructor is called on a background thread (to avoid blocking)

### Memory Pool

TFRT supports memory pooling for common allocation sizes:

```cpp
class MemoryPool {
 public:
  RCReference<Buffer> Allocate(size_t size);
  void Deallocate(RCReference<Buffer> buffer);
};
```

---

## Building and Using TFRT

### CMake Integration

```cmake
add_executable(my_tfrt_app main.cpp)
target_link_libraries(my_tfrt_app
  tfrt_host_context
  tfrt_bef_executor
  tfrt_cpu_ops   # CPU kernel implementations
)
```

### Basic Usage

```cpp
#include "tfrt/host_context/host_context.h"
#include "tfrt/bef_executor/bef_executor.h"
#include "tfrt/bef_converter/bef_to_mlir.h"

// Create host context
auto host = std::make_unique<HostContext>(
    std::make_unique<MallocAllocator>(),
    CreateMultiThreadedWorkQueue(4, 2));

// Load BEF
auto bef_buffer = LoadBefFile("model.bef");

// Create executor
BEFExecutor executor(host.get());

// Create input arguments
std::vector<AsyncValueRef<void>> args;
args.push_back(host->MakeAvailableAsyncValueRef<Tensor>(...));

// Execute
std::vector<AsyncValueRef<void>> results(num_results);
executor.ExecuteSync(args, results);

// Get results
auto result = results[0].get<Tensor>();
```

### Python Integration

```python
# Enable TFRT
import tensorflow as tf
tf.enable_tfrt()

# Use TFRT for eager execution
# Operations are executed via TFRT runtime
x = tf.constant([1.0, 2.0, 3.0])
y = tf.reduce_sum(x)
```

---

## Debugging and Profiling

### BEF Debugging

```bash
# Dump BEF contents
tfrt_bef_dump model.bef

# Convert BEF to text
tfrt_bef_to_text model.bef model.txt
```

### Runtime Profiling

```cpp
// Enable profiling
auto profiler = CreateHostProfiler(host.get());
profiler->Start();

// ... execute ...

auto profile = profiler->Stop();
for (const auto& event : profile.events()) {
  LOG(INFO) << event.kernel_name << ": " << event.duration_us << " us";
}
```

### AsyncValue Tracing

```cpp
// Trace AsyncValue state transitions
async_value.AndThen([]() {
  LOG(INFO) << "Value became available";
});
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `TFRT_ENABLE` | Enable TFRT runtime |
| `TFRT_THREAD_POOL_SIZE` | Number of threads in work queue |
| `TFRT_BLOCKING_THREAD_POOL_SIZE` | Number of blocking threads |
| `TFRT_USE_TFRT_FOR_EAGER` | Use TFRT for eager execution |
| `TFRT_BEF_DUMP` | Dump BEF to file before execution |
| `TFRT_ASYNC_VALUE tracing` | Enable AsyncValue state tracing |
| `TFRT_FALLBACK_FOR_UNIMPLEMENTED` | Use fallback for unimplemented ops |
