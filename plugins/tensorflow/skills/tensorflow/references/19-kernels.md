# TensorFlow Op Kernels

This reference covers TensorFlow's kernel system: the `OpKernel` base class,
the `OpKernelConstruction` and `OpKernelContext` classes for kernel
implementation, `AsyncOpKernel` for asynchronous operations, input/output
management, tensor forwarding, registration macros, and common implementation
patterns.

---

## Table of Contents

1. [OpKernel Base Class](#opkernel-base-class)
2. [OpKernelConstruction](#opkernelconstruction)
3. [OpKernelContext](#opkernelcontext)
4. [AsyncOpKernel](#asynccpkernel)
5. [TensorBuffer](#tensorbuffer)
6. [PersistentTensor](#persistenttensor)
7. [Registration Macros](#registration-macros)
8. [Device-Specific Kernels](#device-specific-kernels)
9. [Input/Output Management](#inputoutput-management)
10. [Resource Kernels](#resource-kernels)
11. [Gradient Kernels](#gradient-kernels)
12. [Common Kernel Utilities](#common-kernel-utilities)
13. [Kernel Implementation Patterns](#kernel-implementation-patterns)

---

## OpKernel Base Class

**Header:** `tensorflow/core/framework/op_kernel.h`
**Namespace:** `tensorflow`

`OpKernel` is the abstract base class for all TensorFlow operation
implementations. Every op registered in TensorFlow must have a corresponding
`OpKernel` subclass that implements the `Compute()` method.

### Class Definition

```cpp
class OpKernel {
 public:
  // Constructor. Called once during graph creation (not per-execution).
  // May perform expensive initialization.
  explicit OpKernel(OpKernelConstruction* context);

  // Constructor for deferred ops.
  OpKernel(OpKernelConstruction* context, bool is_deferred);

  // Constructor with custom NodeDef.
  OpKernel(OpKernelConstruction* context, NodeDef&& custom_def,
           bool is_deferred);

  virtual ~OpKernel();

  // The main computation method.
  // Must be thread-safe (may be called concurrently by multiple graph runs).
  // For synchronous ops: override this method and return after completing work.
  // IMPORTANT: Must NOT block on synchronization from another OpKernel.
  virtual void Compute(OpKernelContext* context) = 0;

  // Returns nullptr for sync ops (override in AsyncOpKernel).
  virtual AsyncOpKernel* AsAsync();

  // Whether this kernel is considered "expensive".
  // Runtime may use this to optimize scheduling.
  virtual bool IsExpensive();

  // Returns pointer to constant tensor (for ConstantOp).
  virtual const Tensor* const_tensor() const;

  // --- Accessors ---

  const NodeDef& def() const;
  const std::string& name() const;
  absl::string_view name_view() const;
  const std::string& type_string() const;
  absl::string_view type_string_view() const;
  const std::string& requested_input(int i) const;
  const std::string& requested_device() const;

  // Input/output counts and types.
  int num_inputs() const;
  DataType input_type(int i) const;
  const DataTypeVector& input_types() const;
  const MemoryTypeVector& input_memory_types() const;

  int num_outputs() const;
  DataType output_type(int o) const;
  const DataTypeVector& output_types() const;
  const MemoryTypeVector& output_memory_types() const;

  // Get the range of inputs/outputs for a named argument.
  absl::Status InputRange(absl::string_view input_name,
                          int* start, int* stop) const;
  absl::Status OutputRange(absl::string_view output_name,
                           int* start, int* stop) const;

  // Whether this kernel uses deferred execution.
  bool is_deferred() const;

  // Trace string for profiler.
  virtual std::string TraceString(const OpKernelContext& ctx,
                                  bool verbose) const;

 protected:
  std::string ShapeTraceString(const OpKernelContext& ctx) const;

 private:
  const std::shared_ptr<const NodeProperties> props_;
  const MemoryTypeVector input_memory_types_;
  const MemoryTypeVector output_memory_types_;
  NameRangeMap input_name_map_;
  NameRangeMap output_name_map_;
  const absl::string_view name_view_;
  const absl::string_view type_string_view_;
  const int graph_def_version_;
  const bool is_deferred_;
  bool expensive_;
};
```

### Thread Safety Requirements

- `Compute()` may be called concurrently by multiple graph executions
- Synchronous ops MUST NOT block on synchronization from another OpKernel
  (may deadlock due to bounded thread pool)
- Blocking ops must use `AsyncOpKernel` instead

### Expensive Kernels

```cpp
virtual bool IsExpensive() { return expensive_; }
```

The runtime uses this to optimize scheduling:
- Inexpensive ops may be "inlined" (executed on the scheduling thread)
- Expensive ops are dispatched to the thread pool

---

## OpKernelConstruction

**Header:** `tensorflow/core/framework/op_kernel.h`

`OpKernelConstruction` provides access to the operation's definition and
construction-time utilities. It is passed to the `OpKernel` constructor.

### Definition

```cpp
class OpKernelConstruction {
 public:
  OpKernelConstruction(DeviceType device_type, DeviceBase* device,
                       Allocator* allocator, FunctionLibraryRuntime* flib,
                       ResourceMgr* resource_mgr,
                       const std::shared_ptr<const NodeProperties>& props,
                       const MemoryTypeSlice& input_memory_types,
                       const MemoryTypeSlice& output_memory_types,
                       int graph_def_version, absl::Status* status);

  // --- Environment ---

  Env* env() const;
  DeviceBase* device() const;
  const DeviceType& device_type() const;
  FunctionLibraryRuntime* function_library() const;
  ResourceMgr* resource_manager() const;
  int graph_def_version() const;

  // --- Node Definition ---

  const NodeDef& def() const;

  // --- Input/Output Types ---

  int num_inputs() const;
  DataType input_type(int i) const;
  const DataTypeSlice& input_types() const;
  const MemoryTypeSlice& input_memory_types() const;

  int num_outputs() const;
  DataType output_type(int i) const;
  const DataTypeSlice& output_types() const;
  const MemoryTypeSlice& output_memory_types() const;

  // --- Attribute Access ---

  // Get a named attribute value.
  template <class T>
  absl::Status GetAttr(absl::string_view attr_name, T* value) const;

  // Check if an attribute exists.
  bool HasAttr(absl::string_view attr_name) const;

  // --- Signature Validation ---

  // Validate that inputs and outputs match expected types.
  absl::Status MatchSignature(const DataTypeSlice expected_inputs,
                              const DataTypeSlice expected_outputs);

  // --- Tensor Allocation ---

  // Allocate a temporary tensor (valid only during construction).
  absl::Status allocate_temp(DataType type, const TensorShape& shape,
                             Tensor* out_temp);
  absl::Status allocate_temp(DataType type, const TensorShape& shape,
                             Tensor* out_temp,
                             AllocatorAttributes allocator_attr);

  // --- Error Handling ---

  void SetStatus(const absl::Status& status);
  const absl::Status& status() const;

  // For OP_REQUIRES macros.
  void CtxFailure(const absl::Status& s);
  void CtxFailureWithWarning(const absl::Status& s);
  void CtxFailure(const char* file, int line, const absl::Status& s);
  void CtxFailureWithWarning(const char* file, int line,
                             const absl::Status& s);
};
```

### Attribute Access Examples

```cpp
class MyOp : public OpKernel {
 public:
  explicit MyOp(OpKernelConstruction* context) : OpKernel(context) {
    // Get a scalar attribute.
    OP_REQUIRES_OK(context, context->GetAttr("stride", &stride_));

    // Get a list attribute.
    OP_REQUIRES_OK(context, context->GetAttr("paddings", &paddings_));

    // Get a type attribute.
    OP_REQUIRES_OK(context, context->GetAttr("T", &dtype_));

    // Validate attribute values.
    OP_REQUIRES(context, stride_ > 0,
                errors::InvalidArgument("Stride must be positive, got ",
                                         stride_));
  }
};
```

---

## OpKernelContext

**Header:** `tensorflow/core/framework/op_kernel.h`

`OpKernelContext` is the primary interface for `Compute()` methods. It provides
access to inputs, output allocation, device information, and execution state.

### Params Structure

The context is initialized from a `Params` struct:

```cpp
struct Params {
  int64_t step_id = 0;                        // Step identifier
  int64_t start_time_usecs = 0;               // Execution start time
  std::optional<absl::Time> deadline;          // Execution deadline

  OpKernel* op_kernel = nullptr;               // The kernel being computed
  DeviceBase* device = nullptr;                // Execution device
  PerOpGpuDevice* eigen_gpu_device = nullptr;  // GPU device wrapper

  bool track_allocations = false;
  bool log_memory = false;

  const AllocatorAttributes* output_attr_array = nullptr;
  ResourceMgr* resource_manager = nullptr;
  ScopedStepContainer* step_container = nullptr;
  RendezvousInterface* rendezvous = nullptr;
  CollectiveExecutor* collective_executor = nullptr;
  const ConfigProto* session_config = nullptr;
  SessionState* session_state = nullptr;
  std::string session_handle;
  const SessionMetadata* session_metadata = nullptr;
  TensorStore* tensor_store = nullptr;
  CancellationManager* cancellation_manager = nullptr;

  absl::Span<const TensorValue> inputs;
  bool is_input_dead = false;
  absl::Span<const AllocatorAttributes> input_alloc_attrs;
  DeviceContext* op_device_context = nullptr;
  FrameAndIter frame_iter;

  CallFrameInterface* call_frame = nullptr;
  FunctionLibraryRuntime* function_library = nullptr;
  std::function<void(std::function<void()>)>* runner = nullptr;
  StepStatsCollectorInterface* stats_collector = nullptr;
  GraphCollector* graph_collector = nullptr;
  bool run_all_kernels_inline = false;
  const std::string* executor_type = nullptr;
  checkpoint::TensorSliceReaderCacheWrapper* slice_reader_cache = nullptr;

  // Input forwarding support.
  static constexpr int kNeverForward = -2;
  static constexpr int kNoReservation = -1;
  const int* forward_from_array = nullptr;

  // Deferred ops support.
  std::function<void()> inc_num_deferred_ops_function;
  std::function<void()> dec_num_deferred_ops_function;

  std::optional<ManagedStackTrace> stack_trace = {};
  bool* outputs_required_array = nullptr;

  tsl::CoordinationServiceAgent* coordination_service_agent = nullptr;
};
```

### Context Construction

```cpp
explicit OpKernelContext(Params* params);
OpKernelContext(Params* params, int num_outputs);
virtual ~OpKernelContext();
```

### Environment Access

```cpp
Env* env() const;
int64_t step_id() const;
int64_t start_time_usecs() const;
std::optional<absl::Time> deadline() const;
const ConfigProto* session_config() const;
const OpKernel& op_kernel() const;
const absl::optional<ManagedStackTrace>& stack_trace() const;
```

### Input Access

```cpp
// Number of inputs.
int num_inputs() const;

// Input data type.
DataType input_dtype(int index) const;
absl::Status input_dtype(absl::string_view name, DataType* dtype) const;

// Input memory type.
MemoryType input_memory_type(int index) const;

// Get immutable input tensor (non-Ref inputs only).
const Tensor& input(int index) const;

// Get immutable input (with Status).
absl::StatusOr<const Tensor*> get_input(int index) const;

// Get named input.
absl::Status input(absl::string_view name, const Tensor** tensor);

// Get named input list.
absl::Status input_list(absl::string_view name, OpInputList* list);

// Check if input exists (for Merge-like ops).
bool has_input(int index) const;

// Mutable input access (for Ref inputs).
mutex* input_ref_mutex(int index);
Tensor mutable_input(int index, bool lock_held);
absl::Status mutable_input(absl::string_view name, Tensor* tensor,
                           bool lock_held);
absl::Status mutable_input_list(absl::string_view name,
                                OpMutableInputList* list);

// Replace a Ref input's tensor.
void replace_ref_input(int index, const Tensor& tensor, bool lock_held);
absl::Status replace_ref_input(absl::string_view name,
                               const Tensor& tensor, bool lock_held);

// Delete a Ref input tensor.
void delete_ref_input(int input_index, bool lock_held);

// Validate all inputs have the same shape.
bool ValidateInputsAreSameShape(OpKernel* op);
```

### Output Access

```cpp
// Number of outputs.
int num_outputs() const;

// Expected output data type.
DataType expected_output_dtype(int index) const;

// Output memory type.
MemoryType output_memory_type(int index) const;

// Check if output is required.
bool output_required(int index) const;

// Check if output expects forwarded input.
bool output_expects_forwarding(int index) const;

// Get named output list.
absl::Status output_list(absl::string_view name, OpOutputList* list);
```

### Output Allocation

```cpp
// Allocate a new output tensor.
// Returns pointer to the allocated tensor (context owns it).
virtual absl::Status allocate_output(int index, const TensorShape& shape,
                                     Tensor** tensor);
virtual absl::Status allocate_output(absl::string_view name,
                                     const TensorShape& shape,
                                     Tensor** tensor);

// With explicit allocator attributes.
virtual absl::Status allocate_output(int index, const TensorShape& shape,
                                     Tensor** tensor,
                                     AllocatorAttributes attr);
virtual absl::Status allocate_output(absl::string_view name,
                                     const TensorShape& shape,
                                     Tensor** tensor,
                                     AllocatorAttributes attr);
```

### Temporary Allocation

```cpp
// Allocate a temporary tensor (scratch space).
virtual absl::Status allocate_temp(DataType type, const TensorShape& shape,
                                   Tensor* out_temp,
                                   AllocatorAttributes allocator_attr,
                                   const AllocationAttributes& allocation_attr);
virtual absl::Status allocate_temp(DataType type, const TensorShape& shape,
                                   Tensor* out_temp);
virtual absl::Status allocate_temp(DataType type, const TensorShape& shape,
                                   Tensor* out_temp,
                                   AllocatorAttributes allocator_attr);
```

### Persistent Allocation

```cpp
// Allocate a persistent tensor (survives across Compute calls).
// Must be stored as a class member (PersistentTensor).
absl::Status allocate_persistent(DataType type,
                                 const TensorShape& shape,
                                 PersistentTensor* persistent_tensor,
                                 Tensor** returned_tensor);
absl::Status allocate_persistent(DataType type,
                                 const TensorShape& shape,
                                 PersistentTensor* persistent_tensor,
                                 Tensor** returned_tensor,
                                 AllocatorAttributes attr);
```

### Output Setting

```cpp
// Set output to an existing tensor.
void set_output(int index, const Tensor& tensor);
void set_output(int index, Tensor&& tensor);

// Set output to a reference tensor.
void set_output_ref(int index, mutex* mu, Tensor* tensor_for_ref);
```

### Input Forwarding

Input forwarding allows an output tensor to reuse an input tensor's buffer,
avoiding unnecessary copies:

```cpp
// Forward a Ref input to a Ref output.
void forward_ref_input_to_ref_output(int input_index, int output_index);

// Forward an input to an output with shape check.
bool forward_input_to_output_with_shape(int input_index, int output_index,
                                        const TensorShape& output_shape,
                                        Tensor** output);
absl::Status forward_input_to_output_with_shape(
    absl::string_view input_name, absl::string_view output_name,
    const TensorShape& output_shape, Tensor** output);

// Get a forwarded input tensor (returns nullptr if forwarding not possible).
std::unique_ptr<Tensor> forward_input(
    int input_index, int output_index, DataType output_dtype,
    const TensorShape& output_shape, MemoryType output_memory_type,
    const AllocatorAttributes& output_attr);

// Try forwarding, fall back to allocation.
absl::Status forward_input_or_allocate_output(
    absl::Span<const int> candidate_input_indices, int output_index,
    const TensorShape& output_shape, Tensor** output,
    int* forwarded_input = nullptr);
absl::Status forward_input_or_allocate_output(
    absl::Span<const absl::string_view> candidate_input_names,
    absl::string_view output_name, const TensorShape& output_shape,
    Tensor** output);

// Try forwarding for temp allocation.
absl::Status forward_input_or_allocate_temp(
    absl::Span<const int> candidate_input_indices, DataType type,
    const TensorShape& shape, const AllocatorAttributes& allocator_attr,
    Tensor* out_temp);
absl::Status forward_input_or_allocate_temp(
    absl::Span<const int> candidate_input_indices, DataType type,
    const TensorShape& shape, Tensor* out_temp);
```

### Status and Error Handling

```cpp
// Set the computation status (used for error reporting).
void SetStatus(const absl::Status& status);
absl::Status status();

// Cancellation support.
CancellationManager* cancellation_manager();

// Device access.
DeviceBase* device() const;

// Eigen device (for Eigen tensor operations).
template <typename EigenDeviceType>
const EigenDeviceType& eigen_device() const;

// Frame and iteration info (for control flow).
FrameAndIter frame_iter() const;

// Graph collector (for function-calling ops).
GraphCollector* graph_collector();

// Function library.
FunctionLibraryRuntime* function_library() const;

// Resource manager.
ResourceMgr* resource_manager() const;

// Step container.
ScopedStepContainer* step_container() const;

// Runner for scheduling work.
std::function<void(std::function<void()>)>* runner() const;

// Whether to run all kernels inline.
bool run_all_kernels_inline() const;

// Executor type.
const std::string& executor_type() const;

// Device context.
DeviceContext* op_device_context() const;

// Input allocation attributes.
const AllocatorAttributes& input_alloc_attr(int index) const;

// Output allocation attributes.
const AllocatorAttributes& output_alloc_attr(int index) const;
```

### Utility Classes

#### OpInputList

```cpp
class OpInputList {
 public:
  const Tensor& operator[](int i) const;
  int size() const;
  Iterator begin() const;
  Iterator end() const;
};
```

#### OpMutableInputList

```cpp
class OpMutableInputList {
 public:
  Tensor at(int i, bool lock_held);
  mutex* ref_mutex(int i);
  int size() const;
  Iterator begin() const;
  Iterator end() const;
};
```

#### OpOutputList

```cpp
class OpOutputList {
 public:
  Tensor* operator[](int i);
  bool required(int i) const;
  DataType expected_output_dtype(int i) const;
  absl::Status allocate(int i, const TensorShape& shape, Tensor** output);
  void set(int i, const Tensor& tensor);
  void set(int i, Tensor&& tensor);
  void set_ref(int i, mutex* mu, Tensor* tensor_for_ref);
  int size() const;
};
```

#### TensorValue

```cpp
struct TensorValue {
  TensorValue();
  explicit TensorValue(Tensor* t);
  TensorValue(mutex* mu, Tensor* t);

  Tensor* operator->() const;
  bool is_ref() const;
  DataType dtype() const;
  DataType dtype_safe() const;  // Acquires lock for refs

  mutex* mutex_if_ref;  // nullptr if not a ref
  Tensor* tensor;
};
```

---

## AsyncOpKernel

**Header:** `tensorflow/core/framework/op_kernel.h`

`AsyncOpKernel` is the base class for operations that must block on the
execution of another op (e.g., `RecvOp`, `DequeueOp`).

### Definition

```cpp
class AsyncOpKernel : public OpKernel {
 public:
  using OpKernel::OpKernel;  // Inherit constructors

  // Asynchronous compute.
  // Implementations MUST call done exactly once when computation completes.
  // ComputeAsync MUST NOT block on another OpKernel's execution.
  // context is guaranteed alive until done is called.
  //
  // WARNING: As soon as done starts, context and this may be deleted.
  // No code depending on these should execute after calling done.
  typedef std::function<void()> DoneCallback;
  virtual void ComputeAsync(OpKernelContext* context, DoneCallback done) = 0;

  AsyncOpKernel* AsAsync() override { return this; }
  void Compute(OpKernelContext* context) override;
};
```

### Example: Asynchronous Kernel

```cpp
class AsyncRecvOp : public AsyncOpKernel {
 public:
  explicit AsyncRecvOp(OpKernelConstruction* context)
      : AsyncOpKernel(context) {}

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    // Receive tensor from another device.
    auto rendezvous = context->rendezvous();
    Rendezvous::ParsedKey key;
    OP_REQUIRES_OK_ASYNC(context, ParseKey(context, &key), done);

    rendezvous->RecvAsync(key, Rendezvous::Args(),
        [context, done](const Status& status,
                        const Rendezvous::Args& send_args,
                        const Rendezvous::Args& recv_args,
                        const Tensor& tensor, bool is_dead) {
          if (!status.ok()) {
            context->SetStatus(status);
          } else {
            context->set_output(0, tensor);
          }
          done();
        });
  }
};
```

### Cancellation Support

Async ops should implement cancellation to handle cases where the unblocking
kernel never runs:

```cpp
void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
  auto* cm = context->cancellation_manager();
  CancellationToken token = cm->get_cancellation_token();
  bool already_cancelled = !cm->RegisterCallback(token, [this, context, done]() {
    // Handle cancellation.
    context->SetStatus(absl::CancelledError("Operation cancelled"));
    done();
  });

  if (already_cancelled) {
    context->SetStatus(absl::CancelledError("Operation cancelled"));
    done();
    return;
  }

  // ... normal async operation ...
  // When complete, call cm->DeregisterCallback(token) then done().
}
```

---

## TensorBuffer

**Header:** `tensorflow/core/framework/tensor.h`

`TensorBuffer` is the reference-counted buffer backing a `Tensor`. See the
[Cpp Framework Reference](16-cpp-framework.md) for full details.

Key points for kernel authors:
- `data()` returns raw pointer (non-virtual, fast)
- `size()` returns buffer size in bytes
- Inherits from `core::RefCounted` for automatic lifetime management
- `root_buffer()` returns the ultimate parent buffer for sub-buffers

---

## PersistentTensor

**Header:** `tensorflow/core/framework/tensor.h`

`PersistentTensor` wraps a `Tensor` that must survive across `Compute()` calls.
It is used for tensors that are allocated once and reused across invocations.

### Definition

```cpp
class PersistentTensor {
 public:
  PersistentTensor();
  explicit PersistentTensor(Tensor tensor);

  // Access the tensor.
  const Tensor& AccessTensor(OpKernelContext* context) const;
  Tensor* AccessTensor(OpKernelContext* context);

  // Check if empty.
  bool IsInitialized() const;

 private:
  Tensor tensor_;
};
```

### Usage

```cpp
class CacheOp : public OpKernel {
 public:
  explicit CacheOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    if (!cache_.IsInitialized()) {
      // Allocate on first call.
      Tensor* cache_tensor = nullptr;
      OP_REQUIRES_OK(context,
        context->allocate_persistent(DT_FLOAT, TensorShape({1000}),
                                     &cache_, &cache_tensor));
      // Initialize cache_tensor...
    }

    // Access cached tensor on subsequent calls.
    const Tensor& cache = cache_.AccessTensor(context);
    // Use cache...
  }

 private:
  PersistentTensor cache_;
};
```

---

## Registration Macros

**Header:** `tensorflow/core/framework/op_kernel.h`

### REGISTER_KERNEL_BUILDER

```cpp
#define REGISTER_KERNEL_BUILDER(kernel_builder, ...) \
  REGISTER_KERNEL_BUILDER_UNIQ_HELPER(__COUNTER__, kernel_builder, __VA_ARGS__)
```

### KernelBuilder

```cpp
class KernelDefBuilder {
 public:
  // Start with device type.
  explicit KernelDefBuilder(const char* op_name);

  // Specify device type.
  KernelDefBuilder& Device(DeviceType device_type);

  // Type constraint: attr must match one of the given types.
  KernelDefBuilder& TypeConstraint(absl::string_view attr_name,
                                   DataTypeVector allowed_types);

  // Label for kernel selection.
  KernelDefBuilder& Label(absl::string_view label);

  // Host memory for specific inputs/outputs.
  KernelDefBuilder& HostMemory(absl::string_view arg_name);

  // Priority (higher = preferred).
  KernelDefBuilder& Priority(int32_t priority);

  // Build the KernelDef.
  const KernelDef* Build();
};
```

### Registration Examples

```cpp
// Register a CPU kernel for MatMul.
REGISTER_KERNEL_BUILDER(
    Name("MatMul").Device(DEVICE_CPU),
    MatMulOp<CPUDevice, float>);

// Register a GPU kernel with type constraint.
REGISTER_KERNEL_BUILDER(
    Name("MatMul")
        .Device(DEVICE_GPU)
        .TypeConstraint("T", {DT_FLOAT, DT_DOUBLE}),
    MatMulOp<GPUDevice, T>);

// Register a kernel with host memory constraint.
REGISTER_KERNEL_BUILDER(
    Name("Shape")
        .Device(DEVICE_GPU)
        .HostMemory("output"),
    ShapeOp);

// Register with priority.
REGISTER_KERNEL_BUILDER(
    Name("Add")
        .Device(DEVICE_CPU)
        .Priority(2),
    FastAddOp);
```

### Registration Macro Details

```cpp
// Full form.
REGISTER_KERNEL_BUILDER(
    Name("OpName")           // Op name (must match REGISTER_OP name)
        .Device(DEVICE_CPU)  // Target device
        .TypeConstraint("T", {DT_FLOAT, DT_DOUBLE})  // Type constraints
        .HostMemory("shape") // Force host memory for specific inputs/outputs
        .Label("custom")     // Custom label for kernel selection
        .Priority(1),        // Higher priority preferred over lower
    MyOpKernel);             // Kernel class name

// Special macro for unbundled kernels.
REGISTER_KERNEL_BUILDER_UNBUNDLE(Name("Op").Device(DEVICE_CPU),
                                 MyKernelClass, InitOp);
```

---

## Device-Specific Kernels

### CPU Kernels

```cpp
// CPU device type.
template <typename Device>
class MyOp : public OpKernel {
  void Compute(OpKernelContext* context) override {
    // Use Eigen::ThreadPoolDevice for computation.
    const Eigen::ThreadPoolDevice& d =
        context->eigen_device<Eigen::ThreadPoolDevice>();

    // Eigen tensor operations.
    auto input = context->input(0).flat<float>();
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
      context->allocate_output(0, context->input(0).shape(), &output));
    auto output_flat = output->flat<float>();

    // Eigen expression.
    output_flat.device(d) = input * 2.0f;
  }
};

REGISTER_KERNEL_BUILDER(Name("MyOp").Device(DEVICE_CPU), MyOp<CPUDevice>);
```

### GPU Kernels

```cpp
// GPU device type.
#if GOOGLE_CUDA
template <>
class MyOp<GPUDevice> : public OpKernel {
  void Compute(OpKernelContext* context) override {
    // Use Eigen::GpuDevice for computation.
    const Eigen::GpuDevice& d =
        context->eigen_device<Eigen::GpuDevice>();

    // GPU computation with Eigen.
    auto input = context->input(0).flat<float>();
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
      context->allocate_output(0, context->input(0).shape(), &output));
    auto output_flat = output->flat<float>();

    output_flat.device(d) = input * 2.0f;
  }
};

REGISTER_KERNEL_BUILDER(Name("MyOp").Device(DEVICE_GPU), MyOp<GPUDevice>);
#endif  // GOOGLE_CUDA
```

### TPU Kernels

TPU kernels follow a similar pattern but use TPU-specific APIs for
computation and data transfer.

---

## Input/Output Management

### Getting Inputs

```cpp
void Compute(OpKernelContext* context) override {
  // Single input.
  const Tensor& input = context->input(0);

  // Named input.
  const Tensor* named_input;
  OP_REQUIRES_OK(context,
    context->input("weights", &named_input));

  // Input list (for ops with N * T input specification).
  OpInputList inputs;
  OP_REQUIRES_OK(context, context->input_list("inputs", &inputs));
  for (const Tensor& t : inputs) {
    // Process each input tensor.
  }

  // Mutable input (for Ref inputs).
  Tensor mutable_input = context->mutable_input(0, /*lock_held=*/false);
}
```

### Allocating Outputs

```cpp
void Compute(OpKernelContext* context) override {
  // Allocate output with same shape as input.
  Tensor* output = nullptr;
  OP_REQUIRES_OK(context,
    context->allocate_output(0, context->input(0).shape(), &output));

  // Allocate output with custom shape.
  OP_REQUIRES_OK(context,
    context->allocate_output(0, TensorShape({batch, height, width, channels}),
                             &output));

  // Allocate output list.
  OpOutputList outputs;
  OP_REQUIRES_OK(context, context->output_list("outputs", &outputs));
  for (int i = 0; i < outputs.size(); ++i) {
    Tensor* out;
    OP_REQUIRES_OK(context, outputs.allocate(i, TensorShape({}), &out));
  }
}
```

### Input Forwarding (Avoiding Copies)

```cpp
void Compute(OpKernelContext* context) override {
  // Try to forward input 0 to output 0.
  Tensor* output = nullptr;
  if (context->forward_input_to_output_with_shape(0, 0,
       context->input(0).shape(), &output)) {
    // Input was forwarded; output shares input's buffer.
    // Modify output in-place.
    output->flat<float>() *= 2.0f;
  } else {
    // Forwarding not possible; allocate new output.
    OP_REQUIRES_OK(context,
      context->allocate_output(0, context->input(0).shape(), &output));
    output->flat<float>() = context->input(0).flat<float>() * 2.0f;
  }

  // Alternative: use forward_input_or_allocate_output.
  int forwarded_from = -1;
  OP_REQUIRES_OK(context,
    context->forward_input_or_allocate_output(
        /*candidate_input_indices=*/{0, 1},
        /*output_index=*/0,
        context->input(0).shape(),
        &output,
        &forwarded_from));
}
```

---

## Resource Kernels

Resource handles allow sharing state across different ops and graph executions.

### ResourceOpKernel

```cpp
template <typename Resource>
class ResourceOpKernel : public OpKernel {
 public:
  explicit ResourceOpKernel(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Create or look up the resource.
    Resource* resource;
    OP_REQUIRES_OK(context, CreateResource(context, &resource));

    // Use the resource.
    ComputeWithResource(context, resource);
  }

 protected:
  virtual absl::Status CreateResource(OpKernelContext* context,
                                      Resource** resource) = 0;
  virtual void ComputeWithResource(OpKernelContext* context,
                                   Resource* resource) = 0;
};
```

### Resource Handle Usage

```cpp
// Create a resource handle.
class CreateVarOp : public OpKernel {
  void Compute(OpKernelContext* context) override {
    // Allocate a variable tensor.
    Tensor tensor;
    OP_REQUIRES_OK(context,
      context->allocate_temp(DT_FLOAT, TensorShape({rows, cols}), &tensor));

    // Create the resource.
    auto* var = new Var(tensor);

    // Create a handle.
    ResourceHandle handle;
    handle.set_device(context->device()->name());
    handle.set_container("variables");
    handle.set_name("my_var");
    handle.set_hash_code(TypeIndex::Make<Var>().hash_code());

    // Store the resource.
    OP_REQUIRES_OK(context,
      context->resource_manager()->Create(handle.container(),
                                          handle.name(), var));

    // Output the handle.
    Tensor* handle_tensor = nullptr;
    OP_REQUIRES_OK(context,
      context->allocate_output(0, TensorShape({}), &handle_tensor));
    handle_tensor->scalar<ResourceHandle>()() = handle;
  }
};

// Use a resource handle.
class ReadVarOp : public OpKernel {
  void Compute(OpKernelContext* context) override {
    const ResourceHandle& handle =
        context->input(0).scalar<ResourceHandle>()();

    Var* var;
    OP_REQUIRES_OK(context,
      context->resource_manager()->Lookup(handle.container(),
                                           handle.name(), &var));

    // Read the variable's tensor.
    context->set_output(0, var->tensor());

    var->Unref();
  }
};
```

---

## Gradient Kernels

TensorFlow provides gradient computation through registered gradient functions.

### Gradient Registration

```cpp
// Register a gradient function for an op.
REGISTER_OP_GRADIENT("MatMul", MatMulGrad);
```

### Gradient Function Signature

```cpp
typedef std::function<absl::Status(const OpGradContext&,
                                   std::vector<Output>* grad_outputs)>
    OpGradFunc;
```

### Example Gradient Implementation

```cpp
absl::Status MatMulGrad(const OpGradContext& context,
                        std::vector<Output>* grad_outputs) {
  // Get the forward op's inputs and attributes.
  auto grad = context.grad(0);
  auto a = context.input(0);
  auto b = context.input(1);
  bool transpose_a, transpose_b;
  TF_RETURN_IF_ERROR(context.GetAttr("transpose_a", &transpose_a));
  TF_RETURN_IF_ERROR(context.GetAttr("transpose_b", &transpose_b));

  // Compute gradients.
  if (!transpose_a && !transpose_b) {
    grad_outputs->push_back(MatMul(scope, grad, b, /*transpose_b=*/true));
    grad_outputs->push_back(MatMul(scope, a, grad, /*transpose_a=*/true));
  }
  // Handle transposed cases...

  return absl::OkStatus();
}
```

### Math Gradient Utilities

```cpp
// Common gradient helper functions.
namespace math_grad {
absl::Status AddGrad(const Scope& scope, const OpGradContext& context,
                     std::vector<Output>* grad_outputs);
absl::Status MatMulGrad(const Scope& scope, const OpGradContext& context,
                        std::vector<Output>* grad_outputs);
// ... many more
}
```

---

## Common Kernel Utilities

### Eigen Tensor Operations

Most kernels use Eigen for computation:

```cpp
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/op_kernel.h"

void Compute(OpKernelContext* context) override {
  const Tensor& input = context->input(0);
  Tensor* output = nullptr;
  OP_REQUIRES_OK(context,
    context->allocate_output(0, input.shape(), &output));

  // Get Eigen device.
  const auto& d = context->eigen_device<Device>();

  // Apply operations.
  output->flat<float>().device(d) = input.flat<float>() * 2.0f;

  // Binary operations.
  output->flat<float>().device(d) =
      context->input(0).flat<float>() + context->input(1).flat<float>();

  // Reduction.
  output->flat<float>().device(d) =
      input.flat<float>().sum(Eigen::array<int, 1>({1}));
}
```

### Functor Pattern

TensorFlow uses a functor pattern to separate device-specific implementations:

```cpp
// In header (e.g., relu_op.h):
namespace functor {
template <typename Device, typename T>
struct Relu {
  void operator()(const Device& d, typename TTypes<T>::ConstFlat input,
                  typename TTypes<T>::Flat output);
};
}  // namespace functor

// CPU implementation (e.g., relu_op.cc):
namespace functor {
template <typename T>
struct Relu<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::ConstFlat input,
                  typename TTypes<T>::Flat output) {
    output.device(d) = input.cwiseMax(static_cast<T>(0));
  }
};
}  // namespace functor

// GPU implementation (e.g., relu_op_gpu.cu.cc):
namespace functor {
template <typename T>
struct Relu<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::ConstFlat input,
                  typename TTypes<T>::Flat output) {
    // GPU-specific implementation (e.g., CUDA kernel).
    ReluKernel(d, input, output);
  }
};
}  // namespace functor

// Kernel using the functor:
template <typename Device, typename T>
class ReluOp : public OpKernel {
  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
      context->allocate_output(0, input.shape(), &output));

    functor::Relu<Device, T>()(
        context->eigen_device<Device>(),
        input.flat<T>(),
        output->flat<T>());
  }
};
```

### Validation Helpers

```cpp
// Validate input shape.
OP_REQUIRES(context, input.dims() == 2,
            errors::InvalidArgument("Input must be 2-D, got shape: ",
                                     input.shape().DebugString()));

// Validate input type.
OP_REQUIRES(context, input.dtype() == DT_FLOAT,
            errors::InvalidArgument("Input must be float, got: ",
                                     DataTypeString(input.dtype())));

// Validate dimensions.
OP_REQUIRES(context, input.dim_size(1) == weights.dim_size(0),
            errors::InvalidArgument("Input columns must match weight rows: ",
                                     input.dim_size(1), " vs ",
                                     weights.dim_size(0)));

// Validate positive.
OP_REQUIRES(context, stride > 0,
            errors::InvalidArgument("Stride must be positive, got: ", stride));

// Validate range.
OP_REQUIRES(context, 0 <= axis && axis < input.dims(),
            errors::InvalidArgument("Axis out of range: ", axis));
```

### OP_REQUIRES Macros

```cpp
// Check a condition.
OP_REQUIRES(context, condition, error_status);

// Check status is OK.
OP_REQUIRES_OK(context, status);

// Async variants (for AsyncOpKernel).
OP_REQUIRES_ASYNC(context, condition, error_status, done_callback);
OP_REQUIRES_OK_ASYNC(context, status, done_callback);
```

---

## Kernel Implementation Patterns

### Pattern 1: Element-wise Unary Op

```cpp
template <typename Device, typename T>
class SquareOp : public OpKernel {
 public:
  explicit SquareOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
      context->allocate_output(0, input.shape(), &output));

    const auto& d = context->eigen_device<Device>();
    output->flat<T>().device(d) = input.flat<T>().square();
  }
};
```

### Pattern 2: Element-wise Binary Op

```cpp
template <typename Device, typename T>
class AddOp : public OpKernel {
 public:
  explicit AddOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& a = context->input(0);
    const Tensor& b = context->input(1);

    // Broadcasting: output shape is the broadcast of a and b shapes.
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
      context->allocate_output(0, a.shape(), &output));

    const auto& d = context->eigen_device<Device>();

    // Simple case: same shapes.
    if (a.shape() == b.shape()) {
      output->flat<T>().device(d) = a.flat<T>() + b.flat<T>();
    } else {
      // Need broadcasting (use Eigen broadcast or custom implementation).
      // ... broadcasting logic ...
    }
  }
};
```

### Pattern 3: Reduction Op

```cpp
template <typename Device, typename T>
class SumOp : public OpKernel {
 public:
  explicit SumOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& axis = context->input(1);

    // Compute output shape.
    TensorShape output_shape;
    for (int i = 0; i < input.dims(); ++i) {
      if (i != axis.scalar<int32>()()) {
        output_shape.AddDim(input.dim_size(i));
      }
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
      context->allocate_output(0, output_shape, &output));

    // Perform reduction.
    const auto& d = context->eigen_device<Device>();
    Eigen::array<int, 1> reduction_axis = {axis.scalar<int32>()()};
    output->flat<T>().device(d) = input.flat_inner_dims<T, 2>().sum(reduction_axis);
  }
};
```

### Pattern 4: Op with Temporaries

```cpp
class ComplexOp : public OpKernel {
 public:
  explicit ComplexOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);

    // Allocate temporary workspace.
    Tensor temp;
    OP_REQUIRES_OK(context,
      context->allocate_temp(DT_FLOAT, input.shape(), &temp));

    // Use temporary.
    const auto& d = context->eigen_device<CPUDevice>();
    temp.flat<float>().device(d) = input.flat<float>().sqrt();

    // Allocate output.
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
      context->allocate_output(0, input.shape(), &output));

    output->flat<float>().device(d) = temp.flat<float>() * 2.0f;
  }
};
```

### Pattern 5: Op with Persistent State

```cpp
class LookupTableOp : public OpKernel {
 public:
  explicit LookupTableOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    if (!table_.IsInitialized()) {
      Tensor* table = nullptr;
      OP_REQUIRES_OK(context,
        context->allocate_persistent(DT_FLOAT, TensorShape({10000, 128}),
                                     &table_, &table));
      // Initialize table values.
      auto table_flat = table->flat<float>();
      for (int i = 0; i < table_flat.size(); ++i) {
        table_flat(i) = static_cast<float>(i);
      }
    }

    const Tensor& indices = context->input(0);
    const Tensor& table = table_.AccessTensor(context);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
      context->allocate_output(0, TensorShape({indices.NumElements(), 128}),
                               &output));

    // Gather from table.
    auto table_matrix = table.matrix<float>();
    auto output_matrix = output->matrix<float>();
    auto indices_flat = indices.flat<int32>();
    for (int i = 0; i < indices_flat.size(); ++i) {
      int idx = indices_flat(i);
      OP_REQUIRES(context, idx >= 0 && idx < 10000,
                  errors::InvalidArgument("Index out of range: ", idx));
      for (int j = 0; j < 128; ++j) {
        output_matrix(i, j) = table_matrix(idx, j);
      }
    }
  }

 private:
  PersistentTensor table_;
};
```

### Pattern 6: Op with Multiple Outputs

```cpp
class SplitOp : public OpKernel {
 public:
  explicit SplitOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("num_split", &num_split_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const int64_t split_dim = input.dim_size(0) / num_split_;

    for (int i = 0; i < num_split_; ++i) {
      Tensor* output = nullptr;
      OP_REQUIRES_OK(context,
        context->allocate_output(i, input.shape(), &output));

      // Copy slice.
      auto slice = output->flat<float>();
      auto input_flat = input.flat<float>();
      int64_t offset = i * split_dim * (input.NumElements() / input.dim_size(0));
      int64_t size = split_dim * (input.NumElements() / input.dim_size(0));
      std::copy_n(input_flat.data() + offset, size, slice.data());
    }
  }

 private:
  int num_split_;
};
```

### Pattern 7: Op with Type Polymorphism

```cpp
template <typename Device, typename T>
class CastOp : public OpKernel {
 public:
  explicit CastOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
      context->allocate_output(0, input.shape(), &output));

    const auto& d = context->eigen_device<Device>();
    output->flat<T>().device(d) = input.flat<SrcT>().template cast<T>();
  }
};

// Register for multiple type pairs.
#define REGISTER_CAST(DstT)                                       \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("Cast")                                                \
          .Device(DEVICE_CPU)                                     \
          .TypeConstraint("DstT", DataTypeToEnum<DstT>::value), \
      CastOp<CPUDevice, DstT>);

REGISTER_CAST(float);
REGISTER_CAST(double);
REGISTER_CAST(int32);
REGISTER_CAST(int64_t);
#undef REGISTER_CAST
```

---

## Error Handling Macros Reference

| Macro                       | Usage                                  |
|----------------------------|----------------------------------------|
| `OP_REQUIRES(ctx, cond, err)` | Check condition, set status on failure |
| `OP_REQUIRES_OK(ctx, status)` | Check status is OK                      |
| `OP_REQUIRES_ASYNC(ctx, cond, err, done)` | Async variant, calls done on failure |
| `OP_REQUIRES_OK_ASYNC(ctx, status, done)` | Async variant, calls done on failure |
| `TF_RETURN_IF_ERROR(status)` | Return status if not OK (non-kernel code) |
| `TF_CHECK_OK(status)`       | Crash if status is not OK              |

---

## Performance Best Practices

1. **Use Input Forwarding**: Avoid unnecessary copies by forwarding input
   buffers to outputs when possible.
2. **Minimize Allocations**: Use `allocate_temp` for scratch space and reuse
   buffers.
3. **Use Eigen Expressions**: Leverage Eigen's lazy evaluation to avoid
   intermediate allocations.
4. **Batch Operations**: Process multiple elements at once rather than one at
   a time.
5. **Device Specialization**: Use the functor pattern for device-specific
   optimizations.
6. **Avoid Blocking**: Use `AsyncOpKernel` for operations that must wait on
   external events.
7. **Check `output_required`**: Skip computation for outputs that are not needed.
8. **Use Persistent Tensors**: For buffers that persist across `Compute()` calls.
9. **Align with Eigen**: Ensure tensors are properly aligned for Eigen operations.

---

## Memory Type Considerations

```cpp
// Host memory: Input tensor is always in CPU memory.
// Device memory: Input tensor is in device memory (GPU for GPU devices).

// Check memory type.
MemoryType mem_type = context->input_memory_type(0);

// Force host memory for specific inputs/outputs via kernel registration.
REGISTER_KERNEL_BUILDER(
    Name("ShapeN")
        .Device(DEVICE_GPU)
        .HostMemory("input")
        .HostMemory("output"),
    ShapeNOp);
```

When `HostMemory` is specified, the framework ensures the tensor is in host
(CPU) memory before passing it to the kernel. This is useful for metadata
operations (Shape, Size, Rank) that need to inspect tensor shapes on the CPU.
