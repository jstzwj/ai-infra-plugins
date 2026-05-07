# TensorFlow Session and Executor Architecture

This reference covers TensorFlow's session-based execution model, including the
Session interface, DirectSession for local execution, distributed session
implementations, the Executor system, graph partitioning, the Rendezvous
mechanism for cross-device communication, and configuration options.

---

## Table of Contents

1. [Session Interface](#session-interface)
2. [SessionOptions](#sessionoptions)
3. [DirectSession](#directsession)
4. [GrpcSession](#grpcsession)
5. [Executor](#executor)
6. [Graph Partitioning for Execution](#graph-partitioning-for-execution)
7. [Rendezvous](#rendezvous)
8. [CostModel](#costmodel)
9. [StepStats](#stepstats)
10. [ThreadPool Configuration](#threadpool-configuration)
11. [RunOptions and RunMetadata](#runoptions-and-runmetadata)
12. [ConfigProto](#configproto)

---

## Session Interface

**Header:** `tensorflow/core/public/session.h`
**Namespace:** `tensorflow`

The `Session` class is the primary interface for executing TensorFlow graphs. A
session encapsulates the environment in which operations are executed, managing
resources, devices, and execution state.

### Core Lifecycle

```cpp
class Session {
 public:
  Session();
  virtual ~Session();

  // Create the graph to be used for the session.
  virtual absl::Status Create(const GraphDef& graph) = 0;
  virtual absl::Status Create(GraphDef&& graph);

  // Add operations to the existing graph.
  virtual absl::Status Extend(const GraphDef& graph) = 0;
  virtual absl::Status Extend(GraphDef&& graph);

  // Execute the graph.
  virtual absl::Status Run(
      const std::vector<std::pair<std::string, Tensor>>& inputs,
      const std::vector<std::string>& output_tensor_names,
      const std::vector<std::string>& target_tensor_names,
      std::vector<Tensor>* outputs) = 0;

  // Close the session and release resources.
  virtual absl::Status Close() = 0;

  // List available devices.
  virtual absl::Status ListDevices(
      std::vector<DeviceAttributes>* response) = 0;
};
```

### Extended Run Methods

```cpp
// Run with options and metadata.
virtual absl::Status Run(
    const RunOptions& run_options,
    const std::vector<std::pair<std::string, Tensor>>& inputs,
    const std::vector<std::string>& output_tensor_names,
    const std::vector<std::string>& target_tensor_names,
    std::vector<Tensor>* outputs,
    RunMetadata* run_metadata);

// Run with custom thread pool.
virtual absl::Status Run(
    const RunOptions& run_options,
    const std::vector<std::pair<std::string, Tensor>>& inputs,
    const std::vector<std::string>& output_tensor_names,
    const std::vector<std::string>& target_tensor_names,
    std::vector<Tensor>* outputs,
    RunMetadata* run_metadata,
    const thread::ThreadPoolOptions& threadpool_options);

// Run with RunOptions (Create/Extend variants).
virtual absl::Status Create(const RunOptions& run_options,
                            const GraphDef& graph);
virtual absl::Status Extend(const RunOptions& run_options,
                            const GraphDef& graph);
virtual absl::Status Close(const RunOptions& run_options);
```

### Partial Run API

```cpp
// Set up partial execution. Returns a handle for subsequent PRun calls.
virtual absl::Status PRunSetup(
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::vector<std::string>& target_nodes,
    std::string* handle);

// Continue partial execution with the given handle.
virtual absl::Status PRun(
    const std::string& handle,
    const std::vector<std::pair<std::string, Tensor>>& inputs,
    const std::vector<std::string>& output_names,
    std::vector<Tensor>* outputs);
```

### Callable API

```cpp
typedef int64_t CallableHandle;

// Create a callable subgraph handle for efficient repeated execution.
virtual absl::Status MakeCallable(
    const CallableOptions& callable_options,
    CallableHandle* out_handle);

// Execute a callable subgraph.
virtual absl::Status RunCallable(
    CallableHandle handle,
    const std::vector<Tensor>& feed_tensors,
    std::vector<Tensor>* fetch_tensors,
    RunMetadata* run_metadata);

// Execute with custom thread pool.
virtual absl::Status RunCallable(
    CallableHandle handle,
    const std::vector<Tensor>& feed_tensors,
    std::vector<Tensor>* fetch_tensors,
    RunMetadata* run_metadata,
    const thread::ThreadPoolOptions& threadpool_options);

// Release callable resources.
virtual absl::Status ReleaseCallable(CallableHandle handle);

// Finalize the session (release graph-related state).
virtual absl::Status Finalize();
```

### Device Management

```cpp
// Get the local device manager.
virtual absl::Status LocalDeviceManager(const DeviceMgr** output);
```

### Session Factory Functions

```cpp
// Create a new session with status reporting.
absl::Status NewSession(const SessionOptions& options,
                        Session** out_session);

// Create a new session (returns nullptr on failure).
Session* NewSession(const SessionOptions& options);

// Reset resource containers associated with a target.
absl::Status Reset(const SessionOptions& options,
                   const std::vector<std::string>& containers);
```

### Usage Example

```cpp
// Create and use a session.
tensorflow::GraphDef graph;
// ... build or load graph ...

tensorflow::SessionOptions options;
std::unique_ptr<tensorflow::Session> session(
    tensorflow::NewSession(options));

// Create the session with the graph.
TF_CHECK_OK(session->Create(graph));

// Run the graph.
std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
    {"input", input_tensor}
};
std::vector<tensorflow::Tensor> outputs;
TF_CHECK_OK(session->Run(inputs, {"output:0"}, {}, &outputs));

// Use the output.
auto result = outputs[0].flat<float>();

// Close the session.
TF_CHECK_OK(session->Close());
```

---

## SessionOptions

**Header:** `tensorflow/core/public/session_options.h`

```cpp
struct SessionOptions {
  // The target to connect to. Empty for local execution.
  // Example: "grpc://localhost:2222" for remote execution.
  std::string target;

  // Configuration options.
  ConfigProto config;

  // Environment (usually Env::Default()).
  Env* env = Env::Default();
};
```

### Target Specification

| Target                | Description                              |
|-----------------------|------------------------------------------|
| `""` (empty)          | Local execution via DirectSession        |
| `"grpc://host:port"`  | Remote execution via GrpcSession         |
| Custom                | Other session implementations            |

---

## DirectSession

**Header:** `tensorflow/core/common_runtime/direct_session.h`

`DirectSession` is the primary session implementation for local execution. It
runs the graph on available local devices (CPU, GPU, TPU).

### Construction

```cpp
DirectSession(const SessionOptions& options,
              const DeviceMgr* device_mgr,
              DirectSessionFactory* factory);
```

### Key Architecture

```
DirectSession
  +-- DeviceMgr (manages available devices)
  +-- GraphExecutionState (manages graph and optimization)
  +-- ExecutorsAndKeys (per feed/fetch pattern)
  |     +-- PerPartitionExecutorsAndLib[]
  |           +-- Graph (partition subgraph)
  |           +-- Device (target device)
  |           +-- Executor (graph executor)
  |           +-- FunctionLibraryRuntime
  +-- CostModelManager
  +-- FunctionInfo[]
```

### Run Flow

When `DirectSession::Run()` is called:

1. **Input Processing**: Convert input name-value pairs to feed tensors
2. **Graph Lookup**: Find or create `ExecutorsAndKeys` for the given
   feed/fetch pattern
3. **Graph Partitioning**: Split the graph by device assignment
4. **Executor Creation**: Create an executor for each partition
5. **Input Feeding**: Send feed tensors to the rendezvous
6. **Execution**: Run all executors concurrently
7. **Output Retrieval**: Fetch output tensors from the rendezvous
8. **Return**: Return output tensors to the caller

### ExecutorsAndKeys

```cpp
struct ExecutorsAndKeys {
  std::atomic_int_fast64_t step_count;

  // The full graph (only kept for partial runs).
  std::unique_ptr<Graph> graph;
  NameNodeMap name_to_node;

  // Per-device executor bundles.
  std::vector<PerPartitionExecutorsAndLib> items;

  // Input/output mapping.
  std::unordered_map<std::string, size_t> input_name_to_index;
  std::unordered_map<std::string, std::string> input_name_to_rendezvous_key;
  std::unordered_map<std::string, size_t> output_name_to_index;
  std::unordered_map<std::string, std::string> output_name_to_rendezvous_key;

  DataTypeVector input_types;
  DataTypeVector output_types;

  CallableOptions callable_options;
  int64_t collective_graph_key;
};
```

### PerPartitionExecutorsAndLib

```cpp
struct PerPartitionExecutorsAndLib {
  std::unique_ptr<Graph> graph = nullptr;
  Device* device = nullptr;                // Not owned.
  FunctionLibraryRuntime* flib = nullptr;  // Not owned.
  std::unique_ptr<Executor> executor;
};
```

### Device Placement

DirectSession performs automatic device placement when no device is specified:

1. Checks user-requested device from NodeDef
2. Falls back to simple placer algorithm
3. Considers device capabilities and op registrations
4. Respects soft/hard device constraints

### Callable API

```cpp
absl::Status MakeCallable(const CallableOptions& callable_options,
                          CallableHandle* out_handle);
absl::Status RunCallable(CallableHandle handle,
                         const std::vector<Tensor>& feed_tensors,
                         std::vector<Tensor>* fetch_tensors,
                         RunMetadata* run_metadata);
absl::Status ReleaseCallable(CallableHandle handle);
absl::Status Finalize();
```

### Cost Model Export

```cpp
void ExportCostModels(CostModelManager::CostModelMap* cost_models);
```

---

## GrpcSession

**Header:** `tensorflow/core/distributed_runtime/rpc/grpc_session.h`

`GrpcSession` is the distributed session implementation that uses gRPC for
communication with remote TensorFlow workers.

### Architecture

```
GrpcSession (client)
  |
  | gRPC
  v
MasterService (master)
  +-- Creates/extends subgraphs on workers
  +-- Coordinates execution across workers
  |
  | gRPC
  v
WorkerService (workers, one per task)
  +-- Local execution via DirectSession-like mechanism
  +-- Manages local devices
  +-- Handles Send/Recv for cross-worker communication
```

### Key Differences from DirectSession

| Aspect           | DirectSession                  | GrpcSession                    |
|-----------------|--------------------------------|--------------------------------|
| Location        | Local only                     | Distributed across machines    |
| Communication   | In-process                     | gRPC over network              |
| Device access   | Local devices only             | All cluster devices            |
| Graph placement | Local device manager           | Master coordinates placement   |
| Execution       | Single process                 | Multiple coordinated processes |

### Usage

```cpp
SessionOptions options;
options.target = "grpc://localhost:2222";  // Master address
Session* session = NewSession(options);
```

---

## Executor

**Header:** `tensorflow/core/common_runtime/executor.h`

The `Executor` is responsible for executing a partitioned graph on a single
device. It manages the scheduling and execution of individual op kernels.

### Executor Interface

```cpp
class Executor {
 public:
  virtual ~Executor();

  struct Args {
    int64_t step_id;
    RendezvousInterface* rendezvous;
    CallFrameInterface* call_frame;
    StepStatsCollectorInterface* stats_collector;
    CancellationManager* cancellation_manager;
    bool is_callback_requested = false;
    std::function<void()> callback;

    // Session-related.
    SessionMetadata* session_metadata = nullptr;
    TensorStore* tensor_store = nullptr;
    ScopedStepContainer* step_container = nullptr;
    CollectiveExecutor* collective_executor = nullptr;
    thread::ThreadPoolOptions threadpool_options;

    // Run options.
    int64_t start_time_usecs = 0;
    std::optional<absl::Time> deadline;
    bool run_all_kernels_inline = false;

    // Function library.
    FunctionLibraryRuntime* flib = nullptr;
  };

  typedef std::function<void()> DoneCallback;

  // Run the graph asynchronously.
  virtual void RunAsync(const Args& args, DoneCallback done) = 0;

  // Run the graph synchronously (blocks until done).
  void Run(const Args& args);
};
```

### ExecutorImpl

The primary implementation of `Executor` is `ExecutorImpl`, which uses a
cost-model-driven scheduling strategy:

```cpp
class ExecutorImpl : public Executor {
 public:
  // Initialize with the graph to execute.
  absl::Status Initialize();

  void RunAsync(const Args& args, DoneCallback done) override;

 private:
  // The graph (immutable after initialization).
  std::unique_ptr<const Graph> graph_;

  // Frame information for control flow.
  std::vector<FrameInfo*> frame_info_;

  // Node scheduling data.
  std::vector<PendingCounts> pending_counts_;
  std::vector<int64_t> mem_tracker_;

  // Total number of pending input edges per node.
  std::vector<int> num_pending_inputs_;
};
```

### Execution Model

The executor uses a dataflow execution model:

1. **Initialization**: Set up initial pending counts for each node
2. **Root nodes**: Schedule nodes with all inputs ready (e.g., source node)
3. **Node execution**:
   a. Pick a ready node from the ready queue
   b. Compute the node's op kernel
   c. Propagate outputs to successor nodes
   d. If a successor has all inputs ready, add to ready queue
4. **Completion**: When sink node is reached, execution is complete

### Frame Management (Control Flow)

The executor manages nested execution frames for while loops:

```cpp
struct FrameInfo {
  int64_t frame_id;
  int iterations;
  int pending_count;
  // ...
};

struct FrameAndIter {
  int64_t frame_id = 0;
  int64_t iter_id = 0;
};
```

### Barrier

For multi-partition execution, the executor uses a barrier mechanism:

```cpp
class ExecutorBarrier {
 public:
  // Create with num_executors participants.
  ExecutorBarrier(size_t num_executors, Rendezvous* rendezvous,
                  DoneCallback done_callback);

  // Called when an executor finishes. Triggers done_callback when
  // all executors have completed.
  void Continue();
};
```

### ExecutorFactory

```cpp
// Register an executor factory for a named executor type.
class ExecutorFactory {
 public:
  virtual absl::Status NewExecutor(const ExecutorOpts& options,
                                   std::unique_ptr<const Graph> graph,
                                   std::unique_ptr<Executor>* executor) = 0;
};

// Create an executor by type name.
absl::Status NewExecutor(const std::string& executor_type,
                         const ExecutorOpts& options,
                         std::unique_ptr<const Graph> graph,
                         std::unique_ptr<Executor>* executor);
```

---

## Graph Partitioning for Execution

**Header:** `tensorflow/core/common_runtime/graph_partition.h`

Before execution, the full graph is partitioned into subgraphs, one per device.

### Partition Process

1. **Assignment**: Each node has an assigned device (via placer or user specification)
2. **Split**: Group nodes by device
3. **Insert Send/Recv**: Add communication ops at partition boundaries
4. **Create Executors**: One executor per partition

### Send/Recv Mechanism

When a tensor produced on device A is needed on device B:

```
Device A:                          Device B:
... -> OpA -> _Send(key="x")  -->  _Recv(key="x") -> OpB -> ...
```

The `_Send` and `_Recv` ops use a `Rendezvous` to exchange tensors. Each
Send/Recv pair has a unique key based on:
- Source node name and output index
- Destination node name and input index
- Frame and iteration IDs (for control flow)

### Partition Options

```cpp
struct PartitionOptions {
  // Function to determine device for each node.
  typedef std::function<string(const Node*)> NodeToLocFunc;
  NodeToLocFunc node_to_loc_func;

  // Function to generate unique names for new nodes.
  typedef std::function<string()> NewNameFunc;
  NewNameFunc new_name_func;

  // Whether to include function library definitions.
  bool include_cvt_funcs = false;

  // Potential additional configurations.
  typedef std::function<bool(const Edge*)> ShouldCopyFunc;
};
```

---

## Rendezvous

**Header:** `tensorflow/core/framework/rendezvous.h`

`Rendezvous` is the mechanism for sending and receiving tensors between
executors (and potentially across devices and processes).

### Interface

```cpp
class RendezvousInterface {
 public:
  virtual ~RendezvousInterface();

  // A parsed key for Send/Recv operations.
  struct ParsedKey {
    string src_device;
    string dst_device;
    // ... other fields
  };

  // Send a tensor to the rendezvous.
  virtual absl::Status Send(const ParsedKey& key,
                            const Rendezvous::Args& args,
                            const Tensor& val,
                            bool is_dead) = 0;

  // Receive a tensor from the rendezvous (blocking).
  virtual void RecvAsync(const ParsedKey& key,
                         const Rendezvous::Args& args,
                         DoneCallback done) = 0;

  // Synchronous receive convenience.
  absl::Status Recv(const ParsedKey& key,
                    const Rendezvous::Args& args,
                    Tensor* val,
                    bool* is_dead);

  // Abort all pending Send/Recv operations.
  virtual void StartAbort(const absl::Status& status) = 0;
};
```

### Rendezvous Key Format

The key format for tensor exchange:

```
frame_id:iter_id:src_incarnation:src_name:src_output:dst_name:dst_input
```

### IntraProcessRendezvous

For local execution within a single process:

```cpp
class IntraProcessRendezvous : public RendezvousInterface {
  // Handles Send/Recv between devices in the same process.
  // For same-device transfers, uses a local map.
  // For cross-device transfers, uses device-to-device copy mechanisms.
};
```

### RemoteRendezvous

For distributed execution:

```cpp
class RemoteRendezvous : public RendezvousInterface {
  // Handles Send/Recv across processes.
  // Uses RPC for cross-process communication.
};
```

### Usage in Execution

```cpp
// Send side (in executor):
Tensor value = ...;
Rendezvous::Args args;
args.device_context = ...;
args.alloc_attrs = ...;
rendezvous->Send(key, args, value, is_dead);

// Recv side (in executor):
rendezvous->RecvAsync(key, args,
    [](const Status& status, const Rendezvous::Args& send_args,
       const Rendezvous::Args& recv_args, const Tensor& val, bool is_dead) {
      // Process received tensor.
    });
```

---

## CostModel

**Header:** `tensorflow/core/common_runtime/costmodel_manager.h`

The `CostModel` tracks execution costs for operations, providing data for
performance analysis and optimization.

### CostModelManager

```cpp
class CostModelManager {
 public:
  // Export all cost models.
  void ExportCostModels(CostModelMap* cost_models);

 private:
  CostModelMap cost_models_;
};
```

### CostModel Data

```cpp
// Per-node cost information.
struct CostModel {
  // Time cost in microseconds for each operation.
  std::vector<int32_t> op_cost;

  // Memory usage per output tensor.
  std::vector<int32_t> output_memory;

  // Maximum temporaries.
  int32_t max_temporary_memory;
};
```

---

## StepStats

**Header:** `tensorflow/core/framework/step_stats.proto`

`StepStats` provides per-step timing and memory information for profiling.

### Protocol Buffer Definition

```protobuf
message StepStats {
  repeated DeviceStepStats dev_stats = 1;
}

message DeviceStepStats {
  string device = 1;
  repeated NodeOutputStats node_stats = 2;
}

message NodeOutputStats {
  string node_name = 1;
  int64 all_start_micros = 2;
  int64 op_end_rel_micros = 3;
  repeated AllocatorMemoryUsed memory = 4;
  int64 all_end_rel_micros = 5;
  int64 op_start_rel_micros = 6;
  int64 timeline_label = 7;
  int64 scheduled_micros = 8;
  int32 thread_id = 9;
}

message AllocatorMemoryUsed {
  string allocator_name = 1;
  int64 total_bytes = 2;
  int64 peak_bytes = 3;
  repeated AllocationRecord allocation_records = 4;
  int64 live_bytes = 5;
}
```

### Collecting StepStats

```cpp
// Enable via RunOptions.
RunOptions run_options;
run_options.set_trace_level(RunOptions::FULL_TRACE);

RunMetadata run_metadata;
session->Run(run_options, inputs, output_names, target_names,
             &outputs, &run_metadata);

// Access stats.
for (const auto& dev_stats : run_metadata.step_stats().dev_stats()) {
  LOG(INFO) << "Device: " << dev_stats.device();
  for (const auto& node_stats : dev_stats.node_stats()) {
    LOG(INFO) << "  Node: " << node_stats.node_name()
              << " start: " << node_stats.all_start_micros() << "us"
              << " duration: " << node_stats.op_end_rel_micros() << "us";
  }
}
```

---

## ThreadPool Configuration

TensorFlow uses thread pools for both inter-op and intra-op parallelism.

### Inter-op Parallelism

Controls how many ops can execute concurrently:

```cpp
// In ConfigProto:
config.set_inter_op_parallelism_threads(num_threads);
```

- Default: Number of CPU cores (or a reasonable default)
- Each ready op is scheduled on a thread from this pool
- Higher values increase concurrency but may increase contention

### Intra-op Parallelism

Controls parallelism within individual ops (e.g., Eigen thread pool):

```cpp
// In ConfigProto:
config.set_intra_op_parallelism_threads(num_threads);
```

- Default: Number of CPU cores
- Used by Eigen for parallelizing internal computation (e.g., matrix multiply)
- Higher values benefit compute-heavy ops

### ThreadPoolOptions

```cpp
struct ThreadPoolOptions {
  // Custom inter-op thread pool.
  thread::ThreadPoolInterface* inter_op_threadpool = nullptr;

  // Custom intra-op thread pool (Eigen).
  Eigen::ThreadPoolInterface* intra_op_threadpool = nullptr;
};
```

### Device Thread Pools

```cpp
// Devices may have their own thread pools.
// GPU devices use stream execution (limited thread usage).
// CPU devices use the configured thread pools.
```

---

## RunOptions and RunMetadata

### RunOptions

**Header:** `tensorflow/core/protobuf/config.proto`

```protobuf
message RunOptions {
  int32 trace_level = 1;  // NO_TRACE, SOFTWARE_TRACE, HARDWARE_TRACE, FULL_TRACE

  // Timeout for the run.
  int64 timeout_in_ms = 2;

  // Inter-op thread pool index (for multiple pools).
  int32 inter_op_thread_pool = 3;

  // Output partition graphs.
  bool output_partition_graphs = 4;

  // Debug options.
  DebugOptions debug_options = 6;

  // Report tensor allocations.
  bool report_tensor_allocations_upon_oom = 7;

  enum TraceLevel {
    NO_TRACE = 0;
    SOFTWARE_TRACE = 1;
    HARDWARE_TRACE = 2;
    FULL_TRACE = 3;
  }
}
```

### RunMetadata

```protobuf
message RunMetadata {
  StepStats step_stats = 1;

  // Cost graph.
  CostGraphDef cost_graph = 2;

  // Partition graphs (if output_partition_graphs was set).
  repeated GraphDef partition_graphs = 3;

  // Function graph information.
  FunctionGraphInfo function_graphs = 5;
}
```

### Usage

```cpp
RunOptions run_options;
run_options.set_trace_level(RunOptions::FULL_TRACE);
run_options.set_timeout_in_ms(5000);

RunMetadata run_metadata;
TF_CHECK_OK(session->Run(run_options, inputs, outputs, targets,
                          &output_tensors, &run_metadata));

// Analyze timing.
for (const auto& dev_stats : run_metadata.step_stats().dev_stats()) {
  // Process per-device stats.
}
```

---

## ConfigProto

**Header:** `tensorflow/core/protobuf/config.proto`

`ConfigProto` is the master configuration for a TensorFlow session. It controls
all aspects of execution behavior.

### Key Configuration Options

```protobuf
message ConfigProto {
  // === Threading ===

  // Number of threads for intra-op parallelism.
  int32 intra_op_parallelism_threads = 2;

  // Number of threads for inter-op parallelism.
  int32 inter_op_parallelism_threads = 3;

  // === GPU Configuration ===
  GPUOptions gpu_options = 4;

  // === Device Configuration ===

  // Override device count per type.
  map<string, int32> device_count = 5;

  // === Execution ===

  // Use soft placement (fall back if preferred device unavailable).
  bool allow_soft_placement = 6;

  // Log device placement.
  bool log_device_placement = 7;

  // === Graph Optimization ===
  GraphOptions graph_options = 8;

  // === RPC Configuration ===
  RPCOptions rpc_options = 12;

  // === Cluster Configuration ===
  ClusterDef cluster_def = 13;

  // === Experimental ===
  Experimental experimental = 20;
}
```

### GPUOptions

```protobuf
message GPUOptions {
  // Fraction of GPU memory to allocate (0.0 to 1.0).
  double per_process_gpu_memory_fraction = 1;

  // Grow GPU memory allocation as needed.
  bool allow_growth = 4;

  // Type of GPU memory allocator.
  string allocator_type = 2;

  // Deferred GPU memory deletion delay (seconds).
  double deferred_deletion_bytes = 3;

  // GPU memory polling.
  int32 polling_in_ms = 6;

  // Force GPU-compatible ops.
  bool force_gpu_compatible = 8;

  // Visible device list (comma-separated GPU indices).
  string visible_device_list = 5;

  // Virtual device configuration.
  repeated VirtualGPUConfiguration virtual_devices = 10;
}
```

### GraphOptions

```protobuf
message GraphOptions {
  // Build cost model after this many steps.
  int64 build_cost_model = 1;
  int64 build_cost_model_after = 3;

  // Minimum number of steps to run before optimization.
  int64 min_optimizer_iterations = 14;

  // Optimizer configuration.
  RewriterConfig rewrite_options = 2;

  // Enable graph optimization.
  bool optimize_for_local_graph = 5;

  // Use TFRT runtime.
  bool use_tfrt = 15;
}
```

### RewriterConfig

```protobuf
message RewriterConfig {
  enum Toggle {
    DEFAULT = 0;
    ON = 1;
    OFF = 2;
    AGGRESSIVE = 3;
  }

  // Layout optimization.
  Toggle layout_optimizer = 1;

  // Constant folding.
  Toggle constant_folding = 2;

  // Shape optimization.
  Toggle shape_optimization = 13;

  // Arithmetic optimization.
  Toggle arithmetic_optimization = 7;

  // Dependency optimization.
  Toggle dependency_optimization = 8;

  // Loop optimization.
  Toggle loop_optimization = 9;

  // Function optimization.
  Toggle function_optimization = 10;

  // Debug stripping.
  Toggle debug_stripper = 11;

  // Scoped allocator optimization.
  Toggle scoped_allocator_optimization = 12;

  // Pin-to-host optimization.
  Toggle pin_to_host_optimization = 14;

  // Auto mixed precision.
  Toggle auto_mixed_precision = 20;
  Toggle auto_mixed_precision_mkl = 23;

  // Disable meta optimizer.
  bool disable_meta_optimizer = 15;

  // Meta optimizer iterations.
  int32 meta_optimizer_iterations = 16;

  // Minimum graph nodes for optimization.
  int64 min_graph_nodes = 17;

  // Custom optimizers.
  repeated CustomOptimizer custom_optimizers = 100;
}
```

### DeviceCount

```cpp
// Specify number of devices of each type.
ConfigProto config;
(*config.mutable_device_count())["CPU"] = 1;
(*config.mutable_device_count())["GPU"] = 2;  // Use 2 GPUs
```

### Experimental Options

```protobuf
message Experimental {
  // Collective ops configuration.
  CollectiveExecutor collective_executor = 1;

  // Executor type override.
  string executor_type = 3;

  // Use TFRT.
  bool use_tfrt = 4;

  // MLIR bridge configuration.
  bool enable_mlir_bridge = 9;

  // Coordination service.
  CoordinationServiceConfig coordination_config = 10;

  // Use composable distributed runtime.
  bool use_composable_tensorflow = 12;
}
```

### Complete Configuration Example

```cpp
SessionOptions options;

// Threading.
options.config.set_intra_op_parallelism_threads(4);
options.config.set_inter_op_parallelism_threads(4);

// GPU.
auto* gpu = options.config.mutable_gpu_options();
gpu->set_per_process_gpu_memory_fraction(0.8);
gpu->set_allow_growth(true);
gpu->set_visible_device_list("0,1");

// Soft placement.
options.config.set_allow_soft_placement(true);
options.config.set_log_device_placement(false);

// Optimization.
auto* rewrite = options.config.mutable_graph_options()
                        ->mutable_rewrite_options();
rewrite->set_constant_folding(RewriterConfig::ON);
rewrite->set_arithmetic_optimization(RewriterConfig::ON);
rewrite->set_layout_optimizer(RewriterConfig::ON);

Session* session = NewSession(options);
```

---

## Session Factory

**Header:** `tensorflow/core/common_runtime/session_factory.h`

TensorFlow uses a factory pattern to create sessions based on the target string.

```cpp
class SessionFactory {
 public:
  // Create a new session.
  virtual absl::Status NewSession(const SessionOptions& options,
                                  Session** out_session) = 0;

  // Check if this factory handles the given target.
  virtual bool AcceptsOptions(const SessionOptions& options) = 0;
};

// Register a session factory.
void RegisterSessionFactory(const string& name, SessionFactory* factory);
```

### Built-in Session Types

| Factory Name     | Target Prefix  | Session Type    |
|-----------------|----------------|-----------------|
| DirectSession   | `""` (empty)   | Local execution |
| GrpcSession     | `"grpc://"`    | Distributed     |

---

## Execution Flow Summary

### Local Execution (DirectSession)

```
Client
  |
  v
Session::Run(inputs, outputs, targets)
  |
  v
DirectSession::Run()
  |
  +-- 1. FindOrCreateExecutors(inputs, outputs, targets)
  |     |-- BuildGraph(inputs, outputs, targets)
  |     |     |-- OptimizeGraph()
  |     |     |-- PartitionGraph()
  |     |-- CreateExecutors(partitions)
  |           |-- For each partition:
  |                 |-- New Executor(graph, device)
  |
  +-- 2. Send inputs to Rendezvous
  |     |-- For each input: rendezvous->Send(key, tensor)
  |
  +-- 3. Run executors
  |     |-- For each partition: executor->RunAsync(args, done)
  |     |-- Wait for all executors (barrier)
  |
  +-- 4. Receive outputs from Rendezvous
  |     |-- For each output: rendezvous->Recv(key, &tensor)
  |
  +-- 5. Return outputs
```

### Distributed Execution (GrpcSession)

```
Client
  |
  v
GrpcSession::Run()
  |
  v
MasterService::RunStep (via gRPC)
  |
  +-- 1. Master processes the request
  |     |-- Build cost model
  |     |-- Partition across workers
  |
  +-- 2. For each worker partition:
  |     |-- WorkerService::RunGraph (via gRPC)
  |           |-- Feed input tensors
  |           |-- Run local executor
  |           |-- Fetch output tensors
  |
  +-- 3. Master collects results
  |     |-- Aggregate outputs from workers
  |
  +-- 4. Return to client
```

---

## Advanced Topics

### Callable API for High-Performance Serving

The callable API avoids the overhead of re-parsing feed/fetch names on each
call:

```cpp
// Setup phase (once).
CallableOptions callable_opts;
callable_opts.add_feed("input");
callable_opts.add_fetch("output");

Session::CallableHandle handle;
session->MakeCallable(callable_opts, &handle);

// Execution phase (repeated).
std::vector<Tensor> inputs = {input_tensor};
std::vector<Tensor> outputs;
session->RunCallable(handle, inputs, &outputs, nullptr);

// Cleanup.
session->ReleaseCallable(handle);
```

### Partial Execution (PRun)

Partial execution allows feeding and fetching intermediate tensors across
multiple calls:

```cpp
// Setup.
std::string handle;
session->PRunSetup({"input_1", "input_2"}, {"output_1", "output_2"},
                   {}, &handle);

// First partial run.
session->PRun(handle, {{"input_1", tensor1}}, {"output_1"}, &outputs);

// Second partial run.
session->PRun(handle, {{"input_2", tensor2}}, {"output_2"}, &outputs);
```

### Session Reset

```cpp
// Reset specific resource containers.
absl::Status Reset(const SessionOptions& options,
                   const std::vector<std::string>& containers);

// Reset all containers.
absl::Status Reset(const SessionOptions& options, {});
```

### Session Finalization

```cpp
// Release graph-related state after setup (reduces memory).
session->Create(graph);

// Warm up callables.
Session::CallableHandle h1, h2;
session->MakeCallable(opts1, &h1);
session->MakeCallable(opts2, &h2);

// Finalize: releases graph construction state.
session->Finalize();

// Run callables normally.
session->RunCallable(h1, inputs, &outputs, nullptr);
session->RunCallable(h2, inputs, &outputs, nullptr);
```

---

## Error Handling

### Common Session Errors

| Error                    | Cause                                         |
|--------------------------|-----------------------------------------------|
| `NotFound`              | Requested output/target not in graph           |
| `InvalidArgument`       | Wrong number/type of inputs                    |
| `FailedPrecondition`    | Session not created or already closed          |
| `Unavailable`           | Device not available                           |
| `Internal`              | Executor or kernel error                       |
| `Aborted`               | Execution cancelled                            |
| `DeadlineExceeded`      | Run timeout exceeded                           |

### Handling Errors

```cpp
absl::Status status = session->Run(inputs, output_names, target_names, &outputs);
if (!status.ok()) {
  if (absl::IsNotFound(status)) {
    LOG(ERROR) << "Output not found: " << status.message();
  } else if (absl::IsInvalidArgument(status)) {
    LOG(ERROR) << "Invalid argument: " << status.message();
  } else {
    LOG(ERROR) << "Run failed: " << status.ToString();
  }
}
```

---

## Thread Safety

| Component            | Thread Safety                                    |
|---------------------|--------------------------------------------------|
| `Session::Run()`    | Thread-safe for concurrent calls                 |
| `Session::Create()` | NOT thread-safe (single-threaded setup)          |
| `Session::Close()`  | NOT thread-safe (must wait for all Runs to finish)|
| `Session::Extend()` | NOT thread-safe (single-threaded)                |
| `Executor::Run()`   | Thread-safe for concurrent executions            |
| `Rendezvous`        | Thread-safe                                      |

---

## Memory Management

### Session Resource Lifecycle

```
Create() -> Allocates:
  - DeviceMgr and devices
  - Graph and function library
  - Per-op resources (variables, queues, etc.)

Run() -> Allocates/Releases:
  - Input/output tensors (caller-owned)
  - Intermediate tensors (executor-managed)
  - Temporary buffers (allocator-managed)

Close() -> Releases:
  - All devices and their resources
  - All cached executors
  - Function library
  - Rendezvous
```

### GPU Memory Management

- **Pre-allocation**: By default, TensorFlow allocates nearly all GPU memory
  (`per_process_gpu_memory_fraction` = 0.95)
- **Allow growth**: Set `allow_growth = true` to allocate incrementally
- **Virtual devices**: Use `visible_device_list` and `virtual_devices` to
  partition GPU memory between logical devices
