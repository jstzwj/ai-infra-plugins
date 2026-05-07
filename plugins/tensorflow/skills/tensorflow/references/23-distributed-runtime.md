# TensorFlow Distributed Runtime Reference

This document provides a comprehensive reference for TensorFlow's distributed
runtime system, which enables distributed training and inference across multiple
machines and devices.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Master Service](#master-service)
3. [Worker Service](#worker-service)
4. [ClusterSpec and Server Configuration](#clusterspec-and-server-configuration)
5. [Session Management](#session-management)
6. [Graph Management](#graph-management)
7. [Rendezvous System](#rendezvous-system)
8. [Collective Operations](#collective-operations)
9. [Device Resolution](#device-resolution)
10. [Worker Cache](#worker-cache)
11. [gRPC Integration](#grpc-integration)
12. [Fault Tolerance](#fault-tolerance)
13. [Coordination Service](#coordination-service)
14. [Partial Run Support](#partial-run-support)

---

## Architecture Overview

TensorFlow's distributed runtime follows a **Master-Worker** architecture where:

```
                 +-----------+
                 |   Client   |
                 +-----+-----+
                       |
                  MasterService
                       |
                 +-----+-----+
                 |   Master   |
                 +-----+-----+
                       |
            +----------+----------+
            |          |          |
       WorkerService  WorkerService  WorkerService
            |          |          |
       +----+----+ +---+----+ +---+----+
       | Worker 0 | | Worker 1| | Worker 2|
       +----------+ +---------+ +---------+
       | GPU/CPU  | | GPU/CPU | | GPU/CPU |
       +----------+ +---------+ +---------+
```

### Key Components

- **Client**: The user-facing API (Python/C++) that creates sessions and submits
  computation requests.
- **Master**: Coordinates distributed execution. Manages sessions, partitions
  graphs across workers, and orchestrates execution.
- **Worker**: Executes graph partitions on local devices. Manages local
  resources, tensors, and inter-worker communication.
- **gRPC**: The primary RPC transport for Master-Worker and Worker-Worker
  communication.

### Core Header Files

| File | Purpose |
|------|---------|
| `tensorflow/core/distributed_runtime/master.h` | Master service implementation |
| `tensorflow/core/distributed_runtime/master_env.h` | Master environment configuration |
| `tensorflow/core/distributed_runtime/master_session.h` | Session management on master |
| `tensorflow/core/distributed_runtime/worker.h` | Worker service implementation |
| `tensorflow/core/distributed_runtime/worker_interface.h` | Worker interface abstraction |
| `tensorflow/core/distributed_runtime/worker_cache.h` | Worker connection caching |
| `tensorflow/core/distributed_runtime/graph_mgr.h` | Graph registration and execution |
| `tensorflow/core/distributed_runtime/session_mgr.h` | Worker session management |
| `tensorflow/core/distributed_runtime/rendezvous_mgr_interface.h` | Tensor transfer interface |

---

## Master Service

The Master is the central coordinator in the distributed runtime. It manages
session lifecycle, graph partitioning, and execution orchestration.

### Class: `Master`

**File**: `tensorflow/core/distributed_runtime/master.h`

```cpp
class Master {
 public:
  explicit Master(MasterEnv* env, double session_gc_seconds);
  virtual ~Master();

  typedef std::function<void(const absl::Status&)> MyClosure;

  void CreateSession(const CreateSessionRequest* req,
                     CreateSessionResponse* resp, MyClosure done);
  void ExtendSession(const ExtendSessionRequest* req,
                     ExtendSessionResponse* resp, MyClosure done);
  void PartialRunSetup(const PartialRunSetupRequest* req,
                       PartialRunSetupResponse* resp, MyClosure done);
  void RunStep(CallOptions* opts, const RunStepRequestWrapper* req,
               MutableRunStepResponseWrapper* resp, MyClosure done);
  void CloseSession(const CloseSessionRequest* req,
                    CloseSessionResponse* resp, MyClosure done);
  void ListDevices(const ListDevicesRequest* req,
                   ListDevicesResponse* resp, MyClosure done);
  void Reset(const ResetRequest* req, ResetResponse* resp, MyClosure done);
  void MakeCallable(const MakeCallableRequest* req,
                    MakeCallableResponse* resp, MyClosure done);
  void RunCallable(CallOptions* opts, const RunCallableRequest* req,
                   RunCallableResponse* resp, MyClosure done);
  void ReleaseCallable(const ReleaseCallableRequest* req,
                       ReleaseCallableResponse* resp, MyClosure done);
};
```

### Master Methods

#### CreateSession

Creates a new session on the master. The session encapsulates a computation graph.

- **Request**: `CreateSessionRequest` contains the initial `GraphDef` and
  `ConfigProto`.
- **Response**: `CreateSessionResponse` returns a session handle and the
  graph version.
- **Process**:
  1. Generates a unique session handle
  2. Creates a `MasterSession` object
  3. Partitions the graph across available devices/workers
  4. Registers subgraphs with workers

#### ExtendSession

Extends the current graph in an existing session.

- **Request**: `ExtendSessionRequest` contains additional nodes and the expected
  current graph version.
- **Response**: `ExtendSessionResponse` returns the new graph version.
- **Precondition**: `req->current_graph_version` must match the session's
  current version.
- **Postcondition**: Graph version is incremented.

#### RunStep

Executes one step of the computation graph.

- **Request**: `RunStepRequest` specifies feed tensors (inputs) and fetch
  tensor names (outputs).
- **Response**: `RunStepResponse` contains the output tensors.
- **Process**:
  1. Lookup session by handle
  2. Send inputs to appropriate workers via rendezvous
  3. Execute graph partitions on workers
  4. Collect outputs from workers
  5. Return results to client

#### CloseSession

Closes and cleans up a session.

- Releases all worker sessions
- Deregisters all graphs
- Frees resources

#### ListDevices

Lists all available devices across the cluster.

- Returns device attributes for all local and remote devices
- Used by clients to determine available hardware

#### Reset

Resets specific or all sessions, cleaning up resources.

- Can target specific device name substrings
- Calls `CleanupAll` on all workers

#### Callable API (MakeCallable, RunCallable, ReleaseCallable)

An optimized interface for repeated execution of the same subgraph:

- **MakeCallable**: Pre-compiles a callable handle for a specific subgraph
- **RunCallable**: Executes the pre-compiled callable with new inputs
- **ReleaseCallable**: Releases the callable handle

### MasterEnv

**File**: `tensorflow/core/distributed_runtime/master_env.h`

The master environment holds per-master state and configuration.

```cpp
struct MasterEnv {
  Env* env = nullptr;
  WorkerCacheInterface* worker_cache = nullptr;
  const OpRegistryInterface* ops = nullptr;
  std::vector<Device*> local_devices;
  int experimental_num_shards = 1;

  // Factory functions
  std::function<MasterSession*(...)> master_session_factory;
  std::function<absl::Status(const WorkerCacheFactoryOptions&,
                             WorkerCacheInterface**)> worker_cache_factory;
  CollectiveExecutorMgrInterface* collective_executor_mgr = nullptr;
};
```

| Field | Description |
|-------|-------------|
| `env` | Platform environment abstraction |
| `worker_cache` | Cache for WorkerInterface instances |
| `ops` | Operation registry |
| `local_devices` | Devices co-located with the master |
| `experimental_num_shards` | Sharding factor for singleton components in large-scale training |
| `master_session_factory` | Factory for creating MasterSession instances |
| `worker_cache_factory` | Factory for creating WorkerCacheInterface instances |
| `collective_executor_mgr` | Manager for collective operations |

### WorkerCacheFactoryOptions

Configuration options passed to the worker cache factory:

```cpp
struct WorkerCacheFactoryOptions {
  ClusterDef cluster_def;
  std::string job_name;
  int task_index;
  int replica_index = 0;
  RPCOptions rpc_options;
};
```

### MasterSession

**File**: `tensorflow/core/distributed_runtime/master_session.h`

A session encapsulates a graph computation including resource allocation,
placement, and execution.

```cpp
class MasterSession : public core::RefCounted {
 public:
  MasterSession(const SessionOptions& options, const MasterEnv* env, ...);
  absl::Status Create(GraphDef&& def, const ClusterDef& cluster_def);
  absl::Status Extend(const ExtendSessionRequest* req, ExtendSessionResponse* resp);
  absl::Status Run(CallOptions* opts, const RunStepRequestWrapper& req,
                   MutableRunStepResponseWrapper* resp);
  absl::Status Close();
  void GarbageCollect();
};
```

Key internal structures:

- **ReffedClientGraph**: A reference-counted, partitioned graph ready for execution
- **PerStepState**: State for a single execution step (cost collection, stats)
- **RunState**: Tracks pending inputs/outputs for partial runs

---

## Worker Service

The Worker service executes registered graph partitions and supports
worker-to-worker tensor transfer.

### Class: `Worker`

**File**: `tensorflow/core/distributed_runtime/worker.h`

```cpp
class Worker : public WorkerInterface {
 public:
  Worker(WorkerEnv* env);

  void GetStatusAsync(...) override;
  void CreateWorkerSessionAsync(...) override;
  void DeleteWorkerSessionAsync(...) override;
  void RegisterGraphAsync(...) override;
  void DeregisterGraphAsync(...) override;
  void RunGraphAsync(...) override;
  void CleanupGraphAsync(...) override;
  void CleanupAllAsync(...) override;
  void RecvTensorAsync(...) override;
  void LoggingAsync(...) override;
  void TracingAsync(...) override;
  void RecvBufAsync(...) override;
  void CompleteGroupAsync(...) override;
  void CompleteInstanceAsync(...) override;
  void GetStepSequenceAsync(...) override;
};
```

### Worker Methods

#### GetStatusAsync

Returns the status and device information of the worker.

- **Request**: `GetStatusRequest` - optionally includes configuration flags
- **Response**: `GetStatusResponse` - device attributes list

#### CreateWorkerSessionAsync

Creates a session on the worker. Called by the master when a new session
is established.

- Associates a session handle with the worker
- Initializes worker-side resources (rendezvous, device managers)

#### DeleteWorkerSessionAsync

Deletes a session on the worker.

- Cleans up registered graphs
- Releases session-specific resources

#### RegisterGraphAsync

Registers a graph partition with the worker for execution.

- **Request**: Contains the subgraph definition, graph options, debug options
- **Response**: Returns a graph handle used for subsequent execution
- **Process**:
  1. Parse the GraphDef
  2. Create executor for each device partition
  3. Store the registered graph in `GraphMgr`

#### DeregisterGraphAsync

Deregisters a previously registered graph.

- **Request**: Graph handle to deregister
- Cleans up executors and resources

#### RunGraphAsync

Executes a registered graph partition.

- **Request**: Graph handle, step_id, input tensors
- **Response**: Output tensors, step stats
- **Process**:
  1. Send input tensors via rendezvous
  2. Start parallel executors for each device partition
  3. Collect output tensors
  4. Return results

#### CleanupGraphAsync

Cleans up resources associated with a completed step.

- Removes rendezvous entries for the step_id
- Called after all workers have completed their part of a step

#### CleanupAllAsync

Cleans up all resources on the worker.

- Used during Reset operations
- Deregisters all graphs
- Clears all rendezvous instances

#### RecvTensorAsync

Receives a tensor from another worker. This is the core data transfer
mechanism for cross-worker communication.

- **Request**: `RecvTensorRequest` - step_id, rendezvous key
- **Response**: `TensorResponse` - tensor data and metadata
- **Optimization**: `GrpcWorker` overrides this for more efficient gRPC
  transfer of large binary data

#### LoggingAsync / TracingAsync

Controls and retrieves execution logging and tracing data.

- **LoggingRequest/Response**: Enable/disable logging, retrieve step logs
- **TracingRequest/Response**: Enable/disable tracing, retrieve trace data

#### CompleteGroupAsync / CompleteInstanceAsync

Support collective operations (all-reduce, etc.) by coordinating
group formation and instance execution.

#### GetStepSequenceAsync

Returns step sequence information for collective operations.

### WorkerInterface

**File**: `tensorflow/core/distributed_runtime/worker_interface.h`

Abstract interface for talking with the TensorFlow Worker service. Provides
both async and synchronous wrappers.

```cpp
class WorkerInterface {
 public:
  // Async methods
  virtual void GetStatusAsync(...) = 0;
  virtual void CreateWorkerSessionAsync(...) = 0;
  // ... (all worker methods)

  // Synchronous wrappers
  absl::Status GetStatus(const GetStatusRequest* request,
                         GetStatusResponse* response);
  absl::Status RegisterGraph(const RegisterGraphRequest* request,
                             RegisterGraphResponse* response);
  // ...

 protected:
  virtual ~WorkerInterface() {}
  friend class WorkerCacheInterface;
};
```

---

## ClusterSpec and Server Configuration

### ClusterDef

The `ClusterDef` proto defines the cluster topology:

```protobuf
message ClusterDef {
  repeated JobDef job = 1;
}

message JobDef {
  string name = 1;
  map<int32, string> tasks = 2;
}
```

### ClusterSpec (Python)

```python
cluster = tf.train.ClusterSpec({
    "worker": [
        "worker0.example.com:2222",
        "worker1.example.com:2222",
        "worker2.example.com:2222"
    ],
    "ps": [
        "ps0.example.com:2222",
        "ps1.example.com:2222"
    ]
})
```

### ServerDef

The `ServerDef` proto defines a single server in the cluster:

```protobuf
message ServerDef {
  ClusterDef cluster = 1;
  string job_name = 2;
  int32 task_index = 3;
  ConfigProto default_session_config = 4;
  string protocol = 5;  // "grpc", "grpc+verbs", etc.
  int32 replica = 6;
}
```

### tf.distribute.Server

Creates a TensorFlow server for distributed training:

```python
server = tf.distribute.Server(
    cluster_or_spec=cluster_spec,
    job_name="worker",
    task_index=0,
    protocol="grpc",
    config=config_proto
)

# Start the server (blocks)
server.start()

# Join the server (blocks forever)
server.join()

# Stop the server
server.stop()
```

**Server methods**:
- `start()`: Starts the server. Non-blocking.
- `join()`: Blocks until the server is stopped.
- `stop()`: Stops the server.
- `target`: Returns the gRPC target string for creating sessions.

---

## Session Management

### SessionMgr

**File**: `tensorflow/core/distributed_runtime/session_mgr.h`

Manages worker sessions. Each master session creates corresponding worker
sessions on all participating workers.

```cpp
class SessionMgr {
 public:
  SessionMgr(WorkerEnv* worker_env,
             const std::string& default_worker_name,
             std::unique_ptr<WorkerCacheInterface> default_worker_cache,
             WorkerCacheFactory worker_cache_factory,
             CoordinationServiceRpcHandler* coordination_handler);

  absl::Status CreateSession(const std::string& session,
                             const ServerDef& server_def,
                             bool isolate_session_state, ...);
  absl::Status WorkerSessionForSession(const std::string& session_handle,
                                       std::shared_ptr<WorkerSession>* out);
  absl::Status DeleteSession(const std::string& session);
  absl::Status DeleteAllSessions();
  absl::Status UpdateSession(const std::string& session,
                             const ServerDef& server_def, ...);
};
```

#### Key Session Management Features

1. **Session Isolation**: When `isolate_session_state` is true, each session
   gets its own device manager and function library, preventing interference.

2. **Master Incarnation Tracking**: Tracks which master created each session.
   If a master restarts (different incarnation), old sessions are automatically
   cleaned up.

3. **Legacy Session**: A default session exists for backward compatibility
   with non-clustered operation.

4. **Coordination Service Integration**: Sessions can register with a
   coordination service for distributed coordination.

### WorkerSession

Represents a single session on a worker. Contains:
- Device manager for the session
- Worker cache for reaching other workers
- Graph manager for registered graphs
- Rendezvous manager for tensor transfer

---

## Graph Management

### GraphMgr

**File**: `tensorflow/core/distributed_runtime/graph_mgr.h`

Manages registered graphs on a worker. Each registered graph is identified
by a unique handle.

```cpp
class GraphMgr {
 public:
  explicit GraphMgr(const WorkerEnv* worker_env, const DeviceMgr* device_mgr);

  absl::Status Register(const std::string& handle, const GraphDef& gdef,
                        const GraphOptions& graph_options,
                        const DebugOptions& debug_options,
                        const ConfigProto& config_proto,
                        int64_t collective_graph_key,
                        WorkerSession* session,
                        DistributedFunctionLibraryRuntime* cluster_flr,
                        std::string* graph_handle);

  void ExecuteAsync(const std::string& handle, const int64_t step_id,
                    const ExecutorOpts& opts, const NamedTensors& in,
                    WorkerSession* session, ...);

  absl::Status Deregister(const std::string& handle);
  absl::Status DeregisterAll();
};
```

#### Graph Registration Process

1. **Parse GraphDef**: Parse the subgraph definition
2. **Create Function Library**: Set up function library for the graph
3. **Partition by Device**: Split the graph into per-device subgraphs
4. **Create Executors**: Instantiate an executor for each device partition
5. **Store in Table**: Map the graph handle to the `Item` structure

#### ExecutionUnit Structure

```cpp
struct ExecutionUnit {
  std::unique_ptr<Graph> graph;
  Device* device;            // not owned
  Executor* root;            // not owned
  FunctionLibraryRuntime* lib;  // not owned
  int64_t build_cost_model;
};
```

#### Execution Flow

1. **Send Inputs**: Input tensors are sent via rendezvous to the appropriate
   device executors.
2. **Start Parallel Executors**: Each device partition's executor runs
   concurrently.
3. **Collect Outputs**: Output tensors are received from the rendezvous.
4. **Return Results**: Results are sent back to the master.

---

## Rendezvous System

The rendezvous system manages tensor transfer between producers and consumers,
both within a single process and across processes.

### Rendezvous Interface

```cpp
class Rendezvous {
 public:
  struct ParsedKey {
    string src_device;
    string dst_device;
    uint64 edge_name;
    // ...
  };

  virtual Status Send(const ParsedKey& key, const Tensor& val,
                      bool is_dead) = 0;
  virtual void RecvAsync(const ParsedKey& key, DoneCallback done) = 0;
};
```

### RemoteRendezvous

**File**: `tensorflow/core/distributed_runtime/rendezvous_mgr_interface.h`

Extends `Rendezvous` for cross-process communication:

```cpp
class RemoteRendezvous : public Rendezvous {
 public:
  virtual absl::Status Initialize(WorkerSession* session) = 0;
  virtual void SetRemoteEagerContextDefault() = 0;
  virtual bool IsRemoteEagerContextDefault() = 0;
 protected:
  bool is_cross_process() override { return true; }
};
```

### RendezvousMgrInterface

Manages rendezvous instances, one per step_id:

```cpp
class RendezvousMgrInterface {
 public:
  virtual RefCountPtr<RemoteRendezvous> Find(int64_t step_id) = 0;
  virtual void RecvLocalAsync(int64_t step_id,
                              const Rendezvous::ParsedKey& parsed,
                              Rendezvous::DoneCallback done) = 0;
  virtual absl::Status RecvLocal(int64_t step_id, ...) = 0;
  virtual void Cleanup(int64_t step_id) = 0;
  virtual void CleanupAll() = 0;
};
```

### Rendezvous Key Encoding

TensorFlow encodes rendezvous keys with the following format:

```
<src_device>;<src_incarnation>;<dst_device>;<dst_incarnation>;<edge_name>
```

Components:
- **src_device**: Source device name (e.g., "/job:worker/replica:0/task:0/device:GPU:0")
- **src_incarnation**: Unique incarnation number of the source device
- **dst_device**: Destination device name
- **dst_incarnation**: Unique incarnation number of the destination device
- **edge_name**: Unique name for the tensor edge

### Rendezvous Types

1. **IntraProcessRendezvous**: For tensors within the same process. Direct
   memory transfer between devices.
2. **BaseRemoteRendezvous**: For tensors across processes. Uses RPC for
   inter-process transfer.
3. **IntraWorkerRendezvous**: Optimized for same-worker, different-device
   transfers.

### Tensor Transfer Flow

```
Producer Worker                     Consumer Worker
     |                                    |
     |-- Send(key, tensor) -->            |
     |   (buffers locally)                |
     |                                    |
     |   <-- RecvTensorAsync(key) -------|
     |                                    |
     |-- RPC: Send tensor data --------> |
     |                                    |
     |                     RecvCallback(tensor)
```

---

## Collective Operations

TensorFlow provides collective operations for distributed training:
all-reduce, all-gather, broadcast, and reduce-scatter.

### CollectiveOpGroupMode

Determines which devices participate in a collective operation:

| Mode | Condition | Description |
|------|-----------|-------------|
| `CROSS_REPLICA` | No channel_id, no use_global_device_ids | Group contains all replicas for current partition |
| `CROSS_PARTITION` | channel_id set, no use_global_device_ids | Group contains all partitions for current replica |
| `CROSS_REPLICA_AND_PARTITION` | channel_id set, use_global_device_ids=false | Group contains all replicas for all partitions |
| `FLATTENED_ID` | channel_id set, use_global_device_ids=true | Group uses flattened device IDs |

### Supported Collective Ops

- **AllReduce**: Sum/multiply/min/max across all devices
- **AllGather**: Gather values from all devices
- **Broadcast**: Send value from one device to all
- **ReduceScatter**: Reduce then scatter chunks to devices
- **CollectivePermute**: Permute data across devices

### CollectiveParamResolver

Resolves collective operation parameters across distributed workers:

```cpp
// Distributed version
class CollectiveParamResolverDistributed {
 public:
  void CompleteGroupAsync(...);
  void CompleteInstanceAsync(...);
};
```

### Collective Execution Flow

1. **Group Formation**: Workers negotiate which devices participate
2. **Instance Resolution**: Determine specific operation parameters
3. **Execution**: All participants execute the collective operation
4. **Synchronization**: Workers wait for all to complete

---

## Device Resolution

### DeviceResolverDistributed

**File**: `tensorflow/core/distributed_runtime/device_resolver_distributed.h`

Resolves device information in the distributed setting:

```cpp
class DeviceResolverDistributed {
 public:
  // Look up device attributes locally or remotely
  absl::Status GetDeviceAttributes(const string& device,
                                   DeviceAttributes* attributes);
  // Get all devices in the cluster
  absl::Status GetAllDeviceAttributes(const string& task,
                                      std::vector<DeviceAttributes>* attributes);
};
```

### Device Placement Process

1. Master receives the full graph
2. SimplePlacer or user-specified constraints determine device for each node
3. Graph is partitioned by device
4. Send/Recv nodes are inserted at partition boundaries
5. Each partition is registered with the appropriate worker

---

## Worker Cache

### WorkerCacheInterface

**File**: `tensorflow/core/distributed_runtime/worker_cache.h`

Manages connections to workers and provides `WorkerInterface` instances.

```cpp
class WorkerCacheInterface {
 public:
  virtual void ListWorkers(std::vector<std::string>* workers) const = 0;
  virtual void ListWorkersInJob(const std::string& job_name,
                                std::vector<std::string>* workers) const = 0;
  virtual WorkerInterface* GetOrCreateWorker(const std::string& target) = 0;
  virtual void ReleaseWorker(const std::string& target,
                             WorkerInterface* worker);
  virtual bool GetDeviceLocalityNonBlocking(const std::string& device,
                                            DeviceLocality* locality) = 0;
  virtual void GetDeviceLocalityAsync(const std::string& device,
                                      DeviceLocality* locality,
                                      StatusCallback done) = 0;
  virtual absl::Status GetEagerClientCache(...) = 0;
  virtual absl::Status GetCoordinationClientCache(...) = 0;
};
```

### Worker Cache Features

1. **Connection Reuse**: Maintains gRPC channels to workers for reuse
2. **Worker Enumeration**: Lists all workers or workers in a specific job
3. **Device Locality**: Provides device locality information for optimized
   placement decisions
4. **Eager Client Cache**: Provides eager execution client connections
5. **Coordination Client Cache**: Provides coordination service connections

### WorkerCachePartial

A partial implementation providing common functionality:

- LRU worker caching
- Channel management
- Logging integration

---

## gRPC Integration

### gRPC Transport

TensorFlow's distributed runtime primarily uses gRPC for communication.

#### Key gRPC Components

1. **GrpcMasterService**: Implements the MasterService gRPC interface
2. **GrpcWorkerService**: Implements the WorkerService gRPC interface
3. **GrpcChannelCache**: Manages gRPC channels to remote workers
4. **GrpcRemoteWorker**: WorkerInterface implementation using gRPC stubs

#### RPC Methods

**MasterService RPCs**:
- `CreateSession`
- `ExtendSession`
- `RunStep`
- `CloseSession`
- `ListDevices`
- `Reset`
- `MakeCallable`
- `RunCallable`
- `ReleaseCallable`

**WorkerService RPCs**:
- `GetStatus`
- `CreateWorkerSession`
- `DeleteWorkerSession`
- `RegisterGraph`
- `DeregisterGraph`
- `RunGraph`
- `CleanupGraph`
- `CleanupAll`
- `RecvTensor`
- `RecvBuf`
- `Logging`
- `Tracing`
- `CompleteGroup`
- `CompleteInstance`
- `GetStepSequence`

#### Message Serialization

- Protobuf messages are used for all RPC requests and responses
- Tensor data can be serialized as raw bytes via `tensor_content` or as
  type-specific repeated fields
- `GrpcWorker::RecvTensorAsync` uses optimized tensor encoding for large
  binary data transfer

#### Channel Configuration

```python
# Configure gRPC channels
config = tf.ConfigProto()
config.rpc_options.num_channels = 4  # Number of gRPC channels per target
config.rpc_options.buffer_size = 64 * 1024 * 1024  # Buffer size
```

---

## Fault Tolerance

### Worker Failure Handling

1. **Detection**: Master detects worker failure via RPC errors or timeouts
2. **Session Cleanup**: Failed sessions are garbage collected
3. **Recovery**: Client can retry or create new sessions

### Master Restart

When a master restarts:
1. New incarnation number is generated
2. Workers detect the incarnation mismatch in `CreateWorkerSession`
3. Old sessions associated with the previous master are automatically removed
4. New sessions are created with the new incarnation

### Session Garbage Collection

```cpp
// Master garbage collection
class Master {
  const double session_gc_seconds_;  // Session timeout
  void GC();  // Periodic cleanup of inactive sessions
};
```

Sessions that are inactive for `session_gc_seconds` are automatically closed
and cleaned up.

### Error Handling Patterns

1. **RPC Errors**: Propagated back to client as `absl::Status` with
   appropriate error codes
2. **Worker Unreachable**: Marked in worker cache, new sessions avoid
   unreachable workers
3. **Step Cancellation**: `CallOptions` supports cancellation of in-flight
   RPCs
4. **Partial Run Abort**: `AbortStep` cleans up partial run state

### CancellationManager

Supports cancellation of running steps:

```cpp
class CancellationManager {
 public:
  bool StartCancel();
  bool IsCancelled();
  void RegisterCallback(CancellationToken token, DoneCallback callback);
};
```

---

## Coordination Service

The coordination service provides distributed coordination for multi-worker
training, enabling features like:

- Barrier synchronization
- Error propagation across workers
- Heartbeat monitoring
- Resource management

### CoordinationServiceAgent

Present on each worker, communicates with the coordination service:

```cpp
class CoordinationServiceAgent {
 public:
  absl::Status Initialize();
  absl::Status Shutdown();
  absl::Status WaitAtBarrier(const std::string& barrier_id, int64_t timeout);
};
```

---

## Partial Run Support

Partial runs allow feeding and fetching tensors incrementally, enabling
interactive debugging and advanced execution patterns.

### PartialRunMgr

**File**: `tensorflow/core/distributed_runtime/partial_run_mgr.h`

Manages state for partial runs:

```cpp
class PartialRunMgr {
 public:
  void FindOrCreate(int64_t step_id, Rendezvous* rendezvous);
  // ...
};
```

### Partial Run Flow

1. **PartialRunSetup**: Client specifies which inputs/outputs will be used
2. **RunStep (partial)**: Client feeds some inputs and/or fetches some outputs
3. **Completion**: Once all declared inputs are fed and outputs fetched,
   the step completes

---

## Execution Flow Summary

### Full Distributed Training Step

```
1. Client: tf.Session.run(feed_dict={x: data}, fetch_list=[loss, train_op])
   |
2. Master: RunStep()
   |-- Lookup MasterSession
   |-- BuildGraphOptions (feeds, fetches)
   |-- ReffedClientGraph lookup/create
   |
3. Master: Partition graph across workers
   |-- Insert Send/Recv nodes at partition boundaries
   |
4. Master: Register partitions with workers
   |-- Worker.RegisterGraph() for each partition
   |
5. Master: Execute
   |-- For each partition:
   |   |-- Worker.RunGraph(step_id, inputs)
   |   |-- Worker sends inputs via Rendezvous
   |   |-- Worker starts executors
   |   |-- Cross-worker tensors via RecvTensor RPC
   |   |-- Worker collects outputs
   |
6. Master: Collect results
   |-- Aggregate outputs from all partitions
   |-- Return to client
   |
7. Master: Cleanup
   |-- Worker.CleanupGraph(step_id)
   |-- Rendezvous cleanup
```

---

## Key Environment Variables

| Variable | Description |
|----------|-------------|
| `TF_GRPC_DEFAULT_OPTIONS` | Default gRPC channel arguments |
| `TF_RPC_DEADLINE_SECS` | RPC deadline in seconds |
| `TF_DISABLE_SESSION_GC` | Disable session garbage collection |
| `TF_SESSION_GC_SEC` | Session GC interval in seconds |
| `TF_GRPC_USE_LOCAL_SUBCHANNEL_POOL` | Use local gRPC subchannel pool |

---

## Thread Safety

All distributed runtime components are thread-safe:

- **Master**: Uses mutexes to protect session maps and state
- **Worker**: Uses atomic operations and mutexes for concurrent graph execution
- **GraphMgr**: Thread-safe registration, execution, and deregistration
- **SessionMgr**: Thread-safe session creation and lookup
- **Rendezvous**: Thread-safe Send/Recv operations

---

## Performance Considerations

1. **gRPC Channels**: Use multiple channels per target for higher throughput
2. **Tensor Encoding**: `tensor_content` field is more efficient for large tensors
3. **Graph Caching**: Reuse registered graphs across steps
4. **Callable API**: Use callable API for repeated execution of the same subgraph
5. **RDMA**: Use `grpc+verbs` protocol for RDMA-capable networks
6. **Compression**: Enable gRPC compression for bandwidth-limited environments
