# TensorFlow Platform Abstraction Layer

This reference covers TensorFlow's platform abstraction layer: the `Env` class
for file system and threading operations, the `Device` interface hierarchy,
device factories, synchronization primitives, logging macros, and other
portable abstractions that allow TensorFlow to run on multiple platforms.

---

## Table of Contents

1. [Env Class](#env-class)
2. [FileSystem](#filesystem)
3. [Thread and ThreadPool](#thread-and-threadpool)
4. [Device Interface](#device-interface)
5. [DeviceAttributes](#deviceattributes)
6. [DeviceFactory](#devicefactory)
7. [DeviceType](#devicetype)
8. [Status](#status)
9. [Mutex and ConditionVariable](#mutex-and-conditionvariable)
10. [Notification](#notification)
11. [Logging](#logging)
12. [StringPiece](#stringpiece)
13. [RefCount](#refcount)
14. [Additional Platform Utilities](#additional-platform-utilities)

---

## Env Class

**Header:** `tensorflow/core/platform/env.h`
**Actual implementation:** `tsl/platform/env.h` (via TSL layer)

The `Env` class is TensorFlow's primary platform abstraction. It provides file
system operations, threading, time, and environment variable access. All
platform-specific code is funneled through this interface.

### Getting the Default Env

```cpp
// Get the default environment.
Env* Env::Default();

// Shorthand (in tensorflow namespace).
Env* env = Env::Default();
```

### File System Operations

```cpp
class Env {
 public:
  // --- File Existence and Metadata ---

  // Check if a file exists.
  absl::Status FileExists(const std::string& fname);

  // Get file size in bytes.
  absl::Status GetFileSize(const std::string& fname, uint64_t* size);

  // Check if a path is a directory.
  absl::Status IsDirectory(const std::string& fname);

  // Get file modification time.
  absl::Status FileStat(const std::string& fname, FileStat* stat);

  // Get all children of a directory.
  absl::Status GetChildren(const std::string& dir,
                           std::vector<std::string>* result);

  // Get all matching paths (glob pattern).
  absl::Status GetMatchingPaths(const std::string& pattern,
                                std::vector<std::string>* results);

  // --- File Creation and Deletion ---

  // Delete a file.
  absl::Status DeleteFile(const std::string& fname);

  // Create a directory.
  absl::Status CreateDir(const std::string& dirname);

  // Delete a directory.
  absl::Status DeleteDir(const std::string& dirname);

  // Recursively create directories.
  absl::Status RecursivelyCreateDir(const std::string& dirname);

  // Recursively delete a directory.
  absl::Status DeleteRecursively(const std::string& dirname,
                                  int64_t* undeleted_files,
                                  int64_t* undeleted_dirs);

  // Rename a file.
  absl::Status RenameFile(const std::string& src,
                          const std::string& target);

  // Copy a file.
  absl::Status CopyFile(const std::string& src,
                        const std::string& target);

  // --- File I/O ---

  // Open a file for random read access.
  absl::Status NewRandomAccessFile(
      const std::string& fname,
      std::unique_ptr<RandomAccessFile>* result);

  // Open a file for sequential read.
  absl::Status NewSequentialFile(
      const std::string& fname,
      std::unique_ptr<SequentialFile>* result);

  // Open a file for writing (creates or truncates).
  absl::Status NewWritableFile(
      const std::string& fname,
      std::unique_ptr<WritableFile>* result);

  // Open a file for appending.
  absl::Status NewAppendableFile(
      const std::string& fname,
      std::unique_ptr<WritableFile>* result);

  // --- Protobuf I/O ---

  // Read a binary protobuf file.
  absl::Status ReadBinaryProto(const std::string& fname,
                               protobuf::Message* proto);

  // Read a text protobuf file.
  absl::Status ReadTextProto(const std::string& fname,
                             protobuf::Message* proto);

  // Read text or binary protobuf (auto-detect).
  absl::Status ReadTextOrBinaryProto(const std::string& fname,
                                     protobuf::Message* proto);

  // Write a binary protobuf file.
  absl::Status WriteBinaryProto(const std::string& fname,
                                const protobuf::Message& proto);

  // Write a text protobuf file.
  absl::Status WriteTextProto(const std::string& fname,
                              const protobuf::Message& proto);

  // --- String I/O ---

  // Read entire file to string.
  absl::Status ReadFileToString(const std::string& fname,
                                std::string* data);

  // Write string to file.
  absl::Status WriteStringToFile(const std::string& fname,
                                 const std::string& data);

  // Copy file between filesystems.
  absl::Status FileSystemCopyFile(FileSystem* src_fs,
                                  const std::string& src,
                                  FileSystem* target_fs,
                                  const std::string& target);
};
```

### Threading Operations

```cpp
class Env {
 public:
  // Start a new thread.
  Thread* StartThread(const ThreadOptions& options,
                      const std::string& name,
                      std::function<void()> fn) ;

  // Schedule a closure for execution (may run on any thread).
  void SchedClosure(std::function<void()> closure);

  // Schedule a closure after a delay.
  void SchedClosureAfter(int64_t micros, std::function<void()> closure);

  // Get current thread ID.
  uint32_t GetCurrentThreadId() const;
};
```

### Time Operations

```cpp
class Env {
 public:
  // Current time in microseconds since epoch.
  uint64_t NowMicros();

  // Current time in seconds since epoch.
  uint64_t NowSeconds();

  // Sleep for specified duration.
  void SleepForMicroseconds(int64_t micros);

  // Get environment variable.
  bool GetEnvVar(const char* var_name, std::string* value);
};
```

### FileStat

```cpp
struct FileStat {
  int64_t length;        // File size in bytes
  double mtime_nsec;     // Modification time in nanoseconds
  bool is_directory;     // Whether path is a directory
};
```

### ThreadOptions

```cpp
struct ThreadOptions {
  size_t stack_size = 0;       // Thread stack size (0 = default)
  int numa_node = -1;          // NUMA node affinity (-1 = no affinity)
};
```

### File I/O Interfaces

```cpp
// Sequential file reading.
class SequentialFile {
 public:
  virtual ~SequentialFile();
  virtual absl::Status Read(size_t n, StringPiece* result, char* scratch) = 0;
  virtual absl::Status Skip(uint64_t n) = 0;
};

// Random access file reading.
class RandomAccessFile {
 public:
  virtual ~RandomAccessFile();
  virtual absl::Status Read(uint64_t offset, size_t n,
                            StringPiece* result, char* scratch) const = 0;
};

// File writing.
class WritableFile {
 public:
  virtual ~WritableFile();
  virtual absl::Status Append(StringPiece data) = 0;
  virtual absl::Status Close() = 0;
  virtual absl::Status Flush() = 0;
  virtual absl::Status Sync() = 0;
};
```

### Environment Variables

```cpp
// Get/set/unset environment variables.
int setenv(const char* name, const char* value, int overwrite);
int unsetenv(const char* name);
```

---

## FileSystem

**Header:** `tensorflow/core/platform/file_system.h`

TensorFlow supports multiple file system implementations. Each provides the
same interface for file operations on different storage backends.

### Supported File Systems

| File System   | Scheme          | Description                              |
|--------------|-----------------|------------------------------------------|
| POSIX        | `file://`       | Local filesystem (default)               |
| HDFS         | `hdfs://`       | Hadoop Distributed File System           |
| GCS          | `gs://`         | Google Cloud Storage                     |
| S3           | `s3://`         | Amazon S3                                |
| Memory       | `memory://`     | In-memory filesystem (for testing)       |
| Azure        | `az://`         | Azure Blob Storage                       |

### FileSystem Interface

```cpp
class FileSystem {
 public:
  virtual ~FileSystem();

  // File operations.
  virtual absl::Status NewRandomAccessFile(
      const std::string& fname,
      std::unique_ptr<RandomAccessFile>* result) = 0;

  virtual absl::Status NewWritableFile(
      const std::string& fname,
      std::unique_ptr<WritableFile>* result) = 0;

  virtual absl::Status NewAppendableFile(
      const std::string& fname,
      std::unique_ptr<WritableFile>* result) = 0;

  virtual absl::Status NewReadOnlyMemoryRegionFromFile(
      const std::string& fname,
      std::unique_ptr<ReadOnlyMemoryRegion>* result) = 0;

  // File existence.
  virtual absl::Status FileExists(const std::string& fname) = 0;

  // Directory operations.
  virtual absl::Status GetChildren(const std::string& dir,
                                   std::vector<std::string>* result) = 0;
  virtual absl::Status GetMatchingPaths(const std::string& pattern,
                                        std::vector<std::string>* results);
  virtual absl::Status CreateDir(const std::string& dirname) = 0;
  virtual absl::Status DeleteDir(const std::string& dirname) = 0;
  virtual absl::Status IsDirectory(const std::string& fname);

  // Delete file.
  virtual absl::Status DeleteFile(const std::string& fname) = 0;

  // File size.
  virtual absl::Status GetFileSize(const std::string& fname,
                                   uint64_t* size) = 0;

  // Rename file.
  virtual absl::Status RenameFile(const std::string& src,
                                  const std::string& target) = 0;

  // Stat.
  virtual absl::Status Stat(const std::string& fname, FileStatistics* stat) = 0;

  // Path manipulation.
  virtual std::string TranslateName(const std::string& name) const;
  virtual bool Match(const std::string& filename,
                     const std::string& pattern) const;
};
```

### File System Registration

```cpp
namespace register_file_system {
// Register a new file system implementation.
void Register(const std::string& scheme, FileSystem* filesystem);
}

// Example: Register a custom filesystem.
REGISTER_FILE_SYSTEM("myfs", MyFileSystem);
```

### Path Resolution

```cpp
// TensorFlow automatically resolves file paths to the appropriate
// filesystem based on the URI scheme:
//   "gs://bucket/path"   -> GCS filesystem
//   "hdfs://namenode/path" -> HDFS filesystem
//   "/local/path"        -> POSIX filesystem
//   "memory://path"      -> Memory filesystem
```

---

## Thread and ThreadPool

**Header:** `tensorflow/core/platform/threadpool.h`, `tensorflow/core/platform/env.h`

### Thread

```cpp
class Thread {
 public:
  virtual ~Thread();

  // Factory method via Env.
  // Env::StartThread creates a new thread.
  // The returned Thread object is owned by the caller.
  // The thread starts immediately upon creation.
};
```

### ThreadPool

```cpp
class ThreadPool {
 public:
  // Create a thread pool.
  ThreadPool(Env* env, const ThreadOptions& options,
             const std::string& name, int num_threads);
  ThreadPool(Env* env, const std::string& name, int num_threads);
  explicit ThreadPool(const ThreadPoolOptions& options);

  ~ThreadPool();

  // Schedule a closure.
  void Schedule(std::function<void()> fn);

  // Parallel for loop.
  void ParallelFor(int64_t total, int64_t cost_per_unit,
                   std::function<void(int64_t, int64_t)> fn);

  // Parallel for with barrier.
  void ParallelForWithOutput(
      int64_t total,
      std::function<void(int64_t, int64_t, ...)> fn);

  // Transform parallel for.
  void TransformRangeConcurrently(
      int64_t block_size, int64_t total,
      const std::function<void(int64_t, int64_t)>& fn);

  // Number of threads.
  int NumThreads() const;

  // Current thread id within pool.
  int CurrentThreadId() const;
};
```

### ParallelFor Example

```cpp
ThreadPool pool(Env::Default(), "my_pool", 4);

// Parallel processing of elements.
int64_t num_elements = 10000;
pool.ParallelFor(
    num_elements,
    /*cost_per_unit=*/100,  // Estimated cost per element
    [](int64_t start, int64_t end) {
      for (int64_t i = start; i < end; ++i) {
        // Process element i.
      }
    });
```

### ThreadPoolOptions

```cpp
struct ThreadPoolOptions {
  // Underlying thread pool implementation.
  Eigen::ThreadPoolInterface* eigen_threadpool = nullptr;

  // Custom thread pool for inter-op execution.
  thread::ThreadPoolInterface* inter_op_threadpool = nullptr;

  // Number of threads for intra-op parallelism.
  int32_t num_intra_op_threads = 0;
};
```

### ThreadPoolInterface

```cpp
class ThreadPoolInterface {
 public:
  virtual ~ThreadPoolInterface();
  virtual void Schedule(std::function<void()> fn) = 0;
  virtual int NumThreads() const = 0;
  virtual int CurrentThreadId() const = 0;
};
```

---

## Device Interface

**Header:** `tensorflow/core/framework/device.h`

TensorFlow's `Device` class represents a physical or logical device that can
perform computations. Devices are responsible for executing op kernels.

### Device Naming Convention

Every device has a unique name with the format:
```
/job:job_name/replica:replica_index/task:task_index/device:device_type:device_index
```

Example: `/job:train/replica:0/task:3/device:GPU:2`

### Device Class Hierarchy

```
DeviceBase               (base class with basic device functionality)
  |
  +-- Device            (full device with graph execution)
       |
       +-- LocalDevice  (local device implementation)
```

### DeviceBase

```cpp
class DeviceBase {
 public:
  explicit DeviceBase(Env* env);
  virtual ~DeviceBase();

  Env* env() const;

  // Device attributes.
  virtual const DeviceAttributes& attributes() const = 0;
  virtual const string& name() const = 0;
  virtual const string& device_type() const = 0;

  // Memory management.
  virtual Allocator* GetAllocator(AllocatorAttributes attr) = 0;
  virtual Allocator* GetScopedAllocator(AllocatorAttributes attr,
                                        int64_t step_id);

  // Eigen device access.
  template <typename EigenDeviceType>
  const EigenDeviceType* eigen_device() const;

  // GPU device creation.
  virtual PerOpGpuDevice* MakeGpuDevice();
  virtual void ReinitializeGpuDevice(OpKernelContext* context,
                                     PerOpGpuDevice* device,
                                     DeviceContext* dc,
                                     Allocator* allocator);

  // Resource manager.
  virtual ResourceMgr* resource_manager();

  // Safe execution.
  virtual bool RequiresAccessCompat();

  // State for stream operations.
  virtual DeviceContext* default_device_context();

  // Whether device uses accelerator streams.
  virtual bool has_accelerator() const { return false; }
};
```

### Device Class

```cpp
class Device : public DeviceBase {
 public:
  // Callback type.
  typedef std::function<void(const absl::Status&)> DoneCallback;

  Device(Env* env, DeviceAttributes device_attributes);
  ~Device() override;

  // Device ordering by parsed name.
  static bool LessByParsedName(const Device& a, const Device& b);

  // Full device name.
  const std::string& name() const override;

  // Parsed name.
  const DeviceNameUtils::ParsedName& parsed_name() const;

  // Human-readable device type.
  const std::string& device_type() const override;

  // Device attributes.
  const DeviceAttributes& attributes() const override;

  // Execute an op kernel.
  virtual void Compute(OpKernel* op_kernel, OpKernelContext* context);

  // Execute an async op kernel.
  virtual void ComputeAsync(AsyncOpKernel* op_kernel,
                            OpKernelContext* context,
                            AsyncOpKernel::DoneCallback done);

  // Synchronize device (wait for all operations to complete).
  virtual absl::Status Sync();

  // Make a Tensor from a TensorProto.
  virtual absl::Status MakeTensorFromProto(const TensorProto& tensor_proto,
                                           const AllocatorAttributes alloc_attrs,
                                           Tensor* tensor);

  // Memory usage tracking.
  virtual size_t AllocatedMemory() const;

  // Device context for specific stream.
  virtual DeviceContext* MaybeGetDeviceContext(OpKernelContext* context);

  // Safe compute with access tracking.
  virtual absl::Status TryGetDeviceContext(DeviceContext** out_context);

  // Resource manager (overridable).
  ResourceMgr* resource_manager() override;

  // Enqueue a tensor to a device (for distributed execution).
  virtual void ConsumeResultSet(TensorBuffer* buffer, int64_t size);
};
```

### LocalDevice

```cpp
class LocalDevice : public Device {
 public:
  LocalDevice(Env* env, DeviceAttributes device_attributes);
  ~LocalDevice() override;

  // Access the local allocator.
  Allocator* GetAllocator(AllocatorAttributes attr) override;
};
```

### DeviceContext

```cpp
class DeviceContext {
 public:
  virtual ~DeviceContext();

  // Maintain a reference.
  virtual void Ref();
  virtual void Unref();

  // Copy tensor from device to host.
  virtual void CopyDeviceTensorToCPUSync(const Tensor* device_tensor,
                                         const Device* device,
                                         Tensor* cpu_tensor);

  // Copy tensor from host to device.
  virtual void CopyCPUTensorToDeviceSync(const Tensor* cpu_tensor,
                                         const Device* device,
                                         Tensor* device_tensor);

  // Async copy operations.
  virtual void CopyDeviceTensorToCPU(const Tensor* device_tensor,
                                     StringPiece tensor_name,
                                     Device* device,
                                     Tensor* cpu_tensor,
                                     StatusCallback done);

  virtual void CopyCPUTensorToDevice(const Tensor* cpu_tensor,
                                     const Device* device,
                                     StringPiece tensor_name,
                                     Tensor* device_tensor,
                                     StatusCallback done,
                                     bool sync_dst_compute);

  // Stream execution.
  virtual void MaintainLifetimeOnStream(const Tensor* t,
                                        se::Stream* stream);

  // Get the underlying stream.
  virtual se::Stream* stream();
  virtual se::Stream* device_to_host_stream();
  virtual se::Stream* host_to_device_stream();
};
```

### CPU Device

```cpp
// CPU device implementation.
class CPUDevice : public LocalDevice {
 public:
  CPUDevice(Env* env, DeviceAttributes attrs);
  ~CPUDevice() override;

  void Compute(OpKernel* op_kernel, OpKernelContext* context) override;
  Allocator* GetAllocator(AllocatorAttributes attr) override;
  // ...
};
```

### GPU Device

```cpp
// GPU device implementation (in tensorflow/core/common_runtime/gpu/).
class GPUDevice : public LocalDevice {
 public:
  GPUDevice(Env* env, DeviceAttributes attrs, ...);
  ~GPUDevice() override;

  void Compute(OpKernel* op_kernel, OpKernelContext* context) override;
  void ComputeAsync(AsyncOpKernel* op_kernel, OpKernelContext* context,
                    AsyncOpKernel::DoneCallback done) override;
  Allocator* GetAllocator(AllocatorAttributes attr) override;
  absl::Status Sync() override;

  // GPU-specific methods.
  se::StreamExecutor* executor();
  // ...
};
```

---

## DeviceAttributes

**Header:** `tensorflow/core/framework/device_attributes.pb.h`

`DeviceAttributes` is a protocol buffer containing device metadata.

```protobuf
message DeviceAttributes {
  string name = 1;              // Full device name
  string device_type = 2;       // "CPU", "GPU", "TPU", etc.
  int64 memory_limit = 4;       // Memory limit in bytes
  uint64 incarnation = 6;       // Unique instance identifier
  string physical_device_desc = 7;  // Hardware description

  // Locality information.
  DeviceLocality locality = 8;
}
```

### DeviceLocality

```protobuf
message DeviceLocality {
  int32 bus_id = 1;           // NUMA node / bus identifier
  int32 numa_node = 3;        // NUMA node number
  // GPU-specific locality.
  repeated int32 links = 2;   // Interconnect topology
}
```

### DeviceNameUtils

**Header:** `tensorflow/core/util/device_name_utils.h`

Utility functions for parsing and formatting device names.

```cpp
class DeviceNameUtils {
 public:
  struct ParsedName {
    bool has_job = false;
    string job;
    bool has_replica = false;
    int replica = -1;
    bool has_task = false;
    int task = -1;
    bool has_type = false;
    string type;
    bool has_id = false;
    int id = -1;

    // Comparison operators.
    bool operator==(const ParsedName& other) const;
    bool operator<(const ParsedName& other) const;
  };

  // Parse a device name string.
  static bool ParseFullName(const string& name, ParsedName* parsed);

  // Parse a partial device name (may omit some fields).
  static bool ParseLocalName(const string& name, ParsedName* parsed);

  // Build a full device name.
  static string FullName(const string& job, int replica, int task,
                         const string& type, int id);

  // Get just the local device name (type:id).
  static string LocalName(const string& type, int id);
  static string LocalName(const ParsedName& parsed);

  // Check if a name specifies a complete device.
  static bool IsCompleteSpecification(const ParsedName& parsed,
                                      const DeviceNameUtils::ParsedName& full);
  // Check if names are compatible.
  static bool IsCompatible(const ParsedName& a, const ParsedName& b);

  // Merge two parsed names.
  static bool MergeUnsets(const ParsedName& name, ParsedName* target);
  static bool MergeDevNames(ParsedName* target, const ParsedName& other,
                            bool allow_soft_placement);

  // Canonicalize a device name.
  static string CanonicalizeDeviceName(const string& name);
};
```

### DeviceAttributes Helpers

```cpp
// Create DeviceAttributes with a unique incarnation.
DeviceAttributes Device::BuildDeviceAttributes(
    const string& name,
    const string& device_type,
    uint64 memory_limit,
    const DeviceLocality& locality);

// Short form.
DeviceAttributes Device::BuildDeviceAttributes(
    const string& name,
    const string& device_type,
    uint64 memory_limit);
```

---

## DeviceFactory

**Header:** `tensorflow/core/common_runtime/device_factory.h`

`DeviceFactory` creates devices based on type and configuration.

### DeviceFactory Interface

```cpp
class DeviceFactory {
 public:
  virtual ~DeviceFactory();

  // Create devices.
  virtual absl::Status CreateDevices(
      const SessionOptions& options,
      const string& name_prefix,
      std::vector<std::unique_ptr<Device>>* devices) = 0;

  // Get the device type this factory creates.
  virtual DeviceType DeviceType() = 0;

  // Get the number of devices of this type.
  virtual int32_t DeviceCount(const SessionOptions& options);

  // Register a factory for a device type.
  static void Register(const string& device_type, DeviceFactory* factory);

  // Get factory for a device type.
  static DeviceFactory* GetFactory(const DeviceType& device_type);

  // Create all devices for a session.
  static absl::Status AddDevices(
      const SessionOptions& options,
      const string& name_prefix,
      std::vector<std::unique_ptr<Device>>* devices);

  // Get device count from config.
  static int32_t DeviceCountForType(const DeviceType& type,
                                    const SessionOptions& options);
};
```

### Device Registration

```cpp
// Register a device factory.
REGISTER_LOCAL_DEVICE_FACTORY("CPU", CPUDeviceFactory);
REGISTER_LOCAL_DEVICE_FACTORY("GPU", GPUDeviceFactory);

// With priority (higher priority wins when multiple factories exist).
REGISTER_LOCAL_DEVICE_FACTORY("GPU", GPUDeviceFactory, /*priority=*/200);
```

### Device Creation Flow

```
SessionOptions
  |
  v
DeviceFactory::AddDevices()
  |
  +-- For each device type:
  |     |-- GetFactory(type)
  |     |-- factory->CreateDevices(options, name_prefix, &devices)
  |
  v
DeviceMgr (manages all created devices)
```

### DeviceMgr

**Header:** `tensorflow/core/common_runtime/device_mgr.h`

```cpp
class DeviceMgr {
 public:
  explicit DeviceMgr(std::vector<std::unique_ptr<Device>> devices);

  // List all devices.
  const std::vector<Device*>& ListDevices() const;

  // Look up a device by name.
  Device* FindDeviceByName(const string& name) const;

  // Look up a device by address (for distributed).
  Device* FindDeviceByAddress(const string& address) const;

  // Number of devices.
  int NumDevices() const;
};
```

---

## DeviceType

**Header:** `tensorflow/core/framework/types.h`

`DeviceType` wraps a device type string for type-safe device identification.

```cpp
class DeviceType {
 public:
  // Construction.
  DeviceType(const char* type);  // e.g., "CPU", "GPU"
  explicit DeviceType(const string& type);

  // Access.
  const string& type() const;

  // Comparison.
  bool operator==(const DeviceType& other) const;
  bool operator!=(const DeviceType& other) const;
  bool operator<(const DeviceType& other) const;

 private:
  string type_;
};
```

### Device Type Constants

```cpp
extern const char* const DEVICE_DEFAULT;     // "DEFAULT"
extern const char* const DEVICE_CPU;         // "CPU"
extern const char* const DEVICE_GPU;         // "GPU"
extern const char* const DEVICE_TPU;         // "TPU"
extern const char* const DEVICE_TPU_SYSTEM;  // "TPU_SYSTEM"
```

### Device Type Vectors

```cpp
typedef absl::InlinedVector<DeviceType, 4UL> DeviceTypeVector;
typedef absl::InlinedVector<std::pair<DeviceType, int32_t>, 4UL>
    PrioritizedDeviceTypeVector;
```

---

## Status

**Header:** `tensorflow/core/platform/status.h` (now `absl/status/status.h`)

TensorFlow uses `absl::Status` for error reporting.

### Key Error Codes

```cpp
absl::StatusCode::kOK               // Success
absl::StatusCode::kCancelled         // Operation cancelled
absl::StatusCode::kUnknown           // Unknown error
absl::StatusCode::kInvalidArgument   // Invalid argument
absl::StatusCode::kDeadlineExceeded  // Deadline exceeded
absl::StatusCode::kNotFound          // Resource not found
absl::StatusCode::kAlreadyExists     // Resource already exists
absl::StatusCode::kPermissionDenied  // Permission denied
absl::StatusCode::kResourceExhausted // Resource exhausted (OOM)
absl::StatusCode::kFailedPrecondition // Precondition failed
absl::StatusCode::kAborted           // Operation aborted
absl::StatusCode::kOutOfRange        // Out of range
absl::StatusCode::kUnimplemented     // Not implemented
absl::StatusCode::kInternal          // Internal error
absl::StatusCode::kUnavailable       // Service unavailable
absl::StatusCode::kDataLoss          // Data loss
absl::StatusCode::kUnauthenticated   // Unauthenticated
```

### Error Factory Functions

```cpp
absl::Status absl::OkStatus();
absl::Status absl::CancelledError(const string& message);
absl::Status absl::UnknownError(const string& message);
absl::Status absl::InvalidArgumentError(const string& message);
absl::Status absl::DeadlineExceededError(const string& message);
absl::Status absl::NotFoundError(const string& message);
absl::Status absl::AlreadyExistsError(const string& message);
absl::Status absl::PermissionDeniedError(const string& message);
absl::Status absl::ResourceExhaustedError(const string& message);
absl::Status absl::FailedPreconditionError(const string& message);
absl::Status absl::AbortedError(const string& message);
absl::Status absl::OutOfRangeError(const string& message);
absl::Status absl::UnimplementedError(const string& message);
absl::Status absl::InternalError(const string& message);
absl::Status absl::UnavailableError(const string& message);
absl::Status absl::DataLossError(const string& message);
absl::Status absl::UnauthenticatedError(const string& message);
```

### TensorFlow Error Utilities

**Header:** `tensorflow/core/lib/core/errors.h`

```cpp
namespace errors {
absl::Status InvalidArgument(StringPiece msg);
absl::Status NotFound(StringPiece msg);
absl::Status AlreadyExists(StringPiece msg);
absl::Status ResourceExhausted(StringPiece msg);
absl::Status Unavailable(StringPiece msg);
absl::Status Unimplemented(StringPiece msg);
absl::Status Internal(StringPiece msg);
absl::Status Aborted(StringPiece msg);
absl::Status OutOfRange(StringPiece msg);
absl::Status InvalidArgument(StringPiece msg, ArgTypes... args);
// ... variadic template versions for formatted messages

// Replacement support.
string* ReplaceErrorInStatus(absl::Status status, const string& message);
}
```

### StatusOr

```cpp
// Return a value or an error.
template <typename T>
using StatusOr = absl::StatusOr<T>;

absl::StatusOr<int> GetDeviceCount() {
  if (!initialized) {
    return absl::InternalError("Not initialized");
  }
  return device_count_;
}

// Usage.
auto result = GetDeviceCount();
if (!result.ok()) {
  return result.status();
}
int count = result.value();
```

---

## Mutex and ConditionVariable

**Header:** `tensorflow/core/platform/mutex.h`

TensorFlow provides portable synchronization primitives.

### Mutex

```cpp
class mutex {
 public:
  mutex();
  ~mutex();

  void lock();
  void unlock();
  bool try_lock();

  // For use with std::unique_lock or std::lock_guard.
};

// RAII lock types.
typedef std::lock_guard<mutex> mutex_lock;
typedef std::unique_lock<mutex> unique_lock;

// Shared (reader-writer) lock support.
class tf_shared_lock {
 public:
  explicit tf_shared_lock(mutex& mu);
  ~tf_shared_lock();
};
```

### ConditionVariable

```cpp
class condition_variable {
 public:
  condition_variable();
  ~condition_variable();

  void wait(unique_lock<mutex>& lock);

  template <class Predicate>
  void wait(unique_lock<mutex>& lock, Predicate pred);

  // Timed wait.
  std::cv_status wait_for(unique_lock<mutex>& lock,
                          std::chrono::duration timeout);
  std::cv_status wait_until(unique_lock<mutex>& lock,
                            std::chrono::time_point timeout);

  void notify_one();
  void notify_all();
};
```

### Usage Patterns

```cpp
// Basic mutex usage.
mutex mu;
int counter TF_GUARDED_BY(mu);

void Increment() {
  mutex_lock lock(mu);
  ++counter;
}

// Reader-writer lock.
mutex rw_mu;
std::string data TF_GUARDED_BY(rw_mu);

std::string Read() {
  tf_shared_lock lock(rw_mu);
  return data;
}

void Write(const std::string& new_data) {
  mutex_lock lock(rw_mu);
  data = new_data;
}

// Condition variable.
mutex mu;
condition_variable cv;
bool ready TF_GUARDED_BY(mu);

void WaitForReady() {
  unique_lock lock(mu);
  cv.wait(lock, [this]() { return ready; });
}

void SignalReady() {
  mutex_lock lock(mu);
  ready = true;
  cv.notify_all();
}
```

### Thread Safety Annotations

```cpp
// Mark a mutex as guarding specific fields.
mutex mu_;
int counter_ TF_GUARDED_BY(mu_);

// Mark functions as requiring specific locks.
void DoWork() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

// Mark functions as NOT requiring specific locks.
void SafeFunction() TF_LOCKS_EXCLUDED(mu_);
```

---

## Notification

**Header:** `absl/synchronization/notification.h`

`Notification` is a one-time synchronization primitive. It allows one thread
to signal completion to multiple waiting threads.

### Definition

```cpp
class Notification {
 public:
  Notification();
  ~Notification();

  // Check if notified.
  bool HasBeenNotified() const;

  // Wait for notification (blocks).
  void WaitForNotification();

  // Wait with timeout.
  bool WaitForNotificationWithTimeout(absl::Duration timeout);

  // Signal (can only be called once).
  void Notify();
};
```

### Usage

```cpp
// One-time completion signal.
absl::Notification done;

// Worker thread.
env->SchedClosure([&done]() {
  DoExpensiveWork();
  done.Notify();
});

// Main thread.
DoOtherWork();
done.WaitForNotification();
LOG(INFO) << "Work completed";
```

---

## Logging

**Header:** `tensorflow/core/platform/logging.h`

TensorFlow provides several logging macros that wrap platform-specific logging.

### Standard Logging

```cpp
// Log at various severity levels.
LOG(INFO) << "Informational message";
LOG(WARNING) << "Warning message";
LOG(ERROR) << "Error message";
LOG(FATAL) << "Fatal error (aborts program)";

// Conditional logging.
LOG_IF(INFO, condition) << "Logged only if condition is true";
LOG_EVERY_N(INFO, n) << "Logged every n-th call";
LOG_FIRST_N(INFO, n) << "Logged first n calls only";
```

### Verbose Logging (VLOG)

```cpp
// Verbose logging at various levels.
// VLOG levels: 0 = most important, higher = less important.
VLOG(1) << "Verbose level 1 message";
VLOG(2) << "Verbose level 2 message";

// Control via --vmodule and --v flags:
//   --v=3              Enable VLOG up to level 3 globally
//   --vmodule=module=3 Enable VLOG level 3 for specific module
```

### CHECK Macros

```cpp
// Assert conditions (abort if false).
CHECK(condition);
CHECK_EQ(a, b);       // a == b
CHECK_NE(a, b);       // a != b
CHECK_LT(a, b);       // a < b
CHECK_LE(a, b);       // a <= b
CHECK_GT(a, b);       // a > b
CHECK_GE(a, b);       // a >= b
CHECK_NOTNULL(ptr);   // ptr != nullptr

// CHECK always evaluates (even in release builds).
// DCHECK only evaluates in debug builds.
DCHECK(condition);
DCHECK_EQ(a, b);
// ...
```

### TensorFlow-Specific Macros

```cpp
// TF_CHECK_OK: Check Status and abort if not OK.
TF_CHECK_OK(status) << "Additional context: " << status;

// OP_REQUIRES: Check condition in kernel (sets status).
OP_REQUIRES(context, condition, errors::InvalidArgument("msg"));

// OP_REQUIRES_OK: Check Status in kernel.
OP_REQUIRES_OK(context, status);
```

### LOG Macros with Severity

| Macro     | Severity | Behavior                                   |
|-----------|----------|--------------------------------------------|
| `LOG(INFO)` | 0      | Informational, always logged               |
| `LOG(WARNING)` | 1   | Warning, always logged                     |
| `LOG(ERROR)` | 2     | Error, always logged                       |
| `LOG(FATAL)` | 3     | Fatal, logs and aborts                     |
| `VLOG(n)` | n >= 0  | Verbose, only if --v >= n                  |
| `CHECK`   | N/A     | Always active, aborts on failure           |
| `DCHECK`  | N/A     | Debug only, aborts on failure in debug     |

---

## StringPiece

**Header:** `tensorflow/core/lib/core/stringpiece.h`

`StringPiece` is a non-owning string reference (string view). See the
[Cpp Framework Reference](16-cpp-framework.md) for complete details.

Key characteristics:
- Does not own or copy string data
- Lightweight (pointer + size)
- Can be implicitly constructed from `const char*`, `std::string`
- Compatible with `absl::string_view`

---

## RefCount

**Header:** `tensorflow/core/lib/core/refcount.h`

TensorFlow's reference counting system. See the
[Cpp Framework Reference](16-cpp-framework.md) for complete details.

Key classes:
- `RefCounted`: Base class with atomic reference count
- `RefCountPtr<T>`: Smart pointer that calls `Unref()` on destruction
- `WeakPtr<T>`: Weak reference that can detect if object is still alive

---

## Additional Platform Utilities

### Protobuf Helpers

```cpp
// Read/write protobuf messages.
absl::Status ReadBinaryProto(Env* env, const string& fname,
                             protobuf::Message* proto);
absl::Status WriteBinaryProto(Env* env, const string& fname,
                              const protobuf::Message& proto);
absl::Status ReadTextProto(Env* env, const string& fname,
                           protobuf::Message* proto);
absl::Status WriteTextProto(Env* env, const string& fname,
                            const protobuf::Message& proto);
```

### Memory Utilities

```cpp
// Platform-specific memory operations.
namespace port {
// Aligned memory allocation.
void* AlignedMalloc(size_t size, int minimum_alignment);
void AlignedFree(void* aligned_memory);

// Memory mapping.
void* Mmap(void* addr, size_t length, int prot, int flags, int fd, off_t offset);
int Munmap(void* addr, size_t length);

// Aligned allocation status.
bool MallocExtension_GetAllocatedSize(const void* p, size_t* size);
}
```

### NUMA Support

```cpp
namespace port {
// NUMA node operations.
int NUMANumNodes();
void NUMABind(int node);
int NUMAGetNode();
void* NUMAMalloc(int node, size_t size, int minimum_alignment);
void NUMAFree(void* ptr, size_t size);
}
```

### Byte Ordering

```cpp
// Network byte order conversion.
uint32_t HostToNet32(uint32_t x);
uint16_t HostToNet16(uint16_t x);
uint32_t NetToHost32(uint32_t x);
uint16_t NetToHost16(uint16_t x);

// Little-endian to host conversion.
uint64_t LittleEndian::FromHost64(uint64_t x);
uint64_t LittleEndian::ToHost64(uint64_t x);
bool LittleEndian::IsLittleEndian();
```

### Stack Trace

```cpp
// Get current stack trace.
string CurrentStackTrace();

// Debug stack trace for debugging.
class AbstractStackTrace {
 public:
  virtual ~AbstractStackTrace();
  virtual string ToString() const = 0;
  virtual bool IsEmpty() const = 0;
};
```

### SubProcess

```cpp
class SubProcess {
 public:
  SubProcess();
  ~SubProcess();

  // Configure I/O channels.
  void SetChannel(Channel channel, ChannelAction action);

  // Start the process.
  bool Start(const string& program, const std::vector<string>& argv);

  // Communicate (send input, receive output).
  bool Communicate(const string& stdin_input,
                   string* stdout_output,
                   string* stderr_output);

  // Wait for process to finish.
  int Wait();

  // Kill the process.
  bool Kill(int signal);
};
```

### Registry Pattern

```cpp
// Generic registration pattern.
template <typename Key, typename Factory>
class Registry {
 public:
  static Registry* Global();

  void Register(const Key& key, Factory factory);
  Factory* LookUp(const Key& key);

 private:
  std::unordered_map<Key, Factory> registry_;
};
```

### Error Handling Utilities

```cpp
// Format error messages with source location.
#define TF_RETURN_IF_ERROR(status)       \
  do {                                   \
    if (!status.ok()) return status;      \
  } while (0)

#define TF_CHECK_OK(status)              \
  do {                                   \
    if (!status.ok()) {                  \
      LOG(FATAL) << status;              \
    }                                    \
  } while (0)
```

---

## Platform Abstraction Summary

### TSL (TensorFlow Serving Library) Layer

Modern TensorFlow delegates many platform abstractions to the TSL layer:

```cpp
// tensorflow/core/platform/env.h re-exports TSL symbols:
using tsl::Env;
using tsl::Thread;
using tsl::ThreadPool;
using tsl::FileSystemCopyFile;
using tsl::ReadBinaryProto;
using tsl::WriteBinaryProto;
// ... etc.
```

This allows TensorFlow and JAX/XLA to share the same platform abstraction
code.

### Platform-Specific Implementations

| Platform    | Implementation Directory                    |
|------------|---------------------------------------------|
| Linux      | `tsl/platform/default/`                     |
| macOS      | `tsl/platform/apple/`                       |
| Windows    | `tsl/platform/windows/`                     |
| Android    | `tensorflow/core/platform/android/`         |
| iOS        | `tensorflow/core/platform/apple/`           |
| Emscripten | `tensorflow/core/platform/emscripten/`      |

Each platform provides:
- `env.cc`: Env implementation (file I/O, threading, time)
- `file_system.cc`: Default filesystem implementation
- `logging.cc`: Logging implementation
- `mutex.cc`: Synchronization primitives
- `notification.cc`: Notification implementation

---

## Device Registration Summary

### Complete Flow

```
1. REGISTER_LOCAL_DEVICE_FACTORY("GPU", GPUDeviceFactory)
   |
   v
2. SessionOptions with config
   |
   v
3. DeviceFactory::AddDevices(options, name_prefix, &devices)
   |
   +-- For each registered device type:
   |     |-- GetFactory(type)
   |     |-- factory->DeviceCount(options) -> num_devices
   |     |-- For i in [0, num_devices):
   |           |-- DeviceNameUtils::FullName(...)
   |           |-- Device::BuildDeviceAttributes(name, type, memory_limit)
   |           |-- factory->CreateDevice(options, name, attrs)
   |
   v
4. DeviceMgr(std::move(devices))
   |
   v
5. Session uses DeviceMgr for execution
```

### Device Placement

When placing ops on devices:

1. Check user-requested device from NodeDef
2. Validate against registered kernels for the op
3. Fall back to soft placement if requested device unavailable
4. Consider device locality for optimal data placement
5. Handle multi-device via Send/Recv insertion

---

## Cross-Reference

| Component         | Header                                        | Reference Document                  |
|------------------|-----------------------------------------------|-------------------------------------|
| Tensor           | `core/framework/tensor.h`                     | [16-cpp-framework.md](16-cpp-framework.md) |
| Graph            | `core/graph/graph.h`                          | [17-graph-construction.md](17-graph-construction.md) |
| Session          | `core/public/session.h`                       | [18-session-and-executor.md](18-session-and-executor.md) |
| OpKernel         | `core/framework/op_kernel.h`                  | [19-kernels.md](19-kernels.md)     |
| REGISTER_OP      | `core/framework/op.h`                         | [20-ops-registration.md](20-ops-registration.md) |
| Env              | `core/platform/env.h`                         | This document                      |
| Device           | `core/framework/device.h`                     | This document                      |
