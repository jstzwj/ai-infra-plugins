# 15 - NCCL Plugin Development: Net, Tuner, Profiler, Env, and Mixed Plugins

## Primary source files

- `sources/nccl/plugins/net/README.md`
- `sources/nccl/plugins/tuner/README.md`
- `sources/nccl/plugins/profiler/README.md`
- `sources/nccl/plugins/env/README.md`
- `sources/nccl/plugins/mixed/README.md`
- `sources/nccl/src/include/plugin/nccl_net.h`
- `sources/nccl/src/include/plugin/nccl_tuner.h`
- `sources/nccl/src/include/plugin/nccl_profiler.h`
- `sources/nccl/src/include/plugin/nccl_env.h`
- `sources/nccl/src/include/plugin/nccl_gin.h`
- `sources/nccl/src/plugin/plugin_open.cc`
- `sources/nccl/src/plugin/net.cc`
- `sources/nccl/src/plugin/tuner.cc`
- `sources/nccl/src/plugin/profiler.cc`
- `sources/nccl/src/plugin/env.cc`

## Plugin architecture overview

NCCL plugins are shared libraries loaded dynamically at runtime. They expose versioned structs filled with
function pointers. NCCL probes symbols by version so one plugin can support multiple NCCL versions and
NCCL can fall back to older plugin ABIs when needed.

Common plugin properties:

- shared library naming convention,
- environment variable for selection/path,
- versioned exported symbol,
- `name` field used in logs,
- NCCL error-code return values,
- NCCL-provided logging callback for consistent logs.

## Plugin library names and variables

| Plugin type | Default library | Selection variable | Versioned symbol pattern |
|---|---|---|---|
| Net | `libnccl-net.so` | `NCCL_NET_PLUGIN` | `ncclNet_vX`, optional CollNet structs |
| Tuner | `libnccl-tuner.so` or named examples | `NCCL_TUNER_PLUGIN` | `ncclTunerPlugin_vX` / tuner version structs |
| Profiler | `libnccl-profiler.so` | `NCCL_PROFILER_PLUGIN` | `ncclProfiler_vX` |
| Env | `libnccl-env.so` | `NCCL_ENV_PLUGIN` | `ncclEnvPlugin_vX` |
| GIN | internal/plugin-specific | GIN parameters | `nccl_gin` related structs |

For suffix-based loading, NCCL looks for names like:

```text
libnccl-net-${NCCL_NET_PLUGIN}.so
libnccl-profiler-${NCCL_PROFILER_PLUGIN}.so
libnccl-env-${NCCL_ENV_PLUGIN}.so
```

Many plugin variables also accept an absolute pathname.

## Net plugin

### Purpose

Net plugins decouple NCCL core builds from network stack builds. They allow NCCL to work on external or
vendor-specific networks without recompiling NCCL.

### Load/selection

```bash
export LD_LIBRARY_PATH=/path/to/plugin:$LD_LIBRARY_PATH
export NCCL_NET_PLUGIN=myplugin
export NCCL_NET=PluginReportedName
```

`NCCL_NET_PLUGIN` controls which library is loaded. `NCCL_NET` controls which implementation name is used
after the plugin reports one.

### v11 interface shape

From the plugin docs, `ncclNet_v11` includes:

```c
typedef struct {
  const char* name;
  ncclResult_t (*init)(void** ctx, uint64_t commId, ncclNetCommConfig_v11_t* config,
                       ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction);
  ncclResult_t (*devices)(int* ndev);
  ncclResult_t (*getProperties)(int dev, ncclNetProperties_v11_t* props);
  ncclResult_t (*listen)(void* ctx, int dev, void* handle, void** listenComm);
  ncclResult_t (*connect)(void* ctx, int dev, void* handle, void** sendComm,
                          ncclNetDeviceHandle_v11_t** sendDevComm);
  ncclResult_t (*accept)(void* listenComm, void** recvComm,
                         ncclNetDeviceHandle_v11_t** recvDevComm);
  ncclResult_t (*regMr)(void* comm, void* data, size_t size, int type, void** mhandle);
  ncclResult_t (*regMrDmaBuf)(void* comm, void* data, size_t size, int type,
                              uint64_t offset, int fd, void** mhandle);
  ncclResult_t (*deregMr)(void* comm, void* mhandle);
  ncclResult_t (*isend)(void* sendComm, void* data, size_t size, int tag,
                        void* mhandle, void* pHandle, void** request);
  ncclResult_t (*irecv)(void* recvComm, int n, void** data, size_t* sizes, int* tags,
                        void** mhandles, void** pHandles, void** request);
  ncclResult_t (*iflush)(void* recvComm, int n, void** data, int* sizes,
                         void** mhandles, void** request);
  ncclResult_t (*test)(void* request, int* done, int* sizes);
  ncclResult_t (*closeSend)(void* sendComm);
  ncclResult_t (*closeRecv)(void* recvComm);
  ncclResult_t (*closeListen)(void* listenComm);
  ncclResult_t (*getDeviceMr)(void* comm, void* mhandle, void** dptr_mhandle);
  ncclResult_t (*irecvConsumed)(void* recvComm, int n, void* request);
  ncclResult_t (*makeVDevice)(int* d, ncclNetVDeviceProps_t* props);
} ncclNet_t;
```

### Operation flow

```text
init
  -> devices
  -> getProperties for each device
  -> listen on receiver
  -> exchange handle through NCCL bootstrap
  -> connect on sender until sendComm != NULL
  -> accept on receiver until recvComm != NULL
  -> regMr/regMrDmaBuf
  -> isend/irecv/iflush/test
  -> closeSend/closeRecv/closeListen
  -> deregMr
```

### Nonblocking requirements

The net docs require several calls not to block:

- `connect`: may return success with `sendComm == NULL`, NCCL retries.
- `accept`: may return success with `recvComm == NULL`, NCCL retries.
- `isend`: may return success with `request == NULL`, NCCL retries.
- `irecv`: may return success with `request == NULL`, NCCL retries.

Blocking in these functions can hang NCCL progress.

### Device properties

Net plugin `getProperties` fields influence topology and scheduling:

| Field | Importance |
|---|---|
| `name` | logs and `NCCL_NET` selection |
| `pciPath` | topology/NIC locality; `NULL` for virtual devices |
| `guid` | detects shared physical ports/endpoints |
| `ptrSupport` | host/CUDA/DMABUF pointer support |
| `regIsGlobal` | registration cache/global registration behavior |
| `forceFlush` | asks NCCL to flush all transfers |
| `speed` | port speed in Mbps |
| `port` | physical port number |
| `latency` | network latency in microseconds |
| `maxComms` | max connections |
| `maxRecvs` | grouped receive capability |
| `netDeviceType`, `netDeviceVersion` | device networking support |
| `maxP2pBytes`, `maxCollBytes` | chunking limits |
| `vProps` | virtual NIC child devices |

### Net plugin error-code guidance

Common plugin return codes:

- `ncclSuccess`: success.
- `ncclSystemError`: kernel/system/network/hardware/library failure.
- `ncclInternalError`: NCCL core used plugin incorrectly or plugin invariant failed.
- `ncclInvalidUsage`: likely user misconfiguration or size mismatch.
- `ncclInvalidArgument`: rarely needed; NCCL core usually checks arguments.
- `ncclUnhandledCudaError`: CUDA error, uncommon for net plugins.

## CollNet plugin support

Network plugins can expose an optional CollNet structure for in-network collective operations. CollNet is
tied to net plugin versioning and shares many functions. It can accelerate inter-node reductions in
AllReduce when network hardware supports it.

## Tuner plugin

### Purpose

Tuner plugins customize NCCL's algorithm/protocol/channel selection without recompiling NCCL.

### Interface

```c
ncclResult_t (*init)(size_t nRanks, size_t nNodes,
                     ncclDebugLogger_t logFunction, void **context);

ncclResult_t (*getCollInfo)(void* context, ncclFunc_t collType, size_t nBytes,
                            int numPipeOps, float** collCostTable,
                            int numAlgo, int numProto,
                            int regBuff, int* nChannels);

ncclResult_t (*destroy)(void* context);
```

### Cost table behavior

- Lower costs are preferred.
- `0.0` strongly prefers a combination.
- `NCCL_ALGO_PROTO_IGNORE` disables a combination.
- `nChannels` can be changed or left as default.

### Loading

```bash
export LD_LIBRARY_PATH=/path/to/plugin:$LD_LIBRARY_PATH
export NCCL_TUNER_PLUGIN=example
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=TUNING
```

## Profiler plugin

### Purpose

Profiler plugins provide structured NCCL performance events to frameworks and analysis tools.

### Load naming

```bash
export NCCL_PROFILER_PLUGIN=myprofiler
# loads libnccl-profiler-myprofiler.so
```

or set an absolute path.

### v5 interface

```c
typedef struct {
  const char* name;
  ncclResult_t (*init)(void** context, uint64_t commId, int* eActivationMask,
                       const char* commName, int nNodes, int nranks, int rank,
                       ncclDebugLogger_t logfn);
  ncclResult_t (*startEvent)(void* context, void** eHandle,
                             ncclProfilerEventDescr_v5_t* eDescr);
  ncclResult_t (*stopEvent)(void* eHandle);
  ncclResult_t (*recordEventState)(void* eHandle,
                                   ncclProfilerEventState_v5_t eState,
                                   ncclProfilerEventStateArgs_v5_t* eStateArgs);
  ncclResult_t (*finalize)(void* context);
} ncclProfiler_v5_t;
```

Profiler errors generally should not alter NCCL behavior; return `ncclSuccess` except `init` may fail to
disable plugin.

## Env plugin

### Purpose

Env plugins customize environment variable resolution, validation, transformation, or integration with
configuration management systems.

### v1 interface

```c
typedef struct {
  const char* name;
  ncclResult_t (*init)(uint8_t ncclMajor, uint8_t ncclMinor, uint8_t ncclPatch,
                       const char* suffix);
  ncclResult_t (*finalize)(void);
  const char* (*getEnv)(const char* name);
} ncclEnv_v1_t;
```

`getEnv` returns a pointer that must remain valid until plugin finalize or another `getEnv` call for the
same variable. Avoid blocking in `getEnv` because NCCL calls it synchronously.

### Loading

```bash
export LD_LIBRARY_PATH=/path/to/plugin:$LD_LIBRARY_PATH
export NCCL_ENV_PLUGIN=myenv
```

## Mixed plugin

The mixed plugin example demonstrates combining multiple plugin APIs in one library, such as Net and
Tuner. This is useful for vendor packages that ship network support and topology-specific tuning together.

When building mixed plugins, keep symbol/version exports clear and test each interface independently.

## Plugin development checklist

1. Copy/fork the relevant NCCL plugin header versions into the plugin source tree.
2. Export all versioned symbols needed by target NCCL versions.
3. Implement `name` consistently with expected `NCCL_NET` or logs.
4. Use NCCL logging callback, not ad hoc stdout spam.
5. Keep retry/nonblocking functions nonblocking.
6. Treat memory ownership and returned pointer lifetime as ABI contracts.
7. Support graceful fallback: failed init should let NCCL choose built-ins where appropriate.
8. Test with `NCCL_DEBUG=INFO` and subsystem-specific logs.
9. Benchmark against built-in plugins and no tuner plugin.
10. Validate under multi-rank, multi-node, CUDA graph, and buffer-registration workloads if supported.

## Source modification map

| Task | Files |
|---|---|
| plugin loading behavior | `src/plugin/plugin_open.cc`, type-specific `src/plugin/*.cc` |
| net ABI update | `src/include/plugin/nccl_net.h`, net wrappers, plugin examples |
| tuner ABI update | `src/include/plugin/nccl_tuner.h`, `src/plugin/tuner.cc`, tuning integration |
| profiler event update | `src/include/plugin/nccl_profiler.h`, instrumentation callsites |
| env behavior | `src/include/plugin/nccl_env.h`, `src/plugin/env.cc`, `param` system |
| example plugin docs | `plugins/<type>/README.md`, examples under `plugins/<type>/` |
