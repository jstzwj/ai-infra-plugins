# Distributed Computing Reference

This document provides comprehensive coverage of vLLM's distributed computing subsystem, including all forms of parallelism, communication primitives, parallel state management, distributed KV cache transfer, executor architecture, and inter-process coordination.

**Source Locations:**
- `vllm/distributed/` - Core distributed infrastructure
- `vllm/config/parallel.py` - Parallel configuration
- `vllm/v1/executor/` - Executor implementations
- `vllm/v1/worker/` - Worker implementations
- `vllm/v1/engine/` - Engine coordination

---

## Table of Contents

1. [Parallel Configuration (ParallelConfig)](#1-parallel-configuration-parallelconfig)
2. [Parallel State Management](#2-parallel-state-management)
3. [GroupCoordinator](#3-groupcoordinator)
4. [Communication Operations](#4-communication-operations)
5. [Device Communicators](#5-device-communicators)
6. [Executor Architecture](#6-executor-architecture)
7. [UniProcExecutor](#7-uniprocexecutor)
8. [MultiprocExecutor](#8-multiprocexecutor)
9. [Worker Architecture](#9-worker-architecture)
10. [Tensor Parallelism (TP)](#10-tensor-parallelism-tp)
11. [Pipeline Parallelism (PP)](#11-pipeline-parallelism-pp)
12. [Data Parallelism (DP)](#12-data-parallelism-dp)
13. [Expert Parallelism (EP) for MoE](#13-expert-parallelism-ep-for-moe)
14. [Expert Parallelism Load Balancing (EPLB)](#14-expert-parallelism-load-balancing-eplb)
15. [Context Parallelism (PCP / DCP)](#15-context-parallelism-pcp--dcp)
16. [KV Cache Transfer / Disaggregated Prefill](#16-kv-cache-transfer--disaggregated-prefill)
17. [KV Connector Base (V1)](#17-kv-connector-base-v1)
18. [KV Connector Implementations](#18-kv-connector-implementations)
19. [Stateless Process Groups](#19-stateless-process-groups)
20. [Elastic Expert Parallelism](#20-elastic-expert-parallelism)
21. [Distributed KV Events](#21-distributed-kv-events)
22. [DP Coordination Utilities](#22-dp-coordination-utilities)
23. [All-to-All Communication (MoE)](#23-all-to-all-communication-moe)
24. [Shared Memory Broadcast](#24-shared-memory-broadcast)
25. [Weight Transfer](#25-weight-transfer)
26. [NIXL Utilities](#26-nixl-utilities)

---

## 1. Parallel Configuration (ParallelConfig)

**File:** `vllm/config/parallel.py`

### Type Aliases

```python
ExpertPlacementStrategy = Literal["linear", "round_robin"]
DistributedExecutorBackend = Literal["ray", "mp", "uni", "external_launcher"]
DataParallelBackend = Literal["ray", "mp"]
EPLBPolicyOption = Literal["default"]
DCPCommBackend = Literal["ag_rs", "a2a"]
EPLBCommunicatorBackend = Literal["torch_nccl", "torch_gloo", "nixl", "pynccl"]
All2AllBackend = Literal[
    "naive", "pplx", "deepep_high_throughput", "deepep_low_latency",
    "mori", "nixl_ep", "allgather_reducescatter",
    "flashinfer_all2allv", "flashinfer_nvlink_two_sided",
    "flashinfer_nvlink_one_sided",
]
```

### EPLBConfig

```python
@config
class EPLBConfig:
    window_size: int = 1000            # Window size for expert load recording
    step_interval: int = 3000          # Interval for rearranging experts
    num_redundant_experts: int = 0     # Number of redundant experts
    log_balancedness: bool = False     # Log balancedness per step
    log_balancedness_interval: int = 1 # Interval for balancedness logging
    use_async: bool = False            # Non-blocking EPLB
    policy: EPLBPolicyOption = "default"
    communicator: EPLBCommunicatorBackend | None = None
```

### ParallelConfig

```python
@config
class ParallelConfig:
    pipeline_parallel_size: int = 1
    tensor_parallel_size: int = 1
    prefill_context_parallel_size: int = 1
    data_parallel_size: int = 1
    data_parallel_size_local: int = 1
    data_parallel_rank: int = 0
    data_parallel_rank_local: int | None = None
    data_parallel_master_ip: str = "127.0.0.1"
    data_parallel_rpc_port: int = 29550
    data_parallel_master_port: int = 29500
    data_parallel_backend: DataParallelBackend = "mp"
    data_parallel_external_lb: bool = False
    data_parallel_hybrid_lb: bool = False
    is_moe_model: bool | None = None
    enable_expert_parallel: bool = False
    enable_ep_weight_filter: bool = False
    enable_eplb: bool = False
    eplb_config: EPLBConfig = EPLBConfig()
    expert_placement_strategy: ExpertPlacementStrategy = "linear"
    all2all_backend: All2AllBackend = "allgather_reducescatter"
    max_parallel_loading_workers: int | None = None
    disable_custom_all_reduce: bool = False
    enable_elastic_ep: bool = False
    enable_dbo: bool = False
    ubatch_size: int = 0
    dbo_decode_token_threshold: int = 32
    dbo_prefill_token_threshold: int = 512
    disable_nccl_for_dp_synchronization: bool | None = None
    ray_workers_use_nsight: bool = False
    ray_runtime_env: RuntimeEnv | None = None
    placement_group: PlacementGroup | None = None
    distributed_executor_backend: str | DistributedExecutorBackend | type[Executor] | None = None
    worker_cls: str = "auto"
    sd_worker_cls: str = "auto"
    worker_extension_cls: str = ""
    master_addr: str = "127.0.0.1"
    master_port: int = 29501
    node_rank: int = 0
    nnodes: int = 1
    numa_bind: bool = False
    numa_bind_nodes: list[int] | None = None
    numa_bind_cpus: list[str] | None = None
    distributed_timeout_seconds: int | None = None
    world_size: int  # computed: TP * PP * PCP
    rank: int = 0
    decode_context_parallel_size: int = 1
    dcp_kv_cache_interleave_size: int = 1
    dcp_comm_backend: DCPCommBackend = "ag_rs"
    cp_kv_cache_interleave_size: int = 1
    data_parallel_index: int  # computed
```

### Key ParallelConfig Properties

| Property | Type | Description |
|---|---|---|
| `world_size_across_dp` | `int` | `TP * PP * DP` |
| `use_ubatching` | `bool` | True if `enable_dbo` or `ubatch_size > 1` |
| `num_ubatches` | `int` | 2 if `enable_dbo`, else `ubatch_size` |
| `local_engines_only` | `bool` | True for external or hybrid LB modes |
| `use_ray` | `bool` | True if backend is Ray |
| `use_sequence_parallel_moe` | `bool` | True for compatible all2all + EP + TP > 1 + DP > 1 |
| `use_batched_dp_moe` | `bool` | True for deepep_low_latency/nixl_ep + EP + DP > 1 |
| `node_rank_within_dp` | `int` | `node_rank % nnodes_within_dp` |
| `nnodes_within_dp` | `int` | `nnodes // (dp_size // dp_size_local)` |
| `local_world_size` | `int` | `world_size // nnodes_within_dp` |

### Key Methods

```python
def get_next_dp_init_port(self) -> int:
    """Pop a port from the prepared port list for DP group init."""

def stateless_init_dp_group(self, return_store: bool = False) -> ProcessGroup | tuple[ProcessGroup, Store]:
    """Initialize a stateless Gloo DP group with retry on EADDRINUSE."""

@staticmethod
def has_unfinished_dp(dp_group: ProcessGroup, has_unfinished: bool) -> bool:
    """OR reduce across DP ranks via MAX all-reduce."""

@staticmethod
def sync_dp_state(dp_group, has_unfinished, pending_pause) -> tuple[bool, bool]:
    """Combined SUM all-reduce for DP state: (has_unfinished_global, pause_consensus)."""

@staticmethod
def sync_kv_cache_memory_size(dp_group, kv_cache_memory) -> int:
    """MIN reduce across DP ranks for KV cache memory."""

def compute_hash(self) -> str:
    """Hash of computation-graph-affecting configs, ignoring topology/networking."""
```

---

## 2. Parallel State Management

**File:** `vllm/distributed/parallel_state.py`

### Module-Level State

```python
_WORLD: GroupCoordinator | None = None       # World group
_INNER_DP_WORLD: GroupCoordinator | None = None  # Inner DP world group
_NODE_COUNT: int | None = None               # Number of nodes
_TP: GroupCoordinator | None = None          # Tensor parallel group
_DCP: GroupCoordinator | None = None         # Decode context parallel group
_PP: GroupCoordinator | None = None          # Pipeline parallel group
_DP: GroupCoordinator | None = None          # Data parallel group
_EP: GroupCoordinator | None = None          # Expert parallel group (MoE only)
_EPLB: GroupCoordinator | None = None        # EPLB group (MoE only)
_PCP: GroupCoordinator | None = None         # Prefill context parallel group
```

### Accessor Functions

```python
def get_world_group() -> GroupCoordinator
def get_inner_dp_world_group() -> GroupCoordinator
def get_tp_group() -> GroupCoordinator
def get_dcp_group() -> GroupCoordinator
def get_pp_group() -> GroupCoordinator
def get_dp_group() -> GroupCoordinator
def get_ep_group() -> GroupCoordinator
def get_eplb_group() -> GroupCoordinator
def get_pcp_group() -> GroupCoordinator
def get_tensor_model_parallel_world_size() -> int
def get_tensor_model_parallel_rank() -> int
def get_decode_context_model_parallel_world_size() -> int
def get_decode_context_model_parallel_rank() -> int
def get_node_count() -> int
def model_parallel_is_initialized() -> bool
```

### Initialization Functions

```python
def init_distributed_environment(
    world_size: int = -1,
    rank: int = -1,
    distributed_init_method: str = "env://",
    local_rank: int = -1,
    backend: str = "nccl",
    timeout: timedelta | None = None,
) -> None:
```

Initializes the global distributed environment via `torch.distributed.init_process_group()`. Handles DP rank offset, multi-node addressing, and elastic EP world group setup.

```python
def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    prefill_context_model_parallel_size: int = 1,
    decode_context_model_parallel_size: int | None = 1,
    backend: str | None = None,
) -> None:
```

Creates all parallel groups from a rank layout of shape `[ExternalDP, DP, PP, PCP, TP]`. Each group is formed by transposing and reshaping this layout to isolate the appropriate dimension:

| Group | Layout Dimension |
|---|---|
| TP | Last dimension (TP) |
| DCP | Reshaped from full layout, splits TP groups |
| PCP | Transposed PCP dimension |
| PP | Transposed PP dimension |
| DP | Transposed DP dimension |
| EP | Merged DP * PCP * TP per PP rank (MoE only) |
| EPLB | Same ranks as EP, separate process group |

```python
def ensure_model_parallel_initialized(
    tensor_model_parallel_size, pipeline_model_parallel_size,
    prefill_context_model_parallel_size, decode_context_model_parallel_size,
    backend,
) -> None:
```

Helper that initializes if needed, or validates sizes match.

### Cleanup Functions

```python
def destroy_model_parallel() -> None:      # Destroys all parallel groups
def destroy_distributed_environment() -> None:  # Destroys world group
def cleanup_dist_env_and_memory(shutdown_ray: bool = False) -> None:  # Full cleanup + GC
```

### Utility Functions

```python
def in_the_same_node_as(pg, source_rank: int = 0) -> list[bool]:
    """Test if each rank is on the same node as source_rank via shared memory."""

def is_global_first_rank() -> bool:
    """True if global rank 0."""

def is_local_first_rank() -> bool:
    """True if local rank 0 on the node."""

def _node_count(pg) -> int:
    """Count total nodes in a process group."""
```

### Context Managers

```python
@contextmanager
def graph_capture(device: torch.device):
    """CUDA graph capture context, sets up stream and CA/AITER contexts."""

@contextmanager
def patch_tensor_parallel_group(tp_group: GroupCoordinator):
    """Temporarily patch the TP group for speculative decoding."""
```

### Registered Custom Ops

```python
torch.ops.vllm.all_reduce(tensor, group_name) -> tensor
torch.ops.vllm.reduce_scatter(tensor, dim, world_size, group_name) -> tensor
torch.ops.vllm.all_gather(tensor, dim, world_size, group_name) -> tensor
torch.ops.vllm.patched_fused_scaled_matmul_reduce_scatter(...)
```

---

## 3. GroupCoordinator

**File:** `vllm/distributed/parallel_state.py`

```python
class GroupCoordinator:
    rank: int                    # Global rank
    ranks: list[int]             # All ranks in the group
    world_size: int              # Size of the group
    local_rank: int              # Local device index
    rank_in_group: int           # Rank inside the group
    cpu_group: ProcessGroup      # Gloo group for CPU communication
    device_group: ProcessGroup   # NCCL/device group
    device_communicator: DeviceCommunicatorBase | None
    mq_broadcaster: MessageQueue | None
```

### Constructor

```python
def __init__(
    self,
    group_ranks: list[list[int]],
    local_rank: int,
    torch_distributed_backend: str | Backend,
    use_device_communicator: bool,
    use_message_queue_broadcaster: bool = False,
    group_name: str | None = None,
) -> None:
```

Creates CPU (Gloo) and device (NCCL) process groups for the specified ranks. If `use_device_communicator` is True and `world_size > 1`, instantiates a platform-specific `DeviceCommunicatorBase`. If `use_message_queue_broadcaster` is True and `world_size > 1`, creates a shared memory `MessageQueue`.

### Properties

| Property | Returns |
|---|---|
| `first_rank` | `self.ranks[0]` |
| `last_rank` | `self.ranks[-1]` |
| `is_first_rank` | `self.rank == self.first_rank` |
| `is_last_rank` | `self.rank == self.last_rank` |
| `next_rank` | `self.ranks[(rank_in_group + 1) % world_size]` |
| `prev_rank` | `self.ranks[(rank_in_group - 1) % world_size]` |

### Communication Methods

```python
def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
    """Out-of-place all-reduce via device communicator or custom op."""

def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Out-of-place all-gather along dim."""

def all_gatherv(self, input_, dim=0, sizes=None) -> torch.Tensor:
    """Variable-size all-gather."""

def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Out-of-place reduce-scatter along dim."""

def reduce_scatterv(self, input_, dim=-1, sizes=None) -> torch.Tensor:
    """Variable-size reduce-scatter."""

def gather(self, input_, dst=0, dim=-1) -> torch.Tensor | None:
    """Gather to dst rank."""

def broadcast(self, input_, src=0) -> torch.Tensor:
    """Broadcast tensor from src."""

def broadcast_object(self, obj=None, src=0) -> Any:
    """Broadcast Python object via MQ or Gloo."""

def broadcast_object_list(self, obj_list, src=0, group=None) -> list[Any]:
    """Broadcast list of objects."""

def broadcast_tensor_dict(self, tensor_dict=None, src=0, group=None, metadata_group=None) -> dict:
    """Broadcast dict of tensors + metadata. Splits into metadata + tensors, uses async broadcast."""
```

### P2P Communication

```python
def send_object(self, obj: Any, dst: int) -> None:
    """Serialize + send Python object via CPU group."""

def recv_object(self, src: int) -> Any:
    """Receive + deserialize Python object via CPU group."""

def send(self, tensor, dst=None) -> None:
    """Blocking tensor send."""

def recv(self, size, dtype, src=None) -> torch.Tensor:
    """Blocking tensor receive."""

def send_tensor_dict(self, tensor_dict, dst=None, all_gather_group=None, all_gather_tensors=None) -> dict | None:
    """Send tensor dict with optional all-gather optimization."""

def recv_tensor_dict(self, src=None, all_gather_group=None, all_gather_tensors=None) -> dict | None:
    """Receive tensor dict with optional all-gather reconstruction."""

def isend_tensor_dict(self, tensor_dict, dst=None, all_gather_group=None, all_gather_tensors=None) -> list[Handle]:
    """Non-blocking send tensor dict."""

def irecv_tensor_dict(self, src=None, all_gather_group=None, all_gather_tensors=None) -> tuple[dict, list[Handle], list[Callable]]:
    """Non-blocking receive tensor dict."""
```

### MoE Dispatch/Combine Methods

```python
def dispatch_router_logits(self, hidden_states, router_logits, is_sequence_parallel=False, extra_tensors=None):
    """Dispatch router logits for MoE expert parallelism."""

def dispatch(self, hidden_states, topk_weights, topk_ids, is_sequence_parallel=False, extra_tensors=None):
    """Dispatch tokens to experts."""

def combine(self, hidden_states, is_sequence_parallel=False) -> torch.Tensor:
    """Combine expert outputs."""
```

### Other Methods

```python
def barrier(self):
    """Barrier via CPU group (NCCL barrier is unreliable)."""

def graph_capture(self, context=None):
    """Context manager for CUDA graph capture with CA/AITER support."""

def destroy(self):
    """Destroy process groups, communicators, and MQ broadcasters."""

def prepare_communication_buffer_for_model(self, model):
    """Initialize MoE communication buffers."""
```

---

## 4. Communication Operations

**File:** `vllm/distributed/communication_op.py`

High-level convenience functions that delegate to the TP group:

```python
def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce across the TP group."""

def tensor_model_parallel_all_gather(input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """All-gather across the TP group."""

def tensor_model_parallel_reduce_scatter(input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Reduce-scatter across the TP group."""

def tensor_model_parallel_gather(input_: torch.Tensor, dst: int = 0, dim: int = -1) -> torch.Tensor | None:
    """Gather to dst rank across the TP group."""

def broadcast_tensor_dict(tensor_dict=None, src: int = 0) -> dict | None:
    """Broadcast tensor dict across the TP group."""
```

---

## 5. Device Communicators

**File:** `vllm/distributed/device_communicators/`

### DeviceCommunicatorBase

**File:** `vllm/distributed/device_communicators/base_device_communicator.py`

```python
class DeviceCommunicatorBase:
    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
        global_ranks: list[int] | None = None,
        global_world_size: int | None = None,
    ):
        self.device = device or torch.device("cpu")
        self.cpu_group = cpu_group
        self.device_group = device_group
        self.unique_name = unique_name
        self.rank: int
        self.world_size: int
        self.ranks: list[int]
        self.is_ep_communicator: bool  # True if group name starts with "ep"
        self.use_all2all: bool          # True for EP with DP > 1
        self.all2all_manager: All2AllManagerBase | None
```

#### Methods

```python
def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
    """Default: dist.all_reduce on device_group."""

def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Default: concat-style all-gather via dist.all_gather_into_tensor."""

def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Default: reduce_scatter_tensor with dim handling."""

def gather(self, input_, dst=0, dim=-1) -> torch.Tensor | None:
    """Default: torch.distributed.gather."""

def send(self, tensor, dst=None) -> None:
    """Blocking send."""

def recv(self, size, dtype, src=None) -> torch.Tensor:
    """Blocking receive."""

def broadcast(self, tensor, src=0) -> torch.Tensor:
    """Broadcast tensor."""

def dispatch(self, hidden_states, topk_weights, topk_ids, is_sequence_parallel=False, extra_tensors=None):
    """MoE dispatch (no-op in base class)."""

def combine(self, hidden_states, is_sequence_parallel=False) -> torch.Tensor:
    """MoE combine (no-op in base class)."""

def prepare_communication_buffer_for_model(self, model) -> None:
    """Initialize MoE modular kernels for all MoE layers."""

def destroy(self): pass
def all_gatherv(self, input_, dim=0, sizes=None): raise NotImplementedError
def reduce_scatterv(self, input_, dim=-1, sizes=None): raise NotImplementedError
def batch_isend_irecv(self, p2p_ops): raise NotImplementedError
```

### All2AllManagerBase

```python
class All2AllManagerBase:
    rank: int
    world_size: int
    dp_group: GroupCoordinator
    tp_group: GroupCoordinator
    internode: bool

    def __init__(self, cpu_group, tcp_store_group=None): ...
    def get_handle(self, kwargs): raise NotImplementedError
    def dispatch(self, hidden_states, topk_weights, topk_ids, ...): raise NotImplementedError
    def combine(self, hidden_states, is_sequence_parallel=False): raise NotImplementedError
    def destroy(self): pass
```

### Concrete Communicator Implementations

| File | Class | Platform | Description |
|---|---|---|---|
| `cuda_communicator.py` | `CudaCommunicator` | CUDA | NCCL + custom all-reduce + CA (CollectiveArgs) |
| `cpu_communicator.py` | `CpuCommunicator` | CPU | Gloo-based communicator |
| `xpu_communicator.py` | `XpuCommunicator` | XPU | Intel XPU communicator |
| `pynccl.py` | `PyNcclCommunicator` | CUDA | Direct NCCL wrapper with send/recv |
| `custom_all_reduce.py` | `CustomAllreduce` | CUDA | Custom all-reduce via shared memory |
| `shm_broadcast.py` | `MessageQueue` | Any | Shared memory broadcast for IPC |
| `symm_mem.py` | - | CUDA | Symmetric memory operations |
| `all2all.py` | Various | CUDA | MoE all-to-all dispatch/combine |
| `flashinfer_all_reduce.py` | - | CUDA | FlashInfer-based all-reduce |
| `ray_communicator.py` | - | Ray | Ray-based communicator |

---

## 6. Executor Architecture

**File:** `vllm/v1/executor/abstract.py`

### Executor (ABC)

```python
class Executor(ABC):
    uses_ray: bool = False
    supports_pp: bool = False

    @staticmethod
    def get_class(vllm_config: VllmConfig) -> type["Executor"]:
        """Resolve executor class from ParallelConfig.distributed_executor_backend.

        Selection logic:
        - type[Executor] subclass -> use directly
        - "ray" -> RayDistributedExecutor (or RayExecutorV2 if VLLM_USE_RAY_V2_EXECUTOR_BACKEND)
        - "mp" -> MultiprocExecutor
        - "uni" -> UniProcExecutor
        - "external_launcher" -> ExecutorWithExternalLauncher
        - str -> resolve by qualified name
        """

    def __init__(self, vllm_config: VllmConfig):
        """Stores all config references and calls _init_executor()."""

    @abstractmethod
    def _init_executor(self) -> None: ...

    def initialize_from_config(self, kv_cache_configs: list[KVCacheConfig]) -> None:
        """Initialize KV caches and compile/warm up model across workers."""

    def determine_available_memory(self) -> list[int]:
        """Profile available memory across workers."""

    def get_kv_cache_specs(self) -> list[dict[str, KVCacheSpec]]:
        """Get KV cache specs from workers."""

    @abstractmethod
    def collective_rpc(self, method, timeout=None, args=(), kwargs=None, non_block=False) -> list | Future:
        """Execute RPC on all workers."""

    def execute_model(self, scheduler_output, non_block=False) -> ModelRunnerOutput | None | Future:
        """Execute one model step."""

    def sample_tokens(self, grammar_output, non_block=False) -> ModelRunnerOutput | Future:
        """Sample tokens from grammar output."""

    def check_health(self) -> None: ...
    def shutdown(self) -> None: ...
    def profile(self, is_start=True, profile_prefix=None): ...
    def sleep(self, level=1) / wake_up(self, tags=None): ...
    def add_lora/remove_lora/pin_lora/list_loras(self): ...
    def init_kv_output_aggregator(self, connector): ...

    @classmethod
    def supports_async_scheduling(cls) -> bool:
        return False
```

### Executor Directory Structure

| File | Class | Description |
|---|---|---|
| `abstract.py` | `Executor` | Abstract base class |
| `uniproc_executor.py` | `UniProcExecutor` | Single-process executor |
| `uniproc_executor.py` | `ExecutorWithExternalLauncher` | SPMD with torchrun |
| `multiproc_executor.py` | `MultiprocExecutor` | Multiprocessing executor |
| `ray_executor.py` | `RayDistributedExecutor` | Ray-based distributed executor |

---

## 7. UniProcExecutor

**File:** `vllm/v1/executor/uniproc_executor.py`

```python
class UniProcExecutor(Executor):
    """Single-process executor for single-GPU or SPMD inference."""
```

### Key Methods

```python
def _init_executor(self) -> None:
    """Creates WorkerWrapperBase, init_worker, init_device, load_model.
    Sets up async output thread if async scheduling enabled."""

def _distributed_args(self) -> tuple[str, int, int]:
    """Returns (distributed_init_method, rank=0, local_rank)."""

def collective_rpc(self, method, timeout=None, args=(), kwargs=None, non_block=False, single_value=False) -> Any:
    """Runs method directly on driver_worker. Returns [result] or Future."""

def execute_model(self, scheduler_output, non_block=False) -> ModelRunnerOutput | None | Future:
    """Delegates to collective_rpc('execute_model')."""

def sample_tokens(self, grammar_output, non_block=False): ...
def take_draft_token_ids(self) -> DraftTokenIds | None: ...
def check_health(self) -> None: return  # Always healthy
def shutdown(self) -> None: ...

@classmethod
def supports_async_scheduling(cls) -> bool: return True
```

### ExecutorWithExternalLauncher

```python
class ExecutorWithExternalLauncher(UniProcExecutor):
    """SPMD executor for torchrun-compatible launchers.
    Each engine has one worker; multiple engines coordinate via env://.
    Scheduling must be deterministic across ranks."""

    def _init_executor(self) -> None:
        """Asserts VLLM_ENABLE_V1_MULTIPROCESSING=0, calls super."""

    def _distributed_args(self) -> tuple[str, int, int]:
        """Reads RANK and LOCAL_RANK from environment."""

    def determine_available_memory(self) -> list[int]:
        """Gets min memory across all ranks via all-reduce."""
```

---

## 8. MultiprocExecutor

**File:** `vllm/v1/executor/multiproc_executor.py`

```python
class MultiprocExecutor(Executor):
    supports_pp: bool = True
```

### Key Methods

```python
def _init_executor(self) -> None:
    """Full initialization:
    1. Compute parallel sizes (TP, PP, PCP)
    2. Set multiprocessing env vars
    3. Create shared worker lock
    4. For leader node (node_rank_within_dp == 0):
       - Create broadcast MessageQueue for scheduler outputs
    5. Spawn WorkerProc processes for each local rank
    6. Wait for all workers to be READY
    7. Start worker monitor thread
    8. Set up response MessageQueues (local + remote)
    9. Wait for all MQs to be ready
    """

def _get_parallel_sizes(self) -> tuple[int, int, int]:
    """Returns (tp_size, pp_size, pcp_size). Computes local_world_size."""

def _get_output_rank(self) -> int:
    """Returns world_size - tp_size * pcp_size (last PP stage, TP rank 0)."""

def execute_model(self, scheduler_output, non_block=False) -> ModelRunnerOutput | None | Future:
    """Broadcasts scheduler_output to all workers via MQ, collects response."""

def collective_rpc(self, method, timeout=None, args=(), kwargs=None, non_block=False, unique_reply_rank=None, kv_output_aggregator=None) -> Any:
    """Enqueues RPC to broadcast MQ, dequeues response from worker MQs.
    Returns single result if unique_reply_rank or kv_output_aggregator provided."""

def start_worker_monitor(self, inline=False) -> None:
    """Monitors worker liveness via process sentinels."""

def register_failure_callback(self, callback): ...
def shutdown(self): ...
def check_health(self) -> None: ...
```

### WorkerProc

```python
class WorkerProc:
    """Runs one Worker in a separate process."""

    READY_STR = "READY"

    def __init__(self, vllm_config, local_rank, rank, distributed_init_method, input_shm_handle, shared_worker_lock, is_driver_worker):
        """Initialize worker:
        1. Create WorkerWrapperBase and init_worker
        2. init_device (sets up distributed env)
        3. load_model
        4. Set up message queues (local or remote)
        5. Enable env cache
        """

    @staticmethod
    def make_worker_process(vllm_config, local_rank, rank, ..., inherited_fds=None) -> UnreadyWorkerProcHandle:
        """Spawn a worker process with ready/death pipes."""

    @staticmethod
    def worker_main(*args, **kwargs):
        """Worker entry point:
        1. Set signal handlers (SIGTERM, SIGINT)
        2. Initialize WorkerProc
        3. Send READY + response MQ handles
        4. Wait for MQs to be ready
        5. Enter worker_busy_loop
        """

    def worker_busy_loop(self):
        """Main loop:
        while True:
            method, args, kwargs, output_rank = rpc_broadcast_mq.dequeue()
            output = getattr(worker, method)(*args, **kwargs)
            if output_rank is None or rank == output_rank:
                handle_output(output)
        """

    def handle_output(self, output):
        """Route to async queue or enqueue directly."""

    def enqueue_output(self, output):
        """Convert to (SUCCESS/FAILURE, result) and enqueue to response MQ."""
```

### FutureWrapper

```python
class FutureWrapper(Future):
    """Future that drains queue in order."""
    def __init__(self, futures_queue, get_response, aggregate): ...
    def result(self, timeout=None): ...
```

### Helper Data Classes

```python
@dataclass
class UnreadyWorkerProcHandle:
    proc: BaseProcess
    rank: int
    ready_pipe: Connection
    death_writer: Connection | None = None

@dataclass
class WorkerProcHandle:
    proc: BaseProcess
    rank: int
    worker_response_mq: MessageQueue | None
    peer_worker_response_mqs: list[MessageQueue | None]
    death_writer: Connection | None = None
```

---

## 9. Worker Architecture

**File:** `vllm/v1/worker/worker_base.py`

### WorkerBase (ABC)

```python
class WorkerBase:
    def __init__(self, vllm_config, local_rank, rank, distributed_init_method, is_driver_worker=False):
        """Stores all config, platform, rank info."""

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]: raise NotImplementedError
    def compile_or_warm_up_model(self) -> CompilationTimes: raise NotImplementedError
    def check_health(self) -> None: return
    def init_device(self) -> None: raise NotImplementedError
    def load_model(self, *, load_dummy_weights=False) -> None: raise NotImplementedError
    def execute_model(self, scheduler_output) -> ModelRunnerOutput | None: raise NotImplementedError
    def sample_tokens(self, grammar_output) -> ModelRunnerOutput: raise NotImplementedError
    def get_cache_block_size_bytes(self) -> int: raise NotImplementedError
    def add_lora/remove_lora/pin_lora/list_loras(self): raise NotImplementedError
    def shutdown(self) -> None: return
```

### WorkerWrapperBase

```python
class WorkerWrapperBase:
    """Lazily initializes a worker. Handles RPC rank mapping."""

    def __init__(self, rpc_rank: int = 0, global_rank: int | None = None):
        self.rpc_rank = rpc_rank
        self.global_rank = rpc_rank if global_rank is None else global_rank

    def init_worker(self, all_kwargs: list[dict]) -> None:
        """Resolves worker class from parallel_config.worker_cls.
        Injects worker_extension_cls via dynamic inheritance.
        Creates mm_receiver_cache if using SHM."""

    def initialize_from_config(self, kv_cache_configs): ...
    def init_device(self): ...
    def execute_model(self, scheduler_output): ...
    def shutdown(self): ...
    def __getattr__(self, attr): return getattr(self.worker, attr)
```

### Worker (GPU)

**File:** `vllm/v1/worker/gpu_worker.py`

```python
class Worker(WorkerBase):
    def __init__(self, vllm_config, local_rank, rank, distributed_init_method, is_driver_worker=False):
        """Sets float32 matmul precision, creates ElasticEPScalingExecutor,
        weight_transfer_engine, profiler config."""

    def init_device(self) -> None:
        """Full GPU initialization:
        1. Set CUDA device
        2. init_worker_distributed_environment (DP, TP, PP, PCP, DCP groups)
        3. Set random seed
        4. Memory snapshot
        5. Create GPUModelRunner
        """

    def load_model(self, *, load_dummy_weights=False) -> None:
        """Load model with optional memory pool context for sleep mode."""

    def determine_available_memory(self) -> int:
        """Profile peak memory and compute available KV cache memory."""

    def initialize_from_config(self, kv_cache_config) -> None:
        """Allocate KV caches, init KV transfer connector."""

    def compile_or_warm_up_model(self) -> CompilationTimes:
        """Compile model, warm up kernels, capture CUDA graphs."""

    def execute_model(self, scheduler_output) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        """Execute model with PP support:
        - Non-first PP rank: irecv_tensor_dict from previous stage
        - Non-last PP rank: isend_tensor_dict to next stage
        - Uses AsyncIntermediateTensors for lazy comm sync
        """

    def sleep(self, level=1) / wake_up(self, tags=None): ...
    def profile(self, is_start, profile_prefix): ...
    def shutdown(self) -> None: ...
```

### AsyncIntermediateTensors

```python
class AsyncIntermediateTensors(IntermediateTensors):
    """IntermediateTensors with lazy comm synchronization."""

    def __init__(self, tensors, comm_handles=None, comm_postprocess=None): ...

    def wait_for_comm(self) -> None:
        """Wait for all async handles and run postprocess callbacks."""

    def __getattribute__(self, name):
        """Ensure .tensors triggers wait_for_comm before use."""
```

### init_worker_distributed_environment

```python
def init_worker_distributed_environment(vllm_config, rank, distributed_init_method=None, local_rank=-1, backend="nccl") -> None:
    """1. Init batch invariance
    2. Override EPLB env vars
    3. Set custom all-reduce flag
    4. init_distributed_environment (NCCL)
    5. ensure_model_parallel_initialized (TP, PP, PCP, DCP groups)
    6. ensure_ec_transfer_initialized (encoder cache transfer)
    """
```

---

## 10. Tensor Parallelism (TP)

Tensor parallelism splits model weights and computations across multiple GPUs.

### Rank Layout

Ranks are arranged as `[ExternalDP, DP, PP, PCP, TP]`. TP groups are the innermost dimension:

```
Example: TP=2, PP=2, DP=2
Rank layout: [DP=0:[PP=0:[TP=0,TP=1], PP=1:[TP=0,TP=1]], DP=1:[...]]
TP groups: [0,1], [2,3], [4,5], [6,7]
```

### Communication Patterns

- **All-reduce**: After each linear layer with column or row parallelism (e.g., QKV proj, output proj)
- **All-gather**: For sequence parallelism (optional)
- **Reduce-scatter**: For sequence parallelism gradient reduction

### TP Group Features

- Uses `MessageQueue` broadcaster for fast CPU-level broadcasting
- Supports custom all-reduce kernels (bypasses NCCL)
- CUDA graph capture compatible via `graph_capture()` context manager

---

## 11. Pipeline Parallelism (PP)

Pipeline parallelism splits model layers across GPUs.

### Layer Distribution

```python
def get_pp_indices(num_hidden_layers: int, pp_rank: int, pp_size: int) -> tuple[int, int]:
    """Compute (start_layer, end_layer) for a PP rank.
    Balances layers, excluding the last PP rank which has output embedding."""
```

Environment variable `VLLM_PP_LAYER_PARTITION` allows manual override with comma-separated layer counts.

### PP Communication

Pipeline stages communicate via P2P tensor dict transfers:

```python
# In Worker.execute_model():
# Non-first PP rank receives from previous stage
tensor_dict, comm_handles, postprocess = get_pp_group().irecv_tensor_dict(
    all_gather_group=get_tp_group(),
    all_gather_tensors=all_gather_tensors,
)

# Non-last PP rank sends to next stage
get_pp_group().isend_tensor_dict(
    output.tensors,
    all_gather_group=get_tp_group(),
    all_gather_tensors=all_gather_tensors,
)
```

The `all_gather_group` and `all_gather_tensors` parameters enable an optimization where each TP rank only sends its shard, and the receiver reconstructs via all-gather.

---

## 12. Data Parallelism (DP)

Data parallelism runs multiple copies of the model in parallel, each processing different requests.

### DP Configuration

| Parameter | Default | Description |
|---|---|---|
| `data_parallel_size` | 1 | Number of DP replicas |
| `data_parallel_size_local` | 1 | Local DP replicas on same node |
| `data_parallel_backend` | "mp" | Backend: "mp" or "ray" |
| `data_parallel_external_lb` | False | External load balancer (K8s) |
| `data_parallel_hybrid_lb` | False | Hybrid LB (per-node) |

### DP Coordination

**File:** `vllm/v1/worker/dp_utils.py`

```python
def coordinate_batch_across_dp(
    num_tokens_unpadded: int,
    allow_microbatching: bool,
    parallel_config: ParallelConfig,
    num_tokens_padded: int | None = None,
    uniform_decode: bool | None = None,
    cudagraph_mode: int = 0,
) -> tuple[bool, torch.Tensor | None, int]:
    """Coordinate across DP ranks:
    1. All-reduce a 4-element tensor: [orig_tokens, padded_tokens, should_ubatch, cudagraph_mode]
    2. Synchronize cudagraph mode (take min)
    3. Decide if microbatching is viable
    4. Optionally pad all ranks to max token count

    Returns: (should_ubatch, num_tokens_after_padding, synced_cudagraph_mode)
    """
```

### DP State Synchronization

```python
@staticmethod
def ParallelConfig.sync_dp_state(dp_group, has_unfinished, pending_pause) -> tuple[bool, bool]:
    """Combined all-reduce for DP state.
    [0] = has_unfinished (SUM > 0 means OR)
    [1] = pending_pause (SUM == dp_size means consensus)
    """

@staticmethod
def ParallelConfig.has_unfinished_dp(dp_group, has_unfinished) -> bool:
    """OR reduce across DP ranks."""
```

### DP Load Balancing Modes

| Mode | Description |
|---|---|
| Internal | Default: API server round-robins across DP ranks |
| External (`data_parallel_external_lb`) | External LB (K8s) routes to specific DP ranks |
| Hybrid (`data_parallel_hybrid_lb`) | Per-node LB + external inter-node LB |

---

## 13. Expert Parallelism (EP) for MoE

Expert parallelism distributes MoE experts across GPUs.

### EP Group Formation

EP groups merge DP * PCP * TP dimensions per PP rank:

```python
group_ranks = all_ranks.transpose(1, 2).reshape(
    -1, data_parallel_size * prefill_context_model_parallel_size * tensor_model_parallel_size
).unbind(0)
```

### All-to-All Backends

| Backend | Description |
|---|---|
| `allgather_reducescatter` | Default: AllGather + ReduceScatter |
| `deepep_high_throughput` | DeepEP high-throughput kernels |
| `deepep_low_latency` | DeepEP low-latency kernels |
| `mori` | MoRI kernels |
| `nixl_ep` | NIXL-EP kernels |
| `flashinfer_nvlink_two_sided` | FlashInfer two-sided for MNNVL |
| `flashinfer_nvlink_one_sided` | FlashInfer one-sided high-throughput |

### Sequence Parallel MoE

When `use_sequence_parallel_moe` is True, input tokens to MoE are sequence-parallel (already scattered across TP ranks) to avoid redundant computation and communication.

### Expert Placement Strategies

| Strategy | Behavior |
|---|---|
| `linear` | Experts placed contiguously: rank 0 = [0,1], rank 1 = [2,3] |
| `round_robin` | Round-robin placement: rank 0 = [0,2], rank 1 = [1,3] |

---

## 14. Expert Parallelism Load Balancing (EPLB)

**Directory:** `vllm/distributed/eplb/`

### Files

| File | Description |
|---|---|
| `eplb_state.py` | EPLB state management |
| `eplb_communicator.py` | EPLB communication backend |
| `eplb_utils.py` | Utility functions |
| `async_worker.py` | Async EPLB worker |
| `rebalance_execute.py` | Expert rebalancing execution |
| `policy/` | Load balancing policies |

### Configuration

```python
# In ParallelConfig:
enable_eplb: bool = False
eplb_config: EPLBConfig = EPLBConfig()

# EPLBConfig:
window_size: int = 1000          # Recording window
step_interval: int = 3000        # Rearrange interval
num_redundant_experts: int = 0   # Redundant experts
log_balancedness: bool = False
use_async: bool = False          # Non-blocking EPLB
communicator: "torch_nccl" | "torch_gloo" | "nixl" | "pynccl" | None
```

### Communicator Auto-Selection

| Condition | Backend |
|---|---|
| Elastic EP | `pynccl` (stateless mode required) |
| Async EPLB | `torch_gloo` (avoids NCCL multi-thread conflicts) |
| Default (sync) | `torch_nccl` |

---

## 15. Context Parallelism (PCP / DCP)

### Prefill Context Parallelism (PCP)

Splits long prefill sequences across GPUs. The PCP group is formed from the PCP dimension of the rank layout.

### Decode Context Parallelism (DCP)

Reuses GPUs within the TP group by splitting one TP group into `tp_size / dcp_size` DCP groups. This means `dcp_size` must divide `tp_size`.

```python
# In ParallelConfig:
decode_context_parallel_size: int = 1
dcp_comm_backend: DCPCommBackend = "ag_rs"  # "ag_rs" or "a2a"
cp_kv_cache_interleave_size: int = 1
```

### DCP Communication Backends

| Backend | Description |
|---|---|
| `ag_rs` | AllGather + ReduceScatter (default, 3 NCCL calls per layer) |
| `a2a` | All-to-All + Triton kernel combine (2 NCCL calls per layer for MLA) |

### KV Cache Interleaving

`cp_kv_cache_interleave_size` controls how tokens are distributed across context parallel ranks:

| Value | Behavior |
|---|---|
| 1 | Token-level: token `i` on rank `i % total_cp_world_size` |
| block_size | Block-level: fill rank i completely before using rank i+1 |

---

## 16. KV Cache Transfer / Disaggregated Prefill

**Directory:** `vllm/distributed/kv_transfer/`

### Architecture

Disaggregated prefill separates prefill (compute-heavy) from decode (memory-bandwidth-heavy) into different engine instances. KV cache is transferred from prefill to decode workers via connectors.

### State Management

**File:** `vllm/distributed/kv_transfer/kv_transfer_state.py`

```python
_KV_CONNECTOR_AGENT: KVConnectorBaseType | None = None

def get_kv_transfer_group() -> KVConnectorBaseType:
    """Get the global KV connector."""

def has_kv_transfer_group() -> bool:
    """Check if KV transfer is initialized."""

def is_v1_kv_transfer_group(connector=None) -> bool:
    """Check if connector is V1 type."""

def ensure_kv_transfer_initialized(vllm_config, kv_cache_config=None) -> None:
    """Initialize KV connector from config if not already done."""

def ensure_kv_transfer_shutdown() -> None:
    """Shutdown KV connector."""
```

### KVOutputAggregator

**File:** `vllm/distributed/kv_transfer/kv_connector/utils.py`

```python
class KVOutputAggregator:
    """Aggregates outputs from all workers into a single output for the scheduler."""

    def __init__(self, expected_finished_count: int): ...

    @classmethod
    def from_connector(cls, connector, world_size) -> "KVOutputAggregator": ...

    def aggregate(self, outputs, output_rank=0) -> ModelRunnerOutput | None:
        """Aggregate:
        - finished_sending/finished_recving across workers
        - kv_connector_stats (aggregate method)
        - kv_connector_worker_meta (aggregate method)
        - kv_cache_events (merge)
        - invalid_block_ids (union)
        """
```

### TransferTopology

```python
@dataclass
class TransferTopology:
    """Single source of truth for local TP identity and per-engine remote info."""
    tp_rank: int
    tp_size: int
    block_size: int
    engine_id: EngineId
    is_mla: bool
    is_mamba: bool
    total_num_kv_heads: int
    attn_backends: list[type[AttentionBackend]]
    tensor_shape: torch.Size | None = None

    def register_remote_engine(self, remote_engine_id, info: EngineTransferInfo) -> EngineTransferInfo: ...
    def tp_ratio(self, remote_tp_size) -> int: ...
    def block_size_ratio(self, remote_block_size) -> int: ...
    def target_remote_ranks(self, remote_engine_id) -> list[int]: ...
    def handshake_target_ranks(self, remote_tp_size) -> list[int]: ...
    def get_transfer_cache_regions(self, cache, layer_spec) -> list[torch.Tensor] | torch.Tensor: ...
```

---

## 17. KV Connector Base (V1)

**File:** `vllm/distributed/kv_transfer/kv_connector/v1/base.py`

### Enums and Types

```python
class KVConnectorRole(enum.Enum):
    SCHEDULER = 0
    WORKER = 1

CopyBlocksOp = Callable[[dict, dict, list[int], list[int], Literal["h2d","d2h"]], None]
```

### Abstract Metadata Classes

```python
class KVConnectorHandshakeMetadata(ABC): pass        # OOB connector handshake
class KVConnectorMetadata(ABC): pass                   # Scheduler -> Worker
class KVConnectorWorkerMetadata(ABC):                  # Worker -> Scheduler
    @abstractmethod
    def aggregate(self, other) -> "KVConnectorWorkerMetadata": ...
```

### KVConnectorBase_V1 (ABC)

```python
class KVConnectorBase_V1(ABC):
    def __init__(self, vllm_config, role: KVConnectorRole, kv_cache_config=None): ...

    @property
    def prefer_cross_layer_blocks(self) -> bool: return False
    @property
    def role(self) -> KVConnectorRole: ...
```

#### Worker-Side Methods

```python
def bind_connector_metadata(self, connector_metadata) -> None: ...
def clear_connector_metadata(self) -> None: ...
def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]): ...
def register_cross_layers_kv_cache(self, kv_cache, attn_backend): ...
def set_host_xfer_buffer_ops(self, copy_operation): ...
def handle_preemptions(self, kv_connector_metadata): ...

@abstractmethod
def start_load_kv(self, forward_context, **kwargs) -> None:
    """Start loading KV from connector to paged buffer (may be async)."""

@abstractmethod
def wait_for_layer_load(self, layer_name: str) -> None:
    """Block until KV for a specific layer is loaded."""

@abstractmethod
def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs) -> None:
    """Start saving KV from paged buffer to connector (may be async)."""

@abstractmethod
def wait_for_save(self) -> None:
    """Block until all saves are complete."""

def get_finished(self, finished_req_ids) -> tuple[set | None, set | None]: ...
def get_block_ids_with_load_errors(self) -> set[int]: ...
def shutdown(self): ...
def get_kv_connector_stats(self): ...
def get_kv_connector_kv_cache_events(self): ...
def get_handshake_metadata(self) -> KVConnectorHandshakeMetadata | None: ...
def build_connector_worker_meta(self) -> KVConnectorWorkerMetadata | None: ...
```

#### Scheduler-Side Methods

```python
@abstractmethod
def get_num_new_matched_tokens(self, request, num_computed_tokens) -> tuple[int | None, bool]:
    """Get number of new tokens loadable from external KV cache.
    Returns (num_tokens, is_async)."""

@abstractmethod
def update_state_after_alloc(self, request, blocks, num_external_tokens): ...
    """Update state after block allocation."""

@abstractmethod
def build_connector_meta(self, scheduler_output) -> KVConnectorMetadata:
    """Build metadata for this step. Resets connector state."""

def update_connector_output(self, connector_output): ...
def request_finished(self, request, block_ids) -> tuple[bool, dict | None]:
    """Called once when request finishes. Returns (save_async, kv_transfer_params)."""

def take_events(self) -> Iterable[KVCacheEvent]: ...
```

### SupportsHMA Mixin

```python
class SupportsHMA(ABC):
    """Indicates connector supports Hybrid Memory Allocator."""

    @abstractmethod
    def request_finished_all_groups(self, request, block_ids) -> tuple[bool, dict | None]: ...
```

---

## 18. KV Connector Implementations

### KVConnectorFactory

**File:** `vllm/distributed/kv_transfer/kv_connector/factory.py`

```python
class KVConnectorFactory:
    _registry: dict[str, Callable[[], type[KVConnectorBase]]] = {}

    @classmethod
    def register_connector(cls, name, module_path, class_name): ...

    @classmethod
    def create_connector(cls, config, role, kv_cache_config=None) -> KVConnectorBase:
        """Create connector from KVTransferConfig.
        Handles deprecated 2-arg signature via introspection."""

    @classmethod
    def get_connector_class(cls, kv_transfer_config) -> type[KVConnectorBaseType]: ...
```

### Registered Connectors

| Name | Module Path | Description |
|---|---|---|
| `ExampleConnector` | `v1.example_connector` | Example/reference implementation |
| `ExampleHiddenStatesConnector` | `v1.example_hidden_states_connector` | Hidden states transfer example |
| `P2pNcclConnector` | `v1.p2p.p2p_nccl_connector` | Direct P2P NCCL KV transfer |
| `LMCacheConnectorV1` | `v1.lmcache_connector` | LMCache integration |
| `LMCacheMPConnector` | `v1.lmcache_mp_connector` | LMCache multi-process |
| `NixlConnector` | `v1.nixl` | NIXL-based high-performance transfer |
| `MultiConnector` | `v1.multi_connector` | Composite of multiple connectors |
| `MoRIIOConnector` | `v1.moriio.moriio_connector` | MoRI I/O connector |
| `OffloadingConnector` | `v1.offloading_connector` | CPU offloading connector |
| `DecodeBenchConnector` | `v1.decode_bench_connector` | Decode benchmarking connector |
| `MooncakeConnector` | `v1.mooncake.mooncake_connector` | Mooncake transfer |
| `FlexKVConnectorV1` | `v1.flexkv_connector` | FlexKV connector |
| `SimpleCPUOffloadConnector` | `v1.simple_cpu_offload_connector` | Simple CPU offload |
| `HF3FSKVConnector` | `v1.hf3fs.hf3fs_connector` | HuggingFace 3FS connector |

### NIXL Connector

**Directory:** `vllm/distributed/kv_transfer/kv_connector/v1/nixl/`

| File | Description |
|---|---|
| `connector.py` | NIXL connector implementation |
| `worker.py` | NIXL worker-side logic |
| `scheduler.py` | NIXL scheduler-side logic |
| `metadata.py` | NIXL transfer metadata |
| `tp_mapping.py` | TP rank mapping for transfers |
| `stats.py` | Transfer statistics |
| `utils.py` | Utility functions |

### P2P NCCL Connector

**Directory:** `vllm/distributed/kv_transfer/kv_connector/v1/p2p/`

| File | Description |
|---|---|
| `p2p_nccl_connector.py` | P2P NCCL connector |
| `p2p_nccl_engine.py` | NCCL send/recv engine |
| `tensor_memory_pool.py` | Memory pool for tensor transfers |

### Offloading Connector

**Directory:** `vllm/distributed/kv_transfer/kv_connector/v1/offloading/`

| File | Description |
|---|---|
| `offloading_connector.py` | CPU offloading connector |
| `common.py` | Common offloading utilities |
| `scheduler.py` | Offload scheduler |
| `worker.py` | Offload worker |
| `metrics.py` | Offloading metrics |

### Utility Functions

**File:** `vllm/distributed/kv_transfer/kv_connector/utils.py`

```python
def copy_kv_blocks(src_kv_caches, dst_kv_caches, src_block_ids, dst_block_ids, direction: Literal["h2d","d2h"]) -> None:
    """Copy KV blocks between host and device buffers."""

def kv_postprocess_blksize_on_receive(cache, indices, block_size_ratio):
    """Transform block size layout on receive (local > remote)."""

def kv_postprocess_layout_on_receive(cache, indices):
    """Transform HND <-> NHD layout on receive."""

def kv_postprocess_blksize_and_layout_on_receive(cache, indices, block_size_ratio):
    """Combined block size + layout transformation."""

def yield_req_data(scheduler_output) -> Iterator[tuple[str, tuple[list[int],...] | None, bool]]:
    """Yield (req_id, new_block_ids, preempted) from scheduler output."""

def get_current_attn_backends(vllm_config, layer_names=None) -> list[type[AttentionBackend]]:
    """Get all distinct attention backends for given layers."""
```

---

## 19. Stateless Process Groups

**File:** `vllm/distributed/utils.py`

### StatelessProcessGroup

```python
@dataclasses.dataclass
class StatelessProcessGroup:
    """Metadata-only process group using TCPStore. No global state pollution."""

    rank: int
    world_size: int
    store: torch._C._distributed_c10d.Store
    data_expiration_seconds: int = 3600

    # Counters
    send_dst_counter: dict[int, int]
    recv_src_counter: dict[int, int]
    broadcast_send_counter: int
    broadcast_recv_src_counter: dict[int, int]
    entries: deque[tuple[str, float]]

    @staticmethod
    def create(host, port, rank, world_size, data_expiration_seconds=3600, store_timeout=300, listen_socket=None) -> "StatelessProcessGroup":
        """Create a stateless process group using TCPStore.
        Unlike torch.distributed.init_process_group, this does not pollute global state."""

    def send_obj(self, obj, dst): ...
    def recv_obj(self, src) -> Any: ...
    def broadcast_obj(self, obj, src) -> Any: ...
    def all_gather_obj(self, obj) -> list[Any]: ...
    def broadcast(self, tensor, src) -> torch.Tensor: ...
    def send(self, tensor, dst): ...
    def recv(self, tensor, src) -> torch.Tensor: ...
    def all_reduce(self, tensor, op=ReduceOp.SUM) -> torch.Tensor: ...
    def barrier(self, timeout=30.0): ...
    def expire_data(self): ...
```

### StatelessGroupCoordinator

**File:** `vllm/distributed/stateless_coordinator.py`

```python
class StatelessGroupCoordinator(GroupCoordinator):
    """Stateless version of GroupCoordinator. Creates CPU, device, and TCPStore
    groups independent of PyTorch's WORLD group."""

    def __init__(self, group_ranks, local_rank, torch_distributed_backend, use_device_communicator, coord_store, use_message_queue_broadcaster=False, group_name=None, host="127.0.0.1", global_rank=0, global_world_size=1):
        """Creates stateless device (NCCL), CPU (Gloo), and TCPStore groups.
        Uses _allocate_group_ports for rank-0 port publishing."""

    # Overridden methods using TCPStore for metadata and NCCL for data:
    def broadcast(self, input_, src=0): ...
    def broadcast_object(self, obj=None, src=0): ...
    def broadcast_object_list(self, obj_list, src=0): ...
    def broadcast_tensor_dict(self, tensor_dict=None, src=0, ...): ...
    def send_object(self, obj, dst): ...
    def recv_object(self, src): ...
    def send_tensor_dict(self, tensor_dict, dst=None, ...): ...
    def recv_tensor_dict(self, src=None, ...): ...
    def barrier(self): ...
    def gather(self, input_, dst=0, dim=-1): ...
    def destroy(self): ...
```

### Utility Functions

```python
def stateless_init_torch_distributed_process_group(host, port, rank, world_size, backend, group_name=None, return_store=False, listen_socket=None) -> ProcessGroup | tuple[ProcessGroup, Store]:
    """Init ProcessGroup without global state pollution.
    Supports NCCL and Gloo backends. Uses PrefixStore for isolation."""

def stateless_destroy_torch_distributed_process_group(pg) -> None:
    """Destroy a stateless ProcessGroup."""

def get_cached_tcp_store_client(host, port) -> TCPStore:
    """Cached TCPStore client (LRU, maxsize=1)."""

def create_tcp_store(host, port, listen_socket=None, **kwargs) -> TCPStore:
    """Create TCPStore, optionally taking ownership of listen_socket."""

def init_gloo_process_group(prefix_store, group_rank, group_size, timeout) -> ProcessGroup:
    """Stateless Gloo ProcessGroup init."""
```

---

## 20. Elastic Expert Parallelism

**Directory:** `vllm/distributed/elastic_ep/`

### Files

| File | Description |
|---|---|
| `elastic_execute.py` | `ElasticEPScalingExecutor` - handles scale up/down |
| `elastic_state.py` | Elastic EP state management |
| `standby_state.py` | Standby worker state |
| `__init__.py` | Module init |

### Configuration

```python
# In ParallelConfig:
enable_elastic_ep: bool = False
```

Requirements:
- `enable_eplb` must be True
- `pipeline_parallel_size` must be 1
- Not compatible with `data_parallel_external_lb` or `data_parallel_hybrid_lb`
- Uses `StatelessGroupCoordinator` for DP/EP groups

---

## 21. Distributed KV Events

**File:** `vllm/distributed/kv_events.py`

### Event Types

```python
class KVCacheEvent(msgspec.Struct, tag=True):
    """Base class for KV cache events."""

class BlockStored(KVCacheEvent):
    block_hashes: list[ExternalBlockHash]
    parent_block_hash: ExternalBlockHash | None
    token_ids: list[int]
    block_size: int
    lora_id: int | None
    medium: str | None         # "GPU" or None
    lora_name: str | None
    extra_keys: list[tuple | None] | None
    group_idx: int | None

class BlockRemoved(KVCacheEvent):
    block_hashes: list[ExternalBlockHash]
    medium: str | None
    group_idx: int | None

class AllBlocksCleared(KVCacheEvent):
    pass
```

### Event Aggregation

```python
class KVEventAggregator:
    """Tracks events across workers, returns those emitted by all."""
    def __init__(self, num_workers: int): ...
    def add_events(self, events: list[KVCacheEvent]) -> None: ...
    def get_common_events(self) -> list[KVCacheEvent]: ...  # Appeared in all workers
    def get_all_events(self) -> list[KVCacheEvent]: ...
    def clear_events(self) -> None: ...
    def increment_workers(self, count=1) -> None: ...
```

### Event Publishing

```python
class EventPublisher(ABC):
    """Lightweight publisher with DP rank support."""
    def publish(self, events: EventBatch) -> None: ...
    def shutdown(self) -> None: ...

class NullEventPublisher(EventPublisher):
    """No-op implementation."""

class ZmqEventPublisher(EventPublisher):
    """ZMQ PUB/ROUTER publisher with replay buffer.
    Supports at-least-once delivery and monotonic ordering."""

    def __init__(self, data_parallel_rank, endpoint="tcp://*:5557", replay_endpoint=None, buffer_steps=10000, hwm=100000, max_queue_size=100000, topic=""): ...

class EventPublisherFactory:
    _registry = {"null": NullEventPublisher, "zmq": ZmqEventPublisher}

    @classmethod
    def create(cls, config, data_parallel_rank=0) -> EventPublisher: ...
```

---

## 22. DP Coordination Utilities

**File:** `vllm/v1/worker/dp_utils.py`

```python
def coordinate_batch_across_dp(
    num_tokens_unpadded: int,
    allow_microbatching: bool,
    parallel_config: ParallelConfig,
    num_tokens_padded: int | None = None,
    uniform_decode: bool | None = None,
    cudagraph_mode: int = 0,
) -> tuple[bool, torch.Tensor | None, int]:
    """Main entry point for DP batch coordination.

    Algorithm:
    1. If DP size is 1, return early
    2. If microbatching allowed, check thresholds
    3. All-reduce [orig_tokens, padded_tokens, should_ubatch, cudagraph_mode] across DP ranks
    4. Sync cudagraph mode (min across ranks)
    5. Check microbatching viability
    6. Optionally pad all ranks to max token count

    Returns:
        (should_ubatch, num_tokens_after_padding_per_rank, synced_cudagraph_mode)
    """
```

### Helper Functions

```python
def _get_device_and_group(parallel_config) -> tuple[str, ProcessGroup]:
    """Get device and group for DP sync. Falls back to CPU if NCCL disabled."""

def _run_ar(should_ubatch, orig_num_tokens, padded_num_tokens, cudagraph_mode, parallel_config) -> torch.Tensor:
    """All-reduce 4-element tensor across DP ranks."""

def _post_process_ubatch(tensor, num_ubatches) -> bool:
    """Check if all DP ranks agree to microbatch and if second ubatch would be empty."""

def _post_process_dp_padding(tensor, should_dp_pad) -> torch.Tensor:
    """Pad all ranks to max tokens if needed."""

def _post_process_cudagraph_mode(tensor) -> int:
    """Sync cudagraph mode (min across ranks)."""
```

---

## 23. All-to-All Communication (MoE)

**Directory:** `vllm/distributed/device_communicators/all2all.py`

The all-to-all communication for MoE expert parallelism is handled by `All2AllManagerBase` subclasses created within the `DeviceCommunicatorBase`. The dispatch and combine operations route tokens to/from the appropriate expert GPUs.

### Dispatch/Combine Flow

1. **Dispatch**: Tokens are sent to the GPUs holding their assigned experts
   - Input: hidden_states, topk_weights, topk_ids
   - Output: received_hidden_states, received_topk_weights, received_topk_ids

2. **Expert Computation**: Each GPU computes its local experts

3. **Combine**: Expert outputs are sent back to the originating GPUs
   - Input: hidden_states (expert outputs)
   - Output: combined_hidden_states

### Backend-Specific Implementations

The `all2all_backend` config determines which implementation is used:

| Backend | Class | Key Feature |
|---|---|---|
| `allgather_reducescatter` | `AllGatherReduceScatter` | Standard fallback |
| `deepep_high_throughput` | `DeepEP` | High throughput mode |
| `deepep_low_latency` | `DeepEPLowLatency` | Low latency mode |
| `mori` | `MoRI` | MoRI I/O |
| `nixl_ep` | `NIXLEP` | NIXL-based EP |

---

## 24. Shared Memory Broadcast

**File:** `vllm/distributed/device_communicators/shm_broadcast.py`

The `MessageQueue` class provides fast inter-process communication via shared memory, used for:

1. **RPC Broadcast**: SchedulerOutput broadcast from executor to all workers
2. **Response Collection**: Worker outputs returned to executor
3. **Object Broadcasting**: Fast CPU-level broadcast within TP groups

### Key Features

- Zero-copy serialization via pickle/cloudpickle
- Chunked transfer for large payloads (`VLLM_MQ_MAX_CHUNK_BYTES_MB`)
- Support for local (single-node) and remote (multi-node) communication
- Thread-safe with blocking and non-blocking operations

---

## 25. Weight Transfer

**Directory:** `vllm/distributed/weight_transfer/`

Weight transfer enables dynamic weight updates from a trainer to running vLLM workers, supporting:

- Live model weight updates without restart
- NCCL-based broadcast for weight distribution
- Checkpoint and kernel format support

### Worker Methods

```python
# In Worker class:
def init_weight_transfer_engine(self, init_info: dict) -> None:
    """Initialize weight transfer engine (NCCL, etc.)."""

def update_weights(self, update_info: dict) -> None:
    """Batched weight update. Supports checkpoint and kernel formats."""
```

---

## 26. NIXL Utilities

**File:** `vllm/distributed/nixl_utils.py`

NIXL (NVIDIA Inference Exchange Library) provides high-performance memory transfer capabilities for:

- KV cache transfer between prefill and decode workers
- Expert weight transfer for EPLB
- All-to-all communication for EP

---

## Appendix: Rank Layout Diagram

For a configuration with `TP=2, PP=2, DP=2, PCP=1`, the rank layout is:

```
all_ranks shape: [ExternalDP=1, DP=2, PP=2, PCP=1, TP=2] = 8 ranks

Rank indices:
  DP=0, PP=0, PCP=0, TP=0  -> rank 0
  DP=0, PP=0, PCP=0, TP=1  -> rank 1
  DP=0, PP=1, PCP=0, TP=0  -> rank 2
  DP=0, PP=1, PCP=0, TP=1  -> rank 3
  DP=1, PP=0, PCP=0, TP=0  -> rank 4
  DP=1, PP=0, PCP=0, TP=1  -> rank 5
  DP=1, PP=1, PCP=0, TP=0  -> rank 6
  DP=1, PP=1, PCP=0, TP=1  -> rank 7

TP groups:  [0,1], [2,3], [4,5], [6,7]
PP groups:  [0,2], [1,3], [4,6], [5,7]
DP groups:  [0,4], [1,5], [2,6], [3,7]
EP groups:  [0,1,4,5], [2,3,6,7]  (MoE: DP*TP per PP rank)
```

## Appendix: Initialization Order

```
1. Executor created (UniProc / Multiproc / Ray)
2. Worker processes spawned (for Multiproc/Ray)
3. init_distributed_environment()
   - torch.distributed.init_process_group()
   - Create WORLD group
4. ensure_model_parallel_initialized()
   - initialize_model_parallel()
     - Create TP group (with MQ broadcaster)
     - Create DCP group
     - Create PCP group
     - Create PP group
     - Create DP group
     - Create EP group (MoE only)
     - Create EPLB group (if EPLB enabled)
5. ensure_kv_transfer_initialized()
   - KVConnectorFactory.create_connector()
6. Model loaded
7. KV cache allocated
8. Model compiled / CUDA graph captured
9. Workers enter busy loop (Multiproc)
```
