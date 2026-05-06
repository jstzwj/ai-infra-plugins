# vLLM Compilation & CUDA Graphs Reference

This document provides comprehensive coverage of vLLM's compilation system and CUDA graph
capture/replay infrastructure. The compilation system integrates with PyTorch's `torch.compile`
through custom backends, piecewise compilation, Inductor passes, and CUDA graph management.

---

## Table of Contents

1. [Compilation Configuration](#1-compilation-configuration)
   - [CompilationMode Enum](#compilationmode-enum)
   - [CUDAGraphMode Enum](#cudagraphmode-enum)
   - [DynamicShapesType Enum](#dynamicshapestype-enum)
   - [DynamicShapesConfig](#dynamicshapesconfig)
   - [PassConfig](#passconfig)
   - [CompilationConfig](#compilationconfig)
2. [Compiler Interface & Adaptors](#2-compiler-interface--adaptors)
   - [CompilerInterface ABC](#compilerinterface-abc)
   - [InductorStandaloneAdaptor](#inductorstandaloneadaptor)
   - [InductorAdaptor](#inductoradaptor)
   - [EagerAdaptor](#eageradaptor)
   - [AlwaysHitShapeEnv](#alwayshitshapeenv)
   - [Helper Functions](#helper-functions)
3. [VllmBackend & Graph Splitting](#3-vllmbackend--graph-splitting)
   - [VllmBackend Class](#vllmbackend-class)
   - [CompilerManager](#compilermanager)
   - [SplitItem Dataclass](#splititem-dataclass)
   - [split_graph Function](#split_graph-function)
   - [PiecewiseCompileInterpreter](#piecewisecompileinterpreter)
   - [wrap_with_cudagraph_if_needed](#wrap_with_cudagraph_if_needed)
   - [Helper Functions](#helper-functions-1)
4. [Piecewise Backend](#4-piecewise-backend)
   - [RangeEntry Dataclass](#rangeentry-dataclass)
   - [PiecewiseBackend Class](#piecewisebackend-class)
   - [create_concrete_args](#create_concrete_args)
   - [get_fake_args_from_graph](#get_fake_args_from_graph)
5. [Code Generation](#5-code-generation)
   - [generate_execution_code_with_name](#generate_execution_code_with_name)
   - [generate_execution_code](#generate_execution_code)
   - [compile_execution_fn](#compile_execution_fn)
   - [_node_ref](#_node_ref)
6. [CUDA Graph System](#6-cuda-graph-system)
   - [CUDAGraphOptions Dataclass](#cudagraphoptions-dataclass)
   - [CUDAGraphEntry Dataclass](#cudagraphentry-dataclass)
   - [CUDAGraphStat Dataclass](#cudagraphstat-dataclass)
   - [CUDAGraphLogging Class](#cudagraphlogging-class)
   - [CUDAGraphWrapper Class](#cudagraphwrapper-class)
7. [CUDA Graph Dispatcher](#7-cuda-graph-dispatcher)
   - [CudagraphDispatcher Class](#cudagraphdispatcher-class)
8. [Compilation Decorators](#8-compilation-decorators)
   - [support_torch_compile Decorator](#support_torch_compile-decorator)
   - [ignore_torch_compile Decorator](#ignore_torch_compile-decorator)
   - [maybe_use_cudagraph_partition_wrapper](#maybe_use_cudagraph_partition_wrapper)
   - [Internal Helper Functions](#internal-helper-functions)
9. [TorchCompileWithNoGuardsWrapper](#9-torchcompilewithnoguardswrapper)
   - [Class Definition](#class-definition)
   - [reset_compile_wrapper Function](#reset_compile_wrapper-function)
10. [Compilation Caching](#10-compilation-caching)
    - [VllmSerializableFunction](#vllmserializablefunction)
    - [StandaloneCompiledArtifacts](#standalonecompiledartifacts)
    - [AOT Compile Functions](#aot-compile-functions)
11. [Compilation Counter](#11-compilation-counter)
    - [CompilationCounter Dataclass](#compilationcounter-dataclass)
12. [Compilation Monitor](#12-compilation-monitor)
    - [monitor_torch_compile](#monitor_torch_compile)
    - [monitor_profiling_run](#monitor_profiling_run)
    - [CUDA Graph Capture Validation](#cuda-graph-capture-validation)
13. [Partition Rules](#13-partition-rules)
    - [should_split](#should_split)
    - [inductor_partition_rule_context](#inductor_partition_rule_context)
14. [Base Static Graph Wrapper](#14-base-static-graph-wrapper)
    - [AbstractStaticGraphWrapper Protocol](#abstractstaticgraphwrapper-protocol)
15. [Inductor Pass Infrastructure](#15-inductor-pass-infrastructure)
    - [InductorPass Base Class](#inductorpass-base-class)
    - [CallableInductorPass](#callableinductorpass)
    - [PassContext](#passcontext)
    - [enable_fake_mode Decorator](#enable_fake_mode-decorator)
16. [VllmInductorPass & Pattern Matching](#16-vllminductorpass--pattern-matching)
    - [VllmInductorPass Class](#vllminductorpass-class)
    - [VllmPatternMatcherPass](#vllmpatternmatcherpass)
    - [VllmFusionPatternMatcherPass](#vllmfusionpatternmatcherpass)
    - [VllmPatternReplacement ABC](#vllmpatternreplacement-abc)
    - [PrinterInductorPass](#printerinductorpass)
    - [InductorCompilationConfig](#inductorcompilationconfig)
17. [Pass Manager](#17-pass-manager)
    - [PostGradPassManager Class](#postgradpassmanager-class)
18. [FX Utilities](#18-fx-utilities)
    - [FX Graph Inspection Functions](#fx-graph-inspection-functions)
19. [Compilation Module Init](#19-compilation-module-init)

---

## 1. Compilation Configuration

**File**: `vllm/config/compilation.py`

### CompilationMode Enum

```python
class CompilationMode(enum.IntEnum):
    NONE = 0
    STOCK_TORCH_COMPILE = 1
    DYNAMO_TRACE_ONCE = 2
    VLLM_COMPILE = 3
```

Enum values:
- `NONE` (0): No torch.compile compilation; model runs in fully eager PyTorch mode.
- `STOCK_TORCH_COMPILE` (1): Standard `torch.compile` compilation pipeline.
- `DYNAMO_TRACE_ONCE` (2): Single Dynamo trace through the model, avoiding recompilation.
- `VLLM_COMPILE` (3): Custom vLLM Inductor-based backend with caching, piecewise compilation,
  shape specialization, and custom passes.

### CUDAGraphMode Enum

```python
class CUDAGraphMode(enum.Enum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2
    FULL_DECODE_ONLY = (FULL, NONE)
    FULL_AND_PIECEWISE = (FULL, PIECEWISE)
```

Enum values:
- `NONE` (0): No CUDA graph capture.
- `PIECEWISE` (1): Piecewise CUDA graphs only, keeping CUDA graph-incompatible ops outside.
- `FULL` (2): Full CUDA graph capture for all batches.
- `FULL_DECODE_ONLY` (`(FULL, NONE)`): Full CUDA graph for decode batches only; mixed prefill-decode runs without.
- `FULL_AND_PIECEWISE` (`(FULL, PIECEWISE)`): Full CUDA graph for decode batches and piecewise CUDA graph for prefill/mixed. Default for V1 engine.

Methods:
- `decode_mode() -> CUDAGraphMode`: Returns the decode component of a compound mode.
- `mixed_mode() -> CUDAGraphMode`: Returns the mixed (non-decode) component of a compound mode.
- `has_mode(mode: CUDAGraphMode) -> bool`: Checks if the mode contains a specific sub-mode.
- `requires_piecewise_compilation() -> bool`: Returns True if PIECEWISE is included.
- `max_cudagraph_mode() -> CUDAGraphMode`: Returns the maximum (most capable) mode.
- `has_full_cudagraphs() -> bool`: True if max mode is FULL.
- `has_piecewise_cudagraphs() -> bool`: True if PIECEWISE is included.
- `separate_routine() -> bool`: True if the value is a tuple (compound mode).
- `valid_runtime_modes() -> frozenset[CUDAGraphMode]`: Returns `{NONE, PIECEWISE, FULL}`.
- `is_valid_runtime_mode() -> bool`: True if this is a valid runtime mode.

### DynamicShapesType Enum

```python
class DynamicShapesType(str, enum.Enum):
    BACKED = "backed"
    UNBACKED = "unbacked"
    BACKED_SIZE_OBLIVIOUS = "backed_size_oblivious"
```

Values:
- `BACKED`: Default PyTorch behavior with potential guards ignored.
- `UNBACKED`: No guards guaranteed (most sound) but may throw data dependent errors.
- `BACKED_SIZE_OBLIVIOUS`: Experimental safer alternative; treats backed symbols as unbacked when explicit unbacked handling is defined.

### DynamicShapesConfig

```python
@config
class DynamicShapesConfig:
    type: DynamicShapesType = DynamicShapesType.BACKED
    evaluate_guards: bool = False
    assume_32_bit_indexing: bool = False
```

Fields:
- `type`: Controls the type of dynamic shapes handling. Default: `BACKED`.
- `evaluate_guards`: Debug mode to detect and fail if Dynamo specializes a dynamic shape by guarding on it. When `True`, guards are not dropped from Dynamo. Requires `VLLM_USE_BYTECODE_HOOK=0`.
- `assume_32_bit_indexing`: Whether all tensor sizes can use 32-bit indexing. Requires PyTorch 2.10+.

Methods:
- `compute_hash() -> str`: Produces a hash unique to the dynamic shapes configuration.

### PassConfig

```python
@config
class PassConfig:
    fuse_norm_quant: bool = None  # Fuse custom RMSNorm + quant ops
    fuse_act_quant: bool = None  # Fuse custom SiluMul + quant ops
    fuse_attn_quant: bool = None  # Fuse custom Attention + quant ops
    eliminate_noops: bool = True  # Eliminate no-op ops
    enable_sp: bool = None  # Enable sequence parallelism
    fuse_gemm_comms: bool = None  # Enable async TP
    fuse_allreduce_rms: bool = None  # Enable flashinfer allreduce fusion
    fuse_minimax_qk_norm: bool = None  # Fused allreduce+RMSNorm for MiniMax QK norm
    enable_qk_norm_rope_fusion: bool = False  # Fused Q/K RMSNorm + RoPE pass
    # ROCm/AITER specific:
    fuse_act_padding: bool = None  # Fuse custom RMSNorm + padding
    fuse_mla_dual_rms_norm: bool = None  # Fuse paired q/kv RMS norms in MLA
    fuse_rope_kvcache: bool = None  # Fuse QK rope + KV cache ops
    rope_kvcache_fusion_max_token_num: int = 256
    fi_allreduce_fusion_max_size_mb: float | None = None
    sp_min_token_num: int | None = None
```

Methods:
- `flashinfer_max_size(world_size: int) -> int | None`: Returns max communication size in bytes for flashinfer allreduce fusion.
- `default_fi_allreduce_fusion_max_size_mb() -> dict[int, float]`: Static; returns default size thresholds by world size.
- `compute_hash() -> str`: Produces a hash unique to pass configuration.
- `_skip_none_validation(value, handler) -> Any`: Class method; skips validation if value is `None`.
- `__post_init__()`: Handles deprecation and defaults, including platform-specific warnings.
- `log_enabled_passes() -> None`: Logs enabled custom fusion passes.

### CompilationConfig

```python
@config
class CompilationConfig:
    # Top-level Compilation control
    mode: CompilationMode = None
    debug_dump_path: Path | None = None
    cache_dir: str = ""
    compile_cache_save_format: Literal["binary", "unpacked"]  # defaults to env var
    backend: str = ""
    custom_ops: list[str] = []
    ir_enable_torch_wrap: bool = None
    splitting_ops: list[str] | None = None
    compile_mm_encoder: bool = False
    # Vision encoder CUDA graph
    cudagraph_mm_encoder: bool = False
    encoder_cudagraph_token_budgets: list[int] = []
    encoder_cudagraph_max_vision_items_per_batch: int = 0
    encoder_cudagraph_max_frames_per_batch: int | None = None
    # Inductor capture
    compile_sizes: list[int | str] | None = None
    compile_ranges_endpoints: list[int] | None = None
    inductor_compile_config: dict = {}
    inductor_passes: dict[str, str] = {}
    # CudaGraph compilation
    cudagraph_mode: CUDAGraphMode = None
    cudagraph_num_of_warmups: int = 0
    cudagraph_capture_sizes: list[int] = None
    cudagraph_copy_inputs: bool = False
    cudagraph_specialize_lora: bool = True
    use_inductor_graph_partition: bool = None
    pass_config: PassConfig = PassConfig()
    max_cudagraph_capture_size: int = None
    dynamic_shapes_config: DynamicShapesConfig = DynamicShapesConfig()
    local_cache_dir: str = None  # init=False
    fast_moe_cold_start: bool | None = None
    # Internal (init=False):
    enabled_custom_ops: Counter[str]
    disabled_custom_ops: Counter[str]
    traced_files: set[str]
    compilation_time: float
    encoder_compilation_time: float
    static_forward_context: dict[str, Any]
    static_all_moe_layers: list[str]
```

Key methods:
- `init_backend(vllm_config: VllmConfig, prefix: str = "", is_encoder: bool = False) -> str | Callable`:
  Initializes the compilation backend from a VllmConfig. For `STOCK_TORCH_COMPILE` or `DYNAMO_TRACE_ONCE`,
  returns the backend string or resolved callable. For `VLLM_COMPILE`, returns a `VllmBackend` instance.
- `post_init_cudagraph_sizes() -> None`: Completes initialization of compile_sizes after cudagraph sizes are set.
- `set_splitting_ops_for_v1(all2all_backend: str, data_parallel_size: int = 1)`: Sets splitting ops for V1 engine based on compilation mode, pass config, and backend.
- `set_splitting_ops_for_attn_fusion()`: Adjusts splitting ops when attention quantization fusion is enabled.
- `splitting_ops_contain_attention() -> bool`: Checks if all attention ops are in splitting_ops.
- `splitting_ops_contain_kv_cache_update() -> bool`: Checks if KV cache update ops are in splitting_ops.
- `is_attention_compiled_piecewise() -> bool`: True if attention is compiled in piecewise mode.
- `custom_op_log_check()`: Logs enabled/disabled custom ops and warns about irrelevant entries.
- `is_custom_op_enabled(op: str) -> bool`: Checks if a specific custom op is enabled.
- `resolve_cudagraph_mode_and_sizes(min_cg_support, min_cg_attn_backend, ...) -> CUDAGraphMode`:
  Resolves the effective cudagraph mode based on attention backend support, speculative decoding config,
  tensor parallelism, and KV cache configuration.
- `adjust_cudagraph_sizes_for_spec_decode(uniform_decode_query_len: int, tensor_parallel_size: int)`: Adjusts cudagraph capture sizes to be multiples of spec-decode query length.
- `get_compile_ranges() -> list[Range]`: Returns compile ranges computed from `compile_ranges_endpoints`.
- `compute_hash() -> str`: Produces a SHA-256 hash unique to all config fields that affect the computation graph.
- `__repr__() -> str` / `__str__() -> str`: Returns a string representation excluding internal/runtime fields.

Validators:
- `validate_mode_before(value) -> Any`: Parses mode from string names (e.g., "NONE", "VLLM_COMPILE").
- `validate_cudagraph_mode_before(value) -> Any`: Parses cudagraph_mode from string.
- `validate_pass_config_before(value) -> Any`: Parses pass_config from dictionary.
- `validate_compile_cache_save_format(value) -> str`: Validates save format is "binary" or "unpacked".

Class variable:
- `_attention_ops: ClassVar[list[str]]`: List of attention operator names used for piecewise cudagraphs:
  `vllm::unified_attention_with_output`, `vllm::unified_mla_attention_with_output`,
  `vllm::mamba_mixer2`, `vllm::mamba_mixer`, `vllm::short_conv`, `vllm::linear_attention`,
  `vllm::plamo2_mamba_mixer`, `vllm::gdn_attention_core`, `vllm::gdn_attention_core_xpu`,
  `vllm::olmo_hybrid_gdn_full_forward`, `vllm::kda_attention`, `vllm::sparse_attn_indexer`,
  `vllm::rocm_aiter_sparse_attn_indexer`, `vllm::deepseek_v4_attention`.

---

## 2. Compiler Interface & Adaptors

**File**: `vllm/compilation/compiler_interface.py`

### CompilerInterface ABC

```python
class CompilerInterface:
    name: str  # Class-level attribute, e.g. "inductor"

    def initialize_cache(self, cache_dir: str, disable_cache: bool = False, prefix: str = "") -> None
    def compute_hash(self, vllm_config: VllmConfig) -> str
    def compile(self, graph: fx.GraphModule, example_inputs: list[Any],
                compiler_config: dict[str, Any], compile_range: Range,
                key: str | None = None) -> tuple[Callable[..., Any] | None, Any | None]
    def load(self, handle: Any, graph: fx.GraphModule, example_inputs: list[Any],
             graph_index: int, compile_range: Range) -> Callable[..., Any]
```

Methods:
- `initialize_cache(cache_dir, disable_cache, prefix)`: Sets up the cache directory. `prefix` can differentiate multiple model parts sharing the same base directory.
- `compute_hash(vllm_config) -> str`: Computes a hash from vLLM config for cache key. Default returns `""`.
- `compile(graph, example_inputs, compiler_config, compile_range, key) -> (callable, handle)`:
  Compiles the FX graph with the given inputs and config. `compile_range` specifies the batch size range.
  Returns `(compiled_callable, cache_handle)`. Returns `(None, None)` on failure.
- `load(handle, graph, example_inputs, graph_index, compile_range) -> Callable`:
  Loads a compiled function from the handle. Raises `NotImplementedError` if caching is not supported.

### InductorStandaloneAdaptor

```python
class InductorStandaloneAdaptor(CompilerInterface):
    name = "inductor_standalone"
```

Constructor:
- `__init__(self, save_format: Literal["binary", "unpacked"])`: Requires PyTorch 2.8+. Patches `CompiledArtifact.save` for atomic writes on older PyTorch.

Methods:
- `compute_hash(vllm_config) -> str`: Returns a 10-char hex hash of Inductor factors.
- `initialize_cache(cache_dir, disable_cache, prefix)`: Sets `self.cache_dir`.
- `compile(graph, example_inputs, compiler_config, compile_range, key) -> (Callable, handle)`:
  Uses `torch._inductor.standalone_compile` for compilation. Supports AOT compilation via `VLLM_USE_MEGA_AOT_ARTIFACT`.
  Key features:
  - Uses `dynamic_shapes="from_example_inputs"` for single sizes, `"from_graph"` for ranges.
  - Patches `FakeTensorMode` to reuse existing mode when inputs are FakeTensors.
  - Disables pre-grad passes on PyTorch < 2.12 for cold compile time optimization.
  - Returns `(compiled_graph, (key, path))` on success with caching, or `(compiled_graph, None)` for AOT.
- `load(handle, graph, example_inputs, graph_index, compile_range) -> Callable`:
  Loads from `CompiledArtifact.load(path)`. Wraps output to handle tuple unpacking.

### InductorAdaptor

```python
class InductorAdaptor(CompilerInterface):
    name = "inductor"
```

Methods:
- `compute_hash(vllm_config) -> str`: Returns a 10-char hex hash of Inductor factors.
- `initialize_cache(cache_dir, disable_cache, prefix)`:
  Creates `{base_cache_dir}/inductor_cache` and `{base_cache_dir}/triton_cache` directories,
  sets `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` environment variables.
- `compile(graph, example_inputs, compiler_config, compile_range, key) -> (Callable, handle)`:
  Uses `torch._inductor.compile_fx.compile_fx` with extensive monkey-patching:
  - Hijacks `compiled_fx_graph_hash` to capture the graph hash.
  - Uses `AlwaysHitShapeEnv` for Inductor code cache lookup.
  - Forces graph caching via `_check_can_cache` override.
  - Disables AOTAutogradCache (required for InductorAdaptor).
  - Clears tracing context to avoid FakeTensorMode mismatches.
  - Returns `(compiled_graph, (hash_str, file_path))`.
- `load(handle, graph, example_inputs, graph_index, compile_range) -> Callable`:
  Loads from Inductor's `FxGraphCache._lookup_graph`. Converts args from `*args` to `list(args)` for Inductor calling convention.
- `metrics_context() -> AbstractContextManager`: Returns Dynamo metrics context for PyTorch >= 2.6.

### EagerAdaptor

```python
class EagerAdaptor(CompilerInterface):
    name = "eager"
```

Methods:
- `compile(graph, example_inputs, compiler_config, compile_range, key) -> (Callable, None)`:
  Returns the graph module itself without compilation. Increments `num_eager_compiles`.

### AlwaysHitShapeEnv

```python
class AlwaysHitShapeEnv:
    def __init__(self) -> None
    def evaluate_guards_expression(self, *args, **kwargs) -> Literal[True]
    def get_pruned_guards(self, *args, **kwargs) -> list[Any]
    def produce_guards_expression(self, *args, **kwargs) -> Literal[""]
```

A dummy shape environment that always hits, enabling Inductor code cache lookup outside Dynamo context.
Used when compiling for specific shapes outside of the Dynamo bytecode compilation context.

### Helper Functions

- `get_inductor_factors() -> list[Any]`: Collects system factors (CacheBase.get_system()),
  PyTorch factors (torch_key()), Inductor config, and functorch config for cache hashing.
- `_get_vllm_functorch_config() -> dict[str, Any]`: Returns functorch config overrides for vLLM.
  Disables `bundled_autograd_cache` unless `VLLM_USE_MEGA_AOT_ARTIFACT` is set.
- `set_inductor_config(config: dict, compile_range: Range) -> None`: Sets Inductor tuning options
  for single-size compilation (max_autotune, coordinate_descent_tuning).
- `set_functorch_config() -> None`: Applies vLLM functorch config overrides.
- `is_compile_cache_enabled(vllm_additional_inductor_config: dict) -> bool`: Checks if compilation
  cache is enabled (not disabled by env, torch config, or user config).
- `_patch_standalone_compile_atomic_save() -> None`: Backports PyTorch fix for atomic saves on < 2.10.

---

## 3. VllmBackend & Graph Splitting

**File**: `vllm/compilation/backends.py`

### VllmBackend Class

```python
class VllmBackend:
    vllm_config: VllmConfig
    compilation_config: CompilationConfig
    _called: bool
    graph: fx.GraphModule
    split_gm: fx.GraphModule
    piecewise_graphs: list[SplitItem]
    returned_callable: Callable[..., Any]
    post_grad_passes: Sequence[Callable[..., Any]]
    compiler_manager: CompilerManager
    inductor_config: dict[str, Any]
```

Constructor:
```python
def __init__(self, vllm_config: VllmConfig, prefix: str = "", is_encoder: bool = False)
```

Parameters:
- `vllm_config`: The vLLM configuration.
- `prefix`: Cache directory prefix (e.g., "language_model", "vision_model"). Falls back to `model_tag`.
- `is_encoder`: Whether this backend is for an encoder module.

Key methods:
- `configure_post_pass() -> None`: Sets up `PostGradPassManager` as the `post_grad_custom_post_pass` in
  Inductor config. Adds `VllmIRInplaceFunctionalizationPass` as pre-grad pass. Configures cache key
  ignore prefixes.
- `__call__(graph: fx.GraphModule, example_inputs: Sequence[Any]) -> Any`:
  Main entry point called by Dynamo. Steps:
  1. Computes cache directory hash from env, config, code, and compiler factors.
  2. Initializes `CompilerManager` cache.
  3. Calls `configure_post_pass()`.
  4. Calls `split_graph()` to partition the FX graph.
  5. Runs `PiecewiseCompileInterpreter` to compile each piecewise subgraph.
  6. Generates execution code via `generate_execution_code()`.
  7. Returns `VllmSerializableFunction` wrapping the compiled callable.

- `collect_standalone_compile_artifacts() -> tuple[Any, dict, dict]`: Collects Inductor cache artifacts
  from all piecewise backends when `VLLM_USE_MEGA_AOT_ARTIFACT` is enabled.
- `_log_compilation_config()`: Logs compilation config for TORCH_TRACE/tlparse.

### CompilerManager

```python
class CompilerManager:
    def __init__(self, compilation_config: CompilationConfig)
```

Fields:
- `cache: dict[tuple[Range, int, str], Any]`: Maps `(compile_range, graph_index, compiler_name)` to cache data.
- `is_cache_updated: bool`: Whether the cache has been modified since last save.
- `compilation_config: CompilationConfig`: The compilation configuration.
- `compiler: CompilerInterface`: The underlying compiler (InductorStandaloneAdaptor, InductorAdaptor, or EagerAdaptor).
- `loaded_artifacts: dict[str, Any]`: Maps cache keys to loaded compiled functions.

Methods:
- `compute_hash(vllm_config) -> str`: Delegates to `self.compiler.compute_hash()`.
- `compile_context(compile_range: Range) -> Generator`: Context manager providing pass context and optionally inductor partition rules.
- `initialize_cache(cache_dir: str, disable_cache: bool = False, prefix: str = "")`: Loads existing cache from `vllm_compile_cache.py` or creates new. Delegates to `self.compiler.initialize_cache()`.
- `save_to_file()`: Saves cache to `vllm_compile_cache.py` using `pprint.PrettyPrinter`.
- `load(graph, example_inputs, graph_index, compile_range) -> Callable | None`: Loads a compiled graph from cache. Returns `None` on cache miss.
- `compile(graph, example_inputs, additional_inductor_config, compilation_config, compile_range, graph_index=0, num_graphs=1, is_encoder=False) -> Any`:
  Compiles a graph with caching. Features:
  - Deduplication via autograd cache key monkey-patching.
  - Compilation time tracking.
  - Cache entry storage with handle and key.

### SplitItem Dataclass

```python
@dataclasses.dataclass
class SplitItem:
    submod_name: str
    graph_id: int
    is_splitting_graph: bool
    graph: fx.GraphModule
```

Fields:
- `submod_name`: Name of the submodule (e.g., "submod_0").
- `graph_id`: Integer graph ID.
- `is_splitting_graph`: Whether this subgraph contains a splitting op.
- `graph`: The FX GraphModule for this subgraph.

### split_graph Function

```python
def split_graph(graph: fx.GraphModule, splitting_ops: list[str]) -> tuple[fx.GraphModule, list[SplitItem]]
```

Steps:
1. Calls `_decompose_size_nodes()` to replace `x.size()` with per-dim `sym_size.int` calls.
2. Assigns each node to a subgraph ID based on splitting ops.
3. Groups consecutive splitting ops together.
4. Calls `_merge_empty_only_subgraphs()` to merge allocation-only subgraphs into the previous partition.
5. Uses `torch.fx.passes.split_module.split_module()` to create the split GraphModule.
6. Returns `(split_gm, sorted SplitItem list)`.

### PiecewiseCompileInterpreter

```python
class PiecewiseCompileInterpreter(torch.fx.Interpreter):
    def __init__(self, module: fx.GraphModule, compile_submod_names: list[str],
                 vllm_config: VllmConfig, vllm_backend: VllmBackend)
```

FX Interpreter that runs the split graph and compiles each submodule:
- `run(*args) -> Any`: Runs the interpreter with tracing instrumentation.
- `call_module(target, args, kwargs) -> Any`: For each submodule in `compile_submod_names`,
  creates a `PiecewiseBackend`, wraps it with `wrap_with_cudagraph_if_needed()`, and replaces
  the module in the parent graph.

### wrap_with_cudagraph_if_needed

```python
def wrap_with_cudagraph_if_needed(
    piecewise_backend: Any,
    vllm_config: VllmConfig,
    compilation_config: CompilationConfig,
    is_first_graph: bool,
    is_last_graph: bool,
) -> Any
```

Wraps a piecewise backend with a `CUDAGraphWrapper` (or platform-specific static graph wrapper)
if piecewise CUDA graphs are enabled and Dynamo-based splitting is used (not Inductor graph partition).
Always assigns `PIECEWISE` runtime mode.

### Helper Functions

- `make_copy_and_call(sym_tensor_indices, input_buffers, callable_fn) -> Callable`: Creates a wrapper
  that copies dynamic tensors to static buffers before calling the compiled function.
- `make_compiler(compilation_config) -> CompilerInterface`: Creates the appropriate compiler adaptor
  based on `compilation_config.backend`.
- `_is_empty_allocation_node(node) -> bool`: Checks if a node is an empty tensor allocation.
- `_merge_empty_only_subgraphs(node_to_subgraph_id, split_op_graphs)`: Merges allocation-only partitions.
- `_decompose_size_nodes(graph)`: Replaces `x.size()` calls with per-dim `sym_size.int` nodes.
- `set_model_tag(tag: str, is_encoder: bool = False)`: Context manager to set the model tag for compilation.

Global state:
- `compilation_start_time: float = 0.0`: Tracks compilation start time.
- `model_tag: str = "backbone"`: Current model tag (e.g., "backbone", "eagle_head").
- `model_is_encoder: bool = False`: Whether the current model is an encoder.

Exception:
- `StopCompiling(BaseException)`: Raised to short-circuit compilation when a duplicate artifact is found.

---

## 4. Piecewise Backend

**File**: `vllm/compilation/piecewise_backend.py`

### RangeEntry Dataclass

```python
@dataclasses.dataclass
class RangeEntry:
    compile_range: Range
    compiled: bool = False
    runnable: Callable[..., Any] = None
```

### PiecewiseBackend Class

```python
class PiecewiseBackend:
    def __init__(
        self,
        graph: fx.GraphModule | None,
        vllm_config: VllmConfig,
        piecewise_compile_index: int,
        total_piecewise_compiles: int,
        sym_shape_indices: list[int],
        vllm_backend: VllmBackend,
        returns_tuple: bool,
        compiled_runnables: dict[str, Callable] | None = None,
        submod_name: str = "",
    )
```

Parameters:
- `graph`: The FX GraphModule to compile (compilation mode), or `None` (precompilation mode).
- `vllm_config`: The vLLM configuration.
- `piecewise_compile_index`: Index of this piece in the sequence (0-based).
- `total_piecewise_compiles`: Total number of piecewise compilations.
- `sym_shape_indices`: Indices of arguments with symbolic (dynamic) shapes.
- `vllm_backend`: The parent `VllmBackend` instance.
- `returns_tuple`: Whether the subgraph returns a tuple.
- `compiled_runnables`: Pre-compiled callables for cache loading (mutually exclusive with `graph`).
- `submod_name`: Name of the submodule.

Two modes:
1. **Compilation mode** (`graph` set, `compiled_runnables=None`): Compiles the graph for each range.
2. **Precompilation mode** (`graph=None`, `compiled_runnables` set): Wraps pre-compiled callables.

Fields:
- `is_first_graph: bool`: Whether this is the first piecewise subgraph.
- `is_last_graph: bool`: Whether this is the last piecewise subgraph.
- `is_full_graph: bool`: Whether there is only one piecewise subgraph (total == 1).
- `is_encoder_compilation: bool`: Whether this is an encoder compilation.
- `compile_ranges: list[Range]`: The compile ranges from config.
- `compile_sizes: list[int] | None`: The compile sizes from config.
- `range_entries: dict[Range, RangeEntry]`: Maps compile ranges to their compiled entries.

Methods:
- `compile_all_ranges() -> None`: Compiles all range entries up front. For single-size ranges, uses `create_concrete_args()`; for general ranges, uses `get_fake_args_from_graph()`.
- `load_all_ranges() -> None`: Loads pre-compiled runnables for all range entries (warm start path).
- `_find_range_for_shape(runtime_shape: int) -> RangeEntry | None`: Finds the matching range entry for a given runtime batch size. Checks compile_sizes first, then ranges.
- `__call__(*args) -> Any`: Dispatches to the appropriate range entry's runnable based on runtime shape.
- `get_compiled_graph_wrapper(compiled_graph) -> Callable`: Wraps a compiled graph to handle tuple unpacking.
- `to_bytes() -> dict[str, bytes]`: Serializes all compiled entries for AOT caching. Handles `CachingAutotuner` pickling.
- `_log_compile_start(compile_range)`: Logs compilation event for TORCH_TRACE/tlparse.

### create_concrete_args

```python
def create_concrete_args(graph: fx.GraphModule, size: int) -> list[Any]
```

Creates Fake example inputs with all symbolic dimensions replaced by the concrete `size` value.
Used for single-size compilation. Creates a new `FakeTensorMode` with fresh `ShapeEnv`.

### get_fake_args_from_graph

```python
def get_fake_args_from_graph(graph: fx.GraphModule) -> list[Any]
```

Extracts fake argument values from placeholder nodes' `example_value` metadata.

---

## 5. Code Generation

**File**: `vllm/compilation/codegen.py`

### generate_execution_code_with_name

```python
def generate_execution_code_with_name(
    split_gm: torch.fx.GraphModule,
    fn_name: str,
    with_submod: bool,
    consts: list[Any] | None = None,
    const_index: dict[int, int] | None = None,
) -> tuple[str, list[str], list[Any]]
```

Generates a Python function from the FX graph that calls submodules directly,
avoiding FX GraphModule overhead. Features:
- Liveness analysis: emits `del` after last use to free memory early.
- Inlines nested `torch.fx.GraphModule` submodules.
- Collects non-primitive constants (torch.device, DTensor placements) into a list
  referenced by index in generated code.

Returns:
- `(code, submod_names, consts)`: Python source code, ordered submodule names, and constants list.

### generate_execution_code

```python
@dynamo_timed("vllm.generate_execution_code")
def generate_execution_code(split_gm: torch.fx.GraphModule) -> tuple[str, list[str], list[Any]]
```

Top-level wrapper that generates execution code with function name `"execution_fn"` and `with_submod=True`.
Prepends `import torch` and `import operator`.

### compile_execution_fn

```python
@dynamo_timed("vllm.compile_execution_fn")
def compile_execution_fn(
    code: str,
    submod_callables: dict[str, Callable[..., Any]],
    submod_names: list[str],
    consts: list[Any] | None = None,
) -> Callable[..., Any]
```

Compiles execution code and binds submodule callables:
1. Executes the generated Python code via `exec()`.
2. Extracts the `execution_fn` from the namespace.
3. Binds submodule callables using `functools.partial` with `__vllm_submods__`.

### _node_ref

```python
def _node_ref(arg: Any, consts: list[Any], const_index: dict[int, int]) -> str
```

Converts an FX node argument to a source code reference. Handles:
- `torch.fx.Node`: Returns node name directly.
- `list`/`tuple`/`dict`: Recursively converts elements.
- Primitive types (`int`, `float`, `bool`, `str`, `bytes`, `None`): Returns `repr()`.
- Other objects: Stores in `consts` list and returns `__vllm_consts__[index]`.

---

## 6. CUDA Graph System

**File**: `vllm/compilation/cuda_graph.py`

### CUDAGraphOptions Dataclass

```python
@dataclasses.dataclass
class CUDAGraphOptions:
    debug_log_enable: bool = True
    gc_disable: bool = False
    weak_ref_output: bool = True
```

Options controlling CUDA graph capture behavior:
- `debug_log_enable`: Whether to log capture events.
- `gc_disable`: Disable garbage collection during capture.
- `weak_ref_output`: Convert output to weak references to save memory.

### CUDAGraphEntry Dataclass

```python
@dataclasses.dataclass
class CUDAGraphEntry:
    batch_descriptor: BatchDescriptor
    cudagraph: torch.cuda.CUDAGraph | None = None
    output: Any | None = None
    input_addresses: list[int] | None = None
```

Stores a captured CUDA graph and its associated data:
- `batch_descriptor`: The batch descriptor key for this entry.
- `cudagraph`: The captured `torch.cuda.CUDAGraph` instance.
- `output`: Weak-referenced output from the last graph replay.
- `input_addresses`: Tensor data pointers for debug validation.

### CUDAGraphStat Dataclass

```python
@dataclasses.dataclass(frozen=True)
class CUDAGraphStat:
    num_unpadded_tokens: int
    num_padded_tokens: int
    num_paddings: int
    runtime_mode: str
```

Immutable record of a CUDA graph execution for metrics.

### CUDAGraphLogging Class

```python
class CUDAGraphLogging:
    COLUMN_HEADERS = ["Unpadded Tokens", "Padded Tokens", "Num Paddings", "Runtime Mode", "Count"]

    def __init__(self, cg_mode: CUDAGraphMode, cg_capture_sizes: list[int] | None)
```

Aggregates and logs CUDA graph metrics in a table format.

Methods:
- `reset() -> None`: Clears collected stats.
- `observe(cudagraph_stat: CUDAGraphStat) -> None`: Records a stat.
- `generate_metric_table() -> str`: Generates a formatted table of stats with counts, sorted by frequency.
- `log(log_fn: Callable = logger.info) -> None`: Logs the table and resets.

### CUDAGraphWrapper Class

```python
class CUDAGraphWrapper:
    _all_instances: ClassVar[weakref.WeakSet["CUDAGraphWrapper"]]

    def __init__(
        self,
        runnable: Callable[..., Any],
        vllm_config: VllmConfig,
        runtime_mode: CUDAGraphMode,
        cudagraph_options: CUDAGraphOptions | None = None,
    )
```

Wraps a callable to add CUDA graph capture and replay. The wrapper:
1. At init, is assigned a runtime mode (FULL or PIECEWISE).
2. At runtime, receives runtime_mode and batch_descriptor from forward context.
3. If runtime_mode matches, performs capture (if key not cached) or replay.
4. If runtime_mode doesn't match or is NONE, calls the underlying runnable directly.

Constructor parameters:
- `runnable`: The callable to wrap.
- `vllm_config`: vLLM configuration.
- `runtime_mode`: The CUDA graph mode for this wrapper (FULL or PIECEWISE).
- `cudagraph_options`: Optional capture options.

Key attributes:
- `graph_pool`: Platform-specific CUDA graph memory pool.
- `concrete_cudagraph_entries: dict[BatchDescriptor, CUDAGraphEntry]`: Cached CUDA graphs keyed by batch descriptor.
- `is_debugging_mode: bool`: Whether to validate input addresses during replay.

Class methods:
- `clear_all_graphs() -> None`: Clears graphs from all CUDAGraphWrapper instances.

Instance methods:
- `__getattr__(key) -> Any`: Proxies attribute access to the underlying runnable.
- `unwrap() -> Callable`: Returns the original runnable.
- `clear_graphs() -> None`: Clears captured graphs.
- `__call__(*args, **kwargs) -> Any | None`:
  Main dispatch method:
  1. Checks forward context availability.
  2. Reads `cudagraph_runtime_mode` and `batch_descriptor` from forward context.
  3. If mode doesn't match, passes through to runnable.
  4. On first call with matching batch_descriptor, captures CUDA graph:
     - Validates capture is legal via `validate_cudagraph_capturing_enabled()`.
     - Records input addresses for debug.
     - Uses `torch.cuda.graph()` context manager with shared graph pool.
     - Optionally disables GC during capture.
     - Converts output to weak references.
     - Syncs offloader's copy stream.
  5. On subsequent calls, replays captured graph after validating input addresses (in debug mode).

---

## 7. CUDA Graph Dispatcher

**File**: `vllm/v1/cudagraph_dispatcher.py`

### CudagraphDispatcher Class

```python
class CudagraphDispatcher:
    def __init__(self, vllm_config: VllmConfig)
```

Runtime CUDA graph dispatcher that manages dispatch keys for multiple sets of CUDA graphs.
Stores keys for both PIECEWISE and FULL modes, initialized based on attention support and config.

Constructor parameters:
- `vllm_config`: The vLLM configuration.

Fields:
- `vllm_config`: The vLLM configuration.
- `compilation_config`: Compilation config shortcut.
- `uniform_decode_query_len: int`: Query length for uniform decode (1 normally, or 1 + num_speculative_tokens).
- `cudagraph_keys: dict[CUDAGraphMode, set[BatchDescriptor]]`: Valid dispatch keys per mode.
- `keys_initialized: bool`: Whether keys have been initialized.
- `specialize_lora_count: bool`: Whether to specialize CUDA graphs by LoRA count.
- `cudagraph_mode: CUDAGraphMode`: Current CUDA graph mode (default NONE until initialized).
- `_bs_to_padded_graph_size: list[int]`: Mapping from batch size to padded graph size.

Key methods:
- `_compute_bs_to_padded_graph_size() -> None`: Pre-computes batch size to padded graph size mapping.
- `_get_lora_cases() -> list[int]`: Returns LoRA capture cases (0 for no LoRA, or specific counts).
- `_create_padded_batch_descriptor(num_tokens, uniform_decode, has_lora, num_active_loras) -> BatchDescriptor`:
  Creates a padded batch descriptor for a given input configuration.
- `add_cudagraph_key(runtime_mode, batch_descriptor)`: Adds a dispatch key for the given mode.
- `initialize_cudagraph_keys(cudagraph_mode, uniform_decode_query_len=1)`:
  Initializes all valid CUDA graph keys based on mode, capture sizes, and LoRA cases.
  Creates keys for:
  - Mixed mode (prefill/mixed): PIECEWISE or FULL depending on config.
  - Decode mode: FULL if `separate_routine()` and decode mode is FULL.
- `dispatch(num_tokens, uniform_decode=False, has_lora=False, num_active_loras=0,
            valid_modes=None, invalid_modes=None) -> tuple[CUDAGraphMode, BatchDescriptor]`:
  Given runtime conditions, dispatches to the appropriate CUDA graph mode and batch descriptor.
  Steps:
  1. Computes allowed modes from `valid_modes` minus `invalid_modes`.
  2. Checks preconditions (keys initialized, mode not NONE, within max size).
  3. Normalizes `num_active_loras` for LoRA specialization.
  4. Creates padded batch descriptor.
  5. Checks FULL mode keys first, then PIECEWISE.
  6. Falls back to NONE if no match.
- `get_capture_descs() -> list[tuple[CUDAGraphMode, list[BatchDescriptor]]]`:
  Returns capture descriptors ordered PIECEWISE first, then FULL, sorted largest-first for memory efficiency.

---

## 8. Compilation Decorators

**File**: `vllm/compilation/decorators.py`

### support_torch_compile Decorator

```python
@overload
def support_torch_compile(*, enable_if: Callable[[VllmConfig], bool] | None = None) -> Callable

@overload
def support_torch_compile(*, dynamic_arg_dims: dict[str, int | list[int] | dict[int, str]] | None) -> Callable

@overload
def support_torch_compile(*, mark_unbacked_dims: dict[str, int | list[int]] | None) -> Callable

@overload
def support_torch_compile(cls: type[_T]) -> type[_T]

def support_torch_compile(
    cls: type[_T] | None = None,
    *,
    dynamic_arg_dims: dict[str, int | list[int] | dict[int, str]] | None = None,
    mark_unbacked_dims: dict[str, int | list[int]] | None = None,
    enable_if: Callable[[VllmConfig], bool] | None = None,
    is_encoder: bool = False,
) -> Callable[[type[_T]], type[_T]] | type[_T]
```

A decorator to add `torch.compile` support to the forward method of a class.

Usage modes:
1. Direct (no arguments): `@support_torch_compile` -- infers dynamic dims from type annotations.
2. With arguments: `@support_torch_compile(dynamic_arg_dims={"x": 0})`.

Parameters:
- `cls`: The class to decorate (when used without arguments).
- `dynamic_arg_dims`: Maps argument names to dynamic dimensions. Values can be:
  - `int`: Single dimension index.
  - `list[int]`: Multiple dimension indices.
  - `dict[int, str]`: Dimension to shape_id mapping (for sharing unbacked symbols).
- `mark_unbacked_dims`: Maps argument names to dimensions that should be marked as unbacked.
- `enable_if`: Function taking `VllmConfig` returning bool; enables compilation conditionally.
- `is_encoder`: Marks this module as a multimodal encoder component.

When `dynamic_arg_dims` is None, it is inferred from forward method type annotations:
- `torch.Tensor` or `Optional[torch.Tensor]`: First dimension marked dynamic.
- `IntermediateTensors`: First dimension of all tensors marked dynamic.

Internal implementation (`_support_torch_compile`):
1. Adds `TorchCompileWithNoGuardsWrapper` to the class's `__bases__`.
2. Replaces `__init__` to initialize compilation state.
3. Replaces `__call__` with compilation-aware dispatch:
   - Checks `do_not_compile` flag.
   - Checks `skip_compiled` from forward context.
   - Handles AOT compilation loading and saving.
   - Marks dynamic inputs based on `DynamicShapesType`.
   - Tracks traced files via patched `InliningInstructionTranslator.inline_call_`.
   - Runs compilation with appropriate config patches.

### ignore_torch_compile Decorator

```python
def ignore_torch_compile(cls: type[_T]) -> type[_T]
```

Marks a class to be excluded from compilation, even if a parent class has `@support_torch_compile`.
Sets `_ignore_compile_vllm = True` on the class.

### maybe_use_cudagraph_partition_wrapper

```python
@contextlib.contextmanager
def maybe_use_cudagraph_partition_wrapper(vllm_config: VllmConfig) -> Generator[None, None, None]
```

Context manager that sets/unsets customized CUDA graph partition wrappers for Inductor-based
graph partitioning. When using piecewise CUDA graphs with `use_inductor_graph_partition=True`,
registers a custom wrapper factory via `torch._inductor.utils.set_customized_partition_wrappers()`.

### Internal Helper Functions

- `should_torch_compile_mm_encoder(vllm_config) -> bool`: Returns `compilation_config.compile_mm_encoder`.
- `_should_ignore_torch_compile(cls) -> bool`: Checks the ignore flag.
- `_model_hash_key(fn) -> str`: Computes SHA-256 hash of vLLM version + function qualname + line number.
- `_verify_source_unchanged(source_info, vllm_config)`: Verifies traced source files haven't changed since compilation.
- `_try_load_aot_compiled_fn(model, aot_compilation_path) -> Any | None`: Attempts to load an AOT-compiled function from disk.

---

## 9. TorchCompileWithNoGuardsWrapper

**File**: `vllm/compilation/wrapper.py`

### Class Definition

```python
class TorchCompileWithNoGuardsWrapper:
    def __init__(self, compile_prefix: str = "", is_encoder: bool = False)
```

A wrapper class for `torch.compile` that drops all guards when not using `STOCK_TORCH_COMPILE` mode.
On first call, triggers a single compilation; Dynamo should never trace again after that.

Constructor steps:
1. Gets current `VllmConfig` and compilation mode.
2. Initializes the backend via `compilation_config.init_backend()`.
3. Sets up guard handling:
   - For `STOCK_TORCH_COMPILE`: Uses default guards.
   - For other modes: Drops all guards via `guard_filter_fn`.
   - If `evaluate_guards`: Only keeps `SHAPE_ENV` guards.
4. Applies `constrain_to_fx_strides` patch.
5. Compiles the forward method with `torch.compile(fullgraph=True, dynamic=False)`.
6. Optionally registers bytecode hook via `VLLM_USE_BYTECODE_HOOK`.

Attributes:
- `compiled: bool`: Whether compilation has occurred.
- `_compile_prefix: str`: Prefix for cache directory.
- `_is_encoder: bool`: Whether this is an encoder module.
- `vllm_config: VllmConfig`: The vLLM configuration.
- `first_compile: bool`: Whether this is the first compile call.
- `evaluate_guards: bool`: Whether to evaluate guards.
- `_compiled_callable`: The torch.compiled function.
- `_compiled_bytecode: CodeType | None`: Cached compiled bytecode (for bytecode hook mode).

Methods:
- `aot_compile(*args, **kwargs) -> Any`: Invokes AOT compilation on the compiled callable.
- `__call__(*args, **kwargs) -> Any`: Dispatches to compiled code:
  - Bytecode hook mode: Uses compiled bytecode directly or triggers compilation.
  - Standard mode: Uses `_compiled_callable` with `fail_on_recompile` stance after first compile.
- `forward(*args, **kwargs) -> Any`: Abstract method (must be overridden by the actual model).
- `original_code_object() -> CodeType`: Returns the original forward method's code object.
- `bytecode_hook(old_code, new_code)`: Saves compiled bytecode and optionally decompiles with depyf.
  Also validates no buffer modifications during forward when CUDA graphs are used.
- `_dispatch_to_compiled_code() -> Generator`: Context manager that temporarily replaces forward's code object.

### reset_compile_wrapper Function

```python
def reset_compile_wrapper(model: torch.nn.Module) -> None
```

Resets compiled model and captured CUDA graphs for elastic EP (Expert Parallelism).
Steps:
1. Resets all `CompilationCounter` fields to 0.
2. Clears AOT compiled function.
3. Resets cache directory.
4. Restores original code object.
5. Reinitializes `TorchCompileWithNoGuardsWrapper`.

---

## 10. Compilation Caching

**File**: `vllm/compilation/caching.py`

### VllmSerializableFunction

```python
class VllmSerializableFunction(SerializableCallable):
    def __init__(
        self,
        graph_module: torch.fx.GraphModule | bytes,
        example_inputs: Sequence[Any],
        prefix: str,
        optimized_call: Callable[..., Any],
        is_encoder: bool = False,
        vllm_backend: Any | None = None,
        sym_tensor_indices: list[int] | None = None,
        aot_autograd_config: dict[str, Any] | None = None,
        execution_code: str | None = None,
        submod_names: list[str] | None = None,
        consts: list[Any] | None = None,
    )
```

A wrapper around a compiled function that supports serialization for PyTorch's precompile
with custom backend. Serializes the Dynamo FX graph plus example inputs.

Key methods:
- `__call__(*args, **kwargs) -> Any`: Delegates to `self.optimized_call`.
- `serialize_graph_module(graph_module) -> bytes`: Class method. Serializes an FX GraphModule using `GraphPickler.dumps()`.
- `deserialize_graph_module(data, fake_mode) -> fx.GraphModule`: Class method. Deserializes an FX GraphModule.
- `serialize_compile_artifacts(compiled_fn) -> bytes`: Class method. Serializes the entire compiled function
  including graph module, example inputs, and optionally standalone compile artifacts.
- `deserialize_compile_artifacts(data) -> VllmSerializableFunction`: Class method. Deserializes a compiled function.
  Supports both mega artifact path (using `reconstruct_serializable_fn_from_mega_artifact`) and standard path
  (using `VllmBackend` recompilation).
- `finalize_loading(vllm_config) -> None`: Eagerly initializes the compiled backend after `traced_files` is populated.
- `co_name -> Literal["VllmSerializableFunction"]`: Property for depyf debugging.

### StandaloneCompiledArtifacts

```python
class StandaloneCompiledArtifacts:
    def __init__(self) -> None
```

Storage for standalone compiled artifacts with content-based deduplication via two-level indirection:
1. `submodule_bytes`: Maps `"{submod_name}_{shape}" -> SHA256 hash`.
2. `submodule_bytes_store`: Maps `SHA256 hash -> actual bytes`.

Fields:
- `submodule_bytes: dict[str, str]`: Cache key to hash mapping.
- `submodule_bytes_store: dict[str, bytes]`: Hash to bytes mapping.
- `loaded_submodule_store: dict[str, Any]`: Hash to loaded module mapping.

Methods:
- `insert(submod_name, shape, entry) -> None`: Inserts a compiled artifact with SHA-256 deduplication.
- `get(submod_name, shape) -> bytes`: Returns raw bytes for a submodule/shape.
- `get_loaded(submod_name, shape) -> Any`: Returns the loaded module for a submodule/shape.
- `size_bytes() -> int`: Total size of stored artifacts.
- `num_artifacts() -> int`: Number of unique artifacts.
- `num_entries() -> int`: Number of submodule/shape entries.
- `submodule_names() -> list[str]`: Returns unique submodule names preserving order.
- `load_all() -> None`: Deserializes all artifacts using `ThreadPoolExecutor` for parallel loading.
- `__getstate__() -> dict`: Pickle support (only serializes bytes, not loaded modules).
- `__setstate__(state)`: Pickle support (resets loaded store).

### AOT Compile Functions

- `aot_compile_hash_factors(vllm_config) -> list[str]`: Computes hash factors for AOT compilation:
  env hash, config hash, and optionally Inductor factors.
- `_compute_code_hash_with_content(file_contents) -> str`: Computes SHA-256 hash of source files and their contents.
- `_compute_code_hash(files) -> str`: Computes code hash from a set of file paths.
- `reconstruct_serializable_fn_from_mega_artifact(state, standalone_compile_artifacts, vllm_config,
    sym_shape_indices_map, returns_tuple_map, fake_mode) -> VllmSerializableFunction`:
  Reconstructs a callable from pre-compiled inductor artifacts without recompilation.
  1. Loads all cached artifacts.
  2. Builds compiled callables for each submodule/shape.
  3. Creates `PiecewiseBackend` instances with pre-compiled runnables.
  4. Wraps with CUDA graph if needed.
  5. Returns the final `VllmSerializableFunction`.

Helper:
- `patch_pytree_map_over_slice()`: Context manager that registers `slice` as a pytree node for serialization.

---

## 11. Compilation Counter

**File**: `vllm/compilation/counter.py`

### CompilationCounter Dataclass

```python
@dataclasses.dataclass
class CompilationCounter:
    num_models_seen: int = 0
    num_graphs_seen: int = 0
    num_piecewise_graphs_seen: int = 0
    num_piecewise_capturable_graphs_seen: int = 0
    num_backend_compilations: int = 0
    num_gpu_runner_capture_triggers: int = 0
    num_cudagraph_captured: int = 0
    num_inductor_compiles: int = 0
    num_eager_compiles: int = 0
    num_cache_entries_updated: int = 0
    num_compiled_artifacts_saved: int = 0
    num_compiled_artifacts_loaded: int = 0
    num_aot_compiles: int = 0
    num_aot_artifacts_saved: int = 0
    num_aot_artifacts_loaded: int = 0
    stock_torch_compile_count: int = 0
```

Fields:
- `num_models_seen`: Number of models with `@support_torch_compile` seen.
- `num_graphs_seen`: Number of FX graphs seen by Dynamo.
- `num_piecewise_graphs_seen`: Number of piecewise subgraphs (including splitting ops).
- `num_piecewise_capturable_graphs_seen`: Number of piecewise subgraphs excluding splitting ops.
- `num_backend_compilations`: Number of backend compilation calls.
- `num_gpu_runner_capture_triggers`: Number of GPU model runner CUDA graph capture attempts.
- `num_cudagraph_captured`: Number of CUDA graphs captured.
- `num_inductor_compiles`: Number of Inductor compilation calls.
- `num_eager_compiles`: Number of eager compilation calls.
- `num_cache_entries_updated`: Number of vLLM compiler cache updates.
- `num_compiled_artifacts_saved`: Number of standalone_compile artifacts saved.
- `num_compiled_artifacts_loaded`: Number of standalone_compile artifacts loaded.
- `num_aot_compiles`: Number of AOT compile invocations.
- `num_aot_artifacts_saved`: Number of AOT artifacts saved to disk.
- `num_aot_artifacts_loaded`: Number of AOT artifacts loaded from disk.
- `stock_torch_compile_count`: Number of models loaded with `STOCK_TORCH_COMPILE` mode.

Methods:
- `clone() -> CompilationCounter`: Returns a deep copy.
- `expect(**kwargs) -> Generator`: Context manager that asserts specific counter differences.

Global instance:
- `compilation_counter: CompilationCounter`: Module-level singleton.

---

## 12. Compilation Monitor

**File**: `vllm/compilation/monitor.py`

### monitor_torch_compile

```python
@contextlib.contextmanager
def monitor_torch_compile(
    vllm_config: VllmConfig,
    message: str = "torch.compile took %.2f s in total",
    is_encoder: bool = False,
) -> Generator[None, None, None]
```

Context manager that:
1. Records start time as `torch_compile_start_time`.
2. Optionally enables depyf debugging for `VLLM_COMPILE` mode when `debug_dump_path` is set.
3. On success: logs compile time, updates `compilation_config.compilation_time` (or `encoder_compilation_time`).
4. On exception: re-raises without logging.
5. Finally: cleans up depyf context.

### monitor_profiling_run

```python
@contextlib.contextmanager
def monitor_profiling_run() -> Generator[None, None, None]
```

Context manager that times the initial profiling run and asserts no backend compilation
occurs during profiling (all compilation should complete before profiling starts).

### CUDA Graph Capture Validation

```python
cudagraph_capturing_enabled: bool = True

def validate_cudagraph_capturing_enabled() -> None
def set_cudagraph_capturing_enabled(enabled: bool) -> None
```

- `validate_cudagraph_capturing_enabled()`: Raises `RuntimeError` if CUDA graph capturing is disabled.
- `set_cudagraph_capturing_enabled(enabled)`: Enables or disables CUDA graph capturing.

Global state:
- `torch_compile_start_time: float = 0.0`: Shared start time accessible from `backends.py`.

---

## 13. Partition Rules

**File**: `vllm/compilation/partition_rules.py`

### should_split

```python
def should_split(node: torch.fx.Node, splitting_ops: list[str]) -> bool
```

Checks if an FX node should be split for Dynamo graph partition.
Only operates on `call_function` nodes with `OpOverload` or `OpOverloadPacket` targets.
Checks both packet names (`aten::add`) and overload names (`aten::add.default`).

### inductor_partition_rule_context

```python
@contextlib.contextmanager
def inductor_partition_rule_context(
    splitting_ops: list[str] | None,
) -> Generator[None, None, None]
```

Context manager that temporarily registers Inductor partition rules for specified operators.
Sets `torch._inductor.config.custom_should_partition_ops` and restores on exit.

---

## 14. Base Static Graph Wrapper

**File**: `vllm/compilation/base_static_graph.py`

### AbstractStaticGraphWrapper Protocol

```python
class AbstractStaticGraphWrapper(Protocol):
    def __init__(
        self,
        runnable: Callable[..., Any],
        vllm_config: VllmConfig,
        runtime_mode: CUDAGraphMode,
        **kwargs: Any,
    ) -> None

    def __call__(self, *args: Any, **kwargs: Any) -> Any
```

A Protocol that defines the interface for platform-specific static graph wrappers.
Allows platforms to wrap a callable for CUDA graph capture/replay.

Constructor parameters:
- `runnable`: The callable to be wrapped and captured.
- `vllm_config`: Global vLLM configuration.
- `runtime_mode`: The CUDA graph mode. Only `NONE`, `PIECEWISE`, and `FULL` are used as concrete runtime modes.

`__call__` behavior:
- If current runtime mode matches this instance's mode: replays or captures CUDA graph.
- Otherwise: calls the original runnable directly.

---

## 15. Inductor Pass Infrastructure

**File**: `vllm/compilation/passes/inductor_pass.py`

### InductorPass Base Class

```python
class InductorPass(CustomGraphPass):
    def uuid(self) -> str
    def is_applicable_for_range(self, compile_range: Range) -> bool
```

Base class for custom graph passes that uses source hash as UUID.

Methods:
- `uuid() -> str`: Returns a unique identifier based on source hash, used in Inductor code cache.
  Defaults to `InductorPass.hash_source(self)`.
- `is_applicable_for_range(compile_range) -> bool`: Returns True by default. Override to skip passes for certain ranges.

Static methods:
- `hash_source(*srcs) -> str`: Hashes source code of functions or objects. Results are cached by type.
- `hash_dict(dict_) -> str`: Hashes a dictionary using JSON serialization with SHA-256.

### CallableInductorPass

```python
class CallableInductorPass(InductorPass):
    def __init__(self, callable: Callable[[fx.Graph], None], uuid: Any | None = None)
    def __call__(self, graph: torch.fx.Graph) -> None
    def uuid(self) -> Any
```

Wraps a plain callable as an `InductorPass`. Automatically provides UUID from source hash.

### PassContext

```python
class PassContext:
    def __init__(self, compile_range: Range)
    compile_range: Range
    donated_input_ids: set[int]
```

Stores pass execution context including the compile range and donated input indices.

Functions:
- `get_pass_context() -> PassContext`: Returns the current pass context.
- `pass_context(compile_range: Range) -> Generator`: Context manager for pass execution.

### enable_fake_mode Decorator

```python
def enable_fake_mode(fn: Callable[P, R]) -> Callable[P, R]
```

Applies `FakeTensorMode` context around a function. Useful for tracing without real tensors.

---

## 16. VllmInductorPass & Pattern Matching

**File**: `vllm/compilation/passes/vllm_inductor_pass.py`

### InductorCompilationConfig

```python
@dataclasses.dataclass
class InductorCompilationConfig:
    splitting_ops: list[str] | None = None
    use_inductor_graph_partition: bool = False
```

Simplified config provided to Inductor passes (avoids exposing full CompilationConfig).

### VllmInductorPass Class

```python
class VllmInductorPass(InductorPass):
    dump_prefix: ClassVar[int | None] = None

    def __init__(self, config: VllmConfig)
```

Base class for vLLM Inductor passes with access to `PassConfig`.

Constructor initializes:
- `compilation_config: InductorCompilationConfig`: Simplified config.
- `pass_config: PassConfig`: Pass configuration.
- `model_dtype`: Model dtype from config.
- `device`: Device from config.
- `pass_name: str`: Class name for logging.

Methods:
- `dump_graph(graph, stage) -> None`: Dumps FX graph with lazy formatting.
- `begin() -> None`: Records start time.
- `end_and_log() -> None`: Logs pass completion time.

Decorators:
- `time_and_log(call_fn) -> Callable`: Wraps a `__call__` method with timing, graph dump before/after, and logging.

### VllmPatternMatcherPass

```python
class VllmPatternMatcherPass(VllmInductorPass):
    matched_count: int = 0
    match_table: ClassVar[defaultdict[str, int]] = defaultdict(int)
```

A pass that uses Inductor's pattern matcher with match counting and debug dumping.

Methods:
- `log_match_summary() -> None`: Class method; logs total match counts.
- `dump_patterns(config, pm_pass) -> None`: Dumps pattern-matcher patterns as Python-like code.

### VllmFusionPatternMatcherPass

```python
class VllmFusionPatternMatcherPass(VllmPatternMatcherPass):
    def __init__(self, config: VllmConfig, pass_name: str)
```

A pass that uses `VllmPatternReplacement` objects for pattern/replacement fusion.

Methods:
- `register(pr: VllmPatternReplacement) -> None`: Registers a pattern/replacement pair under fake mode.
- `uuid() -> str`: Hashes the pass type and all registered pattern types.
- `__call__(graph) -> None`: Applies all registered patterns and tracks match count.

Static methods:
- `_trace_fn(*args, **kwargs) -> fx.GraphModule`: Traces pattern with view-to-reshape and noop permute removal.

### VllmPatternReplacement ABC

```python
class VllmPatternReplacement(ABC, Generic[P, R]):
    @property
    @abstractmethod
    def pattern(self) -> Callable[P, R]: ...

    @property
    @abstractmethod
    def replacement(self) -> Callable[P, R]: ...

    @abstractmethod
    def get_inputs(self) -> list[torch.Tensor]: ...
```

Abstract base for FX graph pattern/replacement pairs.

Abstract properties/methods:
- `pattern`: Returns a closure defining the FX subgraph to search for.
- `replacement`: Returns a closure defining the replacement FX subgraph.
- `get_inputs()`: Returns example tensors for tracing.

Static helper methods:
- `empty(*args, **kwargs) -> torch.Tensor`: Creates empty tensor on CUDA.
- `empty_bf16(*args, **kwargs) -> torch.Tensor`: Creates empty bfloat16 tensor on CUDA.
- `empty_fp16(*args, **kwargs) -> torch.Tensor`: Creates empty float16 tensor on CUDA.
- `empty_fp32(*args, **kwargs) -> torch.Tensor`: Creates empty float32 tensor on CUDA.
- `empty_i32(*args, **kwargs) -> torch.Tensor`: Creates empty int32 tensor on CUDA.

Internal helpers:
- `_fx_view_to_reshape(gm)`: Converts view ops to reshape in FX graph.
- `_remove_noop_permutes(gm)`: Removes identity permute operations.

### PrinterInductorPass

```python
class PrinterInductorPass(VllmInductorPass):
    def __init__(self, name: str, config: VllmConfig)
    def __call__(self, graph: torch.fx.Graph) -> None
```

A pass that only dumps the FX graph at a given stage (for debugging).

### get_match_table

```python
def get_match_table() -> dict[str, int]
```

Returns a snapshot of the global match table.

---

## 17. Pass Manager

**File**: `vllm/compilation/passes/pass_manager.py`

### PostGradPassManager Class

```python
class PostGradPassManager(CustomGraphPass):
    def __init__(self) -> None
```

Manages post-grad passes with configuration, ordering, and UUID support for Inductor code cache.

Pass execution order:
1. User-configured passes (from constructor)
2. Default passes (NoopEliminationPass, fusion passes based on PassConfig)
3. `post_grad_custom_post_pass` from config (if exists)
4. `fix_functionalization`

Methods:
- `__call__(graph: fx.Graph) -> None`: Executes all passes in order, with pattern match debug support.
  Each pass can be skipped based on `is_applicable_for_range()`.
- `configure(config: VllmConfig) -> None`: Configures passes based on `PassConfig`:
  - `eliminate_noops`: NoOpEliminationPass
  - `enable_sp`: SequenceParallelismPass (+ AsyncTPPass if `fuse_gemm_comms`)
  - `fuse_allreduce_rms`: AllReduceFusionPass or RocmAiterAllReduceFusionPass
  - `fuse_minimax_qk_norm`: MiniMaxQKNormPass
  - `fuse_norm_quant`: RMSNormQuantFusionPass (+ RocmAiterRMSNormQuantFusionPass on ROCm)
  - `fuse_act_quant`: ActivationQuantFusionPass (+ RocmAiterSiluMulFp8GroupQuantFusionPass on ROCm)
  - `fuse_act_padding`: RocmAiterTritonAddRMSNormPadFusionPass (ROCm only)
  - `fuse_mla_dual_rms_norm`: MLADualRMSNormFusionPass (ROCm only)
  - `fuse_rope_kvcache`: SplitCoalescingPass + ScatterSplitReplacementPass + RopeKVCacheFusionPass
  - `fuse_attn_quant`: AttnQuantFusionPass + MLAAttnQuantFusionPass
  - `enable_qk_norm_rope_fusion`: SplitCoalescingPass + QKNormRoPEFusionPass
  - Always: VllmIRLoweringPass, UnsafeCloneEliminationPass, PostCleanupPass, FixFunctionalizationPass
- `add(pass_: InductorPass) -> None`: Adds a custom pass.
- `uuid() -> str`: Computes UUID from all pass UUIDs and pass config hash.

Decorator:
- `with_pattern_match_debug(fn)`: Enables/disables Inductor pattern match debug logging.

---

## 18. FX Utilities

**File**: `vllm/compilation/passes/fx_utils.py`

### FX Graph Inspection Functions

- `is_func(node: fx.Node, target: Target) -> bool`: Checks if node is a `call_function` with specific target.
- `is_auto_func(node: fx.Node, op: OpOverload) -> bool`: Checks if node is `auto_functionalized` with specific op.
- `find_auto_fn_maybe(nodes: Iterable[fx.Node], op: OpOverload) -> fx.Node | None`: Finds first auto_functionalized node with given op.
- `find_auto_fn(nodes: Iterable[fx.Node], op: OpOverload) -> fx.Node`: Same as above, but asserts existence.
- `find_getitem_maybe(node: fx.Node, idx: int) -> fx.Node | None`: Finds getitem node extracting idx-th element.
- `find_getitem(node: fx.Node, idx: int) -> fx.Node`: Same as above, but asserts existence.
- `find_op_nodes(op: OpOverload | OpOverloadPacket, graph: fx.Graph) -> Iterator[fx.Node]`: Auto-functionalization-aware node finder. Handles both direct calls and auto_functionalized calls.
- `get_only_user(node: fx.Node) -> fx.Node`: Asserts single user and returns it.

---

## 19. Compilation Module Init

**File**: `vllm/compilation/__init__.py`

Empty module init file (single line).
