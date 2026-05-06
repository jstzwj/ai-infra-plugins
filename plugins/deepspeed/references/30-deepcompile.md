# DeepSpeed DeepCompile Reference

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [CompileConfig](#compileconfig)
4. [FX-Based Compilation Passes](#fx-based-compilation-passes)
5. [PyTorch Inductor Integration](#pytorch-inductor-integration)
6. [Activation Memory Optimization](#activation-memory-optimization)
7. [Double Buffering for Pipeline Parallelism](#double-buffering-for-pipeline-parallelism)
8. [Symmetric Memory for Homogeneous Clusters](#symmetric-memory-for-homogeneous-clusters)
9. [Integration with ZeRO Stages](#integration-with-zero-stages)
10. [Custom Passes](#custom-passes)
11. [Configuration Examples](#configuration-examples)
12. [Troubleshooting](#troubleshooting)

---

## Overview

DeepCompile (`deepspeed/compile/`) is DeepSpeed's compilation framework that applies graph-level optimizations to PyTorch models using PyTorch's FX intermediate representation and the TorchInductor backend. DeepCompile analyzes the entire training computation graph and applies memory-saving transformations that are impossible at the operator level.

### Key Capabilities

1. **Activation memory optimization**: Automatically free or offload intermediate activations during the forward pass to reduce peak GPU memory usage
2. **Double buffering**: Overlap activation computation with pipeline parallelism communication to hide pipeline bubbles
3. **Symmetric memory**: Leverage homogeneous cluster topology for memory-efficient distributed training
4. **ZeRO integration**: Coordinate activation management with ZeRO Stage 1, 2, and 3 parameter partitioning
5. **Custom compilation passes**: Extensible framework for adding new graph-level optimizations
6. **Inductor integration**: Leverage PyTorch's Inductor backend for low-level kernel fusion and optimization

### The Memory Problem

During standard training, all intermediate activations from the forward pass must be retained in memory for the backward pass. For a transformer model with L layers, this means:

```
Activation memory per layer ~ batch_size * seq_length * hidden_size * bytes_per_element

For GPT-3 175B (batch=32, seq=2048, hidden=12288, FP16):
  Per layer: 32 * 2048 * 12288 * 2 = 1.6 GB
  Total (96 layers): 1.6 GB * 96 = 154 GB
  
This is in ADDITION to model parameters (350 GB FP16) and optimizer states.
```

DeepCompile reduces this by:
- **Freeing** activations immediately after they are consumed (recomputed during backward)
- **Offloading** activations to CPU/NVMe when not needed on GPU
- **Double buffering** to overlap activation management with computation

---

## Architecture

### Directory Structure

```
deepspeed/compile/
    __init__.py                  # Public API exports
    config.py                    # CompileConfig dataclass
    backend.py                   # DeepCompileBackend (TorchDynamo backend)
    fx.py                        # FX graph utilities and pass infrastructure
    inductor.py                  # PyTorch Inductor integration
```

### Compilation Pipeline

```
User Training Loop
    |
    v
@torch.compile(model, backend="deepspeed")    # TorchDynamo entry
    |
    v
TorchDynamo captures Python bytecode
    |
    v
Converts to FX Graph (torch.fx.GraphModule)
    |
    v
DeepCompileBackend (backend.py)
    |
    +-- 1. Apply DeepCompile passes (fx.py)
    |       |-- free_activation pass
    |       |-- offload_activation pass
    |       |-- double_buffer pass
    |       |-- symmetric_memory pass
    |       |-- sync_before/after_reduce passes
    |       |-- custom passes
    |
    +-- 2. Delegate to Inductor (inductor.py)
    |       |-- Kernel fusion
    |       |-- Memory planning
    |       |-- Code generation
    |
    v
Optimized executable (TorchInductor compiled kernel)
```

### Class Hierarchy

```
CompileConfig (config.py)           # User-facing configuration
    |
DeepCompileBackend (backend.py)     # TorchDynamo backend implementation
    |
FXPassManager (fx.py)               # Manages compilation passes
    |
DeepSpeedInductor (inductor.py)     # Inductor integration layer
```

---

## CompileConfig

The `CompileConfig` dataclass in `deepspeed/compile/config.py` defines all configuration options for DeepCompile.

### Full Parameter Reference

```python
# deepspeed/compile/config.py

from dataclasses import dataclass, field
from typing import Optional, List, Dict

@dataclass
class CompileConfig:
    """Configuration for DeepSpeed DeepCompile.
    
    Controls activation memory optimization, offloading,
    double buffering, symmetric memory, and synchronization.
    
    All parameters can be set via ds_config.json under the
    "compile" key.
    """
    
    # === Core ===
    deepcompile: bool = False
    """Enable DeepCompile graph optimizations.
    
    When True, DeepCompile intercepts the training step via
    TorchDynamo and applies graph-level optimizations.
    Default: False
    """
    
    # === Activation Memory ===
    free_activation: bool = False
    """Enable activation freeing during forward pass.
    
    When True, activations that are not needed until the backward
    pass are freed immediately after computation. They are
    automatically recomputed during the backward pass.
    
    This trades compute for memory: forward pass uses less memory,
    but backward pass incurs recomputation cost (~30% overhead).
    
    Default: False
    """
    
    free_activation_threshold: int = 10 * 1024 * 1024  # 10 MB
    """Minimum activation tensor size (in bytes) to consider for freeing.
    
    Tensors smaller than this threshold are kept in memory, as the
    overhead of recomputation exceeds the memory savings.
    
    Default: 10 MB (10485760 bytes)
    
    Example values:
      1 MB  = 1048576      (aggressive: free more activations)
      10 MB = 10485760     (default: balance)
      100 MB = 104857600   (conservative: free only large activations)
    """
    
    offload_activation: bool = False
    """Enable activation offloading to CPU memory.
    
    When True, large activations are asynchronously copied to CPU
    memory during the forward pass and copied back to GPU during
    the backward pass. This reduces GPU memory usage without
    recomputation overhead, but requires CPU memory and PCIe bandwidth.
    
    Can be combined with free_activation for hybrid strategies.
    
    Default: False
    """
    
    offload_opt_states: bool = False
    """Offload optimizer states to CPU during forward pass.
    
    When True, optimizer states (momentum, variance) are moved to
    CPU memory during the forward pass and brought back to GPU
    only during the optimizer step. This reduces GPU memory by
    ~12 bytes per parameter (for Adam).
    
    Default: False
    """
    
    offload_parameters: bool = False
    """Offload model parameters to CPU when not in use.
    
    When True, model parameters are kept on CPU and loaded to GPU
    on demand during computation. This is similar to ZeRO-3 offload
    but operates at the graph level for better scheduling.
    
    Default: False
    """
    
    # === Pipeline Parallelism ===
    double_buffer: bool = False
    """Enable double buffering for pipeline parallelism.
    
    When True, two buffers alternate between computation and
    communication. While one buffer is being processed (forward
    or backward), the other is being sent/received over the
    network. This hides pipeline bubble overhead.
    
    Only applicable with pipeline parallelism enabled.
    
    Default: False
    """
    
    # === Distributed Memory ===
    symmetric_memory: bool = False
    """Enable symmetric memory management across homogeneous GPUs.
    
    When True, DeepCompile assumes all GPUs in the training cluster
    have identical memory capacity. This enables:
    - Symmetric activation distribution across GPUs
    - Balanced memory allocation
    - Collective offloading decisions
    
    Default: False
    """
    
    # === Synchronization ===
    sync_before_reduce: bool = False
    """Insert synchronization barrier before gradient reduction.
    
    When True, a torch.cuda.synchronize() call is inserted before
    each all-reduce operation. This ensures all local computation
    completes before communication begins.
    
    Useful for debugging or when using custom kernels that may
    have asynchronous behavior.
    
    Default: False
    """
    
    sync_after_reduce: bool = False
    """Insert synchronization barrier after gradient reduction.
    
    When True, a torch.cuda.synchronize() call is inserted after
    each all-reduce operation. This ensures the reduction is
    complete before any subsequent computation.
    
    Default: False
    """
    
    sync_before_allgather: bool = False
    """Insert synchronization before all-gather operations.
    
    Applicable with ZeRO Stage 3 where parameters are gathered
    before use.
    
    Default: False
    """
    
    sync_after_allgather: bool = False
    """Insert synchronization after all-gather operations.
    
    Ensures parameter gathering is complete before computation.
    
    Default: False
    """
    
    # === Input Tensor Management ===
    keep_int_input_tensors: bool = False
    """Keep integer input tensors materialized in the compiled graph.
    
    When False, integer tensors (e.g., attention masks, position IDs)
    may be optimized away by the compiler. Set to True to preserve
    them, which is needed for some custom operations.
    
    Default: False
    """
    
    keep_all_input_tensors: bool = False
    """Keep all input tensors materialized in the compiled graph.
    
    When True, no input tensor is optimized away. This is the most
    conservative option and may reduce optimization opportunities.
    
    Default: False
    """
    
    # === Custom Passes ===
    passes: Optional[Dict[str, Dict]] = None
    """Custom compilation passes to apply.
    
    Dictionary of pass name to pass configuration. Each pass
    is applied in order during compilation.
    
    Built-in pass names:
    - "z1": ZeRO Stage 1 specific optimizations
    - "z3": ZeRO Stage 3 specific optimizations
    - "autosp": Automatic sequence parallelism detection
    
    Example:
    {
        "z3": {"enabled": true, "prefetch_distance": 2},
        "autosp": {"enabled": true, "min_seq_length": 4096}
    }
    
    Default: None (no custom passes)
    """
    
    # === Debug ===
    debug_log: bool = False
    """Enable detailed debug logging for DeepCompile.
    
    When True, logs:
    - FX graph structure before and after each pass
    - Memory estimates and activation sizes
    - Pass execution times
    - Synchronization points
    
    Default: False
    """
```

### Configuration Table

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `deepcompile` | bool | `False` | Enable DeepCompile |
| `free_activation` | bool | `False` | Free activations during forward |
| `free_activation_threshold` | int | `10485760` | Min size (bytes) for freeing (10 MB) |
| `offload_activation` | bool | `False` | Offload activations to CPU |
| `offload_opt_states` | bool | `False` | Offload optimizer states to CPU |
| `offload_parameters` | bool | `False` | Offload parameters to CPU |
| `double_buffer` | bool | `False` | Double buffering for pipeline parallelism |
| `symmetric_memory` | bool | `False` | Symmetric memory for homogeneous clusters |
| `sync_before_reduce` | bool | `False` | Sync before gradient all-reduce |
| `sync_after_reduce` | bool | `False` | Sync after gradient all-reduce |
| `sync_before_allgather` | bool | `False` | Sync before parameter all-gather |
| `sync_after_allgather` | bool | `False` | Sync after parameter all-gather |
| `keep_int_input_tensors` | bool | `False` | Keep integer input tensors |
| `keep_all_input_tensors` | bool | `False` | Keep all input tensors |
| `passes` | dict | `None` | Custom compilation passes |
| `debug_log` | bool | `False` | Enable debug logging |

---

## FX-Based Compilation Passes

DeepCompile uses PyTorch's FX (torch.fx) framework to capture and transform the training computation graph. Each compilation pass is a graph-to-graph transformation that modifies the FX graph before final compilation.

### FX Graph Basics

PyTorch FX captures a model's forward pass as a directed acyclic graph (DAG) of operations:

```python
import torch.fx

# Capture model as FX graph
graph_module = torch.fx.symbolic_trace(model)

# The graph contains nodes representing operations:
# %input : [users = attn, mlp]
# %linear1 = call_module[target=layer1.attn.q_proj](args = (%input,))
# %linear2 = call_module[target=layer1.attn.k_proj](args = (%input,))
# %matmul = call_function[target=torch.matmul](args = (%linear1, %linear2))
# %output = output(args = (%matmul,))

# Each node has:
# - op: "placeholder", "get_attr", "call_module", "call_function", "call_method", "output"
# - target: The function/module being called
# - args: Input arguments
# - kwargs: Keyword arguments
# - users: Nodes that consume this node's output
# - meta: Metadata (tensor shapes, dtypes)
```

### FXPassManager

```python
# deepspeed/compile/fx.py

class FXPassManager:
    """Manages the application of FX compilation passes.
    
    Applies passes in a specific order and provides utilities
    for graph analysis and transformation.
    """
    
    def __init__(self, config: CompileConfig):
        self.config = config
        self.passes = self._build_pass_pipeline()
    
    def _build_pass_pipeline(self):
        """Build the ordered list of compilation passes.
        
        Pass order matters: earlier passes modify the graph
        in ways that later passes depend on.
        """
        pipeline = []
        
        # 1. Shape propagation (always first)
        pipeline.append(ShapePropagationPass())
        
        # 2. Memory estimation
        pipeline.append(MemoryEstimationPass())
        
        # 3. Activation management passes
        if self.config.free_activation:
            pipeline.append(FreeActivationPass(
                threshold=self.config.free_activation_threshold
            ))
        
        if self.config.offload_activation:
            pipeline.append(OffloadActivationPass())
        
        # 4. Distributed passes
        if self.config.double_buffer:
            pipeline.append(DoubleBufferPass())
        
        if self.config.symmetric_memory:
            pipeline.append(SymmetricMemoryPass())
        
        # 5. Synchronization passes
        if self.config.sync_before_reduce or self.config.sync_after_reduce:
            pipeline.append(SyncPass(
                before_reduce=self.config.sync_before_reduce,
                after_reduce=self.config.sync_after_reduce,
                before_allgather=self.config.sync_before_allgather,
                after_allgather=self.config.sync_after_allgather,
            ))
        
        # 6. ZeRO-specific passes
        if self.config.passes:
            for pass_name, pass_config in self.config.passes.items():
                if pass_name == "z1":
                    pipeline.append(ZeRO1Pass(**pass_config))
                elif pass_name == "z3":
                    pipeline.append(ZeRO3Pass(**pass_config))
                elif pass_name == "autosp":
                    pipeline.append(AutoSPPass(**pass_config))
        
        # 7. Dead code elimination (always last)
        pipeline.append(DeadCodeEliminationPass())
        
        return pipeline
    
    def apply(self, graph_module: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """Apply all passes to the FX graph.
        
        Args:
            graph_module: The captured FX graph
        
        Returns:
            torch.fx.GraphModule: The transformed graph
        """
        for pass_fn in self.passes:
            if self.config.debug_log:
                logger.info(f"Applying pass: {pass_fn.__class__.__name__}")
                logger.info(f"  Nodes before: {len(graph_module.graph.nodes)}")
            
            graph_module = pass_fn(graph_module)
            
            if self.config.debug_log:
                logger.info(f"  Nodes after: {len(graph_module.graph.nodes)}")
        
        graph_module.recompile()
        return graph_module
```

### ShapePropagationPass

```python
class ShapePropagationPass:
    """Propagate tensor shapes through the FX graph.
    
    Uses fake tensors to determine output shapes for every
    operation in the graph. This information is required by
    subsequent passes for memory estimation.
    """
    
    def __call__(self, graph_module):
        from torch.fx.passes.shape_prop import ShapeProp
        
        # Create fake tensors for inputs
        with torch._subclasses.fake_tensor.FakeTensorMode():
            ShapeProp(graph_module).propagate(
                *self._create_fake_inputs(graph_module)
            )
        
        return graph_module
```

### MemoryEstimationPass

```python
class MemoryEstimationPass:
    """Estimate memory usage for each activation tensor.
    
    Annotates each node in the graph with memory metadata:
    - tensor_size: Size in bytes
    - is_activation: Whether this is a forward-pass activation
    - is_parameter: Whether this is a model parameter
    - peak_memory: Peak memory at this point in the graph
    """
    
    def __call__(self, graph_module):
        running_memory = 0
        peak_memory = 0
        
        for node in graph_module.graph.nodes:
            if node.op == "placeholder":
                # Input tensor
                size = self._estimate_tensor_size(node)
                node.meta["tensor_size"] = size
                node.meta["is_activation"] = True
                running_memory += size
            
            elif node.op in ("call_module", "call_function", "call_method"):
                size = self._estimate_tensor_size(node)
                node.meta["tensor_size"] = size
                node.meta["is_activation"] = True
                running_memory += size
                peak_memory = max(peak_memory, running_memory)
            
            # Track when tensors are no longer needed
            self._update_lifetime(node)
        
        graph_module.meta["peak_memory"] = peak_memory
        graph_module.meta["total_activation_memory"] = running_memory
        
        return graph_module
```

### FreeActivationPass

```python
class FreeActivationPass:
    """Free activation tensors during the forward pass.
    
    For each activation tensor:
    1. Check if it exceeds the size threshold
    2. Identify when it is last consumed
    3. Insert a free/delete operation after last use
    4. Insert a recomputation before it's needed in backward
    
    This pass modifies the FX graph to:
    - Delete tensors after their last forward-pass use
    - Recompute them during the backward pass
    
    The recomputation strategy uses the "checkpoint" pattern:
    only the inputs to each subgraph need to be kept, and
    intermediate activations are recomputed from these inputs.
    """
    
    def __init__(self, threshold=10 * 1024 * 1024):
        """
        Args:
            threshold (int): Minimum tensor size in bytes to free.
                             Default: 10 MB
        """
        self.threshold = threshold
    
    def __call__(self, graph_module):
        # 1. Analyze tensor lifetimes
        lifetimes = self._analyze_lifetimes(graph_module)
        
        # 2. Identify freeable tensors
        freeable = self._find_freeable_tensors(graph_module, lifetimes)
        
        # 3. Insert free operations
        for node_name, last_use in freeable.items():
            node = self._find_node(graph_module, node_name)
            if node is not None:
                self._insert_free_op(graph_module, node, last_use)
        
        # 4. Insert recomputation for backward pass
        self._insert_recomputation(graph_module, freeable)
        
        return graph_module
    
    def _analyze_lifetimes(self, graph_module):
        """Determine when each tensor is first and last used."""
        lifetimes = {}
        for i, node in enumerate(graph_module.graph.nodes):
            for user in node.users:
                if node.name not in lifetimes:
                    lifetimes[node.name] = {"first": i, "last": i}
                else:
                    lifetimes[node.name]["last"] = max(
                        lifetimes[node.name]["last"], i
                    )
        return lifetimes
    
    def _find_freeable_tensors(self, graph_module, lifetimes):
        """Identify tensors that should be freed.
        
        Criteria:
        1. Tensor size > threshold
        2. Tensor is an activation (not a parameter or constant)
        3. Tensor has a long lifetime (not immediately consumed)
        """
        freeable = {}
        for node in graph_module.graph.nodes:
            if node.meta.get("is_activation", False):
                size = node.meta.get("tensor_size", 0)
                if size >= self.threshold:
                    lifetime = lifetimes.get(node.name)
                    if lifetime and (lifetime["last"] - lifetime["first"]) > 1:
                        freeable[node.name] = lifetime
        return freeable
    
    def _insert_free_op(self, graph_module, node, last_use):
        """Insert a tensor.free() call after the last use of a node.
        
        This is implemented by inserting a custom op that releases
        the tensor's GPU memory.
        """
        with graph_module.graph.inserting_after(last_use):
            graph_module.graph.call_function(
                deepspeed_compile_free_tensor,
                args=(node,),
            )
```

### OffloadActivationPass

```python
class OffloadActivationPass:
    """Offload activation tensors to CPU memory during forward pass.
    
    For each eligible activation:
    1. Asynchronously copy the tensor from GPU to CPU
    2. Free the GPU copy
    3. During backward, asynchronously copy back from CPU to GPU
    
    This pass is more efficient than free_activation when:
    - PCIe bandwidth is sufficient for the activation sizes
    - CPU memory is available
    - Recomputation cost exceeds transfer cost
    """
    
    def __call__(self, graph_module):
        # 1. Identify offloadable activations
        offloadable = self._find_offloadable(graph_module)
        
        # 2. Insert GPU->CPU copy after forward computation
        for node_name, info in offloadable.items():
            node = self._find_node(graph_module, node_name)
            if node is not None:
                self._insert_offload(graph_module, node)
        
        # 3. Insert CPU->GPU copy before backward consumption
        for node_name, info in offloadable.items():
            node = self._find_node(graph_module, node_name)
            if node is not None:
                self._insert_reload(graph_module, node)
        
        return graph_module
    
    def _insert_offload(self, graph_module, node):
        """Insert async copy: GPU -> CPU + GPU free."""
        with graph_module.graph.inserting_after(node):
            # Async copy to CPU
            cpu_copy = graph_module.graph.call_function(
                deepspeed_compile_offload_to_cpu,
                args=(node,),
            )
            # Free GPU copy
            graph_module.graph.call_function(
                deepspeed_compile_free_tensor,
                args=(node,),
            )
    
    def _insert_reload(self, graph_module, node):
        """Insert async copy: CPU -> GPU before backward use."""
        # Find the backward computation that needs this tensor
        for user in list(node.users):
            if self._is_backward_op(user):
                with graph_module.graph.inserting_before(user):
                    gpu_copy = graph_module.graph.call_function(
                        deepspeed_compile_reload_from_cpu,
                        args=(node,),
                    )
                    # Replace the original tensor reference
                    user.replace_input_with(node, gpu_copy)
```

---

## PyTorch Inductor Integration

DeepCompile integrates with PyTorch's Inductor backend for low-level optimization after applying its own graph-level passes.

### DeepSpeedInductor

```python
# deepspeed/compile/inductor.py

class DeepSpeedInductor:
    """Integration layer between DeepCompile and TorchInductor.
    
    Handles:
    - Converting FX graph to Inductor-compatible format
    - Passing DeepSpeed-specific compilation hints
    - Managing custom ops that Inductor doesn't handle
    """
    
    def __init__(self, config: CompileConfig):
        self.config = config
    
    def compile(self, graph_module, example_inputs):
        """Compile the FX graph using TorchInductor.
        
        Args:
            graph_module: FX GraphModule (after DeepCompile passes)
            example_inputs: Example input tensors for shape inference
        
        Returns:
            Compiled callable
        """
        from torch._inductor import compile as inductor_compile
        
        # Inductor compilation options
        options = {
            "triton.cudagraphs": False,  # Disable CUDA graphs (conflicts with DS)
            "shape_padding": True,        # Pad shapes for better kernel utilization
        }
        
        # Compile
        compiled = inductor_compile(
            graph_module,
            example_inputs,
            options=options,
        )
        
        return compiled
```

### DeepCompileBackend

```python
# deepspeed/compile/backend.py

class DeepCompileBackend:
    """TorchDynamo backend for DeepSpeed compilation.
    
    Registered as a custom backend for torch.compile():
    
        model = torch.compile(model, backend="deepspeed")
    
    Or used implicitly when DeepCompile is enabled in ds_config.
    """
    
    def __init__(self, compile_config: CompileConfig):
        self.config = compile_config
        self.pass_manager = FXPassManager(compile_config)
        self.inductor = DeepSpeedInductor(compile_config)
    
    def __call__(self, graph_module, example_inputs):
        """Compile the FX graph.
        
        Called by TorchDynamo when it captures a new subgraph.
        
        Args:
            graph_module: The captured FX GraphModule
            example_inputs: Example input tensors
        
        Returns:
            Compiled callable
        """
        if self.config.debug_log:
            self._log_graph(graph_module, "Before DeepCompile passes")
        
        # 1. Apply DeepCompile passes
        graph_module = self.pass_manager.apply(graph_module)
        
        if self.config.debug_log:
            self._log_graph(graph_module, "After DeepCompile passes")
        
        # 2. Delegate to Inductor for low-level optimization
        compiled = self.inductor.compile(graph_module, example_inputs)
        
        return compiled
    
    def _log_graph(self, graph_module, label):
        """Log the FX graph for debugging."""
        logger.info(f"=== {label} ===")
        logger.info(f"Nodes: {len(list(graph_module.graph.nodes))}")
        for node in graph_module.graph.nodes:
            size = node.meta.get("tensor_size", 0)
            logger.info(f"  {node.name}: op={node.op}, "
                        f"target={node.target}, size={size} bytes")
```

### Backend Registration

```python
# Registration with TorchDynamo
from torch._dynamo import register_backend

def deepspeed_backend_factory(compile_config):
    """Create a DeepCompile backend with the given configuration."""
    return DeepCompileBackend(compile_config)

# Users can then use:
# model = torch.compile(model, backend="deepspeed")
```

---

## Activation Memory Optimization

DeepCompile provides two complementary strategies for reducing activation memory: **free_activation** and **offload_activation**. Both operate on the same principle -- reducing GPU memory footprint of forward-pass activations -- but use different mechanisms.

### Strategy Comparison

| Strategy | Mechanism | Memory Saving | Compute Overhead | Bandwidth | Use Case |
|----------|-----------|---------------|------------------|-----------|----------|
| `free_activation` | Delete + recompute | 50-80% | ~30% more FLOPs | None | Compute-rich, memory-poor |
| `offload_activation` | Copy to CPU | 50-80% | Minimal | PCIe bandwidth | CPU memory available |
| Both | Hybrid: free small, offload large | 60-90% | Moderate | Partial | Best of both |

### Free Activation Mode

```python
# Enable in config
config = CompileConfig(
    deepcompile=True,
    free_activation=True,
    free_activation_threshold=10 * 1024 * 1024,  # 10 MB
)
```

**How it works**:

```
Standard Forward:
  Layer 0: compute -> keep activation_0 (1.5 GB)
  Layer 1: compute -> keep activation_1 (1.5 GB)
  ...
  Layer 31: compute -> keep activation_31 (1.5 GB)
  Total activation memory: 48 GB

Free Activation Forward:
  Layer 0: compute -> keep activation_0 -> free after Layer 1 uses it
  Layer 1: compute -> keep activation_1 -> free after Layer 2 uses it
  ...
  Layer 31: compute -> keep activation_31
  Total activation memory: ~3 GB (2-3 layers at a time)

Backward (with recomputation):
  Layer 31: backward using activation_31
  Layer 30: recompute activation_30 -> backward -> free
  ...
  Layer 0: recompute activation_0 -> backward -> free
```

### Offload Activation Mode

```python
# Enable in config
config = CompileConfig(
    deepcompile=True,
    offload_activation=True,
)
```

**How it works**:

```
Standard Forward:
  Layer 0: compute -> keep activation_0 on GPU (1.5 GB)
  Layer 1: compute -> keep activation_1 on GPU (1.5 GB)
  ...
  Total GPU activation memory: 48 GB

Offload Forward:
  Layer 0: compute -> async copy to CPU -> free GPU copy
  Layer 1: compute -> async copy to CPU -> free GPU copy
  ...
  Total GPU activation memory: ~0 GB (all on CPU)

Offload Backward:
  Layer 31: async copy activation_31 CPU->GPU -> backward -> free
  Layer 30: async copy activation_30 CPU->GPU -> backward -> free
  ...
```

### Hybrid Strategy

For maximum memory savings with controlled overhead:

```python
config = CompileConfig(
    deepcompile=True,
    free_activation=True,
    free_activation_threshold=50 * 1024 * 1024,  # Free activations > 50 MB
    offload_activation=True,                       # Offload smaller activations to CPU
)
```

This hybrid approach:
1. **Offloads** small-to-medium activations to CPU (low compute cost to recompute, but saves PCIe bandwidth)
2. **Frees** large activations entirely (recomputes during backward, saves both GPU and CPU memory)

### Memory Savings Estimates

For a 7B parameter transformer model (batch=32, seq=2048, FP16):

| Configuration | Activation Memory | Total GPU Memory | Notes |
|--------------|------------------|-----------------|-------|
| Baseline (no optimization) | 48 GB | 62 GB | 14 GB params + 48 GB activations |
| free_activation | 8 GB | 22 GB | ~83% activation reduction |
| offload_activation | 4 GB | 18 GB | Offloaded to CPU |
| free + offload | 3 GB | 17 GB | Best case |

---

## Double Buffering for Pipeline Parallelism

Double buffering is a technique that overlaps computation with communication in pipeline-parallel training to reduce pipeline bubbles.

### The Pipeline Bubble Problem

In standard pipeline parallelism with P stages:

```
Stage 0: [F0][F1][F2][F3]..............[B3][B2][B1][B0]
Stage 1: ....[F0][F1][F2][F3]..........[B3][B2][B1][B0]
Stage 2: ........[F0][F1][F2][F3]......[B3][B2][B1][B0]
Stage 3: ............[F0][F1][F2][F3][B3][B2][B1][B0]

F = Forward, B = Backward
The "bubble" is idle time between last forward and first backward.
Bubble fraction ~ (P-1) / P for P pipeline stages.
```

### How Double Buffering Helps

```python
# Enable in config
config = CompileConfig(
    deepcompile=True,
    double_buffer=True,
)
```

Double buffering splits the activation memory into two buffers:

```
Buffer A: Holds activations for micro-batch 0, 2, 4, ...
Buffer B: Holds activations for micro-batch 1, 3, 5, ...

While Buffer A is being processed (forward/backward):
  - Buffer B is being sent/received over network

This overlaps:
  - Computation on Buffer A with communication for Buffer B
  - And vice versa
```

### Implementation

```python
class DoubleBufferPass:
    """FX pass that implements double buffering for pipeline parallelism.
    
    Modifies the graph to:
    1. Allocate two activation buffers instead of one
    2. Alternate between buffers for consecutive micro-batches
    3. Overlap buffer swaps with computation
    """
    
    def __call__(self, graph_module):
        # 1. Find activation tensors
        activations = self._find_activations(graph_module)
        
        # 2. Create double buffer
        buffer_a, buffer_b = self._create_double_buffer(graph_module, activations)
        
        # 3. Modify forward to alternate buffers
        self._insert_buffer_selection(graph_module, buffer_a, buffer_b)
        
        # 4. Insert async swap operations
        self._insert_async_swaps(graph_module, buffer_a, buffer_b)
        
        return graph_module
```

### Performance Impact

| Pipeline Stages | Without Double Buffer | With Double Buffer | Speedup |
|----------------|----------------------|-------------------|---------|
| 4 | 25% bubble | 12% bubble | 1.15x |
| 8 | 43% bubble | 20% bubble | 1.40x |
| 16 | 56% bubble | 28% bubble | 1.64x |

---

## Symmetric Memory for Homogeneous Clusters

Symmetric memory optimization assumes all GPUs in the cluster have identical memory capacity, enabling more aggressive distributed memory management.

### Configuration

```python
config = CompileConfig(
    deepcompile=True,
    symmetric_memory=True,
)
```

### How It Works

In a homogeneous cluster, DeepCompile can:

1. **Distribute activations evenly**: Split activation storage across GPUs so each holds an equal share
2. **Collective offloading**: When one GPU offloads, all GPUs offload simultaneously (balanced CPU usage)
3. **Pipelined transfers**: Overlap activation transfers between GPU pairs

```python
class SymmetricMemoryPass:
    """FX pass for symmetric memory management.
    
    Assumes all GPUs have identical memory and computes
    a globally optimal activation management strategy.
    """
    
    def __init__(self):
        self.world_size = torch.distributed.get_world_size()
        self.rank = torch.distributed.get_rank()
    
    def __call__(self, graph_module):
        # 1. Estimate total activation memory
        total_memory = graph_module.meta.get("total_activation_memory", 0)
        
        # 2. Divide equally across GPUs
        memory_per_gpu = total_memory // self.world_size
        
        # 3. Assign each activation to a specific GPU
        assignments = self._assign_activations(graph_module, memory_per_gpu)
        
        # 4. Insert send/recv for cross-GPU activation access
        self._insert_communication(graph_module, assignments)
        
        return graph_module
    
    def _assign_activations(self, graph_module, budget):
        """Assign activations to GPUs to balance memory.
        
        Uses a greedy bin-packing algorithm to distribute
        activations across GPUs.
        """
        activations = []
        for node in graph_module.graph.nodes:
            if node.meta.get("is_activation", False):
                activations.append((node.name, node.meta["tensor_size"]))
        
        # Sort by size (largest first for better packing)
        activations.sort(key=lambda x: x[1], reverse=True)
        
        # Assign to GPUs
        gpu_memory = [0] * self.world_size
        assignments = {}
        
        for name, size in activations:
            # Assign to GPU with least memory usage
            target_gpu = gpu_memory.index(min(gpu_memory))
            gpu_memory[target_gpu] += size
            assignments[name] = target_gpu
        
        return assignments
```

---

## Integration with ZeRO Stages

DeepCompile coordinates with ZeRO optimization stages to ensure activation management does not conflict with parameter and optimizer state management.

### ZeRO Stage 1 Integration

ZeRO Stage 1 partitions optimizer states. DeepCompile can offload activation tensors to the memory freed by optimizer state partitioning.

```python
# ZeRO-1 + DeepCompile config
config = CompileConfig(
    deepcompile=True,
    free_activation=True,
    passes={
        "z1": {
            "enabled": True,
            "use_freed_memory": True,  # Use memory freed by optimizer partitioning
        }
    }
)
```

### ZeRO Stage 3 Integration

ZeRO Stage 3 partitions parameters, gradients, and optimizer states. DeepCompile must coordinate parameter gathering with activation management.

```python
# ZeRO-3 + DeepCompile config
config = CompileConfig(
    deepcompile=True,
    free_activation=True,
    offload_activation=True,
    sync_before_allgather=True,  # Ensure params are gathered before use
    passes={
        "z3": {
            "enabled": True,
            "prefetch_distance": 2,      # Prefetch params 2 layers ahead
            "overlap_gather": True,       # Overlap param gathering with compute
            "reduce_scatter": True,       # Use reduce-scatter for gradients
        }
    }
)
```

### ZeRO-3 Pass Detail

```python
class ZeRO3Pass:
    """ZeRO Stage 3 specific optimizations.
    
    Coordinates:
    - Parameter all-gather with activation lifecycle
    - Gradient reduce-scatter scheduling
    - Parameter prefetching during activation recomputation
    """
    
    def __init__(self, enabled=True, prefetch_distance=2, 
                 overlap_gather=True, reduce_scatter=True, **kwargs):
        self.enabled = enabled
        self.prefetch_distance = prefetch_distance
        self.overlap_gather = overlap_gather
        self.reduce_scatter = reduce_scatter
    
    def __call__(self, graph_module):
        if not self.enabled:
            return graph_module
        
        # 1. Identify parameter all-gather points
        gather_points = self._find_allgather_points(graph_module)
        
        # 2. Schedule prefetching
        for i, point in enumerate(gather_points):
            # Prefetch parameters for layer (i + prefetch_distance)
            prefetch_target = min(i + self.prefetch_distance, len(gather_points) - 1)
            self._insert_prefetch(graph_module, point, gather_points[prefetch_target])
        
        # 3. Overlap gather with activation recomputation
        if self.overlap_gather:
            self._overlap_gather_with_recompute(graph_module)
        
        return graph_module
```

### AutoSP Pass

The AutoSP (Automatic Sequence Parallelism) pass detects opportunities for sequence-parallel computation:

```python
class AutoSPPass:
    """Automatic Sequence Parallelism detection and insertion.
    
    Detects operations that can be parallelized along the sequence
    dimension (e.g., LayerNorm, Dropout) and inserts necessary
    communication primitives.
    """
    
    def __init__(self, enabled=True, min_seq_length=4096, **kwargs):
        self.enabled = enabled
        self.min_seq_length = min_seq_length
    
    def __call__(self, graph_module):
        if not self.enabled:
            return graph_module
        
        # Check if sequence length is large enough to benefit
        seq_length = self._get_seq_length(graph_module)
        if seq_length < self.min_seq_length:
            return graph_module
        
        # Find sequence-parallel operations
        sp_ops = self._find_sp_operations(graph_module)
        
        # Insert reduce-scatter and all-gather around SP regions
        for op in sp_ops:
            self._insert_sp_communication(graph_module, op)
        
        return graph_module
```

---

## Custom Passes

DeepCompile provides an extensible framework for adding custom compilation passes.

### Creating a Custom Pass

```python
# custom_pass.py

import torch.fx

class MyCustomPass:
    """Example custom compilation pass.
    
    Implements a custom optimization by modifying the FX graph.
    """
    
    def __init__(self, enabled=True, my_param=42):
        self.enabled = enabled
        self.my_param = my_param
    
    def __call__(self, graph_module: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """Apply the custom pass to the FX graph.
        
        Args:
            graph_module: The FX graph to transform
        
        Returns:
            torch.fx.GraphModule: The transformed graph
        """
        if not self.enabled:
            return graph_module
        
        # Example: Find all linear layers and add a fusion hint
        for node in graph_module.graph.nodes:
            if node.op == "call_module":
                module = getattr(graph_module, node.target)
                if isinstance(module, torch.nn.Linear):
                    node.meta["fusion_hint"] = "prefer_fusion"
        
        # Must recompile after modifying the graph
        graph_module.recompile()
        return graph_module
```

### Registering a Custom Pass

```python
# Method 1: Via configuration
config = CompileConfig(
    deepcompile=True,
    passes={
        "my_pass": {
            "enabled": True,
            "my_param": 42,
        }
    }
)

# Method 2: Directly add to pass pipeline
from deepspeed.compile.fx import FXPassManager

# Extend the pass pipeline
manager = FXPassManager(config)
manager.passes.append(MyCustomPass(enabled=True, my_param=42))
```

### Custom Pass API Reference

```python
class CustomPass:
    """Base class for custom compilation passes.
    
    Subclasses must implement __call__.
    """
    
    def __call__(self, graph_module: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """Transform the FX graph.
        
        Args:
            graph_module: Input FX graph
        
        Returns:
            torch.fx.GraphModule: Transformed FX graph
        
        Guidelines:
        1. Always return a valid GraphModule
        2. Call graph_module.recompile() after structural changes
        3. Preserve node.meta['tensor_size'] annotations
        4. Don't remove output nodes
        """
        raise NotImplementedError
```

### FX Graph Manipulation API

```python
# Key FX graph manipulation operations:

# 1. Insert a new node after another node
with graph_module.graph.inserting_after(target_node):
    new_node = graph_module.graph.call_function(
        torch.ops.my_op,
        args=(input_node,),
        kwargs={"param": value},
    )

# 2. Insert a new node before another node
with graph_module.graph.inserting_before(target_node):
    new_node = graph_module.graph.call_function(
        my_function,
        args=(input_node,),
    )

# 3. Replace a node's uses with another node
old_node.replace_all_uses_with(new_node)

# 4. Replace a specific input
user_node.replace_input_with(old_input, new_input)

# 5. Erase a node
graph_module.graph.erase_node(node)

# 6. Create a new module attribute
graph_module.add_module("new_param", nn.Parameter(torch.empty(10)))

# 7. Get a node by name
node = next(n for n in graph_module.graph.nodes if n.name == "target_name")
```

---

## Configuration Examples

### Basic Activation Freeing

```json
{
    "compile": {
        "deepcompile": true,
        "free_activation": true,
        "free_activation_threshold": 10485760
    },
    "fp16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2
    }
}
```

### Full DeepCompile with All Features

```json
{
    "compile": {
        "deepcompile": true,
        "free_activation": true,
        "free_activation_threshold": 52428800,
        "offload_activation": true,
        "offload_opt_states": false,
        "double_buffer": false,
        "symmetric_memory": true,
        "sync_before_reduce": false,
        "sync_after_reduce": false,
        "sync_before_allgather": true,
        "sync_after_allgather": true,
        "keep_int_input_tensors": true,
        "debug_log": false,
        "passes": {
            "z3": {
                "enabled": true,
                "prefetch_distance": 2,
                "overlap_gather": true
            },
            "autosp": {
                "enabled": true,
                "min_seq_length": 4096
            }
        }
    },
    "train_batch_size": 64,
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "initial_scale_power": 16
    },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e5
    }
}
```

### DeepCompile with Pipeline Parallelism

```json
{
    "compile": {
        "deepcompile": true,
        "free_activation": true,
        "double_buffer": true
    },
    "pipeline": {
        "enabled": true,
        "parallel_size": 4,
        "micro_batches": 8,
        "gradient_accumulation_steps": 8
    },
    "fp16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 1
    }
}
```

### Activation Offloading (Memory-Constrained)

```json
{
    "compile": {
        "deepcompile": true,
        "offload_activation": true,
        "free_activation": true,
        "free_activation_threshold": 104857600
    },
    "fp16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        }
    }
}
```

### Debug Configuration

```json
{
    "compile": {
        "deepcompile": true,
        "free_activation": true,
        "debug_log": true,
        "keep_all_input_tensors": true,
        "sync_before_reduce": true,
        "sync_after_reduce": true
    },
    "fp16": {
        "enabled": true
    }
}
```

### Usage in Python

```python
import deepspeed
import torch

# Method 1: Enable via DeepSpeed config
ds_config = {
    "compile": {
        "deepcompile": True,
        "free_activation": True,
        "free_activation_threshold": 10 * 1024 * 1024,
    },
    "fp16": {"enabled": True},
    "zero_optimization": {"stage": 2},
}

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    config=ds_config,
)

# Method 2: Use torch.compile with DeepSpeed backend
from deepspeed.compile import deepspeed_backend

model = torch.compile(model, backend=deepspeed_backend(ds_config))
```

### Accessing CompileConfig Programmatically

```python
from deepspeed.compile.config import CompileConfig

# Create config
config = CompileConfig(
    deepcompile=True,
    free_activation=True,
    offload_activation=True,
    debug_log=True,
)

# Access fields
print(f"Activation threshold: {config.free_activation_threshold / 1e6:.1f} MB")
print(f"Free activation: {config.free_activation}")
print(f"Offload activation: {config.offload_activation}")

# Create from dict
config = CompileConfig(**{
    "deepcompile": True,
    "free_activation": True,
})
```

---

## Troubleshooting

### DeepCompile Not Activating

**Symptom**: Compilation passes are not being applied; memory usage is unchanged.

**Solutions**:
1. Ensure `deepcompile: true` is set in the configuration
2. Verify PyTorch version supports `torch.compile` (>= 2.0)
3. Check that no other `torch.compile` call is overriding DeepSpeed's backend:
   ```python
   # Don't do this:
   model = torch.compile(model, backend="inductor")  # Overrides DeepSpeed
   ```

### FX Graph Capture Failures

**Symptom**: `torch.fx.TraceError` or `unsupported operator` errors

**Solutions**:
1. Some operations are not FX-traceable. Use `torch.fx.wrap()`:
   ```python
   # Mark custom function as FX-compatible
   torch.fx.wrap('my_custom_function')
   ```
2. Use `keep_all_input_tensors: true` if input tensors are being lost
3. Check for dynamic control flow that FX cannot trace:
   ```python
   # This cannot be traced (data-dependent control flow):
   if tensor.sum() > 0:  # FX can't trace this
       output = fn1(tensor)
   else:
       output = fn2(tensor)
   ```

### Recomputation Errors (free_activation)

**Symptom**: `RuntimeError: Trying to backward through the graph a second time` or shape mismatches during backward

**Solutions**:
1. Increase `free_activation_threshold` to reduce the number of freed tensors
2. Ensure no in-place operations are modifying freed tensors
3. Disable for specific layers:
   ```python
   # Mark layers that should not have activations freed
   layer.train().requires_grad_(True)
   layer._ds_no_free_activation = True
   ```

### Offloading Performance Issues

**Symptom**: Training is slower with `offload_activation` enabled

**Solutions**:
1. Check PCIe bandwidth: `nvidia-sma topo -m` to verify GPU-CPU connectivity
2. Reduce offloaded activation count by increasing threshold
3. Use hybrid strategy: free large activations, offload small ones
4. Enable pinned memory for faster GPU-CPU transfers:
   ```json
   {
       "compile": {
           "offload_activation": true
       },
       "zero_optimization": {
           "offload_param": {
               "pin_memory": true
           }
       }
   }
   ```

### Inductor Compatibility Issues

**Symptom**: `torch._inductor` errors or kernel generation failures

**Solutions**:
1. Update PyTorch to the latest stable version
2. Try disabling Inductor optimizations:
   ```python
   import torch._inductor.config
   torch._inductor.config.triton.unique_kernel_names = True
   ```
3. Check for unsupported custom ops that Inductor cannot handle

### Pipeline Parallelism + Double Buffer Issues

**Symptom**: Deadlocks or incorrect results with `double_buffer: true`

**Solutions**:
1. Ensure pipeline parallelism is properly configured
2. Verify that micro-batch count is even (for double buffer alternation)
3. Check that the pipeline engine supports the double buffer protocol

### ZeRO-3 + DeepCompile Conflicts

**Symptom**: Parameter gathering failures or incorrect gradient computation

**Solutions**:
1. Enable `sync_before_allgather` and `sync_after_allgather`:
   ```json
   {
       "compile": {
           "sync_before_allgather": true,
           "sync_after_allgather": true,
           "passes": {
               "z3": {"enabled": true, "overlap_gather": false}
           }
       }
   }
   ```
2. Disable overlap gathering if stability is a concern
3. Reduce `prefetch_distance` to 1 for more conservative scheduling

### Debug Logging

Enable detailed logging to diagnose issues:

```python
import logging
logging.getLogger("deepspeed.compile").setLevel(logging.DEBUG)
```

Or via configuration:
```json
{
    "compile": {
        "debug_log": true
    }
}
```

This logs:
- FX graph structure before and after each pass
- Activation sizes and memory estimates
- Pass execution order and timing
- Synchronization point insertion
- Offload/reload operations

---

## Performance Tuning Guide

### Choosing the Right Strategy

| Scenario | Recommended Config | Rationale |
|----------|-------------------|-----------|
| **Large model, limited GPU memory** | `free_activation=True` | Trade compute for memory |
| **Large model, CPU memory available** | `offload_activation=True` | Avoid recomputation |
| **Pipeline parallelism** | `double_buffer=True` | Hide pipeline bubbles |
| **Homogeneous cluster** | `symmetric_memory=True` | Balanced memory usage |
| **ZeRO-3 training** | `free_activation=True` + `passes.z3` | Coordinate with param partitioning |
| **Long sequences** | `passes.autosp` | Automatic sequence parallelism |

### Threshold Tuning

The `free_activation_threshold` controls the aggressiveness of activation freeing:

| Threshold | Effect | Recommended For |
|-----------|--------|----------------|
| 1 MB | Very aggressive: free almost all activations | Extreme memory pressure |
| 10 MB | Default: balance memory and compute | General training |
| 50 MB | Conservative: free only large activations | When compute is expensive |
| 100 MB | Very conservative | Minimize recomputation |

### Estimating Memory Savings

```python
# Estimate activation memory before enabling DeepCompile
def estimate_activation_memory(model, batch_size, seq_length, hidden_size, 
                                num_layers, bytes_per_element=2):
    """Rough estimate of activation memory per transformer layer."""
    per_layer = batch_size * seq_length * hidden_size * bytes_per_element
    # Each layer produces ~4 major activations (attn input, attn output, mlp input, mlp output)
    total = per_layer * 4 * num_layers
    return total

# Example: LLaMA-7B
memory = estimate_activation_memory(
    model=None, batch_size=32, seq_length=2048,
    hidden_size=4096, num_layers=32, bytes_per_element=2
)
print(f"Estimated activation memory: {memory / 1e9:.1f} GB")
# Output: Estimated activation memory: 68.7 GB

# With free_activation, expect ~80% reduction:
print(f"After free_activation: {memory * 0.2 / 1e9:.1f} GB")
# Output: After free_activation: 13.7 GB
```

---

## Quick Reference

```python
from deepspeed.compile.config import CompileConfig

# Minimal config for activation freeing
config = CompileConfig(
    deepcompile=True,
    free_activation=True,
)

# Full config
config = CompileConfig(
    deepcompile=True,
    free_activation=True,
    free_activation_threshold=50 * 1024 * 1024,  # 50 MB
    offload_activation=True,
    offload_opt_states=False,
    double_buffer=False,
    symmetric_memory=True,
    sync_before_reduce=False,
    sync_after_reduce=False,
    sync_before_allgather=True,
    sync_after_allgather=True,
    keep_int_input_tensors=True,
    keep_all_input_tensors=False,
    debug_log=False,
    passes={
        "z3": {"enabled": True, "prefetch_distance": 2},
        "autosp": {"enabled": True, "min_seq_length": 4096},
    },
)
```
