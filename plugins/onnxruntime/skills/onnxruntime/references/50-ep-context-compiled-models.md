# ONNX Runtime Reference - Chapter 50: EP Context and Compiled Models

This chapter covers ONNX Runtime's EP Context feature, which enables pre-compiled models for faster session creation, reduced warmup time, and improved deployment performance.

---

## 50.1 EP Context Model Concept

### 50.1.1 Overview

EP Context is a mechanism that allows Execution Providers to serialize their compiled/optimized state into the model file. When the model is loaded again, the EP can directly load the pre-compiled artifacts instead of recompiling, significantly reducing session initialization time.

```
Without EP Context:
  Load Model → Graph Partitioning → EP Compilation → Kernel Setup → Ready
  Time: ~30 seconds (for large models)

With EP Context:
  Load Model → Load Pre-Compiled EP Context → Ready
  Time: ~1 second
```

### 50.1.2 Use Cases

1. **Production deployment**: Pre-compile models during the build pipeline, deploy pre-compiled artifacts
2. **Edge devices**: Avoid on-device compilation overhead
3. **Serverless**: Faster cold starts with pre-compiled models
4. **Reproducibility**: Exact same compiled artifact across deployments
5. **Cost savings**: Reduce GPU compute time during deployment

### 50.1.3 High-Level Workflow

```
Step 1: Generate EP Context (one-time)
┌────────────────────────────────────────────┐
│  Source Model (.onnx)                       │
│  + Session Options (ep.context_enable=1)    │
│  + EP Configuration                         │
│         ↓                                   │
│  Session Creation (triggers compilation)    │
│         ↓                                   │
│  EP Context Model (.onnx with EPContext)    │
│  + External compiled binary (optional)      │
└────────────────────────────────────────────┘

Step 2: Load Pre-Compiled Model (every deployment)
┌────────────────────────────────────────────┐
│  EP Context Model (.onnx)                   │
│         ↓                                   │
│  Session Creation (loads pre-compiled data) │
│         ↓                                   │
│  Ready for inference                        │
└────────────────────────────────────────────┘
```

---

## 50.2 EP Context Configuration

### 50.2.1 Configuration Options

```python
import onnxruntime as ort

options = ort.SessionOptions()

# Enable EP context generation
options.add_config_entry("ep.context_enable", "1")

# Path to save the EP context file (external)
options.add_config_entry("ep.context_file_path", "/path/to/compiled_context.bin")

# Embed mode:
# 0 = EP context saved as external file
# 1 = EP context embedded in the ONNX model
options.add_config_entry("ep.context_embed_mode", "1")

# Session options
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
```

### 50.2.2 All Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `ep.context_enable` | string ("0"/"1") | "0" | Enable EP context generation |
| `ep.context_file_path` | string | "" | Path for external context file |
| `ep.context_embed_mode` | string ("0"/"1") | "0" | Embed mode (0=external, 1=embedded) |
| `ep.disable_model_compile` | string ("0"/"1") | "0" | Disable compilation, only load |
| `ep.fail_on_suboptimal_compiled_model` | string ("0"/"1") | "0" | Fail if pre-compiled model is suboptimal |

### 50.2.3 C++ API Configuration

```cpp
// C++ configuration for EP context
Ort::SessionOptions options;

// Enable EP context generation
options.AddConfigEntry("ep.context_enable", "1");
options.AddConfigEntry("ep.context_file_path", "/path/to/context.bin");
options.AddConfigEntry("ep.context_embed_mode", "0");  // External file

// Register EP (must be done before session creation)
OrtCUDAProviderOptions cuda_options;
cuda_options.device_id = 0;
options.AppendExecutionProvider_CUDA(cuda_options);

// Create session (triggers compilation and saves EP context)
Ort::Session session(env, "model.onnx", options);
// After this, the EP context is saved to the specified path
```

### 50.2.4 C API Configuration

```c
// C API configuration
OrtSessionOptions* options;
OrtCreateSessionOptions(&options);

// Enable EP context
OrtAddConfigEntry(options, "ep.context_enable", "1");
OrtAddConfigEntry(options, "ep.context_file_path", "/path/to/context.bin");
OrtAddConfigEntry(options, "ep.context_embed_mode", "0");

// Create session
OrtSession* session;
OrtCreateSession(env, "model.onnx", options, &session);
```

---

## 50.3 EPContext Node Format

### 50.3.1 EPContext Node Structure

When EP context is generated, the EP replaces its assigned sub-graph nodes with a single `EPContext` node:

```protobuf
// EPContext node in the ONNX graph
// Op type: "EPContext"
// Domain: "com.microsoft"

// Inputs:
//   None (or references to graph inputs)

// Outputs:
//   Same as the original sub-graph outputs

// Attributes:
//   ep_cache_context (string): Serialized EP compilation context
//   ep_cache_external_data (string): Path to external data file
//   ep_cache_embed_mode (int): 0=external, 1=embedded
//   main_context (int): 1 if this is the main EPContext node
//   ep_device_id (int): Device ID for the EP
//   ep_compute_stream (int): Compute stream handle (0=default)
//   source_model_path (string): Original model path for reference
//   ep_partition_name (string): Name of the partition
//   ep_partition_type (string): Type of partition (e.g., "trt", "openvino")
```

### 50.3.2 EPContext Node Example

```python
import onnx
from onnx import helper, TensorProto

# An EPContext node in the graph looks like:
ep_context_node = helper.make_node(
    'EPContext',
    inputs=['input_0', 'input_1'],      # Graph inputs consumed by this partition
    outputs=['output_0'],                # Graph outputs produced by this partition
    name='EPContext_0',
    domain='com.microsoft',
)

# Attributes of EPContext node:
attributes = {
    'ep_cache_context': b'...',          # Binary compiled data (embedded mode)
    'ep_cache_external_data': 'context.bin',  # External file reference
    'ep_cache_embed_mode': 0,            # 0 = external, 1 = embedded
    'main_context': 1,                   # This is the main context node
    'ep_device_id': 0,                   # GPU device ID
    'ep_partition_name': 'trt_partition_0',  # Partition name
    'ep_partition_type': 'trt',          # EP type that generated this context
}
```

### 50.3.3 EPContext Model File Structure

```
EP Context Model (embed_mode = 0, external file):
┌──────────────────────────────┐
│ model.onnx                   │
│ ├── Graph structure           │
│ │   ├── EPContext nodes       │
│ │   ├── CPU fallback nodes    │
│ │   └── Connectors           │
│ ├── Metadata                  │
│ │   ├── ep.context_enable = 1 │
│ │   ├── ep.ep_context_file = "context.bin" │
│ │   └── ep.source_model_path │
│ └── Initializers (weights)    │
├──────────────────────────────┤
│ context.bin (external file)   │
│ ├── EP compilation artifacts  │
│ ├── Optimized kernels         │
│ ├── Pre-packed weights        │
│ └── EP-specific metadata      │
└──────────────────────────────┘

EP Context Model (embed_mode = 1, embedded):
┌──────────────────────────────┐
│ model_ep_context.onnx        │
│ ├── Graph structure           │
│ │   ├── EPContext nodes       │
│ │   │   ├── ep_cache_context │ (contains binary compiled data)
│ │   │   └── ...               │
│ │   ├── CPU fallback nodes    │
│ │   └── Connectors           │
│ ├── Metadata                  │
│ ├── Initializers (weights)    │
│ └── EP Context binary data    │
│     (embedded in EPContext    │
│      node attributes)         │
└──────────────────────────────┘
```

---

## 50.4 Pre-Compiled Model Workflow

### 50.4.1 Step 1: Generate EP Context

```python
import onnxruntime as ort

def generate_ep_context(model_path, output_model_path, context_file_path=None,
                        ep_type="CUDA", embed_mode=0):
    """Generate EP context for a model.

    Args:
        model_path: Path to the source ONNX model.
        output_model_path: Path to save the EP context model.
        context_file_path: Path for external context file (if embed_mode=0).
        ep_type: Execution provider type ("CUDA", "TensorRT", "OpenVINO").
        embed_mode: 0 = external file, 1 = embedded in model.
    """
    options = ort.SessionOptions()

    # Enable EP context generation
    options.add_config_entry("ep.context_enable", "1")

    if embed_mode == 1:
        options.add_config_entry("ep.context_embed_mode", "1")
    else:
        options.add_config_entry("ep.context_embed_mode", "0")
        if context_file_path:
            options.add_config_entry("ep.context_file_path", context_file_path)

    # Save the EP context model
    options.optimized_model_filepath = output_model_path

    # Configure EP
    if ep_type == "CUDA":
        options.append_execution_provider("CUDA", {"device_id": 0})
    elif ep_type == "TensorRT":
        options.append_execution_provider("TensorrtEp", {"device_id": 0})
    elif ep_type == "OpenVINO":
        options.append_execution_provider("OpenVINO", {})

    # Create session (triggers compilation and saves EP context)
    session = ort.InferenceSession(model_path, options)

    print(f"EP context model saved to: {output_model_path}")
    if context_file_path and embed_mode == 0:
        print(f"External context file saved to: {context_file_path}")

    return session
```

### 50.4.2 Step 2: Load Pre-Compiled Model

```python
def load_precompiled_model(ep_context_model_path, ep_type="CUDA"):
    """Load a pre-compiled EP context model.

    Args:
        ep_context_model_path: Path to the EP context model.
        ep_type: Execution provider type (must match the one used during generation).

    Returns:
        InferenceSession ready for inference.
    """
    options = ort.SessionOptions()

    # Disable model compilation (only load pre-compiled)
    options.add_config_entry("ep.disable_model_compile", "1")

    # Configure EP (same configuration as during generation)
    if ep_type == "CUDA":
        options.append_execution_provider("CUDA", {"device_id": 0})
    elif ep_type == "TensorRT":
        options.append_execution_provider("TensorrtEp", {"device_id": 0})

    # Create session (fast - loads pre-compiled data)
    session = ort.InferenceSession(ep_context_model_path, options)

    return session

# Usage
session = load_precompiled_model("model_ep_context.onnx", ep_type="CUDA")
# Session is ready immediately (no compilation overhead)
```

### 50.4.3 Complete Generation Script

```python
#!/usr/bin/env python3
"""Generate EP context for production deployment."""

import argparse
import onnxruntime as ort
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Source ONNX model path")
    parser.add_argument("--output", required=True, help="Output EP context model path")
    parser.add_argument("--context-file", default=None,
                        help="External context file path (for embed_mode=0)")
    parser.add_argument("--embed-mode", type=int, default=1, choices=[0, 1],
                        help="Embed mode: 0=external, 1=embedded")
    parser.add_argument("--ep", default="CUDA", choices=["CUDA", "TensorRT", "OpenVINO"])
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()

    print(f"Generating EP context for: {args.model}")
    print(f"  EP: {args.ep}")
    print(f"  Embed mode: {args.embed_mode}")

    start_time = time.time()

    options = ort.SessionOptions()
    options.add_config_entry("ep.context_enable", "1")
    options.add_config_entry("ep.context_embed_mode", str(args.embed_mode))
    options.optimized_model_filepath = args.output

    if args.embed_mode == 0 and args.context_file:
        options.add_config_entry("ep.context_file_path", args.context_file)

    if args.ep == "CUDA":
        options.append_execution_provider("CUDA", {"device_id": args.device_id})
    elif args.ep == "TensorRT":
        options.append_execution_provider("TensorrtEp", {"device_id": args.device_id})

    session = ort.InferenceSession(args.model, options)

    elapsed = time.time() - start_time
    print(f"EP context generated in {elapsed:.1f} seconds")
    print(f"Output model: {args.output}")


if __name__ == "__main__":
    main()
```

---

## 50.5 Weightless EP Context Nodes

### 50.5.1 Overview

Weightless EP context nodes store only the EP compilation metadata (graph structure, kernel configurations) without embedding the actual weight data. The weights are loaded separately from the original model or external files.

### 50.5.2 Benefits

- **Smaller context files**: Only compilation metadata, not full weights
- **Weight sharing**: Multiple EP context models can share the same weight file
- **Flexible deployment**: Weights can be stored in a central location
- **Model updates**: Weights can be updated without recompiling

### 50.5.3 Weightless EP Context Configuration

```python
import onnxruntime as ort

options = ort.SessionOptions()
options.add_config_entry("ep.context_enable", "1")
options.add_config_entry("ep.context_embed_mode", "0")  # External context

# Make EP context weightless
options.add_config_entry("ep.context_weightless", "1")

# Specify where weights are stored
options.add_config_entry("session.external_initializers_dir", "/path/to/weights")

# The generated EP context model will reference external weights
# rather than embedding them
```

### 50.5.4 Weightless Loading

```python
def load_weightless_ep_context(model_path, weights_dir):
    """Load a weightless EP context model with external weights."""
    options = ort.SessionOptions()
    options.add_config_entry("ep.disable_model_compile", "1")
    options.add_config_entry("session.external_initializers_dir", weights_dir)

    options.append_execution_provider("CUDA", {"device_id": 0})

    session = ort.InferenceSession(model_path, options)
    return session
```

---

## 50.6 EP Context Sharing Across Sessions

### 50.6.1 Overview

EP context data can be shared across multiple InferenceSession instances to reduce memory usage and initialization time.

### 50.6.2 Sharing Strategy

```
Session 1 ──┐
             ├── Shared EP Context (loaded once, referenced by multiple sessions)
Session 2 ──┤
             │
Session 3 ──┘
```

### 50.6.3 Configuration for Sharing

```python
import onnxruntime as ort

# Create shared environment
env = ort.Env(ort.LoggingLevel.ORT_LOGGING_LEVEL_WARNING, "shared_env")

# Enable EP context sharing
options1 = ort.SessionOptions()
options1.add_config_entry("ep.context_enable", "1")
options1.add_config_entry("ep.context_share_across_sessions", "1")
options1.append_execution_provider("CUDA", {"device_id": 0})

session1 = ort.InferenceSession("model.onnx", options1, env)

# Second session shares the EP context
options2 = ort.SessionOptions()
options2.add_config_entry("ep.context_enable", "1")
options2.add_config_entry("ep.context_share_across_sessions", "1")
options2.append_execution_provider("CUDA", {"device_id": 0})

session2 = ort.InferenceSession("model.onnx", options2, env)
# EP context is shared, no additional compilation
```

### 50.6.4 EP Context Cache

```cpp
// Internal EP context cache (C++)
class EpContextCache {
public:
    // Get or create an EP context
    std::shared_ptr<EpContext> GetOrCreate(
        const std::string& model_path,
        const std::string& ep_type,
        const std::string& cache_key) {

        std::lock_guard<std::mutex> lock(mutex_);

        auto cache_key = ComputeCacheKey(model_path, ep_type, cache_key);
        auto it = cache_.find(cache_key);
        if (it != cache_.end()) {
            return it->second;
        }

        // Load EP context
        auto context = LoadEpContext(model_path, ep_type);
        cache_[cache_key] = context;
        return context;
    }

    // Evict old entries
    void Evict(size_t max_entries) {
        std::lock_guard<std::mutex> lock(mutex_);
        while (cache_.size() > max_entries) {
            cache_.erase(cache_.begin());
        }
    }

private:
    std::mutex mutex_;
    std::unordered_map<std::string, std::shared_ptr<EpContext>> cache_;
};
```

---

## 50.7 disable_model_compile Option

### 50.7.1 Overview

The `ep.disable_model_compile` option prevents the EP from compiling the model during session creation. This is used when loading a pre-compiled EP context model.

### 50.7.2 When to Use

```python
# Use case 1: Loading pre-compiled model
options = ort.SessionOptions()
options.add_config_entry("ep.disable_model_compile", "1")
session = ort.InferenceSession("precompiled_model.onnx", options)

# Use case 2: Preventing accidental recompilation
# In production, you want to ensure the pre-compiled model is used
# and not recompiled (which would be slow)
options = ort.SessionOptions()
options.add_config_entry("ep.disable_model_compile", "1")
options.add_config_entry("ep.fail_on_suboptimal_compiled_model", "1")
```

### 50.7.3 Behavior

| Scenario | disable_model_compile=0 | disable_model_compile=1 |
|----------|------------------------|------------------------|
| Loading EP context model | May recompile if EP decides to | Skips compilation, uses cached |
| Loading regular model | Compiles normally | Fails if EP requires compilation |
| EP context not found | Compiles and generates | Returns error |

---

## 50.8 fail_on_suboptimal_compiled_model Option

### 50.8.1 Overview

This option causes session creation to fail if the pre-compiled EP context is suboptimal compared to what a fresh compilation would produce.

### 50.8.2 When to Use

- **Performance-critical deployment**: Ensure the pre-compiled model is optimal
- **Hardware mismatch detection**: Fail if the EP context was compiled for different hardware
- **Validation**: Verify that the pre-compiled model matches expectations

```python
import onnxruntime as ort

options = ort.SessionOptions()
options.add_config_entry("ep.disable_model_compile", "1")
options.add_config_entry("ep.fail_on_suboptimal_compiled_model", "1")
options.append_execution_provider("CUDA", {"device_id": 0})

try:
    session = ort.InferenceSession("precompiled_model.onnx", options)
    print("Pre-compiled model is optimal")
except ort.RuntimeException as e:
    print(f"Pre-compiled model is suboptimal: {e}")
    print("Regenerate the EP context with current hardware")
```

### 50.8.3 Suboptimal Conditions

An EP context may be considered suboptimal if:

1. **Different GPU architecture**: Compiled for SM_80 but running on SM_75
2. **Different cuDNN version**: Algorithm selection may differ
3. **Different CUDA version**: Kernel implementations may have changed
4. **Different driver version**: May affect performance characteristics
5. **Different memory configuration**: Available memory affects algorithm choices
6. **Input shape mismatch**: Pre-compiled shapes differ from runtime shapes

---

## 50.9 Model Compilation Process

### 50.9.1 Compilation Pipeline

```
1. Graph Partitioning
   ├── GetCapability() → determine which nodes the EP can handle
   ├── Group contiguous supported nodes into partitions
   └── Create ComputeCapability for each partition

2. Kernel Compilation
   ├── For each partition:
   │   ├── Analyze node patterns
   │   ├── Select optimal kernels
   │   ├── Tune kernel parameters (tile sizes, etc.)
   │   └── Generate compiled binary (EP-specific)
   └── For individual nodes:
       ├── Select kernel implementation
       └── Configure kernel parameters

3. EP Context Serialization
   ├── Collect compiled artifacts
   ├── Serialize to binary format
   ├── Optionally embed in ONNX model
   └── Save external file (if configured)

4. Model Transformation
   ├── Replace compiled nodes with EPContext nodes
   ├── Add metadata (EP type, device ID, etc.)
   └── Save transformed model
```

### 50.9.2 EP-Specific Compilation

```cpp
// CUDA EP compilation process
class CudaEp::CompilationContext {
public:
    Status Compile(const std::vector<FusedNodeAndGraph>& fused_nodes) {
        for (const auto& [fused_node, graph_viewer] : fused_nodes) {
            CompiledPartition partition;
            partition.name = fused_node->Name();

            // Analyze the fused sub-graph
            auto analysis = AnalyzeGraph(graph_viewer);
            partition.input_shapes = analysis.input_shapes;
            partition.output_shapes = analysis.output_shapes;

            // Select and compile kernels
            for (const auto& node : graph_viewer.Nodes()) {
                auto kernel = SelectKernel(node);
                ORT_RETURN_IF_ERROR(kernel->Compile());
                partition.kernels.push_back(std::move(kernel));
            }

            // Optimize the compiled partition
            ORT_RETURN_IF_ERROR(OptimizePartition(partition));

            // Serialize compiled state
            auto serialized = SerializePartition(partition);
            compiled_partitions_[partition.name] = {
                .partition = std::move(partition),
                .serialized_data = std::move(serialized),
            };
        }

        return Status::OK();
    }

private:
    std::unordered_map<std::string, CompiledPartitionInfo> compiled_partitions_;
};
```

---

## 50.10 EP Context Model Serialization

### 50.10.1 Serialization Flow

```
Session::Initialize()
    │
    ├── Graph Optimization
    ├── Graph Partitioning
    ├── EP Compilation
    │
    ├── Check if ep.context_enable is set
    │   └── Yes: Serialize EP context
    │       │
    │       ├── Collect compiled partitions
    │       ├── Create EPContext nodes
    │       │   ├── For each compiled partition:
    │       │   │   ├── Create EPContext node
    │       │   │   ├── Set ep_cache_context attribute (binary data)
    │       │   │   ├── Set ep_partition_name
    │       │   │   ├── Set ep_partition_type
    │       │   │   └── Replace original nodes with EPContext
    │       │   │
    │       │   └── Set main_context attribute on the first EPContext
    │       │
    │       ├── Handle embed_mode:
    │       │   ├── mode=0: Save binary to external file
    │       │   └── mode=1: Embed in model attribute
    │       │
    │       └── Save transformed model
    │           ├── To optimized_model_filepath (if set)
    │           └── With metadata
    │
    └── Continue session initialization
```

### 50.10.2 Serialization Implementation

```cpp
Status SerializeEpContext(const SessionState& session_state,
                           const std::string& output_path,
                           int embed_mode,
                           const std::string& external_file_path) {
    auto& model = session_state.GetModel();
    auto graph = model.MainGraph();

    // Create a copy of the graph for serialization
    Graph serialized_graph(graph);

    // For each compiled partition, create an EPContext node
    for (const auto& [partition_name, compiled_info] :
         session_state.GetCompiledPartitions()) {

        // Remove original nodes from the serialized graph
        for (const auto& node_idx : compiled_info.node_indices) {
            serialized_graph.RemoveNode(node_idx);
        }

        // Create EPContext node
        auto& ep_context_node = serialized_graph.AddNode(
            partition_name + "_EPContext",
            "EPContext",
            "com.microsoft",
            compiled_info.input_names,
            compiled_info.output_names);

        // Set attributes
        if (embed_mode == 1) {
            // Embed compiled data in the node attribute
            ep_context_node.AddAttribute(
                "ep_cache_context", compiled_info.serialized_data);
            ep_context_node.AddAttribute("ep_cache_embed_mode", int64_t(1));
        } else {
            // Reference external file
            ep_context_node.AddAttribute(
                "ep_cache_external_data", external_file_path);
            ep_context_node.AddAttribute("ep_cache_embed_mode", int64_t(0));

            // Write external file
            std::ofstream file(external_file_path, std::ios::binary);
            file.write(reinterpret_cast<const char*>(compiled_info.serialized_data.data()),
                       compiled_info.serialized_data.size());
        }

        ep_context_node.AddAttribute("main_context", int64_t(1));
        ep_context_node.AddAttribute("ep_device_id", compiled_info.device_id);
        ep_context_node.AddAttribute("ep_partition_name", partition_name);
        ep_context_node.AddAttribute("ep_partition_type", compiled_info.ep_type);
    }

    // Add metadata
    model.AddMetadata("ep.context_enable", "1");
    model.AddMetadata("ep.context_embed_mode", std::to_string(embed_mode));
    if (!external_file_path.empty()) {
        model.AddMetadata("ep.context_file_path", external_file_path);
    }

    // Save the model
    ORT_RETURN_IF_ERROR(model.Save(output_path));

    return Status::OK();
}
```

---

## 50.11 Compatibility Checking

### 50.11.1 Compatibility Verification

When loading a pre-compiled EP context model, ONNX Runtime checks compatibility:

```cpp
Status VerifyEpContextCompatibility(const EpContextInfo& context_info,
                                     const SessionOptions& options) {
    // 1. Check EP type
    auto requested_ep = options.GetExecutionProviderType();
    if (context_info.ep_type != requested_ep) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
            "EP context was compiled for '", context_info.ep_type,
            "' but '", requested_ep, "' was requested");
    }

    // 2. Check device ID
    int requested_device = options.GetDeviceId();
    if (context_info.device_id != requested_device) {
        LOGS_DEFAULT(WARNING)
            << "EP context was compiled for device "
            << context_info.device_id
            << " but device " << requested_device << " was requested";
    }

    // 3. Check CUDA compute capability (for CUDA EP)
    if (context_info.ep_type == "CUDA") {
        cudaDeviceProp prop;
        CUDA_CALL(cudaGetDeviceProperties(&prop, requested_device));
        std::string current_arch = fmt::format("sm_{}{}", prop.major, prop.minor);

        if (context_info.target_architecture != current_arch) {
            if (options.GetConfigEntry("ep.fail_on_suboptimal_compiled_model") == "1") {
                return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                    "EP context was compiled for ", context_info.target_architecture,
                    " but running on ", current_arch);
            } else {
                LOGS_DEFAULT(WARNING)
                    << "EP context architecture mismatch: compiled for "
                    << context_info.target_architecture
                    << " but running on " << current_arch;
            }
        }
    }

    // 4. Check software versions
    if (context_info.cudnn_version != CUDNN_VERSION) {
        LOGS_DEFAULT(WARNING) << "cuDNN version mismatch: compiled with "
                              << context_info.cudnn_version
                              << " but running with " << CUDNN_VERSION;
    }

    // 5. Check input shapes (if pre-compiled for specific shapes)
    // ...

    return Status::OK();
}
```

### 50.11.2 Compatibility Metadata

```python
# The EP context model stores compatibility metadata:
model_metadata = {
    "ep.context_enable": "1",
    "ep.context_embed_mode": "0",
    "ep.context_file_path": "/path/to/context.bin",
    "ep.compilation_info": json.dumps({
        "ep_type": "CUDA",
        "device_id": 0,
        "cuda_version": "12.1",
        "cudnn_version": "8.9.0",
        "target_architecture": "sm_86",
        "driver_version": "535.104.05",
        "ort_version": "1.20.0",
        "compilation_timestamp": "2024-01-15T10:30:00Z",
        "input_shapes": {"input": [1, 3, 224, 224]},
        "optimized_kernels": [
            {"op": "Conv", "algo": "WINOGRAD"},
            {"op": "MatMul", "algo": "CUBLAS_GEMM"},
        ],
    }),
}
```

---

## 50.12 Performance Benefits of Pre-Compiled Models

### 50.12.1 Session Creation Time Comparison

```
Model: ResNet-50 (25M parameters)
GPU: NVIDIA A100

Without EP Context:
  Graph Optimization:     0.5s
  Graph Partitioning:     0.1s
  EP Compilation:        28.0s    ← dominates
  Kernel Setup:           0.3s
  Memory Planning:        0.2s
  ────────────────────────────────
  Total:                 29.1s

With EP Context (external file):
  Load EP Context:        0.1s
  Kernel Setup:           0.1s
  Memory Planning:        0.1s
  ────────────────────────────────
  Total:                  0.3s

Speedup: ~97x faster session creation
```

### 50.12.2 Inference Performance

```
Model: BERT-Large (340M parameters)
GPU: NVIDIA A100

Inference latency (per token):
  Without EP Context:  12.5 ms
  With EP Context:     12.3 ms  (same, pre-compiled kernels)

Memory usage:
  Without EP Context:  2.1 GB   (during compilation peak: 4.5 GB)
  With EP Context:     2.0 GB   (no compilation peak)

Session creation:
  Without EP Context:  45.2 s   (includes cuDNN algorithm search)
  With EP Context:      0.8 s   (loads pre-searched algorithms)
```

### 50.12.3 Deployment Scenarios

| Scenario | Without EP Context | With EP Context |
|----------|-------------------|-----------------|
| Server cold start | 30-60 seconds | <1 second |
| Kubernetes pod restart | 30-60 seconds | <1 second |
| Serverless function cold start | 30-60 seconds | <1 second |
| Edge device boot | 60-120 seconds | <2 seconds |
| Model A/B testing | 60-120 seconds (two compilations) | <2 seconds (two loads) |
| Multi-model serving | N * 30 seconds | N * 1 second |

### 50.12.4 Benchmarking EP Context

```python
import time
import onnxruntime as ort
import numpy as np


def benchmark_session_creation(model_path, use_ep_context=False,
                                num_trials=5):
    """Benchmark session creation time."""
    times = []

    for _ in range(num_trials):
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        if use_ep_context:
            options.add_config_entry("ep.disable_model_compile", "1")

        options.append_execution_provider("CUDA", {"device_id": 0})

        start = time.perf_counter()
        session = ort.InferenceSession(model_path, options)
        elapsed = time.perf_counter() - start

        times.append(elapsed)
        del session

    return {
        "avg_time": np.mean(times),
        "min_time": np.min(times),
        "max_time": np.max(times),
    }


def benchmark_inference(model_path, use_ep_context=False,
                        num_iterations=100, warmup=10):
    """Benchmark inference latency."""
    options = ort.SessionOptions()
    if use_ep_context:
        options.add_config_entry("ep.disable_model_compile", "1")
    options.append_execution_provider("CUDA", {"device_id": 0})

    session = ort.InferenceSession(model_path, options)

    # Prepare input
    input_info = session.get_inputs()[0]
    input_data = np.random.randn(*input_info.shape).astype(np.float32)
    input_name = input_info.name

    # Warmup
    for _ in range(warmup):
        session.run(None, {input_name: input_data})

    # Benchmark
    latencies = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        session.run(None, {input_name: input_data})
        elapsed = (time.perf_counter() - start) * 1000  # ms
        latencies.append(elapsed)

    return {
        "avg_ms": np.mean(latencies),
        "p50_ms": np.percentile(latencies, 50),
        "p95_ms": np.percentile(latencies, 95),
        "p99_ms": np.percentile(latencies, 99),
    }


# Run benchmarks
print("Without EP Context:")
creation = benchmark_session_creation("model.onnx", use_ep_context=False)
print(f"  Session creation: {creation['avg_time']:.2f}s")
inference = benchmark_inference("model.onnx", use_ep_context=False)
print(f"  Inference p50: {inference['p50_ms']:.2f}ms")

print("\nWith EP Context:")
creation = benchmark_session_creation("model_ep_context.onnx",
                                       use_ep_context=True)
print(f"  Session creation: {creation['avg_time']:.2f}s")
inference = benchmark_inference("model_ep_context.onnx", use_ep_context=True)
print(f"  Inference p50: {inference['p50_ms']:.2f}ms")
```

---

## 50.13 EP Context with Different EPs

### 50.13.1 TensorRT EP Context

```python
import onnxruntime as ort

# Generate TensorRT EP context
options = ort.SessionOptions()
options.add_config_entry("ep.context_enable", "1")
options.add_config_entry("ep.context_embed_mode", "0")
options.add_config_entry("ep.context_file_path", "trt_cache.bin")

trt_options = {
    "device_id": 0,
    "trt_max_workspace_size": 4 * 1024 * 1024 * 1024,  # 4GB
    "trt_fp16_enable": True,
    "trt_engine_cache_enable": True,
    "trt_engine_cache_path": "./trt_engines",
}
options.append_execution_provider("TensorrtEp", trt_options)

session = ort.InferenceSession("model.onnx", options)
```

### 50.13.2 OpenVINO EP Context

```python
import onnxruntime as ort

# Generate OpenVINO EP context
options = ort.SessionOptions()
options.add_config_entry("ep.context_enable", "1")
options.add_config_entry("ep.context_embed_mode", "1")

ov_options = {
    "device_type": "GPU",
    "enable_opencl_throttling": True,
}
options.append_execution_provider("OpenVINO", ov_options)

session = ort.InferenceSession("model.onnx", options)
```

### 50.13.3 Multi-EP Context

```python
# Model with multiple EPs, each with its own context
options = ort.SessionOptions()
options.add_config_entry("ep.context_enable", "1")
options.add_config_entry("ep.context_embed_mode", "0")

# TensorRT handles most of the graph
options.append_execution_provider("TensorrtEp", {
    "device_id": 0,
    "trt_fp16_enable": True,
})

# CUDA handles fallback nodes
options.append_execution_provider("CUDA", {
    "device_id": 0,
})

# CPU handles remaining nodes
# CPUExecutionProvider is always available

session = ort.InferenceSession("model.onnx", options)
```

---

## 50.14 Best Practices

### 50.14.1 Generation Best Practices

1. **Generate on target hardware**: Always generate EP context on the same GPU architecture as deployment
2. **Use representative input shapes**: If input shapes are fixed, compile with exact shapes
3. **Set memory limits**: Match production memory limits during generation
4. **Test both modes**: Try embed_mode 0 and 1 to find the best trade-off
5. **Version your contexts**: Include ORT and EP version in the context metadata

### 50.14.2 Deployment Best Practices

1. **Validate EP context**: Use `fail_on_suboptimal_compiled_model` during testing
2. **Handle compatibility errors**: Have fallback to recompilation
3. **Cache external files**: Keep external context files on fast storage
4. **Monitor first-inference latency**: Verify EP context loading is working
5. **Regenerate periodically**: Re-compile when updating drivers or ORT version

### 50.14.3 Error Recovery Pattern

```python
import onnxruntime as ort

def create_session_with_fallback(model_path, ep_context_path=None):
    """Create session with EP context, falling back to fresh compilation."""
    if ep_context_path:
        try:
            options = ort.SessionOptions()
            options.add_config_entry("ep.disable_model_compile", "1")
            options.add_config_entry("ep.fail_on_suboptimal_compiled_model", "1")
            options.append_execution_provider("CUDA", {"device_id": 0})

            session = ort.InferenceSession(ep_context_path, options)
            print("Loaded pre-compiled EP context successfully")
            return session
        except ort.RuntimeException as e:
            print(f"EP context loading failed: {e}")
            print("Falling back to fresh compilation...")

    # Fresh compilation
    options = ort.SessionOptions()
    options.append_execution_provider("CUDA", {"device_id": 0})
    session = ort.InferenceSession(model_path, options)
    print("Created session with fresh compilation")
    return session
```

---

## 50.15 Summary

| Topic | Key Points |
|-------|-----------|
| EP Context Concept | Pre-compiled EP state serialized into model for fast loading |
| Configuration | `ep.context_enable`, `ep.context_file_path`, `ep.context_embed_mode` |
| EPContext Node | Replaces compiled sub-graph with `EPContext` op in `com.microsoft` domain |
| Embed Mode 0 | Context saved as external binary file |
| Embed Mode 1 | Context embedded in ONNX model as node attribute |
| Weightless Mode | Only compilation metadata, weights loaded separately |
| disable_model_compile | Prevents recompilation when loading pre-compiled model |
| fail_on_suboptimal | Fails if pre-compiled context doesn't match current hardware |
| Compatibility Check | Verifies EP type, GPU arch, CUDA version, cuDNN version |
| Performance | ~97x faster session creation, same inference throughput |
| Multi-EP Support | Each EP (CUDA, TensorRT, OpenVINO) has its own context |
| Best Practices | Generate on target hardware, validate contexts, handle fallbacks |
