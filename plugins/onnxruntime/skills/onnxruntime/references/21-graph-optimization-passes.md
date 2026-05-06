# ONNX Runtime Reference - Chapter 21: Graph Optimization Passes

---

## 21.1 Optimization Levels

| Level | Name | Description |
|-------|------|-------------|
| Level 1 | Basic | Constant folding, dead code elimination |
| Level 2 | Extended | Complex fusion, attention fusion |
| Level 3 | Layout | NHWC/NCHWc layout transformation |
| Level 4 | Full | QDQ handling, MatMulNBits |

## 21.2 Level 1 - Basic Optimizations

| Optimizer | Description |
|-----------|-------------|
| ConstantFolding | Evaluate constant subexpressions at compile time |
| DeadCodeElimination | Remove unused nodes |
| IdentityElimination | Remove identity nodes |
| NoopElimination | Remove no-op nodes (Reshape with same shape, etc.) |
| DropoutElimination | Remove dropout nodes (inference only) |
| SliceElimination | Remove unnecessary slice operations |
| UnsqueezeElimination | Remove unnecessary unsqueeze operations |
| ResizeElimination | Remove unnecessary resize operations |
| ExpandElimination | Remove expand with size 1 |

## 21.3 Level 2 - Extended Optimizations

| Optimizer | Description |
|-----------|-------------|
| ConvBNFusion | Fuse Conv + BatchNorm into Conv |
| GemmActivationFusion | Fuse GEMM + activation |
| BiasGeluFusion | Fuse BiasAdd + GELU |
| AttentionFusion | Fuse MultiHeadAttention components |
| EmbedLayerNormFusion | Fuse Embedding + LayerNorm |
| BiasSoftmaxFusion | Fuse BiasAdd + Softmax |
| MatMulAddFusion | Fuse MatMul + BiasAdd |
| ConvMulFusion | Fuse Conv + Mul (scale) |
| FastGeluFusion | Replace GELU with fast approximation |
| GeluFusion | Fuse GELU subgraph |
| MatMulScaleFusion | Fuse MatMul + scale |
| ReluClipFusion | Fuse Relu + Clip |
| QLinearConcatFusion | Fuse quantized concat |

## 21.4 Level 3 - Layout Optimizations

| Optimizer | Description |
|-----------|-------------|
| NHWCTransformer | Convert NCHW → NHWC layout |
| NCHWCTransformer | Convert to NCHWc (blocked) layout |
| InsertCastTransformer | Insert necessary cast operations |

## 21.5 Level 4 - Full Optimizations

| Optimizer | Description |
|-----------|-------------|
| DoubleQDQRemover | Remove Q→DQ→Q→DQ pairs |
| QDQCleanup | Remove remaining QDQ pairs |
| CastChainElimination | Remove unnecessary cast chains |
| GeluApproximation | Approximate GELU for speed |
| MatMulNBitsConversion | Convert DQ+MatMul → MatMulNBits |

## 21.6 Custom Optimizer Registration

```cpp
// Implement GraphTransformer
class MyTransformer : public GraphTransformer {
public:
    Status Apply(Graph& graph, bool& modified, int graph_level,
                 const logging::Logger& logger) override;
};

// Register
auto transformers = std::make_unique<std::vector<std::unique_ptr<GraphTransformer>>>();
transformers->push_back(std::make_unique<MyTransformer>());
```

## 21.7 Controlling Optimizations

```python
# Disable all optimizations
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

# Disable specific optimizers
opts.add_session_config_entry("optimization.disable_specified_optimizers",
    "ConvBNFusion,GemmActivationFusion")

# Control optimization loop
opts.add_session_config_entry("session.graph_optimizations_loop_level", "1")

# Enable GELU approximation
opts.add_session_config_entry("optimization.enable_gelu_approximation", "1")

# Enable Cast chain elimination
opts.add_session_config_entry("optimization.enable_cast_chain_elimination", "1")
```
