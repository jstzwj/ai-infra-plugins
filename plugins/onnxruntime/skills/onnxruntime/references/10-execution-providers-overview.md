# ONNX Runtime Reference - Chapter 10: Execution Providers Overview

Complete reference for the execution provider architecture, registration, and graph partitioning.

---

## 10.1 Execution Provider Architecture

### 10.1.1 IExecutionProvider Interface

```cpp
class IExecutionProvider {
public:
    virtual const std::string& Type() const = 0;

    // Declare supported ops
    virtual std::vector<std::unique_ptr<IndexedSubGraph>>
    GetCapability(const GraphViewer& graph,
                  const IKernelLookup& kernel_lookup) const = 0;

    // Compile subgraph for EP
    virtual common::Status Compile(
        const std::vector<const Node*>& fused_nodes,
        std::vector<NodeComputeInfo>& node_compute_funcs) = 0;

    // Data transfer between devices
    virtual std::unique_ptr<IDataTransfer> GetDataTransfer() const = 0;

    // Allocator management
    virtual AllocatorPtr GetAllocator(int id, OrtMemType mem_type) const = 0;

    // Kernel registry
    virtual std::shared_ptr<KernelRegistry> GetKernelRegistry() const = 0;

    // Stream handling
    virtual std::unique_ptr<onnxruntime::Stream> CreateStream() const;

    // Synchronization
    virtual void SyncStreams(const std::vector<Stream*>& streams) const;

    // Memory profiling
    virtual std::vector<AllocatorPtr> CreateAllocators() = 0;
};
```

### 10.1.2 EP Registration Flow

```
1. Create EP with options
   auto cuda_options = OrtCUDAProviderOptionsV2();
   cuda_options.Update(keys, values);

2. Append to SessionOptions
   opts.AppendExecutionProvider_CUDA(cuda_options);

3. During Session::Initialize():
   a. EP.GetCapability(graph) → returns supported subgraphs
   b. GraphPartitioner assigns nodes to EPs
   c. Unsupported nodes → CPU EP fallback
   d. KernelRegistry maps (op_type, EP) → kernel
   e. EP.Compile() for compiled subgraphs

4. During Session::Run():
   a. Executor dispatches nodes to assigned EPs
   b. IDataTransfer handles cross-EP memory copies
   c. EP-specific kernels execute Compute()
```

---

## 10.2 All Execution Providers

| EP | Hardware | Key Features | Build Flag |
|----|----------|-------------|------------|
| CPU | x86/ARM/PPC | Default fallback, MLAS optimized | Always available |
| CUDA | NVIDIA GPU | cuDNN, cuBLAS, FP16/BF16 | `onnxruntime_USE_CUDA` |
| TensorRT | NVIDIA GPU | INT8/FP16, engine caching | `onnxruntime_USE_TENSORRT` |
| OpenVINO | Intel CPU/GPU/NPU | AUTO device, caching | `onnxruntime_USE_OPENVINO` |
| DNNL (oneDNN) | Intel CPU | Primitive-based | `onnxruntime_USE_DNNL` |
| CoreML | Apple Silicon | Neural Engine, GPU, CPU | `onnxruntime_USE_COREML` |
| NNAPI | Android NPU | GPU/DSP/NPU delegation | `onnxruntime_USE_NNAPI_BUILTIN` |
| DirectML | Windows GPU | DirectX 12 | `onnxruntime_USE_DML` |
| WebGPU | Browser GPU | Dawn, compute shaders | `onnxruntime_USE_WEBGPU` |
| QNN | Qualcomm NPU | HTP, INT8 quantization | `onnxruntime_USE_QNN` |
| ACL | ARM CPU/GPU | ARM Compute Library | `onnxruntime_USE_ACL` |
| CANN | Huawei Ascend | Ascend NPU | `onnxruntime_USE_CANN` |
| Vitis-AI | AMD/Xilinx FPGA | DPU accelerator | `onnxruntime_USE_VITISAI` |
| XNNPACK | Mobile CPU | Optimized mobile inference | `onnxruntime_USE_XNNPACK` |
| RKNPU | Rockchip NPU | RK3588, RK3568 | `onnxruntime_USE_RKNPU` |
| TVM | Multiple | Apache TVM backend | `onnxruntime_USE_TVM` |
| WebNN | Browser | Web Neural Network API | `onnxruntime_USE_WEBNN` |
| MIGraphX | AMD GPU | ROCm inference | `onnxruntime_USE_MIGRAPHX` |
| WinML | Windows | Windows ML integration | Built-in on Windows |

---

## 10.3 Graph Partitioning

### 10.3.1 Partitioning Algorithm

```
1. For each registered EP (in registration order):
   a. Call EP.GetCapability(graph)
   b. EP returns list of IndexedSubGraph (supported node groups)
   c. Mark those nodes as assigned to this EP

2. Unassigned nodes → CPU EP (unless fallback disabled)

3. For compiled EPs (TensorRT, QNN):
   a. Group assigned nodes into subgraphs
   b. Call EP.Compile() to generate EP-specific engines
   c. Replace subgraphs with EPContext nodes
```

### 10.3.2 CPU EP Fallback

```python
# Default: unsupported ops fall back to CPU EP
sess = ort.InferenceSession("model.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

# Disable CPU fallback (fail if CUDA can't handle all ops)
opts = ort.SessionOptions()
opts.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
sess = ort.InferenceSession("model.onnx", sess_options=opts,
                             providers=["CUDAExecutionProvider"])
```

### 10.3.3 Capacity-Aware Partitioning

```python
# Limit CUDA memory usage during partitioning
opts.add_session_config_entry(
    "session.resource_cuda_partitioning_settings",
    "2097152,")  # 2GB limit, no pre-recorded stats
```

### 10.3.4 Layer Assignment

```python
# Guide partitioning with layer annotations
opts.add_session_config_entry(
    "session.layer_assignment_settings",
    "cuda(attention,mlp);cpu(layernorm)")
```

---

## 10.4 EP Device Discovery

### 10.4.1 Hardware Device API

```cpp
// Get all available EP devices
const OrtEpDevice** devices;
size_t num_devices;
ort->GetEpDevices(&devices, &num_devices);

for (size_t i = 0; i < num_devices; i++) {
    const OrtHardwareDevice* hw;
    ort->EpDevice_HardwareDevice(devices[i], &hw);

    OrtHardwareDeviceType type;
    ort->HardwareDevice_Type(hw, &type);

    const char* name;
    ort->HardwareDevice_Name(hw, &name);

    size_t memory_size;
    ort->HardwareDevice_MemorySize(hw, &memory_size);

    const char* ep_name;
    ort->EpDevice_EpName(devices[i], &ep_name);
}
```

### 10.4.2 EP Selection Policies

```c
typedef enum OrtExecutionProviderDevicePolicy {
    OrtExecutionProviderDevicePolicy_DEFAULT,          // OS decides
    OrtExecutionProviderDevicePolicy_PREFER_CPU,       // Prefer CPU
    OrtExecutionProviderDevicePolicy_PREFER_NPU,       // Prefer NPU
    OrtExecutionProviderDevicePolicy_PREFER_GPU,       // Prefer GPU
    OrtExecutionProviderDevicePolicy_MAX_PERFORMANCE,  // Max performance
    OrtExecutionProviderDevicePolicy_MAX_EFFICIENCY,   // Max efficiency
    OrtExecutionProviderDevicePolicy_MIN_OVERALL_POWER, // Min power
} OrtExecutionProviderDevicePolicy;
```

---

## 10.5 Data Transfer Between EPs

### 10.5.1 IDataTransfer Interface

```cpp
class IDataTransfer {
public:
    // Copy tensor from source to destination device
    virtual Status CopyTensor(const Tensor& src, Tensor& dst) const = 0;

    // Async copy with stream
    virtual Status CopyTensorAsync(const Tensor& src, Tensor& dst,
                                    Stream& stream) const;
};
```

### 10.5.2 Automatic Cross-EP Data Transfer

When two connected nodes are on different EPs, ORT automatically:
1. Allocates a buffer on the destination device
2. Copies data from source to destination
3. Uses pinned memory for CPU↔GPU transfers
4. Synchronizes streams if needed

---

## 10.6 Kernel Registry

### 10.6.1 Registry Architecture

```
KernelRegistry (per session)
├── Global kernel registry (built-in ops)
│   ├── CPU kernels (all standard ONNX ops)
│   ├── CUDA kernels (if built with CUDA)
│   └── Other EP kernels
├── EP-specific kernel registries
│   ├── CUDA kernel registry
│   ├── TensorRT kernel registry
│   └── ...
└── Custom op kernel registries
    └── User-registered custom ops
```

### 10.6.2 Kernel Lookup

```cpp
// Lookup: (op_type, domain, EP_type) → KernelCreateInfo
Status KernelRegistry::TryFindKernel(
    const Node& node,
    const std::string& exec_provider_type,
    const KernelCreateInfo** kernel_create_info) const;
```

---

## 10.7 EP Context Models

### 10.7.1 Pre-compiled Models

```python
# Generate EP context model during first run
opts.add_session_config_entry("ep.context_enable", "1")
opts.add_session_config_entry("ep.context_file_path", "model_ctx.onnx")
opts.add_session_config_entry("ep.context_embed_mode", "1")  # Embed in ONNX

sess = ort.InferenceSession("model.onnx", sess_options=opts,
                             providers=["TensorrtExecutionProvider"])
# First run: generates model_ctx.onnx with TensorRT engines embedded

# Subsequent runs: use pre-compiled model (much faster load)
sess2 = ort.InferenceSession("model_ctx.onnx",
    providers=["TensorrtExecutionProvider"])
```

### 10.7.2 EP Context Config Options

| Key | Values | Default | Description |
|-----|--------|---------|-------------|
| `ep.context_enable` | "0"/"1" | "0" | Enable EP context generation |
| `ep.context_file_path` | path | auto | Context model output path |
| `ep.context_embed_mode` | "0"/"1" | "0" | 0=separate file, 1=embed in ONNX |
| `ep.context_node_name_prefix` | string | "" | EPContext node name prefix |
| `ep.share_ep_contexts` | "0"/"1" | "0" | Share EP contexts across sessions |
| `ep.enable_weightless_ep_context_nodes` | "0"/"1" | "0" | Weightless EP context nodes |

---

## 10.8 Dynamic EP Options

```python
# Change EP options at runtime (without recreating session)
sess.set_ep_dynamic_options({
    "ep.dynamic.workload_type": "Efficient",  # or "Default"
    "ep.dynamic.qnn_htp_performance_mode": "high_performance",
})
```

---

## 10.9 EP Compatibility Checking

```python
# Check EP compatibility with hardware
import onnxruntime as ort

# Get available providers
providers = ort.get_available_providers()
print(f"Available: {providers}")

# Check CUDA availability
if "CUDAExecutionProvider" in providers:
    sess_opts = ort.SessionOptions()
    sess = ort.InferenceSession("model.onnx", providers=["CUDAExecutionProvider"])
    active = sess.get_providers()
    if "CUDAExecutionProvider" not in active:
        print("CUDA EP failed to initialize, using CPU")
```
