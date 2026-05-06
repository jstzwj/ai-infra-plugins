# ONNX Runtime Reference - Chapters 14-19: Additional Execution Providers

---

## 14. DNNL (oneDNN) Execution Provider

### Provider Options
```python
providers = [
    ("DnnlExecutionProvider", {
        "use_arena": True,
        "use_pinned_mem": True,
    }),
]
```

### Features
- CPU acceleration using Intel oneDNN primitives
- Optimized for Intel CPUs with AVX2/AVX-512
- JIT-compiled kernels for convolutions, matmul, etc.

---

## 15. CoreML Execution Provider

### Provider Options
```python
providers = [
    ("CoreMLExecutionProvider", {
        "use_cpu_only": False,
        "enable_on_subgraph": True,
        "coreml_flags": 0,
    }),
]
```

### Features
- Accelerates on Apple Neural Engine (ANE), GPU, and CPU
- macOS 12+ and iOS 15+
- Automatic fallback for unsupported ops
- Only available on Apple platforms

---

## 16. NNAPI Execution Provider

### Provider Options
```python
providers = [
    ("NNAPIExecutionProvider", {
        "use_fp16": False,
        "use_nchwc": True,
        "use_cpu_only": False,
    }),
]
```

### Flags
| Flag | Value | Description |
|------|-------|-------------|
| `NNAPI_FLAG_USE_FP16` | 1 | Use FP16 precision |
| `NNAPI_FLAG_USE_NCHW` | 2 | Use NCHW layout |
| `NNAPI_FLAG_CPU_DISABLED` | 4 | Disable CPU fallback |
| `NNAPI_FLAG_CPU_ONLY` | 8 | Use CPU only |

### Stop Ops
```python
opts = ort.SessionOptions()
opts.add_session_config_entry("ep.nnapi.partitioning_stop_ops", "Add,Sub")
```

---

## 17. WebGPU Execution Provider

### Provider Options
```python
providers = [
    ("WebGPUExecutionProvider", {
        "device_id": 0,
        "enable_graph_capture": False,
    }),
]
```

### Features
- Dawn-based WebGPU implementation
- Compute shader operators
- Browser and Node.js deployment
- Available as plugin EP (`plugin-ep-webgpu/`)

---

## 18. DirectML Execution Provider

### Provider Options
```python
providers = [
    ("DmlExecutionProvider", {
        "device_id": 0,
    }),
]
```

### Features
- Windows-only GPU acceleration via DirectX 12
- Works with any DirectX 12 compatible GPU
- Install: `pip install onnxruntime-directml`

---

## 19. Other Execution Providers

### QNN EP (Qualcomm NPU)
```python
providers = [
    ("QNNExecutionProvider", {
        "backend_path": "libQnnHtp.so",
        "htp_performance_mode": "default",
        "soc_model": "SM8550",
    }),
]
```

### ACL EP (ARM Compute Library)
```python
providers = [("ACLExecutionProvider", {"use_arena": "1"})]
```

### CANN EP (Huawei Ascend)
```python
providers = [("CANNExecutionProvider", {"device_id": 0, "npu_peg_size": 262144})]
```

### XNNPACK EP
```python
providers = [
    ("XnnpackExecutionProvider", {
        "num_threads": 4,
    }),
]
```

### Vitis-AI EP (AMD/Xilinx)
```python
providers = [("VitisAIExecutionProvider", {"config_file": "vaip_config.json"})]
```

### RKNPU EP (Rockchip)
```python
providers = [("RknpuExecutionProvider", {})]
```

### TVM EP
```python
providers = [("TvmExecutionProvider", {"target": "llvm"})]
```
