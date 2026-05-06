# ONNX Runtime Reference - Chapter 13: OpenVINO Execution Provider

---

## 13.1 Overview

OpenVINO EP uses Intel's OpenVINO toolkit to accelerate inference on Intel hardware (CPU, GPU, NPU).

## 13.2 Provider Options

```python
providers = [
    ("OpenVINOExecutionProvider", {
        "device_type": "CPU",                  # CPU|GPU|NPU|AUTO|HETERO|MULTI
        "device_id": "",                        # Device ID (for multi-device)
        "enable_opencl_throttling": False,      # GPU task throttling
        "enable_dynamic_shapes": True,          # Dynamic shape support
        "num_of_threads": 0,                    # Thread count (0=auto)
        "cache_dir": "./ov_cache",             # Model cache directory
        "context": None,                        # Remote context
        "load_config": {},                      # OpenVINO config dict
    }),
]
```

## 13.3 Device Types

| Device | Description | Hardware |
|--------|-------------|----------|
| CPU | Intel CPU | All Intel CPUs |
| GPU | Intel Integrated/Discrete GPU | Intel Arc, Iris Xe |
| NPU | Intel Neural Processing Unit | Intel Core Ultra |
| AUTO | Automatic device selection | Selects best available |
| HETERO | Heterogeneous execution | Split across devices |
| MULTI | Multi-device parallel | Use multiple devices |

## 13.4 Usage

```python
# Auto device selection
sess = ort.InferenceSession("model.onnx",
    providers=[("OpenVINOExecutionProvider", {"device_type": "AUTO"})])

# Specific GPU
sess = ort.InferenceSession("model.onnx",
    providers=[("OpenVINOExecutionProvider", {"device_type": "GPU"})])

# With model caching
sess = ort.InferenceSession("model.onnx",
    providers=[("OpenVINOExecutionProvider", {
        "device_type": "GPU",
        "cache_dir": "./ov_cache",
    })])
```
