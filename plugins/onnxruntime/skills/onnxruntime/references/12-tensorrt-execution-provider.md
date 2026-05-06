# ONNX Runtime Reference - Chapter 12: TensorRT Execution Provider

Complete reference for the NVIDIA TensorRT Execution Provider.

---

## 12.1 Overview

TensorRT EP uses NVIDIA TensorRT to optimize ONNX models for NVIDIA GPUs. It performs layer fusion, precision calibration, and kernel auto-tuning for maximum inference performance.

---

## 12.2 Provider Options

```python
providers = [
    ("TensorrtExecutionProvider", {
        "device_id": 0,
        "trt_max_workspace_size": 1 << 30,          # 1GB workspace
        "trt_max_partition_iterations": 1000,
        "trt_min_subgraph_size": 1,
        "trt_fp16_enable": True,
        "trt_int8_enable": False,
        "trt_int8_calibration_table": "",
        "trt_int8_use_native_calibration_table": False,
        "trt_dla_enable": False,
        "trt_dla_core": 0,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": "./trt_cache",
        "trt_engine_update_prefix": "",
        "trt_force_sequential_engine_build": False,
        "trt_enable_layer_norm_fusion": True,
        "trt_timing_cache_enable": True,
        "trt_timing_cache_path": "./trt_cache/timing_cache.bin",
        "trt_dump_subgraphs": False,
        "trt_native_instancenorm": True,
        "trt_context_memory_sharing_enable": False,
        "trt_layer_norm_fp32_fallback": False,
        "trt_heuristic": False,
        "trt_builder_opt_level": 3,
        "trt_extra_plugin_lib_paths": "",
        "trt_ep_context_file_path": "",
        "trt_ep_context_embed_mode": 0,
        "trt_force_timing_cache": False,
        "trt_use_cuda_graph": False,
        "trt_use_dsa": False,
        "trt_use_shape_inputs": False,
    }),
]
```

---

## 12.3 Key Features

### 12.3.1 FP16 Precision

```python
providers = [
    ("TensorrtExecutionProvider", {
        "trt_fp16_enable": True,
    }),
]
# TensorRT will use FP16 for all compatible operations
```

### 12.3.2 INT8 Quantization

```python
providers = [
    ("TensorrtExecutionProvider", {
        "trt_int8_enable": True,
        "trt_int8_calibration_table": "calibration_table.cache",
    }),
]
```

### 12.3.3 Engine Caching

```python
providers = [
    ("TensorrtExecutionProvider", {
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": "./trt_engines",
        "trt_timing_cache_enable": True,
        "trt_timing_cache_path": "./trt_engines/timing_cache.bin",
    }),
]
# First run: builds and caches TensorRT engines (slow)
# Subsequent runs: loads cached engines (fast)
```

### 12.3.4 DLA (Deep Learning Accelerator)

```python
providers = [
    ("TensorrtExecutionProvider", {
        "trt_dla_enable": True,
        "trt_dla_core": 0,          # DLA core 0 or 1
        "trt_fp16_enable": True,     # DLA requires FP16
    }),
]
```

### 12.3.5 Builder Optimization Level

```python
# 0 = shortest build time, lowest optimization
# 5 = longest build time, highest optimization (default: 3)
providers = [
    ("TensorrtExecutionProvider", {
        "trt_builder_opt_level": 5,
    }),
]
```
