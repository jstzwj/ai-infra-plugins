# ONNX Runtime Reference - Chapter 6: Python API - OrtValue and Tensors

Complete reference for OrtValue, tensor operations, IO Binding, and DLPack integration.

---

## 6.1 OrtValue Creation

### 6.1.1 From Numpy Array

```python
import onnxruntime as ort
import numpy as np

# CPU tensor
arr = np.random.randn(1, 3, 224, 224).astype(np.float32)
ort_val = ort.OrtValue.ortvalue_from_numpy(arr)

# GPU tensor (CUDA device 0)
ort_val_gpu = ort.OrtValue.ortvalue_from_numpy(arr, "cuda", 0)

# GPU tensor (CUDA device 1)
ort_val_gpu1 = ort.OrtValue.ortvalue_from_numpy(arr, "cuda", 1)
```

### 6.1.2 From Shape and Type

```python
# Create empty tensor on CPU
ort_val = ort.OrtValue.ortvalue_from_shape_and_type(
    [1, 3, 224, 224],
    np.float32
)

# Create empty tensor on GPU
ort_val_gpu = ort.OrtValue.ortvalue_from_shape_and_type(
    [1, 3, 224, 224],
    np.float32,
    "cuda", 0
)
```

---

## 6.2 OrtValue Properties

```python
ort_val = ort.OrtValue.ortvalue_from_numpy(np.zeros([2, 3], dtype=np.float32))

# Check if tensor
ort_val.is_tensor()        # True

# Get data type
ort_val.dtype()            # numpy.float32

# Get shape
ort_val.shape()            # [2, 3]

# Get device name
ort_val.device_name()      # 'Cpu' or 'Cuda'

# Get raw data pointer
ort_val.data_ptr()         # Memory address

# Check if has value
ort_val.has_value()        # True
```

---

## 6.3 OrtValue to Numpy Conversion

```python
# Convert to numpy (copies data from GPU if needed)
np_array = ort_val.numpy()
# Returns: numpy.ndarray

# For GPU tensors, .numpy() copies data back to CPU
ort_val_gpu = ort.OrtValue.ortvalue_from_numpy(arr, "cuda", 0)
cpu_array = ort_val_gpu.numpy()  # Copies GPU → CPU
```

---

## 6.4 IO Binding

### 6.4.1 IOBinding Class

```python
io_binding = sess.io_binding()
```

### 6.4.2 Binding Inputs

```python
# Bind CPU numpy array
io_binding.bind_cpu_input("input_name", np_array)

# Bind OrtValue
ort_val = ort.OrtValue.ortvalue_from_numpy(np_array)
io_binding.bind_ortvalue_input("input_name", ort_val)

# Bind GPU OrtValue
ort_val_gpu = ort.OrtValue.ortvalue_from_numpy(np_array, "cuda", 0)
io_binding.bind_ortvalue_input("input_name", ort_val_gpu)
```

### 6.4.3 Binding Outputs

```python
# Bind output to device (let ORT allocate)
io_binding.bind_output("output_name", "cuda", 0)

# Bind output to CPU
io_binding.bind_output("output_name")

# Bind output to specific OrtValue
output_val = ort.OrtValue.ortvalue_from_shape_and_type([1, 1000], np.float32, "cuda", 0)
io_binding.bind_ortvalue_output("output_name", output_val)
```

### 6.4.4 Running with IO Binding

```python
# Run
sess.run_with_iobinding(io_binding)

# Get outputs as OrtValues
ort_outputs = io_binding.get_outputs()
for out in ort_outputs:
    print(f"Shape: {out.shape()}, Device: {out.device_name()}")

# Copy all outputs to CPU as numpy arrays
cpu_outputs = io_binding.copy_outputs_to_cpu()
for arr in cpu_outputs:
    print(f"Array shape: {arr.shape}, dtype: {arr.dtype}")
```

### 6.4.5 Complete GPU Inference with IO Binding

```python
import onnxruntime as ort
import numpy as np

# Create session with CUDA
sess = ort.InferenceSession("model.onnx",
    providers=[("CUDAExecutionProvider", {"device_id": 0})])

# Prepare input on GPU
input_array = np.random.randn(1, 3, 224, 224).astype(np.float32)
input_ort = ort.OrtValue.ortvalue_from_numpy(input_array, "cuda", 0)

# Create IO binding
io_binding = sess.io_binding()

# Bind input (already on GPU, no copy)
io_binding.bind_ortvalue_input("input", input_ort)

# Bind output to GPU (let ORT allocate on GPU)
io_binding.bind_output("output", "cuda", 0)

# Run inference
sess.run_with_iobinding(io_binding)

# Get output (still on GPU)
gpu_outputs = io_binding.get_outputs()
output_ort = gpu_outputs[0]
print(f"Output shape: {output_ort.shape()}, device: {output_ort.device_name()}")

# Copy to CPU only when needed
cpu_output = output_ort.numpy()
print(f"CPU output shape: {cpu_output.shape}")
```

---

## 6.5 Sparse Tensor Support

### 6.5.1 Creating Sparse Tensors

```python
# COO format sparse tensor
dense_shape = [4, 4]
values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
indices = np.array([[0, 0], [1, 2], [3, 1]], dtype=np.int64)

sparse = ort.SparseTensor.sparse_coo_from_numpy(dense_shape, values, indices)

# CSR format sparse tensor
values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
inner_indices = np.array([0, 2, 1], dtype=np.int64)
outer_indices = np.array([0, 1, 2, 3], dtype=np.int64)

sparse_csr = ort.SparseTensor.sparse_csr_from_numpy(
    dense_shape, values, inner_indices, outer_indices)
```

---

## 6.6 DLPack Integration

```python
import onnxruntime as ort
import numpy as np

# From DLPack to OrtValue
arr = np.random.randn(2, 3).astype(np.float32)
dlpack_tensor = arr.__dlpack__()  # Get DLPack capsule
ort_val = ort.OrtValue.from_dlpack(dlpack_tensor)

# From OrtValue to DLPack
dlpack_cap = ort_val.to_dlpack()
result = np.from_dlpack(dlpack_cap)

# GPU DLPack (with PyTorch)
import torch
torch_tensor = torch.randn(2, 3, device="cuda")
dlpack_tensor = torch_tensor.__dlpack__()
ort_val = ort.OrtValue.from_dlpack(dlpack_tensor)
```

---

## 6.7 Zero-Copy Patterns

### 6.7.1 CPU Zero-Copy with Numpy

```python
# Create OrtValue sharing memory with numpy array
arr = np.random.randn(1, 3, 224, 224).astype(np.float32)
ort_val = ort.OrtValue.ortvalue_from_numpy(arr)

# The OrtValue shares memory with arr - no copy!
# Modifying arr will affect ort_val
```

### 6.7.2 GPU Zero-Copy with IO Binding

```python
# Input stays on GPU, output stays on GPU
input_ort = ort.OrtValue.ortvalue_from_numpy(input_array, "cuda", 0)

io_binding = sess.io_binding()
io_binding.bind_ortvalue_input("input", input_ort)
io_binding.bind_output("output", "cuda", 0)

sess.run_with_iobinding(io_binding)

# Output is already on GPU - zero copy!
gpu_output = io_binding.get_outputs()[0]
```

### 6.7.3 Pinned Memory for CPU-GPU Transfer

```python
# Allocate pinned memory for faster CPU→GPU transfer
input_array = np.ascontiguousarray(
    np.random.randn(1, 3, 224, 224).astype(np.float32)
)
```

---

## 6.8 Tensor Type Mapping

| ONNX Type | Numpy dtype | C Type |
|-----------|------------|--------|
| FLOAT | float32 | float |
| UINT8 | uint8 | uint8_t |
| INT8 | int8 | int8_t |
| UINT16 | uint16 | uint16_t |
| INT16 | int16 | int16_t |
| INT32 | int32 | int32_t |
| INT64 | int64 | int64_t |
| STRING | object (str) | std::string |
| BOOL | bool | bool |
| FLOAT16 | float16 | IEEE 754 half |
| DOUBLE | float64 | double |
| UINT32 | uint32 | uint32_t |
| UINT64 | uint64 | uint64_t |
| BFLOAT16 | - (custom) | bfloat16 |
